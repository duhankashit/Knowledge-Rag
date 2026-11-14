import asyncio
import math
import re
import time
import uuid
from datetime import datetime, timezone
from statistics import mean
from typing import Any

import tiktoken

from app.clients.grok import GrokClient
from app.core.config import Settings
from app.repositories.vector_store import VectorStore
from app.services.rag import RAGService

_REFUSAL_PATTERN = re.compile(
    r"\b(i (?:cannot|can't|am unable to)|i don['’]t know|i do not know)\b",
    re.IGNORECASE,
)


def _tokenizer(model_name: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _average(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return mean(available) if available else None


def retrieval_metrics(
    retrieved: list[str], relevant: list[str], k: int
) -> dict[str, float | None]:
    if not relevant:
        return {
            "precision_at_k": None,
            "recall_at_k": None,
            "mrr": None,
            "ndcg": None,
        }

    normalized_relevant = [" ".join(value.casefold().split()) for value in relevant]
    hits = [
        any(
            reference in " ".join(context.casefold().split())
            or " ".join(context.casefold().split()) in reference
            for reference in normalized_relevant
        )
        for context in retrieved[:k]
    ]
    relevant_found = sum(hits)
    reciprocal_rank = next(
        (1 / rank for rank, hit in enumerate(hits, start=1) if hit),
        0.0,
    )
    dcg = sum(
        int(hit) / math.log2(rank + 1)
        for rank, hit in enumerate(hits, start=1)
    )
    ideal_relevant = min(len(relevant), k)
    ideal_dcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant + 1)
    )
    return {
        "precision_at_k": relevant_found / k if k else 0.0,
        "recall_at_k": min(relevant_found / len(relevant), 1.0),
        "mrr": reciprocal_rank,
        "ndcg": dcg / ideal_dcg if ideal_dcg else 0.0,
    }


class EvaluationService:
    def __init__(
        self,
        rag: RAGService,
        grok: GrokClient,
        vector_store: VectorStore,
        settings: Settings,
    ) -> None:
        self.rag = rag
        self.grok = grok
        self.vector_store = vector_store
        self.settings = settings

    def run(self, framework: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
        tokenizer = _tokenizer(self.settings.grok_model)
        case_results: list[dict[str, Any]] = []
        for index, case in enumerate(cases, start=1):
            if case.get("answer") is None:
                started = time.perf_counter()
                answer, sources = self.rag.answer(case["question"])
                latency_ms = (time.perf_counter() - started) * 1000
                contexts = [item["text"] for item in sources]
            else:
                answer = case["answer"]
                contexts = case.get("retrieved_contexts") or []
                latency_ms = case.get("generation_latency_ms") or 0
            relevant_contexts = case.get("relevant_contexts") or []
            input_text = "\n".join([case["question"], *contexts])
            input_tokens = len(tokenizer.encode(input_text))
            output_tokens = len(tokenizer.encode(answer))
            estimated_cost = self._estimate_cost(input_tokens, output_tokens)
            case_results.append(
                {
                    "case": index,
                    "question": case["question"],
                    "expected_answer": case.get("expected_answer"),
                    "answer": answer,
                    "retrieved_contexts": contexts,
                    "relevant_contexts": relevant_contexts,
                    "retrieval": retrieval_metrics(
                        contexts, relevant_contexts, self.rag.retrieval_top_k
                    ),
                    "generation": {},
                    "safety": {
                        "refusal": float(bool(_REFUSAL_PATTERN.search(answer)))
                    },
                    "performance": {
                        "latency_ms": latency_ms,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "estimated_cost": estimated_cost,
                    },
                    "reasons": {},
                }
            )

        if framework == "ragas":
            self._run_ragas(case_results)
        elif framework == "deepeval":
            self._run_deepeval(case_results)
        else:
            raise ValueError(f"Unsupported evaluation framework: {framework}.")

        for case in case_results:
            faithfulness = case["generation"].get("faithfulness")
            if faithfulness is not None:
                case["safety"]["hallucination"] = 1 - faithfulness
            case["failed"] = self._case_failed(case)

        summary = self._summarize(case_results)
        run = {
            "run_id": str(uuid.uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "framework": framework,
            "case_count": len(case_results),
            "failure_count": sum(case["failed"] for case in case_results),
            "summary": summary,
            "cases": case_results,
        }
        return run

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float | None:
        input_rate = self.settings.grok_input_cost_per_million
        output_rate = self.settings.grok_output_cost_per_million
        if input_rate is None or output_rate is None:
            return None
        return (
            input_tokens * input_rate + output_tokens * output_rate
        ) / 1_000_000

    def _run_ragas(self, cases: list[dict[str, Any]]) -> None:
        from langchain_openai import ChatOpenAI
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.embeddings.base import BaseRagasEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            AnswerCorrectness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
        from ragas.run_config import RunConfig

        class VectorStoreEmbeddings(BaseRagasEmbeddings):
            def __init__(self, vector_store: VectorStore) -> None:
                super().__init__()
                self.vector_store = vector_store
                self.set_run_config(RunConfig(timeout=120, max_retries=1, max_workers=2))

            def embed_query(self, text: str) -> list[float]:
                vector = self.vector_store.model.encode(
                    [text], convert_to_numpy=True
                )[0]
                return vector.tolist()

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                vectors = self.vector_store.model.encode(
                    texts, convert_to_numpy=True
                )
                return vectors.tolist()

            async def aembed_query(self, text: str) -> list[float]:
                return await asyncio.to_thread(self.embed_query, text)

            async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
                return await asyncio.to_thread(self.embed_documents, texts)

        api_base = self.settings.grok_base_url.rstrip("/")
        if not api_base.endswith("/openai/v1"):
            api_base += "/openai/v1"
        chat_llm = ChatOpenAI(
            model=self.settings.grok_model,
            api_key=self.settings.grok_api_key,
            base_url=api_base,
            temperature=0,
        )
        ragas_llm = LangchainLLMWrapper(chat_llm)
        embeddings = VectorStoreEmbeddings(self.vector_store)
        metrics = [
            Faithfulness(llm=ragas_llm),
            AnswerRelevancy(llm=ragas_llm, embeddings=embeddings),
        ]
        has_expected_answers = all(
            case["expected_answer"] for case in cases
        )
        if has_expected_answers:
            metrics.extend(
                [
                    AnswerCorrectness(llm=ragas_llm, embeddings=embeddings),
                    ContextPrecision(llm=ragas_llm),
                    ContextRecall(llm=ragas_llm),
                ]
            )
        dataset = EvaluationDataset(
            samples=[
                SingleTurnSample(
                    user_input=case["question"],
                    response=case["answer"],
                    retrieved_contexts=case["retrieved_contexts"] or [""],
                    reference=case["expected_answer"],
                )
                for case in cases
            ]
        )
        results = evaluate(
            dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=embeddings,
            run_config=RunConfig(timeout=120, max_retries=1, max_workers=2),
            show_progress=False,
            raise_exceptions=True,
        )
        for case, scores in zip(cases, results.scores):
            self._apply_scores(
                case,
                {
                    "faithfulness": scores.get("faithfulness"),
                    "relevance": scores.get("answer_relevancy"),
                    "correctness": scores.get("answer_correctness"),
                },
                {
                    "context_precision": scores.get("context_precision"),
                    "context_recall": scores.get("context_recall"),
                },
            )

    def _run_deepeval(self, cases: list[dict[str, Any]]) -> None:
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            FaithfulnessMetric,
            GEval,
            HallucinationMetric,
            ToxicityMetric,
        )
        from deepeval.models import DeepEvalBaseLLM
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams

        class GroqJudge(DeepEvalBaseLLM):
            def __init__(self, grok: GrokClient, model_name: str) -> None:
                self.grok = grok
                super().__init__(model_name=model_name)

            def load_model(self) -> GrokClient:
                return self.grok

            def generate(self, prompt: str) -> str:
                return self.grok.generate(
                    [{"role": "user", "content": prompt}],
                    model=self.model_name,
                )

            async def a_generate(self, prompt: str) -> str:
                return await asyncio.to_thread(self.generate, prompt)

            def get_model_name(self) -> str:
                return self.model_name

        judge = GroqJudge(self.grok, self.settings.grok_model)
        for case in cases:
            test_case = LLMTestCase(
                input=case["question"],
                actual_output=case["answer"],
                expected_output=case["expected_answer"],
                context=case["retrieved_contexts"] or [""],
                retrieval_context=case["retrieved_contexts"] or [""],
            )
            metrics = [
                (
                    "faithfulness",
                    FaithfulnessMetric(model=judge, async_mode=False),
                ),
                (
                    "relevance",
                    AnswerRelevancyMetric(model=judge, async_mode=False),
                ),
                (
                    "hallucination",
                    HallucinationMetric(model=judge, async_mode=False),
                ),
                (
                    "toxicity",
                    ToxicityMetric(model=judge, async_mode=False),
                ),
            ]
            if case["expected_answer"] is not None:
                metrics.append(
                    (
                        "correctness",
                        GEval(
                            name="Correctness",
                            criteria=(
                                "Assess whether the actual output correctly answers the "
                                "input using the expected output as ground truth."
                            ),
                            evaluation_params=[
                                LLMTestCaseParams.INPUT,
                                LLMTestCaseParams.ACTUAL_OUTPUT,
                                LLMTestCaseParams.EXPECTED_OUTPUT,
                            ],
                            model=judge,
                            async_mode=False,
                        ),
                    )
                )
            for name, metric in metrics:
                metric.measure(test_case)
                score = _number(metric.score)
                if name in {"faithfulness", "relevance", "correctness"}:
                    case["generation"][name] = score
                else:
                    case["safety"][name] = score
                if metric.reason:
                    case["reasons"][name] = metric.reason

    @staticmethod
    def _apply_scores(
        case: dict[str, Any],
        generation: dict[str, Any],
        retrieval: dict[str, Any],
    ) -> None:
        case["generation"].update(
            {name: _number(score) for name, score in generation.items()}
        )
        case["retrieval"].update(
            {name: _number(score) for name, score in retrieval.items()}
        )

    @staticmethod
    def _case_failed(case: dict[str, Any]) -> bool:
        generation_failure = any(
            score is not None and score < 0.5
            for name, score in case["generation"].items()
            if name in {"faithfulness", "relevance", "correctness"}
        )
        safety_failure = any(
            score is not None and score > 0.5
            for name, score in case["safety"].items()
            if name in {"hallucination", "toxicity"}
        )
        retrieval_recall = case["retrieval"].get("recall_at_k")
        return (
            generation_failure
            or safety_failure
            or (retrieval_recall is not None and retrieval_recall < 0.5)
        )

    @staticmethod
    def _summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
        def average(path: tuple[str, str]) -> float | None:
            key, metric = path
            return _average([case[key].get(metric) for case in cases])

        estimated_costs = [
            case["performance"]["estimated_cost"] for case in cases
        ]
        return {
            "retrieval": {
                name: average(("retrieval", name))
                for name in (
                    "precision_at_k",
                    "recall_at_k",
                    "mrr",
                    "ndcg",
                    "context_precision",
                    "context_recall",
                )
            },
            "generation": {
                name: average(("generation", name))
                for name in ("faithfulness", "correctness", "relevance")
            },
            "safety": {
                name: average(("safety", name))
                for name in ("hallucination", "toxicity", "refusal")
            },
            "performance": {
                "average_latency_ms": _average(
                    [case["performance"]["latency_ms"] for case in cases]
                ),
                "input_tokens": sum(
                    case["performance"]["input_tokens"] for case in cases
                ),
                "output_tokens": sum(
                    case["performance"]["output_tokens"] for case in cases
                ),
                "estimated_cost": (
                    sum(estimated_costs)
                    if all(cost is not None for cost in estimated_costs)
                    else None
                ),
            },
        }
