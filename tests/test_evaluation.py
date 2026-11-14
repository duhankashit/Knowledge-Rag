import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import router
from app.core.config import Settings
from app.services.evaluation import EvaluationService, retrieval_metrics


class RetrievalMetricsTests(unittest.TestCase):
    def test_metrics_match_relevant_retrieved_contexts_at_k(self) -> None:
        metrics = retrieval_metrics(
            ["irrelevant passage", "the target information is here"],
            ["the target information is here"],
            2,
        )

        self.assertEqual(metrics["precision_at_k"], 0.5)
        self.assertEqual(metrics["recall_at_k"], 1.0)
        self.assertEqual(metrics["mrr"], 0.5)
        self.assertIsNotNone(metrics["ndcg"])

    def test_metrics_are_unavailable_without_reference_contexts(self) -> None:
        metrics = retrieval_metrics(["some retrieved passage"], [], 5)

        self.assertEqual(
            metrics,
            {
                "precision_at_k": None,
                "recall_at_k": None,
                "mrr": None,
                "ndcg": None,
            },
        )


class EvaluationFrameworkAdapterTests(unittest.TestCase):
    @staticmethod
    def make_case() -> dict:
        return {
            "question": "What is the fact?",
            "expected_answer": "A fact.",
            "answer": "A fact.",
            "retrieved_contexts": ["A fact."],
            "relevant_contexts": [],
            "retrieval": {},
            "generation": {},
            "safety": {},
            "reasons": {},
        }

    def test_ragas_skips_ground_truth_metrics_without_reference_data(self) -> None:
        class FakeModel:
            def encode(self, texts: list[str], convert_to_numpy: bool = True) -> list:
                return [[0.1, 0.2] for _ in texts]

        class FakeVectorStore:
            model = FakeModel()

        service = EvaluationService(
            None,
            object(),
            FakeVectorStore(),
            Settings(_env_file=None, grok_api_key="test-key"),
        )
        case = self.make_case()
        case["expected_answer"] = None

        with patch(
            "ragas.evaluate",
            return_value=SimpleNamespace(
                scores=[
                    {
                        "faithfulness": 0.9,
                        "answer_relevancy": 0.8,
                    }
                ]
            ),
        ) as evaluate:
            service._run_ragas([case])

        self.assertEqual(len(evaluate.call_args.kwargs["metrics"]), 2)
        self.assertIsNone(evaluate.call_args.args[0].samples[0].reference)
        self.assertEqual(case["generation"]["faithfulness"], 0.9)
        self.assertIsNone(case["generation"]["correctness"])

    def test_ragas_enables_ground_truth_metrics_with_expected_answer(self) -> None:
        class FakeModel:
            def encode(self, texts: list[str], convert_to_numpy: bool = True) -> list:
                return [[0.1, 0.2] for _ in texts]

        class FakeVectorStore:
            model = FakeModel()

        service = EvaluationService(
            None,
            object(),
            FakeVectorStore(),
            Settings(_env_file=None, grok_api_key="test-key"),
        )
        case = self.make_case()
        case["relevant_contexts"] = ["A fact."]

        with patch(
            "ragas.evaluate",
            return_value=SimpleNamespace(
                scores=[
                    {
                        "faithfulness": 0.9,
                        "answer_relevancy": 0.8,
                        "answer_correctness": 0.7,
                        "context_precision": 0.6,
                        "context_recall": 0.5,
                    }
                ]
            ),
        ) as evaluate:
            service._run_ragas([case])

        self.assertEqual(len(evaluate.call_args.kwargs["metrics"]), 5)
        self.assertEqual(
            evaluate.call_args.args[0].samples[0].reference,
            "A fact.",
        )
        self.assertEqual(case["retrieval"]["context_recall"], 0.5)

    def test_deepeval_uses_context_and_custom_judge_for_all_metrics(self) -> None:
        class FakeGroq:
            def __init__(self) -> None:
                self.call_index = 0

            def generate(self, messages: list[dict], model: str | None = None) -> str:
                responses = [
                    '{"truths":["A fact."]}',
                    '{"claims":["A fact."]}',
                    '{"verdicts":[{"verdict":"yes","reason":"Supported."}]}',
                    '{"reason":"Supported."}',
                    '{"statements":["What is the fact?"]}',
                    '{"verdicts":[{"verdict":"yes","reason":"Relevant."}]}',
                    '{"reason":"Relevant."}',
                    '{"verdicts":[{"verdict":"yes","reason":"Grounded."}]}',
                    '{"reason":"Grounded."}',
                    '{"opinions":["A fact."]}',
                    '{"verdicts":[{"verdict":"no","reason":"Not toxic."}]}',
                    '{"reason":"Not toxic."}',
                    '{"steps":["Compare answer with expected output."]}',
                    '{"score":10,"reason":"Correct."}',
                ]
                response = responses[self.call_index]
                self.call_index += 1
                return response

        grok = FakeGroq()
        service = EvaluationService(
            None,
            grok,
            None,
            SimpleNamespace(grok_model="test-model"),
        )
        case = self.make_case()

        service._run_deepeval([case])

        self.assertEqual(grok.call_index, 14)
        self.assertTrue(
            {"faithfulness", "relevance", "correctness"}
            <= case["generation"].keys()
        )
        self.assertTrue({"hallucination", "toxicity"} <= case["safety"].keys())
        self.assertEqual(case["generation"]["correctness"], 1.0)

    def test_inline_evaluation_uses_the_existing_answer(self) -> None:
        class FakeRag:
            retrieval_top_k = 5

            def answer(self, question: str) -> tuple[str, list]:
                raise AssertionError("Inline evaluation must not generate again.")

        service = EvaluationService(
            FakeRag(),
            object(),
            object(),
            Settings(_env_file=None),
        )
        test_case = {
            "question": "What is in this source?",
            "answer": "The original chat answer.",
            "expected_answer": None,
            "retrieved_contexts": ["Original retrieved context."],
            "relevant_contexts": [],
            "generation_latency_ms": 125,
        }

        with patch.object(service, "_run_ragas"):
            result = service.run("ragas", [test_case])

        self.assertEqual(result["cases"][0]["answer"], "The original chat answer.")
        self.assertEqual(
            result["cases"][0]["retrieved_contexts"],
            ["Original retrieved context."],
        )
        self.assertEqual(result["cases"][0]["performance"]["latency_ms"], 125)
        self.assertIsNone(result["summary"]["generation"]["correctness"])
        self.assertIsNone(result["summary"]["retrieval"]["recall_at_k"])


class EvaluationRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        class FakeEvaluationService:
            def __init__(self) -> None:
                self.runs: list[tuple[str, list[dict]]] = []

            def run(self, framework: str, cases: list[dict]) -> dict:
                self.runs.append((framework, cases))
                return {"run_id": "new-run", "framework": framework}

        class FakeRag:
            def answer(self, question: str, history: list[dict]) -> tuple[str, list[dict]]:
                return "A response.", []

        self.evaluation = FakeEvaluationService()
        app = FastAPI()
        app.state.evaluation = self.evaluation
        app.state.rag = FakeRag()
        app.include_router(router, prefix="/api/v1")
        self.client = TestClient(app)

    def test_run_endpoint_passes_framework_and_cases_to_service(self) -> None:
        response = self.client.post(
            "/api/v1/evaluation/run",
            json={
                "framework": "deepeval",
                "cases": [
                    {
                        "question": "What is in the document?",
                        "answer": "The original chat answer.",
                        "retrieved_contexts": ["A source passage."],
                        "generation_latency_ms": 125,
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["run_id"], "new-run")
        self.assertEqual(self.evaluation.runs[0][0], "deepeval")
        self.assertEqual(
            self.evaluation.runs[0][1][0]["relevant_contexts"], []
        )
        self.assertIsNone(self.evaluation.runs[0][1][0]["expected_answer"])
        self.assertEqual(
            self.evaluation.runs[0][1][0]["answer"],
            "The original chat answer.",
        )
        self.assertEqual(
            self.evaluation.runs[0][1][0]["retrieved_contexts"],
            ["A source passage."],
        )
        self.assertEqual(
            self.evaluation.runs[0][1][0]["generation_latency_ms"], 125
        )

    def test_run_endpoint_rejects_empty_case_sets(self) -> None:
        response = self.client.post(
            "/api/v1/evaluation/run",
            json={"framework": "ragas", "cases": []},
        )

        self.assertEqual(response.status_code, 422)

    def test_chat_endpoint_returns_generation_latency(self) -> None:
        response = self.client.post(
            "/api/v1/chat",
            json={"question": "What is in the document?"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "A response.")
        self.assertGreaterEqual(response.json()["generation_latency_ms"], 0)


if __name__ == "__main__":
    unittest.main()
