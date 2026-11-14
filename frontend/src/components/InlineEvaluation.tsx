import { Activity, AlertCircle, LoaderCircle, ShieldAlert, Target } from "lucide-react";
import { useState } from "react";

import { runEvaluation } from "../api";
import type { ChatMessage, EvaluationFramework, EvaluationRun } from "../types";

interface InlineEvaluationProps {
  message: ChatMessage;
  onResult: (run: EvaluationRun) => void;
}

function score(value: number | null | undefined): string {
  return value == null ? "N/A" : `${(value * 100).toFixed(1)}%`;
}

function Metric({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="inline-evaluation-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default function InlineEvaluation({
  message,
  onResult,
}: InlineEvaluationProps) {
  const [open, setOpen] = useState(false);
  const [framework, setFramework] = useState<EvaluationFramework>("ragas");
  const [expectedAnswer, setExpectedAnswer] = useState("");
  const [relevantContexts, setRelevantContexts] = useState("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");

  async function evaluateAnswer() {
    const question = message.evaluationQuestion?.trim();
    if (!question) {
      setError("The question for this answer is unavailable.");
      return;
    }

    setRunning(true);
    setError("");
    try {
      const expected = expectedAnswer.trim();
      const caseInput = {
        question,
        answer: message.content,
        retrieved_contexts: message.sources?.map((source) => source.excerpt) ?? [],
        relevant_contexts: relevantContexts
          .split(/\r?\n/)
          .map((context) => context.trim())
          .filter(Boolean),
        ...(expected ? { expected_answer: expected } : {}),
        ...(message.generationLatencyMs != null
          ? { generation_latency_ms: message.generationLatencyMs }
          : {}),
      };
      const result = await runEvaluation(framework, [caseInput]);
      onResult(result);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Could not evaluate this response.",
      );
    } finally {
      setRunning(false);
    }
  }

  const run = message.evaluation;
  const result = run?.cases[0];
  const summary = run?.summary;

  return (
    <section className="inline-evaluation">
      <button
        className="inline-evaluation-toggle"
        onClick={() => setOpen((current) => !current)}
        type="button"
        aria-expanded={open}
      >
        <Activity size={14} />
        {run ? "Evaluate again" : "Evaluate this answer"}
      </button>

      {open && (
        <div className="inline-evaluation-form">
          <label>
            <span>Framework</span>
            <select
              value={framework}
              onChange={(event) =>
                setFramework(event.target.value as EvaluationFramework)
              }
              disabled={running}
            >
              <option value="ragas">RAGAS</option>
              <option value="deepeval">DeepEval</option>
            </select>
          </label>
          <label>
            <span>Expected answer (optional; needed for correctness)</span>
            <textarea
              value={expectedAnswer}
              onChange={(event) => setExpectedAnswer(event.target.value)}
              placeholder="Enter a reference answer to score correctness."
              maxLength={8000}
              disabled={running}
            />
          </label>
          <label>
            <span>Relevant passages (optional; one per line)</span>
            <textarea
              value={relevantContexts}
              onChange={(event) => setRelevantContexts(event.target.value)}
              placeholder="Paste reference passages to score retrieval."
              disabled={running}
            />
          </label>
          <p className="inline-evaluation-hint">
            Missing reference data is shown as N/A. Evaluation uses the answer and sources above;
            estimated cost excludes judge-model usage, which may incur additional cost.
          </p>
          {error && (
            <p className="inline-evaluation-error" role="alert">
              <AlertCircle size={14} /> {error}
            </p>
          )}
          <button
            className="button-primary inline-evaluation-run"
            onClick={() => void evaluateAnswer()}
            disabled={running}
            type="button"
          >
            {running ? <LoaderCircle className="spin" size={14} /> : <Activity size={14} />}
            {running ? "Evaluating..." : "Run evaluation"}
          </button>
        </div>
      )}

      {run && summary && (
        <div className="inline-evaluation-report">
          <div className="inline-evaluation-report-heading">
            <span>{run.framework.toUpperCase()} evaluation</span>
            <span className={result?.failed ? "evaluation-failures" : "evaluation-success"}>
              {result?.failed ? "Needs review" : "Passed"}
            </span>
          </div>
          <div className="inline-evaluation-groups">
            <section>
              <h3><ShieldAlert size={14} /> Safety evaluation</h3>
              <Metric label="Hallucination" value={score(summary.safety.hallucination)} />
              <Metric label="Toxicity" value={score(summary.safety.toxicity)} />
              <Metric label="Refusal" value={score(summary.safety.refusal)} />
            </section>
            <section>
              <h3><Activity size={14} /> Performance evaluation</h3>
              <Metric
                label="Response latency"
                value={
                  summary.performance.average_latency_ms == null
                    ? "N/A"
                    : `${Math.round(summary.performance.average_latency_ms)} ms`
                }
              />
              <Metric
                label="Tokens (input / output)"
                value={`${summary.performance.input_tokens} / ${summary.performance.output_tokens}`}
              />
              <Metric
                label="Estimated generation cost"
                value={
                  summary.performance.estimated_cost == null
                    ? "N/A"
                    : `$${summary.performance.estimated_cost.toFixed(6)}`
                }
              />
            </section>
            <section>
              <h3><Activity size={14} /> Faithfulness · Correctness · Relevance</h3>
              <Metric label="Faithfulness" value={score(summary.generation.faithfulness)} />
              <Metric label="Correctness" value={score(summary.generation.correctness)} />
              <Metric label="Relevance" value={score(summary.generation.relevance)} />
            </section>
            <section>
              <h3><Target size={14} /> Recall@K · Precision@K · MRR · NDCG</h3>
              <Metric label="Recall@K" value={score(summary.retrieval.recall_at_k)} />
              <Metric label="Precision@K" value={score(summary.retrieval.precision_at_k)} />
              <Metric label="MRR" value={score(summary.retrieval.mrr)} />
              <Metric label="NDCG" value={score(summary.retrieval.ndcg)} />
              <Metric label="Context precision / recall" value={`${score(summary.retrieval.context_precision)} / ${score(summary.retrieval.context_recall)}`} />
            </section>
          </div>
          {result && Object.keys(result.reasons).length > 0 && (
            <details className="inline-evaluation-reasons">
              <summary>Judge reasoning</summary>
              {Object.entries(result.reasons).map(([name, reason]) => (
                <p key={name}><strong>{name}:</strong> {reason}</p>
              ))}
            </details>
          )}
        </div>
      )}
    </section>
  );
}
