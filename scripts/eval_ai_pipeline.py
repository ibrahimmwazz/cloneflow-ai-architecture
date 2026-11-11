"""
CloneFlow AI — CI-Ready LLMOps Evaluation CLI
=============================================
Evaluates the Hybrid RAG pipeline, Multi-Step Scene Director Agent, and
Responsible AI Guardrails end-to-end. Exits 0 on SLA pass, 1 on failure.

Usage:
    python scripts/eval_ai_pipeline.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.services.ai_evaluator import AIEvaluator


async def _main() -> int:
    evaluator = AIEvaluator()
    persist = os.getenv("PERSIST_EVAL_RESULTS", "false").lower() == "true"
    use_remote_rag = os.getenv("EVAL_USE_REMOTE_RAG", "false").lower() == "true"

    report = await evaluator.run_benchmark(
        persist_results=persist,
        use_remote_rag=use_remote_rag,
    )

    print("\n=== CloneFlow AI — RAG, Agent & Guardrails Evaluation ===")
    print(f"Suite   : {report['suite_name']}")
    print(f"Passed  : {report['passed']}")
    print(f"Persisted: {report['persisted']}")
    print("\nMetrics:")
    for key, val in report["metrics"].items():
        print(f"  {key:<28}: {val}")

    print("\nAgent Cases:")
    for case in report["agent_cases"]:
        print(
            f"  - [{case['recommended_model']}] passed={case['verification_passed']} "
            f"grounding={case['grounding_score']} pacing_err={case['pacing_error_ratio']} "
            f"tools={','.join(case['tools_invoked'])}"
        )

    if os.getenv("EVAL_JSON_OUT"):
        Path(os.getenv("EVAL_JSON_OUT")).write_text(json.dumps(report, indent=2), encoding="utf-8")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
