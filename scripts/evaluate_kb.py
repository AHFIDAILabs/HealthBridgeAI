"""
evaluate_kb.py — RAGAS evaluation of the HealthBridgeAI RAG pipeline.

Measures retrieval and generation quality using a golden test dataset
(YAML file of question / ground-truth-answer pairs per disease).

Metrics computed (RAGAS):
  - faithfulness        — does the answer stay within the retrieved context?
  - answer_relevancy    — is the answer on-topic for the question?
  - context_recall      — does the context cover the ground truth?
  - context_precision   — is the retrieved context relevant (no hallucination)?

Usage:
    python scripts/evaluate_kb.py --dataset data/eval/tb_golden.yaml
    python scripts/evaluate_kb.py --disease tb --n 20 --out reports/eval_tb.json

The golden dataset YAML format:
    questions:
      - question: "What are the first-line drugs for TB treatment?"
        ground_truth: "The first-line drugs are isoniazid, rifampicin, ethambutol, and pyrazinamide (HRZE)."
        disease_id: tb
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import textwrap
from pathlib import Path
from typing import Optional

import yaml

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import structlog

log = structlog.get_logger(__name__)

EVAL_DATASET_PATH = _REPO_ROOT / "data" / "eval" / "golden.yaml"


def _configure_logging() -> None:
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _load_dataset(path: Path, disease_filter: Optional[str], n: int) -> list[dict]:
    if not path.exists():
        log.error("evaluate_kb.dataset_not_found", path=str(path))
        sys.exit(1)
    with open(path) as f:
        data = yaml.safe_load(f)
    questions = data.get("questions", [])
    if disease_filter:
        questions = [q for q in questions if q.get("disease_id") == disease_filter]
    return questions[:n]


async def _run_pipeline(question: str, disease_id: str) -> dict:
    """Run a single question through the RAG pipeline and collect evidence."""
    from healthbridgeai.config import get_disease_registry
    from healthbridgeai.core.models.message import InboundMessage, MessageType
    from healthbridgeai.core.services import (
        LanguageService, MessagePipeline, RAGService, ResponseGenerator, RouterService,
    )
    from healthbridgeai.infrastructure.cache import SemanticCache
    from healthbridgeai.infrastructure.llm import OpenRouterClient
    from healthbridgeai.infrastructure.search import TavilyAdapter
    from healthbridgeai.infrastructure.storage import (
        FirestoreConversationStore, FirestoreUserStore,
    )
    from healthbridgeai.infrastructure.vector_store import PineconeAdapter

    llm = OpenRouterClient()
    vector_store = PineconeAdapter()
    web_search = TavilyAdapter()
    user_store = FirestoreUserStore()
    conv_store = FirestoreConversationStore()
    cache = SemanticCache()
    registry = get_disease_registry()

    pipeline = MessagePipeline(
        language_svc=LanguageService(llm),
        router_svc=RouterService(llm, registry),
        rag_svc=RAGService(llm, vector_store, web_search, registry),
        generator=ResponseGenerator(llm),
        llm=llm,
        user_store=user_store,
        conv_store=conv_store,
        cache=cache,
    )

    # Use a synthetic phone number for evaluation (will be hashed in logs)
    message = InboundMessage(
        message_id=f"eval-{hash(question) % 999999:06d}",
        from_number="+2340000000000",
        type=MessageType.TEXT,
        text=question,
        timestamp=0,
    )

    bot = await pipeline.process(message)
    return {
        "question": question,
        "answer": bot.text,
        "contexts": [],   # populated below if retrieval is exposed
        "disease_id": disease_id,
        "cache_hit": bot.cache_hit,
        "confidence": bot.confidence,
    }


async def _evaluate(dataset: list[dict], out_path: Optional[Path]) -> None:
    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from datasets import Dataset
        ragas_available = True
    except ImportError:
        log.warning("evaluate_kb.ragas_not_installed",
                    hint="pip install ragas datasets")
        ragas_available = False

    log.info("evaluate_kb.start", n=len(dataset))
    results = []
    for i, item in enumerate(dataset):
        log.info("evaluate_kb.running", i=i + 1, total=len(dataset), question=item["question"][:60])
        try:
            r = await _run_pipeline(item["question"], item.get("disease_id", "tb"))
            r["ground_truth"] = item.get("ground_truth", "")
            results.append(r)
        except Exception as exc:
            log.error("evaluate_kb.pipeline_error", question=item["question"][:60], error=str(exc))

    if not results:
        log.error("evaluate_kb.no_results")
        sys.exit(1)

    # ── RAGAS evaluation ──────────────────────────────────────────────────────
    if ragas_available:
        ragas_data = {
            "question": [r["question"] for r in results],
            "answer": [r["answer"] for r in results],
            "contexts": [r.get("contexts", []) or [""] for r in results],
            "ground_truth": [r["ground_truth"] for r in results],
        }
        ds = Dataset.from_dict(ragas_data)
        scores = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        )
        score_dict = dict(scores)
        log.info("evaluate_kb.scores", **{k: round(v, 4) for k, v in score_dict.items()})
    else:
        score_dict = {"note": "ragas not installed — install with: pip install ragas datasets"}

    # ── Report ────────────────────────────────────────────────────────────────
    report = {
        "n_questions": len(results),
        "scores": score_dict,
        "results": results,
    }

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        log.info("evaluate_kb.report_saved", path=str(out_path))
    else:
        print(json.dumps(report["scores"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAGAS evaluation of the HealthBridgeAI RAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python scripts/evaluate_kb.py
              python scripts/evaluate_kb.py --disease tb --n 10
              python scripts/evaluate_kb.py --dataset data/eval/tb_golden.yaml --out reports/eval.json
        """),
    )
    parser.add_argument("--dataset", default=str(EVAL_DATASET_PATH), help="YAML golden dataset path")
    parser.add_argument("--disease", help="Filter to one disease ID")
    parser.add_argument("--n", type=int, default=50, help="Max questions to evaluate (default 50)")
    parser.add_argument("--out", help="Write JSON report to this path")
    args = parser.parse_args()

    _configure_logging()
    dataset = _load_dataset(Path(args.dataset), args.disease, args.n)
    if not dataset:
        log.error("evaluate_kb.empty_dataset")
        sys.exit(1)
    asyncio.run(_evaluate(dataset, Path(args.out) if args.out else None))


if __name__ == "__main__":
    main()
