#!/usr/bin/env python3
"""
CLI for the Mini RAG system: index emails, ask questions, run eval.
Usage:
  python cli.py index              # build Qdrant index from emails/
  python cli.py ask "your question" # get answer (requires index + MISTRAL_API_KEY)
  python cli.py ask "question" --subject "Meeting Request"  # with filter
  python cli.py eval               # run quality evaluation (e2e tests)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.config import EMAILS_DIR, MISTRAL_API_KEY
from rag.pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _where_from_args(args: argparse.Namespace) -> dict | None:
    """Build Qdrant payload filter from CLI args."""
    filters = []
    if getattr(args, "subject", None):
        filters.append({"subject": {"$eq": args.subject}})
    if getattr(args, "from_", None):
        filters.append({"from": {"$eq": args.from_}})
    if getattr(args, "to", None):
        filters.append({"to": {"$eq": args.to}})
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def cmd_index(args: argparse.Namespace, pipeline: RAGPipeline) -> None:
    pipeline.index()
    print("Index built successfully.")


def cmd_ask(args: argparse.Namespace, pipeline: RAGPipeline) -> None:
    if not MISTRAL_API_KEY:
        print("Error: MISTRAL_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)
    where = _where_from_args(args)
    answer, results = pipeline.ask(args.query, top_k=args.top_k, where=where)
    print("Retrieved sources:", len(results))
    for i, r in enumerate(results[:3], 1):
        print(f"  {i}. {r.source_file} | {r.subject}")
    print("\nAnswer:", answer)


def cmd_eval(args: argparse.Namespace, pipeline: RAGPipeline) -> None:
    """Run e2e evaluation (imports tests)."""
    import pytest
    test_dir = Path(__file__).resolve().parent / "tests"
    if not test_dir.exists():
        print("No tests/ directory found.", file=sys.stderr)
        sys.exit(1)
    pytest_args = [str(test_dir), "-v", "--tb=short"]
    if args.coverage:
        pytest_args += ["--cov=rag", "--cov-report=term-missing"]
    sys.exit(pytest.main(pytest_args))


def main() -> int:
    parser = argparse.ArgumentParser(description="Mini RAG: index emails, ask questions.")
    sub = parser.add_subparsers(dest="command", required=True)

    # index
    sub.add_parser("index", help="Build Qdrant index from emails/")

    # ask
    ask_p = sub.add_parser("ask", help="Ask a question (requires index and MISTRAL_API_KEY)")
    ask_p.add_argument("query", type=str, help="Your question")
    ask_p.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    ask_p.add_argument("--subject", type=str, help="Filter by subject (exact match)")
    ask_p.add_argument("--from", dest="from_", type=str, help="Filter by sender (exact match)")
    ask_p.add_argument("--to", type=str, help="Filter by receiver (exact match)")

    # eval
    eval_p = sub.add_parser("eval", help="Run end-to-end quality tests")
    eval_p.add_argument("--coverage", action="store_true", help="Report coverage")

    args = parser.parse_args()
    pipeline = RAGPipeline()

    try:
        if args.command == "index":
            cmd_index(args, pipeline)
        elif args.command == "ask":
            cmd_ask(args, pipeline)
        elif args.command == "eval":
            cmd_eval(args, pipeline)
        return 0
    except Exception as e:
        logger.exception("Command failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
