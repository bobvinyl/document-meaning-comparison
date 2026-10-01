#!/usr/bin/env python3
"""Driver script for semantic comparison of two documents."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from semantic_compare.nlp_processor import DocumentSemanticProcessor


def read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise SystemExit(
            "PDF support requires pypdf. Install dependencies from requirements.txt."
        ) from exc

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(pages)


def read_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise SystemExit(
            "DOCX support requires python-docx. Install dependencies from requirements.txt."
        ) from exc

    document = Document(str(path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def maybe_read_file(value: str) -> str:
    """Read from file if value exists as a path; otherwise treat as raw text."""
    path = Path(value)
    if not path.is_file():
        return value

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix == ".docx":
        return read_docx(path)
    return path.read_text(encoding="utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two documents semantically using SBERT cosine similarity."
    )
    parser.add_argument("doc1", nargs="?", help="Document 1: existing file path or raw text")
    parser.add_argument("doc2", nargs="?", help="Document 2: existing file path or raw text")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SBERT model name (default: sentence-transformers/all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable minimal text normalization",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON",
    )
    parser.add_argument(
        "--with-label",
        action="store_true",
        help="Include low/medium/high interpretation in plain output",
    )
    parser.add_argument(
        "--sentence-align",
        action="store_true",
        help="Show top sentence-level semantic alignments",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of sentence alignments to return with --sentence-align (default: 5)",
    )
    parser.add_argument(
        "--batch-csv",
        help="Path to CSV with columns doc1,doc2 and optional pair_id",
    )
    parser.add_argument(
        "--out",
        help="Batch output path (.csv or .jsonl). Defaults to batch_results.csv",
    )
    parser.add_argument(
        "--batch-format",
        choices=["csv", "jsonl"],
        help="Batch output format. If omitted, inferred from --out extension.",
    )
    return parser


def compare_pair(
    doc1_value: str,
    doc2_value: str,
    processor: DocumentSemanticProcessor,
    normalize: bool,
    include_alignments: bool,
    top_k: int,
) -> dict[str, Any]:
    text1 = maybe_read_file(doc1_value)
    text2 = maybe_read_file(doc2_value)
    return processor.compare(
        text1=text1,
        text2=text2,
        normalize=normalize,
        include_alignments=include_alignments,
        top_k=top_k,
    )


def infer_batch_format(output_path: Path, requested_format: str | None) -> str:
    if requested_format:
        return requested_format
    if output_path.suffix.lower() == ".jsonl":
        return "jsonl"
    return "csv"


def write_batch_csv(out_path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "pair_id",
        "doc1",
        "doc2",
        "model",
        "cosine_similarity",
        "interpretation",
        "error",
        "alignments_json",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_batch_jsonl(out_path: Path, rows: list[dict[str, Any]]) -> None:
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = dict(row)
            if payload.get("alignments_json"):
                payload["sentence_alignments"] = json.loads(payload["alignments_json"])
            payload.pop("alignments_json", None)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_batch(args: argparse.Namespace, processor: DocumentSemanticProcessor) -> None:
    if not args.batch_csv:
        raise SystemExit("Batch mode requires --batch-csv.")

    batch_path = Path(args.batch_csv)
    if not batch_path.is_file():
        raise SystemExit(f"Batch CSV not found: {batch_path}")

    out_path = Path(args.out) if args.out else Path("batch_results.csv")
    batch_format = infer_batch_format(out_path, args.batch_format)

    rows_out: list[dict[str, Any]] = []
    with batch_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit("Batch CSV has no header row.")

        required_columns = {"doc1", "doc2"}
        if not required_columns.issubset(set(reader.fieldnames)):
            raise SystemExit("Batch CSV must include headers: doc1, doc2")

        for row_index, row in enumerate(reader, start=1):
            pair_id = row.get("pair_id") or str(row_index)
            doc1_value = row.get("doc1", "")
            doc2_value = row.get("doc2", "")

            base = {
                "pair_id": pair_id,
                "doc1": doc1_value,
                "doc2": doc2_value,
                "model": args.model,
                "cosine_similarity": "",
                "interpretation": "",
                "error": "",
                "alignments_json": "",
            }

            try:
                result = compare_pair(
                    doc1_value=doc1_value,
                    doc2_value=doc2_value,
                    processor=processor,
                    normalize=not args.no_normalize,
                    include_alignments=args.sentence_align,
                    top_k=args.top_k,
                )
                base["cosine_similarity"] = f"{result['cosine_similarity']:.6f}"
                base["interpretation"] = result["interpretation"]
                if "sentence_alignments" in result:
                    base["alignments_json"] = json.dumps(
                        result["sentence_alignments"],
                        ensure_ascii=False,
                    )
            except Exception as exc:  # Keep batch resilient and continue.
                base["error"] = str(exc)

            rows_out.append(base)

    if batch_format == "jsonl":
        write_batch_jsonl(out_path, rows_out)
    else:
        write_batch_csv(out_path, rows_out)

    print(
        f"Batch complete: {len(rows_out)} pairs processed. "
        f"Output written to {out_path} ({batch_format})."
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.batch_csv and (args.doc1 or args.doc2):
        raise SystemExit("Use either positional doc1/doc2 or --batch-csv, not both.")
    if not args.batch_csv and (not args.doc1 or not args.doc2):
        raise SystemExit("Provide doc1 and doc2, or use --batch-csv for batch mode.")

    processor = DocumentSemanticProcessor(model_name=args.model)

    if args.batch_csv:
        run_batch(args, processor)
        return

    result = compare_pair(
        doc1_value=args.doc1,
        doc2_value=args.doc2,
        processor=processor,
        normalize=not args.no_normalize,
        include_alignments=args.sentence_align,
        top_k=args.top_k,
    )
    score = float(result["cosine_similarity"])

    if args.json:
        payload = {
            "model": args.model,
            "cosine_similarity": score,
            "interpretation": result["interpretation"],
        }
        if args.sentence_align:
            payload["sentence_alignments"] = result.get("sentence_alignments", [])
        print(json.dumps(payload, indent=2))
        return

    print(f"Cosine similarity: {score:.6f}")
    if args.with_label:
        print(f"Interpretation: {result['interpretation']}")
    if args.sentence_align:
        alignments = result.get("sentence_alignments", [])
        print("\nTop sentence alignments:")
        if not alignments:
            print("- No sentence alignments found.")
        else:
            for index, item in enumerate(alignments, start=1):
                print(f"{index}. score={item['score']:.4f}")
                print(f"   doc1: {item['doc1_sentence']}")
                print(f"   doc2: {item['doc2_sentence']}")


if __name__ == "__main__":
    main()
