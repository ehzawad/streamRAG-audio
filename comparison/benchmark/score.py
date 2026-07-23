#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CITATION_RE = re.compile(r"\[([^\]\s]+::c\d{4})\]")
CLAUSE_SPLIT_RE = re.compile(r"[.!?;:\n]+|\bbut\b|\bhowever\b|,", re.IGNORECASE)
ABSTENTION_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\binvalid question\b",
        r"\bfalse premise\b|\bpremise (?:is|was) false\b",
        r"\b(?:can not|cannot|can t|could not|unable to)"
        r"(?: [a-z0-9]+){0,4} (?:verify|identify|name|determine|compare|confirm|answer)\b",
        r"\b(?:insufficient|not enough) (?:evidence|information)(?: to [a-z0-9]+)?\b",
        r"\bno (?:available )?(?:evidence|score|source|information)\b",
        r"\b(?:evidence|corpus|sources?|documents?|pages?) (?:does|do|did) not "
        r"(?:establish|provide|support|verify|show|contain)\b",
        r"\bno [a-z0-9 ]{0,60} can be (?:identified|named|verified|determined)\b",
        r"\b(?:i|we) (?:can not|cannot|can t|could not) reliably\b",
    )
)
NON_ABSTENTION_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\bnot an invalid question\b",
        r"\bnot a false premise\b",
        r"\bno evidence problem\b",
        r"\bnot insufficient\b",
    )
)
ADJUDICATION_LABELS = {"perfect", "acceptable", "missing", "incorrect"}
RELATION_PAIRS = (
    ("smaller", "larger"),
    ("lower", "higher"),
    ("less", "more"),
    ("fewer", "more"),
    ("earlier", "later"),
    ("before", "after"),
    ("older", "younger"),
    ("shorter", "longer"),
    ("worse", "better"),
    ("worst", "best"),
    ("least", "most"),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_checksum_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            digest, name = line.split(maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"invalid checksum manifest line {line_number}") from exc
        name = name.lstrip("*").strip()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"invalid SHA-256 at checksum manifest line {line_number}")
        if name in entries:
            raise ValueError(f"duplicate checksum manifest entry: {name}")
        entries[name] = digest
    return entries


def validate_run_integrity(
    rows: list[dict[str, Any]],
    predictions_path: Path,
    manifest_path: Path,
    gold_path: Path,
    evaluation_manifest_path: Path | None = None,
) -> dict[str, Any]:
    issues: list[str] = []
    if not manifest_path.is_file():
        return {
            "status": "missing_or_incomplete",
            "manifest": manifest_path.name,
            "issues": ["adjacent benchmark manifest is missing"],
        }
    try:
        manifest = read_manifest(manifest_path)
    except (json.JSONDecodeError, OSError) as exc:
        return {
            "status": "missing_or_incomplete",
            "manifest": manifest_path.name,
            "issues": [f"benchmark manifest is unreadable: {exc}"],
        }

    if any(key in manifest for key in ("gold", "gold_sha256")):
        issues.append("runner manifest improperly contains scorer-only gold identity")
    if manifest.get("finalized") is not True:
        issues.append("manifest was not finalized after prediction generation")
    if manifest.get("distinct_backend_instances") is not True:
        issues.append("manifest does not prove distinct backend instances")

    manifest_predictions_hash = str(manifest.get("predictions_sha256") or "")
    if (
        not predictions_path.is_file()
        or not manifest_predictions_hash
        or sha256_file(predictions_path) != manifest_predictions_hash
    ):
        issues.append("prediction checksum does not match finalized manifest")

    harness_hashes = manifest.get("benchmark_harness_sha256")
    expected_harness_hashes = {
        "run_benchmark.py": sha256_file(ROOT / "comparison" / "benchmark" / "run_benchmark.py"),
        "typed_trace.py": sha256_file(ROOT / "comparison" / "benchmark" / "typed_trace.py"),
    }
    if harness_hashes != expected_harness_hashes:
        issues.append("benchmark harness source does not match the content-addressed run")

    failed_rows = [row for row in rows if "error" in row]
    if failed_rows:
        issues.append(f"prediction file contains {len(failed_rows)} counted failure row(s)")
    if any(row.get("deadline_exceeded") is True for row in rows):
        issues.append("prediction rows include case-deadline failures")

    query_ids = [str(value) for value in (manifest.get("query_ids") or [])]
    repetitions = int(manifest.get("repetitions") or 0)
    if not query_ids:
        issues.append("manifest query_ids are missing")
    if len(query_ids) != len(set(query_ids)):
        issues.append("manifest query_ids contain duplicates")
    expected_keys = {
        (query_id, repetition, path)
        for query_id in query_ids
        for repetition in range(1, repetitions + 1)
        for path in ("naive", "stream")
    }
    observed_keys = [
        (str(row.get("id")), int(row.get("repetition") or 0), str(row.get("path"))) for row in rows
    ]
    observed_counts = Counter(observed_keys)
    duplicate_keys = [key for key, count in observed_counts.items() if count > 1]
    observed_set = set(observed_keys)
    missing_keys = sorted(expected_keys - observed_set)
    extra_keys = sorted(observed_set - expected_keys)
    if duplicate_keys:
        issues.append(f"duplicate prediction keys: {duplicate_keys[:10]}")
    if missing_keys:
        issues.append(f"missing prediction keys: {missing_keys[:10]}")
    if extra_keys:
        issues.append(f"unexpected prediction keys: {extra_keys[:10]}")
    if int(manifest.get("prediction_rows_observed") or 0) != len(rows):
        issues.append("manifest prediction_rows_observed is inconsistent")

    evaluation_manifest_path = evaluation_manifest_path or gold_path.parent / "checksums.sha256"
    evaluation_checksums: dict[str, str] = {}
    try:
        evaluation_checksums = read_checksum_manifest(evaluation_manifest_path)
    except (OSError, ValueError) as exc:
        issues.append(f"evaluation checksum manifest is unavailable or invalid: {exc}")
    frozen_gold_hash = evaluation_checksums.get(gold_path.name)
    if not gold_path.is_file():
        issues.append("scorer gold file is missing")
    elif not frozen_gold_hash or sha256_file(gold_path) != frozen_gold_hash:
        issues.append("scorer gold checksum does not match the dataset checksum manifest")
    else:
        gold_ids = [str(row.get("id")) for row in read_jsonl(gold_path)]
        if len(gold_ids) != len(set(gold_ids)):
            issues.append("checksummed scorer gold contains duplicate IDs")
        prediction_ids = {str(row.get("id")) for row in rows}
        if not prediction_ids <= set(gold_ids):
            issues.append("scored predictions include IDs absent from gold")

    return {
        "status": "complete" if not issues else "missing_or_incomplete",
        "manifest": manifest_path.name,
        "predictions": predictions_path.name,
        "queries": str(manifest.get("queries") or ""),
        "gold": gold_path.name,
        "gold_sha256": frozen_gold_hash,
        "evaluation_manifest": evaluation_manifest_path.name,
        "expected_prediction_rows": len(expected_keys),
        "observed_prediction_rows": len(rows),
        "duplicate_keys": len(duplicate_keys),
        "missing_keys": len(missing_keys),
        "extra_keys": len(extra_keys),
        "issues": issues,
    }


def read_adjudications(path: Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    adjudications: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        label = str(row.get("label", ""))
        if label not in ADJUDICATION_LABELS:
            raise ValueError(f"invalid adjudication label: {label!r}")
        if not str(row.get("reviewer", "")).strip():
            raise ValueError("manual adjudication requires a reviewer")
        prediction_digest = str(row.get("prediction_sha256", ""))
        if len(prediction_digest) != 64 or any(
            character not in "0123456789abcdef" for character in prediction_digest
        ):
            raise ValueError("manual adjudication requires prediction_sha256")
        key = (str(row["id"]), int(row["repetition"]), str(row["path"]))
        if key in adjudications:
            raise ValueError(f"duplicate adjudication: {key}")
        adjudications[key] = row
    return adjudications


def prediction_sha256(row: dict[str, Any]) -> str:
    """Bind a human label to the exact prediction row that was reviewed."""

    payload = json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_adjudication_integrity(
    rows: list[dict[str, Any]],
    adjudications: dict[tuple[str, int, str], dict[str, Any]],
    path: Path | None,
) -> dict[str, Any]:
    expected_keys = {
        (str(row.get("id")), int(row.get("repetition") or 0), str(row.get("path")))
        for row in rows
        if "error" not in row
    }
    observed_keys = set(adjudications)
    missing = sorted(expected_keys - observed_keys)
    extra = sorted(observed_keys - expected_keys)
    rows_by_key = {
        (str(row.get("id")), int(row.get("repetition") or 0), str(row.get("path"))): row
        for row in rows
        if "error" not in row
    }
    mismatched_predictions = sorted(
        key
        for key in expected_keys & observed_keys
        if str(adjudications[key].get("prediction_sha256")) != prediction_sha256(rows_by_key[key])
    )
    issues: list[str] = []
    resolved_path: str | None = None
    digest: str | None = None
    if path is None:
        return {
            "status": "not_requested",
            "path": None,
            "sha256": None,
            "expected_keys": len(expected_keys),
            "observed_keys": 0,
            "missing_keys": len(expected_keys),
            "extra_keys": 0,
            "issues": [],
        }
    elif not path.is_file():
        issues.append("manual adjudication file is missing")
        resolved_path = path.name
    else:
        resolved_path = path.name
        digest = sha256_file(path)
    if missing:
        issues.append(f"missing adjudication keys: {missing[:10]}")
    if extra:
        issues.append(f"unexpected adjudication keys: {extra[:10]}")
    if mismatched_predictions:
        issues.append(
            f"adjudications do not match current prediction rows: {mismatched_predictions[:10]}"
        )
    return {
        "status": "complete" if not issues else "missing_or_incomplete",
        "path": resolved_path,
        "sha256": digest,
        "expected_keys": len(expected_keys),
        "observed_keys": len(observed_keys),
        "missing_keys": len(missing),
        "extra_keys": len(extra),
        "mismatched_prediction_hashes": len(mismatched_predictions),
        "issues": issues,
    }


def automatic_completeness(
    *,
    failures: int,
    run_integrity: dict[str, Any],
) -> bool:
    return failures == 0 and run_integrity.get("status") == "complete"


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).casefold().replace(",", "")
    return " ".join(re.sub(r"[^a-z0-9.%]+", " ", value).split())


def _contains_phrase(text: str, phrase: str) -> bool:
    return bool(phrase) and f" {phrase} " in f" {text} "


def abstention_detected(answer: str) -> bool:
    for raw_clause in CLAUSE_SPLIT_RE.split(answer):
        clause = normalize(raw_clause)
        if not clause or any(pattern.search(clause) for pattern in NON_ABSTENTION_PATTERNS):
            continue
        if any(pattern.search(clause) for pattern in ABSTENTION_PATTERNS):
            return True
    return False


def false_premise_rejection_detected(answer: str) -> bool:
    """Accept an abstention or an explicit correction of a false premise."""

    if abstention_detected(answer):
        return True
    normalized = normalize(answer)
    return (
        bool(re.match(r"^no(?:\s|[.%]|$)", normalized))
        or normalized.startswith("that premise is incorrect ")
        or normalized.startswith("the premise is incorrect ")
    )


def relation_contradicted(clause: str, question: str) -> bool:
    """Reject a candidate asserted with the opposite queried relation."""

    normalized_question = normalize(question)
    for first, second in RELATION_PAIRS:
        first_asked = _contains_phrase(normalized_question, first)
        second_asked = _contains_phrase(normalized_question, second)
        if first_asked == second_asked:
            continue
        expected, opposite = (first, second) if first_asked else (second, first)
        expected_negated = re.search(
            rf"\b(?:no|not|never)\b(?:\s+[a-z0-9]+){{0,2}}\s+{re.escape(expected)}\b",
            clause,
        )
        if expected_negated:
            return True
        if _contains_phrase(clause, opposite) and not _contains_phrase(clause, expected):
            return True
    return False


def candidate_asserted(answer: str, candidate: str, question: str = "") -> bool:
    expected = normalize(candidate)
    if not expected:
        return False
    escaped = re.escape(expected)
    for raw_clause in CLAUSE_SPLIT_RE.split(answer):
        clause = normalize(raw_clause)
        if not _contains_phrase(clause, expected) or abstention_detected(raw_clause):
            continue
        negated_before = re.search(
            rf"\b(?:no|not|never|cannot|can t|can not|could not)"
            rf"(?: [a-z0-9]+){{0,6}} {escaped}\b",
            clause,
        )
        negated_after = re.search(
            rf"\b{escaped}\b(?: [a-z0-9]+){{0,3}} not\b",
            clause,
        )
        mere_mention = re.search(
            rf"\b(?:evidence|corpus|question|sources?|documents?|pages?) "
            rf"(?:only )?(?:mentions?|contains?|lists?|references?)"
            rf"(?: [a-z0-9]+){{0,4}} {escaped}\b",
            clause,
        )
        if (
            not negated_before
            and not negated_after
            and not mere_mention
            and not relation_contradicted(clause, question)
        ):
            return True
    return False


def expected_hit(answer: str, gold: dict[str, Any], question: str = "") -> bool:
    is_false_premise = normalize(gold["answer"]) == "invalid question"
    if is_false_premise:
        return false_premise_rejection_detected(answer)
    candidates = [gold["answer"], *gold.get("alt_answers", [])]
    for candidate in candidates:
        parts = [part for part in candidate.split(",") if normalize(part)]
        if len(parts) > 1 and all(candidate_asserted(answer, part, question) for part in parts):
            return True
        if candidate_asserted(answer, candidate, question):
            return True
    return False


def cited_chunk_ids(answer: str) -> set[str]:
    return {match.group(1) for match in CITATION_RE.finditer(answer)}


def cited_doc_ids(chunk_ids: set[str]) -> set[str]:
    return {chunk_id.rsplit("::c", 1)[0] for chunk_id in chunk_ids}


def percentile(values: list[float], fraction: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    rank = fraction * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def stdev(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def word_bucket(row: dict[str, Any]) -> str:
    count = int(row.get("word_count") or len(str(row.get("query", "")).split()))
    if count <= 8:
        return "short_1_8"
    if count <= 12:
        return "medium_9_12"
    return "long_13_plus"


def annotate(
    rows: list[dict[str, Any]],
    gold_by_id: dict[str, dict[str, Any]],
    adjudications: dict[tuple[str, int, str], dict[str, Any]] | None = None,
) -> list[dict]:
    adjudications = adjudications or {}
    annotated = []
    for raw in rows:
        row = dict(raw)
        answer = str(row.get("answer", ""))
        gold = gold_by_id.get(str(row.get("id", "")))
        row["abstained"] = "error" not in row and abstention_detected(answer)
        row["false_premise_rejected"] = "error" not in row and false_premise_rejection_detected(
            answer
        )
        row["expected_hit"] = bool(
            gold and "error" not in row and expected_hit(answer, gold, str(row.get("query", "")))
        )
        cited_chunks = cited_chunk_ids(answer)
        source_chunks = {
            str(source.get("chunk_id"))
            for source in row.get("sources", [])
            if source.get("chunk_id")
        }
        valid_chunks = cited_chunks & source_chunks
        row["cited_chunk_ids"] = sorted(cited_chunks)
        row["valid_cited_chunk_ids"] = sorted(valid_chunks)
        row["invalid_cited_chunk_ids"] = sorted(cited_chunks - source_chunks)
        row["cited_doc_ids"] = sorted(cited_doc_ids(valid_chunks))
        row["has_citation_marker"] = bool(cited_chunks)
        row["has_valid_citation"] = bool(valid_chunks)
        row["cited_expected_hit"] = row["expected_hit"] and row["has_valid_citation"]
        supporting = set(gold.get("supporting_doc_ids", []) if gold else [])
        supporting.update(gold.get("acceptable_supporting_doc_ids", []) if gold else [])
        row["support_evaluable"] = bool(supporting)
        row["cites_supporting_doc"] = (
            bool(set(row["cited_doc_ids"]) & supporting) if supporting else None
        )
        row["supported_expected_hit"] = (
            row["expected_hit"] and row["cites_supporting_doc"] if supporting else None
        )
        adjudication = adjudications.get(
            (str(row.get("id", "")), int(row.get("repetition", 0)), str(row.get("path", "")))
        )
        row["manual_adjudication"] = (
            adjudication.get("label")
            if adjudication and adjudication.get("prediction_sha256") == prediction_sha256(raw)
            else None
        )
        row["word_count_bucket"] = word_bucket(row)
        annotated.append(row)
    return annotated


def path_summary(items: list[dict[str, Any]], gold_by_id: dict[str, dict[str, Any]]) -> dict:
    completed = [item for item in items if "error" not in item]
    ttft = [
        float(item["timing"]["submit_to_first_token_ms"])
        for item in completed
        if item.get("timing", {}).get("submit_to_first_token_ms") is not None
    ]
    totals = [float(item["timing"]["total_response_ms"]) for item in completed]
    retrieval = [float(item["timing"].get("retrieval_ms") or 0) for item in completed]
    lead = [
        float(item["timing"].get("accepted_retrieval_lead_at_commit_ms") or 0) for item in completed
    ]
    positive_lead = [value for value in lead if value > 0]
    candidate_lead = [
        float(item["timing"].get("accepted_candidate_retrieval_lead_ms") or 0) for item in completed
    ]
    false_premise = [
        item
        for item in items
        if gold_by_id.get(item["id"], {}).get("question_type") == "false_premise"
        or item.get("question_type") == "false_premise"
    ]
    false_premise_ids = {item["id"] for item in false_premise}
    answerable = [item for item in items if item["id"] not in false_premise_ids]
    calls = [float(item["usage"]["calls"]) for item in completed]
    controller_calls = [int(item.get("controller", {}).get("calls") or 0) for item in completed]
    retrieval_calls = [int(item.get("retrieval", {}).get("calls") or 0) for item in completed]
    dynamic_tool_calls = [len(item.get("tool_traces", [])) for item in completed]
    completed_dynamic_tool_calls = [
        sum(trace.get("status") == "completed" for trace in item.get("tool_traces", []))
        for item in completed
    ]
    post_commit_wall = [float(item.get("post_commit_wall_ms", 0)) for item in completed]
    diagnostics = [item.get("diagnostics", {}) for item in completed]
    cache_known = [item for item in diagnostics if item.get("cache_status") != "unknown"]
    support_evaluable = [item for item in items if item["support_evaluable"]]
    manually_adjudicated = [item for item in completed if item["manual_adjudication"]]
    manual_counts = {
        label: sum(item["manual_adjudication"] == label for item in manually_adjudicated)
        for label in sorted(ADJUDICATION_LABELS)
    }
    return {
        "outputs": len(items),
        "completed": len(completed),
        "failures": len(items) - len(completed),
        "completion_rate": len(completed) / len(items) if items else None,
        "unique_queries": len({item["id"] for item in items}),
        "expected_answer_accuracy": (
            sum(bool(item["expected_hit"]) for item in items) / len(items) if items else None
        ),
        "citation_marker_rate": (
            sum(bool(item["has_citation_marker"]) for item in items) / len(items) if items else None
        ),
        "valid_citation_rate": (
            sum(bool(item["has_valid_citation"]) for item in items) / len(items) if items else None
        ),
        "cited_expected_answer_rate": (
            sum(bool(item["cited_expected_hit"]) for item in items) / len(items) if items else None
        ),
        "support_evaluable_outputs": len(support_evaluable),
        "supporting_doc_citation_rate": (
            sum(bool(item["cites_supporting_doc"]) for item in support_evaluable)
            / len(support_evaluable)
            if support_evaluable
            else None
        ),
        "supported_expected_answer_rate": (
            sum(bool(item["supported_expected_hit"]) for item in support_evaluable)
            / len(support_evaluable)
            if support_evaluable
            else None
        ),
        "manual_adjudication": {
            "completed_outputs": len(completed),
            "adjudicated_outputs": len(manually_adjudicated),
            "coverage": len(manually_adjudicated) / len(completed) if completed else None,
            "labels": manual_counts,
            "perfect_or_acceptable_rate": (
                (manual_counts["perfect"] + manual_counts["acceptable"]) / len(manually_adjudicated)
                if manually_adjudicated
                else None
            ),
        },
        "false_premise_rejection_rate": (
            sum(bool(item["false_premise_rejected"]) for item in false_premise) / len(false_premise)
            if false_premise
            else None
        ),
        "answerable_abstention_rate": (
            sum(bool(item["abstained"]) for item in answerable) / len(answerable)
            if answerable
            else None
        ),
        "median_ttft_ms": median(ttft),
        "mean_ttft_ms": mean(ttft),
        "stdev_ttft_ms": stdev(ttft),
        "p95_ttft_ms": percentile(ttft, 0.95),
        "median_total_ms": median(totals),
        "p95_total_ms": percentile(totals, 0.95),
        "median_retrieval_ms": median(retrieval),
        "median_retrieval_lead_ms": median(lead),
        "median_positive_retrieval_lead_ms": median(positive_lead),
        "median_candidate_retrieval_lead_ms": median(candidate_lead),
        "p95_candidate_retrieval_lead_ms": percentile(candidate_lead, 0.95),
        "speculative_reuse_rate": (
            sum(bool(item.get("speculative_reuse")) for item in diagnostics) / len(diagnostics)
            if diagnostics
            else None
        ),
        "inflight_postcommit_overlap_rate": (
            sum(bool(item.get("inflight_postcommit_overlap")) for item in diagnostics)
            / len(diagnostics)
            if diagnostics
            else None
        ),
        "commit_fallback_rate": (
            sum(bool(item.get("commit_fallback")) for item in diagnostics) / len(diagnostics)
            if diagnostics
            else None
        ),
        "known_cache_hit_rate": (
            sum(item.get("cache_status") in {"hit", "mixed"} for item in cache_known)
            / len(cache_known)
            if cache_known
            else None
        ),
        "mean_input_tokens": mean([float(item["usage"]["input_tokens"]) for item in completed]),
        "mean_output_tokens": mean([float(item["usage"]["output_tokens"]) for item in completed]),
        "mean_usage_accounted_model_calls": mean(calls),
        "usage_accounted_model_calls": sum(calls),
        "mean_controller_calls": mean([float(value) for value in controller_calls]),
        "total_controller_calls": sum(controller_calls),
        "mean_retrieval_calls": mean([float(value) for value in retrieval_calls]),
        "total_retrieval_calls": sum(retrieval_calls),
        "mean_dynamic_tool_calls": mean([float(value) for value in dynamic_tool_calls]),
        "total_dynamic_tool_calls": sum(dynamic_tool_calls),
        "completed_dynamic_tool_calls": sum(completed_dynamic_tool_calls),
        "dynamic_tool_call_rate": (
            sum(value > 0 for value in dynamic_tool_calls) / len(dynamic_tool_calls)
            if dynamic_tool_calls
            else None
        ),
        "post_commit_throughput_queries_per_minute": (
            len(completed) * 60_000 / sum(post_commit_wall)
            if completed and sum(post_commit_wall) > 0
            else None
        ),
    }


def paired_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["id"], int(row["repetition"]))][row["path"]] = row
    complete_pairs = [
        pair
        for pair in grouped.values()
        if {"naive", "stream"} <= pair.keys()
        and "error" not in pair["naive"]
        and "error" not in pair["stream"]
    ]

    def delta(getter: Callable[[dict[str, Any]], float]) -> list[float]:
        return [getter(pair["stream"]) - getter(pair["naive"]) for pair in complete_pairs]

    ttft_delta = delta(lambda item: float(item["timing"]["submit_to_first_token_ms"]))
    total_delta = delta(lambda item: float(item["timing"]["total_response_ms"]))
    call_delta = delta(lambda item: float(item["usage"]["calls"]))
    relative_ttft = [
        (
            float(pair["stream"]["timing"]["submit_to_first_token_ms"])
            - float(pair["naive"]["timing"]["submit_to_first_token_ms"])
        )
        / float(pair["naive"]["timing"]["submit_to_first_token_ms"])
        for pair in complete_pairs
        if float(pair["naive"]["timing"]["submit_to_first_token_ms"]) > 0
    ]
    accuracy_delta = [
        int(pair["stream"]["expected_hit"]) - int(pair["naive"]["expected_hit"])
        for pair in complete_pairs
    ]
    return {
        "candidate_pairs": len(grouped),
        "completed_pairs": len(complete_pairs),
        "stream_ttft_win_rate": (
            sum(value < 0 for value in ttft_delta) / len(ttft_delta) if ttft_delta else None
        ),
        "median_stream_minus_naive_ttft_ms": median(ttft_delta),
        "p95_stream_minus_naive_ttft_ms": percentile(ttft_delta, 0.95),
        "median_stream_minus_naive_ttft_percent": (
            median(relative_ttft) * 100 if relative_ttft else None
        ),
        "median_stream_minus_naive_total_ms": median(total_delta),
        "mean_stream_minus_naive_usage_accounted_calls": mean(call_delta),
        "mean_stream_minus_naive_accuracy": mean(accuracy_delta),
        "accuracy_pairs": {
            "stream_only_correct": sum(value > 0 for value in accuracy_delta),
            "naive_only_correct": sum(value < 0 for value in accuracy_delta),
            "same_outcome": sum(value == 0 for value in accuracy_delta),
        },
    }


def grouped_path_summaries(
    rows: list[dict[str, Any]],
    key: Callable[[dict[str, Any]], str],
    gold_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    result = {}
    for name, items in sorted(groups.items()):
        by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            by_path[item["path"]].append(item)
        result[name] = {
            path: path_summary(path_items, gold_by_id)
            for path, path_items in sorted(by_path.items())
        }
    return result


def grouped_paired_summaries(
    rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], str]
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return {name: paired_summary(items) for name, items in sorted(groups.items())}


def summarize(
    rows: list[dict[str, Any]],
    gold_by_id: dict[str, dict[str, Any]],
    adjudications: dict[tuple[str, int, str], dict[str, Any]] | None = None,
    run_integrity: dict[str, Any] | None = None,
    adjudication_integrity: dict[str, Any] | None = None,
) -> dict:
    rows = annotate(rows, gold_by_id, adjudications)
    by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_path[row["path"]].append(row)
    stream_rows = [row for row in rows if row["path"] == "stream"]
    stream_stage_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stream_cache_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stream_rows:
        stream_stage_groups[row.get("diagnostics", {}).get("evidence_stage", "unknown")].append(row)
        stream_cache_groups[row.get("diagnostics", {}).get("cache_status", "unknown")].append(row)
    completed = [row for row in rows if "error" not in row]
    adjudicated = [row for row in completed if row["manual_adjudication"]]
    failures = len(rows) - len(completed)
    run_integrity = run_integrity or {
        "status": "missing_or_incomplete",
        "issues": ["benchmark manifest was not validated"],
    }
    adjudication_integrity = adjudication_integrity or {
        "status": "not_requested",
        "path": None,
        "sha256": None,
        "issues": [],
    }
    manual_requested = adjudication_integrity.get("status") != "not_requested"
    manual_gate_complete = (
        manual_requested
        and bool(completed)
        and len(adjudicated) == len(completed)
        and adjudication_integrity.get("status") == "complete"
    )
    final_completeness = automatic_completeness(
        failures=failures,
        run_integrity=run_integrity,
    )
    return {
        "schema_version": 4,
        "scorer_sha256": sha256_file(Path(__file__).resolve()),
        "metric_notes": {
            "latency_delta_sign": "negative stream-minus-naive values favor StreamRAG",
            "accuracy": (
                "normalized expected-answer/alias containment with a conservative queried-"
                "relation contradiction guard; failures score incorrect; semantic correctness "
                "still requires prediction-bound manual adjudication"
            ),
            "citation_marker_rate": (
                "checks citation syntax only, not whether the source supports the claim"
            ),
            "supporting_doc_citation_rate": (
                "an exact cited chunk exists in the prediction's retrieved sources and resolves "
                "to a scorer-only gold supporting/acceptable document ID"
            ),
            "final_grounding_gate": (
                "automatic scoring is the required quick-benchmark gate; optional manual review "
                "uses perfect/acceptable/missing/incorrect adjudication"
            ),
            "usage_accounted_model_calls": (
                "counts calls represented in returned provider usage; controller attempts "
                "without returned usage are reported separately and are not silently counted"
            ),
            "throughput": "uses post-commit wall time and excludes simulated typing sleeps",
            "candidate_retrieval_lead": (
                "accepted_candidate_retrieval_lead_ms measures how much retrieval work for "
                "the ultimately accepted candidate completed before Send. It is retrieval "
                "headroom, not accepted/safe evidence lead and not measured TTFT saved"
            ),
            "stabilization_class": (
                "candidate metadata is heuristic/manual-review, assigned without path outputs"
            ),
        },
        "paths": {path: path_summary(items, gold_by_id) for path, items in sorted(by_path.items())},
        "manual_grounding_gate": {
            "status": (
                "complete"
                if manual_gate_complete
                else "missing_or_incomplete"
                if manual_requested
                else "not_requested"
            ),
            "required_for_automatic_benchmark": False,
            "completed_outputs": len(completed),
            "adjudicated_outputs": len(adjudicated),
            "labels": sorted(ADJUDICATION_LABELS),
        },
        "run_integrity_gate": run_integrity,
        "adjudication_integrity_gate": adjudication_integrity,
        "final_completeness_gate": {
            "status": "complete" if final_completeness else "missing_or_incomplete",
            "outputs": len(rows),
            "completed_outputs": len(completed),
            "failures": failures,
            "adjudicated_outputs": len(adjudicated),
            "requirement": (
                "validated gold-blind run and offline evaluation freeze with zero failures; "
                "manual adjudication is an independent gate"
            ),
        },
        "paired": paired_summary(rows),
        "strata": {
            "by_repetition": grouped_path_summaries(
                rows, lambda row: str(row["repetition"]), gold_by_id
            ),
            "by_path_order_position": grouped_path_summaries(
                rows, lambda row: str(row.get("path_order_position", "unknown")), gold_by_id
            ),
            "by_word_count_bucket": grouped_path_summaries(
                rows, lambda row: row["word_count_bucket"], gold_by_id
            ),
            "by_stabilization_class": grouped_path_summaries(
                rows,
                lambda row: str(row.get("stabilization_class", "unreviewed")),
                gold_by_id,
            ),
            "paired_by_stabilization_class": grouped_paired_summaries(
                rows, lambda row: str(row.get("stabilization_class", "unreviewed"))
            ),
            "stream_by_evidence_stage": {
                stage: path_summary(items, gold_by_id)
                for stage, items in sorted(stream_stage_groups.items())
            },
            "stream_by_cache_status": {
                status: path_summary(items, gold_by_id)
                for status, items in sorted(stream_cache_groups.items())
            },
        },
    }


def display(value: float | None, suffix: str = "", digits: int = 0) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}{suffix}"


def percent(value: float | None) -> float | None:
    return value * 100 if value is not None else None


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Benchmark summary",
        "",
        "Generated from frozen predictions and scorer-only golds. Negative paired latency",
        "deltas favor StreamRAG. Automatic expected-answer/alias matching is a proxy, not a",
        "semantic correctness judgment. Support additionally requires a valid exact-chunk",
        "citation resolving to an acceptable gold document. Manual adjudication is optional.",
        "",
        "| Path | Completed | Failures | Automatic match proxy | Support+valid citation | "
        "Manual semantic P/A | "
        "Median TTFT | p95 TTFT | Median total | Pre-Send reuse | In-flight overlap | "
        "Fallback | Usage-accounted calls/output |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for path, values in summary["paths"].items():
        accuracy = display(percent(values["expected_answer_accuracy"]), "%", 1)
        supported_correct = display(percent(values["supported_expected_answer_rate"]), "%", 1)
        manual_correct = display(
            percent(values["manual_adjudication"]["perfect_or_acceptable_rate"]), "%", 1
        )
        reuse = display(percent(values["speculative_reuse_rate"]), "%", 1)
        inflight = display(percent(values["inflight_postcommit_overlap_rate"]), "%", 1)
        fallback = display(percent(values["commit_fallback_rate"]), "%", 1)
        lines.append(
            f"| {path} | {values['completed']} | {values['failures']} | "
            f"{accuracy} | {supported_correct} | {manual_correct} | "
            f"{display(values['median_ttft_ms'], ' ms')} | "
            f"{display(values['p95_ttft_ms'], ' ms')} | "
            f"{display(values['median_total_ms'], ' ms')} | "
            f"{reuse} | {inflight} | {fallback} | "
            f"{display(values['mean_usage_accounted_model_calls'], '', 2)} |"
        )
    gate = summary["manual_grounding_gate"]
    integrity_gate = summary["run_integrity_gate"]
    adjudication_gate = summary["adjudication_integrity_gate"]
    final_gate = summary["final_completeness_gate"]
    lines.extend(
        [
            "",
            f"Manual grounding gate: **{gate['status']}** "
            f"({gate['adjudicated_outputs']}/{gate['completed_outputs']} completed outputs).",
            f"Run integrity gate: **{integrity_gate['status']}** "
            f"({len(integrity_gate.get('issues', []))} issue(s)).",
            f"Adjudication integrity gate: **{adjudication_gate['status']}** "
            f"(SHA-256: {adjudication_gate.get('sha256') or 'missing'}).",
            f"Final completeness gate: **{final_gate['status']}** "
            f"({final_gate['failures']} failures; zero required).",
        ]
    )
    paired = summary["paired"]
    win_rate = display(percent(paired["stream_ttft_win_rate"]), "%", 1)
    paired_ttft = display(paired["median_stream_minus_naive_ttft_ms"], " ms")
    relative_ttft = display(paired["median_stream_minus_naive_ttft_percent"], "%", 1)
    paired_total = display(paired["median_stream_minus_naive_total_ms"], " ms")
    lines.extend(
        [
            "",
            "## Paired StreamRAG deltas",
            "",
            f"- Completed A/B pairs: {paired['completed_pairs']} / {paired['candidate_pairs']}",
            f"- Stream TTFT win rate: {win_rate}",
            f"- Median Stream minus Naive TTFT: {paired_ttft}",
            f"- Median relative TTFT delta: {relative_ttft}",
            f"- Median Stream minus Naive total time: {paired_total}",
            f"- Automatic-proxy discordance: {paired['accuracy_pairs']}",
            "",
            "## Stream evidence stages",
            "",
            "Candidate retrieval lead is work moved before Send for the ultimately accepted",
            "candidate. It is not accepted/safe evidence lead and not measured TTFT saved.",
            "",
            "| Stage | Runs | Median TTFT | Median accepted-safe lead | "
            "Median / p95 candidate retrieval headroom | Automatic match proxy |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for stage, values in summary["strata"]["stream_by_evidence_stage"].items():
        stage_accuracy = display(percent(values["expected_answer_accuracy"]), "%", 1)
        lines.append(
            f"| {stage} | {values['outputs']} | {display(values['median_ttft_ms'], ' ms')} | "
            f"{display(values['median_retrieval_lead_ms'], ' ms')} | "
            f"{display(values['median_candidate_retrieval_lead_ms'], ' ms')} / "
            f"{display(values['p95_candidate_retrieval_lead_ms'], ' ms')} | "
            f"{stage_accuracy} |"
        )
    lines.extend(
        [
            "",
            "## Typed stabilization strata",
            "",
            "Candidate classes are heuristic/manual-review labels assigned without seeing path "
            "outputs; they are not measured stabilization points.",
            "",
            "| Class | Pairs | Naive TTFT | Stream TTFT | Stream reuse | "
            "Extra usage-accounted calls |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    path_strata = summary["strata"]["by_stabilization_class"]
    pair_strata = summary["strata"]["paired_by_stabilization_class"]
    for class_name, paths in path_strata.items():
        naive = paths.get("naive", {})
        stream = paths.get("stream", {})
        pairs = pair_strata[class_name]
        stream_reuse = display(percent(stream.get("speculative_reuse_rate")), "%", 1)
        extra_calls = display(pairs.get("mean_stream_minus_naive_usage_accounted_calls"), "", 2)
        lines.append(
            f"| {class_name} | {pairs['completed_pairs']} | "
            f"{display(naive.get('median_ttft_ms'), ' ms')} | "
            f"{display(stream.get('median_ttft_ms'), ' ms')} | {stream_reuse} | "
            f"{extra_calls} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score saved paired A/B predictions")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "comparison" / "benchmark" / "results" / "predictions.jsonl",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        required=True,
        help="scorer-only gold, supplied explicitly to this offline process",
    )
    parser.add_argument(
        "--evaluation-manifest",
        type=Path,
        help="full frozen checksums.sha256; defaults to the gold file's directory",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="run manifest; defaults to predictions with .manifest.json suffix",
    )
    parser.add_argument(
        "--adjudications",
        type=Path,
        help=(
            "manual JSONL keyed by id/repetition/path with label, reviewer, and the exact "
            "prediction_sha256 returned by prediction_sha256(row)"
        ),
    )
    parser.add_argument(
        "--require-manual-adjudication",
        action="store_true",
        help=("add a strict manual-review extension to the automatic completeness gate"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "comparison" / "benchmark" / "results" / "summary.json",
    )
    args = parser.parse_args()
    manifest_path = args.manifest or args.predictions.with_suffix(".manifest.json")
    rows = read_jsonl(args.predictions)
    gold = read_jsonl(args.gold)
    adjudications = read_adjudications(args.adjudications) if args.adjudications else {}
    run_integrity = validate_run_integrity(
        rows,
        args.predictions,
        manifest_path,
        args.gold,
        args.evaluation_manifest,
    )
    adjudication_integrity = validate_adjudication_integrity(
        rows, adjudications, args.adjudications
    )
    summary = summarize(
        rows,
        {item["id"]: item for item in gold},
        adjudications,
        run_integrity,
        adjudication_integrity,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.output.with_suffix(".md").write_text(markdown(summary), encoding="utf-8")
    if args.require_manual_adjudication and (
        summary["final_completeness_gate"]["status"] != "complete"
        or summary["manual_grounding_gate"]["status"] != "complete"
        or summary["adjudication_integrity_gate"]["status"] != "complete"
    ):
        raise SystemExit(
            "manual-review extension gate failed: require an automatically complete run and "
            "exact content-addressed adjudication for every completed output"
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
