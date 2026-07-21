# Dataset and human review

## Current status

`candidate_pending_human_review` is a metadata value, not a function or file.
Its authoritative location is
`data/crag_eval/dataset_summary.json` at
`.selection.approval_status`; the leakage audit records the same state. The
committed set is a candidate dataset, so services load it with
`ALLOW_UNREVIEWED_DATASET=1`. The shared service gate still checks for approval
or that override, but there is no sealed/final-run ceremony and no
`approved_frozen` freeze transition in the shipped tooling.

| Item | Value |
|---|---|
| Source | pinned CRAG Task 1/2 development release |
| Knowledge base | 250 complete cleaned documents |
| Vector index | exactly 1,000 Qdrant points per path |
| Chunking | 400 tokens with 50-token overlap |
| Embeddings | `text-embedding-3-large`, 3,072 dimensions |
| Development questions | 5 visible rows |
| Final questions | 10 held-out rows |

## What is embedded

The only source collection sent to the embedding API is:

[`data/crag_eval/documents.jsonl.bz2`](../data/crag_eval/documents.jsonl.bz2)

It is compressed JSON Lines: 250 lines, one complete document per line. Each row
contains `doc_id`, `title`, `url`, `text`, `domain`, source timestamps, a snippet,
and a content checksum. Questions, answers, aliases, split labels, and gold are
not stored in those rows.

Inspect the corpus without modifying it:

```bash
# Confirm the document count.
bzcat data/crag_eval/documents.jsonl.bz2 | wc -l

# Inspect rows interactively.
bzcat data/crag_eval/documents.jsonl.bz2 | jq -c . | less

# Inspect one audited support document.
bzcat data/crag_eval/documents.jsonl.bz2 \
  | jq -c 'select(.doc_id == "crag-global-9d22ffbe3ca22f9a9858bd6e")'
```

`shared/data/crag.py` verifies and loads these rows. `chunk_documents` performs
the deterministic split, and each API's `/v1/data/sync` endpoint embeds and
upserts the resulting 1,000 chunks into its own Qdrant service.

## File roles

| File | Consumer |
|---|---|
| `documents.jsonl.bz2` | both indexers; the vector-search knowledge base |
| `dev_queries.jsonl` | development queries (optional `make benchmark` input) |
| `test_queries.jsonl` | the benchmark runner (`make benchmark`) |
| `test_gold.jsonl` | offline scorer only; never an API or inference input |
| `dataset_summary.json` | counts, selection method, input contract, approval state |
| `selection_manifest.json` | deterministic source selection and provenance |
| `leakage_audit.json` | verifies separation between corpus, questions, and gold |
| `REVIEW_SHEET.md` | the human review checklist for all 15 questions |
| `checksums.sha256` | binds every integrity-sensitive dataset file |

## How to review the candidate

Open
[`data/crag_eval/REVIEW_SHEET.md`](../data/crag_eval/REVIEW_SHEET.md). It contains
all 5 development rows and all 10 test rows. Review metadata and source evidence
only; you do not need to run either implementation to review the test questions.

For every row:

1. Confirm the wording and time anchor are unambiguous.
2. Confirm the expected answer and aliases.
3. Open the recorded source pages or locate the audited `doc_id` in the committed
   corpus and confirm that it supports the answer or intended abstention.
4. Check that the candidate stabilization class is plausible without looking at
   Naive or StreamRAG outputs.
5. Confirm the development/test split role.
6. Mark all six checkboxes for that row only after those checks pass.

If a row is wrong, correct or replace it before approval and regenerate every
affected checksum. Do not approve around a known defect.

`REVIEW_SHEET.md` is a human inspection aid only. Completing the sheet records
the review; it does not change any status or trigger any automated step.

## Integrity and leakage boundary

The corpus is checksum-bound by `checksums.sha256`. Each service verifies those
checksums when it loads the dataset (`capture_dataset_snapshot`), covering all
bound files, the 250 source rows, and deterministic chunking to the 1,000-point
target, so no separate verification command is required. There is a single
benchmark flow: the runner (`make benchmark`) writes and finalizes predictions
before the offline scorer reads any gold, and `make score` binds `test_gold.jsonl`
through `data/crag_eval/checksums.sha256`.

## Provenance

The corpus keeps complete cleaned pages; documents are never shortened. Fifteen
preselected evidence documents pending human review cover the questions, and 235
deterministic distractors make retrieval non-trivial. Normal reproduction uses
the committed compressed corpus and does not download the 705 MiB upstream
release.

The construction tooling that built this corpus from Meta's pinned upstream
release (download, clean, select, chunk, and verify) was intentionally removed to
keep the shipped surface minimal. The committed `data/crag_eval/` corpus is
checksum-bound and sufficient to run and reproduce the benchmark from a clean
checkout; the original generation scripts remain in the Git history.

The source is Meta's
[CRAG Task 1/2 development release](https://github.com/facebookresearch/CRAG),
licensed CC BY-NC 4.0. Source IDs and URLs remain in each row for audit and
attribution.
