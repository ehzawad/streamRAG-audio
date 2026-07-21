# CRAG evaluation data

This directory is the complete, committed dataset used by both application paths.
It contains 5 development questions, 10 sealed test questions, and
250 complete CRAG pages. Only `documents.jsonl.bz2` is embedded: its
full pages produce 1,000 chunks under the fixed indexing contract. No page
is shortened.

## Status

`candidate_pending_human_review` means the dataset is proposed but not frozen.
Review every item and the corpus checklist in `REVIEW_SHEET.md` without viewing
either path's predictions. The final unseen benchmark must wait until corrections
are resolved and the status is deliberately changed to `approved_frozen`.

## Files

- `documents.jsonl.bz2`: retrievable full-page corpus; contains no query or gold wrappers.
- `dev_queries.jsonl`: visible questions for development and smoke benchmarks.
- `test_queries.jsonl`: sealed questions without answers.
- `test_gold.jsonl`: scorer-only expected answers and accepted evidence IDs.
- `dataset_summary.json`: corpus statistics, input contract, and approval status.
- `selection_manifest.json`: provenance, split roles, and support-document mapping.
- `leakage_audit.json`: checks that labels and wrapper fields are not retrievable.
- `REVIEW_SHEET.md`: human QA checklist for questions, evidence, classes, and corpus.
- `checksums.sha256`: integrity hashes for every committed dataset artifact.

The selection is fixed independently of Naive RAG and StreamRAG outputs.
Development items may be debugged; sealed test labels must remain scorer-only.
See `../../docs/DATASET.md` for the construction and approval policy.
