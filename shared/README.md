# Shared infrastructure

`shared/` is infrastructure used by both RAG paths. It is not a third product and
does not choose either path's retrieval schedule.

It owns:

- API lifecycle, events, and persistence;
- settings, dataset checks, and reproducibility fingerprints;
- CRAG checksums, chunking, embeddings, index readiness, and search;
- grounded answers, local-corpus tooling, chat memory, and usage/cost accounting;
- schemas and the telemetry contract.

Naive RAG owns committed-text retrieval. StreamRAG owns draft analysis,
speculation, and reuse. `frontend/` owns the GUI, while `comparison/` owns replay
and scoring. `shared/` imports none of those packages and is not runnable alone.
