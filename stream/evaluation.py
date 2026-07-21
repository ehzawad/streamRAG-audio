from shared.metrics import COMMON_EVALUATION_METRICS

EVALUATION_METRICS = {
    "common": COMMON_EVALUATION_METRICS,
    "path_specific": (
        "timing.accepted_retrieval_lead_at_commit_ms",
        "timing.accepted_candidate_retrieval_lead_ms",
        "controller.calls",
        "controller.timeouts",
        "controller.failures",
        "retrieval.raw_candidate_calls",
        "retrieval.settled_draft_retrievals",
        "reuse.mode",
        "reuse.evidence_reuses",
        "reuse.evidence_revalidations",
        "reuse.stale_discards",
        "reuse.retrieval_cancellations",
        "reuse.commit_fallbacks",
    ),
}
