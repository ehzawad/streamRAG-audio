from shared.metrics import COMMON_EVALUATION_METRICS

EVALUATION_METRICS = {
    "common": COMMON_EVALUATION_METRICS,
    "path_specific": (
        "retrieval.started_ms",
        "retrieval.ready_ms",
    ),
}
