from shared.models import RETRIEVAL_QUERY_MAX_CHARS


def bounded_retrieval_query(draft: str) -> str:
    """Bound exact or speculative retrieval text without losing both ends."""

    query = draft.strip()
    if len(query) <= RETRIEVAL_QUERY_MAX_CHARS:
        return query
    separator = "\n...\n"
    remaining = RETRIEVAL_QUERY_MAX_CHARS - len(separator)
    prefix_chars = remaining // 2
    suffix_chars = remaining - prefix_chars
    return f"{query[:prefix_chars].rstrip()}{separator}{query[-suffix_chars:].lstrip()}"


def contextual_retrieval_query(draft: str, conversation_context: str = "") -> str:
    """Rewrite a follow-up into a self-contained retrieval query.

    A bare follow-up ("where did he work before?") embeds poorly because the
    entity lives only in prior turns. Rather than an extra model call, we lead
    with the current text (the primary intent) and append a bounded tail of
    compact conversational state so the embedding can resolve pronouns and
    ellipsis. First turns (no context) are identical to ``bounded_retrieval_query``.
    This rewrites the *query text only*; it never carries prior retrieved
    passages, which stay turn-local.
    """
    question = draft.strip()
    context = conversation_context.strip()
    if not context:
        return bounded_retrieval_query(question)
    # Reserve most of the budget for the current question; use the remainder for
    # the most recent conversational context (entities resolve from the tail).
    q_budget = min(len(question), (RETRIEVAL_QUERY_MAX_CHARS * 3) // 4)
    question = question[:q_budget].rstrip()
    ctx_budget = RETRIEVAL_QUERY_MAX_CHARS - len(question) - len("\n\ncontext: ")
    if ctx_budget <= 0:
        return question
    return f"{question}\n\ncontext: {context[-ctx_budget:].lstrip()}"
