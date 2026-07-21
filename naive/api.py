from naive.config import settings
from naive.path import NaiveRagPath
from shared.api.factory import create_app
from shared.data.vector_store import QdrantVectorStore


def _path_factory(current_settings, store: QdrantVectorStore) -> NaiveRagPath:
    return NaiveRagPath(current_settings, store)


app = create_app(
    implementation="naive",
    api_title="Naive RAG API",
    settings_provider=lambda: settings,
    path_factory=_path_factory,
    supports_snapshots=False,
)
