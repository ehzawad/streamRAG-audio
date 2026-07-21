from shared.api.factory import create_app
from shared.data.vector_store import QdrantVectorStore
from stream.config import StreamSettings, settings
from stream.path import StreamRagPath
from stream.trigger import ModelTrigger


def _path_factory(current_settings: StreamSettings, store: QdrantVectorStore) -> StreamRagPath:
    return StreamRagPath(
        current_settings,
        store,
        ModelTrigger(current_settings),
    )


app = create_app(
    implementation="stream",
    api_title="StreamRAG API",
    settings_provider=lambda: settings,
    path_factory=_path_factory,
    supports_snapshots=True,
)
