import chromadb
from chromadb.config import Settings
from utils.config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME
)

def create_new_index(index_name=CHROMA_COLLECTION_NAME, dimension=768):
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(allow_reset=True)
    )
    collection = client.get_or_create_collection(
        name=index_name,
        metadata={"hnsw:space": "cosine", "dimension": dimension}
    )
    print(f"Connected to Chroma collection: {index_name} at {CHROMA_PERSIST_DIR}")
    return collection