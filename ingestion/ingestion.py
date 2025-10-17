import os
import uuid
import glob
import qdrant_client
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from app.utils.config import settings
from tqdm import tqdm

QDRANT_URL = settings.QDRANT_URL
COLLECTION_NAME = "documents"
DOCS_DIR = "./data"
EMBED_MODEL = settings.EMBEDDING_MODEL
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def read_documents_from_directory(directory: str):
    """Load all .txt, .md files from directory."""
    documents = []
    for path in glob.glob(os.path.join(directory, "**/*"), recursive=True):
        if path.endswith((".txt", ".md")):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
                if text:
                    documents.append({"id": str(uuid.uuid4()), "path": path, "text": text})
    return documents


def chunk_text(text, chunk_size=500, overlap=50):
    """Split large text into overlapping chunks for better embedding context."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def create_collection_if_not_exists(client: qdrant_client.QdrantClient, collection_name: str, vector_size: int):
    """Ensure collection exists."""
    collections = [c.name for c in client.get_collections().collections]
    if collection_name not in collections:
        print(f"Creating new collection '{collection_name}' with vector size {vector_size}...")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
        )
    else:
        print(f"Collection '{collection_name}' already exists.")


def main():
    # Initialize embedding model and Qdrant client
    print("Connecting to Qdrant...")
    client = qdrant_client.QdrantClient(url=QDRANT_URL)

    print("Loading SentenceTransformer model...")
    model = SentenceTransformer(EMBED_MODEL)

    vector_size = model.get_sentence_embedding_dimension()
    create_collection_if_not_exists(client, COLLECTION_NAME, vector_size)

    print(f"Reading documents from {DOCS_DIR}...")
    documents = read_documents_from_directory(DOCS_DIR)

    print(f"Found {len(documents)} documents. Generating embeddings and uploading to Qdrant...")

    all_points = []

    for doc in tqdm(documents):
        chunks = chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        embeddings = model.encode(chunks, show_progress_bar=False)

        for i, emb in enumerate(embeddings):
            all_points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb.tolist(),
                    payload={
                        "source_id": doc["id"],
                        "path": doc["path"],
                        "chunk_id": i,
                        "text": chunks[i],
                    }
                )
            )

    if all_points:
        client.upsert(collection_name=COLLECTION_NAME, points=all_points)
        print(f"Successfully inserted {len(all_points)} chunks into Qdrant collection '{COLLECTION_NAME}'!")
    else:
        print("No documents found to insert.")


if __name__ == "__main__":
    main()
