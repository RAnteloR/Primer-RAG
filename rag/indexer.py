"""Vectoriza los chunks y los guarda en ChromaDB.

Uso:  python -m rag.indexer --chunk-size 900 --overlap 150
"""
import argparse

import chromadb
from sentence_transformers import SentenceTransformer

from . import config
from .chunking import chunkenizar


def get_embedder() -> SentenceTransformer:
    modelo = SentenceTransformer(config.EMBEDDING_MODEL, device=config.EMBEDDING_DEVICE)
    modelo.max_seq_length = config.EMBEDDING_MAX_SEQ_LENGTH
    return modelo


def build_index(chunk_size: int, chunk_overlap: int, embedder: SentenceTransformer | None = None) -> int:
    chunks = chunkenizar(config.DOCS_DIR, chunk_size, chunk_overlap)
    if not chunks:
        return 0
    embedder = embedder or get_embedder()
    textos = [c.page_content for c in chunks]
    vectores = embedder.encode(textos, normalize_embeddings=True, batch_size=16, show_progress_bar=True)

    cliente = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    nombre = config.collection_name(chunk_size, chunk_overlap)
    if nombre in [c.name for c in cliente.list_collections()]:
        cliente.delete_collection(nombre)
    col = cliente.create_collection(name=nombre, metadata={"hnsw:space": "cosine"})

    def meta(c):
        m = {"source": c.metadata["source"], "doc_key": c.metadata["doc_key"],
             "start_index": c.metadata.get("start_index", 0)}
        if "page" in c.metadata:
            m["page"] = c.metadata["page"]
        return m

    LOTE = 1000
    for i in range(0, len(chunks), LOTE):
        col.add(
            ids=[c.metadata["chunk_id"] for c in chunks[i:i + LOTE]],
            embeddings=vectores[i:i + LOTE].tolist(),
            documents=textos[i:i + LOTE],
            metadatas=[meta(c) for c in chunks[i:i + LOTE]],
        )
    print(f"Colección '{nombre}': {col.count()} chunks guardados")
    return col.count()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    args = ap.parse_args()
    build_index(args.chunk_size, args.overlap)
