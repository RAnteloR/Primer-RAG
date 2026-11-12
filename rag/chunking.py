"""Chunking recursivo de la carpeta de documentos."""
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .loaders import load_file

logger = logging.getLogger(__name__)


def chunkenizar(docs_dir: Path = config.DOCS_DIR,
                chunk_size: int = config.CHUNK_SIZE,
                chunk_overlap: int = config.CHUNK_OVERLAP) -> List[Document]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap debe ser menor que chunk_size")

    files = sorted(
        p for p in Path(docs_dir).resolve().rglob("*")
        if p.is_file()
        and p.suffix.lower() in config.SUPPORTED_EXTS
        and not p.name.startswith(("~$", "."))
    )
    if not files:
        logger.warning("No hay documentos en %s", docs_dir)
        return []

    raw_docs: List[Document] = []
    for p in files:
        raw_docs.extend(load_file(p))
    if not raw_docs:
        logger.warning("No se pudo extraer texto de los documentos.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(raw_docs)
    for i, c in enumerate(chunks):
        pagina = f":p{c.metadata['page']}" if "page" in c.metadata else ""
        c.metadata["chunk_id"] = f"{Path(c.metadata['doc_key']).stem}{pagina}:c{i}"

    print(f"{len(files)} archivos -> {len(raw_docs)} docs/páginas -> {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks
