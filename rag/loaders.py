"""Carga de documentos .txt, .md, .pdf y .docx."""
import logging
from pathlib import Path
from typing import List

import docx
from langchain_core.documents import Document
from pypdf import PdfReader

logger = logging.getLogger(__name__)


def _meta(path: Path, **extra) -> dict:
    return {"source": str(path), "doc_key": path.name, **extra}


def load_txt_md(path: Path) -> List[Document]:
    try:
        txt = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        txt = path.read_text(encoding="cp1252", errors="replace")
    return [Document(page_content=txt, metadata=_meta(path))] if txt.strip() else []


def load_pdf(path: Path) -> List[Document]:
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        reader.decrypt("")
    docs = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        if txt.strip():
            docs.append(Document(page_content=txt, metadata=_meta(path, page=i + 1)))
    return docs


def load_docx(path: Path) -> List[Document]:
    d = docx.Document(str(path))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for table in d.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    txt = "\n".join(parts).strip()
    return [Document(page_content=txt, metadata=_meta(path))] if txt else []


LOADERS = {".txt": load_txt_md, ".md": load_txt_md, ".pdf": load_pdf, ".docx": load_docx}


def load_file(path: Path) -> List[Document]:
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        return []
    try:
        return loader(path)
    except Exception as e:
        logger.warning("No se pudo leer %s: %s", path, e)
        return []
