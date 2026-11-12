"""Configuración central del RAG."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
CHROMA_DIR = ROOT / "chroma_rag"

# Chunking (valores por defecto; se pueden sobrescribir por CLI)
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
SUPPORTED_EXTS = {".txt", ".md", ".pdf", ".docx"}

# Embeddings
EMBEDDING_MODEL = "BAAI/bge-m3"   # multilingüe, 1024 dims, ~2,3 GB; no necesita prefijos de query
EMBEDDING_DEVICE = None           # None = automático (cuda si hay GPU, si no cpu)
EMBEDDING_MAX_SEQ_LENGTH = 512    # los chunks (~900 car.) caben de sobra; ahorra RAM/tiempo en CPU

# Recuperación
TOP_K = 3
MIN_SIMILARITY = 0.40  # por debajo, se considera que no hay info relevante

# LLM (API compatible con OpenAI)
API_KEY = os.getenv("GOOGLE_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
LLM_MODEL = "gemini-3.6-flash"  # ajustar al modelo disponible
LLM_TEMPERATURE = 0
LLM_REASONING_EFFORT = "low"  # evita que el modelo "piense" cientos de tokens (lento y variable)
LLM_TIMEOUT = 45              # segundos; si se supera, la UI muestra el error
LLM_MAX_RETRIES = 3           # reintentos con espera creciente (errores 429/503 temporales)
LLM_FALLBACK_MODEL = "gemini-2.5-flash"  # se usa si el principal falla; None para desactivar


def collection_name(chunk_size: int, chunk_overlap: int) -> str:
    """Una colección por combinación de parámetros, para poder compararlas."""
    return f"docs_cs{chunk_size}_ov{chunk_overlap}"
