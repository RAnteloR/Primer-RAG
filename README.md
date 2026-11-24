# Primer-RAG

Sistema RAG (*Retrieval-Augmented Generation*) sobre una pequeña colección de documentos de **comidas tradicionales españolas**, con especial foco en la **tortilla de patatas**. Incluye una interfaz de chat sencilla hecha con **tkinter**.

Es la Tarea RAG 1: chunking recursivo, vectorización con BGE-M3, almacenamiento en ChromaDB, recuperación por similitud y generación de la respuesta con un LLM (Gemini). Si los documentos no contienen información relevante, el sistema lo indica.

## Cómo funciona

```
 docs/ (txt, md, pdf, docx)
        │  loaders.py + chunking.py
        ▼
 chunks (RecursiveCharacterTextSplitter)
        │  indexer.py  (BGE-M3)
        ▼
 ChromaDB  (una colección por tamaño de chunk/solapamiento)
        ▲
        │  pipeline.py (LangGraph)
 pregunta ──► recuperar ──► ¿hay chunks con similitud ≥ umbral?
                               ├─ sí ──► generar (LLM) ──► respuesta + fuentes
                               └─ no ──► "No tengo esa información en los documentos."
```

## Estructura

```
main.py              Lanza el chat (tkinter)
rag/
  config.py          Parámetros: chunking, K, umbral, modelos, rutas
  loaders.py         Lectura de .txt, .md, .pdf y .docx
  chunking.py        Chunking recursivo
  indexer.py         Embeddings BGE-M3 + guardado en ChromaDB (CLI)
  pipeline.py        Grafo LangGraph: recuperar → generar | sin_info
  chat_ui.py         Interfaz de chat con tkinter
docs/                Documentos a consultar
chroma_rag/          Base vectorial (se genera; no se sube al repo)
rag_bge_m3*.ipynb    Notebooks de referencia en los que se basa el proyecto
```

## Instalación

Requiere **Python 3.11 o 3.12** (con 3.14 `torch` y `chromadb` pueden dar problemas) y `tkinter`.

```bash
# Con uv (recomendado)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# O con venv + pip
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

source .venv/bin/activate
```

Si tu Python no incluye `tkinter` (en Debian/Ubuntu): `sudo apt install python3-tk`.

### Clave de API

Copia la plantilla y rellena tu clave de Gemini:

```bash
cp .env.example .env
# edita .env:  GOOGLE_API_KEY=tu_clave  y  GOOGLE_GEMINI_BASE_URL=http://tu_dominio:8338/google
```

El LLM se consume por la API nativa de Gemini (`generateContent`) en `GOOGLE_GEMINI_BASE_URL`. Se usa únicamente `gemini-3.5-flash-lite`.

## Uso

**1. Indexar los documentos** (la primera vez descarga BGE-M3, unos 2,3 GB):

```bash
python -m rag.indexer --chunk-size 900 --overlap 150
```

**2. Abrir el chat** con los mismos parámetros:

```bash
python main.py --chunk-size 900 --overlap 150
```

Ejemplos de preguntas:

- `¿Cuánto tiempo se fríen las patatas?`
- `¿Con cebolla o sin cebolla?`
- `¿Qué lleva el gazpacho?`
- `¿Cuál es la capital de Australia?` → el sistema debe responder que no tiene esa información.

## Probar distintos tamaños de chunk

Cada combinación `chunk-size` / `overlap` se guarda en su propia colección (`docs_cs<size>_ov<overlap>`), así que se pueden comparar sin pisarse:

```bash
python -m rag.indexer --chunk-size 300  --overlap 50
python -m rag.indexer --chunk-size 900  --overlap 150
python -m rag.indexer --chunk-size 2000 --overlap 300

python main.py --chunk-size 300 --overlap 50
```

Los documentos son cortos, por lo que con tamaños grandes cada documento puede quedar en un solo chunk.

## Configuración (`rag/config.py`)

| Parámetro | Por defecto | Descripción |
|---|---|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 900 / 150 | Tamaño y solapamiento (en caracteres) |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Modelo de embeddings (multilingüe, 1024 dims) |
| `EMBEDDING_MAX_SEQ_LENGTH` | 512 | Límite de tokens al vectorizar (ahorra RAM en CPU) |
| `TOP_K` | 3 | Chunks que se pasan al LLM |
| `MIN_SIMILARITY` | 0.40 | Por debajo, se considera que no hay información relevante |
| `LLM_MODEL` | `gemini-3.5-flash-lite` | Modelo de generación; ajusta al que tengas disponible |

## Documentos de ejemplo

| Archivo | Formato | Contenido |
|---|---|---|
| `tortilla_receta.txt` | txt | Receta, tiempos y conservación |
| `tortilla_historia.md` | md | Orígenes y leyendas |
| `tortilla_variantes.docx` | docx | Debate de la cebolla y variantes |
| `comidas_tradicionales_espana.pdf` | pdf | Paella, gazpacho, cocido, pulpo a feira, croquetas |

> Algunos datos de estos documentos son ilustrativos y no están contrastados con fuentes. Para uso real, sustitúyelos por los tuyos.

## Notas

- Si cambias los documentos, vuelve a ejecutar el indexador: recrea la colección correspondiente.
- El umbral `MIN_SIMILARITY` conviene ajustarlo con pruebas reales según el modelo y los documentos.
- La primera carga del chat tarda unos segundos porque inicializa el modelo de embeddings; la ventana avisa cuando está lista.
