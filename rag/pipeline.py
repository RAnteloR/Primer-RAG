"""RAG como grafo LangGraph: recuperar -> (generar | sin_info)."""
import time
from types import SimpleNamespace
from typing import TypedDict

import chromadb
import requests
from langgraph.graph import END, START, StateGraph

from . import config

NO_INFO = "No tengo esa información en los documentos."

INSTRUCCIONES = f"""Eres un asistente que responde preguntas usando ÚNICAMENTE los documentos proporcionados.
- Si la respuesta no está en los documentos, di: "{NO_INFO}"
- Indica entre corchetes el id del documento que usas, por ejemplo [apuntes:p3:c12].
- Responde en el mismo idioma de la pregunta, de forma breve."""


class GeminiLLM:
    """Cliente mínimo de la API nativa de Gemini (generateContent)."""

    def __init__(self, modelo: str):
        self.modelo = modelo
        self.url = f"{config.BASE_URL}/v1beta/models/{modelo}:generateContent"

    def invoke(self, mensajes):
        sistema = "\n".join(t for r, t in mensajes if r == "system")
        cuerpo = {"contents": [{"role": "user", "parts": [{"text": t}]}
                               for r, t in mensajes if r != "system"],
                  "generationConfig": {"temperature": config.LLM_TEMPERATURE}}
        if sistema:
            cuerpo["systemInstruction"] = {"parts": [{"text": sistema}]}
        for intento in range(config.LLM_MAX_RETRIES + 1):
            r = requests.post(self.url, json=cuerpo, timeout=config.LLM_TIMEOUT,
                              headers={"x-goog-api-key": config.API_KEY})
            if r.status_code in (429, 500, 503) and intento < config.LLM_MAX_RETRIES:
                time.sleep(2 ** intento)
                continue
            if not r.ok:
                raise RuntimeError(f"{self.modelo}: HTTP {r.status_code} {r.text[:300]}")
            partes = r.json()["candidates"][0]["content"]["parts"]
            return SimpleNamespace(content="".join(p.get("text", "") for p in partes))


class Estado(TypedDict):
    pregunta: str
    documentos: list[dict]
    respuesta: str


class RAG:
    def __init__(self, embedder, chunk_size: int, chunk_overlap: int):
        self.embedder = embedder
        cliente = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.coleccion = cliente.get_collection(config.collection_name(chunk_size, chunk_overlap))
        self.llm = self._crear_llm(config.LLM_MODEL)
        self.llm_respaldo = (self._crear_llm(config.LLM_FALLBACK_MODEL)
                             if config.LLM_FALLBACK_MODEL else None)
        self.grafo = self._montar()

    @staticmethod
    def _crear_llm(modelo: str) -> GeminiLLM:
        return GeminiLLM(modelo)

    def _invocar_llm(self, mensajes):
        try:
            return self.llm.invoke(mensajes)
        except Exception:
            if self.llm_respaldo is None:
                raise
            return self.llm_respaldo.invoke(mensajes)

    # --- nodos ---
    def _recuperar(self, estado: Estado) -> dict:
        v = self.embedder.encode([estado["pregunta"]], normalize_embeddings=True).tolist()
        res = self.coleccion.query(query_embeddings=v, n_results=config.TOP_K,
                                   include=["documents", "metadatas", "distances"])
        encontrados = []
        for i, t, m, d in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            fuente = m["doc_key"] + (f", pág. {m['page']}" if "page" in m else "")
            encontrados.append({"id": i, "texto": t, "fuente": fuente, "similitud": round(1 - d, 2)})
        relevantes = [d for d in encontrados if d["similitud"] >= config.MIN_SIMILARITY]
        return {"documentos": relevantes}

    def _generar(self, estado: Estado) -> dict:
        contexto = "\n\n".join(f"[{d['id']}] (fuente: {d['fuente']})\n{d['texto']}"
                               for d in estado["documentos"])
        r = self._invocar_llm([("system", INSTRUCCIONES),
                             ("human", f"DOCUMENTOS:\n{contexto}\n\nPREGUNTA: {estado['pregunta']}")])
        return {"respuesta": r.content}

    def _sin_info(self, estado: Estado) -> dict:
        return {"respuesta": NO_INFO}

    def _montar(self):
        g = StateGraph(Estado)
        g.add_node("recuperar", self._recuperar)
        g.add_node("generar", self._generar)
        g.add_node("sin_info", self._sin_info)
        g.add_edge(START, "recuperar")
        g.add_conditional_edges("recuperar", lambda e: "generar" if e["documentos"] else "sin_info")
        g.add_edge("generar", END)
        g.add_edge("sin_info", END)
        return g.compile()

    def preguntar(self, pregunta: str) -> Estado:
        return self.grafo.invoke({"pregunta": pregunta})
