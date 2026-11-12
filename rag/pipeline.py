"""RAG como grafo LangGraph: recuperar -> (generar | sin_info)."""
from typing import TypedDict

import chromadb
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from . import config

NO_INFO = "No tengo esa información en los documentos."

INSTRUCCIONES = f"""Eres un asistente que responde preguntas usando ÚNICAMENTE los documentos proporcionados.
- Si la respuesta no está en los documentos, di: "{NO_INFO}"
- Indica entre corchetes el id del documento que usas, por ejemplo [apuntes:p3:c12].
- Responde en el mismo idioma de la pregunta, de forma breve."""


class Estado(TypedDict):
    pregunta: str
    documentos: list[dict]
    respuesta: str


class RAG:
    def __init__(self, embedder, chunk_size: int, chunk_overlap: int):
        self.embedder = embedder
        cliente = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.coleccion = cliente.get_collection(config.collection_name(chunk_size, chunk_overlap))
        self.llm = ChatOpenAI(model=config.LLM_MODEL, api_key=config.API_KEY,
                              base_url=config.BASE_URL, temperature=config.LLM_TEMPERATURE)
        self.grafo = self._montar()

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
        r = self.llm.invoke([("system", INSTRUCCIONES),
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
