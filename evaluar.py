"""Evalúa la RECUPERACIÓN (sin LLM, sin gastar tokens) para varias configuraciones de chunking.

Uso:  python evaluar.py

Para cada configuración (tamaño de chunk / solapamiento) se (re)indexa en ChromaDB y se lanzan
las mismas preguntas. Una pregunta "acierta" si alguno de los chunks recuperados contiene la
frase clave que responde a la pregunta (criterio independiente del tamaño de chunk).

Métricas:
  Hit@1   -> el chunk nº 1 contiene la respuesta
  Hit@3   -> alguno de los 3 chunks contiene la respuesta
  MRR     -> media de 1/posición del primer chunk correcto (0 si no aparece)
  Prec@3  -> fracción de los 3 chunks recuperados que contienen la respuesta (ruido = 1 - Prec@3)
  Sim@1   -> similitud media del chunk nº 1 en las preguntas con respuesta
  Rechaz. -> preguntas fuera de tema correctamente rechazadas (similitud < MIN_SIMILARITY)
"""
import re

import chromadb
import numpy as np

from rag import config
from rag.indexer import build_index, get_embedder

# (pregunta, frase clave que debe aparecer en el chunk correcto)
PREGUNTAS = [
    ("¿Cuánto tiempo se fríen las patatas?", "unos 20 minutos"),
    ("¿Cuántos huevos lleva la tortilla?", "6 huevos"),
    ("¿Cuánto tiempo se conserva la tortilla en la nevera?", "hasta 2 días"),
    ("¿Cómo debe quedar el centro para una tortilla jugosa?", "70 %"),
    ("¿Quién inventó la tortilla durante el sitio de Bilbao?", "Zumalacárregui"),
    ("¿Cuándo llegó la patata a España?", "siglo XVI"),
    ("¿Qué porcentaje prefirió la tortilla con cebolla en la encuesta?", "60 %"),
    ("¿Cómo es la tortilla de Betanzos?", "Betanzos"),
    ("¿Qué lleva el gazpacho?", "sopa fría de tomate"),
    ("¿Qué es el socarrat?", "socarrat"),
    ("¿Cómo se sirve el pulpo a feira?", "tijera"),
    ("¿Con qué se elaboran las croquetas?", "bechamel espesa"),
    ("¿En cuántos vuelcos se sirve el cocido madrileño?", "tres vuelcos"),
]
FUERA_DE_TEMA = [
    "¿Cuál es la capital de Australia?",
    "¿Quién ganó el Mundial de fútbol de 2010?",
    "¿Cómo se escribe un bucle for en Python?",
    "¿Cuántos habitantes tiene Tokio?",
]

# Tamaño variable con overlap proporcional + overlap variable con tamaño fijo (300)
CONFIGS = [(150, 30), (300, 50), (900, 150), (2000, 300),
           (300, 0), (300, 100)]


def norm(t: str) -> str:
    """Minúsculas y espacios colapsados (los PDF parten las líneas)."""
    return re.sub(r"\s+", " ", t).lower()


def evaluar(cliente, embedder, q_vec, q_vec_fuera, size, overlap):
    n_chunks = build_index(size, overlap, embedder)
    col = cliente.get_collection(config.collection_name(size, overlap))

    # Una sola consulta por lote (vectorizado) en lugar de una por pregunta
    res = col.query(query_embeddings=q_vec, n_results=config.TOP_K,
                    include=["documents", "distances"])
    claves = [norm(k) for _, k in PREGUNTAS]
    hit1, hit3, rr, prec, sim1 = [], [], [], [], []
    for docs, dist, clave in zip(res["documents"], res["distances"], claves):
        ok = np.array([clave in norm(d) for d in docs])
        hit1.append(ok[0])
        hit3.append(ok.any())
        rr.append(1 / (ok.argmax() + 1) if ok.any() else 0.0)
        prec.append(ok.mean())
        sim1.append(1 - dist[0])

    res_f = col.query(query_embeddings=q_vec_fuera, n_results=1, include=["distances"])
    sim_fuera = 1 - np.array([d[0] for d in res_f["distances"]])
    rechazadas = (sim_fuera < config.MIN_SIMILARITY).sum()
    # preguntas con respuesta que NO se rechazarían por el umbral
    aceptadas = (np.array(sim1) >= config.MIN_SIMILARITY).sum()

    return dict(cfg=f"{size}/{overlap}", chunks=n_chunks,
                hit1=np.mean(hit1), hit3=np.mean(hit3), mrr=np.mean(rr),
                prec3=np.mean(prec), sim1=np.mean(sim1),
                aceptadas=f"{aceptadas}/{len(PREGUNTAS)}",
                rechazadas=f"{rechazadas}/{len(FUERA_DE_TEMA)}",
                sim_fuera=sim_fuera.max())


def main():
    embedder = get_embedder()
    q_vec = embedder.encode([p for p, _ in PREGUNTAS], normalize_embeddings=True).tolist()
    q_vec_fuera = embedder.encode(FUERA_DE_TEMA, normalize_embeddings=True).tolist()
    cliente = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

    filas = [evaluar(cliente, embedder, q_vec, q_vec_fuera, s, o) for s, o in CONFIGS]

    print(f"\nEVALUACIÓN DE LA RECUPERACIÓN  ({len(PREGUNTAS)} preguntas con respuesta, "
          f"{len(FUERA_DE_TEMA)} fuera de tema, TOP_K={config.TOP_K}, umbral={config.MIN_SIMILARITY})\n")
    cab = (f"{'size/ovl':<10}{'chunks':>7}{'Hit@1':>8}{'Hit@3':>8}{'MRR':>7}{'Prec@3':>8}"
           f"{'Sim@1':>8}{'Acept.':>8}{'Rechaz.':>9}{'MaxSimFuera':>13}")
    print(cab)
    print("-" * len(cab))
    for f in filas:
        print(f"{f['cfg']:<10}{f['chunks']:>7}{f['hit1']:>8.2f}{f['hit3']:>8.2f}{f['mrr']:>7.2f}"
              f"{f['prec3']:>8.2f}{f['sim1']:>8.2f}{f['aceptadas']:>8}{f['rechazadas']:>9}"
              f"{f['sim_fuera']:>13.2f}")


if __name__ == "__main__":
    main()
