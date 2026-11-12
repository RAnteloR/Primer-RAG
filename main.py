"""Lanza el chat RAG.  Uso: python main.py [--chunk-size 900] [--overlap 150]"""
import argparse

from rag import config
from rag.chat_ui import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    args = ap.parse_args()
    run(args.chunk_size, args.overlap)
