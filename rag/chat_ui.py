"""Interfaz de chat sencilla con tkinter."""
import queue
import threading
import tkinter as tk
from tkinter import scrolledtext

from .pipeline import RAG


class ChatApp:
    def __init__(self, root: tk.Tk, chunk_size: int, chunk_overlap: int):
        self.root = root
        self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap
        self.rag: RAG | None = None
        self.cola: queue.Queue = queue.Queue()

        root.title(f"RAG Chat (chunk={chunk_size}, overlap={chunk_overlap})")
        root.geometry("800x600")

        self.chat = scrolledtext.ScrolledText(root, wrap=tk.WORD, state="disabled", font=("Segoe UI", 10))
        self.chat.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))
        self.chat.tag_config("user", foreground="#0b5394", font=("Segoe UI", 10, "bold"))
        self.chat.tag_config("bot", foreground="#222222")
        self.chat.tag_config("src", foreground="#777777", font=("Segoe UI", 9, "italic"))
        self.chat.tag_config("sys", foreground="#999999")

        marco = tk.Frame(root)
        marco.pack(fill=tk.X, padx=8, pady=(0, 8))
        self.entrada = tk.Entry(marco, font=("Segoe UI", 11), state="disabled")
        self.entrada.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entrada.bind("<Return>", lambda _e: self.enviar())
        self.boton = tk.Button(marco, text="Enviar", command=self.enviar, state="disabled")
        self.boton.pack(side=tk.RIGHT, padx=(6, 0))

        self.escribir("Cargando modelo de embeddings y base vectorial...\n", "sys")
        threading.Thread(target=self._cargar, daemon=True).start()
        self.root.after(100, self._procesar_cola)

    # --- carga e inferencia en hilos para no congelar la UI ---
    def _cargar(self):
        try:
            from .indexer import get_embedder
            self.rag = RAG(get_embedder(), self.chunk_size, self.chunk_overlap)
            self.cola.put(("ready", None))
        except Exception as e:
            self.cola.put(("error", f"No se pudo iniciar: {e}\n¿Has ejecutado python -m rag.indexer?"))

    def _consultar(self, pregunta: str):
        try:
            self.cola.put(("answer", self.rag.preguntar(pregunta)))
        except Exception as e:
            self.cola.put(("error", f"Error: {e}"))

    def _procesar_cola(self):
        try:
            while True:
                tipo, dato = self.cola.get_nowait()
                if tipo == "ready":
                    self.escribir("Listo. Haz tu pregunta sobre los documentos.\n\n", "sys")
                    self._habilitar(True)
                elif tipo == "answer":
                    self._mostrar_respuesta(dato)
                    self._habilitar(True)
                elif tipo == "error":
                    self.escribir(f"{dato}\n\n", "sys")
                    self._habilitar(self.rag is not None)
        except queue.Empty:
            pass
        self.root.after(100, self._procesar_cola)

    # --- helpers ---
    def _habilitar(self, on: bool):
        estado = "normal" if on else "disabled"
        self.entrada.config(state=estado)
        self.boton.config(state=estado)
        if on:
            self.entrada.focus_set()

    def escribir(self, texto: str, tag: str):
        self.chat.config(state="normal")
        self.chat.insert(tk.END, texto, tag)
        self.chat.config(state="disabled")
        self.chat.see(tk.END)

    def enviar(self):
        pregunta = self.entrada.get().strip()
        if not pregunta or self.rag is None:
            return
        self.entrada.delete(0, tk.END)
        self.escribir(f"Tú: {pregunta}\n", "user")
        self.escribir("Pensando...\n", "sys")
        self._habilitar(False)
        threading.Thread(target=self._consultar, args=(pregunta,), daemon=True).start()

    def _mostrar_respuesta(self, estado):
        self.escribir(f"Asistente: {estado['respuesta']}\n", "bot")
        if estado["documentos"]:
            fuentes = "; ".join(f"{d['fuente']} (sim {d['similitud']:.2f})" for d in estado["documentos"])
            self.escribir(f"Fuentes: {fuentes}\n", "src")
        self.escribir("\n", "bot")


def run(chunk_size: int, chunk_overlap: int):
    root = tk.Tk()
    ChatApp(root, chunk_size, chunk_overlap)
    root.mainloop()
