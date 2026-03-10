from __future__ import annotations

import json
import os
from enum import Enum
from typing import Iterable, List, Optional
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv

from .retriever import RetrievedChunk


load_dotenv()

# Message shown when no relevant chunks; also used by API for consistency with frontend.
NO_RELEVANT_CHUNKS_MESSAGE = (
    "I couldn't find anything relevant to your question "
    "in the indexed codebase. Try asking about a specific "
    "function, file, or concept that exists in the repo."
)


class QueryMode(str, Enum):
    """Supported generator modes."""

    EXPLAIN = "explain"
    DOCSTRING = "docstring"
    FIND = "find"
    SUMMARIZE = "summarize"


def _system_prompt() -> str:
    return (
        "You are a code documentation expert. You answer questions ONLY about the provided code snippets. "
        "You are given code from a codebase, each snippet with file name and line numbers. "
        "CRITICAL: If the user asks about a topic, library, framework, or concept (e.g. TensorFlow, pandas, REST APIs) "
        "that is NOT mentioned, imported, or implemented in the snippets, respond with exactly: Not found in indexed codebase. "
        "Do NOT use the code to answer unrelated questions. Do NOT generalize from the code to explain external topics. "
        "Only answer when the snippets clearly contain what the user is asking about. "
        "For answers that use code, always cite file and line numbers, e.g. `file.py:10-25`."
    )


def _build_user_prompt(question: str, mode: QueryMode, chunks: List[RetrievedChunk]) -> str:
    """
    Build a structured user prompt for the LLM given the query mode and context.
    """
    header = f"Query mode: {mode.value}\nQuestion: {question.strip()}\n\n"

    context_blocks: List[str] = []
    for idx, ch in enumerate(chunks, start=1):
        location = f"{ch.file}:{ch.start_line}-{ch.end_line}"
        block = (
            f"Snippet {idx} (score={ch.score:.3f})\n"
            f"Location: {location}\n"
            f"Symbol: {ch.symbol}\n"
            f"Language: {ch.language}\n"
            "Code:\n"
            "```code\n"
            f"{ch.code}\n"
            "```\n"
        )
        context_blocks.append(block)

    context_text = "Context snippets:\n" + "\n".join(context_blocks) if context_blocks else "Context snippets:\n<none>\n"

    instructions = (
        "\nInstructions:\n"
        "- Use only the provided snippets as the source of truth.\n"
        "- Do not invent functions, classes, or behaviours not present in the snippets.\n"
        "- Always mention file and line ranges for any behaviour you describe.\n"
        "- If the snippets do not contain the requested information, reply exactly with: "
        "Not found in indexed codebase.\n"
    )

    if mode == QueryMode.EXPLAIN:
        instructions += (
            "- Explain in clear, concise language what the relevant code is doing.\n"
            "- Focus on behaviour, inputs, outputs, and side effects.\n"
            "- If the question is about something (library, concept, framework) not present in the snippets, reply: Not found in indexed codebase.\n"
        )
    elif mode == QueryMode.DOCSTRING:
        instructions += (
            "- Generate idiomatic docstrings for the functions or methods in the snippets.\n"
            "- Include parameters, return values, and side effects where applicable.\n"
        )
    elif mode == QueryMode.FIND:
        instructions += (
            "- Identify all relevant occurrences or patterns related to the query within the snippets.\n"
            "- For each match, list the file, line range, and a one-line description.\n"
        )
    elif mode == QueryMode.SUMMARIZE:
        instructions += (
            "- Provide a high-level summary of the module or file responsibilities.\n"
            "- Call out key entry points and important interactions.\n"
        )

    return header + context_text + instructions


class OllamaGenerator:
    """
    LLM answer generation via a local Ollama instance.

    This class focuses on prompt construction and streaming responses. It does
    not perform retrieval; callers should supply the `RetrievedChunk` context.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 300,
    ) -> None:
        raw = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").strip()
        # Avoid //host (no scheme) which produces broken URLs and 404s from requests.
        if raw.startswith("//"):
            raw = "http:" + raw
        elif not raw.startswith(("http://", "https://")):
            raw = "http://" + raw.lstrip("/")
        self.base_url = raw.rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "codellama")
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate(
        self,
        question: str,
        mode: str,
        chunks: List[RetrievedChunk],
        stream: bool = False,
    ) -> Iterable[str] | str:
        """
        Generate an LLM answer for the given question and context.

        When `stream=True`, this returns an iterator yielding tokens/chunks
        from Ollama. Otherwise, it returns the full response string.
        """
        if not chunks or len(chunks) == 0:
            return NO_RELEVANT_CHUNKS_MESSAGE

        try:
            qmode = QueryMode(mode)
        except ValueError:
            raise ValueError(f"Unsupported query mode: {mode}")

        prompt = _build_user_prompt(question, qmode, chunks)

        payload = {
            "model": self.model,
            "stream": stream,
            "messages": [
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": prompt},
            ],
        }

        url = urljoin(self.base_url + "/", "api/chat")

        if stream:
            return self._stream_chat(url, payload)
        return self._non_stream_chat(url, payload)

    # ------------------------------------------------------------------ #
    # HTTP helpers
    # ------------------------------------------------------------------ #

    def _stream_chat(self, url: str, payload: dict) -> Iterable[str]:
        """
        Stream the response from Ollama token-by-token (or chunk-by-chunk).
        """
        with requests.post(url, json=payload, stream=True, timeout=self.timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                if data.get("error"):
                    yield f"Error: {data['error']}"
                    return
                message = data.get("message") or {}
                content = message.get("content") or ""
                if content:
                    yield content
                if data.get("done"):
                    break

    def _non_stream_chat(self, url: str, payload: dict) -> str:
        """
        Request a non-streaming completion and return the full content.
        """
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            # Surface 404 etc. with body hint (e.g. wrong Ollama version or URL).
            detail = ""
            r = exc.response
            if r is not None and r.content:
                try:
                    detail = r.text[:500]
                except Exception:
                    pass
            return (
                f"Error: Ollama returned {r.status_code if r else '?'} for {url}. "
                f"Ensure Ollama is running (`ollama serve`) and the model is pulled "
                f"(`ollama pull {self.model}`). {detail}"
            )
        except requests.exceptions.Timeout:
            return "Error: Ollama request timed out. Check that the model is loaded and responsive on the Ollama server."
        except requests.exceptions.RequestException as exc:
            return f"Error: Failed to contact Ollama at {url}: {exc}"

        data = resp.json()
        message = data.get("message", {})
        content = message.get("content", "")
        return content or ""


__all__ = ["NO_RELEVANT_CHUNKS_MESSAGE", "QueryMode", "OllamaGenerator"]

