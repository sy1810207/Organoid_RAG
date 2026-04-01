"""LLM generation via LM Studio OpenAI-compatible API."""

from typing import Iterator

from openai import OpenAI

from config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
from src.retriever import RetrievalResult


SYSTEM_PROMPT = """You are a scientific literature assistant specializing in organoid research.
Answer the user's question based ONLY on the provided context passages.
For each claim, cite the source using [PMID: XXXXX] format.
If the context does not contain enough information to answer, say so explicitly.
Do not fabricate information or citations.
Structure your answer with clear paragraphs.
You can respond in Chinese or English depending on the user's language."""


class Generator:
    def __init__(
        self,
        base_url: str = LLM_BASE_URL,
        api_key: str = LLM_API_KEY,
        model: str = LLM_MODEL,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def _format_context(self, results: list[RetrievalResult]) -> str:
        parts = []
        for i, r in enumerate(results, 1):
            m = r.metadata
            parts.append(
                f"[{i}] {m.get('title', 'Untitled')} "
                f"({m.get('journal', '')}, {m.get('year', '')}; "
                f"PMID: {m.get('pmid', 'N/A')})\n"
                f"Section: {m.get('section', 'unknown')}\n\n"
                f"{r.text}"
            )
        return "\n\n---\n\n".join(parts)

    def generate(
        self,
        query: str,
        results: list[RetrievalResult],
        stream: bool = False,
    ) -> str | Iterator[str]:
        """Generate an answer using retrieved context."""
        context = self._format_context(results)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ]

        # Disable Qwen3's thinking/reasoning mode to get direct answers
        extra = {"chat_template_kwargs": {"enable_thinking": False}}

        if stream:
            return self._stream(messages, extra)
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=2048,
                extra_body=extra,
            )
            return response.choices[0].message.content

    def _stream(self, messages: list[dict], extra: dict) -> Iterator[str]:
        """Stream the response token by token."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=2048,
            stream=True,
            extra_body=extra,
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
