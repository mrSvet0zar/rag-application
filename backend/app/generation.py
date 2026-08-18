"""Answer generation from retrieved context.

Uses the Anthropic SDK directly. When no API key is configured the generator
falls back to a local "demo" answer built from the retrieved context, so the
whole app stays runnable end-to-end without any paid credentials. API failures
degrade the same way rather than surfacing a 500.

Chunking lives in `app.chunking` — this module only turns context into answers.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import anthropic

from app.config import Settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful AI assistant answering questions strictly from the "
    "provided context, which comes from the user's own documents. "
    "Rules:\n"
    "- Answer in the same language as the question.\n"
    "- Base your answer only on the context. If the context does not contain "
    "the answer, say so clearly instead of inventing facts.\n"
    "- Be accurate and concise, and cite the source filename in brackets, "
    "e.g. [source: report.pdf], when you use information from it."
)


class AnswerGenerator:
    """Generates answers with Claude, with demo/error fallbacks."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: anthropic.AsyncAnthropic | None = None
        if not settings.demo_mode:
            self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
            logger.info("Anthropic client ready (model=%s).", settings.chat_model)
        else:
            logger.warning(
                "ANTHROPIC_API_KEY not set -> running in DEMO mode "
                "(answers generated locally from retrieved context)."
            )

    # ---------- Prompt building ----------
    @staticmethod
    def _build_context(chunks: list[dict]) -> str:
        """Format retrieved chunks into a single context block with sources."""
        blocks = []
        for i, chunk in enumerate(chunks, start=1):
            source = chunk.get("filename") or f"document #{chunk.get('document_id')}"
            blocks.append(f"[{i}] (source: {source})\n{chunk['text']}")
        return "\n\n".join(blocks)

    # ---------- Generation ----------
    async def generate_response(self, query: str, chunks: list[dict]) -> tuple[str, int]:
        """Generate an answer from the query + retrieved chunks.

        Returns (answer_text, tokens_used).
        """
        if not chunks:
            return self._no_context_message(), 0

        if self._client is None:
            return self._demo_answer(query, chunks), 0

        context = self._build_context(chunks)
        try:
            message = await self._client.messages.create(
                model=self.settings.chat_model,
                max_tokens=self.settings.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": self._user_prompt(query, context)}],
            )
        except anthropic.APIError as exc:
            logger.error("Anthropic API error: %s", exc)
            return self._llm_error_answer(query, chunks, exc), 0

        answer = "".join(block.text for block in message.content if block.type == "text")
        tokens = message.usage.input_tokens + message.usage.output_tokens
        return answer, tokens

    async def stream_response(
        self, query: str, chunks: list[dict]
    ) -> AsyncIterator[tuple[str, object]]:
        """Stream an answer as it is generated.

        Yields ("token", str) deltas, then a single ("usage", int) total.
        Demo mode streams the local answer word by word for a consistent UX.
        """
        if not chunks:
            yield ("token", self._no_context_message())
            yield ("usage", 0)
            return

        if self._client is None:
            for word in self._demo_answer(query, chunks).split(" "):
                yield ("token", word + " ")
            yield ("usage", 0)
            return

        context = self._build_context(chunks)
        try:
            async with self._client.messages.stream(
                model=self.settings.chat_model,
                max_tokens=self.settings.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": self._user_prompt(query, context)}],
            ) as stream:
                async for text in stream.text_stream:
                    yield ("token", text)
                final = await stream.get_final_message()
                yield ("usage", final.usage.input_tokens + final.usage.output_tokens)
        except anthropic.APIError as exc:
            logger.error("Anthropic API error (stream): %s", exc)
            for word in self._llm_error_answer(query, chunks, exc).split(" "):
                yield ("token", word + " ")
            yield ("usage", 0)

    # ---------- Canned messages ----------
    @staticmethod
    def _user_prompt(query: str, context: str) -> str:
        return (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer using only the context above."
        )

    @staticmethod
    def _no_context_message() -> str:
        return (
            "Je n'ai trouvé aucun passage pertinent dans les documents "
            "indexés pour répondre à cette question. Essayez de reformuler "
            "ou d'ajouter des documents."
        )

    @classmethod
    def _llm_error_answer(cls, query: str, chunks: list[dict], exc: Exception) -> str:
        """Graceful fallback when Claude can't be reached: explain briefly and
        still surface the retrieved context so the request isn't a dead end."""
        reason = str(getattr(exc, "message", "") or exc)
        if "credit balance is too low" in reason.lower():
            hint = (
                "Le compte Anthropic n'a plus de crédits. Ajoutez-en dans "
                "**Plans & Billing** sur console.anthropic.com."
            )
        elif isinstance(exc, anthropic.AuthenticationError):
            hint = "La clé `ANTHROPIC_API_KEY` semble invalide."
        elif isinstance(exc, anthropic.RateLimitError):
            hint = "Limite de débit atteinte, réessayez dans un instant."
        else:
            hint = f"Erreur API : {reason}"

        sources = sorted(
            {c.get("filename") or f"document #{c.get('document_id')}" for c in chunks}
        )
        preview = chunks[0]["text"].strip()
        if len(preview) > 400:
            preview = preview[:400] + "…"
        return (
            "⚠️ **Claude est momentanément indisponible.**\n\n"
            f"{hint}\n\n"
            f"En attendant, voici les {len(chunks)} passage(s) les plus "
            f"pertinents trouvés dans : {', '.join(sources)}.\n\n"
            f"> {preview}"
        )

    @staticmethod
    def _demo_answer(query: str, chunks: list[dict]) -> str:
        """Deterministic offline answer when no LLM key is configured."""
        sources = sorted(
            {c.get("filename") or f"document #{c.get('document_id')}" for c in chunks}
        )
        preview = chunks[0]["text"].strip()
        if len(preview) > 500:
            preview = preview[:500] + "…"
        return (
            "🔧 **Mode démo (aucune clé Anthropic configurée)**\n\n"
            f"Question : *{query}*\n\n"
            f"J'ai retrouvé {len(chunks)} passage(s) pertinent(s) dans "
            f"{len(sources)} document(s) : {', '.join(sources)}.\n\n"
            "Extrait le plus pertinent :\n\n"
            f"> {preview}\n\n"
            "_Ajoutez `ANTHROPIC_API_KEY` dans `backend/.env` pour obtenir une "
            "réponse synthétisée par Claude à partir de ces passages._"
        )
