"""Answer generators.

Every generator returns a structured Draft (answer text + cited chunk ids)
so the pipeline can validate citations regardless of provider.

- ExtractiveGenerator (default): deterministic, offline, used in CI.
- OpenAIGenerator / BedrockGenerator: call a hosted LLM with a versioned
  prompt and require JSON output. Keys come from the environment only.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .retrieval import Hit, tokenize

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Draft:
    answer: str
    cited_ids: list[str]
    model: str


class Generator(Protocol):
    name: str

    def generate(self, question: str, hits: list[Hit]) -> Draft: ...


class ExtractiveGenerator:
    """Picks the context sentences that best cover the question."""

    name = "extractive-v1"

    def __init__(self, max_sentences: int = 2):
        self.max_sentences = max_sentences

    def generate(self, question: str, hits: list[Hit]) -> Draft:
        q = set(tokenize(question))
        scored = []
        for rank, h in enumerate(hits):
            title_terms = set(tokenize(h.chunk.title))
            section_terms = set(tokenize(h.chunk.section)) - title_terms
            for sent in _SENTENCE.split(h.chunk.text):
                matched = q & (set(tokenize(sent)) | section_terms)
                # Words from the document title match every sentence, so they count less.
                overlap = sum(0.3 if t in title_terms else 1.0 for t in matched)
                if overlap:
                    scored.append((overlap - 0.1 * rank, sent.strip(), h.chunk.chunk_id))
        scored.sort(key=lambda s: -s[0])
        if not scored:
            return Draft(answer="", cited_ids=[], model=self.name)
        best = scored[0][0]
        # Extra sentences must be nearly as relevant as the best one and come from the same document.
        best_doc = scored[0][2].split("#")[0]
        chosen = [s for s in scored[: self.max_sentences] if s[0] >= 0.6 * best and s[2].split("#")[0] == best_doc]
        answer = " ".join(f"{s} [{cid}]" for _, s, cid in chosen)
        cited = list(dict.fromkeys(cid for _, _, cid in chosen))  # answer order, de-duplicated
        return Draft(answer=answer, cited_ids=cited, model=self.name)


def load_prompt(version: str = "answer_v1") -> str:
    return (PROMPT_DIR / f"{version}.txt").read_text(encoding="utf-8")


def build_messages(question: str, hits: list[Hit], prompt_version: str = "answer_v1") -> list[dict]:
    context = "\n\n".join(f"[{h.chunk.chunk_id}] ({h.chunk.title} / {h.chunk.section})\n{h.chunk.text}" for h in hits)
    return [
        {"role": "system", "content": load_prompt(prompt_version)},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


def parse_json_draft(raw: str, model: str, allowed_ids: set[str]) -> Draft:
    """Parse and validate the model's JSON; drop citations that were not retrieved."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return valid JSON: {exc}") from exc
    answer = str(data.get("answer", "")).strip()
    cited = [c for c in data.get("citations", []) if c in allowed_ids]
    return Draft(answer=answer, cited_ids=cited, model=model)


class OpenAIGenerator:
    def __init__(self, model: str | None = None, client=None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.name = f"openai:{self.model}"
        if client is None:
            from openai import OpenAI  # optional dependency

            client = OpenAI()  # reads OPENAI_API_KEY from the environment
        self.client = client

    def generate(self, question: str, hits: list[Hit]) -> Draft:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=build_messages(question, hits),
            temperature=0,
            response_format={"type": "json_object"},
        )
        return parse_json_draft(resp.choices[0].message.content, self.name, {h.chunk.chunk_id for h in hits})


class BedrockGenerator:
    def __init__(self, model_id: str | None = None, client=None):
        self.model_id = model_id or os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
        self.name = f"bedrock:{self.model_id}"
        if client is None:
            import boto3  # optional dependency

            client = boto3.client("bedrock-runtime")  # standard AWS credential chain
        self.client = client

    def generate(self, question: str, hits: list[Hit]) -> Draft:
        messages = build_messages(question, hits)
        resp = self.client.converse(
            modelId=self.model_id,
            system=[{"text": messages[0]["content"]}],
            messages=[{"role": "user", "content": [{"text": messages[1]["content"]}]}],
            inferenceConfig={"temperature": 0},
        )
        raw = resp["output"]["message"]["content"][0]["text"]
        return parse_json_draft(raw, self.name, {h.chunk.chunk_id for h in hits})


def get_generator(provider: str | None = None) -> Generator:
    provider = (provider or os.getenv("LLM_PROVIDER", "extractive")).lower()
    if provider == "openai":
        return OpenAIGenerator()
    if provider == "bedrock":
        return BedrockGenerator()
    return ExtractiveGenerator()
