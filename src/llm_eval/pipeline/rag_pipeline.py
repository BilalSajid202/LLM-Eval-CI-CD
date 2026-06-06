"""Reference RAG pipeline under test — LangChain + local retriever."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from llm_eval.config.loader import PipelineConfig, _project_root
from llm_eval.models.types import PipelineOutput
from llm_eval.pipeline.retriever import LocalRetriever


class RAGPipeline:
    def __init__(self, config: PipelineConfig, api_key: str | None = None):
        self.config = config
        root = _project_root()
        self.retriever = LocalRetriever(
            root / config.rag.knowledge_base_path,
            top_k=config.retriever.top_k,
        )
        self.prompt_template = self._load_prompt(root / config.rag.system_prompt_path)
        self.llm = ChatAnthropic(
            model=config.model.name,
            temperature=config.model.temperature,
            max_tokens=config.model.max_tokens,
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
        )

    def _load_prompt(self, path: Path) -> str:
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("template", "")

    async def run(self, question: str) -> PipelineOutput:
        docs = self.retriever.retrieve(question)
        contexts = [d["content"] for d in docs]
        sources = [d["path"] for d in docs]
        context_block = "\n\n".join(
            f"[{src}]\n{ctx}" for src, ctx in zip(sources, contexts, strict=False)
        )
        prompt = (
            self.prompt_template.replace("{context}", context_block).replace(
                "{question}", question
            )
        )

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        answer = response.content if isinstance(response.content, str) else str(response.content)

        usage = getattr(response, "usage_metadata", None) or {}
        prompt_tokens = usage.get("input_tokens", 0) if usage else 0
        completion_tokens = usage.get("output_tokens", 0) if usage else 0

        return PipelineOutput(
            answer=answer,
            sources=sources,
            contexts=contexts,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=self.config.model.name,
        )


def create_pipeline(config: PipelineConfig, api_key: str | None = None) -> RAGPipeline:
    return RAGPipeline(config, api_key=api_key)
