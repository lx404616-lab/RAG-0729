"""RAG 问答流水线：切分 -> BGE 建索引 -> 检索 -> 生成/抽取。"""

from __future__ import annotations

from pathlib import Path

import config
from .chunker import chunk_documents, load_documents
from .generator import AnswerResult, generate_answer
from .indexer import VectorIndex


class RAGPipeline:
    """RAG 知识问答系统入口。"""

    def __init__(
        self,
        kb_dir: Path | None = None,
        index_dir: Path | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> None:
        self.kb_dir = kb_dir or config.KB_DIR
        self.index_dir = index_dir or config.INDEX_DIR
        self.top_k = top_k if top_k is not None else config.TOP_K
        self.score_threshold = (
            score_threshold if score_threshold is not None else config.SCORE_THRESHOLD
        )
        self.index = VectorIndex(
            model_name=config.BGE_MODEL_NAME,
            query_instruction=config.BGE_QUERY_INSTRUCTION,
            batch_size=config.BGE_BATCH_SIZE,
            device=config.BGE_DEVICE,
        )

    def _index_ready(self) -> bool:
        return (
            (self.index_dir / "embeddings.npy").exists()
            and (self.index_dir / "path_embeddings.npy").exists()
            and (self.index_dir / "fact_embeddings.npy").exists()
            and (self.index_dir / "fact_chunk_idx.npy").exists()
            and (self.index_dir / "index_meta.json").exists()
        )

    def build_index(self, force_rebuild: bool = False) -> int:
        """完成文档切分并建立 BGE 向量索引；已有索引时可直接加载。"""
        if self._index_ready() and not force_rebuild:
            self.index.load(self.index_dir)
            return len(self.index.chunks)

        docs = load_documents(self.kb_dir)
        if not docs:
            raise FileNotFoundError(f"知识库目录为空或不存在: {self.kb_dir}")

        chunks = chunk_documents(
            docs,
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP,
        )
        self.index.build(chunks)
        self.index.save(self.index_dir)
        return len(chunks)

    def ask(self, question: str, answer_mode: str | None = None) -> AnswerResult:
        """检索相关内容并生成带引用的回答。

        answer_mode: ai | extractive；默认读取 config.DEFAULT_ANSWER_MODE
        """
        question = question.strip()
        mode = (answer_mode or config.DEFAULT_ANSWER_MODE or "ai").strip().lower()
        if not question:
            return generate_answer(
                question="",
                hits=[],
                score_threshold=self.score_threshold,
                answer_mode=mode,
            )

        if self.index.matrix is None:
            self.build_index()

        hits = self.index.search(question, top_k=self.top_k)
        return generate_answer(
            question=question,
            hits=hits,
            score_threshold=self.score_threshold,
            answer_mode=mode,
            llm_model=config.LLM_MODEL,
            llm_temperature=config.LLM_TEMPERATURE,
            llm_max_tokens=config.LLM_MAX_TOKENS,
            openai_base_url=config.LLM_BASE_URL,
            api_key=config.get_llm_api_key(),
            reasoning_effort=config.LLM_REASONING_EFFORT,
            enable_thinking=config.LLM_ENABLE_THINKING,
        )

    def format_result(self, result: AnswerResult) -> str:
        """将问答结果格式化为可读文本。"""
        lines = [result.answer, ""]
        if result.citations:
            lines.append("—— 引用文档 ——")
            for cite in result.citations:
                ref = cite.get("ref", "")
                path = cite.get("title_path") or cite.get("section", "")
                lines.append(
                    f"[引用{ref}] {cite['source']}｜路径：{path}｜"
                    f"相似度：{cite['score']}｜片段ID：{cite['chunk_id']}"
                )
                lines.append(f"   摘要：{cite['snippet']}")
        else:
            lines.append("—— 引用文档 ——")
            lines.append("（无）")
        lines.append("")
        mode_name = {
            "generative": "生成式",
            "extractive": "抽取式",
            "refuse": "拒答",
        }.get(result.mode, result.mode)
        lines.append(f"[模式: {mode_name}({result.mode}) | 可回答: {result.can_answer}]")
        return "\n".join(lines)
