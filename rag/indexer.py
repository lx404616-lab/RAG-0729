"""向量索引：BGE 稠密向量 Embedding（内容 + 标题路径 + 事实句 max-pool）。"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .chunker import Chunk

_BOILERPLATE = "星澜知识库是一款企业级RAG知识管理产品"


class VectorIndex:
    """基于 BGE 的稠密向量索引。

    检索分 = max(内容与路径加权分, 事实句 max-pool 分)。
    事实句单独编码再按 chunk 取最大，避免多句拼接稀释关键事实。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        query_instruction: str = "为这个句子生成表示以用于检索相关文章：",
        batch_size: int = 32,
        device: str | None = None,
        content_weight: float = 0.55,
        path_weight: float = 0.45,
    ) -> None:
        self.model_name = model_name
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self.device = device
        self.content_weight = content_weight
        self.path_weight = path_weight
        self._model = None
        self.matrix: np.ndarray | None = None
        self.path_matrix: np.ndarray | None = None
        self.fact_vecs: np.ndarray | None = None      # (M, D)
        self.fact_chunk_idx: np.ndarray | None = None  # (M,)
        self.chunks: list[Chunk] = []

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "未安装 sentence-transformers，请执行: "
                "py -3 -m pip install sentence-transformers"
            ) from exc

        kwargs = {}
        if self.device:
            kwargs["device"] = self.device
        self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    def _embed(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        model = self._load_model()
        payload = (
            [self.query_instruction + t for t in texts] if is_query else texts
        )
        vectors = model.encode(
            payload,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    @staticmethod
    def _extract_fact_sentences(body: str) -> list[str]:
        """抽出独特事实句，过滤纯产品介绍模板。"""
        facts: list[str] = []
        for s in re.split(r"(?<=[。！？；])", body):
            s = s.strip()
            if len(s) < 8:
                continue
            compact = re.sub(
                r"^[\u4e00-\u9fffA-Za-z0-9_\-]{1,20}内容\d+[：:]\s*", "", s
            )
            # 模板长句：拆出含关键事实的子句
            if _BOILERPLATE in compact:
                for part in re.split(r"[；;]", compact):
                    part = part.strip(" ，,")
                    if re.search(
                        r"\d+(\.\d+)?%|\d{1,2}:\d{2}|计划维护|退款|SLA|首次响应",
                        part,
                    ):
                        facts.append(part)
                continue
            if re.search(
                r"\d+(\.\d+)?%|\d{1,2}:\d{2}|SLA|PDF|Word|退款|维护|备份|加密|导入",
                compact,
            ):
                facts.append(compact)

        uniq: list[str] = []
        seen = set()
        for f in facts:
            if f not in seen:
                uniq.append(f)
                seen.add(f)
        return uniq[:8]

    def build(self, chunks: list[Chunk]) -> None:
        """对切分结果建立 BGE 稠密向量索引。"""
        if not chunks:
            raise ValueError("知识库切分结果为空，无法建立索引")
        self.chunks = chunks
        content_texts = [c.text for c in chunks]
        path_texts = [f"{c.title_path}。{c.section}" for c in chunks]

        fact_texts: list[str] = []
        fact_owners: list[int] = []
        for i, c in enumerate(chunks):
            sents = self._extract_fact_sentences(c.body)
            if not sents:
                sents = [f"{c.title_path} {c.section}"]
            for s in sents:
                fact_texts.append(s)
                fact_owners.append(i)

        self.matrix = self._embed(content_texts, is_query=False)
        self.path_matrix = self._embed(path_texts, is_query=False)
        self.fact_vecs = self._embed(fact_texts, is_query=False)
        self.fact_chunk_idx = np.asarray(fact_owners, dtype=np.int32)

    def encode_query(self, query: str) -> np.ndarray:
        return self._embed([query], is_query=True)[0]

    def search(self, query: str, top_k: int = 4) -> list[tuple[Chunk, float]]:
        """混合检索：内容/路径加权 与 事实句 max-pool 取较大值。"""
        if (
            self.matrix is None
            or self.path_matrix is None
            or self.fact_vecs is None
            or self.fact_chunk_idx is None
        ):
            raise RuntimeError("索引尚未构建")

        q = self.encode_query(query)
        fused = (
            self.content_weight * (self.matrix @ q)
            + self.path_weight * (self.path_matrix @ q)
        )

        raw_fact = self.fact_vecs @ q
        chunk_fact = np.full(len(self.chunks), -1.0, dtype=np.float32)
        for score, owner in zip(raw_fact, self.fact_chunk_idx):
            idx = int(owner)
            if score > chunk_fact[idx]:
                chunk_fact[idx] = score

        scores = np.maximum(fused, chunk_fact)
        if scores.size == 0:
            return []
        k = min(top_k, scores.size)
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(self.chunks[int(i)], float(scores[int(i)])) for i in top_idx]

    def save(self, index_dir: Path) -> None:
        if (
            self.matrix is None
            or self.path_matrix is None
            or self.fact_vecs is None
            or self.fact_chunk_idx is None
        ):
            raise RuntimeError("索引尚未构建，无法保存")
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "embeddings.npy", self.matrix)
        np.save(index_dir / "path_embeddings.npy", self.path_matrix)
        np.save(index_dir / "fact_embeddings.npy", self.fact_vecs)
        np.save(index_dir / "fact_chunk_idx.npy", self.fact_chunk_idx)
        meta = {
            "model_name": self.model_name,
            "query_instruction": self.query_instruction,
            "content_weight": self.content_weight,
            "path_weight": self.path_weight,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "body": c.body,
                    "source": c.source,
                    "section": c.section,
                    "title_path": c.title_path,
                    "start_char": c.start_char,
                }
                for c in self.chunks
            ],
        }
        (index_dir / "index_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, index_dir: Path) -> None:
        meta_path = index_dir / "index_meta.json"
        emb_path = index_dir / "embeddings.npy"
        path_emb = index_dir / "path_embeddings.npy"
        fact_emb = index_dir / "fact_embeddings.npy"
        fact_idx = index_dir / "fact_chunk_idx.npy"
        if not meta_path.exists() or not emb_path.exists():
            raise FileNotFoundError(f"未找到 BGE 索引文件: {index_dir}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.model_name = meta.get("model_name", self.model_name)
        self.query_instruction = meta.get(
            "query_instruction", self.query_instruction
        )
        self.content_weight = float(meta.get("content_weight", self.content_weight))
        self.path_weight = float(meta.get("path_weight", self.path_weight))
        self.matrix = np.load(emb_path).astype(np.float32)
        self.path_matrix = (
            np.load(path_emb).astype(np.float32) if path_emb.exists() else self.matrix
        )
        if fact_emb.exists() and fact_idx.exists():
            self.fact_vecs = np.load(fact_emb).astype(np.float32)
            self.fact_chunk_idx = np.load(fact_idx).astype(np.int32)
        else:
            # 兼容旧索引
            self.fact_vecs = self.matrix
            self.fact_chunk_idx = np.arange(len(self.matrix), dtype=np.int32)
        self.chunks = [Chunk(**item) for item in meta["chunks"]]
