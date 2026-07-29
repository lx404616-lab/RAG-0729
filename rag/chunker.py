"""文档加载与切分：Markdown 标题感知 + 标题路径注入 + 滑动窗口兜底。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import config


@dataclass
class Chunk:
    """知识库切分后的文本块。"""

    chunk_id: str
    text: str          # 注入标题路径后的检索文本（供 Embedding）
    body: str          # 纯正文（不含路径前缀），供生成/抽取
    source: str
    section: str       # 当前叶子标题
    title_path: str    # 如：产品白皮书 > 总体架构
    start_char: int


def load_documents(kb_dir: Path) -> list[tuple[str, str]]:
    """加载知识库目录下全部 Markdown 文档。"""
    docs: list[tuple[str, str]] = []
    for path in sorted(kb_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append((path.name, text))
    return docs


def _doc_title(source: str, content: str) -> str:
    """优先取一级标题，否则用去扩展名的文件名。"""
    m = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if m:
        return m.group(1).strip()
    return Path(source).stem


def _split_by_heading_path(text: str, doc_title: str) -> list[tuple[str, str, str]]:
    """按 Markdown 标题层级切分，保留标题路径。

    Returns:
        [(叶子标题, 标题路径, 正文), ...]
    """
    pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        body = text.strip()
        return [("全文", doc_title, body)] if body else []

    stack: list[tuple[int, str]] = []
    sections: list[tuple[str, str, str]] = []

    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))

        if stack and stack[0][1] == doc_title:
            path_parts = [doc_title] + [t for _, t in stack[1:]]
        else:
            path_parts = [doc_title] + [t for _, t in stack if t != doc_title]
        title_path = " > ".join(path_parts) if path_parts else doc_title

        if body:
            sections.append((title, title_path, body))

    return sections


def _window_split(
    text: str,
    chunk_size: int,
    overlap: int,
    soft_extend: int = 100,
) -> list[tuple[str, int]]:
    """滑动窗口兜底：超长段落按窗口切分，优先在句读处断开，并允许软扩展凑整句。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [(text, 0)]

    parts: list[tuple[str, int]] = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(text):
        end = min(start + chunk_size, len(text))

        # 1) 优先在窗口后半段找句读回退
        if end < len(text):
            for sep in ("。", "！", "？", "\n", "；"):
                pos = text.rfind(sep, start + chunk_size // 2, end)
                if pos != -1:
                    end = pos + 1
                    break

        # 2) 若仍截断在句中，向前软扩展到下一个句号
        if end < len(text) and text[end - 1] not in "。！？\n":
            forward = -1
            for sep in ("。", "！", "？"):
                pos = text.find(sep, end, min(len(text), end + soft_extend))
                if pos != -1 and (forward == -1 or pos < forward):
                    forward = pos
            if forward != -1:
                end = forward + 1

        piece = text[start:end].strip()
        if piece:
            parts.append((piece, start))
        if end >= len(text):
            break
        next_start = end - overlap if end > start + overlap else start + step
        if next_start <= start:
            next_start = start + step
        start = next_start
    return parts


def _factual_sentences(body: str) -> list[str]:
    """抽出含数字/百分比/时刻的事实句，供 Embedding 强化。"""
    sents = re.split(r"(?<=[。！？；])", body)
    keyed: list[str] = []
    for s in sents:
        s = s.strip()
        if len(s) < 8:
            continue
        if re.search(r"\d+(\.\d+)?%|\d{1,2}:\d{2}|SLA|PDF|Word|退款|维护|备份|加密", s):
            keyed.append(s)
    return keyed


def _build_embed_text(source: str, title_path: str, section: str, body: str) -> str:
    """构造 Embedding 文本：强化标题路径与事实句，避免长正文淹没关键语义。"""
    facts = _factual_sentences(body)
    fact_block = "\n".join(facts[:4]) if facts else ""
    return (
        f"标题路径：{title_path}\n"
        f"章节：{section}\n"
        f"文档：{source}\n"
        f"标题路径：{title_path}\n"
        f"要点：{fact_block}\n"
        f"{body}"
    )


def chunk_documents(
    docs: list[tuple[str, str]],
    chunk_size: int = 480,
    overlap: int = 100,
) -> list[Chunk]:
    """标题感知切分：标题路径注入 + 超长正文滑动窗口兜底。"""
    soft_extend = getattr(config, "CHUNK_SOFT_EXTEND", 100)
    chunks: list[Chunk] = []
    counter = 0

    for source, content in docs:
        doc_title = _doc_title(source, content)
        for section, title_path, body in _split_by_heading_path(content, doc_title):
            windows = _window_split(body, chunk_size, overlap, soft_extend=soft_extend)
            for piece, offset in windows:
                embed_text = _build_embed_text(source, title_path, section, piece)
                chunks.append(
                    Chunk(
                        chunk_id=f"c{counter:04d}",
                        text=embed_text,
                        body=piece,
                        source=source,
                        section=section,
                        title_path=title_path,
                        start_char=offset,
                    )
                )
                counter += 1
    return chunks
