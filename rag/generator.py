"""基于检索结果生成回答：生成式（LLM）与抽取式（本地整理）分流。"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .chunker import Chunk


@dataclass
class AnswerResult:
    """问答结果。"""

    answer: str
    citations: list[dict]
    can_answer: bool
    mode: str  # "generative" | "extractive" | "refuse"


REFUSE_TEXT = "根据现有知识库，我无法回答该问题。"

# 生成式 Prompt（严格按任务要求）
GENERATIVE_PROMPT = """你是一个严谨的知识库问答助手。请基于以下参考资料回答问题。
如果参考资料不足以回答问题，必须明确回答："根据现有知识库，我无法回答该问题。"

参考资料：
{context}

问题：{question}

要求：
1. 优先使用参考资料中的信息，不要编造
2. 引用内容请标注[引用N]
3. 若资料矛盾，请指出并说明依据
4. 回答简洁，控制在300字以内

回答："""

_STOPWORDS = {
    "什么", "怎么", "如何", "是否", "可以", "请问", "一下", "多少", "哪个",
    "哪些", "有没有", "是不是", "今天", "明天", "怎么样", "为什么", "一般",
    "分别", "进行", "一个", "我们", "你们", "他们", "这个", "那个", "如果",
    "时间", "安排", "支持", "系统", "问题", "内容",
}

_DOMAIN_TERMS = [
    "星澜知识库", "星澜", "知识库", "企业版", "标准版", "个人版", "私有化",
    "单点登录", "SSO", "LDAP", "SLA", "文档", "导入", "备份", "增量备份",
    "全量备份", "计划维护", "维护", "加密", "退款", "响应时间", "首次响应",
    "监控", "故障", "权限", "检索", "Chunk", "Embedding", "Markdown",
    "PDF", "Word", "PPT", "TXT", "客户成功", "部门权限", "传输加密",
    "存储加密", "年度SLA", "可用性", "维护窗口",
]


def _unique_citations(hits: list[tuple[Chunk, float]]) -> list[dict]:
    """按来源+路径去重，保留最高分，并编号为引用N。"""
    best: dict[str, dict] = {}
    for chunk, score in hits:
        key = f"{chunk.source}|{chunk.title_path}"
        item = {
            "source": chunk.source,
            "section": chunk.section,
            "title_path": chunk.title_path,
            "chunk_id": chunk.chunk_id,
            "score": round(score, 4),
            "snippet": chunk.body[:160].replace("\n", " "),
        }
        if key not in best or score > best[key]["score"]:
            best[key] = item
    ordered = sorted(best.values(), key=lambda x: x["score"], reverse=True)
    for i, item in enumerate(ordered, 1):
        item["ref"] = i
    return ordered


def _format_context(hits: list[tuple[Chunk, float]]) -> str:
    """构造带 [引用N] 的参考资料文本。"""
    parts = []
    for i, (chunk, score) in enumerate(hits, 1):
        parts.append(
            f"[引用{i}] 来源={chunk.source}｜路径={chunk.title_path}｜相似度={score:.4f}\n"
            f"{chunk.body}"
        )
    return "\n\n".join(parts)


def _extract_keywords(question: str) -> list[str]:
    found: list[str] = []
    q_lower = question.lower()
    for term in _DOMAIN_TERMS:
        if term.lower() in q_lower:
            found.append(term)
    found.extend(re.findall(r"[A-Za-z0-9_\-.%]{2,}", question))
    chars = "".join(re.findall(r"[\u4e00-\u9fff]", question))
    for n in (3, 2):
        for i in range(max(len(chars) - n + 1, 0)):
            gram = chars[i : i + n]
            if gram in _STOPWORDS or gram in found:
                continue
            found.append(gram)
    return found


def _sentence_score(sent: str, keywords: list[str]) -> int:
    return sum(1 for k in keywords if k in sent)


def _is_near_duplicate(a: str, b: str) -> bool:
    def norm(s: str) -> str:
        return re.sub(r"^[\u4e00-\u9fff]{2,12}内容\d+：", "", s)

    na, nb = norm(a), norm(b)
    if na == nb:
        return True
    shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
    return len(shorter) >= 20 and shorter in longer


def _clean_sentence(sent: str) -> str:
    """清理抽取句：去掉编号前缀与残缺开头。"""
    sent = sent.strip(" \n\r\t；，、")
    sent = re.sub(r"^[\u4e00-\u9fffA-Za-z0-9_\-]{1,20}内容\d+[：:]\s*", "", sent)
    sent = re.sub(r"^外[）)]，?", "", sent)
    return sent.strip()


def _extractive_answer(question: str, hits: list[tuple[Chunk, float]]) -> str:
    """抽取式模式：整理要点并标注引用，避免原文整段直出。"""
    keywords = _extract_keywords(question)
    candidates: list[tuple[int, str, int]] = []  # score, sentence, ref_idx
    seen = set()

    for ref_idx, (chunk, _) in enumerate(hits, 1):
        sentences = re.split(r"(?<=[。！？；\n])", chunk.body)
        for sent in sentences:
            sent = _clean_sentence(sent)
            if len(sent) < 12 or sent in seen:
                continue
            if sent[0] in "）)、】》":
                continue
            score = _sentence_score(sent, keywords) if keywords else 1
            if re.search(r"\d+(\.\d+)?%|\d{1,2}:\d{2}|企业版|标准版|支持PDF|SLA|维护", sent):
                score += 2
            if score > 0:
                candidates.append((score, sent, ref_idx))
                seen.add(sent)

    candidates.sort(key=lambda x: x[0], reverse=True)
    selected: list[tuple[str, int]] = []
    for _, sent, ref_idx in candidates:
        if any(_is_near_duplicate(sent, prev) for prev, _ in selected):
            continue
        selected.append((sent, ref_idx))
        if len(selected) >= 3:
            break

    if not selected:
        # 仍无法整理出有效要点 -> 拒答语义由上层阈值兜底；此处给保守摘要
        first = _clean_sentence(hits[0][0].body.split("。")[0] + "。")
        if len(first) < 12:
            return REFUSE_TEXT
        selected = [(first, 1)]

    bullets = []
    for sent, ref_idx in selected:
        if not sent.endswith(("。", "！", "？", "；")):
            sent += "。"
        bullets.append(f"- {sent}[引用{ref_idx}]")

    cite_map = []
    for i, (chunk, score) in enumerate(hits, 1):
        cite_map.append(f"[引用{i}] {chunk.source}｜{chunk.title_path}")

    return (
        "【抽取式整理】以下要点根据检索片段归纳，未经大模型生成改写：\n"
        + "\n".join(bullets)
        + "\n\n引用对照：\n"
        + "\n".join(cite_map)
    )


def _generative_answer(
    question: str,
    hits: list[tuple[Chunk, float]],
    model: str,
    temperature: float,
    max_tokens: int,
    base_url: str | None,
    api_key: str | None = None,
    reasoning_effort: str = "high",
    enable_thinking: bool = True,
) -> str:
    """生成式模式：调用 DeepSeek / OpenAI 兼容接口（规定 Prompt）。"""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("未安装 openai 包，请执行: pip install openai") from exc

    key = (api_key or "").strip()
    if not key:
        key = (
            os.getenv("DEEPSEEK_API_KEY", "").strip()
            or os.getenv("OPENAI_API_KEY", "").strip()
        )
    if not key:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY / OPENAI_API_KEY")

    url = (
        base_url
        or os.getenv("DEEPSEEK_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "https://api.deepseek.com"
    )
    client = OpenAI(api_key=key, base_url=url)
    prompt = GENERATIVE_PROMPT.format(
        context=_format_context(hits),
        question=question,
    )

    create_kwargs: dict = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是一个严谨的知识库问答助手，只依据参考资料作答，不编造。",
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # DeepSeek V4：开启思考模式（按官方调用方式）
    if enable_thinking:
        create_kwargs["reasoning_effort"] = reasoning_effort
        create_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

    resp = client.chat.completions.create(**create_kwargs)
    message = resp.choices[0].message
    content = (getattr(message, "content", None) or "").strip()
    if content:
        return content
    # 部分思考模式下正文可能落在其他字段，做兜底
    reasoning = getattr(message, "reasoning_content", None) or ""
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    raise RuntimeError("模型返回空内容")


def generate_answer(
    question: str,
    hits: list[tuple[Chunk, float]],
    score_threshold: float,
    answer_mode: str = "ai",
    llm_model: str = "deepseek-v4-flash",
    llm_temperature: float = 0.2,
    llm_max_tokens: int = 800,
    openai_base_url: str | None = None,
    api_key: str | None = None,
    reasoning_effort: str = "high",
    enable_thinking: bool = True,
) -> AnswerResult:
    """根据检索命中结果生成最终回答。

    answer_mode:
      - ai / generative: 强制走大模型生成式
      - extractive: 强制走本地抽取式整理
    """
    citations = _unique_citations(hits) if hits else []
    mode = (answer_mode or "ai").strip().lower()
    if mode in {"generative", "llm", "deepseek"}:
        mode = "ai"
    if mode not in {"ai", "extractive"}:
        mode = "ai"

    if not hits or hits[0][1] < score_threshold:
        return AnswerResult(
            answer=REFUSE_TEXT,
            citations=[],
            can_answer=False,
            mode="refuse",
        )

    if mode == "ai":
        try:
            text = _generative_answer(
                question,
                hits,
                model=llm_model,
                temperature=llm_temperature,
                max_tokens=llm_max_tokens,
                base_url=openai_base_url,
                api_key=api_key,
                reasoning_effort=reasoning_effort,
                enable_thinking=enable_thinking,
            )
            refused = REFUSE_TEXT in text and len(text) < len(REFUSE_TEXT) + 40
            return AnswerResult(
                answer=text,
                citations=[] if refused else citations,
                can_answer=not refused,
                mode="generative",
            )
        except Exception as exc:  # noqa: BLE001
            fallback = _extractive_answer(question, hits)
            if fallback == REFUSE_TEXT:
                return AnswerResult(
                    answer=REFUSE_TEXT,
                    citations=[],
                    can_answer=False,
                    mode="refuse",
                )
            return AnswerResult(
                answer=(
                    f"{fallback}\n\n"
                    f"（说明：AI 生成调用失败，已降级为抽取式整理。原因：{exc}）"
                ),
                citations=citations,
                can_answer=True,
                mode="extractive",
            )

    text = _extractive_answer(question, hits)
    if text == REFUSE_TEXT:
        return AnswerResult(
            answer=REFUSE_TEXT,
            citations=[],
            can_answer=False,
            mode="refuse",
        )
    return AnswerResult(
        answer=text,
        citations=citations,
        can_answer=True,
        mode="extractive",
    )
