"""RAG 系统配置参数。"""

from __future__ import annotations

import os
from pathlib import Path

# 路径
BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "知识库"
INDEX_DIR = BASE_DIR / "vector_store"


def _load_dotenv(path: Path) -> None:
    """从 .env 加载环境变量（不覆盖已有环境变量）。"""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(BASE_DIR / ".env")

# 文档切分（Markdown 标题感知 + 滑动窗口兜底）
CHUNK_SIZE = 480
CHUNK_OVERLAP = 100
CHUNK_SOFT_EXTEND = 100

# 检索
TOP_K = 4
SCORE_THRESHOLD = 0.55

# 向量化（BGE 稠密向量 Embedding）
BGE_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
BGE_BATCH_SIZE = 32
BGE_DEVICE = None

# 生成模型：DeepSeek V4 Flash（OpenAI 兼容接口）
LLM_PROVIDER = "deepseek"
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 800
LLM_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "high")
LLM_ENABLE_THINKING = True

# 兼容：优先 DEEPSEEK_API_KEY，其次 OPENAI_API_KEY
def get_llm_api_key() -> str:
    return (
        os.getenv("DEEPSEEK_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )


# 默认回答模式：ai | extractive
DEFAULT_ANSWER_MODE = os.getenv("DEFAULT_ANSWER_MODE", "ai")
