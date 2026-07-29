#!/usr/bin/env python3
"""导出知识库切分为 docs/data/kb.json，供 GitHub Pages 静态演示使用。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from rag.chunker import chunk_documents, load_documents


def main() -> int:
    docs = load_documents(config.KB_DIR)
    chunks = chunk_documents(
        docs,
        chunk_size=config.CHUNK_SIZE,
        overlap=config.CHUNK_OVERLAP,
    )
    payload = {
        "model_note": "static-pages-export",
        "score_threshold": 0.12,
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "body": c.body,
                "source": c.source,
                "section": c.section,
                "title_path": c.title_path,
            }
            for c in chunks
        ],
    }
    out = ROOT / "docs" / "data" / "kb.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"exported {len(chunks)} chunks -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
