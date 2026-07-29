"""轻量级 Web 演示服务。"""

from __future__ import annotations

import config  # noqa: F401  # 触发 .env 加载
from flask import Flask, jsonify, render_template, request

from rag import RAGPipeline

app = Flask(__name__)
pipeline = RAGPipeline()
pipeline.build_index()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    has_key = bool(config.get_llm_api_key())
    return jsonify(
        {
            "ok": True,
            "chunks": len(pipeline.index.chunks),
            "kb_dir": str(pipeline.kb_dir),
            "llm_model": config.LLM_MODEL,
            "llm_ready": has_key,
            "default_mode": config.DEFAULT_ANSWER_MODE,
        }
    )


@app.post("/api/ask")
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    answer_mode = (data.get("mode") or config.DEFAULT_ANSWER_MODE or "ai").strip().lower()
    if answer_mode in {"generative", "llm", "deepseek", "ai生成", "生成式"}:
        answer_mode = "ai"
    if answer_mode in {"抽取式", "extract"}:
        answer_mode = "extractive"
    if answer_mode not in {"ai", "extractive"}:
        return jsonify({"error": "mode 仅支持 ai 或 extractive"}), 400

    if answer_mode == "ai" and not config.get_llm_api_key():
        return jsonify({"error": "未配置 DEEPSEEK_API_KEY，无法使用 AI 生成模式"}), 400

    result = pipeline.ask(question, answer_mode=answer_mode)
    return jsonify(
        {
            "question": question,
            "answer": result.answer,
            "citations": result.citations,
            "can_answer": result.can_answer,
            "mode": result.mode,
            "requested_mode": answer_mode,
            "llm_model": config.LLM_MODEL if answer_mode == "ai" else None,
        }
    )


@app.post("/api/rebuild")
def rebuild():
    n = pipeline.build_index(force_rebuild=True)
    return jsonify({"ok": True, "chunks": n})


if __name__ == "__main__":
    print("RAG Web Demo -> http://127.0.0.1:5000")
    print(f"LLM: {config.LLM_MODEL} @ {config.LLM_BASE_URL}")
    print(f"API Key ready: {bool(config.get_llm_api_key())}")
    app.run(host="127.0.0.1", port=5000, debug=False)
