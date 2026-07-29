"""RAG 知识问答系统 CLI 入口。"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

# 降低第三方库噪音（避免 PowerShell 把进度条 stderr 当成失败）
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="基于本地知识库的简单 RAG 知识问答系统",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="强制重新切分文档并重建向量索引",
    )
    parser.add_argument(
        "-q",
        "--question",
        type=str,
        default=None,
        help="单次提问；不传则进入交互模式",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="运行内置演示问答（含无法回答样例）",
    )
    parser.add_argument(
        "--mode",
        choices=["ai", "extractive"],
        default=None,
        help="回答模式：ai=DeepSeek 生成式，extractive=本地抽取式",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="输出详细调试信息（含完整异常栈）",
    )
    return parser


DEMO_QUESTIONS = [
    "星澜知识库企业版有哪些功能？",
    "标准版和企业版的年度SLA分别是多少？",
    "系统支持哪些文档格式导入？",
    "计划维护一般安排在什么时间？",
    "今天北京的天气怎么样？",
]


def run_demo(pipeline, answer_mode: str | None = None) -> None:
    print("=" * 60)
    print("演示问答（含知识库无法回答的样例）")
    print("=" * 60)
    for q in DEMO_QUESTIONS:
        print(f"\n【问题】{q}")
        print("-" * 40)
        result = pipeline.ask(q, answer_mode=answer_mode)
        print(pipeline.format_result(result))


def interactive_loop(pipeline, answer_mode: str | None = None) -> None:
    print("已进入交互问答模式。输入问题后回车；输入 exit / quit 退出。")
    while True:
        try:
            question = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("已退出。")
            break
        result = pipeline.ask(question, answer_mode=answer_mode)
        print("\n" + pipeline.format_result(result))


def main(argv: list[str] | None = None) -> int:
    # 保证可从任意目录运行：切换到项目根目录
    root = Path(__file__).resolve().parent
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import config
    from rag import RAGPipeline

    args = build_parser().parse_args(argv)
    mode = (args.mode or config.DEFAULT_ANSWER_MODE or "ai").strip().lower()
    has_key = bool(config.get_llm_api_key())

    print(f"[信息] 工作目录: {root}")
    print(f"[信息] 回答模式: {mode}")
    print(f"[信息] DeepSeek API Key: {'已配置' if has_key else '未配置'}")
    if mode == "ai" and not has_key:
        print(
            "[警告] 未检测到 DEEPSEEK_API_KEY。"
            "AI 模式将失败并自动降级为抽取式；"
            "也可加 --mode extractive，或在项目根目录配置 .env。"
        )
    sys.stdout.flush()

    try:
        pipeline = RAGPipeline()
        print("[信息] 正在加载/构建向量索引（首次加载 BGE 模型可能较慢）…")
        sys.stdout.flush()
        n = pipeline.build_index(force_rebuild=args.rebuild)
        print(f"[信息] 向量索引就绪，共 {n} 个文本块。目录: {pipeline.index_dir}")
        sys.stdout.flush()

        if args.demo:
            run_demo(pipeline, answer_mode=args.mode)
            return 0

        if args.question:
            question = args.question.strip()
            print(f"[信息] 问题: {question}")
            print("[信息] 正在检索并生成回答…")
            sys.stdout.flush()
            result = pipeline.ask(question, answer_mode=args.mode)
            print()
            print(pipeline.format_result(result))
            return 0

        interactive_loop(pipeline, answer_mode=args.mode)
        return 0
    except Exception as exc:  # noqa: BLE001
        print("\n[错误] 运行失败：", file=sys.stderr)
        print(f"  类型: {type(exc).__name__}", file=sys.stderr)
        print(f"  详情: {exc}", file=sys.stderr)
        print("\n常见原因：", file=sys.stderr)
        print("  1) 未在项目根目录（含 main.py / 知识库）执行", file=sys.stderr)
        print("  2) 依赖未安装：py -3 -m pip install -r requirements.txt", file=sys.stderr)
        print("  3) AI 模式缺 Key：请配置 .env 中的 DEEPSEEK_API_KEY", file=sys.stderr)
        print("  4) PowerShell 中文参数乱码：可改用交互模式 py -3 main.py", file=sys.stderr)
        if args.verbose:
            print("\n—— 完整异常栈 ——", file=sys.stderr)
            traceback.print_exc()
        else:
            print("\n可加 -v 查看完整异常栈，例如：", file=sys.stderr)
            print('  py -3 main.py -v -q "企业版年度SLA是多少？"', file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
