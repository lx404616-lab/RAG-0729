"""RAG 知识问答系统 CLI 入口。"""

from __future__ import annotations

import argparse
import sys

from rag import RAGPipeline


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
    return parser


DEMO_QUESTIONS = [
    "星澜知识库企业版有哪些功能？",
    "标准版和企业版的年度SLA分别是多少？",
    "系统支持哪些文档格式导入？",
    "计划维护一般安排在什么时间？",
    "今天北京的天气怎么样？",  # 知识库无答案
]


def run_demo(pipeline: RAGPipeline, answer_mode: str | None = None) -> None:
    print("=" * 60)
    print("演示问答（含知识库无法回答的样例）")
    print("=" * 60)
    for q in DEMO_QUESTIONS:
        print(f"\n【问题】{q}")
        print("-" * 40)
        result = pipeline.ask(q, answer_mode=answer_mode)
        print(pipeline.format_result(result))


def interactive_loop(pipeline: RAGPipeline, answer_mode: str | None = None) -> None:
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
    import config  # noqa: F401

    args = build_parser().parse_args(argv)
    pipeline = RAGPipeline()
    n = pipeline.build_index(force_rebuild=args.rebuild)
    print(f"向量索引就绪，共 {n} 个文本块。索引目录: {pipeline.index_dir}")

    if args.demo:
        run_demo(pipeline, answer_mode=args.mode)
        return 0

    if args.question:
        result = pipeline.ask(args.question, answer_mode=args.mode)
        print(pipeline.format_result(result))
        return 0

    interactive_loop(pipeline, answer_mode=args.mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
