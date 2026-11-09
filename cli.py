"""Ask the Black history Q&A from the terminal.

    python cli.py "When was the Emancipation Proclamation issued?"
    python cli.py --mode agent "Where did the Great Migration take Black Americans, and why?"
    python cli.py            # interactive mode
"""
import argparse

from app.service import ask


def show(question: str, mode: str) -> None:
    result = ask(question, mode=mode, verbose=(mode == "agent"))
    print(f"\n{result['answer']}")
    if result.get("citations"):
        print(f"Sources: {', '.join(result['citations'])}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Black history Q&A")
    parser.add_argument("question", nargs="?", help="question to ask (omit for interactive mode)")
    parser.add_argument("--mode", choices=["rag", "agent", "none"], default="rag")
    args = parser.parse_args()

    if args.question:
        show(args.question, args.mode)
        return
    print(f"Black history Q&A ({args.mode} mode). Empty line to quit.")
    while q := input("> ").strip():
        show(q, args.mode)


if __name__ == "__main__":
    main()
