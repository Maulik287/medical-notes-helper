"""Command-line entry point: `python -m app.cli --help`."""
import argparse
import sys
from typing import List, Optional

from app.core import SuggestionEngine, clean_text, milestone_1, prepare_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medical-notes-helper",
        description="Clinic staff use this CLI tool to draft administrative sections of medical notes by generating next-word suggestions, ensuring consistency and reducing manual typing."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    
    # Milestone baseline check
    sub.add_parser("hello", help="check that everything is wired up")
    
    # Milestone 1: Inspect curated administrative corpus
    curate_parser = sub.add_parser("curate", help="inspect the anonymized administrative medical corpus")
    curate_parser.add_argument("--show-samples", action="store_true", help="show sample sentences from corpus")
    
    # Milestone 3 & 4: Next-word suggestions
    suggest_parser = sub.add_parser("suggest", help="suggest the most probable next words for given medical note text")
    suggest_parser.add_argument("prompt", type=str, help="initial phrase or sentence fragment")
    suggest_parser.add_argument("--top-k", type=int, default=3, help="number of next-word suggestions to return")
    
    # Milestone 3 & 4: Auto-complete note
    complete_parser = sub.add_parser("complete", help="auto-complete an administrative medical note from a prompt")
    complete_parser.add_argument("prompt", type=str, help="initial phrase or sentence fragment")
    complete_parser.add_argument("--max-words", type=int, default=6, help="maximum number of words to generate")
    
    # Milestone 4: Interactive drafting loop
    sub.add_parser("interactive", help="start an interactive drafting session with live suggestions")
    
    return parser


def run_interactive() -> None:
    print("\n--- Medical Notes Helper: Interactive Drafting Mode ---")
    print("Type a phrase to receive next-word suggestions. Type 'exit' or 'quit' to stop.\n")
    engine = SuggestionEngine()
    
    while True:
        try:
            line = input("Draft > ").strip()
            if not line:
                continue
            if line.lower() in {"exit", "quit"}:
                print("Exiting interactive draft session.")
                break
                
            suggestions = engine.suggest_next_words(line, top_k=3)
            print("Next-word suggestions:")
            for i, (word, prob) in enumerate(suggestions, 1):
                print(f"  [{i}] {word} (p={prob:.4f})")
            completion = engine.complete_text(line, max_words=5)
            print(f"Auto-completion: {completion}\n")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    
    if args.command == "hello":
        print("Medical Notes Helper is ready. Open README.md for your milestones.")
    elif args.command == "curate":
        info = milestone_1()
        print(f"Administrative Medical Corpus:")
        print(f"  Sentences: {info['num_sentences']}")
        print(f"  Total Tokens: {info['total_tokens']}")
        print(f"  Vocabulary Size: {info['vocab_size']}")
        if args.show_samples:
            print("  Sample Tokenized Sentence:")
            print(f"    {' '.join(info['sample_sentence'])}")
    elif args.command == "suggest":
        engine = SuggestionEngine()
        suggestions = engine.suggest_next_words(args.prompt, top_k=args.top_k)
        print(f"Suggestions for '{args.prompt}':")
        for i, (word, prob) in enumerate(suggestions, 1):
            print(f"  {i}. {word} (p = {prob:.4f})")
    elif args.command == "complete":
        engine = SuggestionEngine()
        completed = engine.complete_text(args.prompt, max_words=args.max_words)
        print(f"Completed note:\n  {completed}")
    elif args.command == "interactive":
        run_interactive()


if __name__ == "__main__":
    main()
