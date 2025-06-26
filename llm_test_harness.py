#!/usr/bin/env python3
"""
Refined test harness for SteamRecs functions with a single LLM checkpoint load,
plus an interactive REPL mode to experiment without restarting.
"""
from utils.price_tracker import parse_steam_price_request
from utils.translation import detect_language, translate_to_english
from data.data_loader import load_games_csv
from utils.steam_utils import analyze_profile
from utils.llm_processing import (
    extract_game_titles,
    infer_user_preferences_with_llm,
    parse_user_intent,
)
from models.transformer_model import load_transformer_model
import sys
import os
import json
import argparse

# Ensure utils and data on path when running standalone
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, 'utils'))
sys.path.append(os.path.join(SCRIPT_DIR, 'data'))


def run_test(test, args, context, games_df):
    if test == 'titles':
        if not args.message:
            print("--message is required for titles test.")
            return
        print("Extracted titles:", extract_game_titles(args.message, context))

    elif test == 'preferences':
        if not args.message:
            print("--message is required for preferences test.")
            return
        print(json.dumps(infer_user_preferences_with_llm(
            args.message, context), indent=2))

    elif test == 'intent':
        if not args.message:
            print("--message is required for intent test.")
            return
        print("Detected intent:", parse_user_intent(args.message, context))

    elif test == 'price':
        if not args.appid:
            print("--appid is required for price test.")
            return
        print(json.dumps(parse_steam_price_request(args.appid), indent=2))

    elif test == 'lang':
        if not args.message:
            print("--message is required for lang test.")
            return
        lang = detect_language(args.message)
        translation = translate_to_english(args.message)
        print("Detected language:", lang)
        print("Translation to English:", translation)

    elif test == 'profile':
        if not args.profile:
            print("--profile is required for profile test.")
            return
        with open(args.profile, 'r', encoding='utf-8') as f:
            profile_data = json.load(f)
        summary = analyze_profile(profile_data, games_df)
        print(json.dumps(summary, indent=2))

    elif test == 'data':
        # Use the preloaded games_df if available, otherwise load it lazily
        if games_df is not None:
            df = games_df
        else:
            df = load_games_csv()
        print(
            f"Loaded games CSV with {len(df)} rows. Columns: {list(df.columns)}")


def repl(context, games_df):
    print("Entering interactive REPL mode. Type 'help' for commands, 'exit' to quit.")
    while True:
        try:
            cmd = input('steamrecs> ').strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting REPL.")
            break
        if not cmd:
            continue
        if cmd.lower() in ('exit', 'quit'):
            print("Exiting REPL.")
            break
        if cmd.lower() == 'help':
            print("Commands:")
            print("  titles <message>")
            print("  preferences <message>")
            print("  intent <message>")
            print("  price <appid>")
            print("  lang <message>")
            print("  profile <path_to_json>")
            print("  data")
            print("  exit")
            continue
        parts = cmd.split(' ', 1)
        test = parts[0]
        rest = parts[1] if len(parts) > 1 else ''
        # build args namespace

        class A:
            pass
        a = A()
        a.message = None
        a.appid = None
        a.profile = None
        # assign from rest
        if test in ('titles', 'preferences', 'intent', 'lang'):
            a.message = rest
        elif test == 'price':
            try:
                a.appid = int(rest)
            except ValueError:
                print("Invalid appid. Must be integer.")
                continue
        elif test == 'profile':
            a.profile = rest
        # run
        run_test(test, a, context, games_df)


def main():
    parser = argparse.ArgumentParser(
        description="Test various SteamRecs functions with one model load or in REPL mode."
    )
    parser.add_argument(
        '-t', '--tests',
        nargs='+',
        choices=[
            'titles', 'preferences', 'intent',
            'price', 'lang', 'profile', 'data'
        ],
        help="Which tests to run; you can specify multiple, e.g. --tests titles intent"
    )
    parser.add_argument(
        '-m', '--message',
        type=str,
        help="User message for titles/preferences/intent/lang tests"
    )
    parser.add_argument(
        '--appid',
        type=int,
        help="Steam appid for price tracking test"
    )
    parser.add_argument(
        '--profile',
        type=str,
        help="Path to a sample Steam profile JSON for profile test"
    )
    parser.add_argument(
        '--repl',
        action='store_true',
        help="Start in interactive REPL mode"
    )
    args = parser.parse_args()

    # Load LLM once if needed
    llm_tests = {'titles', 'preferences', 'intent'}
    context = None
    if args.repl or (args.tests and any(name in llm_tests for name in args.tests)):
        tokenizer, model = load_transformer_model()
        if not tokenizer or not model:
            print("❌ Model/tokenizer failed to load.")
            sys.exit(1)
        context = type(
            'Dummy', (), {'bot_data': {'tokenizer': tokenizer,
                                       'transformer_model': model}}
        )()

    # Preload data if needed
    games_df = None
    if (args.tests and ('data' in args.tests or 'profile' in args.tests)) or args.repl:
        games_df = load_games_csv()

    if args.repl:
        repl(context, games_df)
        return

    if not args.tests:
        print("No tests specified. Use --tests or --repl for interactive mode.")
        sys.exit(1)

    # Run specified tests
    for test in args.tests:
        print(f"\n--- TEST: {test} ---\n")
        run_test(test, args, context, games_df)


if __name__ == '__main__':
    main()
