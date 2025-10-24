#!/usr/bin/env python3
# bin/main.py
"""
エントリポイント（軽量）。
コマンドライン:
  --query "text"        : 単発クエリを投げて結果を表示
  --interactive         : 対話モードを開始 (cli.interactive.InteractiveChat)
  --save-state PATH     : 実行後に LLM の state を保存
"""
from __future__ import annotations
import argparse
import sys
import os

from utils.logging import get_logger
logger = get_logger("bin.main")

try:
    from core.llm import UltraAdvancedLLM
except Exception:
    UltraAdvancedLLM = None

try:
    from cli.interactive import InteractiveChat
except Exception:
    InteractiveChat = None

def main(argv=None):
    parser = argparse.ArgumentParser(prog="ultra-llm")
    parser.add_argument("--query", "-q", type=str, help="single query to LLM")
    parser.add_argument("--interactive", "-i", action="store_true", help="run interactive mode")
    parser.add_argument("--history", type=str, default=os.path.expanduser("~/.ultra_llm_history.json"), help="history path for interactive mode")
    parser.add_argument("--save-state", type=str, help="path to save LLM state after run")
    args = parser.parse_args(argv)

    if args.query:
        if UltraAdvancedLLM is None:
            logger.error("core.llm.UltraAdvancedLLM が見つかりません。core/ を確認してください。")
            return 2
        llm = UltraAdvancedLLM()
        resp = llm.query(args.query, top_k=1)
        if hasattr(resp, "text"):
            print(resp.text)
        else:
            print(str(resp))
        if args.save_state:
            try:
                llm.save_state(args.save_state)
                logger.info("state saved to %s", args.save_state)
            except Exception:
                logger.exception("failed to save state")
        return 0

    if args.interactive:
        if InteractiveChat is None:
            logger.error("cli.interactive.InteractiveChat が見つかりません。cli/ を確認してください。")
            return 2
        try:
            chat = InteractiveChat(llm=UltraAdvancedLLM() if UltraAdvancedLLM is not None else None, history_path=args.history)
            chat.run()
        except Exception:
            logger.exception("interactive mode failed")
            return 3
        return 0

    parser.print_help()
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
