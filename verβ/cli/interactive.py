# cli/interactive.py
"""
対話モード用の簡易 CLI / InteractiveChat クラス。

使い方:
    from core.llm import UltraAdvancedLLM
    from cli.interactive import InteractiveChat
    chat = InteractiveChat(llm=UltraAdvancedLLM())
    chat.run()  # 標準入力ループ
"""
from __future__ import annotations
import readline
import json
import os
import time
from typing import Optional, List, Dict, Any

try:
    from core.llm import UltraAdvancedLLM  # type: ignore
except Exception:
    UltraAdvancedLLM = None  # type: ignore

from utils.logging import get_logger

logger = get_logger("cli.interactive")


class InteractiveChat:
    def __init__(self, llm: Optional[UltraAdvancedLLM] = None, history_path: Optional[str] = None, auto_save: bool = True):
        if llm is None:
            if UltraAdvancedLLM is None:
                raise RuntimeError("UltraAdvancedLLM が見つかりません。core/llm.py を配置してください。")
            llm = UltraAdvancedLLM()
        self.llm = llm
        self.history_path = history_path
        self.auto_save = auto_save
        self.history: List[Dict[str, Any]] = []
        if history_path:
            try:
                if os.path.exists(history_path):
                    with open(history_path, "r", encoding="utf-8") as f:
                        self.history = json.load(f)
            except Exception:
                logger.exception("履歴の読み込みに失敗しました: %s", history_path)

    def _save_history(self):
        if not self.history_path:
            return
        try:
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception:
            logger.exception("履歴の保存に失敗しました: %s", self.history_path)

    def handle_input(self, text: str) -> str:
        """プロンプトを LLM に渡して応答を得る（同期）"""
        t0 = time.time()
        resp = self.llm.query(text, top_k=1)
        latency = time.time() - t0
        out_text = resp.text if hasattr(resp, "text") else str(resp)
        record = {"prompt": text, "response": out_text, "latency": latency, "ts": time.time()}
        self.history.append(record)
        if self.auto_save:
            self._save_history()
        return out_text

    def run(self, welcome: str = "interactive mode. type 'exit' or Ctrl-C to quit."):
        print(welcome)
        try:
            while True:
                try:
                    text = input("> ").strip()
                except EOFError:
                    print()
                    break
                if not text:
                    continue
                if text.lower() in ("exit", "quit", "q"):
                    break
                try:
                    out = self.handle_input(text)
                    print(out)
                except Exception:
                    logger.exception("LLM 呼び出しで例外")
                    print("Error while processing. See logs.")
        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            if self.history_path:
                self._save_history()
