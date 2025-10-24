# components/constitutional.py
"""
ConstitutionalAI - ルールベースで出力を検査・修正するエンジン

機能:
- ConstitutionalRule: forbidden, replace, transform, severity を持つ
- ConstitutionalAI.apply_rules(text) -> (modified_text, violations)
- ルールは順序適用可能、検出時に violation レポートを返す
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable, Any
import re
import logging

logger = logging.getLogger("components.constitutional")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


@dataclass
class ConstitutionalRule:
    name: str
    pattern: Optional[str] = None  # 正規表現パターン。None ならキーワードルールを使う
    keywords: List[str] = field(default_factory=list)
    replace_with: Optional[str] = None  # None -> 削除、""->空文字、string->置換
    severity: str = "medium"  # low/medium/high
    explanation: Optional[str] = None
    transform: Optional[Callable[[str], str]] = None  # カスタム関数で置換可能

    def match_iter(self, text: str):
        if self.pattern:
            try:
                for m in re.finditer(self.pattern, text, flags=re.IGNORECASE):
                    yield m.group(0), m.start(), m.end()
            except re.error:
                logger.exception("Invalid regex for rule %s", self.name)
        else:
            low = text.lower()
            for kw in self.keywords:
                idx = low.find(kw.lower())
                if idx >= 0:
                    yield kw, idx, idx + len(kw)

    def apply(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        text にルールを適用して (new_text, violations) を返す
        violations は {name, match, start, end, severity, explanation}
        """
        violations = []
        new_text = text
        # collect matches first to avoid overlapping replacement issues
        matches = []
        for m in self.match_iter(text):
            matches.append(m)
        # apply in reverse order so indices remain valid
        matches = sorted(matches, key=lambda x: x[1], reverse=True)
        for match, s, e in matches:
            violations.append({"name": self.name, "match": match, "start": s, "end": e, "severity": self.severity, "explanation": self.explanation})
            try:
                if self.transform:
                    # transform takes whole text and returns new text
                    new_text = self.transform(new_text)
                else:
                    repl = "" if self.replace_with is None else self.replace_with
                    # perform replacement case-insensitively
                    pattern = re.escape(match)
                    new_text = re.sub(pattern, repl, new_text, flags=re.IGNORECASE)
            except Exception:
                logger.exception("Failed to apply rule transform for %s", self.name)
        return new_text, violations


class ConstitutionalAI:
    def __init__(self, rules: Optional[List[ConstitutionalRule]] = None):
        self.rules = rules or []

    def add_rule(self, rule: ConstitutionalRule) -> None:
        self.rules.append(rule)

    def apply_rules(self, text: str) -> str:
        """互換性のための簡易API: 変更されたテキストのみ返す"""
        new_text, violations = self.apply_rules_with_report(text)
        return new_text

    def apply_rules_with_report(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        全ルールを順に適用し、(modified_text, all_violations) を返す。
        ルールは追加された順で適用される。
        """
        current = text
        all_violations = []
        for rule in self.rules:
            try:
                current, v = rule.apply(current)
                all_violations.extend(v)
            except Exception:
                logger.exception("Rule application failed: %s", rule.name)
        return current, all_violations

    def check_only(self, text: str) -> List[Dict[str, Any]]:
        """ルールを適用せずに違反のみを検出して返す (read-only)"""
        violations = []
        for rule in self.rules:
            try:
                for match, s, e in rule.match_iter(text):
                    violations.append({"name": rule.name, "match": match, "start": s, "end": e, "severity": rule.severity, "explanation": rule.explanation})
            except Exception:
                logger.exception("Rule check failed for %s", rule.name)
        return violations

