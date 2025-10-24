# tests/test_smoke.py
import unittest

try:
    from core.llm import UltraAdvancedLLM
    CORE_AVAILABLE = True
except Exception:
    CORE_AVAILABLE = False

@unittest.skipUnless(CORE_AVAILABLE, "core/llm が見つかりません")
class TestSmoke(unittest.TestCase):
    def test_generate_stub(self):
        llm = UltraAdvancedLLM()
        resp = llm.query("Hello world", top_k=1)
        # resp は dataclass Resp を期待
        self.assertTrue(hasattr(resp, "text"))
        self.assertIsInstance(resp.text, str)
        self.assertGreaterEqual(len(resp.text), 1)

    def test_save_load_state(self):
        llm = UltraAdvancedLLM()
        tmp = "test_state.json"
        try:
            llm.add_knowledge("k1", "name1", "fact", {"a":1})
            llm.save_state(tmp)
            llm2 = UltraAdvancedLLM()
            llm2.load_state(tmp)
            # kg に k1 があれば成功
            self.assertIn("k1", getattr(llm2.kg, "nodes", {}))
        finally:
            import os
            if os.path.exists(tmp):
                os.remove(tmp)
