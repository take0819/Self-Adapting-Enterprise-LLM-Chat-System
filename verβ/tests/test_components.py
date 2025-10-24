# tests/test_components.py
import unittest

# components が無ければテストをスキップする仕組み
try:
    from components.vdb import VDB
    from components.tree_of_thoughts import TreeOfThoughts, ThoughtNode
    from components.debate import DebateSystem, DebateArgument, DebateResult
    from components.critic import CriticSystem
    from components.constitutional import ConstitutionalAI, ConstitutionalRule
    from components.meta_learning import MetaLearner
    COMPONENTS_AVAILABLE = True
except Exception:
    COMPONENTS_AVAILABLE = False

@unittest.skipUnless(COMPONENTS_AVAILABLE, "components/ が見つかりません")
class TestComponents(unittest.TestCase):
    def test_vdb_add_query(self):
        v = VDB(dim=16, use_numpy=False) if hasattr(VDB, "__init__") else VDB(dim=16)
        v.clear()
        v.add("id1", text="hello world")
        v.add("id2", text="another document")
        res = v.query("hello", top_k=2)
        self.assertIsInstance(res, list)
        self.assertTrue(len(res) >= 1)
        self.assertIn("id1", {r["id"] for r in res})

    def test_tree_of_thoughts(self):
        t = TreeOfThoughts()
        root = t.new_root("start")
        t.expand(root, lambda s, n: [f"{s} candidate {i}" for i in range(n)], n_candidates=3)
        results = t.search(root, mode="beam", beam_width=2, depth=2)
        self.assertIsInstance(results, list)

    def test_debate(self):
        def make_gen(i):
            def gen(prompt, round_idx, prev):
                return f"agent_{i} says about {prompt} in round {round_idx}"
            return gen
        ds = DebateSystem()
        gens = [make_gen(0), make_gen(1)]
        res = ds.debate("topic", gens)
        self.assertIsInstance(res, DebateResult)
        self.assertTrue(hasattr(res, "winner"))

    def test_critic(self):
        c = CriticSystem()
        score = c.assess("This is a reasonably long and informative sentence intended for testing.")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        expl = c.explain("short text")
        self.assertIsInstance(expl, list)

    def test_constitutional(self):
        r = ConstitutionalRule(name="no-bad", keywords=["badword"], replace_with="[redacted]", severity="high")
        ca = ConstitutionalAI([r])
        orig = "this contains badword and should be redacted"
        new = ca.apply_rules(orig)
        self.assertNotIn("badword", new)
        violations = ca.check_only("badword appears")
        self.assertTrue(len(violations) >= 1)

    def test_meta_learner(self):
        ml = MetaLearner(memory_limit=10)
        ml.add_experience("in", "out", reward=0.5)
        summary = ml.summarize_experiences()
        self.assertIn("n", summary)
        res = ml.adapt()
        self.assertIn("summary", res)
