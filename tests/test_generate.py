"""Ported from golem-15 gen/engine/test_controller.py with a synthetic graph fixture
(the 110-node Golem graph stays private), plus teacher-loop tests with a fake server."""
import copy, json, unittest
from pathlib import Path
from character_pipeline.generate.controller import Controller, golem_system, render, hygiene, retained
from character_pipeline.generate import teacher

ROOT = Path(__file__).resolve().parents[1]
G = {"nodes": [{"id": x, "label": x.replace("_", " "), "description": f"{x} description",
                "valence": v, "arousal": a}
               for x, v, a in [("machine_intelligence", .1, .5), ("evolution", -.2, .3),
                               ("body_history", .0, .6), ("task", .2, .2), ("silence", -.1, .1)]],
     "edges": [{"source": "machine_intelligence", "target": "evolution"},
               {"source": "evolution", "target": "body_history"},
               {"source": "machine_intelligence", "target": "task"},
               {"source": "task", "target": "silence"}]}
V = [[1., 0.] for _ in G["nodes"]]


class ControllerTests(unittest.TestCase):
    def test_reproducible_bounded_adjacent_immutable(self):
        snapshot = copy.deepcopy(G); a = Controller(G, V, 1); b = Controller(G, V, 1)
        for text in ["machine intelligence task", "nothing at all", "evolution body history", "unfamiliar"]:
            x = a.step(text, [1., 0.]); self.assertEqual(x, b.step(text, [1., 0.]))
            self.assertTrue(all(0 <= v <= 1 for v in x["activation"].values()))
            self.assertLessEqual(len(x["walk"]), 3); self.assertEqual(len(x["walk"]), len(set(x["walk"])))
            for p, q in zip(x["walk"], x["walk"][1:]):
                self.assertIn(q, a.adj[p])
        self.assertEqual(G, snapshot)

    def test_isolated(self):
        a = Controller(G, V, 1); b = Controller(G, V, 1)
        a.step("intelligence", [1., 0.]); self.assertTrue(all(v == 0 for v in b.a.values()))

    def test_spread_simultaneous_decay(self):
        g = {"nodes": [{"id": x, "label": x, "description": x, "valence": 0, "arousal": 0} for x in ["aaa", "bbb", "ccc"]],
             "edges": [{"source": "aaa", "target": "bbb"}, {"source": "bbb", "target": "ccc"}]}
        c = Controller(g, [[0, 1]] * 3, 1); c.a = {"aaa": 1., "bbb": 0., "ccc": 0.}
        s = c.step("unmatched", [1, 0])
        self.assertAlmostEqual(s["activation"]["aaa"], .65); self.assertAlmostEqual(s["activation"]["bbb"], .0975)
        self.assertEqual(s["activation"]["ccc"], 0)

    def test_roles_and_anchors(self):
        s = {"material": "node", "mood": [0, .4], "target_words": 5}
        p = render(golem_system("CORE\n", "STYLE\n", s), [{"speaker": "partner", "text": "OTHER"},
                   {"speaker": "golem", "text": "SELF"}, {"speaker": "partner", "text": "LATEST"}], "golem")
        self.assertIn("CORE\n\n\nSTYLE\n", p); self.assertIn("<|im_start|>assistant\nSELF", p)
        self.assertIn("<|im_start|>user\nLATEST", p)
        self.assertIn("You are AINEKO,", golem_system("C", "S", s, name="AINEKO"))

    def test_cut_pair(self):
        h = [{"speaker": "partner", "text": "a"}, {"speaker": "golem", "text": "b", "flags": []},
             {"speaker": "partner", "text": "c"}, {"speaker": "golem", "text": "bad", "flags": ["bad"]},
             {"speaker": "partner", "text": "d"}]
        rows, cut = retained(h); self.assertEqual(cut, 3); self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1], {"role": "assistant", "content": "b"})

    def test_filters(self):
        for t in ["- a", "1. b", "I am happy to help you", "the echoes of the void", "hello—friend",
                  "你好", "<think>no", "*smiles*", "GOLEM: hello"]:
            self.assertTrue(hygiene(t), t)
        self.assertEqual(hygiene("A wheel bends under load."), [])
        self.assertIn("role_marker", hygiene("AINEKO: hi", name="AINEKO"))


class TeacherTests(unittest.TestCase):
    def test_scenario_loop_keeps_raw_and_flags(self):
        scenario = json.loads((ROOT / "fixtures/scenarios/g6_ledger.jsonl").read_text().splitlines()[2])
        replies = iter([{"content": "The wheel rubs; no tension was measured here.", "stop_type": "eos"},
                        {"content": "I measured the spoke tension at 110 kgf.", "stop_type": "eos"},
                        {"content": "- check spoke tension", "stop_type": "limit", "stopped_limit": True}])
        seen = []
        def fake(url, body):
            seen.append(body); return next(replies)
        row = teacher.run_scenario(scenario, "SYSTEM", "http://x", 7, "GOLEM", call=fake)
        golem = [t for t in row["history"] if t["speaker"] == "golem"]
        self.assertEqual(len(golem), 3)
        self.assertEqual(golem[0]["flags"], [])
        self.assertIn("bullets", golem[2]["flags"]); self.assertIn("incomplete_or_unknown_finish", golem[2]["flags"])
        self.assertEqual([b["seed"] for b in seen], [7, 8, 9])
        self.assertFalse(row["training_eligible"])
        self.assertEqual(seen[0]["temperature"], 0.7)


if __name__ == "__main__":
    unittest.main()
