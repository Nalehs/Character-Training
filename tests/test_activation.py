import json
import struct
import tempfile
import unittest
from pathlib import Path

from character_pipeline.activation import neuron, persona

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

EVIDENCE = Path(__file__).resolve().parents[1] / "evidence" / "neuron-steer-pilot" / "vectors.json"


def bf16_bytes(matrix):
    f32 = np.asarray(matrix, dtype=np.float32)
    return (f32.view(np.uint32) >> 16).astype("<u2").tobytes()


@unittest.skipIf(np is None, "numpy not installed")
class Neuron(unittest.TestCase):
    def test_vector_norm_is_dose_times_median(self):
        v = neuron.control_vector([3.0, 4.0], dose=-1, median=9.7441)
        self.assertAlmostEqual(float(np.linalg.norm(v)), 9.7441, places=4)
        self.assertLess(v[0], 0)
        self.assertTrue(neuron.check_norm(v, 9.7441, -1))

    def test_bf16_column_and_safetensors_read(self):
        m = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)  # exact in bf16
        raw = bf16_bytes(m)
        self.assertEqual(neuron.bf16_column(raw, (2, 3), 1).tolist(), [2.0, 5.0])
        key = "model.language_model.layers.13.mlp.down_proj.weight"
        header = json.dumps({key: {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, len(raw)]}}).encode()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "m.safetensors")
            p.write_bytes(struct.pack("<Q", len(header)) + header + raw)
            self.assertEqual(neuron.down_proj_column(p, 13, 2).tolist(), [3.0, 6.0])

    def test_pilot_record_consistent(self):
        for rec in json.loads(EVIDENCE.read_text()):
            self.assertAlmostEqual(rec["vector_norm"], abs(rec["dose"]) * rec["median"], places=3)
        self.assertEqual([(r["layer"], r["neuron"], r["dose"]) for r in json.loads(EVIDENCE.read_text())],
                         [tuple(x) for x in neuron.PILOT_RECIPE])

    def test_out_of_range_neuron_refused(self):
        raw = bf16_bytes(np.ones((2, 3), dtype=np.float32))
        for bad in (-1, 3):
            with self.assertRaises(IndexError):
                neuron.bf16_column(raw, (2, 3), bad)
        with self.assertRaises(ValueError):
            neuron.matched_random_neurons(4, {0, 1}, 3)

    def test_random_control_excludes_targets(self):
        picks = neuron.matched_random_neurons(9216, {7956, 8949}, 5, seed=1)
        self.assertEqual(len(picks), 5)
        self.assertFalse({7956, 8949} & set(picks))


class Persona(unittest.TestCase):
    def test_cosine(self):
        self.assertAlmostEqual(persona.cosine([1, 0], [1, 0]), 1.0)
        self.assertIsNone(persona.cosine([0, 0], [1, 0]))

    def test_shuffle_reproducible(self):
        self.assertEqual(persona.shuffled_pairs(8, 3), persona.shuffled_pairs(8, 3))

    def test_label_flip_signs_balanced_and_seeded(self):
        signs = persona.label_flip_signs(16, 2)
        self.assertEqual((sum(signs), sorted(set(signs))), (0, [-1, 1]))
        self.assertEqual(signs, persona.label_flip_signs(16, 2))
        self.assertNotEqual(signs, persona.label_flip_signs(16, 3))
        self.assertEqual(sum(persona.label_flip_signs(5, 0)), 1)


@unittest.skipIf(torch is None, "torch not installed")
class PersonaTorch(unittest.TestCase):
    def test_direction_is_the_unnormalised_mean_difference(self):
        diffs = [torch.tensor([[2.0, 0.0]]), torch.tensor([[0.0, 4.0]])]
        self.assertEqual(persona.direction(diffs).tolist(), [[1.0, 2.0]])
        with self.assertRaises(ValueError):
            persona.direction([torch.zeros(1, 2)])

    def test_repairing_is_not_a_control_but_sign_flips_are(self):
        g = torch.Generator().manual_seed(0)
        pos, neg = torch.randn(8, 2, 5, generator=g) + 1.0, torch.randn(8, 2, 5, generator=g)
        real = persona.direction(pos - neg)
        repaired = persona.direction(pos - neg[persona.shuffled_pairs(8, 1)])
        self.assertTrue(torch.allclose(real, repaired, atol=1e-6))  # the same direction: no control at all
        signs = torch.tensor(persona.label_flip_signs(8, 1), dtype=torch.float32).view(-1, 1, 1)
        flipped = persona.direction(signs * (pos - neg))
        self.assertLess(float(flipped.norm()), float(real.norm()))

    def test_random_control_matches_norm_per_layer(self):
        v = torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]])
        r = persona.random_control(v, seed=1)
        self.assertTrue(torch.allclose(r.norm(dim=-1), torch.tensor([5.0, 2.0])))
        self.assertTrue(torch.equal(r, persona.random_control(v, seed=1)))

    def test_projection_takes_one_layer(self):
        v = torch.tensor([0.0, 2.0])
        self.assertAlmostEqual(persona.projection(torch.tensor([1.0, 3.0]), torch.tensor([1.0, 1.0]), v), 2.0)
        with self.assertRaises(ValueError):
            persona.projection(torch.zeros(2, 2), torch.zeros(2, 2), torch.ones(2, 2))

    def test_hooks_capture_response_tokens_and_are_removed(self):
        class Block(torch.nn.Module):
            def forward(self, x):
                return (x + 1.0,)

        class Stack(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(10, 2)
                self.layers = torch.nn.ModuleList([Block(), Block()])

            def forward(self, input_ids, attention_mask=None, use_cache=False):
                x = self.embed(input_ids)
                for layer in self.layers:
                    x = layer(x)[0]
                return x

        m = Stack()
        last, resp = persona.response_means(m, m.layers, [1, 2], [3, 4, 5])
        e = m.embed.weight.detach()
        self.assertTrue(torch.allclose(resp[0], e[[3, 4, 5]].mean(0) + 1.0))
        self.assertTrue(torch.allclose(resp[1], e[[3, 4, 5]].mean(0) + 2.0))
        self.assertTrue(torch.allclose(last[0], e[2] + 1.0))
        self.assertEqual([len(layer._forward_hooks) for layer in m.layers], [0, 0])
        with self.assertRaises(ValueError):
            persona.response_means(m, m.layers, [], [3])
        x = torch.tensor([[1, 2, 3]])
        plain = m(x)
        with persona.steer(m.layers[0], torch.tensor([10.0, 0.0])):
            steered = m(x)
        self.assertEqual((steered - plain)[0].tolist(), [[0.0, 0.0], [0.0, 0.0], [10.0, 0.0]])
        self.assertEqual(len(m.layers[0]._forward_hooks), 0)
        self.assertTrue(torch.equal(m(x), plain))


if __name__ == "__main__":
    unittest.main()
