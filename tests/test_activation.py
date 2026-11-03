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


if __name__ == "__main__":
    unittest.main()
