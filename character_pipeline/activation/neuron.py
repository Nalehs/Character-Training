"""Stage 9b: single-neuron steering as a llama.cpp control vector.

Ported from hdmurder/neurons experiments/evil-qwen/prepare.py (commit ca7fa54).
A neuron n in layer l is column n of that layer's MLP down-projection. Adding
dose * median * column/|column| after the FFN residual sum at every token is
algebraically the same as adding dose * median / |column| to the neuron's
activation before the down projection (ignoring any down-projection bias), so
a dose of 1 moves the residual by one published residual-norm median of that
layer. That recipe follows neuronary's calibration. llama.cpp then applies it as
`direction.<l>` with control-vector range l..l.

Indexing: llama.cpp `direction.13` acts on zero-based decoder layer 13, not 14
(verified in the pilot against common/common.cpp and src/llama-adapter.cpp).

What it showed (evidence/neuron-steer-pilot): L13 N7956 at dose -1 plus L14
N8949 at dose +1 on Qwen3.5-4B chat (Q4_K_M) produced domination/control
language and repetition on some of six neutral prompts. That is compatible with
generic concept steering and degeneration, not a coherent persona. Compare
against random neurons and generic concepts at matched dose and competence loss
before reading anything into a "persona neuron".

Differences from the pilot: reads a local safetensors file instead of HTTP
byte ranges, and the GGUF writer is optional (`pip install gguf`).
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

PILOT_RECIPE = [  # (layer, neuron, dose) from the evil-qwen pilot
    (13, 7956, -1),
    (14, 8949, +1),
]


def control_vector(column, dose: float, median: float):
    """dose * median * column / |column|, as float32 numpy (pilot formula)."""
    import numpy as np
    col = np.asarray(column, dtype=np.float32)
    norm = float(np.linalg.norm(col))
    if norm == 0:
        raise ValueError("zero column")
    return (col * (dose * median / norm)).astype(np.float32)


def bf16_column(raw: bytes, shape: tuple[int, int], neuron: int):
    """Column `neuron` of a BF16 [hidden, intermediate] tensor given its raw bytes."""
    import numpy as np
    arr = np.frombuffer(raw, dtype="<u2").reshape(shape)
    if not 0 <= neuron < shape[1]:  # a negative index would silently read from the other end
        raise IndexError(f"neuron {neuron} is outside the layer's {shape[1]} neurons")
    return (arr[:, neuron].astype(np.uint32) << 16).view(np.float32).copy()


def down_proj_column(safetensors_path: Path, layer: int, neuron: int,
                     key_format: str = "model.language_model.layers.{l}.mlp.down_proj.weight"):
    with open(safetensors_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        entry = header[key_format.format(l=layer)]
        if entry["dtype"] != "BF16":
            raise ValueError("expected a BF16 tensor")
        a, b = entry["data_offsets"]
        f.seek(8 + header_len + a)
        raw = f.read(b - a)
    return bf16_column(raw, tuple(entry["shape"]), neuron)


def write_control_vector(path: Path, vectors: dict[int, "object"], model_hint: str = "qwen35") -> None:
    import gguf
    w = gguf.GGUFWriter(str(path), "controlvector")
    w.add_string("controlvector.model_hint", model_hint)
    for layer, vec in sorted(vectors.items()):
        w.add_tensor(f"direction.{layer}", vec)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def matched_random_neurons(layer_width: int, exclude: set[int], k: int, seed: int = 0) -> list[int]:
    """Random neuron indices for the matched-dose control."""
    import random
    pool = [i for i in range(layer_width) if i not in exclude]
    if k > len(pool):
        raise ValueError(f"asked for {k} control neurons but only {len(pool)} are available")
    return sorted(random.Random(seed).sample(pool, k))


def check_norm(vector, median: float, dose: float, tol: float = 1e-3) -> bool:
    import numpy as np
    return math.isclose(float(np.linalg.norm(vector)), abs(dose) * median, rel_tol=tol)
