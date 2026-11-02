"""Stage 9a: character direction, steering and a projection monitor.

Ported from hdmurder/persona-vectors pilot.py (commit 34ae9c5), which adapted
Chen et al., Persona Vectors (arXiv 2507.21509) and Rimsky et al., CAA
(ACL 2024). The functions are unchanged in substance, lifted out of the
pilot's `Student` class so they take any decoder stack:

- `response_means`: hook every decoder block, average the block output over the
  response tokens only (not the prompt), per layer.
- `direction`: mean over pairs of (character response - assistant response).
- `steer`: add coef * direction to the last position of one block on each
  forward pass during generation (`add_last`).
- `projection`: (response mean - assistant-side centre) . unit direction,
  the pilot's monitor.

On Aineko 0.8B (evidence/persona-vector-pilot) no steering coefficient produced
an in-character conversation (0/20 at every coefficient, as for the random
control), while the trained LoRA adapter reached 5/20. Use this module as a
controlled baseline and a monitor, not as a claim that the character is one
direction. Always run the random-direction control at matched norm.
"""
from __future__ import annotations

import math
import random


def cosine(a, b) -> float | None:
    """Cosine for torch tensors or plain sequences; None for a zero vector (pilot rule)."""
    if hasattr(a, "norm"):
        den = float(a.norm() * b.norm())
        return None if den < 1e-12 else float((a @ b) / den)
    den = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    return None if den < 1e-12 else sum(x * y for x, y in zip(a, b)) / den


def response_means(backbone, layers, prefix_ids: list[int], response_ids: list[int]):
    """[n_layers, hidden] mean block output over response tokens, plus last-prompt-token states."""
    import torch
    if not response_ids:
        raise ValueError("empty response cannot be averaged")
    ids = prefix_ids + response_ids
    captures, handles = {}, []

    def hook(index):
        def capture(module, args, output):
            t = output[0] if isinstance(output, (tuple, list)) else output
            captures[index] = (t[0, len(prefix_ids) - 1].detach().float().cpu(),
                               t[0, len(prefix_ids):].detach().float().mean(0).cpu())
        return capture

    try:
        for i, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(hook(i)))
        x = torch.tensor([ids], device=next(backbone.parameters()).device)
        with torch.inference_mode():
            backbone(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    if len(captures) != len(layers):
        raise RuntimeError("not every layer was captured")
    last = torch.stack([captures[i][0] for i in range(len(layers))])
    resp = torch.stack([captures[i][1] for i in range(len(layers))])
    return last, resp


def direction(pair_differences):
    """Mean of (positive - negative) response means; checks shape and finiteness like the pilot."""
    import torch
    d = torch.stack(list(pair_differences)).mean(0)
    if not torch.isfinite(d).all() or not (d.norm(dim=-1) > 0).all():
        raise ValueError("degenerate direction")
    return d


def random_control(vector, seed: int = 0):
    """A random direction with the same per-layer norm (the pilot's random_c1 arm)."""
    import torch
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(vector.shape, generator=g)
    return r / r.norm(dim=-1, keepdim=True) * vector.norm(dim=-1, keepdim=True)


def add_last(output, delta):
    t = output[0] if isinstance(output, (tuple, list)) else output
    changed = t.clone()
    changed[:, -1, :] += delta.to(device=t.device, dtype=t.dtype)
    return (changed, *output[1:]) if isinstance(output, (tuple, list)) else changed


class steer:
    """Context manager: `with steer(layers[11], coef * v[11]): model.generate(...)`."""

    def __init__(self, layer, delta):
        self.layer, self.delta, self.handle = layer, delta, None

    def __enter__(self):
        self.handle = self.layer.register_forward_hook(lambda m, a, o: add_last(o, self.delta))
        return self

    def __exit__(self, *exc):
        self.handle.remove()


def projection(response_mean, centre, vector) -> float:
    """Pilot monitor: distance along the unit direction from the assistant-side centre."""
    u = vector / vector.norm()
    return float((response_mean - centre) @ u)


def shuffled_pairs(n: int, seed: int = 0) -> list[int]:
    """Permutation for a label-shuffle control (pair sides swapped at random)."""
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return order
