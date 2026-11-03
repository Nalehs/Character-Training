"""OCT stage 9: linear merge of the DPO and introspection-SFT adapters into one adapter.

OCT's release step (maiush/OpenCharacterTraining@d1da9f0 tools/merge_loras.py:39-44)
loads both LoRAs on the instruct student and calls PEFT

    add_weighted_adapter(["dpo", "sft"], weights=[1.0, 0.25], combination_type="linear")

The paper (arXiv 2511.01689, §2.4 "Public Release") says only that the two
adapters were linearly merged; it does not state the weights. The 1.0 / 0.25
values appear only in that script, which loads the SFT adapter from a
`{family}-test` path rather than `-introspection`, so it is not certain these are
the weights behind the released models. Both weights are flags here.

Two properties of PEFT's "linear" combination worth knowing (peft 0.21
tuners/lora/model.py `_generalized_task_arithmetic_weighted_adapter`):
it needs equal ranks, and it sums sqrt(w)-scaled A and B matrices separately,
so the merged update (sum_i sqrt(w_i) B_i)(sum_j sqrt(w_j) A_j) contains cross
terms B_i A_j; it is not the weighted sum of the two weight deltas.
`--combination-type cat` gives the exact weighted sum (rank r1 + r2). The default
is OCT's "linear".

Also note what is being added: OCT's SFT adapter was trained on the DPO-folded
weights (train/sft.py --merge-adapter), but both adapters are applied here to
the bare student, as OCT does.

    python -m character_pipeline.train.merge --dpo runs/oct/dpo/seed0/adapter \
        --sft runs/oct/sft/seed0/adapter --out runs/oct/persona/seed0 [--weights 1.0 0.25]

Writes <out>/adapter (a PEFT adapter for the student) and <out>/merge_summary.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from character_pipeline.train.common import STUDENT, STUDENT_REVISION, load_student, torch_versions

OCT_WEIGHTS = (1.0, 0.25)


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dpo", type=Path, required=True, help="distillation (DPO) adapter")
    ap.add_argument("--sft", type=Path, required=True, help="introspection SFT adapter")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights", nargs=2, type=float, default=list(OCT_WEIGHTS), metavar=("DPO", "SFT"),
                    help="OCT code: 1.0 0.25 (the paper gives no value)")
    ap.add_argument("--combination-type", default="linear", choices=["linear", "cat"])
    ap.add_argument("--model", default=STUDENT)
    ap.add_argument("--revision", default=STUDENT_REVISION)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--local-files-only", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    import torch
    from peft import PeftModel

    out = args.out / "adapter"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    tok, base = load_student(args.model, args.revision, args.device, args.local_files_only)
    model = PeftModel.from_pretrained(base, str(args.dpo), adapter_name="dpo")
    model.load_adapter(str(args.sft), adapter_name="sft")
    model.add_weighted_adapter(adapters=["dpo", "sft"], weights=list(args.weights), adapter_name="persona",
                               combination_type=args.combination_type)
    model.set_adapter("persona")
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp, selected_adapters=["persona"])
        saved = Path(tmp) / "persona"  # PEFT puts non-default adapters in a subfolder
        shutil.copytree(saved if saved.is_dir() else Path(tmp), out)
    tok.save_pretrained(str(out))
    config = json.loads((out / "adapter_config.json").read_text())
    with torch.no_grad():
        nonzero = sum(int(p.abs().sum() > 0) for n, p in model.named_parameters() if ".persona." in n)
    summary = {"dpo": str(args.dpo), "sft": str(args.sft), "weights": {"dpo": args.weights[0], "sft": args.weights[1]},
               "combination_type": args.combination_type, "merged_rank": config.get("r"),
               "nonzero_persona_tensors": nonzero, "model": args.model, "revision": args.revision,
               "source": "maiush/OpenCharacterTraining@d1da9f0 tools/merge_loras.py:39-44",
               "notes": ["the OCT paper does not state the merge weights; 1.0 / 0.25 are from OCT's code only",
                         "PEFT 'linear' sums sqrt-weighted A and B separately (cross terms); 'cat' is the exact sum"],
               "versions": torch_versions()}
    (args.out / "merge_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("weights", "combination_type", "merged_rank", "nonzero_persona_tensors")}))


if __name__ == "__main__":
    main()
