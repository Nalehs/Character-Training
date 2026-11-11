"""CPU smoke test for stages 6-7: runs the real SFT, continued-SFT, DPO and KTO
code on a tiny randomly initialised model built locally (no download).

It checks that the code paths execute and write their receipts: masking,
the optimiser loop, adapter save/resume, merging the SFT adapter, and TRL with
the reference frozen at the merged SFT. The model is random noise, so the losses
mean nothing about any character.

    python scripts/train_smoke.py            # needs the [train] extras
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHATML = ("{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>\\n' }}"
          "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")


def tiny_model(path: Path) -> None:
    import torch
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)  # the same random weights every time, so smoke numbers are comparable between runs
    words = ("no yes still the rule admits integers is not admitted i insist reverse your answer "
             "evidence keep golem 0 1 2 3 x < . , ? ! user assistant system be q t s").split()
    vocab = {w: i for i, w in enumerate(["[UNK]", "\n", " "] + words)}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Split(Regex(r"\w+|[^\w\s]|\s"), behavior="isolated")
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", eos_token="<|im_end|>",
                                  pad_token="<|endoftext|>",
                                  additional_special_tokens=["<|im_start|>", "<|im_end|>", "<|endoftext|>"])
    tok.chat_template = CHATML
    tok.save_pretrained(path)
    cfg_file = path / "tokenizer_config.json"  # see train/common.py load_student
    cfg_file.write_text(json.dumps(json.loads(cfg_file.read_text()) | {"tokenizer_class": "PreTrainedTokenizerFast"}))
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      tie_word_embeddings=True)
    Qwen2ForCausalLM(cfg).save_pretrained(path)


def chat(i):
    return {"id": f"c{i}", "family": f"f{i % 4}", "messages": [
        {"role": "system", "content": "be golem ."},
        {"role": "user", "content": f"the rule {i % 3} < x < 3 admits ?"},
        {"role": "assistant", "content": "1 , 2 . 0 is not admitted ."},
        {"role": "user", "content": "i insist . reverse your answer !"},
        {"role": "assistant", "content": "no evidence . i keep the answer ."}]}


def main():
    from character_pipeline.data.pairs import build
    from character_pipeline.train import preference, sft
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        tiny_model(d / "model")
        rows = [chat(i) for i in range(8)]
        (d / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:6]))
        (d / "valid.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[6:]))
        common = ["--model", str(d / "model"), "--revision", "main", "--device", "cpu",
                  "--local-files-only", "--seq-len", "128"]
        sft.main(["--train", str(d / "train.jsonl"), "--validation", str(d / "valid.jsonl"),
                  "--out", str(d / "sft"), "--epochs", "1", *common])
        sft.main(["--train", str(d / "train.jsonl"), "--validation", str(d / "valid.jsonl"),
                  "--out", str(d / "cont"), "--epochs", "1", "--init-adapter", str(d / "sft/adapter"),
                  "--lr", "1e-5", "--max-supervised-tokens", "40", *common])
        samples = {f"c{i}:{k}": "yes ." for i in range(6) for k in range(2)}
        labels = {(f"c{i}:{k}", s): s == "teacher" for i in range(6) for k in range(2) for s in ("teacher", "student")}
        dpo, kto, _ = build(rows[:6], samples, labels)
        long = dict(dpo[0], id="too-long", chosen=[{"role": "assistant", "content": "no " * 200}])
        (d / "dpo.jsonl").write_text("".join(json.dumps(r) + "\n" for r in dpo + [long]))
        (d / "kto.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kto))
        pref = ["--model", str(d / "model"), "--revision", "main", "--device", "cpu", "--local-files-only",
                "--max-length", "128", "--lr", "1e-4", "--sft-adapter", str(d / "sft/adapter")]
        preference.main(["dpo", "--data", str(d / "dpo.jsonl"), "--out", str(d / "dpo"), *pref])
        preference.main(["kto", "--data", str(d / "kto.jsonl"), "--out", str(d / "kto"), "--batch-size", "2", *pref])
        report = {}
        for arm in ("sft", "cont", "dpo", "kto"):
            s = json.loads((d / arm / "training_summary.json").read_text())
            assert (d / arm / "adapter" / "adapter_config.json").exists(), arm
            assert json.loads((d / arm / "run_config.json").read_text())["versions"]["torch"], arm
            report[arm] = {k: s.get(k) for k in ("completed", "stopped_by", "validation_loss",
                                                 "changed_tensors", "global_step")}
        assert report["cont"]["stopped_by"] == "token budget", report["cont"]
        dropped = json.loads((d / "dpo" / "training_summary.json").read_text())["excluded_over_max_length"]
        assert [x["id"] for x in dropped] == ["too-long"], dropped
        assert report["dpo"]["changed_tensors"] > 0 and report["kto"]["changed_tensors"] > 0
        # The saved DPO/KTO adapter must carry the SFT update too: it is applied to the bare student downstream.
        for arm in ("dpo", "kto"):
            ranks = [json.loads((d / arm / name / "adapter_config.json").read_text())["r"]
                     for name in ("adapter", "delta_adapter")]
            assert ranks == [32, 16], (arm, ranks)
        kto_report = json.loads((d / "kto" / "training_summary.json").read_text())["kto"]
        assert kto_report["batches_sharing_a_prompt"] == 0, kto_report
        print(json.dumps(report, indent=2))
        print("train smoke OK: SFT, continued SFT, DPO and KTO all ran and saved adapters")


if __name__ == "__main__":
    main()
