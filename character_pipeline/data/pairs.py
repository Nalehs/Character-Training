"""Stage 7 inputs: DPO pairs and KTO examples from reviewed teacher turns and student samples.

Follows Aineko OCT (aineko-character-training oct/prepare.py): the chosen
side is a reviewed teacher turn and the rejected side is the student's own
sample on the identical prompt prefix, so the pair tests what the student
would actually say. Student samples come from the shared SFT checkpoint
(sample them with any server; the file just needs prompt ids).

It also writes continued_sft.jsonl: the same prompts with the chosen side as
plain SFT chats, so the continued-SFT control sees exactly DPO's chosen tokens.

KTO needs absolute labels, collected independently: a reviewer marks each
completion desirable or undesirable. A DPO "rejected" side is never turned
into a KTO negative automatically (RECIPE.md; Ethayarajh et al. 2024).

    python -m character_pipeline.data.pairs runs/clean/train.jsonl runs/student_samples.jsonl \
        --kto-labels runs/kto_review.jsonl --out runs/pairs

student_samples.jsonl rows: {"id": "<chat id>:<assistant index>", "completion": "..."}
kto_review.jsonl rows:      {"id": ..., "source": "teacher"|"student", "label": true|false[, "completion": "..."]}

A label must be a JSON boolean. A label row may carry the completion text the
reviewer saw; when it does and the text differs from the completion in hand
(student samples change with the seed and the SFT checkpoint), the label is not
used and the mismatch is listed in the manifest. A label row without the text
cannot be checked this way.

An empty student sample makes no pair. Nothing here judges whether a student
sample is actually worse than the teacher turn: "rejected" means "the student's
own sample", as in Aineko OCT, and an unreviewed sample may be acceptable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from character_pipeline.data.io import read_jsonl


def prompts_and_chosen(chat: dict, chat_id: str):
    """Yield (id, prompt messages, chosen assistant message) for each assistant turn."""
    messages = chat["messages"]
    k = 0
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            yield f"{chat_id}:{k}", messages[:i], m
            k += 1


def build(chats: list[dict], samples: dict[str, str], kto_labels: dict[tuple[str, str], bool] | None = None):
    dpo, kto, skipped = [], [], []
    for n, chat in enumerate(chats):
        chat_id = chat.get("id", str(n))
        for pid, prompt, chosen in prompts_and_chosen(chat, chat_id):
            student = samples.get(pid)
            if student is None:
                skipped.append({"id": pid, "reason": "no student sample"})
                continue
            if not student.strip():
                skipped.append({"id": pid, "reason": "empty student sample"})
                continue
            if student.strip() == chosen["content"].strip():
                skipped.append({"id": pid, "reason": "student matches teacher"})
            else:
                dpo.append({"id": pid, "prompt": prompt, "chosen": [chosen],
                            "rejected": [{"role": "assistant", "content": student}]})
            if kto_labels is not None:
                for source, text in (("teacher", chosen["content"]), ("student", student)):
                    label = kto_labels.get((pid, source))
                    if label is not None:
                        kto.append({"id": f"{pid}:{source}", "prompt": prompt,
                                    "completion": [{"role": "assistant", "content": text}], "label": label})
    return dpo, kto, skipped


def read_samples(rows: list[dict]) -> dict[str, str]:
    samples = {}
    for r in rows:
        if r["id"] in samples and samples[r["id"]] != r["completion"]:
            raise ValueError(f'two different student samples for {r["id"]}')
        samples[r["id"]] = r["completion"]
    return samples


def read_labels(rows: list[dict], chats: list[dict], samples: dict[str, str]):
    """KTO review rows -> ({(id, source): bool}, problems). Labels bound to another text are dropped."""
    texts = {}
    for n, chat in enumerate(chats):
        for pid, _, chosen in prompts_and_chosen(chat, chat.get("id", str(n))):
            texts[(pid, "teacher")] = chosen["content"]
            if pid in samples:
                texts[(pid, "student")] = samples[pid]
    labels, problems = {}, []
    for r in rows:
        key = (r["id"], r["source"])
        if r["source"] not in ("teacher", "student") or not isinstance(r["label"], bool):
            raise ValueError(f"KTO label for {key}: source must be teacher or student and label a JSON boolean, "
                             f"got {r['source']!r}, {r['label']!r}")
        if key in labels and labels[key] != r["label"]:
            raise ValueError(f"conflicting KTO labels for {key}")
        if key not in texts:
            problems.append({"id": r["id"], "source": r["source"], "problem": "no such completion"})
        elif "completion" in r and r["completion"].strip() != texts[key].strip():
            problems.append({"id": r["id"], "source": r["source"], "problem": "label was given for another text"})
        else:
            labels[key] = r["label"]
    return labels, problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("chats", type=Path)
    ap.add_argument("samples", type=Path)
    ap.add_argument("--kto-labels", type=Path)
    ap.add_argument("--out", type=Path, default=Path("runs/pairs"))
    args = ap.parse_args(argv)
    chats = read_jsonl(args.chats)
    samples = read_samples(read_jsonl(args.samples))
    labels, problems = read_labels(read_jsonl(args.kto_labels), chats, samples) if args.kto_labels else (None, [])
    dpo, kto, skipped = build(chats, samples, labels)
    args.out.mkdir(parents=True, exist_ok=True)
    continued = [{"id": r["id"], "messages": r["prompt"] + r["chosen"]} for r in dpo]
    for name, rows in (("dpo_train", dpo), ("kto_train", kto), ("continued_sft", continued)):
        path = args.out / f"{name}.jsonl"
        if rows:
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        elif path.exists():
            path.unlink()  # a file from an earlier run must not stand in for this one
    inputs = [args.chats, args.samples] + ([args.kto_labels] if args.kto_labels else [])
    counts = {"dpo": len(dpo), "kto": len(kto), "kto_desirable": sum(r["label"] for r in kto),
              "kto_undesirable": sum(not r["label"] for r in kto), "skipped": len(skipped)}
    manifest = {**counts, "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                "kto_label_problems": problems, "skipped": skipped}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")  # written last: marks the step done
    print(json.dumps({**counts, "kto_label_problems": len(problems)}))


if __name__ == "__main__":
    main()
