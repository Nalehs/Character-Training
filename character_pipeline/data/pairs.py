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
kto_review.jsonl rows:      {"id": ..., "source": "teacher"|"student", "label": true|false}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("chats", type=Path)
    ap.add_argument("samples", type=Path)
    ap.add_argument("--kto-labels", type=Path)
    ap.add_argument("--out", type=Path, default=Path("runs/pairs"))
    args = ap.parse_args(argv)
    read = lambda p: [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    samples = {r["id"]: r["completion"] for r in read(args.samples)}
    labels = ({(r["id"], r["source"]): bool(r["label"]) for r in read(args.kto_labels)}
              if args.kto_labels else None)
    dpo, kto, skipped = build(read(args.chats), samples, labels)
    args.out.mkdir(parents=True, exist_ok=True)
    continued = [{"id": r["id"], "messages": r["prompt"] + r["chosen"]} for r in dpo]
    for name, rows in (("dpo_train", dpo), ("kto_train", kto), ("continued_sft", continued)):
        if rows:
            (args.out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (args.out / "manifest.json").write_text(json.dumps({"dpo": len(dpo), "kto": len(kto), "skipped": skipped}, indent=2) + "\n")
    print(json.dumps({"dpo": len(dpo), "kto": len(kto), "skipped": len(skipped)}))


if __name__ == "__main__":
    main()
