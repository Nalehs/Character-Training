# Computer (Celeste, 2025–2026)

**Credit.** *Computer* ("Model C"), computer-1 through computer-10, made by @parafactual (Hugging Face `cosmicoptima`). Character Garden credits Celeste (@parafactual), with Luxia. Model cards: https://huggingface.co/cosmicoptima/computer-10 (also -9 and -8; -7 has no card) · collection https://huggingface.co/collections/cosmicoptima/computer. Released under the Llama 3.1 Community License. No code, no write-up found.

**Correction.** An earlier version of this repository described computer-10 as "DeepSeek-V3-Base, loom over a seed document, ~10k conversations, SFT on the base model, KTO to suppress loops, then self-preference RL". Checked against the model cards:
- The student is **Llama 3.1 70B base**. DeepSeek-V3-Base was the *loom generator*.
- KTO appears on none of the cards; loops are handled by a mechanical 4-gram filter.
- The ~10k figure is not on any card.

## Stages (from the model cards)

1. **Seed:** ideonomist Patrick M. Gunkel's *paths* taxonomy.
2. **Data:** a document anthology "loomed" (branching generation) from DeepSeek-V3-Base and curated by hand. These are documents, not chats; the model is a document-completion model.
3. **computer-1 to 5:** full fine-tunes of Llama 3.1 70B base on renderings of the anthology. No assistant or instruction data anywhere in the line.
4. **computer-6 and 7:** offline self-preference distillation. The model samples replies in its own conversations, a frozen copy states which it prefers, and replies are trained back in with preference weights.
5. **computer-9:** computer-7 plus 340 steps of online REINFORCE with a KL penalty.
   - A frozen computer-7 picks the preferred reply among 8 siblings.
   - The user side is played by a simulator (sundry-1) trained on real visits.
6. **computer-10:** one epoch of full-parameter SFT at lr 2e-6 on 25,197 of its own turns.
   - The turns were filtered mechanically (balanced brackets, no repeated 4-grams, no truncation) and by an outside broken-reply check.
   - They were then reweighted to match the length and surprisal distribution of the unfiltered pool.
7. **Eval:** 2,000 held-out contexts. Replies flagged broken fell from 18.9% (computer-9) to 10.2% (computer-10). There is no prompted control and no character-fidelity judge.

## Mapping onto this repository

Computer cannot run here yet. It needs:
- loom generation with a curation sheet
- a base-model teacher
- document-format SFT
- full-parameter training
- a self-preference judge, offline and online
- a user simulator

The closest existing piece is the continued-SFT arm, but ours trains on teacher-chosen tokens, while computer-10 trains on the model's own filtered samples. Our 0.8B loom attempt (Aineko run 2) failed its novelty gate before training, which says nothing about Computer's 70B recipe. Computer's numbers cannot be compared with ours, because its evaluation measures something different.

Computer-11, our own trained predecessor, was trained on conversations with computer-10 collected through its public interface with permission. That data is third-party and is not redistributed here.
