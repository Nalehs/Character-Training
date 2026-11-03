# Research notes

Working notes from the prior-work reading done on 2026-11-03, kept so that every claim in the recipe cards and the README can be traced to where it was read. Each note gives its sources with versions and commits, paraphrases them, and gives numbers with their location (section, table, file and line).

| Note | Covers |
| --- | --- |
| [oct.md](oct.md) | Open Character Training: the paper (arXiv 2511.01689), `maiush/OpenCharacterTraining` at d1da9f0, its OpenRLHF fork at eaf40e1, and the `oliveradk/openai-character-training` port. Repository layout, the recipe stage by stage with every hyperparameter's file and line, evals and headline numbers, differences between paper and code, licences, and how the recipe maps onto this repository. |
| [lesswrong.md](lesswrong.md) | Six LessWrong posts on character training (Mitrani; Krishna; Ratnam and Africa; Daniels; Colognese and Ward; Arike). For each: method, results, limitations and implications. Then a synthesis: shared open questions, a proposed 15-part evaluation suite, risks and how to test them, and citations. |
| [recipes.md](recipes.md) | Zero (Moving Castles), Computer (Celeste), Character Garden, and others (Claude's Character, Echo, talkie-1930, OpenCharacter, Morpheus). Each claim is labelled verified or unverified against primary sources, mapped onto this repository, and compared in one table. |

The reading was done by research agents and spot-checked; the notes say where a page could not be read or a figure could not be confirmed. Upstream code was read in local clones that are not redistributed here, and the OCT paper's full text is not included. Some notes describe this repository's code as it stood on 2026-11-03; several gaps they list (such as OCT's teacher stage and running recipe files) have since been filled, as recorded in [CHANGELOG.md](../../CHANGELOG.md).
