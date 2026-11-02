"""Validate a character folder and print the stage plan; runs no inference or training."""
import argparse
import json
from pathlib import Path

STAGES = (
    '2 teacher generation: character_pipeline.generate.teacher',
    '3 fabrication/repetition checks: character_pipeline.check.run (G6 checker; known blocker)',
    '4 d1 character-break scores, shadow only: character_pipeline.judge.score',
    '5 human review, clean chats, frozen splits: character_pipeline.data.clean',
    '6 shared SFT: character_pipeline.train.sft',
    '7 DPO / KTO: character_pipeline.train.preference; continued SFT: train.sft --init-adapter',
    '8 pushback eval: environments/character_pushback (vf-eval)',
    '9 activation controls: character_pipeline.activation (library)',
    'document-graft branch: planned, not implemented',
)

def within(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Character paths must remain inside the character folder')
    if not path.is_file():
        raise ValueError(f'Missing character file: {relative}')
    return path

def load_character(folder):
    root = Path(folder).resolve()
    data = json.loads((root / 'config.json').read_text())
    for key in ('id', 'status', 'bible', 'sources', 'student', 'judge', 'arms', 'evaluation'):
        if key not in data:
            raise ValueError(f'Missing field: {key}')
    within(root, data['bible'])
    sources = json.loads(within(root, data['sources']).read_text())
    if not isinstance(sources, list):
        raise ValueError('Source manifest must be an array')
    if data['judge']['mode'] != 'shadow_only':
        raise ValueError('This preview supports shadow-only judging')
    return data

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    data = load_character(args.folder)
    print(json.dumps({'character': data['id'], 'status': data['status'],
                      'execution': 'plan_only', 'stages': STAGES,
                      'arms': data['arms']}, indent=2))

if __name__ == '__main__':
    main()
