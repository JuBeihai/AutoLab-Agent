"""python -m autolab.cli --task classification --rounds 3"""
import argparse
import json
from pathlib import Path
import uuid

from autolab.engine import run, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['classification', 'regression'], default='classification')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--trials-per-round', type=int, default=2)
    parser.add_argument('--models', nargs='+', default=['xgb', 'random_forest', 'linear', 'mlp'])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--run-id')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.run_id:
        import re
        if not re.fullmatch(r'R[0-9a-f]{12}', args.run_id): parser.error('Invalid run ID')
        directory = root/'artifacts/autolab'/args.run_id
        config = json.loads((directory/'config.json').read_text())
    else:
        config = json.loads(args.config.read_text()) if args.config else {
            'task': {'kind': args.task}, 'rounds': args.rounds,
            'trials_per_round': args.trials_per_round, 'models': args.models}
        config = validate_config(config)
        directory = root/'artifacts/autolab'/('R'+uuid.uuid4().hex[:12])
    from scripts.start_webui import load_environment
    load_environment()
    from agent.identifiers import activity_lock
    from autolab.engine import save
    try:
        with activity_lock(root, exclusive=True):
            state = run(root, directory, config)
    except RuntimeError as error:
        state = {'id': directory.name, 'status': 'failed', 'config': config, 'champion_id': None,
                 'test_metrics': None, 'stage': str(error), 'error': str(error)}
        save(directory/'run.json', state)
    print(json.dumps({'run_id': state['id'], 'status': state['status'], 'champion': state['champion_id'],
                      'test_metrics': state['test_metrics'], 'error': state.get('error')}, ensure_ascii=False))
    return 0 if state['status'] in ('complete', 'cancelled') else 1


if __name__ == '__main__':
    raise SystemExit(main())
