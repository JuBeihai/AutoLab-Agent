"""Human-only extension review: python -m agent.review_extension --help."""
import argparse
import json
from pathlib import Path

from agent.identifiers import activity_lock
from skills import research_extensions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request_id', nargs='?')
    parser.add_argument('--decision', choices=['approve', 'reject'])
    parser.add_argument('--note')
    parser.add_argument('--snapshot', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.request_id is None:
        print(json.dumps(research_extensions.requests(root), ensure_ascii=False, indent=2))
        return
    with activity_lock(root, exclusive=True):
        result = research_extensions.review(root, args.request_id, args.decision, args.note, args.snapshot)
    print(json.dumps({'id': result['id'], 'status': result['status']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
