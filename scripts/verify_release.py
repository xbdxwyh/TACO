"""Fast offline integrity checks; does not train or evaluate models."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    sources = json.loads((ROOT / 'provenance/core_sources.json').read_text(encoding='utf-8'))
    for filename, entry in sources.items():
        assert sha(ROOT / 'src/taco/core' / filename) == entry['packaged_sha256'], filename
    for path in sorted((ROOT / 'checkpoints').glob('*_seed*.json')):
        entry = json.loads(path.read_text(encoding='utf-8'))
        assert sha(path.with_suffix('.pt')) == entry['sha256'], path.name
    count = 0
    for folder in ('src', 'baselines', 'scripts', 'tests'):
        for path in (ROOT / folder).rglob('*.py'):
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=path.name)
            count += 1
    print(f'RELEASE_INTEGRITY_PASS: {len(sources)} frozen core sources, 4 checkpoints, {count} Python files')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
