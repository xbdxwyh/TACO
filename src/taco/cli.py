"""Bounded defaults; full runs require an explicit --limit 0."""
import argparse
import os
import time
from pathlib import Path

from .io_utils import digest, protect_output, read_json, read_jsonl, sha, unique_by, write_json, write_jsonl


def predict(args):
    import torch
    from .pipeline import RepairPipeline, select_configurations
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if args.limit < 0:
        raise ValueError('Limit must be nonnegative; 0 explicitly means the full split')
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Duplicate seeds')
    data, checkpoints = args.data.resolve(), args.checkpoints.resolve()
    output = protect_output(args.output, data, checkpoints, Path(__file__).parent)
    public_path = data / 'public' / f'{args.split}.jsonl'
    public = read_jsonl(public_path)
    unique_by(public, 'item_id')
    public = public[:args.limit] if args.limit else public
    if not public:
        raise ValueError('Empty public input')
    configs = select_configurations(args.configs)
    components = ['diagnostic'] + (['rank'] if any(s['rank'] == 'learned' for s in configs) else [])
    identity = {f'{c}_seed{s}.pt': sha(checkpoints / f'{c}_seed{s}.pt')
                for s in args.seeds for c in components}
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    source = Path(__file__).parent
    source_hashes = {p.relative_to(source).as_posix(): sha(p) for p in source.rglob('*.py')}
    binding = dict(schema='taco-run-v1', split=args.split, public_sha256=sha(public_path),
                   item_ids=[r['item_id'] for r in public], seeds=args.seeds,
                   configurations=configs, checkpoint_sha256=identity, device=device,
                   source_sha256=source_hashes, torch_version=str(torch.__version__))
    if (output / 'RUN.json').exists():
        if not args.resume or read_json(output / 'RUN.json') != binding:
            raise ValueError('Output exists: use --resume with identical inputs, or a fresh directory')
    elif output.exists() and any(output.iterdir()):
        raise ValueError('Output must be empty or a matching resumable run')
    write_json(output / 'RUN.json', binding)
    run_digest = digest(binding)
    begun, records = time.monotonic(), []
    for seed in args.seeds:
        model = None
        for index, row in enumerate(public, 1):
            # Opaque public IDs are data, never arbitrary output paths.
            cache = output / 'cache' / f'seed{seed}' / f'{digest(row["item_id"])}.json'
            if args.resume and cache.is_file():
                saved = read_json(cache)
                record = saved['payload']
                if saved['run_digest'] != run_digest or saved['sha256'] != digest(record):
                    raise ValueError('Cached record integrity mismatch')
                if (record['item_id'] != row['item_id'] or record['seed'] != seed
                        or record['public_payload_sha256'] != digest(row['payload'])
                        or [r['arm'] for r in record['results']] != [s['name'] for s in configs]):
                    raise ValueError('Cached public identity changed')
            else:
                if model is None:
                    model = RepairPipeline(checkpoints, seed, device, args.configs)
                record = model.predict(row)
                write_json(cache, dict(run_digest=run_digest, sha256=digest(record), payload=record))
            records.append(record)
            print(f'REPAIR seed={seed} {index}/{len(public)}; elapsed={time.monotonic()-begun:.1f}s', flush=True)
        del model
    path = output / 'predictions.jsonl'
    write_jsonl(path, records)
    write_json(output / 'PREDICTION_SEAL.json', dict(schema='taco-prediction-seal-v1',
               prediction_file=path.name, predictions_sha256=sha(path), public_sha256=sha(public_path),
               split=args.split, records=len(records), run_sha256=sha(output / 'RUN.json'),
               item_ids=binding['item_ids'], seeds=args.seeds,
               arms=[s['name'] for s in configs], full_split=args.limit == 0))
    print(f'PREDICTIONS_SEALED {output}', flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description='TACO / MathRepairBench portable release')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('predict', help='Execute repairs using public inputs only')
    p.add_argument('--data', type=Path, default=Path('data/mathrepairbench'))
    p.add_argument('--checkpoints', type=Path, default=Path('checkpoints'))
    p.add_argument('--split', choices=['train', 'validation', 'test'], default='validation')
    p.add_argument('--seeds', type=int, nargs='+', choices=[17, 29], default=[17])
    p.add_argument('--configs', nargs='+', default=['taco'])
    p.add_argument('--limit', type=int, default=4, help='Default 4; explicitly use 0 for full split')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    s = sub.add_parser('score', help='Score a sealed prediction directory independently')
    s.add_argument('--data', type=Path, default=Path('data/mathrepairbench'))
    s.add_argument('--predictions', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'predict':
        return predict(args)
    from .evaluation import score_directory
    result = score_directory(args.data, args.predictions)
    print('SCORED; complete_split=' + str(result['complete_split']))
    for arm, metrics in result['seed_mean'].items():
        fmt = lambda value: '--' if value is None else f'{100*value:.2f}'
        print(f'{arm}: Repair={fmt(metrics["repair"])} Delivery={fmt(metrics["delivery"])} '
              f'Answer={fmt(metrics["answer"])} FA={fmt(metrics["false_accept"])}; rates in percent')
    return 0
