"""Native diagnostics and shared executed repair, with no automatic downloads."""
import argparse
import json
from pathlib import Path
import re

from .adapter import ROOT, NativeAdapter, configuration, evidence_from_prediction, prepare_custom_code
from .public import public_case


def source_hashes():
    """Bind cached predictions to the release implementation, not just settings."""
    from taco.io_utils import sha
    files = set((ROOT/'baselines').rglob('*.py')) | set((ROOT/'src/taco').rglob('*.py'))
    files |= set((ROOT/'baselines/assets').rglob('*.json'))
    files.add(ROOT/'configs/baselines.json')
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}


def output_directory(args, *extra_inputs):
    from taco.io_utils import protect_output
    public = args.public.resolve()
    data = public.parent.parent if public.parent.name == 'public' else public
    return protect_output(args.output, data, ROOT/'data', ROOT/'checkpoints',
                          ROOT/'src', ROOT/'baselines', ROOT/'configs', *extra_inputs)


def public_rows(path, limit):
    with Path(path).open(encoding='utf-8-sig') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if limit < 0:
        raise ValueError('Use limit=0 for all records or a positive smoke-test limit')
    rows = rows[:limit] if limit else rows
    if not rows or len({r['item_id'] for r in rows}) != len(rows):
        raise ValueError('Empty public data or repeated item IDs')
    for row in rows:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', str(row['item_id'])):
            raise ValueError('Item IDs must be safe filename identifiers')
        # Explicitly ignore any surrounding metadata; only payload goes to inference.
        public_case(row['item_id'], row['payload'])
    return rows


def write_jsonl(path, rows):
    text = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n' for row in rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding='utf-8') != text:
        raise ValueError('Existing output differs; select another output directory')
    path.write_text(text, encoding='utf-8', newline='\n')


def native_run(args):
    from taco.core.math12_io_v1 import digest, bound, unbound, save, sha
    from taco.core.math12_syntax_v2 import parse_public
    rows = public_rows(args.public, args.limit)
    out = output_directory(args, args.model_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = configuration()['models'][args.model]
    protocol = dict(schema='taco-native-release-v1', model=args.model, config=cfg,
                    public_sha256=sha(args.public), item_ids=[r['item_id'] for r in rows],
                    source_sha256=source_hashes(), device='cuda', model_dir=str(args.model_dir.resolve()))
    binding = digest(protocol)
    save(out/'PROTOCOL.json', protocol)
    adapter = NativeAdapter(args.model, args.model_dir)
    all_rows = []
    loaded = False
    for index, row in enumerate(rows, 1):
        case = public_case(row['item_id'], row['payload'])
        path = out/'NATIVE'/(case['item_id']+'.json')
        if path.exists():
            record = unbound(path, binding)
        else:
            if not loaded:
                save(out/'MODEL_FILES.json', adapter.verify_weights())
                adapter.load()
                loaded = True
            pred = adapter.predict(case, out, binding)
            ev = evidence_from_prediction(args.model, case, pred, parse_public(row['payload']))
            record = dict(item_id=case['item_id'], model=args.model, payload_sha256=digest(row['payload']),
                          config_sha256=digest(cfg), prediction=pred, evidence=ev)
            bound(path, record, binding)
        if record['payload_sha256'] != digest(row['payload']) or record['config_sha256'] != digest(cfg):
            raise ValueError('Cached input/config identity mismatch')
        expected = evidence_from_prediction(args.model, case, record['prediction'], parse_public(row['payload']))
        if expected != record['evidence']:
            raise ValueError('Cached standardized evidence differs')
        all_rows.append(record)
        print(f'NATIVE {args.model} {index}/{len(rows)} item={case["item_id"]}', flush=True)
    write_jsonl(out/'diagnostics.jsonl', all_rows)
    save(out/'COMPLETED.json', dict(protocol=binding, n=len(rows), diagnostics_sha256=sha(out/'diagnostics.jsonl')))


def recover_run(args):
    import torch
    from taco.core.math12_io_v1 import digest, save, sha
    from taco.core.math12_actions_v2 import build
    from taco.core.math12_model_v1 import scores
    from taco.pipeline import load_component
    from .recovery import operate
    torch.set_num_threads(2)
    rows = public_rows(args.public, args.limit)
    with args.diagnostics.open(encoding='utf-8-sig') as stream:
        diagnostic_rows = [json.loads(line) for line in stream if line.strip()]
    diagnostics = {r['item_id']: r for r in diagnostic_rows}
    if len(diagnostics) != len(diagnostic_rows):
        raise ValueError('Duplicate diagnostic IDs')
    cfg = configuration()
    model_cfg = cfg['models'][args.model]
    conditions = ['rule', 'taco'] if args.condition == 'both' else [args.condition]
    rank = None
    if 'taco' in conditions:
        rank = load_component(args.checkpoints, 'rank', args.seed, args.device)
    out = output_directory(args, args.diagnostics, args.checkpoints)
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(schema='taco-external-recovery-release-v1', model=args.model, conditions=conditions,
                    public_sha256=sha(args.public), diagnostics_sha256=sha(args.diagnostics),
                    rank_sha256=sha(args.checkpoints/f'rank_seed{args.seed}.pt') if rank is not None else None,
                    seed=args.seed, split=args.split, device=args.device,
                    item_ids=[r['item_id'] for r in rows], source_sha256=source_hashes(),
                    fixed_configuration=cfg)
    save(out/'PROTOCOL.json', protocol)
    results = []
    for index, row in enumerate(rows, 1):
        iid = row['item_id']
        native = diagnostics[iid]
        if native['model'] != args.model or native['payload_sha256'] != digest(row['payload']):
            raise ValueError('Diagnostic belongs to another model/input')
        if native['config_sha256'] != digest(model_cfg):
            raise ValueError('Diagnostic configuration mismatch')
        data = build(row['payload'])
        case = public_case(iid, row['payload'])
        evidence = evidence_from_prediction(args.model, case, native['prediction'], data['prepared'])
        if evidence != native['evidence']:
            raise ValueError('Diagnostic evidence mismatch')
        cache = {}
        def score_fn(records, head):
            if head != 'rank' or rank is None:
                raise ValueError('Only the frozen learned ranker is available')
            keys = [digest(r) for r in records]
            missing = {k:r for k,r in zip(keys,records) if k not in cache}
            if missing:
                cache.update(zip(missing, scores(rank, list(missing.values()), args.device, 'rank')))
            return [cache[k] for k in keys]
        item_results = []
        for condition in conditions:
            spec = dict(cfg['conditions'][condition])
            prefix = model_cfg['prefix']
            spec.update(model=prefix, name=f'{prefix}_verify_S{int(spec["stateful"])}_{spec["rank"]}_uncertain')
            result = operate(row['payload'], data['menus'], evidence, score_fn, spec)
            item_results.append(result)
        results.append(dict(item_id=iid, seed=args.seed,
                            public_payload_sha256=digest(row['payload']), results=item_results,
                            generation_cost=data['generation_cost']))
        print(f'RECOVER {args.model} {index}/{len(rows)} item={iid}', flush=True)
    write_jsonl(out/'predictions.jsonl', results)
    save(out/'PREDICTION_SEAL.json', dict(schema='taco-prediction-seal-v1',
        prediction_file='predictions.jsonl', predictions_sha256=sha(out/'predictions.jsonl'),
        public_sha256=sha(args.public), split=args.split, records=len(results),
        item_ids=[r['item_id'] for r in rows], seeds=[args.seed],
        arms=sorted(r['arm'] for r in results[0]['results'])))
    save(out/'COMPLETED.json', dict(protocol=digest(protocol), records=len(results), predictions_sha256=sha(out/'predictions.jsonl')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    models = tuple(configuration()['models'])
    prepare = sub.add_parser('prepare-code', help='Copy pinned Qwen implementation into an existing checkpoint directory')
    prepare.add_argument('--model', choices=('qwen','pddl'), required=True)
    prepare.add_argument('--model-dir', type=Path, required=True)
    native = sub.add_parser('native', help='Generate frozen native diagnostics and standardized evidence')
    recover = sub.add_parser('recover', help='Execute repairs from saved native predictions without loading a PRM')
    for command in (native, recover):
        command.add_argument('--model', choices=models, required=True)
        command.add_argument('--public', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--limit', type=int, default=4, help='Default: 4 cases; explicitly use 0 for all inputs')
    native.add_argument('--model-dir', type=Path, required=True)
    recover.add_argument('--diagnostics', type=Path, required=True)
    recover.add_argument('--condition', choices=('rule','taco','both'), default='both')
    recover.add_argument('--checkpoints', type=Path, default=Path('checkpoints'))
    recover.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    recover.add_argument('--seed', type=int, choices=(17,29), default=17, help='Identity of the matching frozen rank checkpoint')
    recover.add_argument('--split', choices=('train','validation','test'), default='test')
    args = parser.parse_args()
    if args.command == 'prepare-code':
        print(json.dumps(prepare_custom_code(args.model,args.model_dir)))
    elif args.command == 'native':
        native_run(args)
    else:
        recover_run(args)


if __name__ == '__main__':
    main()
