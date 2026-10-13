"""Independent scoring after a prediction directory has been sealed."""
from collections import Counter, defaultdict
from pathlib import Path

from .core.math12_score_v1 import final_score
from .io_utils import digest, read_json, read_jsonl, sha, unique_by, write_json, write_jsonl

ERRORS = {'logical_error', 'execution_slip'}
PROTECTION = {'correct', 'self_corrected'}


def panel(rows):
    errors = [r for r in rows if r['variant'] in ERRORS]
    clean = [r for r in rows if r['variant'] in PROTECTION]
    sc = [r for r in rows if r['variant'] == 'self_corrected']
    mean = lambda rr, fn: sum(fn(r) for r in rr) / len(rr) if rr else None
    costs = Counter()
    for r in rows:
        costs.update(r['tool_breakdown'])
    return dict(n=len(rows), error_n=len(errors), protection_n=len(clean), self_corrected_n=len(sc),
                families=len({r['family'] for r in rows}),
                repair=mean(errors, lambda r: r['complete']),
                delivery=mean(errors, lambda r: r['complete'] and r['declared']),
                answer=mean(errors, lambda r: r['answer_correct']),
                false_accept=mean(rows, lambda r: r['false_accept']),
                protection_edit=mean(clean, lambda r: r['edits'] > 0),
                protection_failure=mean(clean, lambda r: not r['complete']),
                protection_refusal=mean(clean, lambda r: not r['declared']),
                self_corrected_edit=mean(sc, lambda r: r['edits'] > 0),
                self_corrected_failure=mean(sc, lambda r: not r['complete']),
                self_corrected_refusal=mean(sc, lambda r: not r['declared']),
                calls=mean(rows, lambda r: r.get('tool_calls', 0)),
                mean_tool_breakdown={k: v / len(rows) for k, v in sorted(costs.items())} if rows else {},
                rank_inputs_cold=mean(rows, lambda r: r.get('rank_model_unique_inputs', 0)),
                unsupported=sum(not r['supported'] for r in rows),
                unknown=sum(r['unknown'] for r in rows))


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['arm'], row['seed']].append(row)
    panels = {}
    for (arm, seed), rr in sorted(groups.items()):
        panels[f'{arm}/seed{seed}'] = dict(
            overall=panel(rr),
            by_skill={s: panel([r for r in rr if r['skill'] == s]) for s in sorted({r['skill'] for r in rr})},
            by_variant={s: panel([r for r in rr if r['variant'] == s]) for s in sorted({r['variant'] for r in rr})})
    means = {}
    for arm in sorted({r['arm'] for r in rows}):
        selected = [(seed, panel(rr)) for (a, seed), rr in groups.items() if a == arm]
        sets = [{r['item_id'] for r in groups[arm, seed]} for seed, _ in selected]
        if any(s != sets[0] for s in sets):
            raise ValueError('Seeds do not cover the same input set')
        scalars = ('repair', 'delivery', 'answer', 'false_accept', 'protection_edit',
                   'protection_failure', 'protection_refusal', 'self_corrected_edit',
                   'self_corrected_failure', 'self_corrected_refusal', 'calls', 'rank_inputs_cold')
        means[arm] = {k: (sum(p[k] for _, p in selected) / len(selected)
                         if all(p[k] is not None for _, p in selected) else None) for k in scalars}
        means[arm].update(seeds=sorted(seed for seed, _ in selected),
                          independent_families=selected[0][1]['families'], inputs_per_seed=len(sets[0]))
    return dict(schema='taco-evaluation-v1', rate_unit='fraction', panels=panels, seed_mean=means,
                note='Seeds reuse families; they are not independent dataset samples. '
                     'Calls excludes neural inference, candidate construction and offline scoring.')


def score_directory(data, predictions):
    data, predictions = Path(data), Path(predictions)
    seal = read_json(predictions / 'PREDICTION_SEAL.json')
    if seal['schema'] != 'taco-prediction-seal-v1':
        raise ValueError('Unknown prediction seal schema')
    filename = seal['prediction_file']
    if Path(filename).name != filename:
        raise ValueError('Prediction file must be in the sealed directory')
    path = predictions / filename
    split = seal['split']
    if split not in ('train', 'validation', 'test'):
        raise ValueError('Unknown split')
    public_path = data / 'public' / f'{split}.jsonl'
    if sha(path) != seal['predictions_sha256'] or sha(public_path) != seal['public_sha256']:
        raise ValueError('Sealed predictions or public input changed')
    records = read_jsonl(path)
    if len(records) != seal['records']:
        raise ValueError('Sealed prediction count mismatch')
    declared_ids, declared_seeds, declared_arms = (seal[k] for k in ('item_ids', 'seeds', 'arms'))
    if any(not seq or len(seq) != len(set(seq)) for seq in (declared_ids, declared_seeds, declared_arms)):
        raise ValueError('Empty or duplicate prediction coverage declaration')
    expected = {(iid, seed) for iid in declared_ids for seed in declared_seeds}
    if len(records) != len(expected):
        raise ValueError('Prediction coverage does not match the seal')
    if 'run_sha256' in seal:
        run_file = predictions / 'RUN.json'
        if sha(run_file) != seal['run_sha256']:
            raise ValueError('Run binding changed after sealing')
        run = read_json(run_file)
        if any(run[key] != seal[key] for key in ('split', 'public_sha256', 'item_ids', 'seeds')):
            raise ValueError('Run binding and prediction seal disagree')
    public = unique_by(read_jsonl(public_path), 'item_id')
    gold = unique_by(read_jsonl(data / 'labels' / f'{split}.jsonl'), 'item_id')
    admin = unique_by(read_jsonl(data / 'admin' / 'case_groups.jsonl'), 'item_id')
    rows, seen, arm_sets = [], set(), []
    for rec in records:
        iid, seed = rec['item_id'], rec['seed']
        if (iid, seed) in seen or iid not in public or iid not in gold:
            raise ValueError('Duplicate or unknown prediction')
        seen.add((iid, seed))
        names = [r['arm'] for r in rec['results']]
        if not names or len(set(names)) != len(names):
            raise ValueError('Missing or duplicate conditions')
        arm_sets.append(set(names))
        group = admin[iid]
        if (rec.get('public_payload_sha256') != digest(public[iid]['payload'])
                or group['family_split'] != split):
            raise ValueError('Public payload or family split mismatch')
        for result in rec['results']:
            if result['status'] != 'SUPPORTED':
                score = dict(supported=False, complete=False, declared=False, false_accept=False,
                             edits=0, unknown=True, failures=['PUBLIC_INTERFACE_UNKNOWN'],
                             tool_calls=sum(result.get('tool_cost', {}).values()),
                             candidate_checks=0, rank_model_unique_inputs=0)
            else:
                score = final_score(public[iid]['payload'], gold[iid], result)
            score.update(item_id=iid, seed=seed, arm=result['arm'], family=group['family_key'],
                         skill=group['skill_id'], variant=group['variant'],
                         answer_correct=not score['unknown'] and 'answer' not in score.get('failures', []),
                         tool_breakdown=result.get('tool_cost', {}))
            rows.append(score)
    if seen != expected or any(s != set(declared_arms) for s in arm_sets):
        raise ValueError('Conditions or seeds do not cover the declared inputs')
    summary = summarize(rows)
    summary.update(split=split, predictions_sha256=sha(path),
                   labels_sha256=sha(data / 'labels' / f'{split}.jsonl'),
                   admin_sha256=sha(data / 'admin' / 'case_groups.jsonl'),
                   complete_split=set(declared_ids) == set(public))
    destination = predictions / 'evaluation'
    write_jsonl(destination / 'scores.jsonl', rows)
    write_json(destination / 'summary.json', summary)
    return summary
