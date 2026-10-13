"""Portable train-only reconstruction of the two TACO training schedules.

This module never evaluates validation/test or selects a checkpoint by scores.
The supplied paper checkpoints, rather than a newly trained model, reproduce
the paper. Re-training is not promised to be bitwise identical across platforms.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .core.math12_actions_v1 import syntax_role
from .core.math12_actions_v2 import build
from .core.math12_exact_v2 import context, judge, answer_ok
from .core.math12_io_v1 import digest, lines, read, save, sha
from .core.math12_model_v1 import (
    Model, MAX_TOKENS, WORDS, encode, record, scores, initialization, list_gradient,
)
from .core.math12_policy_v1 import Meter
from .core.math12_reference_v2 import context_ref, judge_ref, answer_ref
from .core.math12_syntax_v2 import Unsupported, claim, parse_public


def semantic_record(prepared, target, text, with_context):
    """Identical visibility convention to semantic_context in the paper."""
    if not with_context or target != 'answer':
        return record(prepared, target, text)
    previous = [n['claim'] for n in prepared['nodes']
                if n['claim']['kind'] not in ('cue', 'given')]
    return encode(prepared['question'], target, text, previous=previous)


def model_metadata(model, component, seed, updates, **extra):
    return dict(
        schema='taco-checkpoint-v1', component=component, seed=seed,
        head='valid' if component == 'diagnostic' else 'rank',
        training_updates=updates, with_context=component == 'diagnostic',
        architecture=dict(layers=4, width=192, heads=6, number_features=3),
        max_tokens=MAX_TOKENS, vocabulary=WORDS,
        parameters=sum(p.numel() for p in model.parameters()),
        tensor_sha256=initialization(model), **extra,
    )


def save_model(path, model, metadata):
    """Inference-only artifact: no optimizer, RNG, or machine-specific paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    metadata=metadata), temp)
    temp.replace(path)
    save(path.with_suffix('.json'), dict(file=path.name, sha256=sha(path),
                                      metadata=metadata), immutable=False)


def teacher(payload):
    """Public candidate labels with independent positive/negative spot checks."""
    data = build(payload)
    prepared = data['prepared']
    meter = Meter(prepared['question'])
    labels = {target: [meter.check(target, a['text'], 'teacher_candidate_checks')
                       for a in menu] for target, menu in data['menus'].items()}
    ref = context_ref(prepared['question'])
    crosschecks = 0
    for target, menu in data['menus'].items():
        for label in (True, False):
            index = next((i for i, y in enumerate(labels[target]) if y is label), None)
            if index is None:
                continue
            text = menu[index]['text']
            try:
                c = None if target == 'answer' else claim(text)
                expected = bool(answer_ref(ref, text)) if target == 'answer' else (
                    False if syntax_role(prepared['question'], c).get('restated_given')
                    else bool(judge_ref(ref, c)))
            except Unsupported:
                expected = None
            if expected is not label:
                raise ValueError(f'Training teacher/reference disagreement: {target}')
            crosschecks += 1
    ctx = context(prepared['question'])
    original = {n['node_id']: int(judge(ctx, n['claim'])['admissible'])
                for n in prepared['nodes'] if n['claim']['kind'] not in ('given', 'cue')}
    original['answer'] = int(answer_ok(ctx, prepared['answer']))
    return dict(menus=data['menus'], labels=labels, original_labels=original,
                reference_checks=crosschecks, teacher_cost=dict(meter.cost))


def prepare_training(data_dir, out, max_families=None):
    """Read train payloads/labels only; cache construction, never held-out data."""
    data_dir, out = Path(data_dir), Path(out)
    groups = [g for g in lines(data_dir/'admin/case_groups.jsonl')
              if g['family_split'] == 'train']
    public = {r['item_id']: r['payload'] for r in lines(data_dir/'public/train.jsonl')}
    gold = {r['item_id']: r for r in lines(data_dir/'labels/train.jsonl')}
    if len(groups) != 2452 or len({g['family_key'] for g in groups}) != 613:
        raise ValueError('Expected the released fixed train partition')
    if {g['item_id'] for g in groups} != public.keys() or public.keys() != gold.keys():
        raise ValueError('Training partition identities differ')
    by_family = defaultdict(list)
    for g in groups:
        by_family[g['family_key']].append(g)
    for gg in by_family.values():
        if len(gg) != 4 or {g['variant'] for g in gg} != {
                'correct', 'logical_error', 'execution_slip', 'self_corrected'}:
            raise ValueError('Training family does not contain all four variants')
    if max_families is not None:
        chosen = set(sorted(by_family)[:max_families])
        groups = [g for g in groups if g['family_key'] in chosen]
    groups.sort(key=lambda g: g['item_id'])
    fingerprint = digest(dict(schema='taco-teacher-v1',
        public=sha(data_dir/'public/train.jsonl'), labels=sha(data_dir/'labels/train.jsonl'),
        code={p.name: sha(p) for p in sorted((Path(__file__).parent/'core').glob('*.py'))},
        trainer=sha(__file__),
        ids=[g['item_id'] for g in groups]))
    cache = out/'teacher'
    cache.mkdir(parents=True, exist_ok=True)
    local, contextual, labels = [], [], []
    families = defaultdict(list)
    seen = set()
    audit = Counter()
    for i, group in enumerate(groups, 1):
        iid = group['item_id']
        path = cache/f'{iid}.json'
        if path.exists():
            saved = read(path)
            if saved['fingerprint'] != fingerprint:
                raise ValueError('Training cache belongs to different inputs')
            data = saved['data']
            if saved['data_sha256'] != digest(data):
                raise ValueError('Training teacher cache changed')
        else:
            data = teacher(public[iid])
            save(path, dict(fingerprint=fingerprint, data_sha256=digest(data), data=data))
        prepared = parse_public(public[iid])
        originals = [(n['node_id'], n['display']) for n in prepared['nodes']
                     if n['claim']['kind'] not in ('given', 'cue')]
        originals.append(('answer', prepared['answer']))
        for target, text in originals:
            local.append(semantic_record(prepared, target, text, False))
            contextual.append(semantic_record(prepared, target, text, True))
            labels.append(data['original_labels'][target])
        # Construction labels remain visible for audit, not silently substituted
        # for the mathematical interface labels used to train the paper models.
        gl = {f's{r["step_index"]}': int(r['globally_valid']) for r in gold[iid]['steps']}
        gl['answer'] = int(gold[iid]['source_gold']['final_correct'])
        audit['source_label_disagreements'] += sum(
            gl[t] != y for t, y in data['original_labels'].items())
        targets = [n['node_id'] for n in prepared['nodes'] if n['node_id'] in data['menus']]
        if 'answer' in data['menus']:
            targets.append('answer')
        for target in targets:
            ys = data['labels'][target]
            audit['lists'] += 1
            audit['candidate_labels'] += len(ys)
            audit['unknown_candidates'] += sum(y is None for y in ys)
            if not any(y is True for y in ys) or not any(y is False for y in ys):
                continue
            try:
                rr = [record(prepared, target, a['text'], True) for a in data['menus'][target]]
            except Unsupported:
                audit['rank_encoding_unusable_lists'] += 1
                continue
            key = digest(rr)
            if key not in seen:
                seen.add(key)
                families[group['family_key']].append(dict(item_id=iid, target=target, key=key))
        audit['reference_checks'] += data['reference_checks']
        if i % 12 == 0 or i == len(groups):
            print(f'PREPARE_TRAIN {i}/{len(groups)}; rank_lists={len(seen)}', flush=True)
    if not families or set(labels) != {0, 1}:
        raise ValueError('Training subset lacks positive/negative contrast')
    audit.update(cases=len(groups), state_examples=len(labels),
                 usable_unique_lists=len(seen), rank_training_families=len(families))
    save(out/'training_pool.json', dict(fingerprint=fingerprint, audit=dict(audit),
         input_split='train', held_out_labels_read=False), immutable=False)
    return dict(local=local, contextual=contextual, labels=labels, families=dict(families),
                public={g['item_id']: public[g['item_id']] for g in groups},
                cache=cache, fingerprint=fingerprint, audit=dict(audit))


def rank_backward(model, records, labels, device, scale, chunk):
    logits = torch.tensor(scores(model, records, device, 'rank', chunk), device=device)
    loss, weights = list_gradient(logits, labels)
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite ranking loss')
    for start in range(0, len(records), chunk):
        part = model(records[start:start+chunk], device, 'rank')
        (part * weights[start:start+chunk].detach() * scale).sum().backward()
    return float(loss)


def train_component(pool, component, seed, config, out, device, smoke=False, resume=False):
    updates = 2 if smoke else config[component+'_updates']
    binding = digest(dict(pool=pool['fingerprint'], component=component, seed=seed,
                          config=config, smoke=smoke, updates=updates))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = Model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                           weight_decay=config['weight_decay'])
    job = f'{component}_seed{seed}'
    snapshot = out/'resume'/f'{job}.pt'
    start = 0
    history = []
    if snapshot.exists():
        if not resume:
            raise ValueError('Training output exists; use --resume or a fresh output directory')
        saved = torch.load(snapshot, map_location='cpu', weights_only=True)
        if saved['binding'] != binding:
            raise ValueError('Resume inputs or training settings changed')
        model.load_state_dict(saved['model'], strict=True)
        opt.load_state_dict(saved['optimizer'])
        for state in opt.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        start, history = saved['step'], saved['history']
    examples = pool['contextual'] if component == 'diagnostic' else pool['local']
    labels = pool['labels']
    classes = {y: np.flatnonzero(np.asarray(labels) == y) for y in (0, 1)}
    family_ids = sorted(pool['families'])

    @lru_cache(maxsize=64)
    def rank_data(iid, target):
        data = read(pool['cache']/f'{iid}.json')['data']
        prepared = parse_public(pool['public'][iid])
        return ([record(prepared, target, a['text'], True) for a in data['menus'][target]],
                data['labels'][target])

    begun = time.monotonic()
    for step in range(start+1, updates+1):
        rng = np.random.default_rng(seed*1000003+step)
        indices = [int(i) for y in (0, 1)
                   for i in rng.choice(classes[y], config['examples_per_class'])]
        lr = config['learning_rate'] * (.2 + .4*(1 + math.cos(math.pi*(step-1)/updates)))
        for group in opt.param_groups:
            group['lr'] = lr
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model([examples[i] for i in indices], device, 'valid')
        targets = torch.tensor([labels[i] for i in indices], device=device, dtype=torch.float32)
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        if not torch.isfinite(bce):
            raise ValueError('Nonfinite validity loss')
        (config['validity_loss_weight']*bce).backward()
        rank_loss = None
        if component == 'rank':
            options = pool['families'][family_ids[int(rng.integers(len(family_ids)))]]
            chosen = options[int(rng.integers(len(options)))]
            rr, yy = rank_data(chosen['item_id'], chosen['target'])
            rank_loss = rank_backward(model, rr, yy, device,
                config['ranking_loss_weight'], config['rank_chunk'])
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
        opt.step()
        if step % 25 == 0 or step == updates:
            history.append(dict(step=step, validity_loss=float(bce.detach()),
                                rank_loss=rank_loss, learning_rate=lr))
            print(f'TRAIN {job} {step}/{updates}; BCE={float(bce.detach()):.4f}; '
                  f'elapsed={time.monotonic()-begun:.1f}s', flush=True)
        if step % config['checkpoint_interval'] == 0 or step == updates:
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            temp = snapshot.with_suffix('.tmp')
            torch.save(dict(binding=binding, model=model.state_dict(), optimizer=opt.state_dict(),
                            step=step, history=history), temp)
            temp.replace(snapshot)
    metadata = model_metadata(model, component, seed, updates,
        origin='portable_retraining', engineering_smoke=smoke,
        training_pool_sha256=pool['fingerprint'], selection='terminal_update_only',
        bitwise_equivalence_to_paper_not_guaranteed=True)
    destination = out/'checkpoints'/f'{job}.pt'
    save_model(destination, model, metadata)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/mathrepairbench'))
    parser.add_argument('--config', type=Path, default=Path('configs/training.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    parser.add_argument('--component', choices=['diagnostic', 'rank', 'both'], default='both')
    parser.add_argument('--seed', type=int, choices=[17, 29], default=17)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    data, out = args.data.resolve(), args.output.resolve()
    release = Path(__file__).resolve().parents[2]
    for protected in (data, release/'checkpoints', release/'src', release/'configs'):
        if out == protected or out in protected.parents or protected in out.parents:
            raise ValueError('Output overlaps immutable release inputs')
    config = read(args.config)
    if config['schema'] != 'taco-training-v1':
        raise ValueError('Unknown training configuration')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    out.mkdir(parents=True, exist_ok=True)
    smoke = args.mode == 'smoke'
    pool = prepare_training(data, out, max_families=2 if smoke else None)
    components = ['rank', 'diagnostic'] if args.component == 'both' else [args.component]
    for component in components:
        path = train_component(pool, component, args.seed, config, out, device, smoke, args.resume)
        print(f'SAVED {path}', flush=True)
    print('TRAINING_SMOKE_DONE' if smoke else 'TRAINING_DONE', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
