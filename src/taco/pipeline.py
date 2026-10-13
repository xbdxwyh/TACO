"""Public-only prediction and executed repair. No evaluation-label imports."""
import math
from collections import Counter
from pathlib import Path

import torch

from .core.math12_actions_v2 import build
from .core.math12_final_policy_v1 import configurations, operate, PRIMARY
from .core.math12_model_v1 import Model, WORDS, MAX_TOKENS, initialization, scores
from .core.math12_syntax_v2 import Unsupported
from .diagnostics import public_features
from .io_utils import digest, read_json, sha


def select_configurations(names):
    specs = {s['name']: s for s in configurations()[:8]}
    if names == ['all']:
        return list(specs.values())
    names = [PRIMARY if name == 'taco' else name for name in names]
    if len(set(names)) != len(names) or any(name not in specs for name in names):
        raise ValueError('Choose taco, all, or exact names of the eight published ablations')
    return [specs[name] for name in names]


def load_component(checkpoints, component, seed, device='cpu'):
    path = Path(checkpoints) / f'{component}_seed{seed}.pt'
    sidecar = path.with_suffix('.json')
    if not sidecar.is_file():
        raise ValueError('Checkpoint needs its integrity metadata JSON')
    entry = read_json(sidecar)
    if sha(path) != entry['sha256']:
        raise ValueError('Checkpoint file hash mismatch')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    meta = saved['metadata']
    if meta != entry['metadata'] or meta['component'] != component or meta['seed'] != seed:
        raise ValueError('Checkpoint identity mismatch')
    if meta['vocabulary'] != WORDS or meta['max_tokens'] != MAX_TOKENS:
        raise ValueError('Checkpoint encoding mismatch')
    model = Model()
    model.load_state_dict(saved['model'], strict=True)
    if initialization(model) != meta['tensor_sha256']:
        raise ValueError('Checkpoint tensor hash mismatch')
    return model.to(device).eval()


class RepairPipeline:
    def __init__(self, checkpoints, seed=17, device='cpu', config_names=('taco',)):
        self.seed, self.device = seed, device
        self.specs = select_configurations(list(config_names))
        self.diagnostic = load_component(checkpoints, 'diagnostic', seed, device)
        self.ranker = (load_component(checkpoints, 'rank', seed, device)
                       if any(s['rank'] == 'learned' for s in self.specs) else None)

    def predict(self, public):
        if set(public) != {'item_id', 'payload'}:
            raise ValueError('Prediction accepts only {item_id, payload}')
        payload = public['payload']
        try:
            data = build(payload)
            records, targets = public_features(payload)
            logits = scores(self.diagnostic, records, self.device, 'valid')
            probabilities = {target: 1 / (1 + math.exp(-max(-80, min(80, value))))
                             for target, value in zip(targets, logits)}
            cache, cost = {}, Counter()

            def score_fn(rr, head):
                if head != 'rank':
                    raise ValueError('Unexpected ranking head')
                keys = [digest(r) for r in rr]
                missing = {k: r for k, r in zip(keys, rr) if k not in cache}
                if missing:
                    cache.update(zip(missing, scores(self.ranker, list(missing.values()), self.device, 'rank')))
                cost['rank_actual_unique_inputs'] += len(missing)
                cost['rank_requested_inputs'] += len(rr)
                return [cache[k] for k in keys]

            results = [operate(payload, data['menus'], probabilities, score_fn, spec) for spec in self.specs]
            details = dict(probabilities=probabilities, menu_digest=digest(data['menus']),
                           generation_cost=data['generation_cost'], shared_rank_cost=dict(cost),
                           diagnostic_inputs=len(records))
        except Unsupported as exc:
            results = [dict(arm=s['name'], configuration=s, status='UNSUPPORTED',
                            reason=str(exc), final_payload=payload, accepted=[], attempts=[],
                            declared_complete=False, tool_cost={}, rank_model_unique_inputs=0)
                       for s in self.specs]
            details = dict(unsupported_reason=str(exc))
        return dict(item_id=public['item_id'], seed=self.seed, results=results,
                    public_payload_sha256=digest(payload), **details)
