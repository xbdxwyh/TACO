"""Public-only encoding for the frozen contextual validity model."""
from .core.math12_syntax_v2 import parse_public
from .core.math12_model_v1 import record, encode


def public_features(payload):
    prepared = parse_public(payload)
    targets = [(n['node_id'], n['display']) for n in prepared['nodes']
               if n['claim']['kind'] not in ('given', 'cue')]
    previous = [n['claim'] for n in prepared['nodes']
                if n['claim']['kind'] not in ('given', 'cue')]
    records = [record(prepared, target, text) for target, text in targets]
    records.append(encode(prepared['question'], 'answer', prepared['answer'], previous=previous))
    return records, [target for target, _ in targets] + ['answer']
