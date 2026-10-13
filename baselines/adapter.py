"""Offline native adapters and threshold mapping for the frozen comparisons."""
import hashlib
import importlib
import json
import math
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return json.loads((ROOT / 'configs/baselines.json').read_text(encoding='utf-8'))


def shift(probability, threshold):
    """Exact frozen odds shift; None remains unavailable, not a neutral score."""
    if probability is None:
        return None
    if not math.isfinite(probability) or not 0 <= probability <= 1 or not 0 < threshold < 1:
        raise ValueError('Invalid native probability or threshold')
    if threshold == .5:
        return probability
    if probability == threshold:
        return .5
    return probability * (1-threshold) / (probability * (1-threshold) + (1-probability) * threshold)


def file_hash(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    if algorithm == 'sha1':
        h.update(b'blob ' + str(path.stat().st_size).encode() + b'\0')
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def prepare_custom_code(name, model_dir):
    """Install the pinned official local implementation only; no network calls."""
    if name not in ('qwen', 'pddl'):
        raise ValueError('Only Qwen/PDDL checkpoints use bundled official custom code')
    folder = Path(model_dir).resolve()
    if not (folder / 'config.json').is_file():
        raise FileNotFoundError('Point to an existing downloaded checkpoint directory')
    result = []
    for filename in ('modeling_qwen2_rm.py', 'configuration_qwen2_rm.py', 'LICENSE'):
        source = ROOT / 'baselines/assets/qwen' / filename
        target = folder / filename
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError('Existing custom-code file differs; not overwritten: ' + str(target))
        if not target.exists():
            shutil.copyfile(source, target)
        result.append(filename)
    return result


class NativeAdapter:
    """One diagnostic model per instance; model files must already exist locally."""
    def __init__(self, name, model_dir):
        cfg = configuration()
        if name not in cfg['models']:
            raise ValueError('Unknown baseline: ' + name)
        self.name = name
        self.config = cfg['models'][name]
        self.model_dir = Path(model_dir).expanduser().resolve()
        if not self.model_dir.is_dir():
            raise FileNotFoundError(self.model_dir)
        self.native = None
        self.model = None
        self.tokenizer = None

    def verify_weights(self):
        manifest = json.loads((ROOT/self.config['manifest']).read_text(encoding='utf-8'))
        raw = manifest['files']
        entries = [dict(value, name=key) for key, value in raw.items()] if isinstance(raw, dict) else raw
        if self.name == 'pddl':
            if (self.model_dir/'adapter_config.json').exists():
                raise ValueError('PDDL uses full merged weights. Create a directory without adapter_* files; do not apply its LoRA again.')
            entries = [entry for entry in entries if not entry['name'].startswith('adapter_')]
            for filename in ('modeling_qwen2_rm.py', 'LICENSE'):
                source = ROOT/'baselines/assets/qwen'/filename
                entries.append(dict(name=filename, size=source.stat().st_size, sha256=file_hash(source)))
        checked = {}
        for entry in entries:
            path = self.model_dir/entry['name']
            if not path.is_file() or path.stat().st_size != entry['size']:
                raise ValueError('Missing or changed checkpoint file: ' + str(path))
            expected = entry.get('sha256') or entry.get('lfs', {}).get('sha256')
            algorithm = 'sha256' if expected else 'sha1'
            expected = expected or entry.get('git_blob_sha1') or entry.get('blobId') or entry.get('git_sha1')
            actual = file_hash(path, algorithm)
            if not expected or actual != expected:
                raise ValueError('Checkpoint hash mismatch: ' + str(path))
            checked[entry['name']] = dict(algorithm=algorithm, hash=actual)
        return checked

    def load(self):
        native = importlib.import_module('.native.'+self.name, 'baselines')
        self.native = native
        if self.name in ('qwen', 'pddl', 'shepherd'):
            native.MODEL = self.model_dir
            self.tokenizer = native.tokenizer()
            self.model = native.load()
        elif self.name == 'thinkprm':
            self.tokenizer, self.model = native.load_model(self.model_dir)
        else:
            self.model, self.tokenizer = native.load_model(self.model_dir)
        return self

    def predict(self, case, output=None, binding=None):
        if self.model is None:
            raise RuntimeError('Call load() before predict()')
        if self.name == 'thinkprm':
            result = self.native.generate_one(self.tokenizer, self.model, case, 2048, 180, 2048)
            result['probabilities'] = [None if value is None else float(value) for value in result['parsed']['labels']]
            result['targets'] = case['targets']
        elif self.name == 'genprm':
            if output is None or binding is None:
                raise ValueError('GenPRM requires a result directory and immutable run binding')
            result = self.native.predict(self.model, self.tokenizer, case, Path(output), binding)
        else:
            result = self.native.predict(self.model, self.tokenizer, case)
        if len(result['probabilities']) != len(case['targets']):
            raise ValueError('Native outputs are not aligned to public targets')
        return result


def evidence_from_prediction(name, case, prediction, prepared):
    config = configuration()['models'][name]
    if prediction['targets'] != case['targets']:
        raise ValueError('Diagnostic target order differs from the public case')
    values = dict(zip(prediction['targets'], prediction['probabilities']))
    required = [n['node_id'] for n in prepared['nodes'] if n['claim']['kind'] not in ('given', 'cue')] + ['answer']
    if len(prediction['probabilities']) != len(case['targets']) or any(t not in values for t in required):
        raise ValueError('Missing or misaligned diagnostic positions')
    return {target: shift(values[target], config['threshold']) for target in required}
