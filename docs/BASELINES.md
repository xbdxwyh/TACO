# Frozen external baselines

The seven external models provide **native process diagnostics**. A PRM does not
itself output an end-to-end edited mathematical solution. The paper compares
the following task adaptations, using the same public candidate menus and
mathematical checks:

| Configuration | Track | Candidate ordering | Original-step Audit |
|---|---|---|---|
| PRM + Rule | Off | Fixed rule order | One selective check |
| PRM + TACO | On | Frozen trained ranker | One selective check |

Thus the Rule row is a reproduction of native diagnostics within the shared
repair interface, not an original-paper end-to-end repair system. The +TACO row
adds our history/goal tracking and trained ranking. Do not label all external
rows +TACO or treat Skywork + TACO as a standalone Skywork repair method.

## Models and frozen settings

| Name | Exact checkpoint ID | Main threshold | Context cap |
|---|---|---:|---:|
| Math-Shepherd | `peiyi9979/math-shepherd-mistral-7b-prm` | 0.5 | 4,096 |
| Skywork-PRM | `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B` | 0.5 | 4,096 |
| Qwen-PRM | `Qwen/Qwen2.5-Math-PRM-7B` | 0.9 | 4,096 |
| SPARE | `UKPLab/Qwen2.5-3b-spare-prm-math` | 0.5 | 1,024 |
| PDDL2PRM | `Babelscape/Qwen2.5-Math-PRM-7B-PDDL-r` | 0.9 | 4,096 |
| GenPRM | `GenPRM/GenPRM-1.5B` | 0.5 | 32,768 |
| ThinkPRM | `launch/ThinkPRM-1.5B` | 0.5 | 2,048 input |

Use the exact revisions in `configs/baselines.json`, not a moving `main` branch.
All native inference uses batch one, BF16, and no quantization. Skywork's value
head remains FP32. Qwen/PDDL/Shepherd use eager attention; the others use SDPA.
The local reference environment used Python 3.11.13, PyTorch 2.7.1+cu126,
Transformers 4.53.2, SymPy 1.13.3, NumPy 2.1.2 and safetensors 0.5.3.
The cloud runs used Transformers 4.53.2, accelerate 1.10.1, safetensors 0.6.2,
huggingface-hub 0.34.4 and tokenizers 0.21.4; Shepherd used sentencepiece 0.2.1.
Qwen/PDDL/Shepherd loading additionally needs `accelerate`; Shepherd needs
`sentencepiece`. Install a PyTorch build appropriate to the actual GPU rather
than downloading the archived author's environment.

ThinkPRM uses greedy generation with at most 2,048 new tokens and a 180-second
case limit. The corrected parser is `scoped_public_judgments_v3`: numbered and
scoped judgments are mapped to original positions; missing or truncated output
remains unknown. An unscoped negative global verdict never identifies a step.

GenPRM uses corrected implementation v2: sequential analyze/verify/execute,
temperature 0.6, top-p 0.95, top-k 20, up to three verification rounds, and a
bounded final judgment fallback. Generated mathematical code is interpreted by
an AST allowlist, never passed to `exec` or `eval`. The reference Windows run
also applied a 1-GiB process limit; the exported executor retains that Windows
limit and a five-second request timeout. Other operating systems retain the
timeout/allowlist but do not thereby reproduce the Windows memory mechanism.
Code execution is an additional native-model cost, separate from repair calls.

Native scores are mapped to the controller's fixed 0.5 decision boundary by the
same odds transformation used in the experiments. `None` remains `None`.
`baselines/recovery.py` is the frozen missing-evidence controller: resolving an
unknown original observation consumes the **same eight-check repair budget**.
Unknowns are not silently filled with a neutral or positive prediction. The
controller reserves one original-step Audit slot; at most seven remaining
checks can be used by missing evidence and candidate verification.

## Run native diagnostics

Install the project editable from the repository root. Acquire the selected
checkpoint separately at its pinned revision and give its local folder using
`--model-dir`. No baseline command downloads, deletes, or swaps checkpoints.

```powershell
python -m baselines native --model skywork --model-dir "D:/models/Skywork-o1-Open-PRM-Qwen-2.5-1.5B" --public data/mathrepairbench/public/test.jsonl --output outputs/skywork-native-smoke --limit 4
```

The public JSONL format is one `{ "item_id": ..., "payload": ... }` record per
line. The diagnostic serializer uses only the public problem, written steps
and submitted answer. It does not read gold labels, variants or paired cases.
The output is `diagnostics.jsonl`, with per-case raw native outputs, aligned
probabilities, unchanged unknowns, threshold-adjusted evidence, input hashes,
and resumable per-case records. Pass `--limit 0` explicitly for all inputs.

For Qwen/PDDL, install the included, pinned official custom Python files into
the local checkpoint folder before inference:

```powershell
python -m baselines prepare-code --model pddl --model-dir "D:/models/Qwen2.5-Math-PRM-7B-PDDL-r"
```

PDDL must use the full merged checkpoint. Its directory must not include the
separately released `adapter_*` files, otherwise Transformers may automatically
apply a second adapter. The preparation command only copies pinned source files
and refuses to overwrite differing files; it does not delete model files.

## Recover from cached diagnostics

Native generation is shared across rank seeds and conditions. Once diagnostics
are saved, the PRM itself is no longer loaded for repair:

```powershell
python -m baselines recover --model skywork --public data/mathrepairbench/public/test.jsonl --diagnostics outputs/skywork-native-smoke/diagnostics.jsonl --condition rule --seed 17 --split test --output outputs/skywork-rule-smoke --limit 4
```

To produce both conditions in the same call, use the bundled **matching frozen
rank checkpoint** for the +TACO branch. Its hash, encoding and seed are checked
against its metadata. The Rule branch never loads or uses a learned ranker:

```powershell
python -m baselines recover --model skywork --public data/mathrepairbench/public/test.jsonl --diagnostics outputs/skywork-native-smoke/diagnostics.jsonl --condition both --checkpoints checkpoints --seed 17 --split test --output outputs/skywork-both-smoke --limit 4
```

Output `predictions.jsonl` contains `{item_id, seed, results:[...]}` records and
is bound by `PREDICTION_SEAL.json`. Each result is the executed controller record,
including actual edits, unresolved targets, declaration, and tool events.
Score the sealed output with the repository's independent scorer:

```powershell
python -m taco score --data data/mathrepairbench --predictions outputs/skywork-both-smoke
```

These commands are small interface checks. Full experiments use all inputs and
both frozen rank seeds 17/29; they are started explicitly by the user. No full
native-model inference was run while preparing this release.

Native generation resumes saved cases only if their public input, configuration,
source-code binding and content checksum still match. Recovery is rerun from the
saved native diagnostics; it does not currently resume individual repair cases.
Existing differing results or protocols are never overwritten. Use a fresh output
directory when changing inputs, settings or implementation. Output directories
must not overlap data, checkpoints, native model files, or source code.

## Interpreting the saved results

The original test uses 488 traces from 122 families; validation uses 516 traces
from 129 families. Seeds share families and do not double the independent sample
size. The historical fixed test had been used previously; it is not described
as a new blind test. The Qwen validation Rule result at 0.9 was not saved and is
left missing. The 0.5 result is not substituted.

`reference_results/` contains the paper tables and focused protection, cost and
error-type records. It does not contain all historical experiments or weight
caches. Native inference time and generated-code execution remain separate from
the common symbolic repair cost; the tables do not claim that all methods have
equal total compute.
