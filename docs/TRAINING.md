# Training and released checkpoints

The paper's compact TACO configuration uses **two separately trained models**
per seed. The validity model and candidate ranker have the same architecture
but different parameters and training histories. Both paper seeds, 17 and 29,
are included in `checkpoints/`; no external model download is needed to use
this configuration.

| Component | Training endpoint | Input | Objective used during training |
|---|---:|---|---|
| Diagnostic | 8,000 updates | Problem and current step; the answer additionally sees all visible mathematical steps | 0.5 × class-balanced validity BCE |
| Ranker | 2,000 updates | Problem, original expression, nearest preceding mathematical step, and candidate | 0.5 × local validity BCE + 0.5 × full-menu multi-positive ranking loss |

Each network has four Transformer layers, width 192, six attention heads,
and 1,865,090 parameters. The typed syntax vocabulary, role embeddings,
position embeddings, and three numeric features are unchanged from the
paper. Inputs longer than 384 tokens or outside the supported syntax are
reported as unsupported, rather than silently truncated. The rank model
still contains its historical validity head, but normal TACO inference uses
the separate diagnostic checkpoint's validity head.

## Frozen paper weights

The following files contain a CPU state dictionary and JSON-compatible
metadata only:

```text
checkpoints/diagnostic_seed17.pt
checkpoints/diagnostic_seed29.pt
checkpoints/rank_seed17.pt
checkpoints/rank_seed29.pt
```

Exporting these weights did not retrain them. Every exported tensor was
compared exactly with the corresponding frozen experiment tensor using
`torch.equal`. Optimizer states, experiment resume state, and local absolute
paths were removed. `manifest.json` and each checkpoint's `.json` companion
record the export hash, original checkpoint hash, original protocol digest,
architecture, vocabulary, seed, training endpoint, and model tensor digest.
The exported file hash differs from its original training file because the
artifact contains fewer fields; the model tensor digest is unchanged.

Use `torch.load(..., map_location="cpu", weights_only=True)` and
`Model.load_state_dict(..., strict=True)`. Verify the manifest file hash and
the metadata's `tensor_sha256` with `math12_model_v1.initialization(model)`.
The tensor digest follows the frozen implementation's state-dictionary
order. No pickle objects beyond supported state-dictionary primitives are
needed.

## Portable reconstruction

`python -m taco.training` reconstructs training from the released text data.
It reads only `public/train.jsonl`, `labels/train.jsonl`, and the training
rows of `admin/case_groups.jsonl`. It never reads validation/test payloads or
labels, never evaluates a held-out partition, and never selects a checkpoint
using a development or test score.

Preparation generates the same public candidate menus, labels each candidate
with the online mathematical interface, and independently cross-checks a
positive and a negative candidate when available in each list. Original-step
and answer validity labels are recomputed using that mathematical interface,
as in the paper's audited training teacher. Construction annotations are
compared for audit and are not silently substituted for these labels.
Unknown candidate judgments are excluded from both positive and negative
sets. Complete candidate lists with positive and negative examples are
deduplicated by their encoded inputs.

The original full training pool contained 2,452 traces, 8,054 validity
examples, 230,473 candidate judgments (278 unknown), and 6,966 unique usable
ranking lists from 613 families. The portable runner writes its own pool
counts to `training_pool.json` so that a reproduction can check these counts.
Teacher caches are bound to the source data and released implementation.

The schedule is specified in `configs/training.json`: AdamW, learning rate
3e-4, weight decay 0.01, gradient clipping at 1, and 16 validity examples per
update (eight per class, sampled with replacement). The learning rate follows
the original cosine schedule, ending near 20% of its initial value. Rank
updates sample one family and one eligible list from that family. The ranking
objective is the log-sum-exp over known candidates minus the log-sum-exp over
positive candidates. It supervises mathematical validity, not final repair
completion; completion checks remain part of the controller.

The public runner retains these objectives, input visibility rules, sampling
scheme, and fixed endpoints. It reconstructs the teacher cache instead of
depending on a chain of private experiment folders. Its candidate-list
enumeration is explicitly canonicalized by written target order. Historic
cache serialization, hardware, and numerical-library versions can affect
ordering or floating-point execution, so **re-training is not promised to
reproduce the paper's weights or scores bit for bit**. Use the supplied frozen
weights for paper-result reproduction. No newly trained result is included
in the paper tables.

## Commands

Run from the repository root after installing the package. A bounded
engineering check processes two complete training families (eight traces)
and runs two updates for each component:

```bash
python -m taco.training --mode smoke --component both --seed 17 --device cpu --output outputs/training-smoke
```

The smoke checkpoints are explicitly marked `engineering_smoke: true`; they
must not replace the paper weights. Use a fresh output directory when changing
from smoke to full training.

Full training is a user-started operation. Run one seed at a time:

```bash
python -m taco.training --mode full --component both --seed 17 --device cuda --output outputs/training-seed17
python -m taco.training --mode full --component both --seed 29 --device cuda --output outputs/training-seed29
```

`--component rank` and `--component diagnostic` can train one component.
`--resume` resumes a matching output folder; mismatched data or configuration
is rejected. Resume checkpoints are stored every 100 updates and at the
terminal update. A process interrupted during an update resumes from the
last saved update. The output contains teacher caches, a pool audit, resume
state, and inference-only terminal checkpoints. It does not overwrite the
release's data, source code, configuration, or frozen checkpoints.

The main inference CLI uses the release's default checkpoint directory.
To use a retrained checkpoint, load the files in that run's `checkpoints/`
directory explicitly rather than replacing the released paper artifacts.

## Release boundary

The repository contains the two training schedules needed for the main
configuration, not every historical development experiment. Full four-arm
training controls, obsolete dataset interfaces, AutoDL launchers, and
experiment-folder integrity chains are not needed for ordinary use of TACO.
Their absence does not change the supplied frozen models. External PRM
backbones use their own upstream model weights and licenses; those weights
are not bundled here.
