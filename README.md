# TACO and MathRepairBench

TACO (Track And COrrect) repairs mathematical solutions by tracking which
errors remain active and which task goals are still open. MathRepairBench
provides matched correct, logical-error, execution-slip and self-corrected
traces for studying both repair and unnecessary intervention.

**Local release preparation.** Licensing is pending the authors' decision.
No code, data or checkpoint license is granted by this preparation alone.
There is no public repository URL or archived release DOI yet. External
models retain their upstream terms. See [release status](docs/RELEASE_STATUS.md).

## What is included

| Directory | Content |
|---|---|
| `src/taco/` | Public-input diagnosis, state tracking, repair, independent scoring and training |
| `data/mathrepairbench/` | Frozen text/LaTeX inputs, separate labels, family splits and provenance |
| `checkpoints/` | Four small frozen TACO checkpoints: diagnostic and ranker, seeds 17 and 29 |
| `baselines/` | Native adapters and common repair interfaces for seven external PRMs |
| `configs/` | Frozen training settings and external-model configurations |
| `reference_results/` | Compact results from completed paper experiments, with source provenance |
| `scripts/`, `tests/` | Dataset verification, release checks and short contract tests |

The text data and our small checkpoints fit in this repository. External
pretrained weights are downloaded separately. Images, simulated learner
trajectories, historical pilot outputs, cloud credentials and old orchestration
scripts are not needed for these text experiments and are not included.

## Install

Use Python 3.11 or later. Install the PyTorch wheel appropriate for your CPU or
CUDA device, then install from this repository's root:

```sh
python -m pip install -e .
python scripts/verify_dataset.py
python scripts/verify_release.py
python -m unittest discover -s tests -v
```

`requirements-core.txt` records the original local package versions. The core
requires no API key or external language-model download. A GPU is optional for
the small included models; it is useful for training and larger runs.

## Four-input smoke run

```sh
python -m taco predict --split validation --seeds 17 --configs taco --limit 4 --device cpu --output outputs/smoke
python -m taco score --predictions outputs/smoke
```

Prediction reads **only** `public/<split>.jsonl`. It saves executed edits,
completion decisions and costs, then seals `predictions.jsonl` with SHA256.
Scoring is a separate command that reads labels after sealing. Results are in
`outputs/smoke/evaluation/summary.json` and `scores.jsonl`.
Four inputs only check the implementation; they do not estimate performance.

## Reproduce the main method and component ablations

These commands run a complete split and are intended to be started explicitly:

```sh
python -m taco predict --split validation --seeds 17 29 --configs all --limit 0 --output outputs/validation --resume
python -m taco score --predictions outputs/validation
python -m taco predict --split test --seeds 17 29 --configs all --limit 0 --output outputs/test --resume
python -m taco score --predictions outputs/test
```

`taco` selects the complete method; `all` selects the published 2 x 2 x 2
Track/Rank/Audit ablations. Exact condition names such as
`S0_rule_uncertain` are also accepted. The same output directory can be resumed
only with identical inputs, settings, code and checkpoints.

Training from scratch is documented in [TRAINING.md](docs/TRAINING.md).
External models and the distinction between their repair adaptations and
PRM + TACO are documented in [BASELINES.md](docs/BASELINES.md).

To inspect the original text construction without rendering images:

```sh
python scripts/generate_dataset.py --families-per-skill 1 --seed 20260827 --output generated/text-demo
```

This generates a separate 48-trace demonstration. The frozen dataset remains
the input for reproducing paper results; do not mix IDs from different
generation settings.

## Dataset and evaluation

MathRepairBench contains 864 families and 3,456 traces across 12 mathematical
skills. Train/validation/test contain 613/129/122 disjoint families
(2,452/516/488 traces). All four variants of a family remain in one split.
These are controlled, programmatically constructed solutions, not collected
student submissions. The split tests new families within the same skill set.

The scorer reports Repair, Delivery and Answer on initially erroneous traces;
False Acceptance on all traces; intervention, final failure and refusal on
correct/self-corrected traces; and symbolic tool costs. An edit to a
self-corrected trace is not automatically counted as damage. Unknown scoring
outcomes remain in the denominator. See [DATASET.md](docs/DATASET.md) and
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for definitions and scope.

## Citation and release status

`CITATION.cff` describes this preparation without inventing a publication year,
DOI or repository URL. The paper is in preparation; replace this metadata with
the final paper and data-release citation when available. The authors will
choose code/data licensing before public distribution.
