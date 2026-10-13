# Release engineering validation

The preparation ran short, bounded checks only. No full experiment, training
run, parameter sweep or external model inference was started.

| Check | Scope | Result |
|---|---|---|
| Data integrity | All frozen files, counts, hashes, ID maps and family partitions; no mathematical rescoring | Pass |
| Checkpoint export | Four models compared tensor by tensor to frozen originals | Exact equality |
| Regression tests | 20 core semantics and prediction-integrity tests | Pass |
| Prediction/scoring CLI | Four public validation inputs, one seed | Pass |
| Original-run parity | One function-composition family; four variants, seeds 17/29, eight component configurations | 64 controller outputs matched |
| Training entry point | Two train families (eight traces); two updates per component | Pass |
| Text generator | One family per skill (48 traces); matched by original family ID and variant | Public content and gold matched |
| External repair path | Four cached Math-Shepherd inputs, Rule and +TACO, one seed | Matched original controller fields |
| Native adapter imports | Seven adapters imported without downloading or loading large weights | Pass |
| Release-content audit | Code, metadata, data and model metadata; excluded temporary outputs | No credentials or personal absolute paths found |

`provenance/ENGINEERING_PARITY.json` records the bounded main-method comparison.
`baselines/RELEASE_CHECKS.json` records the external-adapter checks. Full reported
results in `reference_results/` come from the completed original experiments,
not these smoke runs. Full native inference through each newly packaged adapter
and complete retraining have not been rerun during preparation.

Recheck the public contracts with:

```sh
python scripts/verify_dataset.py
python scripts/verify_release.py
python -m unittest discover -s tests -v
```
