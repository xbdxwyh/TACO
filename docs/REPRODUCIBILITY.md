# Reproducibility and metrics

## Frozen components

The included diagnostic is the terminal contextual-validity checkpoint at
8,000 updates. The ranker comes from the established joint model at 2,000
updates. They are distinct networks. Exported tensors are identical to their
original checkpoints; optimizer state, training resume data and machine paths
have been removed. File and tensor hashes are supplied with each checkpoint.

Core mathematical parsing, public candidate generation, the eight-condition
controller and final scoring are retained from the frozen experiments. Only
package-relative imports were changed in `src/taco/core/`. The portable command
line, checkpoint loading and result aggregation are new release interfaces.
Numerical differences across PyTorch versions or CPU/GPU kernels can affect
scores near a threshold. Use the recorded environment and frozen weights when
comparing with paper results; retraining is not claimed to be bitwise identical.

## Prediction and scoring boundary

1. Prediction reads a public JSONL file containing only `item_id` and `payload`.
   It does not read split-family metadata, reference labels or answers.
2. The diagnostic encodes each written mathematical expression and the submitted
   answer. Answer diagnosis includes the observed mathematical history.
3. Track interprets probabilities in written order. Correct selects candidates
   from the fixed public menu and verifies edits under the fixed budget.
4. A completed prediction file is sealed. Independent scoring then reads the
   original construction labels and checks the edited process with the separate
   symbolic reference implementation.

Full TACO is `S1_learned_uncertain`: Track, learned Rank, and one selective
original-step Audit. Audit reserves one of eight online judgments; up to seven
candidates can then be checked even if no audit-eligible step exists. Without
Audit, up to eight candidates can be checked. Turning off Track removes history
retirement and open-goal management and also changes the eligible audit pool;
it is a controller ablation, not removal of a neural graph network.

## Metrics

Rates in machine-readable summaries are **fractions**; multiply by 100 for the
paper tables. For an edited output, `complete` means that applicable process
steps, the answer and required goal scopes all pass, with no unknown result.
The independent scorer honors genuine *original* correction history. A newly
edited position cannot retroactively exempt another unrepaired position.

| Metric | Numerator / denominator |
|---|---|
| Repair | Complete outputs / initially erroneous traces |
| Delivery | Complete outputs declared complete / initially erroneous traces |
| Answer | Correct final answers with no scoring unknown / initially erroneous traces |
| False Acceptance (FA) | Incomplete outputs declared complete / all traces |
| Protection Edit | Correct/self-corrected traces edited at least once / that subset |
| Protection Failure | Outputs failing complete-process checks / that subset |
| Protection Refusal | Outputs not declared complete / that subset |
| SC Edit/Failure/Refusal | The same three quantities, restricted to self-corrected traces |
| Calls | Sum of symbolic tool calls / all inputs |

Answer follows the paper's conservative unknown policy: a scoring unknown
does not count as answer success. FA is not the conditional error rate among
accepted outputs. Unknown and unsupported cases are retained, not dropped.

Calls includes problem-context solving, original-step checks, candidate checks
and, for external models, paid checks for missing evidence. The complete
breakdown is retained. It excludes neural inference, menu construction,
offline independent scoring and GenPRM's native code execution. Rank inputs
are reported separately; symbolic calls are not a wall-clock speed claim.

Each seed reuses the same inputs. The summary averages seed rates without
counting seeds as extra independent families. Skill and variant panels are
included. The test set has a historical usage record; this release does not
present it as a newly collected blind test or as unseen-skill generalization.
Published family-clustered paired comparisons remain in reference results
where supplied; the portable scorer does not treat seed repetitions as
independent confidence-interval samples.

## Scope of included reproduction

The main runtime covers TACO and its eight component combinations. Optional
baseline tools cover the seven frozen diagnostic adapters and their Rule/TACO
repair conditions. These are adaptations to a common repair interface, not
claims that upstream PRMs natively execute this repair task. Thresholds and
unknown handling are frozen; do not replace the missing Qwen validation Rule
result with a result at another threshold.

Historical exploratory scans, cloud launch/shutdown code, internal reports,
images and model-selection experiments are intentionally outside the runnable
release. Compact paper results retain their origin and configuration names.
