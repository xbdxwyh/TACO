# MathRepairBench: frozen text release

MathRepairBench is a controlled benchmark for mathematical solution repair.
Each fixed problem has four solution histories: `correct`, `logical_error`,
`execution_slip`, and `self_corrected`. The two error-free final states
(`correct` and `self_corrected`) test whether a system preserves work that
already completes the problem. Historical error presence alone does not
determine whether intervention is needed.

The records are synthetic, deterministically constructed mathematical traces,
not collected student work or measurements of real student cognition. The
generator's provenance records identify seed `20260827`,
`source_type=deterministic_programmatic`, `uses_llm=false`, and
`uses_diffusion=false`. These flags describe trace generation; they do not
assert that no AI assistance was used when developing code or the paper.

## Contents and fixed splits

| Split | Families | Traces | Traces per variant |
|---|---:|---:|---:|
| Train | 613 | 2,452 | 613 |
| Validation | 129 | 516 | 129 |
| Test | 122 | 488 | 122 |
| Total | 864 | 3,456 | 864 |

There are 12 skills, 72 problem families per skill, and 12,056 written nodes.
Nodes include restated givens and visible correction cues as well as mathematical
operations. A family is one fixed parameterized problem, not an independent
reasoning template. The 12 skills and 24 predefined logical-error mechanisms
are described in `data/mathrepairbench/skills.json`.

The original family split is preserved exactly. Use the supplied ID lists;
do not randomly split traces, regenerate partitions from renamed IDs, or
place variants of one family in different partitions. `validation` is the
development split called "Val" in the paper. The test split measures held-out
families within the same skill/template domain, not unseen-skill generalization.

```
data/mathrepairbench/
  public/{train,validation,test}.jsonl
  labels/{train,validation,test}.jsonl
  admin/case_groups.jsonl
  admin/id_mapping.jsonl
  splits/{train,validation,test}.json
  skills.json
  provenance.json
  manifest.json
```

The release contains only the text benchmark used by TACO. It excludes the
separate complex-proof branch, rendered images, simulated learner trajectories,
local asset paths, and unrelated experiments. No images or external downloads
are needed to load these records.

## Public input

Each public JSONL line has exactly two top-level fields:

```json
{"item_id": "f3609a24c8baaa0613150aee", "payload": {"...": "..."}}
```

The `payload` is the original `written-observation-v1` object:

| Field | Meaning |
|---|---|
| `question.text`, `question.latex` | Problem statement in Chinese text and LaTeX |
| `steps` | Written nodes in their original order |
| `steps[].node_id`, `position` | Local node ID and one-based written position |
| `steps[].display`, `latex` | Observed expression or cue; not a corrected reference |
| `steps[].surface_features`, `visible_cues` | Features derived from visible text |
| `student_answer` | Submitted answer, whether correct or incorrect |
| `sequence_edges` | `written_after` edges describing written order only |
| `semantic_dependencies`, `retractions` | `null`: not supplied, rather than known to be absent |
| `semantic_status` | `NOT_INFERRED` |

Inference receives only `payload`. `item_id` is a join key for outputs and
auditing, not a learned feature. Do not treat written-order edges as a gold
semantic dependency graph.

## Administrative metadata and reference labels

`admin/case_groups.jsonl` preserves the original administrative records:
`item_id`, `family_key`, `family_split`, `skill_id`, `variant`,
`family_error_mechanism`, `repair_style`, `writer_style_id`,
`writer_ood_split`, and `mechanism_ood_split`. The writer and alternative OOD
fields are retained for provenance; neither defines the paper's text split.
These administrative fields must not be supplied to the diagnostic model,
candidate generator, or repair policy. For example, `variant` and `repair_style`
would directly reveal evaluation categories. A mathematical parser may infer
problem type from the public expressions; it may not use `skill_id` as input.

`admin/id_mapping.jsonl` maps the exported `item_id` and `family_key` to original
`case_id` and `family_id`, records `family_split`, and gives the canonical SHA256
of each original public payload. This preserves provenance despite reorganizing
the exported records into split-specific files.

Each label line retains `{item_id, source_gold, steps}`:

| Label field | Purpose |
|---|---|
| `source_gold.correct_answer` | Reference answer |
| `contains_error`, `error_persisted`, `final_correct` | Original-history outcomes |
| `first_error_step`, `repair_step` | Historical error and observed correction boundaries |
| `error_level`, `error_mechanism`, `gold_diagnosis_zh` | Constructed error description |
| `steps[].step_index` | One-based alignment to public written nodes |
| `locally_valid`, `globally_valid` | Validity relative to the constructed predecessor and original task |
| `relation_to_previous`, `error_origin_step` | Constructed relation and inherited error origin |
| `gold_state_key`, `gold_state_operation`, `evidence_strength` | Legacy simulated-state annotations; not measured student cognition |

Fields abbreviated in the table belong to `source_gold` or `steps[]` as indicated.
The release keeps those legacy label fields to preserve the frozen records;
their presence does not make them inputs to TACO.

**Information boundary:** use train labels for supervision. Use validation
results for development. Keep all test reference labels, reference answers,
correction boundaries, administrative categories, and post-hoc repair coverage
out of inference. Save final predictions and edits before running independent
scoring. Publishing labels for reproducible offline evaluation does not make
them legitimate test-time inputs. No loader should merge all three layers into
a model prompt.

## Evaluation scope

Initially erroneous traces are `logical_error` and `execution_slip`; correct
and self-corrected traces form the protection set. Each test seed therefore
has 244 erroneous and 244 protection traces, including 122 self-corrected
traces. Repeated model seeds evaluate the same families and do not increase
the number of independent test problems.

Complete repair requires the applicable edited process, submitted answer, and
requested goals to pass final checks with no unresolved scoring unknowns.
Correct delivery additionally requires the system to declare completion.
False acceptance is incomplete output declared complete, divided by all traces.
An edited self-corrected trace is not automatically damaged: intervention and
final process failure are different measures.

Genuinely superseded, unedited errors in an observed self-correction may remain
as history. New edits do not acquire this exemption simply by adding a correct
answer elsewhere. Local validity and answer accuracy alone do not establish
complete repair. The online checker and independent scorer operate within the
12-skill mathematical interface, not arbitrary natural-language proofs.

## Provenance and checks

`provenance.json` records the original export hashes and release transformations
without personal filesystem paths. It is a filtered, reorganized export of the
frozen records, not regenerated data. Original public content and labels are
unchanged. The full upstream public export contained an additional complex-proof
branch; its whole-file hash is therefore different from the release file hashes.

`manifest.json` hashes every release data file. Each payload also has a canonical
content hash: SHA256 of UTF-8 JSON encoded with `ensure_ascii=False`,
`sort_keys=True`, and `separators=(",", ":")`. Source item and family ID lists
were checked against the frozen dataset-facts snapshot when packaging.

Run this quick, standard-library-only check from the repository root:

```sh
python scripts/verify_dataset.py
```

It verifies hashes, field boundaries, frozen split membership, four variants per
family, public/label alignment, counts, and the absence of local absolute paths.
It does not train models, regenerate the benchmark, or rerun mathematical scoring.

## Optional text construction

The optional generator makes the construction rules inspectable without the
original image renderer or simulated learner pipeline. It uses only the Python
standard library. From the repository root:

```sh
python scripts/generate_dataset.py --families-per-skill 1 --seed 20260827 --output generated/text-demo
```

This small example generates 12 families and 48 traces, keeping public inputs,
labels, metadata, and splits in separate files. The output must be new or empty;
the command refuses to write into the shipped benchmark directory. It does not
run a model, make an API call, render an image, or score predictions.

`src/taco/generation.py` is extracted from the original
`controlled_logic_dataset.py` and `original_data_contract_v1.py`. The 12 skill
templates, mathematical operations, parameter sampling, deduplication,
`generate_families`, `flatten_cases`, opaque ID construction, and public-field
allowlist are unchanged. The original source SHA256 values are recorded in the
module and generated metadata. The wrapper reproduces the original separate
label/administrative allowlists; unrelated reporting and rendering dependencies
have been removed.

For inspection, `generate_families` selects parameterized problems, each template
constructs the correct and three alternative traces, and `flatten_cases` assigns
the original family splits and labels. `controlled_payload` then exports only
visible problem text, steps, and the submitted answer. Reference annotations
never enter this public object.

The paper's original setting was 72 families per skill and seed 20260827. Use
the shipped, hash-verified export for reproduction. Generating other settings
creates a different collection: the original case IDs are sequential, so an ID
can refer to different content after changing the number of families or seed.
Do not merge generated collections or substitute them for the paper's frozen
train/validation/test records. Generated metadata explicitly marks outputs as
not the frozen paper export.

## Limitations and release status

The traces are short (2--7 written nodes), use controlled templates and
predefined errors, and do not establish coverage of authentic student mistakes.
There was no full independent teacher annotation round. Visible correction cues
can make some self-corrected traces easier; the paper reports cue-based panels.
Problem-family isolation does not imply mathematical-equivalence or template
isolation across splits. Future visual or real-student extensions are outside
this release.

The authors have not yet selected a public-release license. Repository hosting,
a persistent dataset identifier, and the final citation will be added before
public release; no license or public availability is asserted here.
