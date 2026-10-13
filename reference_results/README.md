# Reference results from completed experiments

These files are exports of **previously saved results**, not newly run
experiments. `SOURCE_MANIFEST.json` records original source filenames and SHA256
digests. User-specific absolute paths have been replaced with portable
provenance labels; all numerical records are preserved.

| File | Contents |
|---|---|
| `main.csv` | Main external Rule adaptations and full task-trained TACO; Val/Test |
| `ablation.csv` | Eight Track/Rank/Audit combinations with the same task-trained diagnostic |
| `transfer.csv` | Each frozen external PRM with Rule and with TACO; Val/Test |
| `paper_tables.json` | Unrounded values and source checks for those three tables |
| `all_component_combinations.json` | Wider saved external Track/Rank inventory |
| `protection.json` | Correct/self-corrected intervention, failure and preservation records |
| `costs.json` | Symbolic checks, ranking inputs and separate native-model costs |
| `error_types.json` | Logical-error/execution-slip and skill-level results |
| `figure4_5_source_data.json` | Audited numerical inputs to the protection/efficiency plots |

Rates are percentages. CSV values retain eight decimal places; an empty cell is
unavailable, not zero. The Qwen Val Rule baseline at threshold 0.9 is missing.
No result at a different threshold replaces it.

Each seed sees the same validation/test families. Averages over seeds 17 and 29
do not create additional independent examples. Validation: 516 traces / 129
families; test: 488 traces / 122 families. Repair/Delivery/Answer use initially
erroneous traces; FalseAccept uses all inputs; SC Edit uses self-corrected
traces. Calls is symbolic calls per input and excludes native PRM inference.
See the manuscript and `docs/BASELINES.md` for the experimental distinctions.

The compact transfer table in the final manuscript uses a subset of these
columns and highlights paired changes. The underlying values remain in this
export so that a reader can reconstruct that presentation without new inference.
