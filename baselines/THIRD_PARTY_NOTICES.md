# External diagnostic models and third-party code

This directory provides task adapters, not redistributed model weights. Each
external checkpoint retains its upstream terms. The project's eventual code
license does not replace the licenses of external checkpoints or copied code.

- ThinkPRM prompt/generation adaptation follows `mukhal/ThinkPRM`, revision
  `045b1aed5d72301690a8fc08e846c90b757672c8`. Its upstream MIT notice is retained
  in `assets/thinkprm/UPSTREAM_LICENSE.txt`. Model: `launch/ThinkPRM-1.5B`.
- Skywork encoding/value-head adaptation follows the pinned official
  implementation. Its Apache-2.0 license is retained in
  `assets/skywork/LICENSE`.
- SPARE uses the pinned official instruction and positive/negative token recipe.
  Its Apache-2.0 license and upstream notice are retained in `assets/spare/`.
- GenPRM prompt/reward adaptation follows the pinned official staged inference.
  Its MIT license is retained in `assets/genprm/LICENSE`. The bounded mathematical
  interpreter is a disclosed task adaptation; it is not arbitrary Python.
- Qwen's `configuration_qwen2_rm.py` and `modeling_qwen2_rm.py` are copied without
  modification, with their original copyright headers. The checkpoint's LICENSE
  and model card are retained in `assets/qwen/`. PDDL2PRM uses this same official
  implementation with its full merged reward-model checkpoint; its model card is
  retained in `assets/pddl/`.
- Math-Shepherd follows the published step-marker and two-token probability
  recipe. Its model card is retained in `assets/shepherd/`; obtain its checkpoint
  subject to the upstream terms.

Exact repository revisions and file checksums appear in each model manifest and
in `../configs/baselines.json`. Source-file transformations made for this release
are recorded in `SOURCE_PROVENANCE.json`. No third-party weights, tokenizer
bundles, downloaded environments, or cloud credentials are included.
