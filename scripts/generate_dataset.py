"""Generate a separate, text-only MathRepairBench construction example.

The shipped data/mathrepairbench export remains authoritative for the paper.
This optional command never replaces it. Default: 1 family per skill (48 traces).
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from taco.generation import (ORIGINAL_EXPORT_CONTRACT_SHA256,
                             ORIGINAL_GENERATOR_SHA256, controlled_payload,
                             digest, flatten_cases, generate_families, opaque)


GOLD_FIELDS = ("correct_answer", "final_correct", "contains_error", "error_persisted",
               "first_error_step", "repair_step", "error_mechanism", "error_level",
               "gold_diagnosis_zh", "gold_state_operation", "gold_state_key", "evidence_strength")
GOLD_STEP_FIELDS = ("step_index", "relation_to_previous", "locally_valid",
                    "globally_valid", "error_origin_step")
ADMIN_FIELDS = ("family_split", "writer_ood_split", "mechanism_ood_split", "skill_id",
                "writer_style_id", "variant", "repair_style", "family_error_mechanism")


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")


def save_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                            for row in rows), encoding="utf-8")


def export_generated(cases, output, seed, families_per_skill):
    """Match the original export allowlists; do not pass raw cases to inference."""
    public, labels, admin, mapping = {}, {}, {}, {}
    for case in cases:
        iid = opaque("controlled", case["case_id"])
        family = opaque("family", case["family_id"])
        payload = controlled_payload(case)
        public[iid] = {"item_id": iid, "payload": payload}
        labels[iid] = {
            "item_id": iid,
            "source_gold": {key: case[key] for key in GOLD_FIELDS},
            "steps": [{key: step[key] for key in GOLD_STEP_FIELDS} for step in case["trace_steps"]],
        }
        admin[iid] = {"item_id": iid, "family_key": family,
                      **{key: case[key] for key in ADMIN_FIELDS}}
        mapping[iid] = {
            "item_id": iid, "case_id": case["case_id"], "family_id": case["family_id"],
            "family_key": family, "family_split": case["family_split"],
            "public_payload_sha256": digest(payload),
        }
    if len(public) != len(cases):
        raise ValueError("Generated item ID collision")
    counts = {}
    for split in ("train", "validation", "test"):
        ids = sorted(i for i in public if admin[i]["family_split"] == split)
        family_ids = sorted({mapping[i]["family_id"] for i in ids})
        family_keys = sorted({mapping[i]["family_key"] for i in ids})
        save_jsonl(output / "public" / (split + ".jsonl"), [public[i] for i in ids])
        save_jsonl(output / "labels" / (split + ".jsonl"), [labels[i] for i in ids])
        save_json(output / "splits" / (split + ".json"),
                  {"split": split, "item_ids": ids, "family_ids": family_ids, "family_keys": family_keys})
        counts[split] = {"families": len(family_ids), "traces": len(ids),
                         "by_variant": dict(sorted(Counter(admin[i]["variant"] for i in ids).items()))}
    save_jsonl(output / "admin/case_groups.jsonl", [admin[i] for i in sorted(admin)])
    save_jsonl(output / "admin/id_mapping.jsonl", [mapping[i] for i in sorted(mapping)])
    metadata = {
        "schema": "mathrepairbench-generated-text-v1",
        "seed": seed,
        "families_per_skill": families_per_skill,
        "families": len(cases) // 4,
        "traces": len(cases),
        "counts": counts,
        "is_frozen_paper_export": False,
        "source_type": "deterministic_programmatic",
        "generated_trace_uses_llm": False,
        "generated_trace_uses_diffusion": False,
        "original_generator_sha256": ORIGINAL_GENERATOR_SHA256,
        "original_export_contract_sha256": ORIGINAL_EXPORT_CONTRACT_SHA256,
        "generation_module_sha256": hashlib.sha256((ROOT / "src/taco/generation.py").read_bytes()).hexdigest(),
        "scope": "Construction output; no images, learner trajectories, model calls or scoring.",
        "identity_warning": "IDs follow the original sequential-case scheme and are not globally unique across generator settings. Do not merge with the frozen release or another generated collection.",
        "paper_data": "Use the shipped data/mathrepairbench and its fixed split lists for paper reproduction.",
        "rights_status": "License to be selected by the authors.",
    }
    save_json(output / "GENERATION.json", metadata)
    save_json(output / "checksums.json", {
        p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*")) if p.is_file()
    })
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New/empty output directory outside the frozen data directory")
    parser.add_argument("--families-per-skill", type=int, default=1,
                        help="1--72 families per skill; default 1 is a small construction demo")
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    if not 1 <= args.families_per_skill <= 72:
        parser.error("--families-per-skill must be between 1 and 72")
    output = args.output.resolve()
    frozen = (ROOT / "data/mathrepairbench").resolve()
    if output == frozen or frozen in output.parents or output in frozen.parents:
        parser.error("Output must be separate from the frozen benchmark and its parent directories")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Output must be a new or empty directory; existing content is never overwritten")
    families = generate_families(args.families_per_skill, args.seed)
    cases = flatten_cases(families, args.seed)
    output.mkdir(parents=True, exist_ok=True)
    metadata = export_generated(cases, output, args.seed, args.families_per_skill)
    print(json.dumps({"status": "GENERATED", "output": str(output),
                      "families": metadata["families"], "traces": metadata["traces"],
                      "is_frozen_paper_export": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
