"""Fast, model-free validation of the frozen MathRepairBench text release."""
import argparse
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = {"train": (613, 2452), "validation": (129, 516), "test": (122, 488)}
VARIANTS = {"correct", "logical_error", "execution_slip", "self_corrected"}
PUBLIC_KEYS = {"question", "steps", "student_answer", "schema", "retractions",
               "semantic_dependencies", "semantic_status", "sequence_edges"}
STEP_KEYS = {"node_id", "position", "display", "latex", "surface_features", "visible_cues"}


def canonical_sha256(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def indexed(rows, name):
    result = {row["item_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate item_id in " + name)
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_dataset(data_root=None):
    """Check file/content hashes, schema, split identity and family grouping.

    This checks every small text record, but performs no model inference,
    mathematical rescoring, training or image processing.
    """
    start = time.monotonic()
    root = Path(data_root) if data_root else Path(__file__).resolve().parents[1] / "data/mathrepairbench"
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["schema"] == "mathrepairbench-release-manifest-v1", "Unknown manifest schema")
    expected = set(manifest["files"]) | {"manifest.json"}
    present = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(present == expected, "Unexpected or missing dataset files: " + str(sorted(present ^ expected)))
    for relative, info in manifest["files"].items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root), "Manifest path escapes dataset directory")
        data = path.read_bytes()
        require(len(data) == info["bytes"], "Byte count differs: " + relative)
        require(hashlib.sha256(data).hexdigest() == info["sha256"], "SHA256 differs: " + relative)
        text = data.decode("utf-8")
        require(not re.search(r"(?i)\b[a-z]:[\\/]|/(?:Users|root|home)/", text),
                "Local absolute path in release data: " + relative)
        require(time.monotonic() - start < 30, "Dataset validation exceeded 30-second bound")

    admin = indexed(read_jsonl(root / "admin/case_groups.jsonl"), "admin")
    mapping = indexed(read_jsonl(root / "admin/id_mapping.jsonl"), "mapping")
    require(set(admin) == set(mapping), "Administrative and source mapping IDs differ")
    require(len({m["case_id"] for m in mapping.values()}) == len(mapping), "Duplicate original case ID")
    public, labels, families, split_families = {}, {}, defaultdict(list), {}
    counts = {}
    for split, (family_count, trace_count) in SPLITS.items():
        ids = json.loads((root / "splits" / (split + ".json")).read_text(encoding="utf-8"))
        pub = indexed(read_jsonl(root / "public" / (split + ".jsonl")), split + " public")
        gold = indexed(read_jsonl(root / "labels" / (split + ".jsonl")), split + " labels")
        require(ids["split"] == split, "Wrong split name")
        require(len(ids["item_ids"]) == len(set(ids["item_ids"])) == trace_count, "Wrong split size")
        require(set(pub) == set(gold) == set(ids["item_ids"]), "Public/gold/frozen split IDs differ")
        require(not set(public).intersection(pub), "A trace occurs in several splits")
        require(all(admin[i]["family_split"] == split for i in pub), "Administrative split drift")
        require(sorted({mapping[i]["family_id"] for i in pub}) == ids["family_ids"], "Original family IDs differ")
        require(sorted({admin[i]["family_key"] for i in pub}) == ids["family_keys"], "Export family keys differ")
        require(len(ids["family_ids"]) == len(ids["family_keys"]) == family_count, "Wrong family count")
        split_families[split] = set(ids["family_keys"])
        by_variant = dict(sorted(Counter(admin[i]["variant"] for i in pub).items()))
        require(by_variant == {v: family_count for v in VARIANTS}, "Variant balance differs")
        counts[split] = {"families": family_count, "traces": trace_count, "by_variant": by_variant}
        public.update(pub)
        labels.update(gold)

    require(set(public) == set(admin) == set(mapping), "Coverage differs between release layers")
    require(sum(map(len, split_families.values())) == len(set.union(*split_families.values())),
            "A problem family occurs in several splits")
    node_count = 0
    hashes = {}
    for iid, row in public.items():
        require(set(row) == {"item_id", "payload"}, "Nonpublic top-level field")
        p = row["payload"]
        require(set(p) == PUBLIC_KEYS, "Unknown/nonpublic payload field: " + iid)
        require(set(p["question"]) == {"text", "latex"}, "Question schema differs")
        require(p["schema"] == "written-observation-v1", "Unknown observation schema")
        require(p["semantic_dependencies"] is None and p["retractions"] is None
                and p["semantic_status"] == "NOT_INFERRED", "Gold semantic information in public payload")
        require(all(set(s) == STEP_KEYS for s in p["steps"]), "Nonpublic step field")
        n = len(p["steps"])
        require([s["position"] for s in p["steps"]] == list(range(1, n + 1)), "Nonconsecutive positions")
        require(len({s["node_id"] for s in p["steps"]}) == n, "Duplicate node ID")
        require([s["step_index"] for s in labels[iid]["steps"]] == list(range(1, n + 1)), "Gold alignment differs")
        require(mapping[iid]["family_key"] == admin[iid]["family_key"]
                and mapping[iid]["family_split"] == admin[iid]["family_split"], "Source mapping differs")
        hashes[iid] = canonical_sha256(p)
        require(hashes[iid] == mapping[iid]["public_payload_sha256"], "Original public payload hash differs")
        node_count += n
        families[admin[iid]["family_key"]].append(iid)
    require(len(families) == 864 and len(public) == 3456 and node_count == 12056, "Benchmark scale differs")
    for siblings in families.values():
        require(len(siblings) == 4 and {admin[i]["variant"] for i in siblings} == VARIANTS, "Incomplete family")
        require(len({mapping[i]["family_id"] for i in siblings}) == 1, "Original family mapping differs")
        require(len({public[i]["payload"]["question"]["text"] for i in siblings}) == 1, "Family question differs")
    skill_count = Counter(admin[i]["skill_id"] for i in public)
    require(len(skill_count) == 12 and set(skill_count.values()) == {288}, "Skill counts differ")
    require(counts == manifest["counts"], "Manifest counts differ")
    require(canonical_sha256(hashes) == manifest["canonical_payload_set_sha256"], "Whole payload set differs")
    return {"status": "PASS", "files": len(expected), "skills": len(skill_count),
            "families": len(families), "traces": len(public), "written_nodes": node_count,
            "splits": counts, "elapsed_seconds": round(time.monotonic() - start, 3),
            "model_inference_run": False, "mathematical_rescoring_run": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="Override data/mathrepairbench directory")
    args = parser.parse_args()
    print(json.dumps(verify_dataset(args.data_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
