from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import shutil

# 配置为（输出目录后缀, 方法列表）；后缀留空时使用 merged_results，列表为 None 时合并全部。
# MERGE_CONFIG = ("", None)
# MERGE_CONFIG = ("", ["PE-noInit", "PE-init", "MoEViE-noInit", "MoEViE-init", "DINOv3-noInit"])
# MERGE_CONFIG = ("different_CLIP", [
#     "OpenCLIP-noInit", "OpenCLIP-init", "SigLIP2-noInit", "SigLIP2-init",
#     "MetaCLIP2-noInit", "MetaCLIP2-init", "PE-noInit", "PE-init", "MoEViE-noInit", "MoEViE-init",
# ])
# MERGE_CONFIG = ("probe_ablation", [
#     "PE-noTrain", "MoEViE-noTrain",
#     "PE-linear1-noInit", "PE-linear1-fake", "PE-linear1-diff", "PE-linear1-fake-040", "PE-linear1-fake-075",
#     "MoEViE-linear1-noInit", "MoEViE-linear1-fake", "MoEViE-linear1-diff", "MoEViE-linear1-fake-040", "MoEViE-linear1-fake-075",
#     "PE-mlp-noInit", "MoEViE-mlp-noInit", "MoEViE-mlp-test",
# ])
# MERGE_CONFIG = ("learning_rate", [
#     # "PE-noInit-lr5e-6", "PE-noInit-lr2e-5", "PE-noInit-lr8e-5", "PE-noInit-lr3e-4", "PE-noInit-lr1e-3", "PE-noInit-lr4e-3",
#     "PE-init-lr5e-6", "PE-init-lr2e-5", "PE-init-lr8e-5", "PE-init-lr3e-4", "PE-init-lr1e-3", "PE-init-lr4e-3",
#     # "MoEViE-noInit-lr5e-6", "MoEViE-noInit-lr2e-5", "MoEViE-noInit-lr8e-5", "MoEViE-noInit-lr3e-4", "MoEViE-noInit-lr1e-3", "MoEViE-noInit-lr4e-3",
#     "MoEViE-init-lr5e-6", "MoEViE-init-lr2e-5", "MoEViE-init-lr8e-5", "MoEViE-init-lr3e-4", "MoEViE-init-lr1e-3", "MoEViE-init-lr4e-3",
# ])
MERGE_CONFIG = ("learning_rate_compare", [
    "PE-noInit-lr2e-5", "PE-init-lr2e-5",
    # "PE-noInit-lr1e-3", "PE-init-lr1e-3",
    "MoEViE-noInit-lr2e-5", "MoEViE-init-lr2e-5",
    # "MoEViE-noInit-lr1e-3", "MoEViE-init-lr1e-3",
])
# MERGE_CONFIG = ("prompt", ["PE-prompt-similar", "PE-prompt-opposite", "PE-prompt-unrelated_i", "PE-prompt-unrelated_ii", "MoEViE-prompt-similar", "MoEViE-prompt-opposite", "MoEViE-prompt-unrelated_i", "MoEViE-prompt-unrelated_ii"])

MERGE_NAME, MERGE_METHODS = MERGE_CONFIG
ROOT = Path(__file__).resolve().parents[1] / "outputs"
DETAILS_DIR = ROOT / (f"merged_results_{MERGE_NAME}" if MERGE_NAME else "merged_results")
OUT = DETAILS_DIR / "Controlled.md"


def cells(line):
    return [x.strip() for x in line.strip().strip("|").split("|")]


def read_first_summary(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("|"):
            rows.append(cells(line))
        elif rows:
            break
    if not rows:
        raise ValueError(f"No table found: {path}")
    header = rows[0]
    if header[0] != "Dataset group":
        return None
    if (len(rows) != 3 or rows[2][0] not in ("Avg.", path.relative_to(ROOT).parts[0])
            or any(len(row) != len(header) for row in rows)):
        raise ValueError(f"Invalid summary table: {path}")
    if len(set(header[1:])) != len(header) - 1:
        raise ValueError(f"Duplicate dataset groups: {path}")
    return dict(zip(header[1:], rows[2][1:]))


def read_group_averages(path):
    groups = {}
    for block in path.read_text(encoding="utf-8").split("\n\n"):
        rows = [cells(line) for line in block.splitlines() if line.strip().startswith("|")]
        if not rows or rows[0][0] == "Dataset group":
            continue
        for row in rows[2:]:
            if row[0] in ("Avg.", path.relative_to(ROOT).parts[0]):
                if len(row) != len(rows[0]):
                    raise ValueError(f"Invalid average row: {path}, {rows[0][0]}")
                groups[rows[0][0]] = dict(zip(rows[0][1:], row[1:]))
    return groups


def markdown_table(columns, entries):
    lines = [
        "| Method | " + " | ".join(columns) + " |",
        "| --- | " + " | ".join(["---"] * len(columns)) + " |",
    ]
    lines.extend(
        f"| {name} | " + " | ".join(scores.get(column, "") for column in columns) + " |"
        for name, scores in entries
    )
    return "\n".join(lines) + "\n"


def main():
    merged = []
    groups = []
    details = {}
    paths = sorted(ROOT.glob("*/results/*/results.md"))
    if MERGE_METHODS is not None:
        by_name = {f"{path.parents[2].name}-{path.parent.name}": path for path in paths}
        missing = [name for name in MERGE_METHODS if name not in by_name]
        if missing:
            raise ValueError(f"Results not found: {', '.join(missing)}")
        paths = [by_name[name] for name in MERGE_METHODS]
    for path in paths:
        scores = read_first_summary(path)
        if scores is None:
            if MERGE_METHODS is not None:
                raise ValueError(f"Selected result has no group summary: {path}")
            print(f"Skipped (first table is not a group summary): {path.relative_to(ROOT)}")
            continue
        name = f"{path.parents[2].name}-{path.parent.name}"
        averages = read_group_averages(path)
        for group, score in scores.items():
            if group != "Avg.":
                details.setdefault(group, []).append((name, averages.get(group, {"Avg.": score})))
        scores = {group: score for group, score in scores.items()
                  if not group.casefold().startswith(("in-the-wild", "common perturbations"))
                  and group != "Avg."}
        merged.append((name, scores))
        groups.extend(group for group in scores if group not in groups)
    if not merged:
        raise ValueError(f"No summary tables found under {ROOT}")

    groups.append("Avg.")
    for _, scores in merged:
        values = [Decimal(score) for score in scores.values() if score not in ("", "-")]
        scores["Avg."] = str((sum(values) / len(values)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)) if values else "-"

    if DETAILS_DIR.exists():
        shutil.rmtree(DETAILS_DIR)
    DETAILS_DIR.mkdir()
    OUT.write_text(markdown_table(groups, merged), encoding="utf-8")
    for group, entries in details.items():
        columns = []
        for _, scores in entries:
            columns.extend(column for column in scores if column != "Avg." and column not in columns)
        columns.append("Avg.")
        (DETAILS_DIR / f"{group}.md").write_text(markdown_table(columns, entries), encoding="utf-8")
    print(f"Merged {len(merged)}/{len(paths)} files into {OUT}")
    print(f"Wrote {len(details)} dataset group tables to {DETAILS_DIR}")


if __name__ == "__main__":
    main()
