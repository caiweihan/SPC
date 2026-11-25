"""Evaluate a saved probe and recreate its result tables and figures."""

import argparse
from decimal import Decimal, ROUND_HALF_UP
import os
from pathlib import Path

import h5py
import torch
from torch import nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
EPS = 1e-12
COLOR_REAL = "#1F77B4"
COLOR_FAKE = "#D62728"
WILD_GROUP = "In-the-wild"
WILD_ORDER = ("Chameleon", "SynthWildX", "WildRF", "SocialRF", "CommunityAI")
CONTROLLED_ORDER = ("GenImage", "UnivFakeDetect", "AIGI-Now", "AIGI-Holmes")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 13,
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.linewidth": 0.6,
    "mathtext.fontset": "stix",
})


def select_device(gpu):
    if gpu is not None:
        if gpu < 0:
            raise ValueError("--gpu 必须是非负整数")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        if not torch.cuda.is_available():
            raise RuntimeError(f"指定的 GPU {gpu} 不可用，请检查 GPU 编号和 CUDA 环境")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    if device.type == "cuda":
        print(f"GPU         : CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}; {torch.cuda.get_device_name(0)}")
    return device


def build_head(head, feat_dim, hidden_dim=None):
    """Construct the same architecture for training and checkpoint loading."""
    if head == "linear2":
        return nn.Linear(feat_dim, 2)
    if head == "linear1":
        return nn.Linear(feat_dim, 1)
    if head == "mlp":
        return nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))
    raise ValueError(f"Unknown probe head: {head}")


def collect_test_groups(feature_dir):
    groups = {}
    for folder in sorted(feature_dir.iterdir()):
        if folder.is_dir():
            paths = sorted(folder.glob("*.h5"))
            if paths:
                groups[folder.name] = paths
    return groups


def load_h5_group(h5_path, enable_spc):
    with h5py.File(h5_path, "r") as f:
        if "fake" not in f and "real" not in f:
            raise KeyError(f"{h5_path} 缺少 'fake' 和 'real' 键，现有键: {list(f.keys())}")
        feat_dim = f["fake" if "fake" in f else "real"].shape[1]
        fake = f["fake"][:].astype(np.float32) if "fake" in f else np.empty((0, feat_dim), dtype=np.float32)
        real = f["real"][:].astype(np.float32) if "real" in f else np.empty((0, feat_dim), dtype=np.float32)
    if enable_spc:
        fake = F.normalize(torch.from_numpy(fake), dim=-1).numpy()
        real = F.normalize(torch.from_numpy(real), dim=-1).numpy()
    return fake, real


def wild_dataset_name(name):
    for family in ("SynthWildX", "WildRF"):
        if name == family or name.startswith(family + "-"):
            return family
    return name


def format_score(score):
    if np.isnan(score):
        return "-"
    return str(Decimal(str(score)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def print_markdown_table(method, folder_name, scores, average):
    dataset_names = [p.stem for p in scores]
    header = f"| {folder_name} | " + " | ".join(dataset_names + ["Avg."]) + " |"
    sep = "| --- | " + " | ".join(["---"] * (len(scores) + 1)) + " |"
    avg_row = f"| {method} | " + " | ".join([format_score(score) for score in scores.values()] + [format_score(average)]) + " |"

    table_md = header + "\n" + sep + "\n" + avg_row
    print("\n\n\n" + table_md)
    return table_md


def signed_distances(logits, layer):
    """Positive towards fake, in input space (linear) or final hidden space (MLP)."""
    if layer.out_features == 1:
        score, normal = logits[:, 0], layer.weight[0]
    else:
        score = logits[:, 0] - logits[:, 1]
        normal = layer.weight[0] - layer.weight[1]
    return score / normal.norm().clamp_min(EPS)


def balanced_accuracy(counts):
    per_class = [100.0 * correct / total for correct, total in counts.values() if total]
    return float(np.mean(per_class)) if per_class else float("nan")


def evaluate_split(model, path, device, enable_spc, threshold, collect_distances):
    fake, real = load_h5_group(path, enable_spc)
    counts, distances = {}, {}
    layer = model[-1] if isinstance(model, nn.Sequential) else model
    baseline = isinstance(model, nn.Linear) and model.out_features == 2
    for label, features in (("fake", fake), ("real", real)):
        logits = model(torch.from_numpy(features).to(device))
        if collect_distances or not baseline:
            distance = signed_distances(logits, layer)
            if collect_distances:
                distances[label] = distance.cpu().numpy()
        if baseline:
            # The original two-output probe classifies by comparing logits directly.
            predicts_fake = logits[:, 0] > logits[:, 1]
            correct = predicts_fake if label == "fake" else ~predicts_fake
        else:
            correct = distance > threshold if label == "fake" else distance <= threshold
        counts[label] = (correct.sum().item(), len(features))
    return counts, distances


@torch.no_grad()
def evaluate(model, groups, device, method, enable_spc, threshold, *, collect_distances=False):
    """Read each split once. Only the final report needs the per-sample distances."""
    model.eval()
    curves, folder_avgs, distributions = {}, {}, {}
    tables = []
    for group, paths in groups.items():
        scores, wild_totals, datasets = {}, {}, {}
        for path in paths:
            counts, distances = evaluate_split(model, path, device, enable_spc, threshold, collect_distances)
            scores[path] = balanced_accuracy(counts)
            if collect_distances:
                datasets[path.stem] = distances
            if group == WILD_GROUP:
                # Group subsets by sample count, separately for fake and real.
                totals = wild_totals.setdefault(wild_dataset_name(path.stem), {"fake": [0.0, 0], "real": [0.0, 0]})
                for label, (correct, total) in counts.items():
                    if total:
                        accuracy = 100.0 * correct / total
                        totals[label][0] += accuracy * total
                        totals[label][1] += total

        folder_avgs[group] = float(np.mean(list(scores.values())))
        tables.append(print_markdown_table(method, group, scores, folder_avgs[group]))
        curves[group] = (
            {name: float(np.mean([weighted / n for weighted, n in totals.values() if n]))
             for name, totals in wild_totals.items()}
            if group == WILD_GROUP else {"Avg.": folder_avgs[group]}
        )
        if collect_distances:
            distributions[group] = datasets if group == WILD_GROUP else {"All": pool_distributions(datasets)}

    summary = "\n".join([
        "| Dataset group | " + " | ".join(folder_avgs) + " |",
        "| --- | " + " | ".join(["---"] * len(folder_avgs)) + " |",
        f"| {method} | " + " | ".join(format_score(avg) for avg in folder_avgs.values()) + " |",
    ])
    return curves, "\n\n".join([summary] + tables), distributions


def write_report(directory, markdown, history, distances, method, threshold, save_pdf, *, extra_row=None):
    (directory / "results.md").write_text(markdown + "\n", encoding="utf-8")
    plot_training_summary(history, distances, directory / "training_summary.png",
                          decision_threshold=threshold, save_pdf=save_pdf, method=method, extra_row=extra_row)
    print(f"Saved results.md and training_summary.png{' / .pdf' if save_pdf else ''} in {directory}")


def _ordered_items(data, order):
    names = [name for name in order if name in data]
    names.extend(sorted(set(data) - set(names)))
    return [(name, data[name]) for name in names]


def _merge_wild_distributions(datasets):
    groups = {}
    for name, data in datasets.items():
        parts = groups.setdefault(wild_dataset_name(name), {"fake": [], "real": []})
        for label in parts:
            parts[label].append(data[label])
    return {name: {label: np.concatenate(values) for label, values in parts.items()}
            for name, parts in groups.items()}


def pool_distributions(datasets):
    return {label: np.concatenate([data[label] for data in datasets.values()])
            if datasets else np.array([], dtype=np.float32)
            for label in ("fake", "real")}


def _compute_dprime(fake, real):
    fake, real = np.asarray(fake, dtype=np.float64), np.asarray(real, dtype=np.float64)
    if not len(fake) or not len(real):
        return None
    variance = (fake.var() + real.var()) / 2.0
    return float((fake.mean() - real.mean()) / np.sqrt(variance)) if variance > EPS else None


def _compute_accuracy_at_threshold(fake, real, decision_threshold=0.0):
    fake, real = np.asarray(fake), np.asarray(real)
    if not len(fake) or not len(real):
        return None
    return 50.0 * ((fake > decision_threshold).mean() + (real <= decision_threshold).mean())


def plot_histogram(ax, title, data, *, lines, title_size=20):
    fake, real = data["fake"], data["real"]
    if len(fake) or len(real):
        edges = np.histogram_bin_edges(np.concatenate([fake, real]), bins=50)
        for label, values, color in [("Fake", fake, COLOR_FAKE), ("Real", real, COLOR_REAL)]:
            if len(values):
                ax.hist(values, bins=edges, density=True, alpha=0.5, color=color, label=label)
        ax.legend(loc="upper right", frameon=False, fontsize=13, handlelength=1.0,
                  labelspacing=0.25, handletextpad=0.4, borderaxespad=0.15)
        # Keep the histogram peaks below the annotations and legend.
        ax.set_ylim(0, ax.get_ylim()[1] * 1.5)
    for position in lines:
        ax.axvline(position, color="black", linestyle="--", linewidth=0.8)
    ax.margins(x=0.01)
    ax.set_title(title, fontsize=title_size)
    ax.set_xlabel("Signed distance to trained plane", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    ax.tick_params(labelsize=13)
    ax.grid(True, linestyle="--", alpha=0.3)


def annotate_distances(ax, data, scores):
    dprime = _compute_dprime(data["fake"], data["real"])
    annotations = [f"d' = {dprime:.3f}"] if dprime is not None else []
    annotations.extend(f"Accuracy (τ = {threshold:.3f}) = {score:.1f}%"
                       for threshold, score in scores.items())
    if annotations:
        ax.text(0.02, 0.98, "\n".join(annotations), transform=ax.transAxes, fontsize=13,
                va="top", bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8))


def _plot_distance_panel(ax, title, data, decision_threshold):
    plot_histogram(ax, title, data, lines=(decision_threshold,))
    accuracy = _compute_accuracy_at_threshold(data["fake"], data["real"], decision_threshold)
    annotate_distances(ax, data, {decision_threshold: accuracy} if accuracy is not None else {})


def _plot_curves(ax, series, title):
    for name, values in series.items():
        ax.plot(range(1, len(values)), values[1:], linewidth=1.8, label=name)
    ax.set_title(title, fontsize=24, fontweight="bold")
    ax.set(xlabel="Epoch", ylabel="Balanced Accuracy (%)")
    ax.grid(True, linestyle="--", alpha=0.3)
    if series:
        ax.legend(frameon=False, fontsize=12, ncol=2)


def plot_training_summary(history, dist_data, save_path, decision_threshold=0.0, save_pdf=False, method="", *, extra_row=None):
    """Plot epoch curves and dataset distributions, with an optional final row."""
    wild = _ordered_items(_merge_wild_distributions(dist_data.get(WILD_GROUP, {})), WILD_ORDER)
    controlled, perturbations = {}, {}
    for group, datasets in dist_data.items():
        if group == WILD_GROUP:
            continue
        target = perturbations if group.startswith("Common Perturbations") else controlled
        target[group] = datasets["All"]
    rows = [(wild, "In-the-wild: distance distributions"),
            (_ordered_items(controlled, CONTROLLED_ORDER), "Controlled: distance distributions")]
    if perturbations:
        rows.append((_ordered_items(perturbations, ()), "Common Perturbations: distance distributions"))
    width = max(14, 3.2 * max(len(panels) for panels, _ in rows))
    n_rows = len(rows) + int(extra_row is not None)
    fig = plt.figure(figsize=(width, 3 + 4 * n_rows), layout="constrained")
    sections = fig.subfigures(1 + n_rows, 1, height_ratios=[1] + [1.2] * n_rows)
    curve_axes = sections[0].subplots(1, 2)
    wild_curves = dict(_ordered_items(history.get(WILD_GROUP, {}), WILD_ORDER))
    _plot_curves(curve_axes[0], wild_curves, "In-the-wild")
    controlled_curves = {group: series["Avg."] for group, series in history.items() if group != WILD_GROUP}
    averages = {f"{group} Avg.": values for group, values in _ordered_items(controlled_curves, CONTROLLED_ORDER)}
    _plot_curves(curve_axes[1], averages, "Controlled: Avg.")

    for section, (panels, title) in zip(sections[1:], rows):
        section.suptitle(title, fontsize=26, fontweight="bold")
        axes = section.subplots(1, max(len(panels), 1), squeeze=False)[0]
        if not panels:
            axes[0].text(0.5, 0.5, "No datasets", ha="center", va="center", fontsize=14)
            axes[0].set_axis_off()
        for index, (ax, (name, data)) in enumerate(zip(axes, panels)):
            _plot_distance_panel(ax, name, data, decision_threshold)
            if index > 0:
                ax.set_ylabel("")

    if extra_row is not None:
        extra_row(sections[-1], method, dist_data.get(WILD_GROUP, {}), controlled)
    fig.savefig(save_path, format="png", dpi=200)
    if save_pdf:
        fig.savefig(Path(save_path).with_suffix(".pdf"), format="pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="MoEViE", help="outputs 下的方法目录")
    parser.add_argument("--exp_name", default="test", help="results 下包含 probe.pt 的实验目录")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--save_pdf", action="store_true", help="同时输出 PDF")
    args = parser.parse_args()

    retest(args, parser)


def retest(args, parser, *, extra_row=None):
    directory = OUTPUTS / args.method / "results" / args.exp_name
    checkpoint_path = directory / "probe.pt"
    if not checkpoint_path.is_file():
        parser.error(f"缺少 {checkpoint_path}，请先训练并保存模型")
    device = select_device(args.gpu)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config["head"] == "linear2":
        extra_row = None
    model = build_head(config["head"], checkpoint["feat_dim"], config.get("hidden_dim"))
    model.load_state_dict(checkpoint["state_dict"])
    groups = collect_test_groups(OUTPUTS / args.method / "test-features")
    _, markdown, distances = evaluate(model.to(device), groups, device, args.method, config["enable_spc"],
                                      config["decision_threshold"], collect_distances=True)
    write_report(directory, markdown, checkpoint["history"], distances, args.method,
                 config["decision_threshold"], args.save_pdf or config["save_pdf"], extra_row=extra_row)


if __name__ == "__main__":
    main()
