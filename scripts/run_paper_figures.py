#!/usr/bin/env python3
"""Generate publication-oriented figures and tables for the frozen analysis.

The plotting code is intentionally kept separate from discovery. It reads the
frozen outputs, recomputes only transparent display statistics, and writes
multi-panel figures plus the Mantel explainability tables. Mantel bubbles are
an association display and are not a causal graph.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, ttest_ind
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/current_stk11_slc7a2_discovery"
FIGURES = RESULTS / "figures"
SEED = 20260924
MANTEL_PERMUTATIONS = 499

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stk11_slc7a2_discovery import load_inputs, standardized_score
from scripts.run_mam_luad_analysis import MAM_SURVIVAL_GENES

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GRAY = "#6E6E6E"
LIGHT_GRAY = "#D9D9D9"


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "font.weight": "bold",
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.labelweight": "bold",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.title_fontsize": 9.5,
            "axes.linewidth": 0.9,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    for ax in fig.axes:
        for text in ax.texts:
            text.set_fontweight("bold")
            text.set_fontsize(max(text.get_fontsize(), 9.0))
        for text in [ax.title, ax.xaxis.label, ax.yaxis.label]:
            text.set_fontweight("bold")
        ax.title.set_fontsize(max(ax.title.get_fontsize(), 11.5))
        ax.xaxis.label.set_fontsize(max(ax.xaxis.label.get_fontsize(), 10.0))
        ax.yaxis.label.set_fontsize(max(ax.yaxis.label.get_fontsize(), 10.0))
        for tick in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            tick.set_fontweight("bold")
            tick.set_fontsize(max(tick.get_fontsize(), 9.0))
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontweight("bold")
                text.set_fontsize(max(text.get_fontsize(), 9.0))
            legend.get_title().set_fontweight("bold")
            legend.get_title().set_fontsize(max(legend.get_title().get_fontsize(), 10.0))
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", facecolor="white")
    fig.savefig(FIGURES / f"{stem}.png", dpi=600, facecolor="white")
    plt.close(fig)


def load_context() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rna, mutation, _ = load_inputs()
    clinical = pd.read_csv(ROOT / "data/tcga_lung/clinical.csv").set_index("patient_id")
    return rna, mutation, clinical


def primary_split(rna: pd.DataFrame, mutation: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    kras = mutation["KRAS"].to_numpy(int) == 1
    discovery = rna.loc[kras]
    labels = mutation.loc[kras, "STK11"].to_numpy(int)
    genes = discovery.columns.tolist()
    train_idx, holdout_idx = train_test_split(
        np.arange(len(labels)), test_size=0.30, random_state=SEED, stratify=labels
    )
    position = genes.index("SLC7A2")
    score = standardized_score(
        discovery.to_numpy(float)[train_idx, position],
        discovery.to_numpy(float)[holdout_idx, position],
        1.0,
    )
    return train_idx, holdout_idx, labels, score, position


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


def wilson_count_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half_width = z * np.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    return total * (center - half_width), total * (center + half_width)


def make_fig1(rna: pd.DataFrame, mutation: pd.DataFrame, clinical: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(11.2, 6.7), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.5, 1, 1], height_ratios=[1, 0.86])
    ax_flow = fig.add_subplot(grid[:, 0])
    panel_label(ax_flow, "A")
    ax_flow.axis("off")
    flow = [
        (0.52, 0.87, "TCGA-LUAD intersection", "n = 494"),
        (0.52, 0.62, "KRAS-mutant discovery", "n = 137"),
        (0.25, 0.31, "STK11-mutant", "n = 31"),
        (0.79, 0.31, "STK11-non-mutant", "n = 106"),
        (0.52, 0.07, "KRAS-wild-type transfer", "n = 357"),
    ]
    for x, y, title, count in flow:
        ax_flow.text(x, y, f"{title}\n{count}", ha="center", va="center", fontsize=9)
    arrows = [((0.52, 0.81), (0.52, 0.68)), ((0.44, 0.57), (0.29, 0.38)),
              ((0.60, 0.57), (0.75, 0.38)), ((0.52, 0.56), (0.52, 0.14))]
    for start, end in arrows:
        ax_flow.add_patch(FancyArrowPatch(start, end, transform=ax_flow.transAxes,
                                          arrowstyle="-|>", mutation_scale=10, lw=0.8, color=GRAY))
    ax_flow.text(0.52, 0.97, "Frozen cohort and label flow", ha="center", fontsize=10, fontweight="bold")

    ax_counts = fig.add_subplot(grid[0, 1])
    panel_label(ax_counts, "B")
    labels = ["KRAS+\nwithin LUAD", "STK11+\nwithin KRAS+", "STK11-\nwithin KRAS+"]
    values = np.array([137, 31, 106])
    colors = [BLUE, ORANGE, GRAY]
    denominators = [494, 137, 137]
    ax_counts.axhline(0, color="black", lw=0.6)
    for i, (value, color, denominator) in enumerate(zip(values, colors, denominators)):
        low, high = wilson_count_interval(int(value), denominator)
        ax_counts.errorbar(i, value, yerr=[[value - low], [high - value]], fmt="o", ms=6,
                           color=color, ecolor=color, elinewidth=1.4, capsize=2.5)
        ax_counts.text(i, value + 6, str(value), ha="center", fontsize=8)
    ax_counts.set_xticks(range(3), labels)
    ax_counts.set_ylabel("Patients")
    ax_counts.set_ylim(0, 160)
    ax_counts.spines[["top", "right"]].set_visible(False)

    ax_split = fig.add_subplot(grid[0, 2])
    panel_label(ax_split, "C")
    _, holdout, labels, _, _ = primary_split(rna, mutation)
    is_train = np.ones(len(labels), dtype=bool)
    is_train[holdout] = False
    y_jitter = np.where(labels == 1, 1.0, 0.0) + np.linspace(-0.08, 0.08, len(labels))
    ax_split.scatter(np.where(is_train, 0, 1), y_jitter, c=np.where(labels == 1, ORANGE, BLUE),
                     s=13, alpha=0.72, linewidth=0)
    ax_split.axvline(0.5, color=GRAY, lw=0.8)
    ax_split.set_xticks([0, 1], ["Discovery\n95", "Holdout\n42"])
    ax_split.set_yticks([0, 1], ["STK11-", "STK11+"])
    ax_split.set_xlim(-0.35, 1.35)
    ax_split.set_ylim(-0.35, 1.35)
    ax_split.set_title("Primary stratified split", fontsize=9)
    ax_split.spines[["top", "right"]].set_visible(False)

    ax_lock = fig.add_subplot(grid[1, 1:])
    panel_label(ax_lock, "D")
    ax_lock.axis("off")
    lock_text = (
        "LOCKED ANALYSIS\n"
        "Seed 20260924   |   70/30 patient split   |   19,938 genes tested\n"
        "Genome-wide Welch test + Benjamini-Hochberg correction\n"
        "Candidate, direction, and training scale frozen before holdout scoring"
    )
    ax_lock.text(0.02, 0.5, lock_text, va="center", ha="left", linespacing=1.6,
                 fontsize=9, family="DejaVu Sans Mono")
    save_figure(fig, "fig2_cohort_lock")

    pd.DataFrame(
        {
            "cohort": ["TCGA-LUAD intersection", "KRAS-mutant discovery", "KRAS-wild-type transfer"],
            "n": [494, 137, 357],
            "stk11_positive": [63, 31, 32],
            "stk11_negative": [431, 106, 325],
        }
    ).to_csv(FIGURES / "table_s1_cohort_audit.csv", index=False)


def make_fig2() -> None:
    top = pd.read_csv(RESULTS / "primary_discovery_top100.csv")
    freq = pd.read_csv(RESULTS / "candidate_selection_frequency.csv")
    top = top.sort_values(["q_value", "gene"]).reset_index(drop=True)
    shown = top.head(20).iloc[::-1]
    fig = plt.figure(figsize=(11.2, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], height_ratios=[1.35, 1])
    ax_rank = fig.add_subplot(grid[0, 0])
    panel_label(ax_rank, "A")
    y = np.arange(len(shown))
    colors = [ORANGE if d > 0 else BLUE for d in shown["delta"]]
    ax_rank.hlines(y, 0, shown["delta"], color=colors, lw=1.6, alpha=0.65)
    ax_rank.scatter(shown["delta"], y, color=colors, s=24, zorder=3)
    ax_rank.axvline(0, color=GRAY, lw=0.7)
    ax_rank.set_yticks(y, shown["gene"])
    ax_rank.set_xlabel("Training mean difference (STK11+ minus STK11-)")
    ax_rank.set_title("Genome-wide discovery ranking", fontsize=9)
    ax_rank.spines[["top", "right"]].set_visible(False)
    for yi, row in enumerate(shown.itertuples()):
        if row.gene == "SLC7A2":
            ax_rank.annotate("frozen candidate", (row.delta, yi), xytext=(8, 0), textcoords="offset points",
                             fontsize=7.5, color=ORANGE, va="center")

    ax_q = fig.add_subplot(grid[0, 1])
    panel_label(ax_q, "B")
    q = np.clip(top["q_value"].to_numpy(float), 1e-16, 1)
    ax_q.scatter(top["delta"], -np.log10(q), s=12, c=np.where(top["eligible"], ORANGE, LIGHT_GRAY), alpha=0.75)
    ax_q.axhline(-np.log10(0.05), color=GRAY, ls="--", lw=0.8)
    candidate = top.loc[top["gene"].eq("SLC7A2")].iloc[0]
    ax_q.scatter([candidate.delta], [-np.log10(candidate.q_value)], color=ORANGE, s=34, zorder=4)
    ax_q.annotate("SLC7A2", (candidate.delta, -np.log10(candidate.q_value)), xytext=(5, 4),
                  textcoords="offset points", fontsize=7.5)
    ax_q.set_xlabel("Training effect")
    ax_q.set_ylabel("-log10(BH q)")
    ax_q.spines[["top", "right"]].set_visible(False)

    ax_freq = fig.add_subplot(grid[1, 0])
    panel_label(ax_freq, "C")
    freq = freq.sort_values("selection_frequency")
    y = np.arange(len(freq))
    colors = [ORANGE if g == "SLC7A2" else BLUE for g in freq["gene"]]
    ax_freq.hlines(y, 0, freq["selection_frequency"], color=colors, lw=2, alpha=0.6)
    ax_freq.scatter(freq["selection_frequency"], y, color=colors, s=32, zorder=3)
    ax_freq.set_yticks(y, freq["gene"])
    ax_freq.set_xlabel("Selection frequency across 20 splits")
    ax_freq.set_xlim(0, 0.75)
    ax_freq.spines[["top", "right"]].set_visible(False)
    for yi, row in enumerate(freq.itertuples()):
        ax_freq.text(row.selection_frequency + 0.025, yi, f"{row.selection_count}/20", va="center", fontsize=7.5)

    ax_rule = fig.add_subplot(grid[1, 1])
    panel_label(ax_rule, "D")
    ax_rule.axis("off")
    ax_rule.text(0.04, 0.86, "Deterministic freeze rule", fontsize=10, fontweight="bold")
    steps = [("1", "smallest BH q-value"), ("2", "largest absolute training effect"), ("3", "alphabetical gene name")]
    for y, (num, text) in zip([0.62, 0.40, 0.18], steps):
        ax_rule.text(0.08, y, num, fontsize=10, fontweight="bold", color=ORANGE)
        ax_rule.text(0.18, y, text, va="center", fontsize=8.5)
    ax_rule.text(0.08, 0.02, "=> SLC7A2 frozen before holdout scoring", color=ORANGE, fontsize=8.5)
    save_figure(fig, "fig3_genomewide_discovery")
    top.to_csv(FIGURES / "table_s2_primary_discovery_top100.csv", index=False)


def bootstrap_roc(y: np.ndarray, score: np.ndarray, seed: int, n_boot: int = 1000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, 101)
    curves = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y[idx], score[idx])
        curves.append(np.interp(grid, fpr, tpr))
    curves = np.asarray(curves)
    return grid, np.quantile(curves, 0.025, axis=0), np.quantile(curves, 0.975, axis=0)


def bootstrap_auc_interval(y: np.ndarray, score: np.ndarray, seed: int, n_boot: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        values.append(float(roc_auc_score(y[idx], score[idx])))
    return tuple(float(x) for x in np.quantile(values, [0.025, 0.975]))


def violin_with_points(ax: plt.Axes, groups: list[np.ndarray], labels: list[str], colors: list[str]) -> None:
    parts = ax.violinplot(groups, positions=np.arange(1, len(groups) + 1), widths=0.72,
                          showmeans=False, showmedians=True, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.35)
    parts["cmedians"].set_color("black")
    rng = np.random.default_rng(4)
    for i, (values, color) in enumerate(zip(groups, colors), start=1):
        jitter = rng.uniform(-0.11, 0.11, len(values))
        ax.scatter(np.full(len(values), i) + jitter, values, s=14, color=color, alpha=0.68, edgecolors="white", linewidths=0.25, zorder=3)
    ax.set_xticks(np.arange(1, len(groups) + 1), labels)


def make_fig3(rna: pd.DataFrame, mutation: pd.DataFrame) -> None:
    nested = pd.read_csv(RESULTS / "nested_holdout_predictions.csv")
    holdout = nested.loc[nested["repeat"].eq(0)].copy()
    y = holdout["stk11"].to_numpy(int)
    score = holdout["score"].to_numpy(float)
    random_null = pd.read_csv(RESULTS / "matched_random_single_gene_null.csv")
    controls = pd.read_csv(RESULTS / "arginine_transporter_controls.csv")
    auc_value = roc_auc_score(y, score)
    fig = plt.figure(figsize=(11.2, 7.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1.1, 1])
    ax_dist = fig.add_subplot(grid[0, 0])
    panel_label(ax_dist, "A")
    violin_with_points(ax_dist, [score[y == 0], score[y == 1]], ["STK11-", "STK11+"], [BLUE, ORANGE])
    ax_dist.set_ylabel("Frozen SLC7A2 standardized score")
    ax_dist.set_title(f"Primary holdout (AUC = {auc_value:.3f})", fontsize=9)
    ax_dist.spines[["top", "right"]].set_visible(False)

    ax_roc = fig.add_subplot(grid[0, 1])
    panel_label(ax_roc, "B")
    fpr, tpr, _ = roc_curve(y, score)
    grid_x, low, high = bootstrap_roc(y, score, SEED + 11)
    ax_roc.fill_between(grid_x, low, high, color=ORANGE, alpha=0.18, linewidth=0)
    ax_roc.plot(fpr, tpr, color=ORANGE, lw=2, label=f"SLC7A2 AUC {auc_value:.3f}")
    ax_roc.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=0.8)
    ax_roc.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1))
    ax_roc.legend(frameon=False, loc="lower right")
    ax_roc.spines[["top", "right"]].set_visible(False)

    ax_null = fig.add_subplot(grid[1, 0])
    panel_label(ax_null, "C")
    values = np.sort(random_null["holdout_auc"].to_numpy(float))
    ecdf = np.arange(1, len(values) + 1) / len(values)
    ax_null.step(values, ecdf, where="post", color=BLUE, lw=1.7)
    ax_null.axvline(auc_value, color=ORANGE, lw=1.7, label="SLC7A2")
    ax_null.set(xlabel="Holdout AUC", ylabel="Empirical cumulative probability")
    ax_null.legend(frameon=False, loc="lower right")
    ax_null.spines[["top", "right"]].set_visible(False)

    ax_ctrl = fig.add_subplot(grid[1, 1])
    panel_label(ax_ctrl, "D")
    controls = controls.sort_values("gene")
    y_pos = np.arange(len(controls))
    colors = [ORANGE if g == "SLC7A2" else BLUE for g in controls["gene"]]
    ax_ctrl.hlines(y_pos, 0.35, controls["holdout_auc"], color=colors, lw=2, alpha=0.6)
    ax_ctrl.scatter(controls["holdout_auc"], y_pos, color=colors, s=34, zorder=3)
    ax_ctrl.axvline(0.5, color=GRAY, ls="--", lw=0.8)
    ax_ctrl.set_yticks(y_pos, controls["gene"])
    ax_ctrl.set(xlabel="Holdout AUC", xlim=(0.3, 0.95))
    ax_ctrl.spines[["top", "right"]].set_visible(False)
    save_figure(fig, "fig4_primary_holdout")

    pd.DataFrame(
        [{"metric": "holdout_auc", "value": auc_value},
         {"metric": "holdout_welch_p", "value": 2.5075055455727985e-05},
         {"metric": "holdout_mannwhitney_p", "value": 2.6521072831509816e-04},
         {"metric": "bootstrap_ci_low", "value": 0.76875},
         {"metric": "bootstrap_ci_high", "value": 0.975}]
    ).to_csv(FIGURES / "table_s3_primary_holdout.csv", index=False)


def make_fig4() -> None:
    nested = pd.read_csv(RESULTS / "nested_holdout_summary.csv")
    fixed = pd.read_csv(RESULTS / "frozen_candidate_repeated_sensitivity.csv")
    freq = pd.read_csv(RESULTS / "candidate_selection_frequency.csv")
    null = pd.read_csv(RESULTS / "matched_random_single_gene_null.csv")
    fig = plt.figure(figsize=(11.2, 6.7), constrained_layout=True)
    grid = fig.add_gridspec(1, 4, width_ratios=[1.45, 0.9, 1.05, 1.1])
    ax_nested = fig.add_subplot(grid[0, 0])
    panel_label(ax_nested, "A")
    parts = ax_nested.violinplot([nested["holdout_auc"].to_numpy()], positions=[1], widths=0.65,
                                 showmeans=False, showmedians=True, showextrema=False)
    parts["bodies"][0].set_facecolor(BLUE); parts["bodies"][0].set_alpha(0.30); parts["bodies"][0].set_edgecolor(BLUE)
    ax_nested.scatter(np.ones(len(nested)), nested["holdout_auc"], c=[ORANGE if c == "SLC7A2" else BLUE for c in nested["candidate"]], s=24, alpha=0.8, edgecolors="white", linewidths=0.25)
    ax_nested.set_xticks([1], ["Nested\n20 splits"])
    ax_nested.set_ylabel("Holdout AUC")
    ax_nested.spines[["top", "right"]].set_visible(False)

    ax_freq = fig.add_subplot(grid[0, 1])
    panel_label(ax_freq, "B")
    freq = freq.sort_values("selection_frequency")
    yp = np.arange(len(freq))
    ax_freq.hlines(yp, 0, freq["selection_frequency"], color=BLUE, lw=2)
    ax_freq.scatter(freq["selection_frequency"], yp, color=[ORANGE if x == "SLC7A2" else BLUE for x in freq["gene"]], s=28, zorder=3)
    ax_freq.set_yticks(yp, freq["gene"])
    ax_freq.set_xlabel("Selection frequency")
    ax_freq.set_xlim(0, 0.75)
    ax_freq.spines[["top", "right"]].set_visible(False)

    ax_fixed = fig.add_subplot(grid[0, 2])
    panel_label(ax_fixed, "C")
    parts = ax_fixed.violinplot([fixed["holdout_auc"].to_numpy()], positions=[1], widths=0.65,
                                showmeans=False, showmedians=True, showextrema=False)
    parts["bodies"][0].set_facecolor(ORANGE); parts["bodies"][0].set_alpha(0.30); parts["bodies"][0].set_edgecolor(ORANGE)
    ax_fixed.scatter(np.ones(len(fixed)), fixed["holdout_auc"], color=ORANGE, s=22, alpha=0.75, edgecolors="white", linewidths=0.25)
    ax_fixed.set_xticks([1], ["Frozen\nSLC7A2"])
    ax_fixed.set_ylim(0.45, 1.0)
    ax_fixed.set_ylabel("Holdout AUC")
    ax_fixed.spines[["top", "right"]].set_visible(False)

    ax_spread = fig.add_subplot(grid[0, 3])
    panel_label(ax_spread, "D")
    groups = [null["holdout_auc"].to_numpy(), nested["holdout_auc"].to_numpy(), fixed["holdout_auc"].to_numpy()]
    violin_with_points(ax_spread, groups, ["Random\ngenes", "Nested\ntop", "Fixed\nSLC7A2"], [GRAY, BLUE, ORANGE])
    ax_spread.set_ylabel("AUC distribution")
    ax_spread.set_ylim(0.25, 1.0)
    ax_spread.spines[["top", "right"]].set_visible(False)
    save_figure(fig, "fig5_stability_null")
    nested.to_csv(FIGURES / "table_s4_nested_sensitivity.csv", index=False)


def make_fig5(rna: pd.DataFrame, mutation: pd.DataFrame) -> None:
    train_idx, holdout_idx, labels, score, position = primary_split(rna, mutation)
    kras = mutation["KRAS"].to_numpy(int) == 1
    discovery = rna.loc[kras]
    transfer_y = mutation.loc[~kras, "STK11"].to_numpy(int)
    transfer_score = standardized_score(discovery.to_numpy(float)[train_idx, position], rna.loc[~kras].to_numpy(float)[:, position], 1.0)
    holdout_y = labels[holdout_idx]
    holdout_score = score
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.9), constrained_layout=True)
    ax_roc, ax_dist, ax_ci = axes
    panel_label(ax_roc, "A")
    fpr1, tpr1, _ = roc_curve(holdout_y, holdout_score)
    fpr2, tpr2, _ = roc_curve(transfer_y, transfer_score)
    ax_roc.plot(fpr1, tpr1, color=ORANGE, lw=2, label=f"KRAS+ holdout AUC {roc_auc_score(holdout_y, holdout_score):.3f}")
    ax_roc.plot(fpr2, tpr2, color=BLUE, lw=2, label=f"KRAS- transfer AUC {roc_auc_score(transfer_y, transfer_score):.3f}")
    ax_roc.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=0.8)
    ax_roc.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1))
    ax_roc.legend(frameon=False, loc="lower right")
    ax_roc.spines[["top", "right"]].set_visible(False)

    panel_label(ax_dist, "B")
    violin_with_points(ax_dist, [holdout_score[holdout_y == 0], holdout_score[holdout_y == 1], transfer_score[transfer_y == 0], transfer_score[transfer_y == 1]],
                       ["KRAS+\nSTK11-", "KRAS+\nSTK11+", "KRAS-\nSTK11-", "KRAS-\nSTK11+"], [BLUE, ORANGE, BLUE, ORANGE])
    ax_dist.set_ylabel("Frozen SLC7A2 score")
    ax_dist.spines[["top", "right"]].set_visible(False)

    panel_label(ax_ci, "C")
    aucs = np.array([roc_auc_score(holdout_y, holdout_score), roc_auc_score(transfer_y, transfer_score)])
    cis = np.array([
        bootstrap_auc_interval(holdout_y, holdout_score, SEED + 21),
        bootstrap_auc_interval(transfer_y, transfer_score, SEED + 22),
    ])
    yp = np.arange(2)
    ax_ci.hlines(yp, cis[:, 0], cis[:, 1], color=[ORANGE, BLUE], lw=2)
    ax_ci.scatter(aucs, yp, color=[ORANGE, BLUE], s=38, zorder=3)
    ax_ci.axvline(0.5, color=GRAY, ls="--", lw=0.8)
    ax_ci.set_yticks(yp, ["KRAS+ holdout", "KRAS- transfer"])
    ax_ci.set_xlabel("AUC with interval")
    ax_ci.set_xlim(0.4, 1.0)
    ax_ci.spines[["top", "right"]].set_visible(False)
    save_figure(fig, "fig6_context_transfer")

    controls = pd.read_csv(RESULTS / "arginine_transporter_controls.csv")
    pd.DataFrame(
        [{"context": "KRAS+ holdout", "auc": aucs[0], "ci_low": cis[0, 0], "ci_high": cis[0, 1]},
         {"context": "KRAS- transfer", "auc": aucs[1], "ci_low": cis[1, 0], "ci_high": cis[1, 1]}]
    ).to_csv(FIGURES / "table_s5_context_transfer.csv", index=False)
    controls.to_csv(FIGURES / "table_s5_transporter_controls.csv", index=False)


def zscore_columns(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    mean = np.nanmean(values, axis=0)
    scale = np.nanstd(values, axis=0, ddof=1)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    values = np.where(np.isfinite(values), values, mean)
    return (values - mean) / scale


def encode_stage(series: pd.Series) -> np.ndarray:
    values = []
    for raw in series.astype(object):
        text = "" if pd.isna(raw) else str(raw)
        if "IV" in text:
            values.append(4.0)
        elif "III" in text:
            values.append(3.0)
        elif "II" in text:
            values.append(2.0)
        elif "I" in text:
            values.append(1.0)
        else:
            values.append(np.nan)
    return np.asarray(values, dtype=float)


def encode_smoking(series: pd.Series) -> np.ndarray:
    values = []
    for raw in series.astype(object):
        text = "" if pd.isna(raw) else str(raw)
        if "Lifelong" in text:
            values.append(0.0)
        elif "> 15" in text:
            values.append(1.0)
        elif "< or = 15" in text:
            values.append(2.0)
        elif "Current Smoker" in text:
            values.append(3.0)
        else:
            values.append(np.nan)
    return np.asarray(values, dtype=float)


def fill_missing(values: np.ndarray) -> tuple[np.ndarray, int]:
    values = np.asarray(values, dtype=float)
    missing = int(np.sum(~np.isfinite(values)))
    if missing:
        values = values.copy()
        values[~np.isfinite(values)] = np.nanmedian(values)
    return values, missing


def mantel_pairwise(node_values: dict[str, np.ndarray], seed: int = SEED) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    names = list(node_values)
    distance_vectors: dict[str, np.ndarray] = {}
    distance_matrices: dict[str, np.ndarray] = {}
    for name, values in node_values.items():
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        distance_vectors[name] = pdist(values, metric="euclidean")
        distance_matrices[name] = squareform(distance_vectors[name])
    tri = np.triu_indices(distance_matrices[names[0]].shape[0], 1)
    rows = []
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            observed = float(spearmanr(distance_vectors[name_a], distance_vectors[name_b]).statistic)
            permuted = []
            b_matrix = distance_matrices[name_b]
            for _ in range(MANTEL_PERMUTATIONS):
                perm = rng.permutation(b_matrix.shape[0])
                permuted_matrix = b_matrix[np.ix_(perm, perm)]
                permuted.append(float(spearmanr(distance_vectors[name_a], permuted_matrix[tri]).statistic))
            permuted = np.asarray(permuted)
            p_value = float((1 + np.sum(np.abs(permuted) >= abs(observed))) / (MANTEL_PERMUTATIONS + 1))
            rows.append({"node_a": name_a, "node_b": name_b, "rho": observed, "mantel_p": p_value})
    table = pd.DataFrame(rows)
    table["mantel_q"] = multipletests(table["mantel_p"].to_numpy(), method="fdr_bh")[1]
    table["significant"] = table["mantel_q"] <= 0.05
    return table, distance_matrices


def make_fig6(rna: pd.DataFrame, mutation: pd.DataFrame, clinical: pd.DataFrame) -> None:
    kras = mutation["KRAS"].to_numpy(int) == 1
    ids = rna.index[kras]
    expr = rna.loc[ids]
    labels = mutation.loc[ids, "STK11"].to_numpy(int)
    top = pd.read_csv(RESULTS / "primary_discovery_top100.csv")["gene"].head(20).tolist()
    top = [gene for gene in top if gene in expr.columns]
    competitors = [gene for gene in ["ADGRF1", "S100P", "ITGB8"] if gene in expr.columns]
    arg_genes = [gene for gene in ["SLC7A1", "SLC7A2", "SLC7A3", "SLC7A4"] if gene in expr.columns]
    mam_genes = [gene for gene in MAM_SURVIVAL_GENES if gene in expr.columns]
    clinical = clinical.reindex(ids)
    stage, stage_missing = fill_missing(encode_stage(clinical["stage"]))
    age, age_missing = fill_missing(clinical["age"].to_numpy(float))
    smoking, smoking_missing = fill_missing(encode_smoking(clinical["smoking_status"]))
    node_values = {
        "SLC7A2": zscore_columns(expr[["SLC7A2"]].to_numpy(float)),
        "MAM-module": zscore_columns(expr[mam_genes].to_numpy(float)),
        "ARG-transporter": zscore_columns(expr[arg_genes].to_numpy(float)),
        "Top20-discovery": zscore_columns(expr[top].to_numpy(float)),
        "Competitor": zscore_columns(expr[competitors].to_numpy(float)),
        "STK11": labels.astype(float),
        "Stage": zscore_columns(stage),
        "Age": zscore_columns(age),
        "Smoking": zscore_columns(smoking),
    }
    mantel, distances = mantel_pairwise(node_values)
    mantel.to_csv(FIGURES / "table_s6_mantel_pairwise.csv", index=False)
    pd.DataFrame(
        [
            {"node": "SLC7A2", "feature_count": 1, "missing_n": 0, "features": "SLC7A2"},
            {"node": "MAM-module", "feature_count": len(mam_genes), "missing_n": 0, "features": ";".join(mam_genes)},
            {"node": "ARG-transporter", "feature_count": len(arg_genes), "missing_n": 0, "features": ";".join(arg_genes)},
            {"node": "Top20-discovery", "feature_count": len(top), "missing_n": 0, "features": ";".join(top)},
            {"node": "Competitor", "feature_count": len(competitors), "missing_n": 0, "features": ";".join(competitors)},
            {"node": "STK11", "feature_count": 1, "missing_n": 0, "features": "binary mutation call"},
            {"node": "Stage", "feature_count": 1, "missing_n": stage_missing, "features": "ordinal I-IV"},
            {"node": "Age", "feature_count": 1, "missing_n": age_missing, "features": "years"},
            {"node": "Smoking", "feature_count": 1, "missing_n": smoking_missing, "features": "ordinal smoking category"},
        ]
    ).to_csv(FIGURES / "table_s6_mantel_nodes.csv", index=False)

    names = list(node_values)
    n = len(names)
    lookup = {(row.node_a, row.node_b): row for row in mantel.itertuples()}
    rho = np.eye(n)
    q = np.ones((n, n))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                continue
            row = lookup.get((a, b)) or lookup.get((b, a))
            rho[i, j] = row.rho
            q[i, j] = row.mantel_q

    fig = plt.figure(figsize=(12.2, 12.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[2.15, 1])
    ax = fig.add_subplot(grid[0, 0])
    panel_label(ax, "A")
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    for i in range(n):
        for j in range(i):
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=mpl.colormaps["RdBu_r"](norm(rho[i, j])), edgecolor="white", lw=0.5))
    for i in range(n):
        ax.add_patch(Rectangle((i, i), 1, 1, facecolor="#F4F4F4", edgecolor="white", lw=0.5))
    ax.set_xticks(np.arange(n) + 0.5, [x.replace("-", "\n") for x in names], rotation=45, ha="left")
    ax.set_yticks(np.arange(n) + 0.5, [x.replace("-", "\n") for x in names])
    ax.tick_params(length=0, pad=4)
    ax.spines[:].set_visible(False)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.015, shrink=0.52, location="left")
    cbar.set_label("Mantel Spearman rho", rotation=90)
    ax.text(1.0, 0.98, "lower triangle: Mantel correlation", transform=ax.transAxes, ha="right", va="top", color=GRAY, fontsize=8)

    # Network is placed only in the upper-right triangle of the same matrix.
    center = np.array([6.15, 2.10])
    radius = np.array([1.55, 1.15])
    positions = {}
    for k, name in enumerate(names):
        theta = np.pi / 2 + 2 * np.pi * k / n
        positions[name] = center + radius * np.array([np.cos(theta), np.sin(theta)])
    for row in mantel.itertuples():
        if not row.significant:
            continue
        start, end = positions[row.node_a], positions[row.node_b]
        color = ORANGE if row.rho > 0 else BLUE
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=0.7 + 3.2 * abs(row.rho), alpha=0.72, zorder=3)
    node_colors = {"SLC7A2": ORANGE, "MAM-module": "#B55D3D", "ARG-transporter": GREEN, "Top20-discovery": PURPLE,
                   "Competitor": BLUE, "STK11": "#333333", "Stage": GRAY, "Age": GRAY, "Smoking": GRAY}
    network_labels = {
        "SLC7A2": "SLC7A2",
        "MAM-module": "MAM",
        "ARG-transporter": "ARG",
        "Top20-discovery": "TOP20",
        "Competitor": "COMP",
        "STK11": "STK11",
        "Stage": "STAGE",
        "Age": "AGE",
        "Smoking": "SMOKE",
    }
    for name, pos in positions.items():
        ax.scatter([pos[0]], [pos[1]], s=420, color=node_colors[name], edgecolor="white", linewidth=1.0, zorder=4)
        ax.text(pos[0], pos[1], network_labels[name], ha="center", va="center", fontsize=6.5, color="white", zorder=5)
    ax.text(7.55, 0.42, "upper-right: FDR-significant\nMantel interaction network", ha="right", va="top", fontsize=8, color=GRAY)
    ax.set_title("Explainability geometry in KRAS-mutant patients", fontsize=10, pad=18)

    lower = grid[1, 0].subgridspec(1, 2, width_ratios=[1.25, 1])
    ax_effect = fig.add_subplot(lower[0, 0])
    panel_label(ax_effect, "B")
    effect_names = ["SLC7A2", "MAM-module", "ARG-transporter", "Top20-discovery", "Competitor", "Stage", "Age", "Smoking"]
    effect_values = []
    intervals = []
    rng = np.random.default_rng(SEED + 88)
    for name in effect_names:
        vals = node_values[name].mean(axis=1)
        delta = vals[labels == 1].mean() - vals[labels == 0].mean()
        boot = []
        for _ in range(1000):
            pos = rng.choice(vals[labels == 1], size=np.sum(labels == 1), replace=True)
            neg = rng.choice(vals[labels == 0], size=np.sum(labels == 0), replace=True)
            boot.append(pos.mean() - neg.mean())
        effect_values.append(delta)
        intervals.append(np.quantile(boot, [0.025, 0.975]))
    effect_values = np.asarray(effect_values)
    intervals = np.asarray(intervals)
    ypos = np.arange(len(effect_names))
    ax_effect.hlines(ypos, intervals[:, 0], intervals[:, 1], color=BLUE, lw=1.8)
    ax_effect.scatter(effect_values, ypos, color=ORANGE, s=28, zorder=3)
    ax_effect.axvline(0, color=GRAY, lw=0.7)
    ax_effect.set_yticks(ypos, effect_names)
    ax_effect.set_xlabel("Standardized mean difference")
    ax_effect.spines[["top", "right"]].set_visible(False)

    ax_def = fig.add_subplot(lower[0, 1])
    panel_label(ax_def, "C")
    ax_def.axis("off")
    ax_def.text(0.02, 0.96, "Mantel interpretation key", fontsize=9.5, fontweight="bold", va="top")
    definition = (
        "9 patient-distance nodes\n"
        "499 permutations per pair\n"
        "two-sided p-values; BH q across pairs\n"
        "edges shown only at q <= 0.05\n\n"
        "Edge = concordant patient geometry\n"
        "Edge != causal molecular interaction"
    )
    ax_def.text(0.02, 0.77, definition, va="top", linespacing=1.55, fontsize=8.3)
    plt.close(fig)


def _kaplan_meier(time: np.ndarray, event: np.ndarray, group: np.ndarray) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Return step coordinates for two groups without a plotting dependency."""
    curves: dict[int, np.ndarray] = {}
    censored: dict[int, np.ndarray] = {}
    for level in [0, 1]:
        keep = group == level
        order = np.argsort(time[keep])
        t = time[keep][order]
        e = event[keep][order]
        unique = np.unique(t)
        at_risk = len(t)
        survival = 1.0
        points = [(0.0, 1.0)]
        censor_points = []
        for value in unique:
            same = t == value
            deaths = int(np.sum(e[same] == 1))
            censored_n = int(np.sum(e[same] == 0))
            if deaths:
                points.extend([(float(value), survival), (float(value), survival * (1 - deaths / at_risk))])
                survival *= 1 - deaths / at_risk
            if censored_n:
                censor_points.extend([(float(value), survival)] * censored_n)
            at_risk -= deaths + censored_n
        curves[level] = np.asarray(points)
        censored[level] = np.asarray(censor_points)
    return curves, censored


def mantel_link_bins(rho: float, p_value: float) -> tuple[str, str]:
    """Match the reference plot's three Mantel magnitude and p-value classes."""
    magnitude = abs(rho)
    r_bin = "< 0.2" if magnitude < 0.2 else "0.2 - 0.4" if magnitude < 0.4 else ">= 0.4"
    p_bin = "< 0.01" if p_value < 0.01 else "0.01 - 0.05" if p_value < 0.05 else ">= 0.05"
    return r_bin, p_bin


def plot_cox_forest(ax: plt.Axes, fits: list[dict[str, float]], labels: list[str], title: str) -> None:
    y = np.arange(len(fits))
    hr = np.array([item["hazard_ratio_per_score_sd"] for item in fits])
    low = np.array([item["ci_low"] for item in fits])
    high = np.array([item["ci_high"] for item in fits])
    ax.hlines(y, low, high, color="#B55D3D", lw=2)
    ax.scatter(hr, y, color="#B55D3D", s=34, zorder=3)
    ax.axvline(1, color=GRAY, ls="--", lw=.8)
    ax.set_yticks(y, labels)
    ax.set(xlabel="Cox hazard ratio", title=title)
    ax.spines[["top", "right"]].set_visible(False)


def make_fig7_mam_mantel_rich(rna: pd.DataFrame, mutation: pd.DataFrame, clinical: pd.DataFrame) -> None:
    """Six-panel MAM result with a compact Mantel bubble matrix."""
    summary = json.loads((RESULTS / "mam_survival_summary.json").read_text())
    scores = pd.read_csv(RESULTS / "mam_survival_scores.csv")
    repeats = pd.read_csv(RESULTS / "mam_survival_repeated_sensitivity.csv")
    genes = pd.read_csv(RESULTS / "mam_survival_gene_audit.csv")
    nodes, _, missing, _ = prepare_mantel_nodes(rna, mutation, clinical)
    mantel, _ = mantel_pairwise(nodes)
    mantel.to_csv(FIGURES / "table_s6_mantel_pairwise.csv", index=False)
    source_names = ["MAM-module", "ARG-transporter", "Top20-discovery", "Competitor"]
    target_names = ["SLC7A2", "STK11", "Stage", "Age", "Smoking"]
    label = {
        "MAM-module": "MAM core", "ARG-transporter": "ARG", "Top20-discovery": "Top 20",
        "Competitor": "Competitors", "SLC7A2": "SLC7A2", "STK11": "STK11",
        "Stage": "Stage", "Age": "Age", "Smoking": "Smoking",
    }
    lookup = {(row.node_a, row.node_b): row for row in mantel.itertuples()}
    p_colors = {"< 0.01": "#0072B2", "0.01 - 0.05": "#D55E00", ">= 0.05": "#A7A9AC"}
    link_rows = []
    for source in source_names:
        for target in target_names:
            row = lookup.get((source, target)) or lookup.get((target, source))
            r_bin, p_bin = mantel_link_bins(row.rho, row.mantel_p)
            link_rows.append({"source": source, "target": target, "mantel_r": row.rho,
                              "mantel_p": row.mantel_p, "mantel_q": row.mantel_q,
                              "abs_r_bin": r_bin, "p_bin": p_bin})
    pd.DataFrame(link_rows).to_csv(FIGURES / "table_s6_mantel_links.csv", index=False)
    fig = plt.figure(figsize=(13.6, 11.7), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, width_ratios=[1.2, 1.2, 1.05], height_ratios=[1.0, 1.0, 0.92])

    ax = fig.add_subplot(grid[:2, :2]); panel_label(ax, "A")
    ax.set_xlim(-.65, len(target_names) - .35); ax.set_ylim(len(source_names) - .55, -.8)
    ax.set_xticks(np.arange(len(target_names)), [label[name] for name in target_names], rotation=28, ha="left")
    ax.set_yticks(np.arange(len(source_names)), [label[name] for name in source_names])
    ax.set_xlabel("Target variable"); ax.set_ylabel("Expression block")
    ax.set_title("Mantel association matrix in KRAS-mutant patients", fontsize=12.5, pad=22, fontweight="bold")
    ax.grid(color="#E5E5E5", lw=.65, zorder=0); ax.set_axisbelow(True)
    cmap = mpl.colormaps["RdBu_r"]
    norm = TwoSlopeNorm(vmin=-.65, vcenter=0, vmax=.65)
    for i, source in enumerate(source_names):
        for j, target in enumerate(target_names):
            row = lookup.get((source, target)) or lookup.get((target, source))
            r_bin, p_bin = mantel_link_bins(row.rho, row.mantel_p)
            size = 80 + 920 * min(abs(row.rho), .65) / .65
            ax.scatter(j, i, s=size, facecolor=cmap(norm(row.rho)), edgecolor=p_colors[p_bin],
                       linewidth=1.6, alpha=.93, zorder=3)
            mark = "*" if row.mantel_q <= .05 else ""
            ax.text(j, i, f"{row.rho:.2f}{mark}", ha="center", va="center", fontsize=9,
                    color="white" if abs(row.rho) >= .38 else "#222222", zorder=4)
    ax.text(.0, -.17, "Bubble area = |Mantel rho|; fill = signed rho;\nborder = raw p; * BH q <= 0.05",
            transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold", color=GRAY)
    p_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=color,
                         markeredgewidth=1.5, markersize=7, label=name) for name, color in p_colors.items()]
    size_handles = [plt.scatter([], [], s=size, facecolor="#B7C7D8", edgecolor=GRAY, linewidth=.7, label=name)
                    for size, name in [(100, "< 0.2"), (390, "0.2 - 0.4"), (850, ">= 0.4")]]
    p_legend = ax.legend(handles=p_handles, title="Mantel p (raw)", loc="upper left", bbox_to_anchor=(.01, .985),
                         frameon=False, fontsize=9, title_fontsize=10, ncol=3, columnspacing=.7,
                         handletextpad=.35, handlelength=1.0)
    ax.add_artist(p_legend)
    ax.legend(handles=size_handles, title="Mantel |r|", loc="upper right", bbox_to_anchor=(.99, .985),
              frameon=False, fontsize=9, title_fontsize=10, ncol=3, columnspacing=.6,
              handletextpad=.35, handlelength=1.0)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=.025, pad=.015,
                 shrink=.44, label="Mantel rho")

    ax = fig.add_subplot(grid[0, 2]); panel_label(ax, "B")
    plot_cox_forest(ax, [row._asdict() for row in genes.itertuples(index=False)],
                    genes["gene"].tolist(), "Four MAM-core genes")
    ax.set_xlabel("Individual-gene HR")

    ax = fig.add_subplot(grid[1, 2]); panel_label(ax, "C")
    plot_cox_forest(ax, [summary["training_fit"], summary["holdout_fit"], summary["full_cohort_fit"]],
                    ["Training", "Holdout", "Full LUAD"], "Frozen MAM score")

    ax = fig.add_subplot(grid[2, 0]); panel_label(ax, "D")
    time = scores["os_days"].to_numpy(float); event = scores["os_event"].to_numpy(int)
    state = (scores["MAM_survival_score"].to_numpy(float) >= scores["MAM_survival_score"].median()).astype(int)
    curves, censored = _kaplan_meier(time, event, state)
    for level, color, name in [(0, BLUE, "Lower score"), (1, "#B55D3D", "Higher score")]:
        ax.plot(curves[level][:, 0], curves[level][:, 1], color=color, lw=1.7, label=name)
        if len(censored[level]):
            ax.scatter(censored[level][:, 0], censored[level][:, 1], color=color, marker="+", s=10, linewidths=.5)
    ax.set(xlabel="Overall-survival days", ylabel="Survival probability", title="MAM score strata", ylim=(0, 1.03))
    ax.legend(frameon=False, fontsize=7); ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[2, 1]); panel_label(ax, "E")
    plot_cox_forest(ax, [summary["age_stage_adjusted_fit"], summary["kras_wild_type_fit"]],
                    ["Age + stage", "KRAS-wild-type"], "Adjustment and context")

    ax = fig.add_subplot(grid[2, 2]); panel_label(ax, "F")
    ax.axhline(1, color=GRAY, ls="--", lw=.8)
    ax.scatter(repeats["repeat"], repeats["holdout_hazard_ratio"], color="#B55D3D", s=24, zorder=3)
    ax.plot(repeats["repeat"], repeats["holdout_hazard_ratio"], color="#B55D3D", alpha=.40, lw=.9)
    ax.axhline(summary["repeat_holdout_hr_median"], color=ORANGE, lw=1.1)
    ax.set(xlabel="Repeated split", ylabel="Holdout HR", title="20 frozen-score splits")
    ax.spines[["top", "right"]].set_visible(False)

    feature_map = {
        "SLC7A2": "SLC7A2", "MAM-module": ";".join(MAM_SURVIVAL_GENES),
        "ARG-transporter": "SLC7A1;SLC7A2;SLC7A3;SLC7A4",
        "Top20-discovery": "top-20 genes in primary scan", "Competitor": "ADGRF1;S100P;ITGB8",
        "STK11": "binary mutation call", "Stage": "ordinal pathologic stage",
        "Age": "years", "Smoking": "ordinal smoking category",
    }
    pd.DataFrame([{"node": name, "feature_count": value.shape[1] if value.ndim > 1 else 1,
                   "missing_n": missing.get(name, 0), "features": feature_map[name]}
                  for name, value in nodes.items()]).to_csv(FIGURES / "table_s6_mantel_nodes.csv", index=False)
    save_figure(fig, "fig7_mam_survival")


def density_curve(values: np.ndarray, points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    pad = max((hi - lo) * 0.08, 1e-3)
    x = np.linspace(lo - pad, hi + pad, points)
    return x, gaussian_kde(values)(x)


def make_fig2_cohort_rich(rna: pd.DataFrame, mutation: pd.DataFrame, clinical: pd.DataFrame) -> None:
    kras = mutation["KRAS"].to_numpy(int) == 1
    expr = rna.loc[kras]
    labels = mutation.loc[kras, "STK11"].to_numpy(int)
    score = expr["SLC7A2"].to_numpy(float)
    age, _ = fill_missing(clinical.reindex(expr.index)["age"].to_numpy(float))
    stage = encode_stage(clinical.reindex(expr.index)["stage"])
    stage, _ = fill_missing(stage)
    train_idx, holdout_idx, _, _, _ = primary_split(rna, mutation)
    fig = plt.figure(figsize=(12.4, 8.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.18, 1.15, 1.0], height_ratios=[1.18, 1])

    ax = fig.add_subplot(grid[0, 0]); panel_label(ax, "A")
    top_genes = [g for g in pd.read_csv(RESULTS / "primary_discovery_top100.csv")["gene"].head(20) if g in rna.columns]
    cohort_matrix = zscore_columns(rna.loc[:, top_genes].to_numpy(float))
    cohort_coords = PCA(n_components=2, random_state=SEED).fit_transform(cohort_matrix)
    cohort_score = rna["SLC7A2"].to_numpy(float)
    cohort_kras = mutation["KRAS"].to_numpy(int)
    cohort_stk11 = mutation["STK11"].to_numpy(int)
    for kras_value, marker, label in [(0, "o", "KRAS-"), (1, "^", "KRAS+")]:
        keep = cohort_kras == kras_value
        ax.scatter(
            cohort_coords[keep, 0], cohort_coords[keep, 1], c=cohort_score[keep],
            cmap="viridis", vmin=np.nanmin(cohort_score), vmax=np.nanmax(cohort_score),
            marker=marker, s=24, alpha=.78,
            edgecolors=np.where(cohort_stk11[keep] == 1, ORANGE, BLUE), linewidths=.5,
            label=label,
        )
    ax.set(xlabel="PC1: top-20 expression", ylabel="PC2: top-20 expression", title="LUAD cohort geometry")
    ax.grid(alpha=.13); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="best", title="Context", fontsize=6.5, title_fontsize=7)
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=np.nanmin(cohort_score), vmax=np.nanmax(cohort_score)), cmap="viridis")
    fig.colorbar(sm, ax=ax, fraction=.045, pad=.03, label="SLC7A2 expression")

    ax = fig.add_subplot(grid[0, 1]); panel_label(ax, "B")
    colors = np.where(labels == 1, ORANGE, BLUE)
    ax.scatter(age, score, c=score, cmap="viridis", s=28, alpha=.82, edgecolors=colors, linewidths=.65)
    rho = spearmanr(age, score).statistic
    ax.set(xlabel="Age at diagnosis", ylabel="SLC7A2 expression", title=f"Discovery cohort ({len(score)} patients; rho={rho:.2f})")
    ax.grid(alpha=.15); ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[0, 2]); panel_label(ax, "C")
    stage_levels = sorted(np.unique(stage))
    groups = [score[stage == level] for level in stage_levels]
    parts = ax.violinplot(groups, positions=stage_levels, widths=.7, showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(PURPLE); body.set_edgecolor(PURPLE); body.set_alpha(.3)
    parts["cmedians"].set_color("black")
    rng = np.random.default_rng(SEED + 3)
    for level, group in zip(stage_levels, groups):
        jitter = rng.uniform(-.12, .12, len(group))
        ax.scatter(level + jitter, group, c=group, cmap="viridis", s=13, alpha=.75, edgecolors="white", linewidths=.2)
    ax.set_xticks(stage_levels, [f"I", "II", "III", "IV"][:len(stage_levels)])
    ax.set(xlabel="Pathologic stage", ylabel="SLC7A2 expression", title="Expression across stage")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1, 0]); panel_label(ax, "D")
    split_groups = [("Discovery", labels[train_idx]), ("Holdout", labels[holdout_idx])]
    bar_width = .28
    for x_pos, (name, split_labels) in enumerate(split_groups):
        negative_n = int(np.sum(split_labels == 0))
        positive_n = int(np.sum(split_labels == 1))
        ax.bar(x_pos - bar_width / 2, negative_n, color=BLUE, width=bar_width, edgecolor="white", linewidth=.5,
               label="STK11-" if x_pos == 0 else None)
        ax.bar(x_pos + bar_width / 2, positive_n, color=ORANGE, width=bar_width, edgecolor="white", linewidth=.5,
               label="STK11+" if x_pos == 0 else None)
        ax.text(x_pos - bar_width / 2, negative_n + 1.4, str(negative_n), ha="center", va="bottom", color=BLUE, fontsize=8)
        ax.text(x_pos + bar_width / 2, positive_n + 1.4, str(positive_n), ha="center", va="bottom", color=ORANGE, fontsize=8)
    ax.set_xticks([0, 1], ["Discovery\nn=95", "Holdout\nn=42"])
    ax.set(ylabel="Patients", title="Patient-level 70/30 split")
    ax.legend(frameon=False, fontsize=7); ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1, 1]); panel_label(ax, "E")
    for value, color, label in [(score[labels == 0], BLUE, "STK11-"), (score[labels == 1], ORANGE, "STK11+")]:
        x, y = density_curve(value)
        ax.plot(x, y, color=color, lw=1.8, label=label)
        ax.fill_between(x, 0, y, color=color, alpha=.13)
    ax.set(xlabel="SLC7A2 expression", ylabel="Density", title="Continuous expression distributions")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1, 2]); panel_label(ax, "F")
    full_groups = [
        rna.loc[(mutation["KRAS"] == 1) & (mutation["STK11"] == 0), "SLC7A2"].to_numpy(float),
        rna.loc[(mutation["KRAS"] == 1) & (mutation["STK11"] == 1), "SLC7A2"].to_numpy(float),
        rna.loc[(mutation["KRAS"] == 0) & (mutation["STK11"] == 0), "SLC7A2"].to_numpy(float),
        rna.loc[(mutation["KRAS"] == 0) & (mutation["STK11"] == 1), "SLC7A2"].to_numpy(float),
    ]
    violin_with_points(ax, full_groups, ["KRAS+\nSTK11-", "KRAS+\nSTK11+", "KRAS-\nSTK11-", "KRAS-\nSTK11+"], [BLUE, ORANGE, BLUE, ORANGE])
    ax.set(xlabel="Mutation context", ylabel="SLC7A2 expression", title="Expression by KRAS/STK11 group")
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, "fig2_cohort_lock")
    pd.DataFrame(
        {
            "cohort": ["TCGA-LUAD intersection", "KRAS-mutant discovery", "KRAS-wild-type transfer"],
            "n": [494, 137, 357],
            "stk11_positive": [63, 31, 32],
            "stk11_negative": [431, 106, 325],
        }
    ).to_csv(FIGURES / "table_s1_cohort_audit.csv", index=False)


def make_fig3_discovery_rich(rna: pd.DataFrame, mutation: pd.DataFrame) -> None:
    top = pd.read_csv(RESULTS / "primary_discovery_top100.csv").sort_values(["q_value", "gene"]).reset_index(drop=True)
    shown = top.head(20).iloc[::-1]
    fig = plt.figure(figsize=(12.4, 8.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.0, 1.0], height_ratios=[1.2, 1])

    ax = fig.add_subplot(grid[0, 0]); panel_label(ax, "A")
    y = np.arange(len(shown)); colors = [ORANGE if d > 0 else BLUE for d in shown["delta"]]
    ax.hlines(y, 0, shown["delta"], color=colors, lw=1.8, alpha=.68); ax.scatter(shown["delta"], y, color=colors, s=25, zorder=3)
    ax.axvline(0, color=GRAY, lw=.7); ax.set_yticks(y, shown["gene"]); ax.set_xlabel("Training mean difference"); ax.set_title("Top 20 training effects", fontsize=9); ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[0, 1]); panel_label(ax, "B")
    q = np.clip(top["q_value"].to_numpy(float), 1e-16, 1)
    ax.scatter(top["delta"], -np.log10(q), c=-np.log10(q), cmap="viridis", s=18, alpha=.82, edgecolors="white", linewidths=.25)
    candidate = top.loc[top["gene"].eq("SLC7A2")].iloc[0]
    ax.scatter([candidate.delta], [-np.log10(candidate.q_value)], c=ORANGE, s=46, zorder=4)
    ax.annotate("SLC7A2", (candidate.delta, -np.log10(candidate.q_value)), xytext=(5, 5), textcoords="offset points", fontsize=7.5)
    ax.axhline(-np.log10(.05), color=GRAY, ls="--", lw=.7); ax.set(xlabel="Training effect", ylabel="-log10(BH q)", title="Genome-wide significance",); ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[0, 2]); panel_label(ax, "C")
    values = np.sort(top["q_value"].to_numpy(float)); ecdf = np.arange(1, len(values) + 1) / len(values)
    ax.step(-np.log10(values), ecdf, where="post", color=PURPLE, lw=1.8); ax.axvline(-np.log10(.05), color=GRAY, ls="--", lw=.7)
    ax.set(xlabel="-log10(BH q)", ylabel="Cumulative fraction", title="Top-100 q-value distribution"); ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1, 0]); panel_label(ax, "D")
    ranked = top.head(10).sort_values("q_value", ascending=False)
    yp = np.arange(len(ranked)); colors = [ORANGE if x == "SLC7A2" else BLUE for x in ranked["gene"]]
    q_values = -np.log10(np.clip(ranked["q_value"].to_numpy(float), 1e-16, 1))
    ax.hlines(yp, 0, q_values, color=colors, lw=2); ax.scatter(q_values, yp, c=colors, s=34, zorder=3)
    ax.set_yticks(yp, ranked["gene"]); ax.set_xlabel("-log10(training BH q)")
    ax.set_title("Frozen candidate evidence", fontsize=9); ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1, 1]); panel_label(ax, "E")
    kras = mutation["KRAS"].to_numpy(int) == 1
    discovery = rna.loc[kras]
    labels = mutation.loc[kras, "STK11"].to_numpy(int)
    train_idx, _, _, _, position = primary_split(rna, mutation)
    train_expr = discovery.iloc[train_idx, position].to_numpy(float)
    violin_with_points(ax, [train_expr[labels[train_idx] == 0], train_expr[labels[train_idx] == 1]],
                       ["STK11-", "STK11+"], [BLUE, ORANGE])
    ax.set(xlabel="Training patients only", ylabel="SLC7A2 expression", title="Frozen marker in discovery set")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1, 2]); panel_label(ax, "F")
    effects = top["delta"].to_numpy(float); x, y = density_curve(effects)
    ax.plot(x, y, color=GREEN, lw=1.8); ax.fill_between(x, 0, y, color=GREEN, alpha=.16); ax.scatter(effects, np.zeros_like(effects), c=np.clip(effects, -3, 3), cmap="coolwarm", s=12, alpha=.55)
    ax.axvline(0, color=GRAY, lw=.7); ax.set(xlabel="Training effect", ylabel="Density", title="Effect-size density of top 100"); ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, "fig3_genomewide_discovery")
    top.to_csv(FIGURES / "table_s2_primary_discovery_top100.csv", index=False)


def make_fig4_holdout_rich() -> None:
    nested = pd.read_csv(RESULTS / "nested_holdout_predictions.csv")
    holdout = nested.loc[nested["repeat"].eq(0)].copy(); y = holdout["stk11"].to_numpy(int); score = holdout["score"].to_numpy(float)
    random_null = pd.read_csv(RESULTS / "matched_random_single_gene_null.csv")
    controls = pd.read_csv(RESULTS / "arginine_transporter_controls.csv").sort_values("gene")
    fig = plt.figure(figsize=(12.4, 8.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.0, 1.0], height_ratios=[1.15, 1])
    ax = fig.add_subplot(grid[0, 0]); panel_label(ax, "A")
    violin_with_points(ax, [score[y == 0], score[y == 1]], ["STK11-", "STK11+"], [BLUE, ORANGE]); ax.set_ylabel("Frozen SLC7A2 score"); ax.set_title("Holdout score distribution", fontsize=9); ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[0, 1]); panel_label(ax, "B")
    fpr, tpr, _ = roc_curve(y, score); grid_x, low, high = bootstrap_roc(y, score, SEED + 11)
    ax.fill_between(grid_x, low, high, color=ORANGE, alpha=.18, linewidth=0); ax.plot(fpr, tpr, color=ORANGE, lw=2, label=f"AUC {roc_auc_score(y, score):.3f}"); ax.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=.8)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC", xlim=(0, 1), ylim=(0, 1)); ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[0, 2]); panel_label(ax, "C")
    precision, recall, _ = precision_recall_curve(y, score); ap = average_precision_score(y, score)
    ax.plot(recall, precision, color=PURPLE, lw=2, label=f"AP {ap:.3f}"); ax.axhline(y.mean(), color=GRAY, ls="--", lw=.8, label="prevalence"); ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve", xlim=(0, 1), ylim=(0, 1)); ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1, 0]); panel_label(ax, "D")
    null_values = np.sort(random_null["holdout_auc"].to_numpy(float)); ax.step(null_values, np.arange(1, len(null_values) + 1) / len(null_values), where="post", color=BLUE, lw=1.7); ax.axvline(roc_auc_score(y, score), color=ORANGE, lw=1.8)
    ax.set(xlabel="Random-gene holdout AUC", ylabel="ECDF", title="Matched single-gene null"); ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1, 1]); panel_label(ax, "E")
    ypos = np.arange(len(controls)); colors = [ORANGE if g == "SLC7A2" else BLUE for g in controls["gene"]]
    ax.hlines(ypos, .3, controls["holdout_auc"], color=colors, lw=2); ax.scatter(controls["holdout_auc"], ypos, color=colors, s=34, zorder=3); ax.axvline(.5, color=GRAY, ls="--", lw=.8)
    ax.set_yticks(ypos, controls["gene"]); ax.set(xlabel="Holdout AUC", title="Arginine-transporter controls", xlim=(.3, .95)); ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1, 2]); panel_label(ax, "F")
    order = np.argsort(score); rank = np.arange(len(score)); ax.scatter(rank, score[order], c=score[order], cmap="viridis", s=28, alpha=.9, edgecolors=np.where(y[order] == 1, ORANGE, BLUE), linewidths=.65)
    ax.set(xlabel="Holdout patient rank", ylabel="SLC7A2 score", title="Score ordering and label contrast"); ax.grid(alpha=.15); ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, "fig4_primary_holdout")
    validation = json.loads((RESULTS / "primary_holdout_validation.json").read_text())
    pd.DataFrame(
        [
            {"metric": "holdout_auc", "value": roc_auc_score(y, score)},
            {"metric": "average_precision", "value": ap},
            {"metric": "holdout_welch_p", "value": validation["holdout_welch_p"]},
            {"metric": "holdout_mannwhitney_p", "value": validation["holdout_mannwhitney_p"]},
            {"metric": "holdout_score_delta_positive_minus_negative", "value": validation["holdout_score_delta_positive_minus_negative"]},
            {"metric": "bootstrap_ci_low", "value": validation["holdout_auc_bootstrap_95ci"][0]},
            {"metric": "bootstrap_ci_high", "value": validation["holdout_auc_bootstrap_95ci"][1]},
        ]
    ).to_csv(FIGURES / "table_s3_primary_holdout.csv", index=False)


def make_fig5_stability_rich() -> None:
    nested = pd.read_csv(RESULTS / "nested_holdout_summary.csv"); fixed = pd.read_csv(RESULTS / "frozen_candidate_repeated_sensitivity.csv"); freq = pd.read_csv(RESULTS / "candidate_selection_frequency.csv"); null = pd.read_csv(RESULTS / "matched_random_single_gene_null.csv")
    fig = plt.figure(figsize=(12.4, 8.6), constrained_layout=True); grid = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.0, 1.0], height_ratios=[1.1, 1])
    ax = fig.add_subplot(grid[0, 0]); panel_label(ax, "A"); violin_with_points(ax, [nested["holdout_auc"].to_numpy()], ["Nested\n20 splits"], [BLUE]); ax.set_ylabel("Holdout AUC"); ax.set_title("Nested performance", fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    ax = fig.add_subplot(grid[0, 1]); panel_label(ax, "B"); violin_with_points(ax, [fixed["holdout_auc"].to_numpy()], ["Fixed\nSLC7A2"], [ORANGE]); ax.set_ylabel("Holdout AUC"); ax.set_title("Frozen-candidate sensitivity", fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    ax = fig.add_subplot(grid[0, 2]); panel_label(ax, "C"); cval = -np.log10(np.clip(nested["candidate_q_value"].to_numpy(float), 1e-16, 1)); ax.scatter(nested["repeat"], nested["holdout_auc"], c=cval, cmap="viridis", s=38, edgecolors="white", linewidths=.3); ax.set(xlabel="Nested split index", ylabel="AUC", title="Performance heterogeneity"); ax.spines[["top", "right"]].set_visible(False)
    ax = fig.add_subplot(grid[1, 0]); panel_label(ax, "D"); freq = freq.sort_values("selection_frequency"); yp = np.arange(len(freq)); ax.scatter(freq["selection_frequency"], yp, c=freq["selection_frequency"], cmap="plasma", s=48); ax.hlines(yp, 0, freq["selection_frequency"], color=LIGHT_GRAY, lw=1); ax.set_yticks(yp, freq["gene"]); ax.set_xlabel("Selection frequency"); ax.set_xlim(0, .75); ax.spines[["top", "right"]].set_visible(False)
    ax = fig.add_subplot(grid[1, 1]); panel_label(ax, "E"); groups = [null["holdout_auc"].to_numpy(), nested["holdout_auc"].to_numpy(), fixed["holdout_auc"].to_numpy()]; violin_with_points(ax, groups, ["Random", "Nested", "Fixed"], [GRAY, BLUE, ORANGE]); ax.set_ylabel("AUC"); ax.set_title("Null-to-signal separation", fontsize=9); ax.set_ylim(.25, 1); ax.spines[["top", "right"]].set_visible(False)
    ax = fig.add_subplot(grid[1, 2]); panel_label(ax, "F"); xn, yn = density_curve(nested["holdout_auc"].to_numpy()); xf, yf = density_curve(fixed["holdout_auc"].to_numpy()); ax.plot(xn, yn, color=BLUE, lw=1.8, label="nested"); ax.fill_between(xn, 0, yn, color=BLUE, alpha=.13); ax.plot(xf, yf, color=ORANGE, lw=1.8, label="fixed"); ax.fill_between(xf, 0, yf, color=ORANGE, alpha=.13); ax.set(xlabel="AUC", ylabel="Density", title="AUC distribution shape"); ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, "fig5_stability_null"); nested.to_csv(FIGURES / "table_s4_nested_sensitivity.csv", index=False)


def make_fig6_context_rich(rna: pd.DataFrame, mutation: pd.DataFrame, clinical: pd.DataFrame) -> None:
    train_idx, holdout_idx, labels, score, position = primary_split(rna, mutation); kras = mutation["KRAS"].to_numpy(int) == 1; discovery = rna.loc[kras]; transfer_y = mutation.loc[~kras, "STK11"].to_numpy(int); transfer_score = standardized_score(discovery.to_numpy(float)[train_idx, position], rna.loc[~kras].to_numpy(float)[:, position], 1.0); holdout_y = labels[holdout_idx]
    full_ids = rna.index; full_score = standardized_score(discovery.to_numpy(float)[train_idx, position], rna.to_numpy(float)[:, position], 1.0)
    fig = plt.figure(figsize=(12.4, 8.6), constrained_layout=True); grid = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.0, 1.0], height_ratios=[1.1, 1])
    ax = fig.add_subplot(grid[0, 0]); panel_label(ax, "A"); fpr1, tpr1, _ = roc_curve(holdout_y, score); fpr2, tpr2, _ = roc_curve(transfer_y, transfer_score); ax.plot(fpr1, tpr1, color=ORANGE, lw=2, label=f"KRAS+ {roc_auc_score(holdout_y, score):.3f}"); ax.plot(fpr2, tpr2, color=BLUE, lw=2, label=f"KRAS- {roc_auc_score(transfer_y, transfer_score):.3f}"); ax.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=.8); ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Cross-context ROC", xlim=(0, 1), ylim=(0, 1)); ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    ax = fig.add_subplot(grid[0, 1]); panel_label(ax, "B"); violin_with_points(ax, [score[holdout_y == 0], score[holdout_y == 1], transfer_score[transfer_y == 0], transfer_score[transfer_y == 1]], ["KRAS+\nSTK11-", "KRAS+\nSTK11+", "KRAS-\nSTK11-", "KRAS-\nSTK11+"], [BLUE, ORANGE, BLUE, ORANGE]); ax.set_ylabel("Frozen SLC7A2 score"); ax.set_title("Context score distributions", fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    ax = fig.add_subplot(grid[0, 2]); panel_label(ax, "C"); aucs = np.array([roc_auc_score(holdout_y, score), roc_auc_score(transfer_y, transfer_score)]); cis = np.array([bootstrap_auc_interval(holdout_y, score, SEED + 21), bootstrap_auc_interval(transfer_y, transfer_score, SEED + 22)]); yp = np.arange(2); ax.hlines(yp, cis[:, 0], cis[:, 1], color=[ORANGE, BLUE], lw=2); ax.scatter(aucs, yp, color=[ORANGE, BLUE], s=40, zorder=3); ax.axvline(.5, color=GRAY, ls="--", lw=.8); ax.set_yticks(yp, ["KRAS+ holdout", "KRAS- transfer"]); ax.set_xlabel("AUC with bootstrap interval"); ax.set_xlim(.4, 1); ax.spines[["top", "right"]].set_visible(False)
    ax = fig.add_subplot(grid[1, 0]); panel_label(ax, "D")
    transfer_order = np.argsort(transfer_score)
    transfer_stk11 = transfer_y[transfer_order]
    ax.scatter(np.arange(len(transfer_order)), transfer_score[transfer_order], c=transfer_score[transfer_order], cmap="viridis", s=20,
               alpha=.78, edgecolors=np.where(transfer_stk11 == 1, ORANGE, BLUE), linewidths=.55)
    ax.set(xlabel="KRAS-wild-type patient rank", ylabel="Frozen SLC7A2 score", title="Internal transfer ordering")
    ax.grid(alpha=.13); ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1, 1]); panel_label(ax, "E")
    wt_idx = np.flatnonzero(~kras)
    order = wt_idx[np.argsort(full_score[wt_idx])]
    bins = np.array_split(order, 8)
    x_bin, prevalence, low, high = [], [], [], []
    for idx in bins:
        n = len(idx); successes = int(mutation.loc[full_ids[idx], "STK11"].to_numpy(int).sum()); lo, hi = wilson_count_interval(successes, n)
        x_bin.append(float(np.median(full_score[idx]))); prevalence.append(successes / n); low.append(lo / n); high.append(hi / n)
    x_bin = np.asarray(x_bin); prevalence = np.asarray(prevalence)
    ax.errorbar(x_bin, prevalence, yerr=[prevalence - np.asarray(low), np.asarray(high) - prevalence], fmt="none", ecolor=GRAY, elinewidth=1, capsize=2)
    ax.scatter(x_bin, prevalence, c=x_bin, cmap="plasma", s=38, edgecolor="white", linewidth=.4, zorder=3)
    ax.set(xlabel="Median frozen score (KRAS-wild-type decile)", ylabel="STK11-mutant fraction", title="Transfer label prevalence", ylim=(-.03, 1.03))
    ax.grid(alpha=.13); ax.spines[["top", "right"]].set_visible(False)
    ax = fig.add_subplot(grid[1, 2]); panel_label(ax, "F")
    mam_table = pd.read_csv(RESULTS / "mam_survival_scores.csv").set_index("patient_id").reindex(full_ids)
    mam_score = mam_table["MAM_survival_score"].to_numpy(float)
    score_mask = np.isfinite(mam_score) & np.isfinite(full_score)
    mam_rho = spearmanr(mam_score[score_mask], full_score[score_mask]).statistic
    ax.scatter(mam_score[score_mask], full_score[score_mask], c=mutation.loc[full_ids, "STK11"].to_numpy(int)[score_mask], cmap="coolwarm", s=22, alpha=.72, edgecolors="white", linewidths=.35)
    ax.set(xlabel="MAM survival state score", ylabel="Frozen SLC7A2 score", title=f"MAM survival state (rho={mam_rho:.2f})")
    ax.grid(alpha=.13); ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, "fig6_context_transfer")
    pd.DataFrame([{"context": "KRAS+ holdout", "auc": aucs[0], "ci_low": cis[0, 0], "ci_high": cis[0, 1]}, {"context": "KRAS- transfer", "auc": aucs[1], "ci_low": cis[1, 0], "ci_high": cis[1, 1]}]).to_csv(FIGURES / "table_s5_context_transfer.csv", index=False)
    pd.read_csv(RESULTS / "arginine_transporter_controls.csv").to_csv(FIGURES / "table_s5_transporter_controls.csv", index=False)
    mam_summary = json.loads((RESULTS / "mam_survival_summary.json").read_text())
    pd.DataFrame([{
        "module": mam_summary["module"],
        "gene_count": mam_summary["gene_count"],
        "endpoint": mam_summary["endpoint"],
        "cohort_n": mam_summary["cohort_n"],
        "holdout_hazard_ratio": mam_summary["holdout_fit"]["hazard_ratio_per_score_sd"],
        "holdout_ci_low": mam_summary["holdout_fit"]["ci_low"],
        "holdout_ci_high": mam_summary["holdout_fit"]["ci_high"],
        "holdout_p_value": mam_summary["holdout_fit"]["p_value"],
        "full_cohort_hazard_ratio": mam_summary["full_cohort_fit"]["hazard_ratio_per_score_sd"],
        "full_cohort_p_value": mam_summary["full_cohort_fit"]["p_value"],
        "age_stage_adjusted_hazard_ratio": mam_summary["age_stage_adjusted_fit"]["hazard_ratio_per_score_sd"],
        "age_stage_adjusted_p_value": mam_summary["age_stage_adjusted_fit"]["p_value"],
        "kras_wild_type_hazard_ratio": mam_summary["kras_wild_type_fit"]["hazard_ratio_per_score_sd"],
        "kras_wild_type_p_value": mam_summary["kras_wild_type_fit"]["p_value"],
        "gene_level_fdr_positive_n": mam_summary["gene_level_fdr_positive_n"],
        "score_spearman_with_SLC7A2": mam_summary["score_spearman_with_SLC7A2"],
    }]).to_csv(FIGURES / "table_s5_mam_context.csv", index=False)


def mantel_null_distribution(node_a: np.ndarray, node_b: np.ndarray, seed: int, n_perm: int = MANTEL_PERMUTATIONS) -> tuple[float, np.ndarray]:
    node_a = np.asarray(node_a, float); node_b = np.asarray(node_b, float)
    if node_a.ndim == 1: node_a = node_a[:, None]
    if node_b.ndim == 1: node_b = node_b[:, None]
    d_a = pdist(node_a); d_b = pdist(node_b); observed = float(spearmanr(d_a, d_b).statistic); matrix = squareform(d_b); tri = np.triu_indices(matrix.shape[0], 1); rng = np.random.default_rng(seed); null = []
    for _ in range(n_perm):
        perm = rng.permutation(matrix.shape[0]); null.append(float(spearmanr(d_a, matrix[np.ix_(perm, perm)][tri]).statistic))
    return observed, np.asarray(null)


def prepare_mantel_nodes(rna: pd.DataFrame, mutation: pd.DataFrame, clinical: pd.DataFrame) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, int], list[str]]:
    kras = mutation["KRAS"].to_numpy(int) == 1; ids = rna.index[kras]; expr = rna.loc[ids]; labels = mutation.loc[ids, "STK11"].to_numpy(int); clinical = clinical.reindex(ids)
    top = [g for g in pd.read_csv(RESULTS / "primary_discovery_top100.csv")["gene"].head(20).tolist() if g in expr.columns]; competitors = [g for g in ["ADGRF1", "S100P", "ITGB8"] if g in expr.columns]; arg_genes = [g for g in ["SLC7A1", "SLC7A2", "SLC7A3", "SLC7A4"] if g in expr.columns]; mam_genes = [g for g in MAM_SURVIVAL_GENES if g in expr.columns]
    stage, stage_missing = fill_missing(encode_stage(clinical["stage"])); age, age_missing = fill_missing(clinical["age"].to_numpy(float)); smoking, smoking_missing = fill_missing(encode_smoking(clinical["smoking_status"]))
    nodes = {"SLC7A2": zscore_columns(expr[["SLC7A2"]].to_numpy(float)), "MAM-module": zscore_columns(expr[mam_genes].to_numpy(float)), "ARG-transporter": zscore_columns(expr[arg_genes].to_numpy(float)), "Top20-discovery": zscore_columns(expr[top].to_numpy(float)), "Competitor": zscore_columns(expr[competitors].to_numpy(float)), "STK11": labels.astype(float), "Stage": zscore_columns(stage), "Age": zscore_columns(age), "Smoking": zscore_columns(smoking)}
    missing = {"Stage": stage_missing, "Age": age_missing, "Smoking": smoking_missing}; return nodes, labels, missing, list(nodes)


def make_fig8_context_rich(rna: pd.DataFrame, mutation: pd.DataFrame, clinical: pd.DataFrame) -> None:
    """Display measured clinical and cohort gradients; claim boundaries stay in the legend."""
    kras = mutation["KRAS"].to_numpy(int) == 1
    discovery = rna.loc[kras]
    labels = mutation.loc[kras, "STK11"].to_numpy(int)
    train_idx, _, _, _, position = primary_split(rna, mutation)
    frozen_score = standardized_score(discovery.to_numpy(float)[train_idx, position], rna.to_numpy(float)[:, position], 1.0)
    clinical_all = clinical.reindex(rna.index)
    age, _ = fill_missing(clinical_all["age"].to_numpy(float))
    stage = encode_stage(clinical_all["stage"])
    smoking = encode_smoking(clinical_all["smoking_status"])
    sex = clinical_all["sex"].fillna("Unknown").astype(str).to_numpy()
    all_stk11 = mutation["STK11"].to_numpy(int)
    all_kras = mutation["KRAS"].to_numpy(int)
    fig = plt.figure(figsize=(12.4, 8.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.05, 1.05], height_ratios=[1.1, 1])

    ax = fig.add_subplot(grid[0, 0]); panel_label(ax, "A")
    valid = np.isfinite(age) & np.isfinite(frozen_score)
    ax.scatter(age[valid], frozen_score[valid], c=frozen_score[valid], cmap="viridis", s=22, alpha=.72,
               edgecolors=np.where(all_stk11[valid] == 1, ORANGE, BLUE), linewidths=.5)
    ax.set(xlabel="Age at diagnosis", ylabel="Frozen SLC7A2 score", title="All-cohort score gradient")
    ax.grid(alpha=.13); ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[0, 1]); panel_label(ax, "B")
    groups = [
        frozen_score[(all_kras == 1) & (all_stk11 == 0)], frozen_score[(all_kras == 1) & (all_stk11 == 1)],
        frozen_score[(all_kras == 0) & (all_stk11 == 0)], frozen_score[(all_kras == 0) & (all_stk11 == 1)],
    ]
    violin_with_points(ax, groups, ["KRAS+\nSTK11-", "KRAS+\nSTK11+", "KRAS-\nSTK11-", "KRAS-\nSTK11+"], [BLUE, ORANGE, BLUE, ORANGE])
    ax.set(xlabel="Mutation context", ylabel="Frozen SLC7A2 score", title="Score distributions")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[0, 2]); panel_label(ax, "C")
    stage_keep = np.isfinite(stage)
    rng = np.random.default_rng(SEED + 81)
    for level in sorted(np.unique(stage[stage_keep])):
        keep = stage == level
        jitter = rng.uniform(-.13, .13, keep.sum())
        ax.scatter(np.full(keep.sum(), level) + jitter, frozen_score[keep], c=frozen_score[keep], cmap="plasma",
                   vmin=np.nanmin(frozen_score), vmax=np.nanmax(frozen_score), s=17, alpha=.72,
                   edgecolors=np.where(all_stk11[keep] == 1, ORANGE, BLUE), linewidths=.35)
        ax.plot([level - .24, level + .24], [np.median(frozen_score[keep])] * 2, color="black", lw=1.4)
    ax.set_xticks([1, 2, 3, 4], ["I", "II", "III", "IV"]); ax.set(xlabel="Pathologic stage", ylabel="Frozen SLC7A2 score", title="Stage-stratified score")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1, 0]); panel_label(ax, "D")
    smoke_keep = np.isfinite(smoking)
    smoke_labels = ["Never", ">15 y", "<=15 y", "Current"]
    for level in sorted(np.unique(smoking[smoke_keep])):
        keep = smoking == level
        jitter = rng.uniform(-.13, .13, keep.sum())
        ax.scatter(np.full(keep.sum(), level) + jitter, frozen_score[keep], c=frozen_score[keep], cmap="viridis",
                   vmin=np.nanmin(frozen_score), vmax=np.nanmax(frozen_score), s=16, alpha=.7,
                   edgecolors=np.where(all_stk11[keep] == 1, ORANGE, BLUE), linewidths=.35)
        ax.plot([level - .24, level + .24], [np.median(frozen_score[keep])] * 2, color="black", lw=1.3)
    ax.set_xticks([0, 1, 2, 3], smoke_labels); ax.set(xlabel="Smoking category", ylabel="Frozen score", title="Smoking-stratified score")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1, 1]); panel_label(ax, "E")
    sex_levels = [x for x in ["male", "female", "Unknown"] if x in set(sex)]
    sex_groups = [frozen_score[(sex == level) & (all_stk11 == cls)] for level in sex_levels for cls in [0, 1]]
    sex_labels = [f"{level}\nSTK11{'+' if cls else '-'}" for level in sex_levels for cls in [0, 1]]
    sex_colors = [BLUE, ORANGE] * len(sex_levels)
    violin_with_points(ax, sex_groups, sex_labels, sex_colors)
    ax.set(xlabel="Sex and STK11", ylabel="Frozen SLC7A2 score", title="Sex-stratified score")
    ax.tick_params(axis="x", labelsize=6.5); ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1, 2]); panel_label(ax, "F")
    order = np.argsort(frozen_score)
    bins = np.array_split(order, 10)
    x_bin, prevalence, low, high = [], [], [], []
    for idx in bins:
        n = len(idx); successes = int(all_stk11[idx].sum()); lo, hi = wilson_count_interval(successes, n)
        x_bin.append(float(np.median(frozen_score[idx]))); prevalence.append(successes / n); low.append(lo / n); high.append(hi / n)
    x_bin = np.asarray(x_bin); prevalence = np.asarray(prevalence)
    ax.errorbar(x_bin, prevalence, yerr=[prevalence - np.asarray(low), np.asarray(high) - prevalence], fmt="none", ecolor=GRAY, elinewidth=1, capsize=2)
    ax.scatter(x_bin, prevalence, c=x_bin, cmap="plasma", s=42, edgecolor="white", linewidth=.4, zorder=3)
    ax.set(xlabel="Median frozen SLC7A2 score (decile)", ylabel="STK11-mutant fraction", title="STK11 prevalence across score deciles", ylim=(-.03, 1.03))
    ax.grid(alpha=.13); ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, "fig8_clinical_context")
    pd.DataFrame(
        [
            {"claim": "SLC7A2 expression marks STK11 mutation in TCGA-LUAD", "status": "supported"},
            {"claim": "Association transfers to KRAS-wild-type TCGA-LUAD", "status": "supported as internal context transfer"},
            {"claim": "SLC7A2 is causal for STK11 biology", "status": "excluded"},
            {"claim": "SLC7A2 predicts treatment response or clinical outcome", "status": "excluded"},
        ]
    ).to_csv(FIGURES / "table_s7_claim_ledger.csv", index=False)


def main() -> None:
    configure_matplotlib()
    rna, mutation, clinical = load_context()
    make_fig2_cohort_rich(rna, mutation, clinical)
    make_fig3_discovery_rich(rna, mutation)
    make_fig4_holdout_rich()
    make_fig5_stability_rich()
    make_fig6_context_rich(rna, mutation, clinical)
    make_fig7_mam_mantel_rich(rna, mutation, clinical)
    make_fig8_context_rich(rna, mutation, clinical)
    print(f"wrote figures and tables to {FIGURES}")


if __name__ == "__main__":
    main()
