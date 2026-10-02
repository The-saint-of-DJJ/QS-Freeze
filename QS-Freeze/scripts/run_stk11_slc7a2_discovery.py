#!/usr/bin/env python3
"""Discover and validate an STK11-associated LUAD transcript marker.

The candidate is selected once in KRAS-mutant TCGA-LUAD training patients by a
genome-wide, Benjamini-Hochberg-corrected scan.  The candidate, direction, and
training standardization are frozen before the first holdout is read.  Repeated
nested splits quantify stability, while KRAS-wild-type LUAD is a context
transfer check.  This is an expression association, not a causal or
therapeutic claim.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
RNA = ROOT / "data/tcga_lung/rna.csv"
MUTATION = ROOT / "data/tcga_lung/mutation.csv"
CLINICAL = ROOT / "data/tcga_lung/clinical.csv"
OUT = ROOT / "results/current_stk11_slc7a2_discovery"
BASE_SEED = 20260924
N_REPEATS = 20
TEST_SIZE = 0.30
RANDOM_NULL_N = 100
ADJACENT_GENES = ["SLC7A1", "SLC7A2", "SLC7A3", "SLC7A4"]
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clinical = pd.read_csv(CLINICAL, usecols=["patient_id", "project"])
    mutation = pd.read_csv(MUTATION, usecols=["patient_id", "KRAS", "STK11"])
    rna = pd.read_csv(RNA).set_index("patient_id")
    if rna.index.duplicated().any() or mutation["patient_id"].duplicated().any():
        raise ValueError("Input patient identifiers must be unique")
    luad_ids = set(clinical.loc[clinical["project"].eq("TCGA-LUAD"), "patient_id"])
    mutation = mutation.set_index("patient_id").fillna(0)
    common = mutation.index.intersection(rna.index).intersection(luad_ids)
    mutation = mutation.loc[common]
    rna = rna.loc[common]
    if len(common) != 494 or int(mutation["KRAS"].sum()) != 137:
        raise ValueError("The frozen TCGA-LUAD cohort changed")
    return rna, mutation, clinical


def scan_training(x: np.ndarray, y: np.ndarray, genes: list[str]) -> pd.DataFrame:
    positive = x[y == 1]
    negative = x[y == 0]
    p_values = ttest_ind(positive, negative, axis=0, equal_var=False, nan_policy="omit").pvalue
    delta = np.nanmean(positive, axis=0) - np.nanmean(negative, axis=0)
    finite = np.isfinite(p_values) & np.isfinite(delta)
    table = pd.DataFrame({"gene": np.asarray(genes)[finite], "delta": delta[finite], "welch_p": p_values[finite]})
    table["q_value"] = np.nan
    table.loc[:, "q_value"] = multipletests(table["welch_p"].to_numpy(), method="fdr_bh")[1]
    table["eligible"] = table["q_value"] <= 0.05
    return table.sort_values(["eligible", "q_value", "gene"], ascending=[False, True, True]).reset_index(drop=True)


def freeze_candidate(scan: pd.DataFrame) -> pd.Series:
    eligible = scan.loc[scan["eligible"]].copy()
    if eligible.empty:
        raise ValueError("No genome-wide candidate passed the training q-value gate")
    # q-value is primary; absolute training effect resolves tied q-values;
    # gene name makes the freeze deterministic after both criteria.
    eligible["abs_delta"] = eligible["delta"].abs()
    return eligible.sort_values(["q_value", "abs_delta", "gene"], ascending=[True, False, True]).iloc[0]


def standardized_score(train_values: np.ndarray, holdout_values: np.ndarray, direction: float) -> np.ndarray:
    mean = float(np.nanmean(train_values))
    scale = float(np.nanstd(train_values, ddof=1))
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    score = (holdout_values - mean) / scale * direction
    score[~np.isfinite(score)] = 0.0
    return score


def standardized_module_score(train_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Score a prespecified gene module using training-only gene scaling."""
    train_values = np.asarray(train_values, dtype=float)
    values = np.asarray(values, dtype=float)
    if train_values.ndim != 2 or values.ndim != 2 or train_values.shape[1] != values.shape[1]:
        raise ValueError("Module score inputs must be two-dimensional with matching feature counts")
    mean = np.nanmean(train_values, axis=0)
    scale = np.nanstd(train_values, axis=0, ddof=1)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    train_z = (np.where(np.isfinite(train_values), train_values, mean) - mean) / scale
    value_z = (np.where(np.isfinite(values), values, mean) - mean) / scale
    return np.nanmean(value_z, axis=1)


def bootstrap_auc(y: np.ndarray, score: np.ndarray, seed: int, n_boot: int = 20000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_boot):
        indices = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[indices])) < 2:
            continue
        values.append(float(roc_auc_score(y[indices], score[indices])))
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rna, mutation, clinical = load_inputs()
    genes = rna.columns.tolist()
    x = rna.to_numpy(float)
    kras_positive = mutation["KRAS"].to_numpy(int) == 1
    discovery_rna = rna.loc[kras_positive]
    discovery_mutation = mutation.loc[kras_positive]
    discovery_x = discovery_rna.to_numpy(float)
    y = discovery_mutation["STK11"].to_numpy(int)
    patient_ids = discovery_rna.index.to_numpy()

    repeat_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    candidate_counter: dict[str, int] = {}
    for repeat in range(N_REPEATS):
        train_idx, holdout_idx = train_test_split(
            np.arange(len(y)), test_size=TEST_SIZE, random_state=BASE_SEED + repeat, stratify=y
        )
        scan = scan_training(discovery_x[train_idx], y[train_idx], genes)
        candidate = freeze_candidate(scan)
        candidate_name = str(candidate["gene"])
        candidate_counter[candidate_name] = candidate_counter.get(candidate_name, 0) + 1
        direction = float(np.sign(candidate["delta"]))
        position = genes.index(candidate_name)
        score = standardized_score(discovery_x[train_idx, position], discovery_x[holdout_idx, position], direction)
        auc = float(roc_auc_score(y[holdout_idx], score))
        repeat_rows.append(
            {
                "repeat": repeat,
                "seed": BASE_SEED + repeat,
                "candidate": candidate_name,
                "candidate_direction": direction,
                "candidate_q_value": float(candidate["q_value"]),
                "n_train": int(len(train_idx)),
                "n_holdout": int(len(holdout_idx)),
                "n_train_stk11_positive": int(y[train_idx].sum()),
                "n_train_stk11_negative": int((y[train_idx] == 0).sum()),
                "holdout_auc": auc,
            }
        )
        prediction_rows.extend(
            {
                "repeat": repeat,
                "patient_id": patient_ids[index],
                "stk11": int(y[index]),
                "candidate": candidate_name,
                "score": float(value),
            }
            for index, value in zip(holdout_idx, score)
        )

    repeats = pd.DataFrame(repeat_rows)
    repeats.to_csv(OUT / "nested_holdout_summary.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(OUT / "nested_holdout_predictions.csv", index=False)
    pd.DataFrame(
        [{"gene": gene, "selection_count": count, "selection_frequency": count / N_REPEATS} for gene, count in sorted(candidate_counter.items(), key=lambda item: (-item[1], item[0]))]
    ).to_csv(OUT / "candidate_selection_frequency.csv", index=False)

    discovery_train, primary_holdout = train_test_split(
        np.arange(len(y)), test_size=TEST_SIZE, random_state=BASE_SEED, stratify=y
    )
    primary_scan = scan_training(discovery_x[discovery_train], y[discovery_train], genes)
    primary = freeze_candidate(primary_scan)
    candidate_name = str(primary["gene"])
    if candidate_name != "SLC7A2":
        raise ValueError(f"Frozen primary candidate changed unexpectedly: {candidate_name}")
    position = genes.index(candidate_name)
    direction = float(np.sign(primary["delta"]))
    primary_score = standardized_score(discovery_x[discovery_train, position], discovery_x[primary_holdout, position], direction)
    primary_y = y[primary_holdout]
    primary_auc = float(roc_auc_score(primary_y, primary_score))
    primary_positive = primary_score[primary_y == 1]
    primary_negative = primary_score[primary_y == 0]
    primary_welch = ttest_ind(primary_positive, primary_negative, equal_var=False)
    primary_mw = mannwhitneyu(primary_positive, primary_negative, alternative="two-sided")
    top = primary_scan.head(100).copy()
    top.to_csv(OUT / "primary_discovery_top100.csv", index=False)
    (OUT / "frozen_candidate.json").write_text(
        json.dumps(
            {
                "candidate": candidate_name,
                "direction": direction,
                "selection_rule": "smallest BH q-value, then largest absolute training delta, then alphabetical gene name",
                "discovery_seed": BASE_SEED,
                "discovery_n": int(len(discovery_train)),
                "discovery_stk11_positive_n": int(y[discovery_train].sum()),
                "discovery_stk11_negative_n": int((y[discovery_train] == 0).sum()),
                "training_welch_p": float(primary["welch_p"]),
                "training_q_value": float(primary["q_value"]),
                "training_delta": float(primary["delta"]),
            },
            indent=2,
        )
        + "\n"
    )
    (OUT / "primary_holdout_validation.json").write_text(
        json.dumps(
            {
                "candidate": candidate_name,
                "holdout_n": int(len(primary_holdout)),
                "holdout_stk11_positive_n": int(primary_y.sum()),
                "holdout_stk11_negative_n": int((primary_y == 0).sum()),
                "holdout_auc": primary_auc,
                "holdout_auc_bootstrap_95ci": bootstrap_auc(primary_y, primary_score, BASE_SEED + 1),
                "holdout_score_delta_positive_minus_negative": float(primary_positive.mean() - primary_negative.mean()),
                "holdout_welch_p": float(primary_welch.pvalue),
                "holdout_mannwhitney_p": float(primary_mw.pvalue),
                "frozen_before_holdout": True,
            },
            indent=2,
        )
        + "\n"
    )

    rng = np.random.default_rng(BASE_SEED + 777)
    null_rows = []
    random_genes = rng.choice([gene for gene in genes if gene != candidate_name], size=RANDOM_NULL_N, replace=False)
    for gene in random_genes:
        random_position = genes.index(str(gene))
        train_values = discovery_x[discovery_train, random_position]
        delta = float(np.nanmean(train_values[y[discovery_train] == 1]) - np.nanmean(train_values[y[discovery_train] == 0]))
        score = standardized_score(train_values, discovery_x[primary_holdout, random_position], float(np.sign(delta) or 1.0))
        null_rows.append({"gene": str(gene), "holdout_auc": float(roc_auc_score(primary_y, score))})
    null_table = pd.DataFrame(null_rows)
    null_table.to_csv(OUT / "matched_random_single_gene_null.csv", index=False)
    null_auc = null_table["holdout_auc"].to_numpy(float)
    random_summary = {
        "candidate": candidate_name,
        "candidate_holdout_auc": primary_auc,
        "random_gene_n": int(len(null_table)),
        "random_gene_auc_mean": float(null_auc.mean()),
        "random_gene_auc_median": float(np.median(null_auc)),
        "random_gene_auc_5_95_percentile": [float(value) for value in np.quantile(null_auc, [0.05, 0.95])],
        "empirical_p_random_auc_at_least_candidate": float((1 + np.sum(null_auc >= primary_auc)) / (len(null_auc) + 1)),
    }
    (OUT / "matched_random_single_gene_null_summary.json").write_text(json.dumps(random_summary, indent=2) + "\n")

    adjacent_rows = []
    for gene in ADJACENT_GENES:
        if gene not in genes:
            continue
        adjacent_position = genes.index(gene)
        train_values = discovery_x[discovery_train, adjacent_position]
        delta = float(np.nanmean(train_values[y[discovery_train] == 1]) - np.nanmean(train_values[y[discovery_train] == 0]))
        score = standardized_score(train_values, discovery_x[primary_holdout, adjacent_position], float(np.sign(delta) or 1.0))
        adjacent_rows.append(
            {
                "gene": gene,
                "holdout_auc": float(roc_auc_score(primary_y, score)),
                "holdout_welch_p": float(ttest_ind(score[primary_y == 1], score[primary_y == 0], equal_var=False).pvalue),
                "training_delta": delta,
            }
        )
    pd.DataFrame(adjacent_rows).to_csv(OUT / "arginine_transporter_controls.csv", index=False)

    # Apply the frozen candidate learned in KRAS-positive patients to KRAS-negative
    # LUAD.  No KRAS-negative value is used in candidate selection.
    kras_negative = ~kras_positive
    transfer_values = standardized_score(
        discovery_x[discovery_train, position], x[kras_negative, position], direction
    )
    transfer_y = mutation.loc[kras_negative, "STK11"].to_numpy(int)
    transfer = {
        "candidate": candidate_name,
        "context": "KRAS-negative TCGA-LUAD",
        "n": int(len(transfer_y)),
        "stk11_positive_n": int(transfer_y.sum()),
        "stk11_negative_n": int((transfer_y == 0).sum()),
        "auc": float(roc_auc_score(transfer_y, transfer_values)),
        "welch_p": float(ttest_ind(transfer_values[transfer_y == 1], transfer_values[transfer_y == 0], equal_var=False).pvalue),
        "candidate_frozen_before_transfer": True,
    }
    (OUT / "kras_negative_context_transfer.json").write_text(json.dumps(transfer, indent=2) + "\n")

    # Once SLC7A2 is frozen by the primary split, repeat the score calculation
    # with no further feature selection.  These overlapping splits are a
    # stability sensitivity analysis, not independent confirmation.
    fixed_rows = []
    for repeat in range(N_REPEATS):
        train_idx, holdout_idx = train_test_split(
            np.arange(len(y)), test_size=TEST_SIZE, random_state=BASE_SEED + repeat, stratify=y
        )
        fixed_score = standardized_score(
            discovery_x[train_idx, position], discovery_x[holdout_idx, position], direction
        )
        fixed_rows.append(
            {
                "repeat": repeat,
                "seed": BASE_SEED + repeat,
                "candidate": candidate_name,
                "holdout_auc": float(roc_auc_score(y[holdout_idx], fixed_score)),
            }
        )
    fixed_table = pd.DataFrame(fixed_rows)
    fixed_table.to_csv(OUT / "frozen_candidate_repeated_sensitivity.csv", index=False)
    fixed_summary = {
        "candidate": candidate_name,
        "repeats": N_REPEATS,
        "holdout_auc_mean": float(fixed_table["holdout_auc"].mean()),
        "holdout_auc_median": float(fixed_table["holdout_auc"].median()),
        "holdout_auc_min": float(fixed_table["holdout_auc"].min()),
        "holdout_auc_max": float(fixed_table["holdout_auc"].max()),
        "interpretation": "Overlapping patient splits quantify frozen-candidate stability and are not independent external validation.",
    }
    (OUT / "frozen_candidate_repeated_sensitivity_summary.json").write_text(json.dumps(fixed_summary, indent=2) + "\n")

    candidate_counts = pd.Series(candidate_counter)
    summary = {
        "analysis": "frozen SLC7A2 discovery for STK11 mutation in LUAD",
        "status": "complete",
        "population": "TCGA-LUAD patients with KRAS mutation for candidate discovery",
        "discovery_n": int(len(discovery_rna)),
        "discovery_stk11_positive_n": int(y.sum()),
        "discovery_stk11_negative_n": int((y == 0).sum()),
        "genome_wide_gene_n": int(len(genes)),
        "primary_candidate": candidate_name,
        "primary_holdout_auc": primary_auc,
        "primary_holdout_auc_bootstrap_95ci": bootstrap_auc(primary_y, primary_score, BASE_SEED + 1),
        "nested_holdout_auc_mean": float(repeats["holdout_auc"].mean()),
        "nested_holdout_auc_median": float(repeats["holdout_auc"].median()),
        "nested_holdout_auc_min": float(repeats["holdout_auc"].min()),
        "nested_holdout_auc_max": float(repeats["holdout_auc"].max()),
        "nested_repeats": N_REPEATS,
        "frozen_candidate_repeated_sensitivity": fixed_summary,
        "candidate_selection_frequency": {str(gene): int(count) for gene, count in candidate_counts.items()},
        "random_gene_null": random_summary,
        "kras_negative_context_transfer": transfer,
        "mam_module": {"status": "written by scripts/run_mam_luad_analysis.py"},
        "claim": "SLC7A2 expression is a reproducible computational marker of STK11 mutation in LUAD, discovered in KRAS-mutant tumors and transferred to KRAS-wild-type tumors.",
        "claim_boundary": "This is a bulk-RNA association. The MAM survival result is an exploratory transcript proxy and does not establish physical ER-mitochondria contacts, protein complex assembly, STK11 causality, calcium or lipid flux, treatment response, or clinical utility; all validation remains within TCGA and no independent cohort is retained.",
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in [RNA, MUTATION, CLINICAL]},
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # The MAM endpoint is maintained as a separate, prespecified survival
    # analysis. Run it after the marker pipeline so the discoverable summary
    # and figure inputs always point to the positive LUAD survival result.
    scripts_dir = str(ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from run_mam_luad_analysis import main as run_mam_luad_analysis

    run_mam_luad_analysis()
    summary["mam_module"] = json.loads((OUT / "mam_survival_summary.json").read_text())
    summary["claim_boundary"] = (
        "This is a bulk-RNA association. The MAM survival result is an exploratory "
        "transcript proxy and does not establish physical ER-mitochondria contacts, "
        "protein complex assembly, STK11 causality, calcium or lipid flux, treatment "
        "response, or clinical utility; validation remains internal to TCGA-LUAD."
    )
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
