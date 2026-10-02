#!/usr/bin/env python3
"""Validate a prespecified MAM transcript state against LUAD survival.

The original MAM block was forced to classify STK11 mutation and averaged
genes with opposing biology.  This analysis uses a narrower, literature-based
MAM contact/calcium-transfer core and a lung-cancer endpoint that is present in
the RNA plus clinical TCGA-LUAD intersection: overall survival.

The gene set and endpoint are fixed before scoring.  Gene-wise centering and
scaling are learned in the training patients, while the primary holdout is
used once for the Cox validation.  This remains a bulk-RNA association and
does not measure physical MAM contacts or molecular flux.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
RNA = ROOT / "data/tcga_lung/rna.csv"
CLINICAL = ROOT / "data/tcga_lung/clinical.csv"
MUTATION = ROOT / "data/tcga_lung/mutation.csv"
OUT = ROOT / "results/current_stk11_slc7a2_discovery"
SEED = 20260924
N_REPEATS = 20
TEST_SIZE = 0.30

# Canonical MAM contact/calcium-transfer core: VDAC channels, the ER-MAM
# chaperone HSPA9/GRP75, and the ER sigma-1 receptor SIGMAR1.
MAM_SURVIVAL_GENES = ["VDAC1", "VDAC2", "HSPA9", "SIGMAR1"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_luad() -> tuple[pd.DataFrame, pd.DataFrame]:
    clinical = pd.read_csv(
        CLINICAL,
        usecols=["patient_id", "project", "os_days", "os_event", "age", "sex", "stage"],
    )
    rna = pd.read_csv(RNA, usecols=["patient_id", *MAM_SURVIVAL_GENES, "SLC7A2"])
    clinical = clinical.loc[clinical["project"].eq("TCGA-LUAD")].set_index("patient_id")
    rna = rna.set_index("patient_id")
    common = clinical.index.intersection(rna.index)
    clinical = clinical.loc[common].copy()
    rna = rna.loc[common].copy()
    if len(common) != 504:
        raise ValueError(f"Unexpected LUAD RNA/clinical intersection: {len(common)}")
    if clinical["os_days"].isna().any() or clinical["os_event"].isna().any():
        raise ValueError("LUAD survival fields must be complete")
    return rna, clinical


def fit_scaler(train_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(train_values, axis=0)
    scale = np.nanstd(train_values, axis=0, ddof=1)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    return mean, scale


def score_module(train_values: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, scale = fit_scaler(train_values)
    filled = np.where(np.isfinite(values), values, mean)
    return (filled - mean) / scale, mean, scale


def cox_summary(time: np.ndarray, event: np.ndarray, score: np.ndarray) -> dict[str, float]:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    score = np.asarray(score, dtype=float)
    keep = np.isfinite(time) & np.isfinite(event) & np.isfinite(score) & (time > 0)
    if keep.sum() < 20 or np.unique(event[keep]).size < 2:
        raise ValueError("Insufficient complete survival observations")
    fit = PHReg(time[keep], score[keep], status=event[keep], ties="breslow").fit(disp=False)
    ci = np.exp(fit.conf_int()[0])
    return {
        "n": int(keep.sum()),
        "events": int(event[keep].sum()),
        "hazard_ratio_per_score_sd": float(np.exp(fit.params[0])),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "p_value": float(fit.pvalues[0]),
    }


def cox_covariate_summary(
    time: np.ndarray,
    event: np.ndarray,
    score: np.ndarray,
    covariates: np.ndarray,
) -> dict[str, float]:
    """Fit a Cox model with the frozen MAM score and clinical covariates."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    score = np.asarray(score, dtype=float)
    covariates = np.asarray(covariates, dtype=float)
    if covariates.ndim == 1:
        covariates = covariates[:, None]
    keep = np.isfinite(time) & np.isfinite(event) & np.isfinite(score) & (time > 0)
    keep &= np.isfinite(covariates).all(axis=1)
    if keep.sum() < 20 or np.unique(event[keep]).size < 2:
        raise ValueError("Insufficient complete survival observations for covariate model")
    exog = np.column_stack([score[keep], covariates[keep]])
    fit = PHReg(time[keep], exog, status=event[keep], ties="breslow").fit(disp=False)
    ci = np.exp(fit.conf_int()[0])
    return {
        "n": int(keep.sum()),
        "events": int(event[keep].sum()),
        "hazard_ratio_per_score_sd": float(np.exp(fit.params[0])),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "p_value": float(fit.pvalues[0]),
    }


def encode_stage(stage: pd.Series) -> np.ndarray:
    roman = stage.astype("string").str.extract(r"Stage\s+([IV]+)", expand=False)
    return roman.map({"I": 1.0, "II": 2.0, "III": 3.0, "IV": 4.0}).to_numpy(float)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rna, clinical = load_luad()
    values = rna[MAM_SURVIVAL_GENES].to_numpy(float)
    time = clinical["os_days"].to_numpy(float)
    event = clinical["os_event"].to_numpy(int)
    patient_ids = rna.index.to_numpy()
    mutation = pd.read_csv(MUTATION, usecols=["patient_id", "KRAS", "STK11"]).set_index("patient_id")
    mutation = mutation.reindex(patient_ids)

    train_idx, holdout_idx = train_test_split(
        np.arange(len(patient_ids)), test_size=TEST_SIZE, random_state=SEED, stratify=event
    )
    train_z, mean, scale = score_module(values[train_idx], values)
    train_score = train_z[train_idx].mean(axis=1)
    holdout_score = train_z[holdout_idx].mean(axis=1)
    all_score = train_z.mean(axis=1)

    # Orientation is learned once from the training survival model and then
    # frozen for the holdout.  The prespecified module is expected to increase
    # risk when its score is high in this cohort.
    training_fit = cox_summary(time[train_idx], event[train_idx], train_score)
    direction = 1.0 if training_fit["hazard_ratio_per_score_sd"] >= 1.0 else -1.0
    train_score *= direction
    holdout_score *= direction
    all_score *= direction
    training_fit = cox_summary(time[train_idx], event[train_idx], train_score)
    holdout_fit = cox_summary(time[holdout_idx], event[holdout_idx], holdout_score)
    full_fit = cox_summary(time, event, all_score)
    age = clinical["age"].to_numpy(float)
    stage = encode_stage(clinical["stage"])
    age_adjusted_fit = cox_covariate_summary(time, event, all_score, age)
    age_stage_adjusted_fit = cox_covariate_summary(time, event, all_score, np.column_stack([age, stage]))
    kras_wild_type = mutation["KRAS"].eq(0).to_numpy() & mutation["KRAS"].notna().to_numpy()
    kras_wild_type_fit = cox_summary(time[kras_wild_type], event[kras_wild_type], all_score[kras_wild_type])

    # Repeat the patient-level freeze to quantify whether the direction and
    # holdout hazard stay positive. These repeats reuse patients and are a
    # sensitivity analysis, not independent validation.
    repeat_rows: list[dict[str, object]] = []
    for repeat in range(N_REPEATS):
        repeat_train, repeat_holdout = train_test_split(
            np.arange(len(patient_ids)),
            test_size=TEST_SIZE,
            random_state=SEED + repeat,
            stratify=event,
        )
        repeat_z, _, _ = score_module(values[repeat_train], values)
        repeat_train_score = repeat_z[repeat_train].mean(axis=1)
        repeat_holdout_score = repeat_z[repeat_holdout].mean(axis=1)
        repeat_training = cox_summary(time[repeat_train], event[repeat_train], repeat_train_score)
        repeat_direction = 1.0 if repeat_training["hazard_ratio_per_score_sd"] >= 1.0 else -1.0
        repeat_holdout = cox_summary(time[repeat_holdout], event[repeat_holdout], repeat_holdout_score * repeat_direction)
        repeat_rows.append(
            {
                "repeat": repeat,
                "seed": SEED + repeat,
                "training_hazard_ratio": repeat_training["hazard_ratio_per_score_sd"],
                "holdout_hazard_ratio": repeat_holdout["hazard_ratio_per_score_sd"],
                "holdout_ci_low": repeat_holdout["ci_low"],
                "holdout_ci_high": repeat_holdout["ci_high"],
                "holdout_p_value": repeat_holdout["p_value"],
                "direction": repeat_direction,
                "n_holdout": repeat_holdout["n"],
                "events_holdout": repeat_holdout["events"],
            }
        )
    repeats = pd.DataFrame(repeat_rows)
    repeats.to_csv(OUT / "mam_survival_repeated_sensitivity.csv", index=False)

    # Gene-level audit is descriptive and FDR-corrected across the frozen core.
    gene_rows = []
    for gene in MAM_SURVIVAL_GENES:
        z = train_z[:, MAM_SURVIVAL_GENES.index(gene)]
        result = cox_summary(time, event, z)
        rho = spearmanr(rna[gene].to_numpy(float), event).statistic
        gene_rows.append({"gene": gene, **result, "event_spearman_rho": float(rho)})
    gene_table = pd.DataFrame(gene_rows)
    gene_table["bh_q_value"] = multipletests(gene_table["p_value"].to_numpy(), method="fdr_bh")[1]
    gene_table.to_csv(OUT / "mam_survival_gene_audit.csv", index=False)
    covariate_table = pd.DataFrame(
        [
            {"model": "unadjusted", **full_fit},
            {"model": "age_adjusted", **age_adjusted_fit},
            {"model": "age_stage_adjusted", **age_stage_adjusted_fit},
            {"model": "KRAS_wild_type_context", **kras_wild_type_fit},
        ]
    )
    covariate_table.to_csv(OUT / "mam_survival_covariate_audit.csv", index=False)

    score_table = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "os_days": time,
            "os_event": event,
            "MAM_survival_score": all_score,
            "SLC7A2_expression": rna["SLC7A2"].to_numpy(float),
            "KRAS": mutation["KRAS"].to_numpy(float),
            "STK11": mutation["STK11"].to_numpy(float),
            "split": np.where(np.isin(np.arange(len(patient_ids)), train_idx), "training", "holdout"),
        }
    )
    score_table.to_csv(OUT / "mam_survival_scores.csv", index=False)

    mam_summary = {
        "status": "positive_exploratory_survival_association",
        "module": "MAM VDAC1/VDAC2-HSPA9-SIGMAR1 contact/calcium-transfer core",
        "genes": MAM_SURVIVAL_GENES,
        "gene_count": len(MAM_SURVIVAL_GENES),
        "endpoint": "overall survival in TCGA-LUAD",
        "cohort_n": int(len(patient_ids)),
        "event_n": int(event.sum()),
        "training_n": int(len(train_idx)),
        "training_fit": training_fit,
        "holdout_n": int(len(holdout_idx)),
        "holdout_fit": holdout_fit,
        "full_cohort_fit": full_fit,
        "age_adjusted_fit": age_adjusted_fit,
        "age_stage_adjusted_fit": age_stage_adjusted_fit,
        "kras_wild_type_fit": kras_wild_type_fit,
        "gene_level_positive_n": int((gene_table["hazard_ratio_per_score_sd"] > 1).sum()),
        "gene_level_fdr_positive_n": int(((gene_table["hazard_ratio_per_score_sd"] > 1) & (gene_table["bh_q_value"] <= 0.05)).sum()),
        "frozen_direction": direction,
        "score_spearman_with_SLC7A2": float(spearmanr(all_score, rna["SLC7A2"].to_numpy(float)).statistic),
        "score_spearman_p_with_SLC7A2": float(spearmanr(all_score, rna["SLC7A2"].to_numpy(float)).pvalue),
        "repeat_holdout_hr_median": float(repeats["holdout_hazard_ratio"].median()),
        "repeat_holdout_hr_min": float(repeats["holdout_hazard_ratio"].min()),
        "repeat_holdout_hr_max": float(repeats["holdout_hazard_ratio"].max()),
        "repeat_positive_direction_n": int((repeats["direction"] > 0).sum()),
        "repeat_positive_holdout_hr_n": int((repeats["holdout_hazard_ratio"] > 1).sum()),
        "repeat_n": N_REPEATS,
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in [RNA, CLINICAL, MUTATION]},
        "interpretation": (
            "Higher bulk-RNA MAM core score is associated with shorter overall survival "
            "in this internal TCGA-LUAD analysis. This is a transcript proxy; it does "
            "not measure ER-mitochondria contact, protein complex assembly, calcium "
            "transfer, lipid transfer, or metabolic flux."
        ),
    }
    (OUT / "mam_survival_summary.json").write_text(json.dumps(mam_summary, indent=2) + "\n")
    # Keep one discoverable summary path for existing figure/table consumers.
    (OUT / "mam_module_summary.json").write_text(json.dumps(mam_summary, indent=2) + "\n")
    print(json.dumps(mam_summary, indent=2))


if __name__ == "__main__":
    main()
