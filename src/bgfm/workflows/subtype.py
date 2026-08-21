# -*- coding: utf-8 -*-
"""
OOF sPLS-DA-like brain-gut supervised factor analysis and MDD subtype discovery.

This script is a cleaned, self-contained Python implementation derived from the
OOF PLS-DA analysis script. It removes the model extraction/checkpoint code and
starts directly from OOF node representations saved by the previous pipeline.

Only the normalized bridge feature source is analysed:
    normed_bridge : X = [||B_bridge_normed||, ||M_bridge_normed||]

For this source, a sparse PLS-DA-like pipeline is run:
    - repeated stratified CV searches n_components and keepX
    - keepX grid is 10,20,...,120 by default
    - sparsity is approximated by selecting top keepX variables from PLS weights
      within the training fold and refitting PLSRegression on the selected union
    - final sparse PLS-DA-like model is fitted on all HC/MDD samples
    - factor scores, weights, statistics, bootstrap stability, subtype clustering,
      clinical associations, and node explanations are saved.

Important:
    Python/sklearn does not provide a fully equivalent mixOmics::splsda(). This is
    a practical sPLS-DA-like approximation: PLSRegression + weight-based sparse
    variable selection + refit. For a strict sPLS-DA implementation, use R mixOmics.

Significance:
    All factor/subtype/clinical selection and annotations use raw p values only.
    No q/FDR is used for selection in this script.
"""

import os
import re
import random
import warnings
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)
from sklearn.cluster import KMeans
import networkx as nx


from bgfm.runtime import load_section, apply_globals, apply_mapping

# ============================================================
# CONFIG: only modify paths and main parameters here
# ============================================================
CONFIG: Dict[str, Any] = {
    # OOF outputs from your previous OOF extraction pipeline.
    "NODE_REPRESENTATION_NPZ": r"outputs/alignment/node_representations_raw_and_normed.npz",
    "SAMPLE_INFO_CSV": r"outputs/alignment/aligned_samples.csv",

    # Optional fallback if sample info has no group column.
    "LABEL_CSV": r"data/paired/labels_mdd_hc.csv",
    "LABEL_SAMPLE_COL": "sample_id",
    "LABEL_GROUP_COL": "group",

    # Optional clinical file. If unavailable, clinical analyses are skipped.
    "CLINICAL_CSV": r"data/paired/clinical_biomarkers.csv",
    "CLINICAL_SAMPLE_COL": "sample_id",
    "CLINICAL_COLUMNS": [],  # empty = all numeric columns except sample_id

    # Optional name files. If unavailable, ROI_001... and Taxa_001... are used.
    "ROI_NAME_FILE": r"data/metadata/brain_roi_names.txt",
    "TAXA_NAME_FILE": r"data/metadata/taxa_names.txt",

    "OUT_DIR": r"outputs/subtype",

    # Basic constants.
    "N_ROIS": 90,
    "N_TAXA": 642,
    "HC_GROUP_NAME": "hc",
    "MDD_GROUP_NAME": "mdd",
    "GROUP_ORDER": ["hc", "mdd"],
    "SEED": 42,
    "FIG_DPI": 300,

    # Feature construction.
    # Only normed_bridge is analysed; raw_bridge is intentionally disabled.
    "FEATURE_SOURCES": ["normed_bridge"],
    "BLOCK_SCALE": True,        # divide ROI block by sqrt(90), taxa block by sqrt(642)
    "STANDARDIZE_BLOCKS": True, # z-score ROI and taxa blocks separately within CV/final fit

    # sPLS-DA-like component and keepX search.
    "SPLSDA_MAX_COMPONENTS": 10,
    "SPLSDA_CV_FOLDS": 5,
    "SPLSDA_CV_REPEATS": 10,
    "SPLSDA_SELECT_BY": "ber",  # "ber" or "auc"
    "SPLSDA_TIE_TOL": 0.01,
    "KEEPX_GRID": list(range(10, 301, 10)),

    # Bootstrap stability.
    "BOOTSTRAP_N": 300,
    "BOOTSTRAP_SAMPLE_FRAC": 0.80,
    "TOP_FEATURE_OVERLAP_N": 30,

    # Factor selection thresholds. Use raw p value, not q value.
    "MIN_SCORE_STABILITY": 0.60,
    "STRICT_MIN_SCORE_STABILITY": 0.70,
    "MIN_AUC_BALANCED": 0.70,
    "STRICT_MIN_AUC_BALANCED": 0.70,
    "MAX_P_VALUE": 0.05,
    "MIN_ABS_CLIFF_DELTA": 0.50,
    "STRICT_MIN_ABS_CLIFF_DELTA": 0.50,
    "MIN_SELECTED_FACTORS_FOR_CLUSTER": 1,
    "MAX_SELECTED_FACTORS_FOR_CLUSTER": 2,

    # K-means subtype analysis.
    "KMEANS_K_MIN": 2,
    "KMEANS_K_MAX": 6,
    "KMEANS_N_INIT": 100,
    "KMEANS_BOOTSTRAP_N": 200,
    "KMEANS_BOOTSTRAP_FRAC": 0.80,
    "MIN_CLUSTER_SIZE": 8,
    "MIN_CLUSTER_RATIO": 0.08,

    # Plotting / explanation.
    "TOP_ROI_PER_FACTOR": 10,
    "TOP_TAXA_PER_FACTOR": 10,
    "TOP_NODE_HEATMAP_UNION": 20,
    "BIPARTITE_TOP_ROI": 10,
    "BIPARTITE_TOP_TAXA": 10,
    "CLINICAL_P_THRESHOLD": 0.05,
    "MIN_ABS_SPEARMAN_R": 0.25,

    # Clinical phenotype keyword annotation.
    "PHENOTYPE_KEYWORDS": {
        "suicide_related": ["suicide", "suicidal", "si", "自杀"],
        "depression_severity": ["hamd", "bdi", "depress", "抑郁"],
        "anxiety": ["hama", "anxiety", "anxious", "焦虑"],
        "sleep": ["sleep", "insomnia", "睡眠"],
        "inflammation": ["il6", "il-6", "tnf", "crp", "炎症"],
        "cognition": ["cognition", "cognitive", "mmse", "moca", "认知"],
    },
}


apply_mapping(CONFIG, load_section('subtype'))

# ============================================================
# Basic utilities
# ============================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def sanitize_filename(s: str, max_len: int = 120) -> str:
    s = re.sub(r"[\\/:*?\"<>|]", "_", str(s))
    s = re.sub(r"\s+", "_", s).strip("_")
    return s[:max_len] if s else "unnamed"


def save_df(df: pd.DataFrame, out_dir: str, filename: str) -> str:
    ensure_dir(out_dir)
    fp = os.path.join(out_dir, filename)
    df.to_csv(fp, index=False, encoding="utf-8-sig")
    print("[SAVE]", fp)
    return fp


def save_plot_data(df: pd.DataFrame, out_dir: str, filename: str) -> str:
    if not filename.endswith(".csv"):
        filename += ".csv"
    return save_df(df, out_dir, filename)


def read_name_list(path: Optional[str], n_expected: int, prefix: str) -> List[str]:
    if path is None or str(path).strip() == "" or not os.path.exists(path):
        return [f"{prefix}_{i+1:03d}" for i in range(n_expected)]
    if str(path).lower().endswith(".csv"):
        df = pd.read_csv(path)
        names = df.iloc[:, 0].astype(str).tolist()
    else:
        with open(path, "r", encoding="utf-8") as f:
            names = [x.strip() for x in f if x.strip()]
    if len(names) < n_expected:
        raise ValueError(f"{path} has {len(names)} names, expected >= {n_expected}")
    return names[:n_expected]


def safe_mannwhitneyu(x, y) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 1 or len(y) < 1:
        return np.nan, np.nan
    try:
        u, p = stats.mannwhitneyu(x, y, alternative="two-sided", method="auto")
    except TypeError:
        u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    except Exception:
        return np.nan, np.nan
    return float(u), float(p)


def cohen_d(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / max(nx + ny - 2, 1)
    if pooled <= 0 or not np.isfinite(pooled):
        return np.nan
    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled))


def cliffs_delta(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    try:
        u, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
        return float((2.0 * u) / (nx * ny) - 1.0)
    except Exception:
        return np.nan


def safe_auc(y_true, score) -> float:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true, score))
    except Exception:
        return np.nan


def evaluate_binary_predictions(y_true, score) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    auc_raw = safe_auc(y_true, score)
    auc_bal = max(auc_raw, 1.0 - auc_raw) if np.isfinite(auc_raw) else np.nan
    y_hat = (score >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_hat)
    bacc = balanced_accuracy_score(y_true, y_hat)
    ber = 1.0 - bacc
    cm = confusion_matrix(y_true, y_hat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {
        "auc_raw": float(auc_raw) if np.isfinite(auc_raw) else np.nan,
        "auc_balanced": float(auc_bal) if np.isfinite(auc_bal) else np.nan,
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "BER": float(ber),
        "sensitivity_mdd": float(sens),
        "specificity_hc": float(spec),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def rank_normalize_metric(x: pd.Series, higher_is_better: bool = True) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    if x.notna().sum() <= 1:
        return pd.Series(np.zeros(len(x)), index=x.index)
    r = x.rank(pct=True)
    if not higher_is_better:
        r = 1.0 - r
    return r.fillna(0.0)


def map_phenotype(metric: str) -> str:
    low = str(metric).lower()
    for pheno, keys in CONFIG.get("PHENOTYPE_KEYWORDS", {}).items():
        for key in keys:
            if str(key).lower() in low:
                return pheno
    return "other"


# ============================================================
# Input loading and feature construction
# ============================================================
def load_sample_info() -> pd.DataFrame:
    path = CONFIG["SAMPLE_INFO_CSV"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"SAMPLE_INFO_CSV not found: {path}")
    df = pd.read_csv(path)
    if "sample_id" not in df.columns:
        raise ValueError("SAMPLE_INFO_CSV must contain sample_id column")
    df["sample_id"] = df["sample_id"].astype(str)

    if "group" not in df.columns:
        lab_path = CONFIG.get("LABEL_CSV", "")
        if not lab_path or not os.path.exists(lab_path):
            raise ValueError("sample info has no group column and LABEL_CSV is unavailable")
        lab = pd.read_csv(lab_path)
        sc, gc = CONFIG["LABEL_SAMPLE_COL"], CONFIG["LABEL_GROUP_COL"]
        if sc not in lab.columns or gc not in lab.columns:
            raise ValueError(f"LABEL_CSV must contain {sc} and {gc}")
        lab = lab[[sc, gc]].copy()
        lab[sc] = lab[sc].astype(str)
        lab[gc] = lab[gc].astype(str).str.lower()
        lab = lab.rename(columns={sc: "sample_id", gc: "group"})
        df = df.merge(lab, on="sample_id", how="left")

    df["group"] = df["group"].astype(str).str.lower()
    hc, mdd = CONFIG["HC_GROUP_NAME"].lower(), CONFIG["MDD_GROUP_NAME"].lower()
    df = df[df["group"].isin([hc, mdd])].copy().reset_index(drop=True)
    return df


def choose_npz_key(data: np.lib.npyio.NpzFile, candidates: List[str], required: bool = True) -> Optional[str]:
    keys = set(data.files)
    for c in candidates:
        if c in keys:
            return c
    if required:
        raise KeyError(f"None of keys {candidates} found in NPZ. Available keys: {data.files}")
    return None


def load_bridge_arrays() -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], List[str], List[str], List[str], List[str], List[str]]:
    """Return dict source -> (B, M), plus sample ids/groups/feature info."""
    npz_path = CONFIG["NODE_REPRESENTATION_NPZ"]
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NODE_REPRESENTATION_NPZ not found: {npz_path}")
    info = load_sample_info()
    data = np.load(npz_path)

    n_rois, n_taxa = int(CONFIG["N_ROIS"]), int(CONFIG["N_TAXA"])
    sources: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # Normed bridge. Accept multiple historical key names.
    if "normed_bridge" in CONFIG["FEATURE_SOURCES"]:
        bk = choose_npz_key(data, ["B_bridge_normed", "B_bridge_n", "B_bridge"], required=False)
        mk = choose_npz_key(data, ["M_bridge_normed", "M_bridge_n", "M_bridge"], required=False)
        if bk is not None and mk is not None:
            sources["normed_bridge"] = (np.asarray(data[bk], dtype=np.float64), np.asarray(data[mk], dtype=np.float64))
        else:
            print("[WARN] normed_bridge requested but normed bridge keys are missing; skipped.")

    if not sources:
        raise ValueError("No usable bridge source found in NPZ.")

    # Shape checks and sample count.
    first_source = next(iter(sources))
    N = sources[first_source][0].shape[0]
    if len(info) != N:
        raise ValueError(
            f"Sample info N={len(info)} does not match NPZ N={N}. Use the aligned_samples.csv generated with the NPZ."
        )
    for name, (B, M) in sources.items():
        if B.ndim != 3 or B.shape[1] != n_rois:
            raise ValueError(f"{name}: B expected (N,{n_rois},d), got {B.shape}")
        if M.ndim != 3 or M.shape[1] != n_taxa:
            raise ValueError(f"{name}: M expected (N,{n_taxa},d), got {M.shape}")
        if B.shape[0] != N or M.shape[0] != N:
            raise ValueError(f"{name}: inconsistent sample numbers B={B.shape[0]}, M={M.shape[0]}, expected {N}")

    roi_names = read_name_list(CONFIG.get("ROI_NAME_FILE"), n_rois, "ROI")
    taxa_names = read_name_list(CONFIG.get("TAXA_NAME_FILE"), n_taxa, "Taxa")
    feature_names = roi_names + taxa_names
    feature_types = ["ROI"] * n_rois + ["Taxa"] * n_taxa
    feature_cols = [f"{t}:{n}" for t, n in zip(feature_types, feature_names)]
    sids = info["sample_id"].astype(str).tolist()
    groups = info["group"].astype(str).str.lower().tolist()
    return sources, sids, groups, feature_names, feature_types, feature_cols


def construct_node_matrix(B: np.ndarray, M: np.ndarray, out_dir: str, source_name: str, sids: List[str], groups: List[str], feature_cols: List[str]):
    n_rois, n_taxa = int(CONFIG["N_ROIS"]), int(CONFIG["N_TAXA"])
    B_norm = np.linalg.norm(B, axis=2)
    M_norm = np.linalg.norm(M, axis=2)

    check_df = pd.DataFrame({
        "source": [source_name, source_name],
        "block": ["ROI", "Taxa"],
        "mean_norm": [float(np.nanmean(B_norm)), float(np.nanmean(M_norm))],
        "std_norm": [float(np.nanstd(B_norm)), float(np.nanstd(M_norm))],
        "min_norm": [float(np.nanmin(B_norm)), float(np.nanmin(M_norm))],
        "max_norm": [float(np.nanmax(B_norm)), float(np.nanmax(M_norm))],
    })
    save_df(check_df, out_dir, f"{source_name}_node_norm_diagnostics.csv")

    if bool(CONFIG.get("BLOCK_SCALE", True)):
        B_block = B_norm / np.sqrt(n_rois)
        M_block = M_norm / np.sqrt(n_taxa)
    else:
        B_block = B_norm.copy()
        M_block = M_norm.copy()
    X = np.concatenate([B_block, M_block], axis=1)
    X_raw_norm = np.concatenate([B_norm, M_norm], axis=1)

    aligned = pd.DataFrame({"sample_id": sids, "group": groups})
    save_df(aligned, out_dir, f"{source_name}_aligned_samples.csv")

    mat_df = pd.DataFrame(X, columns=feature_cols)
    mat_df.insert(0, "group", groups)
    mat_df.insert(0, "sample_id", sids)
    save_df(mat_df, out_dir, f"{source_name}_node_feature_matrix_blockscaled.csv")

    raw_df = pd.DataFrame(X_raw_norm, columns=feature_cols)
    raw_df.insert(0, "group", groups)
    raw_df.insert(0, "sample_id", sids)
    save_df(raw_df, out_dir, f"{source_name}_node_feature_matrix_raw_norm.csv")
    return X, X_raw_norm, B_norm, M_norm


def make_fit_mask(groups: List[str]) -> np.ndarray:
    hc, mdd = CONFIG["HC_GROUP_NAME"].lower(), CONFIG["MDD_GROUP_NAME"].lower()
    g = np.array([str(x).lower() for x in groups])
    return np.isin(g, [hc, mdd])


def binary_labels(groups: List[str], fit_mask: Optional[np.ndarray] = None) -> np.ndarray:
    g = np.array([str(x).lower() for x in groups])
    if fit_mask is not None:
        g = g[np.asarray(fit_mask, dtype=bool)]
    hc, mdd = CONFIG["HC_GROUP_NAME"].lower(), CONFIG["MDD_GROUP_NAME"].lower()
    if set(np.unique(g)) - {hc, mdd}:
        raise ValueError(f"sPLS-DA requires only HC/MDD labels, got {sorted(set(np.unique(g)))}")
    return (g == mdd).astype(float)


def standardize_train_valid(X_train: np.ndarray, X_valid: np.ndarray):
    n_rois = int(CONFIG["N_ROIS"])
    if not bool(CONFIG.get("STANDARDIZE_BLOCKS", True)):
        return X_train.copy(), X_valid.copy()
    rs = StandardScaler()
    ts = StandardScaler()
    train_roi = rs.fit_transform(X_train[:, :n_rois])
    valid_roi = rs.transform(X_valid[:, :n_rois])
    train_taxa = ts.fit_transform(X_train[:, n_rois:])
    valid_taxa = ts.transform(X_valid[:, n_rois:])
    return np.concatenate([train_roi, train_taxa], axis=1), np.concatenate([valid_roi, valid_taxa], axis=1)


def standardize_blocks_fit_all(X_raw: np.ndarray, fit_mask: np.ndarray):
    n_rois = int(CONFIG["N_ROIS"])
    if not bool(CONFIG.get("STANDARDIZE_BLOCKS", True)):
        return X_raw[fit_mask].copy(), X_raw.copy()
    roi_scaler = StandardScaler()
    taxa_scaler = StandardScaler()
    X_fit = X_raw[fit_mask]
    fit_roi = roi_scaler.fit_transform(X_fit[:, :n_rois])
    fit_taxa = taxa_scaler.fit_transform(X_fit[:, n_rois:])
    all_roi = roi_scaler.transform(X_raw[:, :n_rois])
    all_taxa = taxa_scaler.transform(X_raw[:, n_rois:])
    return np.concatenate([fit_roi, fit_taxa], axis=1), np.concatenate([all_roi, all_taxa], axis=1)


# ============================================================
# sPLS-DA-like fitting
# ============================================================
def select_top_keepx_by_component(weights: np.ndarray, keepx: int) -> np.ndarray:
    """Return boolean mask: union of top keepX abs weights for each component."""
    p, k = weights.shape
    mask = np.zeros(p, dtype=bool)
    keepx = int(min(max(1, keepx), p))
    for kk in range(k):
        idx = np.argsort(np.abs(weights[:, kk]))[::-1][:keepx]
        mask[idx] = True
    return mask


def fit_sparse_plsda_once(X_train_std: np.ndarray, y_train: np.ndarray, ncomp: int, keepx: int):
    """Approximate sPLS-DA: full PLS weights -> top keepX union -> refit PLS on selected features."""
    ncomp0 = int(min(ncomp, X_train_std.shape[0] - 2, X_train_std.shape[1]))
    if ncomp0 < 1:
        raise ValueError("n_components is invalid for this training fold")
    full = PLSRegression(n_components=ncomp0, scale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full.fit(X_train_std, y_train)
    mask = select_top_keepx_by_component(full.x_weights_, keepx)
    n_selected = int(mask.sum())
    ncomp1 = int(min(ncomp0, X_train_std.shape[0] - 2, n_selected))
    sparse = PLSRegression(n_components=ncomp1, scale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sparse.fit(X_train_std[:, mask], y_train)
    return sparse, mask, ncomp1, full.x_weights_


def select_splsda_params_cv(X_raw_fit: np.ndarray, y_fit: np.ndarray, out_dir: str) -> Tuple[int, int, pd.DataFrame, pd.DataFrame]:
    max_k = int(min(CONFIG["SPLSDA_MAX_COMPONENTS"], X_raw_fit.shape[0] - 2, X_raw_fit.shape[1]))
    keep_grid = [int(x) for x in CONFIG["KEEPX_GRID"]]
    folds = int(CONFIG["SPLSDA_CV_FOLDS"])
    repeats = int(CONFIG.get("SPLSDA_CV_REPEATS", 1))
    seed = int(CONFIG["SEED"])
    if repeats <= 1:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(X_raw_fit, y_fit.astype(int)))
    else:
        splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
        splits = list(splitter.split(X_raw_fit, y_fit.astype(int)))

    detail_rows = []
    summary_rows = []
    for ncomp in range(1, max_k + 1):
        for keepx in keep_grid:
            fold_rows = []
            for fold_id, (tr, va) in enumerate(splits, start=1):
                X_tr, X_va = standardize_train_valid(X_raw_fit[tr], X_raw_fit[va])
                try:
                    model, mask, ncomp_eff, _ = fit_sparse_plsda_once(X_tr, y_fit[tr], ncomp, keepx)
                    pred = model.predict(X_va[:, mask]).ravel()
                    m = evaluate_binary_predictions(y_fit[va], pred)
                    m.update({
                        "n_components": ncomp,
                        "keepX": keepx,
                        "fold_id": fold_id,
                        "n_selected_features": int(mask.sum()),
                        "effective_n_components": int(ncomp_eff),
                        "status": "ok",
                    })
                except Exception as e:
                    m = {
                        "n_components": ncomp,
                        "keepX": keepx,
                        "fold_id": fold_id,
                        "auc_raw": np.nan,
                        "auc_balanced": np.nan,
                        "accuracy": np.nan,
                        "balanced_accuracy": np.nan,
                        "BER": np.nan,
                        "sensitivity_mdd": np.nan,
                        "specificity_hc": np.nan,
                        "tn": np.nan, "fp": np.nan, "fn": np.nan, "tp": np.nan,
                        "n_selected_features": np.nan,
                        "effective_n_components": np.nan,
                        "status": f"error:{e}",
                    }
                fold_rows.append(m)
                detail_rows.append(m)
            fd = pd.DataFrame(fold_rows)
            summary_rows.append({
                "n_components": ncomp,
                "keepX": keepx,
                "mean_auc_balanced": fd["auc_balanced"].mean(),
                "std_auc_balanced": fd["auc_balanced"].std(ddof=0),
                "mean_accuracy": fd["accuracy"].mean(),
                "mean_balanced_accuracy": fd["balanced_accuracy"].mean(),
                "mean_BER": fd["BER"].mean(),
                "std_BER": fd["BER"].std(ddof=0),
                "mean_sensitivity_mdd": fd["sensitivity_mdd"].mean(),
                "mean_specificity_hc": fd["specificity_hc"].mean(),
                "mean_n_selected_features": fd["n_selected_features"].mean(),
                "n_cv_fits": int(fd.shape[0]),
                "n_ok": int((fd["status"] == "ok").sum()),
            })
            print(f"[CV] ncomp={ncomp}, keepX={keepx}: BER={summary_rows[-1]['mean_BER']:.4f}, AUC={summary_rows[-1]['mean_auc_balanced']:.4f}")

    detail_df = pd.DataFrame(detail_rows)
    summary_df = pd.DataFrame(summary_rows)
    save_df(detail_df, out_dir, "splsda_cv_detail.csv")
    save_df(summary_df, out_dir, "splsda_component_keepx_cv_summary.csv")

    select_by = str(CONFIG.get("SPLSDA_SELECT_BY", "ber")).lower()
    tol = float(CONFIG.get("SPLSDA_TIE_TOL", 0.01))
    valid = summary_df[summary_df["n_ok"] > 0].copy()
    if valid.empty:
        raise RuntimeError("All sPLS-DA CV fits failed.")
    if select_by == "auc":
        best_auc = valid["mean_auc_balanced"].max()
        cand = valid[valid["mean_auc_balanced"] >= best_auc - tol].copy()
        cand = cand.sort_values(["n_components", "keepX", "mean_BER"], ascending=[True, True, True])
        rule = f"max mean CV balanced AUC within tolerance {tol}; choose smaller n_components and keepX"
    else:
        best_ber = valid["mean_BER"].min()
        cand = valid[valid["mean_BER"] <= best_ber + tol].copy()
        cand = cand.sort_values(["n_components", "keepX", "mean_auc_balanced"], ascending=[True, True, False])
        rule = f"min mean CV BER within tolerance {tol}; choose smaller n_components and keepX"
    best = cand.iloc[0]
    best_ncomp, best_keepx = int(best["n_components"]), int(best["keepX"])
    best_df = pd.DataFrame({
        "selected_n_components": [best_ncomp],
        "selected_keepX": [best_keepx],
        "selection_rule": [rule],
        "selected_mean_CV_BER": [best["mean_BER"]],
        "selected_mean_CV_AUC_balanced": [best["mean_auc_balanced"]],
    })
    save_df(best_df, out_dir, "splsda_best_parameters.csv")
    return best_ncomp, best_keepx, summary_df, detail_df


def fit_final_splsda(X_raw: np.ndarray, groups: List[str], fit_mask: np.ndarray, out_dir: str):
    y_fit = binary_labels(groups, fit_mask)
    X_raw_fit = X_raw[fit_mask]
    best_ncomp, best_keepx, cv_summary, cv_detail = select_splsda_params_cv(X_raw_fit, y_fit, out_dir)
    X_fit_std, X_all_std = standardize_blocks_fit_all(X_raw, fit_mask)
    model, mask, effective_k, full_weights_initial = fit_sparse_plsda_once(X_fit_std, y_fit, best_ncomp, best_keepx)
    scores_fit = model.transform(X_fit_std[:, mask])
    scores_all = model.transform(X_all_std[:, mask])
    pred_fit = model.predict(X_fit_std[:, mask]).ravel()
    pred_all = model.predict(X_all_std[:, mask]).ravel()

    # Embed final sparse weights back to full feature space.
    weights_full = np.zeros((X_raw.shape[1], effective_k), dtype=float)
    weights_full[mask, :] = model.x_weights_[:, :effective_k]

    # Direction alignment: positive factor direction means MDD tends to be higher than HC.
    g_fit = np.array([str(x).lower() for x in np.array(groups)[fit_mask]])
    hc, mdd = CONFIG["HC_GROUP_NAME"].lower(), CONFIG["MDD_GROUP_NAME"].lower()
    for kk in range(effective_k):
        hc_mean = np.nanmean(scores_fit[g_fit == hc, kk])
        mdd_mean = np.nanmean(scores_fit[g_fit == mdd, kk])
        if np.isfinite(hc_mean) and np.isfinite(mdd_mean) and mdd_mean < hc_mean:
            scores_fit[:, kk] *= -1
            scores_all[:, kk] *= -1
            weights_full[:, kk] *= -1

    fit_metrics = evaluate_binary_predictions(y_fit, pred_fit)
    info = pd.DataFrame({
        "K_requested": [best_ncomp],
        "K_effective": [effective_k],
        "keepX": [best_keepx],
        "n_selected_union_features": [int(mask.sum())],
        "n_fit_samples": [int(fit_mask.sum())],
        "n_all_samples": [len(groups)],
        "n_features_total": [X_raw.shape[1]],
        "fit_auc_balanced": [fit_metrics["auc_balanced"]],
        "fit_BER": [fit_metrics["BER"]],
        "block_scaled": [bool(CONFIG.get("BLOCK_SCALE", True))],
        "standardized_blocks": [bool(CONFIG.get("STANDARDIZE_BLOCKS", True))],
    })
    save_df(info, out_dir, "splsda_model_info.csv")

    mask_df = pd.DataFrame({"feature_index": np.arange(X_raw.shape[1]), "selected_union_feature": mask.astype(int)})
    save_df(mask_df, out_dir, "splsda_selected_union_feature_mask.csv")
    return model, effective_k, best_keepx, mask, scores_all, scores_fit, weights_full, pred_all, pred_fit, X_fit_std, X_all_std, y_fit, cv_summary


# ============================================================
# Statistics, stability, selection
# ============================================================
def factor_statistics(scores_all: np.ndarray, groups: List[str], out_dir: str, prefix: str = "splsda") -> pd.DataFrame:
    g = np.array([str(x).lower() for x in groups])
    hc, mdd = CONFIG["HC_GROUP_NAME"].lower(), CONFIG["MDD_GROUP_NAME"].lower()
    y = (g == mdd).astype(int)
    rows = []
    for kk in range(scores_all.shape[1]):
        x_hc = scores_all[g == hc, kk]
        x_mdd = scores_all[g == mdd, kk]
        u, p = safe_mannwhitneyu(x_mdd, x_hc)
        d = cohen_d(x_mdd, x_hc)
        cd = cliffs_delta(x_mdd, x_hc)
        auc_raw = safe_auc(y, scores_all[:, kk])
        auc_bal = max(auc_raw, 1.0 - auc_raw) if np.isfinite(auc_raw) else np.nan
        rows.append({
            "factor": f"Factor_{kk+1}",
            "component_index": kk + 1,
            "mean_hc": float(np.nanmean(x_hc)),
            "mean_mdd": float(np.nanmean(x_mdd)),
            "median_hc": float(np.nanmedian(x_hc)),
            "median_mdd": float(np.nanmedian(x_mdd)),
            "mannwhitney_u": u,
            "p_value": p,
            "auc_raw_mdd_positive": auc_raw,
            "auc_balanced": auc_bal,
            "cohen_d_mdd_minus_hc": d,
            "cliffs_delta_mdd_vs_hc": cd,
            "abs_cliffs_delta": abs(cd) if np.isfinite(cd) else np.nan,
        })
    df = pd.DataFrame(rows)
    save_df(df, out_dir, f"{prefix}_factor_mdd_hc_statistics_pvalue_only.csv")
    return df


def make_loading_tables(weights: np.ndarray, feature_names: List[str], feature_types: List[str], out_dir: str, prefix: str = "splsda"):
    rows = []
    for kk in range(weights.shape[1]):
        for j in range(weights.shape[0]):
            rows.append({
                "factor": f"Factor_{kk+1}",
                "component_index": kk + 1,
                "feature_index": j,
                "feature_type": feature_types[j],
                "feature_name": feature_names[j],
                "loading": float(weights[j, kk]),
                "abs_loading": float(abs(weights[j, kk])),
                "nonzero_loading": bool(abs(weights[j, kk]) > 1e-12),
            })
    load_df = pd.DataFrame(rows)
    save_df(load_df, out_dir, f"{prefix}_node_weights_long.csv")

    n_rois = int(CONFIG["N_ROIS"])
    contrib_rows = []
    for kk in range(weights.shape[1]):
        L = np.abs(weights[:, kk])
        brain = float(L[:n_rois].sum())
        gut = float(L[n_rois:].sum())
        total = brain + gut + 1e-12
        contrib_rows.append({
            "factor": f"Factor_{kk+1}",
            "brain_abs_loading_sum": brain,
            "taxa_abs_loading_sum": gut,
            "brain_contribution_ratio": brain / total,
            "taxa_contribution_ratio": gut / total,
            "brain_gut_balance": min(brain, gut) / (max(brain, gut) + 1e-12),
            "n_nonzero_roi": int(np.sum(L[:n_rois] > 1e-12)),
            "n_nonzero_taxa": int(np.sum(L[n_rois:] > 1e-12)),
        })
    contrib_df = pd.DataFrame(contrib_rows)
    save_df(contrib_df, out_dir, f"{prefix}_brain_taxa_contribution_ratio.csv")

    wide = pd.DataFrame(weights, columns=[f"Factor_{i+1}" for i in range(weights.shape[1])])
    wide.insert(0, "feature_name", feature_names)
    wide.insert(0, "feature_type", feature_types)
    wide.insert(0, "feature_index", np.arange(weights.shape[0]))
    save_df(wide, out_dir, f"{prefix}_node_weights_wide.csv")
    return load_df, contrib_df


def bootstrap_stability(X_raw: np.ndarray, groups: List[str], fit_mask: np.ndarray, k: int, keepx: int, full_scores_fit: np.ndarray, full_weights: np.ndarray, out_dir: str):
    rng = np.random.default_rng(int(CONFIG["SEED"]) + k * 2027 + keepx)
    n_boot = int(CONFIG["BOOTSTRAP_N"])
    frac = float(CONFIG["BOOTSTRAP_SAMPLE_FRAC"])
    fit_idx = np.where(fit_mask)[0]
    y_fit_full = binary_labels(groups, fit_mask)
    n_fit = len(fit_idx)
    sample_size = max(k + 3, int(round(frac * n_fit)))
    top_n = int(CONFIG["TOP_FEATURE_OVERLAP_N"])

    score_corrs = np.full((n_boot, k), np.nan)
    loading_corrs = np.full((n_boot, k), np.nan)
    sign_consistency = np.full((n_boot, k), np.nan)
    top_overlap = np.full((n_boot, k), np.nan)

    full_top_sets = []
    for kk in range(k):
        full_top_sets.append(set(np.argsort(np.abs(full_weights[:, kk]))[::-1][:top_n].tolist()))

    for b in range(n_boot):
        boot_local = rng.choice(np.arange(n_fit), size=sample_size, replace=True)
        if len(np.unique(y_fit_full[boot_local])) < 2:
            continue
        boot_global = fit_idx[boot_local]
        X_boot_raw = X_raw[boot_global]
        X_fit_raw = X_raw[fit_idx]
        X_boot_std, X_fit_std = standardize_train_valid(X_boot_raw, X_fit_raw)
        try:
            model, mask, k_eff, _ = fit_sparse_plsda_once(X_boot_std, y_fit_full[boot_local], k, keepx)
            if k_eff < k:
                # compare available components only
                pass
            scores_b = model.transform(X_fit_std[:, mask])
            w_b_full = np.zeros_like(full_weights)
            w_b_full[mask, :k_eff] = model.x_weights_[:, :k_eff]
        except Exception:
            continue
        for kk in range(min(k, scores_b.shape[1], w_b_full.shape[1])):
            r = np.corrcoef(scores_b[:, kk], full_scores_fit[:, kk])[0, 1]
            if not np.isfinite(r):
                continue
            sign = 1.0 if r >= 0 else -1.0
            score_corrs[b, kk] = abs(r)
            sign_consistency[b, kk] = 1.0 if r >= 0 else 0.0
            w = w_b_full[:, kk] * sign
            lr = np.corrcoef(w, full_weights[:, kk])[0, 1]
            loading_corrs[b, kk] = abs(lr) if np.isfinite(lr) else np.nan
            top_b = set(np.argsort(np.abs(w))[::-1][:top_n].tolist())
            top_overlap[b, kk] = len(top_b & full_top_sets[kk]) / max(top_n, 1)
        if (b + 1) % 50 == 0 or (b + 1) == n_boot:
            print(f"[Bootstrap sPLS-DA] {b+1}/{n_boot}")

    rows = []
    for kk in range(k):
        rows.append({
            "factor": f"Factor_{kk+1}",
            "component_index": kk + 1,
            "bootstrap_n": n_boot,
            "score_stability_mean_abs_corr": float(np.nanmean(score_corrs[:, kk])),
            "score_stability_median_abs_corr": float(np.nanmedian(score_corrs[:, kk])),
            "loading_stability_mean_abs_corr": float(np.nanmean(loading_corrs[:, kk])),
            "loading_stability_median_abs_corr": float(np.nanmedian(loading_corrs[:, kk])),
            "direction_consistency_rate": float(np.nanmean(sign_consistency[:, kk])),
            "top_feature_overlap_mean": float(np.nanmean(top_overlap[:, kk])),
            "top_feature_overlap_median": float(np.nanmedian(top_overlap[:, kk])),
        })
    stab_df = pd.DataFrame(rows)
    save_df(stab_df, out_dir, "splsda_component_bootstrap_stability.csv")
    return stab_df


def select_factors(stats_df: pd.DataFrame, stability_df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """
    Select factors for downstream MDD subtype clustering using disease-relevance
    criteria only:
        auc_balanced >= MIN_AUC_BALANCED
        p_value <= MAX_P_VALUE
        abs(Cliff's delta) >= MIN_ABS_CLIFF_DELTA

    Bootstrap score stability is retained as an annotation column, but it is not
    used as a hard inclusion/exclusion criterion in this version.
    """
    df = stats_df.merge(stability_df, on=["factor", "component_index"], how="left")

    # Stability is kept for reporting/annotation only.
    df["score_stable_ok"] = df["score_stability_mean_abs_corr"] >= float(CONFIG["MIN_SCORE_STABILITY"])

    # Main disease-relevance rule for factor inclusion.
    df["auc_ok"] = df["auc_balanced"] >= float(CONFIG["MIN_AUC_BALANCED"])
    df["p_value_ok"] = df["p_value"] <= float(CONFIG["MAX_P_VALUE"])
    df["cliff_delta_ok"] = df["cliffs_delta_mdd_vs_hc"].abs() >= float(CONFIG["MIN_ABS_CLIFF_DELTA"])
    df["disease_relevance_ok"] = df["auc_ok"] & df["p_value_ok"] & df["cliff_delta_ok"]
    df["selected_for_subtype"] = df["disease_relevance_ok"]

    # Keep strict columns for compatibility with downstream plotting/output files.
    df["strict_score_stable_ok"] = df["score_stability_mean_abs_corr"] >= float(CONFIG["STRICT_MIN_SCORE_STABILITY"])
    df["strict_disease_ok"] = (
        (df["auc_balanced"] >= float(CONFIG["STRICT_MIN_AUC_BALANCED"]))
        & (df["p_value"] <= float(CONFIG["MAX_P_VALUE"]))
        & (df["cliffs_delta_mdd_vs_hc"].abs() >= float(CONFIG["STRICT_MIN_ABS_CLIFF_DELTA"]))
    )
    df["strict_selected_for_interpretation"] = df["strict_score_stable_ok"] & df["strict_disease_ok"]

    df["selection_note"] = np.where(
        df["selected_for_subtype"],
        "selected_by_auc070_p005_cliff050",
        "not_selected_by_auc070_p005_cliff050"
    )

    # Keep only the first N qualified factors for subtype clustering.
    # "First" means smaller component_index first, e.g., Factor_1 then Factor_2.
    max_sel = int(CONFIG.get("MAX_SELECTED_FACTORS_FOR_CLUSTER", 2))
    selected_idx = df.index[df["selected_for_subtype"]].tolist()
    if len(selected_idx) > max_sel:
        keep_idx = df.loc[selected_idx].sort_values("component_index").head(max_sel).index
        drop_idx = [idx for idx in selected_idx if idx not in set(keep_idx)]
        df.loc[drop_idx, "selected_for_subtype"] = False
        df.loc[drop_idx, "selection_note"] = "passed_threshold_but_not_used_limit_first2"
        df.loc[keep_idx, "selection_note"] = "selected_by_auc070_p005_cliff050_first2"

    # No fallback selection is used in this version. If no factor passes the rule,
    # clustering will stop with an explicit error in run_one_source().
    selected_n = int(df["selected_for_subtype"].sum())
    if selected_n == 0:
        print(
            "[WARN] No sPLS-DA factor passed the disease-relevance selection rule: "
            f"AUC >= {CONFIG['MIN_AUC_BALANCED']}, "
            f"p <= {CONFIG['MAX_P_VALUE']}, "
            f"|Cliff delta| >= {CONFIG['MIN_ABS_CLIFF_DELTA']}."
        )

    save_df(df, out_dir, "splsda_factor_selection_summary_pvalue_only.csv")
    save_df(df[df["selected_for_subtype"]].copy(), out_dir, "splsda_selected_factors_for_subtype.csv")
    counts = pd.DataFrame({
        "category": ["total_factors", "selected_for_subtype", "strict_selected_for_interpretation"],
        "count": [int(df.shape[0]), selected_n, int(df["strict_selected_for_interpretation"].sum())],
    })
    save_df(counts, out_dir, "splsda_selected_factor_counts.csv")
    return df


# ============================================================
# K-means subtype analysis
# ============================================================
def bootstrap_kmeans_stability(X, k, base_labels, seed):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    frac = float(CONFIG["KMEANS_BOOTSTRAP_FRAC"])
    n_boot = int(CONFIG["KMEANS_BOOTSTRAP_N"])
    sample_size = max(k + 2, int(round(frac * n)))
    aris = []
    for i in range(n_boot):
        idx = np.sort(rng.choice(n, size=sample_size, replace=False))
        km = KMeans(n_clusters=k, n_init=int(CONFIG["KMEANS_N_INIT"]), random_state=seed + i + 11)
        lab = km.fit_predict(X[idx])
        aris.append(adjusted_rand_score(base_labels[idx], lab))
    return float(np.nanmean(aris)), float(np.nanstd(aris))


def evaluate_kmeans_and_select(scores_mdd: np.ndarray, sids_mdd: List[str], out_dir: str):
    k_min = int(CONFIG["KMEANS_K_MIN"])
    k_max = int(CONFIG["KMEANS_K_MAX"])
    seed = int(CONFIG["SEED"])
    n = scores_mdd.shape[0]
    rows = []
    labels_by_k = {}
    for k in range(k_min, min(k_max, n - 1) + 1):
        km = KMeans(n_clusters=k, n_init=int(CONFIG["KMEANS_N_INIT"]), random_state=seed)
        labels = km.fit_predict(scores_mdd)
        labels_by_k[k] = labels
        counts = pd.Series(labels).value_counts().sort_index()
        min_size = int(counts.min())
        min_ratio = float(min_size / n)
        valid = bool(min_size >= int(CONFIG["MIN_CLUSTER_SIZE"]) and min_ratio >= float(CONFIG["MIN_CLUSTER_RATIO"]))
        try:
            sil = silhouette_score(scores_mdd, labels)
        except Exception:
            sil = np.nan
        try:
            ch = calinski_harabasz_score(scores_mdd, labels)
        except Exception:
            ch = np.nan
        try:
            db = davies_bouldin_score(scores_mdd, labels)
        except Exception:
            db = np.nan
        ari_mean, ari_std = bootstrap_kmeans_stability(scores_mdd, k, labels, seed + 1000 * k)
        rows.append({
            "K": k,
            "silhouette": sil,
            "calinski_harabasz": ch,
            "davies_bouldin": db,
            "bootstrap_ari_mean": ari_mean,
            "bootstrap_ari_std": ari_std,
            "min_cluster_size": min_size,
            "min_cluster_ratio": min_ratio,
            "valid_cluster_size": valid,
        })
    df = pd.DataFrame(rows)
    df["rank_silhouette"] = rank_normalize_metric(df["silhouette"], True)
    df["rank_calinski_harabasz"] = rank_normalize_metric(df["calinski_harabasz"], True)
    df["rank_davies_bouldin"] = rank_normalize_metric(df["davies_bouldin"], False)
    df["rank_bootstrap_ari"] = rank_normalize_metric(df["bootstrap_ari_mean"], True)
    df["composite_score"] = df["rank_silhouette"] + df["rank_calinski_harabasz"] + df["rank_davies_bouldin"] + df["rank_bootstrap_ari"]
    df.loc[~df["valid_cluster_size"], "composite_score"] = -np.inf
    best_k = int(df.sort_values("composite_score", ascending=False).iloc[0]["K"])
    save_df(df.replace([np.inf, -np.inf], np.nan), out_dir, "mdd_kmeans_K2_to_K6_evaluation.csv")
    save_df(pd.DataFrame({"selected_K": [best_k], "criterion": ["ranked silhouette + CH - DB + bootstrap ARI, with minimum cluster size"]}), out_dir, "mdd_kmeans_selected_K.csv")
    labels = labels_by_k[best_k]
    subtype = np.array([f"Subtype_{x+1}" for x in labels])
    sub_df = pd.DataFrame({"sample_id": sids_mdd, "kmeans_label_zero_based": labels, "subtype": subtype})
    save_df(sub_df, out_dir, "mdd_splsda_kmeans_subtype_assignments.csv")
    return best_k, labels, subtype, df


# ============================================================
# Clinical and subtype analyses
# ============================================================
def load_clinical_table(sids: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    path = CONFIG.get("CLINICAL_CSV", "")
    sc = CONFIG.get("CLINICAL_SAMPLE_COL", "sample_id")
    if path is None or str(path).strip() == "" or not os.path.exists(path):
        print(f"[SKIP] Clinical file not found: {path}")
        return pd.DataFrame(), []
    df = pd.read_csv(path)
    if sc not in df.columns:
        raise ValueError(f"CLINICAL_CSV must contain sample id column: {sc}")
    df[sc] = df[sc].astype(str)
    if CONFIG.get("CLINICAL_COLUMNS"):
        cols = [c for c in CONFIG["CLINICAL_COLUMNS"] if c in df.columns]
    else:
        cols = []
        for c in df.columns:
            if c == sc:
                continue
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().sum() > 0:
                df[c] = vals
                cols.append(c)
    df = df[df[sc].isin([str(s) for s in sids])].copy()
    return df, cols


def clinical_analysis(scores, sids, groups, subtype_map, out_dir):
    clinical_df, clinical_cols = load_clinical_table(sids)
    if clinical_df.empty or not clinical_cols:
        return {"merged": pd.DataFrame(), "mdd": pd.DataFrame(), "subtype_clinical": pd.DataFrame(), "factor_clinical": pd.DataFrame(), "clinical_cols": []}

    sc = CONFIG["CLINICAL_SAMPLE_COL"]
    score_df = pd.DataFrame(scores, columns=[f"Factor_{i+1}" for i in range(scores.shape[1])])
    score_df.insert(0, "group", [str(g).lower() for g in groups])
    score_df.insert(0, "sample_id", [str(s) for s in sids])
    score_df["subtype"] = score_df["sample_id"].map(subtype_map)
    merged = score_df.merge(clinical_df.rename(columns={sc: "sample_id"}), on="sample_id", how="left")
    save_df(merged, out_dir, "splsda_scores_with_clinical_and_subtype.csv")

    mdd = CONFIG["MDD_GROUP_NAME"].lower()
    mdd_df = merged[merged["group"] == mdd].copy()
    subtypes = sorted([x for x in mdd_df["subtype"].dropna().unique()])

    sub_rows = []
    for c in clinical_cols:
        arrays = [mdd_df.loc[mdd_df["subtype"] == st, c].dropna().values for st in subtypes]
        arrays_valid = [a for a in arrays if len(a) > 0]
        if len(arrays_valid) >= 2:
            h, p = stats.kruskal(*arrays_valid)
        else:
            h, p = np.nan, np.nan
        row = {"clinical_variable": c, "kruskal_H": h, "p_value": p}
        for st in subtypes:
            x = mdd_df.loc[mdd_df["subtype"] == st, c].dropna().values
            row[f"mean_{st}"] = float(np.nanmean(x)) if len(x) else np.nan
            row[f"median_{st}"] = float(np.nanmedian(x)) if len(x) else np.nan
            row[f"n_{st}"] = int(len(x))
        sub_rows.append(row)
    subclin_df = pd.DataFrame(sub_rows)
    save_df(subclin_df, out_dir, "mdd_subtype_clinical_kruskal_tests_pvalue_only.csv")

    corr_rows = []
    factor_cols = [f"Factor_{i+1}" for i in range(scores.shape[1])]
    for f in factor_cols:
        for c in clinical_cols:
            tmp = mdd_df[[f, c]].dropna()
            if len(tmp) >= 4:
                r, p = stats.spearmanr(tmp[f].values, tmp[c].values)
            else:
                r, p = np.nan, np.nan
            corr_rows.append({"factor": f, "clinical_variable": c, "phenotype": map_phenotype(c), "spearman_r": r, "p_value": p, "n": int(len(tmp))})
    corr_df = pd.DataFrame(corr_rows)
    save_df(corr_df, out_dir, "splsda_factor_clinical_spearman_mdd_pvalue_only.csv")
    return {"merged": merged, "mdd": mdd_df, "subtype_clinical": subclin_df, "factor_clinical": corr_df, "clinical_cols": clinical_cols}


def subtype_vs_hc_node_differences(X_std, sids, groups, subtype_map, feature_names, feature_types, out_dir):
    g = np.array([str(x).lower() for x in groups])
    hc = CONFIG["HC_GROUP_NAME"].lower()
    subtypes = sorted(set([v for v in subtype_map.values() if isinstance(v, str)]))
    rows = []
    for st in subtypes:
        st_mask = np.array([subtype_map.get(str(s), None) == st for s in sids])
        hc_mask = g == hc
        if st_mask.sum() < 2 or hc_mask.sum() < 2:
            continue
        for j in range(X_std.shape[1]):
            x_st = X_std[st_mask, j]
            x_hc = X_std[hc_mask, j]
            u, p = safe_mannwhitneyu(x_st, x_hc)
            cd = cliffs_delta(x_st, x_hc)
            rows.append({
                "subtype": st,
                "feature_index": j,
                "feature_type": feature_types[j],
                "feature_name": feature_names[j],
                "mean_subtype": float(np.nanmean(x_st)),
                "mean_hc": float(np.nanmean(x_hc)),
                "mean_diff_subtype_minus_hc": float(np.nanmean(x_st) - np.nanmean(x_hc)),
                "p_value": p,
                "cliffs_delta_subtype_vs_hc": cd,
                "abs_cliffs_delta": abs(cd) if np.isfinite(cd) else np.nan,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["rank_abs_effect_within_subtype"] = df.groupby("subtype")["abs_cliffs_delta"].rank(ascending=False, method="min")
    save_df(df, out_dir, "mdd_subtype_vs_hc_node_differences_pvalue_only.csv")
    return df


# ============================================================
# Plotting functions
# ============================================================
def plot_splsda_cv(summary_df, best_ncomp, best_keepx, out_dir):
    if summary_df is None or summary_df.empty:
        return
    save_plot_data(summary_df, out_dir, "plot_data_splsda_component_keepx_cv_summary.csv")

    # For each component, show best BER across keepX.
    best_by_k = summary_df.sort_values("mean_BER").groupby("n_components", as_index=False).first()
    plt.figure(figsize=(7, 5))
    plt.errorbar(best_by_k["n_components"], best_by_k["mean_BER"], yerr=best_by_k["std_BER"], marker="o", label="Best CV BER over keepX")
    plt.axvline(best_ncomp, linestyle="--", label=f"Selected K={best_ncomp}, keepX={best_keepx}")
    plt.xlabel("Number of sPLS-DA components")
    plt.ylabel("Balanced error rate")
    plt.title("sPLS-DA component selection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "splsda_component_selection_cv_BER.png"), dpi=int(CONFIG["FIG_DPI"]))
    plt.close()

    best_by_k_auc = summary_df.sort_values("mean_auc_balanced", ascending=False).groupby("n_components", as_index=False).first()
    plt.figure(figsize=(7, 5))
    plt.plot(best_by_k_auc["n_components"], best_by_k_auc["mean_auc_balanced"], marker="o", label="Best CV balanced AUC over keepX")
    plt.axvline(best_ncomp, linestyle="--", label=f"Selected K={best_ncomp}, keepX={best_keepx}")
    plt.xlabel("Number of sPLS-DA components")
    plt.ylabel("Balanced AUC")
    plt.title("sPLS-DA component selection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "splsda_component_selection_cv_AUC.png"), dpi=int(CONFIG["FIG_DPI"]))
    plt.close()

    # Heatmap: ncomp x keepX BER.
    pivot = summary_df.pivot(index="n_components", columns="keepX", values="mean_BER")
    pivot.to_csv(os.path.join(out_dir, "plot_data_splsda_ncomp_keepx_cv_BER_heatmap.csv"), encoding="utf-8-sig")
    plt.figure(figsize=(max(8, pivot.shape[1] * 0.55), max(4, pivot.shape[0] * 0.45)))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label="Mean CV BER")
    plt.xticks(np.arange(pivot.shape[1]), pivot.columns, rotation=45, ha="right")
    plt.yticks(np.arange(pivot.shape[0]), pivot.index)
    plt.xlabel("keepX")
    plt.ylabel("n_components")
    plt.title("sPLS-DA CV BER heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "splsda_ncomp_keepx_cv_BER_heatmap.png"), dpi=int(CONFIG["FIG_DPI"]), bbox_inches="tight")
    plt.close()


def plot_factor_boxplots(scores, groups, stats_df, out_dir):
    g = np.array([str(x).lower() for x in groups])
    hc, mdd = CONFIG["HC_GROUP_NAME"].lower(), CONFIG["MDD_GROUP_NAME"].lower()
    rows = []
    for kk in range(scores.shape[1]):
        data = [scores[g == hc, kk], scores[g == mdd, kk]]
        for val in data[0]:
            rows.append({"factor": f"Factor_{kk+1}", "group": hc, "score": float(val)})
        for val in data[1]:
            rows.append({"factor": f"Factor_{kk+1}", "group": mdd, "score": float(val)})
        row = stats_df[stats_df["component_index"] == kk + 1].iloc[0]
        plt.figure(figsize=(5, 5))
        plt.boxplot(data, labels=["HC", "MDD"], showfliers=False)
        rng = np.random.default_rng(int(CONFIG["SEED"]) + kk)
        for i, vals in enumerate(data, start=1):
            plt.scatter(np.ones(len(vals)) * i + rng.normal(0, 0.04, len(vals)), vals, s=18, alpha=0.65)
        plt.ylabel("sPLS-DA factor score")
        plt.title(f"Factor {kk+1}: MDD vs HC\nAUC={row['auc_balanced']:.3f}, p={row['p_value']:.3g}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"splsda_factor_{kk+1:02d}_mdd_hc_boxplot.png"), dpi=int(CONFIG["FIG_DPI"]))
        plt.close()
    save_plot_data(pd.DataFrame(rows), out_dir, "plot_data_splsda_factor_boxplots.csv")


def plot_sample_factor_heatmap(scores, sids, groups, out_dir):
    g = np.array([str(x).lower() for x in groups])
    order_map = {CONFIG["HC_GROUP_NAME"].lower(): 0, CONFIG["MDD_GROUP_NAME"].lower(): 1}
    order = np.lexsort((np.arange(len(groups)), np.array([order_map.get(x, 99) for x in g])))
    mat = scores[order]
    row_labels = [f"{groups[i]}:{sids[i]}" for i in order]
    df = pd.DataFrame(mat, columns=[f"Factor_{i+1}" for i in range(scores.shape[1])])
    df.insert(0, "row_label", row_labels)
    save_plot_data(df, out_dir, "plot_data_sample_by_factor_score_heatmap.csv")
    plt.figure(figsize=(max(5, scores.shape[1] * 0.8), max(6, len(sids) * 0.035)))
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="sPLS-DA factor score")
    plt.xticks(np.arange(scores.shape[1]), [f"F{i+1}" for i in range(scores.shape[1])])
    step = max(1, len(row_labels) // 30)
    yticks = np.arange(0, len(row_labels), step)
    plt.yticks(yticks, [row_labels[i] for i in yticks], fontsize=5)
    plt.title("Sample × sPLS-DA factor score heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sample_by_splsda_factor_score_heatmap.png"), dpi=int(CONFIG["FIG_DPI"]), bbox_inches="tight")
    plt.close()


def plot_disease_heatmaps(stats_df, out_dir):
    df = stats_df.copy()
    df["minus_log10_p"] = -np.log10(df["p_value"].clip(lower=1e-300))
    df["effect_abs_cliff"] = df["cliffs_delta_mdd_vs_hc"].abs()
    save_plot_data(df[["factor", "auc_balanced", "p_value", "minus_log10_p", "cliffs_delta_mdd_vs_hc", "effect_abs_cliff"]], out_dir, "plot_data_disease_association_metrics.csv")

    metrics = [
        ("auc_balanced", "Disease association heatmap: balanced AUC", "splsda_disease_association_auc_heatmap.png", "AUC"),
        ("minus_log10_p", "Disease association heatmap: -log10(p)", "splsda_disease_association_pvalue_heatmap.png", "-log10(p)"),
        ("cliffs_delta_mdd_vs_hc", "Disease association heatmap: signed Cliff's delta", "splsda_disease_association_effect_heatmap.png", "Cliff's delta"),
    ]
    for col, title, fname, cbar in metrics:
        mat = df[[col]].T.values
        plt.figure(figsize=(max(6, len(df) * 0.8), 2.6))
        plt.imshow(mat, aspect="auto")
        plt.colorbar(label=cbar)
        plt.xticks(np.arange(len(df)), df["factor"], rotation=45, ha="right")
        plt.yticks([0], [col])
        plt.title(title)
        for j, val in enumerate(df[col].values):
            if np.isfinite(val):
                plt.text(j, 0, f"{val:.2f}", ha="center", va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=int(CONFIG["FIG_DPI"]), bbox_inches="tight")
        plt.close()


def plot_stability_and_selection(selection_df, out_dir):
    if selection_df.empty:
        return
    save_plot_data(selection_df, out_dir, "plot_data_splsda_factor_stability_and_selection.csv")
    factors = selection_df["factor"].tolist()
    x = np.arange(len(factors))
    width = 0.22
    plt.figure(figsize=(max(8, len(factors) * 0.8), 5))
    plt.bar(x - width, selection_df["score_stability_mean_abs_corr"], width, label="Score stability")
    plt.bar(x, selection_df["auc_balanced"], width, label="AUC balanced")
    plt.bar(x + width, selection_df["cliffs_delta_mdd_vs_hc"].abs(), width, label="|Cliff delta|")
    plt.xticks(x, factors, rotation=45, ha="right")
    plt.ylabel("Metric")
    plt.title("sPLS-DA component stability and disease relevance")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "splsda_component_stability_and_selection_metrics.png"), dpi=int(CONFIG["FIG_DPI"]))
    plt.close()

    crit_cols = ["score_stable_ok", "disease_relevance_ok", "selected_for_subtype", "strict_score_stable_ok", "strict_disease_ok", "strict_selected_for_interpretation"]
    mat = selection_df[crit_cols].astype(int).to_numpy()
    plt.figure(figsize=(10, max(3.5, 0.4 * len(factors) + 1)))
    plt.imshow(mat, aspect="auto", vmin=0, vmax=1)
    plt.yticks(np.arange(len(factors)), factors)
    plt.xticks(np.arange(len(crit_cols)), crit_cols, rotation=45, ha="right")
    plt.colorbar(label="0 = no, 1 = yes")
    plt.title("sPLS-DA factor selection criteria")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "splsda_factor_selection_criteria_heatmap.png"), dpi=int(CONFIG["FIG_DPI"]), bbox_inches="tight")
    plt.close()

    counts = pd.DataFrame({
        "category": ["total_factors", "selected_for_subtype", "strict_selected_for_interpretation"],
        "count": [int(selection_df.shape[0]), int(selection_df["selected_for_subtype"].sum()), int(selection_df["strict_selected_for_interpretation"].sum())],
    })
    save_plot_data(counts, out_dir, "plot_data_splsda_selected_factor_counts.csv")
    plt.figure(figsize=(6, 4))
    plt.bar(counts["category"], counts["count"])
    plt.ylabel("Count")
    plt.title("Final selected sPLS-DA component counts")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "splsda_selected_factor_counts.png"), dpi=int(CONFIG["FIG_DPI"]))
    plt.close()


def plot_top_loading_bars(load_df, feature_type, out_dir):
    top_n = int(CONFIG["TOP_ROI_PER_FACTOR"] if feature_type == "ROI" else CONFIG["TOP_TAXA_PER_FACTOR"])
    for factor in sorted(load_df["factor"].unique(), key=lambda x: int(x.split("_")[1])):
        sub = load_df[(load_df["factor"] == factor) & (load_df["feature_type"] == feature_type)].copy()
        sub = sub.sort_values("abs_loading", ascending=False).head(top_n)
        if sub.empty:
            continue
        save_plot_data(sub, out_dir, f"plot_data_splsda_{factor}_{feature_type}_top_loading_bar.csv")
        sub["feature_short"] = sub["feature_name"].astype(str).str.slice(0, 60)
        sub = sub.sort_values("loading")
        plt.figure(figsize=(8, max(4, 0.35 * len(sub) + 1)))
        plt.barh(sub["feature_short"], sub["loading"])
        plt.xlabel("sPLS-DA weight")
        plt.title(f"Top {feature_type} weights - {factor}")
        plt.tight_layout()
        fn = f"splsda_{factor}_{feature_type}_top_loading_bar.png"
        plt.savefig(os.path.join(out_dir, fn), dpi=int(CONFIG["FIG_DPI"]), bbox_inches="tight")
        plt.close()


def plot_node_factor_heatmaps(load_df, feature_type, out_dir):
    top_union = int(CONFIG["TOP_NODE_HEATMAP_UNION"])
    sub = load_df[load_df["feature_type"] == feature_type].copy()
    if sub.empty:
        return
    top_features = []
    for factor in sub["factor"].unique():
        tmp = sub[sub["factor"] == factor].sort_values("abs_loading", ascending=False).head(top_union)
        top_features.extend(tmp["feature_name"].tolist())
    top_features = list(dict.fromkeys(top_features))
    mat = sub[sub["feature_name"].isin(top_features)].pivot(index="feature_name", columns="factor", values="loading").fillna(0.0)
    mat = mat[[f"Factor_{i}" for i in range(1, len(mat.columns) + 1) if f"Factor_{i}" in mat.columns]]
    mat.to_csv(os.path.join(out_dir, f"plot_data_splsda_{feature_type}_node_factor_heatmap.csv"), encoding="utf-8-sig")
    plt.figure(figsize=(max(6, mat.shape[1] * 0.8), max(5, mat.shape[0] * 0.25)))
    plt.imshow(mat.values, aspect="auto")
    plt.colorbar(label="sPLS-DA weight")
    plt.xticks(np.arange(mat.shape[1]), mat.columns, rotation=45, ha="right")
    plt.yticks(np.arange(mat.shape[0]), [str(x)[:50] for x in mat.index], fontsize=6)
    plt.title(f"{feature_type} node × factor loading heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"splsda_{feature_type}_node_factor_heatmap.png"), dpi=int(CONFIG["FIG_DPI"]), bbox_inches="tight")
    plt.close()


def plot_bipartite_chord_like_network(load_df, out_dir):
    top_roi = int(CONFIG["BIPARTITE_TOP_ROI"])
    top_taxa = int(CONFIG["BIPARTITE_TOP_TAXA"])
    for factor in sorted(load_df["factor"].unique(), key=lambda x: int(x.split("_")[1])):
        roi = load_df[(load_df["factor"] == factor) & (load_df["feature_type"] == "ROI")].sort_values("abs_loading", ascending=False).head(top_roi)
        taxa = load_df[(load_df["factor"] == factor) & (load_df["feature_type"] == "Taxa")].sort_values("abs_loading", ascending=False).head(top_taxa)
        if roi.empty or taxa.empty:
            continue
        edges = []
        for _, r in roi.iterrows():
            for _, t in taxa.iterrows():
                w = float(r["loading"] * t["loading"])
                edges.append({
                    "factor": factor,
                    "roi": r["feature_name"],
                    "taxa": t["feature_name"],
                    "edge_weight_outer_product": w,
                    "abs_edge_weight": abs(w),
                })
        edge_df = pd.DataFrame(edges).sort_values("abs_edge_weight", ascending=False)
        save_plot_data(edge_df, out_dir, f"plot_data_splsda_{factor}_brain_gut_chord_like_edges.csv")

        G = nx.Graph()
        for name in roi["feature_name"]:
            G.add_node(f"ROI:{name}", bipartite=0)
        for name in taxa["feature_name"]:
            G.add_node(f"Taxa:{name}", bipartite=1)
        # Keep top edges for readability.
        for _, e in edge_df.head(min(60, len(edge_df))).iterrows():
            G.add_edge(f"ROI:{e['roi']}", f"Taxa:{e['taxa']}", weight=e["abs_edge_weight"])
        pos = {}
        roi_nodes = [n for n in G.nodes if n.startswith("ROI:")]
        taxa_nodes = [n for n in G.nodes if n.startswith("Taxa:")]
        for i, n in enumerate(roi_nodes):
            pos[n] = (-1, i / max(len(roi_nodes) - 1, 1))
        for i, n in enumerate(taxa_nodes):
            pos[n] = (1, i / max(len(taxa_nodes) - 1, 1))
        widths = [max(0.5, 4 * G[u][v]["weight"] / max(edge_df["abs_edge_weight"].max(), 1e-12)) for u, v in G.edges]
        plt.figure(figsize=(10, max(5, 0.35 * max(len(roi_nodes), len(taxa_nodes)))))
        nx.draw_networkx_edges(G, pos, width=widths, alpha=0.45)
        nx.draw_networkx_nodes(G, pos, nodelist=roi_nodes, node_size=220)
        nx.draw_networkx_nodes(G, pos, nodelist=taxa_nodes, node_size=220)
        labels = {n: n.split(":", 1)[1][:28] for n in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=7)
        plt.axis("off")
        plt.title(f"Brain-gut chord-like network - {factor}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"splsda_{factor}_brain_gut_chord_like_network.png"), dpi=int(CONFIG["FIG_DPI"]), bbox_inches="tight")
        plt.close()


def plot_factor_space(scores, groups, sids, subtype_map, out_dir):
    if scores.shape[1] < 2:
        return
    g = np.array([str(x).lower() for x in groups])
    hc = CONFIG["HC_GROUP_NAME"].lower()
    plt.figure(figsize=(7, 6))
    idx_hc = g == hc
    plt.scatter(scores[idx_hc, 0], scores[idx_hc, 1], s=28, alpha=0.55, label="HC")
    subtypes = sorted(set([v for v in subtype_map.values() if isinstance(v, str)]))
    for st in subtypes:
        idx = np.array([subtype_map.get(str(s), None) == st for s in sids])
        plt.scatter(scores[idx, 0], scores[idx, 1], s=35, alpha=0.85, label=st)
    plt.xlabel("sPLS-DA Factor 1")
    plt.ylabel("sPLS-DA Factor 2")
    plt.title("sPLS-DA factor space")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "splsda_factor_space_F1_F2.png"), dpi=int(CONFIG["FIG_DPI"]))
    plt.close()


def plot_k_metrics(k_eval_df, out_dir):
    if k_eval_df.empty:
        return
    save_plot_data(k_eval_df, out_dir, "plot_data_mdd_kmeans_K_selection_metrics.csv")
    plt.figure(figsize=(8, 5))
    for col in ["rank_silhouette", "rank_calinski_harabasz", "rank_davies_bouldin", "rank_bootstrap_ari"]:
        if col in k_eval_df.columns:
            plt.plot(k_eval_df["K"], k_eval_df[col], marker="o", label=col.replace("rank_", ""))
    plt.plot(k_eval_df["K"], k_eval_df["composite_score"] / 4.0, marker="s", linewidth=2, label="composite / 4")
    plt.xlabel("K")
    plt.ylabel("Rank-normalized score")
    plt.title("MDD subtype K selection")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mdd_splsda_kmeans_K_selection_metrics.png"), dpi=int(CONFIG["FIG_DPI"]))
    plt.close()


def plot_mdd_dendrogram(scores_mdd_cluster, sids_mdd, subtype_mdd, out_dir):
    if scores_mdd_cluster.shape[0] < 3:
        return
    Z = linkage(scores_mdd_cluster, method="ward")
    labels = [f"{subtype_mdd[i]}:{sids_mdd[i]}" for i in range(len(sids_mdd))]
    pd.DataFrame(Z, columns=["idx1", "idx2", "distance", "sample_count"]).to_csv(os.path.join(out_dir, "plot_data_mdd_splsda_factor_hierarchical_linkage.csv"), index=False, encoding="utf-8-sig")
    plt.figure(figsize=(max(10, len(sids_mdd) * 0.12), 6))
    dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=5)
    plt.title("MDD sPLS-DA factor hierarchical dendrogram")
    plt.ylabel("Ward distance")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mdd_splsda_factor_hierarchical_dendrogram.png"), dpi=int(CONFIG["FIG_DPI"]), bbox_inches="tight")
    plt.close()


def plot_mdd_subtype_factor_heatmap(scores_mdd, sids_mdd, subtype_mdd, out_dir):
    order = np.argsort(subtype_mdd)
    mat = scores_mdd[order]
    row_labels = [f"{subtype_mdd[i]}:{sids_mdd[i]}" for i in order]
    df = pd.DataFrame(mat, columns=[f"Factor_{i+1}" for i in range(scores_mdd.shape[1])])
    df.insert(0, "row_label", row_labels)
    save_plot_data(df, out_dir, "plot_data_mdd_splsda_factor_score_heatmap_by_subtype.csv")
    plt.figure(figsize=(max(5, scores_mdd.shape[1] * 0.8), max(6, len(sids_mdd) * 0.045)))
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="sPLS-DA factor score")
    plt.xticks(np.arange(scores_mdd.shape[1]), [f"F{i+1}" for i in range(scores_mdd.shape[1])])
    step = max(1, len(row_labels) // 30)
    yticks = np.arange(0, len(row_labels), step)
    plt.yticks(yticks, [row_labels[i] for i in yticks], fontsize=5)
    plt.title("MDD subtype × sPLS-DA factor score heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mdd_splsda_factor_score_heatmap_by_subtype.png"), dpi=int(CONFIG["FIG_DPI"]), bbox_inches="tight")
    plt.close()


def plot_clinical_heatmap(clinical_outputs, out_dir):
    corr = clinical_outputs.get("factor_clinical", pd.DataFrame())
    if corr is None or corr.empty:
        return
    mat = corr.pivot(index="clinical_variable", columns="factor", values="spearman_r")
    mat.to_csv(os.path.join(out_dir, "plot_data_splsda_factor_clinical_spearman_heatmap.csv"), encoding="utf-8-sig")
    plt.figure(figsize=(max(6, mat.shape[1] * 0.8), max(4, mat.shape[0] * 0.35)))
    plt.imshow(mat.values, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(label="Spearman r")
    plt.xticks(np.arange(mat.shape[1]), mat.columns, rotation=45, ha="right")
    plt.yticks(np.arange(mat.shape[0]), mat.index)
    plt.title("MDD clinical correlations with sPLS-DA factors")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "splsda_factor_clinical_spearman_heatmap.png"), dpi=int(CONFIG["FIG_DPI"]), bbox_inches="tight")
    plt.close()


def plot_phenotype_annotation(corr_df, out_dir):
    if corr_df is None or corr_df.empty:
        return
    sig = corr_df.copy()
    sig["minus_log10_p"] = -np.log10(sig["p_value"].clip(lower=1e-300))
    sig["significant_p"] = sig["p_value"] <= float(CONFIG["CLINICAL_P_THRESHOLD"])
    save_plot_data(sig, out_dir, "plot_data_factor_phenotype_annotation_barplot.csv")
    top = sig.sort_values("minus_log10_p", ascending=False).head(40)
    if top.empty:
        return
    top["label"] = top["factor"] + " | " + top["clinical_variable"].astype(str).str.slice(0, 40)
    top = top.sort_values("minus_log10_p")
    plt.figure(figsize=(10, max(5, 0.25 * len(top) + 1)))
    plt.barh(top["label"], top["minus_log10_p"])
    plt.axvline(-np.log10(float(CONFIG["CLINICAL_P_THRESHOLD"])), linestyle="--", label="p threshold")
    plt.xlabel("-log10(p)")
    plt.title("Phenotype annotation of sPLS-DA factors")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "factor_phenotype_annotation_barplot.png"), dpi=int(CONFIG["FIG_DPI"]), bbox_inches="tight")
    plt.close()


def plot_significant_clinical_scatter_and_bars(clinical_outputs, out_dir):
    merged = clinical_outputs.get("merged", pd.DataFrame())
    corr = clinical_outputs.get("factor_clinical", pd.DataFrame())
    if merged is None or merged.empty or corr is None or corr.empty:
        return
    sig = corr[(corr["p_value"] <= float(CONFIG["CLINICAL_P_THRESHOLD"])) & (corr["spearman_r"].abs() >= float(CONFIG["MIN_ABS_SPEARMAN_R"]))].copy()
    save_plot_data(sig, out_dir, "plot_data_significant_clinical_correlations.csv")
    if sig.empty:
        return
    sig_dir = os.path.join(out_dir, "significant_clinical_scatterplots")
    ensure_dir(sig_dir)
    mdd_df = merged[merged["group"] == CONFIG["MDD_GROUP_NAME"].lower()].copy()
    for _, row in sig.iterrows():
        f = row["factor"]
        c = row["clinical_variable"]
        tmp = mdd_df[[f, c, "subtype"]].dropna()
        if len(tmp) < 4:
            continue
        save_plot_data(tmp, sig_dir, f"plot_data_{sanitize_filename(f)}__{sanitize_filename(c)}__scatter.csv")
        plt.figure(figsize=(5, 4))
        for st in sorted(tmp["subtype"].dropna().unique()):
            ss = tmp[tmp["subtype"] == st]
            plt.scatter(ss[f], ss[c], s=28, alpha=0.75, label=st)
        if tmp["subtype"].dropna().empty:
            plt.scatter(tmp[f], tmp[c], s=28, alpha=0.75)
        plt.xlabel(f)
        plt.ylabel(c)
        plt.title(f"{f} vs {c}\nr={row['spearman_r']:.3f}, p={row['p_value']:.3g}")
        if not tmp["subtype"].dropna().empty:
            plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(sig_dir, f"{sanitize_filename(f)}__{sanitize_filename(c)}__scatter.png"), dpi=int(CONFIG["FIG_DPI"]), bbox_inches="tight")
        plt.close()

    for factor, sub in sig.groupby("factor"):
        sub = sub.copy().sort_values("p_value", ascending=False)
        sub["minus_log10_p"] = -np.log10(sub["p_value"].clip(lower=1e-300))
        save_plot_data(sub, out_dir, f"plot_data_{sanitize_filename(factor)}_clinical_features_p_lt_0.05_barplot.csv")
        plt.figure(figsize=(8, max(4, 0.3 * len(sub) + 1)))
        plt.barh(sub["clinical_variable"].astype(str).str.slice(0, 60), sub["minus_log10_p"])
        plt.xlabel("-log10(p)")
        plt.title(f"Significant clinical variables for {factor}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{sanitize_filename(factor)}_clinical_features_p_lt_0.05_barplot.png"), dpi=int(CONFIG["FIG_DPI"]), bbox_inches="tight")
        plt.close()


def plot_subtype_clinical_heatmap_and_radar(clinical_outputs, out_dir):
    mdd_df = clinical_outputs.get("mdd", pd.DataFrame())
    clinical_cols = clinical_outputs.get("clinical_cols", [])
    if mdd_df is None or mdd_df.empty or not clinical_cols or "subtype" not in mdd_df.columns:
        return
    valid = mdd_df.dropna(subset=["subtype"]).copy()
    if valid.empty:
        return
    # z-score clinical variables within MDD.
    z = valid[["sample_id", "subtype"] + clinical_cols].copy()
    for c in clinical_cols:
        vals = pd.to_numeric(z[c], errors="coerce")
        mu, sd = vals.mean(), vals.std(ddof=0)
        z[c] = (vals - mu) / (sd if sd > 0 else 1.0)
    prof = z.groupby("subtype")[clinical_cols].mean()
    prof.to_csv(os.path.join(out_dir, "plot_data_subtype_clinical_variables_heatmap.csv"), encoding="utf-8-sig")
    plt.figure(figsize=(max(6, len(clinical_cols) * 0.45), max(3, prof.shape[0] * 0.5)))
    plt.imshow(prof.values, aspect="auto", vmin=-2, vmax=2)
    plt.colorbar(label="MDD z-scored clinical mean")
    plt.xticks(np.arange(len(clinical_cols)), clinical_cols, rotation=45, ha="right")
    plt.yticks(np.arange(prof.shape[0]), prof.index)
    plt.title("Subtype clinical variable heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "subtype_clinical_variables_heatmap.png"), dpi=int(CONFIG["FIG_DPI"]), bbox_inches="tight")
    plt.close()

    # Radar plot: limit to at most 12 variables for readability, using variables with lowest subtype Kruskal p.
    subclin = clinical_outputs.get("subtype_clinical", pd.DataFrame())
    if subclin is not None and not subclin.empty:
        chosen = subclin.sort_values("p_value")["clinical_variable"].head(12).tolist()
    else:
        chosen = clinical_cols[:12]
    prof2 = prof[chosen]
    angles = np.linspace(0, 2 * np.pi, len(chosen), endpoint=False).tolist()
    angles += angles[:1]
    radar_df = prof2.copy()
    radar_df.to_csv(os.path.join(out_dir, "plot_data_subtype_clinical_radar_plot.csv"), encoding="utf-8-sig")
    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)
    for st, row in prof2.iterrows():
        values = row.values.astype(float).tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, label=st)
        ax.fill(angles, values, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(chosen, fontsize=8)
    ax.set_title("Subtype clinical radar plot")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "subtype_clinical_radar_plot.png"), dpi=int(CONFIG["FIG_DPI"]), bbox_inches="tight")
    plt.close()


def plot_subtype_vs_hc_top_node_differences(node_diff_df, out_dir):
    if node_diff_df is None or node_diff_df.empty:
        return
    for (st, ftype), sub in node_diff_df.groupby(["subtype", "feature_type"]):
        topn = int(CONFIG["TOP_ROI_PER_FACTOR"] if ftype == "ROI" else CONFIG["TOP_TAXA_PER_FACTOR"])
        top = sub.sort_values("abs_cliffs_delta", ascending=False).head(topn).copy()
        if top.empty:
            continue
        save_plot_data(top, out_dir, f"plot_data_{sanitize_filename(st)}_vs_HC_top_{ftype}_difference_barplot.csv")
        top["feature_short"] = top["feature_name"].astype(str).str.slice(0, 60)
        top = top.sort_values("cliffs_delta_subtype_vs_hc")
        plt.figure(figsize=(8, max(4, 0.35 * len(top) + 1)))
        plt.barh(top["feature_short"], top["cliffs_delta_subtype_vs_hc"])
        plt.xlabel("Cliff's delta: subtype vs HC")
        plt.title(f"{st} vs HC top {ftype} node differences")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{sanitize_filename(st)}_vs_HC_top_{ftype}_difference_barplot.png"), dpi=int(CONFIG["FIG_DPI"]), bbox_inches="tight")
        plt.close()


# ============================================================
# Per-source analysis pipeline
# ============================================================
def run_one_source(source_name: str, B: np.ndarray, M: np.ndarray, sids: List[str], groups: List[str], feature_names: List[str], feature_types: List[str], feature_cols: List[str], base_out_dir: str):
    out_dir = os.path.join(base_out_dir, source_name)
    ensure_dir(out_dir)
    print(f"\n========== Run source: {source_name} ==========")
    X_node, X_raw_norm, B_norm, M_norm = construct_node_matrix(B, M, out_dir, source_name, sids, groups, feature_cols)
    fit_mask = make_fit_mask(groups)
    if len(set(np.array(groups)[fit_mask])) < 2:
        raise ValueError("Need both HC and MDD samples for sPLS-DA.")

    print("[STEP] Fit sPLS-DA-like model with CV search over n_components and keepX.")
    model, k_eff, keepx, mask, scores_all, scores_fit, weights, pred_all, pred_fit, X_fit_std, X_all_std, y_fit, cv_summary = fit_final_splsda(X_node, groups, fit_mask, out_dir)
    plot_splsda_cv(cv_summary, k_eff, keepx, out_dir)

    score_df = pd.DataFrame(scores_all, columns=[f"Factor_{i+1}" for i in range(k_eff)])
    score_df.insert(0, "splsda_pred_mdd_score", pred_all)
    score_df.insert(0, "used_for_splsda_fit", fit_mask.astype(int))
    score_df.insert(0, "group", groups)
    score_df.insert(0, "sample_id", sids)
    save_df(score_df, out_dir, "splsda_factor_scores.csv")

    pred_df = pd.DataFrame({
        "sample_id": sids,
        "group": groups,
        "used_for_splsda_fit": fit_mask.astype(int),
        "splsda_pred_mdd_score": pred_all,
    })
    save_df(pred_df, out_dir, "splsda_predicted_mdd_score.csv")
    np.savez_compressed(
        os.path.join(out_dir, "splsda_model_outputs.npz"),
        scores=scores_all,
        scores_fit=scores_fit,
        weights=weights,
        pred_all=pred_all,
        pred_fit=pred_fit,
        fit_mask=fit_mask.astype(np.int8),
        K_effective=k_eff,
        keepX=keepx,
        selected_union_mask=mask.astype(np.int8),
        X_all_std=X_all_std,
        X_fit_std=X_fit_std,
    )
    print("[SAVE]", os.path.join(out_dir, "splsda_model_outputs.npz"))

    print("[STEP] Component statistics, loadings and stability.")
    stats_df = factor_statistics(scores_all, groups, out_dir)
    load_df, contrib_df = make_loading_tables(weights, feature_names, feature_types, out_dir)
    stab_df = bootstrap_stability(X_node, groups, fit_mask, k_eff, keepx, scores_fit, weights, out_dir)
    sel_df = select_factors(stats_df, stab_df, out_dir)

    print("[STEP] MDD subtype clustering using selected sPLS-DA factors.")
    g = np.array([str(x).lower() for x in groups])
    mdd = CONFIG["MDD_GROUP_NAME"].lower()
    mdd_mask = g == mdd
    sids_mdd = list(np.array(sids)[mdd_mask])
    selected_indices = (sel_df.loc[sel_df["selected_for_subtype"], "component_index"].astype(int).values - 1).tolist()
    if len(selected_indices) == 0:
        raise RuntimeError(
            "No factor passed the selection rule for MDD subtype clustering: "
            f"AUC >= {CONFIG['MIN_AUC_BALANCED']}, "
            f"p <= {CONFIG['MAX_P_VALUE']}, "
            f"|Cliff delta| >= {CONFIG['MIN_ABS_CLIFF_DELTA']}. "
            "Please relax the thresholds or inspect splsda_factor_selection_summary_pvalue_only.csv."
        )
    # Extra safety: use only the first MAX_SELECTED_FACTORS_FOR_CLUSTER selected factors.
    selected_indices = sorted(selected_indices)[: int(CONFIG.get("MAX_SELECTED_FACTORS_FOR_CLUSTER", 2))]
    selected_names = [f"Factor_{i+1}" for i in selected_indices]
    save_df(pd.DataFrame({"selected_factor_for_clustering": selected_names}), out_dir, "splsda_factors_used_for_mdd_clustering.csv")

    scores_mdd_cluster = scores_all[mdd_mask][:, selected_indices]
    best_k, labels_mdd, subtype_mdd, k_eval_df = evaluate_kmeans_and_select(scores_mdd_cluster, sids_mdd, out_dir)
    subtype_map = {str(s): str(st) for s, st in zip(sids_mdd, subtype_mdd)}

    mdd_factor_df = pd.DataFrame(scores_all[mdd_mask, :], columns=[f"Factor_{i+1}" for i in range(k_eff)])
    mdd_factor_df.insert(0, "subtype", subtype_mdd)
    mdd_factor_df.insert(0, "sample_id", sids_mdd)
    save_df(mdd_factor_df, out_dir, "mdd_splsda_scores_with_subtypes.csv")
    subtype_profile = mdd_factor_df.groupby("subtype")[[f"Factor_{i+1}" for i in range(k_eff)]].mean().reset_index()
    save_df(subtype_profile, out_dir, "subtype_mean_splsda_factor_profile.csv")

    print("[STEP] Clinical and subtype node analyses.")
    clinical_outputs = clinical_analysis(scores_all, sids, groups, subtype_map, out_dir)
    node_diff_df = subtype_vs_hc_node_differences(X_all_std, sids, groups, subtype_map, feature_names, feature_types, out_dir)

    print("[STEP] Generate figures and plot-data CSVs.")
    plot_factor_boxplots(scores_all, groups, stats_df, out_dir)
    plot_sample_factor_heatmap(scores_all, sids, groups, out_dir)
    plot_disease_heatmaps(stats_df, out_dir)
    plot_stability_and_selection(sel_df, out_dir)
    plot_top_loading_bars(load_df, "ROI", out_dir)
    plot_top_loading_bars(load_df, "Taxa", out_dir)
    plot_node_factor_heatmaps(load_df, "ROI", out_dir)
    plot_node_factor_heatmaps(load_df, "Taxa", out_dir)
    plot_bipartite_chord_like_network(load_df, out_dir)
    plot_k_metrics(k_eval_df, out_dir)
    plot_mdd_dendrogram(scores_mdd_cluster, sids_mdd, subtype_mdd, out_dir)
    plot_mdd_subtype_factor_heatmap(scores_all[mdd_mask, :], sids_mdd, subtype_mdd, out_dir)
    plot_factor_space(scores_all, groups, sids, subtype_map, out_dir)
    plot_clinical_heatmap(clinical_outputs, out_dir)
    plot_phenotype_annotation(clinical_outputs.get("factor_clinical", pd.DataFrame()), out_dir)
    plot_significant_clinical_scatter_and_bars(clinical_outputs, out_dir)
    plot_subtype_clinical_heatmap_and_radar(clinical_outputs, out_dir)
    plot_subtype_vs_hc_top_node_differences(node_diff_df, out_dir)

    summary = pd.DataFrame({
        "source": [source_name],
        "K_effective": [k_eff],
        "keepX": [keepx],
        "n_selected_union_features": [int(mask.sum())],
        "n_selected_for_subtype": [int(sel_df["selected_for_subtype"].sum())],
        "selected_factors_for_clustering": [";".join(selected_names)],
        "selected_K_subtypes": [best_k],
        "n_samples": [len(sids)],
        "n_mdd": [int(mdd_mask.sum())],
        "n_hc": [int((g == CONFIG["HC_GROUP_NAME"].lower()).sum())],
    })
    save_df(summary, out_dir, "splsda_analysis_run_summary.csv")
    return summary


# ============================================================
# Main
# ============================================================
def main():
    base_out = CONFIG["OUT_DIR"]
    ensure_dir(base_out)
    set_seed(int(CONFIG["SEED"]))

    print("[STEP] Load OOF bridge arrays.")
    sources, sids, groups, feature_names, feature_types, feature_cols = load_bridge_arrays()

    all_summaries = []
    for source_name, (B, M) in sources.items():
        summary = run_one_source(source_name, B, M, sids, groups, feature_names, feature_types, feature_cols, base_out)
        all_summaries.append(summary)

    if all_summaries:
        merged = pd.concat(all_summaries, ignore_index=True)
        save_df(merged, base_out, "splsda_normed_summary.csv")
    print("[DONE] sPLS-DA-like normed-bridge OOF analysis finished.")
    print("[OUT]", base_out)


if __name__ == "__main__":
    main()

