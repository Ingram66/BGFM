# -*- coding: utf-8 -*-
"""
Scheme A: small-sample multimodal classification with weighted multi-kernel SVM.

Core design
-----------
1. Full-sample repeated stratified outer CV.
2. Inner stratified CV jointly tunes:
   - microbiome feature-retention percentage (brain modalities keep all 90 ROIs);
   - RBF gamma for each modality;
   - SVM C;
   - fusion kernel weight.
3. Each modality is preprocessed independently inside every inner/outer training fold:
   VarianceThreshold -> StandardScaler -> mutual-information ranking.
   Brain/predicted-brain retain all 90 ROIs; gut/predicted-gut retain a tuned percentage.
4. Each modality forms its own RBF kernel. Fusion modes use:
       K = beta * K_modality_1 + (1-beta) * K_modality_2
   beta includes 0 and 1, so the fused model may ignore either modality.
5. Final outer-fold probabilities come from SVC(kernel='precomputed', probability=True).

Important
---------
This is weighted multi-kernel SVM with nested-CV selection of kernel weights.
It is not a SimpleMKL convex optimizer, but it is usually more stable and easier to
validate for small biomedical datasets.
"""

import os
import json
import random
from itertools import product, combinations
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, f_classif
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import StratifiedKFold, ParameterGrid, ParameterSampler
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC

try:
    from scipy.stats import ttest_ind, mannwhitneyu
except Exception:
    ttest_ind = None
    mannwhitneyu = None


from bgfm.runtime import load_section, apply_globals, apply_mapping

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    # Real modalities
    "TRUE_BOLD_CSV": r"data/paired/bold_roi_mean.csv",
    "TRUE_MICRO_CSV": r"data/paired/microbiome_abundance.csv",

    # Paired in-house OOF cross-modal predictions
    "OOF_PREDBRAIN_CSV": r"outputs/alignment/OOF_bold_pred.csv",
    "OOF_PREDGUT_CSV": r"outputs/alignment/OOF_taxa_pred.csv",

    "LABEL_CSV": r"data/paired/labels_4class.csv",
    "LABEL_COL_SUBJECT": "subject_id",
    "LABEL_COL_CLASS": "group",

    "OUT_DIR": r"outputs/classification",

    "SEED": 1290,
    "N_REPEATS": 500,
    "N_SPLITS": 5,
    "INNER_SPLITS": 3,
    "N_CLASSES": 4,

    # Microbiome input-space handling.
    # The supplied brain->micro model saves softmax relative abundance, not CLR,
    # so predicted gut should be CLR-transformed once here.
    "NORMALIZE_TRUE_MICRO": True,
    "USE_CLR_FOR_TRUE_MICRO": True,
    "NORMALIZE_PREDGUT": False,   # already sums to approximately 1
    "USE_CLR_FOR_PREDGUT": True,

    # Model selection target. If AUC is the primary endpoint, keep auc_ovr.
    "REFIT_METRIC": "f1_macro",  # auc_ovr / balanced_accuracy / f1_macro / accuracy

    # Search type. Randomized is recommended for two-modality modes.
    "TUNING_SEARCH_TYPE": "randomized",  # randomized / grid
    "RANDOM_SEARCH_N_ITER": 100,

    # Feature selection.
    # Real brain and predicted brain are fixed at all 90 ROIs.
    # Real gut and predicted gut are selected by percentage within each training fold.
    "FEATURE_SCORE": "mutual_info",  # mutual_info / f_classif
    "MI_N_NEIGHBORS": 3,
    "BRAIN_K_GRID": [90],
    "PREDBRAIN_K_GRID": [90],
    "GUT_PERCENTILE_GRID": [10, 20, 30, 50, 100],
    "PREDGUT_PERCENTILE_GRID": [10, 20, 30, 50, 100],

    # RBF kernel and SVM search.
    # 'scale' is resolved independently from each modality's current training fold.
    "GAMMA_GRID": ["scale", 1e-3, 1e-2, 1e-1, 1.0],
    "SVM_C_GRID": [0.1, 1.0, 10.0, 100.0],

    # Weight of the first modality listed by get_mode_modality_specs().
    # Including 0 and 1 allows the fusion model to ignore one modality.
    "KERNEL_WEIGHT_GRID": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                           0.6, 0.7, 0.8, 0.9, 1.0],

    # Run the three fusion modes only.
    "RUN_MODES": ["real_fusion", "brain_predgut", "gut_predbrain"],

    "ROC_POINTS": 300,
    "MAKE_PLOTS": True,
    "TOPK_SELECTED_FEATURES": 20,

    # --------------------------------------------------------
    # Feature importance and class-specific analysis
    # --------------------------------------------------------
    # These additions do not change model fitting or the original outputs.
    "MAKE_FEATURE_IMPORTANCE": True,

    # Calculate feature importance for all three fusion modes.
    "IMPORTANCE_MODES": [
        "real_fusion",
        "brain_predgut",
        "gut_predbrain",
    ],

    # Exact feature names to evaluate in every outer fold.
    # Example: ["HIP.L", "PHG.L", "g__Prevotella_copri_clade_A"]
    # When empty, candidates are selected independently inside each outer
    # training fold using the fitted MI ranking below.
    "IMPORTANCE_FEATURE_NAMES": [],

    # Number of selected candidates evaluated per modality in each outer fold.
    # 0 means all features retained by that fold's fitted preprocessor.
    "IMPORTANCE_TOPK_PER_MODALITY": 10,

    # Test-fold permutations per feature. Increase after a pilot run.
    "IMPORTANCE_N_PERMUTATIONS": 3,

    # Importance can be calculated on only the first N repeats while the model
    # itself still runs all N_REPEATS. 0 means all repeats.
    "IMPORTANCE_MAX_REPEATS": 100,
    "IMPORTANCE_RANDOM_STATE": 2026,

    # Number of features displayed in each importance heatmap/bar plot.
    "IMPORTANCE_PLOT_TOPK": 30,

    # Compare the top global permutation-importance features across the
    # three fusion modes. Predicted-feature prefixes are removed only
    # for overlap matching; original names remain unchanged in outputs.
    "MAKE_GLOBAL_IMPORTANCE_OVERLAP": True,
    "GLOBAL_IMPORTANCE_OVERLAP_TOP_N": 10,

    # Full-sample descriptive analyses: one-vs-rest MI, class means/effects,
    # and optional pairwise comparisons. These are descriptive statistics;
    # the unbiased predictive contribution is provided by outer-test permutation.
    "MAKE_CLASS_SPECIFIC_DESCRIPTIVE": True,
    "MAKE_PAIRWISE_CLASS_ANALYSIS": True,
}


apply_mapping(CONFIG, load_section('classification'))

MODE_INFO = {
    "real_fusion": {
        "label": "Real brain + real gut",
        "color": "#D65F5F",
        "group": "fusion",
    },
    "brain_predgut": {
        "label": "Real brain + predicted gut",
        "color": "#4C78A8",
        "group": "fusion",
    },
    "gut_predbrain": {
        "label": "Real gut + predicted brain",
        "color": "#59A14F",
        "group": "fusion",
    },
    "brain_only": {
        "label": "Real brain only",
        "color": "#9CCBE6",
        "group": "single",
    },
    "gut_only": {
        "label": "Real gut only",
        "color": "#F2C879",
        "group": "single",
    },
    "predbrain_only": {
        "label": "Predicted brain only",
        "color": "#B5A0D5",
        "group": "single",
    },
    "predgut_only": {
        "label": "Predicted gut only",
        "color": "#8CD0C3",
        "group": "single",
    },
}

FUSION_MODES = ["real_fusion", "brain_predgut", "gut_predbrain"]
SINGLE_MODES = ["brain_only", "gut_only", "predbrain_only", "predgut_only"]
ALL_MODES = FUSION_MODES + SINGLE_MODES


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_distribution_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, 0.0, None)
    s = np.maximum(x.sum(axis=1, keepdims=True), eps)
    return x / s


def clr_np(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, eps, None)
    logx = np.log(x)
    return logx - logx.mean(axis=1, keepdims=True)


def json_dumps_safe(obj: Any) -> str:
    def _convert(x):
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        return str(x)
    return json.dumps(obj, ensure_ascii=False, default=_convert)


# ============================================================
# Data loading and alignment
# ============================================================
def load_feature_csv(
    csv_path: str,
    subject_col: Optional[str] = None,
    normalize_abundance: bool = False,
    use_clr: bool = False,
) -> Tuple[List[str], np.ndarray, List[str]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    if subject_col is None:
        subject_col = df.columns[0]
    if subject_col not in df.columns:
        raise ValueError(f"{csv_path} has no subject column: {subject_col}")
    if df[subject_col].duplicated().any():
        duplicated = df.loc[df[subject_col].duplicated(), subject_col].astype(str).tolist()[:10]
        raise ValueError(f"{csv_path} contains duplicate subject IDs, examples: {duplicated}")

    sids = df[subject_col].astype(str).tolist()
    feat_cols = [c for c in df.columns if c != subject_col]
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
    if np.isnan(X).any():
        raise ValueError(f"{csv_path} contains NaN values.")
    if normalize_abundance:
        X = normalize_distribution_np(X)
    if use_clr:
        X = clr_np(X)
    return sids, X, feat_cols


def load_label_csv(csv_path: str, subject_col: str, label_col: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    if subject_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"{csv_path} must contain {subject_col} and {label_col}.")
    if df[subject_col].duplicated().any():
        raise ValueError(f"{csv_path} contains duplicate subject IDs.")

    sids = df[subject_col].astype(str).tolist()
    raw = df[label_col].astype(str).tolist()
    unique = sorted(set(raw))
    label_to_int = {lab: i for i, lab in enumerate(unique)}
    int_to_label = {i: lab for lab, i in label_to_int.items()}
    label_map = {sid: label_to_int[raw[i]] for i, sid in enumerate(sids)}
    return label_map, int_to_label


def align_all_modalities():
    bold_sids, X_bold, bold_cols = load_feature_csv(
        CONFIG["TRUE_BOLD_CSV"], normalize_abundance=False, use_clr=False
    )
    gut_sids, X_gut, gut_cols = load_feature_csv(
        CONFIG["TRUE_MICRO_CSV"],
        normalize_abundance=bool(CONFIG["NORMALIZE_TRUE_MICRO"]),
        use_clr=bool(CONFIG["USE_CLR_FOR_TRUE_MICRO"]),
    )
    predb_sids, X_predb, predb_cols = load_feature_csv(
        CONFIG["OOF_PREDBRAIN_CSV"], normalize_abundance=False, use_clr=False
    )
    predg_sids, X_predg, predg_cols = load_feature_csv(
        CONFIG["OOF_PREDGUT_CSV"],
        normalize_abundance=bool(CONFIG["NORMALIZE_PREDGUT"]),
        use_clr=bool(CONFIG["USE_CLR_FOR_PREDGUT"]),
    )

    # Rename predicted outputs according to the corresponding real-feature
    # order. This changes labels only; the numerical prediction matrices are
    # not modified. The prediction-output order must match the real input order.
    if len(predb_cols) != len(bold_cols):
        raise ValueError(
            "Predicted brain and real brain have different feature counts: "
            f"{len(predb_cols)} vs {len(bold_cols)}. Cannot assign ROI names safely."
        )
    if len(predg_cols) != len(gut_cols):
        raise ValueError(
            "Predicted gut and real gut have different feature counts: "
            f"{len(predg_cols)} vs {len(gut_cols)}. Cannot assign taxa names safely."
        )

    original_predb_cols = list(predb_cols)
    original_predg_cols = list(predg_cols)
    predb_cols = [f"pred_{name}" for name in bold_cols]
    predg_cols = [f"pred_{name}" for name in gut_cols]

    print(
        "[FEATURE NAMES] Predicted brain columns renamed by position, e.g.",
        f"{original_predb_cols[0]} -> {predb_cols[0]}" if predb_cols else "none",
    )
    print(
        "[FEATURE NAMES] Predicted gut columns renamed by position, e.g.",
        f"{original_predg_cols[0]} -> {predg_cols[0]}" if predg_cols else "none",
    )

    label_map, int_to_label = load_label_csv(
        CONFIG["LABEL_CSV"],
        subject_col=CONFIG["LABEL_COL_SUBJECT"],
        label_col=CONFIG["LABEL_COL_CLASS"],
    )

    maps = [
        {sid: i for i, sid in enumerate(bold_sids)},
        {sid: i for i, sid in enumerate(gut_sids)},
        {sid: i for i, sid in enumerate(predb_sids)},
        {sid: i for i, sid in enumerate(predg_sids)},
    ]
    common = sorted(set(maps[0]) & set(maps[1]) & set(maps[2]) & set(maps[3]) & set(label_map))
    if not common:
        raise ValueError("No common subject_id across all feature and label files.")

    y = np.asarray([label_map[sid] for sid in common], dtype=np.int64)
    unique_y = sorted(np.unique(y).tolist())
    expected = list(range(len(unique_y)))
    if unique_y != expected:
        raise ValueError(f"Labels must map to consecutive integers; got {unique_y}")
    if len(unique_y) != int(CONFIG["N_CLASSES"]):
        raise ValueError(
            f"N_CLASSES={CONFIG['N_CLASSES']} but aligned data has {len(unique_y)} classes."
        )

    data = {
        "brain_only": np.stack([X_bold[maps[0][sid]] for sid in common]),
        "gut_only": np.stack([X_gut[maps[1][sid]] for sid in common]),
        "predbrain_only": np.stack([X_predb[maps[2][sid]] for sid in common]),
        "predgut_only": np.stack([X_predg[maps[3][sid]] for sid in common]),
    }
    data["real_fusion"] = np.concatenate([data["brain_only"], data["gut_only"]], axis=1)
    data["brain_predgut"] = np.concatenate([data["brain_only"], data["predgut_only"]], axis=1)
    data["gut_predbrain"] = np.concatenate([data["gut_only"], data["predbrain_only"]], axis=1)

    feature_names = {
        "brain_only": bold_cols,
        "gut_only": gut_cols,
        "predbrain_only": predb_cols,
        "predgut_only": predg_cols,
        "real_fusion": list(bold_cols) + list(gut_cols),
        "brain_predgut": list(bold_cols) + list(predg_cols),
        "gut_predbrain": list(gut_cols) + list(predb_cols),
    }
    feature_types = {
        "brain_only": ["Brain ROIs"] * len(bold_cols),
        "gut_only": ["Microbial taxa"] * len(gut_cols),
        "predbrain_only": ["Predicted brain ROIs"] * len(predb_cols),
        "predgut_only": ["Predicted microbial taxa"] * len(predg_cols),
        "real_fusion": ["Brain ROIs"] * len(bold_cols) + ["Microbial taxa"] * len(gut_cols),
        "brain_predgut": ["Brain ROIs"] * len(bold_cols) + ["Predicted microbial taxa"] * len(predg_cols),
        "gut_predbrain": ["Microbial taxa"] * len(gut_cols) + ["Predicted brain ROIs"] * len(predb_cols),
    }
    dims = {
        "brain": len(bold_cols),
        "gut": len(gut_cols),
        "predbrain": len(predb_cols),
        "predgut": len(predg_cols),
    }
    return common, data, y, int_to_label, feature_names, feature_types, dims


# ============================================================
# Mode definitions
# ============================================================
def get_active_modes() -> List[str]:
    modes = list(CONFIG.get("RUN_MODES", []))
    if not modes:
        modes = list(ALL_MODES)
    invalid = [m for m in modes if m not in ALL_MODES]
    if invalid:
        raise ValueError(f"Invalid RUN_MODES: {invalid}. Valid modes: {ALL_MODES}")
    return modes


def get_mode_modality_specs(mode: str, dims: Dict[str, int]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    start = 0

    def add(name: str, feature_type: str):
        nonlocal start
        n = int(dims[name])
        specs.append({
            "name": name,
            "label": feature_type,
            "start": start,
            "end": start + n,
            "indices": np.arange(start, start + n),
        })
        start += n

    if mode == "brain_only":
        add("brain", "Brain ROIs")
    elif mode == "gut_only":
        add("gut", "Microbial taxa")
    elif mode == "predbrain_only":
        add("predbrain", "Predicted brain ROIs")
    elif mode == "predgut_only":
        add("predgut", "Predicted microbial taxa")
    elif mode == "real_fusion":
        add("brain", "Brain ROIs")
        add("gut", "Microbial taxa")
    elif mode == "brain_predgut":
        add("brain", "Brain ROIs")
        add("predgut", "Predicted microbial taxa")
    elif mode == "gut_predbrain":
        add("gut", "Microbial taxa")
        add("predbrain", "Predicted brain ROIs")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return specs


def modality_uses_percentile(modality_name: str) -> bool:
    return modality_name in {"gut", "predgut"}


def get_selection_grid(modality_name: str) -> List[int]:
    """Return candidate selection values for one modality.

    Brain modalities use a fixed feature count (k).
    Microbiome modalities use a percentage of features available after
    VarianceThreshold in the current training fold.
    """
    key = {
        "brain": "BRAIN_K_GRID",
        "gut": "GUT_PERCENTILE_GRID",
        "predbrain": "PREDBRAIN_K_GRID",
        "predgut": "PREDGUT_PERCENTILE_GRID",
    }[modality_name]
    values = [int(v) for v in CONFIG[key]]
    if modality_uses_percentile(modality_name):
        invalid = [v for v in values if v <= 0 or v > 100]
        if invalid:
            raise ValueError(
                f"{key} must contain percentages in (0, 100], got {invalid}"
            )
    return values


def selection_param_name(modality_name: str) -> str:
    prefix = "percentile" if modality_uses_percentile(modality_name) else "k"
    return f"{prefix}__{modality_name}"


def selection_value_to_k(
    modality_name: str,
    requested_value: int,
    n_available: int,
) -> int:
    """Convert a fixed k or microbiome percentage to an effective feature count."""
    if modality_uses_percentile(modality_name):
        percentile = float(requested_value)
        if percentile <= 0 or percentile > 100:
            raise ValueError(
                f"Percentile for {modality_name} must be in (0, 100], got {percentile}"
            )
        # Match SelectPercentile-style behavior: use the integer floor,
        # while retaining at least one feature.
        k = int(np.floor(n_available * percentile / 100.0))
        return max(1, min(k, int(n_available)))
    return effective_k(int(requested_value), int(n_available))


# ============================================================
# Per-modality preprocessing
# ============================================================
def fit_modality_ranker(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Fit variance filtering, scaling and a complete univariate feature ranking.

    The ranking is fitted once per modality per inner fold. Candidate fixed-k
    (brain) or percentile (microbiome) selections then reuse the same ranking,
    which is much faster than recomputing MI for every C/gamma/kernel-weight candidate.
    """
    variance = VarianceThreshold(threshold=0.0)
    X_var = variance.fit_transform(X_train)
    if X_var.shape[1] == 0:
        raise ValueError("All features are zero-variance in the current training fold.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_var)

    score_method = str(CONFIG.get("FEATURE_SCORE", "mutual_info")).lower()
    if score_method == "mutual_info":
        scores = mutual_info_classif(
            X_scaled,
            y_train,
            random_state=seed,
            n_neighbors=int(CONFIG["MI_N_NEIGHBORS"]),
        )
    elif score_method == "f_classif":
        scores, _ = f_classif(X_scaled, y_train)
    else:
        raise ValueError("FEATURE_SCORE must be 'mutual_info' or 'f_classif'.")

    scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    order = np.argsort(scores)[::-1]
    X_ranked = X_scaled[:, order]
    var_original_indices = variance.get_support(indices=True)

    fitted = {
        "variance": variance,
        "scaler": scaler,
        "ranking_after_variance": order,
        "ranking_original_local": var_original_indices[order],
        "scores_ranked": scores[order],
        "n_ranked": int(len(order)),
    }
    return fitted, np.asarray(X_ranked, dtype=np.float64)


def transform_modality_ranked(X: np.ndarray, fitted: Dict[str, Any]) -> np.ndarray:
    X_var = fitted["variance"].transform(X)
    X_scaled = fitted["scaler"].transform(X_var)
    return np.asarray(X_scaled[:, fitted["ranking_after_variance"]], dtype=np.float64)


def effective_k(k_requested: int, n_available: int) -> int:
    return max(1, min(int(k_requested), int(n_available)))


def resolve_gamma(gamma_spec: Any, X_train: np.ndarray) -> float:
    if isinstance(gamma_spec, str):
        if gamma_spec == "scale":
            variance = float(np.var(X_train))
            if variance <= 0 or not np.isfinite(variance):
                return 1.0 / max(X_train.shape[1], 1)
            return 1.0 / (max(X_train.shape[1], 1) * variance)
        if gamma_spec == "auto":
            return 1.0 / max(X_train.shape[1], 1)
        raise ValueError(f"Unsupported gamma string: {gamma_spec}")
    gamma = float(gamma_spec)
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    return gamma


def prepare_inner_fold_cache(
    X_train_fold: np.ndarray,
    y_train_fold: np.ndarray,
    X_val_fold: np.ndarray,
    mode: str,
    dims: Dict[str, int],
    seed: int,
) -> Dict[str, Any]:
    specs = get_mode_modality_specs(mode, dims)
    modality_cache = {}
    for i, spec in enumerate(specs):
        Xi_train = X_train_fold[:, spec["indices"]]
        Xi_val = X_val_fold[:, spec["indices"]]
        fitted, Xtr_ranked = fit_modality_ranker(Xi_train, y_train_fold, seed + i)
        Xval_ranked = transform_modality_ranked(Xi_val, fitted)
        modality_cache[spec["name"]] = {
            "spec": spec,
            "fitted": fitted,
            "Xtr_ranked": Xtr_ranked,
            "Xval_ranked": Xval_ranked,
            "kernel_cache": {},
        }
    return {"modalities": modality_cache, "specs": specs}


def get_cached_inner_kernel(
    modality_entry: Dict[str, Any],
    selection_requested: int,
    gamma_spec: Any,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    modality_name = str(modality_entry["spec"]["name"])
    n_available = modality_entry["Xtr_ranked"].shape[1]
    k_eff = selection_value_to_k(
        modality_name, selection_requested, n_available
    )
    key = (int(k_eff), str(gamma_spec))
    if key not in modality_entry["kernel_cache"]:
        Xtr = modality_entry["Xtr_ranked"][:, :k_eff]
        Xval = modality_entry["Xval_ranked"][:, :k_eff]
        gamma = resolve_gamma(gamma_spec, Xtr)
        modality_entry["kernel_cache"][key] = (
            rbf_kernel(Xtr, Xtr, gamma=gamma),
            rbf_kernel(Xval, Xtr, gamma=gamma),
            float(gamma),
        )
    Ktr, Kval, gamma = modality_entry["kernel_cache"][key]
    return Ktr, Kval, float(gamma), int(k_eff)


def fit_outer_preprocessors_and_kernels(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    mode: str,
    dims: Dict[str, int],
    params: Dict[str, Any],
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[np.ndarray], List[np.ndarray], List[float]]:
    specs = get_mode_modality_specs(mode, dims)
    fitted_list = []
    Ktr_parts = []
    Kte_parts = []
    gamma_list = []

    for i, spec in enumerate(specs):
        name = spec["name"]
        Xi_train = X_train[:, spec["indices"]]
        Xi_test = X_test[:, spec["indices"]]
        fitted, Xtr_ranked = fit_modality_ranker(Xi_train, y_train, seed + i)
        Xte_ranked = transform_modality_ranked(Xi_test, fitted)

        selection_key = selection_param_name(name)
        selection_requested = int(params[selection_key])
        k_eff = selection_value_to_k(
            name, selection_requested, Xtr_ranked.shape[1]
        )
        Xtr = Xtr_ranked[:, :k_eff]
        Xte = Xte_ranked[:, :k_eff]
        gamma = resolve_gamma(params[f"gamma__{name}"], Xtr)

        fitted.update({
            "modality_name": name,
            "modality_label": spec["label"],
            "global_columns": np.asarray(spec["indices"], dtype=int),
            "selected_original_local": fitted["ranking_original_local"][:k_eff],
            "mi_scores_selected": fitted["scores_ranked"][:k_eff],
            "selection_type": (
                "percentile" if modality_uses_percentile(name) else "fixed_k"
            ),
            "selection_requested": int(selection_requested),
            "k_requested": (
                None if modality_uses_percentile(name) else int(selection_requested)
            ),
            "percentile_requested": (
                int(selection_requested) if modality_uses_percentile(name) else None
            ),
            "k_effective": int(k_eff),
            "gamma_resolved": float(gamma),

            # Retained only in memory for outer-test permutation importance.
            # This is the selected/scaled outer-training representation used
            # to reconstruct a perturbed test-to-training RBF kernel.
            "X_train_selected": np.asarray(Xtr, dtype=np.float64),
        })
        fitted_list.append(fitted)
        Ktr_parts.append(rbf_kernel(Xtr, Xtr, gamma=gamma))
        Kte_parts.append(rbf_kernel(Xte, Xtr, gamma=gamma))
        gamma_list.append(float(gamma))

    return fitted_list, Ktr_parts, Kte_parts, gamma_list


# ============================================================
# Multi-kernel construction and scoring
# ============================================================
def build_candidate_space(mode: str, dims: Dict[str, int]) -> List[Dict[str, Any]]:
    specs = get_mode_modality_specs(mode, dims)
    distributions: Dict[str, List[Any]] = {
        "C": list(CONFIG["SVM_C_GRID"]),
    }
    for spec in specs:
        name = spec["name"]
        distributions[selection_param_name(name)] = get_selection_grid(name)
        distributions[f"gamma__{name}"] = list(CONFIG["GAMMA_GRID"])

    if len(specs) == 2:
        distributions["weight_first"] = list(CONFIG["KERNEL_WEIGHT_GRID"])

    search_type = str(CONFIG["TUNING_SEARCH_TYPE"]).lower()
    if search_type == "grid":
        return list(ParameterGrid(distributions))
    if search_type == "randomized":
        total = int(np.prod([len(v) for v in distributions.values()]))
        n_iter = min(int(CONFIG["RANDOM_SEARCH_N_ITER"]), total)
        return list(ParameterSampler(distributions, n_iter=n_iter, random_state=int(CONFIG["SEED"])))
    raise ValueError("TUNING_SEARCH_TYPE must be 'grid' or 'randomized'.")


def combine_kernel_list(kernels: List[np.ndarray], weight_first: Optional[float]) -> Tuple[np.ndarray, List[float]]:
    if len(kernels) == 1:
        return kernels[0], [1.0]
    if len(kernels) != 2:
        raise ValueError("This implementation supports one or two modalities per mode.")
    beta = float(weight_first)
    beta = min(max(beta, 0.0), 1.0)
    return beta * kernels[0] + (1.0 - beta) * kernels[1], [beta, 1.0 - beta]


def multiclass_macro_auc_from_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    n_classes: int,
) -> float:
    ordered = np.zeros((len(y_true), n_classes), dtype=float)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    for j, cls in enumerate(classes):
        ordered[:, int(cls)] = scores[:, j]

    y_bin = label_binarize(y_true, classes=np.arange(n_classes))
    aucs = []
    for c in range(n_classes):
        if y_bin[:, c].min() == y_bin[:, c].max():
            continue
        aucs.append(roc_auc_score(y_bin[:, c], ordered[:, c]))
    return float(np.mean(aucs)) if aucs else 0.0


def score_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    decision_scores: np.ndarray,
    classes: np.ndarray,
    n_classes: int,
    metric: str,
) -> float:
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    if metric == "f1_macro":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "auc_ovr":
        return multiclass_macro_auc_from_scores(
            y_true, decision_scores, classes, n_classes
        )
    raise ValueError(f"Unknown REFIT_METRIC: {metric}")


# ============================================================
# Nested tuning and outer-fold fitting
# ============================================================
def train_one_fold_schemeA_mksvm(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_classes: int,
    seed: int,
    mode: str,
    dims: Dict[str, int],
):
    X_train = X[train_idx].astype(np.float64)
    y_train = y[train_idx].astype(np.int64)
    X_test = X[test_idx].astype(np.float64)
    y_test = y[test_idx].astype(np.int64)

    inner_cv = StratifiedKFold(
        n_splits=int(CONFIG["INNER_SPLITS"]),
        shuffle=True,
        random_state=seed,
    )
    inner_splits = list(inner_cv.split(np.zeros(len(y_train)), y_train))
    candidates = build_candidate_space(mode, dims)
    refit_metric = str(CONFIG["REFIT_METRIC"])
    specs = get_mode_modality_specs(mode, dims)

    # Fit feature ranking once per modality per inner fold and cache kernels.
    fold_caches = []
    for inner_fold, (itr, iva) in enumerate(inner_splits, start=1):
        fold_caches.append({
            "itr": itr,
            "iva": iva,
            "y_train": y_train[itr],
            "y_val": y_train[iva],
            "cache": prepare_inner_fold_cache(
                X_train[itr],
                y_train[itr],
                X_train[iva],
                mode,
                dims,
                seed=seed + inner_fold * 1000,
            ),
        })

    best_score = -np.inf
    best_params: Optional[Dict[str, Any]] = None
    candidate_rows = []

    for cand_id, params in enumerate(candidates, start=1):
        fold_scores = []
        failure_message = ""

        for fold_info in fold_caches:
            try:
                Ktr_parts = []
                Kval_parts = []
                resolved = {}

                for spec in specs:
                    name = spec["name"]
                    entry = fold_info["cache"]["modalities"][name]
                    selection_key = selection_param_name(name)
                    Ktr_i, Kval_i, gamma_i, k_eff_i = get_cached_inner_kernel(
                        entry,
                        int(params[selection_key]),
                        params[f"gamma__{name}"],
                    )
                    Ktr_parts.append(Ktr_i)
                    Kval_parts.append(Kval_i)
                    resolved[f"gamma_resolved__{name}"] = gamma_i
                    resolved[f"k_effective__{name}"] = k_eff_i

                Ktr, _ = combine_kernel_list(Ktr_parts, params.get("weight_first"))
                Kval, _ = combine_kernel_list(Kval_parts, params.get("weight_first"))

                model = SVC(
                    kernel="precomputed",
                    C=float(params["C"]),
                    class_weight="balanced",
                    probability=False,
                    decision_function_shape="ovr",
                    random_state=seed,
                )
                model.fit(Ktr, fold_info["y_train"])
                pred = model.predict(Kval).astype(np.int64)
                decision = model.decision_function(Kval)
                fold_scores.append(
                    score_predictions(
                        fold_info["y_val"], pred, decision,
                        model.classes_, n_classes, refit_metric
                    )
                )
            except Exception as exc:
                failure_message = f"{type(exc).__name__}: {exc}"
                fold_scores = []
                break

        mean_score = float(np.mean(fold_scores)) if fold_scores else -np.inf
        candidate_rows.append({
            "candidate_id": cand_id,
            "mean_inner_score": mean_score,
            "failed": int(not bool(fold_scores)),
            "failure_message": failure_message,
            "params_json": json_dumps_safe(params),
        })
        if mean_score > best_score:
            best_score = mean_score
            best_params = dict(params)

    if best_params is None or not np.isfinite(best_score):
        examples = [r["failure_message"] for r in candidate_rows if r["failure_message"]][:3]
        raise RuntimeError(
            f"All multi-kernel SVM candidates failed for mode={mode}. Examples: {examples}"
        )

    # Refit ranking and kernels on the complete outer training set.
    fitted_list, Ktr_parts, Kte_parts, gamma_list = fit_outer_preprocessors_and_kernels(
        X_train, y_train, X_test, mode, dims, best_params, seed=seed + 99991
    )
    Ktr, weights = combine_kernel_list(Ktr_parts, best_params.get("weight_first"))
    Kte, _ = combine_kernel_list(Kte_parts, best_params.get("weight_first"))

    final_model = SVC(
        kernel="precomputed",
        C=float(best_params["C"]),
        class_weight="balanced",
        probability=True,
        decision_function_shape="ovr",
        random_state=seed,
    )
    final_model.fit(Ktr, y_train)
    prob_native = final_model.predict_proba(Kte)
    ordered_prob = np.zeros((len(y_test), n_classes), dtype=np.float64)
    for j, cls in enumerate(final_model.classes_):
        ordered_prob[:, int(cls)] = prob_native[:, j]
    pred = final_model.predict(Kte).astype(np.int64)

    best_params_out = dict(best_params)
    for spec, fitted, gamma, weight in zip(specs, fitted_list, gamma_list, weights):
        name = spec["name"]
        best_params_out[f"selection_type__{name}"] = fitted["selection_type"]
        best_params_out[f"selection_requested__{name}"] = int(
            fitted["selection_requested"]
        )
        best_params_out[f"k_effective__{name}"] = int(fitted["k_effective"])
        best_params_out[f"gamma_resolved__{name}"] = float(gamma)
        best_params_out[f"kernel_weight__{name}"] = float(weight)

    return {
        "y_true": y_test,
        "y_pred": pred,
        "prob": ordered_prob,
        "model": final_model,
        "preprocessors": fitted_list,
        "best_params": best_params_out,
        "best_score": float(best_score),
        "refit_metric": refit_metric,
        "classifier": "weighted_multikernel_svm",
        "kernel_weights": {
            specs[i]["name"]: float(weights[i]) for i in range(len(specs))
        },
        "candidate_rows": candidate_rows,

        # Transient fold objects used immediately by the outer-loop
        # permutation analysis. They are not written as model files.
        "baseline_kernel_parts": [
            np.asarray(k, dtype=np.float64) for k in Kte_parts
        ],
    }



# ============================================================
# Predictive contribution and class-specific importance
# ============================================================
def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction with NaN preservation."""
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not np.any(valid):
        return q

    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    q[valid] = restored
    return q


def get_importance_modes(active_modes: List[str]) -> List[str]:
    requested = list(CONFIG.get("IMPORTANCE_MODES", []))
    if not requested:
        return list(active_modes)
    invalid = [m for m in requested if m not in active_modes]
    if invalid:
        raise ValueError(
            f"IMPORTANCE_MODES contains modes not being run: {invalid}. "
            f"Active modes: {active_modes}"
        )
    return requested


def ordered_probability_from_precomputed_kernel(
    model: SVC,
    K_test_train: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    native = model.predict_proba(K_test_train)
    ordered = np.zeros((len(K_test_train), n_classes), dtype=np.float64)
    for j, cls in enumerate(model.classes_):
        ordered[:, int(cls)] = native[:, j]
    return ordered


def exact_classwise_auc(
    y_true: np.ndarray,
    prob: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    out = np.full(n_classes, np.nan, dtype=float)
    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        if y_bin.min() == y_bin.max():
            continue
        out[c] = float(roc_auc_score(y_bin, prob[:, c]))
    return out


def predictive_metrics_for_importance(
    y_true: np.ndarray,
    prob: np.ndarray,
    n_classes: int,
) -> Dict[str, Any]:
    pred = np.argmax(prob, axis=1).astype(int)
    class_auc = exact_classwise_auc(y_true, prob, n_classes)
    class_recall = np.full(n_classes, np.nan, dtype=float)
    class_true_prob = np.full(n_classes, np.nan, dtype=float)

    for c in range(n_classes):
        mask = y_true == c
        if not np.any(mask):
            continue
        class_recall[c] = float(np.mean(pred[mask] == c))
        class_true_prob[c] = float(np.mean(prob[mask, c]))

    return {
        "macro_auc": float(np.nanmean(class_auc)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "class_auc": class_auc,
        "class_recall": class_recall,
        "class_true_prob": class_true_prob,
    }


def _find_feature_modality_position(
    result: Dict[str, Any],
    feature_global_index: int,
) -> Tuple[int, Dict[str, Any], int, bool]:
    """Return modality list index, fitted preprocessor, raw local index, selected flag."""
    gi = int(feature_global_index)
    for modality_position, fitted in enumerate(result["preprocessors"]):
        global_columns = np.asarray(fitted["global_columns"], dtype=int)
        hit = np.where(global_columns == gi)[0]
        if len(hit) == 0:
            continue
        raw_local_index = int(hit[0])
        selected_global = global_columns[
            np.asarray(fitted["selected_original_local"], dtype=int)
        ]
        selected = bool(np.any(selected_global == gi))
        return modality_position, fitted, raw_local_index, selected
    raise ValueError(f"Feature index {gi} is not represented by the fitted modalities.")


def select_fold_importance_candidates(
    result: Dict[str, Any],
    mode: str,
    full_feature_names: List[str],
    full_feature_types: List[str],
) -> List[Dict[str, Any]]:
    """Select candidates using outer-training information only.

    Exact names in IMPORTANCE_FEATURE_NAMES are evaluated in every fold. If the
    list is empty, the top selected features from each modality are used.
    """
    exact_names = [str(x) for x in CONFIG.get("IMPORTANCE_FEATURE_NAMES", [])]
    rows: List[Dict[str, Any]] = []

    if exact_names:
        name_to_indices: Dict[str, List[int]] = {}
        for i, name in enumerate(full_feature_names):
            name_to_indices.setdefault(str(name), []).append(i)
        missing = [name for name in exact_names if name not in name_to_indices]
        if missing:
            print(f"[WARNING] {mode}: requested importance features not found: {missing}")
        for name in exact_names:
            for gi in name_to_indices.get(name, []):
                modality_position, fitted, _, selected = _find_feature_modality_position(
                    result, gi
                )
                rows.append({
                    "feature_index": int(gi),
                    "feature": str(full_feature_names[gi]),
                    "feature_type": str(full_feature_types[gi]),
                    "modality": str(fitted["modality_name"]),
                    "modality_label": str(fitted["modality_label"]),
                    "modality_position": int(modality_position),
                    "selected_in_model": int(selected),
                    "training_mi_score": float(
                        _training_mi_for_global_feature(fitted, gi)
                    ),
                    "candidate_source": "explicit_name",
                })
        return rows

    topk = int(CONFIG.get("IMPORTANCE_TOPK_PER_MODALITY", 10))
    for modality_position, fitted in enumerate(result["preprocessors"]):
        global_columns = np.asarray(fitted["global_columns"], dtype=int)
        selected_local = np.asarray(fitted["selected_original_local"], dtype=int)
        selected_global = global_columns[selected_local]
        scores = np.asarray(fitted["mi_scores_selected"], dtype=float)

        n_take = len(selected_global) if topk <= 0 else min(topk, len(selected_global))
        for rank in range(n_take):
            gi = int(selected_global[rank])
            rows.append({
                "feature_index": gi,
                "feature": str(full_feature_names[gi]),
                "feature_type": str(full_feature_types[gi]),
                "modality": str(fitted["modality_name"]),
                "modality_label": str(fitted["modality_label"]),
                "modality_position": int(modality_position),
                "selected_in_model": 1,
                "training_mi_score": float(scores[rank]),
                "training_mi_rank_within_selected": int(rank + 1),
                "candidate_source": "outer_train_top_mi",
            })
    return rows


def _training_mi_for_global_feature(fitted: Dict[str, Any], gi: int) -> float:
    global_columns = np.asarray(fitted["global_columns"], dtype=int)
    hit = np.where(global_columns == int(gi))[0]
    if len(hit) == 0:
        return np.nan
    local_original = int(hit[0])
    ranking_original = np.asarray(fitted["ranking_original_local"], dtype=int)
    rank_hit = np.where(ranking_original == local_original)[0]
    if len(rank_hit) == 0:
        return np.nan
    return float(np.asarray(fitted["scores_ranked"], dtype=float)[int(rank_hit[0])])


def rebuild_perturbed_test_kernel_part(
    X_test_raw: np.ndarray,
    fitted: Dict[str, Any],
    raw_local_index: int,
    permutation: np.ndarray,
) -> np.ndarray:
    global_columns = np.asarray(fitted["global_columns"], dtype=int)
    Xi_test = np.asarray(X_test_raw[:, global_columns], dtype=np.float64).copy()
    Xi_test[:, int(raw_local_index)] = Xi_test[permutation, int(raw_local_index)]

    Xte_ranked = transform_modality_ranked(Xi_test, fitted)
    k_eff = int(fitted["k_effective"])
    Xte_selected = Xte_ranked[:, :k_eff]
    Xtr_selected = np.asarray(fitted["X_train_selected"], dtype=np.float64)
    gamma = float(fitted["gamma_resolved"])
    return rbf_kernel(Xte_selected, Xtr_selected, gamma=gamma)


def compute_outer_fold_permutation_importance(
    result: Dict[str, Any],
    X_test_raw: np.ndarray,
    mode: str,
    repeat: int,
    fold: int,
    feature_names_mode: List[str],
    feature_types_mode: List[str],
    int_to_label: Dict[int, str],
    n_classes: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Permutation importance on an untouched outer test fold.

    No model refitting occurs. The selected/scaled outer-training representation,
    gamma, kernel weight and fitted preprocessing are reused. Only one raw test
    feature is permuted and its modality-specific test-to-training kernel is rebuilt.
    """
    y_true = np.asarray(result["y_true"], dtype=int)
    baseline_prob = np.asarray(result["prob"], dtype=float)
    baseline = predictive_metrics_for_importance(y_true, baseline_prob, n_classes)
    candidates = select_fold_importance_candidates(
        result, mode, feature_names_mode, feature_types_mode
    )
    n_perm = max(1, int(CONFIG.get("IMPORTANCE_N_PERMUTATIONS", 3)))
    rng = np.random.RandomState(int(seed))

    global_rows: List[Dict[str, Any]] = []
    class_rows: List[Dict[str, Any]] = []

    for candidate in candidates:
        gi = int(candidate["feature_index"])
        modality_position, fitted, raw_local_index, selected = _find_feature_modality_position(
            result, gi
        )

        for permutation_id in range(1, n_perm + 1):
            if selected:
                permutation = rng.permutation(len(X_test_raw))
                perturbed_part = rebuild_perturbed_test_kernel_part(
                    X_test_raw=X_test_raw,
                    fitted=fitted,
                    raw_local_index=raw_local_index,
                    permutation=permutation,
                )
                kernel_parts = list(result["baseline_kernel_parts"])
                kernel_parts[modality_position] = perturbed_part
                K_perm, _ = combine_kernel_list(
                    kernel_parts,
                    result["best_params"].get("weight_first"),
                )
                prob_perm = ordered_probability_from_precomputed_kernel(
                    result["model"], K_perm, n_classes
                )
            else:
                # A feature removed by variance filtering/selection cannot affect
                # this fitted fold model; its predictive importance is exactly zero.
                prob_perm = baseline_prob.copy()

            perm_metrics = predictive_metrics_for_importance(
                y_true, prob_perm, n_classes
            )

            common = {
                "repeat": int(repeat),
                "fold": int(fold),
                "outer_fold_id": f"{int(repeat)}_{int(fold)}",
                "mode": mode,
                "mode_label": MODE_INFO[mode]["label"],
                "feature_index": gi,
                "feature": candidate["feature"],
                "feature_type": candidate["feature_type"],
                "modality": candidate["modality"],
                "modality_label": candidate["modality_label"],
                "selected_in_model": int(selected),
                "candidate_source": candidate["candidate_source"],
                "training_mi_score": candidate["training_mi_score"],
                "permutation_id": int(permutation_id),
                "n_test": int(len(y_true)),
            }

            global_rows.append({
                **common,
                "baseline_macro_auc": baseline["macro_auc"],
                "permuted_macro_auc": perm_metrics["macro_auc"],
                "delta_macro_auc": baseline["macro_auc"] - perm_metrics["macro_auc"],
                "baseline_macro_f1": baseline["macro_f1"],
                "permuted_macro_f1": perm_metrics["macro_f1"],
                "delta_macro_f1": baseline["macro_f1"] - perm_metrics["macro_f1"],
                "baseline_balanced_accuracy": baseline["balanced_accuracy"],
                "permuted_balanced_accuracy": perm_metrics["balanced_accuracy"],
                "delta_balanced_accuracy": (
                    baseline["balanced_accuracy"] - perm_metrics["balanced_accuracy"]
                ),
            })

            for c in range(n_classes):
                class_rows.append({
                    **common,
                    "class_index": int(c),
                    "class_name": int_to_label.get(c, str(c)),
                    "baseline_auc_ovr": baseline["class_auc"][c],
                    "permuted_auc_ovr": perm_metrics["class_auc"][c],
                    "delta_auc_ovr": (
                        baseline["class_auc"][c] - perm_metrics["class_auc"][c]
                    ),
                    "baseline_recall": baseline["class_recall"][c],
                    "permuted_recall": perm_metrics["class_recall"][c],
                    "delta_recall": (
                        baseline["class_recall"][c] - perm_metrics["class_recall"][c]
                    ),
                    "baseline_mean_true_class_probability": baseline["class_true_prob"][c],
                    "permuted_mean_true_class_probability": perm_metrics["class_true_prob"][c],
                    "delta_mean_true_class_probability": (
                        baseline["class_true_prob"][c] - perm_metrics["class_true_prob"][c]
                    ),
                })

    return global_rows, class_rows


def _summary_ci(mean: pd.Series, std: pd.Series, n: pd.Series) -> Tuple[pd.Series, pd.Series]:
    se = std.fillna(0.0) / np.sqrt(np.maximum(n.astype(float), 1.0))
    return mean - 1.96 * se, mean + 1.96 * se


def summarize_permutation_importance(
    global_df: pd.DataFrame,
    class_df: pd.DataFrame,
    out_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dir(out_dir)

    if global_df.empty:
        global_summary = pd.DataFrame()
    else:
        grouping = [
            "mode", "mode_label", "feature_index", "feature", "feature_type",
            "modality", "modality_label"
        ]
        global_summary = global_df.groupby(grouping, as_index=False).agg(
            n_evaluations=("delta_macro_auc", "size"),
            n_outer_folds_evaluated=("outer_fold_id", "nunique"),
            selected_fraction=("selected_in_model", "mean"),
            mean_training_mi=("training_mi_score", "mean"),
            mean_delta_macro_auc=("delta_macro_auc", "mean"),
            std_delta_macro_auc=("delta_macro_auc", "std"),
            median_delta_macro_auc=("delta_macro_auc", "median"),
            positive_fraction_delta_macro_auc=("delta_macro_auc", lambda x: float(np.mean(np.asarray(x) > 0))),
            mean_delta_macro_f1=("delta_macro_f1", "mean"),
            std_delta_macro_f1=("delta_macro_f1", "std"),
            mean_delta_balanced_accuracy=("delta_balanced_accuracy", "mean"),
            std_delta_balanced_accuracy=("delta_balanced_accuracy", "std"),
        )
        low, high = _summary_ci(
            global_summary["mean_delta_macro_auc"],
            global_summary["std_delta_macro_auc"],
            global_summary["n_evaluations"],
        )
        global_summary["ci95_low_delta_macro_auc"] = low
        global_summary["ci95_high_delta_macro_auc"] = high
        global_summary = global_summary.sort_values(
            ["mode", "mean_delta_macro_auc"], ascending=[True, False]
        )

    if class_df.empty:
        class_summary = pd.DataFrame()
    else:
        grouping = [
            "mode", "mode_label", "feature_index", "feature", "feature_type",
            "modality", "modality_label", "class_index", "class_name"
        ]
        class_summary = class_df.groupby(grouping, as_index=False).agg(
            n_evaluations=("delta_auc_ovr", "size"),
            selected_fraction=("selected_in_model", "mean"),
            mean_training_mi=("training_mi_score", "mean"),
            mean_delta_auc_ovr=("delta_auc_ovr", "mean"),
            std_delta_auc_ovr=("delta_auc_ovr", "std"),
            median_delta_auc_ovr=("delta_auc_ovr", "median"),
            positive_fraction_delta_auc=("delta_auc_ovr", lambda x: float(np.mean(np.asarray(x) > 0))),
            mean_delta_recall=("delta_recall", "mean"),
            std_delta_recall=("delta_recall", "std"),
            mean_delta_true_class_probability=("delta_mean_true_class_probability", "mean"),
            std_delta_true_class_probability=("delta_mean_true_class_probability", "std"),
        )
        low, high = _summary_ci(
            class_summary["mean_delta_auc_ovr"],
            class_summary["std_delta_auc_ovr"],
            class_summary["n_evaluations"],
        )
        class_summary["ci95_low_delta_auc_ovr"] = low
        class_summary["ci95_high_delta_auc_ovr"] = high
        class_summary = class_summary.sort_values(
            ["mode", "feature", "class_index"]
        )

    global_df.to_csv(
        os.path.join(out_dir, "outer_test_global_permutation_importance_fold_level.csv"),
        index=False, encoding="utf-8-sig"
    )
    class_df.to_csv(
        os.path.join(out_dir, "outer_test_class_specific_permutation_importance_fold_level.csv"),
        index=False, encoding="utf-8-sig"
    )
    global_summary.to_csv(
        os.path.join(out_dir, "outer_test_global_permutation_importance_summary.csv"),
        index=False, encoding="utf-8-sig"
    )
    class_summary.to_csv(
        os.path.join(out_dir, "outer_test_class_specific_permutation_importance_summary.csv"),
        index=False, encoding="utf-8-sig"
    )
    return global_summary, class_summary


def normalize_feature_name_for_overlap(feature: Any) -> str:
    """Return the biological feature name used for cross-mode matching.

    Predicted features are stored as ``pred_<real feature name>``. The prefix
    is removed only for overlap comparisons, so for example ``pred_HIP.L`` is
    matched to ``HIP.L`` and ``pred_g_Blautia_wexlerae`` is matched to
    ``g_Blautia_wexlerae``.
    """
    name = str(feature).strip()
    while name.lower().startswith("pred_"):
        name = name[5:]
    return name


def biological_modality_from_feature_type(feature_type: Any) -> str:
    value = str(feature_type).strip().lower()
    if "brain" in value or "roi" in value:
        return "brain"
    if "micro" in value or "taxa" in value or "gut" in value:
        return "microbial"
    return "other"


def _top_global_importance_features(
    global_summary: pd.DataFrame,
    modes: List[str],
    top_n: int,
    biological_modality: Optional[str] = None,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for mode in modes:
        d = global_summary[global_summary["mode"].astype(str) == str(mode)].copy()
        if biological_modality is not None:
            d["biological_modality"] = d["feature_type"].map(
                biological_modality_from_feature_type
            )
            d = d[d["biological_modality"] == biological_modality].copy()
        else:
            d["biological_modality"] = d["feature_type"].map(
                biological_modality_from_feature_type
            )

        d["mean_delta_macro_auc"] = pd.to_numeric(
            d["mean_delta_macro_auc"], errors="coerce"
        )
        d = d[np.isfinite(d["mean_delta_macro_auc"])].copy()
        d = d.sort_values("mean_delta_macro_auc", ascending=False).head(int(top_n))
        d["rank_within_mode"] = np.arange(1, len(d) + 1, dtype=int)
        d["normalized_feature"] = d["feature"].map(
            normalize_feature_name_for_overlap
        )
        d["is_predicted_feature"] = d["feature"].astype(str).str.lower().str.startswith(
            "pred_"
        ).astype(int)
        rows.append(d)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _pairwise_overlap_table(
    top_df: pd.DataFrame,
    modes: List[str],
    scope: str,
    top_n: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    sets = {
        mode: set(
            top_df.loc[
                top_df["mode"].astype(str) == str(mode), "normalized_feature"
            ].astype(str)
        )
        for mode in modes
    }

    for mode_a, mode_b in combinations(modes, 2):
        shared = sorted(sets[mode_a] & sets[mode_b])
        union = sets[mode_a] | sets[mode_b]
        smaller = min(len(sets[mode_a]), len(sets[mode_b]))
        rows.append({
            "scope": scope,
            "top_n_requested": int(top_n),
            "mode_a": mode_a,
            "mode_b": mode_b,
            "n_mode_a": int(len(sets[mode_a])),
            "n_mode_b": int(len(sets[mode_b])),
            "n_overlap": int(len(shared)),
            "jaccard_index": (
                float(len(shared) / len(union)) if union else np.nan
            ),
            "overlap_coefficient": (
                float(len(shared) / smaller) if smaller > 0 else np.nan
            ),
            "shared_features": " | ".join(shared),
        })
    return pd.DataFrame(rows)


def analyze_global_importance_overlap(
    global_summary: pd.DataFrame,
    out_dir: str,
    modes: Optional[List[str]] = None,
    top_n: int = 10,
) -> Dict[str, pd.DataFrame]:
    """Compare global top features across the three fusion modes.

    Two complementary comparisons are written:
    1. Overall top-N features in each mode.
    2. Top-N brain features and top-N microbial features separately.

    During matching, only the leading ``pred_`` prefix is removed. Thus a
    predicted ROI/taxon is compared with the corresponding real ROI/taxon from
    ``real_fusion`` while the original feature names are retained in the detail
    tables.
    """
    ensure_dir(out_dir)
    if global_summary.empty:
        print("[OVERLAP] Global importance summary is empty; overlap skipped.")
        return {}

    required = {"mode", "feature", "feature_type", "mean_delta_macro_auc"}
    missing = required - set(global_summary.columns)
    if missing:
        raise ValueError(
            "Cannot calculate top-feature overlap; missing columns: "
            + ", ".join(sorted(missing))
        )

    if modes is None:
        modes = list(FUSION_MODES)
    available = set(global_summary["mode"].astype(str).unique())
    missing_modes = [mode for mode in modes if mode not in available]
    if missing_modes:
        raise ValueError(
            "Global importance overlap requires all requested modes. Missing: "
            f"{missing_modes}; available modes: {sorted(available)}"
        )

    outputs: Dict[str, pd.DataFrame] = {}

    # A. Overall top-N comparison.
    overall = _top_global_importance_features(
        global_summary, modes=modes, top_n=top_n, biological_modality=None
    )
    overall_columns = [
        "mode", "mode_label", "rank_within_mode", "feature",
        "normalized_feature", "is_predicted_feature", "feature_type",
        "biological_modality", "mean_delta_macro_auc",
        "ci95_low_delta_macro_auc", "ci95_high_delta_macro_auc",
        "positive_fraction_delta_macro_auc", "n_outer_folds_evaluated",
    ]
    overall_columns = [c for c in overall_columns if c in overall.columns]
    overall_detail = overall[overall_columns].copy()
    overall_detail.to_csv(
        os.path.join(out_dir, "global_importance_top10_overall_by_mode.csv"),
        index=False, encoding="utf-8-sig"
    )
    outputs["overall_detail"] = overall_detail

    overall_pairwise = _pairwise_overlap_table(
        overall, modes=modes, scope="overall", top_n=top_n
    )
    overall_pairwise.to_csv(
        os.path.join(out_dir, "global_importance_top10_overall_pairwise_overlap.csv"),
        index=False, encoding="utf-8-sig"
    )
    outputs["overall_pairwise"] = overall_pairwise

    overall_sets = {
        mode: set(
            overall.loc[
                overall["mode"].astype(str) == str(mode), "normalized_feature"
            ].astype(str)
        )
        for mode in modes
    }
    three_way = sorted(set.intersection(*(overall_sets[m] for m in modes)))
    three_way_df = pd.DataFrame([{
        "scope": "overall",
        "top_n_requested": int(top_n),
        "modes": " | ".join(modes),
        "n_three_way_overlap": int(len(three_way)),
        "shared_features": " | ".join(three_way),
    }])
    three_way_df.to_csv(
        os.path.join(out_dir, "global_importance_top10_overall_three_way_overlap.csv"),
        index=False, encoding="utf-8-sig"
    )
    outputs["overall_three_way"] = three_way_df

    # Membership/rank matrix for convenient inspection.
    membership_rows: List[Dict[str, Any]] = []
    all_features = sorted(set().union(*(overall_sets[m] for m in modes)))
    for feature in all_features:
        row: Dict[str, Any] = {"normalized_feature": feature}
        for mode in modes:
            hit = overall[
                (overall["mode"].astype(str) == str(mode))
                & (overall["normalized_feature"].astype(str) == feature)
            ]
            row[f"in_{mode}"] = int(not hit.empty)
            row[f"rank_{mode}"] = (
                int(hit["rank_within_mode"].iloc[0]) if not hit.empty else np.nan
            )
            row[f"original_name_{mode}"] = (
                str(hit["feature"].iloc[0]) if not hit.empty else ""
            )
            row[f"importance_{mode}"] = (
                float(hit["mean_delta_macro_auc"].iloc[0]) if not hit.empty else np.nan
            )
        membership_rows.append(row)
    membership = pd.DataFrame(membership_rows)
    membership.to_csv(
        os.path.join(out_dir, "global_importance_top10_overall_membership.csv"),
        index=False, encoding="utf-8-sig"
    )
    outputs["overall_membership"] = membership

    # B. Brain and microbial top-N comparisons separately. This is useful when
    # the publication plot shows top-N features per modality.
    per_modality_details: List[pd.DataFrame] = []
    per_modality_pairwise: List[pd.DataFrame] = []
    per_modality_three_way: List[Dict[str, Any]] = []
    for biological_modality in ["brain", "microbial"]:
        d_modality = _top_global_importance_features(
            global_summary,
            modes=modes,
            top_n=top_n,
            biological_modality=biological_modality,
        )
        d_modality["overlap_scope"] = biological_modality
        per_modality_details.append(d_modality)

        pairwise = _pairwise_overlap_table(
            d_modality,
            modes=modes,
            scope=biological_modality,
            top_n=top_n,
        )
        per_modality_pairwise.append(pairwise)

        modality_sets = {
            mode: set(
                d_modality.loc[
                    d_modality["mode"].astype(str) == str(mode),
                    "normalized_feature",
                ].astype(str)
            )
            for mode in modes
        }
        shared = sorted(set.intersection(*(modality_sets[m] for m in modes)))
        per_modality_three_way.append({
            "scope": biological_modality,
            "top_n_requested": int(top_n),
            "modes": " | ".join(modes),
            "n_three_way_overlap": int(len(shared)),
            "shared_features": " | ".join(shared),
        })

    per_modality_detail = pd.concat(per_modality_details, ignore_index=True)
    detail_columns = [
        "overlap_scope", "mode", "mode_label", "rank_within_mode",
        "feature", "normalized_feature", "is_predicted_feature",
        "feature_type", "biological_modality", "mean_delta_macro_auc",
        "ci95_low_delta_macro_auc", "ci95_high_delta_macro_auc",
        "positive_fraction_delta_macro_auc", "n_outer_folds_evaluated",
    ]
    detail_columns = [c for c in detail_columns if c in per_modality_detail.columns]
    per_modality_detail = per_modality_detail[detail_columns]
    per_modality_detail.to_csv(
        os.path.join(out_dir, "global_importance_top10_per_modality_by_mode.csv"),
        index=False, encoding="utf-8-sig"
    )
    outputs["per_modality_detail"] = per_modality_detail

    modality_pairwise_df = pd.concat(per_modality_pairwise, ignore_index=True)
    modality_pairwise_df.to_csv(
        os.path.join(out_dir, "global_importance_top10_per_modality_pairwise_overlap.csv"),
        index=False, encoding="utf-8-sig"
    )
    outputs["per_modality_pairwise"] = modality_pairwise_df

    modality_three_way_df = pd.DataFrame(per_modality_three_way)
    modality_three_way_df.to_csv(
        os.path.join(out_dir, "global_importance_top10_per_modality_three_way_overlap.csv"),
        index=False, encoding="utf-8-sig"
    )
    outputs["per_modality_three_way"] = modality_three_way_df

    print("[OVERLAP] Pairwise overlap of overall top features:")
    for _, row in overall_pairwise.iterrows():
        print(
            f"  {row['mode_a']} vs {row['mode_b']}: "
            f"{int(row['n_overlap'])}/{top_n} shared"
        )
    print(
        "[OVERLAP] Three-way overlap of overall top features:",
        f"{len(three_way)}/{top_n}",
    )
    print("[OVERLAP] Results saved to:", out_dir)
    return outputs


def standardized_effect_size(x_class: np.ndarray, x_rest: np.ndarray) -> Tuple[float, float]:
    x1 = np.asarray(x_class, dtype=float)
    x0 = np.asarray(x_rest, dtype=float)
    n1, n0 = len(x1), len(x0)
    if n1 < 2 or n0 < 2:
        return np.nan, np.nan
    v1 = float(np.var(x1, ddof=1))
    v0 = float(np.var(x0, ddof=1))
    denom_df = n1 + n0 - 2
    if denom_df <= 0:
        return np.nan, np.nan
    pooled_var = ((n1 - 1) * v1 + (n0 - 1) * v0) / denom_df
    if pooled_var <= 0 or not np.isfinite(pooled_var):
        return 0.0, 0.0
    d = (float(np.mean(x1)) - float(np.mean(x0))) / np.sqrt(pooled_var)
    correction = 1.0 - 3.0 / max(4.0 * (n1 + n0) - 9.0, 1.0)
    return float(d), float(correction * d)


def effect_direction(mean_difference: float, tol: float = 1e-12) -> str:
    if mean_difference > tol:
        return "higher_in_target_class"
    if mean_difference < -tol:
        return "lower_in_target_class"
    return "approximately_equal"


def compute_fullsample_class_specific_descriptive(
    data: Dict[str, np.ndarray],
    y: np.ndarray,
    feature_names: Dict[str, List[str]],
    feature_types: Dict[str, List[str]],
    int_to_label: Dict[int, str],
    modes: List[str],
    out_dir: str,
) -> pd.DataFrame:
    """Full-sample descriptive analysis, separate from predictive validation."""
    ensure_dir(out_dir)
    n_classes = int(CONFIG["N_CLASSES"])
    rows: List[Dict[str, Any]] = []

    for mode in modes:
        X = np.asarray(data[mode], dtype=float)
        scaler = StandardScaler()
        Xz = scaler.fit_transform(X)
        names = feature_names[mode]
        types = feature_types[mode]

        for c in range(n_classes):
            y_bin = (y == c).astype(int)
            mi = mutual_info_classif(
                Xz,
                y_bin,
                random_state=int(CONFIG.get("IMPORTANCE_RANDOM_STATE", 2026)) + c,
                n_neighbors=int(CONFIG["MI_N_NEIGHBORS"]),
            )
            class_mask = y == c
            rest_mask = ~class_mask

            p_t_values = np.full(X.shape[1], np.nan, dtype=float)
            p_u_values = np.full(X.shape[1], np.nan, dtype=float)

            if ttest_ind is not None:
                t_result = ttest_ind(
                    X[class_mask], X[rest_mask], axis=0,
                    equal_var=False, nan_policy="omit"
                )
                p_t_values = np.asarray(t_result.pvalue, dtype=float)

            if mannwhitneyu is not None:
                for j in range(X.shape[1]):
                    try:
                        p_u_values[j] = float(mannwhitneyu(
                            X[class_mask, j], X[rest_mask, j],
                            alternative="two-sided"
                        ).pvalue)
                    except Exception:
                        p_u_values[j] = np.nan

            q_t = benjamini_hochberg(p_t_values)
            q_u = benjamini_hochberg(p_u_values)

            for j, name in enumerate(names):
                x_class = X[class_mask, j]
                x_rest = X[rest_mask, j]
                mean_class = float(np.mean(x_class))
                mean_rest = float(np.mean(x_rest))
                mean_diff = mean_class - mean_rest
                cohen_d, hedges_g = standardized_effect_size(x_class, x_rest)
                try:
                    auc_raw = float(roc_auc_score(y_bin, X[:, j]))
                    auc_discrimination = max(auc_raw, 1.0 - auc_raw)
                except Exception:
                    auc_raw = np.nan
                    auc_discrimination = np.nan

                rows.append({
                    "mode": mode,
                    "mode_label": MODE_INFO[mode]["label"],
                    "feature_index": int(j),
                    "feature": str(name),
                    "feature_type": str(types[j]),
                    "class_index": int(c),
                    "class_name": int_to_label.get(c, str(c)),
                    "one_vs_rest_mi": float(mi[j]),
                    "mean_target_class": mean_class,
                    "mean_other_classes": mean_rest,
                    "mean_difference_target_minus_rest": mean_diff,
                    "effect_direction": effect_direction(mean_diff),
                    "cohen_d_target_vs_rest": cohen_d,
                    "hedges_g_target_vs_rest": hedges_g,
                    "single_feature_auc_raw": auc_raw,
                    "single_feature_auc_discrimination": auc_discrimination,
                    "welch_t_p": p_t_values[j],
                    "welch_t_fdr_q": q_t[j],
                    "mannwhitney_p": p_u_values[j],
                    "mannwhitney_fdr_q": q_u[j],
                    "n_target_class": int(np.sum(class_mask)),
                    "n_other_classes": int(np.sum(rest_mask)),
                    "analysis_scope": "full_sample_descriptive_not_outer_test_importance",
                })

    out = pd.DataFrame(rows)
    out.to_csv(
        os.path.join(out_dir, "class_specific_one_vs_rest_mi_mean_effect_fullsample.csv"),
        index=False, encoding="utf-8-sig"
    )
    return out


def compute_fullsample_pairwise_comparisons(
    data: Dict[str, np.ndarray],
    y: np.ndarray,
    feature_names: Dict[str, List[str]],
    feature_types: Dict[str, List[str]],
    int_to_label: Dict[int, str],
    modes: List[str],
    out_dir: str,
) -> pd.DataFrame:
    ensure_dir(out_dir)
    rows: List[Dict[str, Any]] = []
    n_classes = int(CONFIG["N_CLASSES"])

    for mode in modes:
        X = np.asarray(data[mode], dtype=float)
        names = feature_names[mode]
        types = feature_types[mode]

        for class_a, class_b in combinations(range(n_classes), 2):
            mask_a = y == class_a
            mask_b = y == class_b
            p_t_values = np.full(X.shape[1], np.nan, dtype=float)
            p_u_values = np.full(X.shape[1], np.nan, dtype=float)

            if ttest_ind is not None:
                t_result = ttest_ind(
                    X[mask_a], X[mask_b], axis=0,
                    equal_var=False, nan_policy="omit"
                )
                p_t_values = np.asarray(t_result.pvalue, dtype=float)

            if mannwhitneyu is not None:
                for j in range(X.shape[1]):
                    try:
                        p_u_values[j] = float(mannwhitneyu(
                            X[mask_a, j], X[mask_b, j],
                            alternative="two-sided"
                        ).pvalue)
                    except Exception:
                        p_u_values[j] = np.nan

            q_t = benjamini_hochberg(p_t_values)
            q_u = benjamini_hochberg(p_u_values)

            y_pair = np.concatenate([
                np.ones(int(np.sum(mask_a)), dtype=int),
                np.zeros(int(np.sum(mask_b)), dtype=int),
            ])

            for j, name in enumerate(names):
                xa = X[mask_a, j]
                xb = X[mask_b, j]
                mean_a = float(np.mean(xa))
                mean_b = float(np.mean(xb))
                mean_diff = mean_a - mean_b
                cohen_d, hedges_g = standardized_effect_size(xa, xb)
                values_pair = np.concatenate([xa, xb])
                try:
                    auc_raw = float(roc_auc_score(y_pair, values_pair))
                    auc_discrimination = max(auc_raw, 1.0 - auc_raw)
                except Exception:
                    auc_raw = np.nan
                    auc_discrimination = np.nan

                rows.append({
                    "mode": mode,
                    "mode_label": MODE_INFO[mode]["label"],
                    "feature_index": int(j),
                    "feature": str(name),
                    "feature_type": str(types[j]),
                    "class_a_index": int(class_a),
                    "class_a_name": int_to_label.get(class_a, str(class_a)),
                    "class_b_index": int(class_b),
                    "class_b_name": int_to_label.get(class_b, str(class_b)),
                    "class_pair": (
                        f"{int_to_label.get(class_a, str(class_a))} vs "
                        f"{int_to_label.get(class_b, str(class_b))}"
                    ),
                    "mean_class_a": mean_a,
                    "mean_class_b": mean_b,
                    "mean_difference_a_minus_b": mean_diff,
                    "effect_direction": (
                        "higher_in_class_a" if mean_diff > 0 else
                        "lower_in_class_a" if mean_diff < 0 else
                        "approximately_equal"
                    ),
                    "cohen_d_a_vs_b": cohen_d,
                    "hedges_g_a_vs_b": hedges_g,
                    "single_feature_auc_raw_class_a_positive": auc_raw,
                    "single_feature_auc_discrimination": auc_discrimination,
                    "welch_t_p": p_t_values[j],
                    "welch_t_fdr_q": q_t[j],
                    "mannwhitney_p": p_u_values[j],
                    "mannwhitney_fdr_q": q_u[j],
                    "n_class_a": int(np.sum(mask_a)),
                    "n_class_b": int(np.sum(mask_b)),
                    "analysis_scope": "full_sample_pairwise_descriptive",
                })

    out = pd.DataFrame(rows)
    out.to_csv(
        os.path.join(out_dir, "pairwise_class_feature_comparisons_fullsample.csv"),
        index=False, encoding="utf-8-sig"
    )
    return out


def _select_top_features_for_heatmap(
    d: pd.DataFrame,
    score_column: str,
    topk: int,
) -> List[str]:
    if d.empty:
        return []
    score = d.groupby(["feature", "feature_type"])[score_column].apply(
        lambda x: float(np.nanmax(np.abs(np.asarray(x, dtype=float))))
    ).reset_index(name="score")
    feature_types = list(score["feature_type"].drop_duplicates())
    if not feature_types:
        return []
    per_type = max(1, int(np.ceil(topk / len(feature_types))))
    parts = []
    for ft in feature_types:
        parts.append(
            score[score["feature_type"] == ft]
            .sort_values("score", ascending=False)
            .head(per_type)
        )
    selected = pd.concat(parts, ignore_index=True)
    selected = selected.sort_values("score", ascending=False).head(topk)
    return selected["feature"].astype(str).tolist()


def _plot_matrix_heatmap(
    matrix: pd.DataFrame,
    title: str,
    cbar_label: str,
    out_prefix: str,
    diverging: bool,
    annotation_format: str = ".3f",
):
    if matrix.empty:
        return
    set_plot_style()
    values = matrix.values.astype(float)
    fig_height = max(5.5, 0.30 * matrix.shape[0] + 2.0)
    fig, ax = plt.subplots(figsize=(8.4, fig_height))

    if diverging:
        vmax = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
        vmax = max(vmax, 1e-12)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cmap = LinearSegmentedColormap.from_list(
            "importance_diverging",
            ["#5D8FCE", "#D7E5F5", "#FFFFFF", "#F3D0C8", "#CC6A57"],
        )
        im = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm)
    else:
        cmap = LinearSegmentedColormap.from_list(
            "importance_positive",
            ["#F7FBFF", "#D7E8F6", "#9CCBE6", "#4C78A8"],
        )
        im = ax.imshow(values, aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(matrix.columns, rotation=25, ha="right")
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_title(title, fontsize=14, pad=10)

    if matrix.shape[0] <= 35 and matrix.shape[1] <= 8:
        threshold = np.nanmax(np.abs(values)) * 0.55 if np.isfinite(values).any() else np.inf
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = values[i, j]
                if not np.isfinite(v):
                    label = "NA"
                else:
                    label = format(v, annotation_format)
                ax.text(
                    j, i, label,
                    ha="center", va="center", fontsize=7.5,
                    color="white" if np.isfinite(v) and abs(v) >= threshold else "black",
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_prefix + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(out_prefix + ".pdf", bbox_inches="tight")
    fig.savefig(out_prefix + ".svg", bbox_inches="tight")
    plt.close(fig)


def plot_global_permutation_importance(global_summary: pd.DataFrame, out_dir: str):
    if global_summary.empty:
        return
    ensure_dir(out_dir)
    topk = int(CONFIG.get("IMPORTANCE_PLOT_TOPK", 30))
    set_plot_style()

    for mode, d_mode in global_summary.groupby("mode"):
        d = d_mode.sort_values("mean_delta_macro_auc", ascending=False).head(topk)
        d = d.sort_values("mean_delta_macro_auc", ascending=True)
        fig, ax = plt.subplots(figsize=(9.4, max(5.5, 0.31 * len(d) + 1.8)))
        ax.barh(
            np.arange(len(d)),
            d["mean_delta_macro_auc"].values,
            xerr=d["std_delta_macro_auc"].fillna(0).values,
            capsize=2.5,
            color=MODE_INFO[mode]["color"],
            edgecolor="white",
            linewidth=0.6,
        )
        labels = [
            f"{f} [{ft}]" for f, ft in zip(d["feature"], d["feature_type"])
        ]
        ax.set_yticks(np.arange(len(d)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="0.35", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Decrease in outer-test macro-AUC after permutation")
        ax.set_title(f"{MODE_INFO[mode]['label']}: predictive contribution importance")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        prefix = os.path.join(out_dir, f"{mode}_global_permutation_importance")
        fig.savefig(prefix + ".png", dpi=600, bbox_inches="tight")
        fig.savefig(prefix + ".pdf", bbox_inches="tight")
        fig.savefig(prefix + ".svg", bbox_inches="tight")
        plt.close(fig)


def plot_class_specific_permutation_importance(
    class_summary: pd.DataFrame,
    out_dir: str,
):
    if class_summary.empty:
        return
    ensure_dir(out_dir)
    topk = int(CONFIG.get("IMPORTANCE_PLOT_TOPK", 30))
    for mode, d_mode in class_summary.groupby("mode"):
        features = _select_top_features_for_heatmap(
            d_mode, "mean_delta_auc_ovr", topk
        )
        d = d_mode[d_mode["feature"].isin(features)].copy()
        matrix = d.pivot_table(
            index="feature", columns="class_name",
            values="mean_delta_auc_ovr", aggfunc="mean"
        )
        order_score = matrix.abs().max(axis=1).sort_values(ascending=False)
        matrix = matrix.loc[order_score.index]
        _plot_matrix_heatmap(
            matrix=matrix,
            title=(
                f"{MODE_INFO[mode]['label']}: class-specific outer-test "
                "permutation importance"
            ),
            cbar_label="Decrease in one-vs-rest AUC after permutation",
            out_prefix=os.path.join(
                out_dir, f"{mode}_class_specific_permutation_delta_auc_heatmap"
            ),
            diverging=True,
        )


def plot_class_specific_descriptive(
    descriptive_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    out_dir: str,
):
    if descriptive_df.empty:
        return
    ensure_dir(out_dir)
    topk = int(CONFIG.get("IMPORTANCE_PLOT_TOPK", 30))

    for mode, d_mode in descriptive_df.groupby("mode"):
        features_mi = _select_top_features_for_heatmap(
            d_mode, "one_vs_rest_mi", topk
        )
        d_mi = d_mode[d_mode["feature"].isin(features_mi)]
        mi_matrix = d_mi.pivot_table(
            index="feature", columns="class_name",
            values="one_vs_rest_mi", aggfunc="mean"
        )
        mi_order = mi_matrix.max(axis=1).sort_values(ascending=False)
        mi_matrix = mi_matrix.loc[mi_order.index]
        _plot_matrix_heatmap(
            mi_matrix,
            f"{MODE_INFO[mode]['label']}: one-vs-rest mutual information",
            "One-vs-rest mutual information",
            os.path.join(out_dir, f"{mode}_one_vs_rest_mi_heatmap"),
            diverging=False,
        )

        features_effect = _select_top_features_for_heatmap(
            d_mode, "hedges_g_target_vs_rest", topk
        )
        d_eff = d_mode[d_mode["feature"].isin(features_effect)]
        effect_matrix = d_eff.pivot_table(
            index="feature", columns="class_name",
            values="hedges_g_target_vs_rest", aggfunc="mean"
        )
        effect_order = effect_matrix.abs().max(axis=1).sort_values(ascending=False)
        effect_matrix = effect_matrix.loc[effect_order.index]
        _plot_matrix_heatmap(
            effect_matrix,
            f"{MODE_INFO[mode]['label']}: class mean difference and direction",
            "Hedges' g: target class minus other classes",
            os.path.join(out_dir, f"{mode}_one_vs_rest_effect_direction_heatmap"),
            diverging=True,
        )

    if pairwise_df is not None and not pairwise_df.empty:
        for mode, d_mode in pairwise_df.groupby("mode"):
            features = _select_top_features_for_heatmap(
                d_mode, "hedges_g_a_vs_b", topk
            )
            d = d_mode[d_mode["feature"].isin(features)]
            matrix = d.pivot_table(
                index="feature", columns="class_pair",
                values="hedges_g_a_vs_b", aggfunc="mean"
            )
            order_score = matrix.abs().max(axis=1).sort_values(ascending=False)
            matrix = matrix.loc[order_score.index]
            _plot_matrix_heatmap(
                matrix,
                f"{MODE_INFO[mode]['label']}: pairwise class effect sizes",
                "Hedges' g: class A minus class B",
                os.path.join(out_dir, f"{mode}_pairwise_class_effect_heatmap"),
                diverging=True,
            )


# ============================================================
# Selected-feature and kernel-weight summaries
# ============================================================
def extract_selected_feature_rows(
    result: Dict[str, Any],
    mode: str,
    repeat: int,
    fold: int,
    feature_names: Dict[str, List[str]],
    feature_types: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    rows = []
    full_names = feature_names[mode]
    full_types = feature_types[mode]

    for fitted in result["preprocessors"]:
        global_cols = fitted["global_columns"]
        selected_local = fitted["selected_original_local"]
        selected_global = global_cols[selected_local]
        scores = fitted["mi_scores_selected"]

        for gi, score in zip(selected_global, scores):
            rows.append({
                "repeat": int(repeat),
                "fold": int(fold),
                "mode": mode,
                "mode_label": MODE_INFO[mode]["label"],
                "modality": fitted["modality_name"],
                "modality_label": fitted["modality_label"],
                "feature_index": int(gi),
                "feature": str(full_names[int(gi)]),
                "feature_type": str(full_types[int(gi)]),
                "selected": 1,
                "mi_score": float(score),
            })
    return rows


def summarize_selected_features(
    selected_df: pd.DataFrame,
    feature_names: Dict[str, List[str]],
    feature_types: Dict[str, List[str]],
    dims: Dict[str, int],
    active_modes: List[str],
    n_total_folds_per_mode: Dict[str, int],
) -> pd.DataFrame:
    grouped = None
    if not selected_df.empty:
        grouped = selected_df.groupby(["mode", "feature_index"], as_index=False).agg(
            selected_count=("selected", "sum"),
            mean_mi_when_selected=("mi_score", "mean"),
            median_mi_when_selected=("mi_score", "median"),
            max_mi=("mi_score", "max"),
        ).set_index(["mode", "feature_index"])

    rows = []
    for mode in active_modes:
        specs = get_mode_modality_specs(mode, dims)
        n_total = int(n_total_folds_per_mode.get(mode, 0))
        for gi, fname in enumerate(feature_names[mode]):
            modality = "unknown"
            modality_label = feature_types[mode][gi]
            for spec in specs:
                if spec["start"] <= gi < spec["end"]:
                    modality = spec["name"]
                    modality_label = spec["label"]
                    break

            if grouped is not None and (mode, gi) in grouped.index:
                g = grouped.loc[(mode, gi)]
                count = int(g["selected_count"])
                mean_mi = float(g["mean_mi_when_selected"])
                med_mi = float(g["median_mi_when_selected"])
                max_mi = float(g["max_mi"])
            else:
                count = 0
                mean_mi = med_mi = max_mi = 0.0

            rows.append({
                "mode": mode,
                "mode_label": MODE_INFO[mode]["label"],
                "modality": modality,
                "modality_label": modality_label,
                "feature_index": gi,
                "feature": str(fname),
                "feature_type": str(feature_types[mode][gi]),
                "n_total_outer_folds": n_total,
                "selected_count": count,
                "selected_frequency": float(count / n_total) if n_total else 0.0,
                "mean_mi_when_selected": mean_mi,
                "median_mi_when_selected": med_mi,
                "max_mi": max_mi,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["mode", "selected_frequency", "mean_mi_when_selected"],
            ascending=[True, False, False],
        )
    return out


# ============================================================
# Metrics
# ============================================================
def macro_roc_curve_from_prob(y_true, prob, n_classes, roc_points=300):
    fpr_grid = np.linspace(0, 1, roc_points)
    tprs = []
    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        fpr, tpr, _ = roc_curve(y_bin, prob[:, c])
        tpr_interp = np.interp(fpr_grid, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
    if not tprs:
        return fpr_grid, np.zeros_like(fpr_grid), 0.0
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    return fpr_grid, mean_tpr, float(auc(fpr_grid, mean_tpr))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, prob: np.ndarray, n_classes: int):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    sens, spec = [], []
    for c in range(n_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens.append(tp / (tp + fn + 1e-12))
        spec.append(tn / (tn + fp + 1e-12))
    fpr, tpr, macro_auc = macro_roc_curve_from_prob(
        y_true, prob, n_classes, int(CONFIG["ROC_POINTS"])
    )
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "F1-score": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Sensitivity": float(np.mean(sens)),
        "Specificity": float(np.mean(spec)),
        "AUC": float(macro_auc),
        "fpr": fpr,
        "tpr": tpr,
    }


# ============================================================
# Repeated full-sample nested CV
# ============================================================
def run_repeated_fullsample_cv_schemeA(
    data: Dict[str, np.ndarray],
    y: np.ndarray,
    sids: List[str],
    int_to_label: Dict[int, str],
    feature_names: Dict[str, List[str]],
    feature_types: Dict[str, List[str]],
    dims: Dict[str, int],
):
    out_dir = CONFIG["OUT_DIR"]
    ensure_dir(out_dir)
    n_classes = int(CONFIG["N_CLASSES"])
    n_repeats = int(CONFIG["N_REPEATS"])
    n_splits = int(CONFIG["N_SPLITS"])
    base_seed = int(CONFIG["SEED"])
    active_modes = get_active_modes()

    all_pred_rows = []
    repeat_metric_rows = []
    split_rows = []
    best_param_rows = []
    selected_feature_rows = []
    kernel_weight_rows = []
    candidate_search_rows = []
    global_importance_rows = []
    class_importance_rows = []
    n_total_folds_per_mode = {mode: 0 for mode in active_modes}

    importance_modes = get_importance_modes(active_modes)
    importance_max_repeats = int(CONFIG.get("IMPORTANCE_MAX_REPEATS", 0))

    all_indices = np.arange(len(y), dtype=int)

    for rep in range(n_repeats):
        rep_seed = base_seed + rep
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rep_seed)
        repeat_store = {
            mode: {"y_true": [], "y_pred": [], "prob": []}
            for mode in active_modes
        }

        for fold, (train_idx, test_idx) in enumerate(
            skf.split(np.zeros(len(all_indices)), y), start=1
        ):
            for gi in test_idx:
                split_rows.append({
                    "repeat": rep + 1,
                    "fold": fold,
                    "global_index": int(gi),
                    "subject_id": sids[int(gi)],
                    "label_int": int(y[int(gi)]),
                    "label_name": int_to_label.get(int(y[int(gi)]), str(y[int(gi)])),
                })

            for mode in active_modes:
                fold_seed = rep_seed * 1000 + fold * 100 + active_modes.index(mode)
                result = train_one_fold_schemeA_mksvm(
                    X=data[mode],
                    y=y,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    n_classes=n_classes,
                    seed=fold_seed,
                    mode=mode,
                    dims=dims,
                )
                n_total_folds_per_mode[mode] += 1

                repeat_store[mode]["y_true"].append(result["y_true"])
                repeat_store[mode]["y_pred"].append(result["y_pred"])
                repeat_store[mode]["prob"].append(result["prob"])

                best_param_rows.append({
                    "repeat": rep + 1,
                    "fold": fold,
                    "mode": mode,
                    "mode_label": MODE_INFO[mode]["label"],
                    "refit_metric": result["refit_metric"],
                    "best_score": result["best_score"],
                    "best_params_json": json_dumps_safe(result["best_params"]),
                })

                for modality, weight in result["kernel_weights"].items():
                    kernel_weight_rows.append({
                        "repeat": rep + 1,
                        "fold": fold,
                        "mode": mode,
                        "mode_label": MODE_INFO[mode]["label"],
                        "modality": modality,
                        "kernel_weight": float(weight),
                    })

                for cand in result["candidate_rows"]:
                    candidate_search_rows.append({
                        "repeat": rep + 1,
                        "fold": fold,
                        "mode": mode,
                        **cand,
                    })

                selected_feature_rows.extend(
                    extract_selected_feature_rows(
                        result, mode, rep + 1, fold, feature_names, feature_types
                    )
                )

                # Outer-test predictive contribution importance. Candidate
                # selection uses only this outer training fold. Model fitting,
                # hyperparameter tuning and all original outputs are unchanged.
                should_run_importance = (
                    bool(CONFIG.get("MAKE_FEATURE_IMPORTANCE", True))
                    and mode in importance_modes
                    and (importance_max_repeats <= 0 or (rep + 1) <= importance_max_repeats)
                )
                if should_run_importance:
                    imp_seed = (
                        int(CONFIG.get("IMPORTANCE_RANDOM_STATE", 2026))
                        + (rep + 1) * 100000
                        + fold * 1000
                        + active_modes.index(mode) * 10
                    )
                    fold_global_imp, fold_class_imp = compute_outer_fold_permutation_importance(
                        result=result,
                        X_test_raw=data[mode][test_idx].astype(np.float64),
                        mode=mode,
                        repeat=rep + 1,
                        fold=fold,
                        feature_names_mode=feature_names[mode],
                        feature_types_mode=feature_types[mode],
                        int_to_label=int_to_label,
                        n_classes=n_classes,
                        seed=imp_seed,
                    )
                    global_importance_rows.extend(fold_global_imp)
                    class_importance_rows.extend(fold_class_imp)

                for pos, gi in enumerate(test_idx):
                    row = {
                        "repeat": rep + 1,
                        "fold": fold,
                        "mode": mode,
                        "mode_label": MODE_INFO[mode]["label"],
                        "global_index": int(gi),
                        "subject_id": sids[int(gi)],
                        "y_true": int(result["y_true"][pos]),
                        "y_pred": int(result["y_pred"][pos]),
                    }
                    for c in range(n_classes):
                        row[f"prob_c{c}"] = float(result["prob"][pos, c])
                    all_pred_rows.append(row)

        for mode in active_modes:
            yt = np.concatenate(repeat_store[mode]["y_true"])
            yp = np.concatenate(repeat_store[mode]["y_pred"])
            pp = np.concatenate(repeat_store[mode]["prob"])
            metrics = compute_metrics(yt, yp, pp, n_classes)
            for key, value in metrics.items():
                if key in {"fpr", "tpr"}:
                    continue
                repeat_metric_rows.append({
                    "repeat": rep + 1,
                    "mode": mode,
                    "mode_label": MODE_INFO[mode]["label"],
                    "metric": key,
                    "value": float(value),
                })

        print(f"[MK-SVM CV] finished repeat {rep + 1}/{n_repeats}")

    pred_df = pd.DataFrame(all_pred_rows)
    metric_df = pd.DataFrame(repeat_metric_rows)
    split_df = pd.DataFrame(split_rows)
    best_param_df = pd.DataFrame(best_param_rows)
    selected_df = pd.DataFrame(selected_feature_rows)
    weight_df = pd.DataFrame(kernel_weight_rows)
    candidate_df = pd.DataFrame(candidate_search_rows)

    pred_df.to_csv(os.path.join(out_dir, "all_repeated_5fold_predictions.csv"), index=False, encoding="utf-8-sig")
    metric_df.to_csv(os.path.join(out_dir, "repeat_level_metrics.csv"), index=False, encoding="utf-8-sig")
    split_df.to_csv(os.path.join(out_dir, "fullsample_repeated_5fold_subjects.csv"), index=False, encoding="utf-8-sig")
    best_param_df.to_csv(os.path.join(out_dir, "outer_fold_best_params.csv"), index=False, encoding="utf-8-sig")
    selected_df.to_csv(os.path.join(out_dir, "all_outer_fold_selected_features.csv"), index=False, encoding="utf-8-sig")
    weight_df.to_csv(os.path.join(out_dir, "all_outer_fold_kernel_weights.csv"), index=False, encoding="utf-8-sig")
    candidate_df.to_csv(os.path.join(out_dir, "all_inner_search_candidates.csv"), index=False, encoding="utf-8-sig")

    selected_summary = summarize_selected_features(
        selected_df, feature_names, feature_types, dims,
        active_modes, n_total_folds_per_mode
    )
    selected_summary.to_csv(
        os.path.join(out_dir, "selected_feature_stability_summary.csv"),
        index=False, encoding="utf-8-sig"
    )

    if not weight_df.empty:
        weight_summary = weight_df.groupby(
            ["mode", "mode_label", "modality"], as_index=False
        )["kernel_weight"].agg(["mean", "std", "median", "min", "max"]).reset_index()
    else:
        weight_summary = pd.DataFrame()
    weight_summary.to_csv(
        os.path.join(out_dir, "kernel_weight_summary.csv"),
        index=False, encoding="utf-8-sig"
    )

    metric_summary = metric_df.groupby(
        ["mode", "mode_label", "metric"]
    )["value"].agg(["mean", "std", "median", "min", "max"]).reset_index()
    metric_summary.to_csv(
        os.path.join(out_dir, "repeat_level_metrics_summary.csv"),
        index=False, encoding="utf-8-sig"
    )

    importance_out_dir = os.path.join(out_dir, "feature_importance")
    global_importance_df = pd.DataFrame(global_importance_rows)
    class_importance_df = pd.DataFrame(class_importance_rows)
    global_importance_summary, class_importance_summary = summarize_permutation_importance(
        global_importance_df, class_importance_df, importance_out_dir
    )

    if bool(CONFIG.get("MAKE_GLOBAL_IMPORTANCE_OVERLAP", True)):
        analyze_global_importance_overlap(
            global_summary=global_importance_summary,
            out_dir=importance_out_dir,
            modes=list(FUSION_MODES),
            top_n=int(CONFIG.get("GLOBAL_IMPORTANCE_OVERLAP_TOP_N", 10)),
        )

    if bool(CONFIG.get("MAKE_PLOTS", True)):
        plot_global_permutation_importance(
            global_importance_summary, importance_out_dir
        )
        plot_class_specific_permutation_importance(
            class_importance_summary, importance_out_dir
        )

    return pred_df, metric_df, selected_summary, weight_df


# ============================================================
# Subject-level probability ensemble
# ============================================================
def make_subject_level_probability_ensemble(pred_df: pd.DataFrame, n_classes: int, out_dir: str):
    prob_cols = [f"prob_c{i}" for i in range(n_classes)]
    rows = []
    for mode, d_mode in pred_df.groupby("mode"):
        for sid, d_sid in d_mode.groupby("subject_id"):
            true_labels = d_sid["y_true"].unique()
            if len(true_labels) != 1:
                raise ValueError(f"Subject {sid} has inconsistent labels in mode {mode}")
            mean_prob = d_sid[prob_cols].mean(axis=0).values.astype(float)
            std_prob = d_sid[prob_cols].std(axis=0).fillna(0).values.astype(float)
            row = {
                "mode": mode,
                "mode_label": d_sid["mode_label"].iloc[0],
                "subject_id": sid,
                "y_true": int(true_labels[0]),
                "y_pred": int(np.argmax(mean_prob)),
                "n_predictions": int(len(d_sid)),
            }
            for c in range(n_classes):
                row[f"mean_prob_c{c}"] = float(mean_prob[c])
                row[f"std_prob_c{c}"] = float(std_prob[c])
            rows.append(row)
    ens_df = pd.DataFrame(rows)
    ens_df.to_csv(
        os.path.join(out_dir, "subject_level_probability_ensemble_predictions.csv"),
        index=False, encoding="utf-8-sig"
    )
    return ens_df


def compute_subject_level_ensemble_metrics(ens_df: pd.DataFrame, n_classes: int, out_dir: str):
    prob_cols = [f"mean_prob_c{i}" for i in range(n_classes)]
    rows = []
    for mode, d in ens_df.groupby("mode"):
        metrics = compute_metrics(
            d["y_true"].values.astype(int),
            d["y_pred"].values.astype(int),
            d[prob_cols].values.astype(float),
            n_classes,
        )
        for key, value in metrics.items():
            if key in {"fpr", "tpr"}:
                continue
            rows.append({
                "mode": mode,
                "mode_label": d["mode_label"].iloc[0],
                "metric": key,
                "value": float(value),
            })
    out = pd.DataFrame(rows)
    out.to_csv(
        os.path.join(out_dir, "subject_level_probability_ensemble_metrics.csv"),
        index=False, encoding="utf-8-sig"
    )
    return out


# ============================================================
# Plotting
# ============================================================
def set_plot_style():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.linewidth": 1.0,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })


def plot_group_roc(pred_df, modes, out_prefix, title, prob_col_prefix="prob_c"):
    if not modes:
        return
    ensure_dir(os.path.dirname(out_prefix))
    set_plot_style()
    n_classes = int(CONFIG["N_CLASSES"])
    prob_cols = [f"{prob_col_prefix}{i}" for i in range(n_classes)]
    rows = []
    fig, ax = plt.subplots(figsize=(5.8, 5.8))

    for mode in modes:
        d = pred_df[pred_df["mode"] == mode]
        if d.empty:
            continue
        fpr, tpr, macro_auc = macro_roc_curve_from_prob(
            d["y_true"].values.astype(int),
            d[prob_cols].values.astype(float),
            n_classes,
            int(CONFIG["ROC_POINTS"]),
        )
        rows.append(pd.DataFrame({
            "mode": mode,
            "mode_label": MODE_INFO[mode]["label"],
            "fpr": fpr,
            "tpr": tpr,
            "macro_auc": macro_auc,
        }))
        ax.plot(
            fpr, tpr,
            color=MODE_INFO[mode]["color"],
            linewidth=2.1,
            label=f"{MODE_INFO[mode]['label']}   AUC = {macro_auc:.3f}",
        )

    if not rows:
        plt.close(fig)
        return
    pd.concat(rows).to_csv(out_prefix + "_ROC_data.csv", index=False, encoding="utf-8-sig")
    ax.plot([0, 1], [0, 1], "--", color="0.55", linewidth=1.1, label="Chance")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_prefix + "_ROC.png", dpi=600, bbox_inches="tight")
    fig.savefig(out_prefix + "_ROC.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_metrics_bar(metric_df, modes, out_prefix, title):
    if not modes:
        return
    metric_order = ["Accuracy", "Balanced accuracy", "F1-score", "Sensitivity", "Specificity", "AUC"]
    rows = []
    for mode in modes:
        for metric in metric_order:
            vals = metric_df[(metric_df["mode"] == mode) & (metric_df["metric"] == metric)]["value"].values
            if len(vals):
                rows.append({
                    "mode": mode,
                    "metric": metric,
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=0)),
                })
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(out_prefix + "_metrics_bar_data.csv", index=False, encoding="utf-8-sig")

    plot_metrics = [m for m in metric_order if m in set(df["metric"])]
    plot_modes = [m for m in modes if m in set(df["mode"])]
    x = np.arange(len(plot_metrics))
    n = len(plot_modes)
    width = min(0.18, 0.75 / max(n, 1))
    gap = 0.025
    total = n * width + (n - 1) * gap
    start = -total / 2 + width / 2

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    for i, mode in enumerate(plot_modes):
        sub = df[df["mode"] == mode].set_index("metric").loc[plot_metrics]
        pos = x + start + i * (width + gap)
        ax.bar(
            pos, sub["mean"].values, yerr=sub["std"].values,
            width=width, capsize=2.5,
            color=MODE_INFO[mode]["color"],
            edgecolor="white", linewidth=0.8,
            label=MODE_INFO[mode]["label"],
        )
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_metrics, rotation=18, ha="right")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, color="0.85")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=min(n, 3))
    fig.tight_layout()
    fig.savefig(out_prefix + "_metrics_bar.png", dpi=600, bbox_inches="tight")
    fig.savefig(out_prefix + "_metrics_bar.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix_summary_blue(y_true, y_pred, class_labels, title, out_prefix):
    n_class = len(class_labels)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_class))).astype(float)
    pd.DataFrame(cm.astype(int), index=class_labels, columns=class_labels).to_csv(
        out_prefix + "_confusion_counts.csv", encoding="utf-8-sig"
    )
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0) * 100

    cmap = LinearSegmentedColormap.from_list(
        "custom_blue", ["#F5FAFE", "#DCECF8", "#B5D5EC", "#75ADD8", "#2F7FBD"]
    )
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(np.arange(n_class))
    ax.set_yticks(np.arange(n_class))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    for i in range(n_class):
        for j in range(n_class):
            v = cm_norm[i, j]
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center", color="white" if v >= 50 else "black")
    fig.colorbar(im, ax=ax, label="Percentage (%)")
    fig.tight_layout()
    fig.savefig(out_prefix + "_confusion.png", dpi=600, bbox_inches="tight")
    fig.savefig(out_prefix + "_confusion.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fusion_confusions(pred_df, int_to_label, subdir="confusion_matrices"):
    out_dir = os.path.join(CONFIG["OUT_DIR"], subdir)
    ensure_dir(out_dir)
    labels = [int_to_label.get(i, str(i)) for i in range(int(CONFIG["N_CLASSES"]))]
    for mode in FUSION_MODES:
        d = pred_df[pred_df["mode"] == mode]
        if d.empty:
            continue
        plot_confusion_matrix_summary_blue(
            d["y_true"].values.astype(int),
            d["y_pred"].values.astype(int),
            labels,
            MODE_INFO[mode]["label"],
            os.path.join(out_dir, mode),
        )


def plot_selected_feature_stability(summary_df, modes, out_dir):
    if summary_df.empty:
        return
    ensure_dir(out_dir)
    topk = int(CONFIG["TOPK_SELECTED_FEATURES"])
    for mode in modes:
        d_mode = summary_df[summary_df["mode"] == mode]
        if d_mode.empty:
            continue
        parts = []
        for _, d_type in d_mode.groupby("feature_type"):
            parts.append(
                d_type.sort_values(
                    ["selected_frequency", "mean_mi_when_selected"],
                    ascending=False,
                ).head(topk)
            )
        d = pd.concat(parts).sort_values("selected_frequency", ascending=True)
        d.to_csv(
            os.path.join(out_dir, f"{mode}_top_selected_features.csv"),
            index=False, encoding="utf-8-sig"
        )
        fig, ax = plt.subplots(figsize=(9.2, max(5.4, 0.30 * len(d) + 1.8)))
        ax.barh(np.arange(len(d)), d["selected_frequency"].values)
        labels = [
            f"{f} (MI={mi:.3g})"
            for f, mi in zip(d["feature"], d["mean_mi_when_selected"])
        ]
        ax.set_yticks(np.arange(len(d)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("Outer-fold selection frequency")
        ax.set_title(f"{MODE_INFO[mode]['label']}: stable selected features")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{mode}_selected_feature_stability.png"), dpi=600, bbox_inches="tight")
        fig.savefig(os.path.join(out_dir, f"{mode}_selected_feature_stability.pdf"), bbox_inches="tight")
        plt.close(fig)


def plot_kernel_weights(weight_df, out_dir):
    if weight_df.empty:
        return
    ensure_dir(out_dir)
    fusion = weight_df[weight_df["mode"].isin(FUSION_MODES)]
    for mode, d in fusion.groupby("mode"):
        summary = d.groupby("modality")["kernel_weight"].agg(["mean", "std"]).reset_index()
        summary.to_csv(os.path.join(out_dir, f"{mode}_kernel_weight_data.csv"), index=False, encoding="utf-8-sig")
        fig, ax = plt.subplots(figsize=(5.8, 4.8))
        ax.bar(summary["modality"], summary["mean"], yerr=summary["std"].fillna(0), capsize=4)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Kernel weight")
        ax.set_title(MODE_INFO[mode]["label"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{mode}_kernel_weights.png"), dpi=600, bbox_inches="tight")
        fig.savefig(os.path.join(out_dir, f"{mode}_kernel_weights.pdf"), bbox_inches="tight")
        plt.close(fig)


def make_all_plots(pred_df, metric_df, selected_summary, weight_df, ens_df, int_to_label):
    if not bool(CONFIG.get("MAKE_PLOTS", True)):
        return
    out_dir = CONFIG["OUT_DIR"]
    available = set(pred_df["mode"].unique()) & set(ens_df["mode"].unique())
    fusion_modes = [m for m in FUSION_MODES if m in available]
    single_modes = [m for m in SINGLE_MODES if m in available]

    plot_group_roc(
        ens_df, fusion_modes,
        os.path.join(out_dir, "ROC_subject_level_ensemble_fusion_three_modes"),
        "Subject-level ensemble macro-ROC: weighted multi-kernel SVM",
        prob_col_prefix="mean_prob_c",
    )
    plot_group_roc(
        ens_df, single_modes,
        os.path.join(out_dir, "ROC_subject_level_ensemble_single_modalities"),
        "Subject-level ensemble macro-ROC: single-modality SVM",
        prob_col_prefix="mean_prob_c",
    )
    plot_metrics_bar(
        metric_df, fusion_modes,
        os.path.join(out_dir, "metrics_bar_fusion_modes"),
        "Repeated OOF performance: weighted multi-kernel SVM",
    )
    plot_metrics_bar(
        metric_df, single_modes,
        os.path.join(out_dir, "metrics_bar_single_modes"),
        "Repeated OOF performance: single-modality SVM",
    )
    plot_fusion_confusions(ens_df, int_to_label, "confusion_matrices_subject_level_ensemble")
    plot_selected_feature_stability(
        selected_summary,
        list(available),
        os.path.join(out_dir, "selected_feature_stability_plots"),
    )
    plot_kernel_weights(weight_df, os.path.join(out_dir, "kernel_weight_plots"))


# ============================================================
# Main
# ============================================================
def main():
    print("[ENV] Scheme A | weighted multi-kernel RBF-SVM | nested CV")
    fixed_seed = int(CONFIG["SEED"])
    set_seed(fixed_seed)

    sids, data, y, int_to_label, feature_names, feature_types, dims = align_all_modalities()
    class_counts = {
        int_to_label.get(int(c), str(c)): int(np.sum(y == c))
        for c in sorted(np.unique(y))
    }
    print("[DATA] N =", len(sids), "class counts =", class_counts)
    print("[MODES]", get_active_modes())
    print("[CV] repeats =", CONFIG["N_REPEATS"], "outer =", CONFIG["N_SPLITS"], "inner =", CONFIG["INNER_SPLITS"])
    print("[SEARCH]", CONFIG["TUNING_SEARCH_TYPE"], "refit =", CONFIG["REFIT_METRIC"])

    base_out_dir = CONFIG["OUT_DIR"]
    run_out_dir = os.path.join(
        base_out_dir,
        f"fixed_seed_{fixed_seed}",
        "weighted_multikernel_rbf_svm",
    )
    CONFIG["OUT_DIR"] = run_out_dir
    ensure_dir(run_out_dir)

    with open(os.path.join(run_out_dir, "effective_config.json"), "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, ensure_ascii=False, indent=2)

    importance_modes = get_importance_modes(get_active_modes())
    importance_out_dir = os.path.join(run_out_dir, "feature_importance")
    descriptive_df = pd.DataFrame()
    pairwise_df = pd.DataFrame()

    if bool(CONFIG.get("MAKE_CLASS_SPECIFIC_DESCRIPTIVE", True)):
        descriptive_df = compute_fullsample_class_specific_descriptive(
            data=data,
            y=y,
            feature_names=feature_names,
            feature_types=feature_types,
            int_to_label=int_to_label,
            modes=importance_modes,
            out_dir=importance_out_dir,
        )

    if bool(CONFIG.get("MAKE_PAIRWISE_CLASS_ANALYSIS", True)):
        pairwise_df = compute_fullsample_pairwise_comparisons(
            data=data,
            y=y,
            feature_names=feature_names,
            feature_types=feature_types,
            int_to_label=int_to_label,
            modes=importance_modes,
            out_dir=importance_out_dir,
        )

    if bool(CONFIG.get("MAKE_PLOTS", True)):
        plot_class_specific_descriptive(
            descriptive_df, pairwise_df, importance_out_dir
        )

    pred_df, metric_df, selected_summary, weight_df = run_repeated_fullsample_cv_schemeA(
        data, y, sids, int_to_label, feature_names, feature_types, dims
    )

    metric_df["classifier"] = "weighted_multikernel_svm"
    metric_df["seed"] = fixed_seed
    pred_df["classifier"] = "weighted_multikernel_svm"
    pred_df["seed"] = fixed_seed
    pred_df.to_csv(os.path.join(run_out_dir, "all_repeated_5fold_predictions.csv"), index=False, encoding="utf-8-sig")
    metric_df.to_csv(os.path.join(run_out_dir, "repeat_level_metrics.csv"), index=False, encoding="utf-8-sig")

    fixed_seed_summary = metric_df.groupby(
        ["classifier", "seed", "mode", "mode_label", "metric"]
    )["value"].agg(["mean", "std", "median", "min", "max"]).reset_index()
    fixed_seed_summary.to_csv(
        os.path.join(run_out_dir, "fixed_seed_metric_summary.csv"),
        index=False, encoding="utf-8-sig"
    )

    ens_df = make_subject_level_probability_ensemble(
        pred_df, int(CONFIG["N_CLASSES"]), run_out_dir
    )
    compute_subject_level_ensemble_metrics(
        ens_df, int(CONFIG["N_CLASSES"]), run_out_dir
    )
    make_all_plots(
        pred_df, metric_df, selected_summary, weight_df, ens_df, int_to_label
    )

    print("[DONE] Results saved to:", run_out_dir)
    print("[SUMMARY] Metrics:", os.path.join(run_out_dir, "fixed_seed_metric_summary.csv"))
    print("[SUMMARY] Kernel weights:", os.path.join(run_out_dir, "kernel_weight_summary.csv"))
    print("[SUMMARY] Selected features:", os.path.join(run_out_dir, "selected_feature_stability_summary.csv"))
    print("[SUMMARY] Feature importance directory:", os.path.join(run_out_dir, "feature_importance"))
    print("[SUMMARY] Top-10 overlap:", os.path.join(
        run_out_dir, "feature_importance",
        "global_importance_top10_overall_pairwise_overlap.csv"
    ))


if __name__ == "__main__":
    main()

