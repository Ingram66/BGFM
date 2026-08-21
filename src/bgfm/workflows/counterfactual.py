import os
import re
import math
import random
from typing import Dict, Any, List, Optional, Tuple, Iterable, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import pdist, squareform


from bgfm.runtime import load_section, apply_globals, apply_mapping

# ============================================================
# CONFIG: change paths here
# ============================================================
CONFIG = {
    # Original training output directory. Expected: fold_01/best_model.pt ... fold_10/best_model.pt
    "OUT_DIR": r"outputs/alignment",
    "CKPT_DIR": r"outputs/alignment",

    "SEED": 1307,
    "N_FOLDS": 10,
    "BATCH_SIZE": 32,

    # Strict OOF requires the exact validation-fold assignment used when the
    # checkpoints were trained. Preferred columns: subject_id, fold.
    # If the file contains a split column, rows labelled val/valid/validation/test
    # are treated as validation rows. Set to "" to try checkpoint metadata.
    # Preferred exact fold-assignment file. Leave empty to auto-search candidates below.
    "FOLD_ASSIGNMENT_CSV": "",
    "FOLD_ASSIGNMENT_CANDIDATES": [
        r"outputs/alignment/oof_fold_assignment.csv",
        r"outputs/alignment/oof_fold_assignment.csv",
    ],

    # If no exact assignment file/checkpoint metadata is available, reconstruct with
    # the original kfold_indices implementation and SEED. This is exact only when
    # training used the same aligned subjects, the same subject order, and the same
    # splitter before any filtering.
    "ALLOW_SEED_RECONSTRUCTION": True,
    "EXPECTED_ALIGNED_N_FOR_SEED_RECONSTRUCTION": 395,

    # Keep True: if seed reconstruction is disabled, abort rather than silently use
    # a different split.
    "STRICT_OOF_REQUIRE_EXACT_FOLDS": True,

    # Data used by the original cross-modal model training
    "BRAIN_FEAT_PT": r"outputs/paired_brain_features/brain_classification_features.pt",
    "GUT_FEAT_PT": r"outputs/paired_gut_features/gut_classification_features.pt",
    "MICRO_ABUND_CSV": r"data/paired/microbiome_abundance.csv",
    "ROI_MEAN_CSV": r"data/paired/bold_roi_mean.csv",

    # Subject-level group file. Required columns: subject_id, group.
    # group values should include HC, MDD, SZ, BD.
    "GROUP_CSV": r"data/paired/labels_4class.csv",

    # Model hyperparameters: must be identical to training script
    "D_ALIGN": 256,
    "DROPOUT": 0.1,
    "N_HEADS": 4,
    "N_COND_TOKENS": 8,
    "ROI_HEAD_HIDDEN": 256,
    "TAXA_HEAD_HIDDEN": 128,

    # Numerical stability
    "AB_TEMP_MIN": 0.7,
    "AB_TEMP_MAX": 1.3,
    "EPS": 1e-8,
}

ANALYSIS_CONFIG = {
    # Output folder
    "OUT_SUBDIR": r"outputs/counterfactual",

    # Optional names, one name per row; if empty, ROI_001 / Taxa_001 will be used.
    "ROI_NAME_FILE": r"data/metadata/brain_roi_names.txt",
    "TAXA_NAME_FILE": r"data/metadata/taxa_names.txt",

    # Groups
    "CONTROL_GROUP": "HC",
    "DISEASE_GROUPS": ["MDD", "SZ", "BD"],

    # Shift parameters
    # For microbiome, use "relative" first. "clr" is more compositional but requires stable inverse transform.
    "MICRO_SHIFT_SPACE": "relative",  # relative / clr
    "LAMBDA_VALUES": [0.0, 0.25, 0.5, 0.75, 1.0],
    "PI_STEPS": 21,   # path-integrated attribution steps. Increase to 50 for final run if computationally feasible.

    # OOF source samples: HC subjects in each validation fold.
    "SOURCE_GROUP_FOR_SHIFT": "HC",

    # Held-out validation-centroid settings. For fold k, the model receives
    # one validation HC centroid and a path ending at the validation disease
    # centroid. All members of both centroids were unseen by model k.
    "OOF_KEEP_ALL_VALIDATION_SUBJECTS": True,
    "OOF_MIN_VALIDATION_HC": 1,
    "OOF_MIN_VALIDATION_DISEASE": 1,
    "OOF_SAVE_SHIFTED_INPUTS": True,
    # Fold-level results are combined using an effective two-group sample size:
    # n_eff = n_HC * n_disease / (n_HC + n_disease).
    # Alternatives: equal / min_group / total.
    "CENTROID_FOLD_WEIGHT_MODE": "effective_n",

    # ------------------------------------------------------------------
    # Group-wise subject outlier removal before group-center analysis
    # ------------------------------------------------------------------
    # Outliers are detected within each group separately, then removed before
    # computing raw centers, shrinkage centers, group shifts, and attribution.
    "REMOVE_GROUP_OUTLIERS": True,
    # Recommended default for your question: use ROI-level BOLD means Y_all.
    # Optional extra modalities: "micro_a", "brain_feat_mean", "gut_feat_mean".
    "OUTLIER_MODALITIES": ["brain_y"],
    # "mad" is more robust than ordinary z-score for small/biased groups.
    "OUTLIER_METHOD": "mad",  # mad / zscore
    "OUTLIER_Z_THRESHOLD": 3.5,
    # A subject is removed if at least this many features, or this fraction of
    # features, exceed the threshold in any selected modality.
    "OUTLIER_MIN_FEATURE_COUNT": 3,
    "OUTLIER_MAX_FEATURE_FRAC": 0.03,
    # For microbiome abundance, use CLR space for outlier detection by default.
    "OUTLIER_MICRO_SPACE": "clr",  # clr / relative
    # Safety: do not allow filtering to leave too few samples in a group.
    "OUTLIER_MIN_GROUP_N_AFTER_FILTER": 5,
    "OUTLIER_SAVE_REPORT": True,

    # Save individual predicted outputs for every disease/lambda/direction. This can produce large files.
    "SAVE_INDIVIDUAL_PREDICTIONS": True,

    # Plot / selection parameters
    "FIG_DPI": 220,
    "TOPK_TAXA": 20,
    "TOPK_ROI": 20,
    "TOPK_EDGES": 30,
    "TOPK_WATERFALL": 12,
    "TOPK_SHARED_SPECIFIC_EDGES": 30,

    # Heatmap display. To make full 642x90 plots readable, top rows are shown by default.
    # Full matrices are still saved as CSV.
    "HEATMAP_TOP_TAXA": 80,
    "HEATMAP_TOP_ROI": 90,
    "HEATMAP_FIGSIZE": (22, 12),

    # If brain->micro path-integrated attribution is too slow, set to a list of taxa indices or None.
    # None means all 642 taxa will be attributed.
    "BRAIN2MICRO_TARGET_TAXA_INDICES": None,

    # ------------------------------------------------------------------
    # Bootstrap-adaptive shrinkage centers
    # ------------------------------------------------------------------
    # Disease centers are shrunk toward a stable reference before serving as
    # centroid targets. Alpha is computed as:
    #   alpha = effect / (effect + bootstrap_noise)
    "USE_BOOTSTRAP_SHRINKAGE_CENTER": True,
    "SHRINKAGE_TARGET": "grand_mean",  # grand_mean / hc
    "SHRINKAGE_N_BOOT": 1000,
    "SHRINKAGE_BOOTSTRAP_SEED": 1307,
    "SHRINKAGE_ALPHA_MIN": 0.30,
    "SHRINKAGE_ALPHA_MAX": 0.95,
    # Apply shrinkage to control as well. Here HC, MDD, SZ and BD all use
    # bootstrap-adaptive shrinkage centers.
    "SHRINKAGE_APPLY_TO_CONTROL": True,
    "NO_SHRINKAGE_GROUPS": [],

    # ------------------------------------------------------------------
    # Centroid interpolation shift, not centroid endpoint optimization
    # ------------------------------------------------------------------
    # The HC center and disease centers used below are the bootstrap-shrinkage
    # centers when USE_BOOTSTRAP_SHRINKAGE_CENTER=True. No model-driven
    # endpoint optimization is performed.
    "SHIFT_MODE": "centroid_interpolation",
}


# Fixed dimensions used in your original code
N_ROIS = 90
N_TAXA = 642

apply_mapping(CONFIG, load_section('counterfactual'))
apply_mapping(ANALYSIS_CONFIG, load_section('counterfactual_analysis'))


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int = 1307):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def get_analysis_out_dir() -> str:
    subdir = str(ANALYSIS_CONFIG["OUT_SUBDIR"])
    if os.path.isabs(subdir):
        out_dir = subdir
    else:
        out_dir = os.path.join(CONFIG["OUT_DIR"], subdir)
    ensure_dir(out_dir)
    return out_dir


def sanitize_filename(name: str, max_len: int = 90) -> str:
    s = re.sub(r"[\\/:*?\"<>|]", "_", str(name))
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return (s or "unnamed")[:max_len]


def torch_load_any(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_subject_id_from_path(p: str) -> str:
    base = os.path.basename(p)
    return str(os.path.splitext(base)[0])


def read_name_list(path: Optional[str], n_expected: int, prefix: str) -> List[str]:
    if path is None or str(path).strip() == "":
        return [f"{prefix}_{i+1:03d}" for i in range(n_expected)]
    if not os.path.exists(path):
        raise FileNotFoundError(f"name file not found: {path}")
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
        if df.shape[1] < 1:
            raise ValueError(f"empty csv: {path}")
        names = df.iloc[:, 0].astype(str).tolist()
    else:
        with open(path, "r", encoding="utf-8") as f:
            names = [x.strip() for x in f if x.strip()]
    if len(names) < n_expected:
        raise ValueError(f"name file has only {len(names)} entries, expected at least {n_expected}: {path}")
    return names[:n_expected]


def get_roi_names() -> List[str]:
    return read_name_list(ANALYSIS_CONFIG["ROI_NAME_FILE"], N_ROIS, "ROI")


def get_taxa_names() -> List[str]:
    return read_name_list(ANALYSIS_CONFIG["TAXA_NAME_FILE"], N_TAXA, "Taxa")


def save_matrix_csv(mat: np.ndarray, row_names: List[str], col_names: List[str], out_csv: str, index_name: str):
    df = pd.DataFrame(mat, index=row_names, columns=col_names)
    df.index.name = index_name
    df.to_csv(out_csv, encoding="utf-8-sig")


def _require_distribution(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.clamp_min(0.0)
    s = x.sum(dim=1, keepdim=True).clamp_min(eps)
    return x / s


def require_distribution_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.maximum(x, 0.0)
    s = np.maximum(x.sum(axis=-1, keepdims=True), eps)
    return x / s


def clr_np(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.maximum(p, eps)
    lp = np.log(p)
    return lp - lp.mean(axis=-1, keepdims=True)


def inv_clr_np(z: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    x = np.exp(z)
    return require_distribution_np(x, eps=eps)


def pearson_np(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom < eps:
        return 0.0
    return float((x * y).sum() / denom)


def cosine_np(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom < eps:
        return 0.0
    return float(np.dot(x, y) / denom)


def bray_curtis_np(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    return float(np.abs(x - y).sum() / max(np.abs(x + y).sum(), eps))


def js_divergence_np(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = require_distribution_np(np.asarray(p, dtype=np.float64)[None, :], eps=eps)[0]
    q = require_distribution_np(np.asarray(q, dtype=np.float64)[None, :], eps=eps)[0]
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
    return float(0.5 * (kl_pm + kl_qm))


def aitchison_np(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.linalg.norm(clr_np(p, eps=eps) - clr_np(q, eps=eps)))


def pca_2d_np(X: np.ndarray) -> np.ndarray:
    """Small SVD-based PCA, no sklearn dependency."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


# ============================================================
# Loaders
# ============================================================
def load_brain_node_feat(pt_path: str) -> Tuple[torch.Tensor, Optional[List[str]], Dict[str, Any]]:
    obj = torch_load_any(pt_path)
    if not isinstance(obj, dict):
        raise ValueError(f"{pt_path} must be a dict saved by torch.save")
    if "roi_emb" in obj:
        x = obj["roi_emb"]
    elif "emb" in obj:
        x = obj["emb"]
    else:
        raise ValueError(f"{pt_path} has no key 'roi_emb' or 'emb'")
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    if x.dim() != 3 or x.size(1) != N_ROIS:
        raise ValueError(f"[BRAIN] expected (N,{N_ROIS},d), got {tuple(x.shape)}")
    sids = None
    if "paths" in obj and isinstance(obj["paths"], list):
        sids = [extract_subject_id_from_path(p) for p in obj["paths"]]
    elif "sample_ids" in obj and isinstance(obj["sample_ids"], list):
        sids = [str(s) for s in obj["sample_ids"]]
    return x.float(), sids, obj


def load_gut_node_feat(pt_path: str) -> Tuple[torch.Tensor, Optional[List[str]], Dict[str, Any]]:
    obj = torch_load_any(pt_path)
    if not isinstance(obj, dict):
        raise ValueError(f"{pt_path} must be a dict saved by torch.save")
    if "taxa_emb" in obj:
        x = obj["taxa_emb"]
    elif "node_emb" in obj:
        x = obj["node_emb"]
    elif "emb" in obj:
        x = obj["emb"]
    else:
        raise ValueError(f"{pt_path} has no key 'taxa_emb'/'node_emb'/'emb'")
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    if x.dim() != 3 or x.size(1) != N_TAXA:
        raise ValueError(f"[GUT] expected (N,{N_TAXA},d), got {tuple(x.shape)}")
    sids = None
    if "sample_ids" in obj and isinstance(obj["sample_ids"], list):
        sids = [str(s) for s in obj["sample_ids"]]
    elif "paths" in obj and isinstance(obj["paths"], list):
        sids = [extract_subject_id_from_path(p) for p in obj["paths"]]
    return x.float(), sids, obj


def load_abundance_table(csv_path: str) -> Dict[str, torch.Tensor]:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 1 + N_TAXA:
        raise ValueError(f"[ABUND] cols={df.shape[1]} < 1+{N_TAXA}")
    sids = df.iloc[:, 0].astype(str).tolist()
    mat = df.iloc[:, 1:1 + N_TAXA].to_numpy(np.float32)
    x = torch.tensor(mat, dtype=torch.float32)
    x = _require_distribution(x, eps=CONFIG["EPS"])
    return {sids[i]: x[i] for i in range(len(sids))}


def load_roi_mean_table(csv_path: str) -> Dict[str, torch.Tensor]:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 1 + N_ROIS:
        raise ValueError(f"[ROI_MEAN] cols={df.shape[1]} < 1+{N_ROIS}")
    sids = df.iloc[:, 0].astype(str).tolist()
    mat = df.iloc[:, 1:1 + N_ROIS].to_numpy(np.float32)
    y = torch.tensor(mat, dtype=torch.float32)
    return {sids[i]: y[i] for i in range(len(sids))}


def load_group_table(csv_path: str) -> Dict[str, str]:
    df = pd.read_csv(csv_path)
    lower = {c.lower(): c for c in df.columns}
    if "subject_id" not in lower or "group" not in lower:
        raise ValueError("GROUP_CSV must contain columns: subject_id, group")
    sid_col = lower["subject_id"]
    group_col = lower["group"]
    out = {}
    for _, row in df.iterrows():
        out[str(row[sid_col])] = str(row[group_col]).strip().upper()
    return out





def intersect_and_stack_with_groups(
    brain: torch.Tensor,
    brain_sids: Optional[List[str]],
    gut: torch.Tensor,
    gut_sids: Optional[List[str]],
    abund_map: Dict[str, torch.Tensor],
    roi_map: Dict[str, torch.Tensor],
    group_map: Dict[str, str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[str]]:
    if brain_sids is None or gut_sids is None:
        common = sorted(list(set(abund_map) & set(roi_map) & set(group_map)))
        if len(common) == 0:
            raise ValueError("No common subject_ids among abundance/roi_mean/group.")
        N = min(brain.size(0), gut.size(0), len(common))
        use_sids = common[:N]
        B = brain[:N]
        M = gut[:N]
        A = torch.stack([abund_map[s] for s in use_sids], dim=0)
        Y = torch.stack([roi_map[s] for s in use_sids], dim=0)
        groups = [group_map[s] for s in use_sids]
        return B, M, A, Y, use_sids, groups

    b_map = {brain_sids[i]: i for i in range(len(brain_sids))}
    m_map = {gut_sids[i]: i for i in range(len(gut_sids))}
    common = sorted(list(set(b_map) & set(m_map) & set(abund_map) & set(roi_map) & set(group_map)))
    if len(common) == 0:
        raise ValueError("No common subject_ids across brain/gut/abundance/roi_mean/group. Check IDs.")
    B = torch.stack([brain[b_map[s]] for s in common], dim=0)
    M = torch.stack([gut[m_map[s]] for s in common], dim=0)
    A = torch.stack([abund_map[s] for s in common], dim=0)
    Y = torch.stack([roi_map[s] for s in common], dim=0)
    groups = [group_map[s] for s in common]
    return B, M, A, Y, common, groups


def load_oof_data_with_groups():
    """Load aligned B, M, A, Y and group labels. No covariate table is required."""
    print("Loading aligned brain-gut data with group labels.")
    B_all, B_sids, _ = load_brain_node_feat(CONFIG["BRAIN_FEAT_PT"])
    M_all, M_sids, _ = load_gut_node_feat(CONFIG["GUT_FEAT_PT"])
    abund_map = load_abundance_table(CONFIG["MICRO_ABUND_CSV"])
    roi_map = load_roi_mean_table(CONFIG["ROI_MEAN_CSV"])
    group_map = load_group_table(CONFIG["GROUP_CSV"])
    B_all, M_all, A_all, Y_all, sids, groups = intersect_and_stack_with_groups(
        B_all, B_sids, M_all, M_sids, abund_map, roi_map, group_map
    )
    A_all = _require_distribution(A_all, eps=CONFIG["EPS"])
    print(f"[DATA] N={len(sids)} | B={tuple(B_all.shape)} | M={tuple(M_all.shape)} | A={tuple(A_all.shape)} | Y={tuple(Y_all.shape)}")
    print(pd.Series(groups).value_counts().to_string())
    return B_all, M_all, A_all, Y_all, sids, groups


# ============================================================
# Group-wise outlier removal before analysis
# ============================================================
def _robust_zscore_np(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Feature-wise robust z-score using median and MAD."""
    X = np.asarray(X, dtype=np.float64)
    med = np.nanmedian(X, axis=0, keepdims=True)
    mad = np.nanmedian(np.abs(X - med), axis=0, keepdims=True)
    scale = 1.4826 * mad
    # If MAD is zero, fall back to standard deviation for that feature.
    sd = np.nanstd(X, axis=0, keepdims=True)
    scale = np.where(scale < eps, sd, scale)
    scale = np.where(scale < eps, 1.0, scale)
    return (X - med) / scale


def _standard_zscore_np(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Feature-wise ordinary z-score."""
    X = np.asarray(X, dtype=np.float64)
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (X - mu) / sd


def _get_outlier_matrix_for_modality(
    modality: str,
    B_all: torch.Tensor,
    M_all: torch.Tensor,
    A_all: torch.Tensor,
    Y_all: torch.Tensor,
) -> Tuple[np.ndarray, str]:
    """
    Return a subject x feature matrix for group-wise outlier detection.

    Supported modalities:
        brain_y         : ROI-level BOLD means, shape [N, 90]
        micro_a         : microbiome abundance, shape [N, 642], CLR or relative
        brain_feat_mean : mean pooled brain node features, shape [N, d_b]
        gut_feat_mean   : mean pooled gut node features, shape [N, d_m]
    """
    m = str(modality).lower().strip()
    if m == "brain_y":
        return Y_all.detach().cpu().numpy().astype(np.float64), "brain_y_roi_mean"
    if m == "micro_a":
        A_np = _require_distribution(A_all.detach().cpu(), eps=CONFIG["EPS"]).numpy().astype(np.float64)
        if str(ANALYSIS_CONFIG.get("OUTLIER_MICRO_SPACE", "clr")).lower() == "clr":
            return clr_np(A_np, eps=CONFIG["EPS"]).astype(np.float64), "micro_a_clr"
        return A_np, "micro_a_relative"
    if m == "brain_feat_mean":
        return B_all.detach().cpu().mean(dim=1).numpy().astype(np.float64), "brain_feat_node_mean"
    if m == "gut_feat_mean":
        return M_all.detach().cpu().mean(dim=1).numpy().astype(np.float64), "gut_feat_node_mean"
    raise ValueError(
        f"Unsupported OUTLIER_MODALITY: {modality}. "
        "Use one of: brain_y, micro_a, brain_feat_mean, gut_feat_mean."
    )


def remove_groupwise_outlier_subjects(
    B_all: torch.Tensor,
    M_all: torch.Tensor,
    A_all: torch.Tensor,
    Y_all: torch.Tensor,
    sids: List[str],
    groups: List[str],
    out_dir: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[str], pd.DataFrame]:
    """
    Detect and remove subject-level outliers within each diagnostic group.

    A subject is marked as an outlier if, within its own group and within any
    selected modality, the number of features with |z| >= threshold reaches the
    larger of:
        OUTLIER_MIN_FEATURE_COUNT
        ceil(OUTLIER_MAX_FEATURE_FRAC * n_features)

    This keeps disease-control differences from being treated as outliers,
    because each group is evaluated separately.
    """
    if not bool(ANALYSIS_CONFIG.get("REMOVE_GROUP_OUTLIERS", False)):
        report = pd.DataFrame({
            "original_index": np.arange(len(sids), dtype=int),
            "subject_id": sids,
            "group": [str(g).upper() for g in groups],
            "is_outlier": False,
            "outlier_modalities": "",
            "reason": "outlier removal disabled",
        })
        return B_all, M_all, A_all, Y_all, sids, groups, report

    modalities = ANALYSIS_CONFIG.get("OUTLIER_MODALITIES", ["brain_y"])
    modalities = [str(x).lower().strip() for x in modalities]
    method = str(ANALYSIS_CONFIG.get("OUTLIER_METHOD", "mad")).lower()
    z_thr = float(ANALYSIS_CONFIG.get("OUTLIER_Z_THRESHOLD", 3.5))
    min_count_cfg = int(ANALYSIS_CONFIG.get("OUTLIER_MIN_FEATURE_COUNT", 3))
    frac_cfg = float(ANALYSIS_CONFIG.get("OUTLIER_MAX_FEATURE_FRAC", 0.03))
    min_group_n_after = int(ANALYSIS_CONFIG.get("OUTLIER_MIN_GROUP_N_AFTER_FILTER", 5))

    if method not in {"mad", "zscore"}:
        raise ValueError("OUTLIER_METHOD must be 'mad' or 'zscore'.")

    groups_arr = np.array([str(g).upper() for g in groups])
    unique_groups = [ANALYSIS_CONFIG["CONTROL_GROUP"].upper()] + [g.upper() for g in ANALYSIS_CONFIG["DISEASE_GROUPS"]]
    unique_groups = [g for g in unique_groups if g in set(groups_arr)]

    # Prepare modality matrices once.
    modality_data: Dict[str, Tuple[np.ndarray, str]] = {}
    for m in modalities:
        modality_data[m] = _get_outlier_matrix_for_modality(m, B_all, M_all, A_all, Y_all)

    rows = []
    global_outlier = np.zeros(len(sids), dtype=bool)

    for g in unique_groups:
        idx = np.where(groups_arr == g)[0]
        if len(idx) == 0:
            continue
        group_flags = np.zeros(len(idx), dtype=bool)
        modality_flag_names = [set() for _ in idx]
        modality_reasons = [list() for _ in idx]

        for m, (X_all, pretty_name) in modality_data.items():
            Xg = X_all[idx]
            Z = _robust_zscore_np(Xg) if method == "mad" else _standard_zscore_np(Xg)
            absZ = np.abs(Z)
            n_features = int(absZ.shape[1])
            feature_cutoff = max(min_count_cfg, int(math.ceil(frac_cfg * n_features)))
            exceed_count = np.sum(absZ >= z_thr, axis=1).astype(int)
            max_abs_z = np.nanmax(absZ, axis=1)
            flags = exceed_count >= feature_cutoff

            for local_i, original_i in enumerate(idx):
                if flags[local_i]:
                    group_flags[local_i] = True
                    modality_flag_names[local_i].add(pretty_name)
                    modality_reasons[local_i].append(
                        f"{pretty_name}: exceed_count={int(exceed_count[local_i])}/"
                        f"{n_features}, cutoff={feature_cutoff}, max_abs_z={float(max_abs_z[local_i]):.4f}"
                    )

        # Safety rule: never remove so many subjects that the group becomes too small.
        n_flagged = int(group_flags.sum())
        if len(idx) - n_flagged < min_group_n_after:
            print(
                f"[OUTLIER WARNING] group={g}: detected {n_flagged} outliers, "
                f"but keeping all subjects because n_after={len(idx) - n_flagged} "
                f"< OUTLIER_MIN_GROUP_N_AFTER_FILTER={min_group_n_after}."
            )
            group_flags[:] = False
            modality_flag_names = [set() for _ in idx]
            modality_reasons = [["kept by minimum group-size safety rule"] for _ in idx]

        global_outlier[idx] = group_flags

        for local_i, original_i in enumerate(idx):
            rows.append({
                "original_index": int(original_i),
                "subject_id": sids[int(original_i)],
                "group": g,
                "is_outlier": bool(group_flags[local_i]),
                "outlier_modalities": ";".join(sorted(modality_flag_names[local_i])),
                "reason": " | ".join(modality_reasons[local_i]),
            })

    report = pd.DataFrame(rows).sort_values(["group", "original_index"]).reset_index(drop=True)
    keep = ~global_outlier

    print("[OUTLIER] group-wise subject outlier removal summary:")
    summary = report.groupby("group")["is_outlier"].agg(total="count", removed="sum")
    summary["kept"] = summary["total"] - summary["removed"]
    print(summary.to_string())

    if bool(ANALYSIS_CONFIG.get("OUTLIER_SAVE_REPORT", True)):
        outlier_dir = os.path.join(out_dir, "outlier_filtering")
        ensure_dir(outlier_dir)
        report.to_csv(os.path.join(outlier_dir, "groupwise_outlier_subject_report.csv"), index=False, encoding="utf-8-sig")
        summary.to_csv(os.path.join(outlier_dir, "groupwise_outlier_summary.csv"), encoding="utf-8-sig")

    B_f = B_all[torch.tensor(keep, dtype=torch.bool)].clone()
    M_f = M_all[torch.tensor(keep, dtype=torch.bool)].clone()
    A_f = A_all[torch.tensor(keep, dtype=torch.bool)].clone()
    Y_f = Y_all[torch.tensor(keep, dtype=torch.bool)].clone()
    A_f = _require_distribution(A_f, eps=CONFIG["EPS"])
    sids_f = [sids[i] for i in range(len(sids)) if keep[i]]
    groups_f = [groups[i] for i in range(len(groups)) if keep[i]]
    return B_f, M_f, A_f, Y_f, sids_f, groups_f, report


# ============================================================
# Model: must match training script exactly
# ============================================================
class NodeProjector(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_out, d_out), nn.GELU(), nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)


class ConditionalTokenInit(nn.Module):
    def __init__(self, d_seed: int, n_tokens: int, d_token: int, dropout: float = 0.1):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_token = d_token
        self.net = nn.Sequential(
            nn.Linear(d_seed, d_token), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_token, n_tokens * d_token),
        )
    def forward(self, seed):
        out = self.net(seed)
        return out.view(seed.size(0), self.n_tokens, self.d_token)


class ConditionTokenizer(nn.Module):
    def __init__(self, d_in: int, n_tokens: int, d_token: int, dropout: float = 0.1):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_token = d_token
        self.net = nn.Sequential(
            nn.Linear(d_in, d_token), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_token, n_tokens * d_token),
        )
    def forward(self, x):
        out = self.net(x)
        return out.view(x.size(0), self.n_tokens, self.d_token)


class ConditionCrossAttention(nn.Module):
    def __init__(self, d: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
    def forward(self, x_tokens, cond_tokens):
        out, _ = self.attn(query=x_tokens, key=cond_tokens, value=cond_tokens, need_weights=False)
        return self.norm(x_tokens + out)


class CrossBridge(nn.Module):
    def __init__(self, d: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.m2b = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.b2m = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm_b = nn.LayerNorm(d)
        self.norm_m = nn.LayerNorm(d)
    def micro_to_brain(self, Bq, Mk):
        out, _ = self.m2b(query=Bq, key=Mk, value=Mk, need_weights=False)
        return self.norm_b(Bq + out)
    def brain_to_micro(self, Mq, Bk):
        out, _ = self.b2m(query=Mq, key=Bk, value=Bk, need_weights=False)
        return self.norm_m(Mq + out)


class ROIHeadSemiShared(nn.Module):
    def __init__(self, d_align: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(d_align, hidden), nn.GELU(), nn.Dropout(dropout))
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(N_ROIS)])
        self.roi_bias = nn.Parameter(torch.zeros(N_ROIS))
    def forward(self, B_mix):
        h = self.shared(B_mix)
        outs = [self.heads[r](h[:, r, :]) for r in range(N_ROIS)]
        y = torch.cat(outs, dim=1)
        return y + self.roi_bias.view(1, -1)


class TaxaHeadSemiShared(nn.Module):
    def __init__(self, d_align: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(d_align, hidden), nn.GELU(), nn.Dropout(dropout))
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(N_TAXA)])
        self.taxa_bias = nn.Parameter(torch.zeros(N_TAXA))
    def forward(self, M_mix):
        h = self.shared(M_mix)
        outs = [self.heads[t](h[:, t, :]) for t in range(N_TAXA)]
        logits = torch.cat(outs, dim=1)
        return logits + self.taxa_bias.view(1, -1)


class DualMicroBrainBridge(nn.Module):
    def __init__(self, d_b: int, d_m: int, d_align: int = 256, dropout: float = 0.1, n_heads: int = 4):
        super().__init__()
        self.d_align = d_align
        self.f = NodeProjector(d_b, d_align, dropout=dropout)
        self.g = NodeProjector(d_m, d_align, dropout=dropout)
        self.brain_query_tokens = nn.Parameter(torch.randn(N_ROIS, d_align) * 0.02)
        self.micro_query_tokens = nn.Parameter(torch.randn(N_TAXA, d_align) * 0.02)
        self.brain_init_from_micro = ConditionalTokenInit(d_align, N_ROIS, d_align, dropout=dropout)
        self.micro_init_from_brain = ConditionalTokenInit(d_align, N_TAXA, d_align, dropout=dropout)
        self.abund_condition_tokenizer = ConditionTokenizer(N_TAXA, CONFIG['N_COND_TOKENS'], d_align, dropout=dropout)
        self.roi_condition_tokenizer = ConditionTokenizer(N_ROIS, CONFIG['N_COND_TOKENS'], d_align, dropout=dropout)
        self.brain_cond_attn = ConditionCrossAttention(d_align, n_heads=n_heads, dropout=dropout)
        self.micro_cond_attn = ConditionCrossAttention(d_align, n_heads=n_heads, dropout=dropout)
        self.bridge = CrossBridge(d_align, n_heads=n_heads, dropout=dropout)
        self.roi_alpha_logit = nn.Parameter(torch.tensor(0.0))
        self.m_alpha_logit = nn.Parameter(torch.tensor(-0.4))
        self.roi_head = ROIHeadSemiShared(d_align, hidden=CONFIG['ROI_HEAD_HIDDEN'], dropout=dropout)
        self.taxa_head = TaxaHeadSemiShared(d_align, hidden=CONFIG['TAXA_HEAD_HIDDEN'], dropout=dropout)
        self.ab_temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, B_true=None, M_true=None, A_true=None, Y_true=None, mode: str = 'both'):
        eps = CONFIG['EPS']
        assert mode in ('both', 'brain2micro', 'micro2brain')
        if B_true is not None:
            B_lat = self.f(B_true)
        else:
            if M_true is not None:
                M_seed = self.g(M_true).mean(dim=1)
            elif A_true is not None:
                M_seed = self.abund_condition_tokenizer(A_true).mean(dim=1)
            else:
                raise ValueError('Cannot infer B_lat when B_true is None.')
            B_base = self.brain_query_tokens.unsqueeze(0).expand(M_seed.size(0), -1, -1)
            B_offset = self.brain_init_from_micro(M_seed)
            B_lat = B_base + B_offset
        if M_true is not None:
            M_lat = self.g(M_true)
        else:
            if B_true is not None:
                B_seed = self.f(B_true).mean(dim=1)
            elif Y_true is not None:
                B_seed = self.roi_condition_tokenizer(Y_true).mean(dim=1)
            else:
                raise ValueError('Cannot infer M_lat when M_true is None.')
            M_base = self.micro_query_tokens.unsqueeze(0).expand(B_seed.size(0), -1, -1)
            M_offset = self.micro_init_from_brain(B_seed)
            M_lat = M_base + M_offset
        if mode == 'both':
            if A_true is None or Y_true is None:
                raise ValueError("mode='both' requires A_true and Y_true.")
            A_cond = self.abund_condition_tokenizer(A_true)
            Y_cond = self.roi_condition_tokenizer(Y_true)
            B_cond = self.brain_cond_attn(B_lat, A_cond)
            M_cond = self.micro_cond_attn(M_lat, Y_cond)
        elif mode == 'brain2micro':
            if Y_true is None:
                raise ValueError("mode='brain2micro' requires Y_true.")
            Y_cond = self.roi_condition_tokenizer(Y_true)
            B_cond = B_lat
            M_cond = self.micro_cond_attn(M_lat, Y_cond)
        else:
            if A_true is None:
                raise ValueError("mode='micro2brain' requires A_true.")
            A_cond = self.abund_condition_tokenizer(A_true)
            B_cond = self.brain_cond_attn(B_lat, A_cond)
            M_cond = M_lat
        B_bridge = self.bridge.micro_to_brain(B_cond, M_cond)
        M_bridge = self.bridge.brain_to_micro(M_cond, B_cond)
        roi_alpha = torch.sigmoid(self.roi_alpha_logit)
        m_alpha = torch.sigmoid(self.m_alpha_logit)
        B_mix = roi_alpha * B_bridge + (1.0 - roi_alpha) * B_lat
        M_mix = m_alpha * M_bridge + (1.0 - m_alpha) * M_lat
        y_hat = None
        abund_hat = None
        if mode in ('both', 'micro2brain'):
            y_hat = self.roi_head(B_mix)
        if mode in ('both', 'brain2micro'):
            logits = self.taxa_head(M_mix)
            T = self.ab_temp.clamp(float(CONFIG['AB_TEMP_MIN']), float(CONFIG['AB_TEMP_MAX']))
            abund_hat = torch.softmax(logits / T, dim=-1)
            abund_hat = _require_distribution(abund_hat, eps=eps)
        B_lat_n = F.normalize(B_lat, dim=-1)
        M_lat_n = F.normalize(M_lat, dim=-1)
        B_bridge_n = F.normalize(B_bridge, dim=-1)
        M_bridge_n = F.normalize(M_bridge, dim=-1)
        z_b = F.normalize(B_mix.mean(dim=1), dim=-1)
        z_m = F.normalize(M_mix.mean(dim=1), dim=-1)
        z_sh = F.normalize(0.5 * (z_b + z_m), dim=-1)
        return y_hat, abund_hat, B_bridge_n, B_lat_n, M_bridge_n, M_lat_n, roi_alpha, z_b, z_m, z_sh


# ============================================================
# Checkpoint / inference
# ============================================================
def _resolve_ckpt_dir() -> str:
    ckpt_dir = str(CONFIG.get("CKPT_DIR", "") or "").strip()
    return ckpt_dir if ckpt_dir else str(CONFIG["OUT_DIR"])


def _infer_model_dims_from_ckpt(best_model_path: str) -> Tuple[int, int]:
    ckpt = torch.load(best_model_path, map_location="cpu", weights_only=False)
    if "model" not in ckpt:
        raise ValueError(f"Bad checkpoint format (no 'model'): {best_model_path}")
    sd = ckpt["model"]
    k_fb = "f.net.0.weight"
    k_gm = "g.net.0.weight"
    if k_fb not in sd or k_gm not in sd:
        raise KeyError(f"Cannot find '{k_fb}' or '{k_gm}' in checkpoint.")
    return int(sd[k_fb].shape[1]), int(sd[k_gm].shape[1])


def load_single_fold_model(device: torch.device, fold_id: int) -> DualMicroBrainBridge:
    ckpt_root = _resolve_ckpt_dir()
    best_path = os.path.join(ckpt_root, f"fold_{fold_id:02d}", "best_model.pt")
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"best_model.pt not found: {best_path}")
    d_b, d_m = _infer_model_dims_from_ckpt(best_path)
    model = DualMicroBrainBridge(
        d_b=d_b, d_m=d_m,
        d_align=int(CONFIG["D_ALIGN"]),
        dropout=float(CONFIG["DROPOUT"]),
        n_heads=int(CONFIG["N_HEADS"]),
    ).to(device)
    try:
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def kfold_indices(n: int, n_folds: int, seed: int):
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    out = []
    for f in range(n_folds):
        val_idx = folds[f]
        train_idx = np.setdiff1d(idx, val_idx, assume_unique=False)
        out.append((train_idx, val_idx))
    return out




# ============================================================
# OOF fold resolution and held-out validation-centroid preparation
# ============================================================
def _normalize_fold_numbers(fold_values: np.ndarray, n_folds: int) -> np.ndarray:
    folds = np.asarray(fold_values, dtype=int)
    unique = np.sort(np.unique(folds))
    if np.array_equal(unique, np.arange(n_folds)):
        folds = folds + 1
        unique = np.sort(np.unique(folds))
    expected = np.arange(1, n_folds + 1)
    if not np.array_equal(unique, expected):
        raise ValueError(
            f"Fold labels must be 1..{n_folds} or 0..{n_folds-1}; found {unique.tolist()}"
        )
    return folds


def _read_folds_from_csv(csv_path: str, sids: List[str], n_folds: int):
    if not csv_path or not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    lower = {str(c).lower().strip(): c for c in df.columns}
    sid_col = next((lower[k] for k in ["subject_id", "sample_id", "sid", "id"] if k in lower), None)
    fold_col = next((lower[k] for k in ["fold", "fold_id", "cv_fold", "val_fold", "oof_fold"] if k in lower), None)
    if sid_col is None or fold_col is None:
        raise ValueError(
            f"FOLD_ASSIGNMENT_CSV must contain a subject ID column "
            f"(subject_id/sample_id/sid/id) and a fold column "
            f"(fold/fold_id/cv_fold/val_fold/oof_fold): {csv_path}"
        )

    split_col = next((lower[k] for k in ["split", "set", "subset", "role"] if k in lower), None)
    if split_col is not None:
        split = df[split_col].astype(str).str.lower().str.strip()
        is_val = split.str.contains(r"^(val|valid|validation|test|oof)", regex=True)
        if is_val.any():
            df = df.loc[is_val].copy()

    df[sid_col] = df[sid_col].astype(str)
    if df[sid_col].duplicated().any():
        dup = df.loc[df[sid_col].duplicated(), sid_col].tolist()[:10]
        raise ValueError(f"Each subject must appear once as validation subject; duplicates: {dup}")

    df[fold_col] = _normalize_fold_numbers(df[fold_col].to_numpy(), n_folds)
    sid_to_idx = {str(s): i for i, s in enumerate(sids)}
    unknown = sorted(set(df[sid_col]) - set(sid_to_idx))
    if unknown:
        print(f"[FOLD WARNING] Ignoring {len(unknown)} fold-file IDs absent from aligned data.")

    fold_by_sid = {
        str(row[sid_col]): int(row[fold_col])
        for _, row in df.iterrows()
        if str(row[sid_col]) in sid_to_idx
    }
    missing = [s for s in sids if str(s) not in fold_by_sid]
    if missing:
        raise ValueError(
            f"Fold assignment is missing {len(missing)} aligned subjects. Examples: {missing[:10]}"
        )

    all_idx = np.arange(len(sids), dtype=int)
    folds = []
    for fold_id in range(1, n_folds + 1):
        val_idx = np.array([sid_to_idx[s] for s, f in fold_by_sid.items() if f == fold_id], dtype=int)
        if len(val_idx) == 0:
            raise ValueError(f"Fold {fold_id} has no validation subjects in {csv_path}")
        train_idx = np.setdiff1d(all_idx, val_idx, assume_unique=False)
        folds.append((train_idx, np.sort(val_idx)))
    return folds, f"csv:{csv_path}"


def _recursive_find_metadata(obj: Any, candidate_keys: Set[str], max_depth: int = 4, depth: int = 0):
    if depth > max_depth:
        return None
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in candidate_keys:
                return v
        for v in obj.values():
            if isinstance(v, (dict, list, tuple)):
                found = _recursive_find_metadata(v, candidate_keys, max_depth, depth + 1)
                if found is not None:
                    return found
    elif isinstance(obj, (list, tuple)) and len(obj) <= 100:
        for v in obj:
            if isinstance(v, (dict, list, tuple)):
                found = _recursive_find_metadata(v, candidate_keys, max_depth, depth + 1)
                if found is not None:
                    return found
    return None


def _read_folds_from_checkpoints(sids: List[str], n_folds: int):
    sid_to_idx = {str(s): i for i, s in enumerate(sids)}
    all_idx = np.arange(len(sids), dtype=int)
    folds = []
    used_index_metadata = False

    id_keys = {
        "val_subject_ids", "valid_subject_ids", "validation_subject_ids",
        "val_sids", "valid_sids", "oof_subject_ids"
    }
    idx_keys = {"val_idx", "valid_idx", "validation_idx", "val_indices", "valid_indices"}

    for fold_id in range(1, n_folds + 1):
        path = os.path.join(_resolve_ckpt_dir(), f"fold_{fold_id:02d}", "best_model.pt")
        if not os.path.exists(path):
            return None
        ckpt = torch_load_any(path)
        val_ids = _recursive_find_metadata(ckpt, id_keys)
        val_idx_meta = _recursive_find_metadata(ckpt, idx_keys)

        if val_ids is not None:
            if torch.is_tensor(val_ids):
                val_ids = val_ids.detach().cpu().tolist()
            val_ids = [str(x) for x in list(val_ids)]
            missing = [x for x in val_ids if x not in sid_to_idx]
            if missing:
                raise ValueError(
                    f"Checkpoint fold {fold_id} contains validation IDs absent from aligned data: {missing[:10]}"
                )
            val_idx = np.array([sid_to_idx[x] for x in val_ids], dtype=int)
        elif val_idx_meta is not None:
            used_index_metadata = True
            if torch.is_tensor(val_idx_meta):
                val_idx_meta = val_idx_meta.detach().cpu().numpy()
            val_idx = np.asarray(val_idx_meta, dtype=int).ravel()
            if len(val_idx) == 0 or val_idx.min() < 0 or val_idx.max() >= len(sids):
                raise ValueError(f"Invalid validation indices stored in checkpoint fold {fold_id}")
        else:
            return None

        train_idx = np.setdiff1d(all_idx, val_idx, assume_unique=False)
        folds.append((train_idx, np.sort(np.unique(val_idx))))

    all_val = np.concatenate([v for _, v in folds])
    if len(np.unique(all_val)) != len(sids) or len(all_val) != len(sids):
        raise ValueError("Checkpoint validation folds do not form a one-time partition of aligned subjects.")
    source = "checkpoint_subject_ids"
    if used_index_metadata:
        source = "checkpoint_indices; requires identical aligned subject order to training"
    return folds, source


def resolve_strict_oof_folds(sids: List[str], out_dir: str):
    """Resolve the validation folds used by fold_01...fold_K.

    Resolution order:
      1. Explicit FOLD_ASSIGNMENT_CSV.
      2. Existing paths in FOLD_ASSIGNMENT_CANDIDATES.
      3. Validation IDs/indices stored in checkpoints.
      4. Optional deterministic reconstruction with the original kfold_indices.

    Seed reconstruction is exact only if training used the same subject set, the
    same ordered subject list, and the same splitter before any filtering.
    """
    n_folds = int(CONFIG["N_FOLDS"])

    candidate_paths = []
    explicit = str(CONFIG.get("FOLD_ASSIGNMENT_CSV", "") or "").strip()
    if explicit:
        candidate_paths.append(explicit)
    for p in CONFIG.get("FOLD_ASSIGNMENT_CANDIDATES", []):
        p = str(p or "").strip()
        if p and p not in candidate_paths:
            candidate_paths.append(p)

    folds = None
    source = None
    print("[STRICT OOF] Searching fold-assignment files:")
    for p in candidate_paths:
        exists = os.path.exists(p)
        print(f"  - {p} | exists={exists}")
        if not exists:
            continue
        csv_result = _read_folds_from_csv(p, sids, n_folds)
        if csv_result is not None:
            folds, source = csv_result
            break

    if folds is None:
        ckpt_result = _read_folds_from_checkpoints(sids, n_folds)
        if ckpt_result is not None:
            folds, source = ckpt_result

    if folds is None and bool(CONFIG.get("ALLOW_SEED_RECONSTRUCTION", False)):
        expected_n = CONFIG.get("EXPECTED_ALIGNED_N_FOR_SEED_RECONSTRUCTION", None)
        if expected_n is not None and int(expected_n) != len(sids):
            raise ValueError(
                f"Seed reconstruction was requested, but aligned N={len(sids)} differs "
                f"from EXPECTED_ALIGNED_N_FOR_SEED_RECONSTRUCTION={expected_n}. "
                "This would change the folds; provide the exact assignment file instead."
            )
        print(
            "[OOF RECONSTRUCTION] No exact fold file/checkpoint metadata was found. "
            "Recreating folds with the original kfold_indices, SEED, aligned N, and "
            "current ordered subject list."
        )
        folds = kfold_indices(len(sids), n_folds, int(CONFIG["SEED"]))
        source = (
            "reconstructed_from_seed; exact only if training used identical "
            "subject set/order and kfold_indices implementation"
        )

    if folds is None:
        if bool(CONFIG.get("STRICT_OOF_REQUIRE_EXACT_FOLDS", True)):
            raise FileNotFoundError(
                "Strict OOF cannot continue because the exact training validation-fold "
                "assignment was not found and seed reconstruction is disabled. Set "
                "CONFIG['FOLD_ASSIGNMENT_CSV'], add a valid candidate path, store "
                "validation IDs in checkpoints, or enable ALLOW_SEED_RECONSTRUCTION "
                "only after confirming the original subject order and splitter."
            )
        folds = kfold_indices(len(sids), n_folds, int(CONFIG["SEED"]))
        source = "recreated_from_seed_NOT_VERIFIED"

    rows = []
    seen = []
    for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
        seen.extend(np.asarray(val_idx, dtype=int).tolist())
        for pos, idx in enumerate(np.asarray(val_idx, dtype=int)):
            rows.append({
                "fold": fold_id,
                "val_position": pos,
                "aligned_index": int(idx),
                "subject_id": sids[int(idx)],
            })
    if sorted(seen) != list(range(len(sids))):
        raise ValueError(
            "Resolved validation folds are not a complete one-time partition of aligned subjects."
        )

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(
        os.path.join(out_dir, "strict_oof_validation_subject_ids.csv"),
        index=False, encoding="utf-8-sig"
    )
    # Also save the conventional two-column assignment for reuse.
    fold_df[["subject_id", "fold"]].rename(
        columns={"subject_id": "sample_id", "fold": "oof_fold"}
    ).to_csv(
        os.path.join(out_dir, "oof_fold_assignment_resolved.csv"),
        index=False, encoding="utf-8-sig"
    )
    with open(os.path.join(out_dir, "strict_oof_fold_source.txt"), "w", encoding="utf-8") as f:
        f.write(source + "\n")
        f.write(f"seed={CONFIG['SEED']}\n")
        f.write(f"n_subjects={len(sids)}\n")
        f.write("ordered_subject_ids_begin\n")
        for sid in sids:
            f.write(str(sid) + "\n")
        f.write("ordered_subject_ids_end\n")

    print(f"[STRICT OOF] Fold assignment source: {source}")
    print(f"[STRICT OOF] Resolved assignment saved to: "
          f"{os.path.join(out_dir, 'oof_fold_assignment_resolved.csv')}")
    return folds


def _subset_fold_data(
    indices: np.ndarray,
    B_all: torch.Tensor,
    M_all: torch.Tensor,
    A_all: torch.Tensor,
    Y_all: torch.Tensor,
    sids: List[str],
    groups: List[str],
):
    idx_t = torch.tensor(np.asarray(indices, dtype=int), dtype=torch.long)
    return (
        B_all[idx_t].clone(), M_all[idx_t].clone(), A_all[idx_t].clone(), Y_all[idx_t].clone(),
        [sids[int(i)] for i in indices], [groups[int(i)] for i in indices],
    )


def _validation_group_centers_for_fold(
    val_idx: np.ndarray,
    B_all: torch.Tensor,
    M_all: torch.Tensor,
    A_all: torch.Tensor,
    Y_all: torch.Tensor,
    groups: List[str],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Return raw held-out validation centroids for every group present in a fold."""
    groups_arr = np.array([str(g).upper() for g in groups])
    val_idx = np.asarray(val_idx, dtype=int)
    centers: Dict[str, Dict[str, torch.Tensor]] = {}
    requested_groups = [str(ANALYSIS_CONFIG["CONTROL_GROUP"]).upper()] + [
        str(d).upper() for d in ANALYSIS_CONFIG["DISEASE_GROUPS"]
    ]
    for group_name in requested_groups:
        idx = val_idx[groups_arr[val_idx] == group_name]
        if len(idx) == 0:
            continue
        idx_t = torch.tensor(idx, dtype=torch.long)
        centers[group_name] = {
            "B": B_all[idx_t].mean(dim=0, keepdim=True).clone(),
            "M": M_all[idx_t].mean(dim=0, keepdim=True).clone(),
            "A": _require_distribution(
                A_all[idx_t].mean(dim=0, keepdim=True).clone(), eps=CONFIG["EPS"]
            ),
            "Y": Y_all[idx_t].mean(dim=0, keepdim=True).clone(),
            "indices": torch.tensor(idx, dtype=torch.long),
            "n": int(len(idx)),
        }
    return centers


def _centroid_fold_weight(n_hc: int, n_disease: int) -> float:
    """Weight one fold's HC-to-disease centroid pair."""
    mode = str(ANALYSIS_CONFIG.get("CENTROID_FOLD_WEIGHT_MODE", "effective_n")).lower()
    n_hc = int(n_hc)
    n_disease = int(n_disease)
    if n_hc <= 0 or n_disease <= 0:
        return 0.0
    if mode == "equal":
        return 1.0
    if mode == "min_group":
        return float(min(n_hc, n_disease))
    if mode == "total":
        return float(n_hc + n_disease)
    if mode != "effective_n":
        raise ValueError(
            "CENTROID_FOLD_WEIGHT_MODE must be effective_n/equal/min_group/total"
        )
    return float(n_hc * n_disease / (n_hc + n_disease))


def prepare_strict_oof_fold_contexts(
    folds,
    B_all: torch.Tensor,
    M_all: torch.Tensor,
    A_all: torch.Tensor,
    Y_all: torch.Tensor,
    sids: List[str],
    groups: List[str],
    out_dir: str,
    roi_names: List[str],
    taxa_names: List[str],
):
    """
    Build held-out validation group centroids separately in every fold.

    No global centroid is passed to a fold model. In fold k, model k receives:
      validation HC centroid -> validation disease centroid.

    Validation centroids are raw means. They are not bootstrap-shrunk because the
    scientific target is the actually observed held-out group state in that fold.
    """
    contexts = []
    summary_rows = []
    ctrl = str(ANALYSIS_CONFIG["CONTROL_GROUP"]).upper()
    diseases = [str(d).upper() for d in ANALYSIS_CONFIG["DISEASE_GROUPS"]]
    prep_root = os.path.join(out_dir, "validation_centroid_fold_preprocessing")
    ensure_dir(prep_root)

    for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
        fold_dir = os.path.join(prep_root, f"fold_{fold_id:02d}")
        ensure_dir(fold_dir)
        val_idx = np.asarray(val_idx, dtype=int)
        val_centers = _validation_group_centers_for_fold(
            val_idx, B_all, M_all, A_all, Y_all, groups
        )
        if ctrl not in val_centers:
            raise ValueError(f"Fold {fold_id} has no validation {ctrl} subjects.")
        if val_centers[ctrl]["n"] < int(ANALYSIS_CONFIG.get("OOF_MIN_VALIDATION_HC", 1)):
            raise ValueError(
                f"Fold {fold_id} has insufficient validation HC subjects: "
                f"{val_centers[ctrl]['n']}"
            )

        # Save untouched validation membership and the raw A/Y centers used.
        pd.DataFrame({
            "subject_id": [sids[int(i)] for i in val_idx],
            "group": [groups[int(i)] for i in val_idx],
            "split": "validation_centroid_members",
        }).to_csv(
            os.path.join(fold_dir, "validation_subjects_used_for_centroids.csv"),
            index=False, encoding="utf-8-sig"
        )
        for group_name, center in val_centers.items():
            save_matrix_csv(
                center["A"].cpu().numpy(), [group_name], taxa_names,
                os.path.join(fold_dir, f"validation_microbiome_center_{group_name}.csv"),
                "group"
            )
            save_matrix_csv(
                center["Y"].cpu().numpy(), [group_name], roi_names,
                os.path.join(fold_dir, f"validation_brain_center_{group_name}.csv"),
                "group"
            )

        valid_diseases = []
        for d in diseases:
            n_d = int(val_centers[d]["n"]) if d in val_centers else 0
            if n_d < int(ANALYSIS_CONFIG.get("OOF_MIN_VALIDATION_DISEASE", 1)):
                print(
                    f"[CENTROID WARNING] fold={fold_id:02d} has validation {d} n={n_d}; "
                    "this fold will be skipped for that disease."
                )
            else:
                valid_diseases.append(d)

        contexts.append({
            "fold_id": int(fold_id),
            "train_idx": np.asarray(train_idx, dtype=int),
            "val_idx": val_idx,
            "validation_centers": val_centers,
            "valid_diseases": valid_diseases,
            "n_validation": int(len(val_idx)),
        })

        row = {
            "fold": int(fold_id),
            "n_validation": int(len(val_idx)),
            "n_validation_HC": int(val_centers[ctrl]["n"]),
        }
        for d in diseases:
            row[f"n_validation_{d}"] = int(val_centers[d]["n"]) if d in val_centers else 0
            row[f"centroid_weight_{d}"] = _centroid_fold_weight(
                row["n_validation_HC"], row[f"n_validation_{d}"]
            )
        summary_rows.append(row)
        print(
            f"[VALIDATION CENTROIDS] fold={fold_id:02d} | "
            + ", ".join(f"{g}={int(c['n'])}" for g, c in val_centers.items())
        )

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(prep_root, "validation_centroid_fold_summary.csv"),
        index=False, encoding="utf-8-sig"
    )
    return contexts


def interpolate_micro_centers(
    A_hc_center: torch.Tensor,
    A_disease_center: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Exact validation HC centroid -> validation disease centroid interpolation."""
    if str(ANALYSIS_CONFIG["MICRO_SHIFT_SPACE"]).lower() == "clr":
        return shift_micro_subjectwise(A_hc_center, A_disease_center, lam)
    return _require_distribution(
        (1.0 - float(lam)) * A_hc_center + float(lam) * A_disease_center,
        eps=CONFIG["EPS"],
    )


def interpolate_brain_centers(
    Y_hc_center: torch.Tensor,
    Y_disease_center: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Exact validation HC centroid -> validation disease centroid interpolation."""
    return (1.0 - float(lam)) * Y_hc_center + float(lam) * Y_disease_center


@torch.no_grad()
def predict_micro2brain_single_model(model: torch.nn.Module, M_in: torch.Tensor, A_in: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
    y_preds = []
    model.eval()
    for i in range(0, M_in.size(0), batch_size):
        M_b = M_in[i:i + batch_size].to(device)
        A_b = A_in[i:i + batch_size].to(device)
        y_hat, _, *_ = model(B_true=None, M_true=M_b, A_true=A_b, Y_true=None, mode="micro2brain")
        y_preds.append(y_hat.detach().cpu())
    return torch.cat(y_preds, dim=0)


@torch.no_grad()
def predict_brain2micro_single_model(model: torch.nn.Module, B_in: torch.Tensor, Y_in: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
    a_preds = []
    model.eval()
    for i in range(0, B_in.size(0), batch_size):
        B_b = B_in[i:i + batch_size].to(device)
        Y_b = Y_in[i:i + batch_size].to(device)
        _, a_hat, *_ = model(B_true=B_b, M_true=None, A_true=None, Y_true=Y_b, mode="brain2micro")
        a_preds.append(a_hat.detach().cpu())
    return torch.cat(a_preds, dim=0)


# ============================================================
# Group-center and centroid shift construction
# ============================================================









# ============================================================
# Shift construction
# ============================================================
def compute_group_centers(A_all: torch.Tensor, Y_all: torch.Tensor, groups: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    groups_arr = np.array([g.upper() for g in groups])
    all_groups = [ANALYSIS_CONFIG["CONTROL_GROUP"].upper()] + [g.upper() for g in ANALYSIS_CONFIG["DISEASE_GROUPS"]]
    A_np = A_all.cpu().numpy()
    Y_np = Y_all.cpu().numpy()
    centers_A = {}
    centers_Y = {}
    for g in all_groups:
        idx = np.where(groups_arr == g)[0]
        if len(idx) == 0:
            raise ValueError(f"No samples found for group: {g}")
        if ANALYSIS_CONFIG["MICRO_SHIFT_SPACE"].lower() == "clr":
            z = clr_np(A_np[idx], eps=CONFIG["EPS"])
            centers_A[g] = inv_clr_np(z.mean(axis=0, keepdims=True), eps=CONFIG["EPS"])[0]
        else:
            centers_A[g] = require_distribution_np(A_np[idx].mean(axis=0, keepdims=True), eps=CONFIG["EPS"])[0]
        centers_Y[g] = Y_np[idx].mean(axis=0)
    return centers_A, centers_Y
def _bootstrap_center_noise_np(X: np.ndarray, n_boot: int, rng: np.random.RandomState, eps: float = 1e-12) -> Tuple[float, float]:
    """Return mean and 95th percentile bootstrap center displacement from raw center."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n <= 1 or n_boot <= 0:
        return 0.0, 0.0
    raw = X.mean(axis=0)
    dists = []
    for _ in range(int(n_boot)):
        idx = rng.randint(0, n, size=n)
        cb = X[idx].mean(axis=0)
        dists.append(float(np.linalg.norm(cb - raw)))
    dists = np.asarray(dists, dtype=np.float64)
    return float(dists.mean()), float(np.percentile(dists, 95))


def apply_bootstrap_shrinkage_centers(
    centers_A_raw: Dict[str, np.ndarray],
    centers_Y_raw: Dict[str, np.ndarray],
    A_all: torch.Tensor,
    Y_all: torch.Tensor,
    groups: List[str],
    out_dir: str,
    taxa_names: List[str],
    roi_names: List[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], pd.DataFrame]:
    """
    Compute bootstrap-adaptive shrinkage centers for disease groups.

    center_shrink = alpha * center_raw + (1 - alpha) * center_ref
    alpha = effect / (effect + bootstrap_noise + eps)

    For microbiome data, shrinkage is performed in CLR space when MICRO_SHIFT_SPACE='clr',
    otherwise in relative abundance space with renormalization.
    """
    center_dir = os.path.join(out_dir, "group_centers")
    ensure_dir(center_dir)
    ctrl = ANALYSIS_CONFIG["CONTROL_GROUP"].upper()
    diseases = [d.upper() for d in ANALYSIS_CONFIG["DISEASE_GROUPS"]]
    groups_arr = np.array([g.upper() for g in groups])
    A_np = A_all.cpu().numpy()
    Y_np = Y_all.cpu().numpy()

    # Save raw centers for transparency.
    for g, vec in centers_A_raw.items():
        save_matrix_csv(vec.reshape(1, -1), [g], taxa_names, os.path.join(center_dir, f"RAW_microbiome_center_{g}.csv"), "group")
    for g, vec in centers_Y_raw.items():
        save_matrix_csv(vec.reshape(1, -1), [g], roi_names, os.path.join(center_dir, f"RAW_brain_center_{g}.csv"), "group")

    if not bool(ANALYSIS_CONFIG.get("USE_BOOTSTRAP_SHRINKAGE_CENTER", True)):
        info = []
        for g in [ctrl] + diseases:
            info.append({"group": g, "n": int(np.sum(groups_arr == g)), "alpha_A": 1.0, "alpha_Y": 1.0, "applied": False})
        df = pd.DataFrame(info)
        df.to_csv(os.path.join(center_dir, "bootstrap_shrinkage_center_info.csv"), index=False, encoding="utf-8-sig")
        return centers_A_raw, centers_Y_raw, df

    target = str(ANALYSIS_CONFIG.get("SHRINKAGE_TARGET", "grand_mean")).lower()
    n_boot = int(ANALYSIS_CONFIG.get("SHRINKAGE_N_BOOT", 1000))
    seed = int(ANALYSIS_CONFIG.get("SHRINKAGE_BOOTSTRAP_SEED", 2026))
    a_min = float(ANALYSIS_CONFIG.get("SHRINKAGE_ALPHA_MIN", 0.30))
    a_max = float(ANALYSIS_CONFIG.get("SHRINKAGE_ALPHA_MAX", 0.95))
    apply_ctrl = bool(ANALYSIS_CONFIG.get("SHRINKAGE_APPLY_TO_CONTROL", False))
    no_shrink_groups = {str(x).upper() for x in ANALYSIS_CONFIG.get("NO_SHRINKAGE_GROUPS", [])}
    rng = np.random.RandomState(seed)

    if target == "hc":
        A_ref = centers_A_raw[ctrl].copy()
        Y_ref = centers_Y_raw[ctrl].copy()
        A_ref_space = clr_np(A_ref.reshape(1, -1), eps=CONFIG["EPS"])[0] if ANALYSIS_CONFIG["MICRO_SHIFT_SPACE"].lower() == "clr" else A_ref
    elif target == "grand_mean":
        Y_ref = Y_np.mean(axis=0)
        if ANALYSIS_CONFIG["MICRO_SHIFT_SPACE"].lower() == "clr":
            A_ref_space = clr_np(A_np, eps=CONFIG["EPS"]).mean(axis=0)
            A_ref = inv_clr_np(A_ref_space.reshape(1, -1), eps=CONFIG["EPS"])[0]
        else:
            A_ref = require_distribution_np(A_np.mean(axis=0, keepdims=True), eps=CONFIG["EPS"])[0]
            A_ref_space = A_ref
    else:
        raise ValueError("SHRINKAGE_TARGET must be 'grand_mean' or 'hc'.")

    centers_A = {g: v.copy() for g, v in centers_A_raw.items()}
    centers_Y = {g: v.copy() for g, v in centers_Y_raw.items()}
    rows = []

    for g in [ctrl] + diseases:
        idx = np.where(groups_arr == g)[0]
        n_g = len(idx)
        do_apply = not (g == ctrl and not apply_ctrl)
        if g in no_shrink_groups:
            do_apply = False
        if not do_apply:
            alpha_A = alpha_Y = 1.0
            effect_A = effect_Y = noise_A = noise_Y = p95_A = p95_Y = 0.0
            centers_A[g] = centers_A_raw[g].copy()
            centers_Y[g] = centers_Y_raw[g].copy()
        else:
            Y_group = Y_np[idx]
            noise_Y, p95_Y = _bootstrap_center_noise_np(Y_group, n_boot, rng)
            effect_Y = float(np.linalg.norm(centers_Y_raw[g] - Y_ref))
            alpha_Y = effect_Y / (effect_Y + noise_Y + 1e-12)
            alpha_Y = float(np.clip(alpha_Y, a_min, a_max))
            centers_Y[g] = alpha_Y * centers_Y_raw[g] + (1.0 - alpha_Y) * Y_ref

            if ANALYSIS_CONFIG["MICRO_SHIFT_SPACE"].lower() == "clr":
                A_group_space = clr_np(A_np[idx], eps=CONFIG["EPS"])
                A_raw_space = clr_np(centers_A_raw[g].reshape(1, -1), eps=CONFIG["EPS"])[0]
                noise_A, p95_A = _bootstrap_center_noise_np(A_group_space, n_boot, rng)
                effect_A = float(np.linalg.norm(A_raw_space - A_ref_space))
                alpha_A = effect_A / (effect_A + noise_A + 1e-12)
                alpha_A = float(np.clip(alpha_A, a_min, a_max))
                A_new_space = alpha_A * A_raw_space + (1.0 - alpha_A) * A_ref_space
                centers_A[g] = inv_clr_np(A_new_space.reshape(1, -1), eps=CONFIG["EPS"])[0]
            else:
                A_group_space = A_np[idx]
                noise_A, p95_A = _bootstrap_center_noise_np(A_group_space, n_boot, rng)
                effect_A = float(np.linalg.norm(centers_A_raw[g] - A_ref_space))
                alpha_A = effect_A / (effect_A + noise_A + 1e-12)
                alpha_A = float(np.clip(alpha_A, a_min, a_max))
                A_new = alpha_A * centers_A_raw[g] + (1.0 - alpha_A) * A_ref
                centers_A[g] = require_distribution_np(A_new.reshape(1, -1), eps=CONFIG["EPS"])[0]

        rows.append({
            "group": g,
            "n": int(n_g),
            "target": target,
            "applied": bool(do_apply),
            "alpha_A": float(alpha_A),
            "alpha_Y": float(alpha_Y),
            "shrinkage_strength_A": float(1.0 - alpha_A),
            "shrinkage_strength_Y": float(1.0 - alpha_Y),
            "effect_A": float(effect_A),
            "effect_Y": float(effect_Y),
            "noise_A_mean": float(noise_A),
            "noise_Y_mean": float(noise_Y),
            "noise_A_p95": float(p95_A),
            "noise_Y_p95": float(p95_Y),
        })

    info_df = pd.DataFrame(rows)
    info_df.to_csv(os.path.join(center_dir, "bootstrap_shrinkage_center_info.csv"), index=False, encoding="utf-8-sig")

    # Save the centers actually used as centroid targets.
    for g, vec in centers_A.items():
        save_matrix_csv(vec.reshape(1, -1), [g], taxa_names, os.path.join(center_dir, f"microbiome_center_{g}.csv"), "group")
    for g, vec in centers_Y.items():
        save_matrix_csv(vec.reshape(1, -1), [g], roi_names, os.path.join(center_dir, f"brain_center_{g}.csv"), "group")
    return centers_A, centers_Y, info_df





def shift_micro_individual(A: torch.Tensor, delta_A: np.ndarray, lam: float) -> torch.Tensor:
    eps = CONFIG["EPS"]
    if ANALYSIS_CONFIG["MICRO_SHIFT_SPACE"].lower() == "clr":
        A_np = A.detach().cpu().numpy()
        z = clr_np(A_np, eps=eps)
        # delta_A should be a relative-space delta. Convert disease-center difference to CLR outside if needed.
        raise NotImplementedError("For CLR shifts, use shift_micro_individual_clr with delta_Z.")
    else:
        d = torch.tensor(delta_A, dtype=A.dtype, device=A.device).view(1, -1)
        return _require_distribution(A + float(lam) * d, eps=eps)


def shift_micro_individual_clr(A: torch.Tensor, delta_Z: np.ndarray, lam: float) -> torch.Tensor:
    A_np = A.detach().cpu().numpy()
    z = clr_np(A_np, eps=CONFIG["EPS"])
    z2 = z + float(lam) * delta_Z.reshape(1, -1)
    out = inv_clr_np(z2, eps=CONFIG["EPS"])
    return torch.tensor(out, dtype=A.dtype, device=A.device)


def shift_brain_individual(Y: torch.Tensor, delta_Y: np.ndarray, lam: float) -> torch.Tensor:
    d = torch.tensor(delta_Y, dtype=Y.dtype, device=Y.device).view(1, -1)
    return Y + float(lam) * d




def shift_micro_subjectwise(A_src: torch.Tensor, A_tgt: torch.Tensor, lam: float) -> torch.Tensor:
    """Pair-specific relative-abundance interpolation: A_src -> A_tgt."""
    if ANALYSIS_CONFIG["MICRO_SHIFT_SPACE"].lower() == "clr":
        z_src = clr_np(A_src.detach().cpu().numpy(), eps=CONFIG["EPS"])
        z_tgt = clr_np(A_tgt.detach().cpu().numpy(), eps=CONFIG["EPS"])
        out = inv_clr_np(z_src + float(lam) * (z_tgt - z_src), eps=CONFIG["EPS"])
        return torch.tensor(out, dtype=A_src.dtype, device=A_src.device)
    return _require_distribution(A_src + float(lam) * (A_tgt - A_src), eps=CONFIG["EPS"])


def shift_brain_subjectwise(Y_src: torch.Tensor, Y_tgt: torch.Tensor, lam: float) -> torch.Tensor:
    """Pair-specific brain interpolation: Y_src -> Y_tgt."""
    return Y_src + float(lam) * (Y_tgt - Y_src)


def get_micro_delta_for_disease(centers_A: Dict[str, np.ndarray], disease: str, A_all: torch.Tensor, groups: List[str]):
    ctrl = ANALYSIS_CONFIG["CONTROL_GROUP"].upper()
    disease = disease.upper()
    if ANALYSIS_CONFIG["MICRO_SHIFT_SPACE"].lower() == "clr":
        groups_arr = np.array([g.upper() for g in groups])
        A_np = A_all.cpu().numpy()
        z_ctrl = clr_np(A_np[groups_arr == ctrl], eps=CONFIG["EPS"]).mean(axis=0)
        z_dis = clr_np(A_np[groups_arr == disease], eps=CONFIG["EPS"]).mean(axis=0)
        return z_dis - z_ctrl
    return centers_A[disease] - centers_A[ctrl]


# ============================================================
# Metrics and plotting data
# ============================================================
def brain_metrics(pred_mean: np.ndarray, target_mean: np.ndarray) -> Dict[str, float]:
    diff = pred_mean - target_mean
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "pearson_r": pearson_np(pred_mean, target_mean),
        "cosine": cosine_np(pred_mean, target_mean),
    }


def micro_metrics(pred_mean: np.ndarray, target_mean: np.ndarray) -> Dict[str, float]:
    pred_mean = require_distribution_np(pred_mean.reshape(1, -1), eps=CONFIG["EPS"])[0]
    target_mean = require_distribution_np(target_mean.reshape(1, -1), eps=CONFIG["EPS"])[0]
    return {
        "bray_curtis": bray_curtis_np(pred_mean, target_mean),
        "js_divergence": js_divergence_np(pred_mean, target_mean, eps=CONFIG["EPS"]),
        "aitchison": aitchison_np(pred_mean, target_mean, eps=CONFIG["EPS"]),
        "pearson_r": pearson_np(pred_mean, target_mean),
        "spearman_r": float(pd.Series(pred_mean).corr(pd.Series(target_mean), method="spearman")),
    }


def plot_lambda_curve(df: pd.DataFrame, out_png: str, title: str, metric: str, ylabel: str):
    plt.figure(figsize=(8, 5))
    for disease, sub in df.groupby("disease"):
        sub = sub.sort_values("lambda")
        plt.plot(sub["lambda"], sub[metric], marker="o", label=disease)
    plt.xlabel("Lambda: disease-shift strength")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=ANALYSIS_CONFIG["FIG_DPI"], bbox_inches="tight")
    plt.close()


def plot_error_improvement(df: pd.DataFrame, out_png: str, title: str, metric: str):
    rows = []
    for disease, sub in df.groupby("disease"):
        s0 = sub.loc[np.isclose(sub["lambda"], 0.0)]
        s1 = sub.loc[np.isclose(sub["lambda"], 1.0)]
        if len(s0) and len(s1):
            before = float(s0.iloc[0][metric])
            after = float(s1.iloc[0][metric])
            rows.append({"disease": disease, "before": before, "after": after, "improvement": before - after})
    out = pd.DataFrame(rows)
    out.to_csv(out_png.replace(".png", "_data.csv"), index=False, encoding="utf-8-sig")
    x = np.arange(len(out))
    w = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - w/2, out["before"], width=w, label="Before shift")
    plt.bar(x + w/2, out["after"], width=w, label="After shift")
    plt.xticks(x, out["disease"])
    plt.ylabel(metric)
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=ANALYSIS_CONFIG["FIG_DPI"], bbox_inches="tight")
    plt.close()


def plot_pattern_scatter(x: np.ndarray, y: np.ndarray, names: List[str], out_png: str, title: str, xlabel: str, ylabel: str, label_top: int = 10):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    df = pd.DataFrame({"name": names, "real_delta": x, "pred_delta": y, "abs_pred_delta": np.abs(y)})
    df.to_csv(out_png.replace(".png", "_scatter_data.csv"), index=False, encoding="utf-8-sig")
    plt.figure(figsize=(6.5, 6))
    plt.scatter(x, y, s=18, alpha=0.75)
    if len(x) > 1:
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    for _, row in df.sort_values("abs_pred_delta", ascending=False).head(label_top).iterrows():
        plt.text(row["real_delta"], row["pred_delta"], str(row["name"]), fontsize=7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title}\nr = {pearson_np(x, y):.3f}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=ANALYSIS_CONFIG["FIG_DPI"], bbox_inches="tight")
    plt.close()


def plot_trajectory(coords_df: pd.DataFrame, out_png: str, title: str):
    plt.figure(figsize=(7, 6))
    for disease, sub in coords_df[coords_df["kind"] == "predicted_path"].groupby("disease"):
        sub = sub.sort_values("lambda")
        plt.plot(sub["PC1"], sub["PC2"], marker="o", label=f"Pred HC->{disease}")
        for _, r in sub.iterrows():
            plt.text(r["PC1"], r["PC2"], f"{r['lambda']:.2g}", fontsize=7)
    real = coords_df[coords_df["kind"] == "real_group_center"]
    plt.scatter(real["PC1"], real["PC2"], marker="s", s=70, label="Real group centers")
    for _, r in real.iterrows():
        plt.text(r["PC1"], r["PC2"], r["disease"], fontsize=9, fontweight="bold")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=ANALYSIS_CONFIG["FIG_DPI"], bbox_inches="tight")
    plt.close()


def plot_heatmap_top(mat: np.ndarray, row_names: List[str], col_names: List[str], out_png: str, title: str, row_label: str, col_label: str):
    # Select top rows / columns by absolute contribution for readability. Full matrix is saved separately.
    max_rows = min(int(ANALYSIS_CONFIG["HEATMAP_TOP_TAXA"]), mat.shape[0])
    max_cols = min(int(ANALYSIS_CONFIG["HEATMAP_TOP_ROI"]), mat.shape[1])
    row_score = np.abs(mat).sum(axis=1)
    col_score = np.abs(mat).sum(axis=0)
    ridx = np.argsort(-row_score)[:max_rows]
    cidx = np.argsort(-col_score)[:max_cols]
    sub = mat[np.ix_(ridx, cidx)]
    rnames = [row_names[i] for i in ridx]
    cnames = [col_names[i] for i in cidx]
    pd.DataFrame(sub, index=rnames, columns=cnames).to_csv(out_png.replace(".png", "_data.csv"), encoding="utf-8-sig")
    vmax = np.percentile(np.abs(sub), 98) if sub.size else 1.0
    vmax = max(float(vmax), 1e-12)
    plt.figure(figsize=ANALYSIS_CONFIG["HEATMAP_FIGSIZE"])
    plt.imshow(sub, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    plt.colorbar(label="Signed contribution")
    plt.xticks(np.arange(len(cnames)), cnames, rotation=90, fontsize=5)
    plt.yticks(np.arange(len(rnames)), rnames, fontsize=5)
    plt.xlabel(col_label)
    plt.ylabel(row_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=ANALYSIS_CONFIG["FIG_DPI"], bbox_inches="tight")
    plt.close()


def build_top_edges(mat: np.ndarray, row_names: List[str], col_names: List[str], topk: int, src_label: str, tgt_label: str) -> pd.DataFrame:
    flat = np.argsort(-np.abs(mat).reshape(-1))[:min(topk, mat.size)]
    rows, cols = np.unravel_index(flat, mat.shape)
    records = []
    for rank, (r, c) in enumerate(zip(rows, cols), start=1):
        records.append({
            "rank": rank,
            "source_index": int(r),
            "target_index": int(c),
            src_label: row_names[r],
            tgt_label: col_names[c],
            "contribution_signed": float(mat[r, c]),
            "contribution_abs": float(abs(mat[r, c])),
            "edge_id": f"{row_names[r]}|||{col_names[c]}",
        })
    return pd.DataFrame(records)


def plot_top_bar(scores: np.ndarray, names: List[str], out_png: str, title: str, xlabel: str, ylabel: str, topk: int):
    idx = np.argsort(-scores)[:min(topk, len(scores))]
    df = pd.DataFrame({"name": [names[i] for i in idx], "score": scores[idx]})
    df.to_csv(out_png.replace(".png", "_data.csv"), index=False, encoding="utf-8-sig")
    plt.figure(figsize=(9, max(4, 0.32 * len(df))))
    plt.barh(np.arange(len(df)), df["score"])
    plt.yticks(np.arange(len(df)), df["name"], fontsize=8)
    plt.gca().invert_yaxis()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=ANALYSIS_CONFIG["FIG_DPI"], bbox_inches="tight")
    plt.close()


def plot_simple_chord(edges_df: pd.DataFrame, src_col: str, tgt_col: str, out_png: str, title: str):
    """A lightweight chord-like circular diagram using matplotlib Bezier curves."""
    if edges_df.empty:
        return
    edges_df.to_csv(out_png.replace(".png", "_data.csv"), index=False, encoding="utf-8-sig")
    src_nodes = edges_df[src_col].drop_duplicates().tolist()
    tgt_nodes = edges_df[tgt_col].drop_duplicates().tolist()
    nodes = src_nodes + [n for n in tgt_nodes if n not in src_nodes]
    n = len(nodes)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {node: np.array([np.cos(a), np.sin(a)]) for node, a in zip(nodes, angles)}
    max_w = edges_df["contribution_abs"].max() if "contribution_abs" in edges_df.columns else edges_df["edge_abs"].max()
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.set_aspect("equal")
    for node, xy in pos.items():
        ax.scatter([xy[0]], [xy[1]], s=90)
        ha = "left" if xy[0] >= 0 else "right"
        ax.text(1.08 * xy[0], 1.08 * xy[1], node, fontsize=7, ha=ha, va="center")
    for _, row in edges_df.iterrows():
        s = row[src_col]
        t = row[tgt_col]
        p0 = pos[s]
        p2 = pos[t]
        ctrl = np.array([0.0, 0.0])
        signed = float(row.get("contribution_signed", row.get("edge_signed", 0.0)))
        absval = float(row.get("contribution_abs", row.get("edge_abs", abs(signed))))
        lw = 0.5 + 4.5 * absval / max(max_w, 1e-12)
        color = "#d62728" if signed >= 0 else "#1f77b4"
        path = Path([p0, ctrl, p2], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
        patch = PathPatch(path, facecolor="none", edgecolor=color, lw=lw, alpha=0.65)
        ax.add_patch(patch)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=ANALYSIS_CONFIG["FIG_DPI"], bbox_inches="tight")
    plt.close()


def plot_waterfall(contrib_vec: np.ndarray, source_names: List[str], target_name: str, out_png: str, title: str, topk: int):
    order = np.argsort(-np.abs(contrib_vec))[:min(topk, len(contrib_vec))]
    vals = contrib_vec[order]
    names = [source_names[i] for i in order]
    other = float(contrib_vec.sum() - vals.sum())
    wf_names = names + ["Other"]
    wf_vals = np.concatenate([vals, [other]])
    starts = np.r_[0, np.cumsum(wf_vals)[:-1]]
    df = pd.DataFrame({"component": wf_names, "contribution": wf_vals, "start": starts, "end": starts + wf_vals})
    df.to_csv(out_png.replace(".png", "_data.csv"), index=False, encoding="utf-8-sig")
    plt.figure(figsize=(11, 5.5))
    for i, row in df.iterrows():
        plt.bar(i, row["contribution"], bottom=row["start"])
    plt.axhline(0, linewidth=0.8)
    plt.xticks(np.arange(len(df)), df["component"], rotation=60, ha="right", fontsize=8)
    plt.ylabel("Cumulative signed contribution")
    plt.title(f"{title}\nTarget: {target_name}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=ANALYSIS_CONFIG["FIG_DPI"], bbox_inches="tight")
    plt.close()



# ============================================================
# Held-out validation group-centroid shifted prediction
# ============================================================
def _save_oof_matrix_with_metadata(
    matrix: np.ndarray,
    metadata: List[Dict[str, Any]],
    value_names: List[str],
    out_csv: str,
):
    meta_df = pd.DataFrame(metadata).reset_index(drop=True)
    value_df = pd.DataFrame(np.asarray(matrix), columns=value_names)
    pd.concat([meta_df, value_df], axis=1).to_csv(out_csv, index=False, encoding="utf-8-sig")


def _weighted_vector_mean(vectors: List[np.ndarray], weights: List[float]) -> np.ndarray:
    if not vectors:
        raise ValueError("No vectors were supplied for weighted aggregation.")
    stack = np.stack([np.asarray(v).reshape(-1) for v in vectors], axis=0)
    w = np.asarray(weights, dtype=float)
    if len(w) != stack.shape[0] or np.sum(w) <= 0:
        raise ValueError("Invalid fold weights for weighted aggregation.")
    return np.average(stack, axis=0, weights=w)


def _weighted_scalar_mean(values: List[float], weights: List[float]) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    return float(np.average(v, weights=w))


def _pooled_observed_center(
    fold_contexts,
    group_name: str,
    modality: str,
) -> np.ndarray:
    """Pool validation fold centers by their actual group sizes."""
    vecs, ns = [], []
    group_name = str(group_name).upper()
    for ctx in fold_contexts:
        centers = ctx["validation_centers"]
        if group_name not in centers:
            continue
        vecs.append(centers[group_name][modality].cpu().numpy().reshape(-1))
        ns.append(float(centers[group_name]["n"]))
    return _weighted_vector_mean(vecs, ns)


def run_shift_predictions(
    fold_contexts,
    B_all: torch.Tensor,
    M_all: torch.Tensor,
    A_all: torch.Tensor,
    Y_all: torch.Tensor,
    sids: List[str],
    groups: List[str],
    out_dir: str,
    device: torch.device,
    roi_names: List[str],
    taxa_names: List[str],
):
    """
    Held-out validation-centroid transition analysis.

    In fold k and disease d:
      1) average validation HC subjects into one HC centroid;
      2) average validation disease subjects into one disease centroid;
      3) interpolate exactly HC centroid -> disease centroid;
      4) use only fold-k model to predict that fold's centroid path;
      5) combine fold-level predictions using effective two-group sample-size weights.
    """
    pred_dir = os.path.join(out_dir, "validation_centroid_shift_predictions")
    ensure_dir(pred_dir)
    fold_pred_dir = os.path.join(pred_dir, "fold_centroid_predictions")
    fold_pattern_dir = os.path.join(pred_dir, "foldwise_patterns")
    shifted_dir = os.path.join(pred_dir, "shifted_centroid_inputs")
    ensure_dir(fold_pred_dir)
    ensure_dir(fold_pattern_dir)
    if bool(ANALYSIS_CONFIG.get("OOF_SAVE_SHIFTED_INPUTS", True)):
        ensure_dir(shifted_dir)

    batch_size = int(CONFIG["BATCH_SIZE"])
    ctrl = str(ANALYSIS_CONFIG["SOURCE_GROUP_FOR_SHIFT"]).upper()
    diseases = [str(d).upper() for d in ANALYSIS_CONFIG["DISEASE_GROUPS"]]
    lambdas = sorted(float(x) for x in ANALYSIS_CONFIG["LAMBDA_VALUES"])
    if 0.0 not in lambdas or 1.0 not in lambdas:
        raise ValueError("LAMBDA_VALUES must contain 0.0 and 1.0.")

    # One entry per usable fold, not one entry per subject.
    y_pred_store = {d: {lam: [] for lam in lambdas} for d in diseases}
    a_pred_store = {d: {lam: [] for lam in lambdas} for d in diseases}
    y_source_store = {d: {lam: [] for lam in lambdas} for d in diseases}
    a_source_store = {d: {lam: [] for lam in lambdas} for d in diseases}
    y_target_store = {d: {lam: [] for lam in lambdas} for d in diseases}
    a_target_store = {d: {lam: [] for lam in lambdas} for d in diseases}
    fold_ids_store = {d: {lam: [] for lam in lambdas} for d in diseases}
    weights_store = {d: {lam: [] for lam in lambdas} for d in diseases}
    fold_pattern_rows = []
    fold_metric_m2b_rows = []
    fold_metric_b2m_rows = []

    for ctx in fold_contexts:
        fold_id = int(ctx["fold_id"])
        centers = ctx["validation_centers"]
        hc = centers[ctrl]
        B_hc = hc["B"]
        M_hc = hc["M"]
        A_hc = hc["A"]
        Y_hc = hc["Y"]
        n_hc = int(hc["n"])
        model = load_single_fold_model(device, fold_id)

        for d in diseases:
            if d not in centers:
                continue
            dis = centers[d]
            n_dis = int(dis["n"])
            weight = _centroid_fold_weight(n_hc, n_dis)
            if weight <= 0:
                continue
            A_dis = dis["A"]
            Y_dis = dis["Y"]
            fold_y, fold_a = {}, {}

            print(
                f"[VAL CENTROID SHIFT] fold={fold_id:02d} | {ctrl}(n={n_hc}) "
                f"-> {d}(n={n_dis}) | weight={weight:.4f}"
            )

            for lam in lambdas:
                A_shift = interpolate_micro_centers(A_hc, A_dis, lam)
                Y_shift = interpolate_brain_centers(Y_hc, Y_dis, lam)
                y_pred = predict_micro2brain_single_model(
                    model, M_hc, A_shift, device, batch_size
                ).cpu().numpy()
                a_pred = predict_brain2micro_single_model(
                    model, B_hc, Y_shift, device, batch_size
                ).cpu().numpy()
                a_pred = require_distribution_np(a_pred, eps=CONFIG["EPS"])

                fold_y[lam] = y_pred.reshape(-1)
                fold_a[lam] = a_pred.reshape(-1)
                y_pred_store[d][lam].append(y_pred.reshape(-1))
                a_pred_store[d][lam].append(a_pred.reshape(-1))
                y_source_store[d][lam].append(Y_hc.cpu().numpy().reshape(-1))
                a_source_store[d][lam].append(A_hc.cpu().numpy().reshape(-1))
                y_target_store[d][lam].append(Y_dis.cpu().numpy().reshape(-1))
                a_target_store[d][lam].append(A_dis.cpu().numpy().reshape(-1))
                fold_ids_store[d][lam].append(fold_id)
                weights_store[d][lam].append(weight)

                metadata = [{
                    "fold": fold_id,
                    "source_group": ctrl,
                    "target_disease": d,
                    "lambda": lam,
                    "n_validation_HC": n_hc,
                    "n_validation_disease": n_dis,
                    "fold_weight": weight,
                }]
                _save_oof_matrix_with_metadata(
                    y_pred, metadata, roi_names,
                    os.path.join(
                        fold_pred_dir,
                        f"fold{fold_id:02d}_micro_to_brain_Ypred_{ctrl}_to_{d}_lambda_{lam:.2f}.csv"
                    )
                )
                _save_oof_matrix_with_metadata(
                    a_pred, metadata, taxa_names,
                    os.path.join(
                        fold_pred_dir,
                        f"fold{fold_id:02d}_brain_to_micro_Apred_{ctrl}_to_{d}_lambda_{lam:.2f}.csv"
                    )
                )
                if bool(ANALYSIS_CONFIG.get("OOF_SAVE_SHIFTED_INPUTS", True)):
                    _save_oof_matrix_with_metadata(
                        A_shift.cpu().numpy(), metadata, taxa_names,
                        os.path.join(
                            shifted_dir,
                            f"fold{fold_id:02d}_Acenter_{ctrl}_to_{d}_lambda_{lam:.2f}.csv"
                        )
                    )
                    _save_oof_matrix_with_metadata(
                        Y_shift.cpu().numpy(), metadata, roi_names,
                        os.path.join(
                            shifted_dir,
                            f"fold{fold_id:02d}_Ycenter_{ctrl}_to_{d}_lambda_{lam:.2f}.csv"
                        )
                    )

                bm = brain_metrics(y_pred.reshape(-1), Y_dis.cpu().numpy().reshape(-1))
                mm = micro_metrics(a_pred.reshape(-1), A_dis.cpu().numpy().reshape(-1))
                fold_metric_m2b_rows.append({
                    "fold": fold_id, "disease": d, "lambda": lam,
                    "n_validation_HC": n_hc, "n_validation_disease": n_dis,
                    "fold_weight": weight, **bm,
                })
                fold_metric_b2m_rows.append({
                    "fold": fold_id, "disease": d, "lambda": lam,
                    "n_validation_HC": n_hc, "n_validation_disease": n_dis,
                    "fold_weight": weight, **mm,
                })

            real_y_delta = Y_dis.cpu().numpy().reshape(-1) - Y_hc.cpu().numpy().reshape(-1)
            pred_y_delta = fold_y[1.0] - fold_y[0.0]
            real_a_delta = A_dis.cpu().numpy().reshape(-1) - A_hc.cpu().numpy().reshape(-1)
            pred_a_delta = fold_a[1.0] - fold_a[0.0]
            fold_pattern_rows.append({
                "fold": fold_id,
                "disease": d,
                "n_val_HC": n_hc,
                "n_val_disease": n_dis,
                "fold_weight": weight,
                "brain_pearson_r": pearson_np(real_y_delta, pred_y_delta),
                "brain_cosine": cosine_np(real_y_delta, pred_y_delta),
                "brain_direction_consistency": float(
                    np.mean(np.sign(real_y_delta) == np.sign(pred_y_delta))
                ),
                "microbiome_pearson_r": pearson_np(real_a_delta, pred_a_delta),
                "microbiome_cosine": cosine_np(real_a_delta, pred_a_delta),
            })
            save_matrix_csv(
                np.vstack([real_y_delta, pred_y_delta]),
                ["validation_observed_disease_minus_HC", "validation_centroid_pred_lambda1_minus_lambda0"],
                roi_names,
                os.path.join(fold_pattern_dir, f"fold{fold_id:02d}_{d}_brain_pattern.csv"),
                "pattern"
            )
            save_matrix_csv(
                np.vstack([real_a_delta, pred_a_delta]),
                ["validation_observed_disease_minus_HC", "validation_centroid_pred_lambda1_minus_lambda0"],
                taxa_names,
                os.path.join(fold_pattern_dir, f"fold{fold_id:02d}_{d}_microbiome_pattern.csv"),
                "pattern"
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fold_pattern_df = pd.DataFrame(fold_pattern_rows)
    fold_pattern_df.to_csv(
        os.path.join(pred_dir, "validation_centroid_foldwise_spatial_correlations.csv"),
        index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(fold_metric_m2b_rows).to_csv(
        os.path.join(pred_dir, "validation_centroid_micro_to_brain_metrics_by_fold.csv"),
        index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(fold_metric_b2m_rows).to_csv(
        os.path.join(pred_dir, "validation_centroid_brain_to_micro_metrics_by_fold.csv"),
        index=False, encoding="utf-8-sig"
    )

    mean_pred_Y = {d: {} for d in diseases}
    mean_pred_A = {d: {} for d in diseases}
    metrics_m2b, metrics_b2m = [], []

    for d in diseases:
        for lam in lambdas:
            if not y_pred_store[d][lam]:
                continue
            weights = weights_store[d][lam]
            meanY = _weighted_vector_mean(y_pred_store[d][lam], weights)
            meanA = require_distribution_np(
                _weighted_vector_mean(a_pred_store[d][lam], weights).reshape(1, -1),
                eps=CONFIG["EPS"]
            )[0]
            mean_pred_Y[d][lam] = meanY
            mean_pred_A[d][lam] = meanA

            fold_y_metrics = [
                brain_metrics(y_pred_store[d][lam][i], y_target_store[d][lam][i])
                for i in range(len(weights))
            ]
            fold_a_metrics = [
                micro_metrics(a_pred_store[d][lam][i], a_target_store[d][lam][i])
                for i in range(len(weights))
            ]
            metrics_m2b.append({
                "direction": "micro_to_brain",
                "disease": d,
                "lambda": lam,
                "n_valid_folds": int(len(weights)),
                "sum_fold_weight": float(np.sum(weights)),
                "mae": _weighted_scalar_mean([m["mae"] for m in fold_y_metrics], weights),
                "rmse": _weighted_scalar_mean([m["rmse"] for m in fold_y_metrics], weights),
                "pearson_r": _weighted_scalar_mean([m["pearson_r"] for m in fold_y_metrics], weights),
                "cosine": _weighted_scalar_mean([m["cosine"] for m in fold_y_metrics], weights),
            })
            metrics_b2m.append({
                "direction": "brain_to_micro",
                "disease": d,
                "lambda": lam,
                "n_valid_folds": int(len(weights)),
                "sum_fold_weight": float(np.sum(weights)),
                "bray_curtis": _weighted_scalar_mean([m["bray_curtis"] for m in fold_a_metrics], weights),
                "js_divergence": _weighted_scalar_mean([m["js_divergence"] for m in fold_a_metrics], weights),
                "aitchison": _weighted_scalar_mean([m["aitchison"] for m in fold_a_metrics], weights),
                "pearson_r": _weighted_scalar_mean([m["pearson_r"] for m in fold_a_metrics], weights),
                "spearman_r": _weighted_scalar_mean([m["spearman_r"] for m in fold_a_metrics], weights),
            })

            summary_meta = [{
                "aggregation": "effective_sample_size_weighted_fold_mean",
                "disease": d,
                "lambda": lam,
                "n_valid_folds": len(weights),
                "sum_fold_weight": float(np.sum(weights)),
            }]
            _save_oof_matrix_with_metadata(
                meanY.reshape(1, -1), summary_meta, roi_names,
                os.path.join(pred_dir, f"WEIGHTED_MEAN_micro_to_brain_Ypred_HC_to_{d}_lambda_{lam:.2f}.csv")
            )
            _save_oof_matrix_with_metadata(
                meanA.reshape(1, -1), summary_meta, taxa_names,
                os.path.join(pred_dir, f"WEIGHTED_MEAN_brain_to_micro_Apred_HC_to_{d}_lambda_{lam:.2f}.csv")
            )

    metrics_m2b_df = pd.DataFrame(metrics_m2b)
    metrics_b2m_df = pd.DataFrame(metrics_b2m)
    metrics_m2b_df.to_csv(
        os.path.join(pred_dir, "validation_centroid_micro_to_brain_lambda_metrics.csv"),
        index=False, encoding="utf-8-sig"
    )
    metrics_b2m_df.to_csv(
        os.path.join(pred_dir, "validation_centroid_brain_to_micro_lambda_metrics.csv"),
        index=False, encoding="utf-8-sig"
    )
    if not metrics_m2b_df.empty:
        plot_lambda_curve(
            metrics_m2b_df,
            os.path.join(pred_dir, "validation_centroid_micro_to_brain_lambda_MAE_curve.png"),
            "Held-out validation-centroid microbiome shift -> brain prediction",
            "mae", "Weighted foldwise MAE to validation disease brain center"
        )
        plot_error_improvement(
            metrics_m2b_df,
            os.path.join(pred_dir, "validation_centroid_micro_to_brain_before_after_error.png"),
            "Validation-centroid microbiome shift -> brain: before/after error", "mae"
        )
    if not metrics_b2m_df.empty:
        plot_lambda_curve(
            metrics_b2m_df,
            os.path.join(pred_dir, "validation_centroid_brain_to_micro_lambda_JSD_curve.png"),
            "Held-out validation-centroid brain shift -> microbiome prediction",
            "js_divergence", "Weighted foldwise JSD to validation disease microbiome center"
        )
        plot_error_improvement(
            metrics_b2m_df,
            os.path.join(pred_dir, "validation_centroid_brain_to_micro_before_after_error.png"),
            "Validation-centroid brain shift -> microbiome: before/after error", "js_divergence"
        )

    # Pooled observed centers: group-size weighted validation-fold means. Because
    # each subject belongs to one validation fold, these equal the full-cohort raw means.
    pooled_hc_Y = _pooled_observed_center(fold_contexts, ctrl, "Y")
    pooled_hc_A = _pooled_observed_center(fold_contexts, ctrl, "A")
    pooled_pattern_rows = []
    pooled_centers_Y = {ctrl: pooled_hc_Y}
    pooled_centers_A = {ctrl: pooled_hc_A}

    for d in diseases:
        if 0.0 not in mean_pred_Y[d] or 1.0 not in mean_pred_Y[d]:
            continue
        pooled_d_Y = _pooled_observed_center(fold_contexts, d, "Y")
        pooled_d_A = _pooled_observed_center(fold_contexts, d, "A")
        pooled_centers_Y[d] = pooled_d_Y
        pooled_centers_A[d] = pooled_d_A
        real_brain_delta = pooled_d_Y - pooled_hc_Y
        pred_brain_delta = mean_pred_Y[d][1.0] - mean_pred_Y[d][0.0]
        pd.DataFrame({
            "ROI": roi_names,
            "observed_disease_minus_HC": real_brain_delta,
            "validation_centroid_weighted_pred_lambda1_minus_lambda0": pred_brain_delta,
        }).to_csv(
            os.path.join(pred_dir, f"validation_centroid_real_vs_pred_brain_pattern_{d}_data.csv"),
            index=False, encoding="utf-8-sig"
        )
        plot_pattern_scatter(
            real_brain_delta, pred_brain_delta, roi_names,
            os.path.join(pred_dir, f"validation_centroid_real_vs_pred_brain_pattern_{d}.png"),
            f"Observed pattern vs held-out centroid transition: HC->{d}",
            "Pooled observed disease-HC ROI difference",
            "Weighted fold-mean predicted lambda1-lambda0 ROI difference",
            label_top=10,
        )
        pooled_pattern_rows.append({
            "disease": d,
            "n_valid_folds": len(weights_store[d][0.0]),
            "brain_pearson_r": pearson_np(real_brain_delta, pred_brain_delta),
            "brain_cosine": cosine_np(real_brain_delta, pred_brain_delta),
            "brain_direction_consistency": float(
                np.mean(np.sign(real_brain_delta) == np.sign(pred_brain_delta))
            ),
        })

        real_micro_delta = pooled_d_A - pooled_hc_A
        pred_micro_delta = mean_pred_A[d][1.0] - mean_pred_A[d][0.0]
        pd.DataFrame({
            "Taxa": taxa_names,
            "observed_disease_minus_HC": real_micro_delta,
            "validation_centroid_weighted_pred_lambda1_minus_lambda0": pred_micro_delta,
        }).to_csv(
            os.path.join(pred_dir, f"validation_centroid_real_vs_pred_microbiome_pattern_{d}_data.csv"),
            index=False, encoding="utf-8-sig"
        )
        plot_pattern_scatter(
            real_micro_delta, pred_micro_delta, taxa_names,
            os.path.join(pred_dir, f"validation_centroid_real_vs_pred_microbiome_pattern_{d}.png"),
            f"Observed microbiome pattern vs held-out centroid transition: HC->{d}",
            "Pooled observed disease-HC taxon difference",
            "Weighted fold-mean predicted lambda1-lambda0 taxon difference",
            label_top=10,
        )

    pd.DataFrame(pooled_pattern_rows).to_csv(
        os.path.join(pred_dir, "validation_centroid_pooled_spatial_pattern_summary.csv"),
        index=False, encoding="utf-8-sig"
    )

    # PCA trajectories for both directions.
    rows, X = [], []
    for g, vec in pooled_centers_Y.items():
        X.append(vec); rows.append({"kind": "real_group_center", "disease": g, "lambda": np.nan})
    for d in diseases:
        for lam in lambdas:
            if lam in mean_pred_Y[d]:
                X.append(mean_pred_Y[d][lam]); rows.append({"kind": "predicted_path", "disease": d, "lambda": lam})
    if len(X) >= 2:
        coords = pca_2d_np(np.vstack(X))
        coords_df = pd.DataFrame(rows)
        coords_df["PC1"] = coords[:, 0]; coords_df["PC2"] = coords[:, 1]
        coords_df.to_csv(
            os.path.join(pred_dir, "validation_centroid_micro_to_brain_prediction_trajectory_pca.csv"),
            index=False, encoding="utf-8-sig"
        )
        plot_trajectory(
            coords_df,
            os.path.join(pred_dir, "validation_centroid_micro_to_brain_prediction_trajectory_pca.png"),
            "Held-out validation-centroid predicted brain trajectories"
        )

    rows, X = [], []
    for g, vec in pooled_centers_A.items():
        X.append(vec); rows.append({"kind": "real_group_center", "disease": g, "lambda": np.nan})
    for d in diseases:
        for lam in lambdas:
            if lam in mean_pred_A[d]:
                X.append(mean_pred_A[d][lam]); rows.append({"kind": "predicted_path", "disease": d, "lambda": lam})
    if len(X) >= 2:
        coords = pca_2d_np(np.vstack(X))
        coords_df = pd.DataFrame(rows)
        coords_df["PC1"] = coords[:, 0]; coords_df["PC2"] = coords[:, 1]
        coords_df.to_csv(
            os.path.join(pred_dir, "validation_centroid_brain_to_micro_prediction_trajectory_pca.csv"),
            index=False, encoding="utf-8-sig"
        )
        plot_trajectory(
            coords_df,
            os.path.join(pred_dir, "validation_centroid_brain_to_micro_prediction_trajectory_pca.png"),
            "Held-out validation-centroid predicted microbiome trajectories"
        )

    save_oof_disease_direction_projection_outputs(
        y_pred_store, y_source_store, y_target_store,
        fold_ids_store, weights_store, pred_dir
    )
    return mean_pred_Y, mean_pred_A, metrics_m2b_df, metrics_b2m_df


def plot_oof_disease_direction_projection_single(df: pd.DataFrame, out_png: str, disease: str):
    sub = df.sort_values("lambda").copy()
    sub.to_csv(out_png.replace(".png", ".csv"), index=False, encoding="utf-8-sig")
    fig, ax1 = plt.subplots(figsize=(7.2, 5.0))
    ax1.plot(sub["lambda"], sub["projection_normalized_mean"], marker="o", linewidth=2.0, label="Projection")
    lo = sub["projection_normalized_mean"] - sub["projection_normalized_std"]
    hi = sub["projection_normalized_mean"] + sub["projection_normalized_std"]
    ax1.fill_between(sub["lambda"].values, lo.values, hi.values, alpha=0.18, linewidth=0)
    ax1.axhline(0.0, linewidth=0.9, linestyle="--")
    ax1.axhline(1.0, linewidth=0.9, linestyle=":")
    ax1.set_xlabel("Lambda: validation HC-centroid to disease-centroid interpolation")
    ax1.set_ylabel("Weighted mean normalized projection")
    ax1.set_title(f"Held-out centroid disease-direction projection: HC->{disease}")
    ax2 = ax1.twinx()
    ax2.plot(sub["lambda"], sub["orthogonal_residual_norm_mean"], marker="s", linewidth=1.5, alpha=0.55, label="Orthogonal residual")
    ax2.set_ylabel("Weighted mean orthogonal residual norm")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=ANALYSIS_CONFIG["FIG_DPI"], bbox_inches="tight")
    plt.close(fig)


def save_oof_disease_direction_projection_outputs(
    y_pred_store: Dict[str, Dict[float, List[np.ndarray]]],
    y_source_store: Dict[str, Dict[float, List[np.ndarray]]],
    y_target_store: Dict[str, Dict[float, List[np.ndarray]]],
    fold_ids_store: Dict[str, Dict[float, List[int]]],
    weights_store: Dict[str, Dict[float, List[float]]],
    out_dir: str,
):
    """Save fold-centroid disease-direction projections and weighted summaries."""
    proj_dir = os.path.join(out_dir, "validation_centroid_disease_direction_projection")
    ensure_dir(proj_dir)
    all_summary_rows, all_fold_rows = [], []
    for disease, pred_by_lambda in y_pred_store.items():
        lambdas = sorted(float(x) for x in pred_by_lambda if len(pred_by_lambda[x]) > 0)
        if not lambdas:
            continue
        base_lam = 0.0 if 0.0 in lambdas else lambdas[0]
        pred0 = np.stack(y_pred_store[disease][base_lam], axis=0)
        src0 = np.stack(y_source_store[disease][base_lam], axis=0)
        tgt0 = np.stack(y_target_store[disease][base_lam], axis=0)
        v = tgt0 - src0
        v_norm = np.linalg.norm(v, axis=1)
        unit_v = v / np.maximum(v_norm[:, None], 1e-12)
        fold_ids = fold_ids_store[disease][base_lam]
        weights = np.asarray(weights_store[disease][base_lam], dtype=float)
        weights = weights / weights.sum()
        summary_rows, fold_rows = [], []

        for lam in lambdas:
            pred = np.stack(y_pred_store[disease][lam], axis=0)
            delta = pred - pred0
            projection = np.sum(delta * unit_v, axis=1)
            projection_normalized = projection / np.maximum(v_norm, 1e-12)
            orth = delta - projection[:, None] * unit_v
            orth_norm = np.linalg.norm(orth, axis=1)
            delta_norm = np.linalg.norm(delta, axis=1)
            cos_dir = projection / np.maximum(delta_norm, 1e-12)
            for i, fold_id in enumerate(fold_ids):
                row = {
                    "disease": disease,
                    "fold": int(fold_id),
                    "lambda": float(lam),
                    "fold_weight": float(weights_store[disease][lam][i]),
                    "projection": float(projection[i]),
                    "projection_normalized": float(projection_normalized[i]),
                    "cosine_to_validation_centroid_disease_direction": float(cos_dir[i]),
                    "orthogonal_residual_norm": float(orth_norm[i]),
                    "delta_pred_norm": float(delta_norm[i]),
                    "disease_axis_norm": float(v_norm[i]),
                }
                fold_rows.append(row); all_fold_rows.append(row)
            p_mean = float(np.average(projection_normalized, weights=weights))
            p_std = float(np.sqrt(np.average((projection_normalized - p_mean) ** 2, weights=weights)))
            summary = {
                "disease": disease,
                "lambda": float(lam),
                "n_valid_folds": int(len(fold_ids)),
                "projection_normalized_mean": p_mean,
                "projection_normalized_std": p_std,
                "projection_mean": float(np.average(projection, weights=weights)),
                "orthogonal_residual_norm_mean": float(np.average(orth_norm, weights=weights)),
                "delta_pred_norm_mean": float(np.average(delta_norm, weights=weights)),
                "disease_axis_norm_mean": float(np.average(v_norm, weights=weights)),
                "cosine_to_validation_centroid_disease_direction_mean": float(np.average(cos_dir, weights=weights)),
            }
            summary_rows.append(summary); all_summary_rows.append(summary)

        summary_df = pd.DataFrame(summary_rows)
        pd.DataFrame(fold_rows).to_csv(
            os.path.join(proj_dir, f"HC_to_{disease}_validation_centroid_projection_by_fold.csv"),
            index=False, encoding="utf-8-sig"
        )
        summary_df.to_csv(
            os.path.join(proj_dir, f"HC_to_{disease}_validation_centroid_projection.csv"),
            index=False, encoding="utf-8-sig"
        )
        plot_oof_disease_direction_projection_single(
            summary_df,
            os.path.join(proj_dir, f"HC_to_{disease}_validation_centroid_projection.png"),
            disease
        )

    pd.DataFrame(all_summary_rows).to_csv(
        os.path.join(proj_dir, "validation_centroid_projection_all.csv"),
        index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(all_fold_rows).to_csv(
        os.path.join(proj_dir, "validation_centroid_projection_by_fold_all.csv"),
        index=False, encoding="utf-8-sig"
    )


# ============================================================
# Path-integrated attribution
# ============================================================
def integrated_grad_micro_to_brain_single_model(
    model: torch.nn.Module,
    M_src: torch.Tensor,
    A_src: torch.Tensor,
    A_tgt: torch.Tensor,
    device: torch.device,
    pi_steps: int,
    batch_size: int,
) -> np.ndarray:
    """Return Taxa x ROI subject-averaged contribution for HC_i -> centroid microbiome shifts."""
    model.eval()
    lambdas = torch.linspace(0.0, 1.0, steps=pi_steps).tolist()
    contrib_sum = np.zeros((N_TAXA, N_ROIS), dtype=np.float64)
    n_units = 0
    for lam in lambdas:
        for i in range(0, M_src.size(0), batch_size):
            M_b = M_src[i:i + batch_size].to(device)
            A0 = A_src[i:i + batch_size]
            AT = A_tgt[i:i + batch_size]
            A_lam = shift_micro_subjectwise(A0, AT, lam)
            A_b = A_lam.to(device).clone().detach().requires_grad_(True)
            delta_A_b = (AT - A0).detach().cpu().numpy()
            y_hat, _, *_ = model(B_true=None, M_true=M_b, A_true=A_b, Y_true=None, mode="micro2brain")
            for r in range(N_ROIS):
                grad = torch.autograd.grad(y_hat[:, r].sum(), A_b, retain_graph=True, create_graph=False)[0]
                contrib_sum[:, r] += (delta_A_b * grad.detach().cpu().numpy()).mean(axis=0)
            n_units += 1
    return (contrib_sum / max(n_units, 1)).astype(np.float32)


def integrated_grad_brain_to_micro_single_model(
    model: torch.nn.Module,
    B_src: torch.Tensor,
    Y_src: torch.Tensor,
    Y_tgt: torch.Tensor,
    device: torch.device,
    pi_steps: int,
    batch_size: int,
) -> np.ndarray:
    """Return ROI x Taxa subject-averaged contribution for HC_i -> centroid brain shifts."""
    model.eval()
    target_taxa = ANALYSIS_CONFIG["BRAIN2MICRO_TARGET_TAXA_INDICES"]
    if target_taxa is None:
        target_taxa = list(range(N_TAXA))
    else:
        target_taxa = [int(x) for x in target_taxa]
    lambdas = torch.linspace(0.0, 1.0, steps=pi_steps).tolist()
    contrib_sum = np.zeros((N_ROIS, N_TAXA), dtype=np.float64)
    n_units = 0
    for lam in lambdas:
        for i in range(0, B_src.size(0), batch_size):
            B_b = B_src[i:i + batch_size].to(device)
            Y0 = Y_src[i:i + batch_size]
            YT = Y_tgt[i:i + batch_size]
            Y_lam = shift_brain_subjectwise(Y0, YT, lam)
            Y_b = Y_lam.to(device).clone().detach().requires_grad_(True)
            delta_Y_b = (YT - Y0).detach().cpu().numpy()
            _, a_hat, *_ = model(B_true=B_b, M_true=None, A_true=None, Y_true=Y_b, mode="brain2micro")
            for j in target_taxa:
                grad = torch.autograd.grad(a_hat[:, j].sum(), Y_b, retain_graph=True, create_graph=False)[0]
                contrib_sum[:, j] += (delta_Y_b * grad.detach().cpu().numpy()).mean(axis=0)
            n_units += 1
    return (contrib_sum / max(n_units, 1)).astype(np.float32)


def save_contribution_outputs(mat: np.ndarray, row_names: List[str], col_names: List[str], out_dir: str, prefix: str, row_label: str, col_label: str):
    ensure_dir(out_dir)
    save_matrix_csv(mat, row_names, col_names, os.path.join(out_dir, f"{prefix}_signed_contribution.csv"), row_label)
    save_matrix_csv(np.abs(mat), row_names, col_names, os.path.join(out_dir, f"{prefix}_abs_contribution.csv"), row_label)
    plot_heatmap_top(mat, row_names, col_names, os.path.join(out_dir, f"{prefix}_path_integrated_heatmap.png"),
                     f"{prefix}: path-integrated signed contribution", row_label, col_label)
    # Top bars
    row_scores = np.abs(mat).sum(axis=1)
    col_scores = np.abs(mat).sum(axis=0)
    plot_top_bar(row_scores, row_names, os.path.join(out_dir, f"{prefix}_top_{row_label}.png"),
                 f"{prefix}: top contributing {row_label}", "Sum of absolute contribution", row_label, int(ANALYSIS_CONFIG["TOPK_TAXA"] if row_label == "Taxa" else ANALYSIS_CONFIG["TOPK_ROI"]))
    plot_top_bar(col_scores, col_names, os.path.join(out_dir, f"{prefix}_top_{col_label}.png"),
                 f"{prefix}: top affected {col_label}", "Sum of absolute contribution", col_label, int(ANALYSIS_CONFIG["TOPK_ROI"] if col_label == "ROI" else ANALYSIS_CONFIG["TOPK_TAXA"]))
    # Top edges chord
    edges = build_top_edges(mat, row_names, col_names, int(ANALYSIS_CONFIG["TOPK_EDGES"]), row_label, col_label)
    edges.to_csv(os.path.join(out_dir, f"{prefix}_top_edges.csv"), index=False, encoding="utf-8-sig")
    plot_simple_chord(edges, row_label, col_label, os.path.join(out_dir, f"{prefix}_top_edges_chord.png"),
                      f"{prefix}: top {ANALYSIS_CONFIG['TOPK_EDGES']} {row_label}->{col_label} links")
    # Waterfall: choose target with largest total absolute contribution
    target_idx = int(np.argmax(np.abs(mat).sum(axis=0)))
    plot_waterfall(mat[:, target_idx], row_names, col_names[target_idx],
                   os.path.join(out_dir, f"{prefix}_waterfall_for_top_{col_label}.png"),
                   f"{prefix}: waterfall contributions", int(ANALYSIS_CONFIG["TOPK_WATERFALL"]))


def run_path_integrated_attribution(
    fold_contexts,
    B_all: torch.Tensor,
    M_all: torch.Tensor,
    A_all: torch.Tensor,
    Y_all: torch.Tensor,
    sids: List[str],
    groups: List[str],
    out_dir: str,
    device: torch.device,
    roi_names: List[str],
    taxa_names: List[str],
):
    """
    Path-integrated attribution along held-out validation centroid paths.

    In fold k, the path is exactly validation HC centroid -> validation disease
    centroid, and only model k is explained. Fold matrices are combined with the
    same effective two-group sample-size weights used for prediction aggregation.
    """
    attr_dir = os.path.join(out_dir, "validation_centroid_path_integrated_attribution")
    ensure_dir(attr_dir)
    fold_dir = os.path.join(attr_dir, "fold_matrices")
    ensure_dir(fold_dir)
    pi_steps = int(ANALYSIS_CONFIG["PI_STEPS"])
    batch_size = int(CONFIG["BATCH_SIZE"])
    ctrl = str(ANALYSIS_CONFIG["CONTROL_GROUP"]).upper()
    diseases = [str(d).upper() for d in ANALYSIS_CONFIG["DISEASE_GROUPS"]]
    results_m2b, results_b2m = {}, {}

    for d in diseases:
        fold_m2b, fold_b2m, weights, used_folds = [], [], [], []
        for ctx in fold_contexts:
            fold_id = int(ctx["fold_id"])
            centers = ctx["validation_centers"]
            if ctrl not in centers or d not in centers:
                continue
            hc = centers[ctrl]
            dis = centers[d]
            weight = _centroid_fold_weight(hc["n"], dis["n"])
            if weight <= 0:
                continue
            B_src = hc["B"]
            M_src = hc["M"]
            A_src = hc["A"]
            Y_src = hc["Y"]
            A_endpoint = dis["A"]
            Y_endpoint = dis["Y"]

            model = load_single_fold_model(device, fold_id)
            c_m2b = integrated_grad_micro_to_brain_single_model(
                model, M_src, A_src, A_endpoint, device, pi_steps, batch_size
            )
            c_b2m = integrated_grad_brain_to_micro_single_model(
                model, B_src, Y_src, Y_endpoint, device, pi_steps, batch_size
            )
            fold_m2b.append(c_m2b)
            fold_b2m.append(c_b2m)
            weights.append(weight)
            used_folds.append(fold_id)
            save_matrix_csv(
                c_m2b, taxa_names, roi_names,
                os.path.join(fold_dir, f"VAL_CENTROID_HC_to_{d}_micro_to_brain_fold{fold_id:02d}.csv"),
                "Taxa"
            )
            save_matrix_csv(
                c_b2m, roi_names, taxa_names,
                os.path.join(fold_dir, f"VAL_CENTROID_HC_to_{d}_brain_to_micro_fold{fold_id:02d}.csv"),
                "ROI"
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not fold_m2b:
            continue
        weights_np = np.asarray(weights, dtype=float)
        weights_np = weights_np / weights_np.sum()
        m2b_stack = np.stack(fold_m2b, axis=0)
        b2m_stack = np.stack(fold_b2m, axis=0)
        m2b_mean = np.tensordot(weights_np, m2b_stack, axes=(0, 0))
        b2m_mean = np.tensordot(weights_np, b2m_stack, axes=(0, 0))
        m2b_std = np.sqrt(np.tensordot(weights_np, (m2b_stack - m2b_mean) ** 2, axes=(0, 0)))
        b2m_std = np.sqrt(np.tensordot(weights_np, (b2m_stack - b2m_mean) ** 2, axes=(0, 0)))
        results_m2b[d] = m2b_mean
        results_b2m[d] = b2m_mean

        pd.DataFrame({
            "fold": used_folds,
            "raw_weight": weights,
            "normalized_weight": weights_np,
        }).to_csv(
            os.path.join(attr_dir, f"VAL_CENTROID_HC_to_{d}_fold_weights.csv"),
            index=False, encoding="utf-8-sig"
        )
        save_matrix_csv(
            m2b_std, taxa_names, roi_names,
            os.path.join(attr_dir, f"VAL_CENTROID_HC_to_{d}_micro_to_brain_contribution_fold_std.csv"),
            "Taxa"
        )
        save_matrix_csv(
            b2m_std, roi_names, taxa_names,
            os.path.join(attr_dir, f"VAL_CENTROID_HC_to_{d}_brain_to_micro_contribution_fold_std.csv"),
            "ROI"
        )
        save_contribution_outputs(
            m2b_mean, taxa_names, roi_names, attr_dir,
            f"VAL_CENTROID_HC_to_{d}_micro_to_brain", "Taxa", "ROI"
        )
        save_contribution_outputs(
            b2m_mean, roi_names, taxa_names, attr_dir,
            f"VAL_CENTROID_HC_to_{d}_brain_to_micro", "ROI", "Taxa"
        )

    return results_m2b, results_b2m


def edge_sets_from_contrib(results: Dict[str, np.ndarray], row_names: List[str], col_names: List[str], topk: int, row_label: str, col_label: str):
    sets: Dict[str, Set[str]] = {}
    edge_tables: Dict[str, pd.DataFrame] = {}
    for d, mat in results.items():
        df = build_top_edges(mat, row_names, col_names, topk, row_label, col_label)
        sets[d] = set(df["edge_id"].tolist())
        edge_tables[d] = df
    return sets, edge_tables


def plot_shared_specific_chords(results: Dict[str, np.ndarray], row_names: List[str], col_names: List[str], out_dir: str, prefix: str, row_label: str, col_label: str):
    """
    Build shared / pairwise-shared / disease-specific / opposite-direction link outputs.

    Important definition:
        - All edge sets are first selected as the top-K edges within each disease,
          ranked by absolute signed contribution.
        - shared_all: an edge appears in the top-K list of every disease.
        - shared_pairwise_exact: an edge appears in the top-K lists of exactly two diseases.
        - shared_pairwise_inclusive: an edge appears in the top-K lists of a disease pair;
          this also includes shared_all edges if they exist.
        - specific: an edge appears in the top-K list of only one disease.
        - opposite_direction: an edge in the union of top-K lists has opposite signs across diseases.

    These are ranking/overlap definitions, not statistical significance tests.
    """
    ensure_dir(out_dir)
    diseases = list(results.keys())
    topk = int(ANALYSIS_CONFIG["TOPK_SHARED_SPECIFIC_EDGES"])
    sets, edge_tables = edge_sets_from_contrib(results, row_names, col_names, topk, row_label, col_label)

    def _edge_value_row(eid: str, member_diseases: List[str], edge_type: str, pair: str = "") -> Dict[str, Any]:
        """Create one row for a shared edge, including signed values in all diseases."""
        src, tgt = eid.split("|||")
        r = row_names.index(src)
        c = col_names.index(tgt)
        vals_all = {d: float(results[d][r, c]) for d in diseases}
        vals_member = [vals_all[d] for d in member_diseases]
        row = {
            row_label: src,
            col_label: tgt,
            "edge_id": eid,
            "edge_type": edge_type,
            "shared_in": ";".join(member_diseases),
            "n_shared": int(len(member_diseases)),
            "pair": pair,
            "contribution_signed": float(np.mean(vals_member)) if vals_member else 0.0,
            "contribution_abs": float(np.mean(np.abs(vals_member))) if vals_member else 0.0,
        }
        row.update({f"signed_{d}": vals_all[d] for d in diseases})
        row.update({f"in_topk_{d}": bool(eid in sets[d]) for d in diseases})
        return row

    # ------------------------------------------------------------------
    # 1) Shared by all diseases: intersection of all top-K sets.
    # ------------------------------------------------------------------
    all_shared = set.intersection(*(sets[d] for d in diseases)) if diseases else set()
    shared_rows = [_edge_value_row(eid, diseases, "shared_all") for eid in sorted(all_shared)]
    shared_df = pd.DataFrame(shared_rows)
    shared_df = shared_df.sort_values("contribution_abs", ascending=False) if not shared_df.empty else shared_df
    shared_df.to_csv(os.path.join(out_dir, f"{prefix}_shared_all_edges.csv"), index=False, encoding="utf-8-sig")
    if not shared_df.empty:
        plot_simple_chord(
            shared_df.head(topk), row_label, col_label,
            os.path.join(out_dir, f"{prefix}_shared_all_chord.png"),
            f"{prefix}: shared links across all diseases"
        )

    # ------------------------------------------------------------------
    # 2) Pairwise shared links.
    #    inclusive: pair intersection, including links also shared by all.
    #    exact: pair intersection excluding links shared by the remaining disease(s).
    # ------------------------------------------------------------------
    pairwise_inclusive_rows = []
    pairwise_exact_rows = []
    pairwise_summary_rows = []

    for i in range(len(diseases)):
        for j in range(i + 1, len(diseases)):
            d1, d2 = diseases[i], diseases[j]
            pair_name = f"{d1}_{d2}"
            pair_edges_inclusive = sets[d1] & sets[d2]
            other_diseases = [d for d in diseases if d not in {d1, d2}]
            other_union = set.union(*(sets[o] for o in other_diseases)) if other_diseases else set()
            pair_edges_exact = pair_edges_inclusive - other_union

            for eid in sorted(pair_edges_inclusive):
                pairwise_inclusive_rows.append(_edge_value_row(eid, [d1, d2], "shared_pairwise_inclusive", pair=pair_name))
            for eid in sorted(pair_edges_exact):
                pairwise_exact_rows.append(_edge_value_row(eid, [d1, d2], "shared_pairwise_exact", pair=pair_name))

            pair_inc_df = pd.DataFrame([_edge_value_row(eid, [d1, d2], "shared_pairwise_inclusive", pair=pair_name) for eid in sorted(pair_edges_inclusive)])
            pair_exact_df = pd.DataFrame([_edge_value_row(eid, [d1, d2], "shared_pairwise_exact", pair=pair_name) for eid in sorted(pair_edges_exact)])
            if not pair_inc_df.empty:
                pair_inc_df = pair_inc_df.sort_values("contribution_abs", ascending=False)
            if not pair_exact_df.empty:
                pair_exact_df = pair_exact_df.sort_values("contribution_abs", ascending=False)

            pair_inc_df.to_csv(os.path.join(out_dir, f"{prefix}_{pair_name}_shared_pairwise_inclusive_edges.csv"), index=False, encoding="utf-8-sig")
            pair_exact_df.to_csv(os.path.join(out_dir, f"{prefix}_{pair_name}_shared_pairwise_exact_edges.csv"), index=False, encoding="utf-8-sig")

            if not pair_exact_df.empty:
                plot_simple_chord(
                    pair_exact_df.head(topk), row_label, col_label,
                    os.path.join(out_dir, f"{prefix}_{pair_name}_shared_pairwise_exact_chord.png"),
                    f"{prefix}: {d1}-{d2} pairwise shared links"
                )
            elif not pair_inc_df.empty:
                # If the exact pairwise set is empty but the inclusive set is not,
                # still draw the inclusive set so pair-level sharing is visible.
                plot_simple_chord(
                    pair_inc_df.head(topk), row_label, col_label,
                    os.path.join(out_dir, f"{prefix}_{pair_name}_shared_pairwise_inclusive_chord.png"),
                    f"{prefix}: {d1}-{d2} pairwise shared links, inclusive"
                )

            pairwise_summary_rows.append({
                "pair": pair_name,
                "disease_1": d1,
                "disease_2": d2,
                "n_shared_pairwise_inclusive": int(len(pair_edges_inclusive)),
                "n_shared_pairwise_exact": int(len(pair_edges_exact)),
                "n_shared_all_overlap": int(len(pair_edges_inclusive & all_shared)),
            })

    pairwise_inclusive_df = pd.DataFrame(pairwise_inclusive_rows)
    pairwise_exact_df = pd.DataFrame(pairwise_exact_rows)
    if not pairwise_inclusive_df.empty:
        pairwise_inclusive_df = pairwise_inclusive_df.sort_values(["pair", "contribution_abs"], ascending=[True, False])
    if not pairwise_exact_df.empty:
        pairwise_exact_df = pairwise_exact_df.sort_values(["pair", "contribution_abs"], ascending=[True, False])
    pairwise_inclusive_df.to_csv(os.path.join(out_dir, f"{prefix}_shared_pairwise_inclusive_edges.csv"), index=False, encoding="utf-8-sig")
    pairwise_exact_df.to_csv(os.path.join(out_dir, f"{prefix}_shared_pairwise_exact_edges.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(pairwise_summary_rows).to_csv(os.path.join(out_dir, f"{prefix}_shared_pairwise_summary.csv"), index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # 3) Disease-specific links: top-K in one disease but not top-K in any other disease.
    # ------------------------------------------------------------------
    for d in diseases:
        other_union = set.union(*(sets[o] for o in diseases if o != d)) if len(diseases) > 1 else set()
        specific = sets[d] - other_union
        df = edge_tables[d][edge_tables[d]["edge_id"].isin(specific)].copy()
        df = df.sort_values("contribution_abs", ascending=False) if not df.empty else df
        df.to_csv(os.path.join(out_dir, f"{prefix}_{d}_specific_edges.csv"), index=False, encoding="utf-8-sig")
        if not df.empty:
            plot_simple_chord(
                df.head(topk), row_label, col_label,
                os.path.join(out_dir, f"{prefix}_{d}_specific_chord.png"),
                f"{prefix}: {d}-specific links"
            )

    # ------------------------------------------------------------------
    # 4) Opposite-direction matrix/table.
    # ------------------------------------------------------------------
    opp_rows = []
    union_edges = set.union(*(sets[d] for d in diseases)) if diseases else set()
    for eid in sorted(union_edges):
        src, tgt = eid.split("|||")
        r = row_names.index(src)
        c = col_names.index(tgt)
        vals = {d: float(results[d][r, c]) for d in diseases}
        signs = {d: np.sign(v) for d, v in vals.items() if abs(v) > 0}
        if len(set(signs.values())) > 1:
            row = {"edge_id": eid, row_label: src, col_label: tgt, "edge_type": "opposite_direction"}
            row.update({f"signed_{d}": vals[d] for d in diseases})
            row.update({f"in_topk_{d}": bool(eid in sets[d]) for d in diseases})
            opp_rows.append(row)
    opp_df = pd.DataFrame(opp_rows)
    opp_df.to_csv(os.path.join(out_dir, f"{prefix}_opposite_direction_edges.csv"), index=False, encoding="utf-8-sig")
    if not opp_df.empty:
        # Heatmap of signed contributions for opposite edges
        mat = opp_df[[f"signed_{d}" for d in diseases]].to_numpy()
        vmax = np.percentile(np.abs(mat), 98) if mat.size else 1.0
        vmax = max(float(vmax), 1e-12)
        plt.figure(figsize=(7, max(4, 0.25 * len(opp_df))))
        plt.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        plt.colorbar(label="Signed contribution")
        plt.xticks(np.arange(len(diseases)), diseases)
        plt.yticks(np.arange(len(opp_df)), opp_df["edge_id"], fontsize=5)
        plt.title(f"{prefix}: opposite-direction links")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_opposite_direction_heatmap.png"), dpi=ANALYSIS_CONFIG["FIG_DPI"], bbox_inches="tight")
        plt.close()


# ============================================================
# Main pipeline
# ============================================================
def run_individual_distribution_shift_pipeline(device: Optional[torch.device] = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(CONFIG["SEED"]))
    out_dir = get_analysis_out_dir()
    roi_names = get_roi_names()
    taxa_names = get_taxa_names()

    # Load the full aligned cohort. Do not globally filter before resolving folds.
    B_all, M_all, A_all, Y_all, sids, groups = load_oof_data_with_groups()
    pd.DataFrame({
        "aligned_index": np.arange(len(sids), dtype=int),
        "subject_id": sids,
        "group": groups,
    }).to_csv(os.path.join(out_dir, "strict_oof_aligned_subjects.csv"), index=False, encoding="utf-8-sig")

    # Exact training split is mandatory for a defensible strict OOF claim.
    folds = resolve_strict_oof_folds(sids, out_dir)

    # Build raw held-out validation group centroids separately in every fold.
    fold_contexts = prepare_strict_oof_fold_contexts(
        folds, B_all, M_all, A_all, Y_all, sids, groups,
        out_dir, roi_names, taxa_names
    )

    print("Running held-out validation HC-centroid to disease-centroid predictions...")
    run_shift_predictions(
        fold_contexts, B_all, M_all, A_all, Y_all, sids, groups,
        out_dir, device, roi_names, taxa_names
    )

    print("Running validation-centroid path-integrated attribution; this may be computationally expensive.")
    results_m2b, results_b2m = run_path_integrated_attribution(
        fold_contexts, B_all, M_all, A_all, Y_all, sids, groups,
        out_dir, device, roi_names, taxa_names
    )

    print("Building validation-centroid shared / disease-specific / opposite-direction outputs...")
    comp_dir = os.path.join(out_dir, "validation_centroid_shared_specific_opposite_links")
    ensure_dir(comp_dir)
    plot_shared_specific_chords(results_m2b, taxa_names, roi_names, comp_dir, "micro_to_brain", "Taxa", "ROI")
    plot_shared_specific_chords(results_b2m, roi_names, taxa_names, comp_dir, "brain_to_micro", "ROI", "Taxa")
    print(f"Done. Validation-centroid transition results saved to: {out_dir}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_individual_distribution_shift_pipeline(device=device)

