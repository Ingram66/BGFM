import os
import random
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bgfm.runtime import load_section, apply_globals, apply_mapping

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    # data
    'BRAIN_FEAT_PT': r'outputs/paired_brain_features/brain_classification_features.pt',
    'GUT_FEAT_PT': r'outputs/paired_gut_features/gut_classification_features.pt',
    'MICRO_ABUND_CSV': r'data/paired/microbiome_abundance.csv',
    'ROI_MEAN_CSV': r'data/paired/bold_roi_mean.csv',

    'OUT_DIR': r'outputs/alignment',
    'CKPT_DIR': r'outputs/alignment',

    'RUN_TRAIN_CV': False,
    'RUN_EXT_BRAIN2MICRO': True,
    'RUN_EXT_MICRO2BRAIN': True,

    'EXT_BRAIN_FEAT_PT': r'outputs/paired_brain_features/brain_classification_features.pt',
    'EXT_ROI_MEAN_CSV': r'data/paired/bold_roi_mean.csv',
    'EXT_GUT_FEAT_PT': r'outputs/paired_gut_features/gut_classification_features.pt',
    'EXT_MICRO_ABUND_CSV': r'data/paired/microbiome_abundance.csv',

    'EXT_BRAIN2MICRO_TAXA_PRED_CSV': 'EXT_brain2micro_taxa_pred.csv',
    'EXT_MICRO2BRAIN_BOLD_PRED_CSV': 'EXT_micro2brain_bold_pred.csv',
    'EXT_BRAIN2MICRO_METRICS_TXT': 'EXT_brain2micro_metrics.txt',
    'EXT_MICRO2BRAIN_METRICS_TXT': 'EXT_micro2brain_metrics.txt',

    'SEED': 1307,
    'N_FOLDS': 10,
    'EPOCHS': 500,
    'BATCH_SIZE': 16,
    'ACCUM_STEPS': 2,
    'LR': 3e-4,
    'WEIGHT_DECAY': 1e-4,

    'D_ALIGN': 256,
    'DROPOUT': 0.1,
    'N_HEADS': 4,
    'N_COND_TOKENS': 8,
    'ROI_HEAD_HIDDEN': 256,
    'TAXA_HEAD_HIDDEN': 128,

    # task losses
    'LAMBDA_Y': 4.0,
    'LAMBDA_YCORR': 0.80,
    'LAMBDA_AB': 0.5,
    'LAMBDA_ABCORR': 0.20,
    'LAMBDA_AB_CLR_MSE': 0.60,
    'LAMBDA_ALIGN_B': 0.10,
    'LAMBDA_ALIGN_M': 0.05,
    'LAMBDA_DIVERSITY_AB': 0.15,
    'LAMBDA_DIVERSITY_Y': 1.0,
    'Y_SIM_ALIGN_LAMBDA': 2.0,
    'DIVY_UPPER': 0.70,
    'DIVY_MARGIN': 0.02,
    'LAMBDA_Y_VAR': 0.10,
    'Y_VAR_GAMMA': 0.7,
    'ALIGN_SHARED_LAMBDA': 0.20,
    'ALIGN_TAU': 0.07,

    # new losses for individual specificity
    'LAMBDA_Y_DEV': 0.5,
    'LAMBDA_A_DEV': 0.3,
    'LAMBDA_Y_DIST': 0.15,
    'LAMBDA_A_DIST': 0.10,

    # single-modality training branches
    'SINGLE_MODAL_TRAIN_ENABLE': True,
    'SINGLE_MODAL_P_BRAIN2MICRO': 0.25,
    'SINGLE_MODAL_P_MICRO2BRAIN': 0.25,
    'SINGLE_MODAL_BRANCH_WEIGHT': 0.5,

    'USE_HUBER': True,
    'HUBER_DELTA': 1.0,
    'USE_LR_SCHED': True,
    'SCHED_FACTOR': 0.5,
    'SCHED_PATIENCE': 15,
    'MIN_LR': 1e-6,
    'USE_EARLY_STOP': True,
    'EARLY_STOP_PATIENCE': 60,

    # augmentation
    'AUG_ENABLE': True,
    'AUG_MIXUP_PROB': 0.0,
    'AUG_MIXUP_ALPHA': 0.2,
    'AUG_NOISE_PROB': 0.8,
    'AUG_BRAIN_NOISE_STD': 0.01,
    'AUG_GUT_NOISE_STD': 0.01,
    'AUG_NODEDROP_PROB': 0.5,
    'AUG_BRAIN_NODEDROP_P': 0.05,
    'AUG_GUT_NODEDROP_P': 0.02,

    'AB_TEMP_MIN': 0.7,
    'AB_TEMP_MAX': 1.3,
    'EPS': 1e-8,

    # outputs
    'MEAN_PRED_ROI_CSV': 'mean_pred_bold.csv',
    'MEAN_PRED_TAXA_CSV': 'mean_pred_taxa.csv',
    'OOF_TAXA_PRED_CSV': 'OOF_taxa_pred.csv',
    'OOF_TAXA_DIAG_CSV': 'OOF_taxa_pred_diagnostics.csv',
    'OOF_BOLD_PRED_CSV': 'OOF_bold_pred.csv',
    'OOF_BOLD_DIAG_CSV': 'OOF_bold_pred_diagnostics.csv',
    'OOF_EMBEDDING_CSV': 'embedding_oof.csv',
    'OOF_NODE_REPRESENTATION_NPZ': 'node_representations_raw_and_normed.npz',
    'OOF_ALIGNED_SAMPLES_CSV': 'aligned_samples.csv',
}

N_ROIS = 90
N_TAXA = 642

apply_mapping(CONFIG, load_section('bridge'))

# ============================================================
# utilities
# ============================================================
def set_seed(seed: int = 1307):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load_any(path: str):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def extract_subject_id_from_path(p: str) -> str:
    base = os.path.basename(p)
    sid = os.path.splitext(base)[0]
    return str(sid)


def clr_transform(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp_min(eps)
    lp = torch.log(p)
    return lp - lp.mean(dim=1, keepdim=True)


def pearsonr_per_row(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    vx = (x * x).mean(dim=1).clamp_min(eps)
    vy = (y * y).mean(dim=1).clamp_min(eps)
    r = (x * y).mean(dim=1) / torch.sqrt(vx * vy)
    r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    return r.clamp(-1.0, 1.0)


@torch.no_grad()
def pearsonr_per_col(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    x, y: [N, D]
    返回每一列的 Pearson r，shape=[D]
    """
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    vx = (x * x).mean(dim=0).clamp_min(eps)
    vy = (y * y).mean(dim=0).clamp_min(eps)
    r = (x * y).mean(dim=0) / torch.sqrt(vx * vy)
    r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    return r.clamp(-1.0, 1.0)

# ============================================================
# loading
# ============================================================
def load_brain_node_feat(pt_path: str) -> Tuple[torch.Tensor, Optional[List[str]], Dict[str, Any]]:
    obj = torch_load_any(pt_path)
    if not isinstance(obj, dict):
        raise ValueError(f'{pt_path} must be a dict saved by torch.save')
    if 'roi_emb' in obj:
        x = obj['roi_emb']
    elif 'emb' in obj:
        x = obj['emb']
    else:
        raise ValueError(f"{pt_path} has no key 'roi_emb' or 'emb'")
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    if x.dim() != 3 or x.size(1) != N_ROIS:
        raise ValueError(f'[BRAIN] expected (N,{N_ROIS},d), got {tuple(x.shape)}')
    sids = None
    if 'paths' in obj and isinstance(obj['paths'], list):
        sids = [extract_subject_id_from_path(p) for p in obj['paths']]
    elif 'sample_ids' in obj and isinstance(obj['sample_ids'], list):
        sids = [str(s) for s in obj['sample_ids']]
    return x.float(), sids, obj


def load_gut_node_feat(pt_path: str) -> Tuple[torch.Tensor, Optional[List[str]], Dict[str, Any]]:
    obj = torch_load_any(pt_path)
    if not isinstance(obj, dict):
        raise ValueError(f'{pt_path} must be a dict saved by torch.save')
    if 'taxa_emb' in obj:
        x = obj['taxa_emb']
    elif 'node_emb' in obj:
        x = obj['node_emb']
    elif 'emb' in obj:
        x = obj['emb']
    else:
        raise ValueError(f"{pt_path} has no key 'taxa_emb'/'node_emb'/'emb'")
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    if x.dim() != 3 or x.size(1) != N_TAXA:
        raise ValueError(f'[GUT] expected (N,{N_TAXA},d), got {tuple(x.shape)}')
    sids = None
    if 'sample_ids' in obj and isinstance(obj['sample_ids'], list):
        sids = [str(s) for s in obj['sample_ids']]
    elif 'paths' in obj and isinstance(obj['paths'], list):
        sids = [extract_subject_id_from_path(p) for p in obj['paths']]
    return x.float(), sids, obj


def _require_distribution(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.clamp_min(0.0)
    s = x.sum(dim=1, keepdim=True).clamp_min(eps)
    return x / s


def load_abundance_table(csv_path: str) -> Dict[str, torch.Tensor]:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 1 + N_TAXA:
        raise ValueError(f'[ABUND] cols={df.shape[1]} < 1+{N_TAXA}')
    sids = df.iloc[:, 0].astype(str).tolist()
    mat = df.iloc[:, 1:1 + N_TAXA].to_numpy(np.float32)
    x = torch.tensor(mat, dtype=torch.float32)
    x = _require_distribution(x, eps=CONFIG['EPS'])
    return {sids[i]: x[i] for i in range(len(sids))}


def load_roi_mean_table(csv_path: str) -> Dict[str, torch.Tensor]:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 1 + N_ROIS:
        raise ValueError(f'[ROI_MEAN] cols={df.shape[1]} < 1+{N_ROIS}')
    sids = df.iloc[:, 0].astype(str).tolist()
    mat = df.iloc[:, 1:1 + N_ROIS].to_numpy(np.float32)
    y = torch.tensor(mat, dtype=torch.float32)
    return {sids[i]: y[i] for i in range(len(sids))}


def intersect_and_stack_noW(brain, brain_sids, gut, gut_sids, abund_map, roi_map):
    if brain_sids is None or gut_sids is None:
        common = sorted(list(set(abund_map) & set(roi_map)))
        if len(common) == 0:
            raise ValueError('No common subject_ids among abundance/roi_mean.')
        N = min(brain.size(0), gut.size(0), len(common))
        use_sids = common[:N]
        B = brain[:N]
        M = gut[:N]
        A = torch.stack([abund_map[s] for s in use_sids], dim=0)
        Y = torch.stack([roi_map[s] for s in use_sids], dim=0)
        return B, M, A, Y, use_sids

    b_map = {brain_sids[i]: i for i in range(len(brain_sids))}
    m_map = {gut_sids[i]: i for i in range(len(gut_sids))}
    common = sorted(list(set(b_map) & set(m_map) & set(abund_map) & set(roi_map)))
    if len(common) == 0:
        raise ValueError('No common subject_ids across brain/gut/abundance/roi_mean. Check IDs.')
    B = torch.stack([brain[b_map[s]] for s in common], dim=0)
    M = torch.stack([gut[m_map[s]] for s in common], dim=0)
    A = torch.stack([abund_map[s] for s in common], dim=0)
    Y = torch.stack([roi_map[s] for s in common], dim=0)
    return B, M, A, Y, common


def intersect_brain_and_roi(brain, brain_sids, roi_map):
    if brain_sids is None:
        common = sorted(list(set(roi_map)))
        if len(common) == 0:
            raise ValueError('No subject_ids in ROI table.')
        N = min(brain.size(0), len(common))
        use_sids = common[:N]
        B = brain[:N]
        Y = torch.stack([roi_map[s] for s in use_sids], dim=0)
        return B, Y, use_sids
    b_map = {brain_sids[i]: i for i in range(len(brain_sids))}
    common = sorted(list(set(b_map) & set(roi_map)))
    if len(common) == 0:
        raise ValueError('No common subject_ids across brain and roi_mean. Check IDs.')
    B = torch.stack([brain[b_map[s]] for s in common], dim=0)
    Y = torch.stack([roi_map[s] for s in common], dim=0)
    return B, Y, common


def intersect_gut_and_abund(gut, gut_sids, abund_map):
    if gut_sids is None:
        common = sorted(list(set(abund_map)))
        if len(common) == 0:
            raise ValueError('No subject_ids in abundance table.')
        N = min(gut.size(0), len(common))
        use_sids = common[:N]
        M = gut[:N]
        A = torch.stack([abund_map[s] for s in use_sids], dim=0)
        return M, A, use_sids
    m_map = {gut_sids[i]: i for i in range(len(gut_sids))}
    common = sorted(list(set(m_map) & set(abund_map)))
    if len(common) == 0:
        raise ValueError('No common subject_ids across gut and abundance. Check IDs.')
    M = torch.stack([gut[m_map[s]] for s in common], dim=0)
    A = torch.stack([abund_map[s] for s in common], dim=0)
    return M, A, common

# ============================================================
# dataset
# ============================================================
class DualBridgeDatasetCV(Dataset):
    def __init__(self, B, M, A, Y, idx_global: np.ndarray):
        self.B = B
        self.M = M
        self.A = A
        self.Y = Y
        self.idx_global = torch.tensor(idx_global, dtype=torch.long)

    def __len__(self):
        return int(self.B.shape[0])

    def __getitem__(self, idx):
        return self.B[idx], self.M[idx], self.A[idx], self.Y[idx], self.idx_global[idx]


def collate_fn(batch):
    B = torch.stack([x[0] for x in batch], 0)
    M = torch.stack([x[1] for x in batch], 0)
    A = torch.stack([x[2] for x in batch], 0)
    Y = torch.stack([x[3] for x in batch], 0)
    IDX = torch.stack([x[4] for x in batch], 0)
    return B, M, A, Y, IDX

# ============================================================
# losses
# ============================================================
def huber_or_mse(pred, target, use_huber=True, delta=1.0):
    return F.smooth_l1_loss(pred, target, beta=delta) if use_huber else F.mse_loss(pred, target)


def dist_ce_loss(pred_dist, tgt_dist, eps: float = 1e-6):
    pred = pred_dist.clamp_min(eps)
    pred = pred / (pred.sum(dim=1, keepdim=True) + eps)
    tgt = tgt_dist.clamp_min(eps)
    tgt = tgt / (tgt.sum(dim=1, keepdim=True) + eps)
    return -(tgt * pred.log()).sum(dim=1).mean()


def abundance_corr_loss_clr(pred_dist, tgt_dist):
    pred_clr = clr_transform(pred_dist)
    tgt_clr = clr_transform(tgt_dist)
    r = pearsonr_per_row(pred_clr, tgt_clr)
    return 1.0 - r.mean()


def abundance_clr_mse_loss(pred_dist, tgt_dist):
    return F.mse_loss(clr_transform(pred_dist), clr_transform(tgt_dist))


def cosine_align_loss(x_n, y_n):
    return (1.0 - (x_n * y_n).sum(dim=-1)).mean()


def roi_corr_loss(pred, tgt):
    r = pearsonr_per_row(pred, tgt)
    return 1.0 - r.mean()


def batch_diversity_loss_clr(a_hat, eps: float = 1e-6):
    if a_hat.size(0) < 3:
        return torch.tensor(0.0, device=a_hat.device)
    x = clr_transform(a_hat, eps=eps)
    x = F.normalize(x, dim=1)
    sim = x @ x.t()
    b = sim.size(0)
    offdiag = (sim.sum() - sim.diag().sum()) / (b * (b - 1) + 1e-12)
    return offdiag


def batch_diversity_loss_bold(y_hat, eps: float = 1e-6):
    if y_hat.size(0) < 3:
        return torch.tensor(0.0, device=y_hat.device)
    x = y_hat - y_hat.mean(dim=1, keepdim=True)
    x = x / (x.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps))
    x = F.normalize(x, dim=1)
    sim = x @ x.t()
    b = sim.size(0)
    offdiag = (sim.sum() - sim.diag().sum()) / (b * (b - 1) + 1e-12)
    return offdiag


def batch_similarity_alignment_loss_bold(y_hat, y_true, eps: float = 1e-6):
    if y_hat.size(0) < 3:
        return torch.tensor(0.0, device=y_hat.device)

    def z_norm(v):
        v = v - v.mean(dim=1, keepdim=True)
        v = v / (v.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps))
        return F.normalize(v, dim=1)

    zh = z_norm(y_hat)
    zt = z_norm(y_true)
    Sh = zh @ zh.t()
    St = zt @ zt.t()
    b = Sh.size(0)
    mask = ~torch.eye(b, device=Sh.device, dtype=torch.bool)
    return F.mse_loss(Sh[mask], St[mask])


def bold_variance_floor_loss(y_hat, y_true, gamma: float = 0.7, eps: float = 1e-8):
    if y_hat.size(0) < 3:
        return torch.tensor(0.0, device=y_hat.device)
    var_true = y_true.var(dim=0, unbiased=False)
    var_pred = y_hat.var(dim=0, unbiased=False)
    target = (gamma * var_true).clamp_min(eps)
    return F.relu(target - var_pred).mean()


def divy_target_loss(div_y_pred):
    return F.relu(div_y_pred - float(CONFIG['DIVY_UPPER']))


def info_nce_loss(z1, z2, temperature: float = 0.07, eps: float = 1e-8):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    tau = max(float(temperature), float(eps))
    logits = (z1 @ z2.t()) / tau
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def individual_deviation_loss(pred, target):
    pred_dev = pred - pred.mean(dim=0, keepdim=True)
    target_dev = target - target.mean(dim=0, keepdim=True)
    return F.mse_loss(pred_dev, target_dev)


def pairwise_distance_matrix(x, eps: float = 1e-8):
    x2 = (x * x).sum(dim=1, keepdim=True)
    dist2 = x2 + x2.t() - 2.0 * (x @ x.t())
    dist2 = dist2.clamp_min(0.0)
    return torch.sqrt(dist2 + eps)


def individual_distance_preserve_loss(pred, target):
    if pred.size(0) < 3:
        return torch.tensor(0.0, device=pred.device)
    Dp = pairwise_distance_matrix(pred)
    Dt = pairwise_distance_matrix(target)
    mask = ~torch.eye(Dp.size(0), dtype=torch.bool, device=Dp.device)
    return F.mse_loss(Dp[mask], Dt[mask])


def abundance_individual_deviation_loss(a_pred, a_true):
    return individual_deviation_loss(clr_transform(a_pred), clr_transform(a_true))


def abundance_individual_distance_loss(a_pred, a_true):
    return individual_distance_preserve_loss(clr_transform(a_pred), clr_transform(a_true))

# ============================================================
# augmentations
# ============================================================
def _beta_sample(alpha: float, device: torch.device, n: int):
    if alpha <= 0:
        lam = torch.ones(n, device=device)
    else:
        dist = torch.distributions.Beta(alpha, alpha)
        lam = dist.sample((n,)).to(device)
    return torch.maximum(lam, 1.0 - lam)


def vanilla_mixup(B, M, A, Y, alpha: float = 0.2, eps: float = 1e-8):
    device = B.device
    n = B.size(0)
    if n < 2:
        return B, M, A, Y
    perm = torch.randperm(n, device=device)
    lam = _beta_sample(alpha, device, n)
    lam_b = lam.view(-1, 1, 1)
    lam_v = lam.view(-1, 1)
    B2 = lam_b * B + (1.0 - lam_b) * B[perm]
    M2 = lam_b * M + (1.0 - lam_b) * M[perm]
    A2 = lam_v * A + (1.0 - lam_v) * A[perm]
    Y2 = lam_v * Y + (1.0 - lam_v) * Y[perm]
    A2 = _require_distribution(A2, eps=eps)
    return B2, M2, A2, Y2


def add_noise(B, M, brain_std: float, gut_std: float):
    if brain_std > 0:
        B = B + torch.randn_like(B) * brain_std
    if gut_std > 0:
        M = M + torch.randn_like(M) * gut_std
    return B, M


def node_dropout(B, M, p_brain: float, p_gut: float):
    if p_brain > 0:
        keep = (torch.rand(B.size(0), B.size(1), device=B.device) > p_brain).float()
        B = B * keep.unsqueeze(-1)
    if p_gut > 0:
        keep = (torch.rand(M.size(0), M.size(1), device=M.device) > p_gut).float()
        M = M * keep.unsqueeze(-1)
    return B, M

# ============================================================
# model components
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

        # global base templates
        self.brain_query_tokens = nn.Parameter(torch.randn(N_ROIS, d_align) * 0.02)
        self.micro_query_tokens = nn.Parameter(torch.randn(N_TAXA, d_align) * 0.02)

        # conditional template initializers
        self.brain_init_from_micro = ConditionalTokenInit(d_align, N_ROIS, d_align, dropout=dropout)
        self.micro_init_from_brain = ConditionalTokenInit(d_align, N_TAXA, d_align, dropout=dropout)

        # condition tokenizers
        self.abund_condition_tokenizer = ConditionTokenizer(N_TAXA, CONFIG['N_COND_TOKENS'], d_align, dropout=dropout)
        self.roi_condition_tokenizer = ConditionTokenizer(N_ROIS, CONFIG['N_COND_TOKENS'], d_align, dropout=dropout)

        # condition cross-attention
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

        # 1) conditional template init / projection
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

        # 2) condition injection via cross-attention
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
        else:  # micro2brain
            if A_true is None:
                raise ValueError("mode='micro2brain' requires A_true.")
            A_cond = self.abund_condition_tokenizer(A_true)
            B_cond = self.brain_cond_attn(B_lat, A_cond)
            M_cond = M_lat

        # 3) bridge
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
# metrics and plots
# ============================================================
@torch.no_grad()
def compute_roi_metrics(y_pred, y_true):
    mae = (y_pred - y_true).abs().mean().item()
    r = pearsonr_per_row(y_pred, y_true)
    return mae, r.mean().item(), r.std(unbiased=False).item(), r


@torch.no_grad()
def compute_taxa_metrics(a_pred, a_true):
    mae = (a_pred - a_true).abs().mean().item()
    r = pearsonr_per_row(clr_transform(a_pred), clr_transform(a_true))
    return mae, r.mean().item(), r.std(unbiased=False).item(), r


def save_roi_scatter(y_true, y_pred, out_path: str, title_suffix: str = ''):
    yt = y_true.reshape(-1).cpu().numpy()
    yp = y_pred.reshape(-1).cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.scatter(yt, yp, s=8, alpha=0.5)
    vmin = min(yt.min(), yp.min())
    vmax = max(yt.max(), yp.max())
    plt.plot([vmin, vmax], [vmin, vmax], linewidth=1)
    plt.xlabel('True ROI BOLD')
    plt.ylabel('Pred ROI BOLD')
    plt.title(f'ROI True vs Pred {title_suffix}'.strip())
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_taxa_scatter_clr(a_true, a_pred, out_path: str, title_suffix: str = ''):
    at = clr_transform(a_true).reshape(-1).cpu().numpy()
    ap = clr_transform(a_pred).reshape(-1).cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.scatter(at, ap, s=6, alpha=0.5)
    vmin = min(at.min(), ap.min())
    vmax = max(at.max(), ap.max())
    plt.plot([vmin, vmax], [vmin, vmax], linewidth=1)
    plt.xlabel('True CLR(abundance)')
    plt.ylabel('Pred CLR(abundance)')
    plt.title(f'Taxa True vs Pred (CLR) {title_suffix}'.strip())
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_r_hist(r_values, out_path: str, title: str, xlabel: str):
    rv = r_values.cpu().numpy()
    plt.figure(figsize=(6, 4))
    plt.hist(rv, bins=30, alpha=0.9)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_fold_metrics(fold_df: pd.DataFrame, out_path: str):
    x = fold_df['fold'].values
    plt.figure(figsize=(9, 5))
    plt.plot(x, fold_df['ROI_MAE'].values, marker='o', label='ROI_MAE')
    plt.plot(x, fold_df['Taxa_MAE'].values, marker='o', label='Taxa_MAE')
    plt.xlabel('Fold')
    plt.ylabel('MAE')
    plt.title('10-fold CV: MAE per fold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path.replace('.png', '_mae.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(x, fold_df['ROI_r'].values, marker='o', label='ROI_r')
    plt.plot(x, fold_df['Taxa_r_CLR'].values, marker='o', label='Taxa_r(CLR)')
    plt.xlabel('Fold')
    plt.ylabel('Pearson r')
    plt.title('10-fold CV: r per fold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path.replace('.png', '_r.png'), dpi=300)
    plt.close()


def plot_mean_std_bar(summary: Dict[str, Tuple[float, float]], out_path: str):
    keys = list(summary.keys())
    means = [summary[k][0] for k in keys]
    stds = [summary[k][1] for k in keys]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(keys)), means, yerr=stds, alpha=0.9, capsize=5)
    plt.xticks(range(len(keys)), keys, rotation=30, ha='right')
    plt.ylabel('Value')
    plt.title('10-fold CV: mean ± std')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# ----------------------------
# 新增：所有图都保存作图 CSV
# ----------------------------
def save_roi_scatter_with_csv(y_true, y_pred, out_png: str, out_csv: str, title_suffix: str = ''):
    yt = y_true.reshape(-1).cpu().numpy()
    yp = y_pred.reshape(-1).cpu().numpy()

    df = pd.DataFrame({
        'true_roi_bold': yt,
        'pred_roi_bold': yp
    })
    df.to_csv(out_csv, index=False)

    plt.figure(figsize=(6, 6))
    plt.scatter(yt, yp, s=8, alpha=0.5)
    vmin = min(yt.min(), yp.min())
    vmax = max(yt.max(), yp.max())
    plt.plot([vmin, vmax], [vmin, vmax], linewidth=1)
    plt.xlabel('True ROI BOLD')
    plt.ylabel('Pred ROI BOLD')
    plt.title(f'ROI True vs Pred {title_suffix}'.strip())
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def save_taxa_scatter_clr_with_csv(a_true, a_pred, out_png: str, out_csv: str, title_suffix: str = ''):
    at = clr_transform(a_true).reshape(-1).cpu().numpy()
    ap = clr_transform(a_pred).reshape(-1).cpu().numpy()

    df = pd.DataFrame({
        'true_taxa_clr': at,
        'pred_taxa_clr': ap
    })
    df.to_csv(out_csv, index=False)

    plt.figure(figsize=(6, 6))
    plt.scatter(at, ap, s=6, alpha=0.5)
    vmin = min(at.min(), ap.min())
    vmax = max(at.max(), ap.max())
    plt.plot([vmin, vmax], [vmin, vmax], linewidth=1)
    plt.xlabel('True CLR(abundance)')
    plt.ylabel('Pred CLR(abundance)')
    plt.title(f'Taxa True vs Pred (CLR) {title_suffix}'.strip())
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def save_r_hist_with_csv(r_values, out_png: str, out_csv: str, title: str, xlabel: str, col_name: str = 'r'):
    rv = r_values.cpu().numpy()
    pd.DataFrame({col_name: rv}).to_csv(out_csv, index=False)

    plt.figure(figsize=(6, 4))
    plt.hist(rv, bins=30, alpha=0.9)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_fold_metrics_with_csv(fold_df: pd.DataFrame, out_prefix: str):
    mae_df = fold_df[['fold', 'ROI_MAE', 'Taxa_MAE']].copy()
    mae_df.to_csv(out_prefix + '_mae_data.csv', index=False)

    x = mae_df['fold'].values
    plt.figure(figsize=(9, 5))
    plt.plot(x, mae_df['ROI_MAE'].values, marker='o', label='ROI_MAE')
    plt.plot(x, mae_df['Taxa_MAE'].values, marker='o', label='Taxa_MAE')
    plt.xlabel('Fold')
    plt.ylabel('MAE')
    plt.title('10-fold CV: MAE per fold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix + '_mae.png', dpi=300)
    plt.close()

    r_df = fold_df[['fold', 'ROI_r', 'Taxa_r_CLR']].copy()
    r_df.to_csv(out_prefix + '_r_data.csv', index=False)

    x = r_df['fold'].values
    plt.figure(figsize=(9, 5))
    plt.plot(x, r_df['ROI_r'].values, marker='o', label='ROI_r')
    plt.plot(x, r_df['Taxa_r_CLR'].values, marker='o', label='Taxa_r(CLR)')
    plt.xlabel('Fold')
    plt.ylabel('Pearson r')
    plt.title('10-fold CV: r per fold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix + '_r.png', dpi=300)
    plt.close()


def plot_fold_metrics_boxplot_with_csv(fold_df: pd.DataFrame, out_prefix: str):
    mae_long = pd.DataFrame({
        'metric': ['ROI_MAE'] * len(fold_df) + ['Taxa_MAE'] * len(fold_df),
        'value': fold_df['ROI_MAE'].tolist() + fold_df['Taxa_MAE'].tolist()
    })
    mae_long.to_csv(out_prefix + '_mae_box_data.csv', index=False)

    plt.figure(figsize=(7, 5))
    data = [
        fold_df['ROI_MAE'].values,
        fold_df['Taxa_MAE'].values
    ]
    plt.boxplot(data, labels=['ROI_MAE', 'Taxa_MAE'])
    plt.ylabel('MAE')
    plt.title('10-fold CV: MAE boxplot')
    plt.tight_layout()
    plt.savefig(out_prefix + '_mae_box.png', dpi=300)
    plt.close()

    r_long = pd.DataFrame({
        'metric': ['ROI_r'] * len(fold_df) + ['Taxa_r_CLR'] * len(fold_df),
        'value': fold_df['ROI_r'].tolist() + fold_df['Taxa_r_CLR'].tolist()
    })
    r_long.to_csv(out_prefix + '_r_box_data.csv', index=False)

    plt.figure(figsize=(7, 5))
    data = [
        fold_df['ROI_r'].values,
        fold_df['Taxa_r_CLR'].values
    ]
    plt.boxplot(data, labels=['ROI_r', 'Taxa_r_CLR'])
    plt.ylabel('Pearson r')
    plt.title('10-fold CV: r boxplot')
    plt.tight_layout()
    plt.savefig(out_prefix + '_r_box.png', dpi=300)
    plt.close()


def plot_mean_std_bar_with_csv(summary: Dict[str, Tuple[float, float]], out_png: str, out_csv: str):
    keys = list(summary.keys())
    means = [summary[k][0] for k in keys]
    stds = [summary[k][1] for k in keys]

    df = pd.DataFrame({
        'metric': keys,
        'mean': means,
        'std': stds
    })
    df.to_csv(out_csv, index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(keys)), means, yerr=stds, alpha=0.9, capsize=5)
    plt.xticks(range(len(keys)), keys, rotation=30, ha='right')
    plt.ylabel('Value')
    plt.title('10-fold CV: mean ± std')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def save_roi_performance_bar_with_csv(y_true, y_pred, out_png: str, out_csv: str, roi_names: Optional[List[str]] = None):
    mae = (y_pred - y_true).abs().mean(dim=0).cpu().numpy()
    r = pearsonr_per_col(y_pred, y_true).cpu().numpy()

    if roi_names is None:
        roi_names = [f'ROI_{i+1}' for i in range(y_true.shape[1])]

    df = pd.DataFrame({
        'ROI': roi_names,
        'MAE': mae,
        'Pearson_r': r
    })

    df = df.sort_values('Pearson_r', ascending=False).reset_index(drop=True)
    df.to_csv(out_csv, index=False)

    plt.figure(figsize=(18, 6))
    plt.bar(range(len(df)), df['Pearson_r'].values, alpha=0.9)
    plt.xticks(range(len(df)), df['ROI'].values, rotation=90, fontsize=7)
    plt.ylabel('Pearson r')
    plt.title('Performance of each ROI')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def save_top20_taxa_performance_bar_with_csv(a_true, a_pred, out_png: str, out_csv: str,
                                             taxa_names: Optional[List[str]] = None, topk: int = 20):
    mae = (a_pred - a_true).abs().mean(dim=0).cpu().numpy()

    a_true_clr = clr_transform(a_true)
    a_pred_clr = clr_transform(a_pred)
    r_clr = pearsonr_per_col(a_pred_clr, a_true_clr).cpu().numpy()

    if taxa_names is None:
        taxa_names = [f'Taxa_{i+1}' for i in range(a_true.shape[1])]

    df = pd.DataFrame({
        'Taxa': taxa_names,
        'MAE': mae,
        'Pearson_r_CLR': r_clr
    })

    df = df.sort_values('Pearson_r_CLR', ascending=False).reset_index(drop=True)
    df_top = df.head(topk).copy()
    df_top.to_csv(out_csv, index=False)

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(df_top)), df_top['Pearson_r_CLR'].values, alpha=0.9)
    plt.xticks(range(len(df_top)), df_top['Taxa'].values, rotation=60, ha='right', fontsize=9)
    plt.ylabel('Pearson r (CLR)')
    plt.title(f'Top {topk} taxa by performance')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

# ============================================================
# save helpers
# ============================================================
def save_mean_roi_csv(out_dir, roi_mean_pred, fname):
    os.makedirs(out_dir, exist_ok=True)
    v = roi_mean_pred.detach().cpu().numpy().reshape(-1)
    row = {f'roi_mean_pred_{i+1}': float(v[i]) for i in range(v.shape[0])}
    fp = os.path.join(out_dir, fname)
    pd.DataFrame([row]).to_csv(fp, index=False)
    return fp


def save_mean_taxa_csv(out_dir, taxa_mean_pred, fname):
    os.makedirs(out_dir, exist_ok=True)
    v = taxa_mean_pred.detach().cpu().numpy().reshape(-1)
    row = {f'taxa_mean_pred_{i+1}': float(v[i]) for i in range(v.shape[0])}
    fp = os.path.join(out_dir, fname)
    pd.DataFrame([row]).to_csv(fp, index=False)
    return fp


def save_taxa_pred_csv(out_dir, sids, a_pred, fname):
    os.makedirs(out_dir, exist_ok=True)
    A = a_pred.detach().cpu().numpy()
    rows = []
    for i, sid in enumerate(sids):
        row = {'subject_id': str(sid)}
        for t in range(A.shape[1]):
            row[f'taxa_pred_{t+1}'] = float(A[i, t])
        rows.append(row)
    fp = os.path.join(out_dir, fname)
    pd.DataFrame(rows).to_csv(fp, index=False)
    return fp


def save_bold_pred_csv(out_dir, sids, y_pred, fname):
    os.makedirs(out_dir, exist_ok=True)
    Y = y_pred.detach().cpu().numpy()
    rows = []
    for i, sid in enumerate(sids):
        row = {'subject_id': str(sid)}
        for r in range(Y.shape[1]):
            row[f'roi_pred_{r+1}'] = float(Y[i, r])
        rows.append(row)
    fp = os.path.join(out_dir, fname)
    pd.DataFrame(rows).to_csv(fp, index=False)
    return fp


def save_taxa_individual_diagnostics(out_dir, sids, a_pred, eps: float = 1e-8, fname: str = 'taxa_pred_diagnostics.csv'):
    os.makedirs(out_dir, exist_ok=True)
    A = a_pred.detach().cpu().clamp_min(eps)
    A = A / (A.sum(dim=1, keepdim=True) + eps)
    ent = -(A * A.log()).sum(dim=1)
    maxp = A.max(dim=1).values
    meanA = A.mean(dim=0, keepdim=True).clamp_min(eps)
    meanA = meanA / (meanA.sum(dim=1, keepdim=True) + eps)
    kl = (A * (A.log() - meanA.log())).sum(dim=1)
    A_clr = clr_transform(A, eps=1e-6)
    mean_clr = clr_transform(meanA, eps=1e-6).repeat(A_clr.size(0), 1)
    A_n = F.normalize(A_clr, dim=1)
    mean_n = F.normalize(mean_clr, dim=1)
    cos = (A_n * mean_n).sum(dim=1)
    l2 = torch.sqrt(((A_clr - mean_clr) ** 2).mean(dim=1))
    rows = []
    for i, sid in enumerate(sids):
        rows.append({
            'subject_id': str(sid),
            'entropy': float(ent[i].item()),
            'max_prob': float(maxp[i].item()),
            'kl_to_mean': float(kl[i].item()),
            'clr_cosine_to_mean': float(cos[i].item()),
            'clr_l2_to_mean': float(l2[i].item()),
        })
    fp = os.path.join(out_dir, fname)
    pd.DataFrame(rows).to_csv(fp, index=False)
    return fp


def save_bold_individual_diagnostics(out_dir, sids, y_pred, y_true=None, fname: str = 'bold_pred_diagnostics.csv', eps: float = 1e-8):
    os.makedirs(out_dir, exist_ok=True)
    Y = y_pred.detach().cpu()
    meanY = Y.mean(dim=0, keepdim=True)

    def zscore(v):
        v = v - v.mean(dim=1, keepdim=True)
        v = v / (v.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps))
        return v

    Z = zscore(Y)
    Zm = zscore(meanY).repeat(Z.size(0), 1)
    Z_n = F.normalize(Z, dim=1)
    Zm_n = F.normalize(Zm, dim=1)
    zcos = (Z_n * Zm_n).sum(dim=1)
    zl2 = torch.sqrt(((Z - Zm) ** 2).mean(dim=1))
    wstd = Y.std(dim=1, unbiased=False)
    mse = corr = None
    if y_true is not None:
        T = y_true.detach().cpu()
        mse = ((Y - T) ** 2).mean(dim=1)
        corr = pearsonr_per_row(Y, T)
    rows = []
    for i, sid in enumerate(sids):
        row = {
            'subject_id': str(sid),
            'z_cosine_to_mean': float(zcos[i].item()),
            'z_l2_to_mean': float(zl2[i].item()),
            'within_std': float(wstd[i].item())
        }
        if mse is not None:
            row['mse_to_true'] = float(mse[i].item())
        if corr is not None:
            row['corr_to_true'] = float(corr[i].item())
        rows.append(row)
    fp = os.path.join(out_dir, fname)
    pd.DataFrame(rows).to_csv(fp, index=False)
    return fp


def save_embedding_csv(out_dir, sids, z_b, z_m, z_sh, name='embedding.csv'):
    os.makedirs(out_dir, exist_ok=True)
    zb = z_b.numpy()
    zm = z_m.numpy()
    zs = z_sh.numpy()
    rows = []
    for i, sid in enumerate(sids):
        row = {'subject_id': str(sid)}
        for k in range(zs.shape[1]):
            row[f'e{k}'] = float(zs[i, k])
        for k in range(zb.shape[1]):
            row[f'eb{k}'] = float(zb[i, k])
        for k in range(zm.shape[1]):
            row[f'em{k}'] = float(zm[i, k])
        rows.append(row)
    fp = os.path.join(out_dir, name)
    pd.DataFrame(rows).to_csv(fp, index=False)
    return fp

# ============================================================
# kfold
# ============================================================
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
# inference helpers
# ============================================================
@torch.no_grad()
def predict_on_loader(model, loader, device):
    model.eval()
    all_y_true, all_y_pred, all_a_true, all_a_pred, all_idx = [], [], [], [], []
    for B, M, A, Y, IDX in loader:
        B, M, A, Y = B.to(device), M.to(device), A.to(device), Y.to(device)
        y_hat, a_hat, *_ = model(B, M, A, Y, mode='both')
        all_y_true.append(Y.cpu())
        all_y_pred.append(y_hat.cpu())
        all_a_true.append(A.cpu())
        all_a_pred.append(a_hat.cpu())
        all_idx.append(IDX.cpu())
    return (
        torch.cat(all_y_true, 0),
        torch.cat(all_y_pred, 0),
        torch.cat(all_a_true, 0),
        torch.cat(all_a_pred, 0),
        torch.cat(all_idx, 0)
    )


@torch.no_grad()
def predict_on_loader_with_embedding(model, loader, device):
    model.eval()
    all_y_true, all_y_pred, all_a_true, all_a_pred = [], [], [], []
    all_bb, all_mb, all_zb, all_zm, all_zsh, all_idx = [], [], [], [], [], []
    for B, M, A, Y, IDX in loader:
        B, M, A, Y = B.to(device), M.to(device), A.to(device), Y.to(device)
        y_hat, a_hat, B_bridge_n, _B_lat_n, M_bridge_n, _M_lat_n, _roi_alpha, zb, zm, zsh = model(
            B, M, A, Y, mode='both'
        )
        all_y_true.append(Y.cpu())
        all_y_pred.append(y_hat.cpu())
        all_a_true.append(A.cpu())
        all_a_pred.append(a_hat.cpu())
        all_bb.append(B_bridge_n.cpu())
        all_mb.append(M_bridge_n.cpu())
        all_zb.append(zb.cpu())
        all_zm.append(zm.cpu())
        all_zsh.append(zsh.cpu())
        all_idx.append(IDX.cpu())
    return (
        torch.cat(all_y_true, 0),
        torch.cat(all_y_pred, 0),
        torch.cat(all_a_true, 0),
        torch.cat(all_a_pred, 0),
        torch.cat(all_bb, 0),
        torch.cat(all_mb, 0),
        torch.cat(all_zb, 0),
        torch.cat(all_zm, 0),
        torch.cat(all_zsh, 0),
        torch.cat(all_idx, 0)
    )


@torch.no_grad()
def external_brain2micro_predict(models, B_ext, Y_ext, device, batch_size: int):
    a_preds = []
    for i in range(0, B_ext.size(0), batch_size):
        B_b = B_ext[i:i + batch_size].to(device)
        Y_b = Y_ext[i:i + batch_size].to(device)
        a_sum = 0.0
        for m in models:
            m.eval()
            _, a_hat, *_ = m(B_true=B_b, M_true=None, A_true=None, Y_true=Y_b, mode='brain2micro')
            a_sum = a_sum + a_hat
        a_preds.append((a_sum / len(models)).cpu())
    return torch.cat(a_preds, dim=0)


@torch.no_grad()
def external_micro2brain_predict(models, M_ext, A_ext, device, batch_size: int):
    y_preds = []
    for i in range(0, M_ext.size(0), batch_size):
        M_b = M_ext[i:i + batch_size].to(device)
        A_b = A_ext[i:i + batch_size].to(device)
        y_sum = 0.0
        for m in models:
            m.eval()
            y_hat, _, *_ = m(B_true=None, M_true=M_b, A_true=A_b, Y_true=None, mode='micro2brain')
            y_sum = y_sum + y_hat
        y_preds.append((y_sum / len(models)).cpu())
    return torch.cat(y_preds, dim=0)

# ============================================================
# checkpoint utilities
# ============================================================
def _resolve_ckpt_dir() -> str:
    ckpt_dir = CONFIG.get('CKPT_DIR', '')
    if ckpt_dir is None:
        ckpt_dir = ''
    ckpt_dir = str(ckpt_dir).strip()
    return str(CONFIG['OUT_DIR']) if ckpt_dir == '' else ckpt_dir


def _infer_model_dims_from_ckpt(best_model_path: str) -> Tuple[int, int]:
    ckpt = torch.load(best_model_path, map_location='cpu')
    if 'model' not in ckpt:
        raise ValueError(f"Bad checkpoint format (no 'model'): {best_model_path}")
    sd = ckpt['model']
    k_fb = 'f.net.0.weight'
    k_gm = 'g.net.0.weight'
    if k_fb not in sd or k_gm not in sd:
        raise KeyError(f"Cannot find '{k_fb}' or '{k_gm}' in state_dict.")
    return int(sd[k_fb].shape[1]), int(sd[k_gm].shape[1])


def load_all_fold_models(device: torch.device) -> List[DualMicroBrainBridge]:
    ckpt_root = _resolve_ckpt_dir()
    n_folds = int(CONFIG['N_FOLDS'])
    first_best = os.path.join(ckpt_root, 'fold_01', 'best_model.pt')
    if not os.path.exists(first_best):
        for fold_id in range(1, n_folds + 1):
            cand = os.path.join(ckpt_root, f'fold_{fold_id:02d}', 'best_model.pt')
            if os.path.exists(cand):
                first_best = cand
                break
    if not os.path.exists(first_best):
        raise FileNotFoundError(f'No checkpoint found under: {ckpt_root}')
    d_b, d_m = _infer_model_dims_from_ckpt(first_best)
    models = []
    for fold_id in range(1, n_folds + 1):
        best_path = os.path.join(ckpt_root, f'fold_{fold_id:02d}', 'best_model.pt')
        if not os.path.exists(best_path):
            raise FileNotFoundError(f'best_model.pt not found: {best_path}')
        m = DualMicroBrainBridge(
            d_b=d_b,
            d_m=d_m,
            d_align=int(CONFIG['D_ALIGN']),
            dropout=float(CONFIG['DROPOUT']),
            n_heads=int(CONFIG['N_HEADS'])
        ).to(device)
        ckpt = torch.load(best_path, map_location=device)
        m.load_state_dict(ckpt['model'])
        m.eval()
        models.append(m)
    return models

# ============================================================
# train one fold
# ============================================================
def train_one_fold(fold_id: int, B_all, M_all, A_all, Y_all, train_idx, val_idx, device, out_fold_dir: str):
    os.makedirs(out_fold_dir, exist_ok=True)
    train_set = DualBridgeDatasetCV(B_all[train_idx], M_all[train_idx], A_all[train_idx], Y_all[train_idx], idx_global=train_idx)
    val_set = DualBridgeDatasetCV(B_all[val_idx], M_all[val_idx], A_all[val_idx], Y_all[val_idx], idx_global=val_idx)
    train_loader = DataLoader(train_set, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=0, collate_fn=collate_fn, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=0, collate_fn=collate_fn, drop_last=False)

    d_b = B_all.shape[-1]
    d_m = M_all.shape[-1]
    model = DualMicroBrainBridge(d_b=d_b, d_m=d_m, d_align=CONFIG['D_ALIGN'], dropout=CONFIG['DROPOUT'], n_heads=CONFIG['N_HEADS']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['LR'], weight_decay=CONFIG['WEIGHT_DECAY'])
    scheduler = None
    if CONFIG['USE_LR_SCHED']:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=CONFIG['SCHED_FACTOR'],
            patience=CONFIG['SCHED_PATIENCE'],
            min_lr=CONFIG['MIN_LR']
        )

    log_path = os.path.join(out_fold_dir, 'train_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(
            'epoch\tlr\ttrain_loss\tval_roi_mae\tval_taxa_mae\tval_roi_r\tval_taxa_r_CLR\t'
            'roi_alpha\tdiv_ab\tdiv_y_pred\tdiv_y_true\ty_var\tdivy_band\tsim_y\tloss_shared\t'
            'loss_y_dev\tloss_a_dev\tloss_y_dist\tloss_a_dist\n'
        )

    best_score = float('inf')
    best_path = os.path.join(out_fold_dir, 'best_model.pt')
    no_improve = 0
    accum_steps = max(1, int(CONFIG['ACCUM_STEPS']))

    for epoch in range(1, CONFIG['EPOCHS'] + 1):
        model.train()
        loss_list = []
        div_ab_list, div_y_list, div_y_true_list = [], [], []
        yvar_list, divy_band_list, simy_list, shared_list = [], [], [], []
        ydev_list, adev_list, ydist_list, adist_list = [], [], [], []

        optimizer.zero_grad(set_to_none=True)

        for step, (B_b, M_b, A_b, Y_b, _) in enumerate(train_loader):
            B_b, M_b, A_b, Y_b = B_b.to(device), M_b.to(device), A_b.to(device), Y_b.to(device)

            if CONFIG['AUG_ENABLE']:
                if random.random() < CONFIG['AUG_MIXUP_PROB']:
                    B_b, M_b, A_b, Y_b = vanilla_mixup(
                        B_b, M_b, A_b, Y_b,
                        alpha=CONFIG['AUG_MIXUP_ALPHA'],
                        eps=CONFIG['EPS']
                    )
                if random.random() < CONFIG['AUG_NOISE_PROB']:
                    B_b, M_b = add_noise(B_b, M_b, CONFIG['AUG_BRAIN_NOISE_STD'], CONFIG['AUG_GUT_NOISE_STD'])
                if random.random() < CONFIG['AUG_NODEDROP_PROB']:
                    B_b, M_b = node_dropout(B_b, M_b, CONFIG['AUG_BRAIN_NODEDROP_P'], CONFIG['AUG_GUT_NODEDROP_P'])

            y_hat, a_hat, B_bridge_n, B_lat_n, M_bridge_n, M_lat_n, roi_alpha, zb, zm, zsh = model(
                B_true=B_b, M_true=M_b, A_true=A_b, Y_true=Y_b, mode='both'
            )

            loss_y = huber_or_mse(y_hat, Y_b, use_huber=CONFIG['USE_HUBER'], delta=CONFIG['HUBER_DELTA'])
            loss_ycorr = roi_corr_loss(y_hat, Y_b)
            loss_ab = dist_ce_loss(a_hat, A_b)
            loss_abcorr = abundance_corr_loss_clr(a_hat, A_b)
            loss_ab_clrmse = abundance_clr_mse_loss(a_hat, A_b)
            loss_align_b = cosine_align_loss(B_bridge_n, B_lat_n)
            loss_align_m = cosine_align_loss(M_bridge_n, M_lat_n)
            div_ab = batch_diversity_loss_clr(a_hat, eps=1e-6)
            div_y_pred = batch_diversity_loss_bold(y_hat, eps=1e-6)
            div_y_true = batch_diversity_loss_bold(Y_b, eps=1e-6)
            loss_divy_band = divy_target_loss(div_y_pred)
            loss_sim_y = batch_similarity_alignment_loss_bold(y_hat, Y_b, eps=1e-6)
            loss_y_var = bold_variance_floor_loss(y_hat, Y_b, gamma=float(CONFIG['Y_VAR_GAMMA']), eps=CONFIG['EPS'])
            loss_shared = info_nce_loss(zb, zm, temperature=float(CONFIG['ALIGN_TAU']), eps=CONFIG['EPS'])
            loss_y_dev = individual_deviation_loss(y_hat, Y_b)
            loss_a_dev = abundance_individual_deviation_loss(a_hat, A_b)
            loss_y_dist = individual_distance_preserve_loss(y_hat, Y_b)
            loss_a_dist = abundance_individual_distance_loss(a_hat, A_b)

            loss = (
                CONFIG['LAMBDA_Y'] * loss_y
                + CONFIG['LAMBDA_YCORR'] * loss_ycorr
                + CONFIG['LAMBDA_AB'] * loss_ab
                + CONFIG['LAMBDA_ABCORR'] * loss_abcorr
                + CONFIG['LAMBDA_AB_CLR_MSE'] * loss_ab_clrmse
                + CONFIG['LAMBDA_ALIGN_B'] * loss_align_b
                + CONFIG['LAMBDA_ALIGN_M'] * loss_align_m
                + CONFIG['LAMBDA_DIVERSITY_AB'] * div_ab
                + CONFIG['LAMBDA_DIVERSITY_Y'] * loss_divy_band
                + CONFIG['Y_SIM_ALIGN_LAMBDA'] * loss_sim_y
                + CONFIG['LAMBDA_Y_VAR'] * loss_y_var
                + CONFIG['ALIGN_SHARED_LAMBDA'] * loss_shared
                + CONFIG['LAMBDA_Y_DEV'] * loss_y_dev
                + CONFIG['LAMBDA_A_DEV'] * loss_a_dev
                + CONFIG['LAMBDA_Y_DIST'] * loss_y_dist
                + CONFIG['LAMBDA_A_DIST'] * loss_a_dist
            )

            if CONFIG['SINGLE_MODAL_TRAIN_ENABLE']:
                if random.random() < CONFIG['SINGLE_MODAL_P_BRAIN2MICRO']:
                    _, a_hat_b2m, *_ = model(B_true=B_b, M_true=None, A_true=None, Y_true=Y_b, mode='brain2micro')
                    loss_b2m = (
                        CONFIG['LAMBDA_AB'] * dist_ce_loss(a_hat_b2m, A_b)
                        + CONFIG['LAMBDA_ABCORR'] * abundance_corr_loss_clr(a_hat_b2m, A_b)
                        + CONFIG['LAMBDA_AB_CLR_MSE'] * abundance_clr_mse_loss(a_hat_b2m, A_b)
                        + CONFIG['LAMBDA_A_DEV'] * abundance_individual_deviation_loss(a_hat_b2m, A_b)
                        + CONFIG['LAMBDA_A_DIST'] * abundance_individual_distance_loss(a_hat_b2m, A_b)
                    )
                    loss = loss + CONFIG['SINGLE_MODAL_BRANCH_WEIGHT'] * loss_b2m

                if random.random() < CONFIG['SINGLE_MODAL_P_MICRO2BRAIN']:
                    y_hat_m2b, _, *_ = model(B_true=None, M_true=M_b, A_true=A_b, Y_true=None, mode='micro2brain')
                    loss_m2b = (
                        CONFIG['LAMBDA_Y'] * huber_or_mse(y_hat_m2b, Y_b, use_huber=CONFIG['USE_HUBER'], delta=CONFIG['HUBER_DELTA'])
                        + CONFIG['LAMBDA_YCORR'] * roi_corr_loss(y_hat_m2b, Y_b)
                        + CONFIG['LAMBDA_Y_DEV'] * individual_deviation_loss(y_hat_m2b, Y_b)
                        + CONFIG['LAMBDA_Y_DIST'] * individual_distance_preserve_loss(y_hat_m2b, Y_b)
                    )
                    loss = loss + CONFIG['SINGLE_MODAL_BRANCH_WEIGHT'] * loss_m2b

            (loss / accum_steps).backward()

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            loss_list.append(loss.item())
            div_ab_list.append(float(div_ab.detach().cpu().item()))
            div_y_list.append(float(div_y_pred.detach().cpu().item()))
            div_y_true_list.append(float(div_y_true.detach().cpu().item()))
            yvar_list.append(float(loss_y_var.detach().cpu().item()))
            divy_band_list.append(float(loss_divy_band.detach().cpu().item()))
            simy_list.append(float(loss_sim_y.detach().cpu().item()))
            shared_list.append(float(loss_shared.detach().cpu().item()))
            ydev_list.append(float(loss_y_dev.detach().cpu().item()))
            adev_list.append(float(loss_a_dev.detach().cpu().item()))
            ydist_list.append(float(loss_y_dist.detach().cpu().item()))
            adist_list.append(float(loss_a_dist.detach().cpu().item()))

        if len(train_loader) % accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        train_loss = float(np.mean(loss_list)) if loss_list else float('nan')
        train_div_ab = float(np.mean(div_ab_list)) if div_ab_list else float('nan')
        train_div_y = float(np.mean(div_y_list)) if div_y_list else float('nan')
        train_div_y_true = float(np.mean(div_y_true_list)) if div_y_true_list else float('nan')
        train_yvar = float(np.mean(yvar_list)) if yvar_list else float('nan')
        train_divy_band = float(np.mean(divy_band_list)) if divy_band_list else float('nan')
        train_simy = float(np.mean(simy_list)) if simy_list else float('nan')
        train_shared = float(np.mean(shared_list)) if shared_list else float('nan')
        train_ydev = float(np.mean(ydev_list)) if ydev_list else float('nan')
        train_adev = float(np.mean(adev_list)) if adev_list else float('nan')
        train_ydist = float(np.mean(ydist_list)) if ydist_list else float('nan')
        train_adist = float(np.mean(adist_list)) if adist_list else float('nan')

        y_true_v, y_pred_v, a_true_v, a_pred_v, _ = predict_on_loader(model, val_loader, device)
        roi_mae, roi_r_mean, _, _ = compute_roi_metrics(y_pred_v, y_true_v)
        taxa_mae, taxa_r_mean, _, _ = compute_taxa_metrics(a_pred_v, a_true_v)

        val_score = (
            roi_mae
            + 0.40 * (1.0 - roi_r_mean)
            + 0.10 * taxa_mae
            + 0.40 * (1.0 - taxa_r_mean)
        )

        if scheduler is not None:
            scheduler.step(val_score)

        cur_lr = optimizer.param_groups[0]['lr']
        roi_alpha_val = float(roi_alpha.detach().cpu().item())

        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(
                f'{epoch}\t{cur_lr:.8f}\t{train_loss:.6f}\t{roi_mae:.6f}\t{taxa_mae:.6f}\t'
                f'{roi_r_mean:.6f}\t{taxa_r_mean:.6f}\t{roi_alpha_val:.3f}\t{train_div_ab:.6f}\t'
                f'{train_div_y:.6f}\t{train_div_y_true:.6f}\t{train_yvar:.6f}\t{train_divy_band:.6f}\t'
                f'{train_simy:.6f}\t{train_shared:.6f}\t{train_ydev:.6f}\t{train_adev:.6f}\t'
                f'{train_ydist:.6f}\t{train_adist:.6f}\n'
            )

        print(
            f'[Fold {fold_id:02d} | Epoch {epoch:03d}] lr={cur_lr:.2e} train_loss={train_loss:.5f} | '
            f'val_score={val_score:.5f} | '
            f'val ROI: MAE={roi_mae:.5f}, r={roi_r_mean:.5f} | '
            f'val Taxa: MAE={taxa_mae:.5f}, r(CLR)={taxa_r_mean:.5f} | '
            f'roi_alpha={roi_alpha_val:.3f} | div_ab={train_div_ab:.3f} | '
            f'div_y_pred={train_div_y:.3f} (true~{train_div_y_true:.3f}) | '
            f'yvar={train_yvar:.4f} | divy_band={train_divy_band:.4f} | sim_y={train_simy:.4f} | '
            f'shared={train_shared:.4f} | y_dev={train_ydev:.4f} | a_dev={train_adev:.4f} | '
            f'y_dist={train_ydist:.4f} | a_dist={train_adist:.4f}'
        )

        if val_score < best_score:
            best_score = val_score
            no_improve = 0
            torch.save({'model': model.state_dict(), 'config': CONFIG}, best_path)
        else:
            no_improve += 1

        if CONFIG['USE_EARLY_STOP'] and no_improve >= CONFIG['EARLY_STOP_PATIENCE']:
            print(f'[Fold {fold_id:02d}] EARLY STOP at epoch={epoch}, no improve={no_improve}.')
            break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model'])

    y_true_v, y_pred_v, a_true_v, a_pred_v, B_bridge_v, M_bridge_v, z_b_v, z_m_v, z_sh_v, idx_global_v = predict_on_loader_with_embedding(model, val_loader, device)
    roi_mae, roi_r_mean, _, _ = compute_roi_metrics(y_pred_v, y_true_v)
    taxa_mae, taxa_r_mean, _, _ = compute_taxa_metrics(a_pred_v, a_true_v)

    fold_metrics = {
        'fold': fold_id,
        'ROI_MAE': roi_mae,
        'ROI_r': roi_r_mean,
        'Taxa_MAE': taxa_mae,
        'Taxa_r_CLR': taxa_r_mean
    }

    return fold_metrics, y_true_v, y_pred_v, a_true_v, a_pred_v, B_bridge_v, M_bridge_v, z_b_v, z_m_v, z_sh_v, idx_global_v

# ============================================================
# main
# ============================================================
def main():
    os.makedirs(CONFIG['OUT_DIR'], exist_ok=True)
    set_seed(CONFIG['SEED'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('[ENV] device =', device)
    run_train = bool(CONFIG.get('RUN_TRAIN_CV', True))
    print(f"[SWITCH] RUN_TRAIN_CV={run_train}")
    print(f"[SWITCH] RUN_EXT_BRAIN2MICRO={bool(CONFIG.get('RUN_EXT_BRAIN2MICRO', False))}")
    print(f"[SWITCH] RUN_EXT_MICRO2BRAIN={bool(CONFIG.get('RUN_EXT_MICRO2BRAIN', False))}")
    print(f"[CKPT] CKPT_DIR resolved to: {_resolve_ckpt_dir()}")

    # --------------------------------------------------------
    # 1) 训练 + 10折CV
    # --------------------------------------------------------
    if run_train:
        for k in ['BRAIN_FEAT_PT', 'GUT_FEAT_PT', 'MICRO_ABUND_CSV', 'ROI_MEAN_CSV']:
            if not os.path.exists(CONFIG[k]):
                raise FileNotFoundError(f"{k} 不存在：{CONFIG[k]}")

        B_all, brain_sids, _ = load_brain_node_feat(CONFIG['BRAIN_FEAT_PT'])
        M_all, gut_sids, _ = load_gut_node_feat(CONFIG['GUT_FEAT_PT'])
        abund_map = load_abundance_table(CONFIG['MICRO_ABUND_CSV'])
        roi_map = load_roi_mean_table(CONFIG['ROI_MEAN_CSV'])
        B_all, M_all, A_all, Y_all, sids = intersect_and_stack_noW(B_all, brain_sids, M_all, gut_sids, abund_map, roi_map)
        A_all = _require_distribution(A_all, eps=CONFIG['EPS'])

        N = B_all.shape[0]
        print(f'[DATA] aligned subjects N={N}')
        print('[CHECK] first 5 subject_ids:', sids[:5])

        n_folds = int(CONFIG['N_FOLDS'])
        folds = kfold_indices(N, n_folds=n_folds, seed=CONFIG['SEED'])
        print(f'[CV] n_folds={n_folds}')

        y_oof = torch.full((N, N_ROIS), float('nan'), dtype=torch.float32)
        a_oof = torch.full((N, N_TAXA), float('nan'), dtype=torch.float32)
        d_align = int(CONFIG['D_ALIGN'])
        B_bridge_oof = torch.full((N, N_ROIS, d_align), float('nan'), dtype=torch.float32)
        M_bridge_oof = torch.full((N, N_TAXA, d_align), float('nan'), dtype=torch.float32)
        z_b_oof = torch.full((N, d_align), float('nan'), dtype=torch.float32)
        z_m_oof = torch.full((N, d_align), float('nan'), dtype=torch.float32)
        z_sh_oof = torch.full((N, d_align), float('nan'), dtype=torch.float32)

        y_true_all = Y_all.clone().cpu()
        a_true_all = A_all.clone().cpu()
        fold_metrics_list = []

        for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
            print('\n' + '=' * 70)
            print(f'[CV] Fold {fold_id}/{n_folds} | train={len(train_idx)} val={len(val_idx)}')
            print('=' * 70)

            out_fold_dir = os.path.join(CONFIG['OUT_DIR'], f'fold_{fold_id:02d}')
            set_seed(CONFIG['SEED'] + fold_id)

            fold_metrics, y_true_v, y_pred_v, a_true_v, a_pred_v, B_bridge_v, M_bridge_v, z_b_v, z_m_v, z_sh_v, idx_global_v = train_one_fold(
                fold_id=fold_id,
                B_all=B_all,
                M_all=M_all,
                A_all=A_all,
                Y_all=Y_all,
                train_idx=train_idx,
                val_idx=val_idx,
                device=device,
                out_fold_dir=out_fold_dir,
            )

            fold_metrics_list.append(fold_metrics)

            idx = idx_global_v.long()
            y_oof[idx] = y_pred_v.cpu()
            a_oof[idx] = a_pred_v.cpu()
            B_bridge_oof[idx] = B_bridge_v.cpu()
            M_bridge_oof[idx] = M_bridge_v.cpu()
            z_b_oof[idx] = z_b_v.cpu()
            z_m_oof[idx] = z_m_v.cpu()
            z_sh_oof[idx] = z_sh_v.cpu()

        if (torch.isnan(y_oof).any() or torch.isnan(a_oof).any() or torch.isnan(B_bridge_oof).any() or torch.isnan(M_bridge_oof).any() or torch.isnan(z_sh_oof).any()):
            raise RuntimeError('[OOF] has missing predictions/embeddings. Check folds.')

        # ----------------------------------------------------
        # 保存核心数值结果
        # ----------------------------------------------------
        fold_df = pd.DataFrame(fold_metrics_list)
        fold_csv = os.path.join(CONFIG['OUT_DIR'], 'cv_fold_metrics.csv')
        fold_df.to_csv(fold_csv, index=False)

        summary = {
            k: (float(fold_df[k].mean()), float(fold_df[k].std(ddof=0)))
            for k in ['ROI_MAE', 'ROI_r', 'Taxa_MAE', 'Taxa_r_CLR']
        }
        summary_txt = os.path.join(CONFIG['OUT_DIR'], 'cv_summary.txt')
        with open(summary_txt, 'w', encoding='utf-8') as f:
            f.write('==== 10-fold CV mean ± std (fold-level) ====\n')
            for k, (m, s) in summary.items():
                f.write(f'{k}: {m:.6f} ± {s:.6f}\n')

            roi_mae_all, roi_r_mean_all, roi_r_std_all, roi_r_each_subject = compute_roi_metrics(y_oof, y_true_all)
            taxa_mae_all, taxa_r_mean_all, taxa_r_std_all, taxa_r_each_subject = compute_taxa_metrics(a_oof, a_true_all)

            f.write('\n==== OOF overall metrics ====\n')
            f.write(f'ROI_MAE: {roi_mae_all:.6f}\n')
            f.write(f'ROI_r_mean: {roi_r_mean_all:.6f}\n')
            f.write(f'ROI_r_std: {roi_r_std_all:.6f}\n')
            f.write(f'Taxa_MAE: {taxa_mae_all:.6f}\n')
            f.write(f'Taxa_r_CLR_mean: {taxa_r_mean_all:.6f}\n')
            f.write(f'Taxa_r_CLR_std: {taxa_r_std_all:.6f}\n')

        # 保存 OOF 预测
        save_bold_pred_csv(CONFIG['OUT_DIR'], sids, y_oof, CONFIG['OOF_BOLD_PRED_CSV'])
        save_taxa_pred_csv(CONFIG['OUT_DIR'], sids, a_oof, CONFIG['OOF_TAXA_PRED_CSV'])

        # 保存个体诊断信息
        save_bold_individual_diagnostics(
            CONFIG['OUT_DIR'], sids, y_oof, y_true=y_true_all, fname=CONFIG['OOF_BOLD_DIAG_CSV']
        )
        save_taxa_individual_diagnostics(
            CONFIG['OUT_DIR'], sids, a_oof, fname=CONFIG['OOF_TAXA_DIAG_CSV']
        )

        # 保存 embedding
        save_embedding_csv(
            CONFIG['OUT_DIR'], sids, z_b_oof, z_m_oof, z_sh_oof, name=CONFIG['OOF_EMBEDDING_CSV']
        )

        # Save OOF node-level bridge representations for the latent-factor/subtype workflow.
        np.savez_compressed(
            os.path.join(CONFIG['OUT_DIR'], CONFIG['OOF_NODE_REPRESENTATION_NPZ']),
            B_bridge_normed=B_bridge_oof.numpy(),
            M_bridge_normed=M_bridge_oof.numpy(),
        )
        pd.DataFrame({'sample_id': [str(x) for x in sids]}).to_csv(
            os.path.join(CONFIG['OUT_DIR'], CONFIG['OOF_ALIGNED_SAMPLES_CSV']), index=False
        )

        # 保存均值预测
        save_mean_roi_csv(
            CONFIG['OUT_DIR'],
            y_oof.mean(dim=0),
            CONFIG['MEAN_PRED_ROI_CSV']
        )
        save_mean_taxa_csv(
            CONFIG['OUT_DIR'],
            a_oof.mean(dim=0),
            CONFIG['MEAN_PRED_TAXA_CSV']
        )

        # ----------------------------------------------------
        # 作图 + 保存可复现作图 CSV
        # ----------------------------------------------------
        plot_fold_metrics_with_csv(
            fold_df,
            out_prefix=os.path.join(CONFIG['OUT_DIR'], 'cv_fold_metrics')
        )

        plot_fold_metrics_boxplot_with_csv(
            fold_df,
            out_prefix=os.path.join(CONFIG['OUT_DIR'], 'cv_fold_metrics_boxplot')
        )

        plot_mean_std_bar_with_csv(
            summary,
            out_png=os.path.join(CONFIG['OUT_DIR'], 'cv_mean_std_bar.png'),
            out_csv=os.path.join(CONFIG['OUT_DIR'], 'cv_mean_std_bar_data.csv')
        )

        save_roi_scatter_with_csv(
            y_true_all, y_oof,
            out_png=os.path.join(CONFIG['OUT_DIR'], 'cv_oof_roi_scatter.png'),
            out_csv=os.path.join(CONFIG['OUT_DIR'], 'cv_oof_roi_scatter_data.csv'),
            title_suffix='(OOF)'
        )

        save_taxa_scatter_clr_with_csv(
            a_true_all, a_oof,
            out_png=os.path.join(CONFIG['OUT_DIR'], 'cv_oof_taxa_scatter_clr.png'),
            out_csv=os.path.join(CONFIG['OUT_DIR'], 'cv_oof_taxa_scatter_clr_data.csv'),
            title_suffix='(OOF)'
        )

        roi_mae_all, roi_r_mean_all, roi_r_std_all, roi_r_each_subject = compute_roi_metrics(y_oof, y_true_all)
        taxa_mae_all, taxa_r_mean_all, taxa_r_std_all, taxa_r_each_subject = compute_taxa_metrics(a_oof, a_true_all)

        save_r_hist_with_csv(
            roi_r_each_subject,
            out_png=os.path.join(CONFIG['OUT_DIR'], 'cv_oof_roi_r_hist.png'),
            out_csv=os.path.join(CONFIG['OUT_DIR'], 'cv_oof_roi_r_hist_data.csv'),
            title='ROI subject-level Pearson r distribution',
            xlabel='Pearson r',
            col_name='roi_subject_r'
        )

        save_r_hist_with_csv(
            taxa_r_each_subject,
            out_png=os.path.join(CONFIG['OUT_DIR'], 'cv_oof_taxa_r_hist_clr.png'),
            out_csv=os.path.join(CONFIG['OUT_DIR'], 'cv_oof_taxa_r_hist_clr_data.csv'),
            title='Taxa subject-level Pearson r distribution (CLR)',
            xlabel='Pearson r',
            col_name='taxa_subject_r_clr'
        )

        save_roi_performance_bar_with_csv(
            y_true_all, y_oof,
            out_png=os.path.join(CONFIG['OUT_DIR'], 'roi_performance_bar.png'),
            out_csv=os.path.join(CONFIG['OUT_DIR'], 'roi_performance_bar_data.csv'),
            roi_names=None
        )

        save_top20_taxa_performance_bar_with_csv(
            a_true_all, a_oof,
            out_png=os.path.join(CONFIG['OUT_DIR'], 'top20_taxa_performance_bar.png'),
            out_csv=os.path.join(CONFIG['OUT_DIR'], 'top20_taxa_performance_bar_data.csv'),
            taxa_names=None,
            topk=20
        )

        # 同时保留旧版不带 csv 的接口（可选，不影响）
        plot_fold_metrics(fold_df, os.path.join(CONFIG['OUT_DIR'], 'cv_fold_metrics_legacy.png'))
        plot_mean_std_bar(summary, os.path.join(CONFIG['OUT_DIR'], 'cv_mean_std_bar_legacy.png'))
        save_roi_scatter(y_true_all, y_oof, os.path.join(CONFIG['OUT_DIR'], 'cv_oof_roi_scatter_legacy.png'), title_suffix='(OOF)')
        save_taxa_scatter_clr(a_true_all, a_oof, os.path.join(CONFIG['OUT_DIR'], 'cv_oof_taxa_scatter_clr_legacy.png'), title_suffix='(OOF)')
        save_r_hist(roi_r_each_subject, os.path.join(CONFIG['OUT_DIR'], 'cv_oof_roi_r_hist_legacy.png'),
                    title='ROI subject-level Pearson r distribution', xlabel='Pearson r')
        save_r_hist(taxa_r_each_subject, os.path.join(CONFIG['OUT_DIR'], 'cv_oof_taxa_r_hist_clr_legacy.png'),
                    title='Taxa subject-level Pearson r distribution (CLR)', xlabel='Pearson r')

        print('\n[Done] Training / CV / OOF / plotting completed.')
        print(f'[OUT] Results saved to: {CONFIG["OUT_DIR"]}')

    # --------------------------------------------------------
    # 2) 外部 brain -> micro 预测
    # --------------------------------------------------------
    if bool(CONFIG.get('RUN_EXT_BRAIN2MICRO', False)):
        print('\n' + '=' * 70)
        print('[EXT] Running external brain2micro prediction...')
        print('=' * 70)

        for k in ['EXT_BRAIN_FEAT_PT', 'EXT_ROI_MEAN_CSV']:
            if not os.path.exists(CONFIG[k]):
                raise FileNotFoundError(f"{k} 不存在：{CONFIG[k]}")

        models = load_all_fold_models(device)

        B_ext, brain_ext_sids, _ = load_brain_node_feat(CONFIG['EXT_BRAIN_FEAT_PT'])
        roi_ext_map = load_roi_mean_table(CONFIG['EXT_ROI_MEAN_CSV'])
        B_ext, Y_ext, sids_ext = intersect_brain_and_roi(B_ext, brain_ext_sids, roi_ext_map)

        a_pred_ext = external_brain2micro_predict(models, B_ext, Y_ext, device, CONFIG['BATCH_SIZE'])
        save_taxa_pred_csv(CONFIG['OUT_DIR'], sids_ext, a_pred_ext, CONFIG['EXT_BRAIN2MICRO_TAXA_PRED_CSV'])

        mean_pred = a_pred_ext.mean(dim=0)
        save_mean_taxa_csv(CONFIG['OUT_DIR'], mean_pred, 'EXT_mean_pred_taxa.csv')

        if os.path.exists(CONFIG['EXT_MICRO_ABUND_CSV']):
            abund_ext_map = load_abundance_table(CONFIG['EXT_MICRO_ABUND_CSV'])
            common = [sid for sid in sids_ext if sid in abund_ext_map]
            if len(common) > 0:
                idx_map = {sid: i for i, sid in enumerate(sids_ext)}
                A_true_ext = torch.stack([abund_ext_map[sid] for sid in common], dim=0)
                A_pred_common = torch.stack([a_pred_ext[idx_map[sid]] for sid in common], dim=0)

                taxa_mae, taxa_r_mean, taxa_r_std, _ = compute_taxa_metrics(A_pred_common, A_true_ext)
                with open(os.path.join(CONFIG['OUT_DIR'], CONFIG['EXT_BRAIN2MICRO_METRICS_TXT']), 'w', encoding='utf-8') as f:
                    f.write('[EXT brain2micro metrics]\n')
                    f.write(f'N_common={len(common)}\n')
                    f.write(f'Taxa_MAE={taxa_mae:.6f}\n')
                    f.write(f'Taxa_r_CLR_mean={taxa_r_mean:.6f}\n')
                    f.write(f'Taxa_r_CLR_std={taxa_r_std:.6f}\n')

                save_taxa_scatter_clr_with_csv(
                    A_true_ext, A_pred_common,
                    out_png=os.path.join(CONFIG['OUT_DIR'], 'EXT_brain2micro_taxa_scatter_clr.png'),
                    out_csv=os.path.join(CONFIG['OUT_DIR'], 'EXT_brain2micro_taxa_scatter_clr_data.csv'),
                    title_suffix='(EXT brain2micro)'
                )

        print('[EXT] external brain2micro finished.')

    # --------------------------------------------------------
    # 3) 外部 micro -> brain 预测
    # --------------------------------------------------------
    if bool(CONFIG.get('RUN_EXT_MICRO2BRAIN', False)):
        print('\n' + '=' * 70)
        print('[EXT] Running external micro2brain prediction...')
        print('=' * 70)

        for k in ['EXT_GUT_FEAT_PT', 'EXT_MICRO_ABUND_CSV']:
            if not os.path.exists(CONFIG[k]):
                raise FileNotFoundError(f"{k} 不存在：{CONFIG[k]}")

        models = load_all_fold_models(device)

        M_ext, gut_ext_sids, _ = load_gut_node_feat(CONFIG['EXT_GUT_FEAT_PT'])
        abund_ext_map = load_abundance_table(CONFIG['EXT_MICRO_ABUND_CSV'])
        M_ext, A_ext, sids_ext = intersect_gut_and_abund(M_ext, gut_ext_sids, abund_ext_map)

        y_pred_ext = external_micro2brain_predict(models, M_ext, A_ext, device, CONFIG['BATCH_SIZE'])
        save_bold_pred_csv(CONFIG['OUT_DIR'], sids_ext, y_pred_ext, CONFIG['EXT_MICRO2BRAIN_BOLD_PRED_CSV'])

        mean_pred = y_pred_ext.mean(dim=0)
        save_mean_roi_csv(CONFIG['OUT_DIR'], mean_pred, 'EXT_mean_pred_bold.csv')

        if os.path.exists(CONFIG['EXT_ROI_MEAN_CSV']):
            roi_ext_map = load_roi_mean_table(CONFIG['EXT_ROI_MEAN_CSV'])
            common = [sid for sid in sids_ext if sid in roi_ext_map]
            if len(common) > 0:
                idx_map = {sid: i for i, sid in enumerate(sids_ext)}
                Y_true_ext = torch.stack([roi_ext_map[sid] for sid in common], dim=0)
                Y_pred_common = torch.stack([y_pred_ext[idx_map[sid]] for sid in common], dim=0)

                roi_mae, roi_r_mean, roi_r_std, _ = compute_roi_metrics(Y_pred_common, Y_true_ext)
                with open(os.path.join(CONFIG['OUT_DIR'], CONFIG['EXT_MICRO2BRAIN_METRICS_TXT']), 'w', encoding='utf-8') as f:
                    f.write('[EXT micro2brain metrics]\n')
                    f.write(f'N_common={len(common)}\n')
                    f.write(f'ROI_MAE={roi_mae:.6f}\n')
                    f.write(f'ROI_r_mean={roi_r_mean:.6f}\n')
                    f.write(f'ROI_r_std={roi_r_std:.6f}\n')

                save_roi_scatter_with_csv(
                    Y_true_ext, Y_pred_common,
                    out_png=os.path.join(CONFIG['OUT_DIR'], 'EXT_micro2brain_roi_scatter.png'),
                    out_csv=os.path.join(CONFIG['OUT_DIR'], 'EXT_micro2brain_roi_scatter_data.csv'),
                    title_suffix='(EXT micro2brain)'
                )

        print('[EXT] external micro2brain finished.')


if __name__ == '__main__':
    main()

