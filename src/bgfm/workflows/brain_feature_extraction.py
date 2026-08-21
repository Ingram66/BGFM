import os
import glob
import random
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from configuration_brainlm import BrainLMConfig
from modeling_brainlm import BrainLMForPretraining


from bgfm.runtime import load_section, apply_globals, apply_mapping

# =========================
# 一、环境
# =========================
# CUDA device selection is controlled by the caller/environment.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# =========================
# 二、Repository defaults (override in YAML config)
# =========================

# ---- 新数据 BOLD 文件夹 ----
NEW_BOLD_DIR = r"data/paired/bold"

# ---- ROI 坐标文件 ----
COORDS_CSV = r"data/metadata/mni_coordinates.csv"

# ---- 微调后的最佳 checkpoint ----
BEST_CKPT = r"checkpoints/brain_encoder_best"   # 或者 stage1_finetune 的 best_model_checkpoint

# ---- 输出目录 ----
OUT_DIR = r"outputs/paired_brain_features"

# ---- 数据参数 ----
NUM_ROIS = 90
WINDOW_LEN = 100
PATCH_SIZE = 10
CSV_HAS_HEADER = True

# ---- 特征提取参数 ----
FEATURE_BATCH_SIZE = 16
FEATURE_NUM_WORKERS = 2
SEED = 42
USE_DETERMINISTIC = True
FIXED_WINDOW_FOR_FEATURES = True


apply_globals(globals(), load_section('brain_feature_extraction'))

FEATURE_PT = os.path.join(OUT_DIR, "brain_classification_features.pt")
SAMPLE_EMB_CSV = os.path.join(OUT_DIR, "sample_embeddings.csv")
ROI_EMB_CSV = os.path.join(OUT_DIR, "roi_embeddings_long.csv")


# =========================
# 三、通用函数
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
            except Exception:
                pass
            try:
                torch.backends.cudnn.allow_tf32 = False
            except Exception:
                pass
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def read_roi_meta(coords_csv: str, num_rois: int) -> pd.DataFrame:
    df = pd.read_csv(coords_csv)
    for c in ["X", "Y", "Z"]:
        if c not in df.columns:
            raise ValueError(f"MNI.csv must contain column '{c}'")
    if len(df) != num_rois:
        raise ValueError(f"MNI.csv rows={len(df)} != NUM_ROIS={num_rois}")
    if "ROIName" not in df.columns:
        df["ROIName"] = [f"ROI_{i:03d}" for i in range(num_rois)]
    return df[["ROIName", "X", "Y", "Z"]].copy()


# =========================
# 四、数据集
# =========================
class CsvFmriDataset(Dataset):
    """
    输入原始 CSV: [T_total, V]
    输出窗口: [V, T]
    """
    def __init__(
        self,
        csv_paths: List[str],
        coords_csv: str,
        window_len: int,
        num_rois: int,
        has_header: bool,
        seed: int,
        fixed_window: bool = False,
    ):
        self.csv_paths = csv_paths
        self.window_len = window_len
        self.num_rois = num_rois
        self.has_header = has_header
        self.rng = np.random.default_rng(seed)
        self.fixed_window = fixed_window
        self._fixed_start_cache: Dict[int, int] = {}

        coords = pd.read_csv(coords_csv)
        for c in ["X", "Y", "Z"]:
            if c not in coords.columns:
                raise ValueError(f"coords_csv must contain column '{c}'")
        if len(coords) != num_rois:
            raise ValueError(f"coords_csv must have {num_rois} rows, got {len(coords)}")

        self.xyz = torch.tensor(coords[["X", "Y", "Z"]].to_numpy(np.float32))

    def __len__(self):
        return len(self.csv_paths)

    def _read_csv(self, path: str) -> np.ndarray:
        df = pd.read_csv(path) if self.has_header else pd.read_csv(path, header=None)
        arr = df.to_numpy(np.float32)
        if arr.ndim != 2:
            raise ValueError(f"{path}: must be 2D, got {arr.ndim}D")
        if arr.shape[1] != self.num_rois:
            raise ValueError(f"{path}: cols={arr.shape[1]} expected {self.num_rois}")
        return arr

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.csv_paths[idx]
        mat = self._read_csv(path)   # [T_total, V]
        T_total = mat.shape[0]

        if T_total < self.window_len:
            raise ValueError(
                f"{os.path.basename(path)}: T_total={T_total} < window_len={self.window_len}"
            )

        if self.fixed_window:
            if idx in self._fixed_start_cache:
                start = self._fixed_start_cache[idx]
            else:
                start = int(self.rng.integers(0, T_total - self.window_len + 1))
                self._fixed_start_cache[idx] = start
        else:
            start = int(self.rng.integers(0, T_total - self.window_len + 1))

        window = mat[start:start + self.window_len, :]      # [T, V]
        window = torch.tensor(window, dtype=torch.float32).T  # [V, T]

        sample_id = os.path.splitext(os.path.basename(path))[0]

        return {
            "signal_vectors": window,
            "xyz_vectors": self.xyz.clone(),
            "path": path,
            "start": start,
            "sample_id": sample_id,
        }


def collate_fn_feature(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    signal_vectors = torch.stack([e["signal_vectors"] for e in examples], dim=0)  # [B,V,T]
    xyz_vectors = torch.stack([e["xyz_vectors"] for e in examples], dim=0)        # [B,V,3]

    return {
        "signal_vectors": signal_vectors,
        "xyz_vectors": xyz_vectors,
        "paths": [e["path"] for e in examples],
        "starts": [e["start"] for e in examples],
        "sample_ids": [e["sample_id"] for e in examples],
    }


# =========================
# 五、模型
# =========================
def build_brainlm_config(mask_ratio: float = 0.0) -> BrainLMConfig:
    config = BrainLMConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,

        decoder_hidden_size=512,
        decoder_num_hidden_layers=8,
        decoder_num_attention_heads=16,
        decoder_intermediate_size=2048,

        num_timepoints_per_voxel=WINDOW_LEN,
        timepoint_patching_size=PATCH_SIZE,
        mask_ratio=mask_ratio,
        loss_fn="mse",
        use_tanh_decoder=False,
    )
    config.num_brain_voxels = NUM_ROIS
    return config


def _torch_load_weights(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_checkpoint_state_dict(ckpt_path: str, device: str = "cpu"):
    if os.path.isdir(ckpt_path):
        bin_path = os.path.join(ckpt_path, "pytorch_model.bin")
        st_path = os.path.join(ckpt_path, "model.safetensors")

        if os.path.exists(st_path):
            from safetensors.torch import load_file
            return load_file(st_path, device=device)

        if os.path.exists(bin_path):
            return _torch_load_weights(bin_path, map_location=device)

        raise FileNotFoundError(f"No pytorch_model.bin or model.safetensors in: {ckpt_path}")

    return _torch_load_weights(ckpt_path, map_location=device)


def load_vit_only(model: torch.nn.Module, ckpt_path: str, device: str = "cpu"):
    """
    只加载 vit.* 权重，因为提特征只需要编码器部分
    """
    ckpt_sd = _load_checkpoint_state_dict(ckpt_path, device=device)
    model_sd = model.state_dict()

    loaded = {}
    skipped = []

    for k, v in ckpt_sd.items():
        if not k.startswith("vit."):
            continue
        if k not in model_sd:
            skipped.append((k, "not_in_model"))
            continue
        if tuple(v.shape) != tuple(model_sd[k].shape):
            skipped.append((k, "shape_mismatch", tuple(v.shape), tuple(model_sd[k].shape)))
            continue
        loaded[k] = v

    model_sd.update(loaded)
    model.load_state_dict(model_sd, strict=False)

    print("\n===== VIT-only Weight Loading Report =====")
    print(f"loaded vit keys: {len(loaded)}")
    print(f"skipped vit keys: {len(skipped)}")
    return model


# =========================
# 六、特征提取核心
# =========================
@torch.no_grad()
def forward_tokens(model: BrainLMForPretraining, signal: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
    emb, mask, ids_restore = model.vit.embeddings(
        signal_vectors=signal,
        xyz_vectors=xyz,
        do_masking=False,
    )
    out = model.vit.encoder(emb)
    if isinstance(out, (tuple, list)):
        h = out[0]
    else:
        h = getattr(out, "last_hidden_state", out)
    return h


@torch.no_grad()
def infer_token_layout_by_perturbation(
    model: BrainLMForPretraining,
    sample: dict,
    num_rois: int,
    window_len: int,
    patch_size: int,
    device: str,
    roi_to_perturb: int = 0,
    eps: float = 1e-3,
) -> str:
    n_patch = window_len // patch_size
    signal = sample["signal_vectors"].unsqueeze(0).to(device)
    xyz = sample["xyz_vectors"].unsqueeze(0).to(device)

    h0 = forward_tokens(model, signal, xyz)
    signal_pert = signal.clone()
    signal_pert[:, roi_to_perturb, :] += eps
    h1 = forward_tokens(model, signal_pert, xyz)

    dh = (h1 - h0).squeeze(0)
    dh = dh[1:]  # 去掉 cls token
    diff = torch.norm(dh, dim=1)

    expected = num_rois * n_patch
    if diff.numel() != expected:
        return "roi_major"

    score_roi_major = diff.view(num_rois, n_patch).sum(dim=1)
    arg1 = int(torch.argmax(score_roi_major).item())
    conc1 = float(score_roi_major.max().item() / (score_roi_major.mean().item() + 1e-8))

    score_patch_major = diff.view(n_patch, num_rois).sum(dim=0)
    arg2 = int(torch.argmax(score_patch_major).item())
    conc2 = float(score_patch_major.max().item() / (score_patch_major.mean().item() + 1e-8))

    if arg1 == roi_to_perturb and arg2 != roi_to_perturb:
        return "roi_major"
    if arg2 == roi_to_perturb and arg1 != roi_to_perturb:
        return "patch_major"
    return "roi_major" if conc1 >= conc2 else "patch_major"


@torch.no_grad()
def extract_batch_node_features(model: BrainLMForPretraining, batch: dict, layout: str, device: str) -> dict:
    signal = batch["signal_vectors"].to(device)
    xyz = batch["xyz_vectors"].to(device)

    h = forward_tokens(model, signal, xyz)   # [B, 1+tokens, hidden]
    n_patch = WINDOW_LEN // PATCH_SIZE

    patch_tokens = h[:, 1:, :]  # 去掉 cls token
    expected = NUM_ROIS * n_patch
    if patch_tokens.size(1) != expected:
        raise ValueError(f"patch_tokens={patch_tokens.size(1)} != expected={expected}")

    if layout == "roi_major":
        x = patch_tokens.view(patch_tokens.size(0), NUM_ROIS, n_patch, patch_tokens.size(-1))
        roi_emb = x.mean(dim=2)   # [B, NUM_ROIS, hidden]
    elif layout == "patch_major":
        x = patch_tokens.view(patch_tokens.size(0), n_patch, NUM_ROIS, patch_tokens.size(-1))
        roi_emb = x.mean(dim=1)   # [B, NUM_ROIS, hidden]
    else:
        raise ValueError("layout must be 'roi_major' or 'patch_major'")

    sample_emb = roi_emb.mean(dim=1)   # [B, hidden]
    return {
        "roi_emb": roi_emb,
        "sample_emb": sample_emb,
    }


# =========================
# 七、主提取函数
# =========================
def extract_features_from_new_data():
    ensure_dir(OUT_DIR)
    set_global_seed(SEED, deterministic=USE_DETERMINISTIC)
    device = get_device()

    if WINDOW_LEN % PATCH_SIZE != 0:
        raise ValueError(f"WINDOW_LEN({WINDOW_LEN}) must be divisible by PATCH_SIZE({PATCH_SIZE})")

    roi_meta = read_roi_meta(COORDS_CSV, NUM_ROIS)

    paths = sorted(glob.glob(os.path.join(NEW_BOLD_DIR, "*.csv")))
    if len(paths) == 0:
        raise ValueError(f"No CSV files found in: {NEW_BOLD_DIR}")

    print(f"[INFO] Found {len(paths)} csv files.")

    ds = CsvFmriDataset(
        csv_paths=paths,
        coords_csv=COORDS_CSV,
        window_len=WINDOW_LEN,
        num_rois=NUM_ROIS,
        has_header=CSV_HAS_HEADER,
        seed=SEED,
        fixed_window=FIXED_WINDOW_FOR_FEATURES,
    )

    loader = DataLoader(
        ds,
        batch_size=FEATURE_BATCH_SIZE,
        shuffle=False,
        num_workers=FEATURE_NUM_WORKERS,
        collate_fn=collate_fn_feature,
    )

    model = BrainLMForPretraining(build_brainlm_config(mask_ratio=0.0)).to(device)
    model.eval()
    model = load_vit_only(model, BEST_CKPT, device=device)

    if hasattr(model.vit.embeddings, "mask_ratio"):
        model.vit.embeddings.mask_ratio = 0.0

    layout = infer_token_layout_by_perturbation(
        model=model,
        sample=ds[0],
        num_rois=NUM_ROIS,
        window_len=WINDOW_LEN,
        patch_size=PATCH_SIZE,
        device=device,
        roi_to_perturb=0,
        eps=1e-3,
    )
    print("[INFO] TOKEN_LAYOUT =", layout)

    all_roi = []
    all_sample = []
    all_paths = []
    all_starts = []
    all_sample_ids = []

    for batch in loader:
        feats = extract_batch_node_features(model, batch, layout=layout, device=device)

        roi_emb = feats["roi_emb"].cpu()         # [B, NUM_ROIS, hidden]
        sample_emb = feats["sample_emb"].cpu()   # [B, hidden]

        all_roi.append(roi_emb)
        all_sample.append(sample_emb)
        all_paths.extend(batch["paths"])
        all_starts.extend(batch["starts"])
        all_sample_ids.extend(batch["sample_ids"])

    roi_emb = torch.cat(all_roi, dim=0)          # [N, NUM_ROIS, hidden]
    sample_emb = torch.cat(all_sample, dim=0)    # [N, hidden]

    # ---------- 保存 pt ----------
    save_obj = {
        "roi_emb": roi_emb,
        "sample_emb": sample_emb,
        "sample_ids": all_sample_ids,
        "paths": all_paths,
        "starts": all_starts,
        "roi_names": roi_meta["ROIName"].astype(str).tolist(),
        "ckpt": BEST_CKPT,
        "window_len": WINDOW_LEN,
        "patch_size": PATCH_SIZE,
        "num_rois": NUM_ROIS,
        "token_layout": layout,
    }
    torch.save(save_obj, FEATURE_PT)
    print(f"[SAVE] {FEATURE_PT}")

    # ---------- 保存 sample_embeddings.csv ----------
    sample_df = pd.DataFrame(
        sample_emb.numpy(),
        columns=[f"emb_{i:03d}" for i in range(sample_emb.shape[1])]
    )
    sample_df.insert(0, "start", all_starts)
    sample_df.insert(0, "path", all_paths)
    sample_df.insert(0, "sample_id", all_sample_ids)
    sample_df.to_csv(SAMPLE_EMB_CSV, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {SAMPLE_EMB_CSV}")

    # ---------- 保存 roi_embeddings_long.csv ----------
    roi_np = roi_emb.numpy()
    roi_rows = []

    for i, sid in enumerate(all_sample_ids):
        for roi_idx in range(NUM_ROIS):
            row = {
                "sample_id": sid,
                "path": all_paths[i],
                "start": all_starts[i],
                "roi_idx": roi_idx,
                "roi_name": roi_meta.loc[roi_idx, "ROIName"],
            }
            for k in range(roi_np.shape[2]):
                row[f"feat_{k:03d}"] = float(roi_np[i, roi_idx, k])
            roi_rows.append(row)

    roi_df = pd.DataFrame(roi_rows)
    roi_df.to_csv(ROI_EMB_CSV, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {ROI_EMB_CSV}")

    print("========== Feature Extraction Finished ==========")


if __name__ == "__main__":
    extract_features_from_new_data()