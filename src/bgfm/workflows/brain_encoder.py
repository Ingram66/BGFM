import os

# =========================
# 环境
# =========================
# CUDA device selection is controlled by the caller/environment.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import glob
import json
import random
import inspect
import warnings
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments
from transformers.trainer_callback import TrainerCallback

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.manifold import TSNE

from configuration_brainlm import BrainLMConfig
from brainlm_trainer import BrainLMTrainer
from modeling_brainlm import BrainLMForPretraining


from bgfm.runtime import load_section, apply_globals, apply_mapping

# ============================================================
# 一、路径与参数：你主要改这里
# ============================================================

# ---------- 必要路径 ----------
ALL_CSV_DIR = r"data/unimodal/brain/bold"
COORDS_CSV = r"data/metadata/mni_coordinates.csv"
LABELS_CSV = r"data/unimodal/brain/labels.csv"
PRETRAINED_CKPT = r"checkpoints/brainlm_pretrained"

# ---------- 总输出目录 ----------
PIPELINE_OUT_DIR = r"outputs/brain_encoder"

# ---------- 阶段开关 ----------
RUN_FINETUNE = True
RUN_FEATURE_EXTRACTION = True
RUN_ANALYSIS = True

# 如果你已经训练好，只想从已有 ckpt 开始提特征/分析，把这个改成实际路径
EXTERNAL_BEST_CKPT = None  # e.g. r"/path/to/checkpoint-1234"

# ---------- 数据划分 ----------
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42

# ---------- 模型与训练 ----------
NUM_ROIS = 90
WINDOW_LEN = 100
PATCH_SIZE = 10
MASK_RATIO = 0.2

LEARNING_RATE = 1e-4
NUM_EPOCHS = 200
TRAIN_BS = 4
EVAL_BS = 4

EVAL_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"
LOG_STRATEGY = "epoch"
SAVE_TOTAL_LIMIT = 3
CSV_HAS_HEADER = True
USE_DETERMINISTIC = True

FINETUNE_MODE = "emb_decoder"

# ---------- 重建导出 ----------
EXPORT_NUM_RECON_SAMPLES = None   # None = 导出验证集全部样本
PLOT_SIGNAL_SAMPLE_IDX = 0
PLOT_SIGNAL_ROI_LIST = [0, 1, 2]
PLOT_HEATMAP_SAMPLE_IDX = 0

# ---------- 特征导出 ----------
FEATURE_BATCH_SIZE = 16
FEATURE_NUM_WORKERS = 2
FIXED_WINDOW_FOR_FEATURES = True

# ---------- 下游分析 ----------
TSNE_PERPLEXITY = 30
ROI_IMPORTANCE_REPEATS = 10



# Repository/runtime overrides from configs/example.yaml (or BGFM_CONFIG).
apply_globals(globals(), load_section('brain_encoder'))

# ============================================================
# 二、目录
# ============================================================
FINETUNE_OUT_DIR = os.path.join(PIPELINE_OUT_DIR, "stage1_finetune")
FEATURE_OUT_DIR = os.path.join(PIPELINE_OUT_DIR, "stage2_features")
ANALYSIS_OUT_DIR = os.path.join(PIPELINE_OUT_DIR, "stage3_analysis")

FEATURE_PT = os.path.join(FEATURE_OUT_DIR, "brainlm_features.pt")
SAMPLE_EMB_CSV = os.path.join(FEATURE_OUT_DIR, "sample_embeddings.csv")
ROI_EMB_CSV = os.path.join(FEATURE_OUT_DIR, "roi_embeddings_long.csv")


# ============================================================
# 三、通用函数
# ============================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def disable_wandb_completely() -> None:
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_CONSOLE"] = "off"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


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


def load_labels_map(labels_csv: str) -> Dict[str, str]:
    df = pd.read_csv(labels_csv)
    if "sample_id" not in df.columns or "label" not in df.columns:
        raise ValueError("labels.csv must contain columns: sample_id,label")
    df["sample_id"] = df["sample_id"].astype(str)
    df["label"] = df["label"].astype(str)
    return dict(zip(df["sample_id"], df["label"]))


def build_labeled_path_table(csv_dir: str, labels_csv: str) -> pd.DataFrame:
    """
    建立样本表：
      sample_id | path | label
    """
    paths = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if len(paths) == 0:
        raise ValueError(f"No CSV files found in: {csv_dir}")

    label_map = load_labels_map(labels_csv)

    rows = []
    for p in paths:
        sample_id = os.path.splitext(os.path.basename(p))[0]
        if sample_id not in label_map:
            raise ValueError(f"sample_id={sample_id} not found in labels.csv")
        rows.append({
            "sample_id": sample_id,
            "path": p,
            "label": label_map[sample_id],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No matched samples between ALL_CSV_DIR and labels.csv")
    return df


def split_train_val_test_stratified(
    csv_dir: str,
    labels_csv: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    先分 test，再从剩余样本中分 val，保证 train/val/test 互斥且分层。
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    df = build_labeled_path_table(csv_dir, labels_csv)

    y = df["label"].astype(str).values
    idx_all = np.arange(len(df))

    # 1) 先分 test
    idx_trainval, idx_test = train_test_split(
        idx_all,
        test_size=test_ratio,
        random_state=seed,
        stratify=y,
    )

    df_trainval = df.iloc[idx_trainval].reset_index(drop=True)
    df_test = df.iloc[idx_test].reset_index(drop=True)

    # 2) 再从 trainval 中分 val
    y_trainval = df_trainval["label"].astype(str).values
    val_ratio_in_trainval = val_ratio / (train_ratio + val_ratio)

    idx_tv = np.arange(len(df_trainval))
    idx_train, idx_val = train_test_split(
        idx_tv,
        test_size=val_ratio_in_trainval,
        random_state=seed,
        stratify=y_trainval,
    )

    df_train = df_trainval.iloc[idx_train].reset_index(drop=True)
    df_val = df_trainval.iloc[idx_val].reset_index(drop=True)

    return df_train, df_val, df_test


def save_split_csvs(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, out_dir: str):
    df_train.to_csv(os.path.join(out_dir, "split_train.csv"), index=False, encoding="utf-8-sig")
    df_val.to_csv(os.path.join(out_dir, "split_val.csv"), index=False, encoding="utf-8-sig")
    df_test.to_csv(os.path.join(out_dir, "split_test.csv"), index=False, encoding="utf-8-sig")
    print("[SAVE] split_train.csv")
    print("[SAVE] split_val.csv")
    print("[SAVE] split_test.csv")


# ============================================================
# 四、数据集
# ============================================================
class CsvFmriDataset(Dataset):
    """
    原始 CSV: [T_total, V]
    返回: window [V, T]
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

    def __len__(self) -> int:
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
        mat = self._read_csv(path)
        T_total = mat.shape[0]
        if T_total < self.window_len:
            raise ValueError(f"{os.path.basename(path)}: T_total={T_total} < window_len={self.window_len}")

        if self.fixed_window:
            if idx in self._fixed_start_cache:
                start = self._fixed_start_cache[idx]
            else:
                start = int(self.rng.integers(0, T_total - self.window_len + 1))
                self._fixed_start_cache[idx] = start
        else:
            start = int(self.rng.integers(0, T_total - self.window_len + 1))

        window = mat[start:start + self.window_len, :]
        window = torch.tensor(window, dtype=torch.float32).T  # [V,T]

        sample_id = os.path.splitext(os.path.basename(path))[0]

        return {
            "signal_vectors": window,
            "xyz_vectors": self.xyz.clone(),
            "path": path,
            "start": start,
            "sample_id": sample_id,
        }


def collate_fn_train(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    训练/评估用，不带 metadata，避免 Trainer 把多余字段传给 model。
    """
    signal_vectors = torch.stack([e["signal_vectors"] for e in examples], dim=0)  # [B,V,T]
    xyz_vectors = torch.stack([e["xyz_vectors"] for e in examples], dim=0)        # [B,V,3]

    B, V, T = signal_vectors.shape
    P = PATCH_SIZE
    N = T // P
    gt = signal_vectors.reshape(B, V, N, P).flatten(1, 2)  # [B,seq,P]

    return {
        "signal_vectors": signal_vectors,
        "xyz_vectors": xyz_vectors,
        "input_ids": signal_vectors,
        "labels": gt,
    }


def collate_fn_feature(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    signal_vectors = torch.stack([e["signal_vectors"] for e in examples], dim=0)
    xyz_vectors = torch.stack([e["xyz_vectors"] for e in examples], dim=0)
    return {
        "signal_vectors": signal_vectors,
        "xyz_vectors": xyz_vectors,
        "paths": [e["path"] for e in examples],
        "starts": [e["start"] for e in examples],
        "sample_ids": [e["sample_id"] for e in examples],
    }


# ============================================================
# 五、模型与权重
# ============================================================
def build_brainlm_config(mask_ratio: float) -> BrainLMConfig:
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


def _load_checkpoint_state_dict(ckpt_path: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
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


def load_state_dict_with_report(model: torch.nn.Module, ckpt_path: str, device: str = "cpu"):
    ckpt_sd = _load_checkpoint_state_dict(ckpt_path, device=device)
    model_sd = model.state_dict()

    matched = {}
    shape_mismatch = []
    for k, v in ckpt_sd.items():
        if k in model_sd:
            if tuple(v.shape) == tuple(model_sd[k].shape):
                matched[k] = v
            else:
                shape_mismatch.append((k, tuple(v.shape), tuple(model_sd[k].shape)))

    missing_in_ckpt = [k for k in model_sd.keys() if k not in ckpt_sd]
    unexpected_in_ckpt = [k for k in ckpt_sd.keys() if k not in model_sd]

    model_sd.update(matched)
    model.load_state_dict(model_sd, strict=False)

    print("\n===== Pretrained Weight Loading Report =====")
    print(f"Checkpoint params:  {len(ckpt_sd)}")
    print(f"Model params:       {len(model_sd)}")
    print(f"Matched (loaded):   {len(matched)}")
    print(f"Shape mismatch:     {len(shape_mismatch)}")
    print(f"Missing in ckpt:    {len(missing_in_ckpt)}")
    print(f"Unexpected in ckpt: {len(unexpected_in_ckpt)}")


def load_vit_only(model: torch.nn.Module, ckpt_path: str, device: str = "cpu"):
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


def _set_requires_grad(module: torch.nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def apply_partial_finetune_emb_decoder(model: torch.nn.Module) -> None:
    _set_requires_grad(model, False)

    if hasattr(model, "vit") and hasattr(model.vit, "embeddings"):
        _set_requires_grad(model.vit.embeddings, True)

    if hasattr(model, "decoder"):
        _set_requires_grad(model.decoder, True)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[FT] mode=emb_decoder trainable={trainable:,} / total={total:,} ({trainable/total:.2%})")


def build_training_args(**kwargs) -> TrainingArguments:
    sig = inspect.signature(TrainingArguments.__init__).parameters

    if "eval_strategy" in kwargs:
        if "eval_strategy" in sig:
            pass
        elif "evaluation_strategy" in sig:
            kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
        else:
            kwargs.pop("eval_strategy", None)

    if "report_to" in kwargs and "report_to" not in sig:
        kwargs.pop("report_to", None)

    filtered = {k: v for k, v in kwargs.items() if k in sig}
    return TrainingArguments(**filtered)


# ============================================================
# 六、微调阶段指标与曲线
# ============================================================
def compute_masked_metrics(eval_pred):
    preds = eval_pred.predictions
    gt = eval_pred.label_ids

    logits, mask = preds
    pred = logits[0] if isinstance(logits, (tuple, list)) else logits

    pred = torch.as_tensor(pred, dtype=torch.float32)
    mask = torch.as_tensor(mask)
    gt = torch.as_tensor(gt, dtype=torch.float32)

    if pred.ndim == 4:
        pred = pred.flatten(1, 2)
    if mask.ndim == 3:
        mask = mask.flatten(1, 2)

    if pred.shape != gt.shape:
        raise ValueError(f"pred shape {tuple(pred.shape)} != gt shape {tuple(gt.shape)}")
    if mask.ndim != 2 or mask.shape[:2] != pred.shape[:2]:
        raise ValueError(f"mask shape {tuple(mask.shape)} incompatible with pred {tuple(pred.shape)}")

    m_patch = mask.to(dtype=pred.dtype).unsqueeze(-1)
    m_elem = m_patch.expand_as(pred)
    denom = m_elem.sum().clamp_min(1.0)

    diff = pred - gt
    masked_mse = (diff.pow(2) * m_elem).sum() / denom
    masked_mae = (diff.abs() * m_elem).sum() / denom
    masked_rmse = torch.sqrt(masked_mse)

    return {
        "masked_mse": float(masked_mse.detach().cpu()),
        "masked_rmse": float(masked_rmse.detach().cpu()),
        "masked_mae": float(masked_mae.detach().cpu()),
    }


class EpochCurveCallback(TrainerCallback):
    def __init__(self, train_eval_dataset: Dataset, out_dir: str):
        self.train_eval_dataset = train_eval_dataset
        self.out_dir = out_dir
        self.trainer: Optional[BrainLMTrainer] = None
        self.records: List[Dict[str, Any]] = []
        self._internal_eval = False

    def set_trainer(self, trainer: BrainLMTrainer) -> None:
        self.trainer = trainer

    def _epoch_int(self, state) -> Optional[int]:
        if state.epoch is None:
            return None
        return int(round(float(state.epoch)))

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self._internal_eval or self.trainer is None:
            return control

        epoch = self._epoch_int(state)
        eval_metrics = metrics or {}

        try:
            self._internal_eval = True
            train_metrics = self.trainer.evaluate(
                eval_dataset=self.train_eval_dataset,
                metric_key_prefix="train",
            )
        finally:
            self._internal_eval = False

        row = {"epoch": epoch}
        row.update(eval_metrics)
        row.update(train_metrics)

        self.records.append(row)
        df = pd.DataFrame(self.records).sort_values("epoch")
        df.to_csv(os.path.join(self.out_dir, "metrics_by_epoch.csv"), index=False, encoding="utf-8-sig")
        print("[SAVE] metrics_by_epoch.csv")
        return control


def plot_curves_from_csv(out_dir: str) -> None:
    csv_path = os.path.join(out_dir, "metrics_by_epoch.csv")
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path).sort_values("epoch")

    def _plot(train_col, eval_col, ylab, title, out_name):
        plt.figure()
        if train_col in df.columns:
            plt.plot(df["epoch"], df[train_col], label=train_col)
        if eval_col in df.columns:
            plt.plot(df["epoch"], df[eval_col], label=eval_col)
        plt.xlabel("epoch")
        plt.ylabel(ylab)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, out_name), dpi=200)
        plt.close()

    _plot("train_loss", "eval_loss", "loss", "Loss by Epoch", "curve_loss_by_epoch.png")
    _plot("train_masked_mse", "eval_masked_mse", "masked_mse", "Masked MSE by Epoch", "curve_mse_by_epoch.png")
    _plot("train_masked_rmse", "eval_masked_rmse", "masked_rmse", "Masked RMSE by Epoch", "curve_rmse_by_epoch.png")
    _plot("train_masked_mae", "eval_masked_mae", "masked_mae", "Masked MAE by Epoch", "curve_mae_by_epoch.png")


def _normalize_predict_arrays(predictions, label_ids):
    if not isinstance(predictions, (tuple, list)) or len(predictions) < 2:
        raise ValueError("trainer.predict(...).predictions 不是 (logits, mask) 结构，请检查 BrainLMTrainer 输出。")

    logits, mask = predictions
    pred = logits[0] if isinstance(logits, (tuple, list)) else logits

    pred = np.asarray(pred, dtype=np.float32)
    mask = np.asarray(mask)
    gt = np.asarray(label_ids, dtype=np.float32)

    if pred.ndim == 4:
        pred = pred.reshape(pred.shape[0], -1, pred.shape[-1])
    if mask.ndim == 3:
        mask = mask.reshape(mask.shape[0], -1)

    if pred.shape != gt.shape:
        raise ValueError(f"pred shape {pred.shape} != gt shape {gt.shape}")
    if mask.ndim != 2 or mask.shape[:2] != pred.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} incompatible with pred {pred.shape}")

    return pred, mask, gt


def export_reconstruction_artifacts(trainer, dataset, out_dir, num_examples=None):
    pred_output = trainer.predict(dataset)
    pred_patch, mask_patch, gt_patch = _normalize_predict_arrays(pred_output.predictions, pred_output.label_ids)

    V = NUM_ROIS
    T = WINDOW_LEN
    P = PATCH_SIZE
    Np = T // P

    n_samples = len(dataset) if num_examples is None else min(len(dataset), num_examples)

    long_rows = []
    roi_rows = []

    for i in range(n_samples):
        item = dataset[i]
        sample_id = item["sample_id"]
        path = item["path"]
        start = item["start"]
        gt_signal = item["signal_vectors"].numpy()  # [V,T]

        pred_vt = pred_patch[i].reshape(V, Np, P).reshape(V, T)
        mask_vn = mask_patch[i].reshape(V, Np)

        recon_fill = gt_signal.copy()
        for roi in range(V):
            for patch_idx in range(Np):
                if int(mask_vn[roi, patch_idx]) == 1:
                    s = patch_idx * P
                    e = s + P
                    recon_fill[roi, s:e] = pred_vt[roi, s:e]

        for roi in range(V):
            roi_masked_elem = 0
            roi_sq_sum = 0.0
            roi_abs_sum = 0.0

            for t in range(T):
                patch_idx = t // P
                is_masked = int(mask_vn[roi, patch_idx])
                gt_val = float(gt_signal[roi, t])
                pred_val = float(pred_vt[roi, t])
                recon_val = float(recon_fill[roi, t])
                abs_err = abs(pred_val - gt_val)
                sq_err = (pred_val - gt_val) ** 2

                if is_masked == 1:
                    roi_masked_elem += 1
                    roi_sq_sum += sq_err
                    roi_abs_sum += abs_err

                long_rows.append({
                    "sample_idx": i,
                    "sample_id": sample_id,
                    "path": path,
                    "start": int(start),
                    "roi_idx": roi,
                    "time_idx": t,
                    "patch_idx": patch_idx,
                    "is_masked_patch": is_masked,
                    "gt_signal": gt_val,
                    "pred_signal_full": pred_val,
                    "recon_signal_masked_fill": recon_val,
                    "abs_error": float(abs_err),
                    "sq_error": float(sq_err),
                })

            roi_rows.append({
                "sample_idx": i,
                "sample_id": sample_id,
                "path": path,
                "start": int(start),
                "roi_idx": roi,
                "roi_mse_masked_only": float(roi_sq_sum / max(roi_masked_elem, 1)),
                "roi_mae_masked_only": float(roi_abs_sum / max(roi_masked_elem, 1)),
                "num_masked_elements": int(roi_masked_elem),
            })

    long_df = pd.DataFrame(long_rows)
    roi_df = pd.DataFrame(roi_rows)
    summary_df = roi_df.groupby("roi_idx", as_index=False)[["roi_mse_masked_only", "roi_mae_masked_only"]].mean()

    long_df.to_csv(os.path.join(out_dir, "reconstruction_long.csv"), index=False, encoding="utf-8-sig")
    roi_df.to_csv(os.path.join(out_dir, "roi_recon_metrics.csv"), index=False, encoding="utf-8-sig")
    summary_df.to_csv(os.path.join(out_dir, "roi_error_summary.csv"), index=False, encoding="utf-8-sig")

    print("[SAVE] reconstruction_long.csv")
    print("[SAVE] roi_recon_metrics.csv")
    print("[SAVE] roi_error_summary.csv")


def plot_signal_vs_recon_from_csv(out_dir, sample_idx=0, roi_idx=0):
    df = pd.read_csv(os.path.join(out_dir, "reconstruction_long.csv"))
    sub = df[(df["sample_idx"] == sample_idx) & (df["roi_idx"] == roi_idx)].sort_values("time_idx")

    plt.figure(figsize=(10, 4))
    plt.plot(sub["time_idx"], sub["gt_signal"], label="ground_truth")
    plt.plot(sub["time_idx"], sub["recon_signal_masked_fill"], label="reconstruction_fill")

    masked = sub["is_masked_patch"].values.astype(bool)
    if masked.any():
        plt.scatter(sub.loc[masked, "time_idx"], sub.loc[masked, "gt_signal"], s=12, label="masked_points")

    plt.xlabel("time")
    plt.ylabel("signal")
    plt.title(f"Sample {sample_idx} ROI {roi_idx}: Ground Truth vs Reconstruction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"signal_vs_recon_sample{sample_idx}_roi{roi_idx}.png"), dpi=200)
    plt.close()


def plot_recon_error_heatmap(out_dir, sample_idx=0, value_col="abs_error"):
    df = pd.read_csv(os.path.join(out_dir, "reconstruction_long.csv"))
    sub = df[df["sample_idx"] == sample_idx].sort_values(["roi_idx", "time_idx"])
    mat = sub.pivot(index="roi_idx", columns="time_idx", values=value_col).values

    plt.figure(figsize=(10, 6))
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label=value_col)
    plt.xlabel("time")
    plt.ylabel("ROI")
    plt.title(f"Reconstruction Error Heatmap ({value_col}) Sample {sample_idx}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"recon_heatmap_{value_col}_sample{sample_idx}.png"), dpi=200)
    plt.close()


def plot_roi_error_bar(out_dir, metric="roi_mse_masked_only"):
    df = pd.read_csv(os.path.join(out_dir, "roi_error_summary.csv")).sort_values(metric, ascending=False)

    plt.figure(figsize=(12, 4))
    plt.bar(df["roi_idx"].astype(str), df[metric])
    plt.xlabel("ROI")
    plt.ylabel(metric)
    plt.title(f"Mean {metric} across ROIs")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{metric}_roi_bar.png"), dpi=200)
    plt.close()


# ============================================================
# 七、特征提取
# ============================================================
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
    dh = dh[1:]
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

    h = forward_tokens(model, signal, xyz)
    n_patch = WINDOW_LEN // PATCH_SIZE

    patch_tokens = h[:, 1:, :]
    expected = NUM_ROIS * n_patch
    if patch_tokens.size(1) != expected:
        raise ValueError(f"patch_tokens={patch_tokens.size(1)} != expected={expected}")

    if layout == "roi_major":
        x = patch_tokens.view(patch_tokens.size(0), NUM_ROIS, n_patch, patch_tokens.size(-1))
        roi_emb = x.mean(dim=2)
    elif layout == "patch_major":
        x = patch_tokens.view(patch_tokens.size(0), n_patch, NUM_ROIS, patch_tokens.size(-1))
        roi_emb = x.mean(dim=1)
    else:
        raise ValueError("layout must be 'roi_major' or 'patch_major'")

    sample_emb = roi_emb.mean(dim=1)
    return {"roi_emb": roi_emb, "sample_emb": sample_emb}


# ============================================================
# 八、下游分析
# ============================================================
def compute_auc(y_true_int, prob, n_classes):
    if n_classes == 2:
        return float(roc_auc_score(y_true_int, prob[:, 1]))
    y_bin = label_binarize(y_true_int, classes=list(range(n_classes)))
    return float(roc_auc_score(y_bin, prob, average="macro", multi_class="ovr"))


def build_classifier():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=5000, random_state=SEED, multi_class="auto"))
    ])


def plot_confusion_matrix_png(cm, class_names, out_png):
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_roc_binary(y_true_int, prob, out_png, out_csv):
    fpr, tpr, thresholds = roc_curve(y_true_int, prob[:, 1])
    auc_val = roc_auc_score(y_true_int, prob[:, 1])

    roc_df = pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "threshold": thresholds,
        "auc": auc_val,
    })
    roc_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC={auc_val:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def permutation_importance_by_roi(
    clf,
    roi_emb_val,
    y_val_int,
    n_classes,
    repeats=10,
):
    base_prob = clf.predict_proba(roi_emb_val.reshape(roi_emb_val.shape[0], -1))
    if n_classes == 2:
        base_score = roc_auc_score(y_val_int, base_prob[:, 1])
    else:
        y_bin = label_binarize(y_val_int, classes=list(range(n_classes)))
        base_score = roc_auc_score(y_bin, base_prob, average="macro", multi_class="ovr")

    n_samples, n_rois, feat_dim = roi_emb_val.shape
    rng = np.random.default_rng(SEED)
    scores = []

    for roi_idx in range(n_rois):
        drops = []
        for _ in range(repeats):
            roi_copy = roi_emb_val.copy()
            perm = rng.permutation(n_samples)
            roi_copy[:, roi_idx, :] = roi_copy[perm, roi_idx, :]

            prob_perm = clf.predict_proba(roi_copy.reshape(n_samples, n_rois * feat_dim))
            if n_classes == 2:
                score_perm = roc_auc_score(y_val_int, prob_perm[:, 1])
            else:
                y_bin = label_binarize(y_val_int, classes=list(range(n_classes)))
                score_perm = roc_auc_score(y_bin, prob_perm, average="macro", multi_class="ovr")

            drops.append(base_score - score_perm)

        scores.append(float(np.mean(drops)))

    return np.array(scores, dtype=np.float32), float(base_score)


def plot_roi_importance_bar(df, out_png, top_k=20):
    d = df.sort_values("importance_score", ascending=False).head(top_k)

    plt.figure(figsize=(10, 5))
    plt.bar(d["roi_name"], d["importance_score"])
    plt.xticks(rotation=90)
    plt.ylabel("Importance (AUC drop)")
    plt.title(f"Top {top_k} ROI Importance")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_roi_brain2d(df, out_png, plane="xy"):
    if plane == "xy":
        x = df["X"].values
        y = df["Y"].values
        xlabel, ylabel = "X", "Y"
        label_y_col = "Y"
    elif plane == "xz":
        x = df["X"].values
        y = df["Z"].values
        xlabel, ylabel = "X", "Z"
        label_y_col = "Z"
    else:
        raise ValueError("plane must be 'xy' or 'xz'")

    s = 60 + 400 * (df["importance_score"].values - df["importance_score"].min()) / (
        df["importance_score"].max() - df["importance_score"].min() + 1e-8
    )

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(x, y, s=s, c=df["importance_score"].values)
    plt.colorbar(sc, label="Importance")
    for _, row in df.sort_values("importance_score", ascending=False).head(10).iterrows():
        plt.text(row["X"], row[label_y_col], row["roi_name"], fontsize=8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"ROI Importance Brain Map ({plane.upper()})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ============================================================
# 九、阶段函数
# ============================================================
def stage1_finetune() -> str:
    ensure_dir(FINETUNE_OUT_DIR)
    disable_wandb_completely()
    set_global_seed(SEED, deterministic=USE_DETERMINISTIC)

    if WINDOW_LEN % PATCH_SIZE != 0:
        raise ValueError(f"WINDOW_LEN({WINDOW_LEN}) must be divisible by PATCH_SIZE({PATCH_SIZE})")

    # ===== 先一次性分好 train / val / test =====
    df_train, df_val, df_test = split_train_val_test_stratified(
        csv_dir=ALL_CSV_DIR,
        labels_csv=LABELS_CSV,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
    )

    print(f"[SPLIT] train={len(df_train)} val={len(df_val)} test={len(df_test)}")
    save_split_csvs(df_train, df_val, df_test, FINETUNE_OUT_DIR)

    train_paths = df_train["path"].tolist()
    val_paths = df_val["path"].tolist()

    # 微调阶段只用 train + val，test 完全不碰
    train_ds = CsvFmriDataset(
        train_paths, COORDS_CSV, WINDOW_LEN, NUM_ROIS, CSV_HAS_HEADER,
        seed=SEED, fixed_window=False
    )
    val_ds = CsvFmriDataset(
        val_paths, COORDS_CSV, WINDOW_LEN, NUM_ROIS, CSV_HAS_HEADER,
        seed=SEED + 1, fixed_window=True
    )
    train_eval_ds = CsvFmriDataset(
        train_paths, COORDS_CSV, WINDOW_LEN, NUM_ROIS, CSV_HAS_HEADER,
        seed=SEED + 2, fixed_window=True
    )

    config = build_brainlm_config(mask_ratio=MASK_RATIO)
    model = BrainLMForPretraining(config)
    load_state_dict_with_report(model, PRETRAINED_CKPT, device="cpu")

    if FINETUNE_MODE == "emb_decoder":
        apply_partial_finetune_emb_decoder(model)
    else:
        raise ValueError("Set FINETUNE_MODE='emb_decoder'.")

    training_args = build_training_args(
        output_dir=FINETUNE_OUT_DIR,
        remove_unused_columns=False,
        do_train=True,
        do_eval=True,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BS,
        per_device_eval_batch_size=EVAL_BS,
        num_train_epochs=NUM_EPOCHS,
        seed=SEED,

        eval_strategy=EVAL_STRATEGY,
        save_strategy=SAVE_STRATEGY,
        logging_strategy=LOG_STRATEGY,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to=[],
    )

    curve_cb = EpochCurveCallback(train_eval_dataset=train_eval_ds, out_dir=FINETUNE_OUT_DIR)

    trainer = BrainLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn_train,
        compute_metrics=compute_masked_metrics,
        callbacks=[curve_cb],
    )
    curve_cb.set_trainer(trainer)

    trainer.train()
    plot_curves_from_csv(FINETUNE_OUT_DIR)

    metrics = trainer.evaluate(eval_dataset=val_ds)
    best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
    best_metric = getattr(trainer.state, "best_metric", None)
    best_step = getattr(trainer.state, "best_step", None)

    best_row = {
        "best_model_checkpoint": best_ckpt,
        "best_metric(eval_loss)": best_metric,
        "best_step": best_step,
        "eval_loss": metrics.get("eval_loss"),
        "eval_masked_mse": metrics.get("eval_masked_mse"),
        "eval_masked_rmse": metrics.get("eval_masked_rmse"),
        "eval_masked_mae": metrics.get("eval_masked_mae"),
        "global_step": trainer.state.global_step,
        "epoch": float(trainer.state.epoch) if trainer.state.epoch is not None else None,
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "window_len": WINDOW_LEN,
        "patch_size": PATCH_SIZE,
        "mask_ratio": MASK_RATIO,
        "finetune_mode": FINETUNE_MODE,
        "trainable_note": "freeze vit.encoder; train vit.embeddings + decoder",
        "note": "Test set is held out and NOT used during finetuning.",
    }
    pd.DataFrame([best_row]).to_csv(
        os.path.join(FINETUNE_OUT_DIR, "best_metrics.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    print("[SAVE] best_metrics.csv")

    # 这里只导出 val 的重建结果，不碰 test
    export_reconstruction_artifacts(
        trainer=trainer,
        dataset=val_ds,
        out_dir=FINETUNE_OUT_DIR,
        num_examples=EXPORT_NUM_RECON_SAMPLES,
    )

    for roi_idx in PLOT_SIGNAL_ROI_LIST:
        plot_signal_vs_recon_from_csv(FINETUNE_OUT_DIR, sample_idx=PLOT_SIGNAL_SAMPLE_IDX, roi_idx=roi_idx)

    plot_recon_error_heatmap(FINETUNE_OUT_DIR, sample_idx=PLOT_HEATMAP_SAMPLE_IDX, value_col="abs_error")
    plot_recon_error_heatmap(FINETUNE_OUT_DIR, sample_idx=PLOT_HEATMAP_SAMPLE_IDX, value_col="sq_error")
    plot_roi_error_bar(FINETUNE_OUT_DIR, metric="roi_mse_masked_only")
    plot_roi_error_bar(FINETUNE_OUT_DIR, metric="roi_mae_masked_only")

    if best_ckpt is None:
        raise RuntimeError("训练结束后没有拿到 best_model_checkpoint。")
    return best_ckpt


def stage2_extract_features(best_ckpt: str):
    ensure_dir(FEATURE_OUT_DIR)
    device = get_device()
    set_global_seed(SEED, deterministic=USE_DETERMINISTIC)

    label_map = load_labels_map(LABELS_CSV)
    roi_meta = read_roi_meta(COORDS_CSV, NUM_ROIS)

    split_train_csv = os.path.join(FINETUNE_OUT_DIR, "split_train.csv")
    split_val_csv = os.path.join(FINETUNE_OUT_DIR, "split_val.csv")
    split_test_csv = os.path.join(FINETUNE_OUT_DIR, "split_test.csv")

    if not (os.path.exists(split_train_csv) and os.path.exists(split_val_csv) and os.path.exists(split_test_csv)):
        raise FileNotFoundError("找不到 split_train.csv / split_val.csv / split_test.csv，请先运行 stage1_finetune()")

    df_train = pd.read_csv(split_train_csv)
    df_val = pd.read_csv(split_val_csv)
    df_test = pd.read_csv(split_test_csv)

    # 按 split 顺序拼接，后面分析直接使用 split 列
    df_all = pd.concat([
        df_train.assign(split="train"),
        df_val.assign(split="val"),
        df_test.assign(split="test"),
    ], ignore_index=True)

    paths = df_all["path"].tolist()
    split_map = dict(zip(df_all["sample_id"].astype(str), df_all["split"].astype(str)))

    ds = CsvFmriDataset(
        paths, COORDS_CSV, WINDOW_LEN, NUM_ROIS, CSV_HAS_HEADER,
        seed=SEED, fixed_window=FIXED_WINDOW_FOR_FEATURES
    )
    loader = DataLoader(
        ds,
        batch_size=FEATURE_BATCH_SIZE,
        shuffle=False,
        num_workers=FEATURE_NUM_WORKERS,
        collate_fn=collate_fn_feature
    )

    model = BrainLMForPretraining(build_brainlm_config(mask_ratio=0.0)).to(device)
    model.eval()
    model = load_vit_only(model, best_ckpt, device=device)
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
    all_labels = []
    all_splits = []

    for batch in loader:
        feats = extract_batch_node_features(model, batch, layout=layout, device=device)
        roi = feats["roi_emb"].cpu()
        sample_emb = feats["sample_emb"].cpu()

        batch_sample_ids = batch["sample_ids"]

        all_roi.append(roi)
        all_sample.append(sample_emb)
        all_paths.extend(batch["paths"])
        all_starts.extend(batch["starts"])
        all_sample_ids.extend(batch_sample_ids)
        all_labels.extend([label_map[sid] for sid in batch_sample_ids])
        all_splits.extend([split_map[sid] for sid in batch_sample_ids])

    roi_emb = torch.cat(all_roi, dim=0)
    sample_emb = torch.cat(all_sample, dim=0)

    save_obj = {
        "roi_emb": roi_emb,
        "sample_emb": sample_emb,
        "sample_ids": all_sample_ids,
        "labels": all_labels,
        "splits": all_splits,
        "paths": all_paths,
        "starts": all_starts,
        "roi_names": roi_meta["ROIName"].astype(str).tolist(),
        "ckpt": best_ckpt,
        "window_len": WINDOW_LEN,
        "patch_size": PATCH_SIZE,
        "num_rois": NUM_ROIS,
        "token_layout": layout,
    }
    torch.save(save_obj, FEATURE_PT)
    print("[SAVE] brainlm_features.pt")

    sample_df = pd.DataFrame(sample_emb.numpy(), columns=[f"emb_{i:03d}" for i in range(sample_emb.shape[1])])
    sample_df.insert(0, "split", all_splits)
    sample_df.insert(0, "label", all_labels)
    sample_df.insert(0, "sample_id", all_sample_ids)
    sample_df.to_csv(SAMPLE_EMB_CSV, index=False, encoding="utf-8-sig")
    print("[SAVE] sample_embeddings.csv")

    roi_np = roi_emb.numpy()
    roi_rows = []
    for i, sid in enumerate(all_sample_ids):
        for roi_idx in range(NUM_ROIS):
            row = {
                "sample_id": sid,
                "label": all_labels[i],
                "split": all_splits[i],
                "roi_idx": roi_idx,
                "roi_name": roi_meta.loc[roi_idx, "ROIName"],
            }
            for k in range(roi_np.shape[2]):
                row[f"feat_{k:03d}"] = float(roi_np[i, roi_idx, k])
            roi_rows.append(row)

    roi_df = pd.DataFrame(roi_rows)
    roi_df.to_csv(ROI_EMB_CSV, index=False, encoding="utf-8-sig")
    print("[SAVE] roi_embeddings_long.csv")


def stage3_analyze():
    ensure_dir(ANALYSIS_OUT_DIR)
    set_global_seed(SEED, deterministic=USE_DETERMINISTIC)

    obj = torch.load(FEATURE_PT, map_location="cpu")
    roi_emb = obj["roi_emb"].numpy()          # [N,90,768]
    sample_emb = obj["sample_emb"].numpy()    # [N,768]
    sample_ids = np.array(obj["sample_ids"])
    labels_raw = np.array(obj["labels"])
    splits = np.array(obj["splits"])
    roi_names = obj["roi_names"]

    roi_meta = read_roi_meta(COORDS_CSV, roi_emb.shape[1])

    # ---------- t-SNE ----------
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=TSNE_PERPLEXITY)
    Z = tsne.fit_transform(sample_emb)

    tsne_df = pd.DataFrame({
        "sample_id": sample_ids,
        "label": labels_raw,
        "split": splits,
        "tsne_1": Z[:, 0],
        "tsne_2": Z[:, 1],
    })
    tsne_df.to_csv(os.path.join(ANALYSIS_OUT_DIR, "tsne_coordinates.csv"), index=False, encoding="utf-8-sig")

    plt.figure(figsize=(6, 5))
    for lab in sorted(pd.unique(labels_raw)):
        sub = tsne_df[tsne_df["label"] == lab]
        plt.scatter(sub["tsne_1"], sub["tsne_2"], s=25, label=str(lab))
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("t-SNE of Sample Embeddings")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_OUT_DIR, "tsne_plot.png"), dpi=200)
    plt.close()

    # ---------- 分类：只用 train+val 训练，只在 test 评估 ----------
    le = LabelEncoder()
    y_all = le.fit_transform(labels_raw)
    class_names = list(le.classes_)
    n_classes = len(class_names)

    trainval_mask = np.isin(splits, ["train", "val"])
    test_mask = (splits == "test")

    X_trainval = sample_emb[trainval_mask]
    y_trainval = y_all[trainval_mask]
    sid_trainval = sample_ids[trainval_mask]

    X_test = sample_emb[test_mask]
    y_test = y_all[test_mask]
    sid_test = sample_ids[test_mask]

    clf = build_classifier()
    clf.fit(X_trainval, y_trainval)

    y_pred = clf.predict(X_test)
    prob = clf.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    auc_val = compute_auc(y_test, prob, n_classes)

    metrics_df = pd.DataFrame([{
        "classifier": "LogisticRegression(Standardized)",
        "seed": SEED,
        "trainval_samples": int(trainval_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "accuracy_test": acc,
        "macro_f1_test": f1,
        "macro_precision_test": prec,
        "macro_recall_test": rec,
        "auc_test": auc_val,
        "n_classes": n_classes,
        "note": "Classifier trained on train+val features, evaluated on held-out test only.",
    }])
    metrics_df.to_csv(os.path.join(ANALYSIS_OUT_DIR, "classification_metrics.csv"), index=False, encoding="utf-8-sig")

    pred_df = pd.DataFrame({
        "sample_id": sid_test,
        "split": "test",
        "true_label": [class_names[i] for i in y_test],
        "pred_label": [class_names[i] for i in y_pred],
    })
    for j, cls in enumerate(class_names):
        pred_df[f"prob_{cls}"] = prob[:, j]
    pred_df.to_csv(os.path.join(ANALYSIS_OUT_DIR, "classification_predictions.csv"), index=False, encoding="utf-8-sig")

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in class_names],
        columns=[f"pred_{c}" for c in class_names]
    )
    cm_df.to_csv(os.path.join(ANALYSIS_OUT_DIR, "confusion_matrix.csv"), encoding="utf-8-sig")
    plot_confusion_matrix_png(cm, class_names, os.path.join(ANALYSIS_OUT_DIR, "confusion_matrix.png"))

    if n_classes == 2:
        plot_roc_binary(
            y_true_int=y_test,
            prob=prob,
            out_png=os.path.join(ANALYSIS_OUT_DIR, "roc_curve.png"),
            out_csv=os.path.join(ANALYSIS_OUT_DIR, "roc_curve.csv"),
        )

    # ---------- ROI importance：train+val 训练，只在 test 上算贡献 ----------
    roi_emb_trainval = roi_emb[trainval_mask]
    roi_emb_test = roi_emb[test_mask]

    clf_roi = build_classifier()
    clf_roi.fit(roi_emb_trainval.reshape(roi_emb_trainval.shape[0], -1), y_trainval)

    imp_scores, base_score = permutation_importance_by_roi(
        clf=clf_roi,
        roi_emb_val=roi_emb_test,
        y_val_int=y_test,
        n_classes=n_classes,
        repeats=ROI_IMPORTANCE_REPEATS,
    )

    roi_df = pd.DataFrame({
        "roi_idx": np.arange(roi_emb.shape[1]),
        "roi_name": roi_names,
        "importance_score": imp_scores,
    })
    roi_df = pd.concat([roi_df, roi_meta[["X", "Y", "Z"]]], axis=1)
    roi_df["importance_rank"] = roi_df["importance_score"].rank(ascending=False, method="min").astype(int)
    roi_df["base_test_auc"] = base_score
    roi_df.to_csv(os.path.join(ANALYSIS_OUT_DIR, "roi_importance.csv"), index=False, encoding="utf-8-sig")

    plot_roi_importance_bar(roi_df, os.path.join(ANALYSIS_OUT_DIR, "roi_importance_bar.png"), top_k=20)
    plot_roi_brain2d(roi_df, os.path.join(ANALYSIS_OUT_DIR, "roi_importance_brain2d_xy.png"), plane="xy")
    plot_roi_brain2d(roi_df, os.path.join(ANALYSIS_OUT_DIR, "roi_importance_brain2d_xz.png"), plane="xz")

    with open(os.path.join(ANALYSIS_OUT_DIR, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({int(i): c for i, c in enumerate(class_names)}, f, ensure_ascii=False, indent=2)

    print("[SAVE] classification_metrics.csv")
    print("[SAVE] classification_predictions.csv")
    print("[SAVE] confusion_matrix.csv")
    print("[SAVE] tsne_coordinates.csv")
    print("[SAVE] roi_importance.csv")


# ============================================================
# 十、主函数
# ============================================================
def main():
    warnings.filterwarnings("ignore", message=r".*weights_only=False.*", category=FutureWarning)

    ensure_dir(PIPELINE_OUT_DIR)
    ensure_dir(FINETUNE_OUT_DIR)
    ensure_dir(FEATURE_OUT_DIR)
    ensure_dir(ANALYSIS_OUT_DIR)

    if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-8:
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0")

    print("========== BrainLM Full Pipeline (Strict Split) ==========")
    print("ALL_CSV_DIR =", ALL_CSV_DIR)
    print("COORDS_CSV  =", COORDS_CSV)
    print("LABELS_CSV  =", LABELS_CSV)
    print("OUT_DIR     =", PIPELINE_OUT_DIR)

    best_ckpt = EXTERNAL_BEST_CKPT

    if RUN_FINETUNE:
        best_ckpt = stage1_finetune()
    else:
        if best_ckpt is None:
            best_csv = os.path.join(FINETUNE_OUT_DIR, "best_metrics.csv")
            if os.path.exists(best_csv):
                best_ckpt = pd.read_csv(best_csv).loc[0, "best_model_checkpoint"]
            else:
                raise ValueError("RUN_FINETUNE=False 时，需要提供 EXTERNAL_BEST_CKPT 或已有 stage1 的 best_metrics.csv。")

    print("[INFO] best_ckpt =", best_ckpt)

    if RUN_FEATURE_EXTRACTION:
        stage2_extract_features(best_ckpt=best_ckpt)

    if RUN_ANALYSIS:
        if not os.path.exists(FEATURE_PT):
            raise FileNotFoundError(f"找不到特征文件: {FEATURE_PT}")
        stage3_analyze()

    print("========== Pipeline Finished ==========")


if __name__ == "__main__":
    main()