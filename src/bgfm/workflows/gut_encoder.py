import os

# ===== 强制只用一张卡：改成你要用的 GPU 编号 =====
# CUDA device selection is controlled by the caller/environment.
from bgfm.runtime import load_section, apply_globals, apply_mapping

# ===================================================
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import sys
import json
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from pickle import load, dump

from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

from transformers import GPT2ForSequenceClassification, Trainer, TrainingArguments


# =========================================================
# 你需要改这里：路径 + 超参数
# =========================================================
PROJECT_DIR = r"third_party/mgm"

# 用来反序列化拿 tokenizer（不使用 corpus 内的 input_ids）
CORPUS_PKL = r"third_party/mgm/corpus.pkl"

# 丰度矩阵（必须）
# 约定：第一列 sample_id，其余 642 列为 taxa（列名是 taxa 名）
ABUND_CSV = r"data/unimodal/gut/abundance.csv"

# 标签（必须）
# 约定：index 为 sample_id，第一列为标签
LABELS_CSV = r"data/unimodal/gut/labels.csv"

# 预训练模型目录
PRETRAINED_MODEL_DIR = r"checkpoints/mgm_pretrained"

# 输出目录
OUT_DIR = r"outputs/gut_encoder"

SEED = 42
LR = 1e-5
EPOCHS = 500
BATCH_SIZE = 32
WARMUP_STEPS = 0
WEIGHT_DECAY = 0.01

N_TAXA = 642

# 8:1:1
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# 0 bin + K 个非零分箱 => 总 bin = K+1
N_BINS_NONZERO = 31  # => 总 bin=32（<bin_0>.. <bin_31>）
TARGET_MAX_LEN = N_TAXA + 2  # [BOS] + 642 + [EOS] = 644

WEIGHT_ALPHA = 0.1

apply_globals(globals(), load_section('gut_encoder'))

PLOT_DIR = os.path.join(OUT_DIR, "plots")
ANALYSIS_DIR = os.path.join(OUT_DIR, "analysis")
# =========================================================


# ---------------------------------------------------------
# MGM tokenizer compatibility. The upstream MicroCorpus.py is not vendored here.
# PROJECT_DIR can point to a local checkout in third_party/mgm.
# ---------------------------------------------------------
if PROJECT_DIR:
    sys.path.insert(0, PROJECT_DIR)
try:
    from MicroCorpus import MicroCorpus, MicroTokenizer  # noqa: F401
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "MicroCorpus.py is required for MGM corpus deserialization. "
        "Set gut_encoder.project_dir in the YAML config to the upstream MGM code directory."
    ) from exc


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SequenceClassificationDataset(Dataset):
    """返回 CPU tensor；Trainer 内部会自动搬到 GPU。"""
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor):
        assert input_ids.device.type == "cpu"
        assert attention_mask.device.type == "cpu"
        assert labels.device.type == "cpu"
        self.input_ids = input_ids.long()
        self.attention_mask = attention_mask.long()
        self.labels = labels.long()

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def load_tokenizer_from_corpus(corpus_pkl_path: str):
    corpus = load(open(corpus_pkl_path, "rb"))
    tokenizer = getattr(corpus, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("corpus.pkl 中找不到 corpus.tokenizer；请确认 corpus 是 MicroCorpus 对象并包含 tokenizer。")
    return tokenizer


def read_abundance_table(abund_csv: str):
    """
    读取丰度表：
    - 第一列 sample_id
    - 后续 642 列 taxa（列名是 taxa 名）
    """
    df = pd.read_csv(abund_csv)
    if df.shape[1] < 1 + N_TAXA:
        raise ValueError(f"ABUND_CSV cols={df.shape[1]} < 1+{N_TAXA}")
    sample_ids = df.iloc[:, 0].astype(str).tolist()
    taxa_names = list(df.columns[1:1 + N_TAXA])
    X = df.iloc[:, 1:1 + N_TAXA].to_numpy(np.float32)
    return sample_ids, taxa_names, X


def load_and_align_labels(labels_csv: str, sample_ids):
    """
    labels_csv: index=sample_id，第一列为标签
    """
    df = pd.read_csv(labels_csv, index_col=0)
    if df.shape[1] < 1:
        raise ValueError("labels.csv 至少需要 1 列标签（第一列）。")
    y_raw = df.iloc[:, 0].astype(str)

    idx = df.index.astype(str)
    missing = set(sample_ids) - set(idx)
    if len(missing) > 0:
        raise ValueError(f"labels.csv 缺少 {len(missing)} 个样本标签，例如：{list(sorted(missing))[:5]}")
    y_raw = y_raw.loc[pd.Index(sample_ids)]

    le = LabelEncoder()
    y = le.fit_transform(y_raw.values)
    return torch.tensor(y, dtype=torch.long).cpu(), le, y_raw.values


def compute_metrics_multiclass(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    precision, recall, _, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    auc_val = np.nan
    try:
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        n_cls = probs.shape[1]
        if n_cls == 2:
            if len(np.unique(labels)) == 2:
                auc_val = float(roc_auc_score(labels, probs[:, 1]))
        else:
            y_bin = label_binarize(labels, classes=list(range(n_cls)))
            valid_cols = [i for i in range(y_bin.shape[1]) if len(np.unique(y_bin[:, i])) == 2]
            if len(valid_cols) > 0:
                auc_val = float(
                    roc_auc_score(
                        y_bin[:, valid_cols],
                        probs[:, valid_cols],
                        average="macro",
                        multi_class="ovr",
                    )
                )
    except Exception:
        pass

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "auc_macro_or_binary": float(auc_val) if not np.isnan(auc_val) else float("nan"),
    }


def get_vocab_dict(tokenizer):
    if hasattr(tokenizer, "vocab") and isinstance(tokenizer.vocab, dict):
        return tokenizer.vocab
    if hasattr(tokenizer, "get_vocab"):
        v = tokenizer.get_vocab()
        if isinstance(v, dict):
            return v
    raise ValueError("tokenizer 无法提供 vocab dict（缺少 vocab 或 get_vocab）。")


def ensure_special_tokens(tokenizer):
    vocab = get_vocab_dict(tokenizer)

    def add_if_missing(tok: str):
        if tok in vocab:
            return int(vocab[tok])
        tid = len(vocab)
        vocab[tok] = tid
        return tid

    pad_id = getattr(tokenizer, "pad_token_id", None)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)

    if pad_id is None:
        pad_id = add_if_missing("<pad>")
    else:
        pad_id = int(pad_id)

    if bos_id is None:
        bos_id = add_if_missing("<bos>")
    else:
        bos_id = int(bos_id)

    if eos_id is None:
        eos_id = add_if_missing("<eos>")
    else:
        eos_id = int(eos_id)

    try:
        tokenizer.pad_token_id = pad_id
        tokenizer.bos_token_id = bos_id
        tokenizer.eos_token_id = eos_id
    except Exception:
        pass

    return pad_id, bos_id, eos_id


def build_bin_token_ids(tokenizer, n_bins_nonzero: int):
    vocab = get_vocab_dict(tokenizer)

    def add(tok: str):
        if tok in vocab:
            return int(vocab[tok])
        tid = len(vocab)
        vocab[tok] = tid
        return tid

    bin_token_ids = []
    for b in range(0, 1 + int(n_bins_nonzero)):
        bin_token_ids.append(add(f"<bin_{b}>"))
    return bin_token_ids


def make_quantile_edges_nonzero(X_train: np.ndarray, n_bins_nonzero: int):
    nz = X_train[X_train > 0]
    if nz.size == 0:
        return np.array([0.0] * (n_bins_nonzero + 1), dtype=np.float32)

    qs = np.linspace(0.0, 1.0, n_bins_nonzero + 1)
    edges = np.quantile(nz, qs).astype(np.float32)

    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.float32(np.inf))
    return edges


def abundance_to_bin_index(x: np.ndarray, edges: np.ndarray):
    K = edges.shape[0] - 1
    bins = np.zeros_like(x, dtype=np.int64)
    nz_mask = x > 0
    if nz_mask.any():
        inner = edges[1:-1]
        b = np.digitize(x[nz_mask], inner, right=False)
        bins[nz_mask] = b.astype(np.int64) + 1
    if bins.max() > K:
        raise RuntimeError(f"bin overflow: max_bin={bins.max()} > K={K}")
    return bins


def build_fixed_layout_inputs(X: np.ndarray, bin_token_ids: list[int], bos_id: int, eos_id: int, pad_id: int, edges: np.ndarray):
    """
    固定布局输入：
      [BOS] + 642 个 bin token（位置=taxa索引） + [EOS]
    """
    N, V = X.shape
    assert V == N_TAXA

    bins = abundance_to_bin_index(X, edges)
    bin_token_ids_arr = np.array(bin_token_ids, dtype=np.int64)
    bin_ids = np.take(bin_token_ids_arr, bins)

    L = N_TAXA + 2
    input_ids = np.full((N, L), pad_id, dtype=np.int64)
    attention_mask = np.ones((N, L), dtype=np.int64)

    input_ids[:, 0] = bos_id
    input_ids[:, 1:1 + N_TAXA] = bin_ids
    input_ids[:, 1 + N_TAXA] = eos_id

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        bins
    )


def sanity_check_fixed_layout(sample_ids, taxa_names, X, input_ids: torch.Tensor, bins: np.ndarray, bin_token_ids: list[int], bos_id: int, eos_id: int):
    N, L = input_ids.shape
    assert L == N_TAXA + 2

    if not torch.all(input_ids[:, 0] == int(bos_id)):
        raise ValueError("BOS 校验失败：input_ids[:,0] 不是全 BOS")
    if not torch.all(input_ids[:, -1] == int(eos_id)):
        raise ValueError("EOS 校验失败：input_ids[:,-1] 不是全 EOS")

    toks_taxa = input_ids[:, 1:1 + N_TAXA].cpu().numpy()
    bin0_id = int(bin_token_ids[0])
    ok0 = (toks_taxa[X == 0] == bin0_id).all()
    print(f"[CHECK] fixed layout ok: length={L}, BOS/EOS ok, zero->bin0_token ok? {bool(ok0)}")

    for i in range(min(2, N)):
        print(f"\n[CHECK] sample {i} id={sample_ids[i]}")
        for j in range(min(8, N_TAXA)):
            print(f"  taxa[{j:03d}] {taxa_names[j]} | abund={float(X[i,j]):.6g} | bin={int(bins[i,j])} | token_id={int(toks_taxa[i,j])}")

    print(f"[CHECK] global zero ratio={float((X==0).mean()):.4f}")


def expand_gpt2_position_embeddings(model, new_max_len: int):
    old_len = int(model.transformer.wpe.weight.shape[0])
    if new_max_len <= old_len:
        return model, old_len

    device = model.transformer.wpe.weight.device
    dtype = model.transformer.wpe.weight.dtype
    old_wpe = model.transformer.wpe.weight.data
    n_embd = int(old_wpe.shape[1])

    new_wpe = torch.zeros((new_max_len, n_embd), device=device, dtype=dtype)
    new_wpe[:old_len] = old_wpe

    mean = old_wpe.mean(dim=0, keepdim=True)
    std = old_wpe.std(dim=0, keepdim=True) + 1e-6
    new_wpe[old_len:] = mean + torch.randn((new_max_len - old_len, n_embd), device=device, dtype=dtype) * std

    model.transformer.wpe = torch.nn.Embedding(new_max_len, n_embd).to(device)
    model.transformer.wpe.weight.data = new_wpe

    if hasattr(model.config, "n_positions"):
        model.config.n_positions = new_max_len
    if hasattr(model.config, "max_position_embeddings"):
        model.config.max_position_embeddings = new_max_len

    return model, old_len


def set_trainable_layers(model: GPT2ForSequenceClassification):
    """
    只训练：score head + 最后一层 transformer block + token/pos embedding
    """
    for p in model.parameters():
        p.requires_grad = False

    for p in model.score.parameters():
        p.requires_grad = True

    for p in model.transformer.h[-1].parameters():
        p.requires_grad = True

    for p in model.transformer.wte.parameters():
        p.requires_grad = True

    for p in model.transformer.wpe.parameters():
        p.requires_grad = True


class WeightedTrainEvalSamplerTrainer(Trainer):
    """
    train 用 WeightedRandomSampler(replacement=True)
    eval 仅当 eval_dataset 长度与 eval_sample_weights 一致时才使用加权采样，
    否则回退到默认顺序评估，避免 test_set 误用 val 的权重。
    """
    def __init__(self, *args, train_sample_weights=None, eval_sample_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_sample_weights = train_sample_weights
        self.eval_sample_weights = eval_sample_weights

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer requires a train_dataset for training.")
        if self.train_sample_weights is None:
            return super().get_train_dataloader()

        sampler = WeightedRandomSampler(
            weights=self.train_sample_weights,
            num_samples=len(self.train_sample_weights),
            replacement=True
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer requires an eval_dataset for evaluation.")

        if (
            self.eval_sample_weights is None
            or len(self.eval_sample_weights) != len(eval_dataset)
        ):
            return super().get_eval_dataloader(eval_dataset)

        sampler = WeightedRandomSampler(
            weights=self.eval_sample_weights,
            num_samples=len(self.eval_sample_weights),
            replacement=True
        )
        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=False,
        )


def save_training_curves(trainer: Trainer, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(trainer.state.log_history)
    csv_path = os.path.join(out_dir, "trainer_log_history.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print("[SAVE]", csv_path)

    if df.empty or "epoch" not in df.columns:
        print("[WARN] log_history 为空或没有 epoch 字段，跳过训练曲线。")
        return

    df_ep = df.dropna(subset=["epoch"]).copy()
    df_ep["epoch_int"] = df_ep["epoch"].astype(float).astype(int)

    def _group_last_if_exists(col):
        if col is None or col not in df_ep.columns:
            return None
        d = df_ep.dropna(subset=[col]).copy()
        if d.empty:
            return None
        return d.groupby("epoch_int")[col].last()

    plot_items = [
        ("loss", "eval_loss", "Loss", "loss_curve.png"),
        (None, "eval_accuracy", "Accuracy", "accuracy_curve.png"),
        (None, "eval_macro_f1", "Macro F1", "macro_f1_curve.png"),
        (None, "eval_auc_macro_or_binary", "AUC", "auc_curve.png"),
    ]

    for train_col, eval_col, ylabel, fname in plot_items:
        s1 = _group_last_if_exists(train_col)
        s2 = _group_last_if_exists(eval_col)

        if s1 is None and s2 is None:
            print(f"[WARN] {ylabel} 没有可绘制数据，跳过 {fname}")
            continue

        plt.figure()
        if s1 is not None:
            plt.plot(s1.index.values, s1.values, label=train_col)
        if s2 is not None:
            plt.plot(s2.index.values, s2.values, label=eval_col)

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} per Epoch")
        plt.legend()
        plt.tight_layout()
        out_png = os.path.join(out_dir, fname)
        plt.savefig(out_png, dpi=200)
        plt.close()
        print("[SAVE]", out_png)


@torch.no_grad()
def export_eval_artifacts(
    trainer: Trainer,
    model: GPT2ForSequenceClassification,
    test_set,
    test_idx,
    sample_ids,
    label_encoder,
    taxa_names,
    out_dir: str,
    n_taxa: int,
):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    # ===== 1) 逐样本预测 =====
    pred_output = trainer.predict(test_set)
    logits = pred_output.predictions
    y_true = pred_output.label_ids.astype(int)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    y_pred = np.argmax(logits, axis=-1)
    n_cls = probs.shape[1]

    test_sample_ids = [sample_ids[i] for i in test_idx]
    true_names = label_encoder.inverse_transform(y_true)
    pred_names = label_encoder.inverse_transform(y_pred)

    pred_df = pd.DataFrame({
        "sample_id": test_sample_ids,
        "true_label": y_true,
        "true_label_name": true_names,
        "pred_label": y_pred,
        "pred_label_name": pred_names,
    })
    for c in range(n_cls):
        pred_df[f"prob_class_{c}"] = probs[:, c]

    pred_csv = os.path.join(out_dir, "test_predictions.csv")
    pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")
    print("[SAVE]", pred_csv)

    # ===== 2) 混淆矩阵 =====
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_cls)))
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{name}" for name in label_encoder.classes_],
        columns=[f"pred_{name}" for name in label_encoder.classes_],
    )
    cm_csv = os.path.join(out_dir, "confusion_matrix.csv")
    cm_df.to_csv(cm_csv, encoding="utf-8-sig")
    print("[SAVE]", cm_csv)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, aspect="auto")
    plt.colorbar()
    plt.xticks(range(n_cls), label_encoder.classes_, rotation=45)
    plt.yticks(range(n_cls), label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_png = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(cm_png, dpi=200)
    plt.close()
    print("[SAVE]", cm_png)

    # ===== 3) ROC =====
    if n_cls == 2:
        if len(np.unique(y_true)) == 2:
            fpr, tpr, thresholds = roc_curve(y_true, probs[:, 1])
            roc_df = pd.DataFrame({
                "fpr": fpr,
                "tpr": tpr,
                "threshold": thresholds,
            })
            roc_csv = os.path.join(out_dir, "roc_curve_binary.csv")
            roc_df.to_csv(roc_csv, index=False, encoding="utf-8-sig")
            print("[SAVE]", roc_csv)

            auc_val = roc_auc_score(y_true, probs[:, 1])

            plt.figure(figsize=(5, 5))
            plt.plot(fpr, tpr, label=f"AUC={auc_val:.4f}")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title("ROC Curve")
            plt.legend()
            plt.tight_layout()
            roc_png = os.path.join(out_dir, "roc_curve_binary.png")
            plt.savefig(roc_png, dpi=200)
            plt.close()
            print("[SAVE]", roc_png)
        else:
            print("[WARN] test 集只有一个类别，跳过二分类 ROC。")
    else:
        roc_rows = []
        y_bin = label_binarize(y_true, classes=list(range(n_cls)))
        valid_class_count = 0

        plt.figure(figsize=(6, 5))
        for c in range(n_cls):
            if len(np.unique(y_bin[:, c])) < 2:
                print(f"[WARN] 类别 {label_encoder.classes_[c]} 在 test 中缺少正/负样本，跳过该类 ROC。")
                continue

            fpr, tpr, thresholds = roc_curve(y_bin[:, c], probs[:, c])
            auc_val = roc_auc_score(y_bin[:, c], probs[:, c])
            plt.plot(fpr, tpr, label=f"{label_encoder.classes_[c]} AUC={auc_val:.4f}")

            tmp = pd.DataFrame({
                "class_id": c,
                "class_name": label_encoder.classes_[c],
                "fpr": fpr,
                "tpr": tpr,
                "threshold": thresholds,
            })
            roc_rows.append(tmp)
            valid_class_count += 1

        if valid_class_count > 0:
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title("ROC Curve (OvR)")
            plt.legend()
            plt.tight_layout()
            roc_png = os.path.join(out_dir, "roc_curve_multiclass.png")
            plt.savefig(roc_png, dpi=200)
            plt.close()
            print("[SAVE]", roc_png)

            roc_df = pd.concat(roc_rows, ignore_index=True)
            roc_csv = os.path.join(out_dir, "roc_curve_multiclass.csv")
            roc_df.to_csv(roc_csv, index=False, encoding="utf-8-sig")
            print("[SAVE]", roc_csv)
        else:
            plt.close()
            print("[WARN] 多分类 test 集中没有可绘制的 ROC 类别。")

    # ===== 4) 提取 embedding 做 t-SNE =====
    test_loader = DataLoader(
        test_set,
        batch_size=trainer.args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=trainer.args.dataloader_num_workers,
        pin_memory=trainer.args.dataloader_pin_memory,
        drop_last=False,
    )

    sample_emb_list = []

    for batch in test_loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]       # [B, L, H]
        taxa_hidden = last_hidden[:, 1:1 + n_taxa, :] # [B, n_taxa, H]
        sample_emb = taxa_hidden.mean(dim=1)          # [B, H]
        sample_emb_list.append(sample_emb.detach().cpu())

    sample_emb = torch.cat(sample_emb_list, dim=0).numpy()

    emb_df = pd.DataFrame(sample_emb, columns=[f"emb_{i:03d}" for i in range(sample_emb.shape[1])])
    emb_df.insert(0, "pred_label_name", pred_names)
    emb_df.insert(0, "true_label_name", true_names)
    emb_df.insert(0, "sample_id", test_sample_ids)
    emb_csv = os.path.join(out_dir, "test_sample_embeddings.csv")
    emb_df.to_csv(emb_csv, index=False, encoding="utf-8-sig")
    print("[SAVE]", emb_csv)

    if len(test_sample_ids) >= 2:
        perplexity = min(30, max(2, len(test_sample_ids) // 3))
        perplexity = min(perplexity, len(test_sample_ids) - 1)

        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        z = tsne.fit_transform(sample_emb)

        tsne_df = pd.DataFrame({
            "sample_id": test_sample_ids,
            "true_label": y_true,
            "true_label_name": true_names,
            "pred_label": y_pred,
            "pred_label_name": pred_names,
            "tsne_1": z[:, 0],
            "tsne_2": z[:, 1],
        })
        tsne_csv = os.path.join(out_dir, "tsne_coordinates.csv")
        tsne_df.to_csv(tsne_csv, index=False, encoding="utf-8-sig")
        print("[SAVE]", tsne_csv)

        plt.figure(figsize=(6, 5))
        for cls_name in np.unique(true_names):
            sub = tsne_df[tsne_df["true_label_name"] == cls_name]
            plt.scatter(sub["tsne_1"], sub["tsne_2"], s=25, label=str(cls_name))
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title("t-SNE of Test Embeddings")
        plt.legend()
        plt.tight_layout()
        tsne_png = os.path.join(out_dir, "tsne_plot.png")
        plt.savefig(tsne_png, dpi=200)
        plt.close()
        print("[SAVE]", tsne_png)
    else:
        print("[WARN] test 样本数不足，跳过 t-SNE。")

    # ===== 5) 微生物重要性：permutation importance =====
    if n_cls == 2:
        if len(np.unique(y_true)) == 2:
            base_metric = roc_auc_score(y_true, probs[:, 1])
            metric_name = "auc_drop"
        else:
            base_metric = accuracy_score(y_true, y_pred)
            metric_name = "acc_drop"
            print("[WARN] 二分类 test 只有一个类别，importance 改用 accuracy drop。")
    else:
        base_metric = accuracy_score(y_true, y_pred)
        metric_name = "acc_drop"

    test_input_ids = torch.stack([test_set[i]["input_ids"] for i in range(len(test_set))], dim=0).clone()
    test_attention_mask = torch.stack([test_set[i]["attention_mask"] for i in range(len(test_set))], dim=0).clone()
    test_labels = torch.stack([test_set[i]["labels"] for i in range(len(test_set))], dim=0).clone()

    rng = np.random.default_rng(42)
    imp_rows = []

    for taxa_idx in range(n_taxa):
        pos = 1 + taxa_idx
        perm_ids = test_input_ids.clone()
        perm = rng.permutation(len(test_set))
        perm_ids[:, pos] = perm_ids[perm, pos]

        tmp_ds = SequenceClassificationDataset(
            perm_ids.cpu(),
            test_attention_mask.cpu(),
            test_labels.cpu(),
        )
        tmp_out = trainer.predict(tmp_ds)
        tmp_logits = tmp_out.predictions
        tmp_probs = torch.softmax(torch.tensor(tmp_logits), dim=-1).numpy()
        tmp_pred = np.argmax(tmp_logits, axis=-1)

        if metric_name == "auc_drop":
            score = roc_auc_score(y_true, tmp_probs[:, 1])
        else:
            score = accuracy_score(y_true, tmp_pred)

        drop = float(base_metric - score)
        imp_rows.append({
            "taxa_idx": taxa_idx,
            "taxa_name": taxa_names[taxa_idx],
            f"importance_{metric_name}": drop,
        })

    imp_df = pd.DataFrame(imp_rows).sort_values(f"importance_{metric_name}", ascending=False).reset_index(drop=True)
    imp_df["importance_rank"] = np.arange(1, len(imp_df) + 1)
    imp_csv = os.path.join(out_dir, "microbiome_importance.csv")
    imp_df.to_csv(imp_csv, index=False, encoding="utf-8-sig")
    print("[SAVE]", imp_csv)

    top_df = imp_df.head(20)
    plt.figure(figsize=(10, 5))
    plt.bar(top_df["taxa_name"], top_df[f"importance_{metric_name}"])
    plt.xticks(rotation=90)
    plt.ylabel(metric_name)
    plt.title("Top 20 Microbiome Importance")
    plt.tight_layout()
    imp_png = os.path.join(out_dir, "microbiome_importance_top20.png")
    plt.savefig(imp_png, dpi=200)
    plt.close()
    print("[SAVE]", imp_png)


def main():
    seed_everything(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-8:
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO 必须等于 1.0")

    # 1) 读丰度表
    sample_ids, taxa_names, X = read_abundance_table(ABUND_CSV)
    print(f"[DATA] X: N={len(sample_ids)}, shape={X.shape}, taxa={len(taxa_names)}")

    # 2) 标签对齐
    labels, label_encoder, y_raw = load_and_align_labels(LABELS_CSV, sample_ids)
    num_labels = int(labels.max().item() + 1)
    print(f"[DATA] num_labels={num_labels}, classes={list(label_encoder.classes_)}")

    # 3) tokenizer
    tokenizer = load_tokenizer_from_corpus(CORPUS_PKL)
    vocab = get_vocab_dict(tokenizer)
    print(f"[INFO] tokenizer vocab size (before)={len(vocab)}")

    pad_id, bos_id, eos_id = ensure_special_tokens(tokenizer)
    bin_token_ids = build_bin_token_ids(tokenizer, n_bins_nonzero=N_BINS_NONZERO)

    vocab2 = get_vocab_dict(tokenizer)
    print(f"[INFO] pad/bos/eos={pad_id}/{bos_id}/{eos_id}")
    print(f"[INFO] bin tokens count={len(bin_token_ids)} (bin0..bin{N_BINS_NONZERO})")
    print(f"[INFO] tokenizer vocab size (after)={len(vocab2)}")

    # 4) 8:1:1 分层 train/val/test
    all_idx = np.arange(len(sample_ids))
    y_np = labels.numpy()

    train_idx, temp_idx = train_test_split(
        all_idx,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=SEED,
        stratify=y_np
    )

    y_temp = y_np[temp_idx]
    val_ratio_in_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_ratio_in_temp),
        random_state=SEED,
        stratify=y_temp
    )

    print(f"[SPLIT] train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError("train/val/test 至少有一个集合为空，请检查样本量和划分比例。")

    split_train_df = pd.DataFrame({
        "sample_id": [sample_ids[i] for i in train_idx],
        "label": [int(y_np[i]) for i in train_idx],
        "label_name": [y_raw[i] for i in train_idx],
        "split": "train",
    })
    split_val_df = pd.DataFrame({
        "sample_id": [sample_ids[i] for i in val_idx],
        "label": [int(y_np[i]) for i in val_idx],
        "label_name": [y_raw[i] for i in val_idx],
        "split": "val",
    })
    split_test_df = pd.DataFrame({
        "sample_id": [sample_ids[i] for i in test_idx],
        "label": [int(y_np[i]) for i in test_idx],
        "label_name": [y_raw[i] for i in test_idx],
        "split": "test",
    })

    split_train_df.to_csv(os.path.join(OUT_DIR, "split_train.csv"), index=False, encoding="utf-8-sig")
    split_val_df.to_csv(os.path.join(OUT_DIR, "split_val.csv"), index=False, encoding="utf-8-sig")
    split_test_df.to_csv(os.path.join(OUT_DIR, "split_test.csv"), index=False, encoding="utf-8-sig")
    print("[SAVE] split_train.csv")
    print("[SAVE] split_val.csv")
    print("[SAVE] split_test.csv")

    # 5) 仅用训练集非零值计算 edges
    edges = make_quantile_edges_nonzero(X[train_idx], n_bins_nonzero=N_BINS_NONZERO)
    print(f"[BINS] edges len={len(edges)} min={edges[0]:.6g} max={edges[-1]:.6g}")

    # 6) 构造固定布局输入（全量）
    input_ids, attention_mask, bins = build_fixed_layout_inputs(
        X=X,
        bin_token_ids=bin_token_ids,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,
        edges=edges,
    )
    print(f"[TOK] input_ids={tuple(input_ids.shape)} attention_mask={tuple(attention_mask.shape)} (expect N x 644)")

    # 7) 强校验
    sanity_check_fixed_layout(sample_ids, taxa_names, X, input_ids, bins, bin_token_ids, bos_id, eos_id)

    # 8) dataset / subset
    dataset = SequenceClassificationDataset(input_ids.cpu(), attention_mask.cpu(), labels.cpu())
    train_set = Subset(dataset, train_idx.tolist())
    val_set = Subset(dataset, val_idx.tolist())
    test_set = Subset(dataset, test_idx.tolist())

    # 9) 类别加权采样权重（train/val）
    y_train = y_np[train_idx]
    y_val = y_np[val_idx]

    train_counts = np.bincount(y_train, minlength=num_labels)
    val_counts = np.bincount(y_val, minlength=num_labels)

    train_class_weights = 1.0 / (np.power(train_counts, WEIGHT_ALPHA) + 1e-12)
    val_class_weights = 1.0 / (np.power(val_counts, WEIGHT_ALPHA) + 1e-12)

    train_sample_weights = torch.tensor(train_class_weights[y_train], dtype=torch.double)
    eval_sample_weights = torch.tensor(val_class_weights[y_val], dtype=torch.double)

    print(f"[WEIGHT] train_counts={train_counts.tolist()} class_w={train_class_weights.tolist()}")
    print(f"[WEIGHT] val_counts={val_counts.tolist()} class_w={val_class_weights.tolist()}")

    # 10) 模型加载 + resize vocab + pos embedding
    model = GPT2ForSequenceClassification.from_pretrained(PRETRAINED_MODEL_DIR, num_labels=num_labels)

    new_vocab_size = len(get_vocab_dict(tokenizer))
    model.resize_token_embeddings(int(new_vocab_size))
    model.config.pad_token_id = int(pad_id)
    model.config.eos_token_id = int(eos_id)
    if hasattr(model.config, "bos_token_id"):
        model.config.bos_token_id = int(bos_id)

    model, old_pos = expand_gpt2_position_embeddings(model, TARGET_MAX_LEN)
    print(f"[INFO] position embeddings: {old_pos} -> {getattr(model.config,'n_positions', old_pos)}")

    max_tid = int(input_ids.max().item())
    embed_size = int(model.transformer.wte.num_embeddings)
    if max_tid >= embed_size:
        raise ValueError(f"token id 越界：max_token_id={max_tid} >= embedding_size={embed_size}")

    # 11) 冻结/解冻
    set_trainable_layers(model)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable params: {trainable}/{total} ({100.0*trainable/total:.2f}%)")

    # 12) TrainingArguments
    args_common = dict(
        output_dir=os.path.join(OUT_DIR, "checkpoints"),
        logging_dir=os.path.join(OUT_DIR, "logs"),
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=3,
        logging_strategy="epoch",
        report_to="none",
        dataloader_pin_memory=False,
    )
    training_args = TrainingArguments(**args_common)

    # 13) Trainer
    trainer = WeightedTrainEvalSamplerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        compute_metrics=compute_metrics_multiclass,
        train_sample_weights=train_sample_weights,
        eval_sample_weights=eval_sample_weights,
    )

    print("[INFO] Start fine-tuning (fixed 642-position bins)...")
    trainer.train()

    save_training_curves(trainer, PLOT_DIR)

    # 验证集：允许按 val 权重评估
    val_metrics = trainer.evaluate(eval_dataset=val_set)

    # 测试集：直接 predict 后手动算，避免任何 eval sampler 干扰
    test_pred = trainer.predict(test_set)
    test_logits = test_pred.predictions
    test_y = test_pred.label_ids.astype(int)
    test_probs = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()
    test_pred_cls = np.argmax(test_logits, axis=-1)

    test_metrics = {
        "test_accuracy": float(accuracy_score(test_y, test_pred_cls)),
        "test_macro_f1": float(f1_score(test_y, test_pred_cls, average="macro")),
    }

    try:
        if test_probs.shape[1] == 2 and len(np.unique(test_y)) == 2:
            test_metrics["test_auc_macro_or_binary"] = float(roc_auc_score(test_y, test_probs[:, 1]))
        elif test_probs.shape[1] > 2:
            y_bin = label_binarize(test_y, classes=list(range(test_probs.shape[1])))
            valid_cols = [i for i in range(y_bin.shape[1]) if len(np.unique(y_bin[:, i])) == 2]
            if len(valid_cols) > 0:
                test_metrics["test_auc_macro_or_binary"] = float(
                    roc_auc_score(
                        y_bin[:, valid_cols],
                        test_probs[:, valid_cols],
                        average="macro",
                        multi_class="ovr",
                    )
                )
    except Exception as e:
        print("[WARN] test AUC 计算失败：", e)

    summary_metrics = {
        "best_val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    print("[INFO] Validation metrics:", val_metrics)
    print("[INFO] Test metrics:", test_metrics)

    # 14) 保存模型
    trainer.save_model(OUT_DIR)
    dump(label_encoder, open(os.path.join(OUT_DIR, "label_encoder.pkl"), "wb"))
    with open(os.path.join(OUT_DIR, "best_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2, ensure_ascii=False)

    # 15) 导出测试集分析结果
    export_eval_artifacts(
        trainer=trainer,
        model=model,
        test_set=test_set,
        test_idx=test_idx,
        sample_ids=sample_ids,
        label_encoder=label_encoder,
        taxa_names=taxa_names,
        out_dir=ANALYSIS_DIR,
        n_taxa=N_TAXA,
    )

    # ===== 保存固定布局定义（节点级抽特征必须）=====
    edges_path = os.path.join(OUT_DIR, "edges_nonzero.npy")
    np.save(edges_path, edges.astype(np.float32))
    print("[SAVE] edges_nonzero.npy ->", edges_path)

    token_map = {
        "scheme": "fixed_position_bins",
        "n_taxa": int(N_TAXA),
        "seq_len": int(N_TAXA + 2),
        "n_bins_nonzero": int(N_BINS_NONZERO),
        "pad_id": int(pad_id),
        "bos_id": int(bos_id),
        "eos_id": int(eos_id),
        "bin_token_ids": [int(x) for x in bin_token_ids],
        "taxa_names": [str(x) for x in taxa_names],
    }
    token_map_path = os.path.join(OUT_DIR, "token_map.json")
    with open(token_map_path, "w", encoding="utf-8") as f:
        json.dump(token_map, f, indent=2, ensure_ascii=False)
    print("[SAVE] token_map.json ->", token_map_path)

    run_config = dict(
        ABUND_CSV=ABUND_CSV,
        LABELS_CSV=LABELS_CSV,
        PRETRAINED_MODEL_DIR=PRETRAINED_MODEL_DIR,
        OUT_DIR=OUT_DIR,
        TRAIN_RATIO=TRAIN_RATIO,
        VAL_RATIO=VAL_RATIO,
        TEST_RATIO=TEST_RATIO,
        SEED=SEED,
        LR=LR,
        EPOCHS=EPOCHS,
        BATCH_SIZE=BATCH_SIZE,
        N_TAXA=N_TAXA,
        N_BINS_NONZERO=N_BINS_NONZERO,
        WEIGHT_ALPHA=WEIGHT_ALPHA,
        note="Input layout: [BOS] + 642 bin tokens (position=taxa index) + [EOS]. Node features = h[:,1:1+642,:]."
    )
    with open(os.path.join(OUT_DIR, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    print("[INFO] Done. Model+artifacts saved in:", OUT_DIR)


if __name__ == "__main__":
    main()