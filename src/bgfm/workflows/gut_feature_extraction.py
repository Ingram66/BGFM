\
import json
import os
import random
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2ForSequenceClassification

from bgfm.runtime import load_section, apply_globals

FEATURE_DIR = "checkpoints/gut_encoder_best"
TEST_ABUND_CSV = "data/paired/microbiome_abundance.csv"
OUT_PATH = "outputs/paired_gut_features/gut_classification_features.pt"
SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 2

apply_globals(globals(), load_section("gut_feature_extraction"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FeatureDataset(Dataset):
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, sample_ids: List[str]):
        self.input_ids = input_ids.cpu().long()
        self.attention_mask = attention_mask.cpu().long()
        self.sample_ids = [str(x) for x in sample_ids]

    def __len__(self):
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "attention_mask": self.attention_mask[idx],
                "idx": idx, "sample_id": self.sample_ids[idx]}


def load_token_map_and_edges(feature_dir: str):
    token_map_path = os.path.join(feature_dir, "token_map.json")
    edges_path = os.path.join(feature_dir, "edges_nonzero.npy")
    if not os.path.exists(token_map_path): raise FileNotFoundError(token_map_path)
    if not os.path.exists(edges_path): raise FileNotFoundError(edges_path)
    with open(token_map_path, "r", encoding="utf-8") as f: token_map = json.load(f)
    edges = np.load(edges_path).astype(np.float32)
    if token_map.get("scheme") != "fixed_position_bins":
        raise ValueError("token_map.scheme must be fixed_position_bins")
    if edges.shape[0] != int(token_map["n_bins_nonzero"]) + 1:
        raise ValueError("edges_nonzero.npy is inconsistent with token_map.json")
    return token_map, edges


def read_abundance_and_align(csv_path: str, taxa_names_train: List[str]):
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2: raise ValueError("Abundance CSV must contain sample_id and taxa columns")
    sample_col = df.columns[0]
    sample_ids = df[sample_col].astype(str).tolist()
    df_taxa = df.set_index(sample_col)
    missing = [t for t in taxa_names_train if t not in df_taxa.columns]
    if missing: raise ValueError(f"Missing {len(missing)} taxa columns; examples: {missing[:5]}")
    return sample_ids, df_taxa[taxa_names_train].to_numpy(np.float32)


def abundance_to_bin_index(x: np.ndarray, edges: np.ndarray):
    k = edges.shape[0] - 1
    bins = np.zeros_like(x, dtype=np.int64)
    nz = x > 0
    if nz.any(): bins[nz] = np.digitize(x[nz], edges[1:-1], right=False).astype(np.int64) + 1
    if bins.max() > k: raise RuntimeError(f"bin overflow: {bins.max()} > {k}")
    return bins


def build_fixed_layout_inputs(x: np.ndarray, token_map: dict, edges: np.ndarray):
    n_taxa = int(token_map["n_taxa"])
    if x.shape[1] != n_taxa: raise ValueError(f"Expected {n_taxa} taxa, got {x.shape[1]}")
    bins = abundance_to_bin_index(x, edges)
    bin_token_ids = np.asarray(token_map["bin_token_ids"], dtype=np.int64)
    bin_ids = np.take(bin_token_ids, bins)
    ids = np.full((x.shape[0], n_taxa + 2), int(token_map["pad_id"]), dtype=np.int64)
    mask = np.ones_like(ids, dtype=np.int64)
    ids[:, 0] = int(token_map["bos_id"])
    ids[:, 1:1+n_taxa] = bin_ids
    ids[:, 1+n_taxa] = int(token_map["eos_id"])
    return torch.tensor(ids), torch.tensor(mask), bins


def sanity_check_node_layout(input_ids: torch.Tensor, token_map: dict, x: np.ndarray):
    n_taxa = int(token_map["n_taxa"])
    if not torch.all(input_ids[:, 0] == int(token_map["bos_id"])): raise ValueError("BOS check failed")
    if not torch.all(input_ids[:, -1] == int(token_map["eos_id"])): raise ValueError("EOS check failed")
    taxa_tokens = input_ids[:, 1:1+n_taxa].numpy()
    bin0 = int(token_map["bin_token_ids"][0])
    if not (taxa_tokens[x == 0] == bin0).all(): raise ValueError("zero abundance -> bin0 token check failed")


@torch.no_grad()
def extract_node_features(model, loader, n_taxa: int):
    model.eval().to(DEVICE)
    node_feats, idxs, sample_ids = [], [], []
    for batch in loader:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        h = model.transformer(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state
        node_feats.append(h[:, 1:1+n_taxa, :].cpu())
        idxs.extend(batch["idx"].tolist())
        sample_ids.extend(batch["sample_id"])
    return torch.cat(node_feats, dim=0), idxs, sample_ids


def main():
    seed_everything(SEED)
    out_parent = os.path.dirname(OUT_PATH)
    if out_parent: os.makedirs(out_parent, exist_ok=True)
    token_map, edges = load_token_map_and_edges(FEATURE_DIR)
    taxa_names = token_map["taxa_names"]
    sample_ids, x = read_abundance_and_align(TEST_ABUND_CSV, taxa_names)
    input_ids, attention_mask, _ = build_fixed_layout_inputs(x, token_map, edges)
    sanity_check_node_layout(input_ids, token_map, x)
    loader = DataLoader(FeatureDataset(input_ids, attention_mask, sample_ids), batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS)
    model = GPT2ForSequenceClassification.from_pretrained(FEATURE_DIR)
    if int(input_ids.max()) >= int(model.transformer.wte.num_embeddings):
        raise ValueError("Token id exceeds checkpoint embedding size; use the fine-tuned output checkpoint.")
    node_emb, _, out_sample_ids = extract_node_features(model, loader, int(token_map["n_taxa"]))
    torch.save({"node_emb": node_emb.to(torch.float16), "sample_ids": out_sample_ids,
                "taxa_names": taxa_names, "edges_nonzero": edges, "token_map": token_map,
                "ckpt_dir": FEATURE_DIR}, OUT_PATH)
    print(f"[SAVE] {OUT_PATH} | node_emb={tuple(node_emb.shape)}")


if __name__ == "__main__":
    main()
