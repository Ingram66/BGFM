# Paper-to-code pipeline

| Manuscript method | Repository command | Primary output |
|---|---|---|
| Brain encoder fine-tuning/evaluation | `brain-encoder` | BrainLM checkpoint and ROI/subject embeddings |
| Microbiome encoder fine-tuning/evaluation | `gut-encoder` | MGM fine-tuned checkpoint, token map, bin edges |
| Paired brain node extraction | `brain-extract` | `brain_classification_features.pt` |
| Paired microbiome node extraction | `gut-extract` | `gut_classification_features.pt` |
| Bidirectional brain-gut bridging and 10-fold OOF prediction | `align` | OOF brain/microbiome predictions, fold checkpoints, OOF bridge representations |
| Counterfactual disease-state transfer | `counterfactual` | trajectory and attribution analyses |
| Weighted multi-kernel SVM | `classify` | repeated nested-CV metrics and permutation importance |
| sPLS-DA-like latent factors and MDD subtypes | `subtype` | factor scores/loadings, K-means subtype results, clinical associations |

## Important integration change

The original bridge workflow saved OOF predictions and pooled embeddings but did not directly emit the node-level OOF bridge tensor expected by the subtype workflow. This release adds an export of normalized OOF brain and microbiome bridge representations to `node_representations_raw_and_normed.npz` without changing the bridge model or its losses.
