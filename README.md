# BGFM — Brain Gut Foundation Model

Code organization for the manuscript **“Brain Gut Foundation Model for Linking Brain Function and Microbial Ecology.”**

BGFM adapts modality-specific foundation models for BOLD signals and microbial abundance, aligns region-level brain and taxa-level microbiome representations with a bidirectional cross-modal bridge, and supports cross-modal prediction, counterfactual disease-state perturbation, multimodal classification, and MDD latent-factor/subtype analyses.

## Repository design

The release keeps the numerical analysis logic close to the research scripts while separating machine-specific paths from source code. All paths and run switches are controlled from YAML. No user-specific local filesystem paths are stored in the Python source.

```text
BGFM-GitHub/
├── bgfm.py                         # single top-level runner
├── configs/example.yaml            # public, path-neutral configuration
├── src/bgfm/
│   ├── cli.py
│   ├── runtime.py
│   └── workflows/
│       ├── brain_encoder.py
│       ├── brain_feature_extraction.py
│       ├── gut_encoder.py
│       ├── gut_feature_extraction.py
│       ├── bridge.py
│       ├── counterfactual.py
│       ├── classification.py
│       └── subtype.py
├── data/README.md
├── docs/pipeline.md
├── third_party/README.md
└── tests/
```

## Installation

Python 3.9 is recommended for manuscript-level reproducibility.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

BrainLM and MGM source code/weights are external dependencies; see `third_party/README.md`.

## Configure paths

Copy the example config and edit only the local copy:

```bash
cp configs/example.yaml configs/local.yaml
```

`configs/local.yaml` is excluded by `.gitignore`, so machine-specific paths will not be pushed to GitHub.

## One BGFM entry point

```bash
python run_bgfm.py list
python run_bgfm.py brain-encoder --config configs/local.yaml
python run_bgfm.py gut-encoder --config configs/local.yaml
python run_bgfm.py brain-extract --config configs/local.yaml
python run_bgfm.py gut-extract --config configs/local.yaml
python run_bgfm.py align --config configs/local.yaml
python run_bgfm.py counterfactual --config configs/local.yaml
python run_bgfm.py classify --config configs/local.yaml
python run_bgfm.py subtype --config configs/local.yaml
```

To run the configured sequence:

```bash
python run_bgfm.py all --config configs/local.yaml
```

For large experiments, it is usually safer to run each stage separately and verify its output before continuing. `--dry-run` prints stage commands without executing them.

## Paper workflow

1. **Modality adaptation.** Fine-tune BrainLM on fixed-length BOLD segments and MGM/GPT-2 on fixed-position microbial bin-token sequences.
2. **Paired feature extraction.** Extract 90 ROI embeddings and 642 taxa embeddings for the paired cohort.
3. **Bidirectional alignment.** Project both node sets into a common latent dimension, exchange information by cross-attention, and predict microbiome abundance and regional mean BOLD in 10-fold OOF evaluation.
4. **Mechanistic/diagnostic downstream tasks.** Run counterfactual disease-state transfer, weighted multi-kernel SVM classification, and OOF bridge latent-factor/subtype analyses.

See `docs/pipeline.md` for the exact manuscript-to-command mapping.

## Reproducibility notes

- Microbiome bin thresholds are estimated from training data and reused for held-out feature extraction.
- Cross-modal predictions used downstream should be out-of-fold to avoid subject leakage.
- The classification workflow performs feature selection and hyperparameter optimization inside nested CV.
- The subtype workflow uses OOF bridge node representations, not in-sample bridge representations.
- Large datasets, checkpoints, and generated outputs are intentionally git-ignored.

## Citation

If you use this repository, cite the associated BGFM manuscript. A machine-readable `CITATION.cff` is included.
