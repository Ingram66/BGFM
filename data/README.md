# Data layout

Data are intentionally not committed. The default config expects:

```text
data/
├── metadata/
│   ├── mni_coordinates.csv        # ROIName,X,Y,Z; 90 rows
│   ├── brain_roi_names.txt        # 90 ROI names
│   └── taxa_names.txt             # 642 taxa names
├── unimodal/
│   ├── brain/
│   │   ├── bold/                  # one BOLD CSV per participant
│   │   └── labels.csv             # sample_id,label
│   └── gut/
│       ├── abundance.csv          # sample_id + 642 taxa
│       └── labels.csv
└── paired/
    ├── bold/                      # paired cohort BOLD CSVs
    ├── bold_roi_mean.csv          # sample_id + 90 standardized ROI means
    ├── microbiome_abundance.csv   # sample_id + 642 taxa
    ├── labels_4class.csv          # subject_id,group (HC/MDD/SZ/BD)
    ├── labels_mdd_hc.csv
    └── clinical_biomarkers.csv
```

Use a git-ignored `configs/local.yaml` for actual paths if your data live elsewhere.
