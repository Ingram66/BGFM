# Third-party model code

This repository does not vendor upstream BrainLM/MGM implementations or pretrained weights.

- `third_party/brainlm/`: place the BrainLM modules required by the provided workflow (`configuration_brainlm.py`, `modeling_brainlm.py`, `brainlm_trainer.py`) or point `brainlm_code_dir` elsewhere.
- `third_party/mgm/`: place the MGM tokenizer/corpus implementation (`MicroCorpus.py`) and compatible `corpus.pkl`, or point `project_dir` elsewhere.

Check the upstream licenses before redistributing third-party code or model weights.
