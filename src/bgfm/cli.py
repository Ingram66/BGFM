from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path
import yaml

WORKFLOWS = {
    "brain-encoder": "brain_encoder.py",
    "brain-extract": "brain_feature_extraction.py",
    "gut-encoder": "gut_encoder.py",
    "gut-extract": "gut_feature_extraction.py",
    "align": "bridge.py",
    "counterfactual": "counterfactual.py",
    "classify": "classification.py",
    "subtype": "subtype.py",
}
PIPELINE_ORDER = ["brain-encoder", "gut-encoder", "brain-extract", "gut-extract", "align", "counterfactual", "classify", "subtype"]


def _repo_root() -> Path:
    # src/bgfm/cli.py -> repository root
    return Path(__file__).resolve().parents[2]


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_workflow(name: str, config: Path, dry_run: bool = False) -> None:
    if name not in WORKFLOWS:
        raise KeyError(name)
    root = _repo_root()
    script = root / "src" / "bgfm" / "workflows" / WORKFLOWS[name]
    cfg = _read_yaml(config)
    env = os.environ.copy()
    env["BGFM_CONFIG"] = str(config.resolve())
    pythonpath = [str(root / "src")]
    if name.startswith("brain"):
        p = cfg.get("brain_encoder", {}).get("brainlm_code_dir") or cfg.get("brain_feature_extraction", {}).get("brainlm_code_dir")
        if p: pythonpath.append(str((root / p).resolve()) if not os.path.isabs(str(p)) else str(p))
    if name.startswith("gut"):
        p = cfg.get("gut_encoder", {}).get("project_dir")
        if p: pythonpath.append(str((root / p).resolve()) if not os.path.isabs(str(p)) else str(p))
    if env.get("PYTHONPATH"): pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    cmd = [sys.executable, str(script)]
    print("[BGFM]", name, "->", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=root, env=env, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="BGFM paper-reproduction workflow runner")
    parser.add_argument("command", choices=[*WORKFLOWS, "all", "list", "show-config"])
    parser.add_argument("--config", default="configs/example.yaml")
    parser.add_argument("--skip", nargs="*", default=[], choices=list(WORKFLOWS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = _repo_root()
    config = Path(args.config)
    if not config.is_absolute(): config = root / config
    if args.command == "list":
        for i, name in enumerate(PIPELINE_ORDER, 1): print(f"{i}. {name}")
        return
    if not config.exists(): raise FileNotFoundError(config)
    if args.command == "show-config":
        print(yaml.safe_dump(_read_yaml(config), sort_keys=False, allow_unicode=True))
        return
    if args.command == "all":
        for name in PIPELINE_ORDER:
            if name not in args.skip: run_workflow(name, config, args.dry_run)
    else:
        run_workflow(args.command, config, args.dry_run)


if __name__ == "__main__":
    main()
