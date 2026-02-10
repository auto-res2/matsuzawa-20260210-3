"""
Main orchestration script for OA-TSC experiments.
Handles mode selection and invokes train.py.
"""

import sys
import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf


# Pre-process sys.argv to extract mode flags and run_id before Hydra sees them
# This allows workflows to use --sanity_check, --pilot, --main without Hydra errors
_MODE = "main"
_RUN_ID = None
_MODE_FLAGS = ["--sanity_check", "--pilot", "--main"]
_filtered_argv = []
for i, arg in enumerate(sys.argv):
    if arg == "--sanity_check":
        _MODE = "sanity_check"
    elif arg == "--pilot":
        _MODE = "pilot"
    elif arg == "--main":
        _MODE = "main"
    elif arg.startswith("run="):
        _RUN_ID = arg.split("=", 1)[1]
        _filtered_argv.append(arg)
    elif arg not in _MODE_FLAGS:
        _filtered_argv.append(arg)
sys.argv = _filtered_argv


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point for experiments.
    
    Applies mode overrides and invokes training.
    
    Args:
        cfg: Hydra config
    """
    # Handle run config loading
    # When run=<run_id> is passed on command line, Hydra sets cfg.run to that string
    run_id = None
    if _RUN_ID:
        run_id = _RUN_ID
    elif "run" in cfg and isinstance(cfg.run, str):
        run_id = cfg.run
    elif "run" in cfg and hasattr(cfg.run, "run_id"):
        run_id = cfg.run.run_id
    
    if run_id:
        run_config_path = Path(__file__).parent.parent / "config" / "runs" / f"{run_id}.yaml"
        if run_config_path.exists():
            print(f"Loading run config from: {run_config_path}")
            run_cfg = OmegaConf.load(run_config_path)
            # Store the full run config under 'run' for reference
            cfg.run = run_cfg
            # Merge top-level fields from run config into root cfg
            for key in ["method", "model", "dataset", "training", "optuna", "evaluation"]:
                if key in run_cfg:
                    cfg[key] = run_cfg[key]
        else:
            raise FileNotFoundError(f"Run config not found: {run_config_path}")
    else:
        raise ValueError("Run config not specified. Please specify run=<run_id> on command line.")
    
    # Use the mode detected from command-line flags
    mode = _MODE
    
    # Apply mode-specific overrides
    if mode == "sanity_check":
        print("Running in SANITY_CHECK mode")
        # Override config for sanity check
        OmegaConf.update(cfg, "training.mode", "sanity_check", merge=False)
        OmegaConf.update(cfg, "wandb.mode", "disabled", merge=False)
        OmegaConf.update(cfg, "optuna.enabled", False, merge=False)
        if "optuna" in cfg and "n_trials" in cfg.optuna:
            OmegaConf.update(cfg, "optuna.n_trials", 0, merge=False)
    elif mode == "pilot":
        print("Running in PILOT mode")
        OmegaConf.update(cfg, "training.mode", "pilot", merge=False)
        # Pilot mode: reduced scale but WandB enabled
        if "dataset" in cfg and "n_samples" in cfg.dataset:
            OmegaConf.update(cfg, "dataset.n_samples", 50, merge=False)
        if "optuna" in cfg and cfg.get("optuna", {}).get("enabled", False):
            OmegaConf.update(cfg, "optuna.n_trials", 5, merge=False)
    else:
        print("Running in MAIN mode")
        OmegaConf.update(cfg, "training.mode", "main", merge=False)
        OmegaConf.update(cfg, "wandb.mode", "online", merge=False)
    
    # Print config
    print("\n" + "="*60)
    print("CONFIGURATION:")
    print("="*60)
    print(OmegaConf.to_yaml(cfg))
    print("="*60 + "\n")
    
    # Import and run training
    from src.train import train
    
    metrics = train(cfg)
    
    print("\nExperiment completed successfully!")
    return metrics


if __name__ == "__main__":
    main()
