"""
Main orchestration script for OA-TSC experiments.
Handles mode selection and invokes train.py.
"""

import sys
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point for experiments.
    
    Applies mode overrides and invokes training.
    
    Args:
        cfg: Hydra config
    """
    # Check for mode flags
    mode = "main"
    if "--sanity_check" in sys.argv:
        mode = "sanity_check"
        sys.argv.remove("--sanity_check")
    elif "--pilot" in sys.argv:
        mode = "pilot"
        sys.argv.remove("--pilot")
    elif "--main" in sys.argv:
        mode = "main"
        sys.argv.remove("--main")
    
    # Apply mode-specific overrides
    if mode == "sanity_check":
        print("Running in SANITY_CHECK mode")
        # Override config for sanity check
        OmegaConf.update(cfg, "training.mode", "sanity_check")
        OmegaConf.update(cfg, "wandb.mode", "disabled")
        OmegaConf.update(cfg, "optuna.enabled", False)
        OmegaConf.update(cfg, "optuna.n_trials", 0)
    elif mode == "pilot":
        print("Running in PILOT mode")
        OmegaConf.update(cfg, "training.mode", "pilot")
        # Pilot mode: reduced scale but WandB enabled
        OmegaConf.update(cfg, "dataset.n_samples", 50)
        if cfg.optuna.enabled:
            OmegaConf.update(cfg, "optuna.n_trials", 5)
    else:
        print("Running in MAIN mode")
        OmegaConf.update(cfg, "training.mode", "main")
        OmegaConf.update(cfg, "wandb.mode", "online")
    
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
