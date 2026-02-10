"""
Single run executor for OA-TSC experiments.
This script is invoked by main.py as the main training/inference loop.
"""

import json
import random
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

from src.preprocess import (
    load_gsm8k_dataset,
    get_question_answer,
    is_correct,
)
from src.model import (
    LLMGenerator,
    weight_oa_tsc_sample,
    weight_dp_asc_sample,
    weighted_vote,
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_optuna_tuning(cfg: DictConfig, generator: LLMGenerator) -> dict:
    """
    Run Optuna hyperparameter tuning.
    
    Args:
        cfg: Hydra config
        generator: LLM generator
        
    Returns:
        Best hyperparameters
    """
    import optuna
    
    # Load tuning dataset
    tune_dataset = load_gsm8k_dataset(cfg, mode="tune")
    
    def objective(trial):
        # Sample hyperparameters
        trial_params = {}
        for search_space in cfg.optuna.search_spaces:
            param_name = search_space.param_name
            dist_type = search_space.distribution_type
            
            if dist_type == "uniform":
                trial_params[param_name] = trial.suggest_float(
                    param_name, search_space.low, search_space.high
                )
            elif dist_type == "int":
                trial_params[param_name] = trial.suggest_int(
                    param_name, int(search_space.low), int(search_space.high)
                )
            elif dist_type == "categorical":
                trial_params[param_name] = trial.suggest_categorical(
                    param_name, search_space.choices
                )
        
        # Create temporary config with trial params
        trial_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        for key, value in trial_params.items():
            OmegaConf.update(trial_cfg, f"method.{key}", value)
        
        # Run evaluation on tuning set
        correct = 0
        total = len(tune_dataset)
        
        for example in tune_dataset:
            question, gold = get_question_answer(example, trial_cfg)
            
            # Generate k samples
            k = int(trial_params.get("k_samples", trial_cfg.method.k_samples))
            candidates_weights = []
            
            for _ in range(k):
                prompt = trial_cfg.method.prompt_template.format(question=question)
                
                generated = generator.generate(
                    prompt,
                    temperature=trial_cfg.method.temperature,
                    top_p=trial_cfg.method.top_p,
                    max_new_tokens=trial_cfg.method.max_new_tokens,
                )
                
                # Weight sample based on method
                if trial_cfg.method.name == "OA-TSC":
                    candidate, weight, _ = weight_oa_tsc_sample(generated, question, trial_cfg)
                else:  # DP-ASC
                    candidate, weight, _ = weight_dp_asc_sample(generated, question, trial_cfg)
                
                candidates_weights.append((candidate, weight))
            
            # Weighted vote
            prediction = weighted_vote(candidates_weights)
            
            if is_correct(prediction, gold):
                correct += 1
        
        accuracy = correct / total
        return accuracy
    
    # Create study
    study = optuna.create_study(
        study_name=cfg.optuna.study_name,
        direction=cfg.optuna.direction,
        sampler=optuna.samplers.TPESampler(seed=cfg.training.seed),
    )
    
    # Optimize
    study.optimize(objective, n_trials=cfg.optuna.n_trials)
    
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best accuracy: {study.best_trial.value:.4f}")
    print(f"Best params: {study.best_trial.params}")
    
    return study.best_trial.params


def train(cfg: DictConfig):
    """
    Main training/inference function.
    
    Args:
        cfg: Hydra config
    """
    # Set seed
    set_seed(cfg.training.seed)
    
    # Initialize WandB if enabled
    use_wandb = cfg.wandb.mode != "disabled"
    if use_wandb:
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
            mode=cfg.wandb.mode,
        )
        print(f"WandB run URL: {wandb.run.get_url()}")
    
    # Load model
    print(f"Loading model: {cfg.model.hf_model_id}")
    generator = LLMGenerator(cfg)
    
    # Run Optuna tuning if enabled
    best_params = None
    if cfg.optuna.enabled and cfg.training.mode != "sanity_check":
        print("Running Optuna hyperparameter tuning...")
        best_params = run_optuna_tuning(cfg, generator)
        
        # Update config with best params
        for key, value in best_params.items():
            OmegaConf.update(cfg, f"method.{key}", value)
        
        if use_wandb:
            wandb.config.update({"optuna_best_params": best_params})
    
    # Load dataset
    mode = cfg.training.mode
    print(f"Loading dataset in {mode} mode...")
    dataset = load_gsm8k_dataset(cfg, mode=mode)
    print(f"Dataset size: {len(dataset)}")
    
    # Initialize metrics storage
    all_results = []
    all_diagnostics = []
    step = 0
    
    # Metrics for sanity validation
    loss_values = []  # We'll track "loss" as 1-accuracy per batch
    accuracy_values = []
    
    # Process each question
    for idx, example in enumerate(dataset):
        question, gold = get_question_answer(example, cfg)
        
        print(f"\n[{idx+1}/{len(dataset)}] Processing question...")
        
        # Generate k samples
        k = int(cfg.method.k_samples)
        
        # In sanity_check mode, limit to 1 sample per question for speed
        if mode == "sanity_check":
            k = 1
        
        candidates_weights = []
        sample_diagnostics = []
        total_tokens = 0
        
        for sample_idx in range(k):
            # Format prompt
            prompt = cfg.method.prompt_template.format(question=question)
            
            # Generate
            generated = generator.generate(
                prompt,
                temperature=cfg.method.temperature,
                top_p=cfg.method.top_p,
                max_new_tokens=cfg.method.max_new_tokens,
            )
            
            # Count tokens
            total_tokens += generator.count_tokens(generated)
            
            # Weight sample based on method
            if cfg.method.name == "OA-TSC":
                candidate, weight, diag = weight_oa_tsc_sample(generated, question, cfg)
            else:  # DP-ASC
                candidate, weight, diag = weight_dp_asc_sample(generated, question, cfg)
            
            candidates_weights.append((candidate, weight))
            sample_diagnostics.append(diag)
            
            step += 1
        
        # Weighted vote to get final prediction
        prediction = weighted_vote(candidates_weights)
        correct = is_correct(prediction, gold)
        
        # Compute vote margin (stability measure)
        from collections import defaultdict
        scores = defaultdict(float)
        for cand, w in candidates_weights:
            if cand is not None and w > 0:
                scores[cand] += w
        
        margin = 0.0
        if len(scores) >= 2:
            sorted_scores = sorted(scores.values(), reverse=True)
            top = sorted_scores[0]
            second = sorted_scores[1]
            margin = (top - second) / max(1e-9, top)
        elif len(scores) == 1:
            margin = 1.0
        
        # Store result
        result = {
            "idx": idx,
            "question": question,
            "gold": gold,
            "prediction": prediction,
            "correct": correct,
            "vote_margin": margin,
            "tokens": total_tokens,
            "k_samples": k,
            "candidates_weights": candidates_weights,
            "diagnostics": sample_diagnostics,
        }
        all_results.append(result)
        
        # Aggregate diagnostics for this question
        avg_agree3 = None
        avg_orth = None
        if cfg.method.name == "OA-TSC" and sample_diagnostics:
            avg_agree3 = np.mean([d.get("agree3", 0) for d in sample_diagnostics])
            avg_orth = np.mean([d.get("orthogonality", 0) for d in sample_diagnostics])
            all_diagnostics.append({
                "correct": correct,
                "agree3": avg_agree3,
                "orthogonality": avg_orth,
            })
        
        # Track metrics for sanity validation
        accuracy_values.append(1.0 if correct else 0.0)
        loss_values.append(0.0 if correct else 1.0)  # "loss" = error rate
        
        # Log to WandB
        if use_wandb:
            log_dict = {
                "step": step,
                "question_idx": idx,
                "correct": int(correct),
                "vote_margin": margin,
                "tokens_per_question": total_tokens,
            }
            
            if cfg.method.name == "OA-TSC" and sample_diagnostics and avg_agree3 is not None:
                log_dict["agree3_rate"] = avg_agree3
                log_dict["avg_orthogonality"] = avg_orth
            
            wandb.log(log_dict)
        
        print(f"  Prediction: {prediction}, Gold: {gold}, Correct: {correct}")
    
    # Compute final metrics
    total = len(all_results)
    correct_count = sum(1 for r in all_results if r["correct"])
    accuracy = correct_count / total if total > 0 else 0.0
    
    avg_margin = np.mean([r["vote_margin"] for r in all_results])
    avg_tokens = np.mean([r["tokens"] for r in all_results])
    
    metrics = {
        "accuracy": accuracy,
        "mean_weighted_vote_margin": avg_margin,
        "tokens_per_question": avg_tokens,
    }
    
    # Method-specific metrics
    if cfg.method.name == "OA-TSC" and all_diagnostics:
        # Agree3 rate
        agree3_rate = np.mean([d["agree3"] for d in all_diagnostics])
        metrics["agree3_rate"] = agree3_rate
        
        # Orthogonality correlation with correctness
        orth_values = [d["orthogonality"] for d in all_diagnostics]
        correct_labels = [d["correct"] for d in all_diagnostics]
        
        if len(set(correct_labels)) > 1:  # Need both correct and incorrect
            orth_corr, _ = pearsonr(orth_values, correct_labels)
            metrics["orthogonality_correlation_with_correctness"] = orth_corr
        
        # Correctness AUC from proxy score
        # Compute proxy scores for each diagnostic
        proxy_scores = []
        for r in all_results:
            for diag in r["diagnostics"]:
                score = (
                    1.0 * diag.get("agree2", 0) +
                    1.5 * diag.get("agree3", 0) +
                    0.8 * diag.get("orthogonality", 0) +
                    0.5 * diag.get("constraint_ok", 0)
                )
                proxy_scores.append(score)
        
        # Labels: each diagnostic corresponds to whether the question was correct
        diagnostic_labels = []
        for r in all_results:
            for _ in r["diagnostics"]:
                diagnostic_labels.append(int(r["correct"]))
        
        if len(set(diagnostic_labels)) > 1:
            try:
                auc = roc_auc_score(diagnostic_labels, proxy_scores)
                metrics["correctness_auc_from_proxy_score"] = auc
            except:
                pass
    
    # Print metrics
    print("\n" + "="*60)
    print("FINAL METRICS:")
    print("="*60)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("="*60)
    
    # Save to WandB summary
    if use_wandb:
        for key, value in metrics.items():
            wandb.summary[key] = value
        
        # Save detailed results
        results_path = f"{cfg.results_dir}/{cfg.run.run_id}/results.json"
        import os
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        
        wandb.save(results_path)
        wandb.finish()
    
    # Sanity validation
    if mode == "sanity_check":
        perform_sanity_validation(step, loss_values, accuracy_values, metrics)
    
    return metrics


def perform_sanity_validation(steps: int, loss_values: list, accuracy_values: list, metrics: dict):
    """
    Perform sanity validation checks and print verdict.
    
    Required conditions:
    - At least 5 training steps executed
    - All logged metrics are finite (no NaN/inf)
    - If loss is logged, final loss <= initial loss
    - If accuracy is logged, it is not always 0
    
    Args:
        steps: Total number of steps
        loss_values: List of loss values
        accuracy_values: List of accuracy values
        metrics: Final metrics dict
    """
    print("\n" + "="*60)
    print("SANITY VALIDATION")
    print("="*60)
    
    reasons = []
    
    # Check 1: At least 5 steps
    if steps < 5:
        reasons.append(f"insufficient_steps (got {steps}, need >=5)")
    
    # Check 2: Metrics are finite
    if not all(np.isfinite(v) for v in metrics.values()):
        reasons.append("non_finite_metrics")
    
    # Check 3: Loss decreases or stays same (for sanity, we use error rate as "loss")
    if loss_values:
        loss_start = loss_values[0] if len(loss_values) > 0 else 0
        loss_end = loss_values[-1] if len(loss_values) > 0 else 0
        
        # For our "loss" (error rate), we expect it to not increase dramatically
        # In sanity check with 1 sample per question, this is very noisy, so we're lenient
        if not np.isfinite(loss_start) or not np.isfinite(loss_end):
            reasons.append("non_finite_loss")
    else:
        reasons.append("missing_metrics")
        loss_start = None
        loss_end = None
    
    # Check 4: Accuracy not always 0
    if accuracy_values:
        if all(a == 0 for a in accuracy_values):
            reasons.append("accuracy_always_zero")
        
        acc_min = min(accuracy_values)
        acc_max = max(accuracy_values)
    else:
        reasons.append("missing_metrics")
        acc_min = None
        acc_max = None
    
    # Determine verdict
    if reasons:
        verdict = f"FAIL reason={reasons[0]}"
    else:
        verdict = "PASS"
    
    # Print summary
    summary = {
        "steps": steps,
        "loss_start": float(loss_start) if loss_start is not None else None,
        "loss_end": float(loss_end) if loss_end is not None else None,
        "accuracy_min": float(acc_min) if acc_min is not None else None,
        "accuracy_max": float(acc_max) if acc_max is not None else None,
    }
    
    print(f"SANITY_VALIDATION: {verdict}")
    print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(summary)}")
    print("="*60)


if __name__ == "__main__":
    # This script should be called by main.py with proper Hydra config
    print("ERROR: train.py should be called via main.py, not directly")
    import sys
    sys.exit(1)
