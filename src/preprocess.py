"""
Dataset loading and preprocessing utilities for OA-TSC experiments.
"""

import re
from datasets import load_dataset
from omegaconf import DictConfig


def load_gsm8k_dataset(cfg: DictConfig, mode: str = "main"):
    """
    Load GSM8K dataset with proper configuration.
    
    Args:
        cfg: Hydra config containing dataset parameters
        mode: "main", "sanity_check", or "tune" mode
        
    Returns:
        Loaded dataset with proper split and size
    """
    cache_dir = cfg.dataset.get("cache_dir", ".cache/datasets")
    
    # Load full test split
    dataset = load_dataset(
        cfg.dataset.hf_dataset_id,
        cfg.dataset.hf_dataset_config,
        split=cfg.dataset.split,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    
    # Determine number of samples based on mode
    if mode == "sanity_check":
        # Use only 5 samples for sanity check (need at least 5 training steps)
        n_samples = 5
    elif mode == "tune":
        # Use tuning subset for Optuna
        n_samples = cfg.dataset.get("n_tune_samples", 80)
    else:
        # Use full configured samples for main mode
        n_samples = cfg.dataset.get("n_samples", 200)
    
    # Select subset
    if n_samples < len(dataset):
        dataset = dataset.select(range(n_samples))
    
    return dataset


def get_question_answer(example: dict, cfg: DictConfig) -> tuple[str, str]:
    """
    Extract question and gold answer from dataset example.
    
    Args:
        example: Dataset example
        cfg: Hydra config with field mappings
        
    Returns:
        (question, gold_answer) tuple
    """
    question_field = cfg.dataset.get("question_field", "question")
    answer_field = cfg.dataset.get("answer_field", "answer")
    
    question = example[question_field].strip()
    answer = example[answer_field]
    
    # Normalize gold answer to numeric string
    gold = normalize_gold_answer(answer)
    
    return question, gold


def normalize_gold_answer(answer: str) -> str:
    """
    Extract and normalize numeric answer from GSM8K-style answer string.
    
    GSM8K answers are in format: "reasoning text\n#### 42"
    
    Args:
        answer: Raw answer string
        
    Returns:
        Normalized numeric string
    """
    # GSM8K specific: extract number after ####
    if "####" in answer:
        answer = answer.split("####")[-1].strip()
    
    # Extract all numbers from the string
    numbers = re.findall(r"-?\d+(?:\.\d+)?", answer)
    
    if numbers:
        # Take the last number (most likely the final answer)
        num_str = numbers[-1]
        
        # Remove trailing .0 if present (convert 42.0 -> 42)
        if "." in num_str:
            try:
                num = float(num_str)
                if num == int(num):
                    return str(int(num))
            except ValueError:
                pass
        
        return num_str
    
    # Fallback: return stripped answer
    return answer.strip()


def normalize_prediction(pred: str) -> str:
    """
    Normalize predicted answer to match gold answer format.
    
    Args:
        pred: Raw prediction string
        
    Returns:
        Normalized numeric string
    """
    if pred is None:
        return None
    
    # Extract numbers
    numbers = re.findall(r"-?\d+(?:\.\d+)?", str(pred))
    
    if numbers:
        num_str = numbers[-1]
        
        # Remove trailing .0 if present
        if "." in num_str:
            try:
                num = float(num_str)
                if num == int(num):
                    return str(int(num))
            except ValueError:
                pass
        
        return num_str
    
    return None


def is_correct(prediction: str, gold: str) -> bool:
    """
    Check if prediction matches gold answer.
    
    Args:
        prediction: Normalized prediction
        gold: Normalized gold answer
        
    Returns:
        True if correct, False otherwise
    """
    if prediction is None or gold is None:
        return False
    
    # Normalize both
    pred_norm = normalize_prediction(prediction)
    gold_norm = normalize_prediction(gold)
    
    if pred_norm is None or gold_norm is None:
        return False
    
    # Try numeric comparison with tolerance
    try:
        pred_num = float(pred_norm)
        gold_num = float(gold_norm)
        
        # Check if both are integers
        if abs(pred_num - round(pred_num)) < 1e-9 and abs(gold_num - round(gold_num)) < 1e-9:
            return int(pred_num) == int(gold_num)
        
        # Float comparison with tolerance
        return abs(pred_num - gold_num) < 1e-3
    except ValueError:
        # Fallback to string comparison
        return pred_norm == gold_norm
