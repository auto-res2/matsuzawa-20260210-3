"""
Model loading and inference utilities for OA-TSC experiments.
"""

import re
from collections import Counter, defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from omegaconf import DictConfig


# Regular expressions for parsing
NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")  # Support comma separators
A_RE = re.compile(r"FinalA[:\s]+([^\n]*)", re.IGNORECASE)
B_RE = re.compile(r"FinalB[:\s]+([^\n]*)", re.IGNORECASE)
C_RE = re.compile(r"FinalC[:\s]+([^\n]*)", re.IGNORECASE)

# Alternative patterns for extracting answers
ANSWER_PATTERNS = [
    re.compile(r"(?:final answer|answer|result)[:\s]+([^\n]*)", re.IGNORECASE),
    re.compile(r"(?:therefore|thus|so)[,\s]+.*?is[:\s]+([^\n.]+)", re.IGNORECASE),
    re.compile(r"####\s*(\d+)", re.IGNORECASE),  # GSM8K format
]

# Section extraction patterns
SEC_A = re.compile(r"Solve-A\s*:(.*?)(?:Solve-B\s*:|Check-C\s*:|FinalA:|$)", re.IGNORECASE | re.DOTALL)
SEC_B = re.compile(r"Solve-B\s*:(.*?)(?:Check-C\s*:|FinalA:|$)", re.IGNORECASE | re.DOTALL)
SEC_C = re.compile(r"Check-C\s*:(.*?)(?:FinalA:|$)", re.IGNORECASE | re.DOTALL)

TOK_RE = re.compile(r"[a-zA-Z0-9]+")


class LLMGenerator:
    """Wrapper for HuggingFace LLM generation."""
    
    def __init__(self, cfg: DictConfig):
        """
        Initialize model and tokenizer.
        
        Args:
            cfg: Hydra config containing model parameters
        """
        self.cfg = cfg
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.hf_model_id,
            cache_dir=cfg.model.get("cache_dir", ".cache/models"),
            trust_remote_code=True
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Determine dtype
        dtype_str = cfg.model.get("torch_dtype", "bfloat16")
        if dtype_str == "bfloat16":
            dtype = torch.bfloat16
        elif dtype_str == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model.hf_model_id,
            dtype=dtype,
            device_map=cfg.model.get("device_map", "auto"),
            cache_dir=cfg.model.get("cache_dir", ".cache/models"),
            trust_remote_code=True
        )
        
        self.model.eval()
        
    @torch.no_grad()
    def generate(self, prompt: str, temperature: float = 0.7, 
                 top_p: float = 0.95, max_new_tokens: int = 450) -> str:
        """
        Generate completion for a prompt.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text (only the new completion, without the prompt)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        # Decode only the new tokens (exclude prompt)
        new_tokens = outputs[0, input_length:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return generated_text
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))


# Parsing utilities

def normalize_num(s: str) -> str:
    """Extract and normalize numeric value from string."""
    if s is None:
        return None
    
    # Convert to string if not already
    s = str(s).strip()
    
    # Remove commas from numbers
    s = s.replace(',', '')
    
    nums = NUM_RE.findall(s)
    if not nums:
        return None
    
    # Return last number found (often the final answer)
    last_num = nums[-1].replace(',', '')  # Remove any commas
    
    # Clean up: remove trailing .0 for integers
    if '.' in last_num:
        try:
            val = float(last_num)
            if val == int(val):
                return str(int(val))
        except ValueError:
            pass
    
    return last_num


def extract_finals(text: str) -> tuple:
    """
    Extract FinalA, FinalB, FinalC from generated text.
    
    Args:
        text: Generated completion
        
    Returns:
        (finalA, finalB, finalC) tuple of normalized numeric strings or None
    """
    ma = A_RE.search(text)
    mb = B_RE.search(text)
    mc = C_RE.search(text)
    
    a = normalize_num(ma.group(1)) if ma else None
    b = normalize_num(mb.group(1)) if mb else None
    c = normalize_num(mc.group(1)) if mc else None
    
    # Fallback 1: try alternative answer patterns if FinalA not found
    if a is None:
        for pattern in ANSWER_PATTERNS:
            match = pattern.search(text)
            if match:
                a = normalize_num(match.group(1))
                if a is not None:
                    break
    
    # Fallback 2: if still no answer, look for numbers in the last few lines
    if a is None:
        # Look for numbers in the last few lines
        lines = text.strip().split('\n')
        for line in reversed(lines[-10:]):  # Check last 10 lines
            # Skip empty lines
            if not line.strip():
                continue
            nums = NUM_RE.findall(line.replace(',', ''))
            if nums:
                # Use last number found as FinalA
                a = nums[-1]
                break
    
    return a, b, c


def extract_section(text: str, pattern: re.Pattern) -> str:
    """Extract a reasoning section from text."""
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def jaccard_overlap(text_a: str, text_b: str) -> float:
    """
    Compute Jaccard overlap between two text blocks based on token sets.
    
    Args:
        text_a: First text
        text_b: Second text
        
    Returns:
        Jaccard similarity in [0, 1]
    """
    tokens_a = set(t.lower() for t in TOK_RE.findall(text_a))
    tokens_b = set(t.lower() for t in TOK_RE.findall(text_b))
    
    if not tokens_a and not tokens_b:
        return 0.0
    
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    
    return intersection / max(1, union)


def majority_of_three(a, b, c) -> tuple:
    """
    Compute majority answer and agreement strength from three values.
    
    Args:
        a, b, c: Three values (can be None)
        
    Returns:
        (majority_value, agree2, agree3) where:
        - majority_value: Most common value
        - agree2: 1 if at least 2 values match, 0 otherwise
        - agree3: 1 if all 3 values match, 0 otherwise
    """
    values = [x for x in (a, b, c) if x is not None]
    
    if not values:
        return None, 0, 0
    
    counter = Counter(values)
    most_common_value, count = counter.most_common(1)[0]
    
    agree2 = int(count >= 2)
    agree3 = int(count == 3)
    
    return most_common_value, agree2, agree3


def generic_constraint_check(question: str, pred: str) -> int:
    """
    Lightweight generic constraint gate.
    
    Checks if prediction violates obvious constraints from question:
    - Counting questions should have non-negative integers
    
    Args:
        question: Question text
        pred: Predicted answer
        
    Returns:
        1 if constraints satisfied, 0 otherwise
    """
    if pred is None:
        return 0
    
    q_lower = question.lower()
    
    # Check for counting questions
    counting_keywords = ["how many", "number of", "altogether", "total", "count"]
    is_counting = any(kw in q_lower for kw in counting_keywords)
    
    if is_counting:
        # Should be non-negative
        if pred.startswith('-'):
            return 0
        
        # Should be integer (no decimal point)
        if '.' in pred:
            return 0
    
    return 1


def weight_oa_tsc_sample(text: str, question: str, cfg: DictConfig) -> tuple:
    """
    Compute OA-TSC weight for a single generated sample.
    
    Args:
        text: Generated completion
        question: Original question
        cfg: Config with weighting parameters
        
    Returns:
        (candidate_answer, weight, diagnostics_dict)
    """
    # Extract finals
    a, b, c = extract_finals(text)
    
    # Compute majority and agreement
    candidate, agree2, agree3 = majority_of_three(a, b, c)
    
    # Extract reasoning sections
    sec_a = extract_section(text, SEC_A)
    sec_b = extract_section(text, SEC_B)
    sec_c = extract_section(text, SEC_C)
    
    # Compute orthogonality (independence) via lexical overlap
    overlap_ab = jaccard_overlap(sec_a, sec_b)
    overlap_ac = jaccard_overlap(sec_a, sec_c)
    orthogonality = 1.0 - max(overlap_ab, overlap_ac)
    
    # Constraint check
    constraint_ok = generic_constraint_check(question, candidate)
    
    # Get weighting parameters from config
    tau = cfg.method.get("tau_copy_threshold", 0.35)
    b2 = cfg.method.get("b2_agree2_weight", 1.0)
    b3 = cfg.method.get("b3_agree3_weight", 1.5)
    d_orth = cfg.method.get("d_orth_weight", 0.8)
    g_cons = cfg.method.get("g_cons_weight", 0.5)
    copy_penalty = cfg.method.get("copy_penalty_scale", 0.8)
    
    # Compute weight
    weight = 1.0 + b2 * agree2 + b3 * agree3 + d_orth * orthogonality + g_cons * constraint_ok
    
    # Penalize agreement with low orthogonality (likely copied)
    if agree2 == 1 and orthogonality < tau:
        penalty = copy_penalty * (tau - orthogonality) / max(1e-6, tau)
        weight -= penalty
    
    # Zero weight if no valid candidate
    if candidate is None:
        weight = 0.0
    
    diagnostics = {
        "agree2": agree2,
        "agree3": agree3,
        "orthogonality": orthogonality,
        "constraint_ok": constraint_ok,
        "overlap_ab": overlap_ab,
        "overlap_ac": overlap_ac,
    }
    
    return candidate, weight, diagnostics


def weight_dp_asc_sample(text: str, question: str, cfg: DictConfig) -> tuple:
    """
    Compute DP-ASC weight for a single generated sample.
    
    Args:
        text: Generated completion
        question: Original question
        cfg: Config with weighting parameters
        
    Returns:
        (candidate_answer, weight, diagnostics_dict)
    """
    # Extract finals (only A and B for DP-ASC)
    a, b, _ = extract_finals(text + "\nFinalC:")  # Dummy C to reuse extraction
    
    # Fallback: if both A and B are None but we found a number in the fallback, use it for both
    if a is not None and b is None:
        b = a  # If only A found, assume B should be same for agreement
    
    # Agreement check
    agree = int(a is not None and b is not None and a == b)
    
    # Use FinalA as candidate (or B if A is None)
    candidate = a if a is not None else b
    
    # Constraint check
    constraint_ok = generic_constraint_check(question, candidate)
    
    # Get weighting parameters from config
    agree_bonus = cfg.method.get("agree_bonus_weight", 1.5)
    g_cons = cfg.method.get("constraint_gate_weight", 0.5)
    
    # Compute weight
    weight = 1.0 + agree_bonus * agree + g_cons * constraint_ok
    
    # Zero weight if no valid candidate
    if candidate is None:
        weight = 0.0
    
    diagnostics = {
        "agree": agree,
        "constraint_ok": constraint_ok,
        "finalA": a,
        "finalB": b,
    }
    
    return candidate, weight, diagnostics


def weighted_vote(candidates_weights: list) -> str:
    """
    Perform weighted voting over candidate answers.
    
    Args:
        candidates_weights: List of (candidate, weight) tuples
        
    Returns:
        Winning candidate answer
    """
    scores = defaultdict(float)
    
    for candidate, weight in candidates_weights:
        if candidate is None or weight <= 0:
            continue
        scores[candidate] += weight
    
    if not scores:
        return None
    
    # Return candidate with highest score
    winner = max(scores.items(), key=lambda kv: kv[1])
    return winner[0]
