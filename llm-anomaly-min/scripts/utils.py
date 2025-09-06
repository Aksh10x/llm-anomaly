import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from datasets import load_dataset
import yaml


def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_clinc_dataset():
    """Load and preprocess CLINC-OOS dataset."""
    dataset = load_dataset("clinc_oos", "plus")
    
    # Create label mappings
    labels = dataset["train"].features["intent"].names
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    
    return dataset, label2id, id2label


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Compute accuracy and macro-F1 score."""
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1
    }


def compute_msp_scores(logits: torch.Tensor) -> np.ndarray:
    """Compute Maximum Softmax Probability scores for OOD detection."""
    softmax_probs = torch.softmax(logits, dim=-1)
    msp_scores = torch.max(softmax_probs, dim=-1)[0]
    return msp_scores.cpu().numpy()


def evaluate_ood_detection(msp_scores: np.ndarray, ood_labels: np.ndarray) -> Dict[str, float]:
    """Evaluate OOD detection performance using MSP scores."""
    # OOD labels: 1 for OOD, 0 for in-domain
    # MSP scores: higher values indicate more confident (less likely to be OOD)
    # So we use 1 - msp_scores as the OOD scores
    ood_scores = 1 - msp_scores
    
    try:
        auroc = roc_auc_score(ood_labels, ood_scores)
        fpr, tpr, thresholds = roc_curve(ood_labels, ood_scores)
        
        return {
            'auroc': auroc,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
    except ValueError as e:
        print(f"Warning: Could not compute AUROC - {e}")
        return {
            'auroc': 0.0,
            'fpr': np.array([]),
            'tpr': np.array([]),
            'thresholds': np.array([])
        }


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auroc: float, save_path: str):
    """Plot and save ROC curve for OOD detection."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUROC = {auroc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('OOD Detection ROC Curve (MSP)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to {save_path}")


def save_results(results: Dict, save_path: str):
    """Save evaluation results to CSV file."""
    # Convert results to DataFrame
    results_df = pd.DataFrame([results])
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save to CSV
    results_df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")


def preprocess_function(examples, tokenizer, max_length: int = 128):
    """Tokenize examples for training/evaluation."""
    # Tokenize the text
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None  # Don't convert to tensors here, let Trainer handle it
    )
    
    # Add labels if they exist
    if "intent" in examples:
        tokenized["labels"] = examples["intent"]
    
    return tokenized


def is_oos_intent(intent_name: str) -> bool:
    """Check if an intent is out-of-scope."""
    return intent_name == "oos"
