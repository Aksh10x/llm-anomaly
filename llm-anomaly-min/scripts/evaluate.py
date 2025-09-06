import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from datasets import DatasetDict
from rich.console import Console
import matplotlib.pyplot as plt

from utils import (
    seed_everything,
    load_config,
    load_clinc_dataset,
    preprocess_function,
    compute_metrics,
    compute_msp_scores,
    evaluate_ood_detection,
    plot_roc_curve,
    save_results,
    is_oos_intent
)

console = Console()


def load_trained_model(model_dir: str, config: Dict):
    """Load the fine-tuned model and tokenizer."""
    console.print(f"[blue]Loading trained model from {model_dir}...[/blue]")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config['model']['name'],
        num_labels=config['model']['num_labels']
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, model_dir)
    model = model.merge_and_unload()  # Merge LoRA weights for inference
    
    console.print("[green]✓ Model loaded successfully[/green]")
    return model, tokenizer


def evaluate_model(model, tokenizer, test_dataset, config: Dict):
    """Evaluate the model on test set and compute metrics."""
    model.eval()
    device = next(model.parameters()).device
    
    # Prepare data loader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['evaluation']['batch_size'], 
        shuffle=False
    )
    
    all_predictions = []
    all_labels = []
    all_logits = []
    all_msp_scores = []
    
    console.print("[blue]Running evaluation...[/blue]")
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            
            # Compute MSP scores
            msp_scores = compute_msp_scores(logits)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
            all_msp_scores.extend(msp_scores)
    
    return {
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels),
        'logits': np.array(all_logits),
        'msp_scores': np.array(all_msp_scores)
    }


def main():
    # Load configuration
    config_path = "configs/config.yaml"
    config = load_config(config_path)
    
    # Set random seed
    seed_everything(config['seed'])
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[blue]Using device: {device}[/blue]")
    
    # Load dataset
    console.print("[blue]Loading CLINC-OOS dataset...[/blue]")
    dataset, label2id, id2label = load_clinc_dataset()
    
    # Load trained model
    model_dir = config['output']['model_dir']
    if not os.path.exists(model_dir):
        console.print(f"[red]✗ Model directory not found: {model_dir}[/red]")
        console.print("[yellow]Please run training first: python scripts/train.py --config configs/config.yaml[/yellow]")
        return
    
    model, tokenizer = load_trained_model(model_dir, config)
    model.to(device)
    
    # Prepare test dataset
    max_length = config['model']['max_length']
    def tokenize_function(examples):
        return preprocess_function(examples, tokenizer, max_length)
    
    test_dataset = dataset["test"].map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]  # Only remove text column, keep labels
    )
    test_dataset.set_format("torch")
    
    # Get original test data for OOD analysis
    original_test = dataset["test"]
    
    # Evaluate model
    results = evaluate_model(model, tokenizer, test_dataset, config)
    
    # Separate in-domain and OOD examples
    in_domain_mask = []
    ood_mask = []
    
    for i, example in enumerate(original_test):
        intent_name = id2label[example['intent']]
        if is_oos_intent(intent_name):
            ood_mask.append(i)
        else:
            in_domain_mask.append(i)
    
    in_domain_mask = np.array(in_domain_mask)
    ood_mask = np.array(ood_mask)
    
    # Compute in-domain metrics (accuracy, macro-F1)
    if len(in_domain_mask) > 0:
        id_predictions = results['predictions'][in_domain_mask]
        id_labels = results['labels'][in_domain_mask]
        id_metrics = compute_metrics(id_labels, id_predictions)
        
        console.print("[green]In-Domain Results:[/green]")
        console.print(f"  Accuracy: {id_metrics['accuracy']:.4f}")
        console.print(f"  Macro-F1: {id_metrics['macro_f1']:.4f}")
    else:
        id_metrics = {'accuracy': 0.0, 'macro_f1': 0.0}
        console.print("[yellow]No in-domain examples found in test set[/yellow]")
    
    # Compute OOD detection metrics (AUROC)
    if len(ood_mask) > 0 and len(in_domain_mask) > 0:
        # Create binary OOD labels (1 for OOD, 0 for in-domain)
        ood_binary_labels = np.zeros(len(results['labels']))
        ood_binary_labels[ood_mask] = 1
        
        # Evaluate OOD detection using MSP
        ood_results = evaluate_ood_detection(results['msp_scores'], ood_binary_labels)
        
        console.print("[green]OOD Detection Results:[/green]")
        console.print(f"  AUROC: {ood_results['auroc']:.4f}")
        
        # Plot ROC curve
        if ood_results['auroc'] > 0:
            roc_path = os.path.join(config['output']['results_dir'], 'figs', 'roc.png')
            os.makedirs(os.path.dirname(roc_path), exist_ok=True)
            plot_roc_curve(
                ood_results['fpr'], 
                ood_results['tpr'], 
                ood_results['auroc'], 
                roc_path
            )
    else:
        ood_results = {'auroc': 0.0}
        console.print("[yellow]Insufficient data for OOD evaluation[/yellow]")
    
    # Compile all results
    final_results = {
        'accuracy': id_metrics['accuracy'],
        'macro_f1': id_metrics['macro_f1'],
        'auroc_ood': ood_results['auroc'],
        'num_test_samples': len(results['labels']),
        'num_in_domain': len(in_domain_mask),
        'num_ood': len(ood_mask)
    }
    
    # Save results
    results_path = os.path.join(config['output']['results_dir'], 'tables', 'results.csv')
    save_results(final_results, results_path)
    
    # Print summary
    console.print("\n[bold green]Final Results Summary:[/bold green]")
    console.print(f"  In-Domain Accuracy: {final_results['accuracy']:.4f}")
    console.print(f"  In-Domain Macro-F1: {final_results['macro_f1']:.4f}")
    console.print(f"  OOD Detection AUROC: {final_results['auroc_ood']:.4f}")
    console.print(f"\n[blue]Results saved to:[/blue]")
    console.print(f"  Tables: {results_path}")
    if ood_results['auroc'] > 0:
        console.print(f"  ROC Plot: {roc_path}")


if __name__ == "__main__":
    main()
