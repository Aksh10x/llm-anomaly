import argparse
import os
from typing import Dict
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import DatasetDict
import numpy as np
from rich.console import Console
from rich.progress import track

from utils import (
    seed_everything, 
    load_config, 
    load_clinc_dataset, 
    preprocess_function
)

console = Console()


def setup_model_and_tokenizer(config: Dict):
    """Initialize model and tokenizer with LoRA configuration."""
    model_name = config['model']['name']
    num_labels = config['model']['num_labels']
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=int(config['lora']['r']),
        lora_alpha=int(config['lora']['lora_alpha']),
        lora_dropout=float(config['lora']['lora_dropout']),
        target_modules=config['lora']['target_modules'],
        bias="none",  # Don't adapt bias terms
        modules_to_save=[]  # Classification head will be automatically added
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    # Print model information
    trainable_params = model.num_parameters(only_trainable=True)
    total_params = model.num_parameters()
    
    console.print(f"[green]✓[/green] Model loaded: {model_name}")
    console.print(f"[green]✓[/green] Total parameters: {total_params:,}")
    console.print(f"[green]✓[/green] Trainable parameters: {trainable_params:,}")
    console.print(f"[green]✓[/green] Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    return model, tokenizer


def prepare_datasets(config: Dict, tokenizer):
    """Load and preprocess CLINC-OOS dataset."""
    console.print("[blue]Loading CLINC-OOS dataset...[/blue]")
    
    dataset, label2id, id2label = load_clinc_dataset()
    max_length = config['model']['max_length']
    
    # Tokenize datasets
    def tokenize_function(examples):
        return preprocess_function(examples, tokenizer, max_length)
    
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]  # Only remove text column, keep labels as "labels"
    )
    
    # Set format for PyTorch
    tokenized_datasets.set_format("torch")
    
    console.print(f"[green]✓[/green] Dataset loaded and tokenized")
    console.print(f"[green]✓[/green] Train: {len(tokenized_datasets['train'])} samples")
    console.print(f"[green]✓[/green] Validation: {len(tokenized_datasets['validation'])} samples")
    console.print(f"[green]✓[/green] Test: {len(tokenized_datasets['test'])} samples")
    
    return tokenized_datasets, label2id, id2label


def setup_trainer(model, tokenizer, train_dataset, eval_dataset, config: Dict):
    """Setup Trainer with training arguments."""
    
    # Ensure numeric values are properly typed
    learning_rate = float(config['training']['learning_rate'])
    batch_size = int(config['training']['batch_size'])
    eval_batch_size = int(config['evaluation']['batch_size'])
    num_epochs = int(config['training']['num_epochs'])
    warmup_steps = int(config['training']['warmup_steps'])
    weight_decay = float(config['training']['weight_decay'])
    grad_accum_steps = int(config['training']['gradient_accumulation_steps'])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config['output']['model_dir'],
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_epochs,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy=config['output']['save_strategy'],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_accumulation_steps=grad_accum_steps,
        dataloader_pin_memory=False,
        remove_unused_columns=True,  # Let Trainer handle column removal properly
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Train DistilBERT with LoRA on CLINC-OOS")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    seed_everything(config['seed'])
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[blue]Using device: {device}[/blue]")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Prepare datasets
    datasets, label2id, id2label = prepare_datasets(config, tokenizer)
    
    # Setup trainer
    trainer = setup_trainer(
        model, tokenizer, 
        datasets["train"], 
        datasets["validation"], 
        config
    )
    
    # Start training
    console.print("[yellow]Starting training...[/yellow]")
    
    try:
        trainer.train()
        console.print("[green]✓ Training completed successfully![/green]")
        
        # Save the final model
        trainer.save_model()
        tokenizer.save_pretrained(config['output']['model_dir'])
        console.print(f"[green]✓ Model saved to {config['output']['model_dir']}[/green]")
        
    except Exception as e:
        console.print(f"[red]✗ Training failed: {str(e)}[/red]")
        raise


if __name__ == "__main__":
    main()
