# Online Text Anomaly Detection with Lightweight LLMs

A minimal implementation for intent classification and out-of-domain (OOD) detection using DistilBERT with LoRA fine-tuning on the CLINC-OOS dataset.

## Overview

This project implements:
- **Dataset**: CLINC-OOS (150 in-domain intents + out-of-scope detection)
- **Model**: DistilBERT fine-tuned with PEFT-LoRA for efficiency
- **Task**: Multi-class intent classification with OOD detection
- **OOD Method**: Maximum Softmax Probability (MSP) threshold-based detection
- **Metrics**: Accuracy, Macro-F1 (in-domain), AUROC (OOD detection)

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python scripts/train.py --config configs/config.yaml
```

### 3. Evaluate Performance
```bash
python scripts/evaluate.py
```

## Project Structure

```
llm-anomaly-min/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── configs/
│   └── config.yaml             # Hyperparameters and settings
├── scripts/
│   ├── train.py                # Fine-tuning script
│   ├── evaluate.py             # Evaluation and OOD detection
│   └── utils.py                # Helper functions
└── reports/
    ├── figs/                   # Generated plots (ROC curves)
    └── tables/                 # Results CSV files
```

## Results

After running evaluation, check:
- `reports/tables/results.csv` for numerical metrics
- `reports/figs/roc.png` for OOD detection ROC curve

## Configuration

Modify `configs/config.yaml` to adjust:
- Learning rate, batch size, epochs
- LoRA parameters (rank, alpha, dropout)
- Model checkpointing and evaluation settings

## Dependencies

- PyTorch
- Transformers (Hugging Face)
- PEFT (Parameter-Efficient Fine-Tuning)
- Datasets (Hugging Face)
- Scikit-learn
- Pandas, NumPy, Matplotlib
