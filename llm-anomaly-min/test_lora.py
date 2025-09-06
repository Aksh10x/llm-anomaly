import torch
import yaml
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

print("Testing LoRA setup...")

# Load config
config = load_config("configs/config.yaml")

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    config['model']['name'],
    num_labels=config['model']['num_labels']
)

print(f"Base model parameters: {model.num_parameters():,}")

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=int(config['lora']['r']),
    lora_alpha=int(config['lora']['lora_alpha']),
    lora_dropout=float(config['lora']['lora_dropout']),
    target_modules=config['lora']['target_modules'],
    bias="none",  # Don't adapt bias terms
    modules_to_save=[]  # Empty list - don't save classification head
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Check parameters
trainable_params = model.num_parameters(only_trainable=True)
total_params = model.num_parameters()

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")

# Should be around 0.4-0.5% trainable
print(f"\nExpected: ~300K-500K trainable parameters for LoRA")
print(f"Actual: {trainable_params:,} trainable parameters")

if trainable_params < 2000000:  # Less than 2M (reasonable for LoRA + classification head)
    print("✅ LoRA setup is REASONABLE for classification task!")
    print("Note: Classification head is automatically added for SEQ_CLS task")
else:
    print("❌ LoRA setup is WRONG - too many trainable parameters!")

print("\nTrainable modules:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  {name}: {param.numel():,} parameters")
