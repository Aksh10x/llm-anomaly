from transformers import AutoModelForSequenceClassification

# Load DistilBERT to check layer names
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=151)

print("DistilBERT module names:")
for name, module in model.named_modules():
    if any(x in name.lower() for x in ['linear', 'lin', 'query', 'key', 'value', 'q_', 'k_', 'v_']):
        print(f"  {name}: {type(module).__name__}")
