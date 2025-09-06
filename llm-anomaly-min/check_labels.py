from datasets import load_dataset

# Load the dataset
dataset = load_dataset('clinc_oos', 'plus')
labels = dataset['train'].features['intent'].names

print(f'Number of classes: {len(labels)}')
print(f'Label range: 0 to {len(labels)-1}')
print(f'First 5 labels: {labels[:5]}')
print(f'Last 5 labels: {labels[-5:]}')

# Check for OOS
if 'oos' in labels:
    oos_index = labels.index('oos')
    print(f'OOS label index: {oos_index}')
else:
    print('OOS label not found')

# Check actual label values in the dataset
train_labels = set(dataset['train']['intent'])
print(f'Actual label values in train set: min={min(train_labels)}, max={max(train_labels)}')
print(f'Unique labels count: {len(train_labels)}')
