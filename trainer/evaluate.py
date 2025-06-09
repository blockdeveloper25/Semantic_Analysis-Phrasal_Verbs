from sklearn.metrics import accuracy_score, f1_score
import torch

def evaluate_model(model, data_loader):
    model.eval()
    predictions, targets = [], []
    for batch in data_loader:
        with torch.no_grad():
            outputs = model(batch['input_ids'], batch['attention_mask'])
        predictions.extend(torch.argmax(outputs, dim=1).tolist())
        targets.extend(batch['labels'].tolist())

    acc = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, average='weighted')
    return acc, f1
