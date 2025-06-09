import argparse
import yaml
from data.utils import load_and_process_data
from trainer.train import PVDataset, train_model
from trainer.evaluate import evaluate_model
from models.bert_classifier import BERTClassifier
from models.adapter_qlora import inject_qlora
from transformers import BertTokenizer
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_df, val_df, test_df, label2id, id2label = load_and_process_data(cfg['data_path'])

train_dataset = PVDataset(train_df, tokenizer)
val_dataset = PVDataset(val_df, tokenizer)
test_dataset = PVDataset(test_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)
test_loader = DataLoader(test_dataset, batch_size=2)

model = BERTClassifier(num_labels=len(label2id))
model = inject_qlora(model)

train_model(model, train_loader, val_loader, epochs=cfg['epochs'])
acc, f1 = evaluate_model(model, test_loader)
print(f"Test Accuracy: {acc}, F1 Score: {f1}")
