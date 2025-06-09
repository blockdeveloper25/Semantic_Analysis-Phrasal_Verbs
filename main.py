import torch
from transformers import BertTokenizer
from data.utils import load_and_process_data
from models.bert_classifier import BERTClassifier
from trainer.train import PhrasalVerbDataset, train_model
from trainer.evaluate import evaluate_model
import yaml

def main():
    # Load config
    with open('trainer/config.yaml') as f:
        cfg = yaml.safe_load(f)

    # Load and process dataset
    train_df, val_df, test_df, label2id, id2label = load_and_process_data(cfg['dataset_path'], cfg['test_size'], cfg['val_size'])

    # Prepare tokenizer and datasets
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = PhrasalVerbDataset(train_df, tokenizer, max_len=cfg['max_len'])
    val_dataset = PhrasalVerbDataset(val_df, tokenizer, max_len=cfg['max_len'])
    test_dataset = PhrasalVerbDataset(test_df, tokenizer, max_len=cfg['max_len'])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg['batch_size'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg['batch_size'])

    # Initialize model
    num_labels = len(label2id)
    model = BERTClassifier(num_labels=num_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train
    train_model(model, train_loader, val_loader, epochs=cfg['epochs'], device=device)

    # Evaluate
    print("Validation Set:")
    evaluate_model(model, val_loader, device)

    print("Test Set:")
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()
