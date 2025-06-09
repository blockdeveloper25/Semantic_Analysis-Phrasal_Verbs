import torch.nn as nn
from transformers import BertForSequenceClassification

class BERTClassifier(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
