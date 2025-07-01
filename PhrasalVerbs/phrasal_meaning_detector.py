import json
import csv
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

# Load Phrasal Verb descriptions
with open("Phrasal_verbs.json", "r", encoding="utf-8") as f:
    phrasal_verbs = json.load(f)

# Embedding function
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # CLS token

# Input CSV
input_file = "phrasal_examples.csv"
output_file = "predicted_meanings.csv"

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", newline="", encoding="utf-8") as outfile:
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=["phrasal_verb", "example", "predicted_meaning"])
    writer.writeheader()

    for row in reader:
        verb = row["phrasal_verb"].strip()
        example = row["example"].strip()

        descriptions = phrasal_verbs.get(verb, {}).get("descriptions", [])

        if not descriptions:
            predicted = "N/A"
        else:
            example_vec = get_embedding(example)
            max_score = -1
            predicted = "N/A"

            for desc in descriptions:
                desc_vec = get_embedding(desc)
                score = cosine_similarity([example_vec], [desc_vec])[0][0]
                if score > max_score:
                    max_score = score
                    predicted = desc

        writer.writerow({
            "phrasal_verb": verb,
            "example": example,
            "predicted_meaning": predicted
        })

print(f"✅ Done. Results saved to {output_file}")
