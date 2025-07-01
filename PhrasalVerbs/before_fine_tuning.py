import csv
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

def predict_meaning(phrasal_verb, example):
    prompt = f"Define the phrasal verb '{phrasal_verb}' as used in this sentence: '{example}'"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=64,  # Increased length
            num_beams=4,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Process the file
with open("phrasal_examples_reduced.csv", "r") as infile, open("predictions.csv", "w", newline="") as outfile:
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=["phrasal_verb", "example", "description", "predicted_meaning"])
    writer.writeheader()

    for row in reader:
        try:
            predicted = predict_meaning(row["phrasal_verb"], row["example"])
            writer.writerow({
                "phrasal_verb": row["phrasal_verb"],
                "example": row["example"],
                "description": row["description"],
                "predicted_meaning": predicted
            })
        except Exception as e:
            print(f"Error processing {row['phrasal_verb']}: {str(e)}")

print("✅ Done. Predictions saved to 'predictions.csv'")