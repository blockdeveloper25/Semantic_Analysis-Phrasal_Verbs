import pandas as pd
import numpy as np
import nltk
import torch
from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge import Rouge
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher
import os

# =========================
# 🛠️ NLTK Setup & Downloads
# =========================
# Set NLTK data path to ensure proper downloads
nltk_data_dir = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download all required NLTK data
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)
nltk.download('omw-1.4', download_dir=nltk_data_dir)  # Required for METEOR

# =========================
# 📂 Data Loading & Preparation
# =========================
# Load actual and predicted data
df_actual = pd.read_csv("../phrasal_examples.csv")
df_pred = pd.read_csv("../predicted_meanings.csv")

# Merge and clean
df = df_actual.copy()
df["predicted_meaning"] = df_pred["predicted_meaning"]
df = df.dropna(subset=["description", "predicted_meaning"])

# =========================
# 📊 Evaluation Metrics
# =========================
# ✅ Metric 1: Exact Match
df["exact_match"] = df["description"].str.strip().str.lower() == df["predicted_meaning"].str.strip().str.lower()
accuracy = df["exact_match"].mean()

# ✅ Metric 2: BERT Cosine Similarity
bert = SentenceTransformer("all-MiniLM-L6-v2")
actual_embeddings = bert.encode(df["description"].tolist(), convert_to_tensor=True)
predicted_embeddings = bert.encode(df["predicted_meaning"].tolist(), convert_to_tensor=True)
cos_sim = torch.nn.functional.cosine_similarity(actual_embeddings, predicted_embeddings)
df["cosine_similarity"] = cos_sim.tolist()
avg_cosine = df["cosine_similarity"].mean()

# ✅ Metric 3: BLEU with smoothing
smoother = SmoothingFunction().method4
df["bleu"] = [
    sentence_bleu([ref.split()], pred.split(), smoothing_function=smoother)
    for ref, pred in zip(df["description"], df["predicted_meaning"])
]
avg_bleu = df["bleu"].mean()

# ✅ Metric 4: ROUGE-L
rouge = Rouge()
scores = rouge.get_scores(
    [pred for pred in df["predicted_meaning"]],
    [ref for ref in df["description"]],
    avg=True
)
avg_rouge_l = scores['rouge-l']['f']
df["rouge_l"] = [score['rouge-l']['f'] for score in rouge.get_scores(
    df["predicted_meaning"].tolist(),
    df["description"].tolist()
)]

# ✅ Metric 5: Levenshtein similarity
def levenshtein_sim(a, b):
    return SequenceMatcher(None, a, b).ratio()
df["levenshtein_similarity"] = [
    levenshtein_sim(a, b) for a, b in zip(df["description"], df["predicted_meaning"])
]
avg_lev = df["levenshtein_similarity"].mean()

# ✅ Metric 6: Jaccard similarity
def jaccard_sim(a, b):
    a_set = set(a.lower().split())
    b_set = set(b.lower().split())
    return len(a_set & b_set) / len(a_set | b_set) if (a_set | b_set) else 0
df["jaccard_similarity"] = [
    jaccard_sim(a, b) for a, b in zip(df["description"], df["predicted_meaning"])
]
avg_jaccard = df["jaccard_similarity"].mean()

# ✅ Metric 7: METEOR with robust tokenization
try:
    df["meteor"] = [
        meteor_score([word_tokenize(ref)], word_tokenize(pred))
        for ref, pred in zip(df["description"], df["predicted_meaning"])
    ]
except LookupError:
    # Fallback to simple whitespace tokenization if punkt fails
    df["meteor"] = [
        meteor_score([ref.split()], pred.split())
        for ref, pred in zip(df["description"], df["predicted_meaning"])
    ]
avg_meteor = df["meteor"].mean()

# =========================
# 📊 Results Presentation
# =========================
print("\n📊 Evaluation Summary:")
print(f"🔹 Exact Match Accuracy        : {accuracy:.2%}")
print(f"🔹 Avg. BERT Cosine Similarity : {avg_cosine:.4f}")
print(f"🔹 Avg. BLEU Score             : {avg_bleu:.4f}")
print(f"🔹 Avg. ROUGE-L Score          : {avg_rouge_l:.4f}")
print(f"🔹 Avg. Levenshtein Similarity : {avg_lev:.4f}")
print(f"🔹 Avg. Jaccard Similarity     : {avg_jaccard:.4f}")
print(f"🔹 Avg. METEOR Score           : {avg_meteor:.4f}")

# Save to CSV
df.to_csv("evaluation_results.csv", index=False)
print("\n✅ All results saved to evaluation_results.csv")