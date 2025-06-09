import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_process_data(filepath, test_size=0.2, val_size=0.1):
    df = pd.read_csv(filepath)
    df['label'] = df['Meaning'].astype('category').cat.codes
    label2id = dict(enumerate(df['Meaning'].astype('category').cat.categories))
    id2label = {v: k for k, v in label2id.items()}

    # Check label distribution
    label_counts = df['label'].value_counts()
    rare_labels = label_counts[label_counts < 2].index.tolist()

    if rare_labels:
        print(f"[INFO] Skipping stratification: {len(rare_labels)} label(s) occur only once.")
        stratify = None
    else:
        stratify = df['label']

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=stratify)
    train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=42, stratify=None)

    return train_df, val_df, test_df, label2id, id2label
