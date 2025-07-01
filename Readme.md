# Phrasal Verb Meaning Detection

This repository provides tools and scripts for extracting, processing, and fine-tuning models to detect the meanings of phrasal verbs using NLP techniques.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Notebooks](#notebooks)
- [Data](#data)
- [Fine-tuning](#fine-tuning)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project focuses on:
- Extracting phrasal verb examples and meanings from datasets.
- Creating and processing CSV/JSONL datasets for model training.
- Fine-tuning transformer models (e.g., BERT, QLoRA) for phrasal verb meaning detection.
- Evaluating model predictions.

## Project Structure

```
.
├── before_fine_tuning.py
├── csv_creation.py
├── example_extraction.py
├── phrasal_meaning_detector.py
├── qlora_finetune.py
├── training_csv_creation.py
├── requirements.txt
├── Evaluation/
├── File_Operations/
├── Finetune/
│   └── bert_qlora_dataset.json
├── phrasal/
├── *.csv, *.json, *.jsonl
└── *.ipynb
```

- **before_fine_tuning.py**: Preprocessing before model fine-tuning.
- **csv_creation.py**: Scripts for creating CSV datasets.
- **example_extraction.py**: Extracts phrasal verb examples.
- **phrasal_meaning_detector.py**: Main script for meaning detection.
- **qlora_finetune.py**: Fine-tuning using QLoRA.
- **training_csv_creation.py**: Prepares training data.
- **Evaluation/**: Evaluation scripts and results.
- **File_Operations/**: File handling utilities.
- **Finetune/**: Fine-tuning datasets and scripts.
- **phrasal/**: Additional phrasal verb resources.
- **requirements.txt**: Python dependencies.
- **.ipynb files**: Interactive notebooks for exploration and experiments.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/phrasal-verb-meaning-detector.git
    cd phrasal-verb-meaning-detector
    ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Data Preparation

Prepare your dataset using:
```sh
python csv_creation.py
python example_extraction.py
```

### Fine-tuning

Fine-tune a model with:
```sh
python qlora_finetune.py
```

### Meaning Detection

Run the detector:
```sh
python phrasal_meaning_detector.py --input your_input.csv --output predictions.csv
```

## Notebooks

- **basics.ipynb**: Basic data exploration.
- **nltk_download.ipynb**: NLTK resource setup.
- **transformers.ipynb**: Transformer model experiments.

## Data

- **Phrasal_verbs.csv/json**: Main phrasal verb datasets.
- **phrasal_examples*.csv**: Example sentences.
- **qlora_dataset.jsonl**: Dataset for QLoRA fine-tuning.

## Fine-tuning

Fine-tuning scripts and datasets are in the [Finetune/](Finetune/) directory.

## Evaluation

Evaluation scripts and results are in the [Evaluation/](Evaluation/) directory.

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

[MIT License](LICENSE) (update as appropriate)