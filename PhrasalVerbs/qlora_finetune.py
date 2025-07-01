from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import torch

# Load your dataset
dataset = load_dataset("json", data_files={"train": "qlora_dataset_reduced.jsonl"}, split="train")

# Use T5-small model
model_id = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# Prepare model for QLoRA training
model = prepare_model_for_kbit_training(model)

# LoRA configuration
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
model = get_peft_model(model, peft_config)

# Tokenization logic
def tokenize(example):
    # Combine instruction and input for T5
    input_text = f"{example['instruction']} {example['input']}"
    input_ids = tokenizer(input_text, truncation=True, padding="max_length", max_length=128)
    target_ids = tokenizer(example["output"], truncation=True, padding="max_length", max_length=64)
    input_ids["labels"] = target_ids["input_ids"]
    return input_ids

# Apply tokenizer to dataset
tokenized_dataset = dataset.map(tokenize)

# Data collator for batching
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./qlora-t5-phrasal",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="no",
    fp16=torch.cuda.is_available(),  # Enable mixed precision if on GPU
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
model.save_pretrained("qlora-t5-phrasal")
tokenizer.save_pretrained("qlora-t5-phrasal")
