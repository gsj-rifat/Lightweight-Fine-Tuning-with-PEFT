
# # Lightweight Fine-Tuning Project

# * PEFT technique: 
# * Model: 
# * Evaluation approach: 
# * Fine-tuning dataset: 

# ## Loading and Evaluating a Foundation Model

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import torch
import numpy as np
from transformers import Trainer, TrainingArguments



MODEL_NAME = "gpt2"  # Using GPT-2 (small size)
NUM_LABELS = 2  # Modify based on dataset (binary classification)
# Load pre-trained model with classification head
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# GPT-2 does not have a padding token by default, so we define one
tokenizer.pad_token = tokenizer.eos_token


DATASET_NAME = "imdb"  # Choose a text classification dataset
dataset = load_dataset(DATASET_NAME)

# Split into train/test
train_dataset = dataset["train"]
test_dataset = dataset["test"]
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

# Apply tokenization
tokenized_datasets = dataset.map(preprocess_function, batched=True)


# Set padding token for GPT-2
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Define compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred  # Extract logits and labels
    predictions = np.argmax(logits, axis=1)  # Get predicted class
    return {"accuracy": np.mean(predictions == labels)}  # Compute accuracy

# Define training arguments (for evaluation only)
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_eval_batch_size=8
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics
)

# Evaluate baseline performance
baseline_results = trainer.evaluate()
print("Baseline Performance:", baseline_results)



# ## Performing Parameter-Efficient Fine-Tuning
# create a PEFT model from your loaded model, run a training loop, and save the PEFT model weights.


from peft import LoraConfig, get_peft_model, TaskType

# Configure LoRA parameters
lora_config = LoraConfig(
    r=8,  # LoRA rank
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,  # Dropout rate
    task_type=TaskType.SEQ_CLS  # Sequence classification
)

# Wrap the base model with LoRA
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()


training_args = TrainingArguments(
    output_dir="./peft_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    push_to_hub=False
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics
)


# Train model
trainer.train()


# Saving the model
#model.save("/tmp/peft_gpt2_lora")
peft_model.save_pretrained("./tmp/peft_gpt2_lora")
tokenizer.save_pretrained("./tmp/peft_gpt2_lora")


# ## Performing Inference with a PEFT Model
# 
#load the saved PEFT model weights and evaluate the performance of the trained PEFT model. Be sure to compare the results to the results from prior to fine-tuning.


from peft import AutoPeftModelForSequenceClassification
# Define model path
MODEL_PATH = "./tmp/peft_gpt2_lora"

# Load the base GPT-2 model
base_model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=2)

# Load the fine-tuned LoRA model on top of it
peft_model = AutoPeftModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
peft_model.config.pad_token_id = tokenizer.pad_token_id


# Evaluate fine-tuned model
fine_tuned_results = trainer.evaluate()
print("Fine-Tuned Model Performance:", fine_tuned_results)

# Compare baseline vs fine-tuned accuracy
print("Baseline Accuracy:", baseline_results["eval_accuracy"])
print("Fine-Tuned Accuracy:", fine_tuned_results["eval_accuracy"])


