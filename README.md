# Lightweight-Fine-Tuning-with-PEFT
This repository contains an implementation of Parameter-Efficient Fine-Tuning (PEFT) using LoRA (Low-Rank Adaptation) for text classification. The project fine-tunes GPT-2 on the IMDb dataset and evaluates performance before and after fine-tuning.

Project Overview
1. Load and evaluate a pre-trained foundation model (GPT-2) for sequence classification.

2. Apply PEFT using LoRA to efficiently fine-tune the model.

3. Train the LoRA-enhanced model on the IMDb dataset.

4. Save the fine-tuned model to a specified directory.

5. Load the saved PEFT model and evaluate performance against the baseline.

🏗️ Implementation Details
✅ Pre-Trained Model
Model Used: GPT-2 (small model for efficiency).

Task: Binary sequence classification (IMDb sentiment analysis).

Dataset: IMDb dataset from Hugging Face (datasets library).

🔧 Fine-Tuning with LoRA (PEFT)
LoRA Rank (r): 8

Alpha (lora_alpha): 32

Dropout (lora_dropout): 0.1

🔍 Evaluation Metrics
Baseline Model: Evaluated before fine-tuning.

Fine-Tuned Model: Evaluated after applying PEFT.

Key Metric: eval_accuracy (computed using Hugging Face Trainer).

 Project Structure
📦 LightweightFineTuning
│-- 📄 LightweightFineTuning.py  # Main script for loading, fine-tuning, and evaluating the model
│-- 📂 tmp/peft_gpt2_lora/       # Saved fine-tuned model 
│-- 📜 README.md                 # Project documentation
