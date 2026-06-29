import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer

# Configuration Parameters
model_id = "unsloth/llama-3-8b-Instruct-bnb-4bit"  # Clean 4-bit baseline weights
output_dir = "./models/medical_llama3_windows"

# 1. Setup 4-bit quantization configuration for Windows bitsandbytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 2. Load Tokenizer and Base Model
print("[!] Loading model and tokenizer weights...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# 3. Prepare Model for Gradient Checkpointing and apply LoRA configuration
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

# 4. Dataset Template Formatting
alpaca_prompt = """Below is an instruction that describes a medical problem. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs      = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        texts.append(alpaca_prompt.format(instruction, output))
    return { "text" : texts }

dataset = load_dataset("json", data_files="data/medical_training_dataset.json", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# 5. Training Arguments
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=30,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=1,
    output_dir=output_dir,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    report_to="none"
)

# 6. Initialize Trainer and Start Training
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    args=training_args,
)

print("[!] Launching local training loop on Windows...")
trainer.train()

# 7. Save Adapter Modules Locally
model.save_pretrained("./models/Medical-Llama-3-Adapter")
tokenizer.save_pretrained("./models/Medical-Llama-3-Adapter")
print("✓ Custom Medical-Llama-3 adapter weights successfully saved inside your models/ directory.")
