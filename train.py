import torch
from torch.utils.data import Dataset
from PIL import Image
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model # <-- Import LoRA components
import wandb

# Import configurations
from config import (
    model_name, dataset_name, image_folder_path, new_model_name,
    lora_r, lora_alpha, lora_dropout, lora_target_modules,
    output_dir, num_train_epochs, per_device_train_batch_size,
    gradient_accumulation_steps, learning_rate, wandb_project_name
)

class VQADataset(Dataset):
    """Custom PyTorch Dataset for VQA with Qwen-VL."""
    def __init__(self, hf_dataset, processor, image_dir):
        self.dataset = hf_dataset
        self.processor = processor
        self.image_dir = image_dir

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        question, answer = row['question'], row['answer']
        image_path = f"{self.image_dir}/{row['image_id']}.jpg"
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            return None

        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"Question: {question} Answer:"}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        return inputs

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return {}
    return processor.tokenizer.pad(batch, return_tensors="pt")

if __name__ == "__main__":
    wandb.init(project=wandb_project_name, config={"model_name": model_name})

    # <-- Define 4-bit quantization configuration -->
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # Load the VLM with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # <-- Define LoRA Configuration -->
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none"
    )

    # <-- Apply LoRA to the model -->
    print("Applying LoRA configuration to the model...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create custom dataset
    hf_dataset = load_dataset(dataset_name, split="train")
    train_dataset = VQADataset(hf_dataset, processor, image_folder_path)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        bf16=True, # bf16 is recommended for 4-bit training
        logging_steps=10,
        report_to="wandb",
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )

    print("Starting LoRA fine-tuning for VLM...")
    trainer.train()
    print("Fine-tuning complete.")

    print(f"Saving LoRA adapters to {new_model_name}")
    model.save_pretrained(new_model_name)
    processor.save_pretrained(new_model_name)
    wandb.finish()