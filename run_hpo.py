import optuna
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
import wandb

from config import model_name, dataset_name, max_seq_length, wandb_hpo_project_name

def objective(trial: optuna.Trial):
    # --- Hyperparameter Search Space ---
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    lora_r = trial.suggest_categorical("lora_r", [16, 32, 64])
    lora_alpha = trial.suggest_categorical("lora_alpha", [lora_r * 2, lora_r * 4])
    # ---

    with wandb.init(project=wandb_hpo_project_name, config=trial.params, reinit=True) as run:
        # Use a small subset of the data for faster HPO
        dataset = load_dataset(dataset_name, split="train").select(range(500))
        def format_dataset(ex): return {"text": f"Question: {ex['question']}\nAnswer: {ex['answer']}"}
        dataset = dataset.map(format_dataset)

        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        peft_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.1, task_type="CAUSAL_LM")
        training_args = TrainingArguments(
            output_dir=f"./results_trial_{trial.number}", per_device_train_batch_size=2,
            gradient_accumulation_steps=2, learning_rate=learning_rate, num_train_epochs=1,
            bf16=True, logging_steps=10, report_to="wandb", save_strategy="no"
        )

        trainer = SFTTrainer(model=model, train_dataset=dataset, peft_config=peft_config,
                             dataset_text_field="text", max_seq_length=max_seq_length,
                             tokenizer=tokenizer, args=training_args)
        
        result = trainer.train()
        final_loss = result.training_loss
        wandb.log({"final_loss": final_loss})
        
        return final_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10) # Adjust n_trials as needed

    print("HPO Study Complete.")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Loss: {trial.value}")
    print("  Best Parameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")