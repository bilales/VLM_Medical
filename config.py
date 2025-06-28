

# --- Model and Dataset Configuration ---
model_name = "Qwen/Qwen-VL-Chat"
dataset_name = "mmoukouba/MedPix-VQA"

# --- Image Data Configuration ---
image_folder_path = "data/images"

# --- Fine-Tuning Configuration ---
new_model_name = "qwen-vl-chat-medvqa-lora"

# --- LoRA Configuration ---
# The rank of the LoRA update matrices.
lora_r = 16
# The alpha parameter for LoRA scaling.
lora_alpha = 32
# Dropout probability for LoRA layers.
lora_dropout = 0.05
# Modules to target with LoRA. For Qwen-VL, these are common choices.
lora_target_modules = ["c_attn", "attn.c_proj", "w1", "w2"]

# --- Training Arguments ---
output_dir = "./results-lora"
num_train_epochs = 1
per_device_train_batch_size = 1
gradient_accumulation_steps = 4
learning_rate = 1e-4 # Can often use a slightly higher learning rate with LoRA
max_seq_length = 512

# --- Experiment Tracking ---
wandb_project_name = "qwen-vqa-vlm-lora"