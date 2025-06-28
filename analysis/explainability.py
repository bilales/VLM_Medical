import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config import new_model_name, model_name

def plot_attention(attention_matrix, tokens, file_name="attention_map.png"):
    fig, ax = plt.subplots(figsize=(14, 14))
    im = ax.imshow(attention_matrix, cmap='viridis')
    
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))
    
    ax.set_xticklabels(tokens)
    ax.set_yticklabels(tokens)
    
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    
    fig.tight_layout()
    plt.savefig(file_name)
    print(f"Attention map saved to {file_name}")
    plt.close()

def main():
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base_model, new_model_name).merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    question = "What is the differential diagnosis for chest pain that radiates to the left arm?"
    prompt = f"Question: {question}\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=15, output_attentions=True,
            return_dict_in_generate=True, eos_token_id=tokenizer.eos_token_id
        )

    # Visualize attention from the last generated token to all previous tokens
    last_token_attentions = outputs.attentions[-1]
    last_layer_attention = last_token_attentions[-1][0]
    avg_attention = last_layer_attention.mean(dim=0).cpu().numpy()

    all_tokens = tokenizer.convert_ids_to_tokens(outputs.sequences[0])
    
    print("Visualizing attention for the last generated token...")
    plot_attention(avg_attention, all_tokens)

if __name__ == "__main__":
    main()