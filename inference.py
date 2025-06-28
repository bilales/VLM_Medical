import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from config import new_model_name, model_name

def main():
    print("Loading base model and tokenizer...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading fine-tuned LoRA weights from {new_model_name}...")
    model = PeftModel.from_pretrained(base_model, new_model_name)
    model = model.merge_and_unload()
    model.eval()
    print("Model loaded successfully.")

    question = "What are the common side effects of metformin?"
    prompt = f"Question: {question}\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    print("\nGenerating answer...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=75,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_text.split("Answer:")[1].strip()

    print("-" * 20)
    print(f"Question: {question}")
    print(f"Generated Answer: {answer}")
    print("-" * 20)

if __name__ == "__main__":
    main()