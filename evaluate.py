# qwen-vqa-vlm-lora/evaluate.py
import torch
from PIL import Image
from datasets import load_dataset, load_metric
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel # <-- Import PeftModel to load adapters
from tqdm import tqdm

from config import model_name, new_model_name, dataset_name, image_folder_path

def main():
    print("Loading base model and processor...")
    # Load the base model in its original precision
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # <-- Load the LoRA adapters onto the base model -->
    print(f"Loading LoRA adapters from {new_model_name}...")
    model = PeftModel.from_pretrained(base_model, new_model_name)
    model.eval()

    # Load dataset and metrics
    hf_dataset = load_dataset(dataset_name, split="test")
    bleu_metric = load_metric("sacrebleu")
    rouge_metric = load_metric("rouge")

    predictions, references = [], []
    print("Generating predictions on the test set...")
    for i in tqdm(range(len(hf_dataset))):
        row = hf_dataset[i]
        image_path = f"{image_folder_path}/{row['image_id']}.jpg"
        
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            continue

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"Question: {row['question']} Answer:"}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
        
        response = processor.decode(outputs[0], skip_special_tokens=False)
        try:
            answer = response.rsplit("Answer:")[1].strip().replace("</s>", "")
        except IndexError:
            answer = ""
            
        predictions.append(answer)
        references.append(row['answer'])

    print("Computing metrics...")
    bleu_score = bleu_metric.compute(predictions=predictions, references=[[ref] for ref in references])
    rouge_score = rouge_metric.compute(predictions=predictions, references=references)

    print("\n----- VLM LoRA Evaluation Results -----")
    print(f"  BLEU Score: {bleu_score['score']:.2f}")
    print(f"  ROUGE-L F-measure: {rouge_score['rougeL'].mid.fmeasure * 100:.2f}")
    print("--------------------------------------\n")

if __name__ == "__main__":
    main()