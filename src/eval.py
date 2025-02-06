import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

def load_model(model_name="Qwen/Qwen2.5-1.5B"):
    """Load the Qwen2.5-1.5B model with quantization."""
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def evaluate_model(model, tokenizer, dataset_name):
    """Evaluate the model on a given dataset."""
    dataset = load_dataset(dataset_name)
    results = []
    
    for example in dataset["test"]:
        input_text = example["question"]
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=100)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({"input": input_text, "output": generated_text})
    
    return results

def main():
    model, tokenizer = load_model()
    benchmarks = {
        "En-MMLU": "mmlu",
        "Ru-MMLU": "ru_mmlu",
        "CheGeKa": "chegeka",
        "SuperGLUE": "super_glue",
        "MERA": "mera"
    }
    
    for name, dataset in benchmarks.items():
        print(f"Evaluating on {name}...")
        results = evaluate_model(model, tokenizer, dataset)
        print(f"Results for {name}: {results[:5]}")  # Print first 5 results as a sample

if __name__ == "__main__":
    main()
