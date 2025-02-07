from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

BENCHMARKS = [
    {"name": "LiDiRus", "dataset_id": "lidirus"},
    {"name": "RCB", "dataset_id": "rcb"},
    {"name": "PARus", "dataset_id": "parus"},
    {"name": "MuSeRC", "dataset_id": "muserc"},
    {"name": "TERRa", "dataset_id": "terra"},
    {"name": "RUSSE", "dataset_id": "russe"},
    {"name": "RWSD", "dataset_id": "rwsd"},
    {"name": "DaNetQA", "dataset_id": "danetqa"},
    {"name": "RuCoS", "dataset_id": "rucos"}
]

def load_model(model_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    return model, tokenizer, device

def process_example(example, dataset_name):
    if dataset_name == "LiDiRus":
        question = example["question"]
        choices = [example["choice1"], example["choice2"], example["choice3"], example["choice4"]]
        answer = ord(example["correct"]) - ord("A")
        return {"question": question, "choices": choices, "answer": answer}
    
    elif dataset_name == "RCB":
        return {
            "premise": example["premise"],
            "hypothesis": example["hypothesis"],
            "label": example["label"]
        }
    
    elif dataset_name == "RUSSE":
        return {
            "sentence1": example["sentence1"],
            "sentence2": example["sentence2"],
            "label": example["label"]
        }
    
    # Add processing for other datasets here
    else:
        return example

def evaluate_model(model, tokenizer, device, benchmark):
    dataset = load_dataset(benchmark["dataset_id"])
    results = []
    
    # Adjust split based on dataset availability
    split = "test" if "test" in dataset else "validation"
    dataset = dataset[split]
    
    for example in dataset:
        processed = process_example(example, benchmark["name"])
        
        if benchmark["name"] in ["LiDiRus", "RCB", "RUSSE"]:
            # Multiple-choice or classification task
            if benchmark["name"] == "LiDiRus":
                inputs = [f"{processed['question']} {choice}" for choice in processed["choices"]]
            elif benchmark["name"] == "RCB":
                inputs = [f"Premise: {processed['premise']} Hypothesis: {processed['hypothesis']}"]
            elif benchmark["name"] == "RUSSE":
                inputs = [f"Sentence1: {processed['sentence1']} Sentence2: {processed['sentence2']}"]
            
            # Calculate perplexity for each choice
            scores = []
            for input_text in inputs:
                inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs["input_ids"])
                scores.append(-outputs.loss.item())
            
            predicted = np.argmax(scores)
            results.append(predicted == processed["answer"])
        
        # Add evaluation logic for other datasets here
        else:
            # Default generation task
            input_text = processed["question"] if "question" in processed else processed["text"]
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
            
            outputs = model.generate(
                inputs["input_ids"],
                max_length=512,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Add dataset-specific answer extraction logic here
            results.append(generated.strip() == processed["answer"].strip())
    
    accuracy = np.mean(results)
    return accuracy

def main(model_id):
    model, tokenizer, device = load_model(model_id)
    
    print(f"Evaluating model: {model_id}")
    print("==================================")
    
    for benchmark in BENCHMARKS:
        print(f"Evaluating on {benchmark['name']}...")
        accuracy = evaluate_model(model, tokenizer, device, benchmark)
        print(f"{benchmark['name']} Accuracy: {accuracy:.4f}")
        print("----------------------------------")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model identifier")
    args = parser.parse_args()
    
    main(args.model_id)
