import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def format_mmlu_prompt(example):
    """Formats the MMLU prompt as a question with multiple choice answers."""
    prompt = example['question'] + "\n"
    for i in range(4):
        prompt += f"{chr(ord('A') + i)}. {example['choices'][i]}\n"
    prompt += "Answer:"
    return prompt

def predict(model, tokenizer, prompt, device):
    """Predicts the answer using the given model and tokenizer."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1)  # Generate only one token
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split()[-1]  # Return the last token

def evaluate_mmlu(model_name, device):
    """Evaluates the model on the MMLU dataset."""
    dataset = load_dataset("luka-strub/mmlu", "all")['test']

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    correct_predictions = 0
    total_samples = len(dataset)

    for example in tqdm(dataset, desc="Evaluating"):
        prompt = format_mmlu_prompt(example)
        predicted_answer = predict(model, tokenizer, prompt, device)
        
        # The correct answer is a single letter (A, B, C, or D)
        correct_answer = chr(ord('A') + example['answer'])

        if predicted_answer.strip().upper() == correct_answer:
            correct_predictions += 1

    accuracy = correct_predictions / total_samples
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on the MMLU dataset.")
    parser.add_argument("--model_name", type=str, required=True, help="The name of the model to evaluate (Hugging Face model hub).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="The device to run the evaluation on (cuda or cpu).")
    args = parser.parse_args()

    accuracy = evaluate_mmlu(args.model_name, args.device)
    print(f"Accuracy: {accuracy:.4f}")
