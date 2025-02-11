"""
A comprehensive helper script for evaluating LLMs on the SuperGLUE benchmark.
This script loads a Hugging Face model and runs evaluations on the following tasks:
    - BoolQ
    - CB (CommitmentBank)
    - COPA (Choice of Plausible Alternatives)
    - MultiRC (Multi-Sentence Reading Comprehension)
    - ReCoRD (Reading Comprehension with Commonsense Reasoning)
    - RTE (Recognizing Textual Entailment)
    - WiC (Word-in-Context)
    - WSC (Winograd Schema Challenge)

For each task, the appropriate evaluation metric is used (e.g. CB uses the average of accuracy and macro-F1).
Finally, an overall SuperGLUE score is computed as the (unweighted) average of the eight task scores.
Note: This script assumes that the model is a Hugging Face sequence-classification model.
Some tasks (e.g. COPA, ReCoRD) require custom input formatting and may need adjustments
if your model architecture differs.
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import evaluate

def evaluate_boolq(model, tokenizer, batch_size):
    dataset = load_dataset("super_glue", "boolq", trust_remote_code=True)
    metric = evaluate.load("accuracy")

    def preprocess(examples):
        # BoolQ uses "passage" and "question"
        return tokenizer(examples["passage"], examples["question"],
                         truncation=True, padding="max_length")

    dataset = dataset.map(preprocess, batched=True)
    dataloader = DataLoader(dataset["validation"], batch_size=batch_size, pin_memory=True)

    predictions, labels = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    for batch in dataloader:
        inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        predictions.extend(preds.cpu().numpy())
        labels.extend(batch["label"])

    result = metric.compute(predictions=predictions, references=labels)
    return result  # returns {"accuracy": ...}

def evaluate_cb(model, tokenizer, batch_size):
    dataset = load_dataset("super_glue", "cb", trust_remote_code=True)
    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def preprocess(examples):
        # CB uses "premise" and "hypothesis"
        return tokenizer(examples["premise"], examples["hypothesis"],
                         truncation=True, padding="max_length")

    dataset = dataset.map(preprocess, batched=True)
    dataloader = DataLoader(dataset["validation"], batch_size=batch_size, pin_memory=True)

    predictions, labels = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    for batch in dataloader:
        inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        predictions.extend(preds.cpu().numpy())
        labels.extend(batch["label"])

    acc_result = acc_metric.compute(predictions=predictions, references=labels)
    # For F1 we use macro averaging
    f1_result = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    combined_score = (acc_result["accuracy"] + f1_result["f1"]) / 2.0
    return {"accuracy": acc_result["accuracy"], "f1": f1_result["f1"], "cb_score": combined_score}

def evaluate_copa(model, tokenizer, batch_size):
    dataset = load_dataset("super_glue", "copa", trust_remote_code=True)
    metric = evaluate.load("accuracy")

    def preprocess(example):
        # COPA: each example has "premise", "question", "choice1", "choice2", and "label"
        # We'll create two alternative texts.
        alt1 = example["premise"] + " " + example["question"] + " " + example["choice1"]
        alt2 = example["premise"] + " " + example["question"] + " " + example["choice2"]
        return {"alt1": alt1, "alt2": alt2, "label": example["label"]}

    dataset = dataset.map(preprocess)
    # Due to the structure, process examples one-by-one.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions = []
    labels = []
    for example in dataset["validation"]:
        inputs1 = tokenizer(example["alt1"], truncation=True, padding="max_length",
                              return_tensors="pt").to(device)
        inputs2 = tokenizer(example["alt2"], truncation=True, padding="max_length",
                              return_tensors="pt").to(device)
        with torch.no_grad():
            output1 = model(**inputs1)
            output2 = model(**inputs2)
        # Assume binary classification outputs; compare the logit for class 1
        score1 = output1.logits[0, 1].item()
        score2 = output2.logits[0, 1].item()
        pred = 0 if score1 > score2 else 1
        predictions.append(pred)
        labels.append(example["label"])

    result = metric.compute(predictions=predictions, references=labels)
    return result

def evaluate_multirc(model, tokenizer, batch_size):
    dataset = load_dataset("super_glue", "multirc", trust_remote_code=True)
    # MultiRC is a multi-label task. Each row corresponds to one answer candidate,
    # and rows sharing the same "idx" belong to the same question.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # We'll group predictions and labels by question id.
    preds_by_q = {}
    labels_by_q = {}

    # Process each example individually (batching is tricky due to grouping).
    for example in dataset["validation"]:
        # Combine the passage, question, and answer candidate.
        text = example["passage"] + " " + example["question"] + " " + example["answer"]
        inputs = tokenizer(text, truncation=True, padding="max_length",
                           return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # Assume binary classification; use logit for class 1.
        score = outputs.logits[0, 1].item()
        pred_label = 1 if score > 0 else 0  # simple threshold at 0
        qid = example["idx"]
        preds_by_q.setdefault(qid, []).append(pred_label)
        labels_by_q.setdefault(qid, []).append(example["label"])

    # For each question, compute an exact match (EM) and F1 between the predicted and gold sets.
    ems = []
    f1s = []
    for qid in preds_by_q:
        pred = preds_by_q[qid]
        ref = labels_by_q[qid]
        em = 1.0 if pred == ref else 0.0  # exact match per question
        ems.append(em)
        # Compute F1 for the binary predictions (treating them as sets)
        pred = np.array(pred)
        ref = np.array(ref)
        tp = np.sum((pred == 1) & (ref == 1))
        fp = np.sum((pred == 1) & (ref == 0))
        fn = np.sum((pred == 0) & (ref == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    avg_em = float(np.mean(ems))
    avg_f1 = float(np.mean(f1s))
    combined = (avg_em + avg_f1) / 2.0
    return {"exact_match": avg_em, "f1": avg_f1, "multirc_score": combined}

def evaluate_record(model, tokenizer, batch_size):
    dataset = load_dataset("super_glue", "record", trust_remote_code=True)
    # ReCoRD is a cloze-style reading comprehension task.
    # Each example has a "passage", a "query" containing a placeholder (e.g. "@placeholder"),
    # a list of candidate answers in "candidates", and a list of correct answers in "answers".
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions = []
    references = []
    for example in dataset["validation"]:
        scores = []
        # Iterate over candidate answers.
        for candidate in example["candidates"]:
            # Replace the placeholder in the query with the candidate answer.
            query_filled = example["query"].replace("@placeholder", candidate)
            text = example["passage"] + " " + query_filled
            inputs = tokenizer(text, truncation=True, padding="max_length",
                               return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            score = outputs.logits[0, 1].item()  # assume binary classification
            scores.append(score)
        # Choose the candidate with the highest score.
        best_candidate = example["candidates"][np.argmax(scores)]
        predictions.append(best_candidate)
        # The gold answers is a list; if our predicted candidate is among them, consider it correct.
        references.append(example["answers"])

    # A simplified F1: percentage of examples for which our predicted candidate is in the gold answers.
    correct = sum(1 for pred, ans in zip(predictions, references) if pred in ans)
    f1_score = correct / len(predictions) if predictions else 0.0
    return {"record_f1": f1_score}

def evaluate_rte(model, tokenizer, batch_size):
    dataset = load_dataset("super_glue", "rte", trust_remote_code=True)
    metric = evaluate.load("accuracy")

    def preprocess(examples):
        # RTE uses "premise" and "hypothesis"
        return tokenizer(examples["premise"], examples["hypothesis"],
                         truncation=True, padding="max_length")

    dataset = dataset.map(preprocess, batched=True)
    dataloader = DataLoader(dataset["validation"], batch_size=batch_size, pin_memory=True)

    predictions, labels = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    for batch in dataloader:
        inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        predictions.extend(preds.cpu().numpy())
        labels.extend(batch["label"])

    result = metric.compute(predictions=predictions, references=labels)
    return result

def evaluate_wic(model, tokenizer, batch_size):
    dataset = load_dataset("super_glue", "wic", trust_remote_code=True)
    metric = evaluate.load("accuracy")

    def preprocess(examples):
        # WiC: two sentences where a word is used in different contexts.
        return tokenizer(examples["sentence1"], examples["sentence2"],
                         truncation=True, padding="max_length")

    dataset = dataset.map(preprocess, batched=True)
    dataloader = DataLoader(dataset["validation"], batch_size=batch_size, pin_memory=True)

    predictions, labels = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    for batch in dataloader:
        inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        predictions.extend(preds.cpu().numpy())
        labels.extend(batch["label"])

    result = metric.compute(predictions=predictions, references=labels)
    return result

def evaluate_wsc(model, tokenizer, batch_size):
    # Note: In the Hugging Face hub the fixed version is usually named "wsc.fixed"
    dataset = load_dataset("super_glue", "wsc.fixed", trust_remote_code=True)
    metric = evaluate.load("accuracy")

    def preprocess(examples):
        # WSC: typically the whole text (with pronoun and candidate spans) is provided in "text"
        return tokenizer(examples["text"], truncation=True, padding="max_length")

    dataset = dataset.map(preprocess, batched=True)
    dataloader = DataLoader(dataset["validation"], batch_size=batch_size, pin_memory=True)

    predictions, labels = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    for batch in dataloader:
        inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        predictions.extend(preds.cpu().numpy())
        labels.extend(batch["label"])

    result = metric.compute(predictions=predictions, references=labels)
    return result

def evaluate_on_glue(model_file, batch_size):
    # Load the model and tokenizer
    print(f"Loading model and tokenizer from {model_file} ...")
    model = AutoModelForSequenceClassification.from_pretrained(model_file)
    tokenizer = AutoTokenizer.from_pretrained(model_file)

    results = {}
    print("Evaluating BoolQ...")
    results["boolq"] = evaluate_boolq(model, tokenizer, batch_size)

    print("Evaluating CB...")
    results["cb"] = evaluate_cb(model, tokenizer, batch_size)

    print("Evaluating COPA...")
    results["copa"] = evaluate_copa(model, tokenizer, batch_size)

    print("Evaluating MultiRC...")
    results["multirc"] = evaluate_multirc(model, tokenizer, batch_size)

    print("Evaluating ReCoRD...")
    results["record"] = evaluate_record(model, tokenizer, batch_size)

    print("Evaluating RTE...")
    results["rte"] = evaluate_rte(model, tokenizer, batch_size)

    print("Evaluating WiC...")
    results["wic"] = evaluate_wic(model, tokenizer, batch_size)

    print("Evaluating WSC...")
    results["wsc"] = evaluate_wsc(model, tokenizer, batch_size)

    # For overall SuperGLUE score, use the official task scores:
    # - For tasks evaluated via accuracy (BoolQ, COPA, RTE, WiC, WSC) use accuracy.
    # - For CB, use the composite "cb_score".
    # - For MultiRC, use "multirc_score".
    # - For ReCoRD, use "record_f1".
    scores = []
    scores.append(results["boolq"]["accuracy"])
    scores.append(results["cb"]["cb_score"])
    scores.append(results["copa"]["accuracy"])
    scores.append(results["multirc"]["multirc_score"])
    scores.append(results["record"]["record_f1"])
    scores.append(results["rte"]["accuracy"])
    scores.append(results["wic"]["accuracy"])
    scores.append(results["wsc"]["accuracy"])

    overall_score = sum(scores) / len(scores)
    results["superglue_overall_score"] = overall_score
    return results
