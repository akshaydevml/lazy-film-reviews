import argparse
import os
import random
from typing import Dict

import numpy as np
import torch
from datasets import load_dataset, load_metric
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          EvalPrediction, Trainer, TrainingArguments)


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model_and_tokenizer(model="funnel-transformer/small"):
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model, num_labels=2, max_position_embeddings=1024)
    print("Total model parameters", model.num_parameters())

    return model, tokenizer


def compute_metrics(p: EvalPrediction) -> Dict:
    accuracy = load_metric("accuracy")
    precision = load_metric("precision")
    recall = load_metric("recall")
    f1 = load_metric("f1")
    preds = np.argmax(p.predictions, axis=1)

    metrics = dict()
    metrics["accuracy"] = accuracy.compute(predictions=preds,
                                           references=p.label_ids)["accuracy"]
    metrics["precision"] = precision.compute(
        predictions=preds, references=p.label_ids)["precision"]
    metrics["recall"] = recall.compute(predictions=preds,
                                       references=p.label_ids)["recall"]
    metrics["f1"] = f1.compute(predictions=preds, references=p.label_ids)["f1"]

    return metrics


parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed_val', help='Set a seed value', default=42)
parser.add_argument('--batch_size', help='Training batch size', default=32)
parser.add_argument('--eval_batch_size',
                    help='Validation batch size',
                    default=32)
parser.add_argument('--num_train_epochs', help='Training epochs', default=2)
parser.add_argument('--disable_fp16',
                    default=False,
                    action="store_true",
                    help='Disable fp16 training')
parser.add_argument('csvfile', help='Input csv file')
args = parser.parse_args()

seed_all(args.seed_val)

train_test_ds = load_dataset('csv', data_files=args.csvfile, split="train")
print("Dataset format:\n", train_test_ds)
train_test_ds = train_test_ds.rename_column("sentiment", "label")

str_to_int = {"negative": 0, "positive": 1}
train_test_ds = train_test_ds.map(lambda e: {
    "review": e["review"],
    "label": str_to_int[e["label"]]
})
train_test_ds = train_test_ds.train_test_split(test_size=0.3)

training_args = TrainingArguments(
    output_dir="checkpoints",
    num_train_epochs=args.num_train_epochs,
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    load_best_model_at_end=True,
    dataloader_drop_last=True,
    fp16=not args.disable_fp16,
    eval_steps=500,
    save_total_limit=1,
)

model, tokenizer = create_model_and_tokenizer()

train_ds = train_test_ds["train"].map(
    lambda e: tokenizer(e['review'], truncation=True, padding=True),
    batched=True)
train_ds.set_format(type='torch',
                    columns=['input_ids', 'attention_mask', 'label'])
val_ds = train_test_ds["test"].map(
    lambda e: tokenizer(e['review'], truncation=True, padding=True),
    batched=True)
val_ds.set_format(type='torch',
                  columns=['input_ids', 'attention_mask', 'label'])

trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=train_ds,
                  eval_dataset=val_ds,
                  tokenizer=tokenizer,
                  compute_metrics=compute_metrics)
trainer.train()
