import argparse
from pathlib import Path
from collections import Counter
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets import load_dataset, Dataset
from sklearn.metrics import average_precision_score
import evaluate

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    PreTrainedModel,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForTokenClassification,
    set_seed,
)
from transformers.modeling_outputs import TokenClassifierOutput

from dataclasses import dataclass
from typing import Any, Dict, List


# SEED

SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def load_jsonl(path: str) -> Dataset:
    return load_dataset("json", data_files=str(path), split="train")


def build_vocab(dataset: Dataset, tag: str, pad_token="[PAD]", unk_token="[UNK]"):
    values = set(v for row in dataset[tag] for v in row)
    vocab = {pad_token: 0, unk_token: 1}
    for v in sorted(values):
        if v not in vocab:
            vocab[v] = len(vocab)
    return vocab


# Custom model: Roberta based model + lemma embeddings (concat) → token classifier

class RobertaBasedModelForTokenClassification(PreTrainedModel):
    config_class = AutoConfig

    def __init__(
        self,
        config,
        base_model_name,
        num_lemmas,
        lemma_init_matrix,
        num_pos,
        num_labels,
        lemma_pad_id,
        pos_pad_id,
        dropout=0.3,):

        super().__init__(config)
        self.num_labels = num_labels

        self.encoder = AutoModel.from_pretrained(base_model_name, config=config)
        hidden = self.encoder.config.hidden_size

        self.lemma_emb = nn.Embedding(num_lemmas, hidden, padding_idx=lemma_pad_id)
        self.lemma_emb.weight.data.copy_(lemma_init_matrix)

        self.pos_emb = nn.Embedding(num_pos, hidden, padding_idx=pos_pad_id)
        self.nfv_emb = nn.Embedding(2, hidden)  # 0/1

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden * 4, num_labels)

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        lemma_ids_aligned=None,
        pos_ids_aligned=None,
        non_finite_verb_aligned=None):

        
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        H = enc.last_hidden_state
        L = self.lemma_emb(lemma_ids_aligned)
        P = self.pos_emb(pos_ids_aligned)
        NFV = self.nfv_emb(non_finite_verb_aligned)

        X = torch.cat([H, L, NFV, P], dim=-1)
        X = self.dropout(X)
        logits = self.classifier(X)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=enc.hidden_states,
            attentions=enc.attentions,
        )


# Data Collator with features

@dataclass
class DataCollatorForTokenClassificationWithFeatures:
    tokenizer: Any
    lemma_pad_id: int
    pos_pad_id: int
    nfv_pad_id: int = 0

    def __post_init__(self):
        self.base = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        lemma = [f.pop("lemma_ids_aligned") for f in features]
        pos   = [f.pop("pos_ids_aligned") for f in features]
        nfv   = [f.pop("non_finite_verb_aligned") for f in features]

        base = self.base(features)
        max_len = base["input_ids"].shape[1]

        def pad_to(seq, pad_value):
            return seq + [pad_value] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len]

        base["lemma_ids_aligned"] = torch.tensor([pad_to(x, self.lemma_pad_id) for x in lemma], dtype=torch.long)
        base["pos_ids_aligned"] = torch.tensor([pad_to(x, self.pos_pad_id) for x in pos], dtype=torch.long)
        base["non_finite_verb_aligned"] = torch.tensor([pad_to(x, self.nfv_pad_id) for x in nfv], dtype=torch.long)

        return base

# Evaluation metrics

accuracy_m = evaluate.load("accuracy")
precision_m = evaluate.load("precision")
recall_m = evaluate.load("recall")
f1_m = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    mask = labels != -100
    y_true = labels[mask].astype(int)
    y_pred = preds[mask].astype(int)

    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    y_score = probs[..., 1][mask]

    return {
        "accuracy": accuracy_m.compute(predictions=y_pred, references=y_true)["accuracy"],
        "precision": precision_m.compute(predictions=y_pred, references=y_true, average="binary")["precision"],
        "recall": recall_m.compute(predictions=y_pred, references=y_true, average="binary")["recall"],
        "f1": f1_m.compute(predictions=y_pred, references=y_true, average="binary")["f1"],
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }


# Main function

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True, help='e.g. "bowphs/LaBerta"')
    ap.add_argument("--train_file", default="data/train.jsonl")
    ap.add_argument("--val_file", default="data/val.jsonl")
    ap.add_argument("--test_file", default="data/test.jsonl")
    ap.add_argument("--output_dir", default="outputs/run")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    args = ap.parse_args()

    train_dataset = load_jsonl(args.train_file)
    val_dataset   = load_jsonl(args.val_file)
    test_dataset  = load_jsonl(args.test_file)

    # vocabs
    pos2id = build_vocab(train_dataset, "pos")

    lemma_counter = Counter(l for ex in train_dataset["lemmas"] for l in ex)
    lemma2id = {"[PAD]": 0, "[UNK]": 1}
    for lemma, _ in lemma_counter.most_common():
        if lemma not in lemma2id:
            lemma2id[lemma] = len(lemma2id)

    def add_feature_ids(batch):
        batch["pos_ids"] = [[pos2id.get(p, pos2id["[UNK]"]) for p in row] for row in batch["pos"]]
        batch["lemma_ids"] = [[lemma2id.get(l, lemma2id["[UNK]"]) for l in row] for row in batch["lemmas"]]
        return batch

    train_dataset = train_dataset.map(add_feature_ids, batched=True)
    val_dataset   = val_dataset.map(add_feature_ids, batched=True)
    test_dataset  = test_dataset.map(add_feature_ids, batched=True)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)

    MAX_LEN = 512
    STRIDE = 128

    def tokenize_and_align_everything(batch):
        tokenized = tokenizer(
            batch["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LEN,
            stride=STRIDE,
            return_overflowing_tokens=True,
            padding=False,
        )

        sample_mapping = tokenized.pop("overflow_to_sample_mapping")

        aligned_labels, aligned_lemmas, aligned_pos, aligned_nfv, all_word_ids = [], [], [], [], []
        out_micro, out_sent, out_win = [], [], []

        for win_idx, orig_i in enumerate(sample_mapping):
            word_ids = tokenized.word_ids(batch_index=win_idx)
            all_word_ids.append(word_ids)

            labels_i = batch["labels"][orig_i]
            lemma_i  = batch["lemma_ids"][orig_i]
            pos_i    = batch["pos_ids"][orig_i]
            nfv_i    = batch["non_finite_verb"][orig_i]

            lab, lem, pos, nfv = [], [], [], []
            prev_word = None

            for w in word_ids:
                if w is None:
                    lab.append(-100)
                    lem.append(lemma2id["[PAD]"])
                    pos.append(pos2id["[PAD]"])
                    nfv.append(0)
                else:
                    lab.append(labels_i[w] if w != prev_word else -100)
                    prev_word = w

                    lem.append(lemma_i[w])
                    pos.append(pos_i[w])
                    nfv.append(int(nfv_i[w]))

            aligned_labels.append(lab)
            aligned_lemmas.append(lem)
            aligned_pos.append(pos)
            aligned_nfv.append(nfv)

            out_micro.append(batch["microsection"][orig_i])
            out_sent.append(batch["sentence_id"][orig_i])
            out_win.append(win_idx)

        tokenized["labels"] = aligned_labels
        tokenized["lemma_ids_aligned"] = aligned_lemmas
        tokenized["pos_ids_aligned"] = aligned_pos
        tokenized["non_finite_verb_aligned"] = aligned_nfv
        tokenized["word_ids"] = all_word_ids
        tokenized["microsection"] = out_micro
        tokenized["sentence_id"] = out_sent
        tokenized["window_id"] = out_win
        return tokenized

    tokenized_train = train_dataset.map(tokenize_and_align_everything, batched=True, remove_columns=train_dataset.column_names)
    tokenized_val   = val_dataset.map(tokenize_and_align_everything, batched=True, remove_columns=val_dataset.column_names)
    tokenized_test  = test_dataset.map(tokenize_and_align_everything, batched=True, remove_columns=test_dataset.column_names)

    cols = [
        "input_ids", "attention_mask", "labels",
        "lemma_ids_aligned", "pos_ids_aligned", "non_finite_verb_aligned",
        "word_ids", "microsection", "sentence_id", "window_id",
    ]
    tokenized_train = tokenized_train.remove_columns([c for c in tokenized_train.column_names if c not in cols])
    tokenized_val   = tokenized_val.remove_columns([c for c in tokenized_val.column_names if c not in cols])
    tokenized_test  = tokenized_test.remove_columns([c for c in tokenized_test.column_names if c not in cols])

    # lemma init from subtoken embeddings
    base_model = AutoModel.from_pretrained(args.model_name)
    tok_emb = base_model.get_input_embeddings().weight.detach().cpu()
    hidden_size = tok_emb.shape[1]

    def lemma_to_init_vector(lemma: str) -> torch.Tensor:
        ids = tokenizer.encode(lemma, add_special_tokens=False)
        if len(ids) == 0:
            ids = [tokenizer.unk_token_id]
        vecs = tok_emb[torch.tensor(ids, dtype=torch.long)]
        return vecs.mean(dim=0)

    lemma_init = torch.zeros(len(lemma2id), hidden_size)
    lemma_init[lemma2id["[UNK]"]] = tok_emb[tokenizer.unk_token_id]
    for lemma, lid in lemma2id.items():
        if lemma in ("[PAD]", "[UNK]"):
            continue
        lemma_init[lid] = lemma_to_init_vector(lemma)

    # model + collator
    config = AutoConfig.from_pretrained(args.model_name, num_labels=2)

    model = RobertaBasedModelForTokenClassification(
        config=config,
        base_model_name=args.model_name,
        num_lemmas=len(lemma2id),
        lemma_init_matrix=lemma_init,
        num_pos=len(pos2id),
        num_labels=2,
        lemma_pad_id=lemma2id["[PAD]"],
        pos_pad_id=pos2id["[PAD]"],
        dropout=0.3,
    )

    data_collator = DataCollatorForTokenClassificationWithFeatures(
        tokenizer=tokenizer,
        lemma_pad_id=lemma2id["[PAD]"],
        pos_pad_id=pos2id["[PAD]"],
        nfv_pad_id=0,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        overwrite_output_dir=True,
        seed=SEED,
        data_seed=SEED,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        logging_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    test_metrics = trainer.evaluate(eval_dataset=tokenized_test, metric_key_prefix="test")
    print(test_metrics)


if __name__ == "__main__":
    main()
