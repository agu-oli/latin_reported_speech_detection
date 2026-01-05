import os
import json
import random
import unicodedata
from collections import Counter

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from tensor2tensor.data_generators import text_encoder

from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    PreTrainedModel,
    set_seed,
)
from transformers.modeling_outputs import TokenClassifierOutput

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, average_precision_score

import argparse

# SEED (fixed in script)

SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# Latin Bert vocab conventions


SPECIAL = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
OFFSET = 5
UNK_ID = SPECIAL["[UNK]"]


def sanitize_token(w: str) -> str:
    w = unicodedata.normalize("NFKC", w)
    w = " ".join(w.split())
    return w

# encode words to subwords

def encode_words_to_subwords(words, encoder):
    input_ids = [SPECIAL["[CLS]"]]
    word_ids = [None]

    for w_i, w in enumerate(words):
        w = sanitize_token(str(w))
        try:
            sub_ids = encoder.encode(w)
        except AssertionError:
            sub_ids = []

        if len(sub_ids) == 0:
            input_ids.append(UNK_ID)
            word_ids.append(w_i)
            continue

        for sid in sub_ids:
            input_ids.append(sid + OFFSET)
            word_ids.append(w_i)

    input_ids.append(SPECIAL["[SEP]"])
    word_ids.append(None)

    attention_mask = [1] * len(input_ids)
    return input_ids, attention_mask, word_ids

# convert dataframe to examples

def df_to_examples(df: pd.DataFrame):
    examples = []
    for microsection, g in df.groupby("microsection", sort=False):
        examples.append(
            {
                "microsection": microsection,
                "sentence_id": g["sentence_id"].iloc[0],
                "tokens": g["form"].astype(str).tolist(),
                "lemmas": g["lemma"].astype(str).tolist(),
                "labels": g["target"].astype(int).tolist(),
                "pos": g["upos"].astype(str).tolist(),
                "non_finite_verb": g["non_finite_verb"].astype(int).tolist(),
            }
        )
    return examples

# build vocab function

def build_vocab(dataset: Dataset, tag: str, pad_token="[PAD]", unk_token="[UNK]"):
    values = set(v for row in dataset[tag] for v in row)
    vocab = {pad_token: 0, unk_token: 1}
    for v in sorted(values):
        if v not in vocab:
            vocab[v] = len(vocab)
    return vocab

# feature ids function: for PoS and lemmas

def add_feature_ids_factory(pos2id, lemma2id):
    def add_feature_ids(batch):
        batch["pos_ids"] = [[pos2id.get(p, pos2id["[UNK]"]) for p in row] for row in batch["pos"]]
        batch["lemma_ids"] = [[lemma2id.get(l, lemma2id["[UNK]"]) for l in row] for row in batch["lemmas"]]
        return batch

    return add_feature_ids

# sliding window params

MAX_LEN = 512
STRIDE = 128

# sliding window function

def make_windows(input_ids, attention_mask, word_ids, max_len=MAX_LEN, stride=STRIDE):
    cls_id, sep_id = input_ids[0], input_ids[-1]

    content_ids = input_ids[1:-1]
    content_mask = attention_mask[1:-1]
    content_wids = word_ids[1:-1]

    max_content = max_len - 2

    windows = []
    start = 0
    while start < len(content_ids):
        end = min(start + max_content, len(content_ids))

        ids_w = [cls_id] + content_ids[start:end] + [sep_id]
        mask_w = [1] + content_mask[start:end] + [1]
        wids_w = [None] + content_wids[start:end] + [None]

        windows.append((ids_w, mask_w, wids_w))

        if end == len(content_ids):
            break
        start = end - stride

    return windows

# tokenize and align function (with features)

def tokenize_and_align_everything_factory(encoder): # encoder: text_encoder.SubwordTextEncoder is defined in main() 
                                                    # and not as global variable 
    
    def tokenize_and_align_everything(batch):
        out_input_ids = []
        out_attention = []
        out_labels = []
        out_word_ids = []

        out_microsection = []
        out_sentence_id = []
        out_window_id = []

        out_lemma = []
        out_pos = []
        out_nfv = []

        for i in range(len(batch["tokens"])):
            words = batch["tokens"][i]
            labels_word = batch["labels"][i]

            lemma_word = batch["lemma_ids"][i]
            pos_word = batch["pos_ids"][i]
            nfv_word = batch["non_finite_verb"][i]

            input_ids, attention_mask, word_ids = encode_words_to_subwords(words, encoder)
            windows = make_windows(input_ids, attention_mask, word_ids)

            for widx, (ids_w, mask_w, wids_w) in enumerate(windows):
                lab_w = []
                lemma_w = []
                pos_w = []
                nfv_w = []

                prev_word = None
                for wid in wids_w:
                    if wid is None:
                        lab_w.append(-100)
                        lemma_w.append(0)
                        pos_w.append(0)
                        nfv_w.append(0)
                    else:
                        if wid != prev_word:
                            lab_w.append(int(labels_word[wid]))
                        else:
                            lab_w.append(-100)
                        prev_word = wid

                        lemma_w.append(int(lemma_word[wid]))
                        pos_w.append(int(pos_word[wid]))
                        nfv_w.append(int(nfv_word[wid]))

                out_input_ids.append(ids_w)
                out_attention.append(mask_w)
                out_labels.append(lab_w)
                out_word_ids.append(wids_w)

                out_lemma.append(lemma_w)
                out_pos.append(pos_w)
                out_nfv.append(nfv_w)

                out_microsection.append(batch["microsection"][i])
                out_sentence_id.append(batch["sentence_id"][i])
                out_window_id.append(widx)

        return {
            "input_ids": out_input_ids,
            "attention_mask": out_attention,
            "labels": out_labels,
            "word_ids": out_word_ids,
            "lemma_ids_aligned": out_lemma,
            "pos_ids_aligned": out_pos,
            "non_finite_verb_aligned": out_nfv,
            "microsection": out_microsection,
            "sentence_id": out_sentence_id,
            "window_id": out_window_id,
        }

    return tokenize_and_align_everything


def lemma_to_init_vector_factory(tok_emb, encoder):
    def lemma_to_init_vector(lemma: str) -> torch.Tensor:
        sub_ids = encoder.encode(lemma)
        if len(sub_ids) == 0:
            return tok_emb[UNK_ID].clone()
        ids = torch.tensor([sid + OFFSET for sid in sub_ids], dtype=torch.long)
        vecs = tok_emb[ids]
        return vecs.mean(dim=0)

    return lemma_to_init_vector


class LatinBertForTokenClassification(PreTrainedModel):
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
        pos_pad_id,):
    
        super().__init__(config)
        self.num_labels = num_labels

        self.encoder = AutoModel.from_pretrained(base_model_name, config=config)
        hidden = self.encoder.config.hidden_size

        self.lemma_emb = nn.Embedding(num_lemmas, hidden, padding_idx=lemma_pad_id)
        self.lemma_emb.weight.data.copy_(lemma_init_matrix)

        self.pos_emb = nn.Embedding(num_pos, hidden, padding_idx=pos_pad_id)
        self.nfv_emb = nn.Embedding(2, hidden)

        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden * 4, num_labels)

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        lemma_ids_aligned=None,
        pos_ids_aligned=None,
        non_finite_verb_aligned=None,):

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


from dataclasses import dataclass
from typing import Any, Dict, List, Optional

PAD_ID = 0
LABEL_PAD_ID = -100


@dataclass
class LatinBertDataCollatorForTokenClassificationWithFeatures:
    pad_id: int = PAD_ID
    label_pad_id: int = LABEL_PAD_ID
    lemma_pad_id: int = 0
    pos_pad_id: int = 0
    nfv_pad_id: int = 0
    keep_keys: Optional[List[str]] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(f["input_ids"]) for f in features)

        batch_input_ids, batch_attention, batch_labels = [], [], []
        batch_lemma, batch_pos, batch_nfv = [], [], []

        kept = {k: [] for k in (self.keep_keys or [])}

        for f in features:
            input_ids = f["input_ids"]
            attention = f.get("attention_mask", [1] * len(input_ids))
            labels = f.get("labels", [self.label_pad_id] * len(input_ids))

            lemma_ids = f.get("lemma_ids_aligned", [self.lemma_pad_id] * len(input_ids))
            pos_ids = f.get("pos_ids_aligned", [self.pos_pad_id] * len(input_ids))
            nfv_ids = f.get("non_finite_verb_aligned", [self.nfv_pad_id] * len(input_ids))

            pad_n = max_len - len(input_ids)

            batch_input_ids.append(input_ids + [self.pad_id] * pad_n)
            batch_attention.append(attention + [0] * pad_n)
            batch_labels.append(labels + [self.label_pad_id] * pad_n)

            batch_lemma.append(lemma_ids + [self.lemma_pad_id] * pad_n)
            batch_pos.append(pos_ids + [self.pos_pad_id] * pad_n)
            batch_nfv.append(nfv_ids + [self.nfv_pad_id] * pad_n)

            for k in kept:
                kept[k].append(f.get(k))

        out = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "lemma_ids_aligned": torch.tensor(batch_lemma, dtype=torch.long),
            "pos_ids_aligned": torch.tensor(batch_pos, dtype=torch.long),
            "non_finite_verb_aligned": torch.tensor(batch_nfv, dtype=torch.long),
        }
        out.update(kept)
        return out


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    mask = labels != -100
    y_true = labels[mask].astype(int)
    y_pred = preds[mask].astype(int)

    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    y_score = probs[..., 1][mask]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_file", default="data/train.jsonl")
    ap.add_argument("--val_file", default="data/val.jsonl")
    ap.add_argument("--test_file", default="data/test.jsonl")
    ap.add_argument("--output_dir", default="outputs/run")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--weight_decay", type=float, default=0.01)

    # model loading: the model must loaded locally.
    # the model can be downloaded from https://github.com/dbamman/latin-bert following the instructions.

    ap.add_argument(
        "--model_dir",
        default=os.environ.get("LATINBERT_DIR", ""),
        help="Local path to LatinBERT folder (must contain config.json, model weights, vocab.txt).",
    )

    args = ap.parse_args()

    vocab_path = os.path.join(args.model_dir, "vocab.txt")
    encoder = text_encoder.SubwordTextEncoder(vocab_path)

    # load data (jsonl)
    train_dataset = Dataset.from_list([json.loads(line) for line in open(args.train_file, "r", encoding="utf-8")])
    val_dataset = Dataset.from_list([json.loads(line) for line in open(args.val_file, "r", encoding="utf-8")])
    test_dataset = Dataset.from_list([json.loads(line) for line in open(args.test_file, "r", encoding="utf-8")])

    # vocabs
    pos2id = build_vocab(train_dataset, "pos")

    lemma_counter = Counter(l for ex in train_dataset["lemmas"] for l in ex)
    lemma2id = {"[PAD]": 0, "[UNK]": 1}
    for lemma, _ in lemma_counter.most_common():
        if lemma not in lemma2id:
            lemma2id[lemma] = len(lemma2id)

    add_feature_ids = add_feature_ids_factory(pos2id, lemma2id)

    train_dataset = train_dataset.map(add_feature_ids, batched=True)
    val_dataset = val_dataset.map(add_feature_ids, batched=True)
    test_dataset = test_dataset.map(add_feature_ids, batched=True)

    # tokenize + align (with features)
    tokenize_and_align_everything = tokenize_and_align_everything_factory(encoder)

    tokenized_train = train_dataset.map(tokenize_and_align_everything, batched=True, remove_columns=train_dataset.column_names)
    tokenized_val = val_dataset.map(tokenize_and_align_everything, batched=True, remove_columns=val_dataset.column_names)
    tokenized_test = test_dataset.map(tokenize_and_align_everything, batched=True, remove_columns=test_dataset.column_names)

    # lemma embedding init from base model token embeddings
    base_model = AutoModel.from_pretrained(args.model_dir)
    tok_emb = base_model.get_input_embeddings().weight.detach().cpu()
    hidden_size = tok_emb.shape[1]

    lemma_to_init_vector = lemma_to_init_vector_factory(tok_emb, encoder)

    num_lemmas = len(lemma2id)
    lemma_init = torch.zeros(num_lemmas, hidden_size)
    lemma_init[lemma2id["[UNK]"]] = tok_emb[UNK_ID]

    for lemma, lid in lemma2id.items():
        if lemma in ("[PAD]", "[UNK]"):
            continue
        lemma_init[lid] = lemma_to_init_vector(lemma)

    # model
    config = AutoConfig.from_pretrained(args.model_dir, num_labels=2)

    custom_model = LatinBertForTokenClassification(
        config=config,
        base_model_name=args.model_dir,
        num_lemmas=len(lemma2id),
        lemma_init_matrix=lemma_init,
        num_labels=2,
        num_pos=len(pos2id),
        lemma_pad_id=lemma2id["[PAD]"],
        pos_pad_id=pos2id["[PAD]"],
    )

    data_collator = LatinBertDataCollatorForTokenClassificationWithFeatures(
        pad_id=0,
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
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        report_to=None,
        logging_strategy="epoch",
    )

    trainer = Trainer(
        model=custom_model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    metrics = trainer.evaluate()
    print(metrics)

    test_metrics = trainer.evaluate(eval_dataset=tokenized_test, metric_key_prefix="test")
    print(test_metrics)


if __name__ == "__main__":
    main()
