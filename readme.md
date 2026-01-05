# Fine-tuning Latin BERT, PhilBERTa, and LaBERTa

This repository includes the code for finetuning Latin Bert, Philberta and Laberta models for binary token classification. The fine-tuned model detects sequences of reported speech in Latin. The training data includes extracts from Seneca's The Elder Controuersiae et Suasoriae, written around 30 AD. 

## 1. Data preprocessing

Obtain the train, validation, and test splits by running:

```
python src/preprocessing.py \
  --input_csv /path/to/dataset.csv \
  --out_dir data
```

This script creates the following files:

- data/train.jsonl
- data/val.jsonl
- data/test.jsonl

## 2. Train scripts

The training scripts assume the training, validation and test data exist in the data folder.
Data files are loaded from the default folder after having run preprocessing.py: 
data/val.jsonl
data/test.jsonl
outputs/run

Important: This repository uses 'eval_strategy' for Roberta based models and 'evaluation_strategy' for Latin Bert in the trainer arguments. If you get 'TypeError: unexpected keyword argument' switch name.

### Roberta based models: Philberta and Laberta

PhilBERTa and LaBERTa use Hugging Face tokenizers. Indicate the model name:

```
python src/robertabased_finetuning/train_robertabasedmodel.py \
  --model_name bowphs/LaBerta \
  --output_dir outputs/laberta_finetuned
```

bowphs/PhilBerta

python src/robertabased_finetuning/train_robertabasedmodel.py \
  --model_name bowphs/PhilBerta \
  --output_dir outputs/laberta_finetuned

### Latin Bert

A specific finetuning script is necesssary for Latin BERT.
LatiN BERT does not use a standard Hugging Face tokenizer. 
Tokenization is instead done using:
- tensor2tensor.data_generators.text_encoder.SubwordTextEncoder
- a vocab.txt that can be downloaded with the model 
- a custom encoding and alignment step

Also the model must be dowloaded from https://github.com/dbamman/latin-bert and loaded locally.

For this reason in order to run train_latinbert.py you must provide your local path to the model:

```
python src/latinbert_finetuning/train_latinbert.py \
  --model_dir /path/to/latin_bert \
  --output_dir outputs/latinbert_finetuned
```

# Dataset splitting strategy 

The dataset is split at the level of the microsection field, corresponding
to span-level discourse units.We randomly assigned 70% of the 
sections to training, 20% to validation, and 10% to the
test sets using a fixed random seed (42). No section
appears in more than one split. All tokens belonging to the same section
are assigned to the same split. 

# Requirements

Two requirements.txt files are being specified: one for Latin Bert fine-tuning, the other for finetuning PhilBERTa or LaBERTa

