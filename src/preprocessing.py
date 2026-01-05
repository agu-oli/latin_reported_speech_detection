import argparse
import pandas as pd
from pathlib import Path
from datasets import Dataset
from sklearn.model_selection import train_test_split

def df_to_examples(df: pd.DataFrame):
    """
    Convert token-level dataframe into microsection-level examples.
    """
    examples = []
    for microsection, g in df.groupby("microsection", sort=False):
        examples.append({
            "microsection": microsection,
            "sentence_id": int(g["sentence_id"].iloc[0]),
            "tokens": g["form"].astype(str).tolist(),
            "lemmas": g["lemma"].astype(str).tolist(),
            "labels": g["target"].astype(int).tolist(),
            "pos": g["upos"].astype(str).tolist(),
            "non_finite_verb": g["non_finite_verb"].astype(int).tolist(),
        })
    return examples


SEED = 42  # Fixed seed in script


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True, help="Path to raw csv dataset")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_dir = out_dir.parent / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load CSV and build microsection-level dataset
    
    df = pd.read_csv(args.input_csv)
    dataset = Dataset.from_list(df_to_examples(df))

    # 2) Fixed split by microsection (70 / 20 / 10)

    micro = list(dataset["microsection"])

    train_micro, temp_micro = train_test_split(
        micro,
        test_size=0.3,
        random_state=SEED,
        shuffle=True,
    )

    val_micro, test_micro = train_test_split(
        temp_micro,
        test_size=1 / 3,
        random_state=SEED,
        shuffle=True,
    )

    train_set = set(train_micro)
    val_set   = set(val_micro)
    test_set  = set(test_micro)

    train_ds = dataset.filter(lambda x: x["microsection"] in train_set)
    val_ds   = dataset.filter(lambda x: x["microsection"] in val_set)
    test_ds  = dataset.filter(lambda x: x["microsection"] in test_set)

    # 4) Save datasets
    train_ds.to_json(out_dir / "train.jsonl", orient="records", lines=True)
    val_ds.to_json(out_dir / "dev.jsonl", orient="records", lines=True)
    test_ds.to_json(out_dir / "test.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    main()
