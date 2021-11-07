import json

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def sample_writer(data, config, tokenizer, is_train):
    feature = tokenizer(
        data["text"],
        max_length=config.max_seq_length,
        padding="max_length",
        truncation=True,
        add_special_tokens=True,
    )
    write_data = {
        "input_ids": feature["input_ids"],
        "attention_mask": feature["attention_mask"],
        "labels": data["label"],
    }
    return write_data


def make_label_map(labels):
    unique_labels = sorted(list(set(labels)))
    label2id = dict()
    for i, label in enumerate(unique_labels):
        label2id[label] = i
    return label2id


def postprocess():
    def decorator(fn):
        def wrapped(config, data_file, is_train, **kwargs):
            get_label_map = kwargs.get("get_label_map", False)
            texts, labels = fn(config, data_file, is_train)

            try:
                label2id = config.label2id
            except Exception:
                label2id = label2id = make_label_map(labels)

            labels = [label2id[label] for label in labels]

            if get_label_map:
                return label2id

            data = [{"text": text, "label": label} for text, label in zip(texts, labels)]
            pd.DataFrame(data).to_csv(
                "{}_{}_{}.csv".format(data_file, config.dataset, "train" if is_train else "valid"),
                index=False,
                encoding="utf-8-sig",
            )
            if is_train:
                pd.DataFrame(list(label2id.items()), columns=["label", "id"]).to_csv(
                    "{}_{}_label2id.csv".format(data_file, config.dataset), index=False, encoding="utf-8-sig"
                )

            return data

        return wrapped

    return decorator


def train_split(config, texts, labels, is_train):
    x_train, y_train, x_label, y_label = train_test_split(
        texts, labels, test_size=0.2, random_state=config.seed, stratify=labels
    )
    if is_train:
        texts, labels = x_train, x_label
    else:
        texts, labels = y_train, y_label
    return texts, labels


@postprocess()
def process_fake_news_cls(config, data_file, is_train):
    df = pd.read_csv(data_file)
    try:
        labels = df["Label"].astype(str).values.tolist()
    except Exception:
        labels = df["label"].astype(str).values.tolist()
    texts = [
        title + " " + content
        for title, content in zip(df["title"].astype(str).values.tolist(), df["content"].astype(str).values.tolist())
    ]
    texts, labels = train_split(config, texts, labels, is_train)
    return texts, labels


@postprocess()
def process_aihub_sentiment(config, data_file, is_train):
    with open(data_file) as handle:
        data = json.load(handle)
        texts = [" ".join([str(v) for _, v in datum["talk"]["content"].items()]) for datum in data]
        labels = [datum["profile"]["emotion"]["type"] for datum in data]
    return texts, labels


@postprocess()
def process_modu_sentiment(config, data_file, is_train):
    with open(data_file) as handle:
        data = json.load(handle)["document"]
        texts, labels = [], []
        for datum in data:
            texts.append(" ".join(paragraph["paragraph_form"] for paragraph in datum["paragraph"]))
            labels.append(datum["document_score"])
        texts, labels = train_split(config, texts, labels, is_train)
    return texts, labels


@postprocess()
def process_nsmc(config, data_file, is_train):
    data = load_dataset("nsmc", cache_dir=config.cache_dir)
    data = data[("train" if is_train else "test")]
    texts, labels = [], []
    for datum in data:
        labels.append(datum.pop("label"))
        texts.append(datum.pop("document"))
    return texts, labels


process_map = {
    "fake_news": process_fake_news_cls,
    "aihub_sentiment": process_aihub_sentiment,
    "modu_sentiment": process_modu_sentiment,
    "nsmc": process_nsmc,
}


def collate_fn(features):
    input_ids = [sample["input_ids"] for sample in features]
    attention_mask = [sample["attention_mask"] for sample in features]
    labels = [sample["labels"] for sample in features]

    input_ids = torch.tensor(np.array(input_ids).astype(np.int64), dtype=torch.long)
    attention_mask = torch.tensor(np.array(attention_mask).astype(np.int8), dtype=torch.long)
    labels = torch.tensor(np.array(labels).astype(np.int64), dtype=torch.long)
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    return inputs
