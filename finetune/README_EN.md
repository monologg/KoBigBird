# Finetune on Long Sequence Dataset

<p align="left">
    <a href="README.md">한국어</a> |
    <b>English</b>
</p>

## About Dataset

| Dataset            | Task                    | Length (median) | Length (max) |
| ------------------ | ----------------------- | --------------: | -----------: |
| **TyDi QA**        | Question Answering      |           6,165 |       67,135 |
| **Korquad 2.1**    | Question Answering      |           5,777 |      486,730 |
| **Fake News**      | Sequence Classification |             564 |       17,488 |
| **Modu Sentiment** | Sequence Classification |             185 |        5,245 |

- `Length` is calculated based on subword token.
- [TyDi QA](https://github.com/google-research-datasets/tydiqa) is originally `multilingual` and contains `BoolQA` cases. **We only use korean samples and skip BoolQA samples.**

## Setup

### 1. Requirements

```bash
pip3 install -r requirements.txt
```

### 2. Prepare Dataset

#### 1) Question Answering

```bash
bash download_qa_dataset.sh
```

#### 2) Sequence Classification

- **After downloading the data through the link below, place the data in the `--data_dir` path.**
- `Fake news`: [Korean Fake news](https://github.com/2alive3s/Fake_news/blob/b43638105f4802de5773c21afe539157ebed6cc5/data/mission2_train.zip) (`mission2_train.csv`)
- `Modu sentiment corpus`: [감성 분석 말뭉치 2020](https://corpus.korean.go.kr) (`EXSA2002108040.json`)

## How to Run

- We highly recommend to run the scripts on **TPU instance** in order to train and evaluate large and long-sequence datasets.
- We trained and evaluated the models on the [torch-xla-1.8.1](https://github.com/pytorch/xla#-consume-prebuilt-compute-vm-images) environment with `TPU v3-8`.
- Disable `--use_tpu` argument for GPU training.

```bash
bash scripts/run_{$TASK_NAME}.sh  # kobigbird
bash scripts/run_{$TASK_NAME}_short.sh  # klue roberta
```

```bash
bash scripts/run_tydiqa.sh  # tydiqa
bash scripts/run_korquad_2.sh  # korquad 2.1
bash scripts/run_fake_news.sh  # fake news
bash scripts/run_modu_sentiment.sh  # modu sentiment
```

## Results

- In the case of sequence classification, it was evaluated by splitting `train:test=8:2`.
- For `korquad 2.1`, we **only use the subset of the train dataset** because of limited computational resources.
  - Enable `--all_korquad_2_sample` argument in order to use full train dataset.
- In the case of `KoBigBird`, question answering was trained with a length of **4096** and sequence classification was trained with a length of **1024**.
- `KLUE RoBERTa` was trained with a length of **512**.

|                         | TyDi QA<br/>(em/f1) | Korquad 2.1<br/>(em/f1) | Fake News<br/>(f1) | Modu Sentiment<br/>(f1-macro) |
| :---------------------- | :-----------------: | :---------------------: | :----------------: | :---------------------------: |
| KLUE-RoBERTa-Base       |    76.80 / 78.58    |      55.44 / 73.02      |       95.20        |             42.61             |
| **KoBigBird-BERT-Base** |  **79.13 / 81.30**  |    **67.77 / 82.03**    |     **98.85**      |           **45.42**           |

## Reference

- [TyDi QA](https://github.com/google-research-datasets/tydiqa)
- [Korquad](https://korquad.github.io/)
- [Korean Fake news](https://github.com/2alive3s/Fake_news)
- [모두의 말뭉치](https://corpus.korean.go.kr/)
