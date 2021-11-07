# Finetune on Long Sequence Dataset

<p align="left">
    <b>한국어</b> |
    <a href="README_EN.md">English</a>
</p>

## About Dataset

| Dataset            | Task                    | Length (median) | Length (max) |
| ------------------ | ----------------------- | --------------: | -----------: |
| **TyDi QA**        | Question Answering      |           6,165 |       67,135 |
| **Korquad 2.1**    | Question Answering      |           5,777 |      486,730 |
| **Fake News**      | Sequence Classification |             564 |       17,488 |
| **Modu Sentiment** | Sequence Classification |             185 |        5,245 |

- `Length`는 subword token을 기준으로 계산했습니다.
- [TyDi QA](https://github.com/google-research-datasets/tydiqa)는 본래 `다국어(multilingual) 데이터셋`이며 `예-아니오 (BoolQA)` 답변 데이터를 포함합니다. **본 프로젝트에서는 한국어 데이터셋만을 사용했으며, 예-아니오 답변 데이터 또한 제외하였습니다.**

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

- **아래의 링크를 통해 데이터를 다운로드 후, 데이터를 `--data_dir` 경로에 위치시켜 주세요.**
- `Fake news`: [Korean Fake news](https://github.com/2alive3s/Fake_news/blob/b43638105f4802de5773c21afe539157ebed6cc5/data/mission2_train.zip) (`mission2_train.csv`)
- `Modu sentiment corpus`: [감성 분석 말뭉치 2020](https://corpus.korean.go.kr) (`EXSA2002108040.json`)

## How to Run

- 큰 규모의 Long sequence 데이터셋을 학습시키기 위해서 **TPU 인스턴스**에서 실행하는 것을 권장합니다.
- 평가 결과는 모두 [torch-xla-1.8.1](https://github.com/pytorch/xla#-consume-prebuilt-compute-vm-images) 환경에서 `TPU v3-8`을 이용하여 학습 및 평가했습니다.
- TPU가 아닌 GPU로 학습하고 싶을 시 스크립트 안의 `--use_tpu` 인자를 제외하면 됩니다.

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

- Sequence Classification의 경우 `train:test=8:2` 로 split하여 평가했습니다.
- Korquad 2.1 데이터셋의 경우, 컴퓨팅 자원의 한계로 **학습 데이터셋의 일부만으로 모델을 학습**했습니다.
  - `--all_korquad_2_sample` 인자를 스크립트에 추가하면 전체 데이터를 이용하여 학습 가능
- `KoBigBird`의 경우 Question Answering은 **4096**, Sequence Classification은 **1024**의 길이로 학습했습니다.
- `KLUE RoBERTa`는 **512**의 길이로 학습했습니다.

|                         | TyDi QA<br/>(em/f1) | Korquad 2.1<br/>(em/f1) | Fake News<br/>(f1) | Modu Sentiment<br/>(f1-macro) |
| :---------------------- | :-----------------: | :---------------------: | :----------------: | :---------------------------: |
| KLUE-RoBERTa-Base       |    76.80 / 78.58    |      55.44 / 73.02      |       95.20        |             42.61             |
| **KoBigBird-BERT-Base** |  **79.13 / 81.30**  |    **67.77 / 82.03**    |     **98.85**      |           **45.42**           |

## Reference

- [TyDi QA](https://github.com/google-research-datasets/tydiqa)
- [Korquad](https://korquad.github.io/)
- [Korean Fake news](https://github.com/2alive3s/Fake_news)
- [모두의 말뭉치](https://corpus.korean.go.kr/)
