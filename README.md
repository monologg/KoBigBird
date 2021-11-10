<div align="center">

<img src="https://github.com/monologg/KoBigBird/raw/master/.github/images/kobigbird-logo.png" width="400px">

<h1>Pretrained BigBird Model for Korean</h1>

<p align="center">
  <a href="#what-is-bigbird">What is BigBird</a> •
  <a href="#how-to-use">How to Use</a> •
  <a href="#pretraining">Pretraining</a> •
  <a href="#evaluation-result">Evaluation Result</a> •
  <a href="#docs">Docs</a> •
  <a href="#citation">Citation</a>
</p>

<p>
    <b>한국어</b> |
    <a href="README_EN.md">English</a>
</p>

<p align="center">
    <a href="https://github.com/monologg/KoBigBird/blob/master/LICENSE">
        <img alt="Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-yellow.svg">
    </a>
    <a href="https://github.com/monologg/KoBigBird/issues">
        <img alt="Issues" src="https://img.shields.io/github/issues/monologg/KoBigBird">
    </a>
    <a href="https://github.com/monologg/KoBigBird/actions/workflows/linter.yml">
        <img alt="linter" src="https://github.com/monologg/KoBigBird/actions/workflows/linter.yml/badge.svg">
    </a>
    <a href="https://doi.org/10.5281/zenodo.5654154">
        <img alt="DOI" src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.5654154-blue">
    </a>
</p>

</div>

## What is BigBird?

<img width="700px" src="https://github.com/monologg/KoBigBird/raw/master/.github/images/sparse-attention.png">

[BigBird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062)에서 소개된 **sparse-attention** 기반의 모델로, 일반적인 BERT보다 **더 긴 sequence**를 다룰 수 있습니다.

🦅 **Longer Sequence** - 최대 512개의 token을 다룰 수 있는 BERT의 8배인 **최대 4096개의 token**을 다룸

⏱️ **Computational Efficiency** - Full attention이 아닌 **Sparse Attention**을 이용하여 O(n<sup>2</sup>)에서 <b>O(n)</b>으로 개선

## How to Use

- 🤗 [Huggingface Hub](https://huggingface.co/monologg/kobigbird-bert-base)에 업로드된 모델을 곧바로 사용할 수 있습니다:)
- 일부 이슈가 해결된 `transformers>=4.11.0` 사용을 권장합니다. ([MRC 이슈 관련 PR](https://github.com/huggingface/transformers/pull/13143))
- **BigBirdTokenizer 대신에 `BertTokenizer` 를 사용해야 합니다. (`AutoTokenizer` 사용시 `BertTokenizer`가 로드됩니다.)**
- 자세한 사용법은 [BigBird Tranformers documentation](https://huggingface.co/transformers/model_doc/bigbird.html)을 참고해주세요.

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("monologg/kobigbird-bert-base")  # BigBirdModel
tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")  # BertTokenizer
```

## Pretraining

> 자세한 내용은 [[Pretraining BigBird]](pretrain/README.md) 참고

|                         | Hardware | Max len |   LR | Batch | Train Step | Warmup Step |
| :---------------------- | -------: | ------: | ---: | ----: | ---------: | ----------: |
| **KoBigBird-BERT-Base** | TPU v3-8 |    4096 | 1e-4 |    32 |         2M |         20k |

- 모두의 말뭉치, 한국어 위키, Common Crawl, 뉴스 데이터 등 다양한 데이터로 학습
- `ITC (Internal Transformer Construction)` 모델로 학습 ([ITC vs ETC](https://huggingface.co/blog/big-bird#itc-vs-etc))

## Evaluation Result

### 1. Short Sequence (<=512)

> 자세한 내용은 [[Finetune on Short Sequence Dataset]](docs/short_seq_evaluation_ko.md) 참고

|                         | NSMC<br>(acc) | KLUE-NLI<br>(acc) | KLUE-STS<br>(pearsonr) | Korquad 1.0<br>(em/f1) | KLUE MRC<br>(em/rouge-w) |
| :---------------------- | :-----------: | :---------------: | :--------------------: | :--------------------: | :----------------------: |
| KoELECTRA-Base-v3       |     91.13     |       86.87       |       **93.14**        |     85.66 / 93.94      |      59.54 / 65.64       |
| KLUE-RoBERTa-Base       |     91.16     |       86.30       |         92.91          |     85.35 / 94.53      |      69.56 / 74.64       |
| **KoBigBird-BERT-Base** |   **91.18**   |     **87.17**     |         92.61          |   **87.08 / 94.71**    |    **70.33 / 75.34**     |

### 2. Long Sequence (>=1024)

> 자세한 내용은 [[Finetune on Long Sequence Dataset]](finetune/README.md) 참고

|                         | TyDi QA<br/>(em/f1) | Korquad 2.1<br/>(em/f1) | Fake News<br/>(f1) | Modu Sentiment<br/>(f1-macro) |
| :---------------------- | :-----------------: | :---------------------: | :----------------: | :---------------------------: |
| KLUE-RoBERTa-Base       |    76.80 / 78.58    |      55.44 / 73.02      |       95.20        |             42.61             |
| **KoBigBird-BERT-Base** |  **79.13 / 81.30**  |    **67.77 / 82.03**    |     **98.85**      |           **45.42**           |

## Docs

- [Pretraing BigBird](pretrain/README.md)
- [Finetune on Short Sequence Dataset](docs/short_seq_evaluation_ko.md)
- [Finetune on Long Sequence Dataset](finetune/README.md)
- [Download Tensorflow v1 checkpoint](docs/download_tfv1_ckpt.md)
- [GPU Benchmark result](docs/gpu_benchmark.md)

## Citation

KoBigBird를 사용하신다면 아래와 같이 인용해주세요.

```bibtex
@software{jangwon_park_2021_5654154,
  author       = {Jangwon Park and Donggyu Kim},
  title        = {KoBigBird: Pretrained BigBird Model for Korean},
  month        = nov,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.5654154},
  url          = {https://doi.org/10.5281/zenodo.5654154}
}
```

## Contributors

[Jangwon Park](https://github.com/monologg) and [Donggyu Kim](https://github.com/donggyukimc)

## Acknowledgements

KoBigBird는 Tensorflow Research Cloud (TFRC) 프로그램의 Cloud TPU 지원으로 제작되었습니다.

또한 멋진 로고를 제공해주신 [Seyun Ahn](https://www.instagram.com/ahnsy13)님께 감사를 전합니다.
