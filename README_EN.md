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
    <a href="README.md">한국어</a> |
    <b>English</b>
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

<div align="center">
  <img width="700px" src="https://github.com/monologg/KoBigBird/raw/master/.github/images/sparse-attention.png">
</div>

[BigBird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) is a **sparse-attention** based model that can handle **longer sequences** than a normal BERT.

🦅 **Longer Sequence** - Handles **up to 4096 tokens**, 8 times the BERT, which can handle up to 512 tokens

⏱️ **Computational Efficiency** - Improved from O(n<sup>2</sup>) to <b>O(n)</b> using **Sparse Attention** instead of Full Attention

## How to Use

- Available on the 🤗 [Huggingface Hub](https://huggingface.co/monologg/kobigbird-bert-base)!
- Recommend to use `transformers>=4.11.0`, which some issues are fixed ([PR related with MRC issue](https://github.com/huggingface/transformers/pull/13143))
- **You have to use `BertTokenizer` instead of BigBirdTokenizer (`BertTokenizer` will be loaded if you use `AutoTokenizer`)**
- For detail guideline, see [BigBird Tranformers documentation](https://huggingface.co/docs/transformers/v4.20.1/en/model_doc/big_bird).

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("monologg/kobigbird-bert-base")  # BigBirdModel
tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")  # BertTokenizer
```

## Pretraining

> For more information, see [[Pretraining BigBird]](pretrain/README_EN.md)

|                         | Hardware | Max len |   LR | Batch | Train Step | Warmup Step |
| :---------------------- | -------: | ------: | ---: | ----: | ---------: | ----------: |
| **KoBigBird-BERT-Base** | TPU v3-8 |    4096 | 1e-4 |    32 |         2M |         20k |

- Trained with various data such as Everyone's Corpus, Korean Wiki, Common Crawl, and news data
- Use `ITC (Internal Transformer Construction)` model for pretraining. ([ITC vs ETC](https://huggingface.co/blog/big-bird#itc-vs-etc))

## Evaluation Result

### 1. Short Sequence (<=512)

> For more information, see [[Finetune on Short Sequence Dataset]](docs/short_seq_evaluation_en.md)

|                         | NSMC<br>(acc) | KLUE-NLI<br>(acc) | KLUE-STS<br>(pearsonr) | Korquad 1.0<br>(em/f1) | KLUE MRC<br>(em/rouge-w) |
| :---------------------- | :-----------: | :---------------: | :--------------------: | :--------------------: | :----------------------: |
| KoELECTRA-Base-v3       |     91.13     |       86.87       |       **93.14**        |     85.66 / 93.94      |      59.54 / 65.64       |
| KLUE-RoBERTa-Base       |     91.16     |       86.30       |         92.91          |     85.35 / 94.53      |      69.56 / 74.64       |
| **KoBigBird-BERT-Base** |   **91.18**   |     **87.17**     |         92.61          |   **87.08 / 94.71**    |    **70.33 / 75.34**     |

### 2. Long Sequence (>=1024)

> For more information, see [[Finetune on Long Sequence Dataset]](finetune/README_EN.md)

|                         | TyDi QA<br/>(em/f1) | Korquad 2.1<br/>(em/f1) | Fake News<br/>(f1) | Modu Sentiment<br/>(f1-macro) |
| :---------------------- | :-----------------: | :---------------------: | :----------------: | :---------------------------: |
| KLUE-RoBERTa-Base       |    76.80 / 78.58    |      55.44 / 73.02      |       95.20        |             42.61             |
| **KoBigBird-BERT-Base** |  **79.13 / 81.30**  |    **67.77 / 82.03**    |     **98.85**      |           **45.42**           |

## Docs

- [Pretraing BigBird](pretrain/README_EN.md)
- [Finetune on Short Sequence Dataset](docs/short_seq_evaluation_en.md)
- [Finetune on Long Sequence Dataset](finetune/README_EN.md)
- [Download Tensorflow v1 checkpoint](docs/download_tfv1_ckpt.md)
- [GPU Benchmark result](docs/gpu_benchmark.md)

## Citation

If you apply KoBigBird to any project and research, please cite our code as below.

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

KoBigBird is built with Cloud TPU support from the Tensorflow Research Cloud (TFRC) program.

Also, thanks to [Seyun Ahn](https://www.instagram.com/ahnsy13) for a nice logo:)
