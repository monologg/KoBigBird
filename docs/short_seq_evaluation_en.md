# Finetune on Short Sequence Dataset

<p align="left">
    <a href="short_seq_evaluation_ko.md">한국어</a> |
    <b>English</b>
</p>

## Details

- KoBigBird performance evaluation in `max_seq_length<=512` setting

- Evaluated with a total of **5 Datasets**

  - Single Sentence Classification: `NSMC`
  - Sentence Pair Classification: `KLUE-NLI`, `KLUE-STS`
  - Question Answering: `Korquad 1.0`, `KLUE-MRC`

- **Based on the [KLUE-Baseline](https://github.com/KLUE-benchmark/KLUE-baseline) code with some modifications**

  - Add `nsmc` and `korquad 1.0` tasks
  - Fix to be compatible with `transformers==4.11.3`

- Sequence Classification is trained with a length of **128** and Question Answering with a length of **512**

  - **Full Attention** instead of Sparse Attention (Automatically changed to Full Attention with the following log)

  ```text
  Attention type 'block_sparse' is not possible if sequence_length: 300 <= num global tokens: 2 * config.block_size + min. num sliding tokens: 3 * config.block_size
  + config.num_random_blocks * config.block_size + additional buffer: config.num_random_blocks * config.block_size = 704 with config.block_size = 64, config.num_random_blocks = 3.
  Changing attention type to 'original_full'...
  ```

## Result

|                         | NSMC<br>(acc) | KLUE-NLI<br>(acc) | KLUE-STS<br>(pearsonr) | Korquad 1.0<br>(em/f1) | KLUE MRC<br>(em/rouge-w) |
| :---------------------- | :-----------: | :---------------: | :--------------------: | :--------------------: | :----------------------: |
| KoELECTRA-Base-v3       |     91.13     |       86.87       |       **93.14**        |     85.66 / 93.94      |      59.54 / 65.64       |
| KLUE-RoBERTa-Base       |     91.16     |       86.30       |         92.91          |     85.35 / 94.53      |      69.56 / 74.64       |
| **KoBigBird-BERT-Base** |   **91.18**   |     **87.17**     |         92.61          |   **87.08 / 94.71**    |    **70.33 / 75.34**     |

- `KLUE` and `Korquad 1.0` are evaluated with **dev set**.
- For `KoELECTRA-Base-v3` and `KLUE-RoBERTa-Base`, we brought the KLUE dataset score from `A. Dev Set Results` in [KLUE Paper](https://arxiv.org/abs/2105.09680).

## Reference

- [NSMC](https://github.com/e9t/nsmc)
- [KLUE](https://github.com/KLUE-benchmark/KLUE)
- [Korquad 1.0](https://korquad.github.io/KorQuad%201.0/)
- [KoELECTRA-Base-v3](https://huggingface.co/monologg/koelectra-base-v3-discriminator)
- [KLUE-RoBERTa-Base](https://huggingface.co/klue/roberta-base)
