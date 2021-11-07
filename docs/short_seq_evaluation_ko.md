# Finetune on Short Sequence Dataset

<p align="left">
    <b>한국어</b> |
    <a href="short_seq_evaluation_en.md">English</a>
</p>

## Details

- `max_seq_length<=512` 환경에서의 KoBigBird 성능 평가

- 총 **5개의 Dataset**으로 평가

  - Single Sentence Classification: `NSMC`
  - Sentence Pair Classification: `KLUE-NLI`, `KLUE-STS`
  - Question Answering: `Korquad 1.0`, `KLUE-MRC`

- **[KLUE-Baseline](https://github.com/KLUE-benchmark/KLUE-baseline)의 코드를 기반으로 일부 수정하여 학습**

  - `nsmc`와 `korquad 1.0` task 추가
  - `transformers==4.11.3`에 호환되도록 수정

- Sequence Classification은 **128**, Question Answering은 **512**의 길이로 학습

  - Sparse Attention이 아닌 **Full Attention**으로 세팅 (아래의 로그가 나오면서 자동으로 Full Attention으로 변경)

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

- `KLUE`, `Korquad 1.0` 모두 **dev set**으로 평가
- `KoELECTRA-Base-v3`와 `KLUE-RoBERTa-Base`의 KLUE dataset 관련 점수는 [KLUE Paper](https://arxiv.org/abs/2105.09680)의 `A. Dev Set Results`에서 참고

## Reference

- [NSMC](https://github.com/e9t/nsmc)
- [KLUE](https://github.com/KLUE-benchmark/KLUE)
- [Korquad 1.0](https://korquad.github.io/KorQuad%201.0/)
- [KoELECTRA-Base-v3](https://huggingface.co/monologg/koelectra-base-v3-discriminator)
- [KLUE-RoBERTa-Base](https://huggingface.co/klue/roberta-base)
