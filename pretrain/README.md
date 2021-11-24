# Pretraining BigBird

<p align="left">
    <b>한국어</b> |
    <a href="README_EN.md">English</a>
</p>

## 기존 BigBird 코드와의 차이점 및 개선점

> NOTE: 원본 BigBird 코드는 [original bigbird github](https://github.com/google-research/bigbird) 참고

- **RoBERTa가 아닌 BERT를 이용하여 warm start**

  - 자체적으로 새로 학습한 BERT를 이용

- **BERT checkpoint loading 관련 이슈 해결**

  - tf variable name을 수정하여 LMHead가 정상적으로 로딩되지 않는 이슈 해결 (e.g. `transform` -> `transform/dense`)
  - 기존의 512인 position embeddings을 로딩하지 않도록 처리

- **Hard Coding 이슈 해결**

  - `MAX_SEQ_LEN=4096`으로 강제되어 있는 부분
  - Sentencepiece tokenizer 사용이 강제되어 있는 부분
  - RoBERTa Vocab에 맞춰 token id가 하드코딩된 부분

- **`TensorFlow Datasets (tfds)`를 사용하지 않도록 변경**

  - 여러 버전으로 테스트해보았지만 정상 작동하지 않는 이슈 존재
  - 그 대신 TFRecord Builder 코드 추가 ([`create_pretraining_data.py`](./create_pretraining_data.py))

## How to Pretrain

### 0. (Optional) Prepare your own BERT

- Warm Start를 하지 않는다면 BERT가 없어도 상관없습니다.
- BERT를 직접 만들고 싶으면 [BERT Github](https://github.com/google-research/bert)를 참고
- 아래와 같은 형태로 **tensorflow v1 checkpoint**를 준비해야 합니다.

```text
init_checkpoint
├── bert_config.json
├── checkpoint
├── graph.pbtxt
├── model.ckpt-0.data-00000-of-00001
├── model.ckpt-0.index
└── model.ckpt-0.meta
```

### 1. Prepare Tokenizer

- `Hugging Face Transformers`에 호환되는 tokenizer 준비해야 합니다. (KoBigBird의 [tokenizer](./tokenizer)를 sample로 업로드)
- `BigBirdTokenizer`와의 호환을 위해 `BertTokenizer`에 `bos token(=<s>)`, `eos token(=</s>)`을 추가

### 2. Create TFRecord

```bash
bash scripts/build_tfrecord.sh
```

#### Prepare Corpus

[`ko_lm_dataformat`](https://github.com/monologg/ko_lm_dataformat) 혹은 `txt` 파일 사용 가능

- `ko_lm_dataformat`은 sentence split된 document 단위로 들어온다고 가정

```python
for data in rdr.stream_data():
  print(data)

# data
['제임스 얼 카터 주니어(1924년 10월 1일 ~)는 민주당 출신 미국 39대 대통령 (1977년 ~ 1981년)이다.',
  '지미 카터는 조지아주 섬터 카운티 플레인스 마을에서 태어났다.',
  '그 후 해군에 들어가 전함·원자력·잠수함의 승무원으로 일하였다'.,
  ...]
```

- `txt`의 경우 document 사이에 newline이 있다고 가정

```text
제임스 얼 카터 주니어(1924년 10월 1일 ~)는 민주당 출신 미국 39대 대통령 (1977년 ~ 1981년)이다.
지미 카터는 조지아주 섬터 카운티 플레인스 마을에서 태어났다.

수학(數學)은 수, 양, 구조, 공간, 변화 등의 개념을 다루는 학문이다.
널리 받아들여지는 명확한 정의는 없으나 현대 수학은 일반적으로 엄밀한 논리에 근거하여 추상적 대상을 탐구하며, 이는 규칙의 발견과 문제의 제시 및 해결의 과정으로 이루어진다.

문학(文學)은 언어를 예술적 표현의 제재로 삼아 새로운 의미를 창출하여, 인간과 사회를 진실되게 묘사하는 예술의 하위분야이다.
간단하게 설명하면, 언어를 통해 인간의 삶을 미적(美的)으로 형상화한 것이라고 볼 수 있다.
```

#### Details

- `BertTokenizer` 사용을 가정하고 코드 작성 (만일 다른 tokenizer를 사용할 경우 코드를 직접 수정해야 합니다)
- `whole word masking`, `max ngram masking` 적용 가능
- `static masking` (dupe_factor를 통해 다른 마스킹으로 데이터를 몇 배로 만들지 결정)
- `long_seq_threshold`를 통해 example을 만들 때 너무 긴 문장의 경우 여러 개의 example로 나눠줌
- `RoBERTa`에서 사용한 **full sentences** 방식으로 example 생성

### 3. Pretraining with TPU

- **TPU의 tensorflow version은 `2.3.1`을 권장**
- 기본적으로 BERT에서 Warm start 했으며, `position embedding (size 4096)`만 random initialize 처리
- `max_predictions_per_seq=640` 으로 설정 (원 논문의 600보다 크게 설정)
- `tokenizer`, `pretrain_config.json` 모두 `output_dir`에 저장하도록 처리 (추후 huggingface transformers 포맷으로 변환할 때 사용)

#### How to run

- Google Storage에 `tfrecord` 와 `BERT checkpoint`(optional) 업로드

```bash
gsutil -m cp -r pretrain_tfrecords gs://{$BUCKET_NAME}  # tfrecord
gsutil -m cp -r init_checkpoint gs://{$BUCKET_NAME}  # BERT tf v1 ckpt
```

- GCP cloud shell에서 아래의 명령어로 instance와 TPU 세팅

```bash
ctpu up --zone=europe-west4-a --tf-version=2.3.1 --tpu-size=v3-8 --machine-type=n1-standard-1 --disk-size-gb=20 --name={$GCP_NAME} --project={$PROJECT_NAME}
```

- Instance에서 학습 진행

  - BERT tf v1 checkpoint를 warm start에 사용할 시 스크립트에 아래와 같이 인자를 추가해주세요

  ```bash
  --init_checkpoint=gs://$BUCKET_NAME/init_checkpoint/model.ckpt-0
  ```

```bash
cd pretrain
pip3 install -r requirements.txt
bash scripts/base_size_tpu.sh
```

## Convert Tensorflow checkpoint to Hugging Face Transformers format

```bash
python3 convert_bigbird_tf_to_pt.py \
    --checkpoint_dir $ORIG_TF_BERT_CKPT \
    --big_bird_config_file $BIGBIRD_CONFIG_PATH \
    --output_dir $PT_OUTPUT_DIR \
    --tokenizer_dir $TOKENIZER_DIR \
    --step_on_output
```

- `--big_bird_config_file`를 명시하지 않으면, script가 자동으로 tensorflow checkpoint 안의 `pretrain_config.json`을 사용합니다.
- `--tokenizer_dir`를 명시하지 않으면, script가 자동으로 tensorflow checkpoint 안의 `tokenizer`를 사용합니다.

## Reference

- [Original BigBird implementation](https://github.com/google-research/bigbird)
- [BERT tensorflow v1 implementation](https://github.com/google-research/bert)
- [BERT tensorflow v2 implementation](https://github.com/tensorflow/models/tree/d4c5f8975a7b89f01421101882bc8922642c2314/official/nlp/bert)
- [ELECTRA implementation](https://github.com/google-research/electra)
- [Hugging Face Transformers Documentation](https://huggingface.co/transformers/)
- [ko-lm-dataformat](https://github.com/monologg/ko_lm_dataformat)
