# Pretraining BigBird

<p align="left">
    <a href="README.md">한국어</a> |
    <b>English</b>
</p>

## Differences & Improvements from existing BigBird code

> NOTE: Refer [original bigbird github](https://github.com/google-research/bigbird) for the original BigBird code

- **Warm start using BERT instead of RoBERTa**

  - Use newly pretrained BERT for warm start

- **Fix BERT checkpoint loading issues**

  - Fix the issue where LMHead doesn't load properly by modifying tf variable name (e.g. `transform` -> `transform/dense`)
  - Prevent not to load existing 512 position embeddings

- **Fix Hard Coding issues**

  - Forced `MAX_SEQ_LEN=4096`
  - Forced to use Sentencepiece tokenizer
  - Token ids which are hard-coded according to the RoBERTa Vocab

- **Change not to use `Tensorflow Datasets (tfds)`**

  - We've tested with several versions, but doesn't work properly.
  - Instead, we add TFRecord builder code ([`create_pretraining_data.py`](./create_pretraining_data.py))

## How to Pretrain

### 0. (Optional) Prepare your own BERT

- It doesn't matter if you don't have a BERT if you don't use warm starting.
- Refer [BERT Github](https://github.com/google-research/bert) if you want to make BERT by yourself.
- Prepare **tensorflow v1 checkpoint** as below.

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

- Prepare the tokenizer witch is compatible with `Huggingface Transformers`. (Uploaded KoBigBird [tokenizer](./tokenizer) as sample)
- Added `bos token(=<s>)`and `eos token(=</s>)` in `BertTokenizer` for compatibility with `BigBirdTokenizer`

### 2. Create TFRecord

```bash
bash scripts/build_tfrecord.sh
```

#### Prepare Corpus

You can use either [`ko_lm_dataformat`](https://github.com/monologg/ko_lm_dataformat) or `txt` file.

- Assume that `ko_lm_dataformat` comes in sentence-splitted documents

```python
for data in rdr.stream_data():
  print(data)

# data
['제임스 얼 카터 주니어(1924년 10월 1일 ~)는 민주당 출신 미국 39대 대통령 (1977년 ~ 1981년)이다.',
  '지미 카터는 조지아주 섬터 카운티 플레인스 마을에서 태어났다.',
  '그 후 해군에 들어가 전함·원자력·잠수함의 승무원으로 일하였다'.,
  ...]
```

- For `txt`, assume that there is a newline between documents

```text
제임스 얼 카터 주니어(1924년 10월 1일 ~)는 민주당 출신 미국 39대 대통령 (1977년 ~ 1981년)이다.
지미 카터는 조지아주 섬터 카운티 플레인스 마을에서 태어났다.

수학(數學)은 수, 양, 구조, 공간, 변화 등의 개념을 다루는 학문이다.
널리 받아들여지는 명확한 정의는 없으나 현대 수학은 일반적으로 엄밀한 논리에 근거하여 추상적 대상을 탐구하며, 이는 규칙의 발견과 문제의 제시 및 해결의 과정으로 이루어진다.

문학(文學)은 언어를 예술적 표현의 제재로 삼아 새로운 의미를 창출하여, 인간과 사회를 진실되게 묘사하는 예술의 하위분야이다.
간단하게 설명하면, 언어를 통해 인간의 삶을 미적(美的)으로 형상화한 것이라고 볼 수 있다.
```

#### Details

- Code assumes you are using `BertTokenizer`. (If you use a different tokenizer, you need to modify the code by yourself)
- `whole word masking` and `max ngram masking` are available.
- `static masking` (Decide how many times to double the data with different masking via dupe_factor)
- When creating an example through `long_seq_threshold`, a sentence that is too long is divided into multiple examples.
- Create examples with **full sentences**, according to the method used in `RoBERTa`

### 3. Pretraining with TPU

- **Strongly recommend to use `tensorflow==2.3.1` for TPU.**
- Warm started from BERT, and only `position embedding (size 4096)` was random initialized.
- `max_predictions_per_seq=640` (set greater than 600 of the original paper)
- Both `tokenizer` and `pretrain_config.json` will be saved in `output_dir` (Will be used when converting to huggingface transformers format)

#### How to run

- Upload `tfrecord` and `BERT checkpoint` (optional) to Google Storage

```bash
gsutil -m cp -r pretrain_tfrecords gs://{$BUCKET_NAME}  # tfrecord
gsutil -m cp -r init_checkpoint gs://{$BUCKET_NAME}  # BERT tf v1 ckpt
```

- In GCP cloud shell, set the instance and TPU with the following command

```bash
ctpu up --zone=europe-west4-a --tf-version=2.3.1 --tpu-size=v3-8 --machine-type=n1-standard-1 --disk-size-gb=20 --name={$GCP_NAME} --project={$PROJECT_NAME}
```

- Run training on instance

  - When using BERT tf v1 checkpoint for warm start, please add the following argument to the script.

  ```bash
  --init_checkpoint=gs://$BUCKET_NAME/init_checkpoint/model.ckpt-0
  ```

```bash
cd pretrain
pip3 install -r requirements.txt
bash scripts/base_size_tpu.sh
```

## Convert Tensorflow checkpoint to Huggingface Transformers format

```bash
python3 convert_bigbird_tf_to_pt.py \
    --checkpoint_dir $ORIG_TF_BERT_CKPT \
    --big_bird_config_file $BIGBIRD_CONFIG_PATH \
    --output_dir $PT_OUTPUT_DIR \
    --tokenizer_dir $TOKENIZER_DIR \
    --step_on_output
```

- If you don't specify `--big_bird_config_file`, the script will automatically use `pretrain_config.json` in tensorflow checkpoint.
- If `--tokenizer_dir` is not specified, the script will automatically use the `tokenizer` in tensorflow checkpoint.

## Reference

- [Original BigBird implementation](https://github.com/google-research/bigbird)
- [BERT tensorflow v1 implementation](https://github.com/google-research/bert)
- [BERT tensorflow v2 implementation](https://github.com/tensorflow/models/tree/d4c5f8975a7b89f01421101882bc8922642c2314/official/nlp/bert)
- [ELECTRA implementation](https://github.com/google-research/electra)
- [Huggingface Transformers Documentation](https://huggingface.co/transformers/)
- [ko-lm-dataformat](https://github.com/monologg/ko_lm_dataformat)
