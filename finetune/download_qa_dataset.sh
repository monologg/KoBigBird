#!/bin/bash

mkdir cache
cd cache

# Download qa datasets
## 1. tydiqa
mkdir -p tydiqa
wget -P tydiqa https://storage.googleapis.com/tydiqa/v1.0/tydiqa-v1.0-train.jsonl.gz
wget -P tydiqa https://storage.googleapis.com/tydiqa/v1.0/tydiqa-v1.0-dev.jsonl.gz
gzip -d tydiqa/tydiqa-v1.0-train.jsonl.gz
gzip -d tydiqa/tydiqa-v1.0-dev.jsonl.gz

## 2. korquad 2.1
mkdir -p korquad_2/train
for var in {0..12}
do
    var=$(printf %02d $var)
    wget -P korquad_2/train https://github.com/korquad/korquad.github.io/raw/master/dataset/KorQuAD_2.1/train/KorQuAD_2.1_train_${var}.zip
done

mkdir -p korquad_2/dev
for var in {0..1}
do
    var=$(printf %02d $var)
    wget -P korquad_2/dev https://github.com/korquad/korquad.github.io/raw/master/dataset/KorQuAD_2.1/dev/KorQuAD_2.1_dev_${var}.zip
done

cd korquad_2
cd train
unzip '*.zip'
rm *.zip
cd ..

cd dev
unzip '*.zip'
rm *.zip
cd ..

## 3. korquad 1.0
# mkdir korquad_1
# wget -P korquad_1 https://github.com/korquad/korquad.github.io/raw/master/dataset/KorQuAD_v1.0_train.json
# wget -P korquad_1 https://github.com/korquad/korquad.github.io/raw/master/dataset/KorQuAD_v1.0_dev.json

## 4. klue mrc
# mkdir -p kluemrc
# wget -P kluemrc https://raw.githubusercontent.com/KLUE-benchmark/KLUE/v1.1.0/klue_benchmark/klue-mrc-v1.1/klue-mrc-v1.1_train.json
# wget -P kluemrc https://raw.githubusercontent.com/KLUE-benchmark/KLUE/v1.1.0/klue_benchmark/klue-mrc-v1.1/klue-mrc-v1.1_dev.json
