#!/bin/bash

python3 run.py \
--task cls \
--dataset modu_sentiment \
--do_train \
--do_eval_during_train \
--do_eval \
--use_tpu \
--model_name_or_path monologg/kobigbird-bert-base \
--data_dir cache/modu-corpus/sentiment-analysis \
--train_file EXSA2002108040.json \
--predict_file EXSA2002108040.json \
--max_seq_length 1024 \
--train_batch_size 4 \
--eval_batch_size 2 \
--learning_rate 3e-5 \
--gradient_accumulation_steps 2 \
--num_labels 2 \
--num_train_epochs 20
