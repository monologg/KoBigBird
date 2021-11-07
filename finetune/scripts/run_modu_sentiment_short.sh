#!/bin/bash

python3 run.py \
--task cls \
--dataset modu_sentiment \
--do_train \
--do_eval_during_train \
--do_eval \
--use_tpu \
--model_name_or_path klue/roberta-base \
--data_dir cache/modu-corpus/sentiment-analysis \
--train_file EXSA2002108040.json \
--predict_file EXSA2002108040.json \
--max_seq_length 512 \
--train_batch_size 8 \
--eval_batch_size 8 \
--learning_rate 3e-5 \
--gradient_accumulation_steps 1 \
--num_labels 2 \
--num_train_epochs 20
