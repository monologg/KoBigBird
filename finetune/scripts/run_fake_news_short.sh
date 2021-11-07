#!/bin/bash

python3 run.py \
--task cls \
--dataset fake_news \
--do_train \
--do_eval_during_train \
--do_eval \
--use_tpu \
--model_name_or_path klue/roberta-base \
--data_dir cache/fake_news_data \
--train_file mission2_train.csv \
--predict_file mission2_train.csv \
--max_seq_length 512 \
--train_batch_size 8 \
--eval_batch_size 8 \
--learning_rate 3e-5 \
--gradient_accumulation_steps 1 \
--num_labels 2 \
--num_train_epochs 10
