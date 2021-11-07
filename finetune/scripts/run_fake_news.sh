#!/bin/bash

python3 run.py \
--task cls \
--dataset fake_news \
--do_train \
--do_eval_during_train \
--do_eval \
--use_tpu \
--model_name_or_path monologg/kobigbird-bert-base \
--data_dir cache/fake_news_data \
--train_file mission2_train.csv \
--predict_file mission2_train.csv \
--max_seq_length 1024 \
--train_batch_size 4 \
--eval_batch_size 2 \
--learning_rate 3e-5 \
--gradient_accumulation_steps 2 \
--num_labels 2 \
--num_train_epochs 10
