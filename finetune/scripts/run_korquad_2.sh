#!/bin/bash

python3 run.py \
--task qa \
--dataset korquad_2 \
--do_train \
--do_eval_during_train \
--do_eval \
--use_tpu \
--model_name_or_path monologg/kobigbird-bert-base \
--data_dir cache/korquad_2 \
--train_file train \
--predict_file dev \
--max_seq_length 4096 \
--doc_stride 3072 \
--max_answer_length 4096 \
--train_batch_size 2 \
--eval_batch_size 1 \
--learning_rate 3e-5 \
--gradient_accumulation_steps 4 \
--num_train_epochs 5
