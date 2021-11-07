#!/bin/bash

python3 run.py \
--task qa \
--dataset korquad_2 \
--do_train \
--do_eval_during_train \
--do_eval \
--use_tpu \
--model_name_or_path klue/roberta-base \
--data_dir cache/korquad_2 \
--train_file train \
--predict_file dev \
--max_seq_length 512 \
--doc_stride 384 \
--max_answer_length 512 \
--train_batch_size 8 \
--eval_batch_size 8 \
--learning_rate 3e-5 \
--gradient_accumulation_steps 1 \
--num_train_epochs 5
