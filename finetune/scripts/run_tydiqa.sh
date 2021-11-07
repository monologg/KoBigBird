#!/bin/bash

python3 run.py \
--task qa \
--dataset tydiqa \
--do_train \
--do_eval_during_train \
--do_eval \
--use_tpu \
--model_name_or_path monologg/kobigbird-bert-base \
--data_dir cache/tydiqa \
--train_file tydiqa-v1.0-train.jsonl \
--predict_file tydiqa-v1.0-dev.jsonl \
--max_seq_length 4096 \
--doc_stride 3072 \
--max_answer_length 32 \
--version_2_with_negative \
--train_batch_size 2 \
--eval_batch_size 1 \
--learning_rate 3e-5 \
--gradient_accumulation_steps 4 \
--num_train_epochs 5
