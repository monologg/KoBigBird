#!/bin/bash

### GCP Settings (SHOULD change for your own settings) ###
BUCKET_NAME="kobigbird"
MODEL_NAME="kobigbird-bert-base"
TPU_NAME="kobigbird-bert-base"
TPU_ZONE="europe-west4-a"
NUM_TPU_CORES=8
###########################################

ATTENTION_TYPE="block_sparse"
MAX_ENCODER_LENGTH=4096
BLOCK_SIZE=64

NUM_TRAIN_STEPS=2000000
SAVE_CHECKPOINTS_STEPS=100000
NUM_WARMUP_STEPS=20000
ITERATIONS_PER_LOOP=200
LEARNING_RATE=1e-4
KEEP_CHECKPOINT_MAX=20
MAX_PREDICTIONS_PER_SEQ=640
TRAIN_BATCH_SIZE_PER_DEVICE=4
EVAL_BATCH_SIZE_PER_DEVICE=8

USE_GRADIENT_CHECKPOINTING=False
USE_NSP=False
RANDOM_POS_EMB=True

SEED=42

python3 run_pretraining.py \
  --data_dir="gs://$BUCKET_NAME/pretrain_tfrecords" \
  --output_dir="gs://$BUCKET_NAME/models/$MODEL_NAME" \
  --attention_type=$ATTENTION_TYPE \
  --max_encoder_length=$MAX_ENCODER_LENGTH \
  --max_position_embeddings=$MAX_ENCODER_LENGTH \
  --block_size=$BLOCK_SIZE \
  --num_attention_heads=12 \
  --num_hidden_layers=12 \
  --hidden_size=768 \
  --intermediate_size=3072 \
  --do_train=True \
  --do_eval=False \
  --use_tpu=True \
  --tpu_name=$TPU_NAME \
  --tpu_zone=$TPU_ZONE \
  --num_tpu_cores=$NUM_TPU_CORES \
  --train_batch_size=$TRAIN_BATCH_SIZE_PER_DEVICE \
  --eval_batch_size=$EVAL_BATCH_SIZE_PER_DEVICE \
  --num_train_steps=$NUM_TRAIN_STEPS \
  --num_warmup_steps=$NUM_WARMUP_STEPS \
  --save_checkpoints_steps=$SAVE_CHECKPOINTS_STEPS \
  --learning_rate=$LEARNING_RATE \
  --keep_checkpoint_max=$KEEP_CHECKPOINT_MAX \
  --use_gradient_checkpointing=$USE_GRADIENT_CHECKPOINTING \
  --use_nsp=$USE_NSP \
  --max_predictions_per_seq=$MAX_PREDICTIONS_PER_SEQ \
  --random_pos_emb=$RANDOM_POS_EMB \
  --iterations_per_loop=$ITERATIONS_PER_LOOP \
  --seed=$SEED
