#!/bin/bash

### SHOULD change for your own settings ###
INPUT_DIR=data
OUTPUT_DIR=pretrain_tfrecords
TOKENIZER_DIR=tokenizer
###########################################

NUM_PROCESSES=0
DUPE_FACTOR=3

MAX_SEQ_LENGTH=4096
MASKED_LM_PROB=0.15
MAX_PREDICTIONS_PER_SEQ=640

SENTENCE_PAIR_PROB=0.0
SHORT_SEQ_PROB=0.0
MAX_NGRAM_SIZE=2

LONG_SEQ_THRESHOLD=1.8

python3 create_pretraining_data.py \
     --input_dir $INPUT_DIR \
     --tokenizer_dir $TOKENIZER_DIR \
     --output_dir $OUTPUT_DIR \
     --max_seq_length $MAX_SEQ_LENGTH \
     --max_predictions_per_seq $MAX_PREDICTIONS_PER_SEQ \
     --num_processes $NUM_PROCESSES \
     --masked_lm_prob $MASKED_LM_PROB \
     --sentence_pair_prob $SENTENCE_PAIR_PROB \
     --short_seq_prob $SHORT_SEQ_PROB \
     --do_whole_word_mask \
     --max_ngram_size $MAX_NGRAM_SIZE \
     --dupe_factor $DUPE_FACTOR \
     --long_seq_threshold $LONG_SEQ_THRESHOLD \
     --debug
