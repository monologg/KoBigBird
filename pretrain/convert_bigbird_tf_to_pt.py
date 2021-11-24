# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Converts BigBird checkpoint."""


import argparse
import json
import os

import torch
import transformers
from transformers import AutoTokenizer, BigBirdConfig, BigBirdForPreTraining, load_tf_weights_in_big_bird
from transformers.utils import check_min_version, logging
from transformers.utils.versions import require_version

logger = transformers.logging.get_logger()
logging.set_verbosity_info()
transformers.logging.enable_default_handler()
transformers.logging.enable_explicit_format()

# NOTE check library version
require_version("torch>=1.6.0", "To fix: pip install torch>=1.6.0")
check_min_version("4.11.3")

logger.warning("This script is tested on `transformers==4.11.3`. Might not work properly on the other version!")


def load_bigbird_config(config_filepath):
    """Loads bigbird config from original tf checkpoint. (pretrain_config.json)"""
    config_key_lst = [
        "attention_probs_dropout_prob",
        "attention_type",
        "block_size",
        "pad_token_id",
        "bos_token_id",
        "eos_token_id",
        "sep_token_id",
        "gradient_checkpointing",  # Originally `use_gradient_checkpointing`
        "hidden_dropout_prob",
        "hidden_size",
        "initializer_range",
        "intermediate_size",
        "max_position_embeddings",
        "num_attention_heads",
        "num_hidden_layers",
        "num_random_blocks",
        "pad_token_id",
        "rescale_embeddings",
        "type_vocab_size",
        "use_bias",
        "vocab_size",
    ]

    config = {
        "position_embedding_type": "absolute",
        "tokenizer_class": "BertTokenizer",  # NOTE Remove this one if you use other tokenizer.
    }

    with open(config_filepath, "r", encoding="utf-8") as f:
        tf_config = json.load(f)

    for config_key in config_key_lst:
        if config_key in tf_config:
            config[config_key] = tf_config[config_key]
        else:
            if config_key == "gradient_checkpointing":
                config[config_key] = tf_config["use_gradient_checkpointing"]
            elif config_key == "num_random_blocks":
                config[config_key] = tf_config["num_rand_blocks"]
            elif config_key == "rescale_embeddings":
                config[config_key] = tf_config["rescale_embedding"]
            else:
                raise KeyError(f"{config_key} not in tensorflow config!!")

    return dict(sorted(config.items()))


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, big_bird_config_file, output_dir, tokenizer_path):
    # Initialize PyTorch model
    config = BigBirdConfig.from_dict(load_bigbird_config(os.path.join(tf_checkpoint_path, big_bird_config_file)))
    print(f"Building PyTorch model from configuration: {config}")

    model = BigBirdForPreTraining(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_big_bird(model, tf_checkpoint_path, is_trivia_qa=False)

    # Save pytorch-model
    print(f"Save PyTorch model to {output_dir}")
    model.save_pretrained(output_dir)

    # NOTE Convert model which is compatible for torch<1.5
    pytorch_model = torch.load(os.path.join(output_dir, "pytorch_model.bin"))
    torch.save(
        pytorch_model,
        os.path.join(args.output_dir, "pytorch_model.bin"),
        _use_new_zipfile_serialization=False,
    )

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.model_max_length = config.max_position_embeddings  # 1024, 2048, 4096
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_dir", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--big_bird_config_file",
        default="pretrain_config.json",
        type=str,
        help="The config json file corresponding to the pre-trained BigBird model. \n"
        "This specifies the model architecture.",
    )
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--tokenizer_dir",
        default=None,
        type=str,
        help="Tokenizer path (include vocab.txt, tokenizer_config.json, special_tokens_map.json)\n"
        "If not specified, converter will check tokenizer dir in tensorflow checkpoint",
    )
    parser.add_argument(
        "--step_on_output",
        action="store_true",
        help="Whether to write step on pytorch output\ne.g. kobigbird-bert-base-200k, kobigbird-bert-base-600k",
    )
    args = parser.parse_args()

    # Write step on output_dir
    if args.step_on_output:
        with open(os.path.join(args.checkpoint_dir, "checkpoint"), "r", encoding="utf-8") as f:
            line = f.readline()  # read only first line & check with step of checkpoint to convert
            step = line.split("-")[-1][:-1]

        if len(step) <= 4:
            raise ValueError("Step should be bigger than 1k")
        step = step[:-4]

        args.output_dir = args.output_dir + f"-{step}k"

    # Check tokenizer path (If tokenizer_dir is not specified, get tokenizer path from tf checkpoint dir)
    tokenizer_path = args.tokenizer_dir if args.tokenizer_dir else os.path.join(args.checkpoint_dir, "tokenizer")

    convert_tf_checkpoint_to_pytorch(args.checkpoint_dir, args.big_bird_config_file, args.output_dir, tokenizer_path)
