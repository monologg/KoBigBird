# Copyright 2021 The BigBird Authors.
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

"""Run masked LM/next sentence pre-training for BigBird."""

import os
import time

import tensorflow.compat.v2 as tf
from absl import app, logging
from kobigbird import flags, modeling, optimization, training_utils, utils

FLAGS = flags.FLAGS

## Required parameters

flags.DEFINE_string("data_dir", "pretrain_tfrecords", "The input data dir. Should contain the TFRecord files.")

flags.DEFINE_string(
    "output_dir",
    "/tmp/bigb",
    "The output directory where the model checkpoints will be written.",
)

## Other parameters
flags.DEFINE_string(
    "init_checkpoint",
    None,
    "Initial checkpoint (usually from a pre-trained BERT model).",
)

flags.DEFINE_integer(
    "max_encoder_length",
    4096,
    "The maximum total input sequence length after BERT Wordpiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.",
)

flags.DEFINE_integer(
    "max_predictions_per_seq",
    640,
    "Maximum number of masked LM predictions per sequence.",
)

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer(
    "train_batch_size",
    4,
    "Local batch size for training. " "Total batch size will be multiplied by number gpu/tpu cores available.",
)

flags.DEFINE_integer(
    "eval_batch_size",
    4,
    "Local batch size for eval. " "Total batch size will be multiplied by number gpu/tpu cores available.",
)

flags.DEFINE_string(
    "optimizer",
    "AdamWeightDecay",
    "Optimizer to use. Can be Adafactor, Adam, and AdamWeightDecay.",
)

flags.DEFINE_float("learning_rate", 1e-4, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Total number of training steps to perform.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of steps to perform linear warmup.")

flags.DEFINE_integer("save_checkpoints_steps", 1000, "How often to save the model checkpoint.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_bool("use_nsp", False, "Whether to use next sentence prediction loss.")

flags.DEFINE_integer("keep_checkpoint_max", 5, "How many checkpoints to keep.")

flags.DEFINE_bool(
    "random_pos_emb", True, "Random initialize for positional embedding (original 512 doesn't matched to 4096)"
)

flags.DEFINE_integer("seed", 42, "Seed for Pretraining")


def input_fn_builder(
    data_dir,
    max_encoder_length,
    max_predictions_per_seq,
    is_training,
):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    feature_shapes = {
        "input_ids": [max_encoder_length],
        "segment_ids": [max_encoder_length],
        "masked_lm_positions": [max_predictions_per_seq],
        "masked_lm_ids": [max_predictions_per_seq],
        "masked_lm_weights": [max_predictions_per_seq],
        "next_sentence_labels": [1],
    }

    def _decode_record(record):
        """Decodes a record to a TensorFlow example."""
        name_to_features = {
            "input_ids": tf.io.FixedLenFeature([max_encoder_length], tf.int64),
            "segment_ids": tf.io.FixedLenFeature([max_encoder_length], tf.int64),
            "masked_lm_positions": tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_ids": tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_weights": tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
            "next_sentence_labels": tf.io.FixedLenFeature([1], tf.int64),  # NOTE Not using NSP task
        }
        example = tf.io.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # Load dataset and handle tfds separately
        if "tfds://" == data_dir[:7]:
            raise ValueError("We don't support tfds in kobigbird pretraining code")
        else:
            # NOTE Directly read TFRecord
            input_files = tf.io.gfile.glob(os.path.join(data_dir, "pretrain_data.tfrecord*"))

            # For training, we want a lot of parallel reading and shuffling.
            # For eval, we want no shuffling and parallel reading doesn't matter.
            if is_training:
                d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
                d = d.shuffle(buffer_size=len(input_files))

                # Non deterministic mode means that the interleaving is not exact.
                # This adds even more randomness to the training pipeline.
                d = d.interleave(
                    tf.data.TFRecordDataset,
                    deterministic=False,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE,
                )
            else:
                d = tf.data.TFRecordDataset(input_files)

        # NOTE Only accept preprocessed tfrecord
        d = d.map(_decode_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if is_training:
            d = d.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
            d = d.repeat()

        d = d.padded_batch(batch_size, feature_shapes, drop_remainder=True)  # For static shape
        return d

    return input_fn


def model_fn_builder(bert_config):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        model = modeling.BertModel(bert_config)
        masked_lm = MaskedLMLayer(
            bert_config["hidden_size"],
            bert_config["vocab_size"],
            model.embeder,
            initializer=utils.create_initializer(bert_config["initializer_range"]),
            activation_fn=utils.get_activation(bert_config["hidden_act"]),
        )
        next_sentence = NSPLayer(
            bert_config["hidden_size"],
            initializer=utils.create_initializer(bert_config["initializer_range"]),
        )

        sequence_output, pooled_output = model(
            features["input_ids"],
            training=is_training,
            token_type_ids=features.get("segment_ids"),
        )

        masked_lm_loss, masked_lm_log_probs = masked_lm(
            sequence_output,
            label_ids=features.get("masked_lm_ids"),
            label_weights=features.get("masked_lm_weights"),
            masked_lm_positions=features.get("masked_lm_positions"),
        )

        if bert_config["use_nsp"]:
            next_sentence_loss, next_sentence_log_probs = next_sentence(
                pooled_output, features.get("next_sentence_labels")
            )
            total_loss = masked_lm_loss + next_sentence_loss
        else:
            total_loss = masked_lm_loss

        tvars = tf.compat.v1.trainable_variables()
        utils.log_variables(tvars, bert_config["ckpt_var_list"])

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            learning_rate = optimization.get_linear_warmup_linear_decay_lr(
                init_lr=bert_config["learning_rate"],
                num_train_steps=bert_config["num_train_steps"],
                num_warmup_steps=bert_config["num_warmup_steps"],
            )

            optimizer = optimization.get_optimizer(bert_config, learning_rate)

            global_step = tf.compat.v1.train.get_global_step()

            gradients = optimizer.compute_gradients(total_loss, tvars)
            train_op = optimizer.apply_gradients(gradients, global_step=global_step)

            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                host_call=utils.add_scalars_to_summary(bert_config["output_dir"], {"learning_rate": learning_rate}),
                training_hooks=[
                    training_utils.ETAHook(
                        {} if bert_config["use_tpu"] else dict(loss=total_loss),
                        bert_config["num_train_steps"],
                        bert_config["iterations_per_loop"],
                        bert_config["use_tpu"],
                    )
                ],
            )

        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(
                masked_lm_loss_value,
                masked_lm_log_probs,
                masked_lm_ids,
                masked_lm_weights,
                next_sentence_loss_value,
                next_sentence_log_probs,
                next_sentence_labels,
            ):
                """Computes the loss and accuracy of the model."""
                masked_lm_predictions = tf.argmax(masked_lm_log_probs, axis=-1, output_type=tf.int32)
                masked_lm_accuracy = tf.compat.v1.metrics.accuracy(
                    labels=masked_lm_ids,
                    predictions=masked_lm_predictions,
                    weights=masked_lm_weights,
                )
                masked_lm_mean_loss = tf.compat.v1.metrics.mean(values=masked_lm_loss_value)

                next_sentence_predictions = tf.argmax(next_sentence_log_probs, axis=-1, output_type=tf.int32)
                next_sentence_accuracy = tf.compat.v1.metrics.accuracy(
                    labels=next_sentence_labels, predictions=next_sentence_predictions
                )
                next_sentence_mean_loss = tf.compat.v1.metrics.mean(values=next_sentence_loss_value)

                return {
                    "masked_lm_accuracy": masked_lm_accuracy,
                    "masked_lm_loss": masked_lm_mean_loss,
                    "next_sentence_accuracy": next_sentence_accuracy,
                    "next_sentence_loss": next_sentence_mean_loss,
                }

            eval_metrics = (
                metric_fn,
                [
                    masked_lm_loss,
                    masked_lm_log_probs,
                    features["masked_lm_ids"],
                    features["masked_lm_weights"],
                    next_sentence_loss,
                    next_sentence_log_probs,
                    features["next_sentence_labels"],
                ],
            )
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                training_hooks=[
                    training_utils.ETAHook(
                        {} if bert_config["use_tpu"] else dict(loss=total_loss),
                        bert_config["max_eval_steps"],
                        bert_config["iterations_per_loop"],
                        bert_config["use_tpu"],
                        is_training=False,
                    )
                ],
            )
        else:

            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={
                    "log-probabilities": masked_lm_log_probs,
                    "seq-embeddings": sequence_output,
                },
            )

        return output_spec

    return model_fn


class MaskedLMLayer(tf.keras.layers.Layer):
    """Get loss and log probs for the masked LM."""

    def __init__(
        self,
        hidden_size,
        vocab_size,
        embeder,
        initializer=None,
        activation_fn=None,
        name="cls/predictions",
    ):
        super(MaskedLMLayer, self).__init__(name=name)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embeder = embeder

        # NOTE fixed by @monologg
        # change the name of scope for BERT init checkpoint
        with tf.compat.v1.variable_scope(name):
            # We apply one more non-linear transformation before the output layer.
            # This matrix is not used after pre-training.
            self.extra_layer = utils.Dense2dLayer(
                hidden_size, hidden_size, initializer, activation_fn, name="transform/dense"
            )
            self.norm_layer = utils.NormLayer(hidden_size, name="transform/LayerNorm")

            # The output weights are the same as the input embeddings, but there is
            # an output-only bias for each token.
            self.output_bias = tf.compat.v1.get_variable(
                "output_bias", shape=[vocab_size], initializer=tf.zeros_initializer()
            )

    @property
    def trainable_weights(self):
        self._trainable_weights = (
            self.extra_layer.trainable_weights + self.norm_layer.trainable_weights + [self.output_bias]
        )
        return self._trainable_weights

    def call(self, input_tensor, label_ids=None, label_weights=None, masked_lm_positions=None):
        if masked_lm_positions is not None:
            input_tensor = tf.gather(input_tensor, masked_lm_positions, batch_dims=1)

        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        input_tensor = self.extra_layer(input_tensor)
        input_tensor = self.norm_layer(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        logits = self.embeder.linear(input_tensor)
        logits = tf.nn.bias_add(logits, self.output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        if label_ids is not None:
            one_hot_labels = tf.one_hot(label_ids, depth=self.vocab_size, dtype=tf.float32)

            # The `positions` tensor might be zero-padded (if the sequence is too
            # short to have the maximum number of predictions). The `label_weights`
            # tensor has a value of 1.0 for every real prediction and 0.0 for the
            # padding predictions.
            per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=-1)
            numerator = tf.reduce_sum(label_weights * per_example_loss)
            denominator = tf.reduce_sum(label_weights) + 1e-5
            loss = numerator / denominator
        else:
            loss = tf.constant(0.0)

        return loss, log_probs


class NSPLayer(tf.keras.layers.Layer):
    """Get loss and log probs for the next sentence prediction."""

    def __init__(self, hidden_size, initializer=None, name="cls/seq_relationship"):
        super(NSPLayer, self).__init__(name=name)
        self.hidden_size = hidden_size

        # Simple binary classification. Note that 0 is "next sentence" and 1 is
        # "random sentence". This weight matrix is not used after pre-training.
        with tf.compat.v1.variable_scope(name):
            self.output_weights = tf.compat.v1.get_variable(
                "output_weights", shape=[2, hidden_size], initializer=initializer
            )
            self._trainable_weights.append(self.output_weights)
            self.output_bias = tf.compat.v1.get_variable("output_bias", shape=[2], initializer=tf.zeros_initializer())
            self._trainable_weights.append(self.output_bias)

    def call(self, input_tensor, next_sentence_labels=None):
        logits = tf.matmul(input_tensor, self.output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, self.output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        if next_sentence_labels is not None:
            labels = tf.reshape(next_sentence_labels, [-1])
            one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
        else:
            loss = tf.constant(0.0)
        return loss, log_probs


def main(_):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.compat.v1.random.set_random_seed(FLAGS.seed)

    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train`, `do_eval` must be True.")

    bert_config = flags.as_dictionary()

    if FLAGS.max_encoder_length > bert_config["max_position_embeddings"]:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d"
            % (FLAGS.max_encoder_length, bert_config["max_position_embeddings"])
        )

    tf.io.gfile.makedirs(FLAGS.output_dir)
    if FLAGS.do_train:
        # Save pretrain_config.json
        flags.save(os.path.join(FLAGS.output_dir, "pretrain_config.json"))
        training_utils.sys_log(f"Save pretrain_config.json at `{FLAGS.output_dir}`")

        # Save tokenizer to output dir
        tokenizer_output_dir = os.path.join(FLAGS.output_dir, "tokenizer")
        if not tf.io.gfile.exists(tokenizer_output_dir):
            tf.io.gfile.makedirs(tokenizer_output_dir)

            for filename in os.listdir(FLAGS.tokenizer_dir):
                tf.io.gfile.copy(
                    os.path.join(FLAGS.tokenizer_dir, filename),
                    os.path.join(tokenizer_output_dir, filename),
                )
        else:
            training_utils.sys_log(f"Tokenizer is already saved at `{tokenizer_output_dir}`")

    model_fn = model_fn_builder(bert_config)
    estimator = utils.get_estimator(bert_config, model_fn)

    if FLAGS.do_train:
        logging.info("***** Running training *****")
        logging.info("  Batch size = %d", estimator.train_batch_size)
        logging.info("  Num steps = %d", FLAGS.num_train_steps)
        train_input_fn = input_fn_builder(
            data_dir=FLAGS.data_dir,
            max_encoder_length=FLAGS.max_encoder_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=True,
        )
        estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)

    if FLAGS.do_eval:
        logging.info("***** Running evaluation *****")
        logging.info("  Batch size = %d", estimator.eval_batch_size)

        eval_input_fn = input_fn_builder(
            data_dir=FLAGS.data_dir,
            max_encoder_length=FLAGS.max_encoder_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=False,
        )

        # Run continuous evaluation for latest checkpoint as training progresses.
        last_evaluated = None
        while True:
            latest = tf.train.latest_checkpoint(FLAGS.output_dir)
            if latest == last_evaluated:
                if not latest:
                    logging.info("No checkpoints found yet.")
                else:
                    logging.info("Latest checkpoint %s already evaluated.", latest)
                time.sleep(300)
                continue
            else:
                logging.info("Evaluating check point %s", latest)
                last_evaluated = latest

                current_step = int(os.path.basename(latest).split("-")[1])
                output_eval_file = os.path.join(FLAGS.output_dir, "eval_results_{}.txt".format(current_step))
                result = estimator.evaluate(
                    input_fn=eval_input_fn,
                    steps=FLAGS.max_eval_steps,
                    checkpoint_path=latest,
                )

                with tf.io.gfile.GFile(output_eval_file, "w") as writer:
                    logging.info("***** Eval results *****")
                    for key in sorted(result.keys()):
                        logging.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    tf.compat.v1.disable_v2_behavior()
    tf.compat.v1.enable_resource_variables()
    app.run(main)
