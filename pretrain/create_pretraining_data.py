# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Writes out text data as tfrecords"""

import argparse
import collections
import itertools
import multiprocessing
import os
import random
import sys
import time
from pathlib import Path

import ko_lm_dataformat as kldf
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

# A _Gram is a [half-open) interval of token indices which form a word.
# E.g.,
#   words:  ["The", "doghouse"]
#   tokens: ["The", "dog", "##house"]
#   grams:  [(0,1), (1,3)]
_Gram = collections.namedtuple("_Gram", ["begin", "end"])


def _window(iterable, size):
    """Helper to create a sliding window iterator with a given size.
    E.g.,
      input = [1, 2, 3, 4]
      _window(input, 1) => [1], [2], [3], [4]
      _window(input, 2) => [1, 2], [2, 3], [3, 4]
      _window(input, 3) => [1, 2, 3], [2, 3, 4]
      _window(input, 4) => [1, 2, 3, 4]
      _window(input, 5) => None
    Args:
      iterable: elements to iterate over.
      size: size of the window.
    Yields:
      Elements of `iterable` batched into a sliding window of length `size`.
    """
    i = iter(iterable)
    window = []
    try:
        for e in range(0, size):
            window.append(next(i))
        yield window
    except StopIteration:
        # handle the case where iterable's length is less than the window size.
        return
    for e in i:
        window = window[1:] + [e]
        yield window


def _contiguous(sorted_grams):
    """Test whether a sequence of grams is contiguous.
    Args:
      sorted_grams: _Grams which are sorted in increasing order.
    Returns:
      True if `sorted_grams` are touching each other.
    E.g.,
      _contiguous([(1, 4), (4, 5), (5, 10)]) == True
      _contiguous([(1, 2), (4, 5)]) == False
    """
    for a, b in _window(sorted_grams, 2):
        if a.end != b.begin:
            return False
    return True


def _wordpieces_to_grams(tokens, special_tokens):
    """Reconstitue grams (words) from `tokens`.
    E.g.,
       tokens: ['[CLS]', 'That', 'lit', '##tle', 'blue', 'tru', '##ck', '[SEP]']
        grams: [          [1,2), [2,         4),  [4,5) , [5,       6)]
    Args:
      tokens: list of wordpieces
    Returns:
      List of _Grams representing spans of whole words
      (without "[CLS]" and "[SEP]").
    """
    grams = []
    gram_start_pos = None
    for i, token in enumerate(tokens):
        if gram_start_pos is not None and token.startswith("##"):
            continue
        if gram_start_pos is not None:
            grams.append(_Gram(gram_start_pos, i))
        if token not in special_tokens:
            gram_start_pos = i
        else:
            gram_start_pos = None
    if gram_start_pos is not None:
        grams.append(_Gram(gram_start_pos, len(tokens)))
    return grams


def _masking_ngrams(grams, max_ngram_size, max_masked_tokens):
    """Create a list of masking {1, ..., n}-grams from a list of one-grams.
    This is an extention of 'whole word masking' to mask multiple, contiguous
    words such as (e.g., "the red boat").
    Each input gram represents the token indices of a single word,
       words:  ["the", "red", "boat"]
       tokens: ["the", "red", "boa", "##t"]
       grams:  [(0,1), (1,2), (2,4)]
    For a `max_ngram_size` of three, possible outputs masks include:
      1-grams: (0,1), (1,2), (2,4)
      2-grams: (0,2), (1,4)
      3-grams; (0,4)
    Output masks will not overlap and contain less than `max_masked_tokens` total
    tokens.  E.g., for the example above with `max_masked_tokens` as three,
    valid outputs are,
         [(0,1), (1,2)]  # "the", "red" covering two tokens
         [(1,2), (2,4)]  # "red", "boa", "##t" covering three tokens
    The length of the selected n-gram follows a zipf weighting to
    favor shorter n-gram sizes (weight(1)=1, weight(2)=1/2, weight(3)=1/3, ...).
    Args:
      grams: List of one-grams.
      max_ngram_size: Maximum number of contiguous one-grams combined to create
        an n-gram.
      max_masked_tokens: Maximum total number of tokens to be masked.
    Returns:
      A list of n-grams to be used as masks.
    """
    if not grams:
        return None

    grams = sorted(grams)
    num_tokens = grams[-1].end

    # Ensure our grams are valid (i.e., they don't overlap).
    for a, b in _window(grams, 2):
        if a.end > b.begin:
            raise ValueError("overlapping grams: {}".format(grams))

    # Build map from n-gram length to list of n-grams.
    ngrams = {i: [] for i in range(1, max_ngram_size + 1)}
    for gram_size in range(1, max_ngram_size + 1):
        for g in _window(grams, gram_size):
            if _contiguous(g):
                # Add an n-gram which spans these one-grams.
                ngrams[gram_size].append(_Gram(g[0].begin, g[-1].end))

    # Shuffle each list of n-grams.
    for v in ngrams.values():
        random.shuffle(v)

    # Create the weighting for n-gram length selection.
    # Stored cummulatively for `random.choices` below.
    cummulative_weights = list(itertools.accumulate([1.0 / n for n in range(1, max_ngram_size + 1)]))

    output_ngrams = []
    # Keep a bitmask of which tokens have been masked.
    masked_tokens = [False] * num_tokens
    # Loop until we have enough masked tokens or there are no more candidate
    # n-grams of any length.
    # Each code path should ensure one or more elements from `ngrams` are removed
    # to guarentee this loop terminates.
    while sum(masked_tokens) < max_masked_tokens and sum(len(s) for s in ngrams.values()):
        # Pick an n-gram size based on our weights.
        sz = random.choices(range(1, max_ngram_size + 1), cum_weights=cummulative_weights)[0]

        # Ensure this size doesn't result in too many masked tokens.
        # E.g., a two-gram contains _at least_ two tokens.
        if sum(masked_tokens) + sz > max_masked_tokens:
            # All n-grams of this length are too long and can be removed from
            # consideration.
            ngrams[sz].clear()
            continue

        # All of the n-grams of this size have been used.
        if not ngrams[sz]:
            continue

        # Choose a random n-gram of the given size.
        gram = ngrams[sz].pop()
        num_gram_tokens = gram.end - gram.begin

        # Check if this would add too many tokens.
        if num_gram_tokens + sum(masked_tokens) > max_masked_tokens:
            continue

        # Check if any of the tokens in this gram have already been masked.
        if sum(masked_tokens[gram.begin : gram.end]):
            continue

        # Found a usable n-gram!  Mark its tokens as masked and add it to return.
        masked_tokens[gram.begin : gram.end] = [True] * (gram.end - gram.begin)
        output_ngrams.append(gram)
    return output_ngrams


def create_masked_lm_predictions(
    tokens,
    masked_lm_prob,
    max_predictions_per_seq,
    vocab_words,
    do_whole_word_mask,
    tokenizer,
    max_ngram_size=None,
):
    """Creates the predictions for the masked LM objective."""
    if do_whole_word_mask:
        grams = _wordpieces_to_grams(tokens=tokens, special_tokens=tokenizer.special_tokens_map.values())
    else:
        # Here we consider each token to be a word to allow for sub-word masking.
        if max_ngram_size:
            raise ValueError("cannot use ngram masking without whole word masking")
        grams = [
            _Gram(i, i + 1) for i in range(0, len(tokens)) if tokens[i] not in tokenizer.special_tokens_map.values()
        ]

    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))
    # Generate masks.  If `max_ngram_size` in [0, None] it means we're doing
    # whole word masking or token level masking.  Both of these can be treated
    # as the `max_ngram_size=1` case.
    masked_grams = _masking_ngrams(grams, max_ngram_size or 1, num_to_predict)
    masked_lms = []
    output_tokens = list(tokens)
    for gram in masked_grams:
        # 80% of the time, replace all n-gram tokens with [MASK]
        if random.random() < 0.8:
            replacement_action = lambda idx: tokenizer.mask_token
        else:
            # 10% of the time, keep all the original n-gram tokens.
            if random.random() < 0.5:
                replacement_action = lambda idx: tokens[idx]
            # 10% of the time, replace each n-gram token with a random word.
            else:
                replacement_action = lambda idx: random.choice(vocab_words)

        for idx in range(gram.begin, gram.end):
            output_tokens[idx] = replacement_action(idx)
            masked_lms.append(MaskedLmInstance(index=idx, label=tokens[idx]))

    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


class ExampleBuilder(object):
    """Given a stream of input text, creates pretraining examples."""

    def __init__(self, args, tokenizer):
        self.args = args
        self._tokenizer = tokenizer
        self._current_sentences = []
        self._current_length = 0
        self._max_length = args.max_seq_length
        self._long_seq_length_limit = int(self._max_length * args.long_seq_threshold)
        self._target_length = self._max_length
        self.n_build = 0
        self.vocab_for_random_replacement = []

        # NOTE This will except special tokens (e.g. [CLS], [unused0], <s>)
        for token in self._tokenizer.vocab.keys():
            if not (token.startswith("[unused") and token.endswith("]")):
                self.vocab_for_random_replacement.append(token)

        for special_token in self._tokenizer.special_tokens_map.values():
            if special_token not in self.vocab_for_random_replacement:
                self.vocab_for_random_replacement.append(special_token)

    def add_line(self, line):
        """Adds a line of text to the current example being built."""
        line = line.strip().replace("\n", " ")  # NOTE BertTokenizer will cover whitespace cleaning
        if not line:  # NOTE If it is empty string
            return None
        bert_tokens = self._tokenizer.tokenize(line)
        self._current_sentences.append(bert_tokens)
        self._current_length += len(bert_tokens)
        if self._current_length >= self._target_length:
            return self._create_example()
        return None

    def _create_example(self):
        """Creates a pre-training example from the current list of sentences."""
        # small chance to have two segment
        if random.random() < self.args.sentence_pair_prob:
            # -3 due to not yet having [CLS]/[SEP] tokens in the input text
            first_segment_target_length = (self._target_length - 3) // 2
        else:
            # NOTE It will be only one segment for BigBird
            first_segment_target_length = 100000

        first_segment = []
        second_segment = []
        for sentence in self._current_sentences:
            # the sentence goes to the first segment if (1) the first segment is
            # empty, (2) the sentence doesn't put the first segment over length or
            # (3) 50% of the time when it does put the first segment over length
            if (
                len(first_segment) == 0
                or len(first_segment) + len(sentence) < first_segment_target_length
                or (
                    len(second_segment) == 0
                    and len(first_segment) < first_segment_target_length
                    and random.random() < 0.5
                )
            ):
                first_segment += sentence
            else:
                second_segment += sentence

        example_lst = []
        all_segment_len = len(first_segment) + len(second_segment)

        # NOTE
        # If `first_seg + second_seg` is too long, we will make them to multiple chunk of single sentence
        if all_segment_len >= self._long_seq_length_limit:
            all_segment = first_segment + second_segment

            seq_len_for_example = self._max_length - 2  # NOTE -2 for [CLS]/[SEP]
            for i in range(0, all_segment_len, seq_len_for_example):
                example_lst.append(
                    self._make_tf_example(
                        first_segment=all_segment[i : i + seq_len_for_example],
                        second_segment=None,
                    )
                )
        else:
            # trim to max_length while accounting for not-yet-added [CLS]/[SEP] tokens
            first_segment = first_segment[: self._max_length - 2]
            second_segment = second_segment[: max(0, self._max_length - len(first_segment) - 3)]
            # NOTE Put it in List as batch size 1
            example_lst.append(self._make_tf_example(first_segment, second_segment))

        # prepare to start building the next example
        self._current_sentences = []
        self._current_length = 0
        # small chance for random-length instead of max_length-length example
        if random.random() < self.args.short_seq_prob:
            self._target_length = random.randint(5, self._max_length)
        else:
            self._target_length = self._max_length

        return example_lst

    def _make_tf_example(self, first_segment, second_segment):
        """Converts two "segments" of text into a tf.train.Example."""
        tokens = [self._tokenizer.cls_token] + first_segment + [self._tokenizer.sep_token]
        segment_ids = [0] * len(tokens)
        if second_segment:
            tokens += second_segment + [self._tokenizer.sep_token]
            segment_ids += [1] * (len(second_segment) + 1)

        # Masking
        (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(
            tokens=tokens,
            masked_lm_prob=self.args.masked_lm_prob,
            max_predictions_per_seq=self.args.max_predictions_per_seq,
            vocab_words=self.vocab_for_random_replacement,
            do_whole_word_mask=self.args.do_whole_word_mask,
            tokenizer=self._tokenizer,
            max_ngram_size=self.args.max_ngram_size,
        )
        # tokens -> input_ids
        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)

        # Padding
        input_ids += [self._tokenizer.pad_token_id] * (self._max_length - len(tokens))
        segment_ids += [0] * (self._max_length - len(segment_ids))

        masked_lm_positions = list(masked_lm_positions)
        masked_lm_ids = self._tokenizer.convert_tokens_to_ids(masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        while len(masked_lm_positions) < self.args.max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(self._tokenizer.pad_token_id)
            masked_lm_weights.append(0.0)

        assert len(input_ids) == self._max_length
        assert len(segment_ids) == self._max_length
        assert len(masked_lm_positions) == self.args.max_predictions_per_seq
        assert len(masked_lm_ids) == self.args.max_predictions_per_seq
        assert len(masked_lm_weights) == self.args.max_predictions_per_seq

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["segment_ids"] = create_int_feature(segment_ids)
        features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
        features["next_sentence_labels"] = create_int_feature([0])  # NOTE Dummy value

        self.n_build += 1

        if self.args.debug and self.n_build < 3:
            tf.compat.v1.logging.info("*** Example ***")
            tf.compat.v1.logging.info("tokens: %s", " ".join([printable_text(x) for x in tokens]))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.compat.v1.logging.info("%s: %s", feature_name, " ".join([str(x) for x in values]))

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        return tf_example


class ExampleWriter(object):
    """Writes pre-training examples to disk."""

    def __init__(
        self,
        args,
        job_id,
        num_out_files=1000,
    ):
        self.tokenizer_dir = args.tokenizer_dir
        self.output_dir = args.output_dir
        self.max_seq_length = args.max_seq_length
        self.num_jobs = args.num_processes

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir, use_fast=True)
        self._example_builder = ExampleBuilder(args, tokenizer)
        self._writers = []
        for i in range(num_out_files):
            if i % self.num_jobs == job_id:
                output_fname = os.path.join(
                    self.output_dir,
                    "pretrain_data.tfrecord-{:}-of-{:}".format(i, num_out_files),
                )
                self._writers.append(tf.io.TFRecordWriter(output_fname))
        self.n_written = 0

    def write_examples(self, input_file):
        """Writes out examples from the provided input file."""
        if input_file.endswith(".txt"):
            self.write_examples_from_txt(input_file)
        elif input_file.endswith(".jsonl.zst"):
            self.write_examples_from_kldf(input_file)
        else:
            print(f"{input_file} is not supported. Only supports `jsonl.zst` or `txt` file")

        # NOTE Flush
        if self._example_builder._current_length != 0:
            example_lst = self._example_builder._create_example()
            if example_lst:
                for example in example_lst:
                    self._writers[self.n_written % len(self._writers)].write(example.SerializeToString())
                    self.n_written += 1

    def write_examples_from_txt(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    example_lst = self._example_builder.add_line(line)
                    if example_lst:
                        for example in example_lst:
                            self._writers[self.n_written % len(self._writers)].write(example.SerializeToString())
                            self.n_written += 1

    def write_examples_from_kldf(self, input_file):
        rdr = kldf.Reader(input_file)
        for doc in rdr.stream_data(get_meta=False, threaded=False):
            if type(doc) == str:
                doc = [doc]

            for line in doc:
                line = line.strip()
                if line:
                    example_lst = self._example_builder.add_line(line)
                    if example_lst:
                        for example in example_lst:
                            self._writers[self.n_written % len(self._writers)].write(example.SerializeToString())
                            self.n_written += 1

    def finish(self):
        for writer in self._writers:
            writer.close()


def write_examples(fnames, job_id, args):
    """A single process creating and writing out pre-processed examples."""

    def log(*args):
        msg = " ".join(map(str, args))
        print("Job {}:".format(job_id), msg)

    log("Creating example writer")
    example_writer = ExampleWriter(args=args, job_id=job_id)
    log("Writing tf examples")
    fnames = [f for (i, f) in enumerate(fnames) if i % args.num_processes == job_id]

    # https://pytorch.org/docs/stable/data.html#data-loading-randomness
    seed_worker(job_id, args)

    random.shuffle(fnames)
    start_time = time.time()
    # Add dupe factor
    file_processed = 0
    for dupe_idx in range(args.dupe_factor):
        for file_no, fname in enumerate(fnames):
            if file_processed > 0:
                elapsed = time.time() - start_time
                log(
                    "Processed dupe {:}, {:}/{:} files ({:.1f}%), "
                    "{:} examples written, (ELAPSED: {:}s, ETA: {:}s)".format(
                        dupe_idx,
                        file_no,
                        len(fnames),
                        100.0 * file_no / len(fnames),
                        example_writer.n_written,
                        int(elapsed),
                        int((len(fnames) * args.dupe_factor - file_processed) / (file_processed / elapsed)),
                    )
                )
            example_writer.write_examples(fname)
            file_processed += 1
    example_writer.finish()
    log("Done!")


def log_config(config):
    def log(*args):
        msg = ": ".join(map(str, args))
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()

    for key, value in sorted(config.__dict__.items()):
        log(key, value)
    log()


def seed_worker(job_id, args):
    worker_seed = (args.seed + job_id) % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_files(input_dir):
    """
    Get all files from input directory.
    ONLY support `jsonl.zst` and `txt` (kldf format or plain text)
    """
    filetypes = ["jsonl.zst", "txt"]
    files = [list(Path(input_dir).rglob(f"*.{ft}")) for ft in filetypes]
    # flatten list of list -> list and stringify Paths
    flattened_list = [str(item) for sublist in files for item in sublist]
    if not flattened_list:
        raise Exception(
            f"""did not find any files at this path {input_dir}, please also ensure your files are in format {filetypes}"""
        )
    flattened_list = sorted(flattened_list)
    return flattened_list


def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, help="Location of text or ko_lm_dataformat files.")
    parser.add_argument("--tokenizer_dir", required=True, help="Location of tokenizer directory.")
    parser.add_argument("--output_dir", required=True, help="Where to write out the tfrecords.")
    parser.add_argument("--max_seq_length", default=512, type=int, help="Number of tokens per example.")
    parser.add_argument(
        "--max_predictions_per_seq",
        default=76,
        type=int,
        help="Maximum number of masked LM predictions per sequence.",
    )
    parser.add_argument(
        "--num_processes",
        default=0,
        type=int,
        help="Parallelize across multiple processes. 0 will set the number of detected logical CPUs.",
    )
    parser.add_argument(
        "--sentence_pair_prob",
        default=0.05,
        type=float,
        help="Probability for make input with sentence pair ([CLS] text_a [SEP] text_b [SEP])",
    )
    parser.add_argument(
        "--short_seq_prob",
        default=0.01,
        type=float,
        help="Probability of creating sequences which are shorter than the maximum length.",
    )
    parser.add_argument(
        "--do_whole_word_mask",
        action="store_true",
        help="Whether to use whole word masking rather than per-WordPiece masking.",
    )
    parser.add_argument(
        "--max_ngram_size",
        type=int,
        default=None,
        help="Mask contiguous whole words (n-grams) of up to `max_ngram_size` using a weighting scheme to favor shorter n-grams. "
        "Note: `--do_whole_word_mask=True` must also be set when n-gram masking.",
    )
    parser.add_argument(
        "--dupe_factor",
        type=int,
        default=2,
        help="Number of times to duplicate the input data (with different masks).",
    )
    parser.add_argument("--masked_lm_prob", type=float, default=0.15, help="Masked LM probability.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug the result of tokenization, masking etc.",
    )
    parser.add_argument("--seed", default=12345, type=int, help="Random seed for data generation.")
    parser.add_argument(
        "--long_seq_threshold",
        default=1.8,
        type=float,
        help="Threshold for extremely long sequence. "
        "If sequence >= int(max_seq_len * threshold), split the sequence into multiple chunk",
    )

    args = parser.parse_args()

    assert args.long_seq_threshold > 1.0

    log_config(args)

    if tf.io.gfile.exists(args.output_dir):
        tf.io.gfile.rmtree(args.output_dir)
    if not tf.io.gfile.exists(args.output_dir):
        tf.io.gfile.makedirs(args.output_dir)

    fnames = get_files(args.input_dir)
    print(f"Total number of files: {len(fnames)}")

    if args.num_processes == 1:
        write_examples(fnames, 0, args)
    else:
        if args.num_processes == 0:
            args.num_processes = multiprocessing.cpu_count()

        jobs = []
        for i in range(args.num_processes):
            job = multiprocessing.Process(target=write_examples, args=(fnames, i, args))
            jobs.append(job)
            job.start()
        for job in jobs:
            job.join()


if __name__ == "__main__":
    main()
