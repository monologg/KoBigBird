import copy
import json
import re

import numpy as np
import torch
import transformers.data.processors.squad as squad
from tqdm import tqdm


class SquadExample:
    """
    A single training/test example for the Squad dataset, as loaded from disk.

    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
        self,
        qas_id,
        question_text,
        context_text,
        answer_text,
        start_position_character,
        title,
        answers=[],
        is_impossible=False,
        doc_tokens=None,
        char_to_word_offset=None,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers

        self.start_position, self.end_position = 0, 0

        if doc_tokens is None:
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True

            # Split on whitespace so that different tokens may be attributed to their original position.
            for c in self.context_text:
                if squad._is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start and end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]


def squad_convert_example_to_features(
    example,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    padding_strategy,
    is_training,
    tok_to_orig_index=None,
    orig_to_tok_index=None,
    all_doc_tokens=None,
):
    features = []
    if is_training and not example.is_impossible:
        # Get start and end position
        start_position = example.start_position
        end_position = example.end_position

        # If the answer cannot be found in the text, then skip this example.
        actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])
        cleaned_answer_text = " ".join(squad.whitespace_tokenize(example.answer_text))
        if actual_text.find(cleaned_answer_text) == -1:
            return [], None, None, None

    if tok_to_orig_index is None:
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            if tokenizer.__class__.__name__ in [
                "RobertaTokenizer",
                "LongformerTokenizer",
                "BartTokenizer",
                "RobertaTokenizerFast",
                "LongformerTokenizerFast",
                "BartTokenizerFast",
            ]:
                sub_tokens = tokenizer.tokenize(token, add_prefix_space=True)
            else:
                sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = squad._improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
        )

    spans = []

    truncated_query = tokenizer.encode(
        example.question_text, add_special_tokens=False, truncation=True, max_length=max_query_length
    )

    # Tokenizers who insert 2 SEP tokens in-between <context> & <question> need to have special handling
    # in the way they compute mask of added tokens.
    tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
    sequence_added_tokens = (
        tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
        if tokenizer_type in squad.MULTI_SEP_TOKENS_TOKENIZERS_SET
        else tokenizer.model_max_length - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair

    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):

        # Define the side we want to truncate / pad and the text/pair sorting
        if tokenizer.padding_side == "right":
            texts = truncated_query
            pairs = span_doc_tokens
            truncation = squad.TruncationStrategy.ONLY_SECOND.value
        else:
            texts = span_doc_tokens
            pairs = truncated_query
            truncation = squad.TruncationStrategy.ONLY_FIRST.value

        encoded_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
            texts,
            pairs,
            truncation=truncation,
            padding=padding_strategy,
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
            return_token_type_ids=True,
        )

        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            if tokenizer.padding_side == "right":
                non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                last_padding_id_position = (
                    len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
                )
                non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1 :]

        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict or (
            "overflowing_tokens" in encoded_dict and len(encoded_dict["overflowing_tokens"]) == 0
        ):
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = squad._new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        p_mask = np.array([])

        # # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # # Original TF implementation also keep the classification token (set to 0)
        # p_mask = np.ones_like(span["token_type_ids"])
        # if tokenizer.padding_side == "right":
        #     p_mask[len(truncated_query) + sequence_added_tokens :] = 0
        # else:
        #     p_mask[-len(span["tokens"]) : -(len(truncated_query) + sequence_added_tokens)] = 0

        # pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)
        # special_token_indices = np.asarray(
        #     tokenizer.get_special_tokens_mask(span["input_ids"], already_has_special_tokens=True)
        # ).nonzero()

        # p_mask[pad_token_indices] = 1
        # p_mask[special_token_indices] = 1

        # # Set the cls index to 0: the CLS index can be used for impossible answers
        # p_mask[cls_index] = 0

        span_is_impossible = example.is_impossible
        start_position = 0
        end_position = 0
        if is_training and not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = len(truncated_query) + sequence_added_tokens

                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        features.append(
            squad.SquadFeatures(
                span["input_ids"],
                span["attention_mask"],
                span["token_type_ids"],
                cls_index,
                p_mask.tolist(),
                example_index=0,  # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                start_position=start_position,
                end_position=end_position,
                is_impossible=span_is_impossible,
                qas_id=example.qas_id,
            )
        )
    return features, tok_to_orig_index, orig_to_tok_index, all_doc_tokens


def sample_writer(data, config, tokenizer, is_train):
    """process and write single 'paragraph-context' from squad-formed QA dataset"""
    context = data["context"]
    example = None
    tok_to_orig_index = None
    orig_to_tok_index = None
    all_doc_tokens = None
    write_data = []
    for qas in data["qas"]:
        is_impossible = qas.get("is_impossible", False)
        example = SquadExample(
            qas_id=qas["id"],
            question_text=qas["question"],
            context_text=context,
            answer_text="" if is_impossible else qas["answers"][0]["text"],
            start_position_character=0 if is_impossible else qas["answers"][0]["answer_start"],
            title="",
            answers=qas["answers"],
            is_impossible=is_impossible,
            doc_tokens=(None if example is None else example.doc_tokens),
            char_to_word_offset=(None if example is None else example.char_to_word_offset),
        )
        features, tok_to_orig_index, orig_to_tok_index, all_doc_tokens = squad_convert_example_to_features(
            example=example,
            tokenizer=tokenizer,
            max_seq_length=config.max_seq_length,
            doc_stride=config.doc_stride,
            max_query_length=config.max_query_length,
            padding_strategy="max_length",
            is_training=is_train,
            tok_to_orig_index=tok_to_orig_index,
            orig_to_tok_index=orig_to_tok_index,
            all_doc_tokens=all_doc_tokens,
        )
        for i, feature in enumerate(features):
            write_datum = {
                "input_ids": feature.input_ids,
                "attention_mask": feature.attention_mask,
                "token_type_ids": feature.token_type_ids,
                "start_position": feature.start_position,
                "end_position": feature.end_position,
            }
            if not is_train:
                write_datum["id"] = feature.qas_id
                write_datum["unique_id"] = "{}_{}".format(feature.qas_id, i)
                write_datum["tokens"] = feature.tokens
                write_datum["token_to_orig_map"] = feature.token_to_orig_map
                write_datum["paragraph_len"] = feature.paragraph_len
                write_datum["token_is_max_context"] = feature.token_is_max_context
                if i == 0:  # only store example data for the first feature from the example
                    write_datum["is_impossible"] = example.is_impossible
                    write_datum["answers"] = example.answers
                    write_datum["doc_tokens"] = example.doc_tokens
                else:
                    write_datum["is_impossible"] = None
                    write_datum["answers"] = None
                    write_datum["doc_tokens"] = None
            write_data.append(write_datum)

    return write_data


def process_korquad_1(config, data_file, train):
    with open(data_file) as handle:
        load_data = json.load(handle)["data"]
        data = []
        for datum in load_data:
            data.extend(datum["paragraphs"])

    return data


def process_korquad_2(config, data_file, train):
    TAGS = [
        "<!DOCTYPE html>",
        "<html>",
        "</html>",
        "<meta>",
        "<head>",
        "</head>",
        "<body>",
        "</body>",
        "<label>",
        "</label>",
        "<div>",
        "</div>",
        "<a>",
        "</a>",
        "<span>",
        "</span>",
        "<link>",
        "<img>",
        "<sup>",
        "</sup>",
        "<input>",
        "<noscript></noscript>",
        "<p>",
        "</p>",
        "<cite>",
        "</cite>",
        "<tbody>",
        "</tbody>",
        "</br>",
        "<br/>",
        "<h1>",
        "</h1>",
        "<h2>",
        "</h2>",
        "<h3>",
        "</h3>",
        "<form>",
        "</form>",
        "<small>",
        "</small>",
        "<big>",
        "</big>",
        "<b>",
        "</b>",
        "<abbr>",
        "</abbr>",
        "[편집]",
    ]

    def remove_tags(text):
        # only retain meaningful html tags to extract answers
        # tags like <table> <tr> <td> <ul> <li> will remain
        text = re.sub("<title>.*?</title>", " ", text)
        text = text.replace('rowspan="', "r")  # summarize attribute name for col, row spans
        text = text.replace('colspan="', "c")  # which indicate merged table cells
        for tag in TAGS:
            text = text.replace(tag, "")
        text = re.sub(" +", " ", text)
        text = re.sub("\n+", "\n", text)
        return text

    with open(data_file) as handle:
        data = json.load(handle)["data"]

    qas_num = []
    for datum in data:
        context = datum["context"]
        datum["context"] = remove_tags(context)
        del datum["raw_html"]
        if train and not config.all_korquad_2_sample:  # only use first sample from context for train split
            datum["qas"] = datum["qas"][:1]
        for qas in datum["qas"]:
            del qas["answer"]["html_answer_text"]
            del qas["answer"]["html_answer_start"]
            # qas["id"] = "{}-{}".format(i, qas["id"])
            answer_prev_text = context[: qas["answer"]["answer_start"]]
            answer_prev_len = len(answer_prev_text)
            remove_tag_text = remove_tags(answer_prev_text)
            # adjust span position according to text without tags
            qas["answer"]["answer_start"] -= answer_prev_len - len(remove_tag_text)
            qas["answer"]["text"] = remove_tags(qas["answer"]["text"])
            qas["answers"] = [qas.pop("answer")]
        qas_num.append(len(datum["qas"]))
    # some paragraph in korquad_2 have too many question samples
    # it will slow down the progress of multiprocessing job
    # so, i split them
    limit_qas_num = int(np.percentile(np.array(qas_num), 50))
    flat_data = []
    for datum in data:
        qas_num = len(datum["qas"])
        if qas_num > limit_qas_num:
            for j in range(0, qas_num, limit_qas_num):
                _datum = copy.deepcopy(datum)
                _datum["qas"] = datum["qas"][j : j + limit_qas_num]
                flat_data.append(_datum)
        else:
            flat_data.append(datum)
    data = flat_data

    return data


def process_kluemrc(config, data_file, train):
    data = []
    with open(data_file) as handle:
        for datum in json.load(handle)["data"]:
            datum = datum["paragraphs"]
            for qas in datum:
                for q in qas["qas"]:
                    q["id"] = q.pop("guid")
            data.extend(datum)
    return data


def process_tydiqa(config, data_file, train):
    data = []
    total = sum([1 for _ in open(data_file)])
    for line in tqdm(open(data_file), total=total, dynamic_ncols=True):
        datum = json.loads(line)

        if datum["language"].lower().strip() != "korean":
            continue

        span_byte_map = {}
        prev_bytes = 0
        for i, char in enumerate(datum["document_plaintext"]):
            byte_len = len(char.encode("utf-8"))
            for j in range(byte_len):
                span_byte_map[prev_bytes + j] = i
            prev_bytes += byte_len

        answers = []
        bool_answers = []
        is_impossible = False
        for annot in datum["annotations"]:
            spans = annot["minimal_answer"]
            start = spans["plaintext_start_byte"]
            end = spans["plaintext_end_byte"]

            yesno = None if annot["yes_no_answer"] == "NONE" else annot["yes_no_answer"]
            if yesno is not None:
                bool_answers.append(yesno)
                continue

            if spans["plaintext_start_byte"] == -1:
                is_impossible = True
            else:
                start = span_byte_map[start]
                end = span_byte_map[end]
                answers.append(
                    {
                        "text": datum["document_plaintext"][start:end],
                        "answer_start": start,
                    }
                )

        # skip boolqa samples
        if len(bool_answers) != 0:
            continue

        if len(answers) != 0:
            is_impossible = False
        else:
            is_impossible = True

        data.append(
            {
                "context": datum["document_plaintext"],
                "qas": [
                    {
                        "id": len(data),
                        "question": datum["question_text"],
                        "is_impossible": is_impossible,
                        "answers": answers,
                    }
                ],
            }
        )

    return data


process_map = {
    "korquad_1": process_korquad_1,
    "korquad_2": process_korquad_2,
    "tydiqa": process_tydiqa,
    "kluemrc": process_kluemrc,
}


def collate_fn(features):
    input_ids = [sample["input_ids"] for sample in features]
    attention_mask = [sample["attention_mask"] for sample in features]
    token_type_ids = [sample["token_type_ids"] for sample in features]
    start_position = [sample["start_position"] for sample in features]
    end_position = [sample["end_position"] for sample in features]

    input_ids = torch.tensor(np.array(input_ids).astype(np.int64), dtype=torch.long)
    attention_mask = torch.tensor(np.array(attention_mask).astype(np.int8), dtype=torch.long)
    token_type_ids = torch.tensor(np.array(token_type_ids).astype(np.int8), dtype=torch.long)
    start_position = torch.tensor(np.array(start_position).astype(np.int64), dtype=torch.long)
    end_position = torch.tensor(np.array(end_position).astype(np.int64), dtype=torch.long)
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "start_positions": start_position,
        "end_positions": end_position,
    }
    if "unique_id" in features[0]:
        inputs["unique_id"] = [sample["unique_id"] for sample in features]
    return inputs
