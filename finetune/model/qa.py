import json
import os
import shutil

from model.base import BaseModel
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


class QAModel(BaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.pred_dir = os.path.join(self.config.output_dir, self.config.task, "predictions")
        if os.path.exists(self.pred_dir):
            shutil.rmtree(self.pred_dir)
        os.makedirs(self.pred_dir)

    def from_pretrained(self):
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            self.config.model_name_or_path,
            cache_dir=self.config.cache_dir,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path, cache_dir=self.config.cache_dir, use_fast=False
        )

    def forward(self, inputs):
        if self.model.config.model_type in ["roberta"]:
            inputs.pop("token_type_ids", None)
        unique_id = inputs.pop("unique_id", None)
        outputs = self.model(**inputs)
        inputs["unique_id"] = unique_id
        return outputs

    def eval_step(self, inputs, outputs):
        start_logits, end_logits = outputs.start_logits, outputs.end_logits
        start_logits = self.tensor_to_list(start_logits)
        end_logits = self.tensor_to_list(end_logits)
        sample_num = len(inputs["unique_id"])
        for i in range(sample_num):
            qas_id = "_".join(inputs["unique_id"][i].split("_")[:-1])
            writer = open(os.path.join(self.pred_dir, str(qas_id)), "a")
            write_data = {
                "unique_id": inputs["unique_id"][i],
                "start_logits": start_logits[i],
                "end_logits": end_logits[i],
            }
            writer.write(json.dumps(write_data) + "\n")
            writer.close()
        return [None] * sample_num

    @staticmethod
    def add_args(parser):
        parser = BaseModel.add_args(parser)
        parser.add_argument("--version_2_with_negative", action="store_true")
        parser.add_argument("--null_score_diff_threshold", default=0.0, type=float)
        parser.add_argument("--doc_stride", default=384, type=int)
        parser.add_argument("--max_query_length", default=64, type=int)
        parser.add_argument(
            "--n_best_size",
            default=20,
            type=int,
            help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
        )
        parser.add_argument(
            "--max_answer_length",
            default=32,
            type=int,
            help="The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another.",
        )
        parser.add_argument(
            "--all_korquad_2_sample",
            action="store_true",
            help="Use all training samples from korquad2 or not. Do not use all samples by default "
            "because of the limitation on computational resources",
        )
        return parser
