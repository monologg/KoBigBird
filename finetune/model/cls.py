import os

import torch
from data.cls import process_map
from model.base import BaseModel
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


class ClsModel(BaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        config.label2id = self.config.label2id

    def from_pretrained(self):
        data_file = os.path.join(self.config.data_dir, str(self.config.train_file))
        self.config.label2id = process_map[self.config.dataset](self.config, data_file, True, get_label_map=True)
        num_labels = len(self.config.label2id)
        if num_labels != self.config.num_labels:
            print(
                f"given args num_labels({self.config.num_labels}) is not same with num_labels({num_labels}) from dataset."
            )
            print(f"switch num_labels {self.config.num_labels} -> {num_labels}")
            self.config.num_labels = num_labels
        model_config = AutoConfig.from_pretrained(self.config.model_name_or_path, num_labels=self.config.num_labels)
        model_config.label2id = self.config.label2id
        model_config.id2label = {int(v): k for k, v in model_config.label2id.items()}
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name_or_path, config=model_config, cache_dir=self.config.cache_dir
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path, cache_dir=self.config.cache_dir)

    def forward(self, inputs):
        outputs = self.model(**inputs)
        return outputs

    def eval_step(self, inputs, outputs):
        logits = outputs.logits.detach().cpu()
        predictions = self.tensor_to_list(torch.argmax(logits, dim=-1))
        labels = self.tensor_to_list(inputs["labels"])
        results = [{"prediction": prediction, "label": label} for prediction, label in zip(predictions, labels)]
        return results

    @staticmethod
    def add_args(parser):
        parser = BaseModel.add_args(parser)
        parser.add_argument("--num_labels", default=2, type=int)
        return parser
