import torch
from transformers import AdamW, get_linear_schedule_with_warmup


class BaseModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.from_pretrained()

    def save_pretrained(self, save_dir):
        self.model.save_pretrained(save_dir)
        for key in ["special_tokens_map_file", "tokenizer_file"]:
            self.tokenizer.init_kwargs.pop(key, None)
        self.tokenizer.save_pretrained(save_dir)

    def from_pretrained(self):
        raise NotImplementedError

    def forward(self, inputs):
        return self.model(**inputs)

    def eval_step(self, outputs):
        raise NotImplementedError

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            required=True,
        )
        parser.add_argument("--data_dir", default="cache", type=str)
        parser.add_argument("--train_file", default=None, type=str)
        parser.add_argument("--predict_file", default=None, type=str)

        parser.add_argument("--do_lower_case", default=False, type=bool)
        parser.add_argument("--max_seq_length", default=512, type=int)
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--max_grad_norm", default=1.0, type=float)
        parser.add_argument("--num_train_epochs", default=10, type=int)
        parser.add_argument("--train_batch_size", default=8, type=int)
        parser.add_argument("--eval_batch_size", default=16, type=int)
        parser.add_argument("--learning_rate", default=3e-5, type=float)
        parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
        parser.add_argument("--warmup_proportion", default=0.0, type=float)

        return parser

    def get_optimizer(self):
        """Prepare optimizer"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate, eps=self.config.adam_epsilon)
        return optimizer

    def get_scheduler(self, batch_num, optimizer):
        """Prepare scheduler"""
        if self.config.warmup_proportion == 0.0:
            return None

        t_total = batch_num // self.config.gradient_accumulation_steps * self.config.num_train_epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(t_total * self.config.warmup_proportion),
            num_training_steps=t_total,
        )

        return scheduler

    def tensor_to_array(self, tensor):
        return tensor.detach().cpu().numpy()

    def tensor_to_list(self, tensor):
        return self.tensor_to_array(tensor).tolist()
