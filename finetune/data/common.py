import json
import logging
import multiprocessing
import os

import torch
import torch.utils.data as torch_data
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

from data import cls as cls_data
from data import qa as qa_data
from datasets import load_dataset

DATA_PROCESSOR = {"cls": cls_data, "qa": qa_data}


def get_data(config, tokenizer, is_train=True, overwrite=False):
    if is_train:
        data_file = config.train_file
    else:
        data_file = config.predict_file

    data_path = config.data_dir
    if data_file is not None:
        data_path = os.path.join(data_path, data_file)
    else:
        data_path += "/"

    data_processor = DATA_PROCESSOR.get(config.task, None)
    if data_processor is None:
        raise Exception(f"Invalid data task {config.task}!")

    processor = data_processor.process_map.get(config.dataset, None)
    if processor is None:
        raise Exception(f"Invalid task dataset {config.dataset}!")

    comps = [
        data_path,
        config.dataset,
        config.model_name_or_path.replace("/", "_"),
        config.max_seq_length,
        "train" if is_train else "dev",
        "dataset.txt",
    ]
    dataset_file = "_".join([str(comp) for comp in comps])

    if not os.path.exists(dataset_file) or overwrite:
        with open(dataset_file, "w", encoding="utf-8") as writer_file:
            if data_file is None or not os.path.isdir(data_path):
                data = processor(config, data_path, is_train)
                cnt = write_samples(
                    config, tokenizer, is_train, data_processor, writer_file, data, workers=config.threads
                )
            else:
                cnt = 0
                for filename in sorted([f for f in os.listdir(data_path) if f.endswith(".json")]):
                    data = processor(config, os.path.join(data_path, filename), is_train)
                    cnt += write_samples(
                        config, tokenizer, is_train, data_processor, writer_file, data, workers=config.threads
                    )
            logger.info(f"{cnt} features processed from {data_path}")

    dataset = load_dataset("text", data_files=dataset_file)["train"]
    dataset = dataset.map(lambda x: json.loads(x["text"]), batched=False)

    if not is_train:
        # for valid datasets, we pad datasets so that no sample will be skiped in multi-device settings
        dataset = IterableDatasetPad(
            dataset=dataset,
            batch_size=config.train_batch_size if is_train else config.eval_batch_size,
            num_devices=config.world_size,
            seed=config.seed,
        )

    dataloader = torch_data.DataLoader(
        dataset,
        sampler=torch_data.RandomSampler(dataset) if is_train else None,
        drop_last=False,
        batch_size=config.train_batch_size if is_train else config.eval_batch_size,
        collate_fn=(data_processor.collate_fn),
    )

    return dataloader


config = None
tokenizer = None
is_train = None
writer = None


def init_sample_writer(_config, _tokenizer, _is_train, _writer):
    global config
    global tokenizer
    global is_train
    global writer
    config = _config
    tokenizer = _tokenizer
    is_train = _is_train
    writer = _writer


def sample_writer(data):
    global config
    global tokenizer
    global is_train
    global writer
    return writer(data, config, tokenizer, is_train)


def write_samples(config, tokenizer, is_train, processor, writer_file, data, workers=4):
    write_cnt = 0
    with multiprocessing.Pool(
        processes=workers,
        initializer=init_sample_writer,
        initargs=(config, tokenizer, is_train, processor.sample_writer),
    ) as pool:
        for write_data in tqdm(
            pool.imap(sample_writer, data), total=len(data), dynamic_ncols=True, desc="writing samples..."
        ):
            if isinstance(write_data, list):
                for datum in write_data:
                    writer_file.write(json.dumps(datum) + "\n")
                write_cnt += len(write_data)
            else:
                writer_file.write(json.dumps(write_data) + "\n")
                write_cnt += 1
    return write_cnt


class IterableDatasetPad(torch_data.IterableDataset):
    def __init__(
        self,
        dataset: torch_data.IterableDataset,
        batch_size: int = 1,
        num_devices: int = 1,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.num_examples = 0

        chunk_size = self.batch_size * num_devices
        length = len(dataset)
        self.length = length + (chunk_size - length % chunk_size)

    def __len__(self):
        return self.length

    def __iter__(self):
        self.num_examples = 0
        if (
            not hasattr(self.dataset, "set_epoch")
            and hasattr(self.dataset, "generator")
            and isinstance(self.dataset.generator, torch.Generator)
        ):
            self.dataset.generator.manual_seed(self.seed + self.epoch)

        first_batch = None
        current_batch = []
        for element in self.dataset:
            self.num_examples += 1
            current_batch.append(element)
            # Wait to have a full batch before yielding elements.
            if len(current_batch) == self.batch_size:
                for batch in current_batch:
                    yield batch
                    if first_batch is None:
                        first_batch = batch.copy()
                current_batch = []

        # pad the last batch with elements from the beginning.
        while self.num_examples < self.length:
            add_num = self.batch_size - len(current_batch)
            self.num_examples += add_num
            current_batch += [first_batch] * add_num
            for batch in current_batch:
                yield batch
            current_batch = []
