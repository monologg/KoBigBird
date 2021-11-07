import argparse
import copy
import json
import logging
import os

import numpy as np
import torch
import transformers
from data import get_data
from evaluate import EVAL_FUNC_MAP
from model import MODEL_CLASS_MAP
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.data_parallel as xla_dp
except ImportError as e:
    logger.error(f"Failed to import XLA. {e}")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def cal_running_avg_loss(loss, running_avg_loss, decay=0.99):
    if running_avg_loss == 0:
        return loss
    running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    return running_avg_loss


def _run_epoch(model, loader, device=None, context=None, **kwargs):
    config = kwargs["config"]
    is_train = kwargs["is_train"]

    avg_loss = 0
    results = []
    batch_num = len(loader)

    if is_train:
        model.train()
        if config.use_tpu:
            optimizer = context.getattr_or(
                "optimizer",
                lambda: model.get_optimizer(),
            )
            scheduler = context.getattr_or(
                "scheduler",
                lambda: model.get_scheduler(batch_num, optimizer),
            )
        else:
            optimizer = kwargs["optimizer"]
            scheduler = kwargs["scheduler"]
    else:
        model.eval()

    is_master = True
    if config.use_tpu:
        is_master = xm.is_master_ordinal()

    pbar = tqdm(enumerate(loader), total=batch_num, disable=not is_master, dynamic_ncols=True)
    for i, inputs in pbar:

        if not config.use_tpu:
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

        outputs = model(inputs)
        loss = outputs.loss.mean()
        avg_loss = cal_running_avg_loss(loss.item(), avg_loss)
        loss /= config.gradient_accumulation_steps

        if is_train:
            loss.backward()
            if i % config.gradient_accumulation_steps == 0 or i == batch_num - 1:

                if config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                if config.use_tpu:
                    xm.optimizer_step(optimizer)
                else:
                    optimizer.step()
                optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()
        else:
            result = (model.module if hasattr(model, "module") else model).eval_step(inputs, outputs)
            results.extend(result)

        if is_master:
            pbar.set_description(
                f"epoch: {kwargs['epoch'] + 1}, {('train' if is_train else 'valid')} loss: {min(100, round(avg_loss, 4))}"
            )

    return {
        "loss": avg_loss,
        "result": results,
    }


def run_epoch(**kwargs):
    model = kwargs.pop("model")
    if kwargs["config"].use_tpu:
        results = model(_run_epoch, **kwargs)
    else:
        results = _run_epoch(model, **kwargs)

    if isinstance(results, list):
        loss = sum([result["loss"] for result in results]) / len(results)
        result = []
        for res in results:
            result.extend(res["result"])
        results = {"loss": loss, "result": result}

    return results


def run(parser):
    # NOTE Remove redundant bigbird logs
    transformers.logging.get_logger("transformers.models.big_bird.modeling_big_bird").setLevel(logging.ERROR)

    args, _ = parser.parse_known_args()

    model = MODEL_CLASS_MAP.get(args.task, None)
    if model is None:
        raise Exception(f"Invalid model task {args.task}")

    parser = model.add_args(parser)
    config = parser.parse_args()

    set_seed(config.seed)

    model = model(config)

    logger.info(f"configuration: {str(config)}")

    if config.use_tpu:
        devices = xm.get_xla_supported_devices()
        model_dp = xla_dp.DataParallel(model, device_ids=devices)
    else:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"{gpu_count} GPU device detected")
            devices = ["cuda:{}".format(i) for i in range(gpu_count)]
            model_dp = torch.nn.DataParallel(model, device_ids=devices)
            model.to(devices[0])
        else:
            devices = ["cpu"]
            model_dp = model

    config.world_size = len(devices)
    if config.do_train:
        train_loader = get_data(config, tokenizer=model.tokenizer)
    valid_loader = get_data(config, tokenizer=model.tokenizer, is_train=False)

    optimizer = None
    scheduler = None
    if not config.use_tpu and config.do_train:
        optimizer = model.get_optimizer()
        scheduler = model.get_scheduler(len(train_loader), optimizer)

    params = {
        "config": config,
        "model": model_dp,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }
    if not config.use_tpu:
        params["device"] = devices[0]

    def do_eval(epoch):
        with torch.no_grad():
            results = run_epoch(loader=valid_loader, epoch=epoch, is_train=False, **params)["result"]
            results = EVAL_FUNC_MAP[config.task](
                config=config,
                model=model,
                loader=valid_loader,
                tokenizer=model.tokenizer,
                results=results,
            )

        logger.info("Eval results.")
        for k, v in results["results"].items():
            logger.info(f"{k} : {v}")

        return results["best_score"]

    if config.do_train:
        best_score = 0
        for epoch in range(config.num_train_epochs):
            run_epoch(loader=train_loader, epoch=epoch, is_train=True, **params)

            score = 0
            if config.do_eval_during_train:
                score = do_eval(epoch)

            if score >= best_score:
                best_score = score
                output_dir = os.path.join(config.output_dir, config.task, config.dataset, f"{epoch}-{best_score}-ckpt")
                copy.deepcopy(
                    model_dp.module
                    if hasattr(model_dp, "module")
                    else model_dp._models[0]
                    if hasattr(model_dp, "_models")
                    else model_dp
                ).cpu().save_pretrained(output_dir)
                with open(os.path.join(output_dir, "finetune_config.json"), "w") as save_config:
                    json.dump(vars(config), save_config, sort_keys=True, indent=4)
                logger.info(f"Checkpoint {output_dir} saved.")

    if config.do_eval:
        do_eval(-1)


def main(parser):
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    output_dir = os.path.join(args.output_dir, args.task, args.dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    run(parser)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default=None, type=str, required=True)
    parser.add_argument("--dataset", default=None, type=str, required=True)
    parser.add_argument("--cache_dir", default="cache", type=str)
    parser.add_argument("--output_dir", default="output", type=str)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval_during_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--use_tpu", action="store_true")
    parser.add_argument("--threads", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)

    main(parser)
