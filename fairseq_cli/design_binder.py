#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import time
from argparse import Namespace
from itertools import chain
import numpy as np

import torch
from fairseq import checkpoint_utils, distributed_utils, options, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import metrics, progress_bar
from fairseq.utils import reset_logging
from omegaconf import DictConfig
import generate_pdb_file


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.validate")


def main(cfg: DictConfig, override_args=None):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    reset_logging()

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)

    if cfg.distributed_training.distributed_world_size > 1:
        data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
        data_parallel_rank = distributed_utils.get_data_parallel_rank()
    else:
        data_parallel_world_size = 1
        data_parallel_rank = 0

    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    else:
        overrides = None

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [cfg.common_eval.path],
        arg_overrides=overrides,
        suffix=cfg.checkpoint.checkpoint_suffix,
    )
    model = models[0]

    # Move models to GPU
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Print args
    logger.info(saved_cfg)

    # Build criterion
    criterion = task.build_criterion(saved_cfg.criterion)
    criterion.eval()
    start = time.time()
    for sampling_round in range(1, 2):
        print("sampling {}".format(sampling_round))
        try:
            task.load_dataset("test", combine=False, epoch=1, task_cfg=saved_cfg.task)
            dataset = task.dataset("test")
        except KeyError:
            raise Exception("Cannot find dataset: " + "test")

        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=cfg.dataset.max_tokens,
            max_sentences=cfg.dataset.batch_size,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                *[m.max_positions() for m in models],
            ),
            ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
            seed=cfg.common.seed,
            num_shards=data_parallel_world_size,
            shard_id=data_parallel_rank,
            num_workers=cfg.dataset.num_workers,
            data_buffer_size=cfg.dataset.data_buffer_size,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            prefix=f"valid on test subset",
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        )

        log_outputs = []

        os.system("mkdir -p {}{}".format(cfg.common_eval.results_path, sampling_round))
        fw_target = open(os.path.join("{}{}".format(cfg.common_eval.results_path, sampling_round), "target.txt"), "w")
        fw_binder = open(os.path.join("{}{}".format(cfg.common_eval.results_path, sampling_round), "binder.true.txt"), "w")
        fw_generation = open(os.path.join("{}{}".format(cfg.common_eval.results_path, sampling_round), "binder.gen.txt"), "w")

        for i, sample in enumerate(progress):

            sample = utils.move_to_cuda(sample) if use_cuda else sample
            _loss, _sample_size, log_output, _strings, _srcs, coords, target = task.valid_step(sample, model, criterion)
            progress.log(log_output, step=i)
            log_outputs.append(log_output)

            for idx, tgt in enumerate(target):
                assert len(_srcs[idx]) == len(tgt) -2
                index = list(tgt.cpu()).index(0)-1

                target_protein, binder_protein, gen_protein = [], [], []

                for ind in range(index):
                    target_protein.append(_srcs[idx][ind])
                
                for ind in range(index, len(_srcs[idx])):
                    binder_protein.append(_srcs[idx][ind])
                
                for ind in range(index, len(_strings[idx])):
                    gen_protein.append(_strings[idx][ind])

                fw_target.write("".join(target_protein) + "\n")
                fw_binder.write("".join(binder_protein) + "\n")
                fw_generation.write("".join(gen_protein) + "\n")
            fw_target.flush()
            fw_binder.flush()
            fw_generation.flush()

        fw_generation.close()
        fw_binder.close()
        fw_target.close()

        if data_parallel_world_size > 1:
            log_outputs = distributed_utils.all_gather_list(
                log_outputs,
                max_size=cfg.common.all_gather_list_size,
                group=distributed_utils.get_data_parallel_group(),
            )
            log_outputs = list(chain.from_iterable(log_outputs))

        with metrics.aggregate() as agg:
            task.reduce_metrics(log_outputs, criterion)
            log_output = agg.get_smoothed_values()

        print("inference time: {}".format(time.time()-start))
        progress.print(log_output, tag="test", step=i)


def cli_main():
    parser = options.get_validation_parser()
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_validation_parser()
    override_args = options.parse_args_and_arch(
        override_parser, suppress_defaults=True
    )

    main(convert_namespace_to_omegaconf(args), override_args)


if __name__ == "__main__":
    cli_main()