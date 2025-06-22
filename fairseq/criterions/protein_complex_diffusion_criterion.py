# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import math
from omegaconf import II
import numpy as np

import torch
import torch.nn.functional as F
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class ProteinComplexDiffusionCriterionConfig(FairseqDataclass):
    tpu: bool = II("common.tpu")


@register_criterion("protein_complex_diffusion_loss", dataclass=ProteinComplexDiffusionCriterionConfig)
class ProteinComplexDiffusionDesignLoss(FairseqCriterion):

    def __init__(self, cfg: ProteinComplexDiffusionCriterionConfig, task):
        super().__init__(task)
        self.tpu = cfg.tpu

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        """
        seqs = sample["seqs"]
        coors = sample["coors"]    # [B, L, 3]
        target_mask = sample["target"]  # [B, L]
        sample_size = sample["ntokens"]
        n_sentence = sample["nsentences"]
        
        outputs = model.get_diffusion_loss(seqs, coors, target_mask)
        loss, loss_pos, loss_seq = outputs["loss"], outputs["loss_pos"], outputs["loss_residue"]
        
        print(loss_seq.item())
        print(loss_pos.item())
        print(loss.item())
       
        logging_output = {
            "loss": loss.data,
            "loss_seq": loss_seq.data,
            "loss_coor": loss_pos.data,
            "ntokens": sample_size,
            "nsentences": n_sentence}
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss = sum([log["loss"].cpu() * log["nsentences"] for log in logging_outputs])
        loss_seq = sum([log.get("loss_seq").cpu() * log["nsentences"] for log in logging_outputs])
        loss_coor = sum([log.get("loss_coor").cpu() * log["nsentences"] for log in logging_outputs])
        sample_size = sum(log.get("nsentences", 0) for log in logging_outputs)
        tokens = sum(log.get("ntokens", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss / sample_size, round=3
        )
        metrics.log_scalar(
            "sequence loss", loss_seq / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "coordinate loss", loss_coor / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "sample size", tokens)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
