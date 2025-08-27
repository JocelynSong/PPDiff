# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import itertools
import json
import logging
import os
from typing import Optional
from argparse import Namespace
from omegaconf import II
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.data import (
    ProteinComplexDataset,
    data_utils,
)
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from fairseq.models.esm_modules import Alphabet


device = torch.device("cuda")


logger = logging.getLogger(__name__)


def load_protein_dataset(
    data_path,
    split,
    protein,
    dictionary,
    dataset_impl,
    left_pad,
    max_source_positions,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    epoch=1,
):
    protein_complex_dataset = data_utils.load_indexed_dataset(
        data_path, dictionary, dataset_impl, split=split, protein=protein
    )

    
    
    logger.info( "Loading {} {} {} examples".format(
                data_path, split, len(protein_complex_dataset)))

    return ProteinComplexDataset(
            protein_complex_dataset,
            protein_complex_dataset.sizes,
            dictionary,
            left_pad=left_pad,
            eos=dictionary.eos(),
            num_buckets=num_buckets,
            shuffle=shuffle,
            pad_to_multiple=pad_to_multiple,
        )


@dataclass
class ProteinProteinComplexDesignConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "colon separated path to data directories list, will be iterated upon during epochs "
            "in round-robin manner; however, valid and test data are always in the first directory "
            "to avoid the need for repeating them in all directories"
        },
    )
    protein_task: str = field(
        default="PDB",
        metadata={"help": "protein task name"}
    )
    left_pad: bool = field(
        default=False, metadata={"help": "pad the sequencd on the left"}
    )
    max_source_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the sequence"}
    )
    truncate_source: bool = field(
        default=False, metadata={"help": "truncate source to max-source-positions"}
    )
    train_subset: str = II("dataset.train_subset")
    dataset_impl: str= field(
        default="protein_complex", metadata={"help": "data format of source data"}
    )
    eval_aa_recovery: bool = field(
        default=False, metadata={
            "help": "evaluate amino acid recovery or not"
        }
    )


@register_task("protein_protein_complex_design", dataclass=ProteinProteinComplexDesignConfig)
class ProteinProteinComplexDesignTask(FairseqTask):
    """
    protein design task
    """

    cfg: ProteinProteinComplexDesignConfig

    def __init__(self, cfg: ProteinProteinComplexDesignConfig, dictionary):
        super().__init__(cfg)
        self.dictionary = dictionary
        self.mask_idx = self.dictionary.mask_idx
        self.ss_cands = self.get_secondary_candidates()

    @classmethod
    def setup_task(cls, cfg: ProteinProteinComplexDesignConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).
        the dictionary is composed of amino acids
        """

        # load dictionaries
        alphabet = Alphabet.from_architecture("ESM-1b")

        return cls(cfg, alphabet)
    
    def get_secondary_candidates(self, ):
        lines = open("data/secondary_structure.txt").readlines()
        cands = [torch.tensor([self.dictionary.tok_to_idx[char] for char in line.strip()]).to("cuda") for line in lines]
        return cands

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        data_path = self.cfg.data

        protein_task = self.cfg.protein_task

        self.datasets[split] = load_protein_dataset(
            data_path,
            split,
            protein_task,
            self.dictionary,
            dataset_impl=self.cfg.dataset_impl,
            left_pad=self.cfg.left_pad,
            max_source_positions=self.cfg.max_source_positions,
            shuffle=(split != "test"),
            epoch=epoch
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)

        return model

    def valid_step(self, sample, model, criterion):
        # interleaving model
        # loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

        # diffusion model
        sum_loss, sum_loss_pos, sum_loss_res, sum_n = 0, 0, 0, 0
        model.eval()
        with torch.no_grad():
            seqs = sample["seqs"]
            coors = sample["coors"]    # [B, L, 3]
            target_mask = sample["target"]  # [B, L]
            sample_size = sample["ntokens"]
            batch_size = seqs.size(0)
            length = seqs.size(1)

            for t in np.linspace(0, model.num_diffusion_timesteps - 1, 10).astype(int):
                # time_step = torch.full((batch_size, length), t).to("cuda")
                time_step = torch.tensor([t] * batch_size).to("cuda")
                outputs = model.get_diffusion_loss(seqs, coors, target_mask, time_step=time_step)

                loss, loss_pos, loss_seq = outputs["loss"], outputs["loss_pos"], outputs["loss_residue"]
                    
                sum_loss += loss.data 
                sum_loss_pos += loss_pos.data 
                sum_loss_res += loss_seq.data 
                sum_n += 1
            
            loss = sum_loss / sum_n
            avg_loss_pos = sum_loss_pos / sum_n
            avg_loss_res = sum_loss_res / sum_n
            logging_output = {
            "loss": loss,
            "loss_seq": avg_loss_res,
            "loss_coor": avg_loss_pos,
            "ntokens": sample_size,
            "nsentences": batch_size}

        if self.cfg.eval_aa_recovery:
            with torch.no_grad():
                # interleaving network
                # seqs = sample["seqs"]
                # seq_lengths = sample["seq_lengths"]
                # coors = sample["coors"]    # [B, L, 3]
                # target = sample["target"]  # [B, L]
                # sample_size = sample["ntokens"]
                # seq_probs, coords = model(seqs, seq_lengths, coors, target)
                
                # # greedy
                # # seq_probs[:, 1: -1, : 4] = -math.inf
                # # seq_probs[:, :, 24:] = -math.inf
                # # indexes = torch.argmax(seq_probs, dim=-1)   # [batch, length]

                # # sampling
                # seq_probs[:, 1: -1, : 4] = 0
                # seq_probs[:, :, 24:] = 0
                # batch_size = seq_probs.size(0)
                # sorted_logits, sorted_indices = torch.sort(seq_probs, descending=True)
                # cumulative_probs = torch.cumsum(sorted_logits, dim=-1)  # [B,L, vocab]
                # sorted_indices_to_remove = cumulative_probs > 0.4
                # sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                # sorted_indices_to_remove[..., 0] = 0
                # indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                # seq_probs[indices_to_remove] = 0
                # indexes = torch.multinomial(seq_probs.view(-1, seq_probs.size(-1)), 1).reshape(batch_size, -1)

                # indexes = target * seqs + (target == 0).int() * indexes
                # srcs = [model.encoder.alphabet.string(seqs[i]) for i in range(seqs.size(0))]
                # strings = [model.encoder.alphabet.string(indexes[i]) for i in range(len(indexes))]
                # return loss, sample_size, logging_output, strings, srcs, coords, target

                # diffusion model
                seqs = sample["seqs"]
                print(seqs.size())
                coors = sample["coors"]    # [B, L, 3]
                target_mask = sample["target"]  # [B, L]
                sample_size = sample["ntokens"]
                n_sentence = sample["nsentences"]
                batch_size, n_nodes = seqs.size()[0], seqs.size()[1]

                outputs = model.sample_diffusion(seqs, coors, target_mask, ss_initial=self.ss_cands)
                indexes = outputs["residue"]
                print(indexes.size())

                srcs = [model.encoder.alphabet.string(seqs[i]) for i in range(seqs.size(0))]
                strings = [model.encoder.alphabet.string(indexes[i]) for i in range(len(indexes))]
                print(srcs[0])
                print(strings[0])
                return loss, sample_size, logging_output, strings, srcs, outputs["pos"], target_mask
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.cfg.max_source_positions, self.cfg.max_source_positions)
    
    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary


