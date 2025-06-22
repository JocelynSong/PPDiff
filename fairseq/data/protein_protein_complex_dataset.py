# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
from fairseq.data import FairseqDataset, data_utils


logger = logging.getLogger(__name__)


def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, pad_idx=1, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad=left_pad,
            move_eos_to_beginning=move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    def collate_coor(
            value,
            pad_idx,
            pad_to_length=None,
    ):
        """Convert a list of 1d tensors into a padded 2d tensor."""
        values = [s[value] for s in samples]
        size = max(v.size(0) for v in values)
        size = size if pad_to_length is None else max(size, pad_to_length)

        batch_size = len(values)
        coors = values[0].new(batch_size, size, 3).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, coors[i][: len(v)])
        return coors

    def collate_mask(
            value,
            pad_idx,
            pad_to_length=None,
    ):
        values = [s[value] for s in samples]
        size = max(v.size(0) for v in values)
        size = size if pad_to_length is None else max(size, pad_to_length)

        batch_size = len(values)
        masks = values[0].new(batch_size, size).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, masks[i][: len(v)])
        return masks
    
    def collate_length(value):
        values = [s[value] for s in samples]
        lengths = torch.FloatTensor(values)
        return lengths

    id = torch.LongTensor([s["id"] for s in samples])
    seqs = merge(
        "seq",
        pad_idx=pad_idx,  # 1
        left_pad=left_pad,
    )
    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s["seq"].ne(pad_idx).long().sum() for s in samples]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    seqs = seqs.index_select(0, sort_order)

    coors = collate_coor(
        "coor",
        0,
        pad_to_length=pad_to_length["target"]
        if pad_to_length is not None
        else None,
    )
    coors = coors.index_select(0, sort_order)
    
    atoms =  merge(
        "atom",
        pad_idx=4,
        left_pad=left_pad
    )
    atoms = atoms.index_select(0, sort_order)

    target = collate_mask("target", 1)
    target = target.index_select(0, sort_order)
    center_batch = collate_mask("center", 0).index_select(0, sort_order)
    # length_batch = collate_length("length").index_select(0, sort_order)

    ntokens = ((target==0).int().sum().item())

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "seqs": seqs,
        "seq_lengths": src_lengths,
        "coors": coors,
        "atoms": atoms,
        "target": target,
        "center": center_batch
    }
    return batch


class ProteinComplexDataset(FairseqDataset):
    """
    A protein-protein complex dataset of torch.utils.data.Datasets.
    """

    def __init__(
        self,
        protein_complex,
        protein_complex_sizes,
        src_dict=None,
        left_pad=False,
        shuffle=True,
        remove_eos_from_source=False,
        append_eos_to_target=False,
        constraints=None,
        append_bos=False,
        eos=None,
        num_buckets=0,
        pad_to_multiple=1,
    ):
        self.protein_complex = protein_complex
        self.sizes = np.array(protein_complex_sizes)
        self.src_dict = src_dict
        self.left_pad = left_pad 
        self.shuffle = shuffle
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.constraints = constraints
        self.append_bos = append_bos
        self.eos = eos if eos is not None else src_dict.eos()
        self.buckets = None
        self.pad_to_multiple = pad_to_multiple

    def get_batch_shapes(self):
        return self.buckets

    def __getitem__(self, index):
        seq_item, atom_item, coor_item, center_item, target_item = self.protein_complex[index]

        example = {
            "id": index,
            "seq": seq_item,
            "coor": coor_item,
            "atom": atom_item,
            "target": target_item,
            "center": center_item
            }

        return example

    def __len__(self):
        return len(self.protein_complex)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad=self.left_pad,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )

        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.sizes[index]

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        sizes = self.sizes[indices]
        return sizes

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.sizes[index], 0
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            # sort by target length, then source length
            return indices[np.argsort(self.sizes[indices], kind="mergesort")]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind="mergesort")
            ]

    @property
    def supports_prefetch(self):
        return getattr(self.protein_complex, "supports_prefetch", False)

    def prefetch(self, indices):
        self.protein_complex.prefetch(indices)

    def filter_indices_by_size(self, indices, max_sizes):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        return data_utils.filter_paired_dataset_indices_by_size(
            self.sizes, None, indices, max_sizes,
        )
