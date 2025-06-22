# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
import random
import shutil
import struct
from functools import lru_cache
import io

import numpy as np
import torch
from fairseq.dataclass.constants import DATASET_IMPL_CHOICES
from fairseq.file_io import PathManager

from . import FairseqDataset

from typing import Union


def get_available_dataset_impl():
    return list(map(str, DATASET_IMPL_CHOICES))



def make_dataset(path, impl, fix_lua_indexing=False, dictionary=None, source=True, sizes=None, motif_list=None,
                 epoch=1, train=True, split="train", protein=None):
    if impl == "protein_complex" and ProteinComplexDataset.exists(path):
        return ProteinComplexDataset(path, dictionary, split=split, protein=protein)
    elif impl == "protein_complex_pretrain" and ProteinComplexPretrainDataset.exists(path):
        return ProteinComplexPretrainDataset(path, dictionary, split=split, protein=protein)
    elif impl == "binder_design" and TargetProteinBinderDataset.exists(path):
        return TargetProteinBinderDataset(path, dictionary, split=split, protein=protein)
    elif impl == "antibody_design" and AntigenAntibodyDataset.exists(path):
        return AntigenAntibodyDataset(path, dictionary, split=split, protein=protein)
    return None


def read_longs(f, n):
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)
    return a


def write_longs(f, a):
    f.write(np.array(a, dtype=np.int64))
    

class ProteinComplexDataset(FairseqDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path, dictionary, append_eos=True, reverse_order=False, split="train", protein=None):
        self.seqs = []
        self.atoms = []
        self.coors = []
        self.target = []   # 1 for target protein, and 0 for binder protein
        self.sizes = []
        self.centers = []
        self.lengths = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.atom_dict = {"N": 0, "CA": 1, "C": 2, "O": 3}
        self.read_data(path, dictionary, split, protein)
        self.size = len(self.seqs)

    def read_data(self, path, dictionary, split, protein):
        f = open(path)
        data = json.load(f)

        lines = data[protein][split]["seqs"]
        for line in lines:
            line = line.strip()
            tokens = dictionary.encode_line(
                    line,
                    add_if_not_exist=False,
                    prepend_bos=True,
                    append_eos=True,
                    reverse_order=False,
                ).long()
            self.seqs.append(tokens)
            self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)

        atoms = data[protein][split]["atoms"]
        for line in atoms:
            line.insert(0, "CA")
            line.append("CA")
            tokens = torch.IntTensor([self.atom_dict[atom] for atom in line]).long()
            self.atoms.append(tokens)  

        for coor, target in zip(data[protein][split]["coors"], data[protein][split]["target"]):
            center = np.sum(np.array(coor) * np.reshape(target, (-1, 1)), axis=0) / np.sum(target)   # [batch]
            self.centers.append(torch.tensor(center))
           
            array = []
            for coord, label in zip(coor, target):
                if label == 1:
                    array.append(coord)
            max_ = np.max(np.array(array), axis=0)
            min_ = np.min(np.array(array), axis=0) 
            length = np.max(max_ - min_)
            
            target.insert(0, 1)
            target.append(1)
            self.target.append(torch.tensor(target))   # 1 for target and 0 for binder: [N, L]
            
            coor.insert(0, center)
            coor.append(center)
            coor = (coor - np.reshape(center, (1, 3))) / length 
            # coor = (coor - np.reshape(center, (1, 3)))
            self.coors.append(torch.tensor(coor))  # [center, target, binder, center]
            self.lengths.append(length)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return (self.seqs[i], self.atoms[i], self.coors[i], self.centers[i], self.target[i])

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)
    

class ProteinComplexPretrainDataset(FairseqDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path, dictionary, append_eos=True, reverse_order=False, split="train", protein=None):
        self.seqs = []
        self.atoms = []
        self.coors = []
        self.target = []   # 1 for target protein, and 0 for binder protein
        self.sizes = []
        self.centers = []
        self.lengths = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.atom_dict = {"N": 0, "CA": 1, "C": 2, "O": 3}
        self.read_data(path, dictionary, split, protein)
        self.size = len(self.seqs)

    def read_data(self, path, dictionary, split, protein):
        f = open(path)
        data = json.load(f)

        lines = data[protein][split]["seqs"]
        for line in lines:
            line = line.strip()
            if len(line) < 10:
                continue
            tokens = dictionary.encode_line(
                    line,
                    add_if_not_exist=False,
                    prepend_bos=True,
                    append_eos=True,
                    reverse_order=False,
                ).long()
            self.seqs.append(tokens)
            self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)

        atoms = data[protein][split]["atoms"]
        for line in atoms:
            if len(line) < 10:
                continue

            line.insert(0, "CA")
            line.append("CA")
            tokens = torch.IntTensor([self.atom_dict[atom] for atom in line]).long()
            self.atoms.append(tokens)  

        for coor in data[protein][split]["coors"]:
            length = len(coor)

            if length < 10:
                continue

            # half 
            pos = random.choice(range(int(length/2), length-1))
            
            target = np.ones((length), dtype=np.int64)
            target[pos: ] = 0
            center = np.sum(np.array(coor) * np.reshape(target, (-1, 1)), axis=0) / np.sum(target)   # [batch]
            self.centers.append(torch.tensor(center))
           
            array = []
            for coord, label in zip(coor, target):
                if label == 1:
                    array.append(coord)
            max_ = np.max(np.array(array), axis=0)
            min_ = np.min(np.array(array), axis=0) 
            length = np.max(max_ - min_)
            
            target = np.insert(target, 0, 1)
            target = np.insert(target, len(target), 1)
            self.target.append(torch.tensor(target))   # 1 for target and 0 for binder: [N, L]
            
            coor.insert(0, center)
            coor.append(center)
            coor = (coor - np.reshape(center, (1, 3))) / length 
            # coor = (coor - np.reshape(center, (1, 3)))
            self.coors.append(torch.tensor(coor))  # [center, target, binder, center]
            self.lengths.append(length)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return (self.seqs[i], self.atoms[i], self.coors[i], self.centers[i], self.target[i])

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)
    

class TargetProteinBinderDataset(FairseqDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path, dictionary, append_eos=True, reverse_order=False, split="train", protein=None):
        self.seqs = []
        self.atoms = []
        self.coors = []
        self.target = []   # 1 for target protein, and 0 for binder protein
        self.sizes = []
        self.centers = []
        self.lengths = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.atom_dict = {"N": 0, "CA": 1, "C": 2, "O": 3}
        self.read_data(path, dictionary, split, protein)
        self.size = len(self.seqs)

    def read_data(self, path, dictionary, split, protein):
        f = open(path)
        data = json.load(f)

        lines = data[protein][split]["seqs"]
        for line in lines:
            line = line.strip()
            tokens = dictionary.encode_line(
                    line,
                    add_if_not_exist=False,
                    prepend_bos=True,
                    append_eos=True,
                    reverse_order=False,
                ).long()
            self.seqs.append(tokens)
            self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)

        atoms = data[protein][split]["atoms"]
        for line in atoms:
            line.insert(0, "CA")
            line.append("CA")
            tokens = torch.IntTensor([self.atom_dict[atom] for atom in line]).long()
            self.atoms.append(tokens)  

        for coor, target in zip(data[protein][split]["coors"], data[protein][split]["target"]):
            coor = np.array(coor)[:, 1, :]
            center = np.sum(coor * np.reshape(target, (-1, 1)), axis=0) / np.sum(target)   # [batch]
            self.centers.append(torch.tensor(center))
           
            array = []
            for coord, label in zip(coor, target):
                if label == 1:
                    array.append(coord)
            max_ = np.max(np.array(array), axis=0)
            min_ = np.min(np.array(array), axis=0) 
            length = np.max(max_ - min_)
            
            target.insert(0, 1)
            target.append(1)
            self.target.append(torch.tensor(target))   # 1 for target and 0 for binder: [N, L]
            
            coor = np.concatenate((np.reshape(center, (1, 3)), coor, np.reshape(center, (1, 3))), axis=0) 
            coor = (coor - np.reshape(center, (1, 3))) / length 
            # coor = (coor - np.reshape(center, (1, 3)))
            self.coors.append(torch.tensor(coor))  # [center, target, binder, center]
            self.lengths.append(length)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return (self.seqs[i], self.atoms[i], self.coors[i], self.centers[i], self.target[i])

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)


class AntigenAntibodyDataset(FairseqDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path, dictionary, append_eos=True, reverse_order=False, split="train", protein=None):
        self.seqs = []
        self.atoms = []
        self.coors = []
        self.target = []   # 1 for target protein, and 0 for binder protein
        self.sizes = []
        self.centers = []
        self.lengths = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.atom_dict = {"N": 0, "CA": 1, "C": 2, "O": 3}
        self.read_data(path, dictionary, split, protein)
        self.size = len(self.seqs)

    def read_data(self, path, dictionary, split, protein):
        f = open(path)
        data = json.load(f)

        lines = data[protein][split]["seqs"]
        for line in lines:
            line = line.strip()
            tokens = dictionary.encode_line(
                    line,
                    add_if_not_exist=False,
                    prepend_bos=True,
                    append_eos=True,
                    reverse_order=False,
                ).long()
            self.seqs.append(tokens)
            self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)

        atoms = data[protein][split]["atoms"]
        for line in atoms:
            line.insert(0, "CA")
            line.append("CA")
            tokens = torch.IntTensor([self.atom_dict[atom] for atom in line]).long()
            self.atoms.append(tokens)  

        for coor, target in zip(data[protein][split]["coors"], data[protein][split]["target"]):
            coor = np.array(coor)
            center = np.sum(coor * np.reshape(target, (-1, 1)), axis=0) / np.sum(target)   # [batch]
            self.centers.append(torch.tensor(center))
           
            array = []
            for coord, label in zip(coor, target):
                if label == 1:
                    array.append(coord)
            max_ = np.max(np.array(array), axis=0)
            min_ = np.min(np.array(array), axis=0) 
            length = np.max(max_ - min_)
            
            target.insert(0, 1)
            target.append(1)
            self.target.append(torch.IntTensor(target))   # 1 for target and 0 for binder: [N, L]
            
            coor = np.concatenate((np.reshape(center, (1, 3)), coor, np.reshape(center, (1, 3))), axis=0) 
            coor = (coor - np.reshape(center, (1, 3))) / length 
            self.coors.append(torch.tensor(coor))  # [center, target, binder, center]
            self.lengths.append(length)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return (self.seqs[i], self.atoms[i], self.coors[i], self.centers[i], self.target[i], self.lengths[i])

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)


