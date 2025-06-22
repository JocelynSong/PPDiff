#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

data_path=data/protein_protein_complex_data.json

local_root=models
output_path=${local_root}/PPDiff
generation_path=${local_root}/output/PPDiff
mkdir -p ${generation_path}

python3 fairseq_cli/validate.py ${data_path} \
--task protein_protein_complex_design \
--protein-task "PDB" \
--dataset-impl "protein_complex" \
--path ${output_path}/checkpoint_best.pt \
--batch-size 1 \
--results-path ${generation_path} \
--skip-invalid-size-inputs-valid-test \
--valid-subset test \
--eval-aa-recovery