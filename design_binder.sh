#!/bin/bash

data_path=data/binder_design_data.json

local_root=models
output_path=${local_root}/binder_design
generation_path=${local_root}/output/binder_design
mkdir -p ${generation_path}


python3 fairseq_cli/design_binder.py ${data_path} \
--task protein_protein_complex_design \
--protein-task "binder_design" \
--dataset-impl "binder_design" \
--path ${output_path}/checkpoint_best.pt \
--batch-size 1 \
--results-path ${generation_path} \
--skip-invalid-size-inputs-valid-test \
--valid-subset test \
--eval-aa-recovery