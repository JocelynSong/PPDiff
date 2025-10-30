#!/bin/bash

data_path=data/antibody_antigen_complex_cdrh1.json

local_root=models
output_path=${local_root}/antibody_design_cdrh1
generation_path=${local_root}/output/antibody_design_cdrh1
mkdir -p ${output_path}
mkdir -p ${generation_path}

python3 fairseq_cli/design_antibody.py ${data_path} \
--task protein_protein_complex_design \
--protein-task "antibody_design" \
--dataset-impl "antibody_design" \
--path ${output_path}/checkpoint_best.pt \
--batch-size 1 \
--results-path ${generation_path} \
--skip-invalid-size-inputs-valid-test \
--valid-subset test \
--eval-aa-recovery