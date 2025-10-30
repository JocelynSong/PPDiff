#!/bin/bash

data_path=data/antibody_antigen_complex_cdrh1.json

local_root=models
esm_model="esm2_t33_650M_UR50D"
pretrained_model_path=${local_root}/PPDiff
output_path=${local_root}/antibody_design_cdrh1
rm -rf ${output_path}
mkdir ${output_path}

python3 fairseq_cli/train.py ${data_path} \
--finetune-from-model ${pretrained_model_path}/checkpoint_best.pt \
--save-dir ${output_path} \
--task protein_protein_complex_design \
--protein-task "antibody_design" \
--dataset-impl "antibody_design" \
--criterion protein_complex_diffusion_loss \
--arch protein_protein_complex_diffusion_model_esm \
--encoder-embed-dim 1280 \
--egnn-mode "rm-node" \
--decoder-layers 3 \
--autoregressive-layer 1 \
--pretrained-esm-model ${esm_model} \
--model-mean-type "C0" \
--sample-time-method "symmetric" \
--beta-schedule "sigmoid" \
--num-diffusion-timesteps 1500 \
--pos-beta-s 2 --beta-start 1e-7 --beta-end 2e-3 \
--r-beta-schedule "cosine" --r-beta-s 0.01 \
--time-emb-dim 0 --time-emb-mode "simple" \
--loss-r-weight 50 \
--max-source-positions 1024 \
--knn 32 \
--dropout 0.3 \
--optimizer adam --adam-betas '(0.9,0.98)' \
--lr 5e-6 --lr-scheduler inverse_sqrt \
--stop-min-lr '1e-12' --warmup-updates 4000 \
--warmup-init-lr '1e-6' \
--clip-norm 0.0001 \
--ddp-backend legacy_ddp \
--log-format 'simple' --log-interval 10 \
--max-tokens 1024 \
--update-freq 1 \
--max-update 300000 \
--max-epoch 30 \
--validate-after-updates 3000 \
--validate-interval-updates 3000 \
--save-interval-updates 3000 \
--valid-subset valid \
--max-sentences-valid 8 \
--validate-interval 1 \
--save-interval 1 \
--keep-interval-updates	10 \
--keep-last-epochs 1 \
--skip-invalid-size-inputs-valid-test