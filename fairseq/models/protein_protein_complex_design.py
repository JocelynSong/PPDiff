# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os, re
from typing import Any, Dict, Optional, List
from pathlib import Path
import urllib
import warnings
import numpy as np
import random 
from tqdm import tqdm
from math import sqrt, pi

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
# from torch_cluster import knn_graph
from torch_scatter import scatter_sum, scatter_mean

from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerModel,
    base_architecture as transformer_base_architecture,
)
from fairseq.models.esm import ESM2
from fairseq.models.esm_modules import Alphabet
from fairseq.models.egnn import EGNN
from fairseq.modules import transformer_layer, LayerNorm
from fairseq import utils


device = torch.device("cuda")
DEFAULT_MAX_SOURCE_POSITIONS = 1024


def load_hub_workaround(url):
    try:
        data = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu")
    except RuntimeError:
        # Pytorch version issue - see https://github.com/pytorch/pytorch/issues/43106
        fn = Path(url).name
        data = torch.load(
            f"{torch.hub.get_dir()}/checkpoints/{fn}",
            map_location="cpu",
        )
    except urllib.error.HTTPError as e:
        raise Exception(f"Could not load {url}, check if you specified a correct model name?")
    return data


def _has_regression_weights(model_name):
    """Return whether we expect / require regression weights;
    Right now that is all models except ESM-1v and ESM-IF"""
    return not ("esm1v" in model_name or "esm_if" in model_name)


def load_regression_hub(model_name):
    url = f"https://dl.fbaipublicfiles.com/fair-esm/regression/{model_name}-contact-regression.pt"
    regression_data = load_hub_workaround(url)
    return regression_data


def _load_model_and_alphabet_core_v2(model_data):
    def upgrade_state_dict(state_dict):
        """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
        prefixes = ["encoder.sentence_encoder.", "encoder."]
        pattern = re.compile("^" + "|".join(prefixes))
        state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
        return state_dict

    cfg = model_data["cfg"]["model"]
    state_dict = model_data["model"]
    state_dict = upgrade_state_dict(state_dict)
    alphabet = Alphabet.from_architecture("ESM-1b")
    model = ESM2(
        num_layers=cfg.encoder_layers,
        embed_dim=cfg.encoder_embed_dim,
        attention_heads=cfg.encoder_attention_heads,
        alphabet=alphabet,
        token_dropout=cfg.token_dropout,
    )
    return model, alphabet, state_dict


def load_from_pretrained_models(pretrained_model_name):
    def _download_model_and_regression_data(model_name):
        url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
        model_data = load_hub_workaround(url)
        if _has_regression_weights(model_name):
            regression_data = load_regression_hub(model_name)
        else:
            regression_data = None
        return model_data, regression_data

    model_data, regression_data = _download_model_and_regression_data(pretrained_model_name)

    if regression_data is not None:
        model_data["model"].update(regression_data["model"])

    # if pretrained_model_name.startswith("esm2"):
    #     model, alphabet, model_state = _load_model_and_alphabet_core_v2(model_data)
    # else:
    #     model, alphabet, model_state = _load_model_and_alphabet_core_v1(model_data)
    model, alphabet, model_state = _load_model_and_alphabet_core_v2(model_data)

    expected_keys = set(model.state_dict().keys())
    found_keys = set(model_state.keys())

    if regression_data is None:
        expected_missing = {"contact_head.regression.weight", "contact_head.regression.bias"}
        error_msgs = []
        missing = (expected_keys - found_keys) - expected_missing
        if missing:
            error_msgs.append(f"Missing key(s) in state_dict: {missing}.")
        unexpected = found_keys - expected_keys
        if unexpected:
            error_msgs.append(f"Unexpected key(s) in state_dict: {unexpected}.")

        if error_msgs:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        if expected_missing - found_keys:
            warnings.warn(
                "Regression weights not found, predicting contacts will not produce correct results."
            )
    model.load_state_dict(model_state, strict=regression_data is not None)
    return model, alphabet


def get_edges(n_nodes, k, indices):
    rows, cols = [], []

    for i in range(n_nodes):
        for j in range(k):
            rows.append(i)
            cols.append(indices[i][j+1])

    edges = [rows, cols]   # L * 30
    return edges


def get_edges_batch(n_nodes, batch_size, coords, k=30):
    rows, cols = [], []
    # batch = torch.tensor(range(batch_size)).reshape(-1, 1).expand(-1, n_nodes).reshape(-1).to(device)
    # edges = knn_graph(coords, k=k, batch=batch, loop=False)
    # edges = edges[[1, 0]]

    coords = torch.where( torch.isinf(coords), torch.full_like(coords, 0), coords)
    coords = torch.where( torch.isnan(coords), torch.full_like(coords, 0), coords)

    for i in range(batch_size):
        # k = min(k, len(coords[i]))
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords[i])
        distances, indices = nbrs.kneighbors(coords[i])  # [N, 30]
        edges = get_edges(n_nodes, k, indices)  # [[N*N], [N*N]]
        edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
        rows.append(edges[0] + n_nodes * i)  # every sample in batch has its own graph
        cols.append(edges[1] + n_nodes * i)
    edges = [torch.cat(rows).to(device), torch.cat(cols).to(device)]  # B * L * 30
    return edges


def get_knn_distance(batch_size, coords, k=30):
    rows, cols = [], []

    coords = torch.where( torch.isinf(coords), torch.full_like(coords, 0), coords)
    coords = torch.where( torch.isnan(coords), torch.full_like(coords, 0), coords)

    dist = []
    for i in range(batch_size):
        # k = min(k, len(coords[i]))
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords[i])
        distances, indices = nbrs.kneighbors(coords[i])  # [N, 30]
        dist.append(np.average(distances, axis=1)) # [batch, node]
    return dist


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_time_steps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)
    
    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_time_steps,
                dtype=np.float64
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_time_steps, dtype= np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_time_steps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(
            num_diffusion_time_steps, 1, num_diffusion_time_steps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_time_steps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_time_steps, )
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x/steps) + s)/(1+s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1: ] / alphas_cumprod[: -1])
    
    alphas = np.clip(alphas, a_min=0.001, a_max=1.0)
    alphas = np.sqrt(alphas)
    return alphas


def get_distance(pos, edge_index):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)


def to_torch_const(x):
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x 


def index_to_log_one_hot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'

    x_onehot = F.one_hot(x, num_classes)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def cateforical_kl(log_prob1, log_prob2):
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=-1)
    return kl


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=-1)


def normal_kl(mean1, logvar1, mean2, logvar2):
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mean1 - mean2) ** 2 * torch.exp(-logvar2))
    return kl.sum(-1)


def log_normal(values, means, log_scales):
    var = torch.exp(log_scales * 2)
    log_prob = -((values - means) ** 2) / (2*var) - log_scales - np.log(np.sqrt(2 * np.pi))
    return log_prob.sum(-1)


def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample_index = ((gumbel_noise + logits)[:, :, 4: 24]).argmax(dim=-1) + 4  # [B, L]
    return sample_index


def log_1_min_a(a):
    return np.log(1-np.exp(a) + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a-maximum) + torch.exp(b-maximum))


### Time embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim 

    def forward(self, x):
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.log(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb 


def extract(coef, t):
    out = coef[t]
    return out.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]


# %% categorical diffusion related
def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    # permute_order = (0, -1) + tuple(range(1, len(x.size())))
    # x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


@register_model("protein_protein_complex_diffusion_model")
class ProteinProteinComplexdiffusionModel(TransformerModel):
    @staticmethod
    def add_args(parser):
        """
        Add model-specific arguments to the parser.
        """
        TransformerModel.add_args(parser)
        parser.add_argument(
            "--pretrained-esm-model",
            type=str,
            metavar="ESM",
            help="Pretrained protein language model",
        )
        parser.add_argument(
            "--egnn-mode",
            type=str,
            default="full",
            help="version of EGNN architectures, and values could be full, rm-node, rm-edge, rm-all",
        )
        parser.add_argument(
            "--knn",
            type=int,
            default=16,
            help="number of k nearest neighbors",
        )
        parser.add_argument(
            "--model-mean-type",
            type=str, default="C0",
            help="The learning type for model learning, noise for learning noise parameter and C0 for learning mean coordinate"
        )
        parser.add_argument(
            "--sample-time-method",
            type=str,
            default="symmetric"  ## [importance or symmetric] 
        )
        parser.add_argument(
            "--beta-schedule",
            type=str,
            default="sigmoid",
            help="beta schedule for residue alpha carbon coordinate"
        )
        parser.add_argument(
            "--num-diffusion-timesteps",
            type=int, 
            default=1000,
            help="number of diffusion timesteps"
        )
        parser.add_argument(
            "--pos-beta-s",
            default=2
        )
        parser.add_argument(
            "--beta-start",
            type=float,
            default=1.e-7, 
            help="start value of beta"
        )
        parser.add_argument(
            "--beta-end",
            type=float,
            default=2.e-3,
            help="ending value of beta")
        parser.add_argument(
            "--r-beta-schedule",
            type=str, 
            default="cosine",
            help="residue type beta schedule"
        )
        parser.add_argument(
            "--r-beta-s",
            type=float,
            default=0.01,
        )
        parser.add_argument(
            "--time-emb-dim",
            type=int,
            default=0
        )
        parser.add_argument(
            "--time-emb-mode",
            type=str, # ["simple", "sin"]
            default="simple"
        )
        parser.add_argument(
            "--loss-r-weight",
            type=float,
            default=100., 

        )
        parser.add_argument(
            "--autoregressive-layer",
            type=int,
            default=2, 
        )

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.encoder_layers = args.encoder_layers
        self.decoder_layers = args.decoder_layers

        self.mask_index = self.encoder.alphabet.mask_idx
        self.k = args.knn

        self.model_mean_type = args.model_mean_type
        self.sample_time_method = args.sample_time_method
        self.beta_schedule = args.beta_schedule
        self.num_diffusion_timesteps = args.num_diffusion_timesteps
        self.pos_beta_s = args.pos_beta_s

        # residue alpha-C coordinate beta schedule
        if self.beta_schedule == "cosine":
            alphas = cosine_beta_schedule(self.num_diffusion_timesteps, self.pos_beta_s) ** 2
            betas = 1.0 - alphas
        else:
            betas = get_beta_schedule(
                beta_schedule=self.beta_schedule,
                beta_start=args.beta_start,
                beta_end=args.beta_end,
                num_diffusion_time_steps=self.num_diffusion_timesteps 
            )
            alphas = 1.0 - betas 
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[: -1])

        self.betas = to_torch_const(betas)
        self.num_time_steps = self.betas.size(0)   # 1000
        self.alphas_cumprod = to_torch_const(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch_const(alphas_cumprod_prev)

        # statistics for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = to_torch_const(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch_const(np.sqrt(1.0 - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_torch_const(np.sqrt(1.0 / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_torch_const(np.sqrt(1.0 / alphas_cumprod - 1))

        # statistics for posterior q(x_{t-1}|x_t, x_0)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_c0_coef = to_torch_const(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.posterior_mean_ct_coef = to_torch_const((1-alphas_cumprod_prev) * np.sqrt(alphas) / (1-alphas_cumprod))
        self.posterior_var = to_torch_const(self.posterior_variance)
        self.posterior_logvar = to_torch_const(np.log(np.append(self.posterior_var[1], self.posterior_var[1: ])))

        # residue type diffusion beta schedule
        self.r_beta_schedule = args.r_beta_schedule
        self.r_beta_s = args.r_beta_s
        if self.r_beta_schedule == "cosine":
            alphas_r = cosine_beta_schedule(self.num_time_steps, args.r_beta_s)
        else:
            raise NotImplementedError
        log_alphas_r = np.log(alphas_r)
        log_alphas_cumprod_r = np.cumsum(log_alphas_r)
        self.log_alphas_r = to_torch_const(log_alphas_r)
        self.log_one_minus_alphas_r = to_torch_const(log_1_min_a(log_alphas_r))
        self.log_alphas_cumprod_r = to_torch_const(log_alphas_cumprod_r)
        self.log_one_minus_alphas_cumprod_r = to_torch_const(log_1_min_a(log_alphas_cumprod_r))

        self.register_buffer("Lt_history", torch.zeros(self.num_time_steps))
        self.register_buffer("Lt_count", torch.zeros(self.num_time_steps))

        # model parameters
        self.hidden_dim = args.encoder_embed_dim
        self.num_classes = self.encoder.alphabet_size
        self.emb_dim = self.hidden_dim
        self.loss_r_weight = args.loss_r_weight
        # self.time_emb_dim = args.time_emb_dim
        # self.time_emb_mode = args.time_emb_mode

        # topping autoregressive layers
        self.autoregressive_layer_size = args.autoregressive_layer
        self.autoreggresive_layers = nn.ModuleList([])
        self.autoreggresive_layers.extend(
            [
                self.build_autoregressive_layer(args, no_encoder_attn=True)
                for _ in range(self.autoregressive_layer_size)
            ]
        )
        self._future_mask = torch.empty(0)
        self.auto_layer_norm = LayerNorm(self.hidden_dim)

    @classmethod
    def build_model(self, args, task):
        return super().build_model(args, task)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        model, alphabet = load_from_pretrained_models(args.pretrained_esm_model)
        return model

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = EGNN(in_node_nf=args.encoder_embed_dim, hidden_nf=args.encoder_embed_dim, out_node_nf=3,
                       in_edge_nf=0, device=device, n_layers=args.decoder_layers, attention=True, mode=args.egnn_mode)
        return decoder
    
    @classmethod
    def build_autoregressive_layer(cls, args, no_encoder_attn=True):
        layer = transformer_layer.TransformerDecoderLayerBase(args, no_encoder_attn)
        return layer
    
    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]
    
    def autoregressive_layer_forward(self, 
        hidden_outputs,
        incremental_state = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        if alignment_layer is None:
            alignment_layer = self.autoregressive_layer_size - 1

        enc = None
        padding_mask = None

        # B x T x C -> T x B x C
        x = hidden_outputs.transpose(0, 1)

        self_attn_padding_mask = None

        # decoder layers
        for idx, layer in enumerate(self.autoreggresive_layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )

        x = self.auto_layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        return x

    def backbone_model_forward(self, seqs, coords, target_mask):

        batch_size, n_nodes = coords.size()[0], coords.size()[1]
        k = min(n_nodes-1, self.k)
        
        padding_mask = seqs.eq(self.encoder.padding_idx)  # B, T
        embeds = (self.encoder.embed_scale * self.encoder.embed_tokens(seqs)).transpose(0, 1)  # [Batch, length, hidden]  # [Batch, length, hidden]
        
        atom_pos = coords
        edges = get_edges_batch(n_nodes, batch_size, atom_pos.detach().cpu(), k)
        atom_pos = atom_pos.reshape(-1, atom_pos.size()[-1])  # [batch * length, 3]

        # interleaving network - 30 layer model: 6-layer transformer layer + 1 equivariant graph layer
        for layer_idx, layer in enumerate(self.encoder.layers):
            embeds, _ = layer(
                    embeds,
                    self_attn_padding_mask=padding_mask,
                    need_head_weights=False,
                )
            
            if (layer_idx + 1) % 11 == 0:

                embeds = embeds.transpose(0, 1).reshape(-1, embeds.size()[-1])  # [batch * length, hidden]
                embeds, atom_pos, _ = self.decoder._modules["gcl_%d" % int(layer_idx/11)](embeds, edges, atom_pos, edge_attr=None,
                                                                                batch_size=batch_size, k=k)
                embeds = embeds.reshape(batch_size, -1, embeds.size()[-1]).transpose(0, 1)  # [length, batch, hidden]


        embeds = embeds.transpose(0, 1)   # [batch, length, hidden]

        # topping autoregressive layers, after non-autoregressive
        embeds = self.autoregressive_layer_forward(embeds)

        logits = self.encoder.lm_head(embeds)  # [batch, length, hidden]
        atom_pos = atom_pos.reshape(batch_size, -1, atom_pos.size(-1))
        outputs = {
            "coor": atom_pos,
            "residue": logits
        }
        return outputs
    
    def compose_context(h_target, h_binder, pos_target, pos_binder):
        h_ctx = torch.cat([h_target, h_binder], dim=1) # (B, LT+LB, H)
        pos_ctx = torch.cat([pos_target, pos_binder], dim=1)  # (B, LT+LB, 3)
        return h_ctx, pos_ctx
    
    def forward(self, seqs, coords, target_mask, time_step=None, ):
        # r_ctx, pos_ctx = self.compose_context(target_r, init_binder_r, target_pos, init_binder_pos)
        # current model
        outputs = self.backbone_model_forward(seqs, coords, target_mask)
        final_pos, final_r = outputs['coor'], outputs['residue']
        # final_binder_pos, final_binder_r = final_pos[~(target_mask.bool())], final_r[~(target_mask.bool())]

        preds = {
            'pred_pos': final_pos,
            'pred_r': final_r,
        }
        return preds
    
    # residue type diffusion process
    def q_r_pred_one_timestep(self, log_rt_1, t):
        # q(rt|r_{t-1})
        log_alpha_t = extract(self.log_alphas_r, t)
        log_1_min_alpha_t = extract(self.log_one_minus_alphas_r, t)

        log_probs = log_add_exp(
            log_rt_1+log_alpha_t,
            log_1_min_alpha_t-np.log(self.num_classes)
        )
        return log_probs
    
    def q_r_pred(self, log_r0, t):
        # q(rt|r0)
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_r, t)
        log_1_min_cumprod_alpha = extract(self.log_one_minus_alphas_cumprod_r, t)

        log_probs = log_add_exp(
            log_r0 + log_cumprod_alpha_t,  # B, L, C
            log_1_min_cumprod_alpha - np.log(self.num_classes)
        )
        return log_probs
    
    def q_r_sample(self, log_r0, t):
        log_qrt_r0 = self.q_r_pred(log_r0, t)   # [B, L, C]
        sample_index = log_sample_categorical(log_qrt_r0)   # [B, L]
        log_sample = index_to_log_one_hot(sample_index, self.num_classes)  # [B, L, C]
        return sample_index, log_sample
    
    def q_r_posterior(self, log_r0, log_rt, t):
        # q(r_{t-1}| r0, rt)
        t_minus1 = t-1
        t_minus1 = torch.where(t_minus1 < 0, torch.zeros_like(t_minus1), t_minus1)
        log_qrt1_r0 = self.q_r_pred(log_r0, t_minus1)   # q(rt|r0)
        unnormed_logprobs = log_qrt1_r0 + self.q_r_pred_one_timestep(log_rt, t)
        log_rt1_given_rt_r0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)
        return log_rt1_given_rt_r0
    
    def kl_r_prior(self, log_r_start, batch):
        num_graphs = batch.max().item() + 1
        log_qrT_prob = self.q_r_pred(log_r_start, t=[self.num_time_steps-1] * num_graphs, batch=batch)
        log_half_prob = -torch.log(self.num_classes*torch.ones_like(log_qrT_prob))
        kl_prior = cateforical_kl(log_qrT_prob, log_half_prob)
        kl_prior = scatter_mean(kl_prior, batch, dim=0)

    def _predict_x0_from_eps(self, xt, eps, t):
        pos0_from_e = extract(self.sqrt_recip_alphas_cumprod, t) * xt - extract(self.sqrt_recipm1_alphas_cumprod, t) * eps
        return pos0_from_e
    
    def q_pos_posterior(self, x0, xt, t):
        pos_model_mean = extract(self.posterior_mean_c0_coef, t) * x0 + extract(self.posterior_mean_ct_coef, t) * xt
        return pos_model_mean
    
    def kl_pos_prior(self, pos0, batch):
        num_graphs = batch.max().item() + 1
        a_pos = extract(self.alphas_cumprod, [self.num_time_steps - 1] * num_graphs, batch)
        pos_model_mean = a_pos.sqrt() * pos0
        pos_log_var = torch.log((1.0-a_pos).sqrt())
        kl_prior = normal_kl(torch.zeros_like(pos_model_mean), torch.zeros_like(pos_log_var), pos_model_mean, pos_log_var)
        kl_prior = scatter_mean(kl_prior, batch, dim=0)
        return kl_prior
    
    def sample_time(self, num_graphs, device, method):
        if method == "importance":
            if not (self.Lt_count > 10).all():
                return self.sample_time(num_graphs, device, method="symmetric")
            
            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            time_step = torch.multinomial(pt_all, num_samples=num_graphs, replacement=True)
            pt = pt_all.gather(dim=0, index=time_step)
            return time_step, pt
        
        elif method == "symmetric":
            time_step = torch.randint(0, self.num_time_steps, size=(num_graphs//2 +1, )).to("cuda")
            time_step = torch.cat(
                [time_step, self.num_time_steps-time_step-1], dim=0
            )[: num_graphs]
            pt = torch.ones_like(time_step).float() / self.num_time_steps
            return time_step, pt
        
        else:
            raise NotImplementedError
    
    def compute_pos_Lt(self, pos_model_mean, x0, xt, t, batch):
        pos_log_var = extract(self.posterior_logvar, t, batch)
        pos_true_mean = self.q_pos_posterior(x0=x0, xt=xt, t=t, batch=batch)
        kl_pos = normal_kl(pos_true_mean, pos_log_var, pos_model_mean, pos_log_var)
        kl_pos = kl_pos / np.log(2.0)

        decoder_nll_pos = -log_normal(x0, means=pos_model_mean, log_scales=0.5*pos_log_var)
        assert kl_pos.shape == decoder_nll_pos.shape
        mask = (t==0).float()[batch]
        loss_pos = scatter_mean(mask * decoder_nll_pos + (1.0-mask) * kl_pos, batch, dim=0)
        return loss_pos
    
    def compute_r_Lt(self, log_r_model_prob, log_r0, log_r_true_prob, t, target_mask):
        kl_r = cateforical_kl(log_r_true_prob, log_r_model_prob)
        decoder_nll_r = -log_categorical(log_r0, log_r_model_prob)
        assert kl_r.shape == decoder_nll_r.shape
        mask = (t==0).float().unsqueeze(-1)
        loss_r = (mask * decoder_nll_r + (1.0 - mask) * kl_r) * (target_mask==0).int()
        return loss_r

    def get_diffusion_loss(self, seqs, coords, target_mask, time_step=None):
        assert seqs.size() == target_mask.size() == coords.size()[: 2]
        # target_pos = (coords[: , 1: -1, :])[target_mask[:, 1: -1].bool()]
        # target_r = seqs[:, 1: -1][target_mask[:, 1: -1].bool()]
        # binder_pos = coords[~(target_mask.bool())]
        # binder_r = seqs[~(target_mask.bool())]

        num_graphs = coords.size(0)  # batch_size

        # 1. sample noise levels
        if time_step is None:
            time_step, pt = self.sample_time(num_graphs, seqs.device, self.sample_time_method)  # [batch_size, ]
        else:
            pt = torch.ones_like(time_step).float() / self.num_time_steps
        a = self.alphas_cumprod.index_select(0, time_step)  # (num_graphs, )

        # 2. perturb pos and v
        a_pos = a.unsqueeze(-1).unsqueeze(-1)  # (batch_size, 1, 1)

        pos_noise = torch.zeros_like(coords).normal_()
        complex_pos_perturbed = (a_pos.sqrt() * coords + (1.0 - a_pos).sqrt() * pos_noise) * ((~(target_mask.bool())).int()).unsqueeze(-1) \
            + target_mask.unsqueeze(-1) * coords
        log_complex_r0 = index_to_log_onehot(seqs, self.num_classes)
        complex_r_perturbed, log_complex_rt = self.q_r_sample(log_complex_r0, time_step)
        complex_seqs = complex_r_perturbed * (~(target_mask.bool())).int() + seqs * target_mask

        # 3. forward-pass NN, feed perturbed pos and v, output noise
        preds = self.forward(
            complex_seqs,
            complex_pos_perturbed,  
            target_mask=target_mask,
            time_step=time_step, 
        )

        pred_pos, pred_r = preds['pred_pos'], preds['pred_r']
        pred_pos_noise = pred_pos - complex_pos_perturbed
        # residue position
        if self.model_mean_type == 'noise':
            pos0_from_e = self._predict_x0_from_eps(
                xt=complex_pos_perturbed, eps=pred_pos_noise, t=time_step)
            pos_model_mean = self.q_pos_posterior(
                x0=pos0_from_e, xt=complex_pos_perturbed, t=time_step)
        elif self.model_mean_type == 'C0':
            pos_model_mean = self.q_pos_posterior(
                x0=pred_pos, xt=complex_pos_perturbed, t=time_step)
        else:
            raise ValueError

        # atom pos loss
        if self.model_mean_type == 'C0':
            target, pred = coords, pred_pos
        elif self.model_mean_type == 'noise':
            target, pred = pos_noise, pred_pos_noise
        else:
            raise ValueError
        loss_pos = torch.sum(((pred - target) ** 2).sum(-1) * (target_mask==0).int(), dim=1) / (target_mask == 0).int().sum(-1)
        loss_pos = torch.mean(loss_pos)

        # atom type loss
        log_r_recon = F.log_softmax(pred_r, dim=-1)  # predicted r_0
        log_r_model_prob = self.q_r_posterior(log_r_recon, log_complex_rt, time_step)
        log_r_true_prob = self.q_r_posterior(log_complex_r0, log_complex_rt, time_step)
    
        kl_r = self.compute_r_Lt(log_r_model_prob=log_r_model_prob, 
                                 log_r0=log_complex_r0, 
                                 log_r_true_prob=log_r_true_prob,
                                 t=time_step,
                                 target_mask=target_mask)
        loss_r = torch.mean(torch.sum(kl_r, dim=-1) / (target_mask == 0).int().sum(-1))
        loss = loss_pos + loss_r * self.loss_r_weight

        return {
            'loss_pos': loss_pos,
            'loss_residue': loss_r,
            'loss': loss,
            'x0': coords,
            'predd_pos': pred_pos,
            'pred_r': pred_r,
            'pred_pos_noise': pred_pos_noise,
            'r_recon': F.softmax(pred_r, dim=-1)
        }
    
    @torch.no_grad()
    def sample_diffusion(self, seqs, coords, target_mask, time_step=None, ss_initial=None):

        if time_step is None:
            time_step = self.num_time_steps
        num_graphs, length = seqs.size(0), seqs.size(1)

        pos_traj, r_traj = [], []
        r0_pred_traj, rt_pred_traj = [], []

        # sample initial structure based on energy
        scores, structures = [], []
        for _ in range(10):
            pos_noise = torch.zeros_like(coords).normal_()
            complex_pos_perturbed = pos_noise * ((~(target_mask.bool())).int()).unsqueeze(-1) + target_mask.unsqueeze(-1) * coords
            k = min(4, length - 3)
            dist = get_knn_distance(num_graphs, complex_pos_perturbed[:, 1: -1, :].detach().cpu(), k)
            energy = np.sum(np.square(np.array(dist) - 0.06401))
            scores.append(energy)
            structures.append(complex_pos_perturbed)
        indices = np.argsort(np.array(scores))
        complex_pos_perturbed = structures[indices[0]]
            
        index = list(target_mask[0].cpu()).index(0)
        target_protein_len = index -1
        binder_protein_len = seqs.size(1) - index - 1
        pos = 0
        complex_seqs = seqs.clone().detach()
        while pos < binder_protein_len:
            cand = random.choice(ss_initial)
            complex_seqs[:, index+pos: min(index+pos+cand.size(0), index + binder_protein_len)] = cand[: min(cand.size(0), binder_protein_len-pos)]
            pos = pos + cand.size(0)

        # time sequence
        ss_seqs = torch.empty_like(complex_seqs).copy_(complex_seqs)
        time_seq = list(reversed(range(self.num_time_steps)))
        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=coords.device)
            preds = self(
                complex_seqs,
                complex_pos_perturbed,  
                target_mask=target_mask,
                time_step=t, 
            )
            # Compute posterior mean and variance
            if self.model_mean_type == 'noise':
                pred_pos_noise = preds['pred_binder_pos'] - complex_pos_perturbed
                pos0_from_e = self._predict_x0_from_eps(xt=complex_pos_perturbed, eps=pred_pos_noise, t=t)
                r0_from_e = preds['pred_binder_r']
            elif self.model_mean_type == 'C0':
                pos0_from_e = preds['pred_pos']
                r0_from_e = preds['pred_r']
            else:
                raise ValueError

            pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=complex_pos_perturbed, t=t)
            pos_log_variance = extract(self.posterior_logvar, t)
            # no noise when t == 0
            nonzero_mask = (1 - (t == 0).float()).unsqueeze(-1).unsqueeze(-1)
            complex_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(
                complex_pos_perturbed)
            complex_pos_perturbed = complex_pos_next * ((~(target_mask.bool())).int()).unsqueeze(-1) + target_mask.unsqueeze(-1) * complex_pos_perturbed

            log_complex_r_recon = F.log_softmax(r0_from_e, dim=-1)
            log_complex_r = index_to_log_onehot(complex_seqs, self.num_classes)
            log_model_prob = self.q_r_posterior(log_complex_r_recon, log_complex_r, t)
            complex_seq_r_next = log_sample_categorical(log_model_prob)
            complex_seqs = complex_seq_r_next * (~(target_mask.bool())).int() + seqs * target_mask

            r0_pred_traj.append(log_complex_r_recon.clone().cpu())
            rt_pred_traj.append(log_model_prob.clone().cpu())
                
            pos_traj.append(complex_pos_perturbed.clone().cpu())
            r_traj.append(complex_seqs.clone().cpu())

        return {
            'pos': complex_pos_perturbed,
            'residue': complex_seqs,
            'pos_traj': pos_traj,
            'r_traj': r_traj,
            'r0_traj': r0_pred_traj,
            'rt_traj': rt_pred_traj,
            "ss_seq": ss_seqs
        }

    def max_positions(self):
        """Maximum length supported by the model."""
        return (self.encoder.max_positions(), self.encoder.max_positions())
    


@register_model_architecture("protein_protein_complex_diffusion_model", "protein_protein_complex_diffusion_model")
def base_architecture(args):
    transformer_base_architecture(args)


@register_model_architecture("protein_protein_complex_diffusion_model", "protein_protein_complex_diffusion_model_esm")
def transformer_diffusion(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 320)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1280)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 5)
    base_architecture(args)