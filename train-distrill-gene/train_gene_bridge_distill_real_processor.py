'''
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node 4 \
  /data2/xiaoxinyu/project/qformer/train_gene_bridge_distill_real_processor.py \
  --model_path /data2/xiaoxinyu/project/model_cpt_v7_qformer \
  --data_jsonl /data2/xiaoxinyu/project/new_data/gene_data.jsonl \
  --out_dir /data2/xiaoxinyu/project/qformer/distill_out_cpt_v7_real_processor \
  --epochs 1 \
  --batch_size 2 \
  --lr 1e-4 \
  --lambda_ce 0.2 \
  --lambda_cos 1.0 \
  --lambda_nce 1.0 \
  --lambda_cls 0.5 \
  --temp 0.07 \
  --save_step 500
  '''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import time
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from transformers import AutoModel, AutoProcessor


# =========================================================
# distributed helpers
# =========================================================
def ddp_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def ddp_init():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)


def ddp_rank() -> int:
    return dist.get_rank() if ddp_is_initialized() else 0


def ddp_world_size() -> int:
    return dist.get_world_size() if ddp_is_initialized() else 1


def ddp_barrier():
    if ddp_is_initialized():
        dist.barrier()


@torch.no_grad()
def all_gather_variable_batch(x: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
    """
    Gather [B, D] tensors across ranks. Handles uneven final local batches.
    """
    if not ddp_is_initialized():
        return x, [x.shape[0]]

    device = x.device
    b = torch.tensor([x.shape[0]], device=device, dtype=torch.long)
    bs = [torch.zeros_like(b) for _ in range(ddp_world_size())]
    dist.all_gather(bs, b)
    sizes = [int(t.item()) for t in bs]
    max_b = max(sizes)

    if x.shape[0] < max_b:
        pad = torch.zeros((max_b - x.shape[0], x.shape[1]), device=device, dtype=x.dtype)
        x_pad = torch.cat([x, pad], dim=0)
    else:
        x_pad = x

    outs = [torch.zeros_like(x_pad) for _ in range(ddp_world_size())]
    dist.all_gather(outs, x_pad)
    return torch.cat([o[:sizes[i]] for i, o in enumerate(outs)], dim=0), sizes


@torch.no_grad()
def sync_gradients(params: List[torch.nn.Parameter]):
    """
    Manual data-parallel gradient averaging for this script's custom forward path.
    Creates zero grads for locally unused trainable params so every rank participates.
    """
    if not ddp_is_initialized():
        return

    world = ddp_world_size()
    for p in params:
        has_grad = torch.tensor(
            [1 if p.grad is not None else 0],
            device=p.device,
            dtype=torch.long,
        )
        dist.all_reduce(has_grad, op=dist.ReduceOp.SUM)
        if int(has_grad.item()) == 0:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world)


# =========================================================
# label extraction
# =========================================================
LAYER_MAP_ZH2EN = {
    "分子层": "Layer_1",
    "外颗粒层": "Layer_2",
    "外锥体层": "Layer_3",
    "内颗粒层": "Layer_4",
    "内锥体层": "Layer_5",
    "多形层": "Layer_6",
    "白质层": "WM",
}

LABEL2ID = {
    "Layer_1": 0,
    "Layer_2": 1,
    "Layer_3": 2,
    "Layer_4": 3,
    "Layer_5": 4,
    "Layer_6": 5,
    "WM": 6,
    "breast": 7,
    "skin": 8,
    "heart": 9,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_CLASSES = len(LABEL2ID)


def extract_label_from_sample(sample: Dict[str, Any]) -> Tuple[Optional[int], bool]:
    """
    返回:
        label_id: int or None
        has_label: bool
    """
    gene_path = str(sample.get("gene", ""))
    messages = sample.get("messages", [])
    assist_text = ""
    if len(messages) >= 2 and "content" in messages[1]:
        assist_text = str(messages[1]["content"])

    # -------------------------
    # 1) DLPFC: 从文本中提取脑区层 label
    # -------------------------
    if "/data2/xiaoxinyu/data/DLPFC/gene-h5ad/" in gene_path:
        for zh, en in LAYER_MAP_ZH2EN.items():
            if zh in assist_text:
                return LABEL2ID[en], True
        return None, False

    # -------------------------
    # 2) STimage: 直接按路径给组织标签
    # -------------------------
    if "/data2/xiaoxinyu/data/STimage-1K4M/process-data/ST/gene_h5ad/Human_Breast" in gene_path:
        return LABEL2ID["breast"], True

    if "/data2/xiaoxinyu/data/STimage-1K4M/process-data/ST/gene_h5ad/GSE144239" in gene_path:
        return LABEL2ID["skin"], True

    if "/data2/xiaoxinyu/data/STimage-1K4M/process-data/ST/gene_h5ad/Human_Heart" in gene_path:
        return LABEL2ID["heart"], True

    # -------------------------
    # 3) CellWhisperer: 不做分类监督
    # -------------------------
    if "/data2/xiaoxinyu/data/cellwhisperer/one_cell_h5ad" in gene_path:
        return None, False

    return None, False


# =========================================================
# dataset
# =========================================================
class GeneJsonlDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.items.append(json.loads(line))
        print(f"[DATA] loaded {len(self.items)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        return self.items[i]


# =========================================================
# helpers
# =========================================================
def find_first(ids: torch.Tensor, token_id: int) -> Optional[int]:
    pos = (ids == token_id).nonzero(as_tuple=False)
    if pos.numel() == 0:
        return None
    return int(pos[0].item())


def expand_gene_span(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: Optional[torch.Tensor],
    gene_token_id: int,
    unk_id: int,
    span_len: int = 32
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    将文本中的一个 <gene> token 展开成 span_len 个 <unk> 占位。
    返回:
        new_input_ids, new_attention_mask, new_labels, gene_bound [[start, end]]
    """
    pos = find_first(input_ids, gene_token_id)
    if pos is None:
        return input_ids, attention_mask, labels, None

    patch = torch.full((span_len,), unk_id, dtype=input_ids.dtype)
    new_input_ids = torch.cat([input_ids[:pos], patch, input_ids[pos + 1:]], dim=0)

    patch_m = torch.ones((span_len,), dtype=attention_mask.dtype)
    new_attn = torch.cat([attention_mask[:pos], patch_m, attention_mask[pos + 1:]], dim=0)

    new_labels = None
    if labels is not None:
        patch_l = torch.full((span_len,), -100, dtype=labels.dtype)
        new_labels = torch.cat([labels[:pos], patch_l, labels[pos + 1:]], dim=0)

    gene_bound = torch.tensor([[pos, pos + span_len]], dtype=torch.long)
    return new_input_ids, new_attn, new_labels, gene_bound


def build_prompt_and_full_text(tokenizer, messages: List[Dict[str, str]]) -> Tuple[str, str]:
    """
    prompt_text: 只有 user（带 generation prompt）
    full_text: user + assistant（不加 generation prompt）
    """
    prompt_msgs = [messages[0]]
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs,
        tokenize=False,
        add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return prompt_text, full_text


def pool_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    x: [B, L, D]
    mask: [B, L] (1 for keep)
    """
    mask = mask.to(dtype=x.dtype)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (x * mask.unsqueeze(-1)).sum(dim=1) / denom


def load_local_reference_var_names() -> List[str]:
    candidates = [
        "/data2/xiaoxinyu/project/model/gene_tokenizer/model-symbel.h5ad",
        "/data2/xiaoxinyu/project/model/gene_tokenizer/model-ensembl.h5ad",
    ]

    for p in candidates:
        if os.path.exists(p):
            print(f"[INFO] load local reference model from: {p}")
            ref_model = ad.read_h5ad(p)
            return ref_model.var_names.astype(str).tolist()

    raise FileNotFoundError("Cannot find local reference model h5ad.")


def build_aligned_gene_adata(one_gene: ad.AnnData, ref_var_names: List[str]) -> ad.AnnData:
    """
    按 reference var_names 重排/补零，保证输入与真实 processor 路径一致。
    """
    ref_var_names = list(map(str, ref_var_names))
    src_var_names = list(map(str, one_gene.var_names))
    src_index = {g: i for i, g in enumerate(src_var_names)}
    n_ref = len(ref_var_names)

    X_src = one_gene.X

    if sp.issparse(X_src):
        X_src = X_src.tocsr()
        rows, cols, vals = [], [], []
        for j, g in enumerate(ref_var_names):
            if g in src_index:
                src_j = src_index[g]
                v = X_src[0, src_j]
                if v != 0:
                    rows.append(0)
                    cols.append(j)
                    vals.append(float(v))
        X_new = sp.csr_matrix((vals, (rows, cols)), shape=(1, n_ref), dtype=np.float32)
    else:
        X_src = np.asarray(X_src)
        X_new = np.zeros((1, n_ref), dtype=np.float32)
        for j, g in enumerate(ref_var_names):
            if g in src_index:
                X_new[0, j] = float(X_src[0, src_index[g]])

    new_obs = one_gene.obs.copy()
    new_var = pd.DataFrame(index=pd.Index(ref_var_names, name=one_gene.var_names.name))
    out = ad.AnnData(X=X_new, obs=new_obs, var=new_var)
    return out


def build_empty_gene_adata(ref_var_names: List[str]) -> ad.AnnData:
    X = np.zeros((1, len(ref_var_names)), dtype=np.float32)
    var = pd.DataFrame(index=pd.Index(ref_var_names, name="gene"))
    obs = pd.DataFrame(index=pd.Index(["empty"]))
    return ad.AnnData(X=X, obs=obs, var=var)


def tokenize_gene_real_processor(
    processor,
    gene_path: Optional[str],
    ref_var_names: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    用真实 processor 内部的 gene_tokenizer 路径取 gene_input_ids / gene_attention_mask。
    """
    if gene_path is None or (not os.path.exists(gene_path)):
        adata = build_empty_gene_adata(ref_var_names)
    else:
        adata_raw = ad.read_h5ad(gene_path)
        adata = build_aligned_gene_adata(adata_raw, ref_var_names)

    gene_arrays = adata.X
    gene_inputs = processor.gene_tokenizer(gene_arrays)

    gene_input_ids = torch.tensor(gene_inputs["input_ids"][0], dtype=torch.long)
    gene_attention_mask = torch.tensor(gene_inputs["attention_mask"][0], dtype=torch.long)
    return gene_input_ids, gene_attention_mask


# =========================================================
# batch
# =========================================================
@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    position_ids: torch.Tensor

    gene_input_ids: torch.Tensor
    gene_attention_mask: torch.Tensor
    gene_bound: List[Optional[torch.Tensor]]

    teacher_input_ids: torch.Tensor
    teacher_attention_mask: torch.Tensor
    teacher_labels: torch.Tensor
    teacher_position_ids: torch.Tensor

    cls_labels: torch.Tensor
    has_cls_label: torch.Tensor


def collate_fn(
    samples: List[Dict[str, Any]],
    processor,
    gene_span_len: int,
    ref_var_names: List[str],
    device: torch.device,
) -> Batch:
    tok = processor.tokenizer
    gene_token_id = tok.convert_tokens_to_ids("<gene>")
    unk_id = tok.unk_token_id if tok.unk_token_id is not None else tok.convert_tokens_to_ids("<unk>")

    input_ids_list, attn_list, labels_list, pos_list, gene_bound_list = [], [], [], [], []
    teacher_ids_list, teacher_attn_list, teacher_labels_list, teacher_pos_list = [], [], [], []
    gene_ids_list, gene_attn_list = [], []

    cls_labels, has_cls_labels = [], []

    for s in samples:
        messages = s["messages"]
        gene_path = s.get("gene", None)

        # -------- label extraction --------
        label_id, has_label = extract_label_from_sample(s)
        cls_labels.append(-100 if label_id is None else int(label_id))
        has_cls_labels.append(1 if has_label else 0)

        # -------- text tokenize --------
        prompt_text, full_text = build_prompt_and_full_text(tok, messages)

        prompt_ids = tok(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"][0]

        full_enc = tok(
            full_text,
            return_tensors="pt",
            add_special_tokens=False
        )
        full_ids = full_enc["input_ids"][0]
        full_attn = full_enc["attention_mask"][0]

        assist_start = prompt_ids.shape[0]

        labels = full_ids.clone()
        labels[:assist_start] = -100

        # student / teacher 都展开 <gene> 占位，保持长度对齐
        full_ids_s, full_attn_s, labels_s, gene_bound = expand_gene_span(
            full_ids, full_attn, labels,
            gene_token_id, unk_id,
            span_len=gene_span_len
        )
        full_ids_t, full_attn_t, labels_t, _ = expand_gene_span(
            full_ids, full_attn, labels,
            gene_token_id, unk_id,
            span_len=gene_span_len
        )

        pos_s = torch.arange(full_ids_s.shape[0], dtype=torch.long)
        pos_t = torch.arange(full_ids_t.shape[0], dtype=torch.long)

        # -------- real processor gene tokenize --------
        gene_input_ids, gene_attention_mask = tokenize_gene_real_processor(
            processor=processor,
            gene_path=gene_path,
            ref_var_names=ref_var_names,
        )

        input_ids_list.append(full_ids_s)
        attn_list.append(full_attn_s)
        labels_list.append(labels_s)
        pos_list.append(pos_s)
        gene_bound_list.append(gene_bound)

        teacher_ids_list.append(full_ids_t)
        teacher_attn_list.append(full_attn_t)
        teacher_labels_list.append(labels_t)
        teacher_pos_list.append(pos_t)

        gene_ids_list.append(gene_input_ids)
        gene_attn_list.append(gene_attention_mask)

    def pad_1d(xs, pad_val):
        max_len = max(x.shape[0] for x in xs)
        out = []
        for x in xs:
            if x.shape[0] < max_len:
                out.append(F.pad(x, (0, max_len - x.shape[0]), value=pad_val))
            else:
                out.append(x)
        return torch.stack(out, dim=0)

    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0

    input_ids = pad_1d(input_ids_list, pad_id)
    attention_mask = pad_1d(attn_list, 0)
    labels = pad_1d(labels_list, -100)
    position_ids = pad_1d(pos_list, 0)

    teacher_input_ids = pad_1d(teacher_ids_list, pad_id)
    teacher_attention_mask = pad_1d(teacher_attn_list, 0)
    teacher_labels = pad_1d(teacher_labels_list, -100)
    teacher_position_ids = pad_1d(teacher_pos_list, 0)

    gene_input_ids = torch.stack(gene_ids_list, dim=0)
    gene_attention_mask = torch.stack(gene_attn_list, dim=0)

    cls_labels = torch.tensor(cls_labels, dtype=torch.long)
    has_cls_label = torch.tensor(has_cls_labels, dtype=torch.bool)

    return Batch(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        labels=labels.to(device),
        position_ids=position_ids.to(device),

        gene_input_ids=gene_input_ids.to(device),
        gene_attention_mask=gene_attention_mask.to(device),
        gene_bound=gene_bound_list,

        teacher_input_ids=teacher_input_ids.to(device),
        teacher_attention_mask=teacher_attention_mask.to(device),
        teacher_labels=teacher_labels.to(device),
        teacher_position_ids=teacher_position_ids.to(device),

        cls_labels=cls_labels.to(device),
        has_cls_label=has_cls_label.to(device),
    )


# =========================================================
# main
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--data_jsonl", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--device", type=str, default="cuda:0", help="single-process device; ignored under torchrun")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])

    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    ap.add_argument("--gene_span_len", type=int, default=32)
    ap.add_argument("--log_every", type=int, default=20)

    # loss weights
    ap.add_argument("--lambda_ce", type=float, default=0.2)
    ap.add_argument("--lambda_cos", type=float, default=1.0)
    ap.add_argument("--lambda_nce", type=float, default=1.0)
    ap.add_argument("--lambda_cls", type=float, default=0.5)
    ap.add_argument("--temp", type=float, default=0.07)

    ap.add_argument("--save_step", type=int, default=500, help=">0 时每隔该 step 保存一次 checkpoint")

    args = ap.parse_args()

    ddp_init()
    rank = ddp_rank()
    world = ddp_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_master = (rank == 0)

    if is_master:
        os.makedirs(args.out_dir, exist_ok=True)
    ddp_barrier()

    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    curve_path = os.path.join(args.out_dir, "loss_curve.png")
    ckpt_path = os.path.join(args.out_dir, "gene_bridge_distill_real_processor.pt")

    if ddp_is_initialized():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }[args.dtype]

    if is_master:
        print(f"[DDP] world_size={world}, rank={rank}, local_rank={local_rank}, device={device}")
        print("Loading model:", args.model_path)
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype
    ).to(device)

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )

    ref_var_names = load_local_reference_var_names()

    # 新加 gene-side classification head
    hidden_size = model.config.hidden_size
    if not hasattr(model, "gene_cls_head"):
        model.gene_cls_head = nn.Linear(hidden_size, NUM_CLASSES).to(device)
        if is_master:
            print(f"[INFO] add model.gene_cls_head: Linear({hidden_size}, {NUM_CLASSES})")

    # freeze all, train qformer + projector + cls_head
    for p in model.parameters():
        p.requires_grad = False

    model.gene_qformer.requires_grad_(True)
    model.gene_projector.requires_grad_(True)
    model.gene_cls_head.requires_grad_(True)

    if hasattr(model, "nicheformer"):
        model.nicheformer.requires_grad_(False)

    params = [p for p in model.parameters() if p.requires_grad]
    if is_master:
        print(f"[INFO] trainable params = {sum(p.numel() for p in params):,}")

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    dataset = GeneJsonlDataset(args.data_jsonl)
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=False) if ddp_is_initialized() else None
    steps_per_epoch = math.ceil(len(dataset) / (args.batch_size * world))
    total_steps = steps_per_epoch * args.epochs

    def lr_lambda(step):
        warmup = int(0.1 * total_steps)
        if step < warmup:
            return float(step) / max(1, warmup)
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda xs: collate_fn(
            xs,
            processor=processor,
            gene_span_len=args.gene_span_len,
            ref_var_names=ref_var_names,
            device=device,
        )
    )

    loss_hist, ce_hist, cos_hist, nce_hist, cls_hist = [], [], [], [], []

    model.train()
    global_step = 0

    f_log = open(log_path, "w", encoding="utf-8") if is_master else None

    try:
        for ep in range(args.epochs):
            if sampler is not None:
                sampler.set_epoch(ep)

            for batch in loader:
                global_step += 1

                # -------------------------------------------------
                # Teacher: text-only hidden state target
                # -------------------------------------------------
                with torch.no_grad():
                    out_t = model.llm(
                        input_ids=batch.teacher_input_ids,
                        attention_mask=batch.teacher_attention_mask,
                        position_ids=batch.teacher_position_ids,
                        output_hidden_states=True,
                        use_cache=False,
                    )
                    t_hidden = out_t.hidden_states[-1]  # [B, L, D]
                    t_mask = (batch.teacher_labels != -100).long()
                    t_vec = pool_mean(t_hidden, t_mask)
                    t_vec = F.normalize(t_vec, dim=-1)

                # -------------------------------------------------
                # Student: real processor gene path -> get_vllm_embedding
                # -------------------------------------------------
                data = {
                    "input_ids": batch.input_ids,
                    "attention_mask": batch.attention_mask,
                    "position_ids": batch.position_ids,

                    # image branch empty
                    "pixel_values": [[] for _ in range(batch.input_ids.shape[0])],
                    "tgt_sizes": [None] * batch.input_ids.shape[0],
                    "image_bound": [[] for _ in range(batch.input_ids.shape[0])],

                    # real processor gene tokens
                    "gene_input_ids": batch.gene_input_ids,
                    "gene_attention_mask": batch.gene_attention_mask,
                    "gene_bound": batch.gene_bound,
                }

                inputs_embeds, _ = model.get_vllm_embedding(data)
                inputs_embeds = inputs_embeds.to(dtype=dtype)

                out_s = model.llm(
                    input_ids=None,
                    inputs_embeds=inputs_embeds,
                    attention_mask=batch.attention_mask,
                    position_ids=batch.position_ids,
                    labels=batch.labels,
                    output_hidden_states=False,
                    use_cache=False,
                )
                loss_ce = out_s.loss

                # gene vec from injected span
                g_vecs = []
                g_vecs_raw = []
                for i, gb in enumerate(batch.gene_bound):
                    if gb is None:
                        zero = torch.zeros(
                            (inputs_embeds.shape[-1],),
                            device=device,
                            dtype=inputs_embeds.dtype
                        )
                        g_vecs_raw.append(zero)
                        g_vecs.append(F.normalize(zero + 1e-6, dim=-1))
                        continue

                    s, e = int(gb[0, 0].item()), int(gb[0, 1].item())
                    g_tokens = inputs_embeds[i, s:e, :]   # [span, D]
                    g_raw = g_tokens.mean(dim=0)
                    g_vecs_raw.append(g_raw)
                    g_vecs.append(F.normalize(g_raw, dim=-1))

                g_vec_raw = torch.stack(g_vecs_raw, dim=0)   # [B, D]
                g_vec = torch.stack(g_vecs, dim=0)           # [B, D]

                # distillation losses
                loss_cos = (1.0 - (g_vec * t_vec).sum(dim=-1)).mean()

                t_vec_all, t_sizes = all_gather_variable_batch(t_vec)
                nce_offset = sum(t_sizes[:rank])
                logits_nce = (g_vec @ t_vec_all.t()) / args.temp
                targets = torch.arange(logits_nce.shape[0], device=logits_nce.device)
                targets = targets + nce_offset
                loss_nce = F.cross_entropy(logits_nce, targets)

                # gene-side classification loss (only labeled samples)
                logits_cls = model.gene_cls_head(g_vec_raw.float())
                if batch.has_cls_label.any():
                    loss_cls = F.cross_entropy(
                        logits_cls[batch.has_cls_label],
                        batch.cls_labels[batch.has_cls_label]
                    )
                else:
                    loss_cls = torch.tensor(0.0, device=device)

                loss = (
                    args.lambda_ce * loss_ce
                    + args.lambda_cos * loss_cos
                    + args.lambda_nce * loss_nce
                    + args.lambda_cls * loss_cls
                )

                opt.zero_grad(set_to_none=True)
                loss.backward()
                sync_gradients(params)
                grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                opt.step()
                scheduler.step()

                # optional periodic step save
                if is_master and args.save_step > 0 and (global_step % args.save_step == 0):
                    step_ckpt_path = os.path.join(
                        args.out_dir,
                        f"gene_bridge_distill_real_processor_step{global_step}.pt"
                    )
                    torch.save(
                        {
                            "gene_qformer": model.gene_qformer.state_dict(),
                            "gene_projector": model.gene_projector.state_dict(),
                            "gene_cls_head": model.gene_cls_head.state_dict(),
                            "args": vars(args),
                            "global_step": global_step,
                            "epoch": ep,
                            "label2id": LABEL2ID,
                        },
                        step_ckpt_path
                    )
                    print(f"saved step checkpoint at step={global_step}: {step_ckpt_path}")

                if is_master:
                    loss_hist.append(float(loss.item()))
                    ce_hist.append(float(loss_ce.item()))
                    cos_hist.append(float(loss_cos.item()))
                    nce_hist.append(float(loss_nce.item()))
                    cls_hist.append(float(loss_cls.item()))

                if is_master and global_step % args.log_every == 0:
                    rec = {
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "epoch": ep,
                        "step": global_step,
                        "loss": float(loss.item()),
                        "loss_ce": float(loss_ce.item()),
                        "loss_cos": float(loss_cos.item()),
                        "loss_nce": float(loss_nce.item()),
                        "loss_cls": float(loss_cls.item()),
                        "grad_norm": float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm),
                        "lr": float(scheduler.get_last_lr()[0]),
                        "num_labeled_in_batch": int(batch.has_cls_label.sum().item()),
                        "rank0_batch_size": int(batch.input_ids.shape[0]),
                        "world_size": world,
                        "global_nce_batch": int(t_vec_all.shape[0]),
                    }
                    print(
                        f"[ep {ep}] step={global_step} "
                        f"loss={rec['loss']:.4f} "
                        f"ce={rec['loss_ce']:.4f} "
                        f"cos={rec['loss_cos']:.4f} "
                        f"nce={rec['loss_nce']:.4f} "
                        f"cls={rec['loss_cls']:.4f} "
                        f"labeled={rec['num_labeled_in_batch']} "
                        f"grad_norm={rec['grad_norm']:.4e} "
                        f"lr={rec['lr']:.2e} "
                        f"Bg={rec['global_nce_batch']}"
                    )
                    f_log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f_log.flush()

            # epoch save
            if is_master:
                torch.save(
                    {
                        "gene_qformer": model.gene_qformer.state_dict(),
                        "gene_projector": model.gene_projector.state_dict(),
                        "gene_cls_head": model.gene_cls_head.state_dict(),
                        "args": vars(args),
                        "global_step": global_step,
                        "epoch": ep,
                        "label2id": LABEL2ID,
                    },
                    ckpt_path
                )
                print("saved:", ckpt_path)

        if is_master:
            # plot
            plt.figure(figsize=(8, 5))
            plt.plot(loss_hist, label="total")
            plt.plot(ce_hist, label="ce")
            plt.plot(cos_hist, label="cos")
            plt.plot(nce_hist, label="nce")
            plt.plot(cls_hist, label="cls")
            plt.legend()
            plt.xlabel("step")
            plt.ylabel("loss")
            plt.title("Gene bridge distill (real processor) losses")
            plt.tight_layout()
            plt.savefig(curve_path, dpi=200)
            print("saved loss curve:", curve_path)
            print("saved log:", log_path)
    finally:
        if f_log is not None:
            f_log.flush()
            f_log.close()
        ddp_barrier()
        if ddp_is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
