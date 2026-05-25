#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DDP Version B (stabilized minimal edits):
Train ONLY gene_qformer + gene_projector using
(1) CE on frozen LLM (grad flows to inputs_embeds -> gene modules)
(2) cosine alignment between gene-span vector and teacher text vector
(3) global InfoNCE with cross-rank all_gather (negatives = world_size * batch)

Key fixes:
- Assert tokenizer has real "<gene>" token (not UNK)
- Teacher is text-only: DO NOT expand gene span; replace <gene> with UNK token
- InfoNCE only all_gather t_vec (teacher), not g_vec
- Mask cosine loss for samples with gene_bound=None (avoid zero-vector pollution)
"""

import os
import json
import math
import time
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import anndata as ad
from transformers import AutoModel, AutoProcessor

import torch.distributed as dist


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


@torch.no_grad()
def all_gather_variable_batch(x: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
    """
    Gather [b, d] across ranks with potentially different b on last batch.
    Return concatenated [sum(b_i), d] and sizes list.
    """
    assert x.dim() == 2, f"expect [b,d], got {x.shape}"
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

    x_all = torch.cat([o[:sizes[i]] for i, o in enumerate(outs)], dim=0)
    return x_all, sizes


def barrier():
    if ddp_is_initialized():
        dist.barrier()


class GeneTokenizer:
    def __init__(self, vocab_file: str):
        if not os.path.exists(vocab_file):
            raise ValueError(f"Vocab file not found: {vocab_file}")
        with open(vocab_file, "r") as f:
            vocab = json.load(f)
        self.gene_to_id = dict(vocab)

        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        if self.pad_token not in self.gene_to_id:
            self.gene_to_id[self.pad_token] = len(self.gene_to_id)
        if self.unk_token not in self.gene_to_id:
            self.gene_to_id[self.unk_token] = len(self.gene_to_id)

        self.pad_token_id = self.gene_to_id[self.pad_token]
        self.unk_token_id = self.gene_to_id[self.unk_token]

    def __call__(self, gene_list: List[str], max_length: int, padding: bool = True, truncation: bool = True):
        ids = [self.gene_to_id.get(g, self.unk_token_id) for g in gene_list]
        if truncation and len(ids) > max_length:
            ids = ids[:max_length]
        if padding and len(ids) < max_length:
            ids = ids + [self.pad_token_id] * (max_length - len(ids))
        attn = [1 if x != self.pad_token_id else 0 for x in ids]
        return {"input_ids": ids, "attention_mask": attn}


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


class GeneVarCache:
    def __init__(self, max_items: int = 256):
        self.max_items = max_items
        self._cache: Dict[str, List[str]] = {}
        self._keys: List[str] = []

    def get(self, path: str) -> Optional[List[str]]:
        return self._cache.get(path, None)

    def put(self, path: str, genes: List[str]):
        if path in self._cache:
            return
        self._cache[path] = genes
        self._keys.append(path)
        if len(self._keys) > self.max_items:
            old = self._keys.pop(0)
            self._cache.pop(old, None)


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
    Replace one <gene> token with span_len <unk> tokens.
    Return new tensors and gene_bound [[start, start+span_len]] or None.
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
    prompt_msgs = [messages[0]]
    prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return prompt_text, full_text


def pool_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (x * mask.unsqueeze(-1)).sum(dim=1) / denom


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


def collate_fn(
    samples: List[Dict[str, Any]],
    processor,
    gene_tokenizer: GeneTokenizer,
    gene_cache: GeneVarCache,
    gene_span_len: int,
    gene_max_len: int,
    device: torch.device,
    max_seq_len: int
) -> Batch:
    tok = processor.tokenizer

    gene_token_id = tok.convert_tokens_to_ids("<gene>")
    unk_id = tok.unk_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    # ---- FIX 1: hard assert real <gene> token exists ----
    if gene_token_id is None or unk_id is None or gene_token_id == unk_id:
        raise RuntimeError(
            f'Bad token ids: gene_token_id={gene_token_id}, unk_id={unk_id}. '
            'Tokenizer must contain a real "<gene>" token (not UNK).'
        )

    input_ids_list, attn_list, labels_list, pos_list, gene_bound_list = [], [], [], [], []
    teacher_ids_list, teacher_attn_list, teacher_labels_list, teacher_pos_list = [], [], [], []
    gene_ids_list, gene_attn_list = [], []

    for s in samples:
        messages = s["messages"]
        gene_path = s.get("gene", None)

        prompt_text, full_text = build_prompt_and_full_text(tok, messages)

        prompt_ids = tok(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        full_enc = tok(full_text, return_tensors="pt", add_special_tokens=False)
        full_ids = full_enc["input_ids"][0]
        full_attn = full_enc["attention_mask"][0]

        assist_start = prompt_ids.shape[0]

        labels = full_ids.clone()
        labels[:assist_start] = -100

        # ---- Student: expand span ----
        full_ids_s, full_attn_s, labels_s, gene_bound = expand_gene_span(
            full_ids, full_attn, labels, gene_token_id, unk_id, span_len=gene_span_len
        )

        # ---- FIX 2: Teacher text-only, DO NOT expand span; replace <gene> with UNK ----
        full_ids_t = full_ids
        full_attn_t = full_attn
        labels_t = labels
        gene_pos = (full_ids_t == gene_token_id).nonzero(as_tuple=False)
        if gene_pos.numel() > 0:
            full_ids_t = full_ids_t.clone()
            full_ids_t[gene_pos[:, 0]] = unk_id

        def trunc(x: torch.Tensor, L: int):
            return x[:L] if x.shape[0] > L else x

        full_ids_s = trunc(full_ids_s, max_seq_len)
        full_attn_s = trunc(full_attn_s, max_seq_len)
        labels_s = trunc(labels_s, max_seq_len)

        full_ids_t = trunc(full_ids_t, max_seq_len)
        full_attn_t = trunc(full_attn_t, max_seq_len)
        labels_t = trunc(labels_t, max_seq_len)

        if gene_bound is not None:
            s0, e0 = int(gene_bound[0, 0].item()), int(gene_bound[0, 1].item())
            if e0 > full_ids_s.shape[0]:
                gene_bound = None

        pos_s = torch.arange(full_ids_s.shape[0], dtype=torch.long)
        pos_t = torch.arange(full_ids_t.shape[0], dtype=torch.long)

        if gene_path is None or (not os.path.exists(gene_path)):
            gene_names = []
        else:
            cached = gene_cache.get(gene_path)
            if cached is None:
                try:
                    adata = ad.read_h5ad(gene_path, backed="r")
                    gene_names = adata.var_names.tolist()
                    try:
                        adata.file.close()
                    except Exception:
                        pass
                except Exception:
                    gene_names = []
                gene_cache.put(gene_path, gene_names)
            else:
                gene_names = cached

        gene_pack = gene_tokenizer(gene_names, max_length=gene_max_len, padding=True, truncation=True)
        gene_ids = torch.tensor(gene_pack["input_ids"], dtype=torch.long)
        gene_attn = torch.tensor(gene_pack["attention_mask"], dtype=torch.long)

        input_ids_list.append(full_ids_s)
        attn_list.append(full_attn_s)
        labels_list.append(labels_s)
        pos_list.append(pos_s)
        gene_bound_list.append(gene_bound)

        teacher_ids_list.append(full_ids_t)
        teacher_attn_list.append(full_attn_t)
        teacher_labels_list.append(labels_t)
        teacher_pos_list.append(pos_t)

        gene_ids_list.append(gene_ids)
        gene_attn_list.append(gene_attn)

    def pad_1d(xs: List[torch.Tensor], pad_val: int):
        max_len = max(x.shape[0] for x in xs)
        out = []
        for x in xs:
            if x.shape[0] < max_len:
                out.append(F.pad(x, (0, max_len - x.shape[0]), value=pad_val))
            else:
                out.append(x)
        return torch.stack(out, dim=0)

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
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--data_jsonl", type=str, required=True)
    ap.add_argument("--gene_vocab", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    ap.add_argument("--gene_span_len", type=int, default=32)
    ap.add_argument("--gene_max_len", type=int, default=1500)
    ap.add_argument("--max_seq_len", type=int, default=1024)

    ap.add_argument("--log_every", type=int, default=200)

    ap.add_argument("--lambda_ce", type=float, default=0.1)
    ap.add_argument("--lambda_cos", type=float, default=0.5)
    ap.add_argument("--lambda_nce", type=float, default=1.0)
    ap.add_argument("--temp", type=float, default=0.2)

    ap.add_argument("--save_every_epoch", action="store_true")
    ap.add_argument("--save_every_steps", type=int, default=0)
    ap.add_argument("--cache_h5ad", type=int, default=256)

    args = ap.parse_args()

    ddp_init()
    rank = ddp_rank()
    world = ddp_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_master = (rank == 0)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    if is_master:
        os.makedirs(args.out_dir, exist_ok=True)

    barrier()

    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    ckpt_path = os.path.join(args.out_dir, "gene_bridge_distill.pt")

    if is_master:
        print(f"[DDP] world_size={world}, rank={rank}, local_rank={local_rank}, device={device}")
        print("Loading model:", args.model_path)

    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype
    ).to(device)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    for p in model.parameters():
        p.requires_grad = False
    if not hasattr(model, "gene_qformer") or not hasattr(model, "gene_projector"):
        raise RuntimeError("Model does not have gene_qformer / gene_projector.")
    model.gene_qformer.requires_grad_(True)
    model.gene_projector.requires_grad_(True)
    if hasattr(model, "nicheformer"):
        model.nicheformer.requires_grad_(False)

    if ddp_is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False
        )

    m = model.module if hasattr(model, "module") else model

    params = [p for p in m.parameters() if p.requires_grad]
    if is_master:
        print(f"Trainable params: {sum(p.numel() for p in params)/1e6:.2f}M")
        for n, p in m.named_parameters():
            if p.requires_grad:
                print("  ", n, p.shape, p.dtype)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    dataset = GeneJsonlDataset(args.data_jsonl)
    if ddp_is_initialized():
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=False)
    else:
        sampler = None

    gene_tokenizer = GeneTokenizer(args.gene_vocab)
    gene_cache = GeneVarCache(max_items=args.cache_h5ad)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda xs: collate_fn(
            xs,
            processor,
            gene_tokenizer,
            gene_cache,
            args.gene_span_len,
            args.gene_max_len,
            device,
            args.max_seq_len
        )
    )

    steps_per_epoch = math.ceil(len(dataset) / (args.batch_size * world))
    total_steps = max(1, steps_per_epoch * args.epochs)

    def lr_lambda(step):
        warmup = int(0.1 * total_steps)
        if step < warmup:
            return float(step) / max(1, warmup)
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    f_log = open(log_path, "w", encoding="utf-8") if is_master else None

    m.train()
    global_step = 0

    try:
        for ep in range(args.epochs):
            if sampler is not None:
                sampler.set_epoch(ep)

            for batch in loader:
                global_step += 1

                # Teacher (text-only)
                with torch.no_grad():
                    out_t = m.llm(
                        input_ids=batch.teacher_input_ids,
                        attention_mask=batch.teacher_attention_mask,
                        position_ids=batch.teacher_position_ids,
                        output_hidden_states=True,
                        use_cache=False,
                    )
                    t_hidden = out_t.hidden_states[-1]  # [B, L, D]
                    t_mask = (batch.teacher_labels != -100).long()
                    t_vec_local = pool_mean(t_hidden, t_mask)  # [B, D]
                    t_vec_local = F.normalize(t_vec_local, dim=-1)

                # Student inputs_embeds with gene injection
                data = {
                    "input_ids": batch.input_ids,
                    "attention_mask": batch.attention_mask,
                    "position_ids": batch.position_ids,
                    "pixel_values": [[] for _ in range(batch.input_ids.shape[0])],
                    "tgt_sizes": [None] * batch.input_ids.shape[0],
                    "image_bound": [[] for _ in range(batch.input_ids.shape[0])],
                    "gene_input_ids": batch.gene_input_ids,
                    "gene_attention_mask": batch.gene_attention_mask,
                    "gene_bound": batch.gene_bound,
                }

                inputs_embeds, _ = m.get_vllm_embedding(data)  # [B, L, D]

                out_s = m.llm(
                    input_ids=None,
                    inputs_embeds=inputs_embeds,
                    attention_mask=batch.attention_mask,
                    position_ids=batch.position_ids,
                    labels=batch.labels,
                    output_hidden_states=False,
                    use_cache=False,
                )
                loss_ce = out_s.loss

                # gene vector from <gene> span embeddings
                g_vecs = []
                valid_mask = []
                for i, gb in enumerate(batch.gene_bound):
                    if gb is None:
                        # placeholder; will be masked out for cosine
                        g_vecs.append(torch.zeros((inputs_embeds.shape[-1],), device=device, dtype=inputs_embeds.dtype))
                        valid_mask.append(False)
                        continue
                    s0, e0 = int(gb[0, 0].item()), int(gb[0, 1].item())
                    g_tokens = inputs_embeds[i, s0:e0, :]  # [span, D]
                    g_vecs.append(g_tokens.mean(dim=0))
                    valid_mask.append(True)

                g_vec_local = torch.stack(g_vecs, dim=0)  # [B, D]
                g_vec_local = F.normalize(g_vec_local, dim=-1)

                # ---- FIX 4: cosine mask ----
                valid = torch.tensor(valid_mask, device=device)
                if valid.any():
                    loss_cos = (1.0 - (g_vec_local[valid] * t_vec_local[valid]).sum(dim=-1)).mean()
                else:
                    loss_cos = torch.tensor(0.0, device=device)

                # ---- FIX 3: Global InfoNCE only gather teacher vectors ----
                # (teacher is no_grad; gather is cheap and stable)
                t_all, t_sizes = all_gather_variable_batch(t_vec_local)
                offset = sum(t_sizes[:rank])
                b_local = g_vec_local.shape[0]
                logits = (g_vec_local @ t_all.t()) / max(1e-6, args.temp)  # [b_local, B_global]
                targets = torch.arange(b_local, device=device) + offset
                loss_nce = F.cross_entropy(logits, targets)

                loss = args.lambda_ce * loss_ce + args.lambda_cos * loss_cos + args.lambda_nce * loss_nce

                opt.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                opt.step()
                scheduler.step()

                if is_master and (global_step % args.log_every == 0 or global_step == 1):
                    rec = {
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "epoch": ep,
                        "step": global_step,
                        "loss": float(loss.item()),
                        "loss_ce": float(loss_ce.item()),
                        "loss_cos": float(loss_cos.item()) if torch.is_tensor(loss_cos) else float(loss_cos),
                        "loss_nce": float(loss_nce.item()),
                        "grad_norm": float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm),
                        "lr": float(scheduler.get_last_lr()[0]),
                        "b_local": int(b_local),
                        "b_global": int(t_all.shape[0]),
                        "nce_baseline_logBg": float(math.log(max(2, int(t_all.shape[0])))),
                        "valid_cos_frac": float(valid.float().mean().item()),
                    }
                    print(
                        f"[ep {ep}] step={global_step} "
                        f"loss={rec['loss']:.4f} ce={rec['loss_ce']:.4f} cos={rec['loss_cos']:.4f} nce={rec['loss_nce']:.4f} "
                        f"gn={rec['grad_norm']:.3e} lr={rec['lr']:.2e} "
                        f"Bg={rec['b_global']} logBg={rec['nce_baseline_logBg']:.2f} "
                        f"cos_valid={rec['valid_cos_frac']:.2f}"
                    )
                    f_log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if global_step % (args.log_every * 10) == 0:
                        f_log.flush()

                if is_master and args.save_every_steps and (global_step % args.save_every_steps == 0):
                    torch.save(
                        {
                            "gene_qformer": (m.gene_qformer.state_dict()),
                            "gene_projector": (m.gene_projector.state_dict()),
                            "args": vars(args),
                            "step": global_step,
                        },
                        ckpt_path
                    )
                    print("✅ saved step ckpt:", ckpt_path)

            if is_master and args.save_every_epoch:
                torch.save(
                    {
                        "gene_qformer": (m.gene_qformer.state_dict()),
                        "gene_projector": (m.gene_projector.state_dict()),
                        "args": vars(args),
                        "epoch": ep,
                    },
                    ckpt_path
                )
                print("✅ saved epoch ckpt:", ckpt_path)

    finally:
        if f_log is not None:
            try:
                f_log.flush()
                f_log.close()
            except Exception:
                pass
        barrier()
        if ddp_is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

'''
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,7
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1

torchrun --nproc_per_node 6 \
  /data2/xiaoxinyu/project/qformer/train_gene_bridge_distill_ddp.py \
  --model_path /data2/xiaoxinyu/project/model_cpt_v3_qformer \
  --data_jsonl /data2/xiaoxinyu/project/new_data/sft_STimage_gene_only.jsonl \
  --gene_vocab /data2/xiaoxinyu/project/model/gene_tokenizer/vocab.json \
  --out_dir /data2/xiaoxinyu/project/qformer/distill_out_ddp \
  --epochs 1 \
  --batch_size 8 \
  --lr 1e-4 \
  --lambda_ce 0.1 \
  --lambda_cos 0.5 \
  --lambda_nce 1.0 \
  --temp 0.2 \
  --log_every 200
'''
