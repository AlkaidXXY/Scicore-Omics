#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import time
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import anndata as ad
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoProcessor


# -------------------------
# GeneTokenizer (vocab.json)
# -------------------------
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


# -------------------------
# JSONL Dataset
# -------------------------
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


# -------------------------
# helpers
# -------------------------
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
        # gene span 不做 CE 监督（-100）
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
    prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return prompt_text, full_text


def pool_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    x: [B, L, D]
    mask: [B, L] (1 for keep)
    """
    mask = mask.to(dtype=x.dtype)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (x * mask.unsqueeze(-1)).sum(dim=1) / denom


# -------------------------
# collate
# -------------------------
@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    position_ids: torch.Tensor
    gene_input_ids: torch.Tensor
    gene_attention_mask: torch.Tensor
    gene_bound: List[Optional[torch.Tensor]]

    # teacher versions (no gene injection)
    teacher_input_ids: torch.Tensor
    teacher_attention_mask: torch.Tensor
    teacher_labels: torch.Tensor
    teacher_position_ids: torch.Tensor


def collate_fn(samples: List[Dict[str, Any]], processor, gene_tokenizer: GeneTokenizer,
               gene_span_len: int, gene_max_len: int, device: torch.device) -> Batch:
    tok = processor.tokenizer
    gene_token_id = tok.convert_tokens_to_ids("<gene>")
    unk_id = tok.unk_token_id if tok.unk_token_id is not None else tok.convert_tokens_to_ids("<unk>")

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

        # expand <gene> span (student + teacher 都扩，以便对齐)
        full_ids_s, full_attn_s, labels_s, gene_bound = expand_gene_span(
            full_ids, full_attn, labels, gene_token_id, unk_id, span_len=gene_span_len
        )
        full_ids_t, full_attn_t, labels_t, _ = expand_gene_span(
            full_ids, full_attn, labels, gene_token_id, unk_id, span_len=gene_span_len
        )

        pos_s = torch.arange(full_ids_s.shape[0], dtype=torch.long)
        pos_t = torch.arange(full_ids_t.shape[0], dtype=torch.long)

        # gene_input_ids from h5ad
        if gene_path is None or (not os.path.exists(gene_path)):
            gene_pack = gene_tokenizer([], max_length=gene_max_len, padding=True, truncation=True)
        else:
            adata = ad.read_h5ad(gene_path)
            gene_names = adata.var_names.tolist()
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


# -------------------------
# main train
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--data_jsonl", type=str, required=True)
    ap.add_argument("--gene_vocab", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])

    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    ap.add_argument("--gene_span_len", type=int, default=32)
    ap.add_argument("--gene_max_len", type=int, default=1500)
    ap.add_argument("--log_every", type=int, default=20)

    # loss weights
    ap.add_argument("--lambda_ce", type=float, default=0.2)
    ap.add_argument("--lambda_cos", type=float, default=1.0)
    ap.add_argument("--lambda_nce", type=float, default=1.0)
    ap.add_argument("--temp", type=float, default=0.07)

    # NEW: save at step
    ap.add_argument("--save_step", type=int, default=480, help="save checkpoint when global_step == save_step")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    curve_path = os.path.join(args.out_dir, "loss_curve.png")
    ckpt_path = os.path.join(args.out_dir, "gene_bridge_distill.pt")

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print("Loading model:", args.model_path)
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype
    ).to(device)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # freeze all then unfreeze gene_qformer + gene_projector
    for p in model.parameters():
        p.requires_grad = False
    model.gene_qformer.requires_grad_(True)
    model.gene_projector.requires_grad_(True)

    if hasattr(model, "nicheformer"):
        model.nicheformer.requires_grad_(False)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    dataset = GeneJsonlDataset(args.data_jsonl)
    steps_per_epoch = math.ceil(len(dataset) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs

    def lr_lambda(step):
        warmup = int(0.1 * total_steps)
        if step < warmup:
            return float(step) / max(1, warmup)
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    gene_tokenizer = GeneTokenizer(args.gene_vocab)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda xs: collate_fn(
            xs, processor, gene_tokenizer, args.gene_span_len, args.gene_max_len, device
        )
    )

    loss_hist = []
    ce_hist, cos_hist, nce_hist = [], [], []

    model.train()
    global_step = 0
    saved_step_once = False

    with open(log_path, "w", encoding="utf-8") as f_log:
        for ep in range(args.epochs):
            for batch in loader:
                global_step += 1

                # -----------------------
                # Teacher (text-only)
                # -----------------------
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

                # -----------------------
                # Student (gene injected)
                # -----------------------
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

                inputs_embeds, _ = model.get_vllm_embedding(data)  # [B, L, D]
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

                # gene vec from span
                g_vecs = []
                for i, gb in enumerate(batch.gene_bound):
                    if gb is None:
                        g_vecs.append(torch.zeros((inputs_embeds.shape[-1],), device=device, dtype=inputs_embeds.dtype))
                        continue
                    s, e = int(gb[0, 0].item()), int(gb[0, 1].item())
                    g_tokens = inputs_embeds[i, s:e, :]  # [span, D]
                    g_vecs.append(g_tokens.mean(dim=0))
                g_vec = torch.stack(g_vecs, dim=0)
                g_vec = F.normalize(g_vec, dim=-1)

                loss_cos = (1.0 - (g_vec * t_vec).sum(dim=-1)).mean()

                logits = (g_vec @ t_vec.t()) / args.temp
                targets = torch.arange(logits.shape[0], device=logits.device)
                loss_nce = F.cross_entropy(logits, targets)

                loss = args.lambda_ce * loss_ce + args.lambda_cos * loss_cos + args.lambda_nce * loss_nce

                opt.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                opt.step()
                scheduler.step()

                # -----------------------
                # NEW: save at exact step
                # -----------------------
                if (not saved_step_once) and (global_step == args.save_step):
                    step_ckpt_path = os.path.join(args.out_dir, f"gene_bridge_distill_step{global_step}.pt")
                    torch.save(
                        {
                            "gene_qformer": model.gene_qformer.state_dict(),
                            "gene_projector": model.gene_projector.state_dict(),
                            "args": vars(args),
                            "global_step": global_step,
                            "epoch": ep,
                        },
                        step_ckpt_path
                    )
                    print(f"💾 saved step checkpoint at step={global_step}: {step_ckpt_path}")
                    saved_step_once = True

                loss_hist.append(float(loss.item()))
                ce_hist.append(float(loss_ce.item()))
                cos_hist.append(float(loss_cos.item()))
                nce_hist.append(float(loss_nce.item()))

                if global_step % args.log_every == 0:
                    rec = {
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "epoch": ep,
                        "step": global_step,
                        "loss": float(loss.item()),
                        "loss_ce": float(loss_ce.item()),
                        "loss_cos": float(loss_cos.item()),
                        "loss_nce": float(loss_nce.item()),
                        "grad_norm": float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm),
                        "lr": float(scheduler.get_last_lr()[0]),
                    }
                    print(f"[ep {ep}] step={global_step} "
                          f"loss={rec['loss']:.4f} ce={rec['loss_ce']:.4f} cos={rec['loss_cos']:.4f} nce={rec['loss_nce']:.4f} "
                          f"grad_norm={rec['grad_norm']:.4e} lr={rec['lr']:.2e}")
                    f_log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f_log.flush()

            # epoch save
            torch.save(
                {
                    "gene_qformer": model.gene_qformer.state_dict(),
                    "gene_projector": model.gene_projector.state_dict(),
                    "args": vars(args),
                    "global_step": global_step,
                    "epoch": ep,
                },
                ckpt_path
            )
            print("✅ saved:", ckpt_path)

    # plot
    plt.figure()
    plt.plot(loss_hist, label="total")
    plt.plot(ce_hist, label="ce")
    plt.plot(cos_hist, label="cos")
    plt.plot(nce_hist, label="nce")
    plt.legend()
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Gene bridge distill losses")
    plt.tight_layout()
    plt.savefig(curve_path, dpi=200)
    print("📈 saved loss curve:", curve_path)
    print("🧾 saved log:", log_path)


if __name__ == "__main__":
    main()


'''
export CUDA_VISIBLE_DEVICES=7
python /data2/xiaoxinyu/project/qformer/train_gene_bridge_distill.py \
  --model_path /data2/xiaoxinyu/project/model_cpt_v6_qformer \
  --data_jsonl /data2/xiaoxinyu/project/new_data/gene_data.jsonl \
  --gene_vocab /data2/xiaoxinyu/project/model/gene_tokenizer/vocab.json \
  --out_dir /data2/xiaoxinyu/project/qformer/distill_out_cpt_v6 \
  --epochs 1 \
  --batch_size 16 \
  --lr 1e-4 \
  --lambda_ce 0.2 \
  --lambda_cos 1.0 \
  --lambda_nce 1.0 \
  --temp 0.07 \
  --save_step 100
  
  
python /data2/xiaoxinyu/project/qformer/train_gene_bridge_distill.py \
  --model_path /data2/xiaoxinyu/project/model_merged_v75_qformer \
  --data_jsonl /data2/xiaoxinyu/project/new_data/sft_STimage_gene_only.jsonl \
  --gene_vocab /data2/xiaoxinyu/project/model/gene_tokenizer/vocab.json \
  --out_dir /data2/xiaoxinyu/project/qformer/distill_out_cpt_v1 \
  --epochs 1 \
  --batch_size 8 \
  --lr 1e-4 \
  --lambda_ce 0.2 \
  --lambda_cos 1.0 \
  --lambda_nce 1.0 \
  --temp 0.07 \
  --save_step 400
  
  
'''