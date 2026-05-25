#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


LABEL_HINT = (
    "可选标签：分子层（Layer 1）、外颗粒层（Layer 2）、外锥体层（Layer 3）、"
    "内颗粒层（Layer 4）、内锥体层（Layer 5）、多形层（Layer 6）、白质层（WM）。"
)
USER_PROMPT = "请描述样本信息?"
LABELS = ["Layer_1", "Layer_2", "Layer_3", "Layer_4", "Layer_5", "Layer_6", "WM"]
LABEL_DESCRIPTIONS = {
    "Layer_1": "该样本属于分子层（Layer 1）。该层是皮层的最外层，主要由神经胶质细胞和少量神经元组成，功能主要涉及信息传递的调节。",
    "Layer_2": "该样本属于外颗粒层（Layer 2）。该层由小型颗粒神经元组成，主要参与局部信息处理。",
    "Layer_3": "该样本属于外锥体层（Layer 3）。该层含有较多锥体神经元，主要参与皮层间长距离通讯。",
    "Layer_4": "该样本属于内颗粒层（Layer 4）。该层主要接受来自丘脑的感觉信息输入。",
    "Layer_5": "该样本属于内锥体层（Layer 5）。该层负责将信息传递至其他脑区。",
    "Layer_6": "该样本属于多形层（Layer 6）。该层主要将信息反馈给丘脑。",
    "WM": "该样本属于白质层（WM）。该区域主要由神经纤维组成，承担不同脑区之间的信息传导。",
}
_ANSWER_LABEL_PAT = re.compile(
    r"(分子层|外颗粒层|外锥体层|内颗粒层|内锥体层|多形层|白质层|白质)"
    r"|(?:Layer|layer|L)\s*[_ -]?\s*([1-6])"
    r"|\b(WM|white\s+matter)\b",
    re.IGNORECASE,
)
_CN_TO_LABEL = {
    "分子层": "Layer_1",
    "外颗粒层": "Layer_2",
    "外锥体层": "Layer_3",
    "内颗粒层": "Layer_4",
    "内锥体层": "Layer_5",
    "多形层": "Layer_6",
    "白质层": "WM",
    "白质": "WM",
}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def first_text_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                return str(item.get("text", ""))
    return ""


def first_gene_path(content) -> str:
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "gene":
                genes = item.get("genes") or []
                if genes:
                    return str(genes[0])
    return ""


def example_answer(obj: Dict) -> str:
    for msg in obj.get("messages", []):
        if msg.get("role") == "assistant":
            return first_text_content(msg.get("content", "")).strip()
    return ""


def extract_answer_label(answer: str) -> str:
    m = _ANSWER_LABEL_PAT.search(answer)
    if not m:
        return ""
    cn_name, layer_num, wm_name = m.group(1), m.group(2), m.group(3)
    if cn_name:
        return _CN_TO_LABEL.get(cn_name, "")
    if layer_num:
        return f"Layer_{layer_num}"
    if wm_name:
        return "WM"
    return ""


def format_answer(answer: str, answer_style: str) -> str:
    if answer_style == "label":
        return answer
    label = extract_answer_label(answer)
    if answer_style == "full_description" and label in LABEL_DESCRIPTIONS:
        return LABEL_DESCRIPTIONS[label]
    return answer


def example_gene_path(obj: Dict) -> str:
    for msg in obj.get("messages", []):
        if msg.get("role") == "user":
            return first_gene_path(msg.get("content", []))
    return ""


def cell_sentence_from_h5ad(path: str, max_genes: int) -> str:
    a = ad.read_h5ad(path)
    row = a.X[0]
    if sp.issparse(row):
        row = row.toarray().reshape(-1)
    else:
        row = np.asarray(row).reshape(-1)

    nz = np.flatnonzero(row > 0)
    if len(nz) == 0:
        return ""
    order = nz[np.argsort(row[nz])[::-1]]
    if max_genes > 0:
        order = order[:max_genes]
    genes = [str(a.var_names[i]).upper() for i in order]
    return " ".join(genes)


def build_prompt(cell_sentence: str) -> str:
    return (
        "基因表达序列（按表达量从高到低）：\n"
        f"{cell_sentence}\n\n"
        f"{USER_PROMPT}\n"
        "答案："
    )


def build_cache(data_path: Path, cache_path: Path, max_genes: int, overwrite: bool, answer_style: str) -> Path:
    if cache_path.exists() and not overwrite:
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with cache_path.open("w", encoding="utf-8") as out:
        for obj in tqdm(list(read_jsonl(data_path)), desc="building C2S cache"):
            gene_path = example_gene_path(obj)
            answer = example_answer(obj)
            if not gene_path or not answer:
                continue
            answer = format_answer(answer, answer_style)
            sentence = cell_sentence_from_h5ad(gene_path, max_genes=max_genes)
            if not sentence:
                continue
            row = {
                "id": obj.get("id", ""),
                "prompt": build_prompt(sentence),
                "answer": answer,
                "gene_path": gene_path,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[INFO] wrote {n_written} cached examples to {cache_path}", flush=True)
    return cache_path


class C2SDataset(Dataset):
    def __init__(self, cache_path: Path, tokenizer, max_length: int):
        self.rows = list(read_jsonl(cache_path))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        prompt_ids = self.tokenizer(row["prompt"], add_special_tokens=False).input_ids
        target = row["answer"].strip() + self.tokenizer.eos_token
        target_ids = self.tokenizer(target, add_special_tokens=False).input_ids

        budget = self.max_length - len(target_ids)
        if budget < 1:
            target_ids = target_ids[: self.max_length]
            prompt_ids = []
        elif len(prompt_ids) > budget:
            prompt_ids = prompt_ids[:budget]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        return {"input_ids": input_ids, "labels": labels}


@dataclass
class DataCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict]):
        max_len = max(len(x["input_ids"]) for x in features)
        input_ids, labels, attention_mask = [], [], []
        for feat in features:
            n = len(feat["input_ids"])
            pad = max_len - n
            input_ids.append(feat["input_ids"] + [self.pad_token_id] * pad)
            labels.append(feat["labels"] + [-100] * pad)
            attention_mask.append([1] * n + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data1/xiaoxinyu/SOTAModel/C2S")
    parser.add_argument("--dataset", default="/data1/xiaoxinyu/project/data/format_dataset/DLPFC_tri_QA_balanced_train_v5.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_path", default="/data2/xiaoxinyu/project/pretrain-gene/cache/DLPFC_tri_QA_balanced_train_v5.c2s.jsonl")
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--max_genes", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_train_epochs", type=float, default=6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--save_steps", type=int, default=117)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--answer_style", choices=["label", "full_description"], default="label")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    cache_path = build_cache(
        data_path=Path(args.dataset),
        cache_path=Path(args.cache_path),
        max_genes=args.max_genes,
        overwrite=args.overwrite_cache,
        answer_style=args.answer_style,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    train_dataset = C2SDataset(cache_path, tokenizer, args.max_length)
    print(f"[INFO] train examples: {len(train_dataset)}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    collator = DataCollator(tokenizer.pad_token_id)
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    updates_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    total_steps = int(math.ceil(args.num_train_epochs) * updates_per_epoch)
    warmup_steps = int(total_steps * args.warmup_ratio)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    global_step = 0
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    epochs = int(math.ceil(args.num_train_epochs))
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"epoch {epoch + 1}/{epochs}", dynamic_ncols=True)
        for step, batch in enumerate(pbar, start=1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                loss = model(**batch).loss
                loss_to_backprop = loss / args.gradient_accumulation_steps
            loss_to_backprop.backward()
            running_loss += float(loss.detach().cpu())

            should_step = step % args.gradient_accumulation_steps == 0 or step == len(dataloader)
            if not should_step:
                continue

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % args.logging_steps == 0:
                avg_loss = running_loss / max(1, args.logging_steps * args.gradient_accumulation_steps)
                lr = scheduler.get_last_lr()[0]
                print(f"[train] step={global_step} loss={avg_loss:.6f} lr={lr:.8g}", flush=True)
                running_loss = 0.0

            if global_step % args.save_steps == 0:
                ckpt_dir = output_dir / f"checkpoint-{global_step}"
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                print(f"[INFO] saved checkpoint: {ckpt_dir}", flush=True)

            pbar.set_postfix({"step": global_step, "loss": f"{float(loss.detach().cpu()):.4f}"})

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[INFO] saved final adapter: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
