#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
固定 output 上传版 Worker（兼容有 gene / 无 gene）

核心思路：
1. 从 example.json 中采样样本
2. 用 dataset.py 的 GSPODataset + gspo_data_collator 构造 candidate
3. 如果存在 gene 且 gene_bound 是单 token 占位，则扩成 32-token span
4. 不在线 generate；直接使用数据集中已有的 output（labels != -100 对应部分）
5. 计算这条固定 output 的 token-level old-policy log probs
6. reward 默认来自预处理好的 scores；可选叠加 structured reward
7. 上传到 ref_server
8. 打印 PROMPT / FIXED ANSWER / reward 调试信息
"""

import os
import sys
import json
import time
import random
import traceback
from functools import partial
from typing import Dict, List, Any
import re
import requests
import torch
import torch.multiprocessing as mp
from torchvision import transforms
import transformers
from transformers import AutoTokenizer, AutoModel

from dataset import GSPODataset, gspo_data_collator
from gene_tokenizer import GeneTokenizer
from ref_server import tensor_to_bytes, make_bytes_list


GENE_SPAN_LEN = 32


def gen_worker(
    Q,
    model_path: str,
    example_json: str,
    ref_port: int,
    physics_device: int,
    q_batch_size: int = 1,
    max_new_tokens: int = 96,   # 为兼容旧启动参数保留，但不再使用
    temperature: float = 0.7,   # 为兼容旧启动参数保留，但不再使用
    top_p: float = 0.9,         # 为兼容旧启动参数保留，但不再使用
    max_slice_nums: int = 1,
    use_structured_reward: bool = False,
):
    ref_server = f"http://127.0.0.1:{ref_port}"

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{physics_device}"

    cleanup_keys = [
        "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK",
        "NCCL_COMM_ID", "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    ]
    for key in cleanup_keys:
        os.environ.pop(key, None)

    sys.path.append(model_path)

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    print(f"[GEN_WORKER] using physical GPU {physics_device}", flush=True)

    # 仍然加载 model，但只用于计算固定 output 的 old-policy log probs
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # GeneTokenizer 仅在有 gene 时才真正会用到
    gene_vocab_file = os.environ.get(
        "GENE_VOCAB_FILE",
        "/data2/xiaoxinyu/project/model/gene_tokenizer/vocab.json"
    )
    gene_tokenizer = GeneTokenizer(vocab_file=gene_vocab_file)

    if hasattr(model.config, "slice_config"):
        model.config.slice_config.max_slice_nums = max_slice_nums
        slice_config = model.config.slice_config.to_dict()
    else:
        model.config.max_slice_nums = max_slice_nums
        slice_config = model.config.to_dict()

    batch_vision = bool(getattr(model.config, "batch_vision_input", False))
    patch_size = int(getattr(model.config, "patch_size", 14))
    query_num = int(getattr(model.config, "query_num", 64))

    with open(example_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"[GEN_WORKER] loaded {len(dataset)} raw examples", flush=True)

    def has_gene_sample(sample):
        return ("gene" in sample) and (sample["gene"] is not None)

    gene_capable = sum(1 for x in dataset if has_gene_sample(x))
    print(f"[GEN_WORKER] gene-capable examples = {gene_capable} / {len(dataset)}", flush=True)

    if len(dataset) == 0:
        raise RuntimeError("[GEN_WORKER] example_json is empty")

    def move_nested_to_device(data):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        if isinstance(data, list):
            return [move_nested_to_device(x) for x in data]
        if isinstance(data, tuple):
            return tuple(move_nested_to_device(x) for x in data)
        return data

    def build_transform():
        IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
        IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_INCEPTION_MEAN,
                std=IMAGENET_INCEPTION_STD
            ),
        ])

    def convert_tensors_to_list(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, (list, tuple)):
            return [convert_tensors_to_list(item) for item in obj]
        return obj

    def safe_jsonable_bound(bound):
        if isinstance(bound, torch.Tensor):
            return bound.tolist()
        if isinstance(bound, (list, tuple)):
            return convert_tensors_to_list(bound)
        return bound

    def make_supervised_data_module(
        tokenizer: transformers.PreTrainedTokenizer,
        gene_tokenizer: GeneTokenizer,
        gspo_sample,
        transform,
        data_collator=None,
        llm_type="minicpm",
        slice_config=None,
        patch_size=8,
        query_nums=64,
        batch_vision=False,
        max_length=2048,
    ) -> Dict:
        train_dataset = GSPODataset(
            raw_data=gspo_sample,
            transform=transform,
            tokenizer=tokenizer,
            gene_tokenizer=gene_tokenizer,
            slice_config=slice_config,
            llm_type=llm_type,
            patch_size=patch_size,
            query_nums=query_nums,
            batch_vision=batch_vision,
            max_length=max_length,
        )
        return dict(
            train_dataset=train_dataset,
            data_collator=partial(gspo_data_collator, max_length=max_length),
        )

    def try_update_model():
        try:
            new_state_dict = Q.get_nowait()
            print("[GEN_WORKER] receiving new model...", flush=True)
            model.load_state_dict(new_state_dict, strict=False)
            print("[GEN_WORKER] model updated!", flush=True)
            del new_state_dict
            torch.cuda.empty_cache()
        except Exception:
            pass

    def prepare_one_candidate(batch: Dict[str, Any], i: int) -> Dict[str, Any]:
        one = {
            "input_ids": batch["input_ids"][i].clone(),
            "position_ids": batch["position_ids"][i].clone(),
            "labels": batch["labels"][i].clone(),
            "attention_mask": batch["attention_mask"][i].clone(),
            "pixel_values": batch["pixel_values"][i],
            "image_bound": batch["image_bound"][i],
            "tgt_sizes": batch["tgt_sizes"][i],
            "gene_input_ids": (
                batch["gene_input_ids"][i:i+1].clone()
                if batch.get("gene_input_ids", None) is not None else None
            ),
            "gene_attention_mask": (
                batch["gene_attention_mask"][i:i+1].clone()
                if batch.get("gene_attention_mask", None) is not None else None
            ),
            "gene_bound": batch["gene_bound"][i] if batch.get("gene_bound", None) is not None else [],
        }
        return one

    def normalize_bound(bound) -> List[List[int]]:
        if bound is None:
            return []
        if isinstance(bound, torch.Tensor):
            if bound.numel() == 0:
                return []
            return bound.detach().cpu().tolist()
        if isinstance(bound, list):
            if len(bound) == 0:
                return []
            if isinstance(bound[0], int):
                return [bound]
            return bound
        return []

    def shift_bounds_after_insert(bounds, insert_pos: int, delta: int):
        norm = normalize_bound(bounds)
        shifted = []
        for r in norm:
            if len(r) != 2:
                continue
            s, e = int(r[0]), int(r[1])
            if s >= insert_pos:
                shifted.append([s + delta, e + delta])
            elif e > insert_pos:
                shifted.append([s, e + delta])
            else:
                shifted.append([s, e])
        return shifted

    def expand_gene_placeholder(one_sample: Dict[str, Any], span_len: int = GENE_SPAN_LEN) -> Dict[str, Any]:
        """
        仅当存在 gene 且 gene_bound 是单 token 占位时，扩成 32-token span。
        无 gene 样本直接原样返回。
        """
        if one_sample.get("gene_input_ids", None) is None:
            return one_sample

        gene_bound = normalize_bound(one_sample.get("gene_bound", []))
        if len(gene_bound) == 0:
            return one_sample

        start, end = gene_bound[0]
        if end - start == span_len:
            return one_sample

        if end - start != 1:
            raise ValueError(f"[GEN_WORKER] unexpected gene span: {gene_bound}")

        unk_id = tokenizer.unk_token_id
        if unk_id is None:
            raise ValueError("[GEN_WORKER] tokenizer.unk_token_id is None")

        input_ids = one_sample["input_ids"].clone()
        position_ids = one_sample["position_ids"].clone()
        labels = one_sample["labels"].clone()
        attention_mask = one_sample["attention_mask"].clone()

        old_label = labels[start].item()
        old_attn = attention_mask[start].item()

        left_ids = input_ids[:start]
        right_ids = input_ids[end:]
        left_labels = labels[:start]
        right_labels = labels[end:]
        left_attn = attention_mask[:start]
        right_attn = attention_mask[end:]

        new_gene_ids = torch.full((span_len,), unk_id, dtype=input_ids.dtype)
        new_gene_labels = torch.full((span_len,), old_label, dtype=labels.dtype)
        new_gene_attn = torch.full((span_len,), old_attn, dtype=attention_mask.dtype)

        new_input_ids = torch.cat([left_ids, new_gene_ids, right_ids], dim=0)
        new_labels = torch.cat([left_labels, new_gene_labels, right_labels], dim=0)
        new_attention_mask = torch.cat([left_attn, new_gene_attn, right_attn], dim=0)
        new_position_ids = torch.arange(new_input_ids.shape[0], dtype=position_ids.dtype)

        delta = span_len - (end - start)
        new_image_bound = shift_bounds_after_insert(one_sample.get("image_bound", []), start, delta)
        new_gene_bound = [[start, start + span_len]]

        out = dict(one_sample)
        out["input_ids"] = new_input_ids
        out["labels"] = new_labels
        out["attention_mask"] = new_attention_mask
        out["position_ids"] = new_position_ids
        out["image_bound"] = new_image_bound
        out["gene_bound"] = new_gene_bound
        return out

    def extract_prompt_and_answer_text(one_sample: Dict[str, Any]) -> Dict[str, Any]:
        labels = one_sample["labels"]
        input_ids = one_sample["input_ids"]

        prompt_mask = (labels == -100)
        plen = int(prompt_mask.sum().item())

        prompt_ids = input_ids[:plen]
        valid = torch.where(labels != -100)[0]

        answer_ids = input_ids[valid].detach().cpu() if len(valid) > 0 else torch.tensor([], dtype=input_ids.dtype)

        prompt_text = tokenizer.decode(
            prompt_ids.detach().cpu(),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        answer_text = tokenizer.decode(
            answer_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip() if answer_ids.numel() > 0 else ""

        # print("[DEBUG] total_len =", labels.numel(), flush=True)
        # print("[DEBUG] n_valid =", len(valid), flush=True)
        # print("[DEBUG] plen_by_sum =", int((labels == -100).sum().item()), flush=True)
        # print("[DEBUG] first_valid =", int(valid[0]) if len(valid) > 0 else -1, flush=True)

        return {
            "plen": plen,
            "prompt_text": prompt_text,
            "answer_text": answer_text,
            "answer_ids": answer_ids,
        }

    def compute_fixed_output_token_logps(full_sample: Dict[str, Any]) -> List[float]:
        batch_inputs = {
            "input_ids": full_sample["input_ids"].unsqueeze(0).to(device),
            "position_ids": full_sample["position_ids"].unsqueeze(0).to(device),
            "attention_mask": full_sample["attention_mask"].unsqueeze(0).to(device),
            "pixel_values": move_nested_to_device(
                [full_sample["pixel_values"]] if isinstance(full_sample["pixel_values"], list) and len(full_sample["pixel_values"]) > 0 else [[]]
            ),
            "image_bound": [safe_jsonable_bound(full_sample["image_bound"])] if len(normalize_bound(full_sample["image_bound"])) > 0 else [[]],
            "tgt_sizes": move_nested_to_device(
                full_sample["tgt_sizes"] if isinstance(full_sample["tgt_sizes"], list) else [full_sample["tgt_sizes"]]
            ) if len(full_sample["tgt_sizes"]) > 0 else [],
            "gene_input_ids": full_sample["gene_input_ids"].to(device) if full_sample["gene_input_ids"] is not None else None,
            "gene_attention_mask": full_sample["gene_attention_mask"].to(device) if full_sample["gene_attention_mask"] is not None else None,
            "gene_bound": [safe_jsonable_bound(full_sample["gene_bound"])] if full_sample["gene_input_ids"] is not None else [[]],
        }

        labels = full_sample["labels"].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(data=batch_inputs, use_cache=False)
            logits = outputs.logits

        log_probs = torch.log_softmax(logits, dim=-1)
        per_token_logps = torch.gather(
            log_probs,
            dim=2,
            index=labels.unsqueeze(2).clamp(min=0).long()
        ).squeeze(2)

        valid = torch.where(labels[0] != -100)[0]
        if len(valid) == 0:
            return []

        return per_token_logps[0][valid].detach().cpu().tolist()

    def compute_reward_from_scores(batch, candidate_idx):
        import torch.nn.functional as F

        scores = batch["scores"]
        candidate_counts = batch["candidate_counts"]

        cumsum = 0
        question_idx = 0
        for q_idx, count in enumerate(candidate_counts):
            if candidate_idx < cumsum + count:
                question_idx = q_idx
                break
            cumsum += count

        start_idx = sum(candidate_counts[:question_idx])
        end_idx = start_idx + candidate_counts[question_idx]
        question_scores = scores[start_idx:end_idx]

        normalized_scores = F.softmax(question_scores, dim=0)
        local_idx = candidate_idx - start_idx
        return normalized_scores[local_idx].item()

    def extract_target_region(ref_text: str) -> str:
        text = (ref_text or '').lower()
        if '白质层' in ref_text or '（wm）' in ref_text or '(wm)' in text or ' wm' in text:
            return 'wm'
        m = re.search(r'layer[_\s-]?(\d+)', text)
        if m:
            return f"layer_{m.group(1)}"
        m = re.search(r'来自大脑的\s*(layer[_\s-]?\d+)', text)
        if m:
            return m.group(1).replace(' ', '_').replace('-', '_')
        return ''

    def reward_region(pred_text: str, ref_text: str) -> float:
        pred = (pred_text or '').lower()
        region = extract_target_region(ref_text)
        if not region:
            return 0.0
        if region == 'wm':
            keys = ['wm', '白质', '白质层']
            return 1.0 if any(k in pred_text or k in pred for k in keys) else 0.0
        region_keys = [region, region.replace('_', ' '), region.replace('_', '-')]
        if any(k in pred for k in region_keys):
            return 1.0
        return 0.0

    def reward_gene(pred_text: str, ref_text: str) -> float:
        ref_genes = re.findall(r'\b[A-Z0-9-]{3,}\b', ref_text or '')
        ref_genes = [g for g in ref_genes if ('MT-' in g) or g in {'NPY', 'SCGB22', 'SCGB2A2', 'SCGB2B2'}]
        if not ref_genes:
            return 0.0
        pred_upper = (pred_text or '').upper()
        hit = sum(1 for g in set(ref_genes) if g in pred_upper)
        return min(1.0, hit / max(3, len(set(ref_genes))))

    def reward_pathway(pred_text: str, ref_text: str) -> float:
        pathways = ['核糖体', '氧化磷酸化', '蛋白酶体']
        pred = pred_text or ''
        hit = sum(1 for p in pathways if p in pred)
        return hit / len(pathways)

    def penalty_fallback(pred_text: str) -> float:
        bad_patterns = [
            '请上传', '重新发送图片', '上传或描述', '需要看到', '为了准确描述',
            '临床背景', '无法判断', '组织学图像', '显微图片', '上传图片按钮'
        ]
        pred = pred_text or ''
        return -1.0 if any(p in pred for p in bad_patterns) else 0.0

    def penalty_pathology_template(pred_text: str) -> float:
        bad_patterns = [
            '核异型', '腺癌', '病理类型', '染色方法', '巨噬细胞', '肺腺癌',
            '细胞形态', '细胞排列', '核仁明显', '嗜酸性', '核膜不规则'
        ]
        pred = pred_text or ''
        hit = sum(1 for p in bad_patterns if p in pred)
        if hit == 0:
            return 0.0
        return -min(1.0, 0.25 * hit)

    def reward_repetition_penalty(pred_text: str) -> float:
        pred = (pred_text or '').strip()
        if not pred:
            return -1.0
        if '请描述样本信息?AI>' in pred:
            return -0.5
        return 0.0

    def compute_structured_reward(pred_text: str, ref_text: str, base_score: float) -> float:
        r_region = reward_region(pred_text, ref_text)
        r_gene = reward_gene(pred_text, ref_text)
        r_path = reward_pathway(pred_text, ref_text)
        p_fallback = penalty_fallback(pred_text)
        p_pathology = penalty_pathology_template(pred_text)
        p_repeat = reward_repetition_penalty(pred_text)

        reward = (
            0.15 * float(base_score)
            + 0.45 * r_region
            + 0.20 * r_gene
            + 0.20 * r_path
            + p_fallback
            + p_pathology
            + p_repeat
        )
        return float(max(-1.5, min(1.5, reward)))

    def build_fixed_candidates(data_module):
        train_dataset = data_module["train_dataset"]
        data_collator_fn = data_module["data_collator"]

        samples = [train_dataset[idx] for idx in range(len(train_dataset))]
        batch = data_collator_fn(samples)

        total_candidates = batch["input_ids"].shape[0]
        print("+++++++++++++++++++++++++++++++", flush=True)
        print(f"[GEN_WORKER] Total candidates in batch: {total_candidates}", flush=True)
        print("+++++++++++++++++++++++++++++++", flush=True)

        fixed_candidates = []
        reward_values = []
        reward_details = []
        debug_infos = []

        for i in range(total_candidates):
            one_sample = prepare_one_candidate(batch, i)
            one_sample = expand_gene_placeholder(one_sample)

            text_pack = extract_prompt_and_answer_text(one_sample)
            prompt_text = text_pack["prompt_text"]
            fixed_answer_text = text_pack["answer_text"]

            base_score = compute_reward_from_scores(batch, i)
            if use_structured_reward:
                reward = compute_structured_reward(
                    pred_text=fixed_answer_text,
                    ref_text=fixed_answer_text,
                    base_score=base_score,
                )
            else:
                reward = float(base_score)

            answer_logps = compute_fixed_output_token_logps(one_sample)

            fixed_candidates.append({
                "plen": int(text_pack["plen"]),
                "input_ids": one_sample["input_ids"].cpu(),
                "position_ids": one_sample["position_ids"].cpu(),
                "attention_mask": one_sample["attention_mask"].cpu(),
                "labels": one_sample["labels"].cpu(),
                "pixel_values": one_sample["pixel_values"],
                "image_bound": one_sample["image_bound"],
                "tgt_sizes": one_sample["tgt_sizes"],
                "gene_input_ids": one_sample["gene_input_ids"].cpu() if one_sample["gene_input_ids"] is not None else None,
                "gene_attention_mask": one_sample["gene_attention_mask"].cpu() if one_sample["gene_attention_mask"] is not None else None,
                "gene_bound": one_sample["gene_bound"],
                "answer_logps": answer_logps,
            })

            reward_values.append(reward)
            reward_details.append({
                "candidate_idx": int(i),
                "base_score": float(base_score),
                "reward": float(reward),
            })
            debug_infos.append({
                "candidate_idx": i,
                "prompt_text": prompt_text,
                "fixed_answer_text": fixed_answer_text,
                "avg_logp": (sum(answer_logps) / max(len(answer_logps), 1)) if len(answer_logps) > 0 else 0.0,
                "token_count": len(answer_logps),
            })

        rewards_tensor = torch.tensor(reward_values, dtype=torch.float32)
        return fixed_candidates, rewards_tensor, reward_details, debug_infos

    transform_func = build_transform()

    for iteration in range(999999):
        if iteration % 3 == 0:
            try_update_model()

        batch_samples = random.sample(dataset, min(q_batch_size, len(dataset)))

        data_module = make_supervised_data_module(
            tokenizer=tokenizer,
            gene_tokenizer=gene_tokenizer,
            gspo_sample=batch_samples,
            transform=transform_func,
            data_collator=gspo_data_collator,
            slice_config=slice_config,
            llm_type="minicpm",
            patch_size=patch_size,
            query_nums=query_num,
            batch_vision=batch_vision,
        )

        tic = time.time()

        try:
            fixed_candidates, rewards_tensor, reward_details, debug_infos = build_fixed_candidates(data_module)
        except Exception as e:
            print(f"[GEN_WORKER] build_fixed_candidates failed at iter {iteration}: {e}", flush=True)
            traceback.print_exc()
            time.sleep(1)
            continue

        if len(fixed_candidates) == 0:
            print(f"[GEN_WORKER] no valid fixed candidates at iter {iteration}", flush=True)
            time.sleep(0.5)
            continue

        print(f"\n=== Debug Samples (Iter {iteration}) ===", flush=True)
        for idx in range(min(2, len(debug_infos))):
            info = debug_infos[idx]
            print(f"[Candidate {info['candidate_idx']}]", flush=True)
            print("---- PROMPT ----", flush=True)
            print(info["prompt_text"][:1200], flush=True)
            print("---- FIXED ANSWER ----", flush=True)
            print(info["fixed_answer_text"][:1200], flush=True)
            print(f"---- stats: seq_avg_logp={info['avg_logp']:.4f}, token_count={info['token_count']}", flush=True)
            print("-" * 60, flush=True)
        print("=" * 80 + "\n", flush=True)

        print(f"[GEN_WORKER] iter {iteration}, time: {time.time() - tic:.2f}s, rewards: {rewards_tensor.tolist()}", flush=True)
        print(f"[GEN_WORKER] reward_details: {reward_details}", flush=True)

        for i in range(len(fixed_candidates)):
            candidate_data = fixed_candidates[i]
            # print("[UPLOAD DEBUG] i =", i, flush=True)
            # print("[UPLOAD DEBUG] gene_bound =", candidate_data["gene_bound"], flush=True)
            # print("[UPLOAD DEBUG] gene_input_ids_none =", candidate_data["gene_input_ids"] is None, flush=True)
            # print("[UPLOAD DEBUG] gene_input_ids_shape =", None if candidate_data["gene_input_ids"] is None else tuple(candidate_data["gene_input_ids"].shape), flush=True)

            data_list = []

            meta = {"plen": int(candidate_data["plen"])}
            data_list.append(json.dumps(meta).encode())

            data_list.append(tensor_to_bytes(candidate_data["input_ids"].unsqueeze(0)))
            data_list.append(tensor_to_bytes(candidate_data["position_ids"].unsqueeze(0)))
            data_list.append(tensor_to_bytes(candidate_data["attention_mask"].unsqueeze(0)))
            data_list.append(tensor_to_bytes(candidate_data["labels"].unsqueeze(0)))
            data_list.append(tensor_to_bytes(rewards_tensor[i:i+1]))

            pixel_tensors = []
            if isinstance(candidate_data["pixel_values"], list):
                for pv in candidate_data["pixel_values"]:
                    if isinstance(pv, torch.Tensor):
                        pixel_tensors.append(tensor_to_bytes(pv))
                    elif isinstance(pv, list):
                        for x in pv:
                            if isinstance(x, torch.Tensor):
                                pixel_tensors.append(tensor_to_bytes(x))
            elif isinstance(candidate_data["pixel_values"], torch.Tensor):
                pixel_tensors.append(tensor_to_bytes(candidate_data["pixel_values"]))

            data_list.append(make_bytes_list(pixel_tensors) if len(pixel_tensors) > 0 else b"")

            data_list.append(json.dumps(safe_jsonable_bound(candidate_data["image_bound"])).encode())
            data_list.append(json.dumps(convert_tensors_to_list(candidate_data["tgt_sizes"])).encode())

            data_list.append(
                tensor_to_bytes(candidate_data["gene_input_ids"])
                if candidate_data["gene_input_ids"] is not None else b""
            )
            data_list.append(
                tensor_to_bytes(candidate_data["gene_attention_mask"])
                if candidate_data["gene_attention_mask"] is not None else b""
            )
            data_list.append(json.dumps(safe_jsonable_bound(candidate_data["gene_bound"])).encode())

            data_list.append(json.dumps(candidate_data["answer_logps"]).encode())

            xdata = make_bytes_list(data_list)

            try:
                r = requests.post(f"{ref_server}/upload", data=xdata, timeout=60)
                if r.status_code != 200:
                    print(f"[GEN_WORKER] upload failed: {r.status_code} {r.text}", flush=True)
            except Exception as e:
                print(f"[GEN_WORKER] upload exception: {e}", flush=True)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    q = mp.Queue()
    gen_worker(
        q,
        model_path="/data2/xiaoxinyu/project/model_cpt_v5_qformer",
        example_json="/data2/xiaoxinyu/RL/data/pathvqa_open_modelneg_test5.json",
        ref_port=59875,
        physics_device=4,
        q_batch_size=1,
        max_new_tokens=96,
        temperature=0.7,
        top_p=0.9,
        max_slice_nums=1,
        use_structured_reward=False,
    )