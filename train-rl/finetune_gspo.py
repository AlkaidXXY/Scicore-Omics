#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import json
import time
import argparse
import traceback
from typing import Any, Dict, List, Optional

import requests
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoModel

from ref_server import (
    bytes_to_tensor,
    bytes_list_to_list,
    restore_pixel_values,
    restore_image_bound,
    restore_tgt_sizes,
    restore_gene_bound,
)


# =========================
# 基础工具
# =========================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--example_json", type=str, required=True)
    parser.add_argument("--ref_port", type=int, default=59875)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--all_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--rollout_accum_steps", type=int, default=1)
    parser.add_argument("--gen_update_steps", type=int, default=32)
    parser.add_argument("--save_steps", type=int, default=200)

    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # 仅用于启动 gen_worker 的兼容参数
    parser.add_argument("--gen_device", type=int, default=4)
    parser.add_argument("--gen_max_new_tokens", type=int, default=64)
    parser.add_argument("--gen_temperature", type=float, default=0.3)
    parser.add_argument("--gen_top_p", type=float, default=0.85)
    parser.add_argument("--gen_q_batch_size", type=int, default=1)
    parser.add_argument("--gen_max_slice_nums", type=int, default=1)

    parser.add_argument("--gene_vocab_file", type=str, default=None)
    parser.add_argument("--fp16", action="store_true", default=True)

    # 最后一层 LLM 开关
    parser.add_argument("--enable_last_llm_layer", action="store_true", default=True)
    parser.add_argument("--disable_last_llm_layer", action="store_true", default=False)

    return parser.parse_args()


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def is_rank0():
    return get_rank() == 0


def print0(*args, **kwargs):
    if is_rank0():
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)


def broadcast_object(obj, src=0):
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def get_local_rank():
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return 0


def setup_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")


def cleanup_distributed():
    if dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass


# =========================
# 训练参数组控制
# =========================

def freeze_all(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad_(False)


def freeze_for_mixed_rl_static(model: torch.nn.Module):
    """
    先全部冻结。
    真正每一步哪些参数参与训练，不靠这里决定；
    而是在 step 内按 rollout 模态动态切换。
    """
    freeze_all(model)
    groups = {
        "gene": [
            "gene_qformer.",
            "gene_projector.",
        ],
        "image": [
            "resampler.",
        ],
        "llm_last": [
            # 你的日志里最后一层是 27
            "llm.model.layers.27.",
        ],
    }
    return groups


def mark_trainable_by_name_prefix(model: torch.nn.Module, prefixes: List[str]):
    cnt = 0
    hit_names = []
    for name, p in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            p.requires_grad_(True)
            cnt += p.numel()
            hit_names.append(name)
    return cnt, hit_names


def set_trainable_groups(model: torch.nn.Module, groups_to_enable: List[str], group_prefix_map: Dict[str, List[str]]):
    """
    按当前 step 的样本模态，动态打开对应参数组。
    """
    freeze_all(model)

    enabled_stats = {}
    for group_name in groups_to_enable:
        prefixes = group_prefix_map.get(group_name, [])
        cnt, hit_names = mark_trainable_by_name_prefix(model, prefixes)
        enabled_stats[group_name] = {
            "param_count": cnt,
            "n_tensors": len(hit_names),
            "sample_names": hit_names[:10],
        }
    return enabled_stats


def get_trainable_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    raw = model.module if hasattr(model, "module") else model
    state = {}

    allowed_prefixes = [
        "gene_qformer.",
        "gene_projector.",
        "resampler.",
        "llm.model.layers.27.",
    ]
    for k, v in raw.state_dict().items():
        if any(k.startswith(prefix) for prefix in allowed_prefixes):
            state[k] = v.detach().cpu()
    return state


def save_trainable_checkpoint(model, optimizer, step, output_dir):
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    raw_model = model.module if hasattr(model, "module") else model
    save_obj = {
        "model": get_trainable_state_dict(raw_model),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    save_path = os.path.join(ckpt_dir, "rl_trainable.pt")
    torch.save(save_obj, save_path)
    print0(f"[TRAIN] Saved checkpoint to {save_path}")


# =========================
# ref_server 拉取
# =========================

def get_ref_server_url(args) -> str:
    return f"http://127.0.0.1:{args.ref_port}"


def get_batch_from_ref_server(args) -> Optional[Dict[str, Any]]:
    ref_server = get_ref_server_url(args)
    try:
        r = requests.get(f"{ref_server}/get", timeout=(2, 30))
        if r.content == b"empty":
            return None
        dd = bytes_list_to_list(r.content)
    except requests.exceptions.ReadTimeout:
        print0("[TRAIN] ref_server /get read timeout, treat as no batch yet")
        return None
    except Exception as e:
        print0(f"[TRAIN] Cannot get batch from ref_server: {e}")
        return None

    if len(dd) != 14:
        print0(f"[TRAIN] Bad batch payload length: {len(dd)}")
        return None

    try:
        meta = json.loads(dd[0].decode())
        batch = {
            "meta": meta,
            "input_ids": bytes_to_tensor(dd[1]),
            "position_ids": bytes_to_tensor(dd[2]),
            "attention_mask": bytes_to_tensor(dd[3]),
            "labels": bytes_to_tensor(dd[4]),
            "rewards": bytes_to_tensor(dd[5]),
            "pixel_values_bytes": dd[6],
            "image_bound_bytes": dd[7],
            "tgt_sizes_bytes": dd[8],
            "gene_input_ids_bytes": dd[9],
            "gene_attention_mask_bytes": dd[10],
            "gene_bound_bytes": dd[11],
            "ref_logps": bytes_to_tensor(dd[12]),
            "gen_logps_bytes": dd[13],
        }
        return batch
    except Exception as e:
        print0(f"[TRAIN] Failed to decode batch: {e}")
        traceback.print_exc()
        return None


def inspect_rollout_item(batch):
    """
    兼容有 gene / 无 gene：
    - 有 gene：做基本检查
    - 无 gene：也允许通过
    """
    try:
        gene_bytes = batch["gene_input_ids_bytes"]
        gene_bound_bytes = batch["gene_bound_bytes"]

        if len(gene_bytes) == 0:
            return True, "no_gene_ok"

        gene_ids = bytes_to_tensor(gene_bytes)
        if gene_ids.numel() == 0:
            return True, "decoded_empty_gene_ok"

        gene_bound = restore_gene_bound(gene_bound_bytes)
        if gene_bound is None:
            return True, "gene_bound_none_ok"
        if isinstance(gene_bound, list) and len(gene_bound) == 0:
            return True, "gene_bound_empty_ok"

        if isinstance(gene_bound, list) and len(gene_bound) > 0:
            first = gene_bound[0]
            if isinstance(first, list) and len(first) == 2:
                span = int(first[1]) - int(first[0])
                if span != 32:
                    return False, f"bad_gene_span_{span}"

        return True, "gene_ok"
    except Exception as e:
        return False, f"inspect_exception:{e}"


def collect_rollout_batch(args, target_n: int) -> List[Dict[str, Any]]:
    buf = []
    reason_count = {}
    t0 = time.time()

    while len(buf) < target_n:
        if time.time() - t0 > 120:
            print0(f"[TRAIN] rollout timeout: collected {len(buf)}/{target_n}")
            if reason_count:
                print0(f"[TRAIN] rollout inspect reasons: {reason_count}")
            break

        item = get_batch_from_ref_server(args)
        if item is None:
            time.sleep(0.2)
            continue

        ok, reason = inspect_rollout_item(item)
        reason_count[reason] = reason_count.get(reason, 0) + 1
        if not ok:
            print0(f"[TRAIN] skip one rollout item: {reason}")
            continue

        buf.append(item)

    return buf


# =========================
# 模态判断
# =========================

def rollout_has_gene(batch: Dict[str, Any]) -> bool:
    return len(batch["gene_input_ids_bytes"]) > 0


def rollout_has_image(batch: Dict[str, Any]) -> bool:
    return len(batch["pixel_values_bytes"]) > 0


def decide_train_groups_from_rollout(batch_list: List[Dict[str, Any]], enable_last_llm_layer: bool = True):
    """
    只要这个 step 的 rollout 里出现过某种模态，就打开相应参数组。
    默认始终打开 llm_last。
    """
    has_gene = any(rollout_has_gene(x) for x in batch_list)
    has_image = any(rollout_has_image(x) for x in batch_list)

    groups = []
    if enable_last_llm_layer:
        groups.append("llm_last")
    if has_gene:
        groups.append("gene")
    if has_image:
        groups.append("image")

    return {
        "has_gene": has_gene,
        "has_image": has_image,
        "groups": groups,
    }


# =========================
# batch 还原
# =========================

def _build_model_inputs_from_packed_sample(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    input_ids = batch["input_ids"].to(device)
    position_ids = batch["position_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    pixel_values_for_model = restore_pixel_values(batch["pixel_values_bytes"], device)
    image_bound = restore_image_bound(batch["image_bound_bytes"])
    tgt_sizes = restore_tgt_sizes(batch["tgt_sizes_bytes"])

    gene_input_ids = (
        bytes_to_tensor(batch["gene_input_ids_bytes"]).to(device)
        if len(batch["gene_input_ids_bytes"]) > 0 else None
    )
    gene_attention_mask = (
        bytes_to_tensor(batch["gene_attention_mask_bytes"]).to(device)
        if len(batch["gene_attention_mask_bytes"]) > 0 else None
    )
    gene_bound = restore_gene_bound(batch["gene_bound_bytes"])

    batch_inputs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values_for_model,
        "image_bound": [image_bound],
        "tgt_sizes": tgt_sizes,
        "gene_input_ids": gene_input_ids,
        "gene_attention_mask": gene_attention_mask,
        "gene_bound": [gene_bound] if gene_input_ids is not None and gene_bound is not None and len(gene_bound) > 0 else [[]],
    }

    if batch_inputs["gene_input_ids"] is None:
        batch_inputs["gene_bound"] = [[]]

    if len(pixel_values_for_model) == 0:
        batch_inputs["pixel_values"] = [[]]
        batch_inputs["image_bound"] = [[]]
        batch_inputs["tgt_sizes"] = []

    return batch_inputs


# =========================
# PPO / GSPO
# =========================

def GSPO_step_batch(model: torch.nn.Module, batch_list: List[Dict[str, Any]], clip_param: float, beta: float):
    device = next(model.parameters()).device
    if batch_list is None or len(batch_list) == 0:
        raise ValueError("empty rollout batch")

    actor_seq_logps = []
    ref_seq_logps = []
    gen_seq_logps = []
    rewards_all = []

    for batch in batch_list:
        labels = batch["labels"].to(device)
        ref_logps = batch["ref_logps"].to(device)
        rewards = batch["rewards"].to(device).view(-1)

        batch_inputs = _build_model_inputs_from_packed_sample(batch, device)

        outputs = model(data=batch_inputs, use_cache=False)
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        gather_index = labels.unsqueeze(2).clamp(min=0).long()
        actor_per_token = torch.gather(log_probs, dim=2, index=gather_index).squeeze(2)

        answer_mask = (labels != -100).float()

        actor_answer = actor_per_token * answer_mask
        ref_answer = ref_logps * answer_mask

        try:
            gen_logps_list = json.loads(batch["gen_logps_bytes"].decode())
        except Exception:
            gen_logps_list = []

        gen_logps_tensor = torch.tensor(
            gen_logps_list,
            dtype=actor_per_token.dtype,
            device=device,
        )

        gen_answer = torch.zeros_like(actor_per_token)
        answer_indices = torch.where(answer_mask[0] > 0)[0]
        valid_len = min(len(answer_indices), len(gen_logps_tensor))
        if valid_len > 0:
            gen_answer[0, answer_indices[:valid_len]] = gen_logps_tensor[:valid_len]

        answer_len = answer_mask.sum(dim=-1).clamp(min=1)

        actor_seq_logps.append(actor_answer.sum(dim=-1) / answer_len)
        ref_seq_logps.append(ref_answer.sum(dim=-1) / answer_len)
        gen_seq_logps.append(gen_answer.sum(dim=-1) / answer_len)
        rewards_all.append(rewards)

    actor_seq_logp = torch.cat(actor_seq_logps, dim=0)
    ref_seq_logp = torch.cat(ref_seq_logps, dim=0)
    gen_seq_logp = torch.cat(gen_seq_logps, dim=0)
    rewards = torch.cat(rewards_all, dim=0)

    log_ratio = actor_seq_logp - gen_seq_logp
    ratio = torch.exp(log_ratio)
    kl_div = actor_seq_logp - ref_seq_logp

    advantages = rewards
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages

    policy_loss = -torch.min(surr1, surr2).mean()
    kl_penalty = beta * kl_div.mean()
    total_loss = policy_loss + kl_penalty

    stat = {
        "loss": float(total_loss.detach().item()),
        "policy_loss": float(policy_loss.detach().item()),
        "kl_penalty": float(kl_penalty.detach().item()),
        "reward": float(rewards.mean().detach().item()),
        "actor_seq_logp": float(actor_seq_logp.mean().detach().item()),
        "ref_seq_logp": float(ref_seq_logp.mean().detach().item()),
        "gen_seq_logp": float(gen_seq_logp.mean().detach().item()),
        "ratio": float(ratio.mean().detach().item()),
        "kl": float(kl_div.mean().detach().item()),
        "loss_requires_grad": bool(total_loss.requires_grad),
        "loss_grad_fn": str(total_loss.grad_fn) if total_loss.grad_fn is not None else "None",
    }
    return total_loss, stat


def ensure_scalar_loss(loss, model):
    if not torch.is_tensor(loss):
        device = next(model.parameters()).device
        loss = torch.tensor(loss, device=device, dtype=torch.float32)

    if not torch.isfinite(loss).all():
        raise ValueError(f"loss has non-finite values: {loss}")

    if loss.ndim > 0:
        loss = loss.mean()

    loss = loss.squeeze()
    if loss.ndim > 0:
        loss = loss.reshape([])

    return loss


# =========================
# 主训练
# =========================

def train():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    enable_last_llm_layer = args.enable_last_llm_layer and (not args.disable_last_llm_layer)

    setup_distributed()
    rank = get_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        cfg = vars(args).copy()
        cfg["effective_enable_last_llm_layer"] = enable_last_llm_layer
        print("=" * 60, flush=True)
        print("[TRAIN] DDP mixed RL config", flush=True)
        print(json.dumps(cfg, indent=2, ensure_ascii=False), flush=True)
        print(f"[TRAIN] world_size = {world_size}", flush=True)
        print("=" * 60, flush=True)

    Q = None
    p = None

    if rank == 0:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        Q = mp.Queue()

        from gen_worker import gen_worker
        p = mp.Process(
            target=gen_worker,
            args=(Q, args.model_path, args.example_json, args.ref_port, args.gen_device),
            kwargs={
                "q_batch_size": args.gen_q_batch_size,
                "max_new_tokens": args.gen_max_new_tokens,
                "temperature": args.gen_temperature,
                "top_p": args.gen_top_p,
                "max_slice_nums": args.gen_max_slice_nums,
            },
        )
        p.start()
        print("[TRAIN] Started fixed-output worker process.", flush=True)

    dist.barrier()

    print0("[TRAIN] Loading model...")
    sys.path.append(args.model_path)

    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    group_prefix_map = freeze_for_mixed_rl_static(model)
    model = model.to(device)

    # 初始化先只开 llm_last，便于 optimizer 建立
    init_groups = ["llm_last"] if enable_last_llm_layer else []
    init_enabled = set_trainable_groups(model, init_groups, group_prefix_map)

    if rank == 0:
        print("[TRAIN] init enabled groups:", flush=True)
        print(json.dumps(init_enabled, indent=2, ensure_ascii=False), flush=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    progress = range(1, args.all_steps + 1)
    if rank == 0:
        progress = tqdm(progress, dynamic_ncols=True)

    for step in progress:
        if rank == 0:
            rollout_batch = collect_rollout_batch(args, args.rollout_accum_steps)
            print(f"[TRAIN] step={step} collected_rollouts={0 if rollout_batch is None else len(rollout_batch)}", flush=True)
        else:
            rollout_batch = None

        rollout_batch = broadcast_object(rollout_batch, src=0)

        if rollout_batch is None or len(rollout_batch) == 0:
            if rank == 0:
                print(f"[TRAIN] Skip step {step}: empty rollout batch", flush=True)
            continue

        # ===== 动态决定当前 step 打开哪些参数组 =====
        raw_model = model.module if hasattr(model, "module") else model
        modal_info = decide_train_groups_from_rollout(
            rollout_batch,
            enable_last_llm_layer=enable_last_llm_layer,
        )

        enabled_stats = set_trainable_groups(
            raw_model,
            groups_to_enable=modal_info["groups"],
            group_prefix_map=group_prefix_map,
        )

        trainable_params = [p for p in raw_model.parameters() if p.requires_grad]

        # 为了保证参数组变化能生效，这里每步重建 optimizer
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.01,
        )

        if rank == 0:
            total_trainable = sum(p.numel() for p in trainable_params)
            print(
                f"[TRAIN] step={step} modal_info="
                f"has_image={modal_info['has_image']} "
                f"has_gene={modal_info['has_gene']} "
                f"groups={modal_info['groups']} "
                f"total_trainable={total_trainable:,}",
                flush=True,
            )
            print(f"[TRAIN] step={step} enabled_stats={json.dumps(enabled_stats, ensure_ascii=False)}", flush=True)

        try:
            loss, stat = GSPO_step_batch(
                raw_model,
                rollout_batch,
                clip_param=args.clip_param,
                beta=args.beta,
            )

            loss = ensure_scalar_loss(loss, raw_model)

            if rank == 0:
                print(
                    f"[TRAIN][DEBUG] step={step} "
                    f"loss.shape={tuple(loss.shape)} "
                    f"loss={loss.item():.6f} "
                    f"requires_grad={loss.requires_grad} "
                    f"grad_fn={loss.grad_fn}",
                    flush=True,
                )

            if (not loss.requires_grad) or (loss.grad_fn is None):
                if rank == 0:
                    print(f"[TRAIN] Skip step {step}: loss has no grad path to trainable params.", flush=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            if rank == 0:
                print(
                    f"[TRAIN][STAT] step={step} "
                    f"loss={stat['loss']:.6f} "
                    f"policy_loss={stat['policy_loss']:.6f} "
                    f"kl_penalty={stat['kl_penalty']:.6f} "
                    f"reward={stat['reward']:.6f} "
                    f"ratio={stat['ratio']:.6f} "
                    f"kl={stat['kl']:.6f} "
                    f"actor_seq_logp={stat['actor_seq_logp']:.6f} "
                    f"ref_seq_logp={stat['ref_seq_logp']:.6f} "
                    f"gen_seq_logp={stat['gen_seq_logp']:.6f} "
                    f"grad_norm={float(grad_norm):.6f}",
                    flush=True,
                )

        except Exception as e:
            print(f"[TRAIN][rank={rank}] Step {step} failed: {e}", flush=True)
            traceback.print_exc()
            continue

        if rank == 0 and Q is not None and (step % args.gen_update_steps == 0):
            try:
                cpu_state = get_trainable_state_dict(raw_model)
                Q.put(cpu_state)
                print(f"[TRAIN] Sent updated partial model to gen_worker at step {step}", flush=True)
            except Exception as e:
                print(f"[TRAIN] Failed to send model update: {e}", flush=True)

        if rank == 0 and step % args.save_steps == 0:
            try:
                save_trainable_checkpoint(model, optimizer, step, args.output_dir)
            except Exception as e:
                print(f"[TRAIN] Save checkpoint failed: {e}", flush=True)

        if rank == 0:
            progress.set_postfix({
                "loss": f"{stat['loss']:.4f}",
                "reward": f"{stat['reward']:.4f}",
                "ratio": f"{stat['ratio']:.4f}",
                "kl": f"{stat['kl']:.4f}",
            })

    if rank == 0 and p is not None:
        try:
            p.terminate()
            p.join(timeout=5)
            print("[TRAIN] Worker process terminated.", flush=True)
        except Exception:
            pass

    cleanup_distributed()


if __name__ == "__main__":
    train()