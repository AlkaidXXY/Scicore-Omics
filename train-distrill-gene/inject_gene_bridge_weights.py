#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import shutil
from typing import Dict, Any, Set

import torch
from safetensors.torch import load_file, save_file


def read_index(index_path: str) -> Dict[str, Any]:
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    # ============
    # Paths (EDIT)
    # ============
    base_dir = "/data2/xiaoxinyu/project/model_cpt_v7_qformer"
    step_ckpt = "/data2/xiaoxinyu/project/qformer/distill_out_cpt_v7_real_processor/gene_bridge_distill_real_processor_step1000.pt"
    out_dir = "/data2/xiaoxinyu/project/model_cpt_v7_qformer_injected_step1000"  # new dir

    index_path = os.path.join(base_dir, "model.safetensors.index.json")
    assert os.path.exists(index_path), f"index not found: {index_path}"
    assert os.path.exists(step_ckpt), f"ckpt not found: {step_ckpt}"

    # -----------------------------
    # 0) Copy model dir (safe)
    # -----------------------------
    if os.path.exists(out_dir):
        raise RuntimeError(f"out_dir already exists: {out_dir}")
    print(f"[1/5] Copying base_dir -> out_dir ...")
    shutil.copytree(base_dir, out_dir)

    out_index_path = os.path.join(out_dir, "model.safetensors.index.json")
    idx = read_index(out_index_path)
    weight_map: Dict[str, str] = idx["weight_map"]

    # -----------------------------
    # 1) Load distilled gene ckpt
    # -----------------------------
    print(f"[2/5] Loading distilled ckpt ...")
    sd = torch.load(step_ckpt, map_location="cpu")
    gene_qformer_sd: Dict[str, torch.Tensor] = sd["gene_qformer"]
    gene_projector_sd: Dict[str, torch.Tensor] = sd["gene_projector"]

    # Flatten into full model keyspace
    inject_sd: Dict[str, torch.Tensor] = {}
    for k, v in gene_qformer_sd.items():
        inject_sd[f"gene_qformer.{k}"] = v.cpu()
    for k, v in gene_projector_sd.items():
        inject_sd[f"gene_projector.{k}"] = v.cpu()

    inject_keys: Set[str] = set(inject_sd.keys())

    # -----------------------------
    # 2) Find which shard files contain these keys
    # -----------------------------
    shard_to_keys: Dict[str, Set[str]] = {}
    missing_in_index = []
    for k in inject_keys:
        shard = weight_map.get(k, None)
        if shard is None:
            missing_in_index.append(k)
            continue
        shard_to_keys.setdefault(shard, set()).add(k)

    if missing_in_index:
        print("❌ Some inject keys not found in index weight_map (show first 20):")
        for k in missing_in_index[:20]:
            print("  ", k)
        print("Tip: This usually means your module param names differ from base model.")
        raise RuntimeError(f"{len(missing_in_index)} keys missing in index. Cannot inject safely.")

    print(f"[3/5] Will modify {len(shard_to_keys)} shard(s): {list(shard_to_keys.keys())}")

    # -----------------------------
    # 3) For each shard, load safetensors, replace matching keys, save back
    # -----------------------------
    for shard_name, keys in shard_to_keys.items():
        shard_path = os.path.join(out_dir, shard_name)
        assert os.path.exists(shard_path), f"shard not found: {shard_path}"

        print(f"  - Loading shard: {shard_name}")
        shard_sd = load_file(shard_path)  # Dict[str, Tensor]

        # Check all keys exist
        not_in_shard = [k for k in keys if k not in shard_sd]
        if not_in_shard:
            print(f"❌ Keys not found inside shard {shard_name} (first 20):")
            for k in not_in_shard[:20]:
                print("   ", k)
            raise RuntimeError(f"{len(not_in_shard)} keys missing in shard file. Index may be stale?")

        # Replace
        updated = dict(shard_sd)
        for k in keys:
            updated[k] = inject_sd[k].to(updated[k].dtype)  # keep original dtype
        # Save back
        save_file(updated, shard_path)
        print(f"    ✅ Updated {len(keys)} tensors in {shard_name}")

    # -----------------------------
    # 4) Done
    # -----------------------------
    print(f"[5/5] ✅ Injection finished.")
    print("New model dir:", out_dir)
    print("You can now load with:")
    print(f"  AutoModel.from_pretrained('{out_dir}', trust_remote_code=True)")


if __name__ == "__main__":
    main()
