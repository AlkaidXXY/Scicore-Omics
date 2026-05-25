#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
reference server
1. 接收生成端上传的样本
2. 用 fixed reference model 计算 ref token-level log probabilities
3. 入队，供 trainer 拉取
"""

from flask import Flask, request
import os
import sys
import json
import queue
import struct
import argparse
import traceback
from typing import Any, List

import torch
from transformers import AutoModel, AutoTokenizer

from gene_tokenizer import GeneTokenizer

app = Flask(__name__)

data_queue = queue.Queue(maxsize=1000)
ref_model = None
tokenizer = None
gene_tokenizer = None

MODEL_PATH = None
PORT = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--example_json", type=str, required=False, default="")
    parser.add_argument("--port", type=int, default=59875)
    return parser.parse_args()


def tensor_to_bytes(tensor: torch.Tensor | None) -> bytes:
    if tensor is None:
        return b""
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    tensor = tensor.detach().cpu().contiguous()

    shape_bytes = struct.pack("i" * len(tensor.shape), *tensor.shape)
    dtype_str = str(tensor.dtype).encode()
    data_bytes = tensor.numpy().tobytes()

    return (
        struct.pack("i", len(tensor.shape))
        + shape_bytes
        + struct.pack("i", len(dtype_str))
        + dtype_str
        + data_bytes
    )


def bytes_to_tensor(b: bytes) -> torch.Tensor:
    if len(b) == 0:
        return torch.tensor([])

    offset = 0
    shape_len = struct.unpack("i", b[offset:offset + 4])[0]
    offset += 4

    shape = struct.unpack("i" * shape_len, b[offset:offset + 4 * shape_len])
    offset += 4 * shape_len

    dtype_len = struct.unpack("i", b[offset:offset + 4])[0]
    offset += 4

    dtype_str = b[offset:offset + dtype_len].decode()
    offset += dtype_len

    dtype_name = dtype_str.split(".")[-1]
    dtype = getattr(torch, dtype_name)

    data = torch.frombuffer(bytearray(b[offset:]), dtype=dtype).clone()
    return data.reshape(shape)


def make_bytes_list(data_list: List[bytes]) -> bytes:
    result = struct.pack("i", len(data_list))
    for d in data_list:
        result += struct.pack("i", len(d)) + d
    return result


def bytes_list_to_list(data: bytes) -> List[bytes]:
    offset = 0
    count = struct.unpack("i", data[offset:offset + 4])[0]
    offset += 4

    result = []
    for _ in range(count):
        length = struct.unpack("i", data[offset:offset + 4])[0]
        offset += 4
        result.append(data[offset:offset + length])
        offset += length
    return result


def safe_json_loads_bytes(b: bytes, default):
    try:
        return json.loads(b.decode())
    except Exception:
        return default


def get_ref_device():
    return next(ref_model.parameters()).device


def restore_pixel_values(pixel_values_bytes: bytes, ref_device: torch.device):
    """
    恢复为模型期望的二级列表格式：
    [[tensor, tensor, ...]]
    """
    pixel_values_for_model = []

    if len(pixel_values_bytes) == 0:
        return [[]]

    current_sample_pixels = []

    try:
        nested_list = bytes_list_to_list(pixel_values_bytes)
        for b in nested_list:
            if len(b) == 0:
                continue
            tensor = bytes_to_tensor(b)
            if tensor.numel() > 0 and tensor.dim() >= 2:
                current_sample_pixels.append(tensor.to(ref_device))
        pixel_values_for_model.append(current_sample_pixels)
        return pixel_values_for_model
    except Exception:
        pass

    try:
        pv_tensor = bytes_to_tensor(pixel_values_bytes)
        if pv_tensor.numel() > 0 and pv_tensor.dim() >= 2:
            return [[pv_tensor.to(ref_device)]]
    except Exception:
        pass

    return [[]]


def restore_image_bound(image_bound_bytes: bytes):
    if len(image_bound_bytes) == 0:
        return []

    # 先按 JSON 解，因为 gen_worker 上传时就是 json.dumps(...).encode()
    try:
        obj = json.loads(image_bound_bytes.decode())
        return obj
    except Exception:
        pass

    # 再兜底按 tensor bytes 解
    try:
        image_bound = bytes_to_tensor(image_bound_bytes)
        return image_bound.tolist()
    except Exception:
        return []
    

def restore_tgt_sizes(tgt_sizes_bytes: bytes):
    if len(tgt_sizes_bytes) == 0:
        return []

    tgt_sizes_list = safe_json_loads_bytes(tgt_sizes_bytes, [])
    ret = []
    for item in tgt_sizes_list:
        try:
            ret.append(torch.tensor(item, dtype=torch.int32))
        except Exception:
            pass
    return ret


def restore_gene_bound(gene_bound_bytes: bytes):
    if len(gene_bound_bytes) == 0:
        return []

    try:
        return json.loads(gene_bound_bytes.decode())
    except Exception:
        pass

    try:
        t = bytes_to_tensor(gene_bound_bytes)
        return t.tolist()
    except Exception:
        return []
    

@app.route("/upload", methods=["POST"])
def upload():
    """
    接收生成端上传的样本，计算 ref logps，再入队。
    协议：
      dd[0]  meta(json)
      dd[1]  input_ids
      dd[2]  position_ids
      dd[3]  attention_mask
      dd[4]  labels
      dd[5]  rewards
      dd[6]  pixel_values
      dd[7]  image_bound
      dd[8]  tgt_sizes
      dd[9]  gene_input_ids
      dd[10] gene_attention_mask
      dd[11] gene_bound
      dd[12] gen_logps_json
    """
    global ref_model

    try:
        ref_device = get_ref_device()

        raw = request.data
        dd = bytes_list_to_list(raw)

        if len(dd) != 13:
            msg = f"Invalid upload payload length: expected 13, got {len(dd)}"
            print(msg)
            return msg.encode(), 400

        meta = json.loads(dd[0].decode())

        input_ids = bytes_to_tensor(dd[1])
        position_ids = bytes_to_tensor(dd[2])
        attention_mask = bytes_to_tensor(dd[3])
        labels = bytes_to_tensor(dd[4])
        rewards = bytes_to_tensor(dd[5])

        pixel_values_bytes = dd[6]
        image_bound_bytes = dd[7]
        tgt_sizes_bytes = dd[8]
        gene_input_ids_bytes = dd[9]
        gene_attention_mask_bytes = dd[10]
        gene_bound_bytes = dd[11]
        gen_logps_bytes = dd[12]

        pixel_values_for_model = restore_pixel_values(pixel_values_bytes, ref_device)
        image_bound = restore_image_bound(image_bound_bytes)
        tgt_sizes = restore_tgt_sizes(tgt_sizes_bytes)

        gene_input_ids = (
            bytes_to_tensor(gene_input_ids_bytes) if len(gene_input_ids_bytes) > 0 else None
        )
        gene_attention_mask = (
            bytes_to_tensor(gene_attention_mask_bytes) if len(gene_attention_mask_bytes) > 0 else None
        )
        gene_bound = restore_gene_bound(gene_bound_bytes)

        batch_inputs = {
            "input_ids": input_ids.to(ref_device),
            "position_ids": position_ids.to(ref_device),
            "attention_mask": attention_mask.to(ref_device),
            "pixel_values": pixel_values_for_model,
            "image_bound": [image_bound],
            "tgt_sizes": tgt_sizes,
            "gene_input_ids": gene_input_ids.to(ref_device) if gene_input_ids is not None and gene_input_ids.numel() > 0 else None,
            "gene_attention_mask": gene_attention_mask.to(ref_device) if gene_attention_mask is not None and gene_attention_mask.numel() > 0 else None,
            "gene_bound": [gene_bound] if gene_input_ids is not None and len(gene_bound) > 0 else [[]],
        }

        if batch_inputs["gene_input_ids"] is None:
            batch_inputs["gene_bound"] = [[]]

        if len(pixel_values_for_model) == 0:
            batch_inputs["pixel_values"] = [[]]
            batch_inputs["image_bound"] = [[]]
            batch_inputs["tgt_sizes"] = []

        with torch.no_grad():
            ref_outputs = ref_model(data=batch_inputs, use_cache=False)
            ref_logits = ref_outputs.logits

            log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
            per_token_logps = torch.gather(
                log_probs,
                dim=2,
                index=labels.unsqueeze(2).clamp(min=0).long().to(ref_device)
            ).squeeze(2)

            answer_mask = (labels != -100).float().to(ref_device)
            ref_answer_logps = per_token_logps * answer_mask

        queue_data = [
            json.dumps(meta).encode(),
            tensor_to_bytes(input_ids.cpu()),
            tensor_to_bytes(position_ids.cpu()),
            tensor_to_bytes(attention_mask.cpu()),
            tensor_to_bytes(labels.cpu()),
            tensor_to_bytes(rewards.cpu()),
            pixel_values_bytes,
            image_bound_bytes,
            tgt_sizes_bytes,
            gene_input_ids_bytes,
            gene_attention_mask_bytes,
            gene_bound_bytes,
            tensor_to_bytes(ref_answer_logps.cpu()),
            gen_logps_bytes,
        ]

        packed_data = make_bytes_list(queue_data)

        try:
            data_queue.put(packed_data, timeout=1)
            return b"ok", 200
        except queue.Full:
            return b"queue_full", 503
    
    except Exception as e:
        print(f"[REF_SERVER] Error in /upload: {e}")
        traceback.print_exc()
        return b"error", 500
    
    print("[REF DEBUG] gene_bound restored =", gene_bound)
    print("[REF DEBUG] gene_input_ids is None =", gene_input_ids is None)
    print("[REF DEBUG] gene_input_ids numel =", None if gene_input_ids is None else gene_input_ids.numel())
        


@app.route("/get", methods=["GET"])
def get_data():
    try:
        data = data_queue.get_nowait()
        return data
    except queue.Empty:
        return b"empty"


@app.route("/health", methods=["GET"])
def health():
    return b"ok", 200


def init_ref_model():
    global ref_model, tokenizer, gene_tokenizer
    sys.path.append(MODEL_PATH)

    print("[REF_SERVER] Loading reference model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    ref_model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    ref_model.eval()

    gene_tokenizer = GeneTokenizer(
        vocab_file="/data2/xiaoxinyu/project/model/gene_tokenizer/vocab.json"
    )
    print("[REF_SERVER] Reference model loaded!")


if __name__ == "__main__":
    args = parse_args()
    MODEL_PATH = args.model_path
    PORT = args.port
    init_ref_model()
    app.run(host="0.0.0.0", port=PORT, threaded=True)