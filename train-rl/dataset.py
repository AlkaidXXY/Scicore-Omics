import copy
import json
import logging
import math
import os
import re
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import anndata
import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer

logger = logging.getLogger(__name__)

llama3_chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}"


def ensure_gene_special_token(tokenizer):
    """
    确保 tokenizer 中存在 <gene> 特殊 token
    """
    try:
        gene_id = tokenizer.convert_tokens_to_ids("<gene>")
        if gene_id is None or gene_id == tokenizer.unk_token_id:
            tokenizer.add_tokens(["<gene>"], special_tokens=True)
    except Exception:
        tokenizer.add_tokens(["<gene>"], special_tokens=True)


def ensure_gene_placeholder(conversations):
    """
    如果样本带 gene，但第一条 user 消息里没有 <gene>，
    则强制插入到开头：
        <gene>\n原始问题
    """
    conversations = copy.deepcopy(conversations)
    assert len(conversations) > 0 and conversations[0]["role"] == "user"

    user_text = conversations[0]["content"]
    if "<gene>" not in user_text:
        conversations[0]["content"] = "<gene>\n" + user_text
    return conversations


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        raw_data,
        transform,
        tokenizer,
        gene_tokenizer,
        slice_config,
        llm_type="minicpm",
        patch_size=14,
        query_nums=64,
        batch_vision=False,
        max_length=2048,
    ):
        super(SupervisedDataset, self).__init__()
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.gene_tokenizer = gene_tokenizer
        self.transform = transform
        self.slice_config = slice_config
        self.llm_type = llm_type
        self.patch_size = patch_size
        self.query_nums = query_nums
        self.batch_vision = batch_vision
        self.max_length = max_length
        ensure_gene_special_token(self.tokenizer)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            sample = self.raw_data[i]
            has_image = "image" in sample and sample["image"] is not None
            has_gene = "gene" in sample and sample["gene"] is not None

            conversations = copy.deepcopy(sample["conversations"])
            if has_gene:
                conversations = ensure_gene_placeholder(conversations)

            # ====== gene ======
            gene_input_ids = torch.empty(0, dtype=torch.long)
            gene_bound = []

            if has_gene:
                adata = anndata.read_h5ad(sample["gene"])
                gene_names = adata.var_names.tolist()
                gene_tokens = self.gene_tokenizer(
                    gene_names,
                    max_length=1500,
                    padding=True,
                    truncation=True,
                )
                gene_input_ids = torch.tensor(gene_tokens["input_ids"], dtype=torch.long)

            # ====== image/text ======
            pixel_values = []
            tgt_sizes = torch.empty(0, 2, dtype=torch.long)
            image_bound = []

            if has_image:
                if isinstance(sample["image"], str):
                    images_dict = {"<image>": Image.open(sample["image"]).convert("RGB")}
                elif isinstance(sample["image"], Dict):
                    images_dict = {
                        img_name: Image.open(img_path).convert("RGB")
                        for img_name, img_path in sample["image"].items()
                    }
                else:
                    raise ValueError(f"Invalid image format at index {i}")

                image_ret = preprocess(
                    images_dict,
                    conversations,
                    self.tokenizer,
                    self.transform,
                    query_nums=self.query_nums,
                    slice_config=self.slice_config,
                    llm_type=self.llm_type,
                    patch_size=self.patch_size,
                    batch_vision=self.batch_vision,
                    max_length=self.max_length,
                )

                pixel_values = image_ret["pixel_values"]
                tgt_sizes = image_ret["tgt_sizes"]
                image_bound = image_ret["image_bound"]

                # supervised 单样本默认取 preprocess 返回的第一个候选
                input_ids = image_ret["input_ids"][0]
                position_ids = image_ret["position_ids"][0]
                labels = image_ret["target"][0]
                if isinstance(image_bound, list) and len(image_bound) > 0:
                    image_bound = image_bound[0]

            else:
                text_ret = conversation_to_ids(
                    conversations,
                    self.tokenizer,
                    self.llm_type,
                    max_length=self.max_length,
                )
                input_ids = text_ret["input_ids"]
                position_ids = text_ret["position_ids"]
                labels = text_ret["target"]
                image_bound = []
                pixel_values = []
                tgt_sizes = torch.empty(0, 2, dtype=torch.long)

            # ====== find gene bound from tokenized ids ======
            if has_gene:
                gene_token_id = self.tokenizer.convert_tokens_to_ids("<gene>")
                if gene_token_id is None:
                    raise ValueError("tokenizer cannot find <gene> token id")

                gene_indices = torch.where(input_ids == gene_token_id)[0]
                if len(gene_indices) == 1:
                    start = gene_indices[0].item()
                    gene_bound = torch.tensor([[start, start + 1]], dtype=torch.long)
                else:
                    logger.warning(
                        f"Sample {i}: expected exactly one <gene> token, got {len(gene_indices)}"
                    )
                    gene_bound = []

            ret = dict(
                input_ids=input_ids,
                position_ids=position_ids,
                labels=labels,
                attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
                pixel_values=pixel_values,
                tgt_sizes=tgt_sizes,
                image_bound=image_bound,
                gene_input_ids=gene_input_ids,
                gene_bound=gene_bound,
            )
            return ret

        except Exception as e:
            logger.error(f"Data fetch error at index {i}: {e}", exc_info=True)
            if len(self) > 1:
                return self.__getitem__(random.randint(0, len(self) - 1))
            raise


class GSPODataset(Dataset):
    def __init__(
        self,
        raw_data,
        transform,
        tokenizer,
        gene_tokenizer,
        slice_config,
        llm_type="minicpm",
        patch_size=14,
        query_nums=64,
        batch_vision=False,
        max_length=2048,
    ):
        super(GSPODataset, self).__init__()
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.gene_tokenizer = gene_tokenizer
        self.transform = transform
        self.slice_config = slice_config
        self.llm_type = llm_type
        self.patch_size = patch_size
        self.query_nums = query_nums
        self.batch_vision = batch_vision
        self.max_length = max_length
        ensure_gene_special_token(self.tokenizer)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            sample = self.raw_data[i]
            has_image = "image" in sample and sample["image"] is not None
            has_gene = "gene" in sample and sample["gene"] is not None

            conversations = copy.deepcopy(sample["conversations"])
            if has_gene:
                conversations = ensure_gene_placeholder(conversations)

            # ====== gene ======
            gene_input_ids = torch.empty(0, dtype=torch.long)
            if has_gene:
                adata = anndata.read_h5ad(sample["gene"])
                gene_names = adata.var_names.tolist()
                gene_tokens = self.gene_tokenizer(
                    gene_names,
                    max_length=1500,
                    padding=True,
                    truncation=True,
                )
                gene_input_ids = torch.tensor(gene_tokens["input_ids"], dtype=torch.long)

            # ====== image/text ======
            pixel_values = []
            tgt_sizes = torch.empty(0, 2, dtype=torch.long)
            image_bound = []
            all_scores = []

            if has_image:
                if isinstance(sample["image"], str):
                    images_dict = {"<image>": Image.open(sample["image"]).convert("RGB")}
                elif isinstance(sample["image"], Dict):
                    images_dict = {
                        img_name: Image.open(img_path).convert("RGB")
                        for img_name, img_path in sample["image"].items()
                    }
                else:
                    raise ValueError(f"Invalid image format at index {i}")

                image_ret = preprocess(
                    images_dict,
                    conversations,
                    self.tokenizer,
                    self.transform,
                    query_nums=self.query_nums,
                    slice_config=self.slice_config,
                    llm_type=self.llm_type,
                    patch_size=self.patch_size,
                    batch_vision=self.batch_vision,
                    max_length=self.max_length,
                )

                pixel_values = image_ret["pixel_values"]
                tgt_sizes = image_ret["tgt_sizes"]
                image_bound = image_ret["image_bound"]

                all_input_ids = image_ret["input_ids"]
                all_position_ids = image_ret["position_ids"]
                all_labels = image_ret["target"]
                all_scores = image_ret["scores"]

            else:
                assistant_convs = [c for c in conversations[1:] if c["role"] == "assistant"]
                assert len(assistant_convs) > 0, f"Sample {i} has no assistant conversation."

                all_input_ids = []
                all_position_ids = []
                all_labels = []
                all_scores = []

                for assist in assistant_convs:
                    temp_conv = [conversations[0], assist]
                    text_ret = conversation_to_ids(
                        temp_conv,
                        self.tokenizer,
                        self.llm_type,
                        max_length=self.max_length,
                    )
                    all_input_ids.append(text_ret["input_ids"])
                    all_position_ids.append(text_ret["position_ids"])
                    all_labels.append(text_ret["target"])
                    all_scores.append(float(assist.get("score", 0.0)))

                all_scores = torch.tensor(all_scores, dtype=torch.float)
                image_bound = [[] for _ in all_input_ids]
                pixel_values = []
                tgt_sizes = torch.empty(0, 2, dtype=torch.long)

            # ====== find gene bound for each candidate ======
            all_gene_bounds = []
            if has_gene:
                gene_token_id = self.tokenizer.convert_tokens_to_ids("<gene>")
                if gene_token_id is None:
                    raise ValueError("tokenizer cannot find <gene> token id")

                for cur_input_ids in all_input_ids:
                    g_indices = torch.where(cur_input_ids == gene_token_id)[0]
                    if len(g_indices) == 1:
                        top = g_indices[0].item()
                        bound = torch.tensor([[top, top + 1]], dtype=torch.long)
                        all_gene_bounds.append(bound)
                    else:
                        logger.warning(
                            f"Sample {i}: expected exactly one <gene> token, got {len(g_indices)}"
                        )
                        all_gene_bounds.append([])
            else:
                all_gene_bounds = [[] for _ in all_input_ids]

            ret = dict(
                input_ids=all_input_ids,
                position_ids=all_position_ids,
                labels=all_labels,
                attention_mask=[torch.ones_like(ids, dtype=torch.bool) for ids in all_input_ids],
                pixel_values=pixel_values,
                tgt_sizes=tgt_sizes,
                image_bound=image_bound,
                scores=all_scores,
                gene_input_ids=gene_input_ids,
                gene_bound=all_gene_bounds,
            )
            return ret

        except Exception as e:
            logger.error(f"Data fetch error at index {i}: {e}", exc_info=True)
            if len(self) > 1:
                return self.__getitem__(random.randint(0, len(self) - 1))
            raise


def data_collator(examples, padding_value=0, max_length=2048, gene_pad_id=0):
    def trim_and_pad(seq, batch_first, padding_value):
        return pad_sequence([s[:max_length] for s in seq], batch_first=True, padding_value=padding_value)

    input_ids = trim_and_pad(
        [example["input_ids"] for example in examples],
        batch_first=True,
        padding_value=padding_value,
    )
    position_ids = trim_and_pad(
        [example["position_ids"] for example in examples],
        batch_first=True,
        padding_value=padding_value,
    )
    targets = trim_and_pad(
        [example["labels"] for example in examples],
        batch_first=True,
        padding_value=-100,
    )
    attention_mask = trim_and_pad(
        [example["attention_mask"] for example in examples],
        batch_first=True,
        padding_value=padding_value,
    )
    pixel_values = [example["pixel_values"] for example in examples]
    image_bound = [example["image_bound"] for example in examples]
    tgt_sizes = [example["tgt_sizes"] for example in examples]

    gene_input_ids_list = [e["gene_input_ids"] for e in examples]
    has_genes = any(g.numel() > 0 for g in gene_input_ids_list)

    if has_genes:
        max_gene_len = max(len(g) for g in gene_input_ids_list if g.numel() > 0)
        padded_gene_ids = []
        for g in gene_input_ids_list:
            if g.numel() == 0:
                padded_gene_ids.append(torch.full((max_gene_len,), gene_pad_id, dtype=torch.long))
            else:
                pad_len = max_gene_len - len(g)
                padded_gene_ids.append(torch.nn.functional.pad(g, (0, pad_len), value=gene_pad_id))

        gene_input_ids = torch.stack(padded_gene_ids)
        gene_attention_mask = (gene_input_ids != gene_pad_id).long()
    else:
        gene_input_ids = None
        gene_attention_mask = None

    gene_bound = [example["gene_bound"] for example in examples]

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "labels": targets,
        "attention_mask": attention_mask,
        "image_bound": image_bound,
        "tgt_sizes": tgt_sizes,
        "pixel_values": pixel_values,
        "gene_input_ids": gene_input_ids,
        "gene_attention_mask": gene_attention_mask,
        "gene_bound": gene_bound,
    }


def gspo_data_collator(examples, padding_value=0, max_length=2048, gene_pad_id=0):
    flat_input_ids = []
    flat_labels = []
    flat_position_ids = []
    flat_attention_mask = []
    flat_scores = []
    candidate_counts = []

    flat_pixel_values = []
    flat_tgt_sizes = []
    flat_image_bounds = []
    flat_gene_input_ids = []
    flat_gene_bounds = []

    for example in examples:
        candidate_counts.append(len(example["input_ids"]))
        num_candidates = len(example["input_ids"])

        flat_input_ids.extend(example["input_ids"])
        flat_labels.extend(example["labels"])
        flat_position_ids.extend(example["position_ids"])
        flat_attention_mask.extend(example["attention_mask"])
        flat_scores.extend(example["scores"])

        for _ in range(num_candidates):
            flat_pixel_values.append(example["pixel_values"])
            flat_tgt_sizes.append(example["tgt_sizes"])
            flat_gene_input_ids.append(example["gene_input_ids"])

        flat_gene_bounds.extend(example["gene_bound"])
        flat_image_bounds.extend(example["image_bound"])

    def trim_and_pad(seq, batch_first, padding_value):
        return pad_sequence([s[:max_length] for s in seq], batch_first=True, padding_value=padding_value)

    batch_input_ids = trim_and_pad(
        flat_input_ids,
        batch_first=True,
        padding_value=padding_value,
    )
    batch_position_ids = trim_and_pad(
        flat_position_ids,
        batch_first=True,
        padding_value=padding_value,
    )
    batch_labels = trim_and_pad(
        flat_labels,
        batch_first=True,
        padding_value=-100,
    )
    batch_attention_mask = trim_and_pad(
        flat_attention_mask,
        batch_first=True,
        padding_value=padding_value,
    )
    batch_scores = torch.tensor(flat_scores, dtype=torch.float)

    has_genes = any(g.numel() > 0 for g in flat_gene_input_ids)

    if has_genes:
        max_gene_len = max(len(g) for g in flat_gene_input_ids if g.numel() > 0)
        padded_gene_ids = []
        for g in flat_gene_input_ids:
            if g.numel() == 0:
                padded_gene_ids.append(torch.full((max_gene_len,), gene_pad_id, dtype=torch.long))
            else:
                pad_len = max_gene_len - len(g)
                padded_gene_ids.append(torch.nn.functional.pad(g, (0, pad_len), value=gene_pad_id))

        batch_gene_input_ids = torch.stack(padded_gene_ids)
        batch_gene_attention_mask = (batch_gene_input_ids != gene_pad_id).long()
    else:
        batch_gene_input_ids = None
        batch_gene_attention_mask = None

    return {
        "input_ids": batch_input_ids,
        "position_ids": batch_position_ids,
        "labels": batch_labels,
        "attention_mask": batch_attention_mask,
        "candidate_counts": candidate_counts,
        "image_bound": flat_image_bounds,
        "tgt_sizes": flat_tgt_sizes,
        "pixel_values": flat_pixel_values,
        "scores": batch_scores,
        "gene_input_ids": batch_gene_input_ids,
        "gene_attention_mask": batch_gene_attention_mask,
        "gene_bound": flat_gene_bounds,
    }


def conversation_to_ids(conversation, tokenizer, llm_type=None, new_schema=False, max_length=2048):
    """
    for single image multi-turn conversation
    conversation: [{'role': 'user', 'content': 'Describe this image'},
                   {'role': 'assistant', 'content': 'This is a cat.'}]
    """
    if llm_type == "llama3":
        input_ids, context, raw_msg = conversation_to_ids_llama3(conversation, tokenizer)
    elif llm_type == "qwen":
        input_ids, context, raw_msg = conversation_to_ids_qwen2(conversation, tokenizer)
    else:
        input_ids, context, raw_msg = conversation_to_ids_minicpm(conversation, tokenizer)

    ids = torch.from_numpy(np.hstack(input_ids).astype(np.int64))
    context = torch.from_numpy(np.hstack(context, dtype=np.int8))

    if ids.shape[-1] > max_length:
        ids = ids[:max_length]
        context = context[:max_length]
        logger.warning(
            f"The input length ({len(ids)}) exceeds the model's maximum length ({max_length}), so it has been truncated"
        )

    if torch.all(context):
        logger.error("No tokens available to compute loss.")
        raise Exception("No tokens available to compute loss.")

    target = torch.full_like(ids, -100, dtype=torch.int32)

    for i in range(1, len(ids)):
        if context[i] == 0:
            target[i - 1] = ids[i]
        if context[i] == 1 and context[i - 1] == 0:
            if hasattr(tokenizer, "eot_id"):
                target[i - 1] = tokenizer.eot_id
            else:
                target[i - 1] = tokenizer.eos_id

    if new_schema:
        start_cond = (ids == tokenizer.im_start_id) | (ids == tokenizer.slice_start_id)
        end_cond = (ids == tokenizer.im_end_id) | (ids == tokenizer.slice_end_id)
        image_start_tokens = torch.where(start_cond)[0]
        image_start_tokens += 1
        image_end_tokens = torch.where(end_cond)[0]
    else:
        image_start_tokens = torch.where(ids == tokenizer.im_start_id)[0]
        image_start_tokens += 1
        image_end_tokens = torch.where(ids == tokenizer.im_end_id)[0]

    if len(image_start_tokens) != len(image_end_tokens):
        logger.error("image start token != image end tokens")
        raise Exception("image start token != image end tokens")

    if len(image_start_tokens) > 0:
        image_bound = torch.hstack([image_start_tokens.unsqueeze(-1), image_end_tokens.unsqueeze(-1)])
    else:
        image_bound = []

    position_ids = torch.arange(ids.size(0)).long()
    return {
        "input_ids": ids,
        "target": target,
        "image_bound": image_bound,
        "raw_msg": raw_msg,
        "position_ids": position_ids,
    }


def conversation_to_ids_minicpm(conversation, tokenizer):
    raw_msg = ""
    input_ids = []
    context = []

    bos_id = getattr(tokenizer, "bos_token_id", None)

    for idx, msg in enumerate(conversation):
        role = msg["role"]
        message = msg["content"]
        assert role in ["user", "assistant"]

        prefix = "<用户>" if role == "user" else "<AI>"

        # append eos to last turn
        if idx == len(conversation) - 1:
            message = message + tokenizer.eos_token

        # 关键修复：不要盲目 [1:]，只在首 token 真的是 bos 时才去掉
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
        if bos_id is not None and len(prefix_ids) > 0 and prefix_ids[0] == bos_id:
            prefix_ids = prefix_ids[1:]

        message_ids = tokenizer.encode(message, add_special_tokens=True)
        if bos_id is not None and len(message_ids) > 0 and message_ids[0] == bos_id:
            message_ids = message_ids[1:]

        input_ids.append(prefix_ids)
        input_ids.append(message_ids)

        context.append(np.ones((len(prefix_ids),), dtype=np.int8))
        if role == "assistant":
            context.append(np.zeros((len(message_ids),), dtype=np.int8))
        else:
            context.append(np.ones((len(message_ids),), dtype=np.int8))

        raw_msg += prefix + message

    return input_ids, context, raw_msg

def conversation_to_ids_llama3(conversation, tokenizer):
    raw_msg = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False,
        chat_template=llama3_chat_template,
    )
    input_ids = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=False,
        chat_template=llama3_chat_template,
    )
    input_ids = np.array(input_ids)

    start_header_idxs = np.where(input_ids == tokenizer.convert_tokens_to_ids("<|start_header_id|>"))[0]
    assistant_idxs = np.where(input_ids == tokenizer.convert_tokens_to_ids("assistant"))[0]
    end_header_idxs = np.where(input_ids == tokenizer.convert_tokens_to_ids("<|end_header_id|>"))[0]
    eot_idxs = np.where(input_ids == tokenizer.convert_tokens_to_ids("<|eot_id|>"))[0]

    context = np.ones_like(input_ids, dtype=np.int8)

    for assistant_idx in assistant_idxs:
        if assistant_idx in set((start_header_idxs + end_header_idxs) / 2):
            st = assistant_idx + 3
            for eot_idx in eot_idxs:
                if eot_idx > st:
                    context[st: eot_idx + 1] = 0
                    break

    input_ids = np.hstack(input_ids)
    context = np.hstack(context)

    return input_ids, context, raw_msg


def conversation_to_ids_qwen2(conversation, tokenizer):
    raw_msg = ""
    chat = []
    for msg in conversation:
        role = msg["role"]
        message = msg["content"]
        assert role in ["user", "assistant"]
        prefix = "user" if role == "user" else "assistant"
        chat.append({"role": prefix, "content": message})
        raw_msg += prefix + message

    assert set([i["role"] for i in chat]) & set(["assistant"])

    input_ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=False)
    input_ids = np.array(input_ids)

    start_idxs = np.where(input_ids == tokenizer.convert_tokens_to_ids("<|im_start|>"))[0]
    assistant_idxs = np.where(input_ids == tokenizer.convert_tokens_to_ids("assistant"))[0]
    end_idxs = np.where(input_ids == tokenizer.convert_tokens_to_ids("<|im_end|>"))[0]

    context = np.ones_like(input_ids, dtype=np.int8)

    for assistant_idx in assistant_idxs:
        if assistant_idx - 1 in set(start_idxs):
            st = assistant_idx + 1
            for end_idx in end_idxs:
                if end_idx > st:
                    context[st: end_idx + 1] = 0
                    break

    input_ids = np.hstack(input_ids)
    context = np.hstack(context)
    return input_ids, context, raw_msg


def preprocess(
    images_dict,
    conversations,
    tokenizer,
    transform,
    query_nums=64,
    slice_config=None,
    llm_type=None,
    patch_size=14,
    batch_vision=False,
    max_length=2048,
):
    """
    single(multi) image(s) preprocess, the image(s) will be placed at the top of the conversation
    """
    conversations = copy.deepcopy(conversations)
    assert len(conversations) > 1, "conversations length must large than 2"
    assert conversations[0]["role"] == "user", "the first role must be user"

    user_conv = conversations[0]
    assistant_convs = [c for c in conversations[1:] if c["role"] == "assistant"]

    if slice_config is not None:
        assert isinstance(slice_config, Dict)
        assert "patch_size" in slice_config
        assert "max_slice_nums" in slice_config
        assert "scale_resolution" in slice_config

    default_image_placeholder = tokenizer.im_start + tokenizer.unk_token * query_nums + tokenizer.im_end

    new_schema = False
    use_image_id = False
    if llm_type == "qwen":
        new_schema = True
        use_image_id = True

    image_placeholder_dict = {}
    images = []
    image_id_cnt = 0

    for img_name, image in images_dict.items():
        if slice_config:
            source_image, patches, best_grid = slice_image(
                image,
                slice_config["max_slice_nums"],
                slice_config["scale_resolution"],
                slice_config["patch_size"],
            )
            images.append(source_image)
            image_placeholder = default_image_placeholder

            if len(patches) > 0:
                for i in range(len(patches)):
                    for j in range(len(patches[0])):
                        images.append(patches[i][j])

                if use_image_id:
                    image_placeholder = (
                        f"{tokenizer.im_id_start}{image_id_cnt}{tokenizer.im_id_end}" + image_placeholder
                    )
                    image_id_cnt += 1

                image_placeholder += get_grid_placeholder(
                    tokenizer, best_grid, query_nums, new_schema=new_schema
                )

            image_placeholder_dict[img_name] = image_placeholder
        else:
            images.append(image)
            if use_image_id:
                image_placeholder = (
                    f"{tokenizer.im_id_start}{image_id_cnt}{tokenizer.im_id_end}" + default_image_placeholder
                )
                image_id_cnt += 1
            else:
                image_placeholder = default_image_placeholder
            image_placeholder_dict[img_name] = image_placeholder

    images = [transform(i) for i in images]

    if len(images_dict) == 1 and "<image>" in images_dict:
        if "<image>" in conversations[0]["content"]:
            user_text_processed = user_conv["content"].replace("<image>", image_placeholder_dict["<image>"])
        else:
            user_text_processed = image_placeholder_dict["<image>"] + "\n" + user_conv["content"]
    else:
        pattern = r"<image_\d+>"
        content = conversations[0]["content"]
        parts = re.split(f"({pattern})", content)
        for i, part in enumerate(parts):
            if not part.strip():
                continue
            if re.match(pattern, part):
                if part in image_placeholder_dict:
                    parts[i] = image_placeholder_dict[part]
                else:
                    raise Exception(f"not found {part} in image dict")
        user_text_processed = "\n".join(parts)

    user_conv_processed = {"role": "user", "content": user_text_processed}

    all_input_ids = []
    all_targets = []
    all_scores = []
    all_image_bounds = []
    all_position_ids = []

    for assist in assistant_convs:
        temp_conv = [user_conv_processed, assist]
        convers_ret = conversation_to_ids(
            temp_conv, tokenizer, llm_type, new_schema, max_length
        )

        all_input_ids.append(convers_ret["input_ids"])
        all_targets.append(convers_ret["target"])
        all_image_bounds.append(convers_ret["image_bound"])
        all_position_ids.append(convers_ret["position_ids"])
        all_scores.append(float(assist.get("score", 0.0)))

    if batch_vision:
        tgt_sizes = []
        reshape_images = []
        for image in images:
            H, W = image.shape[1:]
            reshape_image = reshape_by_patch(image, patch_size)
            reshape_images.append(reshape_image)
            tgt_sizes.append([H // patch_size, W // patch_size])
        if tgt_sizes:
            tgt_sizes = torch.Tensor(tgt_sizes).type(torch.int32)

        final_pixel_values = reshape_images
        final_tgt_sizes = tgt_sizes
    else:
        final_pixel_values = images
        final_tgt_sizes = []

    return {
        "input_ids": all_input_ids,
        "target": all_targets,
        "position_ids": all_position_ids,
        "image_bound": all_image_bounds,
        "scores": torch.tensor(all_scores, dtype=torch.float),
        "pixel_values": final_pixel_values,
        "tgt_sizes": final_tgt_sizes,
    }


def slice_image(image, max_slice_nums=9, scale_resolution=448, patch_size=14, never_split=False):
    original_size = image.size
    original_width, original_height = original_size
    log_ratio = math.log(original_width / original_height)
    ratio = original_width * original_height / (scale_resolution * scale_resolution)
    multiple = min(math.ceil(ratio), max_slice_nums)

    source_image = None
    best_grid = None
    patches = []

    if multiple <= 1 or never_split:
        best_size = find_best_resize(
            original_size, scale_resolution, patch_size, allow_upscale=True
        )
        source_image = image.resize(best_size, Image.Resampling.BICUBIC)
    else:
        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i == 1 or i > max_slice_nums:
                continue
            candidate_split_grids_nums.append(i)

        best_resize = find_best_resize(original_size, scale_resolution, patch_size)
        source_image = image.copy().resize(best_resize, Image.Resampling.BICUBIC)
        candidate_grids = []

        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([m, split_grids_nums // m])
                m += 1

        best_grid = [1, 1]
        min_error = float("inf")
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error

        refine_size = get_refine_size(
            original_size, best_grid, scale_resolution, patch_size, allow_upscale=True
        )

        refine_image = image.resize(refine_size, Image.Resampling.BICUBIC)
        patches = split_to_patches(refine_image, best_grid)

    return source_image, patches, best_grid


def ensure_divide(length, patch_size):
    return max(round(length / patch_size) * patch_size, patch_size)


def find_best_resize(original_size, scale_resolution, patch_size, allow_upscale=False):
    width, height = original_size
    if (width * height > scale_resolution * scale_resolution) or allow_upscale:
        r = width / height
        height = int(scale_resolution / math.sqrt(r))
        width = int(height * r)
    best_width = ensure_divide(width, patch_size)
    best_height = ensure_divide(height, patch_size)
    return (best_width, best_height)


def get_refine_size(original_size, grid, scale_resolution, patch_size, allow_upscale=False):
    width, height = original_size
    grid_x, grid_y = grid

    refine_width = ensure_divide(width, grid_x)
    refine_height = ensure_divide(height, grid_y)

    grid_width = refine_width / grid_x
    grid_height = refine_height / grid_y

    best_grid_size = find_best_resize(
        (grid_width, grid_height),
        scale_resolution,
        patch_size,
        allow_upscale=allow_upscale,
    )

    refine_size = (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)
    return refine_size


def split_to_patches(image, grid):
    patches = []
    width, height = image.size
    grid_x = int(width / grid[0])
    grid_y = int(height / grid[1])

    for i in range(0, height, grid_y):
        images = []
        for j in range(0, width, grid_x):
            box = (j, i, j + grid_x, i + grid_y)
            patch = image.crop(box)
            images.append(patch)
        patches.append(images)

    return patches


def get_grid_placeholder(tokenizer, grid, query_num, new_schema=False):
    if new_schema:
        image_placeholder = tokenizer.slice_start + tokenizer.unk_token * query_num + tokenizer.slice_end
    else:
        image_placeholder = tokenizer.im_start + tokenizer.unk_token * query_num + tokenizer.im_end

    cols = grid[0]
    rows = grid[1]
    slices = []
    for i in range(rows):
        lines = []
        for j in range(cols):
            lines.append(image_placeholder)
        slices.append("".join(lines))

    if new_schema:
        slice_placeholder = "\n".join(slices)
    else:
        slice_placeholder = tokenizer.slice_start + "\n".join(slices) + tokenizer.slice_end

    return slice_placeholder


def reshape_by_patch(image_tensor, patch_size):
    """
    :param image_tensor: shape [3, H, W]
    :param patch_size:
    :return: [3, patch_size, HW/patch_size]
    """
    patches = torch.nn.functional.unfold(
        image_tensor, (patch_size, patch_size), stride=(patch_size, patch_size)
    )

    patches = patches.reshape(image_tensor.size(0), patch_size, patch_size, -1)
    patches = patches.permute(0, 1, 3, 2).reshape(
        image_tensor.size(0), patch_size, -1
    )
    return patches