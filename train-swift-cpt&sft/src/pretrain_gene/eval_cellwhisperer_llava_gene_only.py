#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList

sys.path.insert(0, "/data1/xiaoxinyu/SOTAModel/cellwhisperer/modules/LLaVA")

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM


SLIDE_ID = os.environ.get("CW_EVAL_SLIDE_ID", "151508")
BASE_MODEL = Path(os.environ.get("CW_BASE_MODEL", "/data1/xiaoxinyu/SOTAModel/cellwhisperer/results/models/llava"))
LORA_PATH = Path(os.environ.get("CW_LORA_PATH", "/data2/xiaoxinyu/project/pretrain-gene/sft_output/cellwhisperer_dlpfc_full_desc_0519_0107"))
FEATURE_NPZ = Path(os.environ.get("CW_FEATURE_NPZ", f"/data2/xiaoxinyu/project/pretrain-gene/cellwhisperer_eval_features/{SLIDE_ID}_cellwhisperer_features.npz"))
META_CSV = Path(os.environ.get("CW_META_CSV", f"/data2/xiaoxinyu/project/pretrain-gene/cellwhisperer_eval_features/{SLIDE_ID}_metadata.csv"))
OUT_DIR = Path(os.environ.get("CW_OUT_DIR", f"/data2/xiaoxinyu/project/eval-model/plot/fig4/cellwhisperer_dlpfc_full_desc_0519_0107_gene_only"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_RAW_JSONL = OUT_DIR / f"{SLIDE_ID}_raw_answers.jsonl"
OUT_RAW_CSV = OUT_DIR / f"{SLIDE_ID}_raw_answers.csv"
OUT_SCORED_CSV = OUT_DIR / f"{SLIDE_ID}_scored_answers.csv"
OUT_METRICS_JSON = OUT_DIR / f"{SLIDE_ID}_metrics.json"
OUT_PRED_TXT_GENE = OUT_DIR / f"{SLIDE_ID}_pred_gene.txt"

USER_PROMPT = "请描述样本信息?"
MAX_NEW_TOKENS = int(os.environ.get("CW_MAX_NEW_TOKENS", "512"))
MAX_SPOTS_RAW = os.environ.get("CW_MAX_SPOTS")
MAX_SPOTS = None if not MAX_SPOTS_RAW or MAX_SPOTS_RAW.lower() in {"none", "all", "full"} else int(MAX_SPOTS_RAW)

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

_LAYER_TOKEN_PAT = re.compile(
    r"\b(?:Layer|layer|L)\s*[_ -]?\s*([1-6])\b|"
    r"\b(Layer_[1-6])\b|"
    r"\b(WM|white\s+matter)\b",
    re.IGNORECASE,
)
_CN_LAYER_PAT = re.compile(
    r"(分子层|外颗粒层|外锥体层|内颗粒层|内锥体层|多形层|白质层|白质)"
    r"(?:\s*[（(]?\s*(?:Layer|layer|L)\s*[_ -]?\s*([1-6])\s*[）)]?)?",
    re.IGNORECASE,
)
_CN_LAYER_TO_LABEL = {
    "分子层": "Layer_1",
    "外颗粒层": "Layer_2",
    "外锥体层": "Layer_3",
    "内颗粒层": "Layer_4",
    "内锥体层": "Layer_5",
    "多形层": "Layer_6",
    "白质层": "WM",
    "白质": "WM",
}


def extract_layer_label(text: str) -> Optional[str]:
    if not text:
        return None
    m_cn = _CN_LAYER_PAT.search(text)
    if m_cn:
        cn_name, layer_num = m_cn.group(1), m_cn.group(2)
        if "白质" in cn_name:
            return "WM"
        if layer_num:
            return f"Layer_{layer_num}"
        return _CN_LAYER_TO_LABEL.get(cn_name)
    m = _LAYER_TOKEN_PAT.search(text)
    if not m:
        return None
    if m.group(1):
        return f"Layer_{m.group(1)}"
    if m.group(2):
        return m.group(2)[:1].upper() + m.group(2)[1:].lower()
    if m.group(3):
        return "WM"
    return None


class StopAfterLayerSentence(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len: int):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        generated = input_ids[0, self.prompt_len :]
        if generated.numel() < 8:
            return False
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return extract_layer_label(text) is not None and ("。" in text or "\n" in text)


def compute_label_metrics(truth: pd.Series, pred: pd.Series) -> Dict[str, Any]:
    truth_arr = truth.astype(str).to_numpy()
    pred_arr = pred.astype(str).to_numpy()
    valid = np.isin(pred_arr, LABELS)
    correct = (truth_arr == pred_arr) & valid
    parsed_n = int(valid.sum())
    total_n = int(len(truth_arr))
    return {
        "n_total": total_n,
        "n_parsed": parsed_n,
        "n_unparsed": int(total_n - parsed_n),
        "n_correct": int(correct.sum()),
        "parse_rate": float(parsed_n / max(1, total_n)),
        "accuracy_overall": float(correct.sum() / max(1, total_n)),
        "accuracy_parsed_only": float(correct[valid].sum() / max(1, parsed_n)),
    }


def build_prompt() -> str:
    conv = conv_templates["mistral_instruct"].copy()
    conv.append_message(conv.roles[0], f"<image>\n{USER_PROMPT}")
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    model = LlavaLlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cuda:0",
    )
    non_lora = torch.load(LORA_PATH / "non_lora_trainables.bin", map_location="cpu")
    non_lora = {(k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora.items()}
    if any(k.startswith("model.model.") for k in non_lora):
        non_lora = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora.items()}
    model.load_state_dict(non_lora, strict=False)
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model = model.merge_and_unload()
    model.eval()
    return tokenizer, model


def generate_one(model, tokenizer, image_vec: np.ndarray) -> str:
    prompt = build_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda")
    image_tensor = torch.from_numpy(image_vec).to(device="cuda", dtype=torch.bfloat16).unsqueeze(0)
    stopping = StoppingCriteriaList([StopAfterLayerSentence(tokenizer, 0)])
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping,
        )
    if output_ids.shape[1] > input_ids.shape[1]:
        new_tokens = output_ids[0, input_ids.shape[1] :]
    else:
        new_tokens = output_ids[0]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    meta = pd.read_csv(META_CSV)
    z = np.load(FEATURE_NPZ, allow_pickle=True)
    embeds = z["transcriptome_embeds"]
    if MAX_SPOTS is not None:
        meta = meta.iloc[:MAX_SPOTS].copy()
        embeds = embeds[:MAX_SPOTS]

    tokenizer, model = load_model()
    rows: List[Dict[str, Any]] = []
    with OUT_RAW_JSONL.open("w", encoding="utf-8") as f:
        for i, row in tqdm(list(meta.iterrows()), total=len(meta), dynamic_ncols=True):
            try:
                answer = generate_one(model, tokenizer, embeds[i])
                out = {
                    "barcode": str(row["barcode"]),
                    "truth": str(row["label"]),
                    "raw_gene": answer,
                    "model_raw_gene": answer,
                }
            except Exception as e:
                out = {"barcode": str(row["barcode"]), "truth": str(row["label"]), "raw_gene": f"[ERROR] {repr(e)}", "model_raw_gene": ""}
            rows.append(out)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()

    pd.DataFrame(rows).to_csv(OUT_RAW_CSV, index=False, encoding="utf-8-sig")
    scored = []
    for row in rows:
        item = dict(row)
        item["pred_gene"] = extract_layer_label(item.get("raw_gene", "")) or "ERROR"
        item["correct_gene"] = int(item["pred_gene"] == item["truth"])
        scored.append(item)
    df = pd.DataFrame(scored)
    df.to_csv(OUT_SCORED_CSV, index=False, encoding="utf-8-sig")
    with OUT_PRED_TXT_GENE.open("w", encoding="utf-8") as f:
        for row in scored:
            f.write(f"{row['barcode']}\t{row['pred_gene']}\n")

    metrics = {
        "slide_id": SLIDE_ID,
        "base_model": str(BASE_MODEL),
        "lora_path": str(LORA_PATH),
        "feature_npz": str(FEATURE_NPZ),
        "metadata_csv": str(META_CSV),
        "prompt": USER_PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_spots": MAX_SPOTS,
        "modalities": {"gene": compute_label_metrics(df["truth"], df["pred_gene"])},
        "outputs": {
            "raw_jsonl": str(OUT_RAW_JSONL),
            "raw_csv": str(OUT_RAW_CSV),
            "scored_csv": str(OUT_SCORED_CSV),
            "pred_gene_txt": str(OUT_PRED_TXT_GENE),
        },
    }
    with OUT_METRICS_JSON.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
