#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from c2s_lora_sft import LABEL_HINT, USER_PROMPT, build_prompt


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


def _env_int_or_none(name: str, default: Optional[int]) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    if raw == "" or raw.lower() in {"none", "all", "full"}:
        return None
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


SLIDE_ID = os.environ.get("C2S_EVAL_SLIDE_ID", "151508")
BASE_ST_DIR = Path(f"/data1/xiaoxinyu/benchmark/DLPFC/stRNA/10xformat/{SLIDE_ID}")
SPATIAL_DIR = BASE_ST_DIR / "spatial"

H5AD_PATH = BASE_ST_DIR / "filtered_feature_bc_matrix.h5ad"
POS_PATH = SPATIAL_DIR / "tissue_positions_list.txt"
TRUTH_PATH = Path(f"/data1/xiaoxinyu/benchmark/DLPFC/annotation/{SLIDE_ID}_truth.txt")

MODEL_PATH = _env_path("C2S_MODEL_PATH", "/data1/xiaoxinyu/SOTAModel/C2S")
LORA_PATH = os.environ.get("C2S_LORA_PATH", "").strip()
OUT_DIR = _env_path("C2S_OUT_DIR", "/data2/xiaoxinyu/project/eval-model/plot/fig4/c2s_gene_only")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_RAW_JSONL = OUT_DIR / f"{SLIDE_ID}_raw_answers.jsonl"
OUT_RAW_CSV = OUT_DIR / f"{SLIDE_ID}_raw_answers.csv"
OUT_SCORED_CSV = OUT_DIR / f"{SLIDE_ID}_scored_answers.csv"
OUT_METRICS_JSON = OUT_DIR / f"{SLIDE_ID}_metrics.json"
OUT_PRED_TXT_GENE = OUT_DIR / f"{SLIDE_ID}_pred_gene.txt"

MAX_SPOTS = _env_int_or_none("C2S_MAX_SPOTS", None)
RESUME = _env_bool("C2S_RESUME", True)
MAX_GENES = int(os.environ.get("C2S_MAX_GENES", "512"))
MAX_NEW_TOKENS = int(os.environ.get("C2S_MAX_NEW_TOKENS", "512"))
CANONICAL_FULL_DESC = _env_bool("C2S_CANONICAL_FULL_DESC", True)
STOP_AFTER_LABEL_SENTENCE = _env_bool("C2S_STOP_AFTER_LABEL_SENTENCE", True)
DTYPE = os.environ.get("C2S_DTYPE", "bfloat16")

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

_STRUCT_TISSUE_PAT = re.compile(r"^\s*TISSUE\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
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


def get_dtype(name: str):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def canon_barcode(x: str) -> str:
    x = str(x).strip()
    x = x.split(",")[0]
    m = re.match(r"^([A-Z0-9]+-\d+)", x)
    if m:
        return m.group(1)
    return x


def normalize_barcode_df(df: pd.DataFrame, col: str = "barcode") -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].astype(str).map(canon_barcode)
    return df


def load_truth_txt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.iloc[:, :2].copy()
    df.columns = ["barcode", "label"]
    df["barcode"] = df["barcode"].astype(str)
    df["label"] = df["label"].astype(str)
    return normalize_barcode_df(df, "barcode")


def load_positions_txt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"[\t,]", header=None, engine="python")
    df = df.iloc[:, :6].copy()
    df.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
    df["barcode"] = df["barcode"].astype(str)
    for c in ["in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["barcode"]).copy()
    return normalize_barcode_df(df, "barcode")


def label_from_match(match: re.Match) -> Optional[str]:
    layer_num = match.group(1)
    layer_label = match.group(2)
    wm_label = match.group(3)
    if layer_num:
        return f"Layer_{layer_num}"
    if layer_label:
        return layer_label[:1].upper() + layer_label[1:].lower().replace("_", "_")
    if wm_label:
        return "WM"
    return None


def extract_layer_label(text: str) -> Optional[str]:
    if not text:
        return None
    candidates = []
    m = _STRUCT_TISSUE_PAT.search(text)
    if m:
        candidates.append(m.group(1))
    candidates.append(text)

    for segment in candidates:
        m_cn = _CN_LAYER_PAT.search(segment)
        if m_cn:
            cn_name, layer_num = m_cn.group(1), m_cn.group(2)
            if "白质" in cn_name:
                return "WM"
            if layer_num:
                return f"Layer_{layer_num}"
            label = _CN_LAYER_TO_LABEL.get(cn_name)
            if label:
                return label

        m_layer = _LAYER_TOKEN_PAT.search(segment)
        if m_layer:
            label = label_from_match(m_layer)
            if label in LABELS:
                return label
    return None


def canonical_answer_from_raw(answer: str) -> str:
    label = extract_layer_label(answer)
    if not label:
        return answer
    return LABEL_DESCRIPTIONS[label]


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


def load_done_barcodes(jsonl_path: Path) -> set:
    done = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["barcode"])
            except Exception:
                pass
    return done


def append_jsonl(path: Path, rows: List[Dict]):
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def cell_sentence_from_row(adata: ad.AnnData, obs_idx: int, max_genes: int) -> str:
    row = adata.X[obs_idx]
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
    return " ".join(str(adata.var_names[i]).upper() for i in order)


def generate_answer(model, tokenizer, prompt: str, device: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    stopping_criteria = None
    if STOP_AFTER_LABEL_SENTENCE:
        stopping_criteria = StoppingCriteriaList(
            [StopAfterLayerSentence(tokenizer, inputs["input_ids"].shape[1])]
        )
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )
    new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


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


def score_and_export(rows: List[Dict]):
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
        "model_path": str(MODEL_PATH),
        "lora_path": str(LORA_PATH),
        "max_spots": MAX_SPOTS,
        "max_genes": MAX_GENES,
        "max_new_tokens": MAX_NEW_TOKENS,
        "canonical_full_description": CANONICAL_FULL_DESC,
        "stop_after_label_sentence": STOP_AFTER_LABEL_SENTENCE,
        "prompt": USER_PROMPT,
        "label_hint": LABEL_HINT,
        "modalities": {
            "gene": compute_label_metrics(df["truth"], df["pred_gene"]),
        },
        "outputs": {
            "raw_jsonl": str(OUT_RAW_JSONL),
            "raw_csv": str(OUT_RAW_CSV),
            "scored_csv": str(OUT_SCORED_CSV),
            "pred_gene_txt": str(OUT_PRED_TXT_GENE),
        },
    }
    with OUT_METRICS_JSON.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


def main():
    if not LORA_PATH:
        raise ValueError("Please set C2S_LORA_PATH to the trained LoRA checkpoint or output directory.")

    print("[1/5] Loading DLPFC data...")
    adata = ad.read_h5ad(H5AD_PATH)
    truth_df = load_truth_txt(TRUTH_PATH)
    pos_df = load_positions_txt(POS_PATH)

    obs_df = pd.DataFrame({"obs_idx": np.arange(adata.n_obs), "barcode": adata.obs_names.astype(str)})
    obs_df = normalize_barcode_df(obs_df, "barcode")
    merged = obs_df.merge(truth_df, on="barcode", how="inner").merge(pos_df, on="barcode", how="inner")
    merged = merged[merged["in_tissue"] == 1].copy().reset_index(drop=True)
    if MAX_SPOTS is not None:
        merged = merged.iloc[:MAX_SPOTS].copy()
    print("[INFO] total spots to infer =", len(merged))

    print("[2/5] Loading C2S + LoRA...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = get_dtype(DTYPE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    model = PeftModel.from_pretrained(base_model, LORA_PATH).to(device)
    model.eval()

    done = load_done_barcodes(OUT_RAW_JSONL) if RESUME else set()
    if done:
        print(f"[INFO] resume mode: skip {len(done)} finished barcodes")

    print("[3/5] Running gene-only inference...")
    rows = []
    buffer = []
    for _, row in tqdm(merged.iterrows(), total=len(merged), dynamic_ncols=True):
        barcode = str(row["barcode"])
        if barcode in done:
            continue
        try:
            sentence = cell_sentence_from_row(adata, int(row["obs_idx"]), MAX_GENES)
            prompt = build_prompt(sentence)
            answer = generate_answer(model, tokenizer, prompt, device)
            raw_gene = canonical_answer_from_raw(answer) if CANONICAL_FULL_DESC else answer
            out = {
                "barcode": barcode,
                "truth": str(row["label"]),
                "raw_gene": raw_gene,
                "model_raw_gene": answer,
            }
        except Exception as e:
            out = {"barcode": barcode, "truth": str(row["label"]), "raw_gene": f"[ERROR] {repr(e)}"}
        buffer.append(out)
        if len(buffer) >= 20:
            append_jsonl(OUT_RAW_JSONL, buffer)
            buffer = []

    if buffer:
        append_jsonl(OUT_RAW_JSONL, buffer)

    print("[4/5] Scoring...")
    with OUT_RAW_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    pd.DataFrame(rows).to_csv(OUT_RAW_CSV, index=False, encoding="utf-8-sig")
    metrics = score_and_export(rows)

    print("[5/5] Done.")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
