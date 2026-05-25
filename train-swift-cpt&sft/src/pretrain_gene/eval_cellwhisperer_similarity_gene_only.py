#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import pickle
import re
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from tqdm.auto import tqdm


CELLWHISPERER_ROOT = Path(os.environ.get("CELLWHISPERER_ROOT", "/data1/xiaoxinyu/SOTAModel/cellwhisperer"))
CELLWHISPERER_SRC = CELLWHISPERER_ROOT / "src"
GENEFORMER_DIR = CELLWHISPERER_ROOT / "modules/Geneformer/geneformer"
GENE_MEDIAN_FILE = GENEFORMER_DIR / "gene_median_dictionary.pkl"
TOKEN_DICTIONARY_FILE = GENEFORMER_DIR / "token_dictionary.pkl"

SLIDE_ID = os.environ.get("CW_EVAL_SLIDE_ID", "151508")
BASE_ST_DIR = Path(f"/data1/xiaoxinyu/benchmark/DLPFC/stRNA/10xformat/{SLIDE_ID}")
SPATIAL_DIR = BASE_ST_DIR / "spatial"
H5AD_PATH = BASE_ST_DIR / "filtered_feature_bc_matrix.h5ad"
POS_PATH = SPATIAL_DIR / "tissue_positions_list.txt"
TRUTH_PATH = Path(f"/data1/xiaoxinyu/benchmark/DLPFC/annotation/{SLIDE_ID}_truth.txt")

MODEL_PATH = Path(
    os.environ.get(
        "CW_MODEL_PATH",
        str(CELLWHISPERER_ROOT / "results/models/jointemb/cellwhisperer_clip_v1.ckpt"),
    )
)
OUT_DIR = Path(
    os.environ.get(
        "CW_OUT_DIR",
        "/data2/xiaoxinyu/project/eval-model/plot/fig4/cellwhisperer_clip_v1_gene_only",
    )
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_RAW_JSONL = OUT_DIR / f"{SLIDE_ID}_raw_answers.jsonl"
OUT_RAW_CSV = OUT_DIR / f"{SLIDE_ID}_raw_answers.csv"
OUT_SCORED_CSV = OUT_DIR / f"{SLIDE_ID}_scored_answers.csv"
OUT_METRICS_JSON = OUT_DIR / f"{SLIDE_ID}_metrics.json"
OUT_PRED_TXT_GENE = OUT_DIR / f"{SLIDE_ID}_pred_gene.txt"
OUT_GENE_MAP = OUT_DIR / "ensembl_gene_symbol_map.csv"

ENSEMBL_H5AD = Path(
    os.environ.get(
        "CW_ENSEMBL_H5AD",
        "/data2/xiaoxinyu/project/model/gene_tokenizer/model-ensembl.h5ad",
    )
)
SYMBOL_H5AD = Path(
    os.environ.get(
        "CW_SYMBOL_H5AD",
        "/data2/xiaoxinyu/project/model/gene_tokenizer/model-symbel.h5ad",
    )
)

MAX_SPOTS = os.environ.get("CW_MAX_SPOTS")
MAX_SPOTS = None if not MAX_SPOTS or MAX_SPOTS.lower() in {"none", "all", "full"} else int(MAX_SPOTS)
BATCH_SIZE = int(os.environ.get("CW_BATCH_SIZE", "32"))
SCORE_BATCH_SIZE = int(os.environ.get("CW_SCORE_BATCH_SIZE", "128"))

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

_LAYER_TOKEN_PAT = re.compile(
    r"\b(?:Layer|layer|L)\s*[_ -]?\s*([1-6])\b|"
    r"\b(Layer_[1-6])\b|"
    r"\b(WM|white\s+matter)\b",
    re.IGNORECASE,
)


def install_geneformer_shims() -> None:
    if str(CELLWHISPERER_SRC) not in sys.path:
        sys.path.insert(0, str(CELLWHISPERER_SRC))

    with GENE_MEDIAN_FILE.open("rb") as f:
        gene_median_dict = pickle.load(f)
    with TOKEN_DICTIONARY_FILE.open("rb") as f:
        gene_token_dict = pickle.load(f)

    geneformer_pkg = types.ModuleType("geneformer")
    tokenizer_mod = types.ModuleType("geneformer.tokenizer")
    perturber_mod = types.ModuleType("geneformer.in_silico_perturber")
    emb_mod = types.ModuleType("geneformer.emb_extractor")
    scanpy_mod = types.ModuleType("scanpy")
    scanpy_pp_mod = types.SimpleNamespace()
    scanpy_queries_mod = types.SimpleNamespace()

    def calculate_qc_metrics(adata, inplace=True, *args, **kwargs):
        counts = np.asarray(adata.X.sum(axis=1)).reshape(-1)
        if inplace:
            adata.obs["total_counts"] = counts
            return None
        return pd.DataFrame({"total_counts": counts}, index=adata.obs_names), pd.DataFrame(index=adata.var_names)

    def biomart_annotations(*args, **kwargs):
        raise RuntimeError("biomart is disabled; using local Ensembl-symbol map")

    scanpy_pp_mod.calculate_qc_metrics = calculate_qc_metrics
    scanpy_queries_mod.biomart_annotations = biomart_annotations
    scanpy_mod.pp = scanpy_pp_mod
    scanpy_mod.queries = scanpy_queries_mod

    class _UnusedTranscriptomeProcessor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("This transcriptome processor is not used for cellwhisperer_clip_v1")

    scgpt_mod = types.ModuleType("cellwhisperer.jointemb.scgpt_model")
    uce_mod = types.ModuleType("cellwhisperer.jointemb.uce_model")
    scgpt_mod.ScGPTTranscriptomeProcessor = _UnusedTranscriptomeProcessor
    uce_mod.UCETranscriptomeProcessor = _UnusedTranscriptomeProcessor

    def rank_genes(gene_vector, gene_tokens):
        return gene_tokens[np.argsort(-gene_vector)]

    class TranscriptomeTokenizer:
        def __init__(self, custom_attr_name_dict=None, nproc=1, *args, **kwargs):
            self.custom_attr_name_dict = custom_attr_name_dict
            self.nproc = nproc
            self.gene_median_dict = gene_median_dict
            self.gene_token_dict = gene_token_dict
            self.gene_keys = list(gene_median_dict.keys())
            self.genelist_dict = dict(zip(self.gene_keys, [True] * len(self.gene_keys)))

    def pad_tensor(tensor, pad_token_id, max_len):
        return torch.nn.functional.pad(
            tensor, pad=(0, max_len - tensor.numel()), mode="constant", value=pad_token_id
        )

    def pad_tensor_list(tensor_list, dynamic_or_constant, pad_token_id, model_input_size):
        if dynamic_or_constant == "dynamic":
            max_len = max(tensor.squeeze().numel() for tensor in tensor_list)
        elif type(dynamic_or_constant) is int:
            max_len = dynamic_or_constant
        else:
            max_len = model_input_size
        return torch.stack([pad_tensor(tensor, pad_token_id, max_len) for tensor in tensor_list])

    def quant_layers(model):
        layer_nums = []
        for name, _ in model.named_parameters():
            if "layer." in name:
                layer_nums.append(int(name.split("layer.")[1].split(".")[0]))
        return int(max(layer_nums)) + 1

    def get_model_input_size(model):
        return int(re.split(r"\(|,", str(model.bert.embeddings.position_embeddings))[1])

    def gen_attention_mask(minibatch_encoding, max_len=None, device="cuda"):
        if max_len is None:
            max_len = max(minibatch_encoding["length"])
        original_lens = minibatch_encoding["length"]
        attention_mask = [
            [1] * int(original_len) + [0] * (int(max_len) - int(original_len))
            if int(original_len) <= int(max_len)
            else [1] * int(max_len)
            for original_len in original_lens
        ]
        return torch.tensor(attention_mask).to(device)

    def mean_nonpadding_embs(embs, original_lens):
        mask = torch.arange(embs.size(1), device=embs.device).unsqueeze(0) < original_lens.to(embs.device).unsqueeze(1)
        masked = embs * mask.unsqueeze(2).expand_as(embs).float()
        return masked.sum(1) / original_lens.to(embs.device).view(-1, 1).float()

    def get_embs(model, input_ids, lengths, emb_mode, layer_to_quant, pad_token_id, forward_batch_size, summary_stat):
        if summary_stat is not None:
            raise NotImplementedError("summary_stat is not needed for this evaluation")
        model_input_size = get_model_input_size(model)
        if forward_batch_size <= 0:
            forward_batch_size = len(input_ids)
        embs_list = []
        for i in range(0, len(input_ids), forward_batch_size):
            max_range = min(i + forward_batch_size, len(input_ids))
            minibatch_length = lengths[i:max_range]
            max_len = max(minibatch_length)
            if isinstance(max_len, torch.Tensor):
                max_len = max_len.item()
            minibatch_input = pad_tensor_list(input_ids[i:max_range], int(max_len), pad_token_id, model_input_size)
            outputs = model(
                input_ids=minibatch_input.to(model.device),
                attention_mask=gen_attention_mask({"length": minibatch_length}, device=model.device),
            )
            hidden = outputs.hidden_states[layer_to_quant]
            if emb_mode != "cell":
                raise NotImplementedError(f"Unsupported emb_mode={emb_mode}")
            embs_list.append(mean_nonpadding_embs(hidden, minibatch_length))
        return torch.cat(embs_list)

    def load_model(*args, **kwargs):
        raise NotImplementedError("load_model shim is not used in this evaluation")

    def downsample_and_sort(data_shuffled, max_ncells):
        return data_shuffled

    tokenizer_mod.TranscriptomeTokenizer = TranscriptomeTokenizer
    tokenizer_mod.rank_genes = rank_genes
    tokenizer_mod.TOKEN_DICTIONARY_FILE = TOKEN_DICTIONARY_FILE
    perturber_mod.pad_tensor_list = pad_tensor_list
    perturber_mod.quant_layers = quant_layers
    perturber_mod.get_model_input_size = get_model_input_size
    perturber_mod.gen_attention_mask = gen_attention_mask
    perturber_mod.mean_nonpadding_embs = mean_nonpadding_embs
    perturber_mod.load_model = load_model
    perturber_mod.downsample_and_sort = downsample_and_sort
    emb_mod.get_embs = get_embs

    geneformer_pkg.tokenizer = tokenizer_mod
    geneformer_pkg.in_silico_perturber = perturber_mod
    geneformer_pkg.emb_extractor = emb_mod
    sys.modules["geneformer"] = geneformer_pkg
    sys.modules["geneformer.tokenizer"] = tokenizer_mod
    sys.modules["geneformer.in_silico_perturber"] = perturber_mod
    sys.modules["geneformer.emb_extractor"] = emb_mod
    sys.modules["scanpy"] = scanpy_mod
    sys.modules["cellwhisperer.jointemb.scgpt_model"] = scgpt_mod
    sys.modules["cellwhisperer.jointemb.uce_model"] = uce_mod


def build_gene_map() -> Tuple[Path, int]:
    ens = ad.read_h5ad(ENSEMBL_H5AD, backed="r")
    sym = ad.read_h5ad(SYMBOL_H5AD, backed="r")
    n = min(ens.n_vars, sym.n_vars)
    df = pd.DataFrame(
        {"ensembl_gene_id": ens.var_names[:n].astype(str)},
        index=pd.Index(sym.var_names[:n].astype(str), name="external_gene_name"),
    )
    df = df[~df.index.duplicated(keep="first")]
    df.to_csv(OUT_GENE_MAP)
    return OUT_GENE_MAP, len(df)


def patch_cellwhisperer_gene_map(gene_map_path: Path) -> None:
    import cellwhisperer.jointemb.geneformer_model as gm

    def custom_get_path(config_keys, **kwargs):
        if list(config_keys) == ["paths", "ensembl_gene_symbol_map"]:
            return gene_map_path
        from cellwhisperer.config import get_path as real_get_path

        return real_get_path(config_keys, **kwargs)

    gm.get_path = custom_get_path


def canon_barcode(x: str) -> str:
    x = str(x).strip().split(",")[0]
    m = re.match(r"^([A-Z0-9]+-\d+)", x)
    return m.group(1) if m else x


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
    return normalize_barcode_df(df.dropna(subset=["barcode"]).copy(), "barcode")


def extract_layer_label(text: str) -> Optional[str]:
    if not text:
        return None
    m = _LAYER_TOKEN_PAT.search(text)
    if not m:
        return None
    layer_num, layer_label, wm_label = m.group(1), m.group(2), m.group(3)
    if layer_num:
        return f"Layer_{layer_num}"
    if layer_label:
        return layer_label[:1].upper() + layer_label[1:].lower()
    if wm_label:
        return "WM"
    return None


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


def subset_adata_for_merged(adata: ad.AnnData, merged: pd.DataFrame) -> ad.AnnData:
    idx = merged["obs_idx"].astype(int).to_numpy()
    sub = adata[idx, :].copy()
    sub.obs_names = merged["barcode"].astype(str).to_list()
    if "gene_name" not in sub.var.columns:
        sub.var["gene_name"] = sub.var_names.astype(str)
    if sp.issparse(sub.X):
        sub.X = sub.X.astype(np.float32)
    else:
        sub.X = np.asarray(sub.X, dtype=np.float32)
    return sub


def main():
    print("[1/6] Preparing CellWhisperer imports and gene map...")
    install_geneformer_shims()
    gene_map_path, n_map = build_gene_map()

    from cellwhisperer.utils.model_io import load_cellwhisperer_model

    patch_cellwhisperer_gene_map(gene_map_path)

    print("[2/6] Loading DLPFC data...")
    adata = ad.read_h5ad(H5AD_PATH)
    truth_df = load_truth_txt(TRUTH_PATH)
    pos_df = load_positions_txt(POS_PATH)
    obs_df = pd.DataFrame({"obs_idx": np.arange(adata.n_obs), "barcode": adata.obs_names.astype(str)})
    obs_df = normalize_barcode_df(obs_df, "barcode")
    merged = obs_df.merge(truth_df, on="barcode", how="inner").merge(pos_df, on="barcode", how="inner")
    merged = merged[merged["in_tissue"] == 1].copy().reset_index(drop=True)
    if MAX_SPOTS is not None:
        merged = merged.iloc[:MAX_SPOTS].copy()
    print("[INFO] total spots to score =", len(merged))

    print("[3/6] Loading CellWhisperer model...")
    pl_model, _, transcriptome_processor = load_cellwhisperer_model(str(MODEL_PATH), eval=True, cache=False)
    model = pl_model.model
    logit_scale = model.logit_scale.exp() if hasattr(model, "logit_scale") else 1.0

    print("[4/6] Computing transcriptome/text similarities...")
    sub = subset_adata_for_merged(adata, merged)
    candidate_texts = [LABEL_DESCRIPTIONS[label] for label in LABELS]
    with torch.inference_mode():
        from cellwhisperer.utils.inference import score_transcriptomes_vs_texts

        scores, _ = score_transcriptomes_vs_texts(
            sub,
            candidate_texts,
            logit_scale=logit_scale,
            model=model,
            average_mode=None,
            transcriptome_processor=transcriptome_processor,
            batch_size=BATCH_SIZE,
            score_norm_method=None,
        )
    scores_np = scores.numpy().T
    pred_idx = scores_np.argmax(axis=1)

    print("[5/6] Exporting predictions...")
    rows: List[Dict[str, Any]] = []
    for i, row in tqdm(list(merged.iterrows()), total=len(merged), dynamic_ncols=True):
        pred_label = LABELS[int(pred_idx[i])]
        score_dict = {f"score_{label}": float(scores_np[i, j]) for j, label in enumerate(LABELS)}
        out = {
            "barcode": str(row["barcode"]),
            "truth": str(row["label"]),
            "raw_gene": LABEL_DESCRIPTIONS[pred_label],
            "model_raw_gene": LABEL_DESCRIPTIONS[pred_label],
            "pred_gene": pred_label,
            **score_dict,
        }
        rows.append(out)
    with OUT_RAW_JSONL.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
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
        "model_path": str(MODEL_PATH),
        "max_spots": MAX_SPOTS,
        "batch_size": BATCH_SIZE,
        "score_batch_size": SCORE_BATCH_SIZE,
        "prompt": USER_PROMPT,
        "label_hint": LABEL_HINT,
        "evaluation_method": "CellWhisperer transcriptome-text similarity over seven canonical full descriptions",
        "ensembl_symbol_map": str(gene_map_path),
        "n_gene_map": n_map,
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

    print("[6/6] Done.")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
