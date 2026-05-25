#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import importlib.machinery
import os
import pickle
import re
import sys
import types
from pathlib import Path
from typing import Dict, List

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

DATASET = Path(os.environ.get("CW_DLPFC_DATASET", "/data1/xiaoxinyu/project/data/format_dataset/DLPFC_tri_QA_balanced_train_v5.jsonl"))
MODEL_PATH = Path(os.environ.get("CW_CLIP_MODEL", str(CELLWHISPERER_ROOT / "results/models/jointemb/cellwhisperer_clip_v1.ckpt")))
OUT_DIR = Path(os.environ.get("CW_PREP_OUT_DIR", "/data2/xiaoxinyu/project/pretrain-gene/cellwhisperer_dlpfc_sft"))
OUT_JSON = OUT_DIR / "dlpfc_cellwhisperer_conversations.json"
OUT_NPZ = OUT_DIR / "dlpfc_cellwhisperer_features.npz"
OUT_GENE_MAP = OUT_DIR / "ensembl_gene_symbol_map.csv"

ENSEMBL_H5AD = Path(os.environ.get("CW_ENSEMBL_H5AD", "/data2/xiaoxinyu/project/model/gene_tokenizer/model-ensembl.h5ad"))
SYMBOL_H5AD = Path(os.environ.get("CW_SYMBOL_H5AD", "/data2/xiaoxinyu/project/model/gene_tokenizer/model-symbel.h5ad"))
BATCH_SIZE = int(os.environ.get("CW_PREP_BATCH_SIZE", "32"))

USER_PROMPT = "请描述样本信息?"
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


def example_gene_path(obj: Dict) -> str:
    for msg in obj.get("messages", []):
        if msg.get("role") == "user":
            return first_gene_path(msg.get("content", []))
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


def install_shims() -> None:
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
        return torch.nn.functional.pad(tensor, pad=(0, max_len - tensor.numel()), mode="constant", value=pad_token_id)

    def pad_tensor_list(tensor_list, dynamic_or_constant, pad_token_id, model_input_size):
        max_len = max(t.squeeze().numel() for t in tensor_list) if dynamic_or_constant == "dynamic" else int(dynamic_or_constant)
        return torch.stack([pad_tensor(tensor, pad_token_id, max_len) for tensor in tensor_list])

    def quant_layers(model):
        nums = [int(name.split("layer.")[1].split(".")[0]) for name, _ in model.named_parameters() if "layer." in name]
        return int(max(nums)) + 1

    def get_model_input_size(model):
        return int(re.split(r"\(|,", str(model.bert.embeddings.position_embeddings))[1])

    def gen_attention_mask(minibatch_encoding, max_len=None, device="cuda"):
        if max_len is None:
            max_len = max(minibatch_encoding["length"])
        return torch.tensor([
            [1] * int(n) + [0] * (int(max_len) - int(n)) if int(n) <= int(max_len) else [1] * int(max_len)
            for n in minibatch_encoding["length"]
        ]).to(device)

    def mean_nonpadding_embs(embs, original_lens):
        mask = torch.arange(embs.size(1), device=embs.device).unsqueeze(0) < original_lens.to(embs.device).unsqueeze(1)
        return (embs * mask.unsqueeze(2).expand_as(embs).float()).sum(1) / original_lens.to(embs.device).view(-1, 1).float()

    def get_embs(model, input_ids, lengths, emb_mode, layer_to_quant, pad_token_id, forward_batch_size, summary_stat):
        if forward_batch_size <= 0:
            forward_batch_size = len(input_ids)
        out = []
        model_input_size = get_model_input_size(model)
        for i in range(0, len(input_ids), forward_batch_size):
            j = min(i + forward_batch_size, len(input_ids))
            lens = lengths[i:j]
            max_len = int(max(lens).item() if isinstance(max(lens), torch.Tensor) else max(lens))
            x = pad_tensor_list(input_ids[i:j], max_len, pad_token_id, model_input_size)
            y = model(input_ids=x.to(model.device), attention_mask=gen_attention_mask({"length": lens}, device=model.device))
            out.append(mean_nonpadding_embs(y.hidden_states[layer_to_quant], lens))
        return torch.cat(out)

    def _unused(*args, **kwargs):
        raise RuntimeError("unused Geneformer helper")

    tokenizer_mod.TranscriptomeTokenizer = TranscriptomeTokenizer
    tokenizer_mod.rank_genes = rank_genes
    tokenizer_mod.TOKEN_DICTIONARY_FILE = TOKEN_DICTIONARY_FILE
    perturber_mod.pad_tensor_list = pad_tensor_list
    perturber_mod.quant_layers = quant_layers
    perturber_mod.get_model_input_size = get_model_input_size
    perturber_mod.gen_attention_mask = gen_attention_mask
    perturber_mod.mean_nonpadding_embs = mean_nonpadding_embs
    perturber_mod.load_model = _unused
    perturber_mod.downsample_and_sort = lambda data, max_ncells: data
    emb_mod.get_embs = get_embs

    geneformer_pkg.tokenizer = tokenizer_mod
    geneformer_pkg.in_silico_perturber = perturber_mod
    geneformer_pkg.emb_extractor = emb_mod
    sys.modules["geneformer"] = geneformer_pkg
    sys.modules["geneformer.tokenizer"] = tokenizer_mod
    sys.modules["geneformer.in_silico_perturber"] = perturber_mod
    sys.modules["geneformer.emb_extractor"] = emb_mod

    scanpy_mod = types.ModuleType("scanpy")
    scanpy_mod.pp = types.SimpleNamespace(
        calculate_qc_metrics=lambda a, inplace=True, **kw: a.obs.__setitem__("total_counts", np.asarray(a.X.sum(axis=1)).reshape(-1))
    )
    scanpy_mod.queries = types.SimpleNamespace(biomart_annotations=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("biomart disabled")))
    sys.modules["scanpy"] = scanpy_mod

    class UnusedConfig:
        model_type = "unused"

    class UnusedModel:
        config_class = UnusedConfig

    class UnusedProcessor:
        pass

    for name, cfg, mdl, proc in [
        ("cellwhisperer.jointemb.scgpt_model", "ScGPTConfig", "ScGPTModel", "ScGPTTranscriptomeProcessor"),
        ("cellwhisperer.jointemb.uce_model", "UCEConfig", "UCEModel", "UCETranscriptomeProcessor"),
    ]:
        mod = types.ModuleType(name)
        setattr(mod, cfg, UnusedConfig)
        setattr(mod, mdl, UnusedModel)
        setattr(mod, proc, UnusedProcessor)
        sys.modules[name] = mod

    validation_mod = types.ModuleType("cellwhisperer.validation")
    validation_mod.initialize_validation_functions = lambda *args, **kwargs: {}
    sys.modules["cellwhisperer.validation"] = validation_mod

    wandb_mod = types.ModuleType("wandb")
    wandb_mod.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)
    wandb_mod.Artifact = type("Artifact", (), {})
    wandb_mod.Table = type("Table", (), {"__init__": lambda self, *args, **kwargs: None})
    sys.modules["wandb"] = wandb_mod


def build_gene_map() -> Path:
    ens = ad.read_h5ad(ENSEMBL_H5AD, backed="r")
    sym = ad.read_h5ad(SYMBOL_H5AD, backed="r")
    n = min(ens.n_vars, sym.n_vars)
    df = pd.DataFrame({"ensembl_gene_id": ens.var_names[:n].astype(str)}, index=pd.Index(sym.var_names[:n].astype(str), name="external_gene_name"))
    df = df[~df.index.duplicated(keep="first")]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_GENE_MAP)
    return OUT_GENE_MAP


def patch_gene_map(path: Path) -> None:
    import cellwhisperer.jointemb.geneformer_model as gm

    real_get_path = gm.get_path
    gm.get_path = lambda keys, **kwargs: path if list(keys) == ["paths", "ensembl_gene_symbol_map"] else real_get_path(keys, **kwargs)


def h5ad_to_single_cell(path: str) -> ad.AnnData:
    a = ad.read_h5ad(path)
    if "gene_name" not in a.var.columns:
        a.var["gene_name"] = a.var_names.astype(str)
    if sp.issparse(a.X):
        a.X = a.X.astype(np.float32)
    else:
        a.X = np.asarray(a.X, dtype=np.float32)
    return a


def main():
    install_shims()
    gene_map = build_gene_map()
    patch_gene_map(gene_map)
    from cellwhisperer.config import model_path_from_name
    import cellwhisperer.jointemb.cellwhisperer_lightning as cw_lightning
    from cellwhisperer.jointemb.cellwhisperer_lightning import TranscriptomeTextDualEncoderLightning
    from cellwhisperer.jointemb.processing import TranscriptomeTextDualEncoderProcessor

    def local_model_path_from_name(name: str):
        if name == "geneformer":
            return str(CELLWHISPERER_ROOT / "modules/Geneformer/geneformer-12L-30M")
        return model_path_from_name(name)

    cw_lightning.model_path_from_name = local_model_path_from_name
    pl_model = TranscriptomeTextDualEncoderLightning.load_from_checkpoint(str(MODEL_PATH), weights_only=False)
    pl_model.freeze()
    pl_model.eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    processor = TranscriptomeTextDualEncoderProcessor(
        pl_model.model.transcriptome_model.config.model_type,
        model_path_from_name(pl_model.model.text_model.config.model_type),
    ).transcriptome_processor
    model = pl_model.model

    rows = []
    adatas = []
    for obj in read_jsonl(DATASET):
        gene_path = example_gene_path(obj)
        label = extract_answer_label(example_answer(obj))
        if not gene_path or label not in LABEL_DESCRIPTIONS:
            continue
        sample_id = str(obj.get("id") or Path(gene_path).stem)
        rows.append({
            "id": sample_id,
            "image": sample_id,
            "conversations": [
                {"from": "human", "value": f"<image>\n{USER_PROMPT}"},
                {"from": "gpt", "value": LABEL_DESCRIPTIONS[label]},
            ],
        })
        a = h5ad_to_single_cell(gene_path)
        a.obs_names = [sample_id]
        adatas.append(a)

    print(f"[INFO] examples = {len(rows)}")
    merged = ad.concat(adatas, axis=0, join="outer", merge="same", fill_value=0)
    if "gene_name" not in merged.var.columns:
        merged.var["gene_name"] = merged.var_names.astype(str)

    features = []
    embeds = []
    for i in tqdm(range(0, merged.n_obs, BATCH_SIZE), desc="CellWhisperer features"):
        batch = merged[i : i + BATCH_SIZE].copy()
        inputs = processor(batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            feat, emb = model.get_transcriptome_features(**inputs, normalize_embeds=True)
        features.append(feat.cpu().float().numpy())
        embeds.append(emb.cpu().float().numpy())

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    np.savez(
        OUT_NPZ,
        orig_ids=np.array([r["id"] for r in rows]),
        transcriptome_features=np.concatenate(features, axis=0),
        transcriptome_embeds=np.concatenate(embeds, axis=0),
    )
    print("[INFO] wrote", OUT_JSON)
    print("[INFO] wrote", OUT_NPZ)


if __name__ == "__main__":
    main()
