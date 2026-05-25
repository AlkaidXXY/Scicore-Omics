#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from tqdm.auto import tqdm

sys.path.insert(0, "/data2/xiaoxinyu/project/pretrain-gene/my_custom_model")
from . import prepare_cellwhisperer_dlpfc_sft as prep


SLIDE_ID = os.environ.get("CW_EVAL_SLIDE_ID", "151508")
BASE_ST_DIR = Path(f"/data1/xiaoxinyu/benchmark/DLPFC/stRNA/10xformat/{SLIDE_ID}")
H5AD_PATH = BASE_ST_DIR / "filtered_feature_bc_matrix.h5ad"
POS_PATH = BASE_ST_DIR / "spatial/tissue_positions_list.txt"
TRUTH_PATH = Path(f"/data1/xiaoxinyu/benchmark/DLPFC/annotation/{SLIDE_ID}_truth.txt")
OUT_DIR = Path(os.environ.get("CW_EVAL_FEATURE_DIR", "/data2/xiaoxinyu/project/pretrain-gene/cellwhisperer_eval_features"))
OUT_NPZ = OUT_DIR / f"{SLIDE_ID}_cellwhisperer_features.npz"
OUT_META = OUT_DIR / f"{SLIDE_ID}_metadata.csv"
BATCH_SIZE = int(os.environ.get("CW_EVAL_BATCH_SIZE", "32"))


def canon_barcode(x: str) -> str:
    x = str(x).strip().split(",")[0]
    return x


def load_truth_txt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python").iloc[:, :2].copy()
    df.columns = ["barcode", "label"]
    df["barcode"] = df["barcode"].astype(str).map(canon_barcode)
    df["label"] = df["label"].astype(str)
    return df


def load_positions_txt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"[\t,]", header=None, engine="python").iloc[:, :6].copy()
    df.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
    df["barcode"] = df["barcode"].astype(str).map(canon_barcode)
    for c in ["in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prep.install_shims()
    gene_map = prep.build_gene_map()
    prep.patch_gene_map(gene_map)

    from cellwhisperer.config import model_path_from_name
    import cellwhisperer.jointemb.cellwhisperer_lightning as cw_lightning
    from cellwhisperer.jointemb.cellwhisperer_lightning import TranscriptomeTextDualEncoderLightning
    from cellwhisperer.jointemb.processing import TranscriptomeTextDualEncoderProcessor

    def local_model_path_from_name(name: str):
        if name == "geneformer":
            return str(prep.CELLWHISPERER_ROOT / "modules/Geneformer/geneformer-12L-30M")
        return model_path_from_name(name)

    cw_lightning.model_path_from_name = local_model_path_from_name
    pl_model = TranscriptomeTextDualEncoderLightning.load_from_checkpoint(str(prep.MODEL_PATH), weights_only=False)
    pl_model.freeze()
    pl_model.eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    processor = TranscriptomeTextDualEncoderProcessor(
        pl_model.model.transcriptome_model.config.model_type,
        model_path_from_name(pl_model.model.text_model.config.model_type),
    ).transcriptome_processor
    model = pl_model.model

    adata = ad.read_h5ad(H5AD_PATH)
    if "gene_name" not in adata.var.columns:
        adata.var["gene_name"] = adata.var_names.astype(str)
    if sp.issparse(adata.X):
        adata.X = adata.X.astype(np.float32)
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)

    truth = load_truth_txt(TRUTH_PATH)
    pos = load_positions_txt(POS_PATH)
    obs = pd.DataFrame({"obs_idx": np.arange(adata.n_obs), "barcode": adata.obs_names.astype(str)})
    obs["barcode"] = obs["barcode"].map(canon_barcode)
    merged = obs.merge(truth, on="barcode", how="inner").merge(pos, on="barcode", how="inner")
    merged = merged[merged["in_tissue"] == 1].copy().reset_index(drop=True)
    sub = adata[merged["obs_idx"].astype(int).to_numpy(), :].copy()
    sub.obs_names = merged["barcode"].astype(str).to_list()

    features = []
    embeds = []
    for i in tqdm(range(0, sub.n_obs, BATCH_SIZE), desc="eval features"):
        batch = sub[i : i + BATCH_SIZE].copy()
        inputs = processor(batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            feat, emb = model.get_transcriptome_features(**inputs, normalize_embeds=True)
        features.append(feat.cpu().float().numpy())
        embeds.append(emb.cpu().float().numpy())

    np.savez(
        OUT_NPZ,
        orig_ids=merged["barcode"].astype(str).to_numpy(),
        transcriptome_features=np.concatenate(features, axis=0),
        transcriptome_embeds=np.concatenate(embeds, axis=0),
    )
    merged.to_csv(OUT_META, index=False, encoding="utf-8-sig")
    print(json.dumps({"features": str(OUT_NPZ), "metadata": str(OUT_META), "n": int(len(merged))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
