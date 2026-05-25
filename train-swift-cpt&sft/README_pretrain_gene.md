# pretrain-gene

This folder contains the cleaned GitHub-facing code for gene-aware MiniCPM-V pretraining/fine-tuning workflows, plus C2S and CellWhisperer baselines for DLPFC gene-only evaluation.

The source files were reorganized from the experimental workspace under `my_custom_model`. Historical CPT experiment scripts, temporary notebooks, caches, logs, datasets, and model weights are intentionally not included here.

## Directory Structure

```text
pretrain-gene/
  README.md
  src/pretrain_gene/
    __init__.py
    swift_minicpm_gene_register.py
    swift_minicpm_gene_qformer_register.py
    train_c2s_lora.py
    eval_c2s_gene_only.py
    prepare_cellwhisperer_dlpfc_sft.py
    prepare_cellwhisperer_eval_features.py
    eval_cellwhisperer_similarity_gene_only.py
    eval_cellwhisperer_llava_gene_only.py
  scripts/
    train_minicpm_gene_qformer.sh
    train_minicpm_vision_qformer.sh
    train_minicpm_gene_vision_qformer.sh
    train_c2s_lora.sh
    train_cellwhisperer_llava_lora.sh
    infer_minicpm_gene.sh
  llava_compat/deepspeed/
```

## Core Python Files

| File | Purpose |
| --- | --- |
| `src/pretrain_gene/swift_minicpm_gene_qformer_register.py` | Swift custom registration for the MiniCPM-V + gene + Q-Former model. It registers the `minicpm_v2_6_gene` model/template, tokenizes `.h5ad` gene inputs, expands the `<gene>` placeholder into a configurable span, builds `gene_input_ids`, `gene_attention_mask`, and `gene_bound`, and controls freezing for Nicheformer, gene projector, and gene Q-Former modules. |
| `src/pretrain_gene/swift_minicpm_gene_register.py` | Legacy Swift custom registration for MiniCPM-V + gene without the Q-Former span expansion. Keep this for older checkpoints or non-Q-Former experiments. |
| `src/pretrain_gene/train_c2s_lora.py` | Trains a C2S LoRA baseline. It reads DLPFC-style JSONL examples, loads each `.h5ad`, converts the expression vector into a high-expression gene sequence prompt, and fine-tunes a causal LM adapter. |
| `src/pretrain_gene/eval_c2s_gene_only.py` | Evaluates C2S + LoRA on DLPFC gene-only inputs. It generates layer descriptions, parses labels into `Layer_1` to `Layer_6` or `WM`, and exports raw answers, scored CSV, prediction TXT, and metrics JSON. |
| `src/pretrain_gene/prepare_cellwhisperer_dlpfc_sft.py` | Prepares CellWhisperer-LLaVA SFT data. It converts DLPFC examples into LLaVA-style conversations and extracts CellWhisperer transcriptome features/embeddings into an `.npz` file. |
| `src/pretrain_gene/prepare_cellwhisperer_eval_features.py` | Extracts CellWhisperer transcriptome features for a DLPFC evaluation slide and writes feature `.npz` plus metadata `.csv`. |
| `src/pretrain_gene/eval_cellwhisperer_similarity_gene_only.py` | Evaluates CellWhisperer directly with transcriptome-text similarity. It compares each spot against seven canonical cortical-layer descriptions and chooses the highest-scoring label. |
| `src/pretrain_gene/eval_cellwhisperer_llava_gene_only.py` | Evaluates a CellWhisperer-LLaVA LoRA model by feeding transcriptome embeddings as visual features, generating layer descriptions, parsing labels, and exporting metrics. |

## Shell Scripts

| File | Purpose |
| --- | --- |
| `scripts/train_minicpm_gene_qformer.sh` | Gene-only MiniCPM-V Q-Former LoRA SFT. Typically freezes the vision tower and trains the gene/Q-Former alignment path plus LoRA modules. |
| `scripts/train_minicpm_vision_qformer.sh` | Vision-focused MiniCPM-V Q-Former LoRA SFT. Useful when visual features should remain trainable while gene/Q-Former modules are frozen or mostly frozen. |
| `scripts/train_minicpm_gene_vision_qformer.sh` | Joint gene + vision MiniCPM-V Q-Former LoRA SFT. Used for multimodal DLPFC-style training where gene and image inputs are both active. |
| `scripts/train_c2s_lora.sh` | Launches C2S LoRA baseline training with `train_c2s_lora.py`. |
| `scripts/train_cellwhisperer_llava_lora.sh` | Launches CellWhisperer-LLaVA LoRA training using prepared conversation JSON and transcriptome feature `.npz` files. |
| `scripts/infer_minicpm_gene.sh` | Launches Swift inference for a MiniCPM-V gene checkpoint. Can run batch evaluation on a dataset or interactive inference depending on script settings. |

## Runtime Configuration

Before running any script, review and update the paths inside that script for your machine:

- `GENE_VOCAB_PATH`: path to the gene tokenizer vocabulary JSON.
- `BASE_MODEL`, `CKPT_DIR`, `MODEL_PATH`, or `LORA_PATH`: local model/checkpoint paths.
- `TRAIN_DATA`, `TEST_DATA`, `DATA_JSON`, `IMAGE_DATA`, `FEATURE_NPZ`, or `META_CSV`: local dataset and feature paths.
- `CUDA_VISIBLE_DEVICES`, `NPROC_PER_NODE`, and `MASTER_PORT`: GPU and distributed training settings.
- CellWhisperer workflows may also require `CELLWHISPERER_ROOT`, `CW_*` environment variables, and the local LLaVA module path.

The scripts preserve the original lab paths as defaults. Treat them as examples and change them before launching jobs in a new environment.

## Important Path Updates After Reorganization

If these files were copied from the old `my_custom_model` directory, make sure old references are updated:

```text
old: /data2/xiaoxinyu/project/pretrain-gene/my_custom_model/my_register_qformer.py
new: /data2/liyunfei/OMICS/Scicore-Omics/pretrain-gene/src/pretrain_gene/swift_minicpm_gene_qformer_register.py

old: /data2/xiaoxinyu/project/pretrain-gene/my_custom_model/my_register.py
new: /data2/liyunfei/OMICS/Scicore-Omics/pretrain-gene/src/pretrain_gene/swift_minicpm_gene_register.py

old: /data2/xiaoxinyu/project/pretrain-gene/my_custom_model/c2s_lora_sft.py
new: /data2/liyunfei/OMICS/Scicore-Omics/pretrain-gene/src/pretrain_gene/train_c2s_lora.py

old: /data2/xiaoxinyu/project/pretrain-gene/my_custom_model/llava_compat
new: /data2/liyunfei/OMICS/Scicore-Omics/pretrain-gene/llava_compat
```

Also update Python imports after renaming:

```python
# eval_c2s_gene_only.py
try:
    from .train_c2s_lora import LABEL_HINT, USER_PROMPT, build_prompt
except ImportError:
    from train_c2s_lora import LABEL_HINT, USER_PROMPT, build_prompt

# prepare_cellwhisperer_eval_features.py
try:
    from . import prepare_cellwhisperer_dlpfc_sft as prep
except ImportError:
    import prepare_cellwhisperer_dlpfc_sft as prep
```

## What Is Intentionally Excluded

Do not commit these files or directories into this GitHub-facing folder:

- Historical `cpt-*` continued-pretraining experiment scripts.
- Temporary notebooks such as `1.ipynb`.
- `__pycache__/`, `.pyc`, logs, generated caches, and temporary outputs.
- Training datasets such as `.jsonl`, `.h5ad`, `.npz`, and `.csv` unless they are tiny documented examples.
- Checkpoints, model weights, and large artifacts such as `.bin`, `.pt`, `.pth`, `.ckpt`, and `.safetensors`.

Recommended `.gitignore` patterns:

```gitignore
__pycache__/
*.pyc
*.log
*.jsonl
*.h5ad
*.npz
*.csv
*.pt
*.pth
*.bin
*.ckpt
*.safetensors
checkpoint-*/
outputs/
logs/
cache/
```

## Suggested Sanity Checks

After copying or editing files, run:

```bash
find pretrain-gene -maxdepth 4 -type f | sort
python3 -m py_compile pretrain-gene/src/pretrain_gene/*.py
bash -n pretrain-gene/scripts/*.sh
grep -R "my_custom_model" pretrain-gene || true
```

The final grep should not show old `my_custom_model` paths in active scripts or Python files.
