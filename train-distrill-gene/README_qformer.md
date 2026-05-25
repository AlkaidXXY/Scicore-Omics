# qformer

This folder contains the gene bridge / Q-Former distillation utilities used to train and inject the gene-side bridge modules for MiniCPM-V-style gene-aware multimodal models.

The cleaned GitHub-facing folder keeps only source code. Training outputs, checkpoints, logs, caches, and temporary files are intentionally excluded.

## Directory Structure

```text
qformer/
  README.md
  train_gene_bridge_distill.py
  train_gene_bridge_distill_ddp.py
  train_gene_bridge_distill_real_processor.py
  inject_gene_bridge_weights.py
```

## Files

| File | Purpose |
| --- | --- |
| `train_gene_bridge_distill.py` | Single-GPU gene bridge distillation script. It freezes the base model and trains only `gene_qformer` and `gene_projector` using CE, cosine alignment, and InfoNCE losses. |
| `train_gene_bridge_distill_ddp.py` | Multi-GPU DDP version of the manual-tokenizer distillation script. It adds distributed training, cross-rank InfoNCE negatives, variable-batch all-gather support, and safer masking for samples without valid gene spans. |
| `train_gene_bridge_distill_real_processor.py` | Main recommended training script. It uses the model's real `AutoProcessor` gene tokenizer path, aligns `.h5ad` genes to a local reference gene list, supports DDP, trains `gene_qformer`, `gene_projector`, and an auxiliary `gene_cls_head`, and combines CE, cosine, InfoNCE, and classification losses. |
| `inject_gene_bridge_weights.py` | Post-training utility that injects distilled `gene_qformer` and `gene_projector` weights from a `.pt` checkpoint into a full sharded `safetensors` model directory. |

## Recommended Workflow

1. Train the gene bridge with the real-processor script:

   ```bash
   torchrun --nproc_per_node 4 train_gene_bridge_distill_real_processor.py \
     --model_path /path/to/model_cpt_v7_qformer \
     --data_jsonl /path/to/gene_data.jsonl \
     --out_dir /path/to/distill_out \
     --epochs 1 \
     --batch_size 2 \
     --lr 1e-4 \
     --lambda_ce 0.2 \
     --lambda_cos 1.0 \
     --lambda_nce 1.0 \
     --lambda_cls 0.5 \
     --temp 0.07 \
     --save_step 500
   ```

2. Pick a saved checkpoint, for example:

   ```text
   /path/to/distill_out/gene_bridge_distill_real_processor_step500.pt
   ```

3. Use `inject_gene_bridge_weights.py` to copy a full base model directory and replace the `gene_qformer` / `gene_projector` tensors with the distilled weights.

4. Load the injected model directory with:

   ```python
   from transformers import AutoModel

   model = AutoModel.from_pretrained(
       "/path/to/injected_model",
       trust_remote_code=True,
   )
   ```

## Training Logic

All training scripts share the same high-level goal: make the gene-side bridge produce representations aligned with the text-side representation of the expected answer.

The student path uses gene inputs:

```text
.h5ad gene expression -> gene tokenizer / processor -> gene_qformer -> gene_projector -> <gene> span embeddings
```

The teacher path uses text:

```text
assistant answer tokens -> frozen LLM hidden states -> pooled text vector
```

The main losses are:

- `loss_ce`: language-model cross entropy through the frozen LLM, with gradients flowing back to the gene bridge through `inputs_embeds`.
- `loss_cos`: cosine alignment between the pooled gene-span vector and the pooled teacher text vector.
- `loss_nce`: contrastive InfoNCE loss between gene vectors and teacher text vectors.
- `loss_cls`: auxiliary classification loss used by `train_gene_bridge_distill_real_processor.py` for samples with known layer/tissue labels.

## Script Details

### `train_gene_bridge_distill.py`

This is the simplest single-GPU version. It manually tokenizes gene names using a `vocab.json` file and trains only:

```text
gene_qformer
gene_projector
```

It saves:

```text
gene_bridge_distill.pt
gene_bridge_distill_step*.pt
train_log.jsonl
loss_curve.png
```

Use this mainly for debugging or small runs.

### `train_gene_bridge_distill_ddp.py`

This is the distributed version of the manual-tokenizer approach. Compared with the single-GPU script, it adds:

- `torch.distributed` / `torchrun` support.
- Global InfoNCE negatives across ranks.
- Variable-batch all-gather handling for uneven final batches.
- Master-only logging and checkpointing.
- Masking for samples where `gene_bound` is missing.

Use this if you need the older manual `vocab.json` tokenization path but want multi-GPU training.

### `train_gene_bridge_distill_real_processor.py`

This is the preferred version for current experiments. It uses:

```python
AutoProcessor.from_pretrained(..., trust_remote_code=True)
processor.gene_tokenizer(...)
```

Instead of only using `.h5ad.var_names`, it aligns each input `.h5ad` to a reference gene list before tokenization. The script looks for reference files such as:

```text
/data2/xiaoxinyu/project/model/gene_tokenizer/model-symbel.h5ad
/data2/xiaoxinyu/project/model/gene_tokenizer/model-ensembl.h5ad
```

It trains:

```text
gene_qformer
gene_projector
gene_cls_head
```

The auxiliary classification head supports labels from:

- DLPFC cortical layers: `Layer_1` to `Layer_6`, `WM`.
- STimage tissue groups: `breast`, `skin`, `heart`.
- CellWhisperer samples are treated as unlabeled for classification.

The checkpoint contains:

```python
{
    "gene_qformer": ...,
    "gene_projector": ...,
    "gene_cls_head": ...,
    "args": ...,
    "global_step": ...,
    "epoch": ...,
    "label2id": ...,
}
```

### `inject_gene_bridge_weights.py`

This script injects distilled bridge weights into a full model directory.

It expects:

- A full sharded `safetensors` model directory with `model.safetensors.index.json`.
- A distilled `.pt` checkpoint containing:

  ```python
  "gene_qformer"
  "gene_projector"
  ```

It creates a new model directory, finds which safetensors shards contain the bridge parameters, replaces those tensors, and saves the updated shards.

Before using it in a new environment, edit these paths in the script:

```python
base_dir = "/path/to/base_model"
step_ckpt = "/path/to/gene_bridge_distill_step.pt"
out_dir = "/path/to/output_injected_model"
```

## Runtime Configuration

Before launching training, update:

- `--model_path`: model directory containing `gene_qformer` and `gene_projector`.
- `--data_jsonl`: training JSONL file with `messages` and gene paths.
- `--out_dir`: output directory for checkpoints and logs.
- `--gene_vocab`: required by `train_gene_bridge_distill.py` and `train_gene_bridge_distill_ddp.py`.
- CUDA variables such as `CUDA_VISIBLE_DEVICES`, `NCCL_ASYNC_ERROR_HANDLING`, and `PYTORCH_CUDA_ALLOC_CONF`.

The scripts currently preserve original lab paths in comments or defaults. Treat them as examples, not portable defaults.

## What Not To Commit

Keep generated artifacts out of git:

```gitignore
__pycache__/
*.pyc
*.pt
*.pth
*.bin
*.ckpt
*.safetensors
*.jsonl
*.h5ad
*.npz
*.csv
*.log
distill_out*/
checkpoint-*/
loss_curve.png
train_log.jsonl
```

The source directory may contain previous outputs like `distill_out_cpt_v6_real_processor/` and `distill_out_cpt_v7_real_processor/`; those are training outputs and should stay outside the cleaned repository folder.

## Suggested Checks

After copying this folder into a repository, run:

```bash
find qformer -maxdepth 3 -type f | sort
python3 -m py_compile qformer/*.py
grep -R "__pycache__\\|distill_out\\|\\.pt" qformer || true
```

The final grep should not show committed generated artifacts.
