# SciCore-Omics

SciCore-Omics is a gene-aware multimodal modeling project built around the MiniCPM-V stack. The central goal of the repository is to make transcriptomic signals usable alongside natural language and tissue imagery within a single instruction-following model. In practice, this repository extends the MiniCPM-V architecture with a dedicated gene branch, provides training code for aligning that branch to the language model, and includes downstream fine-tuning and baseline evaluation pipelines for gene-centric spatial transcriptomics tasks.


## Core Idea

The model augments a MiniCPM-V style vision-language model with a transcriptomics pathway:

```text
gene expression (.h5ad)
  -> gene tokenizer
  -> Nicheformer gene encoder
  -> Gene Q-Former bridge
  -> Gene Projector
  -> <gene> span embeddings in the LLM token space

image
  -> vision tower
  -> resampler
  -> <image> span embeddings in the LLM token space

text prompt
  -> tokenizer

all modalities
  -> merged input embeddings
  -> MiniCPM-V / Qwen2 language model
```

This design allows the model to consume transcriptomic context either alone or together with histology images and text instructions, while preserving the standard autoregressive language-model interface.

## What Is In This Repository

The project is organized around four main code areas:

| Path | Role |
| --- | --- |
| `model/` | Core model and processor definitions for the gene-aware MiniCPM-V variant. |
| `finetune-gene/` | Earlier Hugging Face `Trainer` + DeepSpeed fine-tuning pipeline for multimodal gene experiments. |
| `qformer/` | Gene bridge distillation utilities for training `gene_qformer` and `gene_projector`, plus weight injection into a full model directory. |
| `pretrain-gene/` | Cleaner GitHub-facing training, inference, and baseline evaluation scripts, including C2S and CellWhisperer comparisons. |
| `src/` | TO DO |
| `environment.yml` | Conda environment specification for the research stack. |

If you are new to the codebase, the most useful reading order is:

1. `model/`
2. `qformer/`
3. `pretrain-gene/`
4. `finetune-gene/`

## Quick Start

1. Online demo

   A live demo is available here:

   [http://166.111.5.103:15557/](http://166.111.5.103:15557/)

   This is the quickest way to inspect the current behavior while public weights are not yet released.

2. Environment setup

   To use the model locally, first create the project environment from `environment.yml`:

   ```bash
   conda env create -f environment.yml
   conda activate OMICS
   ```

   The reference environment was developed on Linux with NVIDIA A800-SXM4-80GB GPUs. The `flash-attn` package can be sensitive to the local CUDA, PyTorch, and GPU setup, so it may need to be adjusted for a different machine.

3. Hugging Face release

   TODO: add the public Hugging Face model and checkpoint links once the repository is officially open-sourced.

## Model Architecture

The heart of the repository lives in `model/`, where the multimodal model is defined.

### Key components in `model/`

| File | Purpose |
| --- | --- |
| `model/configuration_minicpm.py` | Defines `MiniCPMVConfig`, extending `Qwen2Config` with `vision_config`, `slice_config`, and `gene_config`. |
| `model/configuration_nicheformer.py` | Defines `NicheformerConfig`, the configuration object for the gene encoder. |
| `model/modeling_nicheformer.py` | Implements `NicheformerModel`, a transformer encoder over gene tokens. |
| `model/gene_qformer_module.py` | Implements `GeneQFormerBiomedBERT`, a learnable-query bridge that compresses variable-length gene token sequences into a fixed set of query tokens. |
| `model/gene_projector_module.py` | Projects Q-Former outputs from the bridge hidden size into the language-model embedding dimension. |
| `model/modeling_minicpmv.py` | Integrates the LLM, vision tower, resampler, Nicheformer, gene Q-Former, and gene projector into one multimodal model. |
| `model/processing_minicpmv.py` | Implements the processor that packages text, image, and gene inputs into model-ready tensors. |
| `model/gene_tokenizer/` | Gene-tokenization resources, tokenizer logic, vocabulary, and reference `.h5ad` assets used by the processor and training scripts. |

### How the gene branch is wired

At a high level, the repository uses the following sequence:

1. Gene expression is tokenized into a gene-token sequence.
2. `NicheformerModel` encodes that sequence into contextual gene embeddings.
3. `GeneQFormerBiomedBERT` compresses those embeddings into a fixed number of query tokens.
4. `GeneProjector` maps the bridge outputs into the hidden space of the MiniCPM-V language model.
5. The projected embeddings are inserted into the language-model input stream at the positions corresponding to the textual placeholder token span for `"<gene>"`.

The multimodal merge happens inside the MiniCPM-V modeling logic, where image features and gene features are both converted into embedding spans and then scattered into the final `inputs_embeds` sequence before language-model forward or generation.

## Training Workflows

### 1. Gene bridge distillation with `qformer/`

The `qformer/` directory isolates training for the gene bridge modules:

- `gene_qformer`
- `gene_projector`
- optionally an auxiliary classification head in the more complete training path

This stage is useful when the core multimodal model already exists but the gene branch needs better alignment with the language-model representation space.

There are three main scripts:

| File | Purpose |
| --- | --- |
| `qformer/train_gene_bridge_distill.py` | Simplest single-GPU bridge distillation. |
| `qformer/train_gene_bridge_distill_ddp.py` | Distributed version with cross-rank negatives. |
| `qformer/train_gene_bridge_distill_real_processor.py` | Preferred current training path using the real processor and reference-gene alignment. |

After distillation, `qformer/inject_gene_bridge_weights.py` copies the trained bridge weights into a full sharded model directory.

### 2. Fine-tuning with `finetune-gene/`

The `finetune-gene/` directory contains an earlier end-to-end training stack built around Hugging Face `Trainer` and DeepSpeed.

Important files include:

| File | Purpose |
| --- | --- |
| `finetune-gene/finetune.py` | Main fine-tuning entrypoint. |
| `finetune-gene/dataset.py` | Multimodal dataset loader for text, image, and gene inputs. |
| `finetune-gene/trainer.py` | Custom trainer wrapper. |
| `finetune-gene/gene_tokenizer.py` | Simpler tokenizer implementation used in this path. |
| `finetune-gene/finetune_1123-2.sh` | Example training launcher. |

Use this directory when reproducing older runs or when you specifically want the Hugging Face `Trainer`-based training flow.

### 3. SFT and training scripts with `pretrain-gene/`

The `pretrain-gene/` directory contains the cleaner GitHub-facing workflow for practical experiments. It includes:

- Swift model registration for MiniCPM-V + gene pipelines
- gene-only, vision-only, and gene+vision SFT launch scripts
- C2S training utilities
- CellWhisperer-LLaVA training utilities

| File / Folder | Purpose |
| --- | --- |
| `pretrain-gene/src/pretrain_gene/swift_minicpm_gene_register.py` | Swift registration for the gene-aware MiniCPM-V path. |
| `pretrain-gene/src/pretrain_gene/swift_minicpm_gene_qformer_register.py` | Swift registration for the gene + Q-Former variant. |
| `pretrain-gene/scripts/` | Shell entrypoints for gene-only, vision-only, and gene+vision SFT, plus inference wrappers. |

## Recommended Starting Points

If your goal is:

- understand the architecture: start with `model/`
- train or improve the gene bridge: start with `qformer/`
- reproduce earlier fine-tuning experiments: start with `finetune-gene/`
- run the cleaned SFT and downstream training scripts: start with `pretrain-gene/`
