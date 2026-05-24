# SciCore-Omics

SciCore-Omics is a gene-aware multimodal modeling project built around the MiniCPM-V2.6 stack. The central goal of the repository is to make transcriptomic signals usable alongside natural language and tissue imagery within a single instruction-following model. In practice, this repository extends the MiniCPM-V architecture with a dedicated gene branch, provides training code for aligning that branch to the language model, and includes downstream fine-tuning and baseline evaluation pipelines for gene-centric spatial transcriptomics tasks.


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
| `src/` | TODO |
| `environment.yml` | Conda environment specification for the research stack. |

If you are new to the codebase, the most useful reading order is:

1. `model/`
2. `qformer/`
3. `pretrain-gene/`
4. `finetune-gene/`

## Quick Start

1. Environment setup

   Create the project environment from `environment.yml`:

   ```bash
   conda env create -f environment.yml
   conda activate <env-name-defined-in-environment.yml>
   ```

2. Online demo

   A live demo is available here:

   [http://166.111.5.103:15557/](http://166.111.5.103:15557/)

   This is the quickest way to inspect the current behavior while public weights are not yet released.

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

## Main Workflows

### 1. Core model loading and multimodal inference

Use `model/` when you need to load the gene-aware MiniCPM-V architecture itself.

Typical loading pattern:

```python
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained(
    "/path/to/model",
    trust_remote_code=True,
)

processor = AutoProcessor.from_pretrained(
    "/path/to/model",
    trust_remote_code=True,
)
```

This path is the right starting point if you are debugging model internals, inspecting how gene spans are inserted, or building a new downstream training or inference entrypoint.

### 2. Gene bridge distillation

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

### 3. Cleaned multimodal SFT and evaluation workflows

The `pretrain-gene/` directory contains the cleaner GitHub-facing workflow for practical experiments. It includes:

- Swift model registration for MiniCPM-V + gene pipelines
- gene-only, vision-only, and gene+vision SFT launch scripts
- C2S baseline training and evaluation
- CellWhisperer and CellWhisperer-LLaVA preparation and evaluation utilities

This is the best starting point for users who want a more organized view of the later training and evaluation stack rather than the older experimental code layout.

### 4. Earlier Hugging Face Trainer based fine-tuning

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

## Data and Input Conventions

The repository assumes several recurring data patterns:

- Transcriptomic inputs are typically stored as `.h5ad`.
- Training instances are commonly described through JSON or JSONL records that reference messages plus gene and optionally image paths.
- The processor expands the `"<gene>"` placeholder in the prompt into a dedicated token span and records the corresponding `gene_bound` indices.
- Image handling follows the MiniCPM-V processor conventions from the vision branch.

Because the code mixes model definitions with lab-specific experiments, data schemas are consistent in spirit but not always normalized into one public API. Expect to inspect the script you plan to run and adapt paths and fields accordingly.

## Environment

The project is designed for a Linux + CUDA research environment and assumes access to a relatively heavy multimodal stack. The supplied `environment.yml` is the right starting point.

At a high level, the environment expects components such as:

- PyTorch
- Transformers
- DeepSpeed
- FlashAttention
- `ms-swift`
- `anndata` and the scientific Python stack for transcriptomics data

Typical setup:

```bash
conda env create -f environment.yml
conda activate <env-name-defined-in-environment.yml>
```

Many scripts assume GPU availability, and several training paths are written with multi-GPU or distributed execution in mind.

## Practical Notes Before Running Anything

### 1. Update hardcoded paths

Several files preserve lab-local absolute paths for:

- model directories
- tokenizer resources
- training and evaluation datasets
- output directories
- custom registration files

Before running a workflow in a new environment, search for and replace those paths.

### 2. Check the gene tokenizer resources

The processor and related scripts rely on the resources under `model/gene_tokenizer/`, including:

- tokenizer implementation
- `vocab.json`
- reference `.h5ad` files used for gene alignment

Make sure these are present and that your scripts point to the correct location.

### 3. Expect mixed maturity across modules

This repository contains both:

- core model code that defines the actual architecture
- experiment-oriented scripts with narrow dataset assumptions

That is normal for a research repository, but it means you should treat script defaults as examples rather than portable production settings.

## Recommended Starting Points

If your goal is:

- understand the architecture: start with `model/`
- train or improve the gene bridge: start with `qformer/`
- run a cleaner downstream training or evaluation workflow: start with `pretrain-gene/`
- reproduce older fine-tuning experiments: start with `finetune-gene/`

## Repository Hygiene

Large artifacts should stay out of Git. In particular, avoid committing:

- checkpoints and model weights (`.pt`, `.pth`, `.bin`, `.ckpt`, `.safetensors`)
- transcriptomics datasets (`.h5ad`, `.npz`, large `.csv`)
- generated logs and outputs
- caches and temporary files

This repository is much easier to share and maintain when code, configuration, and lightweight metadata are kept separate from large training artifacts.

## Companion Documentation

Two subdirectories already benefit from dedicated folder-level documentation:

- `pretrain-gene/`
- `qformer/`

Those folder-specific READMEs are the right place for script-by-script usage details, while this top-level README is intended to explain the overall scientific and engineering structure of the project.

## Summary

SciCore-Omics is best understood as a gene-aware multimodal extension of MiniCPM-V with three layers of functionality:

1. a core model that integrates text, image, and gene embeddings
2. bridge-training utilities that align gene representations to the language model
3. downstream fine-tuning and evaluation pipelines for multimodal and gene-only tasks

If you are entering the codebase for the first time, read `model/` to understand the architecture, then move to `qformer/` and `pretrain-gene/` depending on whether your next step is representation alignment or task-level training.
