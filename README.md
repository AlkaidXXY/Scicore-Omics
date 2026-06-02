<div align="center">

# SciCore-Omics

**A tri-modal foundation model unifying histology, spatial transcriptomics, and biological language for spatial biology.**

[![Hugging Face Model](https://img.shields.io/badge/🤗%20Model-openbmb%2FSciCore--Omics-yellow)](https://huggingface.co/openbmb/SciCore-Omics)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Demo-SciCore--Omics-blue)](https://huggingface.co/spaces/Alkaidxxy/SciCore-Omics)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/OpenBMB/Scicore-Omics?style=social)](https://github.com/OpenBMB/Scicore-Omics)

[Model Weights](https://huggingface.co/openbmb/SciCore-Omics) ·
[Online Demo](https://huggingface.co/spaces/Alkaidxxy/SciCore-Omics) ·
[Quick Start](#quick-start) ·
[Examples](#examples) ·
[Performance](#performance) ·
[Citation](#citation)

[English](README.md) | [中文](README_zh.md)

</div>

---

## News

* **2026-06**: SciCore-Omics model weights are publicly available on Hugging Face: [`openbmb/SciCore-Omics`](https://huggingface.co/openbmb/SciCore-Omics).
* **2026-06**: The online demo is available through Hugging Face Spaces: [`Alkaidxxy/SciCore-Omics`](https://huggingface.co/spaces/Alkaidxxy/SciCore-Omics).
* **2026-06**: Training and inference code has been released in this repository.

---

## Overview

**SciCore-Omics** is a gene-aware multimodal foundation model for joint reasoning over **histology images**, **spatial transcriptomic profiles**, and **biological language**.

Built on a MiniCPM-V-style multimodal language-model stack, SciCore-Omics introduces a dedicated transcriptomic branch that encodes gene-expression profiles with **NicheFormer**, compresses gene representations through a **Gene Q-Former**, and projects them into the language-model token space through a **Gene Projector**.

The model is designed for spatial biology and pathology scenarios where tissue morphology and molecular states should be interpreted together rather than treated as isolated modalities.

<p align="center">
  <img src="figs/fig1.png" width="90%">
</p>

---

## Highlights

* **Tri-modal foundation model for spatial biology**
  SciCore-Omics links histology images, spatial transcriptomics, and biological language in a unified autoregressive modeling framework.

* **Dedicated gene branch**
  The model uses NicheFormer, a Gene Q-Former, and a Gene Projector to transform transcriptomic profiles into LLM-compatible embeddings.

* **Image-gene-text reasoning**
  SciCore-Omics supports image-only, gene-only, and image-gene joint inputs, enabling morphology-aware and molecule-aware biological reasoning.

* **Staged training pipeline**
  The repository provides separate training stages for gene-bridge distillation, Swift-based CPT/SFT, and GSPO/PPO-style reinforcement learning refinement.

* **Public release**
  Model weights, online demo, local inference code, and training entrypoints are publicly available.

---

## What Can SciCore-Omics Do?

SciCore-Omics can be used for research tasks such as:

* histology-conditioned biological description generation;
* transcriptome-conditioned biological description generation;
* joint image-gene reasoning over spatial omics spots;
* spatial domain recognition;
* gene-expression-related reasoning;
* pathology and tissue-level question answering;
* preliminary case-level molecular interpretation from histology images.

> **Note**: SciCore-Omics is released for research use. It is not a standalone clinical diagnostic system.

---

## Model Release

| Item            | Status    | Link                                                                               |
| --------------- | --------- | ---------------------------------------------------------------------------------- |
| Model weights   | Available | [`openbmb/SciCore-Omics`](https://huggingface.co/openbmb/SciCore-Omics)            |
| Online demo     | Available | [`Alkaidxxy/SciCore-Omics`](https://huggingface.co/spaces/Alkaidxxy/SciCore-Omics) |
| Source code     | Available | [`OpenBMB/Scicore-Omics`](https://github.com/OpenBMB/Scicore-Omics)                |
| License         | Available | [Apache-2.0](LICENSE)                                                              |
| Training code   | Available | `train-distill-gene/`, `train-swift-cpt-sft/`, `train-rl/`                         |
| Local inference | Available | `eval/`                                                                  |

---

## Repository Structure

```text
Scicore-Omics/
├── model/                    # Core model, processor, tokenizer, and gene branch definitions
├── eval/                     # Minimal inference and evaluation examples
├── figs/                     # Figures used in README and documentation
├── train-distill-gene/       # Gene bridge distillation scripts
├── train-swift-cpt-sft/      # Swift-based CPT/SFT training scripts
├── train-rl/                 # GSPO/PPO-style RL refinement pipeline
├── environment.yml           # Conda environment specification
├── LICENSE                   # Apache-2.0 license
└── README.md                 # Project documentation
```

If you are new to the codebase, the recommended reading order is:

1. `eval/` — run the model first;
2. `model/` — understand the architecture;
3. `train-distill-gene/` — understand gene-branch alignment;
4. `train-swift-cpt-sft/` — understand CPT/SFT training;
5. `train-rl/` — understand reinforcement learning refinement.

---

## Quick Start

### Option 1: Try the Online Demo

The fastest way to try SciCore-Omics is through the Hugging Face Space:

👉 https://huggingface.co/spaces/Alkaidxxy/SciCore-Omics

The demo supports scientific questions with uploaded gene-expression files and/or histology images.

---

### Option 2: Run Local Inference

#### 1. Clone the repository

```bash
git clone https://github.com/OpenBMB/Scicore-Omics.git
cd Scicore-Omics
```

#### 2. Create the environment

```bash
conda env create -f environment.yml
conda activate OMICS
```

The reference environment was developed on Linux with NVIDIA A800-SXM4-80GB GPUs.

`flash-attn` can be sensitive to CUDA, PyTorch, and GPU versions. If installation fails, please adjust the `flash-attn` version according to your local environment.

#### 3. Download the model weights

You can download the public weights from Hugging Face:

```bash
huggingface-cli download openbmb/SciCore-Omics \
  --local-dir ./weights/SciCore-Omics
```

Alternatively, you can directly load the model by setting:

```text
model_path = "openbmb/SciCore-Omics"
```

#### 4. Run a minimal example

```bash
python eval/example.py \
  --model_path ./weights/SciCore-Omics \
  --image_path examples/assets/example.png \
  --gene_path examples/assets/example.h5ad \
  --prompt "Please describe the tissue morphology and molecular state of this sample."
```

Expected output:

```text
The model generates a natural-language response describing tissue morphology,
transcriptomic context, and potentially relevant biological processes.
```

---

## Before You Run

Please make sure you have:

* a CUDA-enabled GPU environment;
* the SciCore-Omics model weights from Hugging Face;
* a histology image in `.png`, `.jpg`, or `.jpeg` format;
* a spatial transcriptomics file in `.h5ad` format;
* gene names compatible with the gene tokenizer resources under `model/gene_tokenizer/`.

---

## Examples

### Image + Gene Input

```bash
python eval/example.py \
  --model_path openbmb/SciCore-Omics \
  --image_path examples/assets/example.png \
  --gene_path examples/assets/example.h5ad \
  --prompt "Analyze this histology image together with its spatial transcriptomic profile."
```

### Gene-only Prompt

```bash
python eval/example.py \
  --model_path openbmb/SciCore-Omics \
  --gene_path examples/assets/example.h5ad \
  --prompt "Describe the biological state represented by this transcriptomic profile."
```

### Image-only Prompt

```bash
python eval/example.py \
  --model_path openbmb/SciCore-Omics \
  --image_path examples/assets/example.png \
  --prompt "Describe the tissue morphology in this histology image."
```

> The current `eval/example.py` may need to be adapted depending on your local input format. See [Input Format](#input-format) for details.

---

## Input Format

### Histology Image

Supported formats:

```text
.png
.jpg
.jpeg
```

Recommended preprocessing:

* use RGB images;
* avoid extremely large raw whole-slide images for minimal examples;
* crop or tile whole-slide images before inference when needed.

### Gene Expression File

The model expects spatial transcriptomic input in `.h5ad` format.

A typical `.h5ad` file should contain:

```text
adata.X          # gene-expression matrix
adata.var_names  # gene names
adata.obs        # spot/cell metadata, optional
adata.obsm       # spatial coordinates, optional
```

Minimal expected structure:

```python
import anndata as ad

adata = ad.read_h5ad("example.h5ad")
print(adata.X.shape)
print(adata.var_names[:10])
```

The gene names should be compatible with the tokenizer vocabulary and reference resources under:

```text
model/gene_tokenizer/
```

---

## Models

| Model         | Hugging Face                                                            | Modalities          | Main components                                                         | License    |
| ------------- | ----------------------------------------------------------------------- | ------------------- | ----------------------------------------------------------------------- | ---------- |
| SciCore-Omics | [`openbmb/SciCore-Omics`](https://huggingface.co/openbmb/SciCore-Omics) | Image + Gene + Text | MiniCPM-V-style backbone + NicheFormer + Gene Q-Former + Gene Projector | Apache-2.0 |

### Hardware

The reference environment was developed with:

| Item        | Reference setting          |
| ----------- | -------------------------- |
| OS          | Linux                      |
| GPU         | NVIDIA A800-SXM4-80GB      |
| Precision   | bfloat16 / mixed precision |
| Environment | `environment.yml`          |

The minimum GPU memory required for inference depends on image resolution, gene-token length, precision, and generation length.

---

## Core Idea

SciCore-Omics augments a MiniCPM-V-style vision-language model with a transcriptomic pathway:

```text
gene expression (.h5ad)
  -> gene tokenizer
  -> NicheFormer gene encoder
  -> Gene Q-Former
  -> Gene Projector
  -> <gene> span embeddings in the LLM token space

histology image
  -> vision tower
  -> image resampler
  -> <image> span embeddings in the LLM token space

text prompt
  -> tokenizer
  -> text embeddings

all modalities
  -> merged input embeddings
  -> autoregressive language model
  -> biological natural-language response
```

The transcriptomic signal is not simply converted into plain text. Instead, it is encoded as dense multimodal embeddings and injected into the LLM token sequence. This allows the model to reason over molecular and morphological information within a unified language-generation interface.

---

## Model Architecture

The core implementation lives in `model/`.

| File / Folder                        | Purpose                                                                                      |
| ------------------------------------ | -------------------------------------------------------------------------------------------- |
| `model/configuration_minicpm.py`     | Defines the MiniCPM-V-style multimodal configuration with vision and gene settings.          |
| `model/configuration_nicheformer.py` | Defines the NicheFormer gene-encoder configuration.                                          |
| `model/modeling_nicheformer.py`      | Implements the NicheFormer transformer encoder over gene tokens.                             |
| `model/gene_qformer_module.py`       | Implements the Gene Q-Former bridge for compressing variable-length gene embeddings.         |
| `model/gene_projector_module.py`     | Projects Gene Q-Former outputs into the LLM hidden space.                                    |
| `model/modeling_minicpmv.py`         | Integrates the LLM, vision tower, resampler, NicheFormer, Gene Q-Former, and Gene Projector. |
| `model/processing_minicpmv.py`       | Implements the processor that packages text, image, and gene inputs.                         |
| `model/gene_tokenizer/`              | Contains gene-tokenization resources, vocabulary, and reference assets.                      |

### Gene Branch

The gene branch follows five main steps:

1. Gene expression is tokenized into a gene-token sequence.
2. `NicheFormerModel` encodes the sequence into contextual gene embeddings.
3. `GeneQFormerBiomedBERT` compresses variable-length gene embeddings into a fixed number of query tokens.
4. `GeneProjector` maps query outputs into the LLM hidden space.
5. The projected gene embeddings are inserted into the LLM input embedding stream at the gene placeholder span.

### Image Branch

The image branch follows the MiniCPM-V-style visual encoding path:

1. Histology images are processed by the vision tower.
2. Visual features are compressed through the image resampler.
3. The resulting image embeddings are inserted into the LLM input sequence at the image placeholder span.

### Language Interface

After image and gene features are projected into the LLM hidden space, SciCore-Omics keeps the standard autoregressive language-model interface. This makes it possible to use natural-language prompts for biological reasoning, question answering, and description generation.

---

## Performance

The following numbers summarize the main results reported in our current manuscript and release notes. Full benchmark scripts and detailed result tables will be released progressively.

| Task                                 | Input                      | Evaluation                           | Result Summary                                                  |
| ------------------------------------ | -------------------------- | ------------------------------------ | --------------------------------------------------------------- |
| Gene expression prediction           | Histology image            | Task-specific metrics                | 23.6–80.9% relative gains over the strongest external baselines |
| Spatial domain recognition           | Image / gene / joint input | Classification metrics               | Improved spatial-domain prediction with multimodal inputs       |
| Histopathology classification        | Image-only, zero-shot      | Mean accuracy across four benchmarks | +6.16 percentage points over GPT-5                              |
| Breast cancer case-level reasoning   | H&E image only             | Expert evaluation                    | Evaluated on 10 breast cancer cases                             |
| Transcriptome-conditioned generation | Gene input                 | BLEU / ROUGE / BERTScore             | Progressive improvement across staged training                  |

> More detailed benchmark tables, evaluation scripts, and reproduction instructions will be added in future releases.

---

## Training Pipeline

SciCore-Omics uses a staged training strategy instead of a single monolithic training script.

The released training code is organized into three main parts:

```text
train-distill-gene/      # gene bridge distillation
train-swift-cpt-sft/     # Swift-based CPT/SFT
train-rl/                # GSPO/PPO-style RL refinement
```

### Stage 1: Gene Bridge Distillation

The `train-distill-gene/` directory isolates training for the gene-branch alignment modules:

* `gene_qformer`
* `gene_projector`
* optionally `gene_cls_head` in extended settings

This stage is useful when the main multimodal model already exists but the transcriptomic branch needs better alignment with the language-model representation space.

Main scripts:

| Script                                                           | Purpose                                                                        |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `train-distill-gene/train_gene_bridge_distill.py`                | Single-GPU gene-bridge distillation.                                           |
| `train-distill-gene/train_gene_bridge_distill_ddp.py`            | Distributed distillation with cross-rank negatives.                            |
| `train-distill-gene/train_gene_bridge_distill_real_processor.py` | Preferred training path using the real processor and reference-gene alignment. |
| `train-distill-gene/inject_gene_bridge_weights.py`               | Injects trained gene-bridge weights into a full model directory.               |

### Stage 2: CPT/SFT with Swift

The `train-swift-cpt-sft/` directory contains Swift-based CPT/SFT entrypoints for the gene-aware MiniCPM-V model.

These scripts use `ms-swift` through commands such as:

```bash
swift pt
swift sft
```

The gene-specific logic is injected through Swift's custom registration mechanism rather than by modifying the Swift framework itself.

The custom registration file is:

```text
train-swift-cpt-sft/register/my_register_qformer.py
```

This register file defines the gene-aware MiniCPM-V model and template path. It handles `.h5ad` gene inputs, tokenizes gene names, builds `gene_input_ids`, `gene_attention_mask`, and `gene_bound`, expands the gene placeholder into the Gene Q-Former span, and exposes the resulting fields to the model batch.

Main files:

| File / Folder                                         | Purpose                                                              |
| ----------------------------------------------------- | -------------------------------------------------------------------- |
| `train-swift-cpt-sft/register/my_register_qformer.py` | Swift custom register file for gene-aware MiniCPM-V + Gene Q-Former. |
| `train-swift-cpt-sft/script/cpt-example.sh`           | Example continued pretraining launcher.                              |
| `train-swift-cpt-sft/script/sft-example.sh`           | Example supervised fine-tuning launcher.                             |

Before running the training scripts, please replace placeholder paths such as:

```text
YOUR_SOURCE_PATH
YOUR_CONDA_ENV
BASE_MODEL
DATA_DIR
OUTPUT_PATH
LOG_PATH
GENE_VOCAB_PATH
```

### Stage 3: Reinforcement Learning Refinement

The `train-rl/` directory contains a GSPO/PPO-style reinforcement learning pipeline for score-guided multimodal optimization.

It separates rollout generation, reference-model scoring, and distributed actor updates.

Main components:

| File                        | Purpose                                                                                                       |
| --------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `train-rl/gen_worker.py`    | Samples examples, builds candidate batches, computes old-policy log probabilities, and prepares rollout data. |
| `train-rl/ref_server.py`    | Runs a Flask reference server and computes reference-model token log probabilities.                           |
| `train-rl/finetune_gspo.py` | Runs the DDP training loop with a clipped GSPO/PPO-style objective and KL penalty.                            |

The RL script freezes the full model by default and selectively trains the gene bridge, image resampler, and final LLM layers depending on the modality composition of the current rollout.

---

## Responsible Use and Limitations

SciCore-Omics is released for research in spatial biology, pathology AI, and multimodal biomedical modeling.

The model should **not** be used as a standalone clinical diagnostic system. Its outputs may contain errors, incomplete biological interpretations, or unsupported hypotheses. Any biomedical or clinical conclusion should be reviewed by qualified domain experts and validated with appropriate experimental or clinical evidence.

The model is not designed to replace pathologists, molecular biologists, clinicians, or regulatory decision-making processes.

Potential limitations include:

* sensitivity to input image quality and preprocessing;
* sensitivity to gene vocabulary and `.h5ad` formatting;
* possible hallucination in biological explanations;
* limited generalization to tissues, diseases, or platforms not represented during training;
* lack of prospective clinical validation.

---

## Citation

If you find SciCore-Omics useful for your research, please consider citing our work:

```bibtex
@misc{xiao2026scicoreomics,
  title        = {SciCore-Omics: a tri-modal foundation model unifying histology, spatial transcriptomics and language for spatial biology},
  author       = {Xiao, Xinyu and Li, Yunfei and Zeng, Zheni and others},
  year         = {2026},
  note         = {Manuscript in preparation}
}
```

The formal citation will be updated after the manuscript is publicly available.

---

## Contact

If you have questions, suggestions, or bug reports, please open an issue in this repository or contact:

* Xinyu Xiao: [xinyuxiao1@outlook.com](mailto:xinyuxiao1@outlook.com)

---

## License

This project is released under the [Apache-2.0 License](LICENSE).
