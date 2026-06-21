# Comprehensive AI Models Training Framework for Flood Detection

An end-to-end framework for detecting floods and flood-damaged buildings from satellite imagery, covering everything from data acquisition to post-flood damage assessment. The repository also includes a retrieval-based **AI Assistant** that answers natural-language questions about how each module works.

## Table of Contents
- [Overview](#overview)
- [Modules](#modules)
- [AI Assistant (Q&A System)](#ai-assistant-qa-system)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Limitations](#limitations)
- [License](#license)

## Overview

This project trains and deploys deep learning models that fuse **Sentinel-1 (radar)** and **Sentinel-2 (optical)** satellite imagery to detect flooded areas and assess building damage. It is organized into four pipeline modules, plus a supporting AI assistant that helps users (or contributors) understand each module through Q&A.

## Modules

### 1. Data Acquisition
Lets a user draw an Area of Interest (AOI) on a map and automatically finds **paired dates** where both Sentinel-1 and Sentinel-2 captured imagery within a matching time window. Both sensors are needed because radar penetrates cloud cover while optical imagery provides spectral/color detail — together they form the input "chip pairs" used downstream. Before downloading, the system estimates the number of chips, storage size, and download time.

### 2. Data Annotation
Generates water/land masks for the downloaded imagery using a **weak supervision pipeline** — physics-based rules instead of manual labeling. It scans the SAR (`SAR_ALL_BANDS`) and optical (`OPTICAL_ALL_BANDS`) directories, matches tiles by shared index ("paired tiles"), and fuses each matched radar/optical pair into a mask. Unpaired tiles are reported but skipped.

### 3. Training and Preprocessing
Trains **MUST-Former**, a transformer-based flood segmentation model, in five variants:
| Variant | Description |
|---|---|
| `MUST-Former-SAR` | Radar-only, 2 input channels |
| `MUST-Former-Optical` | Optical-only, 15 input channels |
| `MUST-Former-Projector` | Fusion via independent projection of each modality to 3 channels, then summed |
| `MUST-Former-CrossAttn` | Fusion via bi-directional cross-attention between radar and optical features |
| `MUST-Former-PCA` | Fusion via channel standardization + projection to 3 channels |

This module covers preprocessing of the paired chips, training configuration (learning rate, batch size, etc.), and checkpoint export.

### 4. Damaged Building Detection
Runs inference with a trained MUST-Former checkpoint (typically the PCA fusion variant) on new imagery, converts the predicted flood mask into geographic polygons, and cross-references them against the **Google Open Buildings** dataset to flag which buildings fall inside flooded areas. Inference uses GPU if available and falls back to CPU automatically.

## AI Assistant (Q&A System)

The assistant (`the_AI_assitant.ipynb`) is a lightweight retrieval-augmented Q&A tool that lets anyone ask questions about the four modules above and get accurate, sourced answers — without calling a large hosted LLM.

**How it works:**
1. **Knowledge base** — `qa_dataset.json` contains 822 question/answer pairs, each tagged with its module.
2. **Embedding & retrieval** — Each question is embedded with `sentence-transformers/all-MiniLM-L6-v2`. A user query is embedded the same way, and matched via cosine similarity, optionally restricted to a single module to avoid cross-module ambiguity.
3. **Confidence threshold** — If the best match scores below `SIMILARITY_THRESHOLD = 0.55`, the assistant returns a fallback message instead of guessing.
4. **Rephrasing** — The matched answer is passed to a local **Qwen2.5-1.5B-Instruct** model (GGUF, via `llama-cpp-python`) which rewords it conversationally while explicitly preserving every fact, number, and unit. Rephrased answers are cached per record to avoid recomputation.

```python
answer, module, score = answer_question(
    "What learning rate and batch size are used during training?",
    module="Training and Preprocessing"  # optional filter
)
```

## Repository Structure

```
.
├── data_acquisition/              # AOI selection, Sentinel-1/2 download & pairing
├── data_annotation/               # Weak supervision masking pipeline
├── training_preprocessing/        # MUST-Former model variants & training scripts
├── damaged_building_detection/    # Inference + building overlay analysis
├── ai_assistant/
│   ├── the_AI_assitant.ipynb      # Embedding + LLM Q&A assistant
│   └── qa_dataset.json            # 822-pair knowledge base
└── README.md
```

> Adjust this tree to match your actual folder layout before pushing to GitHub.

## Installation

```bash
pip install sentence-transformers huggingface_hub llama-cpp-python
```

The assistant downloads `Qwen/Qwen2.5-1.5B-Instruct-GGUF` (q4_k_m quantization) from the Hugging Face Hub on first run and runs entirely on CPU (`n_threads=4`).

## Usage

1. Place `qa_dataset.json` in the same directory as the notebook (or update the path).
2. Run all cells in `the_AI_assitant.ipynb`.
3. Call `answer_question(question, module=None)`:
   - `question`: natural-language question
   - `module`: one of `Data Acquisition`, `Data Annotation`, `Training and Preprocessing`, `Damaged Building Detection` (optional — omitting it searches all modules)
4. Returns a tuple: `(answer, matched_module, similarity_score)`.

## Dataset

`qa_dataset.json` — 822 question/answer pairs labeled by module:

| Module | # Q&A pairs |
|---|---|
| Data Acquisition | 213 |
| Data Annotation | 155 |
| Training and Preprocessing | 281 |
| Damaged Building Detection | 173 |

Each entry has the schema:
```json
{
  "module": "Data Acquisition",
  "question": "...",
  "answer": "..."
}
```

## Limitations

- The assistant only knows what's in `qa_dataset.json` — it does not generate novel technical answers, only retrieves and rephrases existing ones.
- Out-of-scope or unrelated questions (e.g., general knowledge) are caught by the similarity threshold and return a fallback message, but borderline questions near the threshold may occasionally be misrouted.
- Runs on CPU by default; expect a few seconds of latency per query due to the local LLM rephrasing step.

## License

_Add your chosen license here (e.g., MIT, Apache 2.0)._
