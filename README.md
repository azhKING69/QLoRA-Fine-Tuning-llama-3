# QLoRA Fine-Tuning of Llama 3.1 for Structured Data Extraction from OSHA Incident Narratives

## Project Overview

This project presents a complete machine learning pipeline for automatically extracting structured JSON data from unstructured OSHA (Occupational Safety and Health Administration) construction incident narratives. The pipeline spans the full lifecycle: from raw government data ingestion and weak-supervision labeling, through QLoRA fine-tuning of a Llama 3.1 8B language model, to local offline deployment via a quantized GGUF model served through a Streamlit web interface.

The fine-tuned model achieved a **77.8% extraction accuracy**, representing a **21.5% absolute improvement** over the zero-shot baseline of 64.0%, while completely eliminating JSON formatting errors that plagued the base model.

---

## Problem Statement

OSHA publishes a publicly available dataset of severe workplace injuries reported by employers across the United States. Each record contains a free-text narrative describing the incident in unstructured, often inconsistent language written by different inspectors, employers, and safety officers. Alongside the narrative, OSHA provides categorical codes for event type, injury nature, body part affected, and source of injury, but these codes are drawn from an extremely fragmented taxonomy containing hundreds of granular categories.

The goal of this project was to build a local, offline AI system capable of reading a raw incident narrative and producing a clean, standardized JSON object with six fields, effectively replacing the need for manual human classification.

---

## Dataset

**Source:** [OSHA Severe Injury Reports (January 2015 - December 2023)](https://www.osha.gov/severeinjury)

The raw dataset contains injury records across all industries. For this project, the data was filtered exclusively to the **Construction sector (NAICS code prefix 23)**, yielding approximately **18,000 usable records** after removing entries with missing narratives.

Each record contains:
- A free-text `Final Narrative` describing the incident
- Categorical OSHA codes for Event, Nature of Injury, Part of Body, and Source
- Boolean flags for Hospitalization and Amputation

---

## Schema Design (V3 Taxonomy)

The raw OSHA taxonomy contains hundreds of highly specific codes that are impractical for model training. A consolidated "V3" schema was designed to reduce these into a manageable set of semantically meaningful categories while maintaining greater than 85% coverage across all fields.

| Field | Categories | Examples |
|-------|-----------|----------|
| Event Type | 16 | `fall_to_lower_level`, `struck_by_object`, `electrocution`, `caught_in_between` |
| Injury Nature | 14 | `fracture`, `amputation`, `laceration`, `burn`, `intracranial` |
| Body Part | 15 | `finger_hand`, `head_brain`, `leg`, `multiple`, `back_spine` |
| Source Equipment | 27 | `ladder`, `scaffold`, `power_tool`, `vehicle_heavy_equip`, `rebar` |
| Hospitalized | 2 | `true`, `false` |
| Amputation | 2 | `true`, `false` |

The model was trained to classify free-text across **4 categorical variables spanning 72 distinct classes**, plus two boolean flags, for a total of **6 output fields per prediction**.

---

## Weak Supervision Pipeline

Rather than manually labeling 18,000 training examples, a rule-based weak supervision engine was built in Python. This engine uses keyword matching and regex patterns to map the raw OSHA string codes into the V3 schema categories.

**Coverage metrics achieved:**
- Event Type: >85%
- Injury Nature: >85%
- Body Part: >85%
- Source Equipment: >85%

Records that could not be confidently mapped were assigned the `other` category.

### Noise Quantification

A critical step in this project was quantifying the inherent label noise in the weakly supervised training data. By comparing the automated mappings against human annotations on a 200-row sample, the agreement rate was measured at **87% to 95%** across fields, establishing that the training data contains approximately **10% label noise**. This measurement proved essential for interpreting the learning curve results described below.

---

## Gold Standard Evaluation Set

A stratified random sample of **200 records** was held out from the training pool and manually annotated by a human reviewer to serve as the ground-truth evaluation set. Each record was independently read and labeled across all six fields without reference to the original OSHA codes.

This gold standard dataset was used consistently across all model evaluations to ensure fair, apples-to-apples comparisons. It was never included in any training split.

---

## Baseline Evaluation (Zero-Shot)

Before any fine-tuning, the base `Llama 3.1 8B Instruct` model was evaluated against the full 200-row gold standard to establish a performance floor.

**Zero-Shot Baseline Accuracy: 64.0%**

Failure analysis revealed two primary failure modes:
1. **Enum hallucination:** The base model frequently invented category names not present in the schema (e.g., outputting `"broken_bone"` instead of `"fracture"`).
2. **JSON formatting errors:** The model occasionally wrapped output in markdown code fences, added explanatory text, or produced malformed JSON.

---

## Fine-Tuning Approach

### Architecture: QLoRA (Quantized Low-Rank Adaptation)

The base Llama 3.1 8B model (approximately 15 GB at 16-bit precision) was loaded into GPU memory using **4-bit quantization via bitsandbytes**, reducing its memory footprint by **75%** to approximately 5 GB. This enabled training on a single NVIDIA T4 GPU (15 GB VRAM) available through the free tier of Google Colab.

Small, trainable LoRA adapter weights (kept in 16-bit precision for gradient accuracy) were attached to all attention and feed-forward projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) with rank `r=16`.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | `unsloth/Meta-Llama-3.1-8B-Instruct` |
| Quantization | 4-bit (bitsandbytes NF4) |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 16 |
| Learning Rate | 2e-4 |
| Batch Size | 2 (per device) |
| Gradient Accumulation | 4 steps |
| Optimizer | AdamW 8-bit |
| Epochs | 1 |
| Scheduler | Linear warmup + decay |
| Framework | Unsloth + HuggingFace TRL (SFTTrainer) |

### Training Data Format

Training examples were formatted in the ChatML conversational structure expected by the Llama 3.1 Instruct model:

```json
{
  "messages": [
    {"role": "system", "content": "[extraction instructions with full enum lists]"},
    {"role": "user", "content": "Narrative: Worker fell 15 feet from scaffolding..."},
    {"role": "assistant", "content": "{\"event_type\": \"fall_to_lower_level\", ...}"}
  ]
}
```

---

## Learning Curve Experiment

To determine the optimal training set size and identify the point of diminishing returns, a **strictly nested** learning curve experiment was conducted across three dataset sizes. "Strictly nested" means the 1,000-row subset is fully contained within the 3,000-row subset, which is fully contained within the 5,000-row subset. This ensures differences in performance are attributable solely to data volume, not data composition.

### Results

| Model | Training Rows | Accuracy | Delta vs. Baseline |
|-------|--------------|----------|-------------------|
| Zero-Shot Baseline | 0 | 64.0% | -- |
| Fine-Tuned (1k) | 1,000 | **77.8%** | +13.8% |
| Fine-Tuned (3k) | 3,000 | 77.2% | +13.2% |

### Key Findings

1. **Format mastery at 1k:** Fine-tuning on just 1,000 examples completely eliminated all JSON formatting errors and enum hallucinations that plagued the zero-shot baseline. The model learned to produce valid, schema-compliant JSON 100% of the time.

2. **Accuracy plateau at 1k:** Tripling the training data from 1,000 to 3,000 rows produced no improvement in accuracy. The 3k model actually scored 0.6% lower than the 1k model.

3. **Label noise ceiling:** The plateau is directly attributable to the approximately 10% label noise in the weakly supervised training data. Beyond 1,000 examples, additional data reinforced noisy labels rather than teaching the model new patterns. This finding demonstrates that to exceed the 78% accuracy ceiling, the project would require higher-fidelity human annotations rather than higher data volume.

4. **Training loss trajectory (1k model):** Loss decreased from 1.74 at step 0 to 0.20 at convergence, indicating strong learning signal within the first epoch.

---

## Local Deployment

### Model Export Pipeline

After training, the LoRA adapter weights were fused permanently into the base model. The merged 16-bit model (approximately 15 GB) was then quantized using the `Q4_K_M` quantization scheme and exported to the **GGUF (GPT-Generated Unified Format)**, producing a single standalone file of **4.7 GB**.

GGUF is a binary format specifically designed for efficient CPU and unified-memory inference on consumer hardware, enabling the model to run locally on a MacBook without requiring a dedicated GPU.

### Serving Infrastructure

The quantized GGUF model is served locally using **Ollama**, an open-source local LLM runtime. A custom `Modelfile` registers the fine-tuned model as a named service accessible via a local API endpoint at `localhost:11434`.

### Web Interface

A **Streamlit** web application (`05_app.py`) provides an interactive browser-based interface for testing the model. Users paste an OSHA incident narrative into a text field, and the application returns a formatted JSON extraction in real time, powered entirely by local inference with no internet connection required.

---

## Project Architecture

```
Raw OSHA CSV
     |
     v
[01_data_prep.py] --- Filters for Construction (NAICS 23)
     |                 Applies V3 weak supervision mappings
     |                 Exports train_pool.csv + gold_eval_set.csv
     v
[02_build_dataset.py] --- Converts CSV to ChatML JSONL format
     |                     Generates nested 1k / 3k / 5k splits
     v
[03_train_qlora.ipynb] --- QLoRA fine-tuning on Google Colab (T4 GPU)
     |                      Evaluates on gold standard
     |                      Exports merged GGUF model
     v
[04_evaluate.py] --- Local evaluation via Ollama
     |                Grades predictions against gold standard
     v
[05_app.py] --- Streamlit web UI for interactive inference
```

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| Base Model | Meta Llama 3.1 8B Instruct |
| Fine-Tuning | Unsloth, HuggingFace TRL (SFTTrainer), PEFT |
| Quantization (Training) | bitsandbytes (4-bit NF4) |
| Quantization (Export) | llama.cpp (Q4_K_M) |
| Model Format | GGUF |
| Local Inference | Ollama |
| Web Interface | Streamlit |
| Training Hardware | Google Colab (NVIDIA T4, 15 GB VRAM) |
| Deployment Hardware | Apple MacBook (CPU / Unified Memory) |

---

## Repository Structure

```
.
├── 01_data_prep.py            # Data cleaning, filtering, and weak supervision mapping
├── 02_build_dataset.py        # ChatML JSONL generation for LLM training
├── 03_train_qlora.ipynb       # QLoRA fine-tuning notebook (Google Colab)
├── 04_evaluate.py             # Model evaluation and accuracy grading
├── 05_app.py                  # Streamlit web UI for local inference
├── Modelfile                  # Ollama model registration file
├── gold_eval_set.csv          # 200-row ground-truth evaluation dataset with human annotation columns
├── train_1k.jsonl             # Sample ChatML formatted training data (1,000 rows)
└── README.md
```

*(Note: The massive 4.7GB `.gguf` model file and the 57MB raw OSHA `dataset.csv` are intentionally excluded from this repository. The raw data can be downloaded via the OSHA link above).*

---

## Conclusion

This project demonstrates that a relatively small, carefully constructed dataset of 1,000 weakly supervised examples is sufficient to transform a general-purpose LLM into a domain-specific structured extraction tool, achieving a 21.5% accuracy improvement while eliminating all formatting errors. The learning curve experiment further reveals that scaling synthetic training data beyond this threshold provides no benefit when the underlying labels contain noise, establishing a clear direction for future improvement: investing in annotation quality over annotation quantity.
