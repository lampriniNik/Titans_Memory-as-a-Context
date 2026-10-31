# Titans: A Small-Scale Reproduction Study
### Learning to Memorize at Test Time — Retrieval, Reasoning & Long-Term Forecasting

A small-scale experimental reproduction of **[Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)** by Behrouz, Zhong, and Mirrokni (Google Research, 2025).

This project investigates the capabilities and limitations of the Titans architecture under constrained computational resources, using a single Google Colab T4 GPU.

**Course project:** Speech and Natural Language Processing  
**Institution:** School of Electrical and Computer Engineering, National Technical University of Athens (NTUA)  
**Date:** June 2026

---

## Overview

Transformers rely on self-attention to integrate information across a sequence. However, the computational and memory requirements of attention grow quadratically with context length.

Titans introduces a **Neural Memory** module that learns and updates its parameters at test time, allowing information to be retained beyond the conventional attention window.

The objective of this project is to investigate whether the main capabilities demonstrated in the original Titans paper can be reproduced using substantially smaller models and limited computational resources.

We focus on three primary research directions:

1. Long-context information retrieval.
2. In-context multi-hop reasoning.
3. Long-term time-series forecasting.

An additional synthetic retrieval experiment provides a controlled environment for studying associative recall.

## Key Findings

| Experiment | Task | Main Finding |
|---|---|---|
| I | Needle-in-a-Haystack | Positive — MAC Titan extends retrieval beyond its training context. |
| II | Multi-hop Reasoning | Negative — MAC remains near chance under the available training budget. |
| III | Time-Series Forecasting | Positive — LMM outperforms the Transformer at the longest prediction horizon. |
| Supplementary | Synthetic S-NIAH | Controlled investigation of multi-key associative retrieval. |

**Central observation:** Neural Memory can provide advantages when information must persist beyond the attention window. However, the ability to learn discrete cross-segment retrieval appears substantially more demanding than compressing continuous temporal structure.

---

## Repository Structure

```text
Titans-Research/
│
├── README.md
│
├── notebooks/
│   ├── experiment1_niah_wikitext.ipynb
│   ├── experiment1_synthetic_NIAH.ipynb
│   ├── experiment2_reasoning.ipynb
│   └── experiment3_timeseries_ETTh1.ipynb
│
├── report.pdf
│
└── requirements.txt
```

All four Jupyter notebooks are included to document the complete experimental work, including the primary experiments and supplementary investigations.

## Experiments

### Experiment I — Long-Context Retrieval

**Notebook:** `experiment1_niah_wikitext.ipynb`

This experiment investigates the ability of a Memory-as-Context (MAC) Titan to retrieve information from sequences substantially longer than its training context.

A baseline Transformer and a MAC Titan are trained on WikiText-2 using a context length of 1,024 tokens.

Evaluation is performed across:

- Context lengths: 1K, 2K, 4K, 8K and 16K tokens.
- Needle positions: 10%, 30%, 50%, 70% and 90%.
- Attention visualizations to investigate retrieval behavior.

**Main finding:**

The MAC Titan maintains strong retrieval performance beyond its training context, while the baseline Transformer becomes increasingly limited by its attention window.

The reported results demonstrate successful retrieval up to four times the training context, although both architectures struggle at substantially larger context lengths.

### Supplementary Experiment — Synthetic Multi-Key Retrieval

**Notebook:** `experiment1_synthetic_NIAH.ipynb`

This experiment investigates associative recall using synthetically generated sequences.

Unlike the WikiText-based retrieval experiment, the synthetic benchmark provides a controlled environment in which relevant information and distractors can be systematically varied.

The evaluation investigates:

- **Depth Test:** Retrieval performance at different needle positions.
- **Scaling Test:** Performance as context length increases.
- **Multi-Needle Test:** Retrieval under increasing numbers of distractors.

The purpose of this experiment is to provide an additional controlled investigation of retrieval behavior and the effect of context scaling.

### Experiment II — In-Context Multi-Hop Reasoning

**Notebook:** `experiment2_reasoning.ipynb`

This experiment investigates whether the MAC Titan can combine multiple supporting facts embedded in long, noisy contexts.

The benchmark follows a BABILong-style approach, using WikiText-2 as background text and synthetic bAbI-style facts.

Two tasks are evaluated:

- **QA1:** Single supporting fact (sanity check).
- **QA2:** Two supporting facts (multi-hop reasoning).

The evaluation considers different context lengths and distances between supporting facts.

**Main finding: Negative result.**

The baseline Transformer achieves approximately 94% mean accuracy, whereas the MAC Titan remains near chance level at approximately 15%.

The MAC training loss plateaus at:

`ln(6) ≈ 1.7918`

This indicates that the model fails to learn the correct answer among the six candidate locations.

Several potential implementation issues were investigated, including:

- Data leakage.
- Initialization problems.
- Learning-rate scheduler inconsistencies.
- An overly difficult prediction objective.

None of these modifications resolved the failure.

We attribute this result primarily to the computational and training requirements of learning cross-segment retrieval through the MAC inner loop. The original paper uses substantially larger models and billions of training tokens, which are beyond the scope of our available resources.

### Experiment III — Long-Term Time-Series Forecasting

**Notebook:** `experiment3_timeseries_ETTh1.ipynb`

This experiment investigates the ability of Neural Memory to capture long-term temporal dependencies.

Following the forecasting direction of the original Titans paper, the standalone Long-term Memory Module (LMM) is evaluated within a Simba-style forecasting pipeline on the **ETTh1 dataset**.

Experimental configuration:

- Dataset: ETTh1.
- Input sequence length: 512.
- Variables: 7.
- Prediction horizons: 96, 192, 336 and 720 hours.
- Channel-independent patching.
- Parameter-matched model comparisons.

The evaluated architectures include:

- LMM (Titans).
- PatchTransformer.
- PatchLSTM.
- NaïveLast baseline.

**Main finding: Positive result.**

The LMM demonstrates an increasing relative advantage over the Transformer as the prediction horizon grows.

| Model | 96h | 192h | 336h | 720h | Mean |
|---|---:|---:|---:|---:|---:|
| LMM (Titans) | 0.440 | 0.487 | 0.553 | 0.635 | 0.529 |
| PatchTransformer | 0.423 | 0.470 | 0.545 | **0.666** | 0.526 |
| PatchLSTM | 0.436 | 0.494 | **0.521** | **0.621** | **0.518** |
| NaïveLast | 1.705 | 1.780 | 1.832 | 1.986 | 1.826 |

*Test-set MSE — lower values indicate better forecasting performance.*

At the longest prediction horizon (720 hours), LMM achieves an MSE of 0.635 compared with 0.666 for the Transformer, corresponding to approximately **4.7% lower forecasting error**.

**Important qualification:** LMM does not achieve the best overall mean MSE. PatchLSTM obtains the lowest mean error (0.518), compared with 0.529 for LMM and 0.526 for PatchTransformer.

Therefore, the main finding is not that Titans outperforms all baselines overall, but that its relative performance improves with increasing prediction length and it overtakes the Transformer at the longest horizon.

This qualitatively reproduces the long-term memory advantage discussed in the original paper.

---

## Experimental Environment

The experiments were designed to run under constrained computational resources.

| Component | Configuration |
|---|---|
| Compute | Google Colab T4 GPU |
| Framework | PyTorch |
| Titans implementation | titans-pytorch |
| Language models / tokenization | Hugging Face Transformers |
| Dataset utilities | Hugging Face Datasets |
| Numerical computing | NumPy, Pandas |
| Visualization | Matplotlib |
| Forecasting dataset | ETTh1 |

The notebooks contain their respective installation commands and experimental configurations.

### Running the Experiments

1. Open the desired notebook in Google Colab.
2. Select a GPU runtime (T4 where available).
3. Execute the notebook cells in order.
4. Set the `MODE` variable to select the execution procedure where supported.

Typical execution modes:

- `all`: Training and evaluation.
- `train`: Training only.
- `eval`: Evaluation using previously saved checkpoints.

Some notebooks also provide additional visualization or checkpoint-saving functionality.

**Note:** Evaluation-only execution requires the corresponding trained checkpoints.

---

## Limitations

This project is a small-scale research reproduction rather than an official implementation of Titans.

The experimental results should be interpreted in the context of the available computational budget.

Important limitations include:

- Significantly smaller models than those used in the original paper.
- Reduced training-token budgets and training iterations.
- Limited ability to reproduce large-scale cross-segment memory learning.
- Differences in model implementations and experimental pipelines.

The negative reasoning result is reported alongside the positive retrieval and forecasting findings to provide a balanced account of what can and cannot be reproduced under constrained resources.

The forecasting results also show that the LMM does not outperform every baseline across all prediction horizons, highlighting the importance of evaluating performance across multiple metrics and forecasting settings.

---

## References

1. Behrouz, A., Zhong, P., & Mirrokni, V. (2025). *Titans: Learning to Memorize at Test Time*. Google Research.
2. Di Nepi et al. (2025). *Titans Revisited*.
3. [Titans PyTorch Implementation — lucidrains](https://github.com/lucidrains/titans-pytorch)
4. [ETTh1 / ETDataset](https://github.com/zhouhaoyi/ETDataset)

---

**Project:** Titans — Learning to Memorize at Test Time  
**Institution:** National Technical University of Athens (NTUA)  
**Department:** School of Electrical and Computer Engineering
