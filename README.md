# data-science-ml-ai999x — offline, verified data science and machine learning solver with step-by-step answers

[![tests](https://github.com/AI999X/data-science-ml-ai999x/actions/workflows/ci.yml/badge.svg)](https://github.com/AI999X/data-science-ml-ai999x/actions/workflows/ci.yml) [![PyPI](https://img.shields.io/pypi/v/data-science-ml-ai999x)](https://pypi.org/project/data-science-ml-ai999x/) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/AI999X/data-science-ml-ai999x)

**[Install](https://pypi.org/project/data-science-ml-ai999x/)** · **[Capability guide](https://huggingface.co/AI999X/data-science-ml-ai999x/tree/main/docs)** · **[Evaluation](https://huggingface.co/AI999X/data-science-ml-ai999x/tree/main/evaluation)** · **[AI999X on Hugging Face](https://huggingface.co/AI999X)** · **[Website](https://ai999x.com)** · **[Support: Buy Me a Coffee](https://buymeacoffee.com/ibnuljaiffarabi)**

License: Apache-2.0 · Author: Ibnul Jaif Farabi (creator and engineer) · Organization: AI999X · Version 1.0.1

Field: **Data Science & Machine Learning** · Tasks: Question Answering, Token Classification, Text Classification, Tabular Classification, Tabular Regression, Sentence Similarity, Feature Extraction

data-science-ml-ai999x is an offline, deterministic Python solver that answers data science and machine learning questions with verified results from 107 certified capabilities, gives 711 documented answers identically on Windows and Linux, and refuses rather than guesses when an answer cannot be verified.

**data-science-ml-ai999x** is a compact, deterministic data science and machine learning engine for Python. It reads a question in plain English (or a typed call), picks a certified data science and machine learning capability, computes the answer exactly, checks it, and returns the result with its steps, units, LaTeX and assumptions — or an honest refusal that says what is missing. It runs fully offline on any CPU, needs no GPU, no API key and no internet, and is built to give **zero wrong answers**: when it is not sure, it refuses instead of guessing.

[![Buy me a coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee-support%20development-FFDD00?logo=buymeacoffee&logoColor=black)](https://buymeacoffee.com/ibnuljaiffarabi) ![License](https://img.shields.io/badge/license-Apache--2.0-blue) ![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue) ![Offline](https://img.shields.io/badge/runs-offline%20on%20CPU-green)

## Highlights

- **Verified, not generated.** Every answer is computed by a certified data science and machine learning capability and checked (substituted back, dimension-checked or compared with an independent computation) before it is returned.
- **Zero wrong answers by design.** Seven explicit outcomes — SOLVED, NEEDS_INPUT, AMBIGUOUS, INVALID_INPUT, OUT_OF_SCOPE, UNRECOGNIZED_FORMAT, INTERNAL_ERROR — never a silent guess. Every one of its 711 documented answers is reproduced exactly on every tested platform.
- **Plain-English questions and a guaranteed typed interface.** `ask()` reads documented question forms; `solve(capability_id, **inputs)` always solves a supported task with valid inputs.
- **Explains itself.** Steps, formulas in LaTeX, units, assumptions and the model used, ready for notebooks, tutoring apps and homework checkers.
- **Small and portable.** Pure Python (sympy, mpmath); identical answers on Windows and Linux (Python 3.10, 3.11, 3.12, 3.13, 3.14).
- **Coverage:** classification and regression metrics, loss functions and gradients, information theory, clustering, classical machine-learning algorithms, neural-network arithmetic and ranking metrics.

## Model overview

| Property | Value |
|---|---|
| Package | `data-science-ml-ai999x` 1.0.1 (import `data_science_ml_ai999x`) |
| Type | Deterministic neuro-symbolic data science and machine learning engine (rule-based reading + certified symbolic capabilities) |
| Capabilities | 107 certified procedures |
| Knowledge pack | version 1.0.0 |
| Golden answers | 711 documented answers, identical on every tested platform |
| Platforms tested | Windows (Python 3.13), Linux x86-64 (Python 3.10, 3.11, 3.12, 3.13, 3.14) |
| Runtime | CPU only, offline; no GPU, no network, no API key |
| License | Apache-2.0 |
| Creator / engineer | Ibnul Jaif Farabi |
| Organization | AI999X |

## Quickstart

```bash
pip install data-science-ml-ai999x
```

```python
import data_science_ml_ai999x as data_science_ml

data_science_ml.ask("Binary metrics from counts: TP = 40, FP = 10, FN = 5, TN = 45").show()
# -> SOLVED — accuracy = 0.85, precision = 0.8, recall = 0.888889, specificity = 0.818182  (ml_binary_metrics_from_counts)
data_science_ml.ask("ROC AUC: y_true = [0, 1, 1, 0, 1], scores = [0.1, 0.9, 0.8, 0.3, 0.2]").show()
# -> SOLVED — auc = 0.833333, gini = 0.666667, n_pos = 3, n_neg = 2  (ml_roc_auc)
data_science_ml.ask("K-means clustering: X = [[1, 1], [1.5, 2], [3, 4], [5, 7], [3.5, 5], [4.5, 5], [3.5, 4.5]], C = [[1, 1], [5, 6]]").show()
# -> SOLVED — centroids = [[1.25, 1.5], [3.9, 5.1]], labels = [0, 0, 1, 1, 1, 1, 1], inertia = 8.525  (ml_kmeans)
data_science_ml.ask("Scaled dot-product attention: Q = [[1, 0], [0, 1]], K = [[1, 0], [0, 1], [1, 1]], V = [[1, 2], [3, 4], [5, 6]], causal = no").show()
# -> SOLVED — weights = [[0.40111209268, 0.19777581464, 0.40111209268], [0.19777581464, 0.40111209268, 0.40111209268]], output = [[3.0, 4.0], [3.40667255608, 4.40667255608]], output_sum = 14.8133  (ml_scaled_dot_product_attention)
data_science_ml.solve("ml_conv_output_size", n=224, kernel=7, stride=2, padding=3, dilation=1).show()
# -> SOLVED — output_size = 112, effective_kernel = 7  (ml_conv_output_size)
data_science_ml.ask("Who created you?").message
# -> data-science-ml-ai999x was created and engineered by Ibnul Jaif Farabi at AI999X.
```

Command line: `data-science-ml-ai999x "<question>"`, `data-science-ml-ai999x --capabilities`. The outputs above were produced by running this exact code on data-science-ml-ai999x 1.0.1.

## How every answer is verified

1. **Reading** — the question is turned into typed quantities (value, unit, kind, role) and one asked quantity; every stated number must be used or the question is refused.
2. **Planning** — only certified capabilities may answer; a capability applies only under the conditions the wording states (for example a pendulum, a liquid at rest, boiling).
3. **Computing** — exact symbolic or high-precision numeric evaluation.
4. **Checking** — dimensions, substitution back into the defining equation, and agreement between independent readings; any disagreement becomes a refusal, never a number.

Capabilities are certified before they ship: each must reproduce hand-worked reference cases, reject a deliberately wrong variant (a control), and leave every benchmark answer unchanged.

## Capabilities

107 certified procedures. A few of them:

- **Procedures:** 2-D convolution of an image with a kernel (deep-learning convention); Adam optimizer: one update; AdamW optimizer (decoupled weight decay): one update; Adjusted R^2; Adjusted mutual information of two clusterings; Agglomerative hierarchical clustering merge heights; Average precision and precision-recall AUC; Averaged precision, recall and F1 (multi-class); Backpropagation gradients of a one-hidden-layer network; Balanced accuracy (multi-class); Balanced class weights; Batch normalisation forward pass (training mode); Bernoulli naive Bayes prediction (binary features); Best decision-tree split (classification, Gini or entropy); Best regression-tree split (squared error); Bias-variance decomposition of predictions; Binary classification metrics from confusion counts; Binary confusion matrix from labels; Binary cross-entropy with logits and its gradient; Binary focal loss and its gradient; Binary log loss (cross-entropy); Brier score; Calinski-Harabasz index; Cohen's kappa (unweighted, linear or quadratic) …

`data_science_ml_ai999x.templates()` lists every question form `ask()` reads; the capability guide (PDF, every example executed) is included in this repository.

## Out of scope

- Open-ended explanation, essay writing or tutoring dialogue: it is a solver, not a chatbot.
- Questions outside its certified data science and machine learning capabilities, and any question whose stated numbers it cannot all use — it refuses with the reason instead.
- High-stakes engineering, medical or legal decisions without independent review.

## Limitations

- It answers what its certified capabilities cover (classification and regression metrics, loss functions and gradients, information theory, clustering, classical machine-learning algorithms, neural-network arithmetic and ranking metrics); anything else is refused with a reason.
- `ask()` reads documented question forms and common phrasings; unusual wording may be refused — use `solve()` for guaranteed results.
- Results are as correct as the certified relations and their stated conditions; a certified relation is a checked model, not a claim about every physical situation.

## Citation

```bibtex
@software{data_science_ml_ai999x_1_0_1,
  title  = {data-science-ml-ai999x: an offline, verified data science and machine learning solver},
  author = {Farabi, Ibnul Jaif}, organization = {AI999X},
  year   = {2026}, version = {1.0.1},
  url    = {https://huggingface.co/AI999X/data-science-ml-ai999x}
}
```

## Contact

Questions, bug reports and requests: open a discussion on https://huggingface.co/AI999X/data-science-ml-ai999x/discussions or visit https://ai999x.com.

## Support the project

data-science-ml-ai999x is free and open source. If it helps you, you can support further development on [Buy Me a Coffee](https://buymeacoffee.com/ibnuljaiffarabi).

Created and engineered by **Ibnul Jaif Farabi** · **[AI999X](https://huggingface.co/AI999X)** · [ai999x.com](https://ai999x.com)

## Source and releases

This repository holds the installable source of each release, exactly as published on [PyPI](https://pypi.org/project/data-science-ml-ai999x/) (`pip install .` builds it). Releases are tagged `v<version>`; the wheel is attached to each [GitHub release](https://github.com/AI999X/data-science-ml-ai999x/releases). The optional ONNX models (formula graphs and, where measured, a decision model) are on [Hugging Face](https://huggingface.co/AI999X/data-science-ml-ai999x/tree/main/onnx).
