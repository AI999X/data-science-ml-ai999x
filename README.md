# Data Science and Machine Learning (data-science-ml-ai999x)

[![tests](https://github.com/AI999X/data-science-ml-ai999x/actions/workflows/ci.yml/badge.svg)](https://github.com/AI999X/data-science-ml-ai999x/actions/workflows/ci.yml) [![PyPI](https://img.shields.io/pypi/v/data-science-ml-ai999x)](https://pypi.org/project/data-science-ml-ai999x/) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/AI999X/data-science-ml-ai999x)

Data Science and Machine Learning covers evaluation metrics, loss functions and gradients, classical machine-learning algorithms, dimensionality reduction, cross-validation, information theory and neural-network arithmetic: 107 calculations in 14 topics, each with its formula, its steps and its units, and a plain reason instead of a number when an input is missing or out of range. One compiled engine, four ways to use it: the desktop app, the command line, the browser and Python. Everything runs on your own computer, offline.

[![The Data Science and Machine Learning desktop app: Binary confusion matrix from labels, with its worked example and live scene](https://www.ai999x.com/?ax_shot=data-science-ml-ai999x/hero.webp)](https://www.ai999x.com/models/data-science-ml-ai999x/)

## Four ways to use it

| | |
|---|---|
| **Desktop app** (Windows, Linux) | [Windows](https://github.com/AI999X/ai999x-desktop/releases/download/v1.6.4/ai999x-1.6.4-windows-x86_64.zip) · [Linux](https://github.com/AI999X/ai999x-desktop/releases/download/v1.6.4/ai999x-1.6.4-linux-x86_64.tar.gz) |
| **Command line** `data-science-ml` (Windows, Linux) | [Windows](https://github.com/AI999X/ai999x-desktop/releases/download/v1.6.4/ai999x-cli-1.6.4-windows-x86_64.zip) · [Linux](https://github.com/AI999X/ai999x-desktop/releases/download/v1.6.4/ai999x-cli-1.6.4-linux-x86_64.tar.gz) |
| **Browser** (any device) | [Try it on the model page](https://www.ai999x.com/models/data-science-ml-ai999x/#try) |
| **Python** 3.10+ | `pip install data-science-ml-ai999x`: on Windows and Linux (x86-64) `pip` installs the same compiled engine; on other systems (macOS, ARM) it installs a pure-Python build that gives the same answers |

## Ask in words: 15 examples

Each question below was asked on this release; the comment under it is the line it printed.

```python
import data_science_ml_ai999x as data_science_ml

a = data_science_ml.ask(
    "Adjusted R squared: y_true = [10, 20, 30, 40, 50], y_pred = [12, "
    "18, 33, 37, 52], p = 1"
)
print(a.outcome, a.values)
# SOLVED {'r2': 0.97, 'adjusted_r2': 0.96}

a = data_science_ml.ask(
    "Mean absolute percentage error: y_true = [100, 200, 300], y_pred "
    "= [110, 190, 330]"
)
print(a.outcome, a.values)
# SOLVED {'mape': 0.0833333333333, 'mape_percent': 8.33333333333}

a = data_science_ml.ask(
    "Balanced accuracy: y_true = [0, 0, 0, 0, 0, 0, 1, 1], y_pred = "
    "[0, 0, 0, 0, 0, 0, 0, 1]"
)
print(a.outcome, a.values)
# SOLVED {'balanced_accuracy': 0.75, 'adjusted': 0.5}

a = data_science_ml.ask(
    "Davies-Bouldin index: X = [[1, 1], [1.5, 2], [3, 4], [5, 7], "
    "[3.5, 5], [4.5, 5]], labels = [0, 0, 1, 1, 1, 1]"
)
print(a.outcome, a.values)
# SOLVED {'davies_bouldin': 0.37473357715}

a = data_science_ml.ask(
    "Cosine learning rate schedule: lr_max = 0.1, eta_min = 0, T_max = "
    "100, epoch = 50"
)
print(a.outcome, a.values)
# SOLVED {'lr': 0.05}

a = data_science_ml.ask(
    "Mean squared log error: y_true = [3, 5, 2.5, 7], y_pred = [2.5, "
    "5, 4, 8]"
)
print(a.outcome, a.values)
# SOLVED {'msle': 0.0397301229846, 'rmsle': 0.199324165581}

a = data_science_ml.ask(
    "Leave-one-out cross-validation of linear regression: X = [[1], "
    "[2], [3], [4], [5], [6]], y = [1.1, 1.9, 3.2, 3.9, 5.1, 5.8]"
)
print(a.outcome, a.values)
# SOLVED {'press': 0.210910888993, 'loocv_mse': 0.0351518148321}

a = data_science_ml.ask(
    "Convolution output size: n = 32, kernel = 3, stride = 1, padding "
    "= 1, dilation = 1"
)
print(a.outcome, a.values)
# SOLVED {'output_size': 32, 'effective_kernel': 3}

a = data_science_ml.ask(
    "Warmup cosine schedule: lr_max = 0.001, start_factor = 0.1, "
    "warmup = 10, total = 100, eta_min = 0, step = 5"
)
print(a.outcome, a.values)
# SOLVED {'lr': 0.00055}

a = data_science_ml.ask(
    "Tweedie deviance: y_true = [2, 0, 1, 4], y_pred = [0.5, 0.5, 2, "
    "2], power = 1"
)
print(a.outcome, a.values)
# SOLVED {'deviance': 1.42601513196}

a = data_science_ml.ask(
    "Normalized mutual information: labels_true = [0, 0, 1, 1, 2, 2], "
    "labels_pred = [0, 0, 1, 1, 1, 1], method = arithmetic"
)
print(a.outcome, a.values)
# SOLVED {'nmi': 0.733680436651, 'mutual_information': 0.636514168295}

a = data_science_ml.ask(
    "Hinge loss: y_true = [1, -1, 1, -1], scores = [2, -0.5, 0.3, 0.8]"
)
print(a.outcome, a.values)
# SOLVED {'hinge': 0.75, 'squared_hinge': 0.995, 'n_violations': 3}

a = data_science_ml.ask(
    "Train test split sizes: n = 100, test_size = 0.2"
)
print(a.outcome, a.values)
# SOLVED {'n_test': 20, 'n_train': 80}

a = data_science_ml.ask(
    "Dense network forward pass: x = [1, 2], W1 = [[0.5, -0.25], [0.1, "
    "0.3]], b1 = [0.1, -0.2], activation = relu, W2 = [[1, -1]], b2 = "
    "[0.05], output = identity"
)
print(a.outcome, a.values)
# SOLVED {'hidden': [0.1, 0.5], 'output': [-0.35], 'max_output': -0.35}

a = data_science_ml.ask(
    "SGD momentum step: w = [1, -2], grad = [0.5, -1], buf = [0.1, "
    "0.2], lr = 0.1, momentum = 0.9, nesterov = no, weight_decay = 0"
)
print(a.outcome, a.values)
# SOLVED {'buf_new': [0.59, -0.82], 'w_new': [0.941, -1.918]}
```

## Screenshots

[![The Data Science and Machine Learning desktop app: Binary confusion matrix from labels, with its worked example and live scene](https://www.ai999x.com/?ax_shot=data-science-ml-ai999x/hero-dark.webp)](https://www.ai999x.com/models/data-science-ml-ai999x/)

[![Agglomerative hierarchical clustering merge heights: the live scene in the Data Science and Machine Learning desktop app](https://www.ai999x.com/?ax_shot=data-science-ml-ai999x/see-1.webp)](https://www.ai999x.com/models/data-science-ml-ai999x/)

*Agglomerative hierarchical clustering merge heights*

[![Binary cross-entropy with logits and its gradient: the live scene in the Data Science and Machine Learning desktop app](https://www.ai999x.com/?ax_shot=data-science-ml-ai999x/see-2.webp)](https://www.ai999x.com/models/data-science-ml-ai999x/)

*Binary cross-entropy with logits and its gradient*

[![Convolution output size: the live scene in the Data Science and Machine Learning desktop app](https://www.ai999x.com/?ax_shot=data-science-ml-ai999x/see-3.webp)](https://www.ai999x.com/models/data-science-ml-ai999x/)

*Convolution output size*

Every calculation, with its formula, a worked example and its figures: **[Data Science and Machine Learning on ai999x.com](https://www.ai999x.com/models/data-science-ml-ai999x/)**. Install guides for the app and the command line: [https://www.ai999x.com/desktop-cli/](https://www.ai999x.com/desktop-cli/).

Version 1.6.4 · License: Apache-2.0 · Author: Ibnul Jaif Farabi · Organization: AI999X · Contact: support@ai999x.com · [Support the project](https://buymeacoffee.com/ibnuljaiffarabi)

*The Future Is Self-Improving.*

## Source and releases

This repository holds the installable source of each release, exactly as published on [PyPI](https://pypi.org/project/data-science-ml-ai999x/) (`pip install .` builds it). Releases are tagged `v1.6.4`; the [GitHub release](https://github.com/AI999X/data-science-ml-ai999x/releases/tag/v1.6.4) carries the wheel, the desktop app and the command-line program for Windows and Linux, and their checksums.

## Build the program from source

`rust/` holds the Rust source of the `data-science-ml` command-line program and browser module: the AI999X engine (`rust/ax-engine`), this model (`rust/data-science-ml`) and its data (`rust/models/data_science_ml/pack.json`). With Rust 1.80 or later:

```bash
cd rust
cargo build --release -p data-science-ml-ai999x
./target/release/data-science-ml --help
cargo test -p data-science-ml-ai999x
```
