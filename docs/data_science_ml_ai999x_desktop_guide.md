# Data Science and Machine Learning: the desktop app, user guide

**Data Science and Machine Learning**, desktop app 1.6.5 for Windows and Linux. **Creator / engineer:** Ibnul Jaif Farabi · **Company:** AI999X · support@ai999x.com

Data Science and Machine Learning covers evaluation metrics, loss functions and gradients, classical machine-learning algorithms, dimensionality reduction, cross-validation, information theory and neural-network arithmetic. Its desktop app holds all 107 calculations in 14 topics: pick one, change its inputs and the answer, a live scene and charts follow at once. It runs on your own computer with no network connection and nothing else to install. The pictures in this guide were taken in this app.

## Contents

- 1. Get it and start it
- 2. The window
- 3. Compute a calculation
- 4. Ask in words
- 5. The live scene and the charts
- 6. Light and dark
- 7. Every calculation
- 8. Other ways to use it

## 1. Get it and start it

| system | download |
|---|---|
| Windows x86-64 | https://github.com/AI999X/ai999x-desktop/releases/download/v1.6.5/ai999x-data-science-ml-1.6.5-windows-x86_64.zip |
| Linux x86-64 | https://github.com/AI999X/ai999x-desktop/releases/download/v1.6.5/ai999x-data-science-ml-1.6.5-linux-x86_64.tar.gz |

Windows: Download ai999x-data-science-ml-1.6.5-windows-x86_64.zip, right-click it → Extract All, then open the folder and start ai999x-data-science-ml.exe. If Windows SmartScreen asks, choose More info → Run anyway.

Linux:

```shell
curl -LO https://github.com/AI999X/ai999x-desktop/releases/download/v1.6.5/ai999x-data-science-ml-1.6.5-linux-x86_64.tar.gz
curl -LO https://github.com/AI999X/ai999x-desktop/releases/download/v1.6.5/SHA256SUMS
sha256sum -c --ignore-missing SHA256SUMS
tar xzf ai999x-data-science-ml-1.6.5-linux-x86_64.tar.gz
./ai999x-data-science-ml-1.6.5-linux-x86_64/ai999x-data-science-ml
```

Checksums of every download: https://github.com/AI999X/ai999x-desktop/releases/download/v1.6.5/SHA256SUMS. Install steps for every app: https://www.ai999x.com/desktop-cli/

## 2. The window

![The Data Science and Machine Learning app, opened on Binary classification metrics from confusion counts with its worked example.](https://www.ai999x.com/?ax_shot=data-science-ml-ai999x/hero.webp)

*The Data Science and Machine Learning app, opened on Binary classification metrics from confusion counts with its worked example.*

On the left, a search box and the topics with their calculations. In the middle, the chosen calculation: its answer at the top, then the live scene (or its charts), what the answer means and the formula. On the right, its inputs, each with a slider where that makes sense, and the Compute and Reset buttons. At the bottom, a box for a question in words.

## 3. Compute a calculation

Open Classification metrics › Binary classification metrics from confusion counts (or type part of its name in the search box). It opens with a worked example filled in: “Binary metrics from counts: TP = 40, FP = 10, FN = 5, TN = 45” The answer: accuracy = 0.85, precision = 0.8, recall = 0.8888888889, specificity = 0.8181818182.

Change any input (type a number with its unit, or drag its slider) and the answer, the scene and the charts follow; Compute recomputes after typing, Reset brings back the worked example. An input outside the range the formula holds for is refused with the reason, never answered with a guessed number.

![Binary classification metrics from confusion counts: the live scene of the worked example.](https://www.ai999x.com/?ax_shot=data-science-ml-ai999x/scene-1.webp)

*Binary classification metrics from confusion counts: the live scene of the worked example.*

## 4. Ask in words

Type a question as you would say it, for example “Binary metrics from counts: TP = 40, FP = 10, FN = 5, TN = 45”. The app reads the quantities and their units, picks the calculation and answers; the command-line program reads it the same way and printed:

```text
Read as: Binary classification metrics from confusion counts
accuracy = 0.85
precision = 0.8
```

When a question names a quantity the app cannot place, it says what is missing instead of guessing.

## 5. The live scene and the charts

Every calculation draws a scene from its answer: it moves as an input changes. The Charts button shows charts that vary one input around your values; Enlarge opens one in a large view. Scenes from other topics:

![k-means clustering from given initial centroids](https://www.ai999x.com/?ax_shot=data-science-ml-ai999x/scene-2.webp)

*k-means clustering from given initial centroids*

![Scaled dot-product attention](https://www.ai999x.com/?ax_shot=data-science-ml-ai999x/scene-3.webp)

*Scaled dot-product attention*

![Adjusted mutual information of two clusterings](https://www.ai999x.com/?ax_shot=data-science-ml-ai999x/scene-4.webp)

*Adjusted mutual information of two clusterings*

![Principal component analysis (explained variance and first component)](https://www.ai999x.com/?ax_shot=data-science-ml-ai999x/scene-5.webp)

*Principal component analysis (explained variance and first component)*

![One gradient-descent step for linear regression](https://www.ai999x.com/?ax_shot=data-science-ml-ai999x/scene-6.webp)

*One gradient-descent step for linear regression*

![Lasso regression coefficients (stated alpha)](https://www.ai999x.com/?ax_shot=data-science-ml-ai999x/scene-7.webp)

*Lasso regression coefficients (stated alpha)*

![Binary cross-entropy with logits and its gradient](https://www.ai999x.com/?ax_shot=data-science-ml-ai999x/scene-8.webp)

*Binary cross-entropy with logits and its gradient*

![Binary classification metrics from confusion counts: a chart, enlarged.](https://www.ai999x.com/?ax_shot=data-science-ml-ai999x/chart-1.webp)

*Binary classification metrics from confusion counts: a chart, enlarged.*

![k-means clustering from given initial centroids: a chart, enlarged.](https://www.ai999x.com/?ax_shot=data-science-ml-ai999x/chart-2.webp)

*k-means clustering from given initial centroids: a chart, enlarged.*

## 6. Light and dark

The app follows a light or a dark theme (the switch is in its top bar).

![The Data Science and Machine Learning app in the dark theme.](https://www.ai999x.com/?ax_shot=data-science-ml-ai999x/hero-dark.webp)

*The Data Science and Machine Learning app in the dark theme.*

## 7. Every calculation

107 calculations in 14 topics; each has its own entry on the model's page (https://www.ai999x.com/models/data-science-ml-ai999x/) with its formula, worked example and charts.

| topic | calculations |
|---|---|
| Classical algorithms | Agglomerative hierarchical clustering merge heights, Bernoulli naive Bayes prediction (binary features), Best decision-tree split (classification, Gini or entropy), Best regression-tree split (squared error), Distances and cosine similarity between two vectors, Gaussian naive Bayes prediction, k-means clustering from given initial centroids, k-nearest-neighbour classification of a query point, k-nearest-neighbour regression of a query point, Multinomial naive Bayes prediction (word counts), One k-means iteration (assignment and update), Polynomial kernel, RBF (Gaussian) kernel matrix |
| Classification metrics | Average precision and precision-recall AUC, Averaged precision, recall and F1 (multi-class), Balanced accuracy (multi-class), Balanced class weights, Binary classification metrics from confusion counts, Binary confusion matrix from labels, Binary log loss (cross-entropy), Brier score, Cohen's kappa (unweighted, linear or quadratic), Confusion counts, precision and recall at a score threshold, Expected calibration error, F-beta score, Hamming loss and subset accuracy (multi-label), Matthews correlation coefficient, Multi-class confusion matrix, Multi-class log loss (categorical cross-entropy), Per-class precision, recall, F1 and support, ROC AUC from scores, ROC curve points, Top-k accuracy |
| Clustering metrics | Adjusted mutual information of two clusterings, Calinski-Harabasz index, Davies-Bouldin index, Homogeneity, completeness and V-measure, Normalised mutual information of two clusterings, Rand index, adjusted Rand index and Fowlkes-Mallows index, Silhouette score |
| Dimensionality reduction | Principal component analysis (explained variance and first component) |
| Gradients | Logistic regression loss and gradient (L2-regularised), Mean squared error loss and gradient of a linear model, One gradient-descent step for linear regression |
| Information theory | Gini impurity decrease of a split, Information gain and gain ratio of a split, Joint, marginal and conditional entropy and mutual information, KL divergence, Jensen-Shannon divergence and cross-entropy of discrete distributions, Mutual information between two label sequences, Node impurity from class counts (entropy, Gini, misclassification), Shannon entropy and perplexity of a distribution |
| Linear models | Lasso regression coefficients (stated alpha), Linear SVM decision values, classes and margin, Logistic regression fit (L2, stated C), Logistic regression prediction from given weights, Perceptron training epoch, Ridge regression coefficients (stated alpha) |
| Loss functions and activations | Binary cross-entropy with logits and its gradient, Binary focal loss and its gradient, Cross-entropy loss from logits and its gradient, Cross-entropy with label smoothing, Hinge and squared hinge loss, Huber, smooth L1 and pseudo-Huber loss, Log-softmax, Multi-class hinge loss (Crammer-Singer and Weston-Watkins), Quantile (pinball) loss, Sigmoid (logistic function) and its derivative, Softmax, Softmax with temperature, Triplet margin loss |
| Model selection | Bias-variance decomposition of predictions, k-fold cross-validated MSE of linear regression, k-fold cross-validation fold sizes, Leave-one-out cross-validation of linear regression (PRESS), Train/test split sizes |
| Neural networks | 2-D convolution of an image with a kernel (deep-learning convention), Backpropagation gradients of a one-hidden-layer network, Batch normalisation forward pass (training mode), Convolution output size, Forward pass of a one-hidden-layer dense network, Max pooling and average pooling, Multi-head attention (identity projections), Parameter count of a 2-D convolution layer, Parameter count of a dense (fully connected) network, Parameter count of a transformer encoder, Parameter count of an LSTM, GRU or simple RNN, Receptive field of stacked convolutions, Scaled dot-product attention |
| Optimizers | Adam optimizer: one update, AdamW optimizer (decoupled weight decay): one update, Cosine-annealing learning-rate schedule, Linear warm-up followed by cosine decay, SGD with momentum: one update, Step learning-rate schedule |
| Preprocessing | Label encoding and one-hot encoding of categories, Min-max scaling to [0, 1], Robust scaling (median and interquartile range), Standardisation (z-scores of features) |
| Ranking metrics | Mean average precision (MAP) of rankings, Mean reciprocal rank (MRR), NDCG@k (normalised discounted cumulative gain), Precision@k and recall@k of a ranking |
| Regression metrics | Adjusted R^2, Mean absolute percentage error, Mean squared logarithmic error, Mean Tweedie deviance (normal, Poisson, gamma, compound Poisson-gamma), Regression metrics (MAE, MSE, RMSE, R^2, explained variance) |

## 8. Other ways to use it

The same engine runs on the command line (`data-science-ml`, its own guide), in Python (`pip install data-science-ml-ai999x`, `import data_science_ml_ai999x`, its own guide) and in the browser on https://www.ai999x.com/models/data-science-ml-ai999x/.

Licence: Apache-2.0 · AI999X · https://www.ai999x.com/models/data-science-ml-ai999x/
