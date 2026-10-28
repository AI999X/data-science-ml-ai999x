"""data-science-ml-ai999x — offline data science and machine learning answers with their formula, steps and verification.

    import data_science_ml_ai999x as m
    m.ask("<capability phrase>: name = value, ...").show()
    m.formulas()                       # every certified capability

107 certified capabilities, school level to graduate level (the checkable part of machine learning):
- classification metrics: binary and multi-class confusion matrices, accuracy, precision, recall, specificity, F1
  and F-beta (macro, micro, weighted), balanced accuracy, MCC, Cohen's kappa (unweighted, linear, quadratic), log
  loss, Brier score, ROC AUC and ROC curves, average precision, top-k accuracy, Hamming loss, calibration error
- regression metrics: MAE, MSE, RMSE, R^2, adjusted R^2, explained variance, MAPE, MSLE, Huber, pinball and
  Tweedie deviances
- loss functions and gradients: softmax, log-softmax, sigmoid, cross-entropy from logits, binary cross-entropy with
  logits, hinge (binary and multi-class), focal loss, label smoothing, KL and Jensen-Shannon divergence, gradients
  of linear and logistic models, a gradient-descent step
- information theory and clustering: entropy, joint and conditional entropy, mutual information, information gain
  and gain ratio, Gini decrease, normalised and adjusted mutual information, Rand and adjusted Rand index,
  V-measure, silhouette, Davies-Bouldin and Calinski-Harabasz indices
- classical algorithms: k-means (one step or to convergence from given centroids), kNN classification and
  regression, Gaussian, multinomial and Bernoulli naive Bayes, decision-tree best splits, hierarchical clustering,
  PCA, logistic, ridge and lasso regression, linear SVM, the perceptron, kernels and distances, scaling and
  encoding, train/test and k-fold splits, cross-validated and leave-one-out MSE, bias-variance decomposition
- neural networks: dense forward pass and backpropagation, parameter counts (dense, convolution, LSTM/GRU,
  transformer), convolution output sizes and receptive fields, 2-D convolution and pooling, scaled dot-product and
  multi-head attention, batch normalisation, SGD with momentum, Adam and AdamW, step, cosine and warm-up schedules
- ranking metrics: precision@k, recall@k, MAP, NDCG@k, MRR

Questions use one documented form: the capability phrase, a colon, then `name = value` pairs (lists in brackets,
matrices as bracketed rows, conventions as words). Examples (both run as shown):

    m.ask("Binary metrics from counts: TP = 40, FP = 10, FN = 5, TN = 45").values
    # {'accuracy': 0.85, 'precision': 0.8, 'recall': 0.888888888889, 'specificity': 0.818181818182, 'f1': 0.842105263158, ...}
    m.ask("ROC AUC: y_true = [0, 1, 1, 0, 1], scores = [0.1, 0.9, 0.8, 0.3, 0.2]").values
    # {'auc': 0.833333333333, 'gini': 0.666666666667, 'n_pos': 3, 'n_neg': 2}

Built by the AI999X model machine from the same engine as every domain; a question that is not determined is
refused rather than guessed.

Creator / engineer: Ibnul Jaif Farabi. Company: AI999X.
"""
from __future__ import annotations

from pathlib import Path

try:                                       # the installed package carries its own copy of the engine
    from ._engine import Answer
    from ._engine.product import Product
except ImportError:                        # the factory's source tree, during development
    from ai999x_factory.runtime import Answer
    from ai999x_factory.runtime.product import Product

__version__ = "1.0.2"
_P = Product("data_science_ml", [Path(__file__).resolve().parent / "pack", Path(__file__).resolve().parents[3]])
pack, ask, solve, compile_problem = _P.pack, _P.ask, _P.solve, _P.compile_problem
formula, formulas, quantities, conditions, capabilities = _P.formula, _P.formulas, _P.quantities, _P.conditions, _P.capabilities
examples, family_examples, structured_example, to_si = _P.examples, _P.family_examples, _P.structured_example, _P.to_si
templates, procedure_examples = _P.templates, _P.procedure_examples
prove, derivations, about = _P.prove, _P.derivations, (lambda: {**_P.about(), "version": __version__})
knowledge, lookup, related = _P.knowledge, _P.lookup, _P.related
plot, plot_relation, search = _P.plot, _P.plot_relation, _P.search   # figures (SVG) and capability search ([search])
__all__ = ["ask", "solve", "prove", "derivations", "about", "formula", "formulas", "quantities", "conditions", "capabilities",
           "compile_problem", "pack", "Answer", "plot", "plot_relation", "search", "__version__"]
