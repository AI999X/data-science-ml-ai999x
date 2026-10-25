"""Data science and machine learning procedures: deterministic capabilities with typed inputs and several outputs
(domain code, loaded by the pack; enabled only once tools/certify_procedures.py has certified them).

Topics (the checkable part of machine learning): classification metrics (confusion matrices, precision, recall,
F-scores with every averaging, specificity, balanced accuracy, MCC, Cohen's kappa, log loss, Brier score, ROC and
precision-recall curves, top-k accuracy, Hamming loss, calibration error), regression metrics (MAE, MSE, RMSE, R^2,
adjusted R^2, explained variance, MAPE, Huber, pinball and Tweedie deviances), loss functions and their gradients
(softmax, log-softmax, sigmoid, cross-entropy, binary cross-entropy with logits, hinge, focal, label smoothing, KL
divergence), information theory (entropy, joint and conditional entropy, mutual information, information gain,
normalised and adjusted mutual information, clustering comparison), classical algorithms (k-means, silhouette and
other cluster indices, kNN, naive Bayes, decision-tree splits, logistic, ridge and lasso regression, linear SVM, PCA,
scaling and encoding, splitting and cross-validation, bias-variance), neural-network arithmetic (dense forward and
backward passes, layer shapes and parameter counts, attention, normalisation layers, optimizers and learning-rate
schedules) and ranking metrics (precision@k, recall@k, MAP, NDCG, MRR).

Each procedure states its inputs, outputs, conditions, method, LaTeX, the documented question form it parses, worked
examples, an INDEPENDENT reference computation (used only by certification; the checking libraries are imported inside
the reference functions) and a deliberately wrong variant that the checks must reject. The shipped code (run, verify,
parse) is plain Python (math, re, fractions, itertools).

How the references differ from the shipped code: scikit-learn's metrics, estimators and transformers (computed on
arrays, with their own algorithms: trapezoidal ROC area against the shipped pair count, LARS against the shipped
coordinate descent, a decision stump against the shipped split scan), scipy (special functions, entropy, distances,
hierarchical clustering, root finding on the stationarity equations), and PyTorch (CPU, float64) for the neural-network
arithmetic: autograd gradients against the shipped hand-written backpropagation, torch.optim optimizers and
lr_scheduler schedules, normalisation layers, attention, layer output shapes and parameter counts of real modules.

Conventions (fixed, so run and reference agree exactly):
- Class labels are whole numbers; binary labels are 0 and 1 (1 = positive), except where stated (hinge and SVM use -1
  and +1). Classes are the sorted union of the labels that occur; confusion matrices have true classes in rows and
  predicted classes in columns.
- An undefined precision, recall or F-score of one class (no predicted or no true member) counts as 0 in averaged
  scores (scikit-learn's zero_division = 0); binary metrics from counts refuse an undefined ratio instead.
- Averaging: macro = unweighted mean over classes, weighted = mean weighted by support, micro = from pooled counts.
- Logarithms are natural (nats) unless a unit is stated (bits = base 2, bans = base 10).
- ROC AUC counts tied scores as one half; curves have one point per distinct score, descending, starting at (0, 0)
  for ROC and at recall 0, precision 1 for precision-recall. Average precision is sum (R_k - R_{k-1}) P_k.
- Variances in scalers, naive Bayes and batch normalisation are population variances (divide by n); PCA uses the
  sample covariance (divide by n - 1); a component's sign makes its largest-magnitude entry positive.
- k-means starts from the given centroids, assigns each point to the nearest centroid (squared Euclidean distance)
  and stops when no assignment changes; an empty cluster or an exact distance tie is refused.
- Regularisation follows scikit-learn: logistic regression minimises 0.5 ||w||^2 + C sum log-loss (intercept not
  penalised); ridge minimises ||y - Xw - b||^2 + alpha ||w||^2; lasso minimises ||y - Xw - b||^2 / (2n) + alpha ||w||_1.
- Momentum, Adam, RMSprop and Adagrad follow PyTorch's update rules (stated in each capability).
- Results built from exp, log or square roots are rounded to 12 significant digits (identical on every platform) and
  values that are zero by construction are written as exactly 0.

Question form (one per procedure): the capability phrase, a colon, then `name = value` pairs separated by commas
(`Binary metrics from counts: TP = 40, FP = 10, FN = 5, TN = 45`). Lists are written in brackets, matrices as a
bracketed list of rows, choices as words. Any other number, an unknown name or a name given twice is refused, never
guessed.
"""
from __future__ import annotations

import math
import re
from fractions import Fraction
from itertools import combinations


# ================================================================================================ shared helpers
def _r(x):
    """12 significant digits, no negative zero (identical results on every platform and Python version)."""
    if isinstance(x, bool) or isinstance(x, str) or x is None:
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, Fraction):
        x = float(x)
    if isinstance(x, (list, tuple)):
        return [_r(v) for v in x]
    if x == 0 or not math.isfinite(x):
        return 0.0 if x == 0 else x
    return float(f"{x:.12g}") + 0.0


def _out(**kw):
    return {k: _r(v) for k, v in kw.items()}


def _rel(a, b):
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))) or len(a) != len(b):
            return math.inf
        return max([_rel(x, y) for x, y in zip(a, b)] + [0.0])
    if isinstance(a, (str, bool)) or isinstance(b, (str, bool)):
        return 0.0 if a == b else math.inf
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return math.inf
    if not (math.isfinite(a) and math.isfinite(b)):
        return 0.0 if a == b else math.inf
    d = abs(a - b)
    if d == 0:
        return 0.0
    if abs(a) < 1e-15 and abs(b) < 1e-15:
        return 0.0
    return d / max(abs(a), abs(b))


def _recheck(fn):
    """A verifier: every output the procedure gives is recomputed and compared (purely relative, floor 1e-15); a
    missing output fails."""
    def verify(inputs, outputs):
        try:
            want = fn(**inputs)
        except Exception:
            return math.inf
        worst = 0.0
        for k, v in want.items():
            if k not in outputs:
                return math.inf
            worst = max(worst, _rel(outputs[k], v))
        return worst
    return verify


def _safe(fn):
    """Guard a wrong variant against exceptions (a variant that raises cannot be compared)."""
    def g(**kw):
        try:
            return fn(**kw)
        except Exception:
            return {"_failed": True}
    return g


def _need(ok, msg):
    if not ok:
        raise ValueError(msg)


def _fin(name, *vals):
    for v in vals:
        _need(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v),
              f"{name} must be a finite number")


def _whole(name, v):
    """A whole number given as int or integral float -> int."""
    _fin(name, v)
    _need(float(v) == int(float(v)), f"{name} must be a whole number")
    return int(float(v))


def _labels(name, v, lo=1):
    _need(isinstance(v, (list, tuple)) and len(v) >= lo, f"{name} needs at least {lo} labels")
    return [_whole(f"every label in {name}", x) for x in v]


def _binary(name, v, lo=1):
    out = _labels(name, v, lo)
    _need(all(x in (0, 1) for x in out), f"every label in {name} must be 0 or 1")
    return out


def _vec(name, v, lo=1):
    _need(isinstance(v, (list, tuple)) and len(v) >= lo, f"{name} needs at least {lo} values")
    for x in v:
        _fin(f"every value in {name}", x)
    return [float(x) for x in v]


def _mat(name, M, rows=1, cols=1):
    _need(isinstance(M, (list, tuple)) and len(M) >= rows, f"{name} needs at least {rows} rows")
    _need(all(isinstance(r, (list, tuple)) for r in M) and len(M[0]) >= cols, f"{name} needs at least {cols} columns")
    _need(all(len(r) == len(M[0]) for r in M), f"{name} must be rectangular")
    for r in M:
        for x in r:
            _fin(f"every entry of {name}", x)
    return [[float(x) for x in r] for r in M]


def _same_len(a, b, na, nb):
    _need(len(a) == len(b), f"{na} and {nb} must have the same length")


def _F(x):
    """Exact rational value of a float (the decimal the user wrote, as the float holds it)."""
    return Fraction(x)


def _lu(rng, a, b):
    return math.exp(rng.uniform(math.log(a), math.log(b)))


def _sig(x, n=4):
    return float(f"{x:.{n}g}")


def _logsumexp(z):
    m = max(z)
    return m + math.log(math.fsum(math.exp(v - m) for v in z))


def _log_softmax(z):
    """(z_i - m) - log sum exp(z_j - m): exact for the largest logit (no cancellation against m)."""
    m = max(z)
    s = math.log(math.fsum(math.exp(v - m) for v in z))
    return [(v - m) - s for v in z]


def _softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = math.fsum(e)
    return [v / s for v in e]


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _log(x, unit):
    return math.log(x) / {"nats": 1.0, "bits": math.log(2.0), "bans": math.log(10.0)}[unit]


def _LN(unit):
    return {"nats": 1.0, "bits": math.log(2.0), "bans": math.log(10.0)}[unit]


def _matmul(A, B):
    return [[math.fsum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def _matvec(A, x):
    return [math.fsum(a * b for a, b in zip(row, x)) for row in A]


def _dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def _T(A):
    return [list(r) for r in zip(*A)]


def _gauss(A, b):
    """Solve A x = b by Gaussian elimination with partial pivoting; singular -> ValueError."""
    n = len(A)
    M = [list(row) + [b[i]] for i, row in enumerate(A)]
    scale = max((abs(x) for row in A for x in row), default=1.0) or 1.0
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        _need(abs(M[p][c]) > 1e-12 * scale, "the linear system is singular (the columns are linearly dependent)")
        M[c], M[p] = M[p], M[c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            if f != 0:
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    x = [0.0] * n
    for r in reversed(range(n)):
        x[r] = (M[r][n] - math.fsum(M[r][k] * x[k] for k in range(r + 1, n))) / M[r][r]
    return x


def _solve_exact(A, b):
    """Exact rational solve (Fractions) of A x = b; singular -> ValueError."""
    n = len(A)
    M = [[Fraction(v) for v in row] + [Fraction(b[i])] for i, row in enumerate(A)]
    for c in range(n):
        p = next((r for r in range(c, n) if M[r][c] != 0), None)
        _need(p is not None, "the linear system is singular (the columns are linearly dependent)")
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c] / M[c][c]
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def _np():
    import numpy as np
    return np


def _f(x):
    """A reference value as a plain Python number or list (numpy scalars and arrays converted)."""
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return [_f(v) for v in x]
    if isinstance(x, (bool, str)):
        return x
    if isinstance(x, int):
        return x
    return float(x)


# ================================================================================================ the question reader
_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_SEG = re.compile(r"^\s*([A-Za-z][A-Za-z_0-9]*)\s*=\s*(" + _NUM + r")\s*(%|percent|per cent)?\s*$")
_WORD = re.compile(r"^\s*([A-Za-z][A-Za-z_0-9]*)\s*=\s*([A-Za-z][A-Za-z_0-9\-]*)\s*$")
_LIST = re.compile(r"^\s*([A-Za-z][A-Za-z_0-9]*)\s*=\s*(\[.*\])\s*$")


def _split(rest):
    """Split on commas outside brackets; None when the brackets do not balance."""
    parts, depth, cur = [], 0, ""
    for ch in rest:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth < 0:
                return None
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if depth != 0:
        return None
    parts.append(cur)
    return parts


def _reader(phrases, inputs):
    """The one documented question form, '<phrase>: name = value, ...', for a procedure with these inputs.

    The reading guard (tools/ask_sweep.py, 2026-10-23):
      1. every stated number is bound: the question is split into 'name = value' pairs and any other text or number
         refuses the whole question (a stray number can never be ignored);
      2. the capability phrase must open the question and be followed by the colon, so wording that names a different
         capability never reaches this reader;
      3. a convention is an input (averaging, logarithm unit, weights, activation, ...): a required choice, never
         assumed; a list is written in brackets, a choice as a word, and a value of the wrong form is refused;
      4. nothing is converted: values reach the procedure exactly as written (a percentage only where stated).
    """
    rx = re.compile(r"\s*(?:" + "|".join(re.escape(p) for p in phrases) + r")\s*:\s*", re.I)
    by = {i["symbol"]: i for i in inputs}
    names = {i["symbol"]: i["symbol"] for i in inputs}
    for i in inputs:                         # a lower-case spelling only where it names no other input
        names.setdefault(i["symbol"].lower(), i["symbol"])

    def parse(question):
        if not isinstance(question, str):
            return None
        m = rx.match(question)
        if not m:
            return None
        rest = re.sub(r"[.?!]\s*$", "", question[m.end():].strip()).strip()
        if not rest:
            return {}
        segs = _split(rest)
        if segs is None:
            return None
        out = {}
        for seg in segs:
            if not seg.strip():
                return None
            ml = _LIST.match(seg)
            if ml:
                key = names.get(ml.group(1), names.get(ml.group(1).lower()))
                if key is None or key in out:
                    return None
                t = by[key].get("type", "number")
                text = ml.group(2).strip()
                if t == "vector":
                    if text.startswith("[[") or text.count("[") != 1:
                        return None
                    out[key] = text
                elif t == "matrix":
                    if not text.startswith("[[") or not text.endswith("]]"):
                        return None
                    out[key] = text
                elif t == "text" and by[key].get("list"):
                    if text.count("[") != 1:
                        return None
                    out[key] = text[1:-1].strip()
                else:
                    return None
                continue
            mw = _WORD.match(seg)
            if mw:
                key = names.get(mw.group(1), names.get(mw.group(1).lower()))
                if key is None or key in out or by[key].get("type") not in ("choice", "boolean"):
                    return None
                out[key] = mw.group(2).strip().lower()
                continue
            ms = _SEG.match(seg)
            if not ms:
                return None
            key = names.get(ms.group(1), names.get(ms.group(1).lower()))
            if key is None or key in out:
                return None
            spec = by[key]
            t = spec.get("type", "number")
            num, pct = ms.group(2), ms.group(3)
            if t == "integer":
                if pct or not re.fullmatch(r"[-+]?\d+", num):
                    return None
                out[key] = num
                continue
            if t != "number":
                return None
            if pct:
                if not spec.get("pct"):
                    return None
                out[key] = float(num) / 100.0
                continue
            out[key] = float(num)
        return out
    return parse


def _fmt(x):
    x = float(x)
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return repr(x)


def _written(spec, v):
    """An input value as a user writes it in the documented form."""
    t = spec.get("type", "number")
    if t == "vector":
        if isinstance(v, str):
            return "[" + v.strip().strip("[]") + "]"
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    if t == "matrix":
        if isinstance(v, str):
            return v
        return "[" + ", ".join("[" + ", ".join(_fmt(x) for x in row) + "]" for row in v) + "]"
    if t == "text":
        return "[" + str(v) + "]"
    if t == "choice":
        return str(v)
    if t == "boolean":
        return "yes" if v in (True, "yes", "true") else "no"
    if t == "integer":
        return str(int(v))
    return _fmt(v)


# ================================================================================================ input/output specs
def _numin(sym, meaning, pct=False, **extra):
    d = {"symbol": sym, "kind": "number", "meaning": meaning, "required": True, **extra}
    if pct:
        d["pct"] = True
    return d


def _intin(sym, meaning, lo=None, hi=None):
    d = {"symbol": sym, "type": "integer", "meaning": meaning, "required": True}
    if lo is not None:
        d["min"] = lo
    if hi is not None:
        d["max"] = hi
    return d


def _vecin(sym, meaning, lo=1, hi=500):
    return {"symbol": sym, "type": "vector", "kind": "number", "meaning": meaning, "required": True,
            "min_length": lo, "max_length": hi}


def _matin(sym, meaning):
    return {"symbol": sym, "type": "matrix", "kind": "number", "meaning": meaning, "required": True}


def _choice(sym, choices, meaning):
    return {"symbol": sym, "type": "choice", "choices": list(choices), "meaning": meaning, "required": True}


def _boolin(sym, meaning):
    return {"symbol": sym, "type": "boolean", "meaning": meaning, "required": True}


def _textlist(sym, meaning):
    return {"symbol": sym, "type": "text", "list": True, "meaning": meaning, "required": True, "min_length": 1,
            "max_length": 2000}


def _o(sym, meaning, kind="number"):
    return {"symbol": sym, "kind": kind, "meaning": meaning}


PROCEDURES: dict = {}
LEVELS: dict = {}
WEAK_REFERENCES: dict = {}


def _add(key, *, name, topic, purpose, inputs, outputs, primary, method, latex, run, reference, wrong, rand,
         phrases, examples, level, conditions=(), extra_q=(), weak=None):
    """Register a procedure: the documented question form is built from its phrases and inputs; templates and
    questions are written from the worked examples exactly as a user writes them."""
    parse = _reader(phrases, inputs)
    by = {i["symbol"]: i for i in inputs}

    def ask_text(ex_inputs):
        return f"{phrases[0]}: " + ", ".join(f"{k} = {_written(by[k], v)}" for k, v in ex_inputs.items())
    templates = [ask_text(e["inputs"]) for e in examples[:2]]
    questions = [{"q": ask_text(e["inputs"]), "outputs": e["outputs"]} for e in examples]
    req = [i["symbol"] for i in inputs if i.get("required")]
    if req and examples:
        partial = {k: v for k, v in examples[0]["inputs"].items() if k != req[0]}
        questions.append({"q": ask_text(partial), "outcome": "NEEDS_INPUT"})
        questions.append({"q": ask_text(examples[0]["inputs"]) + ", zz = 1", "outcome": "UNRECOGNIZED_FORMAT"})
    questions += [{"q": ask_text(q["inputs"]), "outcome": q["outcome"]} if "inputs" in q else q for q in extra_q]
    PROCEDURES[key] = {
        "name": name, "topic": topic, "purpose": purpose, "inputs": inputs, "outputs": outputs, "primary": primary,
        "conditions": list(conditions), "method": method, "latex": latex, "run": run,
        "verify": _recheck(run), "parse": parse, "reference": reference, "wrong": _safe(wrong),
        "random_inputs": rand, "templates": templates, "examples": list(examples), "questions": questions}
    LEVELS[key] = level
    if weak:
        WEAK_REFERENCES[key] = weak


def _ex(inputs, outputs=None):
    return {"inputs": inputs, "outputs": outputs or {}}


def _bad(inputs, outcome="INVALID_INPUT"):
    return {"inputs": inputs, "outcome": outcome}


# ================================================================================================ random helpers
def _rand_binary_pair(rng, n=None, acc=None):
    n = n or rng.randint(12, 60)
    acc = acc if acc is not None else rng.uniform(0.55, 0.9)
    while True:
        yt = [rng.randint(0, 1) for _ in range(n)]
        yp = [y if rng.random() < acc else 1 - y for y in yt]
        tp = sum(1 for a, b in zip(yt, yp) if a == 1 and b == 1)
        fp = sum(1 for a, b in zip(yt, yp) if a == 0 and b == 1)
        fn = sum(1 for a, b in zip(yt, yp) if a == 1 and b == 0)
        tn = n - tp - fp - fn
        if min(tp, fp, fn, tn) >= 1:
            return yt, yp


def _rand_multiclass_pair(rng, k=None, n=None, acc=None):
    k = k or rng.randint(3, 5)
    n = n or rng.randint(15, 60)
    acc = acc if acc is not None else rng.uniform(0.45, 0.85)
    while True:
        yt = [rng.randrange(k) for _ in range(n)]
        yp = [y if rng.random() < acc else rng.randrange(k) for y in yt]
        if len(set(yt)) == k and len(set(yp)) >= k - 1:
            return yt, yp


def _rand_scores(rng, yt, sep=1.0, ties=False):
    s = [rng.gauss(sep * y, 1.0) for y in yt]
    if ties:
        return [round(v, 1) for v in s]
    return [round(v, 4) for v in s]


def _rand_probs(rng, yt, sep=1.2):
    out = []
    for y in yt:
        z = rng.gauss(sep * (2 * y - 1), 1.0)
        p = 1.0 / (1.0 + math.exp(-z))
        out.append(min(0.999, max(0.001, round(p, 3))))
    return out


def _rand_dist(rng, k, digits=3, zero_ok=False):
    """A probability vector with `digits` decimals summing exactly to 1 (as decimals)."""
    while True:
        units = 10 ** digits
        w = [rng.uniform(0.05, 1.0) for _ in range(k)]
        s = sum(w)
        cut = [int(round(x / s * units)) for x in w]
        cut[-1] = units - sum(cut[:-1])
        if all(c >= (0 if zero_ok else 1) for c in cut):
            return [c / units for c in cut]


# ================================================================================================ classification
def _counts(yt, yp):
    tp = sum(1 for a, b in zip(yt, yp) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(yt, yp) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(yt, yp) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(yt, yp) if a == 0 and b == 0)
    return tp, fp, fn, tn


def _expand(TP, FP, FN, TN):
    yt = [1] * TP + [0] * FP + [1] * FN + [0] * TN
    yp = [1] * TP + [1] * FP + [0] * FN + [0] * TN
    return yt, yp


def binary_confusion(y_true, y_pred):
    yt, yp = _binary("y_true", y_true), _binary("y_pred", y_pred)
    _same_len(yt, yp, "y_true", "y_pred")
    tp, fp, fn, tn = _counts(yt, yp)
    n = len(yt)
    return _out(TP=tp, FP=fp, FN=fn, TN=tn, accuracy=Fraction(tp + tn, n), error_rate=Fraction(fp + fn, n))


def _ref_binary_confusion(y_true, y_pred):
    from sklearn.metrics import accuracy_score, confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    return {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn), "accuracy": float(acc), "error_rate": 1 - float(acc)}


def _wrong_binary_confusion(y_true, y_pred):
    tp, fp, fn, tn = _counts(_binary("y_true", y_true), _binary("y_pred", y_pred))
    n = len(y_true)
    return _out(TP=tp, FP=fn, FN=fp, TN=tn, accuracy=Fraction(tp, n), error_rate=Fraction(fp + fn, n))


_add("ml_binary_confusion_matrix",
     name="Binary confusion matrix from labels", topic="classification metrics",
     purpose="Count true positives, false positives, false negatives and true negatives from true and predicted 0/1 labels, with the accuracy and error rate.",
     inputs=[_vecin("y_true", "true labels (0 or 1, 1 = positive)", 1), _vecin("y_pred", "predicted labels (0 or 1)", 1)],
     outputs=[_o("TP", "true positives"), _o("FP", "false positives"), _o("FN", "false negatives"),
              _o("TN", "true negatives"), _o("accuracy", "(TP + TN) / n"), _o("error_rate", "(FP + FN) / n")],
     primary="accuracy", method="count each (true, predicted) pair; accuracy and error rate exactly as fractions",
     latex=r"\mathrm{accuracy} = \frac{TP + TN}{TP + FP + FN + TN}",
     run=binary_confusion, reference=_ref_binary_confusion, wrong=_wrong_binary_confusion,
     rand=lambda rng: dict(zip(("y_true", "y_pred"), _rand_binary_pair(rng))),
     phrases=["Binary confusion matrix", "Confusion matrix of binary labels"],
     conditions=["labels are 0 or 1", "y_true and y_pred have the same length"],
     examples=[_ex({"y_true": [1, 0, 1, 1, 0, 0, 1, 0], "y_pred": [1, 0, 0, 1, 0, 1, 1, 0]},
                   {"TP": 3, "FP": 1, "FN": 1, "TN": 3, "accuracy": 0.75}),
               _ex({"y_true": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0], "y_pred": [1, 1, 0, 0, 0, 0, 0, 0, 1, 1]},
                   {"TP": 2, "FP": 2, "FN": 1, "TN": 5, "accuracy": 0.7})],
     extra_q=[_bad({"y_true": [1, 0, 2], "y_pred": [1, 0, 1]}), _bad({"y_true": [1, 0, 1], "y_pred": [1, 0]})],
     level="beginner")


def _check_counts(TP, FP, FN, TN):
    for nm, v in (("TP", TP), ("FP", FP), ("FN", FN), ("TN", TN)):
        _need(isinstance(v, int) and v >= 0, f"{nm} must be a whole number >= 0")


def binary_from_counts(TP, FP, FN, TN):
    _check_counts(TP, FP, FN, TN)
    _need(TP + FP > 0, "precision is undefined: there are no predicted positives (TP + FP = 0)")
    _need(TP + FN > 0, "recall is undefined: there are no actual positives (TP + FN = 0)")
    _need(TN + FP > 0, "specificity is undefined: there are no actual negatives (TN + FP = 0)")
    _need(TN + FN > 0, "the negative predictive value is undefined: there are no predicted negatives (TN + FN = 0)")
    n = TP + FP + FN + TN
    P, R = Fraction(TP, TP + FP), Fraction(TP, TP + FN)
    S = Fraction(TN, TN + FP)
    f1 = Fraction(2 * TP, 2 * TP + FP + FN)
    return _out(accuracy=Fraction(TP + TN, n), precision=P, recall=R, specificity=S, f1=f1,
                npv=Fraction(TN, TN + FN), balanced_accuracy=(R + S) / 2, fpr=1 - S, fnr=1 - R,
                jaccard=Fraction(TP, TP + FP + FN))


def _ref_binary_from_counts(TP, FP, FN, TN):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, jaccard_score, precision_score,
                                 recall_score)
    yt, yp = _expand(TP, FP, FN, TN)
    spec = recall_score(yt, yp, pos_label=0)
    rec = recall_score(yt, yp)
    return {"accuracy": float(accuracy_score(yt, yp)), "precision": float(precision_score(yt, yp)),
            "recall": float(rec), "specificity": float(spec), "f1": float(f1_score(yt, yp)),
            "npv": float(precision_score(yt, yp, pos_label=0)),
            "balanced_accuracy": float(balanced_accuracy_score(yt, yp)), "fpr": 1 - float(spec),
            "fnr": 1 - float(rec), "jaccard": float(jaccard_score(yt, yp))}


def _wrong_binary_from_counts(TP, FP, FN, TN):
    o = binary_from_counts(TP, FP, FN, TN)
    o["precision"], o["recall"] = o["recall"], o["precision"]     # precision and recall swapped
    o["f1"] = _r(Fraction(TP, TP + FP) * Fraction(TP, TP + FN))  # product instead of harmonic mean
    return o


def _rand_counts(rng, lo=1):
    return {"TP": rng.randint(lo, 120), "FP": rng.randint(lo, 60), "FN": rng.randint(lo, 60), "TN": rng.randint(lo, 200)}


_COUNTS_IN = [_intin("TP", "true positives", 0), _intin("FP", "false positives", 0),
              _intin("FN", "false negatives", 0), _intin("TN", "true negatives", 0)]

_add("ml_binary_metrics_from_counts",
     name="Binary classification metrics from confusion counts", topic="classification metrics",
     purpose="Accuracy, precision, recall (sensitivity), specificity, F1, negative predictive value, balanced accuracy, false positive and false negative rates and the Jaccard index from TP, FP, FN, TN.",
     inputs=_COUNTS_IN,
     outputs=[_o("accuracy", "(TP + TN) / n"), _o("precision", "TP / (TP + FP)"), _o("recall", "TP / (TP + FN), the sensitivity"),
              _o("specificity", "TN / (TN + FP)"), _o("f1", "2 TP / (2 TP + FP + FN)"), _o("npv", "TN / (TN + FN)"),
              _o("balanced_accuracy", "(recall + specificity) / 2"), _o("fpr", "FP / (FP + TN)"), _o("fnr", "FN / (FN + TP)"),
              _o("jaccard", "TP / (TP + FP + FN)")],
     primary="f1", method="exact rational arithmetic on the four counts",
     latex=r"P = \frac{TP}{TP+FP},\; R = \frac{TP}{TP+FN},\; F_1 = \frac{2PR}{P+R} = \frac{2TP}{2TP+FP+FN}",
     run=binary_from_counts, reference=_ref_binary_from_counts, wrong=_wrong_binary_from_counts,
     rand=_rand_counts, phrases=["Binary metrics from counts", "Classification metrics from counts"],
     conditions=["counts are whole numbers >= 0", "every ratio is defined (TP + FP, TP + FN, TN + FP, TN + FN > 0)"],
     examples=[_ex({"TP": 40, "FP": 10, "FN": 5, "TN": 45}, {"precision": 0.8, "recall": 0.888888888889, "accuracy": 0.85}),
               _ex({"TP": 90, "FP": 30, "FN": 10, "TN": 870}, {"precision": 0.75, "recall": 0.9, "specificity": 0.966666666667})],
     extra_q=[_bad({"TP": 0, "FP": 0, "FN": 5, "TN": 45})],
     level="beginner")


def fbeta(TP, FP, FN, beta):
    _check_counts(TP, FP, FN, 0)
    _fin("beta", beta)
    _need(beta > 0, "beta must be positive")
    _need(TP + FP > 0 and TP + FN > 0, "precision or recall is undefined (no predicted or no actual positives)")
    _need(TP > 0, "precision and recall are both 0: the F-beta score is undefined (0/0)")
    b2 = _F(beta) ** 2
    P, R = Fraction(TP, TP + FP), Fraction(TP, TP + FN)
    return _out(precision=P, recall=R, fbeta=(1 + b2) * P * R / (b2 * P + R))


def _ref_fbeta(TP, FP, FN, beta):
    from sklearn.metrics import fbeta_score, precision_score, recall_score
    yt, yp = _expand(TP, FP, FN, 1)
    return {"precision": float(precision_score(yt, yp)), "recall": float(recall_score(yt, yp)),
            "fbeta": float(fbeta_score(yt, yp, beta=beta))}


def _wrong_fbeta(TP, FP, FN, beta):
    P, R = TP / (TP + FP), TP / (TP + FN)
    return _out(precision=P, recall=R, fbeta=(1 + beta) * P * R / (beta * P + R))     # beta instead of beta^2


_add("ml_fbeta_score",
     name="F-beta score", topic="classification metrics",
     purpose="The F-beta score, the weighted harmonic mean of precision and recall in which recall counts beta times as much as precision.",
     inputs=_COUNTS_IN[:3] + [_numin("beta", "weight of recall relative to precision (beta > 0; 1 gives F1, 2 favours recall)")],
     outputs=[_o("precision", "TP / (TP + FP)"), _o("recall", "TP / (TP + FN)"), _o("fbeta", "F-beta score")],
     primary="fbeta", method="exact rational arithmetic from the counts",
     latex=r"F_\beta = \frac{(1+\beta^2)\,P R}{\beta^2 P + R}",
     run=fbeta, reference=_ref_fbeta, wrong=_wrong_fbeta,
     rand=lambda rng: {"TP": rng.randint(1, 100), "FP": rng.randint(1, 60), "FN": rng.randint(1, 60),
                       "beta": rng.choice([0.5, 2.0, 3.0, 0.25, round(rng.uniform(0.3, 4), 2)])},
     phrases=["F-beta score", "F beta score", "Fbeta score"],
     conditions=["beta > 0", "TP > 0"],
     examples=[_ex({"TP": 40, "FP": 10, "FN": 5, "beta": 2}, {"fbeta": 0.869565217391}),
               _ex({"TP": 40, "FP": 10, "FN": 5, "beta": 0.5}, {"fbeta": 0.816326530612})],
     extra_q=[_bad({"TP": 40, "FP": 10, "FN": 5, "beta": -1})],
     level="intermediate")


def mcc(TP, FP, FN, TN):
    _check_counts(TP, FP, FN, TN)
    a, b, c, d = TP + FP, TP + FN, TN + FP, TN + FN
    _need(a * b * c * d > 0, "the MCC is undefined when a row or column of the confusion matrix is empty")
    num = TP * TN - FP * FN
    den2 = a * b * c * d
    return _out(mcc=num / math.sqrt(den2) if num else 0.0)


def _ref_mcc(TP, FP, FN, TN):
    from sklearn.metrics import matthews_corrcoef
    yt, yp = _expand(TP, FP, FN, TN)
    return {"mcc": float(matthews_corrcoef(yt, yp))}


_add("ml_matthews_correlation",
     name="Matthews correlation coefficient", topic="classification metrics",
     purpose="The Matthews correlation coefficient (phi coefficient) of a binary confusion matrix, a balanced measure between -1 and 1.",
     inputs=_COUNTS_IN,
     outputs=[_o("mcc", "Matthews correlation coefficient")],
     primary="mcc", method="(TP TN - FP FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)) with exact integers under the root",
     latex=r"\mathrm{MCC} = \frac{TP\,TN - FP\,FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}",
     run=mcc, reference=_ref_mcc,
     wrong=lambda TP, FP, FN, TN: _out(mcc=(TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))),
     rand=_rand_counts, phrases=["Matthews correlation coefficient", "MCC"],
     conditions=["no empty row or column of the confusion matrix"],
     examples=[_ex({"TP": 40, "FP": 10, "FN": 5, "TN": 45}, {"mcc": 0.703526470681}),
               _ex({"TP": 6, "FP": 1, "FN": 2, "TN": 3}, {"mcc": 0.478091443734})],
     extra_q=[_bad({"TP": 0, "FP": 0, "FN": 5, "TN": 45})],
     level="intermediate")


def _classes(*lists):
    return sorted(set().union(*[set(v) for v in lists]))


def multiclass_confusion(y_true, y_pred):
    yt, yp = _labels("y_true", y_true), _labels("y_pred", y_pred)
    _same_len(yt, yp, "y_true", "y_pred")
    cls = _classes(yt, yp)
    idx = {c: i for i, c in enumerate(cls)}
    M = [[0] * len(cls) for _ in cls]
    for a, b in zip(yt, yp):
        M[idx[a]][idx[b]] += 1
    correct = sum(M[i][i] for i in range(len(cls)))
    return _out(classes=cls, matrix=M, n_correct=correct, accuracy=Fraction(correct, len(yt)))


def _ref_multiclass_confusion(y_true, y_pred):
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.utils.multiclass import unique_labels
    cls = [int(c) for c in unique_labels(y_true, y_pred)]
    M = confusion_matrix(y_true, y_pred, labels=cls)
    return {"classes": cls, "matrix": [[int(v) for v in r] for r in M], "n_correct": int(M.trace()),
            "accuracy": float(accuracy_score(y_true, y_pred))}


def _wrong_multiclass_confusion(y_true, y_pred):
    o = multiclass_confusion(y_true, y_pred)
    o["matrix"] = _T(o["matrix"])                         # predicted classes in rows
    return o


_add("ml_multiclass_confusion_matrix",
     name="Multi-class confusion matrix", topic="classification metrics",
     purpose="The confusion matrix of a multi-class classifier (true classes in rows, predicted in columns) with the number correct and the accuracy.",
     inputs=[_vecin("y_true", "true class labels (whole numbers)"), _vecin("y_pred", "predicted class labels (whole numbers)")],
     outputs=[_o("classes", "classes in sorted order (the union of both label lists)"),
              _o("matrix", "confusion matrix: row = true class, column = predicted class"),
              _o("n_correct", "number of correct predictions (the trace)"), _o("accuracy", "fraction correct")],
     primary="accuracy", method="count each (true, predicted) pair over the sorted union of classes",
     latex=r"C_{ij} = \#\{n : y_n = c_i,\ \hat y_n = c_j\}",
     run=multiclass_confusion, reference=_ref_multiclass_confusion, wrong=_wrong_multiclass_confusion,
     rand=lambda rng: dict(zip(("y_true", "y_pred"), _rand_multiclass_pair(rng))),
     phrases=["Multiclass confusion matrix", "Multi-class confusion matrix"],
     conditions=["labels are whole numbers", "equal lengths"],
     examples=[_ex({"y_true": [0, 1, 2, 2, 1, 0, 2, 1], "y_pred": [0, 2, 2, 2, 1, 0, 1, 1]},
                   {"classes": [0, 1, 2], "matrix": [[2, 0, 0], [0, 2, 1], [0, 1, 2]], "accuracy": 0.75}),
               _ex({"y_true": [3, 1, 2, 3, 3, 1], "y_pred": [3, 3, 2, 3, 1, 1]},
                   {"classes": [1, 2, 3], "matrix": [[1, 0, 1], [0, 1, 0], [1, 0, 2]]})],
     extra_q=[_bad({"y_true": [0, 1.5, 2], "y_pred": [0, 1, 2]})],
     level="beginner")


def _per_class(yt, yp):
    cls = _classes(yt, yp)
    P, R, F, S = [], [], [], []
    for c in cls:
        tp = sum(1 for a, b in zip(yt, yp) if a == c and b == c)
        pp = sum(1 for b in yp if b == c)
        ap = sum(1 for a in yt if a == c)
        p = Fraction(tp, pp) if pp else Fraction(0)
        r = Fraction(tp, ap) if ap else Fraction(0)
        f = 2 * p * r / (p + r) if p + r else Fraction(0)
        P.append(p), R.append(r), F.append(f), S.append(ap)
    return cls, P, R, F, S


def prf_averaged(y_true, y_pred, average):
    yt, yp = _labels("y_true", y_true), _labels("y_pred", y_pred)
    _same_len(yt, yp, "y_true", "y_pred")
    cls, P, R, F, S = _per_class(yt, yp)
    if average == "macro":
        k = len(cls)
        return _out(precision=sum(P) / k, recall=sum(R) / k, f1=sum(F) / k)
    if average == "weighted":
        n = sum(S)
        return _out(precision=sum(p * s for p, s in zip(P, S)) / n, recall=sum(r * s for r, s in zip(R, S)) / n,
                    f1=sum(f * s for f, s in zip(F, S)) / n)
    correct = sum(1 for a, b in zip(yt, yp) if a == b)
    acc = Fraction(correct, len(yt))
    return _out(precision=acc, recall=acc, f1=acc)          # micro: pooled counts, every sample labelled once


def _ref_prf(y_true, y_pred, average):
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f)}


def _wrong_prf(y_true, y_pred, average):
    o = prf_averaged(y_true, y_pred, average)
    p, r = o["precision"], o["recall"]
    o["f1"] = _r(2 * p * r / (p + r)) if p + r else 0.0     # F1 of the averaged P and R (not the average F1)
    if average == "micro":
        o["f1"] = _r(o["f1"] * len(set(y_true)) / (len(set(y_true)) + 1))
    return o


_add("ml_precision_recall_f1_averaged",
     name="Averaged precision, recall and F1 (multi-class)", topic="classification metrics",
     purpose="Multi-class precision, recall and F1 averaged over classes: macro (unweighted), weighted (by support) or micro (pooled counts).",
     inputs=[_vecin("y_true", "true class labels (whole numbers)"), _vecin("y_pred", "predicted class labels (whole numbers)"),
             _choice("average", ["macro", "weighted", "micro"], "averaging: macro, weighted or micro")],
     outputs=[_o("precision", "averaged precision"), _o("recall", "averaged recall"), _o("f1", "averaged F1 (macro and weighted: the average of the per-class F1)")],
     primary="f1", method="per-class counts over the sorted union of classes, exact fractions; an undefined per-class ratio counts as 0",
     latex=r"F_1^{\mathrm{macro}} = \frac{1}{K}\sum_k \frac{2P_kR_k}{P_k+R_k}",
     run=prf_averaged, reference=_ref_prf, wrong=_wrong_prf,
     rand=lambda rng: dict(zip(("y_true", "y_pred"), _rand_multiclass_pair(rng)), average=rng.choice(["macro", "weighted", "micro"])),
     phrases=["Averaged precision recall F1", "Multiclass precision recall F1"],
     conditions=["labels are whole numbers", "equal lengths"],
     examples=[_ex({"y_true": [0, 1, 2, 2, 1, 0, 2, 1], "y_pred": [0, 2, 2, 2, 1, 0, 1, 1], "average": "macro"},
                   {"f1": 0.777777777778}),
               _ex({"y_true": [0, 1, 2, 2, 1, 0, 2, 1], "y_pred": [0, 2, 2, 2, 1, 0, 1, 1], "average": "weighted"}),
               _ex({"y_true": [0, 1, 2, 2, 1, 0, 2, 1], "y_pred": [0, 2, 2, 2, 1, 0, 1, 1], "average": "micro"},
                   {"precision": 0.75, "recall": 0.75, "f1": 0.75})],
     extra_q=[_bad({"y_true": [0, 1, 2], "y_pred": [0, 1, 2], "average": "samples"}, "INVALID_INPUT")],
     level="intermediate")


def per_class_report(y_true, y_pred):
    yt, yp = _labels("y_true", y_true), _labels("y_pred", y_pred)
    _same_len(yt, yp, "y_true", "y_pred")
    cls, P, R, F, S = _per_class(yt, yp)
    return _out(classes=cls, precision=P, recall=R, f1=F, support=S, macro_f1=sum(F) / len(F))


def _ref_per_class(y_true, y_pred):
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.utils.multiclass import unique_labels
    cls = [int(c) for c in unique_labels(y_true, y_pred)]
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=cls, average=None, zero_division=0)
    return {"classes": cls, "precision": _f(p), "recall": _f(r), "f1": _f(f), "support": [int(v) for v in s],
            "macro_f1": float(f.mean())}


def _wrong_per_class(y_true, y_pred):
    o = per_class_report(y_true, y_pred)
    o["precision"], o["recall"] = o["recall"], o["precision"]
    o["macro_f1"] = _r(sum(o["f1"]) / (len(o["f1"]) + 1))
    return o


_add("ml_classification_report",
     name="Per-class precision, recall, F1 and support", topic="classification metrics",
     purpose="A classification report: precision, recall, F1 and support for every class, with the macro-averaged F1.",
     inputs=[_vecin("y_true", "true class labels (whole numbers)"), _vecin("y_pred", "predicted class labels (whole numbers)")],
     outputs=[_o("classes", "classes in sorted order"), _o("precision", "precision of each class"), _o("recall", "recall of each class"),
              _o("f1", "F1 of each class"), _o("support", "number of true members of each class"), _o("macro_f1", "mean of the per-class F1")],
     primary="macro_f1", method="per-class counts, exact fractions; an undefined ratio counts as 0",
     latex=r"P_k = \frac{TP_k}{TP_k + FP_k},\quad R_k = \frac{TP_k}{TP_k + FN_k}",
     run=per_class_report, reference=_ref_per_class, wrong=_wrong_per_class,
     rand=lambda rng: dict(zip(("y_true", "y_pred"), _rand_multiclass_pair(rng))),
     phrases=["Classification report", "Per-class precision recall F1"],
     conditions=["labels are whole numbers", "equal lengths"],
     examples=[_ex({"y_true": [0, 1, 2, 2, 1, 0, 2, 1], "y_pred": [0, 2, 2, 2, 1, 0, 1, 1]},
                   {"classes": [0, 1, 2], "support": [2, 3, 3]}),
               _ex({"y_true": [1, 1, 2, 2, 3, 3, 3], "y_pred": [1, 2, 2, 2, 3, 1, 3]})],
     level="intermediate")


def cohen_kappa(y1, y2, weights):
    a, b = _labels("y1", y1), _labels("y2", y2)
    _same_len(a, b, "y1", "y2")
    cls = _classes(a, b)
    k = len(cls)
    _need(k >= 2, "kappa needs at least two classes")
    idx = {c: i for i, c in enumerate(cls)}
    n = len(a)
    O = [[0] * k for _ in range(k)]
    for u, v in zip(a, b):
        O[idx[u]][idx[v]] += 1
    rows = [sum(r) for r in O]
    cols = [sum(O[i][j] for i in range(k)) for j in range(k)]

    def w(i, j):
        if weights == "none":
            return 0 if i == j else 1
        return abs(i - j) if weights == "linear" else (i - j) ** 2
    obs = sum(w(i, j) * O[i][j] for i in range(k) for j in range(k))
    exp = sum(w(i, j) * Fraction(rows[i] * cols[j], n) for i in range(k) for j in range(k))
    _need(exp != 0, "kappa is undefined: the expected disagreement is 0 (both raters use one class)")
    return _out(kappa=1 - Fraction(obs) / exp)


def _ref_kappa(y1, y2, weights):
    from sklearn.metrics import cohen_kappa_score
    return {"kappa": float(cohen_kappa_score(y1, y2, weights=None if weights == "none" else weights))}


def _wrong_kappa(y1, y2, weights):
    a, b = _labels("y1", y1), _labels("y2", y2)
    n = len(a)
    po = Fraction(sum(1 for u, v in zip(a, b) if u == v), n)
    pe = sum(Fraction(a.count(c), n) ** 2 for c in set(a))       # chance agreement from one rater only
    return _out(kappa=(po - pe) / (1 - pe))


_add("ml_cohen_kappa",
     name="Cohen's kappa (unweighted, linear or quadratic)", topic="classification metrics",
     purpose="Cohen's kappa, the agreement between two raters (or a classifier and the truth) corrected for chance, optionally weighted by the distance between classes.",
     inputs=[_vecin("y1", "labels of the first rater (whole numbers)"), _vecin("y2", "labels of the second rater (whole numbers)"),
             _choice("weights", ["none", "linear", "quadratic"], "disagreement weights: none, linear or quadratic (distance between class positions in sorted order)")],
     outputs=[_o("kappa", "Cohen's kappa")],
     primary="kappa", method="kappa = 1 - (weighted observed disagreement) / (weighted chance disagreement), exact fractions",
     latex=r"\kappa = 1 - \frac{\sum_{ij} w_{ij} O_{ij}}{\sum_{ij} w_{ij} E_{ij}},\quad E_{ij} = \frac{r_i c_j}{n}",
     run=cohen_kappa, reference=_ref_kappa, wrong=_wrong_kappa,
     rand=lambda rng: dict(zip(("y1", "y2"), _rand_multiclass_pair(rng)), weights=rng.choice(["none", "linear", "quadratic"])),
     phrases=["Cohen's kappa", "Cohen kappa", "Cohens kappa"],
     conditions=["labels are whole numbers", "at least two classes", "equal lengths"],
     examples=[_ex({"y1": [0, 1, 2, 2, 1, 0, 2, 1], "y2": [0, 2, 2, 2, 1, 0, 1, 1], "weights": "none"}, {"kappa": 0.619047619048}),
               _ex({"y1": [1, 2, 3, 3, 2, 1, 3, 2, 1, 1], "y2": [1, 2, 3, 2, 2, 1, 3, 3, 2, 1], "weights": "quadratic"})],
     level="advanced")


def balanced_accuracy(y_true, y_pred):
    yt, yp = _labels("y_true", y_true), _labels("y_pred", y_pred)
    _same_len(yt, yp, "y_true", "y_pred")
    present = sorted(set(yt))
    rec = [Fraction(sum(1 for a, b in zip(yt, yp) if a == c and b == c), yt.count(c)) for c in present]
    k = len(present)
    ba = sum(rec) / k
    _need(k >= 2, "the adjusted balanced accuracy needs at least two true classes")
    adj = (ba - Fraction(1, k)) / (1 - Fraction(1, k))
    return _out(balanced_accuracy=ba, adjusted=adj)


def _ref_balanced(y_true, y_pred):
    import warnings
    from sklearn.metrics import balanced_accuracy_score
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {"balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "adjusted": float(balanced_accuracy_score(y_true, y_pred, adjusted=True))}


def _wrong_balanced(y_true, y_pred):
    yt, yp = _labels("y_true", y_true), _labels("y_pred", y_pred)
    acc = Fraction(sum(1 for a, b in zip(yt, yp) if a == b), len(yt))     # plain accuracy
    k = len(set(yt))
    return _out(balanced_accuracy=acc, adjusted=(acc - Fraction(1, k)) / (1 - Fraction(1, k)))


def _rand_imbalanced(rng):
    k = rng.randint(2, 4)
    w = [rng.uniform(0.1, 1.0) for _ in range(k)]
    n = rng.randint(20, 60)
    while True:
        yt = [rng.choices(range(k), w)[0] for _ in range(n)]
        if len(set(yt)) == k:
            break
    yp = [y if rng.random() < 0.7 else rng.randrange(k) for y in yt]
    return {"y_true": yt, "y_pred": yp}


_add("ml_balanced_accuracy",
     name="Balanced accuracy (multi-class)", topic="classification metrics",
     purpose="Balanced accuracy, the mean recall over the classes present in the true labels, and its chance-adjusted form.",
     inputs=[_vecin("y_true", "true class labels (whole numbers)"), _vecin("y_pred", "predicted class labels (whole numbers)")],
     outputs=[_o("balanced_accuracy", "mean per-class recall over the true classes"),
              _o("adjusted", "(balanced accuracy - 1/K) / (1 - 1/K), K = number of true classes")],
     primary="balanced_accuracy", method="per-class recall as exact fractions, averaged",
     latex=r"\mathrm{BA} = \frac{1}{K}\sum_{k} \frac{TP_k}{TP_k + FN_k}",
     run=balanced_accuracy, reference=_ref_balanced, wrong=_wrong_balanced, rand=_rand_imbalanced,
     phrases=["Balanced accuracy"],
     conditions=["at least two true classes", "equal lengths"],
     examples=[_ex({"y_true": [0, 0, 0, 0, 0, 0, 1, 1], "y_pred": [0, 0, 0, 0, 0, 0, 0, 1]}, {"balanced_accuracy": 0.75, "adjusted": 0.5}),
               _ex({"y_true": [0, 1, 2, 2, 1, 0, 2, 1], "y_pred": [0, 2, 2, 2, 1, 0, 1, 1]})],
     extra_q=[_bad({"y_true": [1, 1, 1], "y_pred": [1, 0, 1]})],
     level="intermediate")


def log_loss_binary(y_true, p):
    yt = _binary("y_true", y_true)
    pv = _vec("p", p)
    _same_len(yt, pv, "y_true", "p")
    for q in pv:
        _need(0 < q < 1, "every probability must lie strictly between 0 and 1")
    ll = -math.fsum(math.log(q) if y == 1 else math.log1p(-q) for y, q in zip(yt, pv)) / len(yt)
    return _out(log_loss=ll, log_loss_bits=ll / math.log(2.0))


def _ref_log_loss_binary(y_true, p):
    from sklearn.metrics import log_loss
    v = float(log_loss(y_true, p, labels=[0, 1]))
    return {"log_loss": v, "log_loss_bits": v / math.log(2.0)}


def _rand_ll(rng):
    yt = [rng.randint(0, 1) for _ in range(rng.randint(5, 40))]
    return {"y_true": yt, "p": _rand_probs(rng, yt)}


_add("ml_log_loss_binary",
     name="Binary log loss (cross-entropy)", topic="classification metrics",
     purpose="The mean binary cross-entropy (log loss) of predicted probabilities of the positive class, in nats and in bits.",
     inputs=[_vecin("y_true", "true labels (0 or 1)"), _vecin("p", "predicted probability of class 1 for each sample (0 < p < 1)")],
     outputs=[_o("log_loss", "mean log loss in nats"), _o("log_loss_bits", "mean log loss in bits")],
     primary="log_loss", method="-mean(y log p + (1 - y) log(1 - p)) with log1p for the negative class",
     latex=r"\mathcal{L} = -\frac{1}{n}\sum_i \left[y_i \ln p_i + (1-y_i)\ln(1-p_i)\right]",
     run=log_loss_binary, reference=_ref_log_loss_binary,
     wrong=lambda y_true, p: _out(log_loss=-math.fsum(math.log10(q) if y == 1 else math.log10(1 - q) for y, q in zip(y_true, p)) / len(p),
                                  log_loss_bits=0.0),
     rand=_rand_ll, phrases=["Binary log loss", "Log loss", "Binary cross-entropy of probabilities"],
     conditions=["labels 0 or 1", "0 < p < 1"],
     examples=[_ex({"y_true": [1, 0, 1, 1], "p": [0.9, 0.2, 0.7, 0.6]}, {"log_loss": 0.299001158669}),
               _ex({"y_true": [0, 0, 1], "p": [0.1, 0.4, 0.8]})],
     extra_q=[_bad({"y_true": [1, 0], "p": [1.0, 0.2]})],
     level="intermediate")


def log_loss_multiclass(y_true, P):
    yt = _labels("y_true", y_true)
    M = _mat("P", P, 1, 2)
    _same_len(yt, M, "y_true", "the rows of P")
    k = len(M[0])
    tot = []
    for y, row in zip(yt, M):
        _need(0 <= y < k, "every label must be a column index of P (0 to K-1)")
        _need(all(0 <= q <= 1 for q in row), "probabilities must lie between 0 and 1")
        _need(abs(math.fsum(row) - 1) <= 1e-9, "every row of P must sum to 1")
        _need(row[y] > 0, "the probability of the true class must be positive")
        tot.append(math.log(row[y]))
    ll = -math.fsum(tot) / len(yt)
    return _out(log_loss=ll, perplexity=math.exp(ll))


def _ref_log_loss_multi(y_true, P):
    from sklearn.metrics import log_loss
    v = float(log_loss(y_true, P, labels=list(range(len(P[0])))))
    return {"log_loss": v, "perplexity": math.exp(v)}


def _rand_llm(rng):
    k = rng.randint(3, 5)
    n = rng.randint(3, 15)
    yt = [rng.randrange(k) for _ in range(n)]
    P = []
    for y in yt:
        d = _rand_dist(rng, k)
        j = max(range(k), key=lambda i: d[i]) if rng.random() < 0.6 else rng.randrange(k)
        d[y], d[j] = d[j], d[y]
        P.append(d)
    return {"y_true": yt, "P": P}


_add("ml_log_loss_multiclass",
     name="Multi-class log loss (categorical cross-entropy)", topic="classification metrics",
     purpose="The mean categorical cross-entropy of predicted class probabilities (rows of P), with the perplexity exp(loss).",
     inputs=[_vecin("y_true", "true class index of each sample (0 to K-1)"), _matin("P", "predicted probabilities: one row per sample, one column per class, rows sum to 1")],
     outputs=[_o("log_loss", "mean log loss in nats"), _o("perplexity", "exp(log loss)")],
     primary="log_loss", method="-mean of ln P[i, y_i]",
     latex=r"\mathcal{L} = -\frac{1}{n}\sum_i \ln P_{i, y_i}",
     run=log_loss_multiclass, reference=_ref_log_loss_multi,
     wrong=lambda y_true, P: _out(log_loss=-math.fsum(math.log(max(r)) for r in P) / len(P), perplexity=0.0),
     rand=_rand_llm, phrases=["Multiclass log loss", "Categorical cross-entropy"],
     conditions=["labels are column indices of P", "rows of P sum to 1", "P[i, y_i] > 0"],
     examples=[_ex({"y_true": [0, 2, 1], "P": [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.25, 0.5, 0.25]]}, {"log_loss": 0.520215916088}),
               _ex({"y_true": [1, 1], "P": [[0.2, 0.8], [0.6, 0.4]]})],
     extra_q=[_bad({"y_true": [0, 3], "P": [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]]})],
     level="advanced")


def brier(y_true, p):
    yt = _binary("y_true", y_true)
    pv = _vec("p", p)
    _same_len(yt, pv, "y_true", "p")
    _need(all(0 <= q <= 1 for q in pv), "every probability must lie between 0 and 1")
    return _out(brier=sum((_F(q) - y) ** 2 for y, q in zip(yt, pv)) / len(yt))


def _ref_brier(y_true, p):
    from sklearn.metrics import brier_score_loss
    return {"brier": float(brier_score_loss(y_true, p, pos_label=1))}


_add("ml_brier_score",
     name="Brier score", topic="classification metrics",
     purpose="The Brier score, the mean squared difference between predicted probabilities of the positive class and the 0/1 outcomes.",
     inputs=[_vecin("y_true", "true labels (0 or 1)"), _vecin("p", "predicted probability of class 1 (0 to 1)")],
     outputs=[_o("brier", "mean squared error of the probabilities")],
     primary="brier", method="exact mean of (p - y)^2",
     latex=r"\mathrm{BS} = \frac{1}{n}\sum_i (p_i - y_i)^2",
     run=brier, reference=_ref_brier,
     wrong=lambda y_true, p: _out(brier=math.fsum(abs(q - y) for y, q in zip(y_true, p)) / len(p)),
     rand=_rand_ll, phrases=["Brier score"],
     conditions=["labels 0 or 1", "0 <= p <= 1"],
     examples=[_ex({"y_true": [1, 0, 1, 1], "p": [0.9, 0.2, 0.7, 0.6]}, {"brier": 0.075}),
               _ex({"y_true": [0, 1, 0], "p": [0.3, 0.6, 0.1]})],
     extra_q=[_bad({"y_true": [1, 0], "p": [1.2, 0.2]})],
     level="beginner")


def roc_auc(y_true, scores):
    yt = _binary("y_true", y_true)
    s = _vec("scores", scores)
    _same_len(yt, s, "y_true", "scores")
    pos = [v for y, v in zip(yt, s) if y == 1]
    neg = [v for y, v in zip(yt, s) if y == 0]
    _need(pos and neg, "ROC AUC needs at least one positive and one negative sample")
    wins = Fraction(0)
    for a in pos:
        for b in neg:
            wins += 1 if a > b else (Fraction(1, 2) if a == b else 0)
    auc = wins / (len(pos) * len(neg))
    return _out(auc=auc, gini=2 * auc - 1, n_pos=len(pos), n_neg=len(neg))


def _ref_roc_auc(y_true, scores):
    from sklearn.metrics import roc_auc_score
    a = float(roc_auc_score(y_true, scores))
    return {"auc": a, "gini": 2 * a - 1, "n_pos": int(sum(y_true)), "n_neg": int(len(y_true) - sum(y_true))}


def _wrong_roc_auc(y_true, scores):
    # rank-sum with the wrong offset n1(n1-1)/2 instead of n1(n1+1)/2
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2 + 1
        i = j + 1
    n1 = sum(1 for y in y_true if y == 1)
    n0 = len(y_true) - n1
    r1 = sum(r for r, y in zip(ranks, y_true) if y == 1)
    auc = (r1 - n1 * (n1 - 1) / 2) / (n1 * n0)
    return _out(auc=auc, gini=2 * auc - 1, n_pos=n1, n_neg=n0)


def _rand_scored(rng):
    yt, _ = _rand_binary_pair(rng, n=rng.randint(8, 50))
    return {"y_true": yt, "scores": _rand_scores(rng, yt, sep=rng.uniform(0.3, 1.5), ties=rng.random() < 0.4)}


_add("ml_roc_auc",
     name="ROC AUC from scores", topic="classification metrics",
     purpose="The area under the ROC curve from 0/1 labels and classifier scores: the probability that a random positive scores above a random negative (ties count one half), with the Gini coefficient 2 AUC - 1.",
     inputs=[_vecin("y_true", "true labels (0 or 1)"), _vecin("scores", "classifier scores (higher = more likely positive)")],
     outputs=[_o("auc", "area under the ROC curve"), _o("gini", "2 AUC - 1"), _o("n_pos", "number of positives"), _o("n_neg", "number of negatives")],
     primary="auc", method="exact count over all positive-negative pairs (Mann-Whitney), ties counted 1/2",
     latex=r"\mathrm{AUC} = \frac{1}{n_+ n_-}\sum_{i \in +}\sum_{j \in -}\left[\mathbb{1}(s_i > s_j) + \tfrac12 \mathbb{1}(s_i = s_j)\right]",
     run=roc_auc, reference=_ref_roc_auc, wrong=_wrong_roc_auc, rand=_rand_scored,
     phrases=["ROC AUC", "Area under the ROC curve"],
     conditions=["labels 0 or 1", "at least one positive and one negative"],
     examples=[_ex({"y_true": [0, 1, 1, 0, 1], "scores": [0.1, 0.9, 0.8, 0.3, 0.2]}, {"auc": 0.833333333333}),
               _ex({"y_true": [0, 0, 1, 1], "scores": [0.1, 0.4, 0.35, 0.8]}, {"auc": 0.75}),
               _ex({"y_true": [1, 0, 1, 0], "scores": [0.5, 0.5, 0.7, 0.2]}, {"auc": 0.875})],
     extra_q=[_bad({"y_true": [1, 1, 1], "scores": [0.2, 0.5, 0.7]})],
     level="intermediate")


def _curve(yt, s):
    """Cumulative (tp, fp) at each distinct score, descending."""
    pairs = sorted(zip(s, yt), key=lambda t: -t[0])
    th, tps, fps = [], [], []
    tp = fp = 0
    i = 0
    while i < len(pairs):
        v = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == v:
            tp += pairs[i][1]
            fp += 1 - pairs[i][1]
            i += 1
        th.append(v), tps.append(tp), fps.append(fp)
    return th, tps, fps


def roc_curve(y_true, scores):
    yt = _binary("y_true", y_true)
    s = _vec("scores", scores)
    _same_len(yt, s, "y_true", "scores")
    P, N = sum(yt), len(yt) - sum(yt)
    _need(P > 0 and N > 0, "the ROC curve needs at least one positive and one negative sample")
    th, tps, fps = _curve(yt, s)
    fpr = [Fraction(0)] + [Fraction(f, N) for f in fps]
    tpr = [Fraction(0)] + [Fraction(t, P) for t in tps]
    auc = sum((fpr[i + 1] - fpr[i]) * (tpr[i + 1] + tpr[i]) / 2 for i in range(len(fpr) - 1))
    return _out(thresholds=th, fpr=fpr, tpr=tpr, auc=auc)


def _ref_roc_curve(y_true, scores):
    from sklearn.metrics import auc, roc_curve as rc
    fpr, tpr, th = rc(y_true, scores, drop_intermediate=False)
    return {"thresholds": _f(th[1:]), "fpr": _f(fpr), "tpr": _f(tpr), "auc": float(auc(fpr, tpr))}


def _wrong_roc_curve(y_true, scores):
    o = roc_curve(y_true, scores)
    n = len(y_true)
    N = n - sum(y_true)
    o["fpr"] = [_r(v * N / n) for v in o["fpr"]]                        # FP / n instead of FP / N
    o["auc"] = _r(sum((o["fpr"][i + 1] - o["fpr"][i]) * (o["tpr"][i + 1] + o["tpr"][i]) / 2 for i in range(len(o["fpr"]) - 1)))
    return o


_add("ml_roc_curve",
     name="ROC curve points", topic="classification metrics",
     purpose="The ROC curve of a scored classifier: the false and true positive rates at every distinct score threshold (a score at or above the threshold is positive), and the trapezoidal area under it.",
     inputs=[_vecin("y_true", "true labels (0 or 1)"), _vecin("scores", "classifier scores")],
     outputs=[_o("thresholds", "distinct scores, descending (one per curve point after the origin)"),
              _o("fpr", "false positive rates, starting at 0"), _o("tpr", "true positive rates, starting at 0"),
              _o("auc", "trapezoidal area under the curve")],
     primary="auc", method="sort by score, accumulate TP and FP per distinct score, exact fractions",
     latex=r"\mathrm{TPR}(t) = \frac{TP(t)}{P},\ \mathrm{FPR}(t) = \frac{FP(t)}{N}",
     run=roc_curve, reference=_ref_roc_curve, wrong=_wrong_roc_curve, rand=_rand_scored,
     phrases=["ROC curve", "ROC curve points"],
     conditions=["labels 0 or 1", "at least one positive and one negative"],
     examples=[_ex({"y_true": [0, 1, 1, 0, 1], "scores": [0.1, 0.9, 0.8, 0.3, 0.2]},
                   {"auc": 0.833333333333}),
               _ex({"y_true": [1, 0, 1, 0, 0, 1], "scores": [3, 1, 2, 2, 0, 4]})],
     level="advanced")


def average_precision(y_true, scores):
    yt = _binary("y_true", y_true)
    s = _vec("scores", scores)
    _same_len(yt, s, "y_true", "scores")
    P = sum(yt)
    _need(P > 0, "average precision needs at least one positive sample")
    th, tps, fps = _curve(yt, s)
    prec = [Fraction(t, t + f) for t, f in zip(tps, fps)]
    rec = [Fraction(t, P) for t in tps]
    ap = sum((rec[i] - (rec[i - 1] if i else 0)) * prec[i] for i in range(len(th)))
    R = [Fraction(0)] + rec
    Pr = [Fraction(1)] + prec
    trap = sum((R[i + 1] - R[i]) * (Pr[i + 1] + Pr[i]) / 2 for i in range(len(R) - 1))
    return _out(average_precision=ap, pr_auc_trapezoid=trap)


def _ref_ap(y_true, scores):
    from sklearn.metrics import auc, average_precision_score, precision_recall_curve
    p, r, _ = precision_recall_curve(y_true, scores)
    return {"average_precision": float(average_precision_score(y_true, scores)), "pr_auc_trapezoid": float(auc(r, p))}


def _wrong_ap(y_true, scores):
    yt, s = list(y_true), list(scores)
    th, tps, fps = _curve(yt, s)
    prec = [t / (t + f) for t, f in zip(tps, fps)]
    return _out(average_precision=sum(prec) / len(prec), pr_auc_trapezoid=0.0)   # plain mean of the precisions


_add("ml_average_precision",
     name="Average precision and precision-recall AUC", topic="classification metrics",
     purpose="Average precision (the step-wise area under the precision-recall curve, sum of (R_k - R_{k-1}) P_k over distinct score thresholds) and the trapezoidal area under the precision-recall curve.",
     inputs=[_vecin("y_true", "true labels (0 or 1)"), _vecin("scores", "classifier scores")],
     outputs=[_o("average_precision", "average precision"), _o("pr_auc_trapezoid", "trapezoidal area under the precision-recall curve, from (recall 0, precision 1)")],
     primary="average_precision", method="precision and recall at every distinct threshold, exact fractions",
     latex=r"\mathrm{AP} = \sum_k (R_k - R_{k-1})\,P_k",
     run=average_precision, reference=_ref_ap, wrong=_wrong_ap, rand=_rand_scored,
     phrases=["Average precision", "PR AUC", "Precision-recall AUC"],
     conditions=["labels 0 or 1", "at least one positive"],
     examples=[_ex({"y_true": [0, 1, 1, 0, 1], "scores": [0.1, 0.9, 0.8, 0.3, 0.2]}, {"average_precision": 0.916666666667}),
               _ex({"y_true": [1, 0, 1, 0, 0, 1], "scores": [3, 1, 2, 2, 0, 4]})],
     level="advanced")


def top_k_accuracy(y_true, S, k):
    yt = _labels("y_true", y_true)
    M = _mat("S", S, 1, 3)
    _same_len(yt, M, "y_true", "the rows of S")
    K = len(M[0])
    _need(1 <= k < K, "k must be at least 1 and less than the number of classes")
    hits = 0
    for y, row in zip(yt, M):
        _need(0 <= y < K, "every label must be a column index of S (0 to K-1)")
        _need(len(set(row)) == K, "scores within a row must be distinct (a tie makes the top k ambiguous)")
        rank = sum(1 for v in row if v > row[y])
        hits += rank < k
    return _out(top_k_accuracy=Fraction(hits, len(yt)), n_hits=hits)


def _ref_topk(y_true, S, k):
    from sklearn.metrics import top_k_accuracy_score
    a = float(top_k_accuracy_score(y_true, S, k=k, labels=list(range(len(S[0])))))
    return {"top_k_accuracy": a, "n_hits": int(round(a * len(y_true)))}


def _wrong_topk(y_true, S, k):
    return top_k_accuracy(y_true, S, k - 1) if k > 1 else top_k_accuracy(y_true, S, k + 1)


def _rand_topk(rng):
    K = rng.randint(4, 7)
    n = rng.randint(5, 20)
    yt = [rng.randrange(K) for _ in range(n)]
    S = []
    for y in yt:
        while True:
            row = [round(rng.uniform(0, 1), 3) for _ in range(K)]
            row[y] = round(min(0.999, row[y] + rng.uniform(0, 0.4)), 3)
            if len(set(row)) == K:
                break
        S.append(row)
    return {"y_true": yt, "S": S, "k": rng.randint(2, K - 2)}


_add("ml_top_k_accuracy",
     name="Top-k accuracy", topic="classification metrics",
     purpose="The fraction of samples whose true class is among the k highest-scored classes.",
     inputs=[_vecin("y_true", "true class index of each sample (0 to K-1)"), _matin("S", "class scores: one row per sample, one column per class (K >= 3)"),
             _intin("k", "number of top classes that count as a hit", 1)],
     outputs=[_o("top_k_accuracy", "fraction of samples with the true class in the top k"), _o("n_hits", "number of such samples")],
     primary="top_k_accuracy", method="rank of the true class = number of classes scored strictly higher; a hit when rank < k",
     latex=r"\mathrm{top}\text{-}k = \frac{1}{n}\sum_i \mathbb{1}\left[\mathrm{rank}_i(y_i) \le k\right]",
     run=top_k_accuracy, reference=_ref_topk, wrong=_wrong_topk, rand=_rand_topk,
     phrases=["Top-k accuracy", "Top k accuracy"],
     conditions=["K >= 3 classes", "1 <= k < K", "distinct scores within each row"],
     examples=[_ex({"y_true": [0, 1, 2, 2], "S": [[0.5, 0.2, 0.2001], [0.3, 0.4, 0.2], [0.2, 0.4, 0.3], [0.7, 0.2, 0.1]], "k": 2},
                   {"top_k_accuracy": 0.75, "n_hits": 3}),
               _ex({"y_true": [3, 0, 1], "S": [[0.1, 0.2, 0.3, 0.4], [0.2, 0.5, 0.1, 0.25], [0.6, 0.1, 0.2, 0.15]], "k": 2})],
     extra_q=[_bad({"y_true": [0, 1], "S": [[0.5, 0.2, 0.3], [0.3, 0.4, 0.2]], "k": 3})],
     level="intermediate")


def hamming(Y_true, Y_pred):
    A, B = _mat("Y_true", Y_true), _mat("Y_pred", Y_pred)
    _need(len(A) == len(B) and len(A[0]) == len(B[0]), "Y_true and Y_pred must have the same shape")
    for M in (A, B):
        _need(all(v in (0.0, 1.0) for r in M for v in r), "entries must be 0 or 1")
    n, L = len(A), len(A[0])
    miss = sum(1 for i in range(n) for j in range(L) if A[i][j] != B[i][j])
    exact = sum(1 for i in range(n) if A[i] == B[i])
    return _out(hamming_loss=Fraction(miss, n * L), subset_accuracy=Fraction(exact, n), n_mismatches=miss)


def _ref_hamming(Y_true, Y_pred):
    np = _np()
    from sklearn.metrics import accuracy_score, hamming_loss
    A, B = np.array(Y_true, dtype=int), np.array(Y_pred, dtype=int)
    h = float(hamming_loss(A, B))
    return {"hamming_loss": h, "subset_accuracy": float(accuracy_score(A, B)), "n_mismatches": int(round(h * A.size))}


def _rand_multilabel(rng):
    n, L = rng.randint(3, 10), rng.randint(2, 6)
    A = [[rng.randint(0, 1) for _ in range(L)] for _ in range(n)]
    B = [[v if rng.random() < 0.8 else 1 - v for v in r] for r in A]
    return {"Y_true": A, "Y_pred": B}


_add("ml_hamming_loss",
     name="Hamming loss and subset accuracy (multi-label)", topic="classification metrics",
     purpose="For multi-label predictions (0/1 indicator matrices): the Hamming loss (fraction of wrong labels) and the subset accuracy (fraction of samples with every label right).",
     inputs=[_matin("Y_true", "true label indicators: one row per sample, one column per label (0 or 1)"),
             _matin("Y_pred", "predicted label indicators, same shape")],
     outputs=[_o("hamming_loss", "wrong label entries / (samples x labels)"), _o("subset_accuracy", "fraction of rows predicted exactly"),
              _o("n_mismatches", "number of wrong label entries")],
     primary="hamming_loss", method="count mismatching entries and exactly matching rows",
     latex=r"\mathrm{HL} = \frac{1}{nL}\sum_{i,j} \mathbb{1}(Y_{ij} \neq \hat Y_{ij})",
     run=hamming, reference=_ref_hamming,
     wrong=lambda Y_true, Y_pred: dict(hamming(Y_true, Y_pred), hamming_loss=_r(Fraction(hamming(Y_true, Y_pred)["n_mismatches"], len(Y_true)))),
     rand=_rand_multilabel, phrases=["Hamming loss", "Multilabel Hamming loss"],
     conditions=["entries 0 or 1", "same shape"],
     examples=[_ex({"Y_true": [[1, 0, 1], [0, 1, 0]], "Y_pred": [[1, 0, 0], [0, 1, 0]]},
                   {"hamming_loss": 0.166666666667, "subset_accuracy": 0.5}),
               _ex({"Y_true": [[1, 1], [0, 1], [1, 0]], "Y_pred": [[1, 0], [0, 1], [0, 1]]})],
     extra_q=[_bad({"Y_true": [[1, 2], [0, 1]], "Y_pred": [[1, 0], [0, 1]]})],
     level="intermediate")


def class_weights(y):
    yl = _labels("y", y)
    cls = sorted(set(yl))
    n, k = len(yl), len(cls)
    return _out(classes=cls, weights=[Fraction(n, k * yl.count(c)) for c in cls])


def _ref_class_weights(y):
    np = _np()
    from sklearn.utils.class_weight import compute_class_weight
    cls = np.unique(y)
    return {"classes": [int(c) for c in cls], "weights": _f(compute_class_weight("balanced", classes=cls, y=np.array(y)))}


_add("ml_balanced_class_weights",
     name="Balanced class weights", topic="classification metrics",
     purpose="Class weights that balance an imbalanced training set: n / (K n_c) for each class c.",
     inputs=[_vecin("y", "training labels (whole numbers)")],
     outputs=[_o("classes", "classes in sorted order"), _o("weights", "weight of each class, n / (K n_c)")],
     primary="weights", method="exact fractions n / (K n_c)",
     latex=r"w_c = \frac{n}{K\, n_c}",
     run=class_weights, reference=_ref_class_weights,
     wrong=lambda y: dict(class_weights(y), weights=[_r(Fraction(len(y), y.count(c))) for c in sorted(set(y))]),
     rand=lambda rng: {"y": _rand_imbalanced(rng)["y_true"]}, phrases=["Balanced class weights", "Class weights"],
     conditions=["labels are whole numbers"],
     examples=[_ex({"y": [0, 0, 0, 0, 0, 0, 1, 1]}, {"weights": [0.666666666667, 2.0]}),
               _ex({"y": [1, 2, 2, 3, 3, 3]}, {"weights": [2.0, 1.0, 0.666666666667]})],
     level="intermediate")


def ece(y_true, p, n_bins):
    yt = _binary("y_true", y_true)
    pv = _vec("p", p)
    _same_len(yt, pv, "y_true", "p")
    _need(all(0 <= q <= 1 for q in pv), "probabilities must lie between 0 and 1")
    _need(1 <= n_bins <= 100, "n_bins must be between 1 and 100")
    bins = {}
    for y, q in zip(yt, pv):
        b = min(int(math.floor(_F(q) * n_bins)), n_bins - 1)
        bins.setdefault(b, []).append((y, _F(q)))
    n = len(yt)
    gaps = []
    for b, items in sorted(bins.items()):
        acc = Fraction(sum(y for y, _ in items), len(items))
        conf = sum(q for _, q in items) / len(items)
        gaps.append((Fraction(len(items), n), abs(acc - conf)))
    return _out(ece=sum(w * g for w, g in gaps), mce=max(g for _, g in gaps), n_nonempty_bins=len(gaps))


def _ref_ece(y_true, p, n_bins):
    np = _np()
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy="uniform")
    edges = np.linspace(0, 1, n_bins + 1)
    ids = np.searchsorted(edges[1:-1], np.array(p), side="right")
    counts = np.bincount(ids, minlength=n_bins)
    counts = counts[counts > 0]
    gaps = np.abs(prob_true - prob_pred)
    return {"ece": float(np.sum(counts / len(p) * gaps)), "mce": float(gaps.max()), "n_nonempty_bins": int(len(counts))}


def _rand_ece(rng):
    n_bins = rng.choice([5, 10, 4, 8])
    yt = [rng.randint(0, 1) for _ in range(rng.randint(10, 60))]
    p = []
    for y in yt:
        while True:
            q = round(min(0.9999, max(0.0001, rng.betavariate(2 + 3 * y, 5 - 3 * y))), 4)
            if abs(q * n_bins - round(q * n_bins)) > 1e-6:
                break
        p.append(q)
    return {"y_true": yt, "p": p, "n_bins": n_bins}


_add("ml_expected_calibration_error",
     name="Expected calibration error", topic="classification metrics",
     purpose="The expected calibration error (support-weighted mean gap between accuracy and mean confidence over equal-width probability bins) and the maximum calibration error.",
     inputs=[_vecin("y_true", "true labels (0 or 1)"), _vecin("p", "predicted probability of class 1"),
             _intin("n_bins", "number of equal-width bins on [0, 1]", 1, 100)],
     outputs=[_o("ece", "expected calibration error"), _o("mce", "maximum calibration error over non-empty bins"),
              _o("n_nonempty_bins", "number of non-empty bins")],
     primary="ece", method="bin b = floor(p * n_bins) (p = 1 in the last bin); per bin |mean(y) - mean(p)|, weighted by the bin's share",
     latex=r"\mathrm{ECE} = \sum_b \frac{n_b}{n}\left|\bar y_b - \bar p_b\right|",
     run=ece, reference=_ref_ece,
     wrong=lambda y_true, p, n_bins: dict(ece(y_true, p, n_bins), ece=ece(y_true, p, n_bins)["mce"] / 2 + _r(sum(y_true) / len(y_true) / 7)),
     rand=_rand_ece, phrases=["Expected calibration error", "ECE"],
     conditions=["labels 0 or 1", "0 <= p <= 1", "bins are [b/n, (b+1)/n); a probability on a bin edge goes to the upper bin"],
     examples=[_ex({"y_true": [1, 0, 1, 1, 0, 0, 1, 0], "p": [0.92, 0.13, 0.67, 0.81, 0.36, 0.08, 0.44, 0.58], "n_bins": 5}),
               _ex({"y_true": [0, 1, 1, 0, 1], "p": [0.15, 0.85, 0.65, 0.45, 0.95], "n_bins": 2})],
     level="advanced")


# ================================================================================================ regression metrics
def _pair(y_true, y_pred, lo=1):
    yt, yp = _vec("y_true", y_true, lo), _vec("y_pred", y_pred, lo)
    _same_len(yt, yp, "y_true", "y_pred")
    return yt, yp


def _median(vals):
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def regression_metrics(y_true, y_pred):
    yt, yp = _pair(y_true, y_pred, 2)
    Y, P = [_F(v) for v in yt], [_F(v) for v in yp]
    n = len(Y)
    e = [a - b for a, b in zip(Y, P)]
    ybar = sum(Y) / n
    sst = sum((a - ybar) ** 2 for a in Y)
    _need(sst != 0, "R^2 is undefined: every true value is the same")
    sse = sum(v * v for v in e)
    ebar = sum(e) / n
    mse = sse / n
    return _out(mae=sum(abs(v) for v in e) / n, mse=mse, rmse=math.sqrt(mse), r2=1 - sse / sst,
                explained_variance=1 - sum((v - ebar) ** 2 for v in e) / sst,
                median_ae=_median([abs(v) for v in e]), max_error=max(abs(v) for v in e))


def _ref_regression(y_true, y_pred):
    from sklearn import metrics as m
    return {"mae": float(m.mean_absolute_error(y_true, y_pred)), "mse": float(m.mean_squared_error(y_true, y_pred)),
            "rmse": float(m.root_mean_squared_error(y_true, y_pred)), "r2": float(m.r2_score(y_true, y_pred)),
            "explained_variance": float(m.explained_variance_score(y_true, y_pred)),
            "median_ae": float(m.median_absolute_error(y_true, y_pred)), "max_error": float(m.max_error(y_true, y_pred))}


def _wrong_regression(y_true, y_pred):
    o = regression_metrics(y_true, y_pred)
    sse = math.fsum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    o["r2"] = _r(1 - sse / math.fsum(a * a for a in y_true))          # uncentred total sum of squares
    return o


def _rand_reg(rng, lo=-50, hi=50, n=None, positive=False):
    n = n or rng.randint(4, 30)
    a = rng.uniform(lo, hi)
    sc = rng.uniform(1, 20)
    yt = [round(a + rng.gauss(0, sc), 2) for _ in range(n)]
    if positive:
        yt = [abs(v) + 0.5 for v in yt]
        yt = [round(v, 2) for v in yt]
    yp = [round(v + rng.gauss(0, sc * rng.uniform(0.2, 0.9)), 2) for v in yt]
    if positive:
        yp = [round(abs(v) + 0.1, 2) for v in yp]
    return {"y_true": yt, "y_pred": yp}


_add("ml_regression_metrics",
     name="Regression metrics (MAE, MSE, RMSE, R^2, explained variance)", topic="regression metrics",
     purpose="The standard regression error metrics of predictions against true values: mean absolute error, mean squared error, root mean squared error, R^2, explained variance, median absolute error and maximum error.",
     inputs=[_vecin("y_true", "true values", 2), _vecin("y_pred", "predicted values", 2)],
     outputs=[_o("mae", "mean absolute error"), _o("mse", "mean squared error"), _o("rmse", "root mean squared error"),
              _o("r2", "coefficient of determination 1 - SS_res / SS_tot"), _o("explained_variance", "1 - Var(y - y_hat) / Var(y)"),
              _o("median_ae", "median absolute error"), _o("max_error", "largest absolute error")],
     primary="r2", method="exact rational sums of the errors; the square root rounded once",
     latex=r"R^2 = 1 - \frac{\sum (y_i - \hat y_i)^2}{\sum (y_i - \bar y)^2},\quad \mathrm{RMSE} = \sqrt{\tfrac{1}{n}\sum (y_i - \hat y_i)^2}",
     run=regression_metrics, reference=_ref_regression, wrong=_wrong_regression, rand=_rand_reg,
     phrases=["Regression metrics", "Regression error metrics"],
     conditions=["at least two values", "the true values are not all equal"],
     examples=[_ex({"y_true": [3, -0.5, 2, 7], "y_pred": [2.5, 0, 2, 8]}, {"mae": 0.5, "mse": 0.375, "r2": 0.948608137045}),
               _ex({"y_true": [10, 20, 30, 40, 50], "y_pred": [12, 18, 33, 37, 52]}, {"mae": 2.4, "max_error": 3})],
     extra_q=[_bad({"y_true": [5, 5, 5], "y_pred": [4, 5, 6]})],
     level="beginner")


def mape(y_true, y_pred):
    yt, yp = _pair(y_true, y_pred)
    _need(all(v != 0 for v in yt), "MAPE is undefined when a true value is 0")
    m = sum(abs(_F(a) - _F(b)) / abs(_F(a)) for a, b in zip(yt, yp)) / len(yt)
    return _out(mape=m, mape_percent=100 * m)


def _ref_mape(y_true, y_pred):
    from sklearn.metrics import mean_absolute_percentage_error
    v = float(mean_absolute_percentage_error(y_true, y_pred))
    return {"mape": v, "mape_percent": 100 * v}


_add("ml_mape",
     name="Mean absolute percentage error", topic="regression metrics",
     purpose="The mean absolute percentage error of predictions, as a fraction and in per cent.",
     inputs=[_vecin("y_true", "true values (none equal to 0)"), _vecin("y_pred", "predicted values")],
     outputs=[_o("mape", "mean of |y - y_hat| / |y| (a fraction)"), _o("mape_percent", "the same in per cent")],
     primary="mape", method="exact mean of the relative absolute errors",
     latex=r"\mathrm{MAPE} = \frac{1}{n}\sum_i \frac{|y_i - \hat y_i|}{|y_i|}",
     run=mape, reference=_ref_mape,
     wrong=lambda y_true, y_pred: _out(mape=math.fsum(abs(a - b) / abs(b) for a, b in zip(y_true, y_pred)) / len(y_true), mape_percent=0.0),
     rand=lambda rng: _rand_reg(rng, 20, 200, positive=True),
     phrases=["Mean absolute percentage error", "MAPE"],
     conditions=["no true value equal to 0"],
     examples=[_ex({"y_true": [100, 200, 300], "y_pred": [110, 190, 330]}, {"mape": 0.0833333333333, "mape_percent": 8.33333333333}),
               _ex({"y_true": [3, -0.5, 2, 7], "y_pred": [2.5, 0, 2, 8]})],
     extra_q=[_bad({"y_true": [0, 2], "y_pred": [1, 2]})],
     level="beginner")


def adjusted_r2(y_true, y_pred, p):
    yt, yp = _pair(y_true, y_pred, 3)
    n = len(yt)
    _need(isinstance(p, int) and p >= 1, "p (number of predictors) must be a whole number >= 1")
    _need(n - p - 1 > 0, "adjusted R^2 needs more samples than predictors plus one (n - p - 1 > 0)")
    r2 = _F(regression_metrics(yt, yp)["r2"]) if False else None
    Y, P = [_F(v) for v in yt], [_F(v) for v in yp]
    ybar = sum(Y) / n
    sst = sum((a - ybar) ** 2 for a in Y)
    _need(sst != 0, "R^2 is undefined: every true value is the same")
    r2 = 1 - sum((a - b) ** 2 for a, b in zip(Y, P)) / sst
    return _out(r2=r2, adjusted_r2=1 - (1 - r2) * Fraction(n - 1, n - p - 1))


def _ref_adj_r2(y_true, y_pred, p):
    np = _np()
    from sklearn.metrics import r2_score
    y, yh = np.array(y_true), np.array(y_pred)
    n = len(y)
    return {"r2": float(r2_score(y, yh)),
            "adjusted_r2": float(1 - (np.sum((y - yh) ** 2) / (n - p - 1)) / np.var(y, ddof=1))}


_add("ml_adjusted_r2",
     name="Adjusted R^2", topic="regression metrics",
     purpose="R^2 and the adjusted R^2 that penalises the number of predictors p of the model.",
     inputs=[_vecin("y_true", "true values", 3), _vecin("y_pred", "fitted values", 3), _intin("p", "number of predictors (excluding the intercept)", 1)],
     outputs=[_o("r2", "R^2"), _o("adjusted_r2", "1 - (1 - R^2)(n - 1)/(n - p - 1)")],
     primary="adjusted_r2", method="exact rational arithmetic",
     latex=r"\bar R^2 = 1 - (1 - R^2)\frac{n-1}{n-p-1}",
     run=adjusted_r2, reference=_ref_adj_r2,
     wrong=lambda y_true, y_pred, p: dict(adjusted_r2(y_true, y_pred, p), adjusted_r2=_r(1 - (1 - adjusted_r2(y_true, y_pred, p)["r2"]) * (len(y_true) - 1) / (len(y_true) - p))),
     rand=lambda rng: dict(_rand_reg(rng, n=rng.randint(8, 30)), p=rng.randint(1, 5)),
     phrases=["Adjusted R squared", "Adjusted R^2", "Adjusted R2"],
     conditions=["n - p - 1 > 0", "true values not all equal"],
     examples=[_ex({"y_true": [10, 20, 30, 40, 50], "y_pred": [12, 18, 33, 37, 52], "p": 1}, {"r2": 0.97, "adjusted_r2": 0.96}),
               _ex({"y_true": [3, 5, 7, 6, 9, 11, 10, 14], "y_pred": [3.5, 4.5, 6.5, 7, 8.5, 10.5, 11, 13], "p": 2})],
     extra_q=[_bad({"y_true": [1, 2, 4], "y_pred": [1, 2, 3], "p": 2})],
     weak="adjusted R^2 is checked by the same definition rearranged with numpy (residual variance over the sample variance); R^2 by scikit-learn",
     level="intermediate")


def huber(y_true, y_pred, delta):
    yt, yp = _pair(y_true, y_pred)
    _fin("delta", delta)
    _need(delta > 0, "delta must be positive")
    d = _F(delta)
    h, pseudo = [], []
    for a, b in zip(yt, yp):
        r = _F(a) - _F(b)
        h.append(r * r / 2 if abs(r) <= d else d * (abs(r) - d / 2))
        x2 = float(r * r / (d * d))
        pseudo.append(delta * delta * x2 / (math.sqrt(1 + x2) + 1))
    n = len(yt)
    H = sum(h) / n
    return _out(huber=H, smooth_l1=H / d, pseudo_huber=math.fsum(pseudo) / n)


def _ref_huber(y_true, y_pred, delta):
    import torch
    from scipy.special import pseudo_huber
    np = _np()
    a, b = torch.tensor(y_pred, dtype=torch.float64), torch.tensor(y_true, dtype=torch.float64)
    h = torch.nn.functional.huber_loss(a, b, reduction="mean", delta=delta).item()
    s = torch.nn.functional.smooth_l1_loss(a, b, reduction="mean", beta=delta).item()
    ph = float(np.mean(pseudo_huber(delta, np.array(y_true) - np.array(y_pred))))
    return {"huber": h, "smooth_l1": s, "pseudo_huber": ph}


def _wrong_huber(y_true, y_pred, delta):
    h = [((a - b) ** 2 / 2 if abs(a - b) <= delta else delta * abs(a - b)) for a, b in zip(y_true, y_pred)]   # no -delta/2
    H = math.fsum(h) / len(h)
    return _out(huber=H, smooth_l1=H / delta, pseudo_huber=0.0)


_add("ml_huber_loss",
     name="Huber, smooth L1 and pseudo-Huber loss", topic="loss functions and activations",
     purpose="The mean Huber loss with threshold delta (quadratic for small errors, linear for large), the smooth L1 loss (Huber / delta) and the pseudo-Huber loss.",
     inputs=[_vecin("y_true", "true values"), _vecin("y_pred", "predicted values"), _numin("delta", "threshold between the quadratic and linear parts (delta > 0)")],
     outputs=[_o("huber", "mean Huber loss"), _o("smooth_l1", "mean smooth L1 loss with beta = delta"), _o("pseudo_huber", "mean pseudo-Huber loss")],
     primary="huber", method="exact piecewise Huber terms; pseudo-Huber in the cancellation-free form delta^2 x^2 / (sqrt(1 + x^2) + 1)",
     latex=r"L_\delta(r) = \begin{cases} \tfrac12 r^2 & |r| \le \delta \\ \delta\,(|r| - \tfrac12\delta) & |r| > \delta \end{cases}",
     run=huber, reference=_ref_huber, wrong=_wrong_huber,
     rand=lambda rng: dict(_rand_reg(rng, -5, 5), delta=rng.choice([1.0, 0.5, 2.0, round(rng.uniform(0.3, 5), 2)])),
     phrases=["Huber loss"],
     conditions=["delta > 0"],
     examples=[_ex({"y_true": [3, -0.5, 2, 7], "y_pred": [2.5, 0, 2, 8], "delta": 1}, {"huber": 0.1875}),
               _ex({"y_true": [1, 2, 3], "y_pred": [4, 2, 2.5], "delta": 1.5})],
     extra_q=[_bad({"y_true": [1, 2], "y_pred": [1, 3], "delta": 0})],
     level="intermediate")


def pinball(y_true, y_pred, alpha):
    yt, yp = _pair(y_true, y_pred)
    _fin("alpha", alpha)
    _need(0 < alpha < 1, "alpha (the quantile level) must lie strictly between 0 and 1")
    a = _F(alpha)
    tot = sum(max(a * (_F(u) - _F(v)), (a - 1) * (_F(u) - _F(v))) for u, v in zip(yt, yp))
    return _out(pinball=tot / len(yt))


def _ref_pinball(y_true, y_pred, alpha):
    from sklearn.metrics import mean_pinball_loss
    return {"pinball": float(mean_pinball_loss(y_true, y_pred, alpha=alpha))}


_add("ml_quantile_loss",
     name="Quantile (pinball) loss", topic="loss functions and activations",
     purpose="The mean pinball loss of predictions of the alpha-quantile: under-predictions cost alpha per unit, over-predictions 1 - alpha.",
     inputs=[_vecin("y_true", "true values"), _vecin("y_pred", "predicted quantiles"), _numin("alpha", "quantile level (0 < alpha < 1)", pct=True)],
     outputs=[_o("pinball", "mean pinball loss")],
     primary="pinball", method="exact mean of max(alpha (y - q), (alpha - 1)(y - q))",
     latex=r"L_\alpha(y, q) = \max\{\alpha (y - q),\ (\alpha - 1)(y - q)\}",
     run=pinball, reference=_ref_pinball,
     wrong=lambda y_true, y_pred, alpha: pinball(y_true, y_pred, 1 - alpha) if abs(alpha - 0.5) > 0.05 else _out(pinball=0.0),
     rand=lambda rng: dict(_rand_reg(rng), alpha=rng.choice([0.1, 0.9, 0.25, 0.75, 0.05, 0.95, round(rng.uniform(0.02, 0.98), 2)])),
     phrases=["Pinball loss", "Quantile loss"],
     conditions=["0 < alpha < 1"],
     examples=[_ex({"y_true": [1, 2, 3], "y_pred": [0, 2, 3], "alpha": 0.1}, {"pinball": 0.0333333333333}),
               _ex({"y_true": [10, 20, 30], "y_pred": [12, 15, 33], "alpha": 0.9})],
     extra_q=[_bad({"y_true": [1, 2], "y_pred": [1, 3], "alpha": 1.5})],
     level="intermediate")


def tweedie(y_true, y_pred, power):
    yt, yp = _pair(y_true, y_pred)
    _fin("power", power)
    _need(power == 0 or power >= 1, "power must be 0 (normal) or at least 1 (1 Poisson, 2 gamma, 3 inverse Gaussian)")
    if power > 0:
        _need(all(m > 0 for m in yp), "predictions must be positive for power > 0")
    if 1 <= power < 2:
        _need(all(y >= 0 for y in yt), "true values must be non-negative for 1 <= power < 2")
    if power >= 2:
        _need(all(y > 0 for y in yt), "true values must be positive for power >= 2")
    p = power
    dev = []
    for y, m in zip(yt, yp):
        if p == 0:
            dev.append((y - m) ** 2)
        elif p == 1:
            dev.append(2 * ((y * math.log(y / m) if y > 0 else 0.0) - y + m))
        elif p == 2:
            dev.append(2 * (math.log(m / y) + y / m - 1))
        else:
            dev.append(2 * (max(y, 0.0) ** (2 - p) / ((1 - p) * (2 - p)) - y * m ** (1 - p) / (1 - p) + m ** (2 - p) / (2 - p)))
    return _out(deviance=math.fsum(dev) / len(dev))


def _ref_tweedie(y_true, y_pred, power):
    from sklearn.metrics import mean_tweedie_deviance
    return {"deviance": float(mean_tweedie_deviance(y_true, y_pred, power=power))}


def _rand_tweedie(rng):
    p = rng.choice([0, 1, 1.5, 2, 3, round(rng.uniform(1.1, 1.9), 2), round(rng.uniform(2.2, 3.5), 2)])
    n = rng.randint(4, 20)
    yt = [round(rng.uniform(0.5, 12), 2) for _ in range(n)]
    yp = [round(max(0.2, v * rng.uniform(0.3, 2.0)), 2) for v in yt]
    if 1 <= p < 2 and rng.random() < 0.5:
        yt[0] = 0.0
    return {"y_true": yt, "y_pred": yp, "power": p}


_add("ml_tweedie_deviance",
     name="Mean Tweedie deviance (normal, Poisson, gamma, compound Poisson-gamma)", topic="regression metrics",
     purpose="The mean Tweedie deviance of predictions for a stated power: 0 squared error, 1 Poisson, 2 gamma, 3 inverse Gaussian, between 1 and 2 compound Poisson-gamma.",
     inputs=[_vecin("y_true", "true values"), _vecin("y_pred", "predicted means (positive for power > 0)"),
             _numin("power", "Tweedie power p (0, or p >= 1)")],
     outputs=[_o("deviance", "mean Tweedie deviance")],
     primary="deviance", method="unit deviance of the Tweedie family for each sample, averaged",
     latex=r"d_p(y, \mu) = 2\left(\frac{y^{2-p}}{(1-p)(2-p)} - \frac{y\mu^{1-p}}{1-p} + \frac{\mu^{2-p}}{2-p}\right)",
     run=tweedie, reference=_ref_tweedie,
     wrong=lambda y_true, y_pred, power: _out(deviance=tweedie(y_true, y_pred, power)["deviance"] / 2 + (1.0 if power == 0 else 0.0) * math.fsum(y_true) / len(y_true)),
     rand=_rand_tweedie, phrases=["Tweedie deviance", "Mean Tweedie deviance"],
     conditions=["power 0 or >= 1", "positive predictions (power > 0)", "y >= 0 (1 <= p < 2) or y > 0 (p >= 2)"],
     examples=[_ex({"y_true": [2, 0, 1, 4], "y_pred": [0.5, 0.5, 2, 2], "power": 1}),
               _ex({"y_true": [2, 0.5, 1, 4], "y_pred": [0.5, 0.5, 2, 2], "power": 2}),
               _ex({"y_true": [2, 0, 1, 4], "y_pred": [0.5, 0.5, 2, 2], "power": 1.5})],
     extra_q=[_bad({"y_true": [2, 1], "y_pred": [0.5, 0.5], "power": 0.5})],
     level="expert")


def msle(y_true, y_pred):
    yt, yp = _pair(y_true, y_pred)
    _need(all(v >= 0 for v in yt + yp), "the squared log error needs non-negative values")
    m = math.fsum((math.log1p(a) - math.log1p(b)) ** 2 for a, b in zip(yt, yp)) / len(yt)
    return _out(msle=m, rmsle=math.sqrt(m))


def _ref_msle(y_true, y_pred):
    from sklearn.metrics import mean_squared_log_error, root_mean_squared_log_error
    return {"msle": float(mean_squared_log_error(y_true, y_pred)), "rmsle": float(root_mean_squared_log_error(y_true, y_pred))}


_add("ml_mean_squared_log_error",
     name="Mean squared logarithmic error", topic="regression metrics",
     purpose="The mean squared logarithmic error ln(1 + y) - ln(1 + y_hat) and its square root (RMSLE), for non-negative targets such as counts and prices.",
     inputs=[_vecin("y_true", "true values (>= 0)"), _vecin("y_pred", "predicted values (>= 0)")],
     outputs=[_o("msle", "mean squared log error"), _o("rmsle", "root mean squared log error")],
     primary="msle", method="log1p differences squared and averaged",
     latex=r"\mathrm{MSLE} = \frac{1}{n}\sum_i \left(\ln(1+y_i) - \ln(1+\hat y_i)\right)^2",
     run=msle, reference=_ref_msle,
     wrong=lambda y_true, y_pred: _out(msle=math.fsum((math.log(a + 2) - math.log(b + 2)) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true), rmsle=0.0),
     rand=lambda rng: _rand_reg(rng, 2, 60, positive=True),
     phrases=["Mean squared log error", "MSLE", "Mean squared logarithmic error"],
     conditions=["values >= 0"],
     examples=[_ex({"y_true": [3, 5, 2.5, 7], "y_pred": [2.5, 5, 4, 8]}),
               _ex({"y_true": [100, 200], "y_pred": [110, 150]})],
     extra_q=[_bad({"y_true": [-3, 5], "y_pred": [2.5, 5]})],
     level="intermediate")


# ================================================================================================ activations and losses
def softmax(z):
    zv = _vec("z", z, 2)
    return _out(probabilities=_softmax(zv), log_sum_exp=_logsumexp(zv))


def _ref_softmax(z):
    from scipy.special import logsumexp, softmax as sm
    return {"probabilities": _f(sm(z)), "log_sum_exp": float(logsumexp(z))}


def _rand_logits(rng, lo=2, hi=8, scale=4.0):
    return [round(rng.uniform(-scale, scale), 3) for _ in range(rng.randint(lo, hi))]


_add("ml_softmax",
     name="Softmax", topic="loss functions and activations",
     purpose="The softmax of a vector of logits (numerically stable: the largest logit is subtracted first), with log-sum-exp.",
     inputs=[_vecin("z", "logits (at least two)", 2)],
     outputs=[_o("probabilities", "softmax probabilities exp(z_i) / sum exp(z_j)"), _o("log_sum_exp", "log sum exp(z_j)")],
     primary="log_sum_exp", method="subtract max(z), exponentiate, normalise (fsum)",
     latex=r"\sigma(z)_i = \frac{e^{z_i - m}}{\sum_j e^{z_j - m}},\ m = \max_j z_j",
     run=softmax, reference=_ref_softmax,
     wrong=lambda z: _out(probabilities=[math.exp(v) / (1 + math.fsum(math.exp(u) for u in z)) for v in z],
                          log_sum_exp=math.log1p(math.fsum(math.exp(u) for u in z))),
     rand=lambda rng: {"z": _rand_logits(rng)}, phrases=["Softmax"],
     conditions=["at least two logits"],
     examples=[_ex({"z": [1, 2, 3]}, {"probabilities": [0.0900305731704, 0.244728471055, 0.665240955775]}),
               _ex({"z": [1000, 1001]}, {"probabilities": [0.268941421370, 0.731058578630]})],
     level="beginner")


def log_softmax(z):
    zv = _vec("z", z, 2)
    m = max(zv)
    s = math.log(math.fsum(math.exp(v - m) for v in zv))
    return _out(log_probabilities=[(v - m) - s for v in zv], log_sum_exp=m + s)


def _ref_log_softmax(z):
    import torch
    t = torch.tensor(z, dtype=torch.float64)
    return {"log_probabilities": torch.log_softmax(t, 0).tolist(), "log_sum_exp": torch.logsumexp(t, 0).item()}


_add("ml_log_softmax",
     name="Log-softmax", topic="loss functions and activations",
     purpose="The log-softmax of logits, z_i - log sum exp(z), computed stably even for very large logits.",
     inputs=[_vecin("z", "logits", 2)],
     outputs=[_o("log_probabilities", "z_i - logsumexp(z)"), _o("log_sum_exp", "logsumexp(z)")],
     primary="log_sum_exp", method="log-sum-exp with the maximum subtracted",
     latex=r"\log\sigma(z)_i = z_i - m - \log\sum_j e^{z_j - m}",
     run=log_softmax, reference=_ref_log_softmax,
     wrong=lambda z: _out(log_probabilities=[v - math.fsum(z) / len(z) for v in z], log_sum_exp=math.fsum(z) / len(z)),
     rand=lambda rng: {"z": _rand_logits(rng, scale=rng.choice([3.0, 50.0, 800.0]))}, phrases=["Log-softmax", "Log softmax"],
     conditions=["at least two logits"],
     examples=[_ex({"z": [1, 2, 3]}, {"log_probabilities": [-2.40760596444, -1.40760596444, -0.407605964448]}),
               _ex({"z": [1000, 0, -1000]})],
     level="intermediate")


def _log_sigmoid(z):
    return -math.log1p(math.exp(-z)) if z >= 0 else z - math.log1p(math.exp(z))


def sigmoid(z):
    _fin("z", z)
    e = math.exp(-abs(z))
    return _out(sigmoid=_sigmoid(z), derivative=e / (1 + e) ** 2, log_sigmoid=_log_sigmoid(z))


def _ref_sigmoid(z):
    from scipy.special import expit, log_expit
    return {"sigmoid": float(expit(z)), "derivative": float(expit(z) * expit(-z)), "log_sigmoid": float(log_expit(z))}


_add("ml_sigmoid",
     name="Sigmoid (logistic function) and its derivative", topic="loss functions and activations",
     purpose="The logistic sigmoid 1 / (1 + e^-z), its derivative sigma(1 - sigma) and log sigma(z), computed stably for large |z|.",
     inputs=[_numin("z", "input (logit)")],
     outputs=[_o("sigmoid", "1 / (1 + exp(-z))"), _o("derivative", "sigma(z)(1 - sigma(z))"), _o("log_sigmoid", "log sigma(z)")],
     primary="sigmoid", method="branch on the sign of z so no exponential overflows; derivative as e^-|z| / (1 + e^-|z|)^2",
     latex=r"\sigma(z) = \frac{1}{1 + e^{-z}},\quad \sigma'(z) = \sigma(z)\,(1 - \sigma(z))",
     run=sigmoid, reference=_ref_sigmoid,
     wrong=lambda z: _out(sigmoid=1 / (1 + math.exp(min(z, 700))), derivative=0.0, log_sigmoid=0.0),
     rand=lambda rng: {"z": round(rng.choice([rng.uniform(-6, 6), rng.uniform(-40, 40)]), 3)}, phrases=["Sigmoid", "Logistic function"],
     conditions=["finite z"],
     examples=[_ex({"z": 0}, {"sigmoid": 0.5, "derivative": 0.25}), _ex({"z": 2}, {"sigmoid": 0.880797077978}), _ex({"z": -35})],
     level="beginner")


def ce_logits(z, y):
    zv = _vec("z", z, 2)
    _need(isinstance(y, int) and 0 <= y < len(zv), "y must be the index of the true class (0 to K-1)")
    p = _softmax(zv)
    g = list(p)
    g[y] = -math.fsum(p[j] for j in range(len(p)) if j != y)
    return _out(loss=-_log_softmax(zv)[y], probability=p[y], gradient=g)


def _ref_ce_logits(z, y):
    import torch
    t = torch.tensor([z], dtype=torch.float64, requires_grad=True)
    loss = torch.nn.functional.cross_entropy(t, torch.tensor([y]))
    loss.backward()
    return {"loss": loss.item(), "probability": torch.softmax(t.detach(), 1)[0, y].item(), "gradient": t.grad[0].tolist()}


def _wrong_ce_logits(z, y):
    p = _softmax(z)
    return _out(loss=-math.log(p[y]), probability=p[y], gradient=[q - (1 if j != y else 0) for j, q in enumerate(p)])


_add("ml_cross_entropy_logits",
     name="Cross-entropy loss from logits and its gradient", topic="loss functions and activations",
     purpose="The cross-entropy loss of one sample from its logits and true class, the predicted probability of that class, and the gradient of the loss with respect to the logits (softmax minus one-hot).",
     inputs=[_vecin("z", "logits of the classes", 2), _intin("y", "index of the true class (0 to K-1)", 0)],
     outputs=[_o("loss", "logsumexp(z) - z_y"), _o("probability", "softmax probability of the true class"),
              _o("gradient", "dL/dz = softmax(z) - onehot(y)")],
     primary="loss", method="log-sum-exp form; the true-class gradient as minus the sum of the other probabilities (no cancellation)",
     latex=r"L = -\log\frac{e^{z_y}}{\sum_j e^{z_j}},\quad \frac{\partial L}{\partial z_j} = p_j - \mathbb{1}(j = y)",
     run=ce_logits, reference=_ref_ce_logits, wrong=_wrong_ce_logits,
     rand=lambda rng: (lambda z: {"z": z, "y": rng.randrange(len(z))})(_rand_logits(rng, 2, 6, 3.0)),
     phrases=["Cross-entropy from logits", "Softmax cross-entropy"],
     conditions=["y is a class index"],
     examples=[_ex({"z": [2, 1, 0.1], "y": 0}, {"loss": 0.417030016278}), _ex({"z": [0.5, -1, 3, 0], "y": 2})],
     extra_q=[_bad({"z": [2, 1, 0.1], "y": 3})],
     level="intermediate")


def bce_logits(y, z):
    yv, zv = _vec("y", y), _vec("z", z)
    _same_len(yv, zv, "y", "z")
    _need(all(0 <= v <= 1 for v in yv), "targets y must lie between 0 and 1")
    n = len(yv)
    loss = math.fsum(max(b, 0.0) - b * a + math.log1p(math.exp(-abs(b))) for a, b in zip(yv, zv)) / n
    return _out(loss=loss, gradient=[(_sigmoid(b) - a) / n for a, b in zip(yv, zv)])


def _ref_bce_logits(y, z):
    import torch
    t = torch.tensor(z, dtype=torch.float64, requires_grad=True)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(t, torch.tensor(y, dtype=torch.float64))
    loss.backward()
    return {"loss": loss.item(), "gradient": t.grad.tolist()}


def _rand_bce(rng):
    n = rng.randint(1, 10)
    y = [rng.randint(0, 1) if rng.random() < 0.7 else round(rng.uniform(0, 1), 2) for _ in range(n)]
    z = [round(rng.uniform(-4, 4), 3) for _ in range(n)]
    return {"y": y, "z": z}


_add("ml_bce_with_logits",
     name="Binary cross-entropy with logits and its gradient", topic="loss functions and activations",
     purpose="The mean binary cross-entropy computed directly from logits (stable for large |z|; soft targets allowed) and its gradient with respect to each logit.",
     inputs=[_vecin("y", "targets between 0 and 1"), _vecin("z", "logits")],
     outputs=[_o("loss", "mean of max(z, 0) - z y + log(1 + e^-|z|)"), _o("gradient", "dL/dz_i = (sigma(z_i) - y_i) / n")],
     primary="loss", method="the log-sum-exp form max(z, 0) - z y + log1p(e^-|z|)",
     latex=r"\ell(z, y) = \max(z, 0) - z y + \log(1 + e^{-|z|}),\quad \frac{\partial \ell}{\partial z} = \sigma(z) - y",
     run=bce_logits, reference=_ref_bce_logits,
     wrong=lambda y, z: _out(loss=math.fsum(math.log1p(math.exp(-b)) * a for a, b in zip(y, z) if b > -700) / len(y),
                             gradient=[_sigmoid(b) - a for a, b in zip(y, z)]),
     rand=_rand_bce, phrases=["Binary cross-entropy with logits", "BCE with logits"],
     conditions=["0 <= y <= 1"],
     examples=[_ex({"y": [1, 0, 1], "z": [2, -1, -0.5]}), _ex({"y": [0.8, 0.1], "z": [1.5, 0.3]})],
     extra_q=[_bad({"y": [2, 0], "z": [1, 1]})],
     level="intermediate")


def hinge(y_true, scores):
    yt = _labels("y_true", y_true)
    _need(all(v in (-1, 1) for v in yt), "labels must be -1 or +1")
    s = _vec("scores", scores)
    _same_len(yt, s, "y_true", "scores")
    _need(len(set(yt)) == 2, "both classes (-1 and +1) must occur")
    m = [max(Fraction(0), 1 - y * _F(v)) for y, v in zip(yt, s)]
    n = len(m)
    return _out(hinge=sum(m) / n, squared_hinge=sum(v * v for v in m) / n, n_violations=sum(1 for v in m if v > 0))


def _ref_hinge(y_true, scores):
    np = _np()
    from sklearn.metrics import hinge_loss
    y, s = np.array(y_true), np.array(scores)
    m = np.maximum(0, 1 - y * s)
    return {"hinge": float(hinge_loss(y, s)), "squared_hinge": float(np.mean(m ** 2)), "n_violations": int(np.sum(y * s < 1))}


def _rand_hinge(rng):
    n = rng.randint(4, 25)
    while True:
        y = [rng.choice([-1, 1]) for _ in range(n)]
        if len(set(y)) == 2:
            break
    return {"y_true": y, "scores": [round(v * rng.uniform(-0.5, 2.5), 2) for v in y]}


_add("ml_hinge_loss",
     name="Hinge and squared hinge loss", topic="loss functions and activations",
     purpose="The mean hinge loss max(0, 1 - y f) of a margin classifier with labels -1/+1, the mean squared hinge loss and the number of margin violations.",
     inputs=[_vecin("y_true", "true labels (-1 or +1)"), _vecin("scores", "decision values f(x)")],
     outputs=[_o("hinge", "mean max(0, 1 - y f)"), _o("squared_hinge", "mean max(0, 1 - y f)^2"), _o("n_violations", "samples with y f < 1")],
     primary="hinge", method="exact fractions",
     latex=r"\ell(y, f) = \max(0,\ 1 - y f)",
     run=hinge, reference=_ref_hinge,
     wrong=lambda y_true, scores: _out(hinge=math.fsum(max(0.0, -y * v) for y, v in zip(y_true, scores)) / len(scores), squared_hinge=0.0, n_violations=0),
     rand=_rand_hinge, phrases=["Hinge loss"],
     conditions=["labels -1 or +1, both present"],
     examples=[_ex({"y_true": [1, -1, 1, -1], "scores": [2, -0.5, 0.3, 0.8]}, {"hinge": 0.75, "n_violations": 3}),
               _ex({"y_true": [1, 1, -1], "scores": [0.9, 1.5, -2]})],
     extra_q=[_bad({"y_true": [1, 0, 1], "scores": [2, -0.5, 0.3]})],
     weak="the squared hinge loss is checked by a numpy re-derivation of the same definition; the hinge loss by scikit-learn",
     level="beginner")


def multiclass_hinge(y_true, D):
    yt = _labels("y_true", y_true)
    M = _mat("D", D, 1, 3)
    _same_len(yt, M, "y_true", "the rows of D")
    K = len(M[0])
    cs, ww = [], []
    for y, row in zip(yt, M):
        _need(0 <= y < K, "every label must be a column index of D (0 to K-1)")
        fy = _F(row[y])
        others = [_F(row[j]) for j in range(K) if j != y]
        cs.append(max(Fraction(0), 1 - fy + max(others)))
        ww.append(sum(max(Fraction(0), 1 - fy + o) for o in others) / K)
    n = len(yt)
    return _out(crammer_singer=sum(cs) / n, weston_watkins=sum(ww) / n)


def _ref_mc_hinge(y_true, D):
    import torch
    from sklearn.metrics import hinge_loss
    cs = float(hinge_loss(y_true, D, labels=list(range(len(D[0])))))
    ww = torch.nn.functional.multi_margin_loss(torch.tensor(D, dtype=torch.float64), torch.tensor(y_true, dtype=torch.long)).item()
    return {"crammer_singer": cs, "weston_watkins": ww}


def _rand_mc_hinge(rng):
    K = rng.randint(3, 5)
    n = rng.randint(3, 12)
    while True:
        y = [rng.randrange(K) for _ in range(n)]
        if len(set(y)) >= 2:
            break
    D = [[round(rng.uniform(-2, 2) + (1.0 if j == t else 0.0), 2) for j in range(K)] for t in y]
    return {"y_true": y, "D": D}


_add("ml_multiclass_hinge_loss",
     name="Multi-class hinge loss (Crammer-Singer and Weston-Watkins)", topic="loss functions and activations",
     purpose="The multi-class hinge losses of a decision matrix: Crammer-Singer max(0, 1 - f_y + max_{j != y} f_j) and Weston-Watkins sum_{j != y} max(0, 1 - f_y + f_j) / K, averaged over samples.",
     inputs=[_vecin("y_true", "true class index of each sample (0 to K-1)"), _matin("D", "decision values: one row per sample, one column per class (K >= 3)")],
     outputs=[_o("crammer_singer", "mean Crammer-Singer hinge loss"), _o("weston_watkins", "mean Weston-Watkins hinge loss (sum over wrong classes divided by K)")],
     primary="crammer_singer", method="exact fractions",
     latex=r"\ell_{CS} = \max\left(0,\ 1 - f_y + \max_{j \ne y} f_j\right),\quad \ell_{WW} = \frac{1}{K}\sum_{j \ne y}\max(0,\ 1 - f_y + f_j)",
     run=multiclass_hinge, reference=_ref_mc_hinge,
     wrong=lambda y_true, D: (lambda o: dict(o, crammer_singer=_r(o["crammer_singer"] * 0.5 + 0.25)))(multiclass_hinge(y_true, D)),
     rand=_rand_mc_hinge, phrases=["Multiclass hinge loss", "Multi-class hinge loss"],
     conditions=["K >= 3 classes", "labels are column indices"],
     examples=[_ex({"y_true": [0, 1, 2], "D": [[1.2, 0.3, -0.5], [0.4, 0.9, 1.1], [-0.2, 0.1, 2.0]]}),
               _ex({"y_true": [2, 0], "D": [[0.5, 0.2, 0.4], [1.0, 1.5, -1.0]]})],
     level="advanced")


def _dist(name, v):
    p = _vec(name, v, 2)
    _need(all(x > 0 for x in p), f"every probability in {name} must be positive")
    _need(abs(math.fsum(p) - 1) <= 1e-9, f"{name} must sum to 1")
    return p


def kl_discrete(p, q, unit):
    P, Q = _dist("p", p), _dist("q", q)
    _same_len(P, Q, "p", "q")
    L = _LN(unit)
    kl = lambda a, b: math.fsum(x * math.log(x / y) for x, y in zip(a, b)) / L
    M = [(x + y) / 2 for x, y in zip(P, Q)]
    return _out(kl_pq=kl(P, Q), kl_qp=kl(Q, P), js_divergence=(kl(P, M) + kl(Q, M)) / 2,
                cross_entropy=-math.fsum(x * math.log(y) for x, y in zip(P, Q)) / L,
                entropy_p=-math.fsum(x * math.log(x) for x in P) / L)


def _ref_kl(p, q, unit):
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import entropy
    base = {"nats": None, "bits": 2, "bans": 10}[unit]
    return {"kl_pq": float(entropy(p, q, base=base)), "kl_qp": float(entropy(q, p, base=base)),
            "js_divergence": float(jensenshannon(p, q, base=base)) ** 2,
            "cross_entropy": float(entropy(p, base=base) + entropy(p, q, base=base)), "entropy_p": float(entropy(p, base=base))}


_UNIT = lambda: _choice("unit", ["nats", "bits", "bans"], "unit of information: nats (natural log), bits (log base 2) or bans (log base 10)")

_add("ml_kl_divergence_discrete",
     name="KL divergence, Jensen-Shannon divergence and cross-entropy of discrete distributions", topic="information theory",
     purpose="The Kullback-Leibler divergence of two discrete distributions in both directions, the Jensen-Shannon divergence, the cross-entropy H(p, q) and the entropy H(p).",
     inputs=[_vecin("p", "first distribution (positive, sums to 1)", 2), _vecin("q", "second distribution (positive, sums to 1)", 2), _UNIT()],
     outputs=[_o("kl_pq", "KL(p || q)"), _o("kl_qp", "KL(q || p)"), _o("js_divergence", "Jensen-Shannon divergence"),
              _o("cross_entropy", "H(p, q) = -sum p log q"), _o("entropy_p", "H(p)")],
     primary="kl_pq", method="direct sums with fsum",
     latex=r"D_{KL}(p \| q) = \sum_i p_i \log\frac{p_i}{q_i},\quad H(p, q) = H(p) + D_{KL}(p \| q)",
     run=kl_discrete, reference=_ref_kl,
     wrong=lambda p, q, unit: (lambda o: dict(o, kl_pq=o["kl_qp"]))(kl_discrete(p, q, unit)),
     rand=lambda rng: (lambda k: {"p": _rand_dist(rng, k), "q": _rand_dist(rng, k), "unit": rng.choice(["nats", "bits", "bans"])})(rng.randint(2, 6)),
     phrases=["KL divergence", "Kullback-Leibler divergence"],
     conditions=["positive probabilities summing to 1", "same length"],
     examples=[_ex({"p": [0.5, 0.5], "q": [0.9, 0.1], "unit": "bits"}, {"kl_pq": 0.736965594166}),
               _ex({"p": [0.36, 0.48, 0.16], "q": [0.333, 0.333, 0.334], "unit": "nats"})],
     extra_q=[_bad({"p": [0.5, 0.6], "q": [0.9, 0.1], "unit": "bits"})],
     level="intermediate")


def label_smoothing_ce(z, y, epsilon):
    zv = _vec("z", z, 2)
    K = len(zv)
    _need(isinstance(y, int) and 0 <= y < K, "y must be the index of the true class (0 to K-1)")
    _fin("epsilon", epsilon)
    _need(0 <= epsilon < 1, "epsilon must lie in [0, 1)")
    logp = _log_softmax(zv)
    q = [epsilon / K + (1 - epsilon if j == y else 0.0) for j in range(K)]
    p = _softmax(zv)
    g = [p[j] - q[j] for j in range(K)]
    return _out(loss=-math.fsum(a * b for a, b in zip(q, logp)), gradient=g)


def _ref_ls_ce(z, y, epsilon):
    import torch
    t = torch.tensor([z], dtype=torch.float64, requires_grad=True)
    loss = torch.nn.functional.cross_entropy(t, torch.tensor([y]), label_smoothing=epsilon)
    loss.backward()
    return {"loss": loss.item(), "gradient": t.grad[0].tolist()}


_add("ml_label_smoothing_cross_entropy",
     name="Cross-entropy with label smoothing", topic="loss functions and activations",
     purpose="The cross-entropy of logits against a smoothed target (1 - epsilon on the true class plus epsilon spread evenly over all K classes) and its gradient with respect to the logits.",
     inputs=[_vecin("z", "logits", 2), _intin("y", "index of the true class (0 to K-1)", 0), _numin("epsilon", "smoothing amount (0 <= epsilon < 1)")],
     outputs=[_o("loss", "-sum_j q_j log softmax(z)_j"), _o("gradient", "softmax(z) - q")],
     primary="loss", method="smoothed target q_j = epsilon/K + (1 - epsilon) 1(j = y); stable log-softmax",
     latex=r"q_j = \frac{\varepsilon}{K} + (1-\varepsilon)\mathbb{1}(j=y),\quad L = -\sum_j q_j \log \sigma(z)_j",
     run=label_smoothing_ce, reference=_ref_ls_ce,
     wrong=lambda z, y, epsilon: (lambda K: label_smoothing_ce(z, y, 0.0) if epsilon > 0.02 else _out(loss=0.0, gradient=[0.0]))(len(z)),
     rand=lambda rng: (lambda z: {"z": z, "y": rng.randrange(len(z)), "epsilon": rng.choice([0.1, 0.2, 0.05, round(rng.uniform(0.03, 0.5), 2)])})(_rand_logits(rng, 2, 6, 3.0)),
     phrases=["Label smoothing cross-entropy", "Cross-entropy with label smoothing"],
     conditions=["0 <= epsilon < 1", "y is a class index"],
     examples=[_ex({"z": [2, 1, 0.1], "y": 0, "epsilon": 0.1}), _ex({"z": [0.5, -1, 3, 0], "y": 1, "epsilon": 0.2})],
     extra_q=[_bad({"z": [2, 1, 0.1], "y": 0, "epsilon": 1.2})],
     level="advanced")


def focal(y, z, gamma, alpha):
    yv = _binary("y", y)
    zv = _vec("z", z)
    _same_len(yv, zv, "y", "z")
    _fin("gamma", gamma), _fin("alpha", alpha)
    _need(gamma >= 0, "gamma must be >= 0")
    _need(0 <= alpha <= 1, "alpha must lie between 0 and 1")
    n = len(yv)
    L, G = [], []
    for t, v in zip(yv, zv):
        s = 1 if t == 1 else -1
        logpt = _log_sigmoid(s * v)
        pt = _sigmoid(s * v)
        q = _sigmoid(-s * v)                   # 1 - p_t without cancellation
        at = alpha if t == 1 else 1 - alpha
        w = q ** gamma
        L.append(-at * w * logpt)
        G.append(s * at * w * (gamma * pt * logpt - q) / n)
    return _out(loss=math.fsum(L) / n, gradient=G)


def _ref_focal(y, z, gamma, alpha):
    import torch
    t = torch.tensor(z, dtype=torch.float64, requires_grad=True)
    tg = torch.tensor(y, dtype=torch.float64)
    p = torch.sigmoid(t)                         # the torchvision sigmoid_focal_loss formula
    ce = torch.nn.functional.binary_cross_entropy_with_logits(t, tg, reduction="none")
    p_t = p * tg + (1 - p) * (1 - tg)
    loss = ce * ((1 - p_t) ** gamma)
    loss = (alpha * tg + (1 - alpha) * (1 - tg)) * loss
    loss = loss.mean()
    loss.backward()
    return {"loss": loss.item(), "gradient": t.grad.tolist()}


def _rand_focal(rng):
    n = rng.randint(1, 10)
    return {"y": [rng.randint(0, 1) for _ in range(n)], "z": [round(rng.uniform(-3, 3), 3) for _ in range(n)],
            "gamma": rng.choice([2.0, 1.0, 0.5, 3.0, round(rng.uniform(0.2, 4), 2)]), "alpha": rng.choice([0.25, 0.5, 0.75, round(rng.uniform(0.05, 0.95), 2)])}


_add("ml_focal_loss",
     name="Binary focal loss and its gradient", topic="loss functions and activations",
     purpose="The mean binary focal loss -alpha_t (1 - p_t)^gamma log p_t from logits (it down-weights easy examples) and its gradient with respect to each logit.",
     inputs=[_vecin("y", "labels (0 or 1)"), _vecin("z", "logits"), _numin("gamma", "focusing parameter (>= 0; 0 gives weighted cross-entropy)"),
             _numin("alpha", "weight of the positive class (0 to 1; negatives get 1 - alpha)")],
     outputs=[_o("loss", "mean focal loss"), _o("gradient", "dL/dz_i")],
     primary="loss", method="p_t = sigma(+-z), 1 - p_t = sigma(-+z) (no cancellation); analytic gradient alpha_t (1 - p_t)^gamma (gamma p_t log p_t - (1 - p_t)) with the sign of the label",
     latex=r"FL = -\alpha_t (1 - p_t)^\gamma \log p_t",
     run=focal, reference=_ref_focal,
     wrong=lambda y, z, gamma, alpha: _out(loss=math.fsum(-(alpha if t == 1 else 1 - alpha) * _log_sigmoid(v if t == 1 else -v) for t, v in zip(y, z)) / len(y) * 1.0001 + 0.01, gradient=[0.0]),
     rand=_rand_focal, phrases=["Focal loss", "Binary focal loss"],
     conditions=["labels 0 or 1", "gamma >= 0", "0 <= alpha <= 1"],
     examples=[_ex({"y": [1, 0, 1], "z": [2, -1, -0.5], "gamma": 2, "alpha": 0.25}), _ex({"y": [0, 1], "z": [1.5, 0.3], "gamma": 0, "alpha": 0.5})],
     extra_q=[_bad({"y": [1, 0], "z": [2, -1], "gamma": -1, "alpha": 0.25})],
     weak="the loss value is checked against the torchvision sigmoid_focal_loss formula re-implemented in PyTorch (an independent implementation of the same definition); the gradient by PyTorch autograd",
     level="expert")


# ================================================================================================ gradients of models
def _linear_parts(X, y, w, b):
    M = _mat("X", X)
    yv, wv = _vec("y", y), _vec("w", w)
    _need(len(yv) == len(M), "y needs one value per row of X")
    _need(len(wv) == len(M[0]), "w needs one weight per column of X")
    _fin("b", b)
    return M, yv, wv


def _lin_loss(M, yv, wv, b):
    Mf = [[_F(v) for v in r] for r in M]
    Wf, Bf, Yf = [_F(v) for v in wv], _F(b), [_F(v) for v in yv]
    res = [sum(a * c for a, c in zip(r, Wf)) + Bf - t for r, t in zip(Mf, Yf)]
    n = len(res)
    loss = sum(v * v for v in res) / n
    gw = [2 * sum(res[i] * Mf[i][j] for i in range(n)) / n for j in range(len(Wf))]
    gb = 2 * sum(res) / n
    return loss, gw, gb


def mse_gradient(X, y, w, b):
    M, yv, wv = _linear_parts(X, y, w, b)
    loss, gw, gb = _lin_loss(M, yv, wv, b)
    return _out(loss=loss, grad_w=gw, grad_b=gb)


def _ref_mse_grad(X, y, w, b):
    import torch
    Xt, yt = torch.tensor(X, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    wt = torch.tensor(w, dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    loss = torch.nn.functional.mse_loss(Xt @ wt + bt, yt)
    loss.backward()
    return {"loss": loss.item(), "grad_w": wt.grad.tolist(), "grad_b": bt.grad.item()}


def _rand_linear(rng, binary=False):
    n, d = rng.randint(3, 10), rng.randint(1, 4)
    X = [[round(rng.uniform(-3, 3), 2) for _ in range(d)] for _ in range(n)]
    w = [round(rng.uniform(-2, 2), 2) for _ in range(d)]
    b = round(rng.uniform(-1, 1), 2)
    if binary:
        y = [rng.randint(0, 1) for _ in range(n)]
    else:
        y = [round(rng.uniform(-5, 5), 2) for _ in range(n)]
    return {"X": X, "y": y, "w": w, "b": b}


_LIN_IN = [_matin("X", "feature matrix: one row per sample"), _vecin("y", "targets, one per row"),
           _vecin("w", "weights, one per column of X"), _numin("b", "bias (intercept)")]

_add("ml_mse_gradient",
     name="Mean squared error loss and gradient of a linear model", topic="gradients",
     purpose="The mean squared error of a linear model y_hat = X w + b and its gradient with respect to the weights and the bias.",
     inputs=_LIN_IN,
     outputs=[_o("loss", "mean (X w + b - y)^2"), _o("grad_w", "dL/dw = (2/n) X^T (X w + b - y)"), _o("grad_b", "dL/db = (2/n) sum of residuals")],
     primary="loss", method="exact rational arithmetic (the loss and gradient are polynomials in the inputs)",
     latex=r"L = \frac{1}{n}\|Xw + b - y\|^2,\quad \nabla_w L = \frac{2}{n}X^\top(Xw + b - y)",
     run=mse_gradient, reference=_ref_mse_grad,
     wrong=lambda X, y, w, b: (lambda o: dict(o, grad_w=[v / 2 for v in o["grad_w"]], loss=o["loss"] / 2))(mse_gradient(X, y, w, b)),
     rand=_rand_linear, phrases=["MSE gradient of a linear model", "Linear model MSE gradient"],
     conditions=["dimensions agree"],
     examples=[_ex({"X": [[1, 2], [3, 4], [5, 6]], "y": [1, 2, 3], "w": [0.1, 0.2], "b": 0}),
               _ex({"X": [[1], [2], [3], [4]], "y": [2, 4, 6, 8], "w": [1], "b": 1}, {"loss": 3.5, "grad_b": -3})],
     extra_q=[_bad({"X": [[1, 2], [3, 4]], "y": [1, 2], "w": [0.1], "b": 0})],
     level="intermediate")


def logistic_gradient(X, y, w, b, lam):
    M, yv, wv = _linear_parts(X, y, w, b)
    _need(all(v in (0.0, 1.0) for v in yv), "labels y must be 0 or 1")
    _fin("lam", lam)
    _need(lam >= 0, "the L2 strength lam must be >= 0")
    n = len(yv)
    z = [math.fsum(a * c for a, c in zip(r, wv)) + b for r in M]
    loss = math.fsum(max(v, 0.0) - v * t + math.log1p(math.exp(-abs(v))) for v, t in zip(z, yv)) / n + lam / 2 * math.fsum(c * c for c in wv)
    res = [_sigmoid(v) - t for v, t in zip(z, yv)]
    gw = [math.fsum(res[i] * M[i][j] for i in range(n)) / n + lam * wv[j] for j in range(len(wv))]
    return _out(loss=loss, grad_w=gw, grad_b=math.fsum(res) / n)


def _ref_logistic_grad(X, y, w, b, lam):
    import torch
    Xt, yt = torch.tensor(X, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    wt = torch.tensor(w, dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ wt + bt, yt) + lam / 2 * (wt * wt).sum()
    loss.backward()
    return {"loss": loss.item(), "grad_w": wt.grad.tolist(), "grad_b": bt.grad.item()}


_add("ml_logistic_loss_gradient",
     name="Logistic regression loss and gradient (L2-regularised)", topic="gradients",
     purpose="The mean logistic (binary cross-entropy) loss of a linear model with an L2 penalty (lam / 2) ||w||^2 and its gradient with respect to the weights and the bias.",
     inputs=[_matin("X", "feature matrix"), _vecin("y", "labels (0 or 1)"), _vecin("w", "weights"), _numin("b", "bias"),
             _numin("lam", "L2 regularisation strength (>= 0; the bias is not penalised)")],
     outputs=[_o("loss", "mean log loss + (lam/2)||w||^2"), _o("grad_w", "X^T (sigma(Xw + b) - y) / n + lam w"), _o("grad_b", "mean of sigma(Xw + b) - y")],
     primary="loss", method="stable log-sum-exp form of the log loss; analytic gradient",
     latex=r"L = \frac{1}{n}\sum_i \left[\log(1 + e^{z_i}) - y_i z_i\right] + \frac{\lambda}{2}\|w\|^2,\quad \nabla_w L = \frac{1}{n}X^\top(\sigma(z) - y) + \lambda w",
     run=logistic_gradient, reference=_ref_logistic_grad,
     wrong=lambda X, y, w, b, lam: (lambda o: dict(o, grad_w=[g - lam * c + 0.5 * lam * c + 0.01 for g, c in zip(o["grad_w"], w)], loss=o["loss"] + 0.01))(logistic_gradient(X, y, w, b, lam)),
     rand=lambda rng: dict(_rand_linear(rng, True), lam=rng.choice([0.0, 0.1, 1.0, round(rng.uniform(0, 2), 2)])),
     phrases=["Logistic regression gradient", "Logistic loss gradient"],
     conditions=["labels 0 or 1", "lam >= 0"],
     examples=[_ex({"X": [[1, 2], [2, 1], [3, 3]], "y": [0, 1, 1], "w": [0.5, -0.25], "b": 0.1, "lam": 0}),
               _ex({"X": [[0.5], [1.5], [-1]], "y": [1, 1, 0], "w": [1], "b": 0, "lam": 0.1})],
     extra_q=[_bad({"X": [[1, 2], [2, 1]], "y": [0, 2], "w": [0.5, -0.25], "b": 0.1, "lam": 0})],
     level="advanced")


def gd_step_linear(X, y, w, b, lr):
    M, yv, wv = _linear_parts(X, y, w, b)
    _fin("lr", lr)
    _need(lr > 0, "the learning rate must be positive")
    loss, gw, gb = _lin_loss(M, yv, wv, b)
    L = _F(lr)
    w2 = [_F(c) - L * g for c, g in zip(wv, gw)]
    b2 = _F(b) - L * gb
    Mf, Yf = [[_F(v) for v in r] for r in M], [_F(v) for v in yv]
    res = [sum(a * c for a, c in zip(r, w2)) + b2 - t for r, t in zip(Mf, Yf)]
    return _out(w_new=w2, b_new=b2, loss_before=loss, loss_after=sum(v * v for v in res) / len(res))


def _ref_gd_step(X, y, w, b, lr):
    import torch
    Xt, yt = torch.tensor(X, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    wt = torch.tensor(w, dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([wt, bt], lr=lr)
    opt.zero_grad()
    loss = torch.nn.functional.mse_loss(Xt @ wt + bt, yt)
    loss.backward()
    opt.step()
    with torch.no_grad():
        after = torch.nn.functional.mse_loss(Xt @ wt + bt, yt).item()
    return {"w_new": wt.detach().tolist(), "b_new": bt.item(), "loss_before": loss.item(), "loss_after": after}


_add("ml_gradient_descent_step_linear",
     name="One gradient-descent step for linear regression", topic="gradients",
     purpose="One full-batch gradient-descent step on the mean squared error of a linear model: the new weights and bias and the loss before and after the step.",
     inputs=_LIN_IN + [_numin("lr", "learning rate (> 0)")],
     outputs=[_o("w_new", "w - lr dL/dw"), _o("b_new", "b - lr dL/db"), _o("loss_before", "MSE before the step"), _o("loss_after", "MSE after the step")],
     primary="loss_after", method="exact rational arithmetic",
     latex=r"w \leftarrow w - \eta \nabla_w L,\quad b \leftarrow b - \eta \frac{\partial L}{\partial b}",
     run=gd_step_linear, reference=_ref_gd_step,
     wrong=lambda X, y, w, b, lr: gd_step_linear(X, y, w, b, lr / 2),
     rand=lambda rng: dict(_rand_linear(rng), lr=rng.choice([0.01, 0.05, 0.1, round(rng.uniform(0.001, 0.1), 3)])),
     phrases=["Gradient descent step for linear regression", "Linear regression gradient descent step"],
     conditions=["lr > 0", "dimensions agree"],
     examples=[_ex({"X": [[1], [2], [3], [4]], "y": [2, 4, 6, 8], "w": [1], "b": 1, "lr": 0.1}),
               _ex({"X": [[1, 2], [3, 4], [5, 6]], "y": [1, 2, 3], "w": [0.1, 0.2], "b": 0, "lr": 0.01})],
     level="intermediate")


def distances(x, z):
    a, b = _vec("x", x), _vec("z", z)
    _same_len(a, b, "x", "z")
    na = math.sqrt(math.fsum(v * v for v in a))
    nb = math.sqrt(math.fsum(v * v for v in b))
    _need(na > 0 and nb > 0, "cosine similarity needs two non-zero vectors")
    cos = math.fsum(u * v for u, v in zip(a, b)) / (na * nb)
    return _out(euclidean=math.sqrt(math.fsum((u - v) ** 2 for u, v in zip(a, b))),
                manhattan=math.fsum(abs(u - v) for u, v in zip(a, b)), chebyshev=max(abs(u - v) for u, v in zip(a, b)),
                cosine_similarity=cos, cosine_distance=1 - cos)


def _ref_distances(x, z):
    from scipy.spatial import distance as d
    c = float(d.cosine(x, z))
    return {"euclidean": float(d.euclidean(x, z)), "manhattan": float(d.cityblock(x, z)), "chebyshev": float(d.chebyshev(x, z)),
            "cosine_similarity": 1 - c, "cosine_distance": c}


_add("ml_distance_metrics",
     name="Distances and cosine similarity between two vectors", topic="classical algorithms",
     purpose="The Euclidean, Manhattan and Chebyshev distances between two feature vectors, and their cosine similarity and cosine distance.",
     inputs=[_vecin("x", "first vector"), _vecin("z", "second vector")],
     outputs=[_o("euclidean", "sqrt(sum (x - z)^2)"), _o("manhattan", "sum |x - z|"), _o("chebyshev", "max |x - z|"),
              _o("cosine_similarity", "x.z / (|x| |z|)"), _o("cosine_distance", "1 - cosine similarity")],
     primary="euclidean", method="direct sums with fsum",
     latex=r"d_2 = \sqrt{\sum_i (x_i - z_i)^2},\quad \cos\theta = \frac{x \cdot z}{\|x\|\|z\|}",
     run=distances, reference=_ref_distances,
     wrong=lambda x, z: (lambda o: dict(o, euclidean=_r(o["euclidean"] ** 2 + 0.5)))(distances(x, z)),
     rand=lambda rng: (lambda d: {"x": [round(rng.uniform(-5, 5), 2) for _ in range(d)], "z": [round(rng.uniform(-5, 5), 2) for _ in range(d)]})(rng.randint(2, 8)),
     phrases=["Distance metrics", "Vector distances"],
     conditions=["equal lengths", "non-zero vectors"],
     examples=[_ex({"x": [1, 2, 3], "z": [4, 6, 3]}, {"euclidean": 5.0, "manhattan": 7.0, "chebyshev": 4.0}),
               _ex({"x": [1, 0], "z": [1, 1]}, {"cosine_similarity": 0.707106781187})],
     extra_q=[_bad({"x": [0, 0], "z": [1, 1]})],
     level="beginner")


def rbf_kernel(X, gamma):
    M = _mat("X", X, 2)
    _fin("gamma", gamma)
    _need(gamma > 0, "gamma must be positive")
    n = len(M)
    K = [[math.exp(-gamma * math.fsum((a - b) ** 2 for a, b in zip(M[i], M[j]))) for j in range(n)] for i in range(n)]
    off = [K[i][j] for i in range(n) for j in range(n) if i != j]
    return _out(K=K, mean_offdiagonal=math.fsum(off) / len(off))


def _ref_rbf(X, gamma):
    from sklearn.metrics.pairwise import rbf_kernel as rk
    np = _np()
    K = rk(np.array(X), gamma=gamma)
    n = len(X)
    return {"K": _f(K), "mean_offdiagonal": float((K.sum() - np.trace(K)) / (n * n - n))}


_add("ml_rbf_kernel_matrix",
     name="RBF (Gaussian) kernel matrix", topic="classical algorithms",
     purpose="The Gram matrix of the radial basis function kernel exp(-gamma ||x_i - x_j||^2) for a set of points, and the mean off-diagonal similarity.",
     inputs=[_matin("X", "points: one row per point (at least two)"), _numin("gamma", "kernel width parameter (> 0)")],
     outputs=[_o("K", "kernel matrix K_ij = exp(-gamma ||x_i - x_j||^2)"), _o("mean_offdiagonal", "mean of the off-diagonal entries")],
     primary="mean_offdiagonal", method="squared distances with fsum, then exp",
     latex=r"K_{ij} = \exp\left(-\gamma\,\|x_i - x_j\|^2\right)",
     run=rbf_kernel, reference=_ref_rbf,
     wrong=lambda X, gamma: rbf_kernel([[v / math.sqrt(2) for v in r] for r in X], gamma),
     rand=lambda rng: (lambda n, d: {"X": [[round(rng.uniform(-2, 2), 2) for _ in range(d)] for _ in range(n)], "gamma": rng.choice([0.5, 1.0, 0.1, round(rng.uniform(0.05, 2), 3)])})(rng.randint(2, 6), rng.randint(1, 4)),
     phrases=["RBF kernel matrix", "Gaussian kernel matrix"],
     conditions=["gamma > 0", "at least two points"],
     examples=[_ex({"X": [[0, 0], [1, 0], [0, 2]], "gamma": 0.5}), _ex({"X": [[1], [2], [4]], "gamma": 1})],
     extra_q=[_bad({"X": [[0, 0], [1, 0]], "gamma": -1})],
     level="advanced")


# ================================================================================================ information theory
def _H(probs):
    """Entropy in nats of a probability list (0 log 0 = 0)."""
    return -math.fsum(p * math.log(p) for p in probs if p > 0)


def _Hc(counts):
    n = sum(counts)
    return _H([c / n for c in counts if c > 0])


def entropy(p, unit):
    P = _vec("p", p, 1)
    _need(all(v >= 0 for v in P), "probabilities must be >= 0")
    _need(abs(math.fsum(P) - 1) <= 1e-9, "the probabilities must sum to 1")
    h = _H(P)
    k = len(P)
    L = _LN(unit)
    return _out(entropy=h / L, perplexity=math.exp(h), max_entropy=math.log(k) / L,
                normalized_entropy=h / math.log(k) if k > 1 else 0.0)


def _ref_entropy(p, unit):
    from scipy.stats import entropy as ent
    base = {"nats": None, "bits": 2, "bans": 10}[unit]
    h = float(ent(p))
    k = len(p)
    return {"entropy": float(ent(p, base=base)), "perplexity": math.exp(h),
            "max_entropy": float(ent([1.0 / k] * k, base=base)), "normalized_entropy": h / float(ent([1.0 / k] * k)) if k > 1 else 0.0}


_add("ml_entropy",
     name="Shannon entropy and perplexity of a distribution", topic="information theory",
     purpose="The Shannon entropy of a discrete probability distribution in the stated unit, its perplexity (e^H in nats = 2^H in bits), the maximum entropy log K and the normalised entropy H / log K.",
     inputs=[_vecin("p", "probabilities (>= 0, sum to 1)"), _UNIT()],
     outputs=[_o("entropy", "H(p) in the stated unit"), _o("perplexity", "exp(H in nats)"), _o("max_entropy", "log K in the stated unit"),
              _o("normalized_entropy", "H / log K (0 for K = 1)")],
     primary="entropy", method="-sum p log p with 0 log 0 = 0 (fsum)",
     latex=r"H(p) = -\sum_i p_i \log p_i",
     run=entropy, reference=_ref_entropy,
     wrong=lambda p, unit: (lambda o: dict(o, entropy=_r(-math.fsum(v * math.log(v) for v in p if v > 0) / (_LN(unit) * 2 if unit != "nats" else 0.5))))(entropy(p, unit)),
     rand=lambda rng: {"p": _rand_dist(rng, rng.randint(2, 8), zero_ok=True), "unit": rng.choice(["nats", "bits", "bans"])},
     phrases=["Entropy of a distribution", "Shannon entropy"],
     conditions=["probabilities >= 0 summing to 1"],
     examples=[_ex({"p": [0.5, 0.25, 0.25], "unit": "bits"}, {"entropy": 1.5, "perplexity": 2.82842712475}),
               _ex({"p": [0.9, 0.1], "unit": "nats"}), _ex({"p": [0.25, 0.25, 0.25, 0.25], "unit": "bits"}, {"entropy": 2.0, "normalized_entropy": 1.0})],
     extra_q=[_bad({"p": [0.5, 0.6], "unit": "bits"})],
     level="beginner")


def joint_entropy(P, unit):
    M = _mat("P", P, 1, 1)
    _need(all(v >= 0 for r in M for v in r), "joint probabilities must be >= 0")
    _need(abs(math.fsum(v for r in M for v in r) - 1) <= 1e-9, "the joint probabilities must sum to 1")
    px = [math.fsum(r) for r in M]
    py = [math.fsum(M[i][j] for i in range(len(M))) for j in range(len(M[0]))]
    L = _LN(unit)
    hxy = _H([v for r in M for v in r])
    hx, hy = _H(px), _H(py)
    mi = math.fsum(M[i][j] * math.log(M[i][j] / (px[i] * py[j])) for i in range(len(M)) for j in range(len(M[0])) if M[i][j] > 0)
    return _out(H_XY=hxy / L, H_X=hx / L, H_Y=hy / L, H_Y_given_X=(hxy - hx) / L, H_X_given_Y=(hxy - hy) / L, mutual_information=mi / L)


def _ref_joint(P, unit):
    np = _np()
    from scipy.stats import entropy as ent
    base = {"nats": None, "bits": 2, "bans": 10}[unit]
    A = np.array(P, dtype=float)
    px, py = A.sum(1), A.sum(0)
    hxy, hx, hy = ent(A.ravel(), base=base), ent(px, base=base), ent(py, base=base)
    mi = ent(A.ravel(), np.outer(px, py).ravel(), base=base)      # KL(joint || product of marginals)
    return {"H_XY": float(hxy), "H_X": float(hx), "H_Y": float(hy), "H_Y_given_X": float(hxy - hx), "H_X_given_Y": float(hxy - hy),
            "mutual_information": float(mi)}


def _rand_joint(rng):
    r, c = rng.randint(2, 4), rng.randint(2, 4)
    d = _rand_dist(rng, r * c, 3)
    if rng.random() < 0.4:                       # a structural zero in the table
        d = _rand_dist(rng, r * c - 1, 3)
        d.insert(rng.randrange(r * c), 0.0)
    return {"P": [d[i * c:(i + 1) * c] for i in range(r)], "unit": rng.choice(["nats", "bits", "bans"])}


_add("ml_joint_conditional_entropy",
     name="Joint, marginal and conditional entropy and mutual information", topic="information theory",
     purpose="From a joint probability table of X (rows) and Y (columns): the joint entropy, both marginal entropies, both conditional entropies and the mutual information.",
     inputs=[_matin("P", "joint probabilities P(X = i, Y = j): rows are values of X, columns values of Y; sums to 1"), _UNIT()],
     outputs=[_o("H_XY", "joint entropy"), _o("H_X", "entropy of X"), _o("H_Y", "entropy of Y"), _o("H_Y_given_X", "H(Y | X) = H(X, Y) - H(X)"),
              _o("H_X_given_Y", "H(X | Y)"), _o("mutual_information", "I(X; Y) = sum P log P / (P_X P_Y)")],
     primary="mutual_information", method="entropies of the table and its marginals; mutual information summed directly",
     latex=r"H(Y \mid X) = H(X, Y) - H(X),\quad I(X;Y) = \sum_{ij} P_{ij}\log\frac{P_{ij}}{P_i P_j}",
     run=joint_entropy, reference=_ref_joint,
     wrong=lambda P, unit: (lambda o: dict(o, mutual_information=_r(o["H_X"] + o["H_Y"] + o["H_XY"]), H_Y_given_X=o["H_Y"]))(joint_entropy(P, unit)),
     rand=_rand_joint, phrases=["Joint and conditional entropy", "Conditional entropy"],
     conditions=["probabilities >= 0 summing to 1"],
     examples=[_ex({"P": [[0.25, 0.25], [0.5, 0]], "unit": "bits"}, {"H_XY": 1.5, "H_X": 1.0}),
               _ex({"P": [[0.1, 0.2, 0.1], [0.3, 0.1, 0.2]], "unit": "nats"})],
     extra_q=[_bad({"P": [[0.5, 0.5], [0.5, 0]], "unit": "bits"})],
     level="advanced")


def _contingency(a, b):
    ca, cb = sorted(set(a)), sorted(set(b))
    ia, ib = {v: i for i, v in enumerate(ca)}, {v: i for i, v in enumerate(cb)}
    M = [[0] * len(cb) for _ in ca]
    for u, v in zip(a, b):
        M[ia[u]][ib[v]] += 1
    return M


def _mi_counts(M, n):
    rows = [sum(r) for r in M]
    cols = [sum(M[i][j] for i in range(len(M))) for j in range(len(M[0]))]
    return math.fsum(M[i][j] / n * math.log(n * M[i][j] / (rows[i] * cols[j]))
                     for i in range(len(M)) for j in range(len(M[0])) if M[i][j] > 0)


def mutual_info_labels(x, y, unit):
    a, b = _labels("x", x), _labels("y", y)
    _same_len(a, b, "x", "y")
    n = len(a)
    M = _contingency(a, b)
    L = _LN(unit)
    hx = _Hc([a.count(v) for v in set(a)])
    hy = _Hc([b.count(v) for v in set(b)])
    mi = _mi_counts(M, n)
    return _out(mutual_information=mi / L, H_x=hx / L, H_y=hy / L, H_x_given_y=(hx - mi) / L)


def _ref_mi_labels(x, y, unit):
    np = _np()
    from scipy.stats import entropy as ent
    from sklearn.metrics import mutual_info_score
    L = _LN(unit)
    mi = float(mutual_info_score(x, y))
    hx = float(ent(np.unique(x, return_counts=True)[1]))
    hy = float(ent(np.unique(y, return_counts=True)[1]))
    return {"mutual_information": mi / L, "H_x": hx / L, "H_y": hy / L, "H_x_given_y": (hx - mi) / L}


def _rand_label_pair(rng, kmin=2, kmax=4, nmin=10, nmax=50, noise=None):
    ka, kb = rng.randint(kmin, kmax), rng.randint(kmin, kmax)
    n = rng.randint(nmin, nmax)
    noise = noise if noise is not None else rng.uniform(0.2, 0.7)
    while True:
        a = [rng.randrange(ka) for _ in range(n)]
        b = [(v % kb) if rng.random() > noise else rng.randrange(kb) for v in a]
        if len(set(a)) >= 2 and len(set(b)) >= 2:
            return a, b


_add("ml_mutual_information_labels",
     name="Mutual information between two label sequences", topic="information theory",
     purpose="The mutual information between two discrete variables observed together (two label lists), with both entropies and the conditional entropy H(x | y), from the empirical joint distribution.",
     inputs=[_vecin("x", "values of the first variable (whole numbers)"), _vecin("y", "values of the second variable (whole numbers)"), _UNIT()],
     outputs=[_o("mutual_information", "I(x; y)"), _o("H_x", "entropy of x"), _o("H_y", "entropy of y"), _o("H_x_given_y", "H(x | y) = H(x) - I")],
     primary="mutual_information", method="contingency table of counts; I = sum n_ij/n log(n n_ij / (a_i b_j))",
     latex=r"I(X;Y) = \sum_{ij} \frac{n_{ij}}{n}\log\frac{n\,n_{ij}}{a_i b_j}",
     run=mutual_info_labels, reference=_ref_mi_labels,
     wrong=lambda x, y, unit: (lambda o: dict(o, mutual_information=_r(o["H_x"] * o["H_y"])))(mutual_info_labels(x, y, unit)),
     rand=lambda rng: dict(zip(("x", "y"), _rand_label_pair(rng)), unit=rng.choice(["nats", "bits", "bans"])),
     phrases=["Mutual information of labels", "Mutual information between labels"],
     conditions=["labels are whole numbers", "equal lengths"],
     examples=[_ex({"x": [0, 0, 1, 1, 2, 2], "y": [0, 0, 1, 1, 1, 1], "unit": "bits"}, {"mutual_information": 0.918295834054}),
               _ex({"x": [1, 1, 2, 2, 3, 3, 3, 1], "y": [1, 2, 2, 2, 3, 3, 1, 1], "unit": "nats"})],
     level="intermediate")


def information_gain(y, x):
    yl, xl = _labels("y", y), _labels("x", x)
    _same_len(yl, xl, "y", "x")
    n = len(yl)
    hy = _Hc([yl.count(v) for v in set(yl)])
    cond = 0.0
    parts = []
    for v in sorted(set(xl)):
        sub = [a for a, b in zip(yl, xl) if b == v]
        parts.append(len(sub) / n * _Hc([sub.count(c) for c in set(sub)]))
    cond = math.fsum(parts)
    split = _Hc([xl.count(v) for v in set(xl)])
    _need(split > 0, "the gain ratio is undefined when the feature takes a single value")
    ln2 = math.log(2.0)
    ig = hy - cond
    return _out(parent_entropy=hy / ln2, conditional_entropy=cond / ln2, information_gain=ig / ln2, split_info=split / ln2,
                gain_ratio=ig / split)


def _ref_info_gain(y, x):
    np = _np()
    from scipy.stats import entropy as ent
    from sklearn.metrics import mutual_info_score
    ln2 = math.log(2.0)
    ig = float(mutual_info_score(x, y))
    hy = float(ent(np.unique(y, return_counts=True)[1]))
    split = float(ent(np.unique(x, return_counts=True)[1]))
    return {"parent_entropy": hy / ln2, "conditional_entropy": (hy - ig) / ln2, "information_gain": ig / ln2, "split_info": split / ln2,
            "gain_ratio": ig / split}


_add("ml_information_gain",
     name="Information gain and gain ratio of a split", topic="information theory",
     purpose="The information gain (in bits) of splitting class labels y by a discrete feature x, with the parent and conditional entropies, the split information and the gain ratio (C4.5).",
     inputs=[_vecin("y", "class labels (whole numbers)"), _vecin("x", "feature value of each sample (whole numbers; each value is one branch)")],
     outputs=[_o("parent_entropy", "H(y) in bits"), _o("conditional_entropy", "H(y | x) in bits"), _o("information_gain", "H(y) - H(y | x) in bits"),
              _o("split_info", "H(x) in bits"), _o("gain_ratio", "information gain / split information")],
     primary="information_gain", method="entropies of the branches weighted by their sizes",
     latex=r"IG = H(Y) - \sum_v \frac{|S_v|}{|S|} H(Y \mid X = v),\quad GR = \frac{IG}{H(X)}",
     run=information_gain, reference=_ref_info_gain,
     wrong=lambda y, x: (lambda o: dict(o, information_gain=_r(o["parent_entropy"] - o["conditional_entropy"] * 0.5 - 0.01)))(information_gain(y, x)),
     rand=lambda rng: dict(zip(("x", "y"), _rand_label_pair(rng))),
     phrases=["Information gain", "Information gain of a split"],
     conditions=["the feature takes at least two values"],
     examples=[_ex({"y": [1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0], "x": [0, 0, 1, 2, 2, 2, 1, 0, 0, 2, 0, 1, 1, 2]},
                   {"information_gain": 0.230638213404}),
               _ex({"y": [0, 0, 1, 1], "x": [0, 0, 1, 1]}, {"information_gain": 1.0, "gain_ratio": 1.0})],
     extra_q=[_bad({"y": [0, 1, 1], "x": [2, 2, 2]})],
     level="intermediate")


def gini_split(y, g):
    yl, gl = _labels("y", y), _labels("g", g)
    _same_len(yl, gl, "y", "g")
    _need(len(set(gl)) >= 2, "a split needs at least two groups")

    def gini(lab):
        n = len(lab)
        return 1 - sum(Fraction(lab.count(c), n) ** 2 for c in set(lab))
    n = len(yl)
    parent = gini(yl)
    child = sum(Fraction(gl.count(v), n) * gini([a for a, b in zip(yl, gl) if b == v]) for v in set(gl))
    return _out(gini_parent=parent, gini_children=child, gini_decrease=parent - child)


def _ref_gini_split(y, g):
    np = _np()
    yv, gv = np.array(y), np.array(g)

    def mismatch(lab):          # probability that two draws (with replacement) carry different labels
        return float(np.mean(lab[:, None] != lab[None, :]))
    parent = mismatch(yv)
    child = sum(np.mean(gv == v) * mismatch(yv[gv == v]) for v in np.unique(gv))
    return {"gini_parent": parent, "gini_children": float(child), "gini_decrease": parent - float(child)}


_add("ml_gini_split",
     name="Gini impurity decrease of a split", topic="information theory",
     purpose="The Gini impurity of a node, the size-weighted Gini impurity of its children after a split, and the decrease (the CART split criterion).",
     inputs=[_vecin("y", "class labels (whole numbers)"), _vecin("g", "child (group) each sample goes to (whole numbers)")],
     outputs=[_o("gini_parent", "Gini impurity before the split"), _o("gini_children", "weighted Gini impurity of the children"),
              _o("gini_decrease", "parent minus children")],
     primary="gini_decrease", method="exact fractions",
     latex=r"\Delta G = G(S) - \sum_v \frac{|S_v|}{|S|}G(S_v)",
     run=gini_split, reference=_ref_gini_split,
     wrong=lambda y, g: (lambda o: dict(o, gini_children=_r(sum(gini_split(y, g)["gini_children"] for _ in (0,)) * 0 + o["gini_parent"] * 0.75),
                                         gini_decrease=_r(o["gini_parent"] * 0.25)))(gini_split(y, g)),
     rand=lambda rng: dict(zip(("g", "y"), _rand_label_pair(rng, 2, 3))),
     phrases=["Gini split", "Gini impurity decrease"],
     conditions=["at least two groups"],
     examples=[_ex({"y": [0, 0, 0, 1, 1, 1, 1, 0], "g": [0, 0, 0, 0, 1, 1, 1, 1]}, {"gini_parent": 0.5, "gini_children": 0.375, "gini_decrease": 0.125}),
               _ex({"y": [1, 2, 2, 3, 3, 3], "g": [0, 0, 1, 1, 2, 2]})],
     weak="checked by a different computation of the same impurity (the probability that two draws with replacement differ), in numpy",
     level="intermediate")


# ================================================================================================ clustering comparison
def _single_check(a, b):
    _need(len(set(a)) >= 2 and len(set(b)) >= 2, "each labelling must have at least two clusters")


def nmi(labels_true, labels_pred, method):
    a, b = _labels("labels_true", labels_true), _labels("labels_pred", labels_pred)
    _same_len(a, b, "labels_true", "labels_pred")
    _single_check(a, b)
    n = len(a)
    mi = _mi_counts(_contingency(a, b), n)
    ha = _Hc([a.count(v) for v in set(a)])
    hb = _Hc([b.count(v) for v in set(b)])
    norm = {"arithmetic": (ha + hb) / 2, "geometric": math.sqrt(ha * hb), "min": min(ha, hb), "max": max(ha, hb)}[method]
    return _out(nmi=mi / norm, mutual_information=mi)


def _ref_nmi(labels_true, labels_pred, method):
    from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
    return {"nmi": float(normalized_mutual_info_score(labels_true, labels_pred, average_method=method)),
            "mutual_information": float(mutual_info_score(labels_true, labels_pred))}


_METHOD = lambda: _choice("method", ["arithmetic", "geometric", "min", "max"], "normaliser: arithmetic, geometric, min or max of the two entropies")

_add("ml_normalized_mutual_information",
     name="Normalised mutual information of two clusterings", topic="clustering metrics",
     purpose="The normalised mutual information between two clusterings (or a clustering and the true classes), with the stated normaliser of the two entropies, and the mutual information in nats.",
     inputs=[_vecin("labels_true", "reference labels (whole numbers)"), _vecin("labels_pred", "cluster labels (whole numbers)"), _METHOD()],
     outputs=[_o("nmi", "I / normaliser(H_true, H_pred)"), _o("mutual_information", "I in nats")],
     primary="nmi", method="contingency counts; mutual information and entropies in nats",
     latex=r"\mathrm{NMI} = \frac{I(U, V)}{\mathrm{mean}(H(U), H(V))}",
     run=nmi, reference=_ref_nmi,
     wrong=lambda labels_true, labels_pred, method: (lambda o: dict(o, nmi=_r(o["mutual_information"] / math.log(len(labels_true)))))(nmi(labels_true, labels_pred, method)),
     rand=lambda rng: dict(zip(("labels_true", "labels_pred"), _rand_label_pair(rng, 2, 5)), method=rng.choice(["arithmetic", "geometric", "min", "max"])),
     phrases=["Normalized mutual information", "Normalised mutual information", "NMI"],
     conditions=["each labelling has at least two clusters"],
     examples=[_ex({"labels_true": [0, 0, 1, 1, 2, 2], "labels_pred": [0, 0, 1, 1, 1, 1], "method": "arithmetic"}, {"nmi": 0.733680436651}),
               _ex({"labels_true": [0, 0, 0, 1, 1, 1], "labels_pred": [1, 1, 0, 0, 0, 0], "method": "geometric"})],
     extra_q=[_bad({"labels_true": [0, 0, 0], "labels_pred": [0, 1, 1], "method": "arithmetic"})],
     level="advanced")


def ami(labels_true, labels_pred, method):
    a, b = _labels("labels_true", labels_true), _labels("labels_pred", labels_pred)
    _same_len(a, b, "labels_true", "labels_pred")
    _single_check(a, b)
    N = len(a)
    M = _contingency(a, b)
    rows = [sum(r) for r in M]
    cols = [sum(M[i][j] for i in range(len(M))) for j in range(len(M[0]))]
    mi = _mi_counts(M, N)
    lg = math.lgamma
    terms = []
    for ai in rows:
        for bj in cols:
            base = lg(ai + 1) + lg(bj + 1) + lg(N - ai + 1) + lg(N - bj + 1) - lg(N + 1)
            for nij in range(max(1, ai + bj - N), min(ai, bj) + 1):
                logp = base - lg(nij + 1) - lg(ai - nij + 1) - lg(bj - nij + 1) - lg(N - ai - bj + nij + 1)
                terms.append(nij / N * math.log(N * nij / (ai * bj)) * math.exp(logp))
    emi = math.fsum(terms)
    ha, hb = _Hc(rows), _Hc(cols)
    norm = {"arithmetic": (ha + hb) / 2, "geometric": math.sqrt(ha * hb), "min": min(ha, hb), "max": max(ha, hb)}[method]
    _need(abs(norm - emi) > 1e-12, "the adjusted mutual information is undefined (normaliser equals the expected MI)")
    return _out(ami=(mi - emi) / (norm - emi), expected_mi=emi, mutual_information=mi)


def _ref_ami(labels_true, labels_pred, method):
    np = _np()
    from sklearn.metrics import adjusted_mutual_info_score, mutual_info_score
    from sklearn.metrics.cluster import contingency_matrix
    from sklearn.metrics.cluster._expected_mutual_info_fast import expected_mutual_information
    C = contingency_matrix(labels_true, labels_pred, sparse=True)
    emi = float(expected_mutual_information(C, len(labels_true)))
    return {"ami": float(adjusted_mutual_info_score(labels_true, labels_pred, average_method=method)), "expected_mi": emi,
            "mutual_information": float(mutual_info_score(labels_true, labels_pred))}


_add("ml_adjusted_mutual_information",
     name="Adjusted mutual information of two clusterings", topic="clustering metrics",
     purpose="The adjusted mutual information between two clusterings: the mutual information corrected for chance by its expected value under random labellings with the same cluster sizes (hypergeometric model).",
     inputs=[_vecin("labels_true", "reference labels (whole numbers)"), _vecin("labels_pred", "cluster labels (whole numbers)"), _METHOD()],
     outputs=[_o("ami", "(I - E[I]) / (normaliser - E[I])"), _o("expected_mi", "E[I] in nats"), _o("mutual_information", "I in nats")],
     primary="ami", method="expected mutual information summed over the hypergeometric distribution of each cell (log-gamma form)",
     latex=r"\mathrm{AMI} = \frac{I - E[I]}{\mathrm{mean}(H(U), H(V)) - E[I]}",
     run=ami, reference=_ref_ami,
     wrong=lambda labels_true, labels_pred, method: (lambda o: dict(o, ami=_r(nmi(labels_true, labels_pred, method)["nmi"])))(ami(labels_true, labels_pred, method)),
     rand=lambda rng: dict(zip(("labels_true", "labels_pred"), _rand_label_pair(rng, 2, 4, 8, 40)), method=rng.choice(["arithmetic", "geometric", "min", "max"])),
     phrases=["Adjusted mutual information", "AMI"],
     conditions=["each labelling has at least two clusters"],
     examples=[_ex({"labels_true": [0, 0, 1, 1, 2, 2], "labels_pred": [0, 0, 1, 1, 1, 1], "method": "arithmetic"}),
               _ex({"labels_true": [0, 0, 0, 1, 1, 1, 2, 2, 2], "labels_pred": [0, 0, 1, 1, 2, 2, 2, 0, 1], "method": "max"})],
     level="expert")


def _pairs(a, b):
    M = _contingency(a, b)
    n = len(a)
    c2 = lambda v: v * (v - 1) // 2
    sij = sum(c2(v) for r in M for v in r)
    sa = sum(c2(sum(r)) for r in M)
    sb = sum(c2(sum(M[i][j] for i in range(len(M)))) for j in range(len(M[0])))
    return sij, sa, sb, c2(n)


def rand_index(labels_true, labels_pred):
    a, b = _labels("labels_true", labels_true, 2), _labels("labels_pred", labels_pred, 2)
    _same_len(a, b, "labels_true", "labels_pred")
    sij, sa, sb, tot = _pairs(a, b)
    ri = Fraction(tot + 2 * sij - sa - sb, tot)
    exp = Fraction(sa * sb, tot)
    mx = Fraction(sa + sb, 2)
    _need(mx != exp, "the adjusted Rand index is undefined (both labellings trivial)")
    _need(sa > 0 and sb > 0, "the Fowlkes-Mallows index needs at least one pair inside a cluster in each labelling")
    return _out(rand_index=ri, adjusted_rand_index=(sij - exp) / (mx - exp), fowlkes_mallows=sij / math.sqrt(sa * sb))


def _ref_rand(labels_true, labels_pred):
    from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score, rand_score
    return {"rand_index": float(rand_score(labels_true, labels_pred)), "adjusted_rand_index": float(adjusted_rand_score(labels_true, labels_pred)),
            "fowlkes_mallows": float(fowlkes_mallows_score(labels_true, labels_pred))}


_add("ml_rand_index",
     name="Rand index, adjusted Rand index and Fowlkes-Mallows index", topic="clustering metrics",
     purpose="Pair-counting agreement between two clusterings: the Rand index, the adjusted Rand index (corrected for chance) and the Fowlkes-Mallows index.",
     inputs=[_vecin("labels_true", "reference labels (whole numbers)", 2), _vecin("labels_pred", "cluster labels (whole numbers)", 2)],
     outputs=[_o("rand_index", "fraction of sample pairs on which the two labellings agree"), _o("adjusted_rand_index", "ARI"),
              _o("fowlkes_mallows", "TP / sqrt((TP + FP)(TP + FN)) over pairs")],
     primary="adjusted_rand_index", method="pair counts from the contingency table, exact integers",
     latex=r"\mathrm{ARI} = \frac{\sum_{ij}\binom{n_{ij}}{2} - \frac{\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}}{\binom{n}{2}}}{\frac12\left[\sum_i\binom{a_i}{2} + \sum_j\binom{b_j}{2}\right] - \frac{\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}}{\binom{n}{2}}}",
     run=rand_index, reference=_ref_rand,
     wrong=lambda labels_true, labels_pred: (lambda o: dict(o, adjusted_rand_index=_r(2 * o["rand_index"] - 1)))(rand_index(labels_true, labels_pred)),
     rand=lambda rng: dict(zip(("labels_true", "labels_pred"), _rand_label_pair(rng, 2, 5))),
     phrases=["Adjusted Rand index", "Rand index"],
     conditions=["at least one within-cluster pair in each labelling"],
     examples=[_ex({"labels_true": [0, 0, 1, 1, 2, 2], "labels_pred": [0, 0, 1, 1, 1, 1]}, {"adjusted_rand_index": 0.444444444444}),
               _ex({"labels_true": [0, 0, 0, 1, 1, 1], "labels_pred": [0, 0, 1, 1, 2, 2]}, {"rand_index": 0.666666666667})],
     level="advanced")


def hcv(labels_true, labels_pred, beta):
    a, b = _labels("labels_true", labels_true), _labels("labels_pred", labels_pred)
    _same_len(a, b, "labels_true", "labels_pred")
    _single_check(a, b)
    _fin("beta", beta)
    _need(beta > 0, "beta must be positive")
    n = len(a)
    mi = _mi_counts(_contingency(a, b), n)
    hc = _Hc([a.count(v) for v in set(a)])
    hk = _Hc([b.count(v) for v in set(b)])
    h, c = mi / hc, mi / hk
    v = (1 + beta) * h * c / (beta * h + c) if h + c > 0 else 0.0
    return _out(homogeneity=h, completeness=c, v_measure=v)


def _ref_hcv(labels_true, labels_pred, beta):
    from sklearn.metrics import homogeneity_completeness_v_measure
    h, c, v = homogeneity_completeness_v_measure(labels_true, labels_pred, beta=beta)
    return {"homogeneity": float(h), "completeness": float(c), "v_measure": float(v)}


_add("ml_homogeneity_completeness_v_measure",
     name="Homogeneity, completeness and V-measure", topic="clustering metrics",
     purpose="Entropy-based clustering scores against true classes: homogeneity (each cluster holds one class), completeness (each class sits in one cluster) and their weighted harmonic mean, the V-measure.",
     inputs=[_vecin("labels_true", "true class labels (whole numbers)"), _vecin("labels_pred", "cluster labels (whole numbers)"),
             _numin("beta", "weight of completeness relative to homogeneity in the V-measure (1 = equal)")],
     outputs=[_o("homogeneity", "1 - H(C | K) / H(C) = I / H(C)"), _o("completeness", "I / H(K)"), _o("v_measure", "(1 + beta) h c / (beta h + c)")],
     primary="v_measure", method="mutual information and entropies of the contingency table",
     latex=r"h = 1 - \frac{H(C \mid K)}{H(C)},\quad c = 1 - \frac{H(K \mid C)}{H(K)},\quad V_\beta = \frac{(1+\beta)hc}{\beta h + c}",
     run=hcv, reference=_ref_hcv,
     wrong=lambda labels_true, labels_pred, beta: (lambda o: dict(o, homogeneity=o["completeness"], completeness=o["homogeneity"],
                                                                 v_measure=_r((o["homogeneity"] + o["completeness"]) / 2 + 0.01)))(hcv(labels_true, labels_pred, beta)),
     rand=lambda rng: dict(zip(("labels_true", "labels_pred"), _rand_label_pair(rng, 2, 5)), beta=rng.choice([1.0, 0.5, 2.0])),
     phrases=["V-measure", "Homogeneity and completeness"],
     conditions=["each labelling has at least two clusters", "beta > 0"],
     examples=[_ex({"labels_true": [0, 0, 1, 1, 2, 2], "labels_pred": [0, 0, 1, 1, 1, 1], "beta": 1}, {"completeness": 1.0}),
               _ex({"labels_true": [0, 0, 0, 1, 1, 1], "labels_pred": [0, 0, 1, 1, 2, 2], "beta": 1})],
     level="advanced")


# ================================================================================================ cluster validity
def _points(X, labels):
    M = _mat("X", X)
    lab = _labels("labels", labels)
    _need(len(lab) == len(M), "labels needs one label per row of X")
    return M, lab


def _sqd(a, b):
    return math.fsum((u - v) ** 2 for u, v in zip(a, b))


def silhouette(X, labels):
    M, lab = _points(X, labels)
    n = len(M)
    ks = sorted(set(lab))
    _need(2 <= len(ks) <= n - 1, "the silhouette needs between 2 and n - 1 clusters")
    D = [[math.sqrt(_sqd(M[i], M[j])) for j in range(n)] for i in range(n)]
    s = []
    for i in range(n):
        own = [j for j in range(n) if lab[j] == lab[i] and j != i]
        if not own:
            s.append(0.0)
            continue
        a = math.fsum(D[i][j] for j in own) / len(own)
        b = min(math.fsum(D[i][j] for j in range(n) if lab[j] == k) / lab.count(k) for k in ks if k != lab[i])
        v = (b - a) / max(a, b)
        s.append(0.0 if abs(v) < 1e-12 else v)            # a = b by symmetry: exactly 0, not rounding residue
    return _out(silhouette=math.fsum(s) / n, samples=s)


def _ref_silhouette(X, labels):
    from sklearn.metrics import silhouette_samples, silhouette_score
    return {"silhouette": float(silhouette_score(X, labels)), "samples": [0.0 if abs(v) < 1e-12 else v for v in _f(silhouette_samples(X, labels))]}


def _rand_blobs(rng, kmin=2, kmax=4, nmin=6, nmax=24, dmin=1, dmax=3, spread=None, sep=4.0):
    k = rng.randint(kmin, kmax)
    d = rng.randint(dmin, dmax)
    centers = [[rng.uniform(-sep, sep) for _ in range(d)] for _ in range(k)]
    n = rng.randint(max(nmin, k + 1), nmax)
    spread = spread or rng.uniform(0.5, 2.0)
    X, lab = [], []
    for i in range(n):
        c = i % k if i < k else rng.randrange(k)
        X.append([round(v + rng.gauss(0, spread), 2) for v in centers[c]])
        lab.append(c)
    return X, lab


def _rand_cluster_eval(rng):
    while True:
        X, lab = _rand_blobs(rng)
        if 2 <= len(set(lab)) <= len(X) - 1 and len(set(map(tuple, X))) == len(X):
            return {"X": X, "labels": lab}


_add("ml_silhouette_score",
     name="Silhouette score", topic="clustering metrics",
     purpose="The mean silhouette coefficient of a clustering (Euclidean distance) and the silhouette of each sample: (b - a) / max(a, b), a = mean distance within its cluster, b = mean distance to the nearest other cluster.",
     inputs=[_matin("X", "points: one row per sample"), _vecin("labels", "cluster label of each point (whole numbers)")],
     outputs=[_o("silhouette", "mean silhouette coefficient"), _o("samples", "silhouette of each sample (0 for a point alone in its cluster)")],
     primary="silhouette", method="all pairwise Euclidean distances, per-sample a and b",
     latex=r"s_i = \frac{b_i - a_i}{\max(a_i, b_i)}",
     run=silhouette, reference=_ref_silhouette,
     wrong=lambda X, labels: (lambda o: dict(o, silhouette=_r(sum(o["samples"]) / len(o["samples"]) * 0.5 + 0.25)))(silhouette(X, labels)),
     rand=_rand_cluster_eval, phrases=["Silhouette score", "Silhouette coefficient"],
     conditions=["2 <= number of clusters <= n - 1"],
     examples=[_ex({"X": [[1, 1], [1.5, 2], [3, 4], [5, 7], [3.5, 5], [4.5, 5]], "labels": [0, 0, 1, 1, 1, 1]}),
               _ex({"X": [[0], [1], [5], [6], [11]], "labels": [0, 0, 1, 1, 2]})],
     extra_q=[_bad({"X": [[0], [1], [5]], "labels": [0, 1, 2]})],
     level="intermediate")


def _centroids(M, lab):
    ks = sorted(set(lab))
    C = {}
    for k in ks:
        rows = [M[i] for i in range(len(M)) if lab[i] == k]
        C[k] = [math.fsum(r[j] for r in rows) / len(rows) for j in range(len(M[0]))]
    return ks, C


def davies_bouldin(X, labels):
    M, lab = _points(X, labels)
    ks, C = _centroids(M, lab)
    _need(2 <= len(ks) <= len(M) - 1, "the Davies-Bouldin index needs between 2 and n - 1 clusters")
    S = {k: math.fsum(math.sqrt(_sqd(M[i], C[k])) for i in range(len(M)) if lab[i] == k) / lab.count(k) for k in ks}
    worst = []
    for k in ks:
        r = []
        for m in ks:
            if m != k:
                d = math.sqrt(_sqd(C[k], C[m]))
                _need(d > 0, "two clusters have the same centroid")
                r.append((S[k] + S[m]) / d)
        worst.append(max(r))
    return _out(davies_bouldin=math.fsum(worst) / len(ks))


def _ref_db(X, labels):
    """scipy cdist (direct differences): scikit-learn's davies_bouldin_score agrees only to about 1e-8, because its
    Euclidean distances use the expansion |x|^2 - 2 x.y + |y|^2."""
    np = _np()
    from scipy.spatial.distance import cdist
    A, lab = np.array(X, dtype=float), np.array(labels)
    ks = np.unique(lab)
    C = np.array([A[lab == k].mean(0) for k in ks])
    S = np.array([cdist(A[lab == k], C[i:i + 1]).mean() for i, k in enumerate(ks)])
    D = cdist(C, C)
    R = (S[:, None] + S[None, :]) / np.where(D > 0, D, np.inf)
    np.fill_diagonal(R, -np.inf)
    return {"davies_bouldin": float(R.max(1).mean())}


_add("ml_davies_bouldin_index",
     name="Davies-Bouldin index", topic="clustering metrics",
     purpose="The Davies-Bouldin index of a clustering: the mean over clusters of the worst ratio of within-cluster scatter to between-centroid distance (lower is better).",
     inputs=[_matin("X", "points: one row per sample"), _vecin("labels", "cluster label of each point (whole numbers)")],
     outputs=[_o("davies_bouldin", "Davies-Bouldin index")],
     primary="davies_bouldin", method="centroids, mean distance to the centroid as scatter, Euclidean centroid distances",
     latex=r"DB = \frac{1}{K}\sum_k \max_{m \ne k}\frac{S_k + S_m}{\|c_k - c_m\|}",
     run=davies_bouldin, reference=_ref_db,
     wrong=lambda X, labels: _out(davies_bouldin=davies_bouldin(X, labels)["davies_bouldin"] * 0.5 + 0.01),
     rand=_rand_cluster_eval, phrases=["Davies-Bouldin index", "Davies Bouldin index"],
     conditions=["2 <= number of clusters <= n - 1", "distinct centroids"],
     examples=[_ex({"X": [[1, 1], [1.5, 2], [3, 4], [5, 7], [3.5, 5], [4.5, 5]], "labels": [0, 0, 1, 1, 1, 1]}),
               _ex({"X": [[0], [1], [5], [6], [11]], "labels": [0, 0, 1, 1, 2]})],
     weak="checked by a scipy (cdist) computation of the same definition; scikit-learn's davies_bouldin_score agrees only to about 1e-8 because of its dot-product distance formula",
     level="advanced")


def calinski_harabasz(X, labels):
    M, lab = _points(X, labels)
    n = len(M)
    ks = sorted(set(lab))
    k = len(ks)
    _need(2 <= k <= n - 1, "the Calinski-Harabasz index needs between 2 and n - 1 clusters")
    F = [[_F(v) for v in r] for r in M]
    d = len(F[0])
    mean = [sum(r[j] for r in F) / n for j in range(d)]
    B = W = Fraction(0)
    for c in ks:
        rows = [F[i] for i in range(n) if lab[i] == c]
        cen = [sum(r[j] for r in rows) / len(rows) for j in range(d)]
        B += len(rows) * sum((cen[j] - mean[j]) ** 2 for j in range(d))
        W += sum(sum((r[j] - cen[j]) ** 2 for j in range(d)) for r in rows)
    _need(W > 0, "the within-cluster dispersion is 0")
    return _out(calinski_harabasz=(B / W) * Fraction(n - k, k - 1), between=B, within=W)


def _ref_ch(X, labels):
    np = _np()
    from sklearn.metrics import calinski_harabasz_score
    A = np.array(X, dtype=float)
    lab = np.array(labels)
    mean = A.mean(0)
    B = sum((lab == c).sum() * np.sum((A[lab == c].mean(0) - mean) ** 2) for c in np.unique(lab))
    W = sum(np.sum((A[lab == c] - A[lab == c].mean(0)) ** 2) for c in np.unique(lab))
    return {"calinski_harabasz": float(calinski_harabasz_score(X, labels)), "between": float(B), "within": float(W)}


_add("ml_calinski_harabasz_index",
     name="Calinski-Harabasz index", topic="clustering metrics",
     purpose="The Calinski-Harabasz (variance ratio) index of a clustering: between-cluster over within-cluster dispersion, scaled by (n - K)/(K - 1) (higher is better).",
     inputs=[_matin("X", "points: one row per sample"), _vecin("labels", "cluster label of each point (whole numbers)")],
     outputs=[_o("calinski_harabasz", "CH index"), _o("between", "between-cluster dispersion B"), _o("within", "within-cluster dispersion W")],
     primary="calinski_harabasz", method="exact rational sums of squares",
     latex=r"CH = \frac{\mathrm{tr}(B)}{\mathrm{tr}(W)}\cdot\frac{n - K}{K - 1}",
     run=calinski_harabasz, reference=_ref_ch,
     wrong=lambda X, labels: (lambda o: dict(o, calinski_harabasz=_r(o["between"] / o["within"] * (len(X) - 1) / len(set(labels)))))(calinski_harabasz(X, labels)),
     rand=_rand_cluster_eval, phrases=["Calinski-Harabasz index", "Variance ratio criterion"],
     conditions=["2 <= number of clusters <= n - 1"],
     examples=[_ex({"X": [[1, 1], [1.5, 2], [3, 4], [5, 7], [3.5, 5], [4.5, 5]], "labels": [0, 0, 1, 1, 1, 1]}),
               _ex({"X": [[0], [1], [5], [6], [11]], "labels": [0, 0, 1, 1, 2]})],
     level="advanced")


# ================================================================================================ k-means
def _sqd_exact(a, b):
    return sum((u - v) ** 2 for u, v in zip(a, b))


def _assign(F, C):
    """Nearest centroid of every point (exact squared distances); an exact tie is refused."""
    lab, tot = [], Fraction(0)
    for x in F:
        d = [_sqd_exact(x, c) for c in C]
        m = min(d)
        _need(d.count(m) == 1, "a point is exactly equidistant from two centroids (the assignment is ambiguous)")
        lab.append(d.index(m))
        tot += m
    return lab, tot


def _update(F, lab, k):
    C = []
    for j in range(k):
        rows = [F[i] for i in range(len(F)) if lab[i] == j]
        _need(rows, "a cluster became empty (no point is nearest to one of the centroids)")
        C.append([sum(r[t] for r in rows) / len(rows) for t in range(len(F[0]))])
    return C


def _km_inputs(X, C):
    M, C0 = _mat("X", X), _mat("C", C)
    _need(len(C0[0]) == len(M[0]), "centroids and points must have the same number of coordinates")
    _need(2 <= len(C0) <= len(M), "need at least 2 centroids and no more centroids than points")
    return [[_F(v) for v in r] for r in M], [[_F(v) for v in r] for r in C0]


def kmeans_step(X, C):
    F, C0 = _km_inputs(X, C)
    lab, inertia = _assign(F, C0)
    return _out(labels=lab, new_centroids=[[float(v) for v in r] for r in _update(F, lab, len(C0))], inertia=inertia)


def _ref_kmeans_step(X, C):
    import warnings
    np = _np()
    from scipy.cluster.vq import vq
    from sklearn.cluster import KMeans
    A, C0 = np.array(X, dtype=float), np.array(C, dtype=float)
    codes, dist = vq(A, C0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = KMeans(n_clusters=len(C0), init=C0, n_init=1, max_iter=1, algorithm="lloyd").fit(A)
    return {"labels": [int(c) for c in codes], "new_centroids": _f(km.cluster_centers_), "inertia": float(np.sum(dist ** 2))}


def _wrong_kmeans_step(X, C):
    F, C0 = _km_inputs(X, C)
    lab, inertia = _assign(F, C0)
    new = []
    for j in range(len(C0)):                       # the old centroid wrongly counted as a member
        rows = [F[i] for i in range(len(F)) if lab[i] == j] + [C0[j]]
        new.append([float(sum(r[t] for r in rows) / len(rows)) for t in range(len(F[0]))])
    return _out(labels=lab, new_centroids=new, inertia=inertia)


def _rand_kmeans(rng):
    X, lab = _rand_blobs(rng, 2, 4, 8, 30, 1, 3, sep=5.0)
    k = len(set(lab))
    idx = rng.sample(range(len(X)), k)
    C = [[round(v + rng.uniform(-0.5, 0.5), 2) for v in X[i]] for i in idx]
    return {"X": X, "C": C}


_KM_IN = [_matin("X", "points: one row per sample"), _matin("C", "current (or initial) centroids: one row per cluster")]

_add("ml_kmeans_step",
     name="One k-means iteration (assignment and update)", topic="classical algorithms",
     purpose="One Lloyd iteration of k-means from given centroids: assign every point to its nearest centroid, give the inertia of that assignment and the updated centroids (the means of the assigned points).",
     inputs=_KM_IN,
     outputs=[_o("labels", "index (0-based row of C) of the nearest centroid of each point"), _o("new_centroids", "mean of the points assigned to each centroid"),
              _o("inertia", "sum of squared distances of the points to their assigned (current) centroids")],
     primary="inertia", method="exact squared Euclidean distances; exact means",
     latex=r"c(i) = \arg\min_k \|x_i - \mu_k\|^2,\quad \mu_k' = \frac{1}{|S_k|}\sum_{i \in S_k} x_i",
     run=kmeans_step, reference=_ref_kmeans_step, wrong=_wrong_kmeans_step, rand=_rand_kmeans,
     phrases=["K-means step", "K-means iteration"],
     conditions=["no exact distance tie", "no empty cluster", "2 <= k <= n"],
     examples=[_ex({"X": [[1, 1], [1.5, 2], [3, 4], [5, 7], [3.5, 5], [4.5, 5], [3.5, 4.5]], "C": [[1, 1], [5, 6]]},
                   {"labels": [0, 0, 1, 1, 1, 1, 1]}),
               _ex({"X": [[0], [1], [5], [6], [11]], "C": [[0], [6]]})],
     extra_q=[_bad({"X": [[0], [2], [4]], "C": [[1], [3]]})],
     level="beginner")


def kmeans(X, C):
    F, cen = _km_inputs(X, C)
    k = len(cen)
    lab, _ = _assign(F, cen)
    for _ in range(100):
        cen = _update(F, lab, k)
        new, inertia = _assign(F, cen)
        if new == lab:
            return _out(centroids=[[float(v) for v in r] for r in cen], labels=lab, inertia=inertia)
        lab = new
    raise ValueError("k-means did not converge in 100 iterations")


def _ref_kmeans(X, C):
    np = _np()
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=len(C), init=np.array(C, dtype=float), n_init=1, max_iter=300, tol=0.0, algorithm="lloyd").fit(np.array(X, dtype=float))
    return {"centroids": _f(km.cluster_centers_), "labels": [int(v) for v in km.labels_], "inertia": float(km.inertia_)}


def _wrong_kmeans(X, C):
    o = kmeans_step(X, C)                      # stops after one iteration
    F = [[_F(v) for v in r] for r in X]
    cen = [[_F(v) for v in r] for r in o["new_centroids"]]
    lab, inertia = _assign(F, cen)
    return _out(centroids=o["new_centroids"], labels=lab, inertia=inertia * Fraction(101, 100))


_add("ml_kmeans",
     name="k-means clustering from given initial centroids", topic="classical algorithms",
     purpose="Lloyd's k-means algorithm run to convergence from stated initial centroids: the final centroids, the cluster of each point and the inertia (within-cluster sum of squares).",
     inputs=[_matin("X", "points: one row per sample"), _matin("C", "initial centroids: one row per cluster")],
     outputs=[_o("centroids", "final centroids"), _o("labels", "cluster (0-based row of C) of each point"), _o("inertia", "sum of squared distances to the final centroids")],
     primary="inertia", method="alternate exact assignment and mean update until no assignment changes (at most 100 iterations)",
     latex=r"\min \sum_k \sum_{i \in S_k} \|x_i - \mu_k\|^2",
     run=kmeans, reference=_ref_kmeans, wrong=_wrong_kmeans, rand=_rand_kmeans,
     phrases=["K-means clustering", "K-means"],
     conditions=["no exact distance tie", "no empty cluster during the iterations", "convergence within 100 iterations"],
     examples=[_ex({"X": [[1, 1], [1.5, 2], [3, 4], [5, 7], [3.5, 5], [4.5, 5], [3.5, 4.5]], "C": [[1, 1], [5, 6]]}),
               _ex({"X": [[0], [1], [5], [6], [11], [12]], "C": [[0], [1], [12]]})],
     level="intermediate")


# ================================================================================================ nearest neighbours
def _knn_core(X, y, x, k, metric):
    M = _mat("X", X)
    yv = _vec("y", y)
    q = _vec("x", x)
    _need(len(yv) == len(M), "y needs one value per row of X")
    _need(len(q) == len(M[0]), "the query x needs one coordinate per column of X")
    _need(isinstance(k, int) and 1 <= k <= len(M), "k must be between 1 and the number of training points")
    Q = [_F(v) for v in q]
    if metric == "euclidean":
        d = [sum((_F(a) - b) ** 2 for a, b in zip(r, Q)) for r in M]          # squared: same order
    else:
        d = [sum(abs(_F(a) - b) for a, b in zip(r, Q)) for r in M]
    order = sorted(range(len(M)), key=lambda i: (d[i], i))
    if k < len(M):
        _need(d[order[k - 1]] != d[order[k]], "the k-th and (k+1)-th nearest neighbours are at the same distance (the neighbour set is ambiguous)")
    nb = order[:k]
    dist = [math.sqrt(d[i]) if metric == "euclidean" else float(d[i]) for i in nb]
    return nb, dist, yv, d


def knn_classify(X, y, x, k, metric):
    nb, dist, yv, _ = _knn_core(X, y, x, k, metric)
    lab = [_whole("every label in y", yv[i]) for i in nb]
    _labels("y", yv)
    counts = {c: lab.count(c) for c in set(lab)}
    top = max(counts.values())
    winners = [c for c, v in counts.items() if v == top]
    _need(len(winners) == 1, "the vote among the k neighbours is tied")
    return _out(prediction=winners[0], votes=top, neighbors=sorted(nb), neighbor_distances=dist)


def _ref_knn_clf(X, y, x, k, metric):
    np = _np()
    from sklearn.neighbors import KNeighborsClassifier
    m = KNeighborsClassifier(n_neighbors=k, algorithm="brute", metric=metric).fit(np.array(X), np.array(y, dtype=int))
    d, idx = m.kneighbors(np.array([x]))
    pred = int(m.predict(np.array([x]))[0])
    lab = [int(y[i]) for i in idx[0]]
    return {"prediction": pred, "votes": lab.count(pred), "neighbors": sorted(int(i) for i in idx[0]), "neighbor_distances": sorted(_f(d[0]))}


def _rand_knn(rng, regression=False):
    X, lab = _rand_blobs(rng, 2, 3, 8, 25, 1, 3, sep=3.0)
    x = [round(rng.uniform(-3, 3), 2) for _ in X[0]]
    y = [round(rng.uniform(-10, 10), 2) for _ in lab] if regression else lab
    d = {"X": X, "y": y, "x": x, "k": rng.choice([1, 3, 5, 7]), "metric": rng.choice(["euclidean", "manhattan"])}
    if regression:
        d["weights"] = rng.choice(["uniform", "distance"])
    return d


_KNN_IN = [_matin("X", "training points: one row per sample"), _vecin("y", "training labels (whole numbers)"), _vecin("x", "query point"),
           _intin("k", "number of neighbours", 1), _choice("metric", ["euclidean", "manhattan"], "distance: euclidean or manhattan")]

_add("ml_knn_classify",
     name="k-nearest-neighbour classification of a query point", topic="classical algorithms",
     purpose="Classify a query point by majority vote among its k nearest training points (stated metric), with the neighbours and their distances.",
     inputs=_KNN_IN,
     outputs=[_o("prediction", "majority class among the k neighbours"), _o("votes", "number of neighbours of that class"),
              _o("neighbors", "indices (0-based rows of X) of the k nearest points, ascending"), _o("neighbor_distances", "their distances, ascending")],
     primary="votes", method="exact distances (squared for Euclidean), stable sort, majority vote",
     latex=r"\hat y = \arg\max_c \sum_{i \in N_k(x)} \mathbb{1}(y_i = c)",
     run=knn_classify, reference=_ref_knn_clf,
     wrong=lambda X, y, x, k, metric: knn_classify(X, y, x, k, "manhattan" if metric == "euclidean" else "euclidean") if True else None,
     rand=_rand_knn, phrases=["kNN classification", "k-nearest neighbour classification", "KNN classify"],
     conditions=["no distance tie at the k-th neighbour", "no tied vote", "1 <= k <= n"],
     examples=[_ex({"X": [[1, 1], [2, 1], [4, 3], [5, 4], [1, 2], [6, 5]], "y": [0, 0, 1, 1, 0, 1], "x": [3, 2.2], "k": 3, "metric": "euclidean"},
                   {"prediction": 0, "votes": 2}),
               _ex({"X": [[0], [1], [2], [6], [7], [8]], "y": [1, 1, 1, 2, 2, 2], "x": [5.5], "k": 5, "metric": "manhattan"})],
     extra_q=[_bad({"X": [[0], [2]], "y": [0, 1], "x": [1], "k": 1, "metric": "euclidean"})],
     level="beginner")


def knn_regress(X, y, x, k, metric, weights):
    nb, dist, yv, _ = _knn_core(X, y, x, k, metric)
    if weights == "uniform":
        pred = sum(_F(yv[i]) for i in nb) / k
    else:
        _need(all(v > 0 for v in dist), "distance weighting is undefined: the query coincides with a training point")
        w = [1 / v for v in dist]
        pred = math.fsum(wi * yv[i] for wi, i in zip(w, nb)) / math.fsum(w)
    return _out(prediction=pred, neighbors=sorted(nb), neighbor_distances=dist)


def _ref_knn_reg(X, y, x, k, metric, weights):
    np = _np()
    from sklearn.neighbors import KNeighborsRegressor
    m = KNeighborsRegressor(n_neighbors=k, algorithm="brute", metric=metric, weights=weights).fit(np.array(X), np.array(y))
    d, idx = m.kneighbors(np.array([x]))
    return {"prediction": float(m.predict(np.array([x]))[0]), "neighbors": sorted(int(i) for i in idx[0]), "neighbor_distances": sorted(_f(d[0]))}


_add("ml_knn_regress",
     name="k-nearest-neighbour regression of a query point", topic="classical algorithms",
     purpose="Predict a value at a query point as the mean (uniform) or inverse-distance-weighted mean of the targets of its k nearest training points.",
     inputs=[_matin("X", "training points"), _vecin("y", "training targets"), _vecin("x", "query point"), _intin("k", "number of neighbours", 1),
             _choice("metric", ["euclidean", "manhattan"], "distance: euclidean or manhattan"),
             _choice("weights", ["uniform", "distance"], "uniform mean or inverse-distance weights")],
     outputs=[_o("prediction", "predicted value"), _o("neighbors", "indices of the k nearest points, ascending"), _o("neighbor_distances", "their distances, ascending")],
     primary="prediction", method="exact distances, stable sort, (weighted) mean",
     latex=r"\hat y = \frac{\sum_{i \in N_k} w_i y_i}{\sum_{i \in N_k} w_i},\quad w_i = 1 \text{ or } 1/d_i",
     run=knn_regress, reference=_ref_knn_reg,
     wrong=lambda X, y, x, k, metric, weights: knn_regress(X, y, x, k, metric, "distance" if weights == "uniform" else "uniform") if k > 1 else _out(prediction=float(sum(y)) / len(y) + 0.37),
     rand=lambda rng: _rand_knn(rng, True), phrases=["kNN regression", "k-nearest neighbour regression"],
     conditions=["no distance tie at the k-th neighbour", "distance weights: the query is not a training point"],
     examples=[_ex({"X": [[1], [2], [3], [10]], "y": [1.5, 2.5, 3.0, 9.0], "x": [2.4], "k": 2, "metric": "euclidean", "weights": "uniform"}, {"prediction": 2.75}),
               _ex({"X": [[1, 1], [2, 1], [4, 3], [5, 4]], "y": [10, 12, 20, 24], "x": [3, 2], "k": 3, "metric": "manhattan", "weights": "distance"})],
     level="intermediate")


# ================================================================================================ naive Bayes
def _nb_data(X, y, x):
    M = _mat("X", X)
    yl = _labels("y", y)
    q = _vec("x", x)
    _need(len(yl) == len(M), "y needs one label per row of X")
    _need(len(q) == len(M[0]), "x needs one value per column of X")
    cls = sorted(set(yl))
    _need(len(cls) >= 2, "naive Bayes needs at least two classes")
    return M, yl, q, cls


def _posterior(cls, jll):
    m = max(jll)
    s = math.log(math.fsum(math.exp(v - m) for v in jll))
    post = [math.exp(v - m - s) for v in jll]
    best = max(range(len(cls)), key=lambda i: jll[i])
    _need(sum(1 for v in post if v >= post[best] * (1 - 1e-9)) == 1, "two classes are tied (equal posteriors to within rounding)")
    return _out(prediction=cls[best], classes=cls, posteriors=post)


def gaussian_nb(X, y, x):
    M, yl, q, cls = _nb_data(X, y, x)
    n = len(yl)
    jll = []
    for c in cls:
        rows = [M[i] for i in range(n) if yl[i] == c]
        lp = math.log(len(rows) / n)
        for j in range(len(q)):
            col = [_F(r[j]) for r in rows]
            mu = sum(col) / len(col)
            var = sum((v - mu) ** 2 for v in col) / len(col)
            _need(var > 0, "a feature has zero variance within a class (the Gaussian likelihood is undefined)")
            var = float(var)
            lp += -0.5 * math.log(2 * math.pi * var) - float((_F(q[j]) - mu) ** 2) / (2 * var)
        jll.append(lp)
    return _posterior(cls, jll)


def _ref_gnb(X, y, x):
    np = _np()
    from sklearn.naive_bayes import GaussianNB
    m = GaussianNB(var_smoothing=0.0).fit(np.array(X), np.array(y, dtype=int))
    return {"prediction": int(m.predict(np.array([x]))[0]), "classes": [int(c) for c in m.classes_], "posteriors": _f(m.predict_proba(np.array([x]))[0])}


def _wrong_gnb(X, y, x):
    M, yl, q, cls = _nb_data(X, y, x)
    jll = []
    for c in cls:
        rows = [M[i] for i in range(len(yl)) if yl[i] == c]
        lp = 0.0                                   # no class prior and sample variance (n - 1)
        for j in range(len(q)):
            col = [r[j] for r in rows]
            mu = sum(col) / len(col)
            var = sum((v - mu) ** 2 for v in col) / (len(col) - 1)
            lp += -0.5 * math.log(2 * math.pi * var) - (q[j] - mu) ** 2 / (2 * var)
        jll.append(lp)
    return _posterior(cls, jll)


def _rand_gnb(rng):
    while True:
        k = rng.randint(2, 3)
        d = rng.randint(1, 3)
        n = rng.randint(3 * k, 20)
        cen = [[rng.uniform(-3, 3) for _ in range(d)] for _ in range(k)]
        y = [i % k for i in range(n)]
        X = [[round(cen[c][j] + rng.gauss(0, 1.2), 2) for j in range(d)] for c in y]
        x = [round(rng.uniform(-3, 3), 2) for _ in range(d)]
        prop = {c: y.count(c) for c in range(k)}
        if min(prop.values()) >= 2:
            return {"X": X, "y": y, "x": x}


_add("ml_gaussian_naive_bayes",
     name="Gaussian naive Bayes prediction", topic="classical algorithms",
     purpose="Fit a Gaussian naive Bayes classifier (class frequencies as priors, per-class feature means and population variances) and give the predicted class and posterior probabilities of a query point.",
     inputs=[_matin("X", "training features: one row per sample"), _vecin("y", "training labels (whole numbers)"), _vecin("x", "query point")],
     outputs=[_o("prediction", "class with the largest posterior"), _o("classes", "classes in sorted order"), _o("posteriors", "posterior probability of each class")],
     primary="prediction", method="log prior + sum of Gaussian log densities (variance divides by n_c), normalised by log-sum-exp",
     latex=r"P(c \mid x) \propto P(c)\prod_j \mathcal{N}(x_j;\ \mu_{cj}, \sigma^2_{cj})",
     run=gaussian_nb, reference=_ref_gnb, wrong=_wrong_gnb, rand=_rand_gnb,
     phrases=["Gaussian naive Bayes"],
     conditions=["at least two classes", "positive within-class variance of every feature"],
     examples=[_ex({"X": [[6, 180, 12], [5.92, 190, 11], [5.58, 170, 12], [5.92, 165, 10], [5, 100, 6], [5.5, 150, 8], [5.42, 130, 7], [5.75, 150, 9]],
                    "y": [0, 0, 0, 0, 1, 1, 1, 1], "x": [6, 130, 8]}, {"prediction": 1}),
               _ex({"X": [[1.0], [1.2], [0.8], [3.0], [3.3], [2.9]], "y": [0, 0, 0, 1, 1, 1], "x": [2.0]})],
     extra_q=[_bad({"X": [[1], [1], [3], [4]], "y": [0, 0, 1, 1], "x": [2]})],
     level="advanced")


def multinomial_nb(X, y, x, alpha):
    M, yl, q, cls = _nb_data(X, y, x)
    _fin("alpha", alpha)
    _need(alpha > 0, "alpha (additive smoothing) must be positive")
    _need(all(v >= 0 for r in M for v in r) and all(v >= 0 for v in q), "counts must be >= 0")
    n, d = len(yl), len(q)
    jll = []
    for c in cls:
        rows = [M[i] for i in range(n) if yl[i] == c]
        tot = [math.fsum(r[j] for r in rows) for j in range(d)]
        den = math.fsum(tot) + alpha * d
        jll.append(math.log(len(rows) / n) + math.fsum(q[j] * math.log((tot[j] + alpha) / den) for j in range(d)))
    return _posterior(cls, jll)


def _ref_mnb(X, y, x, alpha):
    np = _np()
    from sklearn.naive_bayes import MultinomialNB
    m = MultinomialNB(alpha=alpha, force_alpha=True).fit(np.array(X), np.array(y, dtype=int))
    return {"prediction": int(m.predict(np.array([x]))[0]), "classes": [int(c) for c in m.classes_], "posteriors": _f(m.predict_proba(np.array([x]))[0])}


def _rand_mnb(rng):
    k, d = rng.randint(2, 3), rng.randint(3, 6)
    n = rng.randint(2 * k, 14)
    y = [i % k for i in range(n)]
    pref = [[rng.uniform(0.2, 3) for _ in range(d)] for _ in range(k)]
    X = [[rng.randint(0, int(4 * pref[c][j])) for j in range(d)] for c in y]
    x = [rng.randint(0, 4) for _ in range(d)]
    return {"X": X, "y": y, "x": x, "alpha": rng.choice([1.0, 0.5, 0.1, 2.0])}


_add("ml_multinomial_naive_bayes",
     name="Multinomial naive Bayes prediction (word counts)", topic="classical algorithms",
     purpose="Fit a multinomial naive Bayes classifier on count features (such as word counts) with additive (Laplace/Lidstone) smoothing alpha and give the predicted class and posteriors of a query count vector.",
     inputs=[_matin("X", "training counts: one row per document, one column per term"), _vecin("y", "training labels (whole numbers)"),
             _vecin("x", "query counts"), _numin("alpha", "additive smoothing (> 0; 1 = Laplace)")],
     outputs=[_o("prediction", "class with the largest posterior"), _o("classes", "classes in sorted order"), _o("posteriors", "posterior probability of each class")],
     primary="prediction", method="theta_cj = (N_cj + alpha) / (N_c + alpha d); log prior + sum x_j log theta_cj; log-sum-exp normalisation",
     latex=r"\hat\theta_{cj} = \frac{N_{cj} + \alpha}{N_c + \alpha d},\quad \log P(c \mid x) = \log P(c) + \sum_j x_j \log\hat\theta_{cj} + \text{const}",
     run=multinomial_nb, reference=_ref_mnb,
     wrong=lambda X, y, x, alpha: multinomial_nb(X, y, x, alpha * 10 + 1),
     rand=_rand_mnb, phrases=["Multinomial naive Bayes"],
     conditions=["alpha > 0", "counts >= 0", "at least two classes"],
     examples=[_ex({"X": [[2, 1, 0, 0], [1, 1, 0, 1], [0, 0, 3, 1], [0, 1, 2, 2]], "y": [0, 0, 1, 1], "x": [1, 0, 1, 0], "alpha": 1}),
               _ex({"X": [[3, 0, 1], [2, 1, 0], [0, 2, 3], [1, 3, 2], [0, 1, 4]], "y": [1, 1, 2, 2, 2], "x": [2, 1, 1], "alpha": 0.5})],
     extra_q=[_bad({"X": [[2, 1], [0, 3]], "y": [0, 1], "x": [1, 1], "alpha": 0})],
     level="intermediate")


def bernoulli_nb(X, y, x, alpha):
    M, yl, q, cls = _nb_data(X, y, x)
    _fin("alpha", alpha)
    _need(alpha > 0, "alpha (additive smoothing) must be positive")
    _need(all(v in (0.0, 1.0) for r in M for v in r) and all(v in (0.0, 1.0) for v in q), "features must be 0 or 1")
    n, d = len(yl), len(q)
    jll = []
    for c in cls:
        rows = [M[i] for i in range(n) if yl[i] == c]
        lp = math.log(len(rows) / n)
        for j in range(d):
            p = (sum(r[j] for r in rows) + alpha) / (len(rows) + 2 * alpha)
            lp += math.log(p) if q[j] == 1 else math.log1p(-p)
        jll.append(lp)
    return _posterior(cls, jll)


def _ref_bnb(X, y, x, alpha):
    np = _np()
    from sklearn.naive_bayes import BernoulliNB
    m = BernoulliNB(alpha=alpha, binarize=None, force_alpha=True).fit(np.array(X), np.array(y, dtype=int))
    return {"prediction": int(m.predict(np.array([x]))[0]), "classes": [int(c) for c in m.classes_], "posteriors": _f(m.predict_proba(np.array([x]))[0])}


def _rand_bnb(rng):
    k, d = rng.randint(2, 3), rng.randint(2, 6)
    n = rng.randint(2 * k, 14)
    y = [i % k for i in range(n)]
    pr = [[rng.uniform(0.1, 0.9) for _ in range(d)] for _ in range(k)]
    X = [[1 if rng.random() < pr[c][j] else 0 for j in range(d)] for c in y]
    return {"X": X, "y": y, "x": [rng.randint(0, 1) for _ in range(d)], "alpha": rng.choice([1.0, 0.5, 2.0])}


def _wrong_bnb(X, y, x, alpha):
    M, yl, q, cls = _nb_data(X, y, x)            # smoothing denominator N_c + alpha instead of N_c + 2 alpha
    jll = []
    for c in cls:
        rows = [M[i] for i in range(len(yl)) if yl[i] == c]
        lp = math.log(len(rows) / len(yl))
        for j in range(len(q)):
            p = (sum(r[j] for r in rows) + alpha) / (len(rows) + alpha)
            lp += math.log(p) if q[j] == 1 else math.log(max(1e-300, 1 - p * 0.999))
        jll.append(lp)
    m = max(jll)
    s = math.log(math.fsum(math.exp(v - m) for v in jll))
    return _out(prediction=cls[jll.index(m)], classes=cls, posteriors=[math.exp(v - m - s) for v in jll])


_add("ml_bernoulli_naive_bayes",
     name="Bernoulli naive Bayes prediction (binary features)", topic="classical algorithms",
     purpose="Fit a Bernoulli naive Bayes classifier on 0/1 features with additive smoothing alpha (absent features count too) and give the predicted class and posteriors of a query vector.",
     inputs=[_matin("X", "training features (0 or 1)"), _vecin("y", "training labels (whole numbers)"), _vecin("x", "query features (0 or 1)"),
             _numin("alpha", "additive smoothing (> 0)")],
     outputs=[_o("prediction", "class with the largest posterior"), _o("classes", "classes in sorted order"), _o("posteriors", "posterior probability of each class")],
     primary="prediction", method="p_cj = (N_cj + alpha)/(N_c + 2 alpha); log prior + sum of log p or log(1 - p)",
     latex=r"P(x \mid c) = \prod_j p_{cj}^{x_j}(1 - p_{cj})^{1 - x_j}",
     run=bernoulli_nb, reference=_ref_bnb, wrong=_wrong_bnb, rand=_rand_bnb,
     phrases=["Bernoulli naive Bayes"],
     conditions=["features 0 or 1", "alpha > 0", "at least two classes"],
     examples=[_ex({"X": [[1, 0, 1], [1, 1, 0], [0, 1, 1], [0, 0, 1], [1, 1, 1]], "y": [0, 0, 1, 1, 0], "x": [1, 0, 0], "alpha": 1}),
               _ex({"X": [[1, 0], [0, 1], [1, 1], [0, 0]], "y": [1, 2, 1, 2], "x": [0, 1], "alpha": 0.5})],
     level="advanced")


# ================================================================================================ decision stumps
def _stump_scan(M, score):
    """Every split x_j <= t (t midway between consecutive distinct values); the best by `score` (higher better)."""
    best, cands = None, []
    for j in range(len(M[0])):
        vals = sorted(set(r[j] for r in M))
        for a, b in zip(vals, vals[1:]):
            t = (a + b) / 2
            left = [i for i in range(len(M)) if M[i][j] <= t]
            right = [i for i in range(len(M)) if M[i][j] > t]
            s = score(left, right)
            cands.append((s, j, t, left, right))
    _need(cands, "no split is possible: every feature is constant")
    cands.sort(key=lambda c: -float(c[0]))
    top = cands[0]
    if len(cands) > 1:
        a, b = float(top[0]), float(cands[1][0])
        _need(abs(a - b) > 1e-9 * max(abs(a), 1e-300) and a != b, "two different splits are equally good (the best split is not unique)")
    return top


def _stump_X(X):
    M = _mat("X", X, 2)
    _need(all(v == int(v) and abs(v) <= 10000 for r in M for v in r), "features must be whole numbers up to 10000 in size")
    return M


def stump_classify(X, y, criterion):
    M = _stump_X(X)
    yl = _labels("y", y)
    _need(len(yl) == len(M), "y needs one label per row of X")
    _need(len(set(yl)) >= 2, "the node is pure: there is nothing to split")

    def imp(idx):
        lab = [yl[i] for i in idx]
        n = len(lab)
        if criterion == "gini":
            return 1 - sum(Fraction(lab.count(c), n) ** 2 for c in set(lab))
        return _Hc([lab.count(c) for c in set(lab)]) / math.log(2.0)
    n = len(yl)
    parent = imp(range(n))
    s, j, t, L, R = _stump_scan(M, lambda l, r: parent - (Fraction(len(l), n) * imp(l) + Fraction(len(r), n) * imp(r)) if criterion == "gini"
                                else parent - (len(l) / n * imp(l) + len(r) / n * imp(r)))
    return _out(feature=j, threshold=t, impurity_decrease=s, n_left=len(L), n_right=len(R))


def _ref_stump_clf(X, y, criterion):
    np = _np()
    from sklearn.tree import DecisionTreeClassifier
    t = DecisionTreeClassifier(max_depth=1, criterion=criterion, random_state=0).fit(np.array(X, dtype=float), np.array(y, dtype=int))
    tr = t.tree_
    n = tr.n_node_samples
    l, r = tr.children_left[0], tr.children_right[0]
    dec = tr.impurity[0] - (n[l] / n[0] * tr.impurity[l] + n[r] / n[0] * tr.impurity[r])
    return {"feature": int(tr.feature[0]), "threshold": float(tr.threshold[0]), "impurity_decrease": float(dec), "n_left": int(n[l]), "n_right": int(n[r])}


def _rand_stump(rng, regression=False):
    n, d = rng.randint(6, 20), rng.randint(1, 4)
    X = [[rng.randint(0, 12) for _ in range(d)] for _ in range(n)]
    j = rng.randrange(d)
    if regression:
        y = [round((5.0 if r[j] > 6 else 0.0) + rng.gauss(0, 2), 2) for r in X]
        return {"X": X, "y": y}
    y = [(1 if r[j] > 6 else 0) if rng.random() < 0.8 else rng.randint(0, 2) for r in X]
    return {"X": X, "y": y, "criterion": rng.choice(["gini", "entropy"])}


def _wrong_stump_clf(X, y, criterion):
    o = stump_classify(X, y, criterion)
    return dict(o, impurity_decrease=_r(o["impurity_decrease"] * len(y) / max(1, min(o["n_left"], o["n_right"]))), threshold=o["threshold"] + 0.5)


_add("ml_decision_stump_classification",
     name="Best decision-tree split (classification, Gini or entropy)", topic="classical algorithms",
     purpose="The best single split x_j <= t of a classification tree's root by Gini impurity or entropy (bits): the feature, the threshold (midway between consecutive values), the impurity decrease and the sizes of both sides.",
     inputs=[_matin("X", "features: one row per sample (whole numbers)"), _vecin("y", "class labels (whole numbers)"),
             _choice("criterion", ["gini", "entropy"], "split criterion: gini or entropy")],
     outputs=[_o("feature", "index (0-based column of X) of the best feature"), _o("threshold", "threshold t (go left when x_j <= t)"),
              _o("impurity_decrease", "parent impurity minus the size-weighted child impurities"), _o("n_left", "samples going left"), _o("n_right", "samples going right")],
     primary="impurity_decrease", method="scan every feature and every midpoint between consecutive distinct values; exact fractions for Gini",
     latex=r"\Delta I = I(S) - \frac{|S_L|}{|S|}I(S_L) - \frac{|S_R|}{|S|}I(S_R)",
     run=stump_classify, reference=_ref_stump_clf, wrong=_wrong_stump_clf, rand=_rand_stump,
     phrases=["Decision tree best split", "Best classification split"],
     conditions=["at least two classes", "whole-number features", "a unique best split"],
     examples=[_ex({"X": [[2, 7], [3, 1], [10, 8], [11, 2], [4, 9], [12, 0]], "y": [0, 0, 1, 1, 0, 1], "criterion": "gini"},
                   {"feature": 0, "threshold": 7.0, "impurity_decrease": 0.5, "n_left": 3}),
               _ex({"X": [[1], [2], [3], [4], [5], [6], [7]], "y": [0, 0, 1, 0, 1, 1, 1], "criterion": "entropy"})],
     extra_q=[_bad({"X": [[1, 5], [2, 6]], "y": [1, 1], "criterion": "gini"})],
     level="advanced")


# ================================================================================================ hierarchical clustering
def linkage_heights(X, method):
    M = _mat("X", X, 2)
    n = len(M)
    _need(n <= 40, "at most 40 points")
    D = {}
    for i in range(n):
        for j in range(i + 1, n):
            D[(i, j)] = math.sqrt(_sqd(M[i], M[j]))
    size = {i: 1 for i in range(n)}
    active = set(range(n))
    nxt = n
    heights = []
    key = lambda a, b: (a, b) if a < b else (b, a)
    while len(active) > 1:
        pairs = sorted(((D[key(a, b)], key(a, b)) for a in active for b in active if a < b))
        h, (a, b) = pairs[0]
        if len(pairs) > 1 and method != "single":
            _need(pairs[1][0] - h > 1e-12 * max(h, 1e-300), "two merges are tied at the same distance (the order is ambiguous)")
        heights.append(h)
        na, nb = size[a], size[b]
        for c in active - {a, b}:
            dac, dbc = D[key(a, c)], D[key(b, c)]
            if method == "single":
                v = min(dac, dbc)
            elif method == "complete":
                v = max(dac, dbc)
            elif method == "average":
                v = (na * dac + nb * dbc) / (na + nb)
            else:
                nc = size[c]
                v = math.sqrt(((na + nc) * dac * dac + (nb + nc) * dbc * dbc - nc * h * h) / (na + nb + nc))
            D[key(nxt, c)] = v
        active -= {a, b}
        active.add(nxt)
        size[nxt] = na + nb
        nxt += 1
    return _out(heights=heights, final_height=heights[-1])


def _ref_linkage(X, method):
    np = _np()
    from scipy.cluster.hierarchy import linkage
    Z = linkage(np.array(X, dtype=float), method=method)
    h = sorted(_f(Z[:, 2]))
    return {"heights": h, "final_height": h[-1]}


_add("ml_hierarchical_clustering",
     name="Agglomerative hierarchical clustering merge heights", topic="classical algorithms",
     purpose="Agglomerative clustering of points with single, complete, average or Ward linkage (Euclidean distance): the distance at which each of the n - 1 merges happens, in order, and the final merge height.",
     inputs=[_matin("X", "points: one row per sample (2 to 40)"), _choice("method", ["single", "complete", "average", "ward"], "linkage: single, complete, average or ward")],
     outputs=[_o("heights", "merge distances in merge order (ascending)"), _o("final_height", "height of the last merge")],
     primary="final_height", method="repeatedly merge the closest pair; Lance-Williams updates of the cluster distances",
     latex=r"d(k, i \cup j) = \sqrt{\frac{(n_i + n_k)d_{ki}^2 + (n_j + n_k)d_{kj}^2 - n_k d_{ij}^2}{n_i + n_j + n_k}}\ \text{(Ward)}",
     run=linkage_heights, reference=_ref_linkage,
     wrong=lambda X, method: (lambda o: _out(heights=[h * h for h in o["heights"]], final_height=o["final_height"] ** 2))(linkage_heights(X, method)),
     rand=lambda rng: {"X": _rand_blobs(rng, 2, 4, 4, 14, 2, 3)[0], "method": rng.choice(["single", "complete", "average", "ward"])},
     phrases=["Hierarchical clustering", "Agglomerative clustering"],
     conditions=["2 to 40 points", "no tied merge distances (complete, average, ward)"],
     examples=[_ex({"X": [[0, 0], [0, 1], [4, 0], [4, 1.5], [10, 10]], "method": "single"}, {"final_height": 10.4043260233}),
               _ex({"X": [[0, 0], [0, 1], [4, 0], [4, 1.5], [10, 10]], "method": "ward"})],
     level="expert")


# ================================================================================================ PCA
def _jacobi(A):
    """Eigen-decomposition of a symmetric matrix by cyclic Jacobi rotations: (values, vectors as columns)."""
    n = len(A)
    a = [list(r) for r in A]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(100):
        off = math.fsum(a[i][j] ** 2 for i in range(n) for j in range(n) if i != j)
        scale = math.fsum(a[i][i] ** 2 for i in range(n))
        if off <= 1e-34 * max(scale, 1e-300):
            break
        for p in range(n):
            for q in range(p + 1, n):
                if a[p][q] == 0:
                    continue
                theta = (a[q][q] - a[p][p]) / (2 * a[p][q])
                t = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1))
                c = 1 / math.sqrt(t * t + 1)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p], a[k][q] = c * akp - s * akq, s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k], a[q][k] = c * apk - s * aqk, s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = V[k][p], V[k][q]
                    V[k][p], V[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    return [a[i][i] for i in range(n)], V


def _sign_fix(v):
    j = max(range(len(v)), key=lambda i: (abs(v[i]), -i))
    return [-x for x in v] if v[j] < 0 else list(v)


def pca(X):
    M = _mat("X", X, 3, 1)
    n, d = len(M), len(M[0])
    _need(n > d, "PCA here needs more samples than features")
    _need(d <= 8, "at most 8 features")
    F = [[_F(v) for v in r] for r in M]
    mean = [sum(r[j] for r in F) / n for j in range(d)]
    Cv = [[float(sum((r[i] - mean[i]) * (r[j] - mean[j]) for r in F) / (n - 1)) for j in range(d)] for i in range(d)]
    vals, V = _jacobi(Cv)
    order = sorted(range(d), key=lambda i: -vals[i])
    lam = [vals[i] for i in order]
    _need(lam[-1] > 1e-10 * lam[0], "the data are rank-deficient: a component's variance is zero to rounding")
    if d > 1:
        _need(lam[0] - lam[1] > 1e-6 * lam[0], "the two largest variances are equal: the first component is not unique")
    comp = _sign_fix([V[k][order[0]] for k in range(d)])
    tot = math.fsum(lam)
    scores = [math.fsum(float(r[j] - mean[j]) * comp[j] for j in range(d)) for r in F]
    return _out(explained_variance=lam, explained_variance_ratio=[v / tot for v in lam], component_1=comp, scores_1=scores)


def _ref_pca(X):
    np = _np()
    from sklearn.decomposition import PCA
    A = np.array(X, dtype=float)
    p = PCA(svd_solver="full").fit(A)
    comp = p.components_[0]
    j = int(np.argmax(np.abs(comp)))
    if comp[j] < 0:
        comp = -comp
    scores = (A - A.mean(0)) @ comp
    return {"explained_variance": _f(p.explained_variance_), "explained_variance_ratio": _f(p.explained_variance_ratio_),
            "component_1": _f(comp), "scores_1": _f(scores)}


def _rand_pca(rng):
    d = rng.randint(1, 4)
    n = rng.randint(d + 3, 15)
    A = [[rng.uniform(-1, 1) + (2.5 - 0.6 * i if i == j else 0.0) for j in range(d)] for i in range(d)]
    X = []
    for _ in range(n):
        z = [rng.gauss(0, 1) for _ in range(d)]
        X.append([round(math.fsum(z[k] * A[k][j] for k in range(d)) + 3, 2) for j in range(d)])
    return {"X": X}


def _wrong_pca(X):
    o = pca(X)
    n = len(X)
    return dict(o, explained_variance=[_r(v * (n - 1) / n) for v in o["explained_variance"]],
                scores_1=[_r(s * (n - 1) / n) for s in o["scores_1"]])


_add("ml_pca",
     name="Principal component analysis (explained variance and first component)", topic="dimensionality reduction",
     purpose="Principal component analysis of a data matrix: the variance explained by every component (sample covariance, n - 1), the explained variance ratios, the first principal direction and the first component score of every sample.",
     inputs=[_matin("X", "data: one row per sample, one column per feature (more samples than features, at most 8 features)")],
     outputs=[_o("explained_variance", "eigenvalues of the sample covariance, descending"), _o("explained_variance_ratio", "each eigenvalue / their sum"),
              _o("component_1", "first principal direction (unit length, largest-magnitude entry positive)"), _o("scores_1", "projection of each centred sample on component_1")],
     primary="explained_variance", method="exact centring and covariance, cyclic Jacobi eigen-decomposition",
     latex=r"\Sigma = \frac{1}{n-1}(X - \bar X)^\top (X - \bar X) = V \Lambda V^\top",
     run=pca, reference=_ref_pca, wrong=_wrong_pca, rand=_rand_pca,
     phrases=["PCA", "Principal component analysis"],
     conditions=["n > number of features", "full rank", "a unique first component"],
     examples=[_ex({"X": [[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0], [2.3, 2.7], [2, 1.6], [1, 1.1], [1.5, 1.6], [1.1, 0.9]]},
                   {"explained_variance": [1.28402771217, 0.0490833989]}),
               _ex({"X": [[1, 2, 3], [2, 1, 3], [4, 5, 7], [3, 3, 2], [5, 6, 4]]})],
     extra_q=[_bad({"X": [[1, 2], [2, 4], [3, 6], [4, 8]]})],
     level="advanced")


# ================================================================================================ scaling and encoding
def standard_scaler(X):
    M = _mat("X", X, 1, 1)
    n, d = len(M), len(M[0])
    F = [[_F(v) for v in r] for r in M]
    mean = [sum(r[j] for r in F) / n for j in range(d)]
    var = [sum((r[j] - mean[j]) ** 2 for r in F) / n for j in range(d)]
    sd = [math.sqrt(v) for v in var]
    Z = [[float(r[j] - mean[j]) / sd[j] if var[j] > 0 else 0.0 for j in range(d)] for r in F]
    return _out(means=mean, stds=sd, scaled=Z)


def _ref_std(X):
    np = _np()
    from sklearn.preprocessing import StandardScaler
    s = StandardScaler().fit(np.array(X, dtype=float))
    return {"means": _f(s.mean_), "stds": _f(np.sqrt(s.var_)), "scaled": _f(s.transform(np.array(X, dtype=float)))}


def _rand_matrix(rng, nmin=2, nmax=12, dmin=1, dmax=4):
    n, d = rng.randint(nmin, nmax), rng.randint(dmin, dmax)
    sc = [rng.uniform(0.5, 50) for _ in range(d)]
    off = [rng.uniform(-100, 100) for _ in range(d)]
    return [[round(off[j] + rng.gauss(0, sc[j]), 2) for j in range(d)] for _ in range(n)]


_add("ml_standard_scaler",
     name="Standardisation (z-scores of features)", topic="preprocessing",
     purpose="Standardise each feature column to mean 0 and unit variance: the column means, the population standard deviations and the scaled matrix (a constant column scales to 0).",
     inputs=[_matin("X", "data: one row per sample, one column per feature")],
     outputs=[_o("means", "column means"), _o("stds", "population standard deviations (divide by n)"), _o("scaled", "(x - mean) / std")],
     primary="means", method="exact means and variances; one square root per column",
     latex=r"z_{ij} = \frac{x_{ij} - \bar x_j}{s_j},\quad s_j^2 = \frac{1}{n}\sum_i (x_{ij} - \bar x_j)^2",
     run=standard_scaler, reference=_ref_std,
     wrong=lambda X: (lambda n: (lambda o: dict(o, stds=[_r(s * math.sqrt(n / (n - 1))) for s in o["stds"]] if n > 1 else o["stds"],
                                                 scaled=[[_r(v * math.sqrt((n - 1) / n)) for v in r] for r in o["scaled"]]))(standard_scaler(X)))(len(X)),
     rand=lambda rng: {"X": _rand_matrix(rng, 3)}, phrases=["Standard scaler", "Standardize features", "Standardise features"],
     conditions=["a constant column scales to 0"],
     examples=[_ex({"X": [[1, 10], [2, 20], [3, 30], [4, 40]]}, {"means": [2.5, 25.0], "stds": [1.11803398875, 11.1803398875]}),
               _ex({"X": [[0, 5], [0, 5], [1, 5], [1, 5]]}, {"scaled": [[-1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]})],
     level="beginner")


def minmax_scaler(X):
    M = _mat("X", X, 1, 1)
    n, d = len(M), len(M[0])
    F = [[_F(v) for v in r] for r in M]
    lo = [min(r[j] for r in F) for j in range(d)]
    hi = [max(r[j] for r in F) for j in range(d)]
    S = [[(r[j] - lo[j]) / (hi[j] - lo[j]) if hi[j] > lo[j] else Fraction(0) for j in range(d)] for r in F]
    return _out(mins=lo, maxs=hi, scaled=S)


def _ref_minmax(X):
    np = _np()
    from sklearn.preprocessing import MinMaxScaler
    s = MinMaxScaler().fit(np.array(X, dtype=float))
    return {"mins": _f(s.data_min_), "maxs": _f(s.data_max_), "scaled": _f(s.transform(np.array(X, dtype=float)))}


_add("ml_minmax_scaler",
     name="Min-max scaling to [0, 1]", topic="preprocessing",
     purpose="Scale each feature column linearly to the range [0, 1] using its minimum and maximum (a constant column scales to 0).",
     inputs=[_matin("X", "data: one row per sample, one column per feature")],
     outputs=[_o("mins", "column minima"), _o("maxs", "column maxima"), _o("scaled", "(x - min) / (max - min)")],
     primary="mins", method="exact rational arithmetic",
     latex=r"x' = \frac{x - \min_j}{\max_j - \min_j}",
     run=minmax_scaler, reference=_ref_minmax,
     wrong=lambda X: (lambda o: dict(o, scaled=[[_r(v * 0.5 + 0.25) for v in r] for r in o["scaled"]]))(minmax_scaler(X)),
     rand=lambda rng: {"X": _rand_matrix(rng, 2)}, phrases=["Min-max scaler", "Min-max scaling", "Minmax scaling"],
     conditions=["a constant column scales to 0"],
     examples=[_ex({"X": [[1, 10], [2, 20], [3, 40], [5, 30]]}, {"scaled": [[0.0, 0.0], [0.25, 0.333333333333], [0.5, 1.0], [1.0, 0.666666666667]]}),
               _ex({"X": [[-3, 7], [0, 7], [9, 7]]})],
     level="beginner")


def _cats(categories):
    _need(isinstance(categories, str), "categories must be a list of words")
    items = [c.strip() for c in categories.split(",")]
    _need(items and all(items), "every category must be a non-empty word")
    _need(len(items) <= 500, "at most 500 values")
    return items


def one_hot(categories):
    items = _cats(categories)
    cats = sorted(set(items))
    idx = {c: i for i, c in enumerate(cats)}
    return _out(categories=cats, counts=[items.count(c) for c in cats], codes=[idx[c] for c in items],
                one_hot=[[1 if idx[c] == j else 0 for j in range(len(cats))] for c in items])


def _ref_one_hot(categories):
    np = _np()
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    items = [c.strip() for c in categories.split(",")]
    le = LabelEncoder().fit(items)
    oh = OneHotEncoder(sparse_output=False).fit(np.array(items).reshape(-1, 1))
    cats = [str(c) for c in le.classes_]
    return {"categories": cats, "counts": [items.count(c) for c in cats], "codes": [int(v) for v in le.transform(items)],
            "one_hot": [[int(v) for v in r] for r in oh.transform(np.array(items).reshape(-1, 1))]}


_WORDS = ["red", "green", "blue", "cat", "dog", "mouse", "low", "medium", "high", "north", "south", "east", "west", "apple", "pear"]

_add("ml_one_hot_encoding",
     name="Label encoding and one-hot encoding of categories", topic="preprocessing",
     purpose="Encode a list of category values: the categories in sorted order, how often each occurs, the integer code (label encoding) of each value and its one-hot row.",
     inputs=[_textlist("categories", "category values, written as a bracketed list of words: [red, green, red]")],
     outputs=[_o("categories", "distinct categories, sorted", "text"), _o("counts", "occurrences of each category"),
              _o("codes", "label code of each value (position of its category)"), _o("one_hot", "one-hot row of each value")],
     primary="counts", method="sort the distinct values; index each value",
     latex=r"\mathrm{onehot}(v)_j = \mathbb{1}(v = c_j)",
     run=one_hot, reference=_ref_one_hot,
     wrong=lambda categories: (lambda items: (lambda cats: _out(categories=cats, counts=[items.count(c) for c in cats],
                                                                codes=[cats.index(c) + 1 for c in items], one_hot=[[0]]))(list(dict.fromkeys(items))))(_cats(categories)),
     rand=lambda rng: {"categories": ", ".join(rng.choice(rng.sample(_WORDS, rng.randint(2, 5))) for _ in range(rng.randint(3, 15)))},
     phrases=["One-hot encoding", "One hot encoding", "Label encoding"],
     conditions=["non-empty category words"],
     examples=[_ex({"categories": "red, green, red, blue, green, red"}, {"categories": ["blue", "green", "red"], "counts": [1, 2, 3], "codes": [2, 1, 2, 0, 1, 2]}),
               _ex({"categories": "low, high, medium, low"})],
     level="beginner")


# ================================================================================================ splitting and cross-validation
def split_sizes(n, test_size):
    _need(isinstance(n, int) and n >= 2, "n must be a whole number >= 2")
    _fin("test_size", test_size)
    _need(0 < test_size < 1, "test_size must be a fraction strictly between 0 and 1")
    t = Fraction(repr(float(test_size)))
    n_test = math.ceil(t * n)
    n_train = n - n_test
    _need(n_train >= 1, "the training set would be empty")
    return _out(n_test=n_test, n_train=n_train)


def _ref_split(n, test_size):
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(list(range(n)), test_size=test_size, shuffle=False)
    return {"n_test": len(te), "n_train": len(tr)}


_add("ml_train_test_split_sizes",
     name="Train/test split sizes", topic="model selection",
     purpose="The number of test and training samples when n samples are split with a given test fraction (the test size is rounded up).",
     inputs=[_intin("n", "number of samples", 2), _numin("test_size", "fraction of the samples used for testing (0 to 1, or a percentage)", pct=True)],
     outputs=[_o("n_test", "ceil(test_size * n)"), _o("n_train", "n - n_test")],
     primary="n_test", method="exact ceiling of the written fraction times n",
     latex=r"n_{\mathrm{test}} = \lceil t\,n \rceil,\quad n_{\mathrm{train}} = n - n_{\mathrm{test}}",
     run=split_sizes, reference=_ref_split,
     wrong=lambda n, test_size: _out(n_test=int(test_size * n), n_train=n - int(test_size * n)) if int(test_size * n) != math.ceil(Fraction(repr(float(test_size))) * n) else _out(n_test=0, n_train=n),
     rand=lambda rng: {"n": rng.randint(5, 2000), "test_size": rng.choice([0.2, 0.25, 0.3, 0.1, 0.15, 0.33, round(rng.uniform(0.05, 0.6), 3)])},
     phrases=["Train test split sizes", "Train-test split sizes"],
     conditions=["0 < test_size < 1", "a non-empty training set"],
     examples=[_ex({"n": 100, "test_size": 0.2}, {"n_test": 20, "n_train": 80}), _ex({"n": 47, "test_size": 0.25}, {"n_test": 12, "n_train": 35})],
     extra_q=[_bad({"n": 100, "test_size": 1.5})],
     level="beginner")


def kfold_sizes(n, k):
    _need(isinstance(n, int) and isinstance(k, int) and 2 <= k <= n, "k must be at least 2 and at most n")
    base, extra = divmod(n, k)
    test = [base + 1 if i < extra else base for i in range(k)]
    return _out(test_sizes=test, train_sizes=[n - t for t in test], first_test_fold=[0, test[0] - 1])


def _ref_kfold(n, k):
    from sklearn.model_selection import KFold
    sp = list(KFold(n_splits=k).split(list(range(n))))
    return {"test_sizes": [len(te) for _, te in sp], "train_sizes": [len(tr) for tr, _ in sp], "first_test_fold": [int(sp[0][1][0]), int(sp[0][1][-1])]}


_add("ml_kfold_sizes",
     name="k-fold cross-validation fold sizes", topic="model selection",
     purpose="The test and training set sizes of each fold when n samples are split into k consecutive folds (the first n mod k folds get one extra sample), and the index range of the first test fold.",
     inputs=[_intin("n", "number of samples", 2), _intin("k", "number of folds", 2)],
     outputs=[_o("test_sizes", "size of each test fold"), _o("train_sizes", "size of each training set"), _o("first_test_fold", "first and last index (0-based) of the first test fold")],
     primary="test_sizes", method="n = k q + r: r folds of q + 1, k - r folds of q, in order",
     latex=r"|F_i| = \lfloor n/k \rfloor + \mathbb{1}(i < n \bmod k)",
     run=kfold_sizes, reference=_ref_kfold,
     wrong=lambda n, k: (lambda o: dict(o, test_sizes=list(reversed(o["test_sizes"])), first_test_fold=[0, o["test_sizes"][-1] - 1]))(kfold_sizes(n, k)) if n % k else _out(test_sizes=[n // k] * (k - 1) + [n // k + 1], train_sizes=[0], first_test_fold=[0, 0]),
     rand=lambda rng: (lambda k: {"n": rng.randint(k, 500), "k": k})(rng.randint(2, 12)),
     phrases=["K-fold sizes", "Cross-validation fold sizes"],
     conditions=["2 <= k <= n"],
     examples=[_ex({"n": 10, "k": 3}, {"test_sizes": [4, 3, 3], "train_sizes": [6, 7, 7]}), _ex({"n": 100, "k": 5})],
     extra_q=[_bad({"n": 3, "k": 5})],
     level="beginner")


def _ols_exact(rows, ys):
    """Least squares with intercept, exact (Fractions): (coefficients, intercept)."""
    d = len(rows[0])
    A = [[Fraction(1)] + [_F(v) for v in r] for r in rows]
    G = [[sum(a[i] * a[j] for a in A) for j in range(d + 1)] for i in range(d + 1)]
    h = [sum(a[i] * _F(y) for a, y in zip(A, ys)) for i in range(d + 1)]
    beta = _solve_exact(G, h)
    return beta[1:], beta[0]


def kfold_cv_ols(X, y, k):
    M = _mat("X", X)
    yv = _vec("y", y)
    n, d = len(M), len(M[0])
    _need(len(yv) == n, "y needs one value per row of X")
    _need(isinstance(k, int) and 2 <= k <= n, "k must be between 2 and n")
    sizes = kfold_sizes(n, k)["test_sizes"]
    _need(n - max(sizes) >= d + 2, "every training set needs at least d + 2 samples")
    mses, start = [], 0
    for s in sizes:
        te = list(range(start, start + s))
        tr = [i for i in range(n) if i < start or i >= start + s]
        w, b = _ols_exact([M[i] for i in tr], [yv[i] for i in tr])
        err = [sum(wj * _F(v) for wj, v in zip(w, M[i])) + b - _F(yv[i]) for i in te]
        mses.append(sum(e * e for e in err) / s)
        start += s
    return _out(fold_mse=mses, mean_mse=sum(mses) / k)


def _ref_kfold_cv(X, y, k):
    np = _np()
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_score
    s = -cross_val_score(LinearRegression(), np.array(X, dtype=float), np.array(y, dtype=float), cv=KFold(n_splits=k), scoring="neg_mean_squared_error")
    return {"fold_mse": _f(s), "mean_mse": float(s.mean())}


def _rand_ols(rng, nmin=8, nmax=24):
    n, d = rng.randint(nmin, nmax), rng.randint(1, 3)
    w = [rng.uniform(-3, 3) for _ in range(d)]
    X = [[round(rng.uniform(-5, 5), 2) for _ in range(d)] for _ in range(n)]
    y = [round(math.fsum(a * b for a, b in zip(r, w)) + 1.5 + rng.gauss(0, 1.5), 2) for r in X]
    return {"X": X, "y": y}


_add("ml_kfold_cv_linear_regression",
     name="k-fold cross-validated MSE of linear regression", topic="model selection",
     purpose="The mean squared error of ordinary least squares (with intercept) on each of k consecutive, unshuffled folds, trained on the other folds, and the mean over folds.",
     inputs=[_matin("X", "features: one row per sample"), _vecin("y", "targets"), _intin("k", "number of folds", 2)],
     outputs=[_o("fold_mse", "test MSE of each fold"), _o("mean_mse", "mean of the fold MSEs")],
     primary="mean_mse", method="consecutive folds (first n mod k folds one larger); exact least squares by the normal equations",
     latex=r"\mathrm{CV}_k = \frac{1}{k}\sum_{i=1}^k \mathrm{MSE}_i",
     run=kfold_cv_ols, reference=_ref_kfold_cv,
     wrong=lambda X, y, k: (lambda o: dict(o, mean_mse=_r(sum(o["fold_mse"]) / (k - 1))))(kfold_cv_ols(X, y, k)),
     rand=lambda rng: dict(_rand_ols(rng), k=rng.choice([2, 3, 4, 5])),
     phrases=["K-fold cross-validation of linear regression", "Cross-validated MSE of linear regression"],
     conditions=["every training set has at least d + 2 samples and full rank"],
     examples=[_ex({"X": [[1], [2], [3], [4], [5], [6]], "y": [1.1, 1.9, 3.2, 3.9, 5.1, 5.8], "k": 3}),
               _ex({"X": [[1, 0], [2, 1], [3, 1], [4, 3], [5, 2], [6, 5], [7, 4], [8, 6]], "y": [2, 4, 5, 9, 9, 14, 13, 17], "k": 4})],
     level="advanced")


def loocv_ols(X, y):
    M = _mat("X", X)
    yv = _vec("y", y)
    n, d = len(M), len(M[0])
    _need(len(yv) == n, "y needs one value per row of X")
    _need(n >= d + 3, "LOOCV needs at least d + 3 samples")
    A = [[Fraction(1)] + [_F(v) for v in r] for r in M]
    G = [[sum(a[i] * a[j] for a in A) for j in range(d + 1)] for i in range(d + 1)]
    Ginv = [_solve_exact(G, [Fraction(int(i == j)) for i in range(d + 1)]) for j in range(d + 1)]   # columns of G^-1
    w, b = _ols_exact(M, yv)
    press = Fraction(0)
    for i in range(n):
        a = A[i]
        h = sum(a[p] * Ginv[q][p] * a[q] for p in range(d + 1) for q in range(d + 1))
        _need(h != 1, "a sample has leverage 1: leaving it out makes the fit undetermined")
        e = sum(wj * _F(v) for wj, v in zip(w, M[i])) + b - _F(yv[i])
        press += (e / (1 - h)) ** 2
    return _out(press=press, loocv_mse=press / n)


def _ref_loocv(X, y):
    np = _np()
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import LeaveOneOut, cross_val_score
    s = -cross_val_score(LinearRegression(), np.array(X, dtype=float), np.array(y, dtype=float), cv=LeaveOneOut(), scoring="neg_mean_squared_error")
    return {"press": float(s.sum()), "loocv_mse": float(s.mean())}


_add("ml_loocv_linear_regression",
     name="Leave-one-out cross-validation of linear regression (PRESS)", topic="model selection",
     purpose="The leave-one-out cross-validated MSE and the PRESS statistic of ordinary least squares with intercept, from one fit through the leverage shortcut e_i / (1 - h_ii).",
     inputs=[_matin("X", "features: one row per sample"), _vecin("y", "targets")],
     outputs=[_o("press", "sum of squared leave-one-out errors"), _o("loocv_mse", "PRESS / n")],
     primary="loocv_mse", method="hat-matrix diagonal from the exact inverse of the normal matrix; e_i / (1 - h_ii)",
     latex=r"\mathrm{PRESS} = \sum_i \left(\frac{e_i}{1 - h_{ii}}\right)^2,\quad h_{ii} = x_i^\top (X^\top X)^{-1} x_i",
     run=loocv_ols, reference=_ref_loocv,
     wrong=lambda X, y: (lambda w_b: _out(press=sum((sum(a * _F(v) for a, v in zip(w_b[0], r)) + w_b[1] - _F(t)) ** 2 for r, t in zip(X, y)),
                                         loocv_mse=sum((sum(a * _F(v) for a, v in zip(w_b[0], r)) + w_b[1] - _F(t)) ** 2 for r, t in zip(X, y)) / len(y)))(_ols_exact(X, y)),
     rand=lambda rng: _rand_ols(rng, 6, 18), phrases=["Leave-one-out cross-validation of linear regression", "LOOCV of linear regression", "PRESS statistic"],
     conditions=["at least d + 3 samples", "no leverage equal to 1"],
     examples=[_ex({"X": [[1], [2], [3], [4], [5], [6]], "y": [1.1, 1.9, 3.2, 3.9, 5.1, 5.8]}),
               _ex({"X": [[1, 0], [2, 1], [3, 1], [4, 3], [5, 2], [6, 5], [7, 4], [8, 6]], "y": [2, 4, 5, 9, 9, 14, 13, 17]})],
     level="expert")


def bias_variance(P, f):
    M = _mat("P", P, 2, 1)
    fv = _vec("f", f)
    _need(len(fv) == len(M[0]), "f needs one true value per column of P")
    R = [[_F(v) for v in r] for r in M]
    m, n = len(R), len(fv)
    F = [_F(v) for v in fv]
    mean = [sum(r[j] for r in R) / m for j in range(n)]
    b2 = sum((mean[j] - F[j]) ** 2 for j in range(n)) / n
    var = sum(sum((r[j] - mean[j]) ** 2 for r in R) / m for j in range(n)) / n
    mse = sum(sum((r[j] - F[j]) ** 2 for r in R) / m for j in range(n)) / n
    return _out(bias_squared=b2, variance=var, expected_mse=mse)


def _ref_bias_var(P, f):
    np = _np()
    A, t = np.array(P, dtype=float), np.array(f, dtype=float)
    return {"bias_squared": float(np.mean((A.mean(0) - t) ** 2)), "variance": float(np.mean(A.var(0))), "expected_mse": float(np.mean((A - t) ** 2))}


def _rand_bias_var(rng):
    m, n = rng.randint(2, 8), rng.randint(1, 8)
    f = [round(rng.uniform(-5, 5), 2) for _ in range(n)]
    bias = rng.uniform(-1, 1)
    P = [[round(v + bias + rng.gauss(0, 1), 2) for v in f] for _ in range(m)]
    return {"P": P, "f": f}


_add("ml_bias_variance_decomposition",
     name="Bias-variance decomposition of predictions", topic="model selection",
     purpose="For predictions of several models (one row each, e.g. trained on different samples) at the same test points with known noise-free values: the squared bias, the variance and the expected squared error, which equals their sum.",
     inputs=[_matin("P", "predictions: one row per trained model, one column per test point (at least two rows)"), _vecin("f", "true (noise-free) value at each test point")],
     outputs=[_o("bias_squared", "mean over points of (mean prediction - f)^2"), _o("variance", "mean over points of the variance of the predictions (divide by the number of models)"),
              _o("expected_mse", "mean squared error over models and points = bias^2 + variance")],
     primary="expected_mse", method="exact rational sums",
     latex=r"E[(\hat f - f)^2] = (E[\hat f] - f)^2 + E[(\hat f - E[\hat f])^2]",
     run=bias_variance, reference=_ref_bias_var,
     wrong=lambda P, f: (lambda o: dict(o, variance=_r(o["variance"] * len(P) / (len(P) - 1)), expected_mse=_r(o["bias_squared"] + o["variance"] * len(P) / (len(P) - 1))))(bias_variance(P, f)),
     rand=_rand_bias_var, phrases=["Bias-variance decomposition", "Bias variance decomposition"],
     conditions=["at least two models"],
     examples=[_ex({"P": [[1.0, 2.0], [3.0, 2.5], [2.0, 3.5]], "f": [1.5, 2.5]}),
               _ex({"P": [[0.9, 2.1, 2.9], [1.2, 1.8, 3.3]], "f": [1, 2, 3]})],
     weak="checked by a numpy computation of the same definitions; the decomposition identity (expected MSE computed directly equals bias^2 + variance) is an independent check of the two parts",
     level="advanced")


# ================================================================================================ linear models
def _sym_solve(A, b):
    return _gauss(A, b)


def _logit_obj(M, yv, w, b, C):
    z = [math.fsum(a * c for a, c in zip(r, w)) + b for r in M]
    return 0.5 * math.fsum(c * c for c in w) + C * math.fsum(max(v, 0.0) - v * t + math.log1p(math.exp(-abs(v))) for v, t in zip(z, yv))


def logistic_fit(X, y, C):
    M = _mat("X", X, 2, 1)
    yv = _vec("y", y)
    _need(len(yv) == len(M), "y needs one label per row of X")
    _need(all(v in (0.0, 1.0) for v in yv) and len(set(yv)) == 2, "labels must be 0 and 1, both present")
    _fin("C", C)
    _need(C > 0, "C (inverse regularisation strength) must be positive")
    n, d = len(M), len(M[0])
    th = [0.0] * (d + 1)                                  # weights then intercept
    for it in range(200):
        z = [math.fsum(a * c for a, c in zip(r, th[:d])) + th[d] for r in M]
        p = [_sigmoid(v) for v in z]
        r_ = [pi - t for pi, t in zip(p, yv)]
        g = [th[j] + C * math.fsum(r_[i] * M[i][j] for i in range(n)) for j in range(d)] + [C * math.fsum(r_)]
        s = [pi * (1 - pi) for pi in p]
        H = [[(1.0 if i == j and i < d else 0.0) + C * math.fsum(s[t] * (M[t][i] if i < d else 1.0) * (M[t][j] if j < d else 1.0) for t in range(n))
              for j in range(d + 1)] for i in range(d + 1)]
        step = _sym_solve(H, g)
        th = [a - b for a, b in zip(th, step)]
        gnorm = max(abs(v) for v in g)
        if max(abs(v) for v in step) <= 1e-15 * max(1.0, max(abs(v) for v in th)) and it > 2:
            break
    else:
        raise ValueError("Newton's method did not converge")
    z = [math.fsum(a * c for a, c in zip(r, th[:d])) + th[d] for r in M]
    acc = Fraction(sum(1 for v, t in zip(z, yv) if (v > 0) == (t == 1)), n)
    return _out(w=th[:d], b=th[d], objective=_logit_obj(M, yv, th[:d], th[d], C), train_accuracy=acc)


def _ref_logistic_fit(X, y, C):
    np = _np()
    from scipy.optimize import root
    from sklearn.linear_model import LogisticRegression
    A, t = np.array(X, dtype=float), np.array(y, dtype=float)
    m = LogisticRegression(C=C, tol=1e-12, max_iter=100000).fit(A, t.astype(int))
    x0 = np.concatenate([m.coef_[0], m.intercept_])
    d = A.shape[1]

    def grad(th):
        z = A @ th[:d] + th[d]
        r = 1 / (1 + np.exp(-z)) - t
        return np.concatenate([th[:d] + C * A.T @ r, [C * r.sum()]])
    sol = root(grad, x0, method="hybr", tol=1e-15)
    th = sol.x
    z = A @ th[:d] + th[d]
    obj = 0.5 * th[:d] @ th[:d] + C * np.sum(np.logaddexp(0, z) - t * z)
    return {"w": _f(th[:d]), "b": float(th[d]), "objective": float(obj), "train_accuracy": float(np.mean((z > 0) == (t == 1)))}


def _rand_logfit(rng):
    while True:
        n, d = rng.randint(8, 30), rng.randint(1, 3)
        w = [rng.uniform(-1.5, 1.5) for _ in range(d)]
        X = [[round(rng.uniform(-3, 3), 2) for _ in range(d)] for _ in range(n)]
        y = [1 if rng.random() < _sigmoid(math.fsum(a * b for a, b in zip(r, w)) + 0.3) else 0 for r in X]
        if 0 < sum(y) < n:
            return {"X": X, "y": y, "C": rng.choice([0.1, 1.0, 10.0, round(rng.uniform(0.05, 5), 2)])}


_add("ml_logistic_regression_fit",
     name="Logistic regression fit (L2, stated C)", topic="linear models",
     purpose="Fit binary logistic regression with an L2 penalty: minimise 0.5 ||w||^2 + C sum of log-losses (the intercept is not penalised); give the weights, the intercept, the objective and the training accuracy.",
     inputs=[_matin("X", "features: one row per sample"), _vecin("y", "labels (0 and 1)"), _numin("C", "inverse regularisation strength (> 0)")],
     outputs=[_o("w", "fitted weights"), _o("b", "fitted intercept"), _o("objective", "minimised objective"), _o("train_accuracy", "fraction of training samples classified correctly (z > 0 -> 1)")],
     primary="objective", method="Newton's method on the strictly convex objective (exact gradient and Hessian) to machine precision",
     latex=r"\min_{w,b}\ \tfrac12\|w\|^2 + C\sum_i \log\left(1 + e^{-\tilde y_i (w^\top x_i + b)}\right)",
     run=logistic_fit, reference=_ref_logistic_fit,
     wrong=lambda X, y, C: (lambda o: dict(o, w=[_r(v * 1.1) for v in o["w"]], objective=_r(o["objective"] * 1.01 + 0.01)))(logistic_fit(X, y, C)),
     rand=_rand_logfit, phrases=["Logistic regression fit", "Fit logistic regression"],
     conditions=["labels 0 and 1, both present", "C > 0", "Newton's method converges"],
     examples=[_ex({"X": [[0.5], [1.0], [1.5], [2.0], [2.5], [3.0], [3.5], [4.0]], "y": [0, 0, 1, 0, 1, 0, 1, 1], "C": 1}),
               _ex({"X": [[1, 2], [2, 1], [3, 3], [4, 5], [5, 4], [6, 6], [1, 4], [5, 1]], "y": [0, 0, 1, 1, 1, 1, 0, 1], "C": 0.5})],
     extra_q=[_bad({"X": [[1], [2]], "y": [1, 1], "C": 1})],
     weak="checked against scikit-learn's fit polished by scipy's root finder on the stationarity equations (a different algorithm for the same optimum)",
     level="expert")


def logistic_predict(w, b, X):
    wv = _vec("w", w)
    _fin("b", b)
    M = _mat("X", X)
    _need(len(M[0]) == len(wv), "w needs one weight per column of X")
    z = [math.fsum(a * c for a, c in zip(r, wv)) + b for r in M]
    return _out(probabilities=[_sigmoid(v) for v in z], classes=[1 if v > 0 else 0 for v in z], logits=z)


def _ref_logistic_predict(w, b, X):
    np = _np()
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression()
    m.coef_, m.intercept_, m.classes_ = np.array([w], dtype=float), np.array([b], dtype=float), np.array([0, 1])
    A = np.array(X, dtype=float)
    return {"probabilities": _f(m.predict_proba(A)[:, 1]), "classes": [int(v) for v in m.predict(A)], "logits": _f(m.decision_function(A))}


_add("ml_logistic_regression_predict",
     name="Logistic regression prediction from given weights", topic="linear models",
     purpose="The logits, predicted probabilities of class 1 and predicted classes of a logistic regression model with given weights and intercept.",
     inputs=[_vecin("w", "weights"), _numin("b", "intercept"), _matin("X", "samples to predict: one row each")],
     outputs=[_o("probabilities", "sigma(w.x + b) for each sample"), _o("classes", "1 when w.x + b > 0, else 0"), _o("logits", "w.x + b")],
     primary="probabilities", method="stable sigmoid of the linear score",
     latex=r"P(y = 1 \mid x) = \sigma(w^\top x + b)",
     run=logistic_predict, reference=_ref_logistic_predict,
     wrong=lambda w, b, X: (lambda o: dict(o, probabilities=[_r(_sigmoid(v - 2 * b)) for v in o["logits"]]))(logistic_predict(w, b, X)) if abs(b) > 0.05 else _out(probabilities=[0.0], classes=[0], logits=[0.0]),
     rand=lambda rng: (lambda d: {"w": [round(rng.uniform(-2, 2), 2) for _ in range(d)], "b": round(rng.uniform(-2, 2), 2),
                                  "X": [[round(rng.uniform(-3, 3), 2) for _ in range(d)] for _ in range(rng.randint(1, 6))]})(rng.randint(1, 4)),
     phrases=["Logistic regression prediction", "Logistic regression predict"],
     conditions=["w has one weight per feature"],
     examples=[_ex({"w": [0.8, -0.4], "b": -0.2, "X": [[1, 2], [3, 1], [0, 0]]}, {"classes": [0, 1, 0]}),
               _ex({"w": [2], "b": -3, "X": [[1], [1.5], [2]]})],
     level="beginner")


def ridge(X, y, alpha):
    M = _mat("X", X, 2, 1)
    yv = _vec("y", y)
    n, d = len(M), len(M[0])
    _need(len(yv) == n, "y needs one value per row of X")
    _fin("alpha", alpha)
    _need(alpha >= 0, "alpha must be >= 0")
    F = [[_F(v) for v in r] for r in M]
    Y = [_F(v) for v in yv]
    mx = [sum(r[j] for r in F) / n for j in range(d)]
    my = sum(Y) / n
    Xc = [[r[j] - mx[j] for j in range(d)] for r in F]
    G = [[sum(r[i] * r[j] for r in Xc) + (_F(alpha) if i == j else 0) for j in range(d)] for i in range(d)]
    h = [sum(r[i] * (t - my) for r, t in zip(Xc, Y)) for i in range(d)]
    w = _solve_exact(G, h)
    b = my - sum(a * c for a, c in zip(mx, w))
    mse = sum((sum(a * c for a, c in zip(r, w)) + b - t) ** 2 for r, t in zip(F, Y)) / n
    return _out(w=w, b=b, train_mse=mse)


def _ref_ridge(X, y, alpha):
    np = _np()
    from sklearn.linear_model import Ridge
    A, t = np.array(X, dtype=float), np.array(y, dtype=float)
    m = Ridge(alpha=alpha, solver="cholesky").fit(A, t)
    return {"w": _f(m.coef_), "b": float(m.intercept_), "train_mse": float(np.mean((m.predict(A) - t) ** 2))}


_add("ml_ridge_regression",
     name="Ridge regression coefficients (stated alpha)", topic="linear models",
     purpose="Fit ridge regression: minimise ||y - X w - b||^2 + alpha ||w||^2 (the intercept is not penalised); give the coefficients, the intercept and the training MSE.",
     inputs=[_matin("X", "features: one row per sample"), _vecin("y", "targets"), _numin("alpha", "L2 penalty strength (>= 0; 0 is least squares)")],
     outputs=[_o("w", "coefficients"), _o("b", "intercept"), _o("train_mse", "training mean squared error")],
     primary="train_mse", method="centre X and y, solve (Xc^T Xc + alpha I) w = Xc^T yc exactly, b = mean(y) - mean(X) w",
     latex=r"\hat w = (X_c^\top X_c + \alpha I)^{-1} X_c^\top y_c",
     run=ridge, reference=_ref_ridge,
     wrong=lambda X, y, alpha: ridge(X, y, alpha / len(X)) if alpha > 0.05 else ridge(X, y, alpha + 1.0),
     rand=lambda rng: dict(_rand_ols(rng, 5, 20), alpha=rng.choice([0.1, 1.0, 10.0, round(rng.uniform(0.05, 20), 2)])),
     phrases=["Ridge regression"],
     conditions=["alpha >= 0", "a non-singular system (alpha > 0 always is)"],
     examples=[_ex({"X": [[1], [2], [3], [4], [5]], "y": [1.2, 1.9, 3.1, 3.9, 5.2], "alpha": 1}),
               _ex({"X": [[1, 0], [2, 1], [3, 1], [4, 3], [5, 2], [6, 5]], "y": [2, 4, 5, 9, 9, 14], "alpha": 0.5})],
     extra_q=[_bad({"X": [[1], [2]], "y": [1, 2], "alpha": -1})],
     level="advanced")


def lasso(X, y, alpha):
    M = _mat("X", X, 3, 1)
    yv = _vec("y", y)
    n, d = len(M), len(M[0])
    _need(len(yv) == n, "y needs one value per row of X")
    _fin("alpha", alpha)
    _need(alpha > 0, "alpha must be positive")
    mx = [math.fsum(r[j] for r in M) / n for j in range(d)]
    my = math.fsum(yv) / n
    Xc = [[r[j] - mx[j] for j in range(d)] for r in M]
    yc = [v - my for v in yv]
    nrm = [math.fsum(r[j] ** 2 for r in Xc) for j in range(d)]
    _need(all(v > 0 for v in nrm), "a feature is constant")
    w = [0.0] * d
    res = list(yc)
    lam = n * alpha
    for _ in range(20000):
        delta = 0.0
        for j in range(d):
            rho = math.fsum(Xc[i][j] * (res[i] + Xc[i][j] * w[j]) for i in range(n))
            new = math.copysign(max(abs(rho) - lam, 0.0), rho) / nrm[j]
            if new != w[j]:
                for i in range(n):
                    res[i] -= Xc[i][j] * (new - w[j])
                delta = max(delta, abs(new - w[j]))
                w[j] = new
        if delta <= 1e-13 * max(1.0, max(abs(v) for v in w)):
            break
    # exact polish on the active set, then the optimality (KKT) conditions are checked
    A = [j for j in range(d) if w[j] != 0]
    s = {j: (1 if w[j] > 0 else -1) for j in A}
    F = [[_F(v) for v in r] for r in M]
    Y = [_F(v) for v in yv]
    mxF = [sum(r[j] for r in F) / n for j in range(d)]
    myF = sum(Y) / n
    XcF = [[r[j] - mxF[j] for j in range(d)] for r in F]
    ycF = [t - myF for t in Y]
    lamF = n * _F(alpha)
    wF = [Fraction(0)] * d
    if A:
        G = [[sum(r[i] * r[j] for r in XcF) for j in A] for i in A]
        h = [sum(r[i] * t for r, t in zip(XcF, ycF)) - lamF * s[i] for i in A]
        sol = _solve_exact(G, h)
        for j, v in zip(A, sol):
            _need(v != 0 and (v > 0) == (s[j] > 0), "the lasso solution is degenerate (a coefficient sits exactly at zero)")
            wF[j] = v
    resF = [t - sum(r[j] * wF[j] for j in range(d)) for r, t in zip(XcF, ycF)]
    for j in range(d):
        if j not in s:
            c = abs(sum(r[j] * e for r, e in zip(XcF, resF)))
            _need(c < lamF * (1 - Fraction(1, 10 ** 9)), "the lasso solution is degenerate (an inactive feature is exactly at the threshold)")
    b = myF - sum(a * c for a, c in zip(mxF, wF))
    return _out(w=wF, b=b, n_nonzero=len(A))


def _ref_lasso(X, y, alpha):
    np = _np()
    from sklearn.linear_model import LassoLars
    m = LassoLars(alpha=alpha, max_iter=10000).fit(np.array(X, dtype=float), np.array(y, dtype=float))
    coef = [0.0 if abs(v) < 1e-14 else float(v) for v in m.coef_]
    return {"w": coef, "b": float(m.intercept_), "n_nonzero": int(sum(1 for v in coef if v != 0))}


def _rand_lasso(rng):
    n, d = rng.randint(8, 25), rng.randint(2, 4)
    w = [rng.choice([0.0, rng.uniform(-3, 3)]) for _ in range(d)]
    X = [[round(rng.uniform(-3, 3), 2) for _ in range(d)] for _ in range(n)]
    y = [round(math.fsum(a * b for a, b in zip(r, w)) + rng.gauss(0, 1), 2) for r in X]
    return {"X": X, "y": y, "alpha": rng.choice([0.1, 0.5, 1.0, round(rng.uniform(0.05, 1.5), 2)])}


_add("ml_lasso_regression",
     name="Lasso regression coefficients (stated alpha)", topic="linear models",
     purpose="Fit lasso regression: minimise ||y - X w - b||^2 / (2n) + alpha ||w||_1 (the intercept is not penalised); give the coefficients (some exactly zero), the intercept and the number of non-zero coefficients.",
     inputs=[_matin("X", "features: one row per sample"), _vecin("y", "targets"), _numin("alpha", "L1 penalty strength (> 0)")],
     outputs=[_o("w", "coefficients"), _o("b", "intercept"), _o("n_nonzero", "number of non-zero coefficients")],
     primary="b", method="coordinate descent to find the active set and signs, then the exact solution on that set; the optimality conditions are checked",
     latex=r"\min_{w,b}\ \frac{1}{2n}\|y - Xw - b\|^2 + \alpha\|w\|_1",
     run=lasso, reference=_ref_lasso,
     wrong=lambda X, y, alpha: lasso(X, y, alpha * len(X)) if False else (lambda o: dict(o, w=[_r(v * 0.9) for v in o["w"]], b=_r(o["b"] + 0.1)))(lasso(X, y, alpha)),
     rand=_rand_lasso, phrases=["Lasso regression", "Lasso"],
     conditions=["alpha > 0", "non-constant features", "a non-degenerate solution"],
     examples=[_ex({"X": [[1, 0.5], [2, 1.5], [3, 1.0], [4, 3.0], [5, 2.0], [6, 3.5]], "y": [1.1, 2.3, 2.9, 4.2, 5.1, 5.8], "alpha": 0.1}),
               _ex({"X": [[1, 2, 0], [2, 1, 1], [3, 3, 0], [4, 1, 1], [5, 4, 0], [6, 2, 1], [7, 5, 1]], "y": [3, 4, 6, 6, 9, 8, 12], "alpha": 0.5})],
     level="expert")


def svm_decision(w, b, X):
    wv = _vec("w", w)
    _fin("b", b)
    M = _mat("X", X)
    _need(len(M[0]) == len(wv), "w needs one weight per column of X")
    nw = math.sqrt(math.fsum(v * v for v in wv))
    _need(nw > 0, "w must not be the zero vector")
    f = [sum(_F(a) * _F(c) for a, c in zip(r, wv)) + _F(b) for r in M]
    return _out(decision=f, classes=[1 if v > 0 else -1 for v in f], margin_width=2 / nw, distances=[abs(float(v)) / nw for v in f])


def _ref_svm(w, b, X):
    np = _np()
    from sklearn.svm import LinearSVC
    m = LinearSVC()
    m.coef_, m.intercept_, m.classes_ = np.array([w], dtype=float), np.array([b], dtype=float), np.array([-1, 1])
    A = np.array(X, dtype=float)
    f = m.decision_function(A)
    nw = float(np.linalg.norm(w))
    return {"decision": _f(f), "classes": [int(v) for v in m.predict(A)], "margin_width": 2 / nw, "distances": _f(np.abs(f) / nw)}


_add("ml_linear_svm_decision",
     name="Linear SVM decision values, classes and margin", topic="linear models",
     purpose="For a linear SVM with given weights and bias: the decision value w.x + b of each sample, its class (+1 when the value is positive, else -1), the margin width 2 / ||w|| and each sample's distance to the hyperplane.",
     inputs=[_vecin("w", "weights"), _numin("b", "bias"), _matin("X", "samples: one row each")],
     outputs=[_o("decision", "w.x + b"), _o("classes", "+1 if w.x + b > 0, else -1"), _o("margin_width", "2 / ||w||"), _o("distances", "|w.x + b| / ||w||")],
     primary="margin_width", method="exact decision values; one square root for ||w||",
     latex=r"f(x) = w^\top x + b,\quad \text{margin} = \frac{2}{\|w\|}",
     run=svm_decision, reference=_ref_svm,
     wrong=lambda w, b, X: (lambda o: dict(o, margin_width=_r(1 / math.sqrt(math.fsum(v * v for v in w)))))(svm_decision(w, b, X)),
     rand=lambda rng: (lambda d: {"w": [round(rng.uniform(-2, 2), 2) for _ in range(d)], "b": round(rng.uniform(-2, 2), 2),
                                  "X": [[round(rng.uniform(-3, 3), 2) for _ in range(d)] for _ in range(rng.randint(1, 6))]})(rng.randint(1, 4)),
     phrases=["Linear SVM decision", "SVM decision values"],
     conditions=["w is not zero"],
     examples=[_ex({"w": [2, -1], "b": -1, "X": [[1, 0], [0, 1], [2, 2]]}, {"decision": [1.0, -2.0, 1.0], "classes": [1, -1, 1], "margin_width": 0.894427191}),
               _ex({"w": [0.5, 0.5, 1], "b": 0.25, "X": [[1, 1, -1], [0, 0, 0]]})],
     weak="decision values and classes by scikit-learn's LinearSVC with the given coefficients; the margin width and distances by numpy's norm of the same weights",
     level="intermediate")


def perceptron_epoch(X, y, w, b, lr):
    M = _mat("X", X)
    yl = _labels("y", y)
    _need(len(yl) == len(M), "y needs one label per row of X")
    _need(all(v in (-1, 1) for v in yl) and len(set(yl)) == 2, "labels must be -1 and +1, both present")
    wv = [_F(v) for v in _vec("w", w)]
    _need(len(wv) == len(M[0]), "w needs one weight per column of X")
    _fin("b", b), _fin("lr", lr)
    _need(lr > 0, "the learning rate must be positive")
    B, L = _F(b), _F(lr)
    mistakes = 0
    for r, t in zip(M, yl):
        f = sum(a * _F(c) for a, c in zip(wv, r)) + B
        if t * f <= 0:
            wv = [a + L * t * _F(c) for a, c in zip(wv, r)]
            B += L * t
            mistakes += 1
    return _out(w_new=wv, b_new=B, n_updates=mistakes)


def _ref_perceptron(X, y, w, b, lr):
    import warnings
    np = _np()
    from sklearn.linear_model import Perceptron
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Perceptron(eta0=lr, max_iter=1, shuffle=False, tol=None, penalty=None, fit_intercept=True)
        m.fit(np.array(X, dtype=float), np.array(y, dtype=int), coef_init=np.array([w], dtype=float), intercept_init=np.array([b], dtype=float))
    return {"w_new": _f(m.coef_[0]), "b_new": float(m.intercept_[0])}


def _rand_perceptron(rng):
    while True:
        n, d = rng.randint(4, 12), rng.randint(1, 3)
        wt = [rng.uniform(-2, 2) for _ in range(d)]
        X = [[round(rng.uniform(-3, 3), 1) for _ in range(d)] for _ in range(n)]
        y = [1 if math.fsum(a * c for a, c in zip(r, wt)) + 0.2 > 0 else -1 for r in X]
        if len(set(y)) == 2:
            return {"X": X, "y": y, "w": [round(rng.uniform(-1, 1), 1) for _ in range(d)], "b": round(rng.uniform(-1, 1), 1),
                    "lr": rng.choice([1.0, 0.5, 0.1])}


_add("ml_perceptron_epoch",
     name="Perceptron training epoch", topic="linear models",
     purpose="One pass of the perceptron learning rule through the samples in order (labels -1/+1): whenever y (w.x + b) <= 0, update w += lr y x and b += lr y; the final weights, bias and number of updates.",
     inputs=[_matin("X", "samples in presentation order"), _vecin("y", "labels (-1 or +1)"), _vecin("w", "initial weights"), _numin("b", "initial bias"),
             _numin("lr", "learning rate (> 0)")],
     outputs=[_o("w_new", "weights after the epoch"), _o("b_new", "bias after the epoch"), _o("n_updates", "number of updates (mistakes, including y f = 0)")],
     primary="b_new", method="exact rational arithmetic, samples in the given order",
     latex=r"\text{if } y_i(w^\top x_i + b) \le 0:\ w \leftarrow w + \eta y_i x_i,\ b \leftarrow b + \eta y_i",
     run=perceptron_epoch, reference=_ref_perceptron,
     wrong=lambda X, y, w, b, lr: (lambda o: dict(o, b_new=_r(o["b_new"] + lr), w_new=[_r(v + lr * 0.5) for v in o["w_new"]]))(perceptron_epoch(X, y, w, b, lr)),
     rand=_rand_perceptron, phrases=["Perceptron epoch", "Perceptron training epoch"],
     conditions=["labels -1 and +1", "lr > 0"],
     examples=[_ex({"X": [[2, 1], [-1, -2], [1, -1], [-2, 1]], "y": [1, -1, 1, -1], "w": [0, 0], "b": 0, "lr": 1}),
               _ex({"X": [[1], [3], [-1], [-2]], "y": [1, 1, -1, -1], "w": [-0.5], "b": 0.5, "lr": 0.5})],
     weak="the weights and bias are checked by scikit-learn's Perceptron (one epoch, no shuffling); the update count by the shipped simulation only (no library reports it)",
     level="intermediate")


# ================================================================================================ neural networks
def _act(name, v):
    if name == "relu":
        return v if v > 0 else 0.0
    if name == "tanh":
        return math.tanh(v)
    return _sigmoid(v)


def _dact(name, z, a):
    if name == "relu":
        return 1.0 if z > 0 else 0.0
    if name == "tanh":
        return 1.0 - a * a
    return a * (1.0 - a)


def _net_inputs(x, W1, b1, W2, b2):
    xv = _vec("x", x)
    A, B = _mat("W1", W1), _mat("W2", W2)
    c1, c2 = _vec("b1", b1), _vec("b2", b2)
    _need(len(A[0]) == len(xv), "W1 needs one column per input (shape hidden x inputs)")
    _need(len(c1) == len(A), "b1 needs one entry per row of W1")
    _need(len(B[0]) == len(A), "W2 needs one column per hidden unit (shape outputs x hidden)")
    _need(len(c2) == len(B), "b2 needs one entry per row of W2")
    return xv, A, c1, B, c2


def _forward(xv, A, c1, B, c2, activation):
    z1 = [math.fsum(a * v for a, v in zip(r, xv)) + b for r, b in zip(A, c1)]
    h = [_act(activation, v) for v in z1]
    z2 = [math.fsum(a * v for a, v in zip(r, h)) + b for r, b in zip(B, c2)]
    return z1, h, z2


def dense_forward(x, W1, b1, activation, W2, b2, output):
    xv, A, c1, B, c2 = _net_inputs(x, W1, b1, W2, b2)
    _, h, z2 = _forward(xv, A, c1, B, c2, activation)
    if output == "sigmoid":
        y = [_sigmoid(v) for v in z2]
    elif output == "softmax":
        y = _softmax(z2)
    else:
        y = z2
    return _out(hidden=h, output=y, max_output=max(y))


def _ref_dense_forward(x, W1, b1, activation, W2, b2, output):
    import torch
    T = lambda v: torch.tensor(v, dtype=torch.float64)
    l1, l2 = torch.nn.Linear(len(x), len(b1), dtype=torch.float64), torch.nn.Linear(len(b1), len(b2), dtype=torch.float64)
    with torch.no_grad():
        l1.weight.copy_(T(W1)), l1.bias.copy_(T(b1)), l2.weight.copy_(T(W2)), l2.bias.copy_(T(b2))
        act = {"relu": torch.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid}[activation]
        h = act(l1(T(x)))
        z = l2(h)
        y = torch.sigmoid(z) if output == "sigmoid" else (torch.softmax(z, 0) if output == "softmax" else z)
    return {"hidden": h.tolist(), "output": y.tolist(), "max_output": float(y.max())}


def _rand_net(rng, loss=None):
    d, hdim, o = rng.randint(1, 4), rng.randint(1, 5), rng.randint(1, 3)
    R = lambda: round(rng.uniform(-1.5, 1.5), 2)
    while True:
        x = [R() for _ in range(d)]
        W1 = [[R() for _ in range(d)] for _ in range(hdim)]
        b1 = [R() for _ in range(hdim)]
        z1 = [math.fsum(a * v for a, v in zip(r, x)) + b for r, b in zip(W1, b1)]
        if all(abs(v) > 1e-9 for v in z1):
            break
    out = {"x": x, "W1": W1, "b1": b1, "activation": rng.choice(["relu", "tanh", "sigmoid"]),
           "W2": [[R() for _ in range(hdim)] for _ in range(o)], "b2": [R() for _ in range(o)]}
    if loss is None:
        out["output"] = rng.choice(["identity", "sigmoid", "softmax"])
    else:
        lo = rng.choice(["mse", "cross_entropy"])
        out["loss"] = lo
        if lo == "cross_entropy":
            if o == 1:
                out["W2"].append([R() for _ in range(hdim)]), out["b2"].append(R())
            out["t"] = _rand_dist(rng, len(out["b2"]), 2)
        else:
            out["t"] = [R() for _ in range(o)]
    return out


_NET_IN = [_vecin("x", "input vector"), _matin("W1", "hidden-layer weights (rows = hidden units, columns = inputs)"), _vecin("b1", "hidden-layer biases"),
           _choice("activation", ["relu", "tanh", "sigmoid"], "hidden activation: relu, tanh or sigmoid"),
           _matin("W2", "output-layer weights (rows = outputs, columns = hidden units)"), _vecin("b2", "output biases")]

_add("ml_dense_forward_pass",
     name="Forward pass of a one-hidden-layer dense network", topic="neural networks",
     purpose="The forward pass of a dense network x -> activation(W1 x + b1) -> W2 h + b2 with an identity, sigmoid or softmax output: the hidden activations and the outputs.",
     inputs=_NET_IN + [_choice("output", ["identity", "sigmoid", "softmax"], "output activation: identity, sigmoid or softmax")],
     outputs=[_o("hidden", "hidden activations h"), _o("output", "network outputs"), _o("max_output", "largest output")],
     primary="max_output", method="matrix-vector products with fsum, then the activations",
     latex=r"h = \phi(W_1 x + b_1),\quad \hat y = g(W_2 h + b_2)",
     run=dense_forward, reference=_ref_dense_forward,
     wrong=lambda x, W1, b1, activation, W2, b2, output: dense_forward(x, W1, [0.0] * len(b1), activation, W2, b2, output),
     rand=_rand_net, phrases=["Dense network forward pass", "Neural network forward pass"],
     conditions=["shapes agree (W1: hidden x inputs, W2: outputs x hidden)"],
     examples=[_ex({"x": [1, 2], "W1": [[0.5, -0.25], [0.1, 0.3]], "b1": [0.1, -0.2], "activation": "relu", "W2": [[1, -1]], "b2": [0.05], "output": "identity"},
                   {"hidden": [0.1, 0.5], "output": [-0.35]}),
               _ex({"x": [0.5, -1, 2], "W1": [[0.2, 0.4, -0.1], [-0.3, 0.1, 0.5]], "b1": [0, 0.1], "activation": "tanh",
                    "W2": [[0.7, -0.2], [0.1, 0.9], [-0.5, 0.3]], "b2": [0, 0.1, -0.1], "output": "softmax"})],
     extra_q=[_bad({"x": [1, 2], "W1": [[0.5, -0.25, 1]], "b1": [0.1], "activation": "relu", "W2": [[1]], "b2": [0], "output": "identity"})],
     level="intermediate")


def backprop(x, t, W1, b1, activation, W2, b2, loss):
    xv, A, c1, B, c2 = _net_inputs(x, W1, b1, W2, b2)
    tv = _vec("t", t)
    _need(len(tv) == len(B), "t needs one target per output")
    z1, h, z2 = _forward(xv, A, c1, B, c2, activation)
    if activation == "relu":
        _need(all(v != 0 for v in z1), "a ReLU pre-activation is exactly 0 (its derivative is undefined there)")
    o = len(z2)
    if loss == "mse":
        L = math.fsum((a - b) ** 2 for a, b in zip(z2, tv)) / o
        d2 = [2 * (a - b) / o for a, b in zip(z2, tv)]
    else:
        _need(o >= 2, "cross-entropy needs at least two outputs")
        _need(all(v >= 0 for v in tv) and abs(math.fsum(tv) - 1) <= 1e-9, "the target t must be a probability distribution for cross-entropy")
        lp = _log_softmax(z2)
        L = -math.fsum(a * b for a, b in zip(tv, lp))
        p = _softmax(z2)
        d2 = [pi - ti for pi, ti in zip(p, tv)]
    dW2 = [[d * hj for hj in h] for d in d2]
    dh = [math.fsum(d2[k] * B[k][j] for k in range(o)) for j in range(len(h))]
    d1 = [g * _dact(activation, z, a) for g, z, a in zip(dh, z1, h)]
    dW1 = [[d * xi for xi in xv] for d in d1]
    return _out(loss=L, dW1=dW1, db1=d1, dW2=dW2, db2=d2)


def _ref_backprop(x, t, W1, b1, activation, W2, b2, loss):
    import torch
    T = lambda v, g=True: torch.tensor(v, dtype=torch.float64, requires_grad=g)
    P = [T(W1), T(b1), T(W2), T(b2)]
    act = {"relu": torch.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid}[activation]
    z = P[2] @ act(P[0] @ T(x, False) + P[1]) + P[3]
    if loss == "mse":
        L = torch.nn.functional.mse_loss(z, T(t, False))
    else:
        L = torch.nn.functional.cross_entropy(z[None], T([t], False))
    L.backward()
    return {"loss": L.item(), "dW1": P[0].grad.tolist(), "db1": P[1].grad.tolist(), "dW2": P[2].grad.tolist(), "db2": P[3].grad.tolist()}


def _wrong_backprop(x, t, W1, b1, activation, W2, b2, loss):
    o = backprop(x, t, W1, b1, activation, W2, b2, loss)
    xv, A, c1, B, c2 = _net_inputs(x, W1, b1, W2, b2)
    z1, h, z2 = _forward(xv, A, c1, B, c2, activation)
    d2 = o["db2"]
    dh = [math.fsum(d2[k] * B[k][j] for k in range(len(d2))) for j in range(len(h))]       # activation derivative forgotten
    return dict(o, db1=_r(dh), dW1=_r([[d * xi for xi in xv] for d in dh]))


_add("ml_backpropagation",
     name="Backpropagation gradients of a one-hidden-layer network", topic="neural networks",
     purpose="One backpropagation pass of a dense network with one hidden layer: the loss (mean squared error with identity outputs, or cross-entropy of softmax outputs against a target distribution) and its gradients with respect to every weight and bias.",
     inputs=[_vecin("x", "input vector"), _vecin("t", "target: values for mse, a probability distribution for cross-entropy")] + _NET_IN[1:] +
            [_choice("loss", ["mse", "cross_entropy"], "loss: mse (identity outputs, mean over outputs) or cross_entropy (softmax outputs)")],
     outputs=[_o("loss", "loss value"), _o("dW1", "dL/dW1"), _o("db1", "dL/db1"), _o("dW2", "dL/dW2"), _o("db2", "dL/db2")],
     primary="loss", method="chain rule backwards through the output layer, the activation and the hidden layer",
     latex=r"\delta_2 = \frac{\partial L}{\partial z_2},\ \nabla_{W_2} = \delta_2 h^\top,\ \delta_1 = (W_2^\top \delta_2) \odot \phi'(z_1),\ \nabla_{W_1} = \delta_1 x^\top",
     run=backprop, reference=_ref_backprop, wrong=_wrong_backprop,
     rand=lambda rng: _rand_net(rng, loss=True), phrases=["Backpropagation gradients", "Backpropagation"],
     conditions=["shapes agree", "no ReLU pre-activation exactly 0", "cross-entropy: target sums to 1, at least two outputs"],
     examples=[_ex({"x": [1, 2], "t": [0.5], "W1": [[0.5, -0.25], [0.1, 0.3]], "b1": [0.1, -0.2], "activation": "relu", "W2": [[1, -1]], "b2": [0.05], "loss": "mse"},
                   {"loss": 0.7225}),
               _ex({"x": [0.5, -1], "t": [0, 1, 0], "W1": [[0.2, 0.4], [-0.3, 0.1]], "b1": [0, 0.1], "activation": "sigmoid",
                    "W2": [[0.7, -0.2], [0.1, 0.9], [-0.5, 0.3]], "b2": [0, 0.1, -0.1], "loss": "cross_entropy"})],
     level="advanced")


def dense_params(sizes, bias):
    s = _labels("sizes", sizes, 2)
    _need(all(v >= 1 for v in s), "layer sizes must be positive")
    per = [a * b + (b if bias else 0) for a, b in zip(s, s[1:])]
    return _out(per_layer=per, total=sum(per))


def _ref_dense_params(sizes, bias):
    import torch
    s = [int(v) for v in sizes]
    layers = [torch.nn.Linear(a, b, bias=bias, device="meta") for a, b in zip(s, s[1:])]
    per = [sum(p.numel() for p in l.parameters()) for l in layers]
    return {"per_layer": per, "total": sum(per)}


_add("ml_dense_parameter_count",
     name="Parameter count of a dense (fully connected) network", topic="neural networks",
     purpose="The number of trainable parameters of each dense layer and of the whole network, from the layer sizes (input size first), with or without biases.",
     inputs=[_vecin("sizes", "layer sizes from input to output, e.g. [784, 128, 10]", 2), _boolin("bias", "whether every layer has biases (yes or no)")],
     outputs=[_o("per_layer", "parameters of each layer: n_in n_out (+ n_out biases)"), _o("total", "total parameters")],
     primary="total", method="n_in * n_out + n_out per layer",
     latex=r"P = \sum_l (n_{l-1} n_l + n_l)",
     run=dense_params, reference=_ref_dense_params,
     wrong=lambda sizes, bias: (lambda s: _out(per_layer=[a * b for a, b in zip(s, s[1:])], total=sum(a * b + a for a, b in zip(s, s[1:]))))([int(v) for v in sizes]),
     rand=lambda rng: {"sizes": [rng.choice([rng.randint(1, 64), rng.randint(64, 1024)]) for _ in range(rng.randint(2, 6))], "bias": rng.choice([True, False])},
     phrases=["Dense network parameter count", "Parameter count of a dense network"],
     conditions=["positive whole-number sizes"],
     examples=[_ex({"sizes": [784, 128, 10], "bias": True}, {"per_layer": [100480, 1290], "total": 101770}),
               _ex({"sizes": [3, 4, 4, 1], "bias": False}, {"total": 32})],
     level="beginner")


def conv_out(n, kernel, stride, padding, dilation):
    for nm, v, lo in (("n", n, 1), ("kernel", kernel, 1), ("stride", stride, 1), ("padding", padding, 0), ("dilation", dilation, 1)):
        _need(isinstance(v, int) and v >= lo, f"{nm} must be a whole number >= {lo}")
    eff = dilation * (kernel - 1) + 1
    _need(n + 2 * padding >= eff, "the (dilated) kernel is larger than the padded input")
    return _out(output_size=(n + 2 * padding - eff) // stride + 1, effective_kernel=eff)


def _ref_conv_out(n, kernel, stride, padding, dilation):
    import torch
    y = torch.nn.functional.conv1d(torch.zeros(1, 1, n), torch.zeros(1, 1, kernel), stride=stride, padding=padding, dilation=dilation)
    return {"output_size": int(y.shape[-1]), "effective_kernel": dilation * (kernel - 1) + 1}


_add("ml_conv_output_size",
     name="Convolution output size", topic="neural networks",
     purpose="The output length (per spatial dimension) of a convolution or pooling window from the input size, kernel size, stride, zero padding and dilation.",
     inputs=[_intin("n", "input size", 1), _intin("kernel", "kernel size", 1), _intin("stride", "stride", 1), _intin("padding", "zero padding on each side", 0),
             _intin("dilation", "dilation (1 = ordinary)", 1)],
     outputs=[_o("output_size", "floor((n + 2p - d(k - 1) - 1) / s) + 1"), _o("effective_kernel", "d(k - 1) + 1")],
     primary="output_size", method="integer floor division",
     latex=r"n_{\mathrm{out}} = \left\lfloor \frac{n + 2p - d(k-1) - 1}{s} \right\rfloor + 1",
     run=conv_out, reference=_ref_conv_out,
     wrong=lambda n, kernel, stride, padding, dilation: _out(output_size=(n + padding - kernel) // stride + 1, effective_kernel=kernel),
     rand=lambda rng: (lambda k, d: {"n": rng.randint(d * (k - 1) + 1, 300), "kernel": k, "stride": rng.randint(1, 4), "padding": rng.randint(1, 3), "dilation": d})(rng.choice([1, 3, 5, 7, 2]), rng.choice([1, 1, 2, 3])),
     phrases=["Convolution output size", "Conv output size"],
     conditions=["the dilated kernel fits in the padded input"],
     examples=[_ex({"n": 32, "kernel": 3, "stride": 1, "padding": 1, "dilation": 1}, {"output_size": 32}),
               _ex({"n": 224, "kernel": 7, "stride": 2, "padding": 3, "dilation": 1}, {"output_size": 112}),
               _ex({"n": 64, "kernel": 3, "stride": 1, "padding": 0, "dilation": 2}, {"output_size": 60})],
     extra_q=[_bad({"n": 2, "kernel": 5, "stride": 1, "padding": 0, "dilation": 1})],
     level="beginner")


def conv_params(in_channels, out_channels, kernel_h, kernel_w, groups, bias):
    for nm, v in (("in_channels", in_channels), ("out_channels", out_channels), ("kernel_h", kernel_h), ("kernel_w", kernel_w), ("groups", groups)):
        _need(isinstance(v, int) and v >= 1, f"{nm} must be a whole number >= 1")
    _need(in_channels % groups == 0 and out_channels % groups == 0, "in_channels and out_channels must both be divisible by groups")
    w = out_channels * (in_channels // groups) * kernel_h * kernel_w
    return _out(weights=w, biases=out_channels if bias else 0, total=w + (out_channels if bias else 0))


def _ref_conv_params(in_channels, out_channels, kernel_h, kernel_w, groups, bias):
    import torch
    c = torch.nn.Conv2d(in_channels, out_channels, (kernel_h, kernel_w), groups=groups, bias=bias, device="meta")
    return {"weights": c.weight.numel(), "biases": c.bias.numel() if bias else 0, "total": sum(p.numel() for p in c.parameters())}


_add("ml_conv_parameter_count",
     name="Parameter count of a 2-D convolution layer", topic="neural networks",
     purpose="The number of weights, biases and total parameters of a 2-D convolution layer, including grouped and depthwise convolutions.",
     inputs=[_intin("in_channels", "input channels", 1), _intin("out_channels", "output channels (filters)", 1), _intin("kernel_h", "kernel height", 1),
             _intin("kernel_w", "kernel width", 1), _intin("groups", "groups (1 = ordinary; in_channels = depthwise)", 1), _boolin("bias", "whether the layer has a bias per filter")],
     outputs=[_o("weights", "C_out (C_in / g) k_h k_w"), _o("biases", "C_out or 0"), _o("total", "weights + biases")],
     primary="total", method="integer arithmetic",
     latex=r"P = C_{\mathrm{out}}\,\frac{C_{\mathrm{in}}}{g}\,k_h k_w + C_{\mathrm{out}}",
     run=conv_params, reference=_ref_conv_params,
     wrong=lambda in_channels, out_channels, kernel_h, kernel_w, groups, bias: _out(weights=out_channels * in_channels * kernel_h * kernel_w, biases=0, total=out_channels * in_channels * kernel_h * kernel_w + in_channels),
     rand=lambda rng: (lambda g: {"in_channels": g * rng.randint(1, 32), "out_channels": g * rng.randint(1, 32), "kernel_h": rng.choice([1, 3, 5, 7]),
                                  "kernel_w": rng.choice([1, 3, 5, 7]), "groups": g, "bias": rng.choice([True, False])})(rng.choice([1, 1, 2, 4])),
     phrases=["Convolution parameter count", "Conv layer parameter count"],
     conditions=["channels divisible by groups"],
     examples=[_ex({"in_channels": 3, "out_channels": 64, "kernel_h": 3, "kernel_w": 3, "groups": 1, "bias": True}, {"total": 1792}),
               _ex({"in_channels": 32, "out_channels": 32, "kernel_h": 3, "kernel_w": 3, "groups": 32, "bias": False}, {"total": 288})],
     extra_q=[_bad({"in_channels": 3, "out_channels": 64, "kernel_h": 3, "kernel_w": 3, "groups": 2, "bias": True})],
     level="beginner")


def conv2d(image, kernel, stride, padding):
    I, K = _mat("image", image), _mat("kernel", kernel)
    _need(isinstance(stride, int) and stride >= 1, "stride must be a whole number >= 1")
    _need(isinstance(padding, int) and padding >= 0, "padding must be a whole number >= 0")
    H, W, kh, kw = len(I), len(I[0]), len(K), len(K[0])
    _need(H + 2 * padding >= kh and W + 2 * padding >= kw, "the kernel is larger than the padded image")
    P = [[Fraction(0)] * (W + 2 * padding) for _ in range(H + 2 * padding)]
    for i in range(H):
        for j in range(W):
            P[i + padding][j + padding] = _F(I[i][j])
    KF = [[_F(v) for v in r] for r in K]
    oh, ow = (H + 2 * padding - kh) // stride + 1, (W + 2 * padding - kw) // stride + 1
    out = [[sum(P[i * stride + a][j * stride + b] * KF[a][b] for a in range(kh) for b in range(kw)) for j in range(ow)] for i in range(oh)]
    return _out(output=out, output_sum=sum(sum(r) for r in out))


def _ref_conv2d(image, kernel, stride, padding):
    import torch
    y = torch.nn.functional.conv2d(torch.tensor([[image]], dtype=torch.float64), torch.tensor([[kernel]], dtype=torch.float64), stride=stride, padding=padding)[0, 0]
    return {"output": y.tolist(), "output_sum": float(y.sum())}


def _rand_conv2d(rng):
    H, W = rng.randint(3, 7), rng.randint(3, 7)
    kh, kw = rng.randint(1, 3), rng.randint(1, 3)
    return {"image": [[rng.randint(-5, 9) for _ in range(W)] for _ in range(H)], "kernel": [[rng.choice([-1, 0, 1, 2, 0.5]) for _ in range(kw)] for _ in range(kh)],
            "stride": rng.randint(1, 2), "padding": rng.randint(0, 1)}


_add("ml_conv2d_forward",
     name="2-D convolution of an image with a kernel (deep-learning convention)", topic="neural networks",
     purpose="The output feature map of a single-channel 2-D convolution as computed in deep learning (cross-correlation: the kernel is not flipped), with a stride and zero padding.",
     inputs=[_matin("image", "input image (rows of pixel values)"), _matin("kernel", "kernel (filter) weights"), _intin("stride", "stride", 1), _intin("padding", "zero padding on each side", 0)],
     outputs=[_o("output", "output feature map"), _o("output_sum", "sum of all outputs")],
     primary="output_sum", method="exact sums of products over each window",
     latex=r"Y_{ij} = \sum_{a,b} X_{is + a,\ js + b}\, K_{ab}",
     run=conv2d, reference=_ref_conv2d,
     wrong=lambda image, kernel, stride, padding: conv2d(image, [list(reversed(r)) for r in reversed(kernel)], stride, padding),
     rand=_rand_conv2d, phrases=["2D convolution", "Conv2d forward"],
     conditions=["the kernel fits in the padded image"],
     examples=[_ex({"image": [[1, 2, 3, 0], [0, 1, 2, 3], [3, 0, 1, 2], [2, 3, 0, 1]], "kernel": [[2, 0, 1], [0, 1, 2], [1, 0, 2]], "stride": 1, "padding": 0},
                   {"output": [[15, 16], [6, 15]]}),
               _ex({"image": [[1, 2, 0], [4, 5, 6], [7, 8, 9]], "kernel": [[1, -1], [0, 2]], "stride": 2, "padding": 1})],
     level="advanced")


def pooling(X, size, stride):
    M = _mat("X", X)
    _need(isinstance(size, int) and size >= 1 and isinstance(stride, int) and stride >= 1, "size and stride must be whole numbers >= 1")
    H, W = len(M), len(M[0])
    _need(H >= size and W >= size, "the pooling window is larger than the input")
    oh, ow = (H - size) // stride + 1, (W - size) // stride + 1
    win = lambda i, j: [_F(M[i * stride + a][j * stride + b]) for a in range(size) for b in range(size)]
    return _out(max_pool=[[max(win(i, j)) for j in range(ow)] for i in range(oh)],
                avg_pool=[[sum(win(i, j)) / (size * size) for j in range(ow)] for i in range(oh)])


def _ref_pool(X, size, stride):
    import torch
    t = torch.tensor([[X]], dtype=torch.float64)
    return {"max_pool": torch.nn.functional.max_pool2d(t, size, stride)[0, 0].tolist(), "avg_pool": torch.nn.functional.avg_pool2d(t, size, stride)[0, 0].tolist()}


_add("ml_pooling",
     name="Max pooling and average pooling", topic="neural networks",
     purpose="Max pooling and average pooling of a 2-D feature map with a square window, a stride and no padding.",
     inputs=[_matin("X", "feature map"), _intin("size", "window size", 1), _intin("stride", "stride", 1)],
     outputs=[_o("max_pool", "maximum of each window"), _o("avg_pool", "mean of each window")],
     primary="max_pool", method="exact window maxima and means",
     latex=r"Y_{ij} = \max_{a,b < k} X_{is+a,\ js+b}",
     run=pooling, reference=_ref_pool,
     wrong=lambda X, size, stride: (lambda o: dict(o, max_pool=[[_r(v - 1) for v in r] for r in o["max_pool"]], avg_pool=[[_r(v * 2) for v in r] for r in o["avg_pool"]]))(pooling(X, size, stride)),
     rand=lambda rng: (lambda H, W: {"X": [[rng.randint(-9, 9) for _ in range(W)] for _ in range(H)], "size": rng.randint(1, min(H, W, 3)), "stride": rng.randint(1, 2)})(rng.randint(2, 7), rng.randint(2, 7)),
     phrases=["Max pooling", "Pooling"],
     conditions=["the window fits in the input"],
     examples=[_ex({"X": [[1, 3, 2, 4], [5, 6, 1, 2], [7, 2, 9, 0], [1, 8, 3, 4]], "size": 2, "stride": 2}, {"max_pool": [[6, 4], [8, 9]], "avg_pool": [[3.75, 2.25], [4.5, 4.0]]}),
               _ex({"X": [[1, 2, 3], [4, 5, 6], [7, 8, 9]], "size": 2, "stride": 1}, {"max_pool": [[5, 6], [8, 9]]})],
     level="intermediate")


def attention(Q, K, V, causal):
    Qm, Km, Vm = _mat("Q", Q), _mat("K", K), _mat("V", V)
    _need(len(Qm[0]) == len(Km[0]), "Q and K need the same number of columns (the key dimension)")
    _need(len(Km) == len(Vm), "K and V need the same number of rows")
    if causal:
        _need(len(Qm) == len(Km), "a causal mask needs as many queries as keys")
    d = len(Qm[0])
    sc = 1 / math.sqrt(d)
    Wt = []
    for i, q in enumerate(Qm):
        s = [math.fsum(a * b for a, b in zip(q, k)) * sc for k in Km]
        if causal:
            s = s[:i + 1]
        p = _softmax(s) + [0.0] * (len(Km) - len(s))
        Wt.append(p)
    O = [[math.fsum(Wt[i][j] * Vm[j][c] for j in range(len(Km))) for c in range(len(Vm[0]))] for i in range(len(Qm))]
    return _out(weights=Wt, output=O, output_sum=math.fsum(v for r in O for v in r))


def _ref_attention(Q, K, V, causal):
    import torch
    T = lambda v: torch.tensor(v, dtype=torch.float64)
    q, k, v = T(Q), T(K), T(V)
    out = torch.nn.functional.scaled_dot_product_attention(q[None], k[None], v[None], is_causal=bool(causal))[0]
    s = q @ k.T / math.sqrt(q.shape[1])
    if causal:
        s = s.masked_fill(torch.ones_like(s, dtype=torch.bool).triu(1), float("-inf"))
    return {"weights": torch.softmax(s, 1).tolist(), "output": out.tolist(), "output_sum": float(out.sum())}


def _rand_attention(rng):
    n, m, d, dv = rng.randint(1, 4), rng.randint(1, 4), rng.randint(1, 4), rng.randint(1, 3)
    causal = rng.random() < 0.4
    if causal:
        m = n
    R = lambda: round(rng.uniform(-2, 2), 2)
    return {"Q": [[R() for _ in range(d)] for _ in range(n)], "K": [[R() for _ in range(d)] for _ in range(m)], "V": [[R() for _ in range(dv)] for _ in range(m)], "causal": causal}


_add("ml_scaled_dot_product_attention",
     name="Scaled dot-product attention", topic="neural networks",
     purpose="Scaled dot-product attention softmax(Q K^T / sqrt(d)) V for given query, key and value matrices, optionally with a causal mask (query i sees keys 0..i): the attention weights and the output.",
     inputs=[_matin("Q", "queries: one row per query"), _matin("K", "keys: one row per key"), _matin("V", "values: one row per key"),
             _boolin("causal", "apply a causal mask (yes or no)")],
     outputs=[_o("weights", "attention weights (rows sum to 1)"), _o("output", "weights V"), _o("output_sum", "sum of all output entries")],
     primary="output_sum", method="scores scaled by 1/sqrt(d), stable softmax per row, weighted sum of values",
     latex=r"\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V",
     run=attention, reference=_ref_attention,
     wrong=lambda Q, K, V, causal: (lambda d: attention([[v * math.sqrt(d) for v in r] for r in Q], K, V, causal))(len(Q[0])) if len(Q[0]) > 1 else (lambda o: dict(o, output_sum=_r(o["output_sum"] + 1)))(attention([[2 * v for v in r] for r in Q], K, V, causal)),
     rand=_rand_attention, phrases=["Scaled dot-product attention", "Attention output"],
     conditions=["Q and K have the same width", "K and V have the same rows", "causal: as many queries as keys"],
     examples=[_ex({"Q": [[1, 0], [0, 1]], "K": [[1, 0], [0, 1], [1, 1]], "V": [[1, 2], [3, 4], [5, 6]], "causal": False}),
               _ex({"Q": [[1, 0], [0, 1], [1, 1]], "K": [[1, 0], [0, 1], [1, 1]], "V": [[1], [2], [3]], "causal": True})],
     level="advanced")


def multihead(Q, K, V, heads):
    Qm, Km, Vm = _mat("Q", Q), _mat("K", K), _mat("V", V)
    D = len(Qm[0])
    _need(len(Km[0]) == D and len(Vm[0]) == D, "Q, K and V need the same width (the model dimension)")
    _need(len(Km) == len(Vm), "K and V need the same number of rows")
    _need(isinstance(heads, int) and heads >= 1 and D % heads == 0, "the model dimension must be divisible by the number of heads")
    dh = D // heads
    out = [[0.0] * D for _ in Qm]
    for h in range(heads):
        sl = slice(h * dh, (h + 1) * dh)
        o = attention([r[sl] for r in Qm], [r[sl] for r in Km], [r[sl] for r in Vm], False)["output"]
        for i in range(len(Qm)):
            out[i][h * dh:(h + 1) * dh] = o[i]
    return _out(output=out, output_sum=math.fsum(v for r in out for v in r))


def _ref_multihead(Q, K, V, heads):
    import torch
    D = len(Q[0])
    m = torch.nn.MultiheadAttention(D, heads, bias=False, batch_first=True, dtype=torch.float64)
    with torch.no_grad():
        m.in_proj_weight.copy_(torch.cat([torch.eye(D, dtype=torch.float64)] * 3))
        m.out_proj.weight.copy_(torch.eye(D, dtype=torch.float64))
        T = lambda v: torch.tensor([v], dtype=torch.float64)
        out, _ = m(T(Q), T(K), T(V), need_weights=False)
    return {"output": out[0].tolist(), "output_sum": float(out.sum())}


def _rand_mha(rng):
    h = rng.choice([1, 2, 2, 3, 4])
    D = h * rng.randint(1, 2)
    n, m = rng.randint(1, 4), rng.randint(1, 4)
    R = lambda: round(rng.uniform(-2, 2), 2)
    return {"Q": [[R() for _ in range(D)] for _ in range(n)], "K": [[R() for _ in range(D)] for _ in range(m)], "V": [[R() for _ in range(D)] for _ in range(m)], "heads": h}


_add("ml_multihead_attention",
     name="Multi-head attention (identity projections)", topic="neural networks",
     purpose="Multi-head attention with identity input and output projections: the model dimension is split into equal heads, scaled dot-product attention runs in each head (scale 1/sqrt(head size)) and the head outputs are concatenated.",
     inputs=[_matin("Q", "queries (rows), width = model dimension"), _matin("K", "keys"), _matin("V", "values"), _intin("heads", "number of heads (divides the model dimension)", 1)],
     outputs=[_o("output", "concatenated head outputs"), _o("output_sum", "sum of all output entries")],
     primary="output_sum", method="per head: softmax(Q_h K_h^T / sqrt(d_h)) V_h; concatenate",
     latex=r"\mathrm{MHA} = [\mathrm{head}_1, \dots, \mathrm{head}_h],\quad \mathrm{head}_i = \mathrm{softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{d/h}}\right)V_i",
     run=multihead, reference=_ref_multihead,
     wrong=lambda Q, K, V, heads: (lambda o: dict(o, output_sum=_r(o["output_sum"] * 1.5 + 0.1)))(multihead(Q, K, V, 1)) if heads == 1 else multihead(Q, K, V, 1),
     rand=_rand_mha, phrases=["Multi-head attention", "Multihead attention"],
     conditions=["equal widths", "model dimension divisible by heads"],
     examples=[_ex({"Q": [[1, 0, 1, 0]], "K": [[1, 0, 0, 1], [0, 1, 1, 0]], "V": [[1, 2, 3, 4], [5, 6, 7, 8]], "heads": 2}),
               _ex({"Q": [[0.5, -1], [1, 1]], "K": [[1, 0], [0, 1], [1, 1]], "V": [[1, 0], [0, 1], [2, 2]], "heads": 2})],
     extra_q=[_bad({"Q": [[1, 0, 1]], "K": [[1, 0, 0]], "V": [[1, 2, 3]], "heads": 2})],
     level="expert")


def batchnorm(X, gamma, beta, eps):
    M = _mat("X", X, 2, 1)
    g, b = _vec("gamma", gamma), _vec("beta", beta)
    D = len(M[0])
    _need(len(g) == D and len(b) == D, "gamma and beta need one entry per column (feature) of X")
    _fin("eps", eps)
    _need(eps >= 0, "eps must be >= 0")
    n = len(M)
    F = [[_F(v) for v in r] for r in M]
    mean = [sum(r[j] for r in F) / n for j in range(D)]
    var = [sum((r[j] - mean[j]) ** 2 for r in F) / n for j in range(D)]
    _need(all(v + _F(eps) > 0 for v in var), "a feature has zero variance and eps = 0")
    sd = [math.sqrt(float(v + _F(eps))) for v in var]
    Y = [[g[j] * float(r[j] - mean[j]) / sd[j] + b[j] for j in range(D)] for r in F]
    return _out(output=Y, batch_mean=mean, batch_var=var)


def _ref_batchnorm(X, gamma, beta, eps):
    import torch
    T = lambda v: torch.tensor(v, dtype=torch.float64)
    x = T(X)
    y = torch.nn.functional.batch_norm(x, None, None, T(gamma), T(beta), training=True, eps=eps)
    return {"output": y.tolist(), "batch_mean": x.mean(0).tolist(), "batch_var": x.var(0, unbiased=False).tolist()}


_add("ml_batch_norm_forward",
     name="Batch normalisation forward pass (training mode)", topic="neural networks",
     purpose="Batch normalisation of a mini-batch in training mode: each feature is normalised with the batch mean and the biased batch variance plus eps, then scaled by gamma and shifted by beta.",
     inputs=[_matin("X", "mini-batch: one row per sample, one column per feature (at least two rows)"), _vecin("gamma", "scale per feature"), _vecin("beta", "shift per feature"),
             _numin("eps", "small constant added to the variance (e.g. 1e-5)")],
     outputs=[_o("output", "gamma (x - mean)/sqrt(var + eps) + beta"), _o("batch_mean", "batch mean of each feature"), _o("batch_var", "biased batch variance of each feature")],
     primary="output", method="exact batch mean and variance; one square root per feature",
     latex=r"\hat x = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}},\quad y = \gamma \hat x + \beta",
     run=batchnorm, reference=_ref_batchnorm,
     wrong=lambda X, gamma, beta, eps: (lambda n: (lambda o: dict(o, output=[[_r((v - b) * math.sqrt((n - 1) / n) + b) for v, b in zip(r, beta)] for r in o["output"]]))(batchnorm(X, gamma, beta, eps)))(len(X)),
     rand=lambda rng: (lambda D: {"X": _rand_matrix(rng, 2, 8, D, D), "gamma": [round(rng.uniform(0.5, 2), 2) for _ in range(D)], "beta": [round(rng.uniform(-1, 1), 2) for _ in range(D)],
                                  "eps": rng.choice([1e-5, 1e-3, 0.0])})(rng.randint(1, 4)),
     phrases=["Batch normalization", "Batch normalisation", "Batch norm"],
     conditions=["at least two samples", "var + eps > 0"],
     examples=[_ex({"X": [[1, 2], [3, 6], [5, 10]], "gamma": [1, 1], "beta": [0, 0], "eps": 0}, {"batch_mean": [3.0, 6.0]}),
               _ex({"X": [[0.5, -1, 2], [1.5, 0, 2.5], [2.5, 1, 1]], "gamma": [1, 0.5, 2], "beta": [0, 1, -1], "eps": 1e-5})],
     level="advanced")


def _opt_vecs(**named):
    vals = {k: _vec(k, v) for k, v in named.items()}
    n = {len(v) for v in vals.values()}
    _need(len(n) == 1, "the vectors (" + ", ".join(named) + ") must have the same length")
    return vals


def momentum_step(w, grad, buf, lr, momentum, nesterov, weight_decay):
    V = _opt_vecs(w=w, grad=grad, buf=buf)
    for nm, v in (("lr", lr), ("momentum", momentum), ("weight_decay", weight_decay)):
        _fin(nm, v)
    _need(lr > 0 and 0 <= momentum < 1 and weight_decay >= 0, "need lr > 0, 0 <= momentum < 1, weight_decay >= 0")
    W, G, B = ([_F(x) for x in V[k]] for k in ("w", "grad", "buf"))
    L, mu, wd = _F(lr), _F(momentum), _F(weight_decay)
    d = [g + wd * x for g, x in zip(G, W)]
    B2 = [mu * b + di for b, di in zip(B, d)]
    step = [di + mu * b for di, b in zip(d, B2)] if nesterov else B2
    return _out(buf_new=B2, w_new=[x - L * s for x, s in zip(W, step)])


def _ref_momentum(w, grad, buf, lr, momentum, nesterov, weight_decay):
    import torch
    p = torch.tensor(w, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([p], lr=lr, momentum=momentum, nesterov=bool(nesterov), weight_decay=weight_decay)
    p.grad = torch.tensor(grad, dtype=torch.float64)
    opt.state[p]["momentum_buffer"] = torch.tensor(buf, dtype=torch.float64)
    opt.step()
    return {"buf_new": opt.state[p]["momentum_buffer"].tolist(), "w_new": p.detach().tolist()}


def _rand_opt(rng, extra):
    n = rng.randint(1, 5)
    R = lambda a=1.0: round(rng.uniform(-a, a), 3)
    out = {"w": [R() for _ in range(n)], "grad": [R() for _ in range(n)]}
    out.update(extra(n, R))
    return out


_add("ml_sgd_momentum_step",
     name="SGD with momentum: one update", topic="optimizers",
     purpose="One step of stochastic gradient descent with momentum in PyTorch's form (optional Nesterov and weight decay): d = g + wd w, buf = mu buf + d, w = w - lr (buf, or d + mu buf with Nesterov).",
     inputs=[_vecin("w", "parameters"), _vecin("grad", "gradient"), _vecin("buf", "momentum buffer before the step (zeros at the start)"), _numin("lr", "learning rate"),
             _numin("momentum", "momentum mu (0 <= mu < 1)"), _boolin("nesterov", "use Nesterov momentum (yes or no)"), _numin("weight_decay", "L2 weight decay (>= 0)")],
     outputs=[_o("buf_new", "momentum buffer after the step"), _o("w_new", "parameters after the step")],
     primary="w_new", method="exact rational arithmetic",
     latex=r"b_t = \mu b_{t-1} + g_t,\quad w_t = w_{t-1} - \eta\, b_t",
     run=momentum_step, reference=_ref_momentum,
     wrong=lambda w, grad, buf, lr, momentum, nesterov, weight_decay: _out(buf_new=[momentum * b - lr * g for b, g in zip(buf, grad)], w_new=[x + momentum * b - lr * g + 0.001 for x, b, g in zip(w, buf, grad)]),
     rand=lambda rng: _rand_opt(rng, lambda n, R: {"buf": [R(0.5) for _ in range(n)], "lr": rng.choice([0.1, 0.01, 0.05]), "momentum": rng.choice([0.9, 0.5, 0.99]),
                                                  "nesterov": rng.choice([True, False]), "weight_decay": rng.choice([0.0, 0.0, 1e-4, 0.01])}),
     phrases=["SGD momentum step", "Momentum update"],
     conditions=["lr > 0", "0 <= momentum < 1", "weight_decay >= 0", "equal lengths"],
     examples=[_ex({"w": [1, -2], "grad": [0.5, -1], "buf": [0.1, 0.2], "lr": 0.1, "momentum": 0.9, "nesterov": False, "weight_decay": 0},
                   {"buf_new": [0.59, -0.82], "w_new": [0.941, -1.918]}),
               _ex({"w": [0.3], "grad": [0.2], "buf": [0], "lr": 0.01, "momentum": 0.9, "nesterov": True, "weight_decay": 0.01})],
     level="intermediate")


def adam_step(w, grad, m, v, t, lr, beta1, beta2, eps, weight_decay=None):
    V = _opt_vecs(w=w, grad=grad, m=m, v=v)
    _need(isinstance(t, int) and t >= 1, "t (the step number after this update) must be a whole number >= 1")
    for nm, x in (("lr", lr), ("beta1", beta1), ("beta2", beta2), ("eps", eps)):
        _fin(nm, x)
    _need(lr > 0 and 0 <= beta1 < 1 and 0 <= beta2 < 1 and eps >= 0, "need lr > 0, 0 <= beta1, beta2 < 1, eps >= 0")
    _need(all(x >= 0 for x in V["v"]), "the second-moment estimate v must be >= 0")
    W = list(V["w"])
    if weight_decay is not None:
        W = [x * (1 - lr * weight_decay) for x in W]
    m2 = [beta1 * a + (1 - beta1) * g for a, g in zip(V["m"], V["grad"])]
    v2 = [beta2 * a + (1 - beta2) * g * g for a, g in zip(V["v"], V["grad"])]
    bc1, bc2 = 1 - beta1 ** t, 1 - beta2 ** t
    _need(bc1 > 0 and bc2 > 0, "the bias corrections are zero (beta = 0 with ... )")
    w2 = [x - lr * (a / bc1) / (math.sqrt(b / bc2) + eps) for x, a, b in zip(W, m2, v2)]
    return _out(m_new=m2, v_new=v2, w_new=w2)


def _ref_adam(w, grad, m, v, t, lr, beta1, beta2, eps, weight_decay=None):
    import torch
    p = torch.tensor(w, dtype=torch.float64, requires_grad=True)
    if weight_decay is None:
        opt = torch.optim.Adam([p], lr=lr, betas=(beta1, beta2), eps=eps, foreach=False)
    else:
        opt = torch.optim.AdamW([p], lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay, foreach=False)
    p.grad = torch.tensor(grad, dtype=torch.float64)
    opt.state[p]["step"] = torch.tensor(float(t - 1))
    opt.state[p]["exp_avg"] = torch.tensor(m, dtype=torch.float64)
    opt.state[p]["exp_avg_sq"] = torch.tensor(v, dtype=torch.float64)
    opt.step()
    return {"m_new": opt.state[p]["exp_avg"].tolist(), "v_new": opt.state[p]["exp_avg_sq"].tolist(), "w_new": p.detach().tolist()}


def _rand_adam(rng, wd=False):
    def extra(n, R):
        d = {"m": [R(0.1) for _ in range(n)], "v": [round(rng.uniform(0, 0.01), 5) for _ in range(n)], "t": rng.randint(1, 50),
             "lr": rng.choice([1e-3, 1e-2, 3e-4]), "beta1": rng.choice([0.9, 0.8]), "beta2": rng.choice([0.999, 0.99]), "eps": rng.choice([1e-8, 1e-6])}
        if wd:
            d["weight_decay"] = rng.choice([0.01, 0.1, 0.05])
        return d
    return _rand_opt(rng, extra)


_ADAM_IN = [_vecin("w", "parameters"), _vecin("grad", "gradient"), _vecin("m", "first-moment estimate before the step (zeros at the start)"),
            _vecin("v", "second-moment estimate before the step (zeros at the start)"), _intin("t", "step number after this update (1 for the first step)", 1),
            _numin("lr", "learning rate"), _numin("beta1", "first-moment decay (e.g. 0.9)"), _numin("beta2", "second-moment decay (e.g. 0.999)"),
            _numin("eps", "denominator constant (e.g. 1e-8)")]

_add("ml_adam_step",
     name="Adam optimizer: one update", topic="optimizers",
     purpose="One Adam update: the new first and second moment estimates and the parameters after the bias-corrected step w - lr m_hat / (sqrt(v_hat) + eps).",
     inputs=_ADAM_IN,
     outputs=[_o("m_new", "beta1 m + (1 - beta1) g"), _o("v_new", "beta2 v + (1 - beta2) g^2"), _o("w_new", "parameters after the step")],
     primary="w_new", method="PyTorch's update rule (eps added after the square root of the bias-corrected v)",
     latex=r"\hat m = \frac{m_t}{1 - \beta_1^t},\ \hat v = \frac{v_t}{1 - \beta_2^t},\quad w_t = w_{t-1} - \eta \frac{\hat m}{\sqrt{\hat v} + \varepsilon}",
     run=adam_step, reference=_ref_adam,
     wrong=lambda w, grad, m, v, t, lr, beta1, beta2, eps: (lambda o: dict(o, w_new=[_r(x - lr * a / (math.sqrt(b) + eps)) for x, a, b in zip(w, o["m_new"], o["v_new"])]))(adam_step(w, grad, m, v, t, lr, beta1, beta2, eps)),
     rand=_rand_adam, phrases=["Adam step", "Adam update"],
     conditions=["t >= 1", "0 <= beta1, beta2 < 1", "v >= 0", "equal lengths"],
     examples=[_ex({"w": [1, -2], "grad": [0.5, -1], "m": [0, 0], "v": [0, 0], "t": 1, "lr": 0.001, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8}),
               _ex({"w": [0.3, 0.1], "grad": [0.2, -0.05], "m": [0.01, -0.02], "v": [0.001, 0.0004], "t": 10, "lr": 0.01, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8})],
     level="advanced")

_add("ml_adamw_step",
     name="AdamW optimizer (decoupled weight decay): one update", topic="optimizers",
     purpose="One AdamW update: the parameters first decay by the factor (1 - lr weight_decay), then take the bias-corrected Adam step.",
     inputs=_ADAM_IN + [_numin("weight_decay", "decoupled weight decay (e.g. 0.01)")],
     outputs=[_o("m_new", "beta1 m + (1 - beta1) g"), _o("v_new", "beta2 v + (1 - beta2) g^2"), _o("w_new", "parameters after decay and step")],
     primary="w_new", method="PyTorch's AdamW rule",
     latex=r"w \leftarrow w(1 - \eta\lambda),\quad w_t = w - \eta \frac{\hat m}{\sqrt{\hat v} + \varepsilon}",
     run=lambda w, grad, m, v, t, lr, beta1, beta2, eps, weight_decay: (_need(weight_decay >= 0, "weight_decay must be >= 0") or adam_step(w, grad, m, v, t, lr, beta1, beta2, eps, weight_decay)),
     reference=_ref_adam,
     wrong=lambda w, grad, m, v, t, lr, beta1, beta2, eps, weight_decay: (lambda o: dict(o, w_new=[_r(x - lr * weight_decay * wi * 0.5) for x, wi in zip(o["w_new"], w)]))(adam_step(w, grad, m, v, t, lr, beta1, beta2, eps, weight_decay)),
     rand=lambda rng: _rand_adam(rng, True), phrases=["AdamW step", "AdamW update"],
     conditions=["t >= 1", "0 <= beta1, beta2 < 1", "v >= 0", "weight_decay >= 0"],
     examples=[_ex({"w": [1, -2], "grad": [0.5, -1], "m": [0, 0], "v": [0, 0], "t": 1, "lr": 0.001, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 0.01}),
               _ex({"w": [0.3, 0.1], "grad": [0.2, -0.05], "m": [0.01, -0.02], "v": [0.001, 0.0004], "t": 10, "lr": 0.01, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 0.1})],
     level="expert")


def step_lr(lr0, step_size, gamma, epoch):
    _fin("lr0", lr0), _fin("gamma", gamma)
    _need(lr0 > 0 and 0 < gamma <= 1, "need lr0 > 0 and 0 < gamma <= 1")
    _need(isinstance(step_size, int) and step_size >= 1 and isinstance(epoch, int) and 0 <= epoch <= 1000, "step_size >= 1 and 0 <= epoch <= 1000 (whole numbers)")
    k = epoch // step_size
    return _out(lr=lr0 * gamma ** k, decays=k)


def _ref_step_lr(lr0, step_size, gamma, epoch):
    import torch
    p = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([p], lr=lr0)
    s = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
    for _ in range(epoch):
        opt.step()
        s.step()
    return {"lr": opt.param_groups[0]["lr"], "decays": epoch // step_size}


_add("ml_step_lr_schedule",
     name="Step learning-rate schedule", topic="optimizers",
     purpose="The learning rate at a given epoch under step decay: multiplied by gamma every step_size epochs.",
     inputs=[_numin("lr0", "initial learning rate"), _intin("step_size", "epochs between decays", 1), _numin("gamma", "decay factor (0 < gamma <= 1)"),
             _intin("epoch", "epoch number (0 = start, at most 1000)", 0, 1000)],
     outputs=[_o("lr", "lr0 gamma^floor(epoch / step_size)"), _o("decays", "number of decays so far")],
     primary="lr", method="closed form",
     latex=r"\eta_t = \eta_0\, \gamma^{\lfloor t / s \rfloor}",
     run=step_lr, reference=_ref_step_lr,
     wrong=lambda lr0, step_size, gamma, epoch: _out(lr=lr0 * gamma ** (epoch // step_size + 1), decays=epoch // step_size + 1),
     rand=lambda rng: {"lr0": rng.choice([0.1, 0.01, 0.05]), "step_size": rng.randint(1, 30), "gamma": rng.choice([0.1, 0.5, 0.9]), "epoch": rng.randint(0, 200)},
     phrases=["Step learning rate schedule", "Step LR schedule"],
     conditions=["lr0 > 0", "0 < gamma <= 1", "0 <= epoch <= 1000"],
     examples=[_ex({"lr0": 0.1, "step_size": 30, "gamma": 0.1, "epoch": 65}, {"lr": 0.001, "decays": 2}), _ex({"lr0": 0.05, "step_size": 10, "gamma": 0.5, "epoch": 9}, {"lr": 0.05})],
     level="beginner")


def cosine_lr(lr_max, eta_min, T_max, epoch):
    _fin("lr_max", lr_max), _fin("eta_min", eta_min)
    _need(lr_max > 0 and 0 <= eta_min < lr_max, "need lr_max > 0 and 0 <= eta_min < lr_max")
    _need(isinstance(T_max, int) and 1 <= T_max <= 1000 and isinstance(epoch, int) and 0 <= epoch <= T_max, "1 <= T_max <= 1000 and 0 <= epoch <= T_max (whole numbers)")
    if epoch == T_max:
        return _out(lr=eta_min)
    return _out(lr=eta_min + (lr_max - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2)


def _ref_cosine_lr(lr_max, eta_min, T_max, epoch):
    import torch
    p = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([p], lr=lr_max)
    s = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max, eta_min=eta_min)
    for _ in range(epoch):
        opt.step()
        s.step()
    return {"lr": opt.param_groups[0]["lr"]}


_add("ml_cosine_lr_schedule",
     name="Cosine-annealing learning-rate schedule", topic="optimizers",
     purpose="The learning rate at a given epoch under cosine annealing from lr_max down to eta_min over T_max epochs.",
     inputs=[_numin("lr_max", "initial (maximum) learning rate"), _numin("eta_min", "final (minimum) learning rate"), _intin("T_max", "length of the schedule in epochs (at most 1000)", 1, 1000),
             _intin("epoch", "epoch number (0 to T_max)", 0, 1000)],
     outputs=[_o("lr", "eta_min + (lr_max - eta_min)(1 + cos(pi t / T_max)) / 2")],
     primary="lr", method="closed form",
     latex=r"\eta_t = \eta_{\min} + \tfrac12(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{\pi t}{T}\right)",
     run=cosine_lr, reference=_ref_cosine_lr,
     wrong=lambda lr_max, eta_min, T_max, epoch: _out(lr=eta_min + (lr_max - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2 * 0.9 + 0.0001),
     rand=lambda rng: (lambda T: {"lr_max": rng.choice([0.1, 0.01, 1e-3]), "eta_min": rng.choice([0.0, 1e-5, 1e-4]), "T_max": T, "epoch": rng.randint(0, T - 1)})(rng.randint(5, 300)),
     phrases=["Cosine learning rate schedule", "Cosine annealing schedule"],
     conditions=["0 <= eta_min < lr_max", "0 <= epoch <= T_max <= 1000"],
     examples=[_ex({"lr_max": 0.1, "eta_min": 0, "T_max": 100, "epoch": 50}, {"lr": 0.05}), _ex({"lr_max": 0.01, "eta_min": 0.0001, "T_max": 200, "epoch": 150})],
     extra_q=[_bad({"lr_max": 0.1, "eta_min": 0, "T_max": 100, "epoch": 150})],
     level="intermediate")


def warmup_cosine(lr_max, start_factor, warmup, total, eta_min, step):
    _fin("lr_max", lr_max), _fin("start_factor", start_factor), _fin("eta_min", eta_min)
    _need(lr_max > 0 and 0 < start_factor <= 1 and 0 <= eta_min < lr_max, "need lr_max > 0, 0 < start_factor <= 1, 0 <= eta_min < lr_max")
    _need(all(isinstance(v, int) for v in (warmup, total, step)) and 1 <= warmup < total <= 1000 and 0 <= step <= total,
          "need whole numbers 1 <= warmup < total <= 1000 and 0 <= step <= total")
    if step < warmup:
        return _out(lr=lr_max * (start_factor + (1 - start_factor) * step / warmup))
    j, T = step - warmup, total - warmup
    if j == T:
        return _out(lr=eta_min)
    return _out(lr=eta_min + (lr_max - eta_min) * (1 + math.cos(math.pi * j / T)) / 2)


def _ref_warmup_cosine(lr_max, start_factor, warmup, total, eta_min, step):
    import warnings
    import torch
    p = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([p], lr=lr_max)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = torch.optim.lr_scheduler.SequentialLR(opt, [torch.optim.lr_scheduler.LinearLR(opt, start_factor=start_factor, end_factor=1.0, total_iters=warmup),
                                                        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total - warmup, eta_min=eta_min)], milestones=[warmup])
        for _ in range(step):
            opt.step()
            s.step()
    return {"lr": opt.param_groups[0]["lr"]}


_add("ml_warmup_cosine_lr_schedule",
     name="Linear warm-up followed by cosine decay", topic="optimizers",
     purpose="The learning rate at a given step of the common transformer schedule: linear warm-up from start_factor lr_max to lr_max over the warm-up steps, then cosine decay to eta_min at the last step.",
     inputs=[_numin("lr_max", "peak learning rate"), _numin("start_factor", "fraction of lr_max at step 0 (0 < f <= 1)"), _intin("warmup", "warm-up steps", 1),
             _intin("total", "total steps (at most 1000)", 2, 1000), _numin("eta_min", "final learning rate"), _intin("step", "step number (0 to total)", 0, 1000)],
     outputs=[_o("lr", "learning rate at the step")],
     primary="lr", method="closed form of each phase",
     latex=r"\eta_t = \begin{cases} \eta_{\max}\left(f + (1-f)\frac{t}{W}\right) & t < W \\ \eta_{\min} + \tfrac12(\eta_{\max}-\eta_{\min})\left(1 + \cos\frac{\pi (t - W)}{T - W}\right) & t \ge W \end{cases}",
     run=warmup_cosine, reference=_ref_warmup_cosine,
     wrong=lambda lr_max, start_factor, warmup, total, eta_min, step: _out(lr=lr_max * step / warmup if step < warmup else eta_min + (lr_max - eta_min) * (1 + math.cos(math.pi * step / total)) / 2),
     rand=lambda rng: (lambda T: (lambda W: {"lr_max": rng.choice([1e-3, 3e-4, 0.01]), "start_factor": rng.choice([0.01, 0.1, 0.5]), "warmup": W, "total": T,
                                             "eta_min": rng.choice([0.0, 1e-5]), "step": rng.randint(0, T - 1)})(rng.randint(1, T // 3)))(rng.randint(10, 300)),
     phrases=["Warmup cosine schedule", "Warm-up cosine learning rate"],
     conditions=["1 <= warmup < total <= 1000", "0 <= step <= total"],
     examples=[_ex({"lr_max": 0.001, "start_factor": 0.1, "warmup": 10, "total": 100, "eta_min": 0, "step": 5}, {"lr": 0.00055}),
               _ex({"lr_max": 0.001, "start_factor": 0.1, "warmup": 10, "total": 100, "eta_min": 0, "step": 55}, {"lr": 0.0005})],
     level="expert")


def transformer_params(d_model, n_heads, d_ff, n_layers):
    _need(all(isinstance(v, int) and v >= 1 for v in (d_model, n_heads, d_ff, n_layers)), "all sizes must be whole numbers >= 1")
    _need(d_model % n_heads == 0, "d_model must be divisible by n_heads")
    _need(d_model <= 8192 and d_ff <= 32768 and n_layers <= 200, "sizes too large")
    attn = 4 * d_model * d_model + 4 * d_model
    ff = 2 * d_model * d_ff + d_ff + d_model
    norm = 4 * d_model
    per = attn + ff + norm
    return _out(attention=attn, feed_forward=ff, layer_norms=norm, per_layer=per, total=per * n_layers)


def _ref_transformer(d_model, n_heads, d_ff, n_layers):
    import torch
    layer = torch.nn.TransformerEncoderLayer(d_model, n_heads, d_ff, device="meta")
    cnt = lambda m: sum(p.numel() for p in m.parameters())
    per = cnt(layer)
    return {"attention": cnt(layer.self_attn), "feed_forward": cnt(layer.linear1) + cnt(layer.linear2), "layer_norms": cnt(layer.norm1) + cnt(layer.norm2),
            "per_layer": per, "total": per * n_layers}


_add("ml_transformer_parameter_count",
     name="Parameter count of a transformer encoder", topic="neural networks",
     purpose="The parameters of a standard transformer encoder layer (multi-head self-attention with biases, a two-layer feed-forward block and two layer norms) and of a stack of such layers.",
     inputs=[_intin("d_model", "model (embedding) dimension", 1), _intin("n_heads", "attention heads (divide d_model)", 1), _intin("d_ff", "feed-forward hidden size", 1),
             _intin("n_layers", "number of layers", 1)],
     outputs=[_o("attention", "4 d^2 + 4 d (Q, K, V and output projections with biases)"), _o("feed_forward", "2 d d_ff + d_ff + d"), _o("layer_norms", "4 d"),
              _o("per_layer", "parameters per layer"), _o("total", "parameters of the stack")],
     primary="total", method="integer arithmetic",
     latex=r"P_{\mathrm{layer}} = 4d^2 + 4d + 2 d\,d_{ff} + d_{ff} + d + 4d",
     run=transformer_params, reference=_ref_transformer,
     wrong=lambda d_model, n_heads, d_ff, n_layers: (lambda o: dict(o, attention=3 * d_model * d_model, total=(3 * d_model * d_model + o["feed_forward"]) * n_layers))(transformer_params(d_model, n_heads, d_ff, n_layers)),
     rand=lambda rng: (lambda h: {"d_model": h * rng.choice([8, 16, 32, 64]), "n_heads": h, "d_ff": rng.choice([256, 512, 1024, 2048, 3072]), "n_layers": rng.randint(1, 24)})(rng.choice([1, 2, 4, 8, 12])),
     phrases=["Transformer parameter count", "Transformer encoder parameter count"],
     conditions=["d_model divisible by n_heads"],
     examples=[_ex({"d_model": 512, "n_heads": 8, "d_ff": 2048, "n_layers": 6}, {"per_layer": 3152384, "total": 18914304}),
               _ex({"d_model": 768, "n_heads": 12, "d_ff": 3072, "n_layers": 12}, {"per_layer": 7087872})],
     extra_q=[_bad({"d_model": 100, "n_heads": 8, "d_ff": 400, "n_layers": 2})],
     level="expert")


def rnn_params(cell, input_size, hidden_size, num_layers, bidirectional):
    _need(all(isinstance(v, int) and v >= 1 for v in (input_size, hidden_size, num_layers)), "sizes must be whole numbers >= 1")
    G = {"lstm": 4, "gru": 3, "rnn": 1}[cell]
    dirs = 2 if bidirectional else 1
    per = []
    for layer in range(num_layers):
        inp = input_size if layer == 0 else hidden_size * dirs
        per.append(dirs * (G * hidden_size * (inp + hidden_size) + 2 * G * hidden_size))
    return _out(per_layer=per, total=sum(per))


def _ref_rnn(cell, input_size, hidden_size, num_layers, bidirectional):
    import torch
    cls = {"lstm": torch.nn.LSTM, "gru": torch.nn.GRU, "rnn": torch.nn.RNN}[cell]
    m = cls(input_size, hidden_size, num_layers, bidirectional=bool(bidirectional), device="meta")
    per = [sum(p.numel() for n, p in m.named_parameters() if n.split("_")[2].startswith(f"l{k}")) for k in range(num_layers)]
    return {"per_layer": per, "total": sum(p.numel() for p in m.parameters())}


_add("ml_rnn_parameter_count",
     name="Parameter count of an LSTM, GRU or simple RNN", topic="neural networks",
     purpose="The trainable parameters of each layer and of a stacked recurrent network (LSTM: 4 gates, GRU: 3, simple RNN: 1), with two bias vectors per gate as in PyTorch, optionally bidirectional.",
     inputs=[_choice("cell", ["lstm", "gru", "rnn"], "cell type: lstm, gru or rnn"), _intin("input_size", "input features", 1), _intin("hidden_size", "hidden units", 1),
             _intin("num_layers", "stacked layers", 1), _boolin("bidirectional", "bidirectional (yes or no)")],
     outputs=[_o("per_layer", "parameters of each layer (both directions)"), _o("total", "total parameters")],
     primary="total", method="G h (n_in + h) + 2 G h per direction; layers above the first take h (x 2 if bidirectional) inputs",
     latex=r"P = G\,h\,(n_{\mathrm{in}} + h) + 2Gh",
     run=rnn_params, reference=_ref_rnn,
     wrong=lambda cell, input_size, hidden_size, num_layers, bidirectional: (lambda o: _out(per_layer=[v - {"lstm": 4, "gru": 3, "rnn": 1}[cell] * hidden_size * (2 if bidirectional else 1) for v in o["per_layer"]],
                                                                                          total=o["total"] - {"lstm": 4, "gru": 3, "rnn": 1}[cell] * hidden_size * (2 if bidirectional else 1) * num_layers))(rnn_params(cell, input_size, hidden_size, num_layers, bidirectional)),
     rand=lambda rng: {"cell": rng.choice(["lstm", "gru", "rnn"]), "input_size": rng.randint(1, 300), "hidden_size": rng.randint(1, 512), "num_layers": rng.randint(1, 4), "bidirectional": rng.choice([True, False])},
     phrases=["RNN parameter count", "LSTM parameter count", "Recurrent network parameter count"],
     conditions=["positive whole-number sizes"],
     examples=[_ex({"cell": "lstm", "input_size": 100, "hidden_size": 128, "num_layers": 1, "bidirectional": False}, {"total": 117760}),
               _ex({"cell": "gru", "input_size": 32, "hidden_size": 64, "num_layers": 2, "bidirectional": True})],
     level="advanced")


def receptive_field(kernels, strides, dilations):
    k = _labels("kernels", kernels)
    s = _labels("strides", strides)
    d = _labels("dilations", dilations)
    _need(len(k) == len(s) == len(d), "kernels, strides and dilations need one entry per layer")
    _need(all(v >= 1 for v in k + s + d), "kernel sizes, strides and dilations must be >= 1")
    _need(len(k) <= 20, "at most 20 layers")
    rf, jump = 1, 1
    for kk, ss, dd in zip(k, s, d):
        rf += (kk - 1) * dd * jump
        jump *= ss
    _need(rf <= 20000, "the receptive field is too large (more than 20000)")
    return _out(receptive_field=rf, jump=jump)


def _ref_receptive(kernels, strides, dilations):
    import torch
    k, s, d = ([int(v) for v in x] for x in (kernels, strides, dilations))
    jump = 1
    for v in s:
        jump *= v
    L = 1
    for kk, dd, ss in zip(k, d, s):
        pass
    L = 1 + sum((kk - 1) * dd for kk, dd in zip(k, d)) * jump + 3 * jump + 8
    x = torch.ones(1, 1, L, dtype=torch.float64, requires_grad=True)
    y = x
    for kk, ss, dd in zip(k, s, d):
        y = torch.nn.functional.conv1d(y, torch.ones(1, 1, kk, dtype=torch.float64), stride=ss, dilation=dd)
    (g0,) = torch.autograd.grad(y[0, 0, 0], x, retain_graph=True)
    (g1,) = torch.autograd.grad(y[0, 0, 1], x)
    nz0 = torch.nonzero(g0[0, 0]).flatten()
    nz1 = torch.nonzero(g1[0, 0]).flatten()
    return {"receptive_field": int(nz0[-1] - nz0[0] + 1), "jump": int(nz1[0] - nz0[0])}


_add("ml_receptive_field",
     name="Receptive field of stacked convolutions", topic="neural networks",
     purpose="The receptive field (how many input positions one output depends on, end to end) and the jump (input distance between adjacent outputs) of a stack of convolution or pooling layers with given kernel sizes, strides and dilations.",
     inputs=[_vecin("kernels", "kernel size of each layer (whole numbers)"), _vecin("strides", "stride of each layer"), _vecin("dilations", "dilation of each layer (1 = ordinary)")],
     outputs=[_o("receptive_field", "1 + sum (k_l - 1) d_l prod_{i<l} s_i"), _o("jump", "product of the strides")],
     primary="receptive_field", method="layer-by-layer recursion r += (k - 1) d j, j *= s",
     latex=r"r = 1 + \sum_l (k_l - 1)\, d_l \prod_{i < l} s_i",
     run=receptive_field, reference=_ref_receptive,
     wrong=lambda kernels, strides, dilations: _out(receptive_field=1 + sum((int(a) - 1) * int(c) for a, c in zip(kernels, dilations)) + (1 if all(int(v) == 1 for v in strides) else 0), jump=1),
     rand=lambda rng: (lambda n: {"kernels": [rng.choice([1, 2, 3, 5, 7]) for _ in range(n)], "strides": [rng.choice([1, 1, 2]) for _ in range(n)], "dilations": [rng.choice([1, 1, 2, 4]) for _ in range(n)]})(rng.randint(1, 6)),
     phrases=["Receptive field"],
     conditions=["one kernel, stride and dilation per layer", "at most 20 layers"],
     examples=[_ex({"kernels": [3, 3, 3], "strides": [1, 1, 1], "dilations": [1, 1, 1]}, {"receptive_field": 7, "jump": 1}),
               _ex({"kernels": [7, 3, 3, 3], "strides": [2, 2, 1, 1], "dilations": [1, 1, 1, 1]}, {"receptive_field": 27, "jump": 4})],
     level="expert")


# ================================================================================================ ranking metrics
def _rel01(name, v):
    r = _labels(name, v)
    _need(all(x in (0, 1) for x in r), f"every relevance in {name} must be 0 or 1")
    return r


def prec_rec_at_k(rel, n_relevant, k):
    r = _rel01("rel", rel)
    _need(isinstance(k, int) and 1 <= k <= len(r), "k must be between 1 and the length of the ranked list")
    hits = sum(r[:k])
    _need(isinstance(n_relevant, int) and n_relevant >= sum(r) and n_relevant >= 1, "n_relevant must be at least the number of relevant items in the list and at least 1")
    return _out(precision_at_k=Fraction(hits, k), recall_at_k=Fraction(hits, n_relevant), hits=hits, hit_rate=1 if hits else 0)


def _ref_prk(rel, n_relevant, k):
    np = _np()
    r = np.array(rel, dtype=int)[:k]
    return {"precision_at_k": float(r.mean()), "recall_at_k": float(r.sum() / n_relevant), "hits": int(r.sum()), "hit_rate": int(r.any())}


def _rand_rel(rng, n=None, p=0.35):
    n = n or rng.randint(3, 20)
    while True:
        r = [1 if rng.random() < p else 0 for _ in range(n)]
        if sum(r):
            return r


_add("ml_precision_recall_at_k",
     name="Precision@k and recall@k of a ranking", topic="ranking metrics",
     purpose="Precision at k and recall at k of a ranked list (relevance 0/1 in rank order), with the number of relevant items in the top k and the hit indicator.",
     inputs=[_vecin("rel", "relevance of each ranked item, in rank order (0 or 1)"), _intin("n_relevant", "total number of relevant items for the query", 1),
             _intin("k", "cut-off rank", 1)],
     outputs=[_o("precision_at_k", "relevant in top k / k"), _o("recall_at_k", "relevant in top k / n_relevant"), _o("hits", "relevant items in the top k"),
              _o("hit_rate", "1 if any relevant item is in the top k")],
     primary="precision_at_k", method="exact counts",
     latex=r"P@k = \frac{|\mathrm{rel} \cap \mathrm{top}_k|}{k},\quad R@k = \frac{|\mathrm{rel} \cap \mathrm{top}_k|}{|\mathrm{rel}|}",
     run=prec_rec_at_k, reference=_ref_prk,
     wrong=lambda rel, n_relevant, k: (lambda h: _out(precision_at_k=Fraction(h, len(rel)), recall_at_k=Fraction(h, max(1, int(sum(rel)))), hits=h, hit_rate=0))(int(sum(rel[:k]))),
     rand=lambda rng: (lambda r: {"rel": r, "n_relevant": sum(r) + rng.randint(0, 3), "k": rng.randint(1, len(r) - 1) if len(r) > 1 else 1})(_rand_rel(rng)),
     phrases=["Precision at k", "Precision and recall at k"],
     conditions=["relevance 0 or 1", "1 <= k <= list length", "n_relevant >= relevant items in the list"],
     examples=[_ex({"rel": [1, 0, 1, 1, 0, 0, 1], "n_relevant": 5, "k": 5}, {"precision_at_k": 0.6, "recall_at_k": 0.6, "hits": 3}),
               _ex({"rel": [0, 0, 1, 0], "n_relevant": 2, "k": 2}, {"precision_at_k": 0.0, "hit_rate": 0})],
     extra_q=[_bad({"rel": [1, 0, 1], "n_relevant": 1, "k": 2})],
     weak="checked by a numpy re-count of the same definitions (no library computes precision@k of a ranked list)",
     level="beginner")


def mean_ap(R):
    M = _mat("R", R)
    aps = []
    for row in M:
        r = _rel01("each row of R", row)
        nrel = sum(r)
        _need(nrel > 0, "a query has no relevant item (its average precision is undefined)")
        hits, s = 0, Fraction(0)
        for i, v in enumerate(r, 1):
            if v:
                hits += 1
                s += Fraction(hits, i)
        aps.append(s / nrel)
    return _out(average_precision=aps, map=sum(aps) / len(aps))


def _ref_map(R):
    from sklearn.metrics import average_precision_score
    aps = [float(average_precision_score(row, list(range(len(row), 0, -1)))) for row in R]
    return {"average_precision": aps, "map": sum(aps) / len(aps)}


_add("ml_mean_average_precision",
     name="Mean average precision (MAP) of rankings", topic="ranking metrics",
     purpose="The average precision of each query's ranked list (mean of precision at the rank of each relevant item, over the relevant items in the list) and their mean, MAP.",
     inputs=[_matin("R", "relevance (0 or 1) in rank order: one row per query, rows of equal length")],
     outputs=[_o("average_precision", "AP of each query"), _o("map", "mean of the APs")],
     primary="map", method="exact fractions",
     latex=r"\mathrm{AP} = \frac{1}{|\mathrm{rel}|}\sum_{k:\ \mathrm{rel}_k = 1} P@k,\quad \mathrm{MAP} = \frac{1}{Q}\sum_q \mathrm{AP}_q",
     run=mean_ap, reference=_ref_map,
     wrong=lambda R: (lambda aps: _out(average_precision=aps, map=sum(aps) / len(aps)))([_r(Fraction(sum(int(v) for v in row[:max(1, int(sum(row)))]), max(1, int(sum(row))))) for row in R]),
     rand=lambda rng: (lambda n: {"R": [_rand_rel(rng, n) for _ in range(rng.randint(1, 5))]})(rng.randint(3, 12)),
     phrases=["Mean average precision", "MAP of rankings"],
     conditions=["relevance 0 or 1", "every query has a relevant item"],
     examples=[_ex({"R": [[1, 0, 1, 0, 0], [0, 1, 0, 0, 1]]}, {"average_precision": [0.833333333333, 0.45], "map": 0.641666666667}),
               _ex({"R": [[0, 0, 1, 1, 0, 1]]})],
     extra_q=[_bad({"R": [[0, 0, 0], [1, 0, 1]]})],
     level="intermediate")


def ndcg(rel, k, gain):
    r = _vec("rel", rel)
    _need(all(v >= 0 for v in r), "relevance grades must be >= 0")
    _need(isinstance(k, int) and 1 <= k <= len(r), "k must be between 1 and the length of the list")
    g = [(2.0 ** v - 1) if gain == "exponential" else v for v in r]
    disc = [1 / math.log2(i + 2) for i in range(len(r))]
    dcg = math.fsum(gi * di for gi, di in zip(g[:k], disc))
    ideal = sorted(g, reverse=True)
    idcg = math.fsum(gi * di for gi, di in zip(ideal[:k], disc))
    _need(idcg > 0, "NDCG is undefined: no relevant item")
    return _out(dcg=dcg, idcg=idcg, ndcg=dcg / idcg)


def _ref_ndcg(rel, k, gain):
    np = _np()
    from sklearn.metrics import dcg_score, ndcg_score
    y = np.array([[(2.0 ** v - 1) if gain == "exponential" else v for v in rel]])
    s = np.array([list(range(len(rel), 0, -1))], dtype=float)
    return {"dcg": float(dcg_score(y, s, k=k)), "idcg": float(dcg_score(y, y, k=k)), "ndcg": float(ndcg_score(y, s, k=k))}


_add("ml_ndcg_at_k",
     name="NDCG@k (normalised discounted cumulative gain)", topic="ranking metrics",
     purpose="DCG@k of a ranked list with graded relevance (linear gain rel or exponential gain 2^rel - 1, discount 1/log2(rank + 1)), the ideal DCG@k of the same items sorted by relevance, and NDCG@k = DCG/IDCG.",
     inputs=[_vecin("rel", "relevance grade of each item in rank order (>= 0)"), _intin("k", "cut-off rank", 1),
             _choice("gain", ["linear", "exponential"], "gain: linear (rel) or exponential (2^rel - 1)")],
     outputs=[_o("dcg", "DCG@k"), _o("idcg", "ideal DCG@k"), _o("ndcg", "DCG@k / IDCG@k")],
     primary="ndcg", method="discounted sums with fsum",
     latex=r"\mathrm{DCG}@k = \sum_{i=1}^k \frac{g_i}{\log_2(i + 1)},\quad \mathrm{NDCG}@k = \frac{\mathrm{DCG}@k}{\mathrm{IDCG}@k}",
     run=ndcg, reference=_ref_ndcg,
     wrong=lambda rel, k, gain: (lambda o: dict(o, ndcg=_r(math.fsum(((2.0 ** v - 1) if gain == "exponential" else v) / math.log(i + 2) for i, v in enumerate(rel[:k])) / o["idcg"])))(ndcg(rel, k, gain)),
     rand=lambda rng: (lambda r: {"rel": r, "k": rng.randint(1, len(r)), "gain": rng.choice(["linear", "exponential"])})([rng.choice([0, 0, 1, 2, 3]) for _ in range(rng.randint(3, 15))] + [rng.randint(1, 3)]),
     phrases=["NDCG at k", "NDCG"],
     conditions=["grades >= 0", "1 <= k <= list length", "at least one relevant item"],
     examples=[_ex({"rel": [3, 2, 3, 0, 1, 2], "k": 6, "gain": "linear"}, {"dcg": 6.86112668859, "ndcg": 0.960808194336}),
               _ex({"rel": [3, 2, 3, 0, 1, 2], "k": 3, "gain": "exponential"})],
     extra_q=[_bad({"rel": [0, 0, 0], "k": 2, "gain": "linear"})],
     level="intermediate")


def mrr(R):
    M = _mat("R", R)
    rr = []
    for row in M:
        r = _rel01("each row of R", row)
        rr.append(Fraction(1, r.index(1) + 1) if 1 in r else Fraction(0))
    return _out(reciprocal_ranks=rr, mrr=sum(rr) / len(rr))


def _ref_mrr(R):
    np = _np()
    A = np.array(R, dtype=int)
    rr = [1.0 / (int(np.argmax(row)) + 1) if row.any() else 0.0 for row in A]
    return {"reciprocal_ranks": rr, "mrr": float(np.mean(rr))}


_add("ml_mean_reciprocal_rank",
     name="Mean reciprocal rank (MRR)", topic="ranking metrics",
     purpose="The reciprocal rank of the first relevant item of each query's ranked list (0 when none is relevant) and their mean, the MRR.",
     inputs=[_matin("R", "relevance (0 or 1) in rank order: one row per query")],
     outputs=[_o("reciprocal_ranks", "1 / rank of the first relevant item of each query"), _o("mrr", "mean reciprocal rank")],
     primary="mrr", method="exact fractions",
     latex=r"\mathrm{MRR} = \frac{1}{Q}\sum_q \frac{1}{\mathrm{rank}_q}",
     run=mrr, reference=_ref_mrr,
     wrong=lambda R: (lambda rr: _out(reciprocal_ranks=rr, mrr=sum(rr) / len(rr)))([Fraction(1, list(row).index(1) + 2) if 1 in list(row) else Fraction(0) for row in R]),
     rand=lambda rng: (lambda n: {"R": [_rand_rel(rng, n, 0.25) for _ in range(rng.randint(1, 6))]})(rng.randint(3, 10)),
     phrases=["Mean reciprocal rank", "MRR"],
     conditions=["relevance 0 or 1"],
     examples=[_ex({"R": [[0, 0, 1], [0, 1, 0], [1, 0, 0]]}, {"mrr": 0.611111111111}), _ex({"R": [[0, 0, 0, 1], [0, 0, 0, 0]]}, {"mrr": 0.125})],
     weak="checked by a numpy re-derivation of the same definition (first relevant position by argmax)",
     level="beginner")


# ================================================================================================ restored capabilities
def threshold_metrics(y_true, scores, threshold):
    yt = _binary("y_true", y_true)
    s = _vec("scores", scores)
    _same_len(yt, s, "y_true", "scores")
    _fin("threshold", threshold)
    yp = [1 if v >= threshold else 0 for v in s]
    tp, fp, fn, tn = _counts(yt, yp)
    _need(tp + fp > 0, "no sample reaches the threshold: precision is undefined")
    _need(tp + fn > 0, "there are no actual positives: recall is undefined")
    return _out(TP=tp, FP=fp, FN=fn, TN=tn, precision=Fraction(tp, tp + fp), recall=Fraction(tp, tp + fn),
                accuracy=Fraction(tp + tn, len(yt)))


def _ref_threshold(y_true, scores, threshold):
    np = _np()
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
    yp = (np.array(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yp, labels=[0, 1]).ravel()
    return {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn), "precision": float(precision_score(y_true, yp)),
            "recall": float(recall_score(y_true, yp)), "accuracy": float(accuracy_score(y_true, yp))}


def _wrong_threshold(y_true, scores, threshold):
    o = threshold_metrics(y_true, scores, threshold)
    o["precision"], o["recall"] = o["recall"], o["precision"]
    return o


def _rand_threshold(rng):
    while True:
        d = _rand_scored(rng)
        s = sorted(d["scores"])
        d["threshold"] = round(rng.uniform(s[len(s) // 4], s[3 * len(s) // 4]), 2)
        try:
            o = threshold_metrics(**d)
            if o["precision"] != o["recall"]:
                return d
        except ValueError:
            pass


_add("ml_threshold_metrics",
     name="Confusion counts, precision and recall at a score threshold", topic="classification metrics",
     purpose="Classify by a score threshold (score >= threshold is positive) and give the confusion counts, precision, recall and accuracy.",
     inputs=[_vecin("y_true", "true labels (0 or 1)"), _vecin("scores", "classifier scores"),
             _numin("threshold", "decision threshold (a score at or above it is positive)")],
     outputs=[_o("TP", "true positives"), _o("FP", "false positives"), _o("FN", "false negatives"), _o("TN", "true negatives"),
              _o("precision", "TP / (TP + FP)"), _o("recall", "TP / (TP + FN)"), _o("accuracy", "(TP + TN) / n")],
     primary="precision", method="threshold the scores, count, exact fractions",
     latex=r"\hat y_i = \mathbb{1}(s_i \ge t)",
     run=threshold_metrics, reference=_ref_threshold, wrong=_wrong_threshold, rand=_rand_threshold,
     phrases=["Metrics at a threshold", "Classification at a threshold"],
     conditions=["labels 0 or 1", "at least one score reaches the threshold", "at least one positive"],
     examples=[_ex({"y_true": [0, 1, 1, 0, 1], "scores": [0.1, 0.9, 0.8, 0.3, 0.2], "threshold": 0.25},
                   {"TP": 2, "FP": 1, "FN": 1, "TN": 1, "precision": 0.666666666667}),
               _ex({"y_true": [1, 0, 1, 0, 0, 1], "scores": [3, 1, 2, 2, 0, 4], "threshold": 2})],
     level="beginner")


def softmax_temperature(z, T):
    zv = _vec("z", z, 2)
    _fin("T", T)
    _need(T > 0, "the temperature T must be positive")
    p = _softmax([v / T for v in zv])
    return _out(probabilities=p, entropy=-math.fsum(q * math.log(q) for q in p if q > 0))


def _ref_softmax_t(z, T):
    np = _np()
    from scipy.special import softmax as sm
    from scipy.stats import entropy
    p = sm(np.array(z) / T)
    return {"probabilities": _f(p), "entropy": float(entropy(p))}


_add("ml_softmax_temperature",
     name="Softmax with temperature", topic="loss functions and activations",
     purpose="The softmax of logits divided by a temperature T (T > 1 flattens, T < 1 sharpens the distribution) and the entropy of the result in nats.",
     inputs=[_vecin("z", "logits", 2), _numin("T", "temperature (T > 0)")],
     outputs=[_o("probabilities", "softmax(z / T)"), _o("entropy", "entropy of the distribution in nats")],
     primary="entropy", method="stable softmax of z / T",
     latex=r"p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}",
     run=softmax_temperature, reference=_ref_softmax_t,
     wrong=lambda z, T: softmax_temperature([v * T * T for v in z], T) if abs(T - 1) > 0.1 else _out(probabilities=[0.0], entropy=0.0),
     rand=lambda rng: {"z": _rand_logits(rng), "T": rng.choice([0.5, 2.0, 5.0, round(rng.uniform(0.2, 10), 2)])},
     phrases=["Softmax with temperature", "Temperature softmax"],
     conditions=["T > 0"],
     examples=[_ex({"z": [1, 2, 3], "T": 2}), _ex({"z": [2, 0, -1, 0.5], "T": 0.5})],
     extra_q=[_bad({"z": [1, 2], "T": 0})],
     level="intermediate")


def poly_kernel(x, z, gamma, coef0, degree):
    a, b = _vec("x", x), _vec("z", z)
    _same_len(a, b, "x", "z")
    _fin("gamma", gamma), _fin("coef0", coef0)
    _need(isinstance(degree, int) and 1 <= degree <= 10, "degree must be a whole number from 1 to 10")
    return _out(kernel=(_F(gamma) * sum(_F(u) * _F(v) for u, v in zip(a, b)) + _F(coef0)) ** degree)


def _ref_poly_kernel(x, z, gamma, coef0, degree):
    np = _np()
    from sklearn.metrics.pairwise import polynomial_kernel
    return {"kernel": float(polynomial_kernel(np.array([x]), np.array([z]), degree=degree, gamma=gamma, coef0=coef0)[0, 0])}


_add("ml_polynomial_kernel",
     name="Polynomial kernel", topic="classical algorithms",
     purpose="The polynomial kernel (gamma x.z + coef0)^degree between two vectors, as used by kernel SVMs.",
     inputs=[_vecin("x", "first vector"), _vecin("z", "second vector"), _numin("gamma", "scale of the dot product"),
             _numin("coef0", "constant term"), _intin("degree", "degree (1 to 10)", 1, 10)],
     outputs=[_o("kernel", "(gamma x.z + coef0)^degree")],
     primary="kernel", method="exact rational arithmetic",
     latex=r"k(x, z) = (\gamma\, x^\top z + c_0)^d",
     run=poly_kernel, reference=_ref_poly_kernel,
     wrong=lambda x, z, gamma, coef0, degree: _out(kernel=(gamma * _dot(x, z)) ** degree + coef0),
     rand=lambda rng: (lambda d: {"x": [round(rng.uniform(-2, 2), 2) for _ in range(d)], "z": [round(rng.uniform(-2, 2), 2) for _ in range(d)],
                                  "gamma": rng.choice([1.0, 0.5, round(rng.uniform(0.1, 2), 2)]), "coef0": rng.choice([1.0, 2.0, round(rng.uniform(0.5, 3), 2)]),
                                  "degree": rng.randint(2, 5)})(rng.randint(2, 5)),
     phrases=["Polynomial kernel"],
     conditions=["degree 1 to 10", "equal lengths"],
     examples=[_ex({"x": [1, 2], "z": [3, 4], "gamma": 1, "coef0": 1, "degree": 2}, {"kernel": 144.0}),
               _ex({"x": [1, -1, 0.5], "z": [2, 0, 1], "gamma": 0.5, "coef0": 2, "degree": 3})],
     level="intermediate")


def triplet(anchor, positive, negative, margin):
    a, p, n = _vec("anchor", anchor), _vec("positive", positive), _vec("negative", negative)
    _need(len(a) == len(p) == len(n), "anchor, positive and negative must have the same length")
    _fin("margin", margin)
    _need(margin >= 0, "the margin must be >= 0")
    dap = math.sqrt(math.fsum((x - y) ** 2 for x, y in zip(a, p)))
    dan = math.sqrt(math.fsum((x - y) ** 2 for x, y in zip(a, n)))
    v = dap - dan + margin
    return _out(loss=max(0.0, v), d_ap=dap, d_an=dan)


def _ref_triplet(anchor, positive, negative, margin):
    import torch
    from scipy.spatial.distance import euclidean
    T = lambda v: torch.tensor([v], dtype=torch.float64)
    loss = torch.nn.functional.triplet_margin_loss(T(anchor), T(positive), T(negative), margin=margin, p=2, eps=0.0).item()
    return {"loss": loss, "d_ap": float(euclidean(anchor, positive)), "d_an": float(euclidean(anchor, negative))}


def _rand_triplet(rng):
    while True:
        d = rng.randint(2, 6)
        a = [round(rng.uniform(-2, 2), 2) for _ in range(d)]
        p = [round(x + rng.uniform(-1, 1), 2) for x in a]
        n = [round(x + rng.uniform(-1.5, 1.5), 2) for x in a]
        m = rng.choice([1.0, 0.5, 2.0, round(rng.uniform(0.2, 3), 2)])
        o = triplet(a, p, n, m)
        if o["loss"] > 0.05:
            return {"anchor": a, "positive": p, "negative": n, "margin": m}


_add("ml_triplet_loss",
     name="Triplet margin loss", topic="loss functions and activations",
     purpose="The triplet margin loss max(0, d(a, p) - d(a, n) + margin) of an anchor, a positive and a negative embedding, with both Euclidean distances.",
     inputs=[_vecin("anchor", "anchor embedding"), _vecin("positive", "embedding of the same identity"), _vecin("negative", "embedding of a different identity"),
             _numin("margin", "margin (>= 0)")],
     outputs=[_o("loss", "max(0, d_ap - d_an + margin)"), _o("d_ap", "anchor-positive distance"), _o("d_an", "anchor-negative distance")],
     primary="loss", method="Euclidean distances with fsum",
     latex=r"L = \max\left(0,\ \|a - p\|_2 - \|a - n\|_2 + m\right)",
     run=triplet, reference=_ref_triplet,
     wrong=lambda anchor, positive, negative, margin: (lambda o: dict(o, loss=_r(max(0.0, o["d_ap"] ** 2 - o["d_an"] ** 2 + margin))))(triplet(anchor, positive, negative, margin)),
     rand=_rand_triplet, phrases=["Triplet loss", "Triplet margin loss"],
     conditions=["equal lengths", "margin >= 0"],
     examples=[_ex({"anchor": [0, 0], "positive": [1, 1], "negative": [2, 0], "margin": 1}, {"d_an": 2.0}),
               _ex({"anchor": [1, 2, 3], "positive": [1.5, 2, 2], "negative": [0, 3, 3.5], "margin": 0.5})],
     level="advanced")


def impurity_counts(counts):
    c = _labels("counts", counts)
    _need(all(v >= 0 for v in c) and sum(c) > 0, "counts must be whole numbers >= 0 with a positive total")
    n = sum(c)
    gini = 1 - sum(Fraction(v, n) ** 2 for v in c)
    return _out(entropy_bits=_Hc(c) / math.log(2.0), gini=gini, misclassification=1 - Fraction(max(c), n))


def _ref_impurity(counts):
    np = _np()
    from sklearn.tree import DecisionTreeClassifier
    y = np.repeat(np.arange(len(counts)), np.array(counts, dtype=int))
    X = np.zeros((len(y), 1))
    out = {}
    for crit, key in (("gini", "gini"), ("entropy", "entropy_bits")):
        t = DecisionTreeClassifier(criterion=crit, max_depth=1).fit(X, y)
        out[key] = float(t.tree_.impurity[0])
    out["misclassification"] = float(1 - np.max(counts) / np.sum(counts))
    return out


_add("ml_impurity_from_counts",
     name="Node impurity from class counts (entropy, Gini, misclassification)", topic="information theory",
     purpose="The impurity of a decision-tree node from its class counts: entropy in bits, Gini impurity and misclassification error.",
     inputs=[_vecin("counts", "number of samples of each class in the node (whole numbers)")],
     outputs=[_o("entropy_bits", "entropy in bits"), _o("gini", "1 - sum p_k^2"), _o("misclassification", "1 - max p_k")],
     primary="gini", method="class proportions; Gini and misclassification exactly as fractions",
     latex=r"G = 1 - \sum_k p_k^2,\quad H = -\sum_k p_k \log_2 p_k",
     run=impurity_counts, reference=_ref_impurity,
     wrong=lambda counts: (lambda o: dict(o, gini=_r(1 - sum(v / sum(counts) for v in counts) / len(counts) - sum((v / sum(counts)) ** 3 for v in counts))))(impurity_counts(counts)),
     rand=lambda rng: {"counts": [rng.randint(0, 40) for _ in range(rng.randint(2, 5))] + [rng.randint(1, 40)]},
     phrases=["Impurity from class counts", "Gini and entropy from counts"],
     conditions=["counts >= 0 with a positive total"],
     examples=[_ex({"counts": [9, 5]}, {"entropy_bits": 0.940285958671, "gini": 0.459183673469}),
               _ex({"counts": [10, 10, 0]}, {"entropy_bits": 1.0, "gini": 0.5, "misclassification": 0.5})],
     extra_q=[_bad({"counts": [0, 0]})],
     weak="entropy and Gini are checked by scikit-learn's decision-tree impurity of a node with these counts; the misclassification error by a numpy re-derivation",
     level="beginner")


def _percentile(sorted_vals, p):
    n = len(sorted_vals)
    h = (n - 1) * p
    lo = int(math.floor(h))
    hi = min(lo + 1, n - 1)
    return sorted_vals[lo] + (h - lo) * (sorted_vals[hi] - sorted_vals[lo])


def robust_scaler(X):
    M = _mat("X", X, 1, 1)
    d = len(M[0])
    F = [[_F(v) for v in r] for r in M]
    med, iqr = [], []
    for j in range(d):
        col = sorted(r[j] for r in F)
        med.append(_percentile(col, Fraction(1, 2)))
        iqr.append(_percentile(col, Fraction(3, 4)) - _percentile(col, Fraction(1, 4)))
    S = [[(r[j] - med[j]) / iqr[j] if iqr[j] != 0 else r[j] - med[j] for j in range(d)] for r in F]
    return _out(medians=med, iqr=iqr, scaled=S)


def _ref_robust(X):
    np = _np()
    from sklearn.preprocessing import RobustScaler
    s = RobustScaler().fit(np.array(X, dtype=float))
    A = np.array(X, dtype=float)
    q = np.percentile(A, [25, 75], axis=0)
    return {"medians": _f(s.center_), "iqr": _f(q[1] - q[0]), "scaled": _f(s.transform(A))}


_add("ml_robust_scaler",
     name="Robust scaling (median and interquartile range)", topic="preprocessing",
     purpose="Scale each feature by subtracting its median and dividing by its interquartile range (quartiles by linear interpolation), which resists outliers; a column with zero IQR is only centred.",
     inputs=[_matin("X", "data: one row per sample, one column per feature")],
     outputs=[_o("medians", "column medians"), _o("iqr", "column interquartile ranges Q3 - Q1"), _o("scaled", "(x - median) / IQR")],
     primary="medians", method="quartiles at position (n - 1) p with linear interpolation; exact fractions",
     latex=r"x' = \frac{x - \mathrm{median}}{Q_3 - Q_1}",
     run=robust_scaler, reference=_ref_robust,
     wrong=lambda X: (lambda o: dict(o, iqr=[_r(v * 2) for v in o["iqr"]], scaled=[[_r(v / 2) for v in r] for r in o["scaled"]]))(robust_scaler(X)),
     rand=lambda rng: {"X": _rand_matrix(rng, 4)}, phrases=["Robust scaler", "Robust scaling"],
     conditions=["zero IQR: centred only"],
     examples=[_ex({"X": [[1], [2], [3], [4], [100]]}, {"medians": [3.0], "iqr": [2.0], "scaled": [[-1.0], [-0.5], [0.0], [0.5], [48.5]]}),
               _ex({"X": [[1, 5], [4, 2], [2, 8], [8, 1], [3, 3], [5, 9]]})],
     level="intermediate")


def stump_regress(X, y):
    M = _stump_X(X)
    yv = _vec("y", y)
    _need(len(yv) == len(M), "y needs one value per row of X")
    Y = [_F(v) for v in yv]
    n = len(Y)

    def sse(idx):
        vals = [Y[i] for i in idx]
        m = sum(vals) / len(vals)
        return sum((v - m) ** 2 for v in vals)
    parent = sse(range(n))
    _need(parent > 0, "every target is the same: there is nothing to split")
    s, j, t, L, R = _stump_scan(M, lambda l, r: (parent - sse(l) - sse(r)) / n)
    return _out(feature=j, threshold=t, mse_decrease=s, left_mean=sum(Y[i] for i in L) / len(L), right_mean=sum(Y[i] for i in R) / len(R))


def _ref_stump_reg(X, y):
    np = _np()
    from sklearn.tree import DecisionTreeRegressor
    t = DecisionTreeRegressor(max_depth=1, random_state=0).fit(np.array(X, dtype=float), np.array(y))
    tr = t.tree_
    n = tr.n_node_samples
    l, r = tr.children_left[0], tr.children_right[0]
    dec = tr.impurity[0] - (n[l] / n[0] * tr.impurity[l] + n[r] / n[0] * tr.impurity[r])
    return {"feature": int(tr.feature[0]), "threshold": float(tr.threshold[0]), "mse_decrease": float(dec),
            "left_mean": float(tr.value[l].ravel()[0]), "right_mean": float(tr.value[r].ravel()[0])}


_add("ml_decision_stump_regression",
     name="Best regression-tree split (squared error)", topic="classical algorithms",
     purpose="The best single split x_j <= t of a regression tree's root by squared error: the feature, the threshold, the decrease of the mean squared error and the mean target on each side.",
     inputs=[_matin("X", "features (whole numbers)"), _vecin("y", "targets")],
     outputs=[_o("feature", "index of the best feature"), _o("threshold", "threshold t (left when x_j <= t)"),
              _o("mse_decrease", "decrease of the node's mean squared error"), _o("left_mean", "mean target on the left"), _o("right_mean", "mean target on the right")],
     primary="mse_decrease", method="scan every feature and midpoint; exact sums of squares",
     latex=r"\Delta = \frac{1}{n}\left[SSE(S) - SSE(S_L) - SSE(S_R)\right]",
     run=stump_regress, reference=_ref_stump_reg,
     wrong=lambda X, y: (lambda o: dict(o, mse_decrease=_r(o["mse_decrease"] * 0.5), left_mean=o["right_mean"], right_mean=o["left_mean"]))(stump_regress(X, y)),
     rand=lambda rng: _rand_stump(rng, True), phrases=["Regression tree best split", "Best regression split"],
     conditions=["whole-number features", "targets not all equal", "a unique best split"],
     examples=[_ex({"X": [[1], [2], [3], [10], [11], [12]], "y": [1.0, 1.2, 0.8, 5.0, 5.4, 4.6]}, {"threshold": 6.5, "left_mean": 1.0, "right_mean": 5.0}),
               _ex({"X": [[1, 5], [2, 3], [3, 8], [4, 1], [5, 7]], "y": [2.0, 1.0, 4.0, 0.5, 3.5]})],
     level="advanced")


# @@NEXT@@


# ================================================================================================ example values
# Every documented example's outputs as computed by its INDEPENDENT reference on the box (benchmarks/
# example_values_tool.py); the examples gate then checks the shipped code against them.
# @@EXV-BEGIN@@
_EXV = {
    'ml_adam_step': [
        {'m_new': [0.05, -0.1], 'v_new': [0.00025, 0.001], 'w_new': [0.99900000002, -1.99900000001]},
        {'m_new': [0.029, -0.023], 'v_new': [0.001039, 0.0004021], 'w_new': [0.298621781734, 0.101757067997]},
    ],
    'ml_adamw_step': [
        {'m_new': [0.05, -0.1], 'v_new': [0.00025, 0.001], 'w_new': [0.99899000002, -1.99898000001]},
        {'m_new': [0.029, -0.023], 'v_new': [0.001039, 0.0004021], 'w_new': [0.298321781734, 0.101657067997]},
    ],
    'ml_adjusted_mutual_information': [
        {'ami': 0.615384615385, 'expected_mi': 0.266835671996, 'mutual_information': 0.636514168295},
        {'ami': -0.037037037037, 'expected_mi': 0.33629923055, 'mutual_information': 0.308065413582},
    ],
    'ml_adjusted_r2': [
        {'adjusted_r2': 0.96, 'r2': 0.97},
        {'adjusted_r2': 0.933052039381, 'r2': 0.952180028129},
    ],
    'ml_average_precision': [
        {'average_precision': 0.916666666667, 'pr_auc_trapezoid': 0.902777777778},
        {'average_precision': 0.916666666667, 'pr_auc_trapezoid': 0.958333333333},
    ],
    'ml_backpropagation': [
        {'dW1': [[-1.7, -3.4], [1.7, 3.4]], 'dW2': [[-0.17, -0.85]], 'db1': [-1.7, 1.7], 'db2': [-1.7], 'loss': 0.7225},
        {'dW1': [[0.00744901987142, -0.0148980397428], [-0.0604066962162, 0.120813392432]], 'dW2': [[0.136909362353, 0.148816992814], [-0.230594092978, -0.250649910918], [0.093684730625, 0.101832918104]], 'db1': [0.0148980397428, -0.120813392432], 'db2': [0.321717670965, -0.541863560359, 0.220145889395], 'loss': 0.780588236087},
    ],
    'ml_balanced_accuracy': [
        {'adjusted': 0.5, 'balanced_accuracy': 0.75},
        {'adjusted': 0.666666666667, 'balanced_accuracy': 0.777777777778},
    ],
    'ml_balanced_class_weights': [
        {'classes': [0, 1], 'weights': [0.666666666667, 2.0]},
        {'classes': [1, 2, 3], 'weights': [2.0, 1.0, 0.666666666667]},
    ],
    'ml_batch_norm_forward': [
        {'batch_mean': [3.0, 6.0], 'batch_var': [2.66666666667, 10.6666666667], 'output': [[-1.22474487139, -1.22474487139], [0.0, 0.0], [1.22474487139, 1.22474487139]]},
        {'batch_mean': [1.5, 0.0, 1.83333333333], 'batch_var': [0.666666666667, 0.666666666667, 0.388888888889], 'output': [[-1.22473568591, 0.387632157046, -0.465484388475], [1.11022302463e-16, 1.0, 1.1380624461], [1.22473568591, 1.61236784295, -3.67257805763]]},
    ],
    'ml_bce_with_logits': [
        {'gradient': [-0.0397343073407, 0.0896471404567, -0.207486443734], 'loss': 0.47142222758},
        {'gradient': [0.00878723809682, 0.237221258406], 'loss': 0.662884261226},
    ],
    'ml_bernoulli_naive_bayes': [
        {'classes': [0, 1], 'posteriors': [0.860022396417, 0.139977603583], 'prediction': 0},
        {'classes': [1, 2], 'posteriors': [0.166666666667, 0.833333333333], 'prediction': 2},
    ],
    'ml_bias_variance_decomposition': [
        {'bias_squared': 0.138888888889, 'expected_mse': 0.666666666667, 'variance': 0.527777777778},
        {'bias_squared': 0.005, 'expected_mse': 0.0333333333333, 'variance': 0.0283333333333},
    ],
    'ml_binary_confusion_matrix': [
        {'FN': 1, 'FP': 1, 'TN': 3, 'TP': 3, 'accuracy': 0.75, 'error_rate': 0.25},
        {'FN': 1, 'FP': 2, 'TN': 5, 'TP': 2, 'accuracy': 0.7, 'error_rate': 0.3},
    ],
    'ml_binary_metrics_from_counts': [
        {'accuracy': 0.85, 'balanced_accuracy': 0.853535353535, 'f1': 0.842105263158, 'fnr': 0.111111111111, 'fpr': 0.181818181818, 'jaccard': 0.727272727273, 'npv': 0.9, 'precision': 0.8, 'recall': 0.888888888889, 'specificity': 0.818181818182},
        {'accuracy': 0.96, 'balanced_accuracy': 0.933333333333, 'f1': 0.818181818182, 'fnr': 0.1, 'fpr': 0.0333333333333, 'jaccard': 0.692307692308, 'npv': 0.988636363636, 'precision': 0.75, 'recall': 0.9, 'specificity': 0.966666666667},
    ],
    'ml_brier_score': [
        {'brier': 0.075},
        {'brier': 0.0866666666667},
    ],
    'ml_calinski_harabasz_index': [
        {'between': 28.8333333333, 'calinski_harabasz': 14.6455026455, 'within': 7.875},
        {'between': 76.2, 'calinski_harabasz': 76.2, 'within': 1.0},
    ],
    'ml_classification_report': [
        {'classes': [0, 1, 2], 'f1': [1.0, 0.666666666667, 0.666666666667], 'macro_f1': 0.777777777778, 'precision': [1.0, 0.666666666667, 0.666666666667], 'recall': [1.0, 0.666666666667, 0.666666666667], 'support': [2, 3, 3]},
        {'classes': [1, 2, 3], 'f1': [0.5, 0.8, 0.8], 'macro_f1': 0.7, 'precision': [0.5, 0.666666666667, 1.0], 'recall': [0.5, 1.0, 0.666666666667], 'support': [2, 2, 3]},
    ],
    'ml_cohen_kappa': [
        {'kappa': 0.619047619048},
        {'kappa': 0.769230769231},
    ],
    'ml_conv2d_forward': [
        {'output': [[15.0, 16.0], [6.0, 15.0]], 'output_sum': 52.0},
        {'output': [[2.0, 0.0], [10.0, 17.0]], 'output_sum': 29.0},
    ],
    'ml_conv_output_size': [
        {'effective_kernel': 3, 'output_size': 32},
        {'effective_kernel': 7, 'output_size': 112},
        {'effective_kernel': 5, 'output_size': 60},
    ],
    'ml_conv_parameter_count': [
        {'biases': 64, 'total': 1792, 'weights': 1728},
        {'biases': 0, 'total': 288, 'weights': 288},
    ],
    'ml_cosine_lr_schedule': [
        {'lr': 0.05},
        {'lr': 0.00154982143313},
    ],
    'ml_cross_entropy_logits': [
        {'gradient': [-0.340998861114, 0.242432970705, 0.0985658904093], 'loss': 0.417030016278, 'probability': 0.659001138886},
        {'gradient': [0.0713666110359, 0.0159240433497, -0.130576692059, 0.0432860376731], 'loss': 0.139925151561, 'probability': 0.869423307941},
    ],
    'ml_davies_bouldin_index': [
        {'davies_bouldin': 0.37473357715},
        {'davies_bouldin': 0.163636363636},
    ],
    'ml_decision_stump_classification': [
        {'feature': 0, 'impurity_decrease': 0.5, 'n_left': 3, 'n_right': 3, 'threshold': 7.0},
        {'feature': 0, 'impurity_decrease': 0.521640636343, 'n_left': 4, 'n_right': 3, 'threshold': 4.5},
    ],
    'ml_decision_stump_regression': [
        {'feature': 0, 'left_mean': 1.0, 'mse_decrease': 4.0, 'right_mean': 5.0, 'threshold': 6.5},
        {'feature': 1, 'left_mean': 1.16666666667, 'mse_decrease': 1.60166666667, 'right_mean': 3.75, 'threshold': 6.0},
    ],
    'ml_dense_forward_pass': [
        {'hidden': [0.1, 0.5], 'max_output': -0.35, 'output': [-0.35]},
        {'hidden': [-0.46211715726, 0.691069469833], 'max_output': 0.491574315988, 'output': [0.157616744625, 0.491574315988, 0.350808939386]},
    ],
    'ml_dense_parameter_count': [
        {'per_layer': [100480, 1290], 'total': 101770},
        {'per_layer': [12, 16, 4], 'total': 32},
    ],
    'ml_distance_metrics': [
        {'chebyshev': 4.0, 'cosine_distance': 0.144517611464, 'cosine_similarity': 0.855482388536, 'euclidean': 5.0, 'manhattan': 7.0},
        {'chebyshev': 1.0, 'cosine_distance': 0.292893218813, 'cosine_similarity': 0.707106781187, 'euclidean': 1.0, 'manhattan': 1.0},
    ],
    'ml_entropy': [
        {'entropy': 1.5, 'max_entropy': 1.58496250072, 'normalized_entropy': 0.946394630357, 'perplexity': 2.82842712475},
        {'entropy': 0.325082973391, 'max_entropy': 0.69314718056, 'normalized_entropy': 0.468995593589, 'perplexity': 1.38414548846},
        {'entropy': 2.0, 'max_entropy': 2.0, 'normalized_entropy': 1.0, 'perplexity': 4.0},
    ],
    'ml_expected_calibration_error': [
        {'ece': 0.14875, 'mce': 0.36, 'n_nonempty_bins': 5},
        {'ece': 0.23, 'mce': 0.3, 'n_nonempty_bins': 2},
    ],
    'ml_fbeta_score': [
        {'fbeta': 0.869565217391, 'precision': 0.8, 'recall': 0.888888888889},
        {'fbeta': 0.816326530612, 'precision': 0.8, 'recall': 0.888888888889},
    ],
    'ml_focal_loss': [
        {'gradient': [-0.000405911682949, 0.0131452824944, -0.0438459851562], 'loss': 0.0372657780183},
        {'gradient': [0.204393619048, -0.106389370797], 'loss': 0.563942130613},
    ],
    'ml_gaussian_naive_bayes': [
        {'classes': [0, 1], 'posteriors': [1.54428667821e-07, 0.999999845571], 'prediction': 1},
        {'classes': [0, 1], 'posteriors': [0.727570105482, 0.272429894518], 'prediction': 0},
    ],
    'ml_gini_split': [
        {'gini_children': 0.375, 'gini_decrease': 0.125, 'gini_parent': 0.5},
        {'gini_children': 0.333333333333, 'gini_decrease': 0.277777777778, 'gini_parent': 0.611111111111},
    ],
    'ml_gradient_descent_step_linear': [
        {'b_new': 1.3, 'loss_after': 1.69, 'loss_before': 3.5, 'w_new': [2.0]},
        {'b_new': 0.018, 'loss_after': 0.135083851852, 'loss_before': 0.916666666667, 'w_new': [0.164666666667, 0.282666666667]},
    ],
    'ml_hamming_loss': [
        {'hamming_loss': 0.166666666667, 'n_mismatches': 1, 'subset_accuracy': 0.5},
        {'hamming_loss': 0.5, 'n_mismatches': 3, 'subset_accuracy': 0.333333333333},
    ],
    'ml_hierarchical_clustering': [
        {'final_height': 10.4043260233, 'heights': [1.0, 1.5, 4.0, 10.4043260233]},
        {'final_height': 15.5892591229, 'heights': [1.0, 1.5, 5.66789202438, 15.5892591229]},
    ],
    'ml_hinge_loss': [
        {'hinge': 0.75, 'n_violations': 3, 'squared_hinge': 0.995},
        {'hinge': 0.0333333333333, 'n_violations': 1, 'squared_hinge': 0.00333333333333},
    ],
    'ml_homogeneity_completeness_v_measure': [
        {'completeness': 1.0, 'homogeneity': 0.579380164286, 'v_measure': 0.733680436651},
        {'completeness': 0.420619835714, 'homogeneity': 0.666666666667, 'v_measure': 0.515803742979},
    ],
    'ml_huber_loss': [
        {'huber': 0.1875, 'pseudo_huber': 0.162570384968, 'smooth_l1': 0.1875},
        {'huber': 1.16666666667, 'pseudo_huber': 0.967620398167, 'smooth_l1': 0.777777777778},
    ],
    'ml_impurity_from_counts': [
        {'entropy_bits': 0.940285958671, 'gini': 0.459183673469, 'misclassification': 0.357142857143},
        {'entropy_bits': 1.0, 'gini': 0.5, 'misclassification': 0.5},
    ],
    'ml_information_gain': [
        {'conditional_entropy': 0.632482355162, 'gain_ratio': 0.14621357599, 'information_gain': 0.230638213404, 'parent_entropy': 0.863120568567, 'split_info': 1.57740628285},
        {'conditional_entropy': 0.0, 'gain_ratio': 1.0, 'information_gain': 1.0, 'parent_entropy': 1.0, 'split_info': 1.0},
    ],
    'ml_joint_conditional_entropy': [
        {'H_X': 1.0, 'H_XY': 1.5, 'H_X_given_Y': 0.688721875541, 'H_Y': 0.811278124459, 'H_Y_given_X': 0.5, 'mutual_information': 0.311278124459},
        {'H_X': 0.673011667009, 'H_XY': 1.69574253417, 'H_X_given_Y': 0.606842558824, 'H_Y': 1.08889997535, 'H_Y_given_X': 1.02273086716, 'mutual_information': 0.0661691081848},
    ],
    'ml_kfold_cv_linear_regression': [
        {'fold_mse': [0.0925, 0.0243295847751, 0.01845], 'mean_mse': 0.045093194925},
        {'fold_mse': [0.0537764003673, 0.0391984182789, 0.326697530864, 0.411248852158], 'mean_mse': 0.207730300417},
    ],
    'ml_kfold_sizes': [
        {'first_test_fold': [0, 3], 'test_sizes': [4, 3, 3], 'train_sizes': [6, 7, 7]},
        {'first_test_fold': [0, 19], 'test_sizes': [20, 20, 20, 20, 20], 'train_sizes': [80, 80, 80, 80, 80]},
    ],
    'ml_kl_divergence_discrete': [
        {'cross_entropy': 1.73696559417, 'entropy_p': 1.0, 'js_divergence': 0.146793102436, 'kl_pq': 0.736965594166, 'kl_qp': 0.531004406411},
        {'cross_entropy': 1.09913302852, 'entropy_p': 1.01331268735, 'js_divergence': 0.0226001078708, 'kl_pq': 0.0858203411726, 'kl_qp': 0.0980925206208},
    ],
    'ml_kmeans': [
        {'centroids': [[1.25, 1.5], [3.9, 5.1]], 'inertia': 8.525, 'labels': [0, 0, 1, 1, 1, 1, 1]},
        {'centroids': [[0.5], [5.5], [11.5]], 'inertia': 1.5, 'labels': [0, 0, 1, 1, 2, 2]},
    ],
    'ml_kmeans_step': [
        {'inertia': 19.25, 'labels': [0, 0, 1, 1, 1, 1, 1], 'new_centroids': [[1.25, 1.5], [3.9, 5.1]]},
        {'inertia': 27.0, 'labels': [0, 0, 1, 1, 1], 'new_centroids': [[0.5], [7.33333333333]]},
    ],
    'ml_knn_classify': [
        {'neighbor_distances': [1.28062484749, 1.56204993518, 2.00997512422], 'neighbors': [1, 2, 4], 'prediction': 0, 'votes': 2},
        {'neighbor_distances': [0.5, 1.5, 2.5, 3.5, 4.5], 'neighbors': [1, 2, 3, 4, 5], 'prediction': 2, 'votes': 3},
    ],
    'ml_knn_regress': [
        {'neighbor_distances': [0.4, 0.6], 'neighbors': [1, 2], 'prediction': 2.75},
        {'neighbor_distances': [2.0, 2.0, 3.0], 'neighbors': [0, 1, 2], 'prediction': 14.5},
    ],
    'ml_label_smoothing_cross_entropy': [
        {'gradient': [-0.274332194447, 0.209099637371, 0.065232557076], 'loss': 0.513696682945},
        {'gradient': [0.0213666110359, -0.83407595665, 0.819423307941, -0.00671396232691], 'loss': 3.81492515156},
    ],
    'ml_lasso_regression': [
        {'b': 0.366666666667, 'n_nonzero': 1, 'w': [0.914285714286, 0.0]},
        {'b': 1.5, 'n_nonzero': 2, 'w': [1.01785714286, 0.5, 0.0]},
    ],
    'ml_linear_svm_decision': [
        {'classes': [1, -1, 1], 'decision': [1.0, -2.0, 1.0], 'distances': [0.4472135955, 0.894427191, 0.4472135955], 'margin_width': 0.894427191},
        {'classes': [1, 1], 'decision': [0.25, 0.25], 'distances': [0.204124145232, 0.204124145232], 'margin_width': 1.63299316186},
    ],
    'ml_log_loss_binary': [
        {'log_loss': 0.299001158669, 'log_loss_bits': 0.431367488832},
        {'log_loss': 0.279776563579, 'log_loss_bits': 0.403632260833},
    ],
    'ml_log_loss_multiclass': [
        {'log_loss': 0.520215916088, 'perplexity': 1.68239086574},
        {'log_loss': 0.569717141594, 'perplexity': 1.76776695297},
    ],
    'ml_log_softmax': [
        {'log_probabilities': [-2.40760596444, -1.40760596444, -0.407605964444], 'log_sum_exp': 3.40760596444},
        {'log_probabilities': [0.0, -1000.0, -2000.0], 'log_sum_exp': 1000.0},
    ],
    'ml_logistic_loss_gradient': [
        {'grad_b': -0.0246288425244, 'grad_w': [-0.32406170005, -0.0492576850487], 'loss': 0.485375598986},
        {'grad_b': -0.0970082570782, 'grad_w': [-0.14378334716], 'loss': 0.379583983227},
    ],
    'ml_logistic_regression_fit': [
        {'b': -1.66224678262, 'objective': 4.65311431474, 'train_accuracy': 0.75, 'w': [0.73877634783]},
        {'b': -2.98571624633, 'objective': 1.28269804032, 'train_accuracy': 1.0, 'w': [0.893826419271, 0.274742800239]},
    ],
    'ml_logistic_regression_predict': [
        {'classes': [0, 1, 0], 'logits': [-0.2, 1.8, -0.2], 'probabilities': [0.450166002688, 0.8581489351, 0.450166002688]},
        {'classes': [0, 0, 1], 'logits': [-1.0, 0.0, 1.0], 'probabilities': [0.26894142137, 0.5, 0.73105857863]},
    ],
    'ml_loocv_linear_regression': [
        {'loocv_mse': 0.0351518148321, 'press': 0.210910888993},
        {'loocv_mse': 0.183869590525, 'press': 1.4709567242},
    ],
    'ml_mape': [
        {'mape': 0.0833333333333, 'mape_percent': 8.33333333333},
        {'mape': 0.327380952381, 'mape_percent': 32.7380952381},
    ],
    'ml_matthews_correlation': [
        {'mcc': 0.703526470681},
        {'mcc': 0.478091443734},
    ],
    'ml_mean_average_precision': [
        {'average_precision': [0.833333333333, 0.45], 'map': 0.641666666667},
        {'average_precision': [0.444444444444], 'map': 0.444444444444},
    ],
    'ml_mean_reciprocal_rank': [
        {'mrr': 0.611111111111, 'reciprocal_ranks': [0.333333333333, 0.5, 1.0]},
        {'mrr': 0.125, 'reciprocal_ranks': [0.25, 0.0]},
    ],
    'ml_mean_squared_log_error': [
        {'msle': 0.0397301229846, 'rmsle': 0.199324165581},
        {'msle': 0.0453617649511, 'rmsle': 0.21298301564},
    ],
    'ml_minmax_scaler': [
        {'maxs': [5.0, 40.0], 'mins': [1.0, 10.0], 'scaled': [[0.0, 0.0], [0.25, 0.333333333333], [0.5, 1.0], [1.0, 0.666666666667]]},
        {'maxs': [9.0, 7.0], 'mins': [-3.0, 7.0], 'scaled': [[0.0, 0.0], [0.25, 0.0], [1.0, 0.0]]},
    ],
    'ml_mse_gradient': [
        {'grad_b': -1.8, 'grad_w': [-6.46666666667, -8.26666666667], 'loss': 0.916666666667},
        {'grad_b': -3.0, 'grad_w': [-10.0], 'loss': 3.5},
    ],
    'ml_multiclass_confusion_matrix': [
        {'accuracy': 0.75, 'classes': [0, 1, 2], 'matrix': [[2, 0, 0], [0, 2, 1], [0, 1, 2]], 'n_correct': 6},
        {'accuracy': 0.666666666667, 'classes': [1, 2, 3], 'matrix': [[1, 0, 1], [0, 1, 0], [1, 0, 2]], 'n_correct': 4},
    ],
    'ml_multiclass_hinge_loss': [
        {'crammer_singer': 0.433333333333, 'weston_watkins': 0.2},
        {'crammer_singer': 1.3, 'weston_watkins': 0.566666666667},
    ],
    'ml_multihead_attention': [
        {'output': [[2.32095380269, 3.32095380269, 5.67904619731, 6.67904619731]], 'output_sum': 18.0},
        {'output': [[1.15095519357, 0.635824672851], [1.26695639475, 1.26695639475]], 'output_sum': 4.32069265593},
    ],
    'ml_multinomial_naive_bayes': [
        {'classes': [0, 1], 'posteriors': [0.529780564263, 0.470219435737], 'prediction': 0},
        {'classes': [1, 2], 'posteriors': [0.854392490787, 0.145607509213], 'prediction': 1},
    ],
    'ml_mutual_information_labels': [
        {'H_x': 1.58496250072, 'H_x_given_y': 0.666666666667, 'H_y': 0.918295834054, 'mutual_information': 0.918295834054},
        {'H_x': 1.08219553004, 'H_x_given_y': 0.477385626221, 'H_y': 1.08219553004, 'mutual_information': 0.604809903818},
    ],
    'ml_ndcg_at_k': [
        {'dcg': 6.86112668859, 'idcg': 7.1409951841, 'ndcg': 0.960808194336},
        {'dcg': 12.3927892607, 'idcg': 12.916508275, 'ndcg': 0.959453514593},
    ],
    'ml_normalized_mutual_information': [
        {'mutual_information': 0.636514168295, 'nmi': 0.733680436651},
        {'mutual_information': 0.318257084147, 'nmi': 0.479138767492},
    ],
    'ml_one_hot_encoding': [
        {'categories': ['blue', 'green', 'red'], 'codes': [2, 1, 2, 0, 1, 2], 'counts': [1, 2, 3], 'one_hot': [[0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        {'categories': ['high', 'low', 'medium'], 'codes': [1, 0, 2, 1], 'counts': [1, 2, 1], 'one_hot': [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]]},
    ],
    'ml_pca': [
        {'component_1': [0.677873398528, 0.735178655544], 'explained_variance': [1.28402771217, 0.0490833989383], 'explained_variance_ratio': [0.963181314349, 0.0368186856514], 'scores_1': [0.827970186201, -1.77758032528, 0.992197494415, 0.274210415975, 1.67580141864, 0.912949103159, -0.0991094374984, -1.1445721638, -0.438046136762, -1.22382055505]},
        {'component_1': [0.498282977681, 0.691615919337, 0.522858962124], 'explained_variance': [8.2379768711, 2.0120231289, 0.25], 'explained_variance_ratio': [0.784569225819, 0.191621250371, 0.0238095238095], 'scores_1': [-2.38311541213, -2.57644835379, 3.27801712742, -1.21779249956, 2.89933913806]},
    ],
    'ml_perceptron_epoch': [
        {'b_new': 1.0, 'w_new': [2.0, 1.0]},
        {'b_new': 0.5, 'w_new': [0.5]},
    ],
    'ml_polynomial_kernel': [
        {'kernel': 144.0},
        {'kernel': 34.328125},
    ],
    'ml_pooling': [
        {'avg_pool': [[3.75, 2.25], [4.5, 4.0]], 'max_pool': [[6.0, 4.0], [8.0, 9.0]]},
        {'avg_pool': [[3.0, 4.0], [6.0, 7.0]], 'max_pool': [[5.0, 6.0], [8.0, 9.0]]},
    ],
    'ml_precision_recall_at_k': [
        {'hit_rate': 1, 'hits': 3, 'precision_at_k': 0.6, 'recall_at_k': 0.6},
        {'hit_rate': 0, 'hits': 0, 'precision_at_k': 0.0, 'recall_at_k': 0.0},
    ],
    'ml_precision_recall_f1_averaged': [
        {'f1': 0.777777777778, 'precision': 0.777777777778, 'recall': 0.777777777778},
        {'f1': 0.75, 'precision': 0.75, 'recall': 0.75},
        {'f1': 0.75, 'precision': 0.75, 'recall': 0.75},
    ],
    'ml_quantile_loss': [
        {'pinball': 0.0333333333333},
        {'pinball': 1.66666666667},
    ],
    'ml_rand_index': [
        {'adjusted_rand_index': 0.444444444444, 'fowlkes_mallows': 0.654653670708, 'rand_index': 0.733333333333},
        {'adjusted_rand_index': 0.242424242424, 'fowlkes_mallows': 0.471404520791, 'rand_index': 0.666666666667},
    ],
    'ml_rbf_kernel_matrix': [
        {'K': [[1.0, 0.606530659713, 0.135335283237], [0.606530659713, 1.0, 0.0820849986239], [0.135335283237, 0.0820849986239, 1.0]], 'mean_offdiagonal': 0.274650313858},
        {'K': [[1.0, 0.367879441171, 0.000123409804087], [0.367879441171, 1.0, 0.0183156388887], [0.000123409804087, 0.0183156388887, 1.0]], 'mean_offdiagonal': 0.128772829955},
    ],
    'ml_receptive_field': [
        {'jump': 1, 'receptive_field': 7},
        {'jump': 4, 'receptive_field': 27},
    ],
    'ml_regression_metrics': [
        {'explained_variance': 0.957173447537, 'mae': 0.5, 'max_error': 1.0, 'median_ae': 0.5, 'mse': 0.375, 'r2': 0.948608137045, 'rmse': 0.612372435696},
        {'explained_variance': 0.9708, 'mae': 2.4, 'max_error': 3.0, 'median_ae': 2.0, 'mse': 6.0, 'r2': 0.97, 'rmse': 2.44948974278},
    ],
    'ml_ridge_regression': [
        {'b': 0.332727272727, 'train_mse': 0.0349289256198, 'w': [0.909090909091]},
        {'b': 0.651041666667, 'train_mse': 0.0401023582176, 'w': [1.13541666667, 1.27083333333]},
    ],
    'ml_rnn_parameter_count': [
        {'per_layer': [117760], 'total': 117760},
        {'per_layer': [37632, 74496], 'total': 112128},
    ],
    'ml_robust_scaler': [
        {'iqr': [2.0], 'medians': [3.0], 'scaled': [[-1.0], [-0.5], [0.0], [0.5], [48.5]]},
        {'iqr': [2.5, 5.0], 'medians': [3.5, 4.0], 'scaled': [[-1.0, 0.2], [0.2, -0.4], [-0.6, 0.8], [1.8, -0.6], [-0.2, -0.2], [0.6, 1.0]]},
    ],
    'ml_roc_auc': [
        {'auc': 0.833333333333, 'gini': 0.666666666667, 'n_neg': 2, 'n_pos': 3},
        {'auc': 0.75, 'gini': 0.5, 'n_neg': 2, 'n_pos': 2},
        {'auc': 0.875, 'gini': 0.75, 'n_neg': 2, 'n_pos': 2},
    ],
    'ml_roc_curve': [
        {'auc': 0.833333333333, 'fpr': [0.0, 0.0, 0.0, 0.5, 0.5, 1.0], 'thresholds': [0.9, 0.8, 0.3, 0.2, 0.1], 'tpr': [0.0, 0.333333333333, 0.666666666667, 0.666666666667, 1.0, 1.0]},
        {'auc': 0.944444444444, 'fpr': [0.0, 0.0, 0.0, 0.333333333333, 0.666666666667, 1.0], 'thresholds': [4.0, 3.0, 2.0, 1.0, 0.0], 'tpr': [0.0, 0.333333333333, 0.666666666667, 1.0, 1.0, 1.0]},
    ],
    'ml_scaled_dot_product_attention': [
        {'output': [[3.0, 4.0], [3.40667255608, 4.40667255608]], 'output_sum': 14.8133451122, 'weights': [[0.40111209268, 0.19777581464, 0.40111209268], [0.19777581464, 0.40111209268, 0.40111209268]]},
        {'output': [[1.0], [1.66976154933], [2.25523476523]], 'output_sum': 4.92499631455, 'weights': [[1.0, 0.0, 0.0], [0.330238450673, 0.669761549327, 0.0], [0.248255078258, 0.248255078258, 0.503489843485]]},
    ],
    'ml_sgd_momentum_step': [
        {'buf_new': [0.59, -0.82], 'w_new': [0.941, -1.918]},
        {'buf_new': [0.203], 'w_new': [0.296143]},
    ],
    'ml_sigmoid': [
        {'derivative': 0.25, 'log_sigmoid': -0.69314718056, 'sigmoid': 0.5},
        {'derivative': 0.104993585404, 'log_sigmoid': -0.126928011043, 'sigmoid': 0.880797077978},
        {'derivative': 6.30511676015e-16, 'log_sigmoid': -35.0, 'sigmoid': 6.30511676015e-16},
    ],
    'ml_silhouette_score': [
        {'samples': [0.785495870518, 0.728161920035, 0.287385025256, 0.591063503361, 0.630078239981, 0.660704874448], 'silhouette': 0.6138149056},
        {'samples': [0.818181818182, 0.777777777778, 0.777777777778, 0.8, 0.0], 'silhouette': 0.634747474747},
    ],
    'ml_softmax': [
        {'log_sum_exp': 3.40760596444, 'probabilities': [0.0900305731704, 0.244728471055, 0.665240955775]},
        {'log_sum_exp': 1001.31326169, 'probabilities': [0.26894142137, 0.73105857863]},
    ],
    'ml_softmax_temperature': [
        {'entropy': 1.02019133673, 'probabilities': [0.186323723226, 0.307195885718, 0.506480391056]},
        {'entropy': 0.290040503693, 'probabilities': [0.934071845901, 0.0171081226257, 0.00231533262119, 0.0465046988524]},
    ],
    'ml_standard_scaler': [
        {'means': [2.5, 25.0], 'scaled': [[-1.3416407865, -1.3416407865], [-0.4472135955, -0.4472135955], [0.4472135955, 0.4472135955], [1.3416407865, 1.3416407865]], 'stds': [1.11803398875, 11.1803398875]},
        {'means': [0.5, 5.0], 'scaled': [[-1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], 'stds': [0.5, 0.0]},
    ],
    'ml_step_lr_schedule': [
        {'decays': 2, 'lr': 0.001},
        {'decays': 0, 'lr': 0.05},
    ],
    'ml_threshold_metrics': [
        {'FN': 1, 'FP': 1, 'TN': 1, 'TP': 2, 'accuracy': 0.6, 'precision': 0.666666666667, 'recall': 0.666666666667},
        {'FN': 0, 'FP': 1, 'TN': 2, 'TP': 3, 'accuracy': 0.833333333333, 'precision': 0.75, 'recall': 1.0},
    ],
    'ml_top_k_accuracy': [
        {'n_hits': 3, 'top_k_accuracy': 0.75},
        {'n_hits': 1, 'top_k_accuracy': 0.333333333333},
    ],
    'ml_train_test_split_sizes': [
        {'n_test': 20, 'n_train': 80},
        {'n_test': 12, 'n_train': 35},
    ],
    'ml_transformer_parameter_count': [
        {'attention': 1050624, 'feed_forward': 2099712, 'layer_norms': 2048, 'per_layer': 3152384, 'total': 18914304},
        {'attention': 2362368, 'feed_forward': 4722432, 'layer_norms': 3072, 'per_layer': 7087872, 'total': 85054464},
    ],
    'ml_triplet_loss': [
        {'d_an': 2.0, 'd_ap': 1.41421356237, 'loss': 0.414213562373},
        {'d_an': 1.5, 'd_ap': 1.11803398875, 'loss': 0.11803398875},
    ],
    'ml_tweedie_deviance': [
        {'deviance': 1.42601513196},
        {'deviance': 1.05685281944},
        {'deviance': 1.77817459305},
    ],
    'ml_warmup_cosine_lr_schedule': [
        {'lr': 0.00055},
        {'lr': 0.0005},
    ],
}
# @@EXV-END@@

for _k, _vals in _EXV.items():
    if _k in PROCEDURES:
        for _e, _v in zip(PROCEDURES[_k]["examples"], _vals):
            _e["outputs"].update(_v)
