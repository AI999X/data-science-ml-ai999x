#![allow(unused_imports)]
//! Data Science and ML's own live figures, continued: tables and matrices (contingency tables, confusion matrices,
//! inputs beside outputs), bars of scores and counts, networks with their weights, before-and-after scalings.

use crate::figures::{hbars, hheat, hfield, hdecision, htable, ans, fl, it, mat_out, nin, tl, tm, tx, words_of};
use crate::plot::{self as pl, list, BAR, DOT};
use ax_engine::figkit::{self as k, num, Canvas, Solved};
use ax_engine::figures::Figure;

pub fn live(s: &Solved) -> Vec<Option<Figure>> {
    match s.key {
        "ml_adjusted_mutual_information" | "ml_normalized_mutual_information" => vec![contingency(s), method_scores(s)],
        "ml_cohen_kappa" => vec![contingency(s), method_scores(s)],
        "ml_homogeneity_completeness_v_measure" => vec![contingency(s)],
        "ml_information_gain" | "ml_mutual_information_labels" | "ml_rand_index" => vec![contingency(s), out_scores(s)],
        "ml_regression_metrics" => vec![out_scores(s)],
        "ml_top_k_accuracy" => vec![top_k(s)],
        "ml_backpropagation" => vec![panels(s, &[("W1", true), ("W2", true), ("dW1", false), ("dW2", false)]), network(s)],
        "ml_dense_forward_pass" => vec![network(s)],
        "ml_batch_norm_forward" => vec![panels(s, &[("X", true), ("output", false)]), before_after(s)],
        "ml_log_loss_multiclass" => vec![panels(s, &[("P", true)]), per_sample(s)],
        "ml_multiclass_hinge_loss" => vec![panels(s, &[("D", true)]), per_sample(s)],
        "ml_mape" | "ml_mean_squared_log_error" => vec![per_sample(s)],
        "ml_minmax_scaler" | "ml_standard_scaler" => vec![before_after(s), panels(s, &[("X", true), ("scaled", false)])],
        "ml_robust_scaler" => vec![before_after(s)],
        "ml_binary_confusion_matrix" | "ml_matthews_correlation" => vec![confusion(s)],
        "ml_binary_metrics_from_counts" => vec![confusion(s), rates(s)],
        "ml_precision_recall_f1_averaged" => vec![averages(s), confusion(s)],
        "ml_multiclass_confusion_matrix" => vec![multi_confusion(s, false), multi_confusion(s, true)],
        "ml_dense_parameter_count" | "ml_one_hot_encoding" | "ml_rnn_parameter_count" | "ml_transformer_parameter_count" => {
            vec![counts(s)]
        }
        "ml_bernoulli_naive_bayes" | "ml_multinomial_naive_bayes" => vec![nb_data(s)],
        "ml_softmax" | "ml_softmax_temperature" => vec![probabilities(s)],
        "ml_mean_average_precision" | "ml_mean_reciprocal_rank" => vec![ranks(s)],
        "ml_joint_conditional_entropy" => vec![info_diagram(s), joint(s)],
        "ml_entropy" => vec![distribution(s)],
        "ml_kl_divergence_discrete" => vec![distributions(s)],
        "ml_label_smoothing_cross_entropy" | "ml_cross_entropy_logits" => vec![logits(s)],
        "ml_hierarchical_clustering" => vec![heights(s)],
        "ml_gini_split" => vec![impurity(s)],
        "ml_hamming_loss" => vec![labels(s)],
        "ml_balanced_accuracy" => vec![recalls(s)],
        "ml_classification_report" => vec![report(s)],
        "ml_balanced_class_weights" => vec![weights(s)],
        _ => vec![],
    }
}

fn classes(v: &[f64]) -> Vec<f64> {
    let mut c = v.to_vec();
    c.sort_by(|a, b| a.partial_cmp(b).unwrap());
    c.dedup();
    c
}

/// Bars with their values (kit), one or several groups.
fn group_bars(labels: &[String], groups: &[(String, Vec<f64>)], xlab: &str, ylab: &str, header: &str) -> String {
    let g: Vec<(String, Vec<Option<f64>>)> = groups.iter().map(|(n, v)| (n.clone(), v.iter().map(|x| Some(*x)).collect())).collect();
    k::bar_groups(labels, &g, xlab, ylab, &pl::clip(header, 44))
}

/// The two label lists of a comparison.
fn two_lists(s: &Solved) -> Option<(Vec<f64>, Vec<f64>, &'static str, &'static str)> {
    for (a, b, na, nb) in [("labels_true", "labels_pred", "true", "cluster"), ("y1", "y2", "rater 1", "rater 2"),
                           ("x", "y", "x", "y")] {
        if let (Some(u), Some(v)) = (s.vec(a), s.vec(b)) {
            return (u.len() == v.len()).then_some((u, v, na, nb));
        }
    }
    None
}

/// A table of counts of each pair (row label, column label), tallied from the stated lists.
fn contingency(s: &Solved) -> Option<Figure> {
    let (u, v, na, nb) = two_lists(s)?;
    let (cu, cv) = (classes(&u), classes(&v));
    if cu.len() > 12 || cv.len() > 12 {
        return None;
    }
    let mut m = vec![vec![0.0; cv.len()]; cu.len()];
    for (a, b) in u.iter().zip(&v) {
        let i = cu.iter().position(|c| c == a)?;
        let j = cv.iter().position(|c| c == b)?;
        m[i][j] += 1.0;
    }
    let short = |n: &str| match n { "cluster" => "c", "rater 1" => "r1", "rater 2" => "r2", other => other }.to_string();
    let rows: Vec<String> = cu.iter().map(|c| format!("{} {}", short(na), num(*c))).collect();
    let cols: Vec<String> = cv.iter().map(|c| format!("{} {}", short(nb), num(*c))).collect();
    let rets: Vec<String> = s.a.values.iter().filter_map(|x| x.1.as_f64().map(|v| format!("{} {}", x.0, num(v)))).collect();
    let body = hheat(&m, &rows, &cols, &pl::clip(&rets.iter().take(2).cloned().collect::<Vec<_>>().join(", "), 46));
    Some(s.chart("contingency", "Counts of each pair of labels",
                 &format!("Counts of each pair ({na}, {nb}) (rows {na}, columns {nb}) tallied from the stated labels {na} = {} and {nb} = {}; returned: {}.",
                          list(&u, 16), list(&v, 16), rets.join(", ")), &body))
}

/// The capability under each of its choices of method, one run each.
fn method_scores(s: &Solved) -> Option<Figure> {
    let (sym, choices, out, what): (&str, &[&str], &str, &str) = match s.key {
        "ml_adjusted_mutual_information" => ("method", &["arithmetic", "geometric", "min", "max"], "ami", "adjusted mutual information"),
        "ml_normalized_mutual_information" => ("method", &["arithmetic", "geometric", "min", "max"], "nmi", "normalised mutual information"),
        _ => ("weights", &["none", "linear", "quadratic"], "kappa", "Cohen's κ"),
    };
    let mine = s.txt(sym)?;
    let mut labels = vec![];
    let mut vals = vec![];
    let mut hl = vec![];
    for c in choices {
        if let Some(v) = s.with_input(&[(sym, tx(c))]).and_then(|a| ans(&a, out)) {
            if *c == mine {
                hl.push(labels.len());
            }
            labels.push(c.to_string());
            vals.push(v);
        }
    }
    if vals.len() < 2 {
        return None;
    }
    let body = hbars(&labels, &vals, &hl, sym, what, &format!("{mine}: {}", num(s.out(out)?)));
    Some(s.chart("scores", &format!("{} by {sym}", cap1(what)),
                 &format!("The {what} of the stated labels with each {sym}, one run each: {}; the visitor's choice in the first colour.",
                          labels.iter().zip(&vals).map(|(l, v)| format!("{l} {}", num(*v))).collect::<Vec<_>>().join(", ")), &body))
}

fn cap1(t: &str) -> String {
    let mut c = t.chars();
    c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
}

/// The number outputs of the answer as bars.
fn out_scores(s: &Solved) -> Option<Figure> {
    let vals: Vec<(String, f64)> = s.a.values.iter().filter_map(|x| x.1.as_f64().filter(|v| v.is_finite()).map(|v| (x.0.clone(), v))).collect();
    if vals.len() < 2 || vals.len() > 12 {
        return None;
    }
    let labels: Vec<String> = vals.iter().map(|x| match x.0.as_str() {
        "parent_entropy" => "H(y)", "conditional_entropy" => "H(y|x)", "information_gain" => "gain",
        "split_info" => "split info", "gain_ratio" => "gain ratio", "mutual_information" => "I", "H_x" => "H(x)",
        "H_y" => "H(y)", "H_x_given_y" => "H(x|y)", "rand_index" => "Rand", "adjusted_rand_index" => "adj. Rand",
        "fowlkes_mallows" => "F-M", "mae" => "MAE", "mse" => "MSE", "rmse" => "RMSE", "r2" => "R²",
        "explained_variance" => "expl. var.", "median_ae" => "median AE", "max_error" => "max error", o => o,
    }.to_string()).collect();
    let v: Vec<f64> = vals.iter().map(|x| x.1).collect();
    let body = hbars(&labels, &v, &[], "", "value", "returned values");
    let title = match s.key {
        "ml_regression_metrics" => "Regression metrics of one set of predictions",
        "ml_rand_index" => "Pair-counting agreement scores",
        "ml_information_gain" => "Information gain of a split",
        _ => "Entropies and mutual information of two label lists",
    };
    Some(s.chart("scores", title, &format!("The values the model returns for the stated inputs: {}.",
                                           vals.iter().map(|(n, v)| format!("{n} {}", num(*v))).collect::<Vec<_>>().join(", ")), &body))
}

fn top_k(s: &Solved) -> Option<Figure> {
    let sm = s.mat("S")?;
    let y = s.vec("y_true")?;
    if sm.len() != y.len() || sm.len() > 20 || sm[0].len() > 12 {
        return None;
    }
    let rows: Vec<String> = y.iter().enumerate().map(|(i, c)| format!("s{} (y {})", i + 1, num(*c))).collect();
    let cols: Vec<String> = (0..sm[0].len()).map(|j| format!("class {j}")).collect();
    let body = hheat(&sm, &rows, &cols, &format!("top-{} accuracy {}", num(nin(s, "k")?), num(s.out("top_k_accuracy")?)));
    Some(s.chart("scores", "Class scores of each sample",
                 &format!("The stated scores S, one row per sample, labelled with its true class y = {}; returned top-{} accuracy {} \
                           ({} hits).", list(&y, 12), num(nin(s, "k")?), num(s.out("top_k_accuracy")?), num(s.out("n_hits")?)), &body))
}

/// Matrices side by side as one heat map (gaps between panels): inputs as read, outputs as returned.
fn panels(s: &Solved, names: &[(&str, bool)]) -> Option<Figure> {
    let mut mats = vec![];
    for (n, input) in names {
        let m = if *input { s.mat(n) } else { mat_out(s, n) };
        if let Some(m) = m {
            if m.iter().all(|r| r.len() == m[0].len()) {
                mats.push((n.to_string(), m, *input));
            }
        }
    }
    if mats.is_empty() {
        return None;
    }
    let rows = mats.iter().map(|m| m.1.len()).max()?;
    let width: usize = mats.iter().map(|m| m.1[0].len()).sum::<usize>() + mats.len() - 1;
    if rows > 20 || width > 24 {
        return None;
    }
    let mut big = vec![vec![f64::NAN; width]; rows];
    let mut cols = vec![String::new(); width];
    let mut at = 0;
    for (n, m, _) in &mats {
        for (i, r) in m.iter().enumerate() {
            for (j, v) in r.iter().enumerate() {
                big[i][at + j] = *v;
            }
        }
        for j in 0..m[0].len() {
            cols[at + j] = if j == 0 { n.clone() } else { String::new() };
        }
        at += m[0].len() + 1;
    }
    let rl: Vec<String> = (1..=rows).map(|i| format!("row {i}")).collect();
    let body = hheat(&big, &rl, &cols, &pl::clip(&mats.iter().map(|m| m.0.clone()).collect::<Vec<_>>().join(", "), 44));
    let rets: Vec<String> = s.a.values.iter().filter_map(|x| x.1.as_f64().map(|v| format!("{} = {}", x.0, num(v)))).collect();
    Some(s.chart("matrix", "The matrices",
                 &format!("The matrices as heat maps, side by side (each cell shaded by its value): {}.{}",
                          mats.iter().map(|m| format!("{} ({})", m.0, if m.2 { "input" } else { "returned" })).collect::<Vec<_>>().join("; "),
                          if rets.is_empty() { String::new() } else { format!(" Returned with them: {}.", rets.join(", ")) }), &body))
}

/// A one-hidden-layer network: nodes, every edge labelled with its weight (and, for backpropagation, its gradient).
fn network(s: &Solved) -> Option<Figure> {
    let (x, w1, b1, w2, b2) = (s.vec("x")?, s.mat("W1")?, s.vec("b1")?, s.mat("W2")?, s.vec("b2")?);
    let (ni, nh, no) = (x.len(), w1.len(), w2.len());
    if ni > 4 || nh > 4 || no > 3 || w1.iter().any(|r| r.len() != ni) || w2.iter().any(|r| r.len() != nh) || b1.len() != nh || b2.len() != no {
        return None;
    }
    let back = s.key == "ml_backpropagation";
    let (dw1, dw2) = if back { (mat_out(s, "dW1")?, mat_out(s, "dW2")?) } else { (vec![], vec![]) };
    // the hidden activations from the model's own forward pass
    let fw = if back {
        s.solve_other("ml_dense_forward_pass", &[("x", tl(&x)), ("W1", tm(&w1)), ("b1", tl(&b1)), ("activation", tx(&s.txt("activation")?)),
                                                ("W2", tm(&w2)), ("b2", tl(&b2)), ("output", tx("identity"))])?
    } else {
        s.a.clone()
    };
    let hid = k::list_of(&fw, "hidden")?;
    let outv = k::list_of(&fw, "output")?;
    let h = 110.0 + 80.0 * ni.max(nh).max(no) as f64;
    let mut c = Canvas::new(h);
    let col = [80.0, 260.0, 440.0];
    let ys = |n: usize, i: usize| 70.0 + (h - 110.0) * (i as f64 + 0.5) / n as f64;
    for j in 0..nh {
        for i in 0..ni {
            let (a, b) = ((col[0] + 22.0, ys(ni, i)), (col[1] - 22.0, ys(nh, j)));
            c.line(a, b);
            let t = 0.26;
            let lab = if back { format!("{} | {}", num(w1[j][i]), num(dw1[j][i])) } else { num(w1[j][i]) };
            c.value(a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t - 4.0, "middle", &lab);
        }
    }
    for o in 0..no {
        for j in 0..nh {
            let (a, b) = ((col[1] + 22.0, ys(nh, j)), (col[2] - 22.0, ys(no, o)));
            c.line(a, b);
            let t = 0.3;
            let lab = if back { format!("{} | {}", num(w2[o][j]), num(dw2[o][j])) } else { num(w2[o][j]) };
            c.value(a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t - 4.0, "middle", &lab);
        }
    }
    for i in 0..ni {
        c.ring((col[0], ys(ni, i)), 20.0, &num(x[i]));
    }
    for j in 0..nh {
        c.ring((col[1], ys(nh, j)), 20.0, &num(*hid.get(j)?));
    }
    for o in 0..no {
        c.ring((col[2], ys(no, o)), 20.0, &num(*outv.get(o)?));
    }
    c.name(col[0], 40.0, "middle", "input x");
    c.name(col[1], 40.0, "middle", &format!("hidden ({})", s.txt("activation")?));
    c.name(col[2], 40.0, "middle", "output");
    let foot = if back {
        format!("loss {} ({}); edges: weight | gradient", num(s.out("loss")?), s.txt("loss")?)
    } else {
        format!("{} output; edges: weights", s.txt("output")?)
    };
    c.bold(260.0, h - 14.0, "middle", &foot);
    let body = c.body().to_string();
    let title = if back { "Backpropagation: weight | gradient on every edge" } else { "Forward pass through a small network" };
    let caption = if back {
        format!("The {ni}-{nh}-{no} network of the visitor's example (x = ({}), target ({}), {} hidden units, {} loss). Each edge: its \
                 stated weight | the returned gradient of the loss with respect to it; hidden activations ({}) from the model's \
                 forward pass; returned loss {}, bias gradients db1 = ({}), db2 = ({}).", list(&x, 4), list(&s.vec("t")?, 3),
                s.txt("activation")?, s.txt("loss")?, list(&hid, 4), num(s.out("loss")?), list(&s.out_list("db1")?, 4),
                list(&s.out_list("db2")?, 3))
    } else {
        format!("The {ni}-{nh}-{no} network of the visitor's example (x = ({}), {} hidden units, {} output): every edge carries its \
                 stated weight; the nodes hold the returned hidden activations ({}) and output ({}).", list(&x, 4),
                s.txt("activation")?, s.txt("output")?, list(&hid, 4), list(&outv, 3))
    };
    Some(s.figure("network", title, &caption, &body, c.h))
}

/// Each feature before (top line) and after (bottom line) a scaling, each line on its own range.
fn before_after(s: &Solved) -> Option<Figure> {
    let xm = s.mat("X")?;
    let sc = mat_out(s, if s.key == "ml_batch_norm_forward" { "output" } else { "scaled" })?;
    let nf = xm[0].len();
    if sc.len() != xm.len() || nf > 4 || xm.len() > 60 || sc.iter().any(|r| r.len() != nf) {
        return None;
    }
    let ph = 120.0;
    let mut c = Canvas::new(60.0 + ph * nf as f64);
    let (xa, xb) = (130.0, 490.0);
    for j in 0..nf {
        let y0 = 70.0 + ph * j as f64;
        for (row, (vals, name)) in [(xm.iter().map(|r| r[j]).collect::<Vec<f64>>(), "before"),
                                    (sc.iter().map(|r| r[j]).collect::<Vec<f64>>(), "after")].iter().enumerate() {
            let y = y0 + 44.0 * row as f64;
            let (lo, hi) = pl::minmax(vals);
            let span = (hi - lo).max(1e-12);
            c.line((xa, y), (xb, y));
            c.name(xa - 10.0, y + 4.0, "end", &format!("feature {} {name}", j + 1));
            c.value(xa, y + 18.0, "middle", &num(lo));
            c.value(xb, y + 18.0, "middle", &num(hi));
            for (i, v) in vals.iter().enumerate() {
                c.raw(&format!("<circle class=\"{}\" cx=\"{:.1}\" cy=\"{y:.1}\" r=\"5\"/>", DOT[(i % 2) * 2],
                               xa + (v - lo) / span * (xb - xa)));
            }
        }
    }
    let rets: Vec<String> = ["mins", "maxs", "means", "stds", "medians", "iqr", "batch_mean", "batch_var"].iter()
        .filter_map(|n| s.out_list(n).map(|v| format!("{} ({})", n.replace('_', " "), list(&v, 4)))).collect();
    c.bold(16.0, 26.0, "start", &pl::clip(&rets.join(", "), 66));
    let body = c.body().to_string();
    let (kind, title) = match s.key {
        "ml_batch_norm_forward" => ("normalise", "Batch normalisation of each feature"),
        "ml_minmax_scaler" => ("before-after", "Min-max scaling of each feature"),
        "ml_robust_scaler" => ("before-after", "Robust scaling: median and IQR"),
        _ => ("before-after", "Standardising each feature"),
    };
    Some(s.figure(kind, title,
                  &format!("Each feature of X = {} before (upper line) and after (lower line) the scaling, the scaled values as returned, \
                            each line on its own range (its ends labelled); also returned: {}.", pl::text_mat(&xm), rets.join(", ")),
                  &body, c.h))
}

/// The loss of each sample, one run of the model on that sample alone.
fn per_sample(s: &Solved) -> Option<Figure> {
    let mut labels = vec![];
    let mut groups: Vec<(String, Vec<f64>)> = vec![];
    let (title, caption, header);
    match s.key {
        "ml_log_loss_multiclass" | "ml_multiclass_hinge_loss" => {
            let msym = if s.key == "ml_log_loss_multiclass" { "P" } else { "D" };
            let (y, m) = (s.vec("y_true")?, s.mat(msym)?);
            if y.len() != m.len() || y.len() > 20 {
                return None;
            }
            let outs: Vec<&str> = if s.key == "ml_log_loss_multiclass" { vec!["log_loss"] } else { vec!["crammer_singer", "weston_watkins"] };
            groups = outs.iter().map(|o| (o.replace('_', " "), vec![])).collect();
            for (i, (yy, r)) in y.iter().zip(&m).enumerate() {
                let a = s.with_input(&[("y_true", tl(&[*yy])), (msym, tm(&[r.clone()]))])?;
                for (g, o) in outs.iter().enumerate() {
                    groups[g].1.push(ans(&a, o)?);
                }
                labels.push(format!("sample {}", i + 1));
            }
            if s.key == "ml_log_loss_multiclass" {
                title = "Multiclass log loss of each sample";
                header = format!("mean {}, perplexity {}", num(s.out("log_loss")?), num(s.out("perplexity")?));
                caption = format!("Log loss of each sample of the stated probabilities with true classes {}, one run per sample (-ln of \
                                   the probability given to the true class); mean = returned log loss {}, perplexity {}.",
                                  list(&y, 12), num(s.out("log_loss")?), num(s.out("perplexity")?));
            } else {
                title = "Multiclass hinge loss of each sample";
                header = format!("means: Crammer-Singer {}, Weston-Watkins {}", num(s.out("crammer_singer")?), num(s.out("weston_watkins")?));
                caption = format!("Multiclass hinge loss of each sample of the decision values with true classes {}, one run per sample; \
                                   returned means: Crammer-Singer {}, Weston-Watkins {}.", list(&y, 12),
                                  num(s.out("crammer_singer")?), num(s.out("weston_watkins")?));
            }
        }
        _ => {
            let (t, p) = (s.vec("y_true")?, s.vec("y_pred")?);
            if t.len() != p.len() || t.len() > 30 {
                return None;
            }
            let (out, name) = if s.key == "ml_mape" { ("mape_percent", "percentage error") } else { ("msle", "squared log error") };
            let mut v = vec![];
            for (i, (a, b)) in t.iter().zip(&p).enumerate() {
                v.push(s.with_input(&[("y_true", tl(&[*a])), ("y_pred", tl(&[*b]))]).and_then(|r| ans(&r, out))?);
                labels.push(format!("{} → {}", num(*a), num(*b)));
                let _ = i;
            }
            groups.push((name.to_string(), v));
            if s.key == "ml_mape" {
                title = "Percentage error of each prediction";
                header = format!("MAPE {} %", num(s.out("mape_percent")?));
                caption = format!("Absolute percentage error of each sample (y_true {}, y_pred {}), each one run of the model on that \
                                   sample alone; their mean is the returned MAPE {} %.", list(&t, 12), list(&p, 12),
                                  num(s.out("mape_percent")?));
            } else {
                title = "Squared log error of each prediction";
                header = format!("MSLE {}, RMSLE {}", num(s.out("msle")?), num(s.out("rmsle")?));
                caption = format!("Squared log error of each sample (y_true {}, y_pred {}), one run per sample; their mean is the \
                                   returned MSLE {} (RMSLE {}).", list(&t, 12), list(&p, 12), num(s.out("msle")?), num(s.out("rmsle")?));
            }
        }
    }
    let body = if groups.len() == 1 { hbars(&labels, &groups[0].1, &[], "sample", &groups[0].0, &header) } else {
        group_bars(&labels, &groups, "sample", "loss", &header)
    };
    Some(s.chart("per-sample", title, &caption, &body))
}

/// A 2 x 2 confusion matrix (actual rows, predicted columns).
fn confusion(s: &Solved) -> Option<Figure> {
    if s.key == "ml_precision_recall_f1_averaged" {
        let (t, p) = (s.vec("y_true")?, s.vec("y_pred")?);
        let a = s.solve_other("ml_multiclass_confusion_matrix", &[("y_true", tl(&t)), ("y_pred", tl(&p))])?;
        let m = crate::figures::mat_of(&a, "matrix")?;
        let cl = k::list_of(&a, "classes")?;
        if m.len() > 12 {
            return None;
        }
        let rows: Vec<String> = cl.iter().map(|c| format!("true {}", num(*c))).collect();
        let cols: Vec<String> = cl.iter().map(|c| format!("pred {}", num(*c))).collect();
        let body = hheat(&m, &rows, &cols, "precision down a column, recall along a row");
        return Some(s.chart("confusion", "Confusion matrix behind the averages",
                            &format!("The confusion matrix of y_true = {} and y_pred = {} (from the model's confusion-matrix \
                                      capability): precision is read down each column, recall along each row.", list(&t, 16),
                                     list(&p, 16)), &body));
    }
    let g = |n: &str| s.get(n).or_else(|| nin(s, n));
    let (tp, fp, fn_, tn) = (g("TP")?, g("FP")?, g("FN")?, g("TN")?);
    let m = vec![vec![tp, fn_], vec![fp, tn]];
    let rows = vec!["actual +".to_string(), "actual -".to_string()];
    let cols = vec!["pred +".to_string(), "pred -".to_string()];
    let (title, header) = match s.key {
        "ml_matthews_correlation" => ("Confusion matrix and its Matthews correlation", format!("MCC = {}", num(s.out("mcc")?))),
        "ml_binary_confusion_matrix" => ("Binary confusion matrix", format!("accuracy {}", num(s.out("accuracy")?))),
        _ => ("The four counts as a confusion matrix", format!("F1 = {}", num(s.out("f1")?))),
    };
    let body = hheat(&m, &rows, &cols, &header);
    Some(s.chart("confusion", title,
                 &format!("The confusion matrix TP = {}, FN = {}, FP = {}, TN = {} (rows the actual class, columns the predicted); \
                           returned {header}.", num(tp), num(fn_), num(fp), num(tn)), &body))
}

fn rates(s: &Solved) -> Option<Figure> {
    let names = ["accuracy", "precision", "recall", "specificity", "f1", "npv", "balanced_accuracy", "jaccard"];
    let mut labels = vec![];
    let mut v = vec![];
    for n in names {
        if let Some(x) = s.out(n) {
            labels.push(match n { "accuracy" => "acc.", "precision" => "prec.", "specificity" => "spec.", "f1" => "F1",
                                  "npv" => "NPV", "balanced_accuracy" => "bal.acc", "jaccard" => "Jacc.", o => o }.to_string());
            v.push(x);
        }
    }
    let body = hbars(&labels, &v, &[], "", "rate", &format!("TP {}, FP {}, FN {}, TN {}", num(nin(s, "TP")?), num(nin(s, "FP")?),
                                                              num(nin(s, "FN")?), num(nin(s, "TN")?)));
    Some(s.chart("rates", "Rates from a confusion matrix",
                 &format!("The rates the model returns for TP = {}, FP = {}, FN = {}, TN = {}: {}.", num(nin(s, "TP")?),
                          num(nin(s, "FP")?), num(nin(s, "FN")?), num(nin(s, "TN")?),
                          labels.iter().zip(&v).map(|(l, x)| format!("{l} {}", num(*x))).collect::<Vec<_>>().join(", ")), &body))
}

fn averages(s: &Solved) -> Option<Figure> {
    let mine = s.txt("average")?;
    let mut labels = vec![];
    let mut g: Vec<(String, Vec<f64>)> = vec![("precision".into(), vec![]), ("recall".into(), vec![]), ("F1".into(), vec![])];
    for c in ["macro", "weighted", "micro"] {
        let a = s.with_input(&[("average", tx(c))])?;
        for (i, o) in ["precision", "recall", "f1"].iter().enumerate() {
            g[i].1.push(ans(&a, o)?);
        }
        labels.push(c.to_string());
    }
    let body = group_bars(&labels, &g, "averaging", "score", &format!("{mine}: F1 = {}", num(s.out("f1")?)));
    Some(s.chart("averages", "Precision, recall and F1 by averaging",
                 &format!("Precision, recall and F1 of the stated labels under each averaging, one run each: {}.",
                          labels.iter().enumerate().map(|(i, l)| format!("{l}: P {}, R {}, F1 {}", num(g[0].1[i]), num(g[1].1[i]),
                                                                         num(g[2].1[i]))).collect::<Vec<_>>().join("; ")), &body))
}

fn multi_confusion(s: &Solved, norm: bool) -> Option<Figure> {
    let m = mat_out(s, "matrix")?;
    let cl = s.out_list("classes")?;
    if m.len() > 12 || m.len() != cl.len() {
        return None;
    }
    let shown: Vec<Vec<f64>> = if norm {
        m.iter().map(|r| {
            let t: f64 = r.iter().sum();
            r.iter().map(|v| if t > 0.0 { v / t } else { 0.0 }).collect()
        }).collect()
    } else {
        m.clone()
    };
    let rows: Vec<String> = cl.iter().map(|c| format!("true {}", num(*c))).collect();
    let cols: Vec<String> = cl.iter().map(|c| format!("pred {}", num(*c))).collect();
    let (acc, nc) = (s.out("accuracy")?, s.out("n_correct")?);
    let body = hheat(&shown, &rows, &cols, &format!("accuracy {}, {} correct", num(acc), num(nc)));
    if norm {
        Some(s.chart("normalised", "Confusion matrix as fractions of each true class",
                     &format!("The confusion matrix the model returns, each row divided by its total (so the diagonal is each class's \
                               recall); accuracy {}, {} correct.", num(acc), num(nc)), &body))
    } else {
        Some(s.chart("heatmap", "Confusion matrix",
                     &format!("The confusion matrix the model returns for true labels {} and predictions {}: rows the true classes, \
                               columns the predicted ones; {} on the diagonal, accuracy {}.", list(&s.vec("y_true")?, 16),
                              list(&s.vec("y_pred")?, 16), num(nc), num(acc)), &body))
    }
}

fn counts(s: &Solved) -> Option<Figure> {
    match s.key {
        "ml_dense_parameter_count" => {
            let (sz, pl_) = (s.vec("sizes")?, s.out_list("per_layer")?);
            let labels: Vec<String> = sz.windows(2).map(|w| format!("{} → {}", num(w[0]), num(w[1]))).collect();
            if labels.len() != pl_.len() || labels.len() > 16 {
                return None;
            }
            let body = hbars(&labels, &pl_, &[], "layer", "parameters", &format!("total {}", num(s.out("total")?)));
            Some(s.chart("counts", "Parameters of each dense layer",
                         &format!("Trainable parameters of the dense network with layer sizes {} (biases: {}): returned per layer {}, \
                                   total {}.", list(&sz, 10), if nin(s, "bias")? > 0.0 { "yes" } else { "no" }, list(&pl_, 10),
                                  num(s.out("total")?)), &body))
        }
        "ml_one_hot_encoding" => {
            let cats = words_of(s, "categories")?;
            let cn = s.out_list("counts")?;
            if cats.len() != cn.len() || cats.len() > 30 {
                return None;
            }
            let labels: Vec<String> = cats.iter().enumerate().map(|(i, c)| format!("{c} ({i})")).collect();
            let body = hbars(&labels, &cn, &[], "category (code)", "count", &format!("{} categories", cats.len()));
            Some(s.chart("counts", "Categories, codes and counts",
                         &format!("The categories in sorted order with their returned codes and counts ({}); each becomes one column of \
                                   the one-hot matrix.", cats.iter().zip(&cn).map(|(c, n)| format!("{c}: {}", num(*n))).collect::<Vec<_>>().join(", ")),
                         &body))
        }
        "ml_rnn_parameter_count" => {
            let (mut uni, mut bi) = (vec![], vec![]);
            let cells = ["rnn", "gru", "lstm"];
            for c in cells {
                uni.push(s.with_input(&[("cell", tx(c)), ("bidirectional", it(0))]).and_then(|a| ans(&a, "total"))?);
                bi.push(s.with_input(&[("cell", tx(c)), ("bidirectional", it(1))]).and_then(|a| ans(&a, "total"))?);
            }
            let labels: Vec<String> = cells.iter().map(|c| c.to_string()).collect();
            let body = group_bars(&labels, &[("one direction".into(), uni.clone()), ("bidirectional".into(), bi.clone())], "cell",
                                  "parameters", &format!("visitor's {}: {}", s.txt("cell")?, num(s.out("total")?)));
            Some(s.chart("counts", "Recurrent cells: parameters by gate count",
                         &format!("Parameters of {} recurrent layer(s) with {} inputs and {} hidden units for each cell type, one run \
                                   each: {}.", num(nin(s, "num_layers")?), num(nin(s, "input_size")?), num(nin(s, "hidden_size")?),
                                  cells.iter().enumerate().map(|(i, c)| format!("{c} {} ({} bidirectional)", num(uni[i]), num(bi[i])))
                                      .collect::<Vec<_>>().join(", ")), &body))
        }
        _ => {
            let (d0, h) = (nin(s, "d_model")?, nin(s, "n_heads")?);
            let mut ds = vec![256.0, 512.0, 768.0, 1024.0];
            if !ds.contains(&d0) {
                ds.push(d0);
                ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
            }
            let (mut at, mut ff, mut labels) = (vec![], vec![], vec![]);
            for d in ds {
                if (d / h).fract() != 0.0 {
                    continue;
                }
                let ffn = if d == d0 { nin(s, "d_ff")? } else { 4.0 * d };
                if let Some(a) = s.with_input(&[("d_model", it(d as i64)), ("d_ff", it(ffn as i64)), ("n_layers", it(1))]) {
                    at.push(ans(&a, "attention")?);
                    ff.push(ans(&a, "feed_forward")?);
                    labels.push(format!("{}", num(d)));
                }
            }
            if labels.len() < 2 {
                return None;
            }
            let body = group_bars(&labels, &[("attention".into(), at.clone()), ("feed-forward".into(), ff.clone())], "d_model",
                                  "parameters per layer", &format!("visitor's total {}", num(s.out("total")?)));
            Some(s.chart("counts", "Transformer layer: attention against feed-forward",
                         &format!("Parameters of one transformer layer by part for several d_model (d_ff = 4·d_model except the \
                                   visitor's own d_ff, {} heads), one run each: {}. Visitor's example ({} layers): total {}.", num(h),
                                  labels.iter().enumerate().map(|(i, l)| format!("{l}: attention {}, feed-forward {}", num(at[i]), num(ff[i])))
                                      .collect::<Vec<_>>().join("; "), num(nin(s, "n_layers")?), num(s.out("total")?)), &body))
        }
    }
}

fn nb_data(s: &Solved) -> Option<Figure> {
    let xm = s.mat("X")?;
    let y = s.vec("y")?;
    let q = s.vec("x")?;
    if xm.len() != y.len() || xm.len() > 14 || q.len() > 6 || xm.iter().any(|r| r.len() != q.len()) {
        return None;
    }
    let mut head = vec!["row".to_string()];
    head.extend((1..=q.len()).map(|j| format!("x{j}")));
    head.push("class".into());
    let mut rows: Vec<Vec<String>> = xm.iter().zip(&y).enumerate().map(|(i, (r, c))| {
        let mut v = vec![format!("{}", i + 1)];
        v.extend(r.iter().map(|x| num(*x)));
        v.push(num(*c));
        v
    }).collect();
    let mut qr = vec!["query".to_string()];
    qr.extend(q.iter().map(|x| num(*x)));
    qr.push(format!("→ {}", num(s.out("prediction")?)));
    rows.push(qr);
    let cl = s.out_list("classes")?;
    let post = s.out_list("posteriors")?;
    let pw = cl.iter().zip(&post).map(|(c, p)| format!("class {} {}", num(*c), num(*p))).collect::<Vec<_>>().join(", ");
    let (body, h) = htable(&head, &rows, Some(rows.len() - 1), &format!("posteriors: {pw}"));
    let bern = s.key == "ml_bernoulli_naive_bayes";
    Some(s.figure("data", if bern { "Bernoulli naive Bayes: training rows, query and posterior" } else {
                      "Multinomial naive Bayes: counts, query and posterior" },
                  &format!("The stated training rows (labelled with their class) and the query x = ({}) outlined; returned posteriors \
                            {pw}, prediction class {} (smoothing α = {}).", list(&q, 6), num(s.out("prediction")?),
                           num(s.inp("alpha")?)), &body, h))
}

fn probabilities(s: &Solved) -> Option<Figure> {
    let z = s.vec("z")?;
    if z.len() > 12 {
        return None;
    }
    let labels: Vec<String> = z.iter().map(|v| format!("z = {}", num(*v))).collect();
    if s.key == "ml_softmax" {
        let p = s.out_list("probabilities")?;
        let body = hbars(&labels, &p, &[], "logit", "probability", &format!("log-sum-exp {}", num(s.out("log_sum_exp")?)));
        return Some(s.chart("probabilities", "Softmax probabilities of the logits",
                            &format!("Softmax of the logits z = ({}): returned probabilities {} (log-sum-exp {}); each unit of z \
                                      multiplies the probability by e.", list(&z, 12), list(&p, 12), num(s.out("log_sum_exp")?)), &body));
    }
    let t0 = s.inp("T")?;
    let mut ts = vec![0.5, 1.0, 2.0, 5.0];
    if !ts.contains(&t0) {
        ts.push(t0);
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }
    let mut groups = vec![];
    let mut ent = vec![];
    for t in ts {
        let a = s.with(&[("T", t)])?;
        groups.push((format!("T = {}", num(t)), k::list_of(&a, "probabilities")?));
        ent.push(format!("T = {}: {}", num(t), num(ans(&a, "entropy")?)));
    }
    let body = group_bars(&labels, &groups, "logit", "probability", &format!("visitor's T = {}: entropy {}", num(t0), num(s.out("entropy")?)));
    Some(s.chart("probabilities", "Softmax at several temperatures",
                 &format!("Softmax of z = ({}) at several temperatures, one run each; returned entropies {}. Low temperature sharpens \
                           the distribution, high temperature flattens it.", list(&z, 12), ent.join(", ")), &body))
}

/// Relevance of each query's ranked results as a grid, each row labelled with its returned score.
fn ranks(s: &Solved) -> Option<Figure> {
    let r = s.mat("R")?;
    let ap = s.out_list(if s.key == "ml_mean_average_precision" { "average_precision" } else { "reciprocal_ranks" })?;
    let nr = r.len();
    let nc = r.iter().map(|x| x.len()).max()?;
    if nr > 14 || nc > 20 || ap.len() != nr {
        return None;
    }
    let cw = (300.0 / nc as f64).min(34.0);
    let rh = 26.0;
    let mut c = Canvas::new(80.0 + rh * nr as f64 + 20.0);
    let x0 = 110.0;
    for (j, _) in (0..nc).enumerate() {
        c.name(x0 + cw * (j as f64 + 0.5), 58.0, "middle", &format!("{}", j + 1));
    }
    for (i, row) in r.iter().enumerate() {
        let y = 66.0 + rh * i as f64;
        c.name(x0 - 8.0, y + rh * 0.65, "end", &format!("query {}", i + 1));
        for (j, v) in row.iter().enumerate() {
            c.raw(&format!("<rect class=\"{}\" x=\"{:.1}\" y=\"{y:.1}\" width=\"{:.1}\" height=\"{:.1}\"/>",
                           if *v > 0.0 { "axp-b" } else { "axp-b4" }, x0 + cw * j as f64, cw - 2.0, rh - 4.0));
        }
        c.value(x0 + cw * nc as f64 + 12.0, y + rh * 0.65, "start", &num(ap[i]));
    }
    let mean = s.out(if s.key == "ml_mean_average_precision" { "map" } else { "mrr" })?;
    let (title, what) = if s.key == "ml_mean_average_precision" { ("Ranked results and average precision", "average precision") } else {
        ("Ranked results and reciprocal ranks", "reciprocal rank")
    };
    c.bold(16.0, 26.0, "start", &format!("rank across, relevant dark; right: {what}; mean {}", num(mean)));
    let body = c.body().to_string();
    Some(s.figure("ranks", title,
                  &format!("The stated relevance of each query's ranked results (1 = relevant, dark) and the returned {what} of each \
                            query ({}); mean {}.", list(&ap, 12), num(mean)), &body, c.h))
}

fn info_diagram(s: &Solved) -> Option<Figure> {
    let (hx, hy, hxy) = (s.out("H_X")?, s.out("H_Y")?, s.out("H_XY")?);
    let (mi, hxgy, hygx) = (s.out("mutual_information")?, s.out("H_X_given_Y")?, s.out("H_Y_given_X")?);
    if !(hxy > 0.0) {
        return None;
    }
    let mut c = Canvas::new(250.0);
    let (x0, w) = (60.0, 400.0);
    let px = |v: f64| x0 + v / hxy * w;
    c.raw(&format!("<rect class=\"axp-b\" x=\"{:.1}\" y=\"70\" width=\"{:.1}\" height=\"34\"/>", px(0.0), px(hx) - px(0.0)));
    c.raw(&format!("<rect class=\"axp-b3\" x=\"{:.1}\" y=\"120\" width=\"{:.1}\" height=\"34\"/>", px(hxgy), px(hxy) - px(hxgy)));
    c.dashed((px(hxgy), 60.0), (px(hxgy), 165.0));
    c.dashed((px(hx), 60.0), (px(hx), 165.0));
    c.value(px(hx / 2.0), 92.0, "middle", &format!("H(X) {}", num(hx)));
    c.value(px(hxgy + hy / 2.0), 142.0, "middle", &format!("H(Y) {}", num(hy)));
    c.line((px(0.0), 190.0), (px(hxy), 190.0));
    c.name(px(hxgy / 2.0), 182.0, "middle", &format!("H(X|Y) {}", num(hxgy)));
    c.name(px(hxgy + mi / 2.0), 210.0, "middle", &format!("I {}", num(mi)));
    c.name(px(hx + hygx / 2.0), 182.0, "middle", &format!("H(Y|X) {}", num(hygx)));
    c.bold(260.0, 236.0, "middle", &format!("H(X, Y) = {} {}", num(hxy), s.txt("unit")?));
    c.bold(16.0, 30.0, "start", "Information diagram");
    let body = c.body().to_string();
    Some(s.figure("diagram", "Joint, conditional and mutual information",
                  &format!("Information diagram of the stated joint distribution (rows X, columns Y): H(X) = {} and H(Y) = {} drawn as \
                            overlapping bars; the overlap is the mutual information {}, the rest the conditional entropies H(X|Y) = \
                            {} and H(Y|X) = {}; the line spans H(X, Y) = {} {}, all returned values.", num(hx), num(hy), num(mi),
                           num(hxgy), num(hygx), num(hxy), s.txt("unit")?), &body, c.h))
}

fn joint(s: &Solved) -> Option<Figure> {
    let m = s.mat("P")?;
    if m.len() > 12 || m[0].len() > 12 {
        return None;
    }
    let rows: Vec<String> = (0..m.len()).map(|i| format!("X = {i}")).collect();
    let cols: Vec<String> = (0..m[0].len()).map(|j| format!("Y = {j}")).collect();
    let body = hheat(&m, &rows, &cols, &format!("mutual information {}", num(s.out("mutual_information")?)));
    Some(s.chart("joint", "Joint distribution of X and Y",
                 &format!("The stated joint distribution P(X, Y) = {}, rows the values of X and columns the values of Y.", pl::text_mat(&m)),
                 &body))
}

fn distribution(s: &Solved) -> Option<Figure> {
    let p = s.vec("p")?;
    if p.len() > 30 {
        return None;
    }
    let labels: Vec<String> = (1..=p.len()).map(|i| format!("{i}")).collect();
    let (h, u) = (s.out("entropy")?, s.txt("unit")?);
    let body = hbars(&labels, &p, &[], "outcome", "probability", &format!("entropy {} {u}, perplexity {}", num(h), num(s.out("perplexity")?)));
    Some(s.chart("distribution", "A distribution and its entropy",
                 &format!("The stated distribution p = ({}): returned entropy {} {u} (maximum {} for {} outcomes, normalised {}), \
                           perplexity {}: as uncertain as {} equally likely outcomes.", list(&p, 12), num(h), num(s.out("max_entropy")?),
                          p.len(), num(s.out("normalized_entropy")?), num(s.out("perplexity")?), num(s.out("perplexity")?)), &body))
}

fn distributions(s: &Solved) -> Option<Figure> {
    let (p, q) = (s.vec("p")?, s.vec("q")?);
    if p.len() != q.len() || p.len() > 20 {
        return None;
    }
    let labels: Vec<String> = (1..=p.len()).map(|i| format!("{i}")).collect();
    let u = s.txt("unit")?;
    let body = group_bars(&labels, &[("p".into(), p.clone()), ("q".into(), q.clone())], "outcome", "probability",
                          &format!("KL(p‖q) = {}, KL(q‖p) = {}", num(s.out("kl_pq")?), num(s.out("kl_qp")?)));
    Some(s.chart("distributions", "Two distributions and their divergences",
                 &format!("The stated distributions p = ({}) and q = ({}): returned KL(p‖q) = {}, KL(q‖p) = {} (not symmetric), \
                           Jensen-Shannon {}, cross-entropy {} = H(p) {} + KL(p‖q), in {u}.", list(&p, 10), list(&q, 10),
                          num(s.out("kl_pq")?), num(s.out("kl_qp")?), num(s.out("js_divergence")?), num(s.out("cross_entropy")?),
                          num(s.out("entropy_p")?)), &body))
}

fn logits(s: &Solved) -> Option<Figure> {
    let z = s.vec("z")?;
    let y = nin(s, "y")?;
    if z.len() > 12 {
        return None;
    }
    let labels: Vec<String> = (0..z.len()).map(|i| if i as f64 == y { format!("{i} (true)") } else { format!("{i}") }).collect();
    if s.key == "ml_cross_entropy_logits" {
        let p = k::list_of(&s.solve_other("ml_softmax", &[("z", tl(&z))])?, "probabilities")?;
        let g = s.out_list("gradient")?;
        let body = group_bars(&labels, &[("probability".into(), p.clone()), ("gradient".into(), g.clone())], "class", "value",
                              &format!("loss {}", num(s.out("loss")?)));
        return Some(s.chart("logits", "Cross-entropy: probabilities and gradient per class",
                            &format!("Logits z = ({}) with true class {}: the softmax probabilities (from the model's softmax: {}) and \
                                      the returned gradient of the loss, probability minus one-hot ({}); returned loss {}.", list(&z, 12),
                                     num(y), list(&p, 12), list(&g, 12), num(s.out("loss")?)), &body));
    }
    let ce = s.solve_other("ml_cross_entropy_logits", &[("z", tl(&z)), ("y", it(y as i64))])?;
    let (g0, l0) = (k::list_of(&ce, "gradient")?, ans(&ce, "loss")?);
    let g = s.out_list("gradient")?;
    let e = s.inp("epsilon")?;
    let body = group_bars(&labels, &[("plain cross-entropy".into(), g0.clone()), (format!("smoothed, ε = {}", num(e)), g.clone())],
                          "class", "gradient", &format!("loss {} (plain {})", num(s.out("loss")?), num(l0)));
    Some(s.chart("gradients", "Label smoothing softens the gradient",
                 &format!("Gradient of the loss for logits z = ({}), true class {}: plain cross-entropy (the model's cross-entropy, loss \
                           {}) against label smoothing with ε = {} (returned loss {}); smoothing shrinks the push on the true class \
                           and on the others.", list(&z, 12), num(y), num(l0), num(e), num(s.out("loss")?)), &body))
}

fn heights(s: &Solved) -> Option<Figure> {
    let mine = s.txt("method")?;
    let ms = ["single", "complete", "average", "ward"];
    let mut groups = vec![];
    let mut words = vec![];
    for m in ms {
        let h = s.with_input(&[("method", tx(m))]).and_then(|a| k::list_of(&a, "heights"))?;
        words.push(format!("{m}: {}", list(&h, 8)));
        groups.push((if m == mine { format!("{m} (yours)") } else { m.to_string() }, h));
    }
    let n = groups[0].1.len();
    if n > 12 || groups.iter().any(|g| g.1.len() != n) {
        return None;
    }
    let labels: Vec<String> = (1..=n).map(|i| format!("merge {i}")).collect();
    let body = group_bars(&labels, &groups, "merge", "height", &format!("{mine}: final height {}", num(s.out("final_height")?)));
    Some(s.chart("heights", "Merge heights under four linkages",
                 &format!("The merge heights the model returns for the {} stated points under four linkages, one run each: {}.",
                          n + 1, words.join("; ")), &body))
}

fn impurity(s: &Solved) -> Option<Figure> {
    let (y, g) = (s.vec("y")?, s.vec("g")?);
    if y.len() != g.len() {
        return None;
    }
    let (cy, cg) = (classes(&y), classes(&g));
    if cg.len() > 8 || cy.len() > 8 {
        return None;
    }
    let mut labels = vec!["parent".to_string()];
    let mut vals = vec![s.out("gini_parent")?];
    let mut words = vec![];
    for c in &cg {
        let cnt: Vec<f64> = cy.iter().map(|k_| y.iter().zip(&g).filter(|(a, b)| *a == k_ && *b == c).count() as f64).collect();
        let gi = s.solve_other("ml_impurity_from_counts", &[("counts", tl(&cnt))]).and_then(|a| ans(&a, "gini"))?;
        labels.push(format!("child {}", num(*c)));
        vals.push(gi);
        words.push(format!("child {}: [{}]", num(*c), list(&cnt, 8)));
    }
    labels.push("children (weighted)".into());
    vals.push(s.out("gini_children")?);
    let body = hbars(&labels, &vals, &[0, labels.len() - 1], "", "Gini impurity", &format!("decrease {}", num(s.out("gini_decrease")?)));
    Some(s.chart("impurity", "Gini impurity before and after a split",
                 &format!("Gini impurity of the parent y = {} and of the children given by g = {}: class counts per child tallied from \
                           the stated lists ({}), each child's Gini from the model's impurity capability; returned: parent {}, weighted \
                           children {}, decrease {}.", list(&y, 16), list(&g, 16), words.join("; "), num(s.out("gini_parent")?),
                          num(s.out("gini_children")?), num(s.out("gini_decrease")?)), &body))
}

fn labels(s: &Solved) -> Option<Figure> {
    let (t, p) = (s.mat("Y_true")?, s.mat("Y_pred")?);
    if t.len() != p.len() || t.len() > 16 || t[0].len() > 10 || t.iter().zip(&p).any(|(a, b)| a.len() != b.len()) {
        return None;
    }
    let nl = t[0].len();
    let mut big = vec![];
    for (a, b) in t.iter().zip(&p) {
        let mut row = a.clone();
        row.push(f64::NAN);
        row.extend(b.iter().cloned());
        row.push(f64::NAN);
        row.extend(a.iter().zip(b).map(|(u, v)| if u != v { 1.0 } else { 0.0 }));
        big.push(row);
    }
    let mut cols = vec![String::new(); 3 * nl + 2];
    cols[0] = "true".into();
    cols[nl + 1] = "pred".into();
    cols[2 * nl + 2] = "diff".into();
    let rows: Vec<String> = (1..=t.len()).map(|i| format!("sample {i}")).collect();
    let body = hheat(&big, &rows, &cols, &format!("Hamming loss {}", num(s.out("hamming_loss")?)));
    Some(s.chart("labels", "True and predicted label sets",
                 &format!("Stated true and predicted label sets of each sample (Y_true = {}, Y_pred = {}) and their mismatches; returned \
                           Hamming loss {} ({} mismatched entries), subset accuracy {}.", pl::text_mat(&t), pl::text_mat(&p),
                          num(s.out("hamming_loss")?), num(s.out("n_mismatches")?), num(s.out("subset_accuracy")?)), &body))
}

fn recalls(s: &Solved) -> Option<Figure> {
    let (t, p) = (s.vec("y_true")?, s.vec("y_pred")?);
    let a = s.solve_other("ml_classification_report", &[("y_true", tl(&t)), ("y_pred", tl(&p))])?;
    let (cl, rc) = (k::list_of(&a, "classes")?, k::list_of(&a, "recall")?);
    if cl.len() > 16 || cl.len() != rc.len() {
        return None;
    }
    let ba = s.out("balanced_accuracy")?;
    let mut labels: Vec<String> = cl.iter().map(|c| format!("class {}", num(*c))).collect();
    let mut v = rc.clone();
    labels.push("mean".into());
    v.push(ba);
    let body = hbars(&labels, &v, &[v.len() - 1], "", "recall", &format!("balanced accuracy {} (adjusted {})", num(ba), num(s.out("adjusted")?)));
    Some(s.chart("recalls", "Balanced accuracy: the mean of per-class recalls",
                 &format!("For y_true = {} and y_pred = {}: the recall of each class (from the model's classification report: {}); their \
                           mean is the returned balanced accuracy {} (adjusted {}).", list(&t, 16), list(&p, 16), list(&rc, 10),
                          num(ba), num(s.out("adjusted")?)), &body))
}

fn report(s: &Solved) -> Option<Figure> {
    let cl = s.out_list("classes")?;
    let (pr, rc, f1) = (s.out_list("precision")?, s.out_list("recall")?, s.out_list("f1")?);
    if cl.len() > 12 {
        return None;
    }
    let labels: Vec<String> = cl.iter().map(|c| format!("class {}", num(*c))).collect();
    let body = group_bars(&labels, &[("precision".into(), pr), ("recall".into(), rc), ("F1".into(), f1)], "class", "score",
                          &format!("macro F1 {}", num(s.out("macro_f1")?)));
    Some(s.chart("report", "Per-class precision, recall and F1",
                 &format!("Per-class precision, recall and F1 the model returns for y_true = {} and y_pred = {} (support {}); macro F1 {}.",
                          list(&s.vec("y_true")?, 16), list(&s.vec("y_pred")?, 16), list(&s.out_list("support")?, 10),
                          num(s.out("macro_f1")?)), &body))
}

/// Train and test sizes for several test fractions, one run each.
pub fn sizes(s: &Solved) -> Option<Figure> {
    let (n, f0) = (nin(s, "n")?, s.inp("test_size")?);
    let mut fs = vec![0.1, 0.2, 0.25, 0.3, 0.5];
    if !fs.iter().any(|f| (f - f0).abs() < 1e-12) {
        fs.push(f0);
        fs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }
    let (mut tr, mut te, mut labels) = (vec![], vec![], vec![]);
    for f in fs {
        if let Some(a) = s.with(&[("test_size", f)]) {
            tr.push(ans(&a, "n_train")?);
            te.push(ans(&a, "n_test")?);
            labels.push(if (f - f0).abs() < 1e-12 { format!("{} (yours)", num(f)) } else { num(f) });
        }
    }
    if labels.len() < 2 {
        return None;
    }
    let words = labels.iter().enumerate().map(|(i, l)| format!("{l}: {} / {}", num(tr[i]), num(te[i]))).collect::<Vec<_>>().join("; ");
    let body = group_bars(&labels, &[("train".into(), tr), ("test".into(), te)], "test fraction", "samples", &format!("n = {}", num(n)));
    Some(s.chart("sizes", "Train and test sizes for several fractions",
                 &format!("Train and test sizes of {} samples for several test fractions, one run each (train / test): {words}.", num(n)),
                 &body))
}

fn weights(s: &Solved) -> Option<Figure> {
    let y = s.vec("y")?;
    let (cl, w) = (s.out_list("classes")?, s.out_list("weights")?);
    if cl.len() != w.len() || cl.len() > 16 {
        return None;
    }
    let cnt: Vec<f64> = cl.iter().map(|c| y.iter().filter(|v| *v == c).count() as f64).collect();
    let labels: Vec<String> = cl.iter().map(|c| format!("class {}", num(*c))).collect();
    let body = group_bars(&labels, &[("count".into(), cnt.clone()), ("weight × count".into(), w.iter().zip(&cnt).map(|(a, b)| a * b).collect())],
                          "class", "samples", &format!("weights {}", list(&w, 8)));
    let _ = (BAR, fl);
    Some(s.chart("weights", "Balanced class weights against class counts",
                 &format!("Class counts tallied from y = {} ({}) and the returned balanced weights {}: weight × count is the same for \
                           every class.", list(&y, 16), cl.iter().zip(&cnt).map(|(c, n)| format!("class {}: {}", num(*c), num(*n)))
                          .collect::<Vec<_>>().join(", "), list(&w, 8)), &body))
}
