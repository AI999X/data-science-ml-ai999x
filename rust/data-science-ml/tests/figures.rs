//! Every Data Science and ML capability's own live figures (src/figures*.rs): its worked example gives
//! each published figure kind, a changed input changes the drawing, and no drawing holds a non-finite number, text
//! outside its canvas or wording the public pages do not use.

use ax_engine::figures::{figures, Figure};
use ax_engine::Input;
use data_science_ml_ai999x::pack;

/// The published figure kinds of every capability that the model draws itself (publish/website/models/
/// data-science-ml-ai999x/figures, except the generic "curve" and the kinds the generic figures already give), as
/// tools/figure_parity.py lists them.
const KINDS: &[(&str, &[&str])] = &[
    ("ml_adam_step", &["path"]),
    ("ml_adamw_step", &["path"]),
    ("ml_adjusted_mutual_information", &["contingency", "scores"]),
    ("ml_adjusted_r2", &["parity", "penalty"]),
    ("ml_average_precision", &["pr"]),
    ("ml_backpropagation", &["matrix", "network"]),
    ("ml_balanced_accuracy", &["recalls"]),
    ("ml_balanced_class_weights", &["weights"]),
    ("ml_batch_norm_forward", &["matrix", "normalise"]),
    ("ml_bce_with_logits", &["loss"]),
    ("ml_bernoulli_naive_bayes", &["data"]),
    ("ml_bias_variance_decomposition", &["spread"]),
    ("ml_binary_confusion_matrix", &["confusion"]),
    ("ml_binary_metrics_from_counts", &["confusion", "rates"]),
    ("ml_brier_score", &["loss"]),
    ("ml_calinski_harabasz_index", &["clusters"]),
    ("ml_classification_report", &["report"]),
    ("ml_cohen_kappa", &["contingency", "scores"]),
    ("ml_cosine_lr_schedule", &["schedule"]),
    ("ml_cross_entropy_logits", &["logits"]),
    ("ml_davies_bouldin_index", &["clusters"]),
    ("ml_decision_stump_classification", &["split"]),
    ("ml_decision_stump_regression", &["split"]),
    ("ml_dense_forward_pass", &["network"]),
    ("ml_dense_parameter_count", &["counts"]),
    ("ml_distance_metrics", &["geometry"]),
    ("ml_entropy", &["distribution"]),
    ("ml_expected_calibration_error", &["reliability"]),
    ("ml_focal_loss", &["loss"]),
    ("ml_gaussian_naive_bayes", &["features"]),
    ("ml_gini_split", &["impurity"]),
    ("ml_gradient_descent_step_linear", &["fit"]),
    ("ml_hamming_loss", &["labels"]),
    ("ml_hierarchical_clustering", &["heights"]),
    ("ml_hinge_loss", &["loss"]),
    ("ml_homogeneity_completeness_v_measure", &["contingency"]),
    ("ml_huber_loss", &["loss"]),
    ("ml_information_gain", &["contingency", "scores"]),
    ("ml_joint_conditional_entropy", &["diagram", "joint"]),
    ("ml_kfold_cv_linear_regression", &["data", "folds"]),
    ("ml_kfold_sizes", &["folds"]),
    ("ml_kl_divergence_discrete", &["distributions"]),
    ("ml_kmeans", &["clusters", "regions"]),
    ("ml_kmeans_step", &["step"]),
    ("ml_knn_classify", &["neighbours", "regions"]),
    ("ml_knn_regress", &["fit"]),
    ("ml_label_smoothing_cross_entropy", &["gradients"]),
    ("ml_lasso_regression", &["path"]),
    ("ml_linear_svm_decision", &["boundary"]),
    ("ml_log_loss_binary", &["loss"]),
    ("ml_log_loss_multiclass", &["matrix", "per-sample"]),
    ("ml_logistic_loss_gradient", &["landscape"]),
    ("ml_logistic_regression_fit", &["fit"]),
    ("ml_logistic_regression_predict", &["probability"]),
    ("ml_loocv_linear_regression", &["comparison"]),
    ("ml_mape", &["per-sample"]),
    ("ml_matthews_correlation", &["confusion"]),
    ("ml_mean_average_precision", &["ranks"]),
    ("ml_mean_reciprocal_rank", &["ranks"]),
    ("ml_mean_squared_log_error", &["per-sample"]),
    ("ml_minmax_scaler", &["before-after", "matrix"]),
    ("ml_mse_gradient", &["landscape"]),
    ("ml_multiclass_confusion_matrix", &["heatmap", "normalised"]),
    ("ml_multiclass_hinge_loss", &["matrix", "per-sample"]),
    ("ml_multinomial_naive_bayes", &["data"]),
    ("ml_mutual_information_labels", &["contingency", "scores"]),
    ("ml_ndcg_at_k", &["over-k"]),
    ("ml_normalized_mutual_information", &["contingency", "scores"]),
    ("ml_one_hot_encoding", &["counts"]),
    ("ml_pca", &["pca"]),
    ("ml_perceptron_epoch", &["boundary"]),
    ("ml_precision_recall_at_k", &["over-k"]),
    ("ml_precision_recall_f1_averaged", &["averages", "confusion"]),
    ("ml_quantile_loss", &["loss"]),
    ("ml_rand_index", &["contingency", "scores"]),
    ("ml_receptive_field", &["growth"]),
    ("ml_regression_metrics", &["parity", "scores"]),
    ("ml_ridge_regression", &["fit"]),
    ("ml_rnn_parameter_count", &["counts"]),
    ("ml_robust_scaler", &["before-after"]),
    ("ml_roc_auc", &["roc"]),
    ("ml_roc_curve", &["roc", "thresholds"]),
    ("ml_sgd_momentum_step", &["path"]),
    ("ml_silhouette_score", &["clusters", "silhouette"]),
    ("ml_softmax", &["probabilities"]),
    ("ml_softmax_temperature", &["probabilities"]),
    ("ml_standard_scaler", &["before-after", "matrix"]),
    ("ml_step_lr_schedule", &["schedule"]),
    ("ml_top_k_accuracy", &["scores"]),
    ("ml_train_test_split_sizes", &["field", "sizes"]),
    ("ml_transformer_parameter_count", &["counts"]),
    ("ml_triplet_loss", &["geometry"]),
    ("ml_tweedie_deviance", &["loss"]),
    ("ml_warmup_cosine_lr_schedule", &["schedule"]),
];

/// Inputs to change for a capability whose numbers cannot simply be scaled (a second, different example).
const OTHER: &[(&str, &str, &str)] = &[
    ("ml_cohen_kappa", "y2", "[0, 2, 2, 1, 1, 0, 1, 1]"),
    ("ml_adjusted_mutual_information", "labels_pred", "[0, 1, 1, 1, 2, 2]"),
    ("ml_normalized_mutual_information", "labels_pred", "[0, 1, 1, 1, 2, 2]"),
    ("ml_homogeneity_completeness_v_measure", "labels_pred", "[0, 1, 1, 1, 2, 2]"),
    ("ml_rand_index", "labels_pred", "[0, 1, 1, 1, 2, 2]"),
    ("ml_mutual_information_labels", "y", "[0, 1, 1, 1, 1, 0]"),
    ("ml_information_gain", "x", "[0, 1, 1, 2, 2, 2, 1, 0, 0, 2, 0, 1, 1, 0]"),
    ("ml_one_hot_encoding", "categories", "red, green, red, blue, green, blue, blue"),
    ("ml_rnn_parameter_count", "hidden_size", "64"),
    ("ml_multiclass_confusion_matrix", "y_pred", "[0, 2, 2, 1, 1, 0, 1, 1]"),
    ("ml_precision_recall_f1_averaged", "y_pred", "[0, 2, 2, 1, 1, 0, 1, 1]"),
    ("ml_classification_report", "y_pred", "[0, 2, 2, 1, 1, 0, 1, 1]"),
    ("ml_balanced_accuracy", "y_pred", "[0, 0, 0, 0, 0, 1, 1, 1]"),
    ("ml_binary_confusion_matrix", "y_pred", "[1, 1, 0, 1, 0, 1, 1, 0]"),
    ("ml_hamming_loss", "Y_pred", "[[1, 1, 0], [0, 1, 1]]"),
    ("ml_balanced_class_weights", "y", "[0, 0, 0, 0, 1, 1, 1, 2]"),
    ("ml_gini_split", "g", "[0, 0, 1, 0, 1, 1, 1, 1]"),
    ("ml_silhouette_score", "labels", "[0, 0, 0, 1, 1, 1]"),
    ("ml_calinski_harabasz_index", "labels", "[0, 0, 0, 1, 1, 1]"),
    ("ml_davies_bouldin_index", "labels", "[0, 0, 0, 1, 1, 1]"),
    ("ml_mean_average_precision", "R", "[[0, 1, 1, 0, 0], [1, 1, 0, 0, 1]]"),
    ("ml_mean_reciprocal_rank", "R", "[[0, 1, 0], [0, 0, 1], [1, 0, 0]]"),
    ("ml_hierarchical_clustering", "X", "[[0, 0], [0, 2], [4, 0], [4, 1.5], [9, 10]]"),
    ("ml_decision_stump_classification", "y", "[0, 1, 1, 1, 0, 1]"),
    ("ml_top_k_accuracy", "y_true", "[1, 1, 2, 0]"),
    ("ml_entropy", "p", "[0.6, 0.3, 0.1]"),
    ("ml_joint_conditional_entropy", "P", "[[0.4, 0.1], [0.3, 0.2]]"),
    ("ml_kl_divergence_discrete", "q", "[0.7, 0.3]"),
    ("ml_log_loss_multiclass", "P", "[[0.6, 0.3, 0.1], [0.2, 0.2, 0.6], [0.25, 0.5, 0.25]]"),
];

/// The arguments of a structured example: solve('key', a='1 V', b=[1, 2], c="(A·B)'"); a bracketed list is one
/// value.
pub fn example(key: &str) -> Vec<(String, Input)> {
    let p = pack();
    let ex = &p.procedure(key).expect(key).structured_example;
    let cs: Vec<char> = ex.chars().collect();
    let mut i = ex.find(',').map(|x| ex[..x].chars().count() + 1).unwrap_or(cs.len());
    let mut out = vec![];
    while i < cs.len() {
        while i < cs.len() && (cs[i] == ' ' || cs[i] == ',') {
            i += 1;
        }
        if i >= cs.len() || cs[i] == ')' {
            break;
        }
        let st = i;
        while cs[i] != '=' {
            i += 1;
        }
        let name: String = cs[st..i].iter().collect();
        i += 1;
        let value: String;
        if cs[i] == '\'' || cs[i] == '"' {
            let qch = cs[i];
            let s0 = i + 1;
            i += 1;
            while cs[i] != qch {
                i += 1;
            }
            value = cs[s0..i].iter().collect();
            i += 1;
        } else {
            let s0 = i;
            let mut depth = 0;
            while i < cs.len() && (depth > 0 || (cs[i] != ',' && cs[i] != ')')) {
                match cs[i] {
                    '[' => depth += 1,
                    ']' => depth -= 1,
                    _ => {}
                }
                i += 1;
            }
            value = cs[s0..i].iter().collect();
        }
        out.push((name.trim().to_string(), Input::Text(value.trim().to_string())));
    }
    out
}

/// Every text inside the canvas, no non-finite number (ax_engine::figkit::check_figure), public wording.
fn check(f: &Figure) {
    if let Err(e) = problems(f) {
        panic!("{e}");
    }
}

fn problems(f: &Figure) -> Result<(), String> {
    ax_engine::figkit::check_figure(f)?;
    let words = format!("{} {}", f.title, f.caption).to_lowercase();
    for bad in ["—", " we ", " our ", " us ", "package", "stax"] {
        if words.contains(bad) {
            return Err(format!("{}: '{bad}' in the public text: {}", f.slug, words));
        }
    }
    Ok(())
}

#[test]
fn every_capability_draws_its_published_kinds() {
    let p = pack();
    let mut missing = vec![];
    for (key, kinds) in KINDS {
        let given = example(key);
        let figs = figures(p, key, &given, None, None);
        for kind in *kinds {
            match figs.iter().find(|f| f.kind == *kind && !f.slug.contains("-live-")) {
                Some(f) => {
                    assert_eq!(f.slug, format!("{}-{kind}", key.replace('_', "-")));
                    if let Err(e) = problems(f) {
                        missing.push(e);
                    }
                }
                None => missing.push(format!("{key}: {kind}")),
            }
        }
        if let Some(i) = figs.iter().position(|f| f.slug.contains("-live-")) {
            assert!(figs[i..].iter().all(|f| f.slug.contains("-live-")), "{key}: generic figures before own ones");
        }
    }
    assert!(missing.is_empty(), "missing figures: {missing:?}");
}

/// Numbers in a text multiplied by f ("12, 3" -> "16.44, 4.11").
fn scale_text(t: &str, f: f64) -> Option<String> {
    let mut out = String::new();
    let mut cur = String::new();
    let mut changed = false;
    let flush = |cur: &mut String, out: &mut String, changed: &mut bool| {
        if !cur.is_empty() {
            match cur.parse::<f64>() {
                Ok(x) if x != 0.0 => {
                    out.push_str(&format!("{}", x * f));
                    *changed = true;
                }
                _ => out.push_str(cur),
            }
            cur.clear();
        }
    };
    for c in t.chars() {
        if c.is_ascii_digit() || c == '.' || (c == '-' && cur.is_empty()) || ((c == 'e' || c == 'E') && !cur.is_empty()) {
            cur.push(c);
        } else {
            flush(&mut cur, &mut out, &mut changed);
            out.push(c);
        }
    }
    flush(&mut cur, &mut out, &mut changed);
    changed.then_some(out)
}

/// A second example: a listed replacement, else each input's numbers scaled until the model solves it and every
/// kind is drawn differently.
fn changed_example(key: &str, kinds: &[&str], base: &[Figure]) -> Option<Vec<(String, Input)>> {
    let p = pack();
    let given = example(key);
    let mut tries: Vec<Vec<(String, Input)>> = vec![];
    if let Some((_, sym, val)) = OTHER.iter().find(|o| o.0 == key) {
        let mut g = given.clone();
        for e in g.iter_mut() {
            if e.0 == *sym {
                e.1 = Input::Text(val.to_string());
            }
        }
        tries.push(g);
    }
    for f in [1.37, 0.73, 1.1, 2.0] {
        for i in 0..given.len() {
            if let Input::Text(t) = &given[i].1 {
                if let Some(t2) = scale_text(t, f) {
                    let mut g = given.clone();
                    g[i].1 = Input::Text(t2);
                    tries.push(g);
                }
            }
        }
    }
    for g in tries {
        if !ax_engine::solve(p, key, &g, None).solved() {
            continue;
        }
        let figs = figures(p, key, &g, None, None);
        let differs = kinds.iter().all(|kind| match (base.iter().find(|f| f.kind == *kind), figs.iter().find(|f| f.kind == *kind)) {
            (Some(a), Some(b)) => a.svg != b.svg,
            _ => false,
        });
        if differs {
            for f in figs.iter().filter(|f| !f.slug.contains("-live-")) {
                check(f);
            }
            return Some(g);
        }
    }
    None
}

#[test]
fn a_changed_input_changes_the_drawing() {
    let p = pack();
    let mut unchanged = vec![];
    for (key, kinds) in KINDS {
        let base = figures(p, key, &example(key), None, None);
        if changed_example(key, kinds, &base).is_none() {
            unchanged.push(*key);
        }
    }
    assert!(unchanged.is_empty(), "drawing did not follow a changed input: {unchanged:?}");
}

#[test]
fn a_refused_input_draws_nothing() {
    // lists of different lengths, a probability outside [0, 1]: refused by the model, so no figure at all
    let p = pack();
    for (key, sym, bad) in [("ml_brier_score", "p", "[0.9, 0.2, 1.7, 0.6]"), ("ml_regression_metrics", "y_pred", "[2.5, 0]"),
                            ("ml_softmax_temperature", "T", "-1")] {
        let mut g = example(key);
        for e in g.iter_mut() {
            if e.0 == sym {
                e.1 = Input::Text(bad.into());
            }
        }
        let a = ax_engine::solve(p, key, &g, None);
        assert!(!a.solved(), "{key}: {sym} = {bad} was solved");
        assert!(figures(p, key, &g, None, None).is_empty(), "{key}: a figure for a refused input");
    }
}
