//! data_science_ml: procedures translated from the package's Python by tools/py2rust.py (do not edit; regenerate).
//! Source: models/data_science_ml/capabilities/procedures.py. Each keeps Python's number rules through
//! ax_engine::py and is checked against the package by tools/rust_accuracy.py before the build is listed.
#![allow(clippy::all, non_snake_case, unused_parens, unused_braces, unused_mut, unused_variables, unreachable_code)]

use ax_engine::py::{self, Py, PyErr, R};
use ax_engine::Val;
use serde_json::Value;
use std::collections::HashMap;

/// The procedures this crate implements (tools/export_rust_pack.py reads this list).
pub const NATIVE: &[&str] = &[
    "ml_adam_step",
    "ml_adamw_step",
    "ml_adjusted_mutual_information",
    "ml_adjusted_r2",
    "ml_average_precision",
    "ml_backpropagation",
    "ml_balanced_accuracy",
    "ml_balanced_class_weights",
    "ml_batch_norm_forward",
    "ml_bce_with_logits",
    "ml_bernoulli_naive_bayes",
    "ml_bias_variance_decomposition",
    "ml_binary_confusion_matrix",
    "ml_binary_metrics_from_counts",
    "ml_brier_score",
    "ml_calinski_harabasz_index",
    "ml_classification_report",
    "ml_cohen_kappa",
    "ml_conv2d_forward",
    "ml_conv_output_size",
    "ml_conv_parameter_count",
    "ml_cosine_lr_schedule",
    "ml_cross_entropy_logits",
    "ml_davies_bouldin_index",
    "ml_decision_stump_classification",
    "ml_decision_stump_regression",
    "ml_dense_forward_pass",
    "ml_dense_parameter_count",
    "ml_distance_metrics",
    "ml_entropy",
    "ml_expected_calibration_error",
    "ml_fbeta_score",
    "ml_focal_loss",
    "ml_gaussian_naive_bayes",
    "ml_gini_split",
    "ml_gradient_descent_step_linear",
    "ml_hamming_loss",
    "ml_hierarchical_clustering",
    "ml_hinge_loss",
    "ml_homogeneity_completeness_v_measure",
    "ml_huber_loss",
    "ml_impurity_from_counts",
    "ml_information_gain",
    "ml_joint_conditional_entropy",
    "ml_kfold_cv_linear_regression",
    "ml_kfold_sizes",
    "ml_kl_divergence_discrete",
    "ml_kmeans",
    "ml_kmeans_step",
    "ml_knn_classify",
    "ml_knn_regress",
    "ml_label_smoothing_cross_entropy",
    "ml_lasso_regression",
    "ml_linear_svm_decision",
    "ml_log_loss_binary",
    "ml_log_loss_multiclass",
    "ml_log_softmax",
    "ml_logistic_loss_gradient",
    "ml_logistic_regression_fit",
    "ml_logistic_regression_predict",
    "ml_loocv_linear_regression",
    "ml_mape",
    "ml_matthews_correlation",
    "ml_mean_average_precision",
    "ml_mean_reciprocal_rank",
    "ml_mean_squared_log_error",
    "ml_minmax_scaler",
    "ml_mse_gradient",
    "ml_multiclass_confusion_matrix",
    "ml_multiclass_hinge_loss",
    "ml_multihead_attention",
    "ml_multinomial_naive_bayes",
    "ml_mutual_information_labels",
    "ml_ndcg_at_k",
    "ml_normalized_mutual_information",
    "ml_one_hot_encoding",
    "ml_pca",
    "ml_perceptron_epoch",
    "ml_polynomial_kernel",
    "ml_pooling",
    "ml_precision_recall_at_k",
    "ml_precision_recall_f1_averaged",
    "ml_quantile_loss",
    "ml_rand_index",
    "ml_rbf_kernel_matrix",
    "ml_receptive_field",
    "ml_regression_metrics",
    "ml_ridge_regression",
    "ml_rnn_parameter_count",
    "ml_robust_scaler",
    "ml_roc_auc",
    "ml_roc_curve",
    "ml_scaled_dot_product_attention",
    "ml_sgd_momentum_step",
    "ml_sigmoid",
    "ml_silhouette_score",
    "ml_softmax",
    "ml_softmax_temperature",
    "ml_standard_scaler",
    "ml_step_lr_schedule",
    "ml_threshold_metrics",
    "ml_top_k_accuracy",
    "ml_train_test_split_sizes",
    "ml_transformer_parameter_count",
    "ml_triplet_loss",
    "ml_tweedie_deviance",
    "ml_warmup_cosine_lr_schedule",
];

type Out = Result<(Vec<(String, Value)>, f64), String>;

/// The run's result as the engine's answer: a dict of outputs and the residual of the package's own check; a
/// ValueError is the refusal's reason; any other error is an internal error (None).
fn finish(r: R, residual: impl Fn(&Py, &Py) -> f64, v: &HashMap<String, Val>) -> Option<Out> {
    let o = match r {
        Ok(o) => o,
        Err(PyErr::Value(m)) => return Some(Err(m)),
        Err(e) => {
            if std::env::var_os("AX_DEBUG").is_some() {        // the error behind an internal error, for diagnosis
                eprintln!("internal error: {e:?}");
            }
            return None;
        }
    };
    let Py::Dict(d) = &o else { return None };
    let outs: Vec<(String, Value)> = d.borrow().iter().map(|(k, x)| (py::to_str(k), py::to_json(x))).collect();
    let i = Py::dict(v.iter().map(|(k, x)| (Py::str(k), py::from_val(x))).collect());
    let res = residual(&i, &o);
    if std::env::var_os("AX_DEBUG").is_some() && !(res <= 1e-9) {
        eprintln!("check failed: residual {res}");
    }
    Some(Ok((outs, res)))
}

/// The engine's entry point (ax_engine::pack::NativeFn): None for a key that is not one of these procedures.
pub fn native(key: &str, v: &HashMap<String, Val>) -> Option<Out> {
    match key {
        "ml_adam_step" => {
            let r = f_adam_step(match v.get("w") { Some(x) => py::from_val(x), None => { return None } }, match v.get("grad") { Some(x) => py::from_val(x), None => { return None } }, match v.get("m") { Some(x) => py::from_val(x), None => { return None } }, match v.get("v") { Some(x) => py::from_val(x), None => { return None } }, match v.get("t") { Some(x) => py::from_val(x), None => { return None } }, match v.get("lr") { Some(x) => py::from_val(x), None => { return None } }, match v.get("beta1") { Some(x) => py::from_val(x), None => { return None } }, match v.get("beta2") { Some(x) => py::from_val(x), None => { return None } }, match v.get("eps") { Some(x) => py::from_val(x), None => { return None } }, (|| -> R { Ok(Py::None) })().ok()?);
            finish(r, |i: &Py, o: &Py| -> f64 { match s_62(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_adamw_step" => {
            let r = s_91(match v.get("w") { Some(x) => py::from_val(x), None => { return None } }, match v.get("grad") { Some(x) => py::from_val(x), None => { return None } }, match v.get("m") { Some(x) => py::from_val(x), None => { return None } }, match v.get("v") { Some(x) => py::from_val(x), None => { return None } }, match v.get("t") { Some(x) => py::from_val(x), None => { return None } }, match v.get("lr") { Some(x) => py::from_val(x), None => { return None } }, match v.get("beta1") { Some(x) => py::from_val(x), None => { return None } }, match v.get("beta2") { Some(x) => py::from_val(x), None => { return None } }, match v.get("eps") { Some(x) => py::from_val(x), None => { return None } }, match v.get("weight_decay") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_156(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_adjusted_mutual_information" => {
            let r = f_ami(match v.get("labels_true") { Some(x) => py::from_val(x), None => { return None } }, match v.get("labels_pred") { Some(x) => py::from_val(x), None => { return None } }, match v.get("method") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_247(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_adjusted_r2" => {
            let r = f_adjusted_r2(match v.get("y_true") { Some(x) => py::from_val(x), None => { return None } }, match v.get("y_pred") { Some(x) => py::from_val(x), None => { return None } }, match v.get("p") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_327(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_average_precision" => {
            let r = f_average_precision(match v.get("y_true") { Some(x) => py::from_val(x), None => { return None } }, match v.get("scores") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_404(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_backpropagation" => {
            let r = f_backprop(match v.get("x") { Some(x) => py::from_val(x), None => { return None } }, match v.get("t") { Some(x) => py::from_val(x), None => { return None } }, match v.get("W1") { Some(x) => py::from_val(x), None => { return None } }, match v.get("b1") { Some(x) => py::from_val(x), None => { return None } }, match v.get("activation") { Some(x) => py::from_val(x), None => { return None } }, match v.get("W2") { Some(x) => py::from_val(x), None => { return None } }, match v.get("b2") { Some(x) => py::from_val(x), None => { return None } }, match v.get("loss") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_539(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_balanced_accuracy" => {
            let r = f_balanced_accuracy(match v.get("y_true") { Some(x) => py::from_val(x), None => { return None } }, match v.get("y_pred") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_602(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_balanced_class_weights" => {
            let r = f_class_weights(match v.get("y") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_654(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_batch_norm_forward" => {
            let r = f_batchnorm(match v.get("X") { Some(x) => py::from_val(x), None => { return None } }, match v.get("gamma") { Some(x) => py::from_val(x), None => { return None } }, match v.get("beta") { Some(x) => py::from_val(x), None => { return None } }, match v.get("eps") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_737(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_bce_with_logits" => {
            let r = f_bce_logits(match v.get("y") { Some(x) => py::from_val(x), None => { return None } }, match v.get("z") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_799(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_bernoulli_naive_bayes" => {
            let r = f_bernoulli_nb(match v.get("X") { Some(x) => py::from_val(x), None => { return None } }, match v.get("y") { Some(x) => py::from_val(x), None => { return None } }, match v.get("x") { Some(x) => py::from_val(x), None => { return None } }, match v.get("alpha") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_901(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_bias_variance_decomposition" => {
            let r = f_bias_variance(match v.get("P") { Some(x) => py::from_val(x), None => { return None } }, match v.get("f") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_977(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_binary_confusion_matrix" => {
            let r = f_binary_confusion(match v.get("y_true") { Some(x) => py::from_val(x), None => { return None } }, match v.get("y_pred") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_1062(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_binary_metrics_from_counts" => {
            let r = f_binary_from_counts(match v.get("TP") { Some(x) => py::from_val(x), None => { return None } }, match v.get("FP") { Some(x) => py::from_val(x), None => { return None } }, match v.get("FN") { Some(x) => py::from_val(x), None => { return None } }, match v.get("TN") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_1117(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_brier_score" => {
            let r = f_brier(match v.get("y_true") { Some(x) => py::from_val(x), None => { return None } }, match v.get("p") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_1183(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_calinski_harabasz_index" => {
            let r = f_calinski_harabasz(match v.get("X") { Some(x) => py::from_val(x), None => { return None } }, match v.get("labels") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_1269(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_classification_report" => {
            let r = f_per_class_report(match v.get("y_true") { Some(x) => py::from_val(x), None => { return None } }, match v.get("y_pred") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_1339(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_cohen_kappa" => {
            let r = f_cohen_kappa(match v.get("y1") { Some(x) => py::from_val(x), None => { return None } }, match v.get("y2") { Some(x) => py::from_val(x), None => { return None } }, match v.get("weights") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_1417(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_conv2d_forward" => {
            let r = f_conv2d(match v.get("image") { Some(x) => py::from_val(x), None => { return None } }, match v.get("kernel") { Some(x) => py::from_val(x), None => { return None } }, match v.get("stride") { Some(x) => py::from_val(x), None => { return None } }, match v.get("padding") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_1501(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_conv_output_size" => {
            let r = f_conv_out(match v.get("n") { Some(x) => py::from_val(x), None => { return None } }, match v.get("kernel") { Some(x) => py::from_val(x), None => { return None } }, match v.get("stride") { Some(x) => py::from_val(x), None => { return None } }, match v.get("padding") { Some(x) => py::from_val(x), None => { return None } }, match v.get("dilation") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_1549(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_conv_parameter_count" => {
            let r = f_conv_params(match v.get("in_channels") { Some(x) => py::from_val(x), None => { return None } }, match v.get("out_channels") { Some(x) => py::from_val(x), None => { return None } }, match v.get("kernel_h") { Some(x) => py::from_val(x), None => { return None } }, match v.get("kernel_w") { Some(x) => py::from_val(x), None => { return None } }, match v.get("groups") { Some(x) => py::from_val(x), None => { return None } }, match v.get("bias") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_1600(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_cosine_lr_schedule" => {
            let r = f_cosine_lr(match v.get("lr_max") { Some(x) => py::from_val(x), None => { return None } }, match v.get("eta_min") { Some(x) => py::from_val(x), None => { return None } }, match v.get("T_max") { Some(x) => py::from_val(x), None => { return None } }, match v.get("epoch") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_1661(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_cross_entropy_logits" => {
            let r = f_ce_logits(match v.get("z") { Some(x) => py::from_val(x), None => { return None } }, match v.get("y") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_1721(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_davies_bouldin_index" => {
            let r = f_davies_bouldin(match v.get("X") { Some(x) => py::from_val(x), None => { return None } }, match v.get("labels") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_1810(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_decision_stump_classification" => {
            let r = f_stump_classify(match v.get("X") { Some(x) => py::from_val(x), None => { return None } }, match v.get("y") { Some(x) => py::from_val(x), None => { return None } }, match v.get("criterion") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_1918(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_decision_stump_regression" => {
            let r = f_stump_regress(match v.get("X") { Some(x) => py::from_val(x), None => { return None } }, match v.get("y") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_2017(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_dense_forward_pass" => {
            let r = f_dense_forward(match v.get("x") { Some(x) => py::from_val(x), None => { return None } }, match v.get("W1") { Some(x) => py::from_val(x), None => { return None } }, match v.get("b1") { Some(x) => py::from_val(x), None => { return None } }, match v.get("activation") { Some(x) => py::from_val(x), None => { return None } }, match v.get("W2") { Some(x) => py::from_val(x), None => { return None } }, match v.get("b2") { Some(x) => py::from_val(x), None => { return None } }, match v.get("output") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_2116(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_dense_parameter_count" => {
            let r = f_dense_params(match v.get("sizes") { Some(x) => py::from_val(x), None => { return None } }, match v.get("bias") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_2171(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_distance_metrics" => {
            let r = f_distances(match v.get("x") { Some(x) => py::from_val(x), None => { return None } }, match v.get("z") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_2238(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_entropy" => {
            let r = f_entropy(match v.get("p") { Some(x) => py::from_val(x), None => { return None } }, match v.get("unit") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_2297(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_expected_calibration_error" => {
            let r = f_ece(match v.get("y_true") { Some(x) => py::from_val(x), None => { return None } }, match v.get("p") { Some(x) => py::from_val(x), None => { return None } }, match v.get("n_bins") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_2376(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_fbeta_score" => {
            let r = f_fbeta(match v.get("TP") { Some(x) => py::from_val(x), None => { return None } }, match v.get("FP") { Some(x) => py::from_val(x), None => { return None } }, match v.get("FN") { Some(x) => py::from_val(x), None => { return None } }, match v.get("beta") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_2435(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_focal_loss" => {
            let r = f_focal(match v.get("y") { Some(x) => py::from_val(x), None => { return None } }, match v.get("z") { Some(x) => py::from_val(x), None => { return None } }, match v.get("gamma") { Some(x) => py::from_val(x), None => { return None } }, match v.get("alpha") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_2511(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_gaussian_naive_bayes" => {
            let r = f_gaussian_nb(match v.get("X") { Some(x) => py::from_val(x), None => { return None } }, match v.get("y") { Some(x) => py::from_val(x), None => { return None } }, match v.get("x") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_2603(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_gini_split" => {
            let r = f_gini_split(match v.get("y") { Some(x) => py::from_val(x), None => { return None } }, match v.get("g") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_2664(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_gradient_descent_step_linear" => {
            let r = f_gd_step_linear(match v.get("X") { Some(x) => py::from_val(x), None => { return None } }, match v.get("y") { Some(x) => py::from_val(x), None => { return None } }, match v.get("w") { Some(x) => py::from_val(x), None => { return None } }, match v.get("b") { Some(x) => py::from_val(x), None => { return None } }, match v.get("lr") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_2759(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_hamming_loss" => {
            let r = f_hamming(match v.get("Y_true") { Some(x) => py::from_val(x), None => { return None } }, match v.get("Y_pred") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_2836(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_hierarchical_clustering" => {
            let r = f_linkage_heights(match v.get("X") { Some(x) => py::from_val(x), None => { return None } }, match v.get("method") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_2927(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_hinge_loss" => {
            let r = f_hinge(match v.get("y_true") { Some(x) => py::from_val(x), None => { return None } }, match v.get("scores") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_2995(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_homogeneity_completeness_v_measure" => {
            let r = f_hcv(match v.get("labels_true") { Some(x) => py::from_val(x), None => { return None } }, match v.get("labels_pred") { Some(x) => py::from_val(x), None => { return None } }, match v.get("beta") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_3084(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_huber_loss" => {
            let r = f_huber(match v.get("y_true") { Some(x) => py::from_val(x), None => { return None } }, match v.get("y_pred") { Some(x) => py::from_val(x), None => { return None } }, match v.get("delta") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_3144(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_impurity_from_counts" => {
            let r = f_impurity_counts(match v.get("counts") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_3207(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_information_gain" => {
            let r = f_information_gain(match v.get("y") { Some(x) => py::from_val(x), None => { return None } }, match v.get("x") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_3276(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_joint_conditional_entropy" => {
            let r = f_joint_entropy(match v.get("P") { Some(x) => py::from_val(x), None => { return None } }, match v.get("unit") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_3355(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_kfold_cv_linear_regression" => {
            let r = f_kfold_cv_ols(match v.get("X") { Some(x) => py::from_val(x), None => { return None } }, match v.get("y") { Some(x) => py::from_val(x), None => { return None } }, match v.get("k") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_3481(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_kfold_sizes" => {
            let r = f_kfold_sizes(match v.get("n") { Some(x) => py::from_val(x), None => { return None } }, match v.get("k") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_3532(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_kl_divergence_discrete" => {
            let r = f_kl_discrete(match v.get("p") { Some(x) => py::from_val(x), None => { return None } }, match v.get("q") { Some(x) => py::from_val(x), None => { return None } }, match v.get("unit") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_3596(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_kmeans" => {
            let r = f_kmeans(match v.get("X") { Some(x) => py::from_val(x), None => { return None } }, match v.get("C") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_3685(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_kmeans_step" => {
            let r = f_kmeans_step(match v.get("X") { Some(x) => py::from_val(x), None => { return None } }, match v.get("C") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_3770(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_knn_classify" => {
            let r = f_knn_classify(match v.get("X") { Some(x) => py::from_val(x), None => { return None } }, match v.get("y") { Some(x) => py::from_val(x), None => { return None } }, match v.get("x") { Some(x) => py::from_val(x), None => { return None } }, match v.get("k") { Some(x) => py::from_val(x), None => { return None } }, match v.get("metric") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_3872(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_knn_regress" => {
            let r = f_knn_regress(match v.get("X") { Some(x) => py::from_val(x), None => { return None } }, match v.get("y") { Some(x) => py::from_val(x), None => { return None } }, match v.get("x") { Some(x) => py::from_val(x), None => { return None } }, match v.get("k") { Some(x) => py::from_val(x), None => { return None } }, match v.get("metric") { Some(x) => py::from_val(x), None => { return None } }, match v.get("weights") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_3969(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_label_smoothing_cross_entropy" => {
            let r = f_label_smoothing_ce(match v.get("z") { Some(x) => py::from_val(x), None => { return None } }, match v.get("y") { Some(x) => py::from_val(x), None => { return None } }, match v.get("epsilon") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_4035(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_lasso_regression" => {
            let r = f_lasso(match v.get("X") { Some(x) => py::from_val(x), None => { return None } }, match v.get("y") { Some(x) => py::from_val(x), None => { return None } }, match v.get("alpha") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_4188(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_linear_svm_decision" => {
            let r = f_svm_decision(match v.get("w") { Some(x) => py::from_val(x), None => { return None } }, match v.get("b") { Some(x) => py::from_val(x), None => { return None } }, match v.get("X") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_4263(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_log_loss_binary" => {
            let r = f_log_loss_binary(match v.get("y_true") { Some(x) => py::from_val(x), None => { return None } }, match v.get("p") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_4331(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_log_loss_multiclass" => {
            let r = f_log_loss_multiclass(match v.get("y_true") { Some(x) => py::from_val(x), None => { return None } }, match v.get("P") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_4410(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_log_softmax" => {
            let r = f_log_softmax(match v.get("z") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_4461(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_logistic_loss_gradient" => {
            let r = f_logistic_gradient(match v.get("X") { Some(x) => py::from_val(x), None => { return None } }, match v.get("y") { Some(x) => py::from_val(x), None => { return None } }, match v.get("w") { Some(x) => py::from_val(x), None => { return None } }, match v.get("b") { Some(x) => py::from_val(x), None => { return None } }, match v.get("lam") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_4547(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_logistic_regression_fit" => {
            let r = f_logistic_fit(match v.get("X") { Some(x) => py::from_val(x), None => { return None } }, match v.get("y") { Some(x) => py::from_val(x), None => { return None } }, match v.get("C") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_4690(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_logistic_regression_predict" => {
            let r = f_logistic_predict(match v.get("w") { Some(x) => py::from_val(x), None => { return None } }, match v.get("b") { Some(x) => py::from_val(x), None => { return None } }, match v.get("X") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_4764(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_loocv_linear_regression" => {
            let r = f_loocv_ols(match v.get("X") { Some(x) => py::from_val(x), None => { return None } }, match v.get("y") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_4878(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_mape" => {
            let r = f_mape(match v.get("y_true") { Some(x) => py::from_val(x), None => { return None } }, match v.get("y_pred") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_4936(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_matthews_correlation" => {
            let r = f_mcc(match v.get("TP") { Some(x) => py::from_val(x), None => { return None } }, match v.get("FP") { Some(x) => py::from_val(x), None => { return None } }, match v.get("FN") { Some(x) => py::from_val(x), None => { return None } }, match v.get("TN") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_4985(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_mean_average_precision" => {
            let r = f_mean_ap(match v.get("R") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_5058(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_mean_reciprocal_rank" => {
            let r = f_mrr(match v.get("R") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_5128(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_mean_squared_log_error" => {
            let r = f_msle(match v.get("y_true") { Some(x) => py::from_val(x), None => { return None } }, match v.get("y_pred") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_5186(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_minmax_scaler" => {
            let r = f_minmax_scaler(match v.get("X") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_5255(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_mse_gradient" => {
            let r = f_mse_gradient(match v.get("X") { Some(x) => py::from_val(x), None => { return None } }, match v.get("y") { Some(x) => py::from_val(x), None => { return None } }, match v.get("w") { Some(x) => py::from_val(x), None => { return None } }, match v.get("b") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_5337(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_multiclass_confusion_matrix" => {
            let r = f_multiclass_confusion(match v.get("y_true") { Some(x) => py::from_val(x), None => { return None } }, match v.get("y_pred") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_5399(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_multiclass_hinge_loss" => {
            let r = f_multiclass_hinge(match v.get("y_true") { Some(x) => py::from_val(x), None => { return None } }, match v.get("D") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_5475(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_multihead_attention" => {
            let r = f_multihead(match v.get("Q") { Some(x) => py::from_val(x), None => { return None } }, match v.get("K") { Some(x) => py::from_val(x), None => { return None } }, match v.get("V") { Some(x) => py::from_val(x), None => { return None } }, match v.get("heads") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_5575(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_multinomial_naive_bayes" => {
            let r = f_multinomial_nb(match v.get("X") { Some(x) => py::from_val(x), None => { return None } }, match v.get("y") { Some(x) => py::from_val(x), None => { return None } }, match v.get("x") { Some(x) => py::from_val(x), None => { return None } }, match v.get("alpha") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_5676(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_mutual_information_labels" => {
            let r = f_mutual_info_labels(match v.get("x") { Some(x) => py::from_val(x), None => { return None } }, match v.get("y") { Some(x) => py::from_val(x), None => { return None } }, match v.get("unit") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_5755(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_ndcg_at_k" => {
            let r = f_ndcg(match v.get("rel") { Some(x) => py::from_val(x), None => { return None } }, match v.get("k") { Some(x) => py::from_val(x), None => { return None } }, match v.get("gain") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_5821(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_normalized_mutual_information" => {
            let r = f_nmi(match v.get("labels_true") { Some(x) => py::from_val(x), None => { return None } }, match v.get("labels_pred") { Some(x) => py::from_val(x), None => { return None } }, match v.get("method") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_5905(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_one_hot_encoding" => {
            let r = f_one_hot(match v.get("categories") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_5958(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_pca" => {
            let r = f_pca(match v.get("X") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_6070(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_perceptron_epoch" => {
            let r = f_perceptron_epoch(match v.get("X") { Some(x) => py::from_val(x), None => { return None } }, match v.get("y") { Some(x) => py::from_val(x), None => { return None } }, match v.get("w") { Some(x) => py::from_val(x), None => { return None } }, match v.get("b") { Some(x) => py::from_val(x), None => { return None } }, match v.get("lr") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_6161(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_polynomial_kernel" => {
            let r = f_poly_kernel(match v.get("x") { Some(x) => py::from_val(x), None => { return None } }, match v.get("z") { Some(x) => py::from_val(x), None => { return None } }, match v.get("gamma") { Some(x) => py::from_val(x), None => { return None } }, match v.get("coef0") { Some(x) => py::from_val(x), None => { return None } }, match v.get("degree") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_6219(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_pooling" => {
            let r = f_pooling(match v.get("X") { Some(x) => py::from_val(x), None => { return None } }, match v.get("size") { Some(x) => py::from_val(x), None => { return None } }, match v.get("stride") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_6297(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_precision_recall_at_k" => {
            let r = f_prec_rec_at_k(match v.get("rel") { Some(x) => py::from_val(x), None => { return None } }, match v.get("n_relevant") { Some(x) => py::from_val(x), None => { return None } }, match v.get("k") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_6360(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_precision_recall_f1_averaged" => {
            let r = f_prf_averaged(match v.get("y_true") { Some(x) => py::from_val(x), None => { return None } }, match v.get("y_pred") { Some(x) => py::from_val(x), None => { return None } }, match v.get("average") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_6444(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_quantile_loss" => {
            let r = f_pinball(match v.get("y_true") { Some(x) => py::from_val(x), None => { return None } }, match v.get("y_pred") { Some(x) => py::from_val(x), None => { return None } }, match v.get("alpha") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_6502(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_rand_index" => {
            let r = f_rand_index(match v.get("labels_true") { Some(x) => py::from_val(x), None => { return None } }, match v.get("labels_pred") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_6579(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_rbf_kernel_matrix" => {
            let r = f_rbf_kernel(match v.get("X") { Some(x) => py::from_val(x), None => { return None } }, match v.get("gamma") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_6647(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_receptive_field" => {
            let r = f_receptive_field(match v.get("kernels") { Some(x) => py::from_val(x), None => { return None } }, match v.get("strides") { Some(x) => py::from_val(x), None => { return None } }, match v.get("dilations") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_6710(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_regression_metrics" => {
            let r = f_regression_metrics(match v.get("y_true") { Some(x) => py::from_val(x), None => { return None } }, match v.get("y_pred") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_6776(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_ridge_regression" => {
            let r = f_ridge(match v.get("X") { Some(x) => py::from_val(x), None => { return None } }, match v.get("y") { Some(x) => py::from_val(x), None => { return None } }, match v.get("alpha") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_6886(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_rnn_parameter_count" => {
            let r = f_rnn_params(match v.get("cell") { Some(x) => py::from_val(x), None => { return None } }, match v.get("input_size") { Some(x) => py::from_val(x), None => { return None } }, match v.get("hidden_size") { Some(x) => py::from_val(x), None => { return None } }, match v.get("num_layers") { Some(x) => py::from_val(x), None => { return None } }, match v.get("bidirectional") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_6942(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_robust_scaler" => {
            let r = f_robust_scaler(match v.get("X") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_7009(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_roc_auc" => {
            let r = f_roc_auc(match v.get("y_true") { Some(x) => py::from_val(x), None => { return None } }, match v.get("scores") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_7084(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_roc_curve" => {
            let r = f_roc_curve(match v.get("y_true") { Some(x) => py::from_val(x), None => { return None } }, match v.get("scores") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_7163(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_scaled_dot_product_attention" => {
            let r = f_attention(match v.get("Q") { Some(x) => py::from_val(x), None => { return None } }, match v.get("K") { Some(x) => py::from_val(x), None => { return None } }, match v.get("V") { Some(x) => py::from_val(x), None => { return None } }, match v.get("causal") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_7240(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_sgd_momentum_step" => {
            let r = f_momentum_step(match v.get("w") { Some(x) => py::from_val(x), None => { return None } }, match v.get("grad") { Some(x) => py::from_val(x), None => { return None } }, match v.get("buf") { Some(x) => py::from_val(x), None => { return None } }, match v.get("lr") { Some(x) => py::from_val(x), None => { return None } }, match v.get("momentum") { Some(x) => py::from_val(x), None => { return None } }, match v.get("nesterov") { Some(x) => py::from_val(x), None => { return None } }, match v.get("weight_decay") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_7317(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_sigmoid" => {
            let r = f_sigmoid(match v.get("z") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_7365(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_silhouette_score" => {
            let r = f_silhouette(match v.get("X") { Some(x) => py::from_val(x), None => { return None } }, match v.get("labels") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_7455(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_softmax" => {
            let r = f_softmax(match v.get("z") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_7507(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_softmax_temperature" => {
            let r = f_softmax_temperature(match v.get("z") { Some(x) => py::from_val(x), None => { return None } }, match v.get("T") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_7564(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_standard_scaler" => {
            let r = f_standard_scaler(match v.get("X") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_7634(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_step_lr_schedule" => {
            let r = f_step_lr(match v.get("lr0") { Some(x) => py::from_val(x), None => { return None } }, match v.get("step_size") { Some(x) => py::from_val(x), None => { return None } }, match v.get("gamma") { Some(x) => py::from_val(x), None => { return None } }, match v.get("epoch") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_7692(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_threshold_metrics" => {
            let r = f_threshold_metrics(match v.get("y_true") { Some(x) => py::from_val(x), None => { return None } }, match v.get("scores") { Some(x) => py::from_val(x), None => { return None } }, match v.get("threshold") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_7788(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_top_k_accuracy" => {
            let r = f_top_k_accuracy(match v.get("y_true") { Some(x) => py::from_val(x), None => { return None } }, match v.get("S") { Some(x) => py::from_val(x), None => { return None } }, match v.get("k") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_7869(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_train_test_split_sizes" => {
            let r = f_split_sizes(match v.get("n") { Some(x) => py::from_val(x), None => { return None } }, match v.get("test_size") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_7921(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_transformer_parameter_count" => {
            let r = f_transformer_params(match v.get("d_model") { Some(x) => py::from_val(x), None => { return None } }, match v.get("n_heads") { Some(x) => py::from_val(x), None => { return None } }, match v.get("d_ff") { Some(x) => py::from_val(x), None => { return None } }, match v.get("n_layers") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_7976(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_triplet_loss" => {
            let r = f_triplet(match v.get("anchor") { Some(x) => py::from_val(x), None => { return None } }, match v.get("positive") { Some(x) => py::from_val(x), None => { return None } }, match v.get("negative") { Some(x) => py::from_val(x), None => { return None } }, match v.get("margin") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_8035(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_tweedie_deviance" => {
            let r = f_tweedie(match v.get("y_true") { Some(x) => py::from_val(x), None => { return None } }, match v.get("y_pred") { Some(x) => py::from_val(x), None => { return None } }, match v.get("power") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_8119(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        "ml_warmup_cosine_lr_schedule" => {
            let r = f_warmup_cosine(match v.get("lr_max") { Some(x) => py::from_val(x), None => { return None } }, match v.get("start_factor") { Some(x) => py::from_val(x), None => { return None } }, match v.get("warmup") { Some(x) => py::from_val(x), None => { return None } }, match v.get("total") { Some(x) => py::from_val(x), None => { return None } }, match v.get("eta_min") { Some(x) => py::from_val(x), None => { return None } }, match v.get("step") { Some(x) => py::from_val(x), None => { return None } });
            finish(r, |i: &Py, o: &Py| -> f64 { match s_8188(i.clone(), o.clone()).and_then(|r| r.f()) { Ok(x) => x, Err(e) => { if std::env::var_os("AX_DEBUG").is_some() { eprintln!("check error: {e:?}"); } f64::NAN } } }, v)
        }
        _ => None,
    }
}


#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__F(mut v_x: Py) -> R {
    return Ok(py::fraction(&[v_x.clone()])?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__H(mut v_probs: Py) -> R {
    return Ok(py::neg(&py::fsum(&{ let mut out = vec![]; for t5885 in py::iter(&(v_probs.clone()))? { let mut v_p: Py; 
v_p = t5885;
if ({ let t5886 = v_p.clone(); Py::Bool({ let t5887 = Py::Int(0i64); if py::gt(&t5886, &t5887)? { true } else { false } }) }).truthy() { out.push(py::mul(&v_p.clone(), &py::math1("log", &v_p.clone())?)?); }} Py::list(out) })?)?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__Hc(mut v_counts: Py) -> R {
    let mut v_n: Py = Py::None;
    v_n = py::sum(&v_counts.clone(), None)?;
    return Ok(f__H({ let mut out = vec![]; for t5888 in py::iter(&(v_counts.clone()))? { let mut v_c: Py; 
v_c = t5888;
if ({ let t5889 = v_c.clone(); Py::Bool({ let t5890 = Py::Int(0i64); if py::gt(&t5889, &t5890)? { true } else { false } }) }).truthy() { out.push(py::truediv(&v_c.clone(), &v_n.clone())?); }} Py::list(out) })?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__LN(mut v_unit: Py) -> R {
    return Ok(py::getitem(&Py::dict(vec![(Py::str("nats"), Py::Float(1.0)), (Py::str("bits"), py::math1("log", &Py::Float(2.0))?), (Py::str("bans"), py::math1("log", &Py::Float(10.0))?)]), &v_unit.clone())?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__act(mut v_name: Py, mut v_v: Py) -> R {
    if ({ let t2084 = v_name.clone(); Py::Bool({ let t2085 = Py::str("relu"); if py::eq(&t2084, &t2085) { true } else { false } }) }).truthy() {
        return Ok((if ({ let t2086 = v_v.clone(); Py::Bool({ let t2087 = Py::Int(0i64); if py::gt(&t2086, &t2087)? { true } else { false } }) }).truthy() { v_v.clone() } else { Py::Float(0.0) }));
    }
    if ({ let t2088 = v_name.clone(); Py::Bool({ let t2089 = Py::str("tanh"); if py::eq(&t2088, &t2089) { true } else { false } }) }).truthy() {
        return Ok(py::math1("tanh", &v_v.clone())?);
    }
    return Ok(f__sigmoid(v_v.clone())?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__assign(mut v_F: Py, mut v_C: Py) -> R {
    let mut v_d: Py = Py::None;
    let mut v_lab: Py = Py::None;
    let mut v_m: Py = Py::None;
    let mut v_tot: Py = Py::None;
    let mut v_x: Py = Py::None;
    let t3742 = (Py::tuple(vec![Py::list(vec![]), py::fraction(&[Py::Int(0i64)])?])).items()?;
    if t3742.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_lab = t3742[0].clone();
    v_tot = t3742[1].clone();
    for t3743 in py::iter(&(v_F.clone()))? {
        v_x = t3743;
        v_d = { let mut out = vec![]; for t3744 in py::iter(&(v_C.clone()))? { let mut v_c: Py; 
v_c = t3744;
out.push(f__sqd_exact(v_x.clone(), v_c.clone())?); } Py::list(out) };
        v_m = py::minmax(&[v_d.clone()], false)?;
        let _ = f__need({ let t3747 = py::method(&v_d.clone(), "count", &[v_m.clone()])?; Py::Bool({ let t3748 = Py::Int(1i64); if py::eq(&t3747, &t3748) { true } else { false } }) }, Py::str("a point is exactly equidistant from two centroids (the assignment is ambiguous)"))?;
        let _ = py::method(&v_lab.clone(), "append", &[py::method(&v_d.clone(), "index", &[v_m.clone()])?])?;
        v_tot = py::iadd(&v_tot, &v_m.clone())?;
    }
    return Ok(Py::tuple(vec![v_lab.clone(), v_tot.clone()]));
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__binary(mut v_name: Py, mut v_v: Py, mut v_lo: Py) -> R {
    let mut v_out: Py = Py::None;
    v_out = f__labels(v_name.clone(), v_v.clone(), v_lo.clone())?;
    let _ = f__need(Py::Bool(({ let mut out = vec![]; for t7730 in py::iter(&(v_out.clone()))? { let mut v_x: Py; 
v_x = t7730;
out.push({ let t7731 = v_x.clone(); Py::Bool({ let t7732 = Py::tuple(vec![Py::Int(0i64), Py::Int(1i64)]); if py::contains(&t7732, &t7731)? { true } else { false } }) }); } Py::list(out) }).items()?.iter().all(|x| x.truthy())), Py::str(&["every label in ".to_string(), py::format(&v_name.clone(), "")?, " must be 0 or 1".to_string()].concat()))?;
    return Ok(v_out.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__cats(mut v_categories: Py) -> R {
    let mut v_items: Py = Py::None;
    let _ = f__need(Py::Bool(matches!(v_categories.clone(), Py::Str(_))), Py::str("categories must be a list of words"))?;
    v_items = { let mut out = vec![]; for t5934 in py::iter(&(py::method(&v_categories.clone(), "split", &[Py::str(",")])?))? { let mut v_c: Py; 
v_c = t5934;
out.push(py::method(&v_c.clone(), "strip", &[])?); } Py::list(out) };
    let _ = f__need({ let t5935 = v_items.clone(); if !t5935.truthy() { t5935 } else { Py::Bool((v_items.clone()).items()?.iter().all(|x| x.truthy())) } }, Py::str("every category must be a non-empty word"))?;
    let _ = f__need({ let t5936 = py::len(&v_items.clone())?; Py::Bool({ let t5937 = Py::Int(500i64); if py::le(&t5936, &t5937)? { true } else { false } }) }, Py::str("at most 500 values"))?;
    return Ok(v_items.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__centroids(mut v_M: Py, mut v_lab: Py) -> R {
    let mut v_C: Py = Py::None;
    let mut v_k: Py = Py::None;
    let mut v_ks: Py = Py::None;
    let mut v_rows: Py = Py::None;
    v_ks = py::sorted(&py::set(&v_lab.clone())?, (Py::Bool(false)).truthy())?;
    v_C = Py::dict(vec![]);
    for t1776 in py::iter(&(v_ks.clone()))? {
        v_k = t1776;
        v_rows = { let mut out = vec![]; for t1777 in py::range_iter(&[(py::len(&v_M.clone())?).i()?]) { let mut v_i: Py; 
v_i = t1777;
if ({ let t1778 = py::getitem(&v_lab.clone(), &v_i.clone())?; Py::Bool({ let t1779 = v_k.clone(); if py::eq(&t1778, &t1779) { true } else { false } }) }).truthy() { out.push(py::getitem(&v_M.clone(), &v_i.clone())?); }} Py::list(out) };
        py::setitem(&v_C.clone(), &v_k.clone(), { let mut out = vec![]; for t1780 in py::range_iter(&[(py::len(&py::getitem(&v_M.clone(), &Py::Int(0i64))?)?).i()?]) { let mut v_j: Py; 
v_j = t1780;
out.push(py::truediv(&py::fsum(&{ let mut out = vec![]; for t1781 in py::iter(&(v_rows.clone()))? { let mut v_r: Py; 
v_r = t1781;
out.push(py::getitem(&v_r.clone(), &v_j.clone())?); } Py::list(out) })?, &py::len(&v_rows.clone())?)?); } Py::list(out) })?;
    }
    return Ok(Py::tuple(vec![v_ks.clone(), v_C.clone()]));
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__check_counts(mut v_TP: Py, mut v_FP: Py, mut v_FN: Py, mut v_TN: Py) -> R {
    let mut v_nm: Py = Py::None;
    let mut v_v: Py = Py::None;
    for t4965 in py::iter(&(Py::tuple(vec![Py::tuple(vec![Py::str("TP"), v_TP.clone()]), Py::tuple(vec![Py::str("FP"), v_FP.clone()]), Py::tuple(vec![Py::str("FN"), v_FN.clone()]), Py::tuple(vec![Py::str("TN"), v_TN.clone()])])))? {
        let t4969 = (t4965).items()?;
        if t4969.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_nm = t4969[0].clone();
        v_v = t4969[1].clone();
        let _ = f__need({ let t4968 = Py::Bool(matches!(v_v.clone(), Py::Int(_) | Py::Big(_) | Py::Bool(_))); if !t4968.truthy() { t4968 } else { { let t4966 = v_v.clone(); Py::Bool({ let t4967 = Py::Int(0i64); if py::ge(&t4966, &t4967)? { true } else { false } }) } } }, Py::str(&[py::format(&v_nm.clone(), "")?, " must be a whole number >= 0".to_string()].concat()))?;
    }
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__classes(mut v_lists: Py) -> R {
    return Ok(py::sorted(&py::method(&py::set(&Py::list(vec![]))?, "union", &({ let mut v: Vec<Py> = vec![]; v.extend(({ let mut out = vec![]; for t6401 in py::iter(&(v_lists.clone()))? { let mut v_v: Py; 
v_v = t6401;
out.push(py::set(&v_v.clone())?); } Py::list(out) }).items()?); v })[..])?, (Py::Bool(false)).truthy())?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__contingency(mut v_a: Py, mut v_b: Py) -> R {
    let mut v_M: Py = Py::None;
    let mut v_ca: Py = Py::None;
    let mut v_cb: Py = Py::None;
    let mut v_ia: Py = Py::None;
    let mut v_ib: Py = Py::None;
    let mut v_u: Py = Py::None;
    let mut v_v: Py = Py::None;
    let t6543 = (Py::tuple(vec![py::sorted(&py::set(&v_a.clone())?, (Py::Bool(false)).truthy())?, py::sorted(&py::set(&v_b.clone())?, (Py::Bool(false)).truthy())?])).items()?;
    if t6543.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_ca = t6543[0].clone();
    v_cb = t6543[1].clone();
    let t6548 = (Py::tuple(vec![{ let mut out = vec![]; for t6544 in py::iter(&(py::enumerate(&v_ca.clone(), 0)?))? { let mut v_i: Py; let mut v_v: Py; 
let t6545 = (t6544).items()?;
if t6545.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_i = t6545[0].clone();
v_v = t6545[1].clone();
out.push((v_v.clone(), v_i.clone())); } Py::dict(out) }, { let mut out = vec![]; for t6546 in py::iter(&(py::enumerate(&v_cb.clone(), 0)?))? { let mut v_i: Py; let mut v_v: Py; 
let t6547 = (t6546).items()?;
if t6547.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_i = t6547[0].clone();
v_v = t6547[1].clone();
out.push((v_v.clone(), v_i.clone())); } Py::dict(out) }])).items()?;
    if t6548.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_ia = t6548[0].clone();
    v_ib = t6548[1].clone();
    v_M = { let mut out = vec![]; for t6549 in py::iter(&(v_ca.clone()))? { let mut v__: Py; 
v__ = t6549;
out.push(py::mul(&Py::list(vec![Py::Int(0i64)]), &py::len(&v_cb.clone())?)?); } Py::list(out) };
    for t6550 in py::iter(&(py::zip(&[v_a.clone(), v_b.clone()])?))? {
        let t6553 = (t6550).items()?;
        if t6553.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_u = t6553[0].clone();
        v_v = t6553[1].clone();
        let t6551 = py::getitem(&v_M.clone(), &py::getitem(&v_ia.clone(), &v_u.clone())?)?; let t6552 = py::getitem(&v_ib.clone(), &v_v.clone())?;
        py::setitem(&t6551, &t6552, py::iadd(&py::getitem(&t6551, &t6552)?, &Py::Int(1i64))?)?;
    }
    return Ok(v_M.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__counts(mut v_yt: Py, mut v_yp: Py) -> R {
    let mut v_fn: Py = Py::None;
    let mut v_fp: Py = Py::None;
    let mut v_tn: Py = Py::None;
    let mut v_tp: Py = Py::None;
    v_tp = py::sum(&{ let mut out = vec![]; for t7743 in py::iter(&(py::zip(&[v_yt.clone(), v_yp.clone()])?))? { let mut v_a: Py; let mut v_b: Py; 
let t7744 = (t7743).items()?;
if t7744.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t7744[0].clone();
v_b = t7744[1].clone();
if ({ let t7749 = { let t7745 = v_a.clone(); Py::Bool({ let t7746 = Py::Int(1i64); if py::eq(&t7745, &t7746) { true } else { false } }) }; if !t7749.truthy() { t7749 } else { { let t7747 = v_b.clone(); Py::Bool({ let t7748 = Py::Int(1i64); if py::eq(&t7747, &t7748) { true } else { false } }) } } }).truthy() { out.push(Py::Int(1i64)); }} Py::list(out) }, None)?;
    v_fp = py::sum(&{ let mut out = vec![]; for t7750 in py::iter(&(py::zip(&[v_yt.clone(), v_yp.clone()])?))? { let mut v_a: Py; let mut v_b: Py; 
let t7751 = (t7750).items()?;
if t7751.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t7751[0].clone();
v_b = t7751[1].clone();
if ({ let t7756 = { let t7752 = v_a.clone(); Py::Bool({ let t7753 = Py::Int(0i64); if py::eq(&t7752, &t7753) { true } else { false } }) }; if !t7756.truthy() { t7756 } else { { let t7754 = v_b.clone(); Py::Bool({ let t7755 = Py::Int(1i64); if py::eq(&t7754, &t7755) { true } else { false } }) } } }).truthy() { out.push(Py::Int(1i64)); }} Py::list(out) }, None)?;
    v_fn = py::sum(&{ let mut out = vec![]; for t7757 in py::iter(&(py::zip(&[v_yt.clone(), v_yp.clone()])?))? { let mut v_a: Py; let mut v_b: Py; 
let t7758 = (t7757).items()?;
if t7758.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t7758[0].clone();
v_b = t7758[1].clone();
if ({ let t7763 = { let t7759 = v_a.clone(); Py::Bool({ let t7760 = Py::Int(1i64); if py::eq(&t7759, &t7760) { true } else { false } }) }; if !t7763.truthy() { t7763 } else { { let t7761 = v_b.clone(); Py::Bool({ let t7762 = Py::Int(0i64); if py::eq(&t7761, &t7762) { true } else { false } }) } } }).truthy() { out.push(Py::Int(1i64)); }} Py::list(out) }, None)?;
    v_tn = py::sum(&{ let mut out = vec![]; for t7764 in py::iter(&(py::zip(&[v_yt.clone(), v_yp.clone()])?))? { let mut v_a: Py; let mut v_b: Py; 
let t7765 = (t7764).items()?;
if t7765.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t7765[0].clone();
v_b = t7765[1].clone();
if ({ let t7770 = { let t7766 = v_a.clone(); Py::Bool({ let t7767 = Py::Int(0i64); if py::eq(&t7766, &t7767) { true } else { false } }) }; if !t7770.truthy() { t7770 } else { { let t7768 = v_b.clone(); Py::Bool({ let t7769 = Py::Int(0i64); if py::eq(&t7768, &t7769) { true } else { false } }) } } }).truthy() { out.push(Py::Int(1i64)); }} Py::list(out) }, None)?;
    return Ok(Py::tuple(vec![v_tp.clone(), v_fp.clone(), v_fn.clone(), v_tn.clone()]));
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__curve(mut v_yt: Py, mut v_s: Py) -> R {
    let mut v_fp: Py = Py::None;
    let mut v_fps: Py = Py::None;
    let mut v_i: Py = Py::None;
    let mut v_pairs: Py = Py::None;
    let mut v_th: Py = Py::None;
    let mut v_tp: Py = Py::None;
    let mut v_tps: Py = Py::None;
    let mut v_v: Py = Py::None;
    v_pairs = py::sorted_key(&py::zip(&[v_s.clone(), v_yt.clone()])?, &{ let v_s = v_s.clone(); let v_yt = v_yt.clone(); py::func(move |a: &[Py]| -> R {
            #[allow(unused_mut, unused_variables, unused_assignments)]
{
            let mut v_t: Py = a.get(0).cloned().unwrap_or(Py::None);
            return Ok(py::neg(&py::getitem(&v_t.clone(), &Py::Int(0i64))?)?);
            #[allow(unreachable_code)]
            Ok(Py::None)
        }
        }) }, (Py::Bool(false)).truthy())?;
    let t7138 = (Py::tuple(vec![Py::list(vec![]), Py::list(vec![]), Py::list(vec![])])).items()?;
    if t7138.len() != 3 { return Err(PyErr::Value("unpack".into())); }
    v_th = t7138[0].clone();
    v_tps = t7138[1].clone();
    v_fps = t7138[2].clone();
    let t7139 = Py::Int(0i64);
    v_tp = t7139.clone();
    v_fp = t7139.clone();
    v_i = Py::Int(0i64);
    while ({ let t7145 = v_i.clone(); Py::Bool({ let t7146 = py::len(&v_pairs.clone())?; if py::lt(&t7145, &t7146)? { true } else { false } }) }).truthy() {
        v_v = py::getitem(&py::getitem(&v_pairs.clone(), &v_i.clone())?, &Py::Int(0i64))?;
        while ({ let t7144 = { let t7140 = v_i.clone(); Py::Bool({ let t7141 = py::len(&v_pairs.clone())?; if py::lt(&t7140, &t7141)? { true } else { false } }) }; if !t7144.truthy() { t7144 } else { { let t7142 = py::getitem(&py::getitem(&v_pairs.clone(), &v_i.clone())?, &Py::Int(0i64))?; Py::Bool({ let t7143 = v_v.clone(); if py::eq(&t7142, &t7143) { true } else { false } }) } } }).truthy() {
            v_tp = py::iadd(&v_tp, &py::getitem(&py::getitem(&v_pairs.clone(), &v_i.clone())?, &Py::Int(1i64))?)?;
            v_fp = py::iadd(&v_fp, &py::sub(&Py::Int(1i64), &py::getitem(&py::getitem(&v_pairs.clone(), &v_i.clone())?, &Py::Int(1i64))?)?)?;
            v_i = py::iadd(&v_i, &Py::Int(1i64))?;
        }
        let _ = Py::tuple(vec![py::method(&v_th.clone(), "append", &[v_v.clone()])?, py::method(&v_tps.clone(), "append", &[v_tp.clone()])?, py::method(&v_fps.clone(), "append", &[v_fp.clone()])?]);
    }
    return Ok(Py::tuple(vec![v_th.clone(), v_tps.clone(), v_fps.clone()]));
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__dact(mut v_name: Py, mut v_z: Py, mut v_a: Py) -> R {
    if ({ let t519 = v_name.clone(); Py::Bool({ let t520 = Py::str("relu"); if py::eq(&t519, &t520) { true } else { false } }) }).truthy() {
        return Ok((if ({ let t521 = v_z.clone(); Py::Bool({ let t522 = Py::Int(0i64); if py::gt(&t521, &t522)? { true } else { false } }) }).truthy() { Py::Float(1.0) } else { Py::Float(0.0) }));
    }
    if ({ let t523 = v_name.clone(); Py::Bool({ let t524 = Py::str("tanh"); if py::eq(&t523, &t524) { true } else { false } }) }).truthy() {
        return Ok(py::sub(&Py::Float(1.0), &py::mul(&v_a.clone(), &v_a.clone())?)?);
    }
    return Ok(py::mul(&v_a.clone(), &py::sub(&Py::Float(1.0), &v_a.clone())?)?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__dist(mut v_name: Py, mut v_v: Py) -> R {
    let mut v_p: Py = Py::None;
    v_p = f__vec(v_name.clone(), v_v.clone(), Py::Int(2i64))?;
    let _ = f__need(Py::Bool(({ let mut out = vec![]; for t3569 in py::iter(&(v_p.clone()))? { let mut v_x: Py; 
v_x = t3569;
out.push({ let t3570 = v_x.clone(); Py::Bool({ let t3571 = Py::Int(0i64); if py::gt(&t3570, &t3571)? { true } else { false } }) }); } Py::list(out) }).items()?.iter().all(|x| x.truthy())), Py::str(&["every probability in ".to_string(), py::format(&v_name.clone(), "")?, " must be positive".to_string()].concat()))?;
    let _ = f__need({ let t3572 = py::abs(&py::sub(&py::fsum(&v_p.clone())?, &Py::Int(1i64))?)?; Py::Bool({ let t3573 = Py::Float(1e-09); if py::le(&t3572, &t3573)? { true } else { false } }) }, Py::str(&[py::format(&v_name.clone(), "")?, " must sum to 1".to_string()].concat()))?;
    return Ok(v_p.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__fin(mut v_name: Py, mut v_vals: Py) -> R {
    let mut v_v: Py = Py::None;
    for t8148 in py::iter(&(v_vals.clone()))? {
        v_v = t8148;
        let _ = f__need({ let t8150 = Py::Bool(matches!(v_v.clone(), Py::Int(_) | Py::Big(_) | Py::Bool(_) | Py::Float(_))); if !t8150.truthy() { t8150 } else { { let t8149 = py::not(&Py::Bool(matches!(v_v.clone(), Py::Bool(_)))); if !t8149.truthy() { t8149 } else { py::math1("isfinite", &v_v.clone())? } } } }, Py::str(&[py::format(&v_name.clone(), "")?, " must be a finite number".to_string()].concat()))?;
    }
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__forward(mut v_xv: Py, mut v_A: Py, mut v_c1: Py, mut v_B: Py, mut v_c2: Py, mut v_activation: Py) -> R {
    let mut v_h: Py = Py::None;
    let mut v_z1: Py = Py::None;
    let mut v_z2: Py = Py::None;
    v_z1 = { let mut out = vec![]; for t2079 in py::iter(&(py::zip(&[v_A.clone(), v_c1.clone()])?))? { let mut v_b: Py; let mut v_r: Py; 
let t2080 = (t2079).items()?;
if t2080.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_r = t2080[0].clone();
v_b = t2080[1].clone();
out.push(py::add(&py::fsum(&{ let mut out = vec![]; for t2081 in py::iter(&(py::zip(&[v_r.clone(), v_xv.clone()])?))? { let mut v_a: Py; let mut v_v: Py; 
let t2082 = (t2081).items()?;
if t2082.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t2082[0].clone();
v_v = t2082[1].clone();
out.push(py::mul(&v_a.clone(), &v_v.clone())?); } Py::list(out) })?, &v_b.clone())?); } Py::list(out) };
    v_h = { let mut out = vec![]; for t2083 in py::iter(&(v_z1.clone()))? { let mut v_v: Py; 
v_v = t2083;
out.push(f__act(v_activation.clone(), v_v.clone())?); } Py::list(out) };
    v_z2 = { let mut out = vec![]; for t2092 in py::iter(&(py::zip(&[v_B.clone(), v_c2.clone()])?))? { let mut v_b: Py; let mut v_r: Py; 
let t2093 = (t2092).items()?;
if t2093.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_r = t2093[0].clone();
v_b = t2093[1].clone();
out.push(py::add(&py::fsum(&{ let mut out = vec![]; for t2094 in py::iter(&(py::zip(&[v_r.clone(), v_h.clone()])?))? { let mut v_a: Py; let mut v_v: Py; 
let t2095 = (t2094).items()?;
if t2095.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t2095[0].clone();
v_v = t2095[1].clone();
out.push(py::mul(&v_a.clone(), &v_v.clone())?); } Py::list(out) })?, &v_b.clone())?); } Py::list(out) };
    return Ok(Py::tuple(vec![v_z1.clone(), v_h.clone(), v_z2.clone()]));
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__gauss(mut v_A: Py, mut v_b: Py) -> R {
    let mut v_M: Py = Py::None;
    let mut v_c: Py = Py::None;
    let mut v_f: Py = Py::None;
    let mut v_k: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_p: Py = Py::None;
    let mut v_r: Py = Py::None;
    let mut v_scale: Py = Py::None;
    let mut v_x: Py = Py::None;
    v_n = py::len(&v_A.clone())?;
    v_M = { let mut out = vec![]; for t4634 in py::iter(&(py::enumerate(&v_A.clone(), 0)?))? { let mut v_i: Py; let mut v_row: Py; 
let t4635 = (t4634).items()?;
if t4635.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_i = t4635[0].clone();
v_row = t4635[1].clone();
out.push(py::add(&Py::list((v_row.clone()).items()?), &Py::list(vec![py::getitem(&v_b.clone(), &v_i.clone())?]))?); } Py::list(out) };
    v_scale = { let t4638 = { let t = { let mut out = vec![]; for t4636 in py::iter(&(v_A.clone()))? { let mut v_row: Py; 
v_row = t4636;
for t4637 in py::iter(&(v_row.clone()))? { let mut v_x: Py; 
v_x = t4637;
out.push(py::abs(&v_x.clone())?); }} Py::list(out) }; if t.items()?.is_empty() { Py::Float(1.0) } else { py::minmax(&[t], true)? } }; if t4638.truthy() { t4638 } else { Py::Float(1.0) } };
    for t4639 in py::range_iter(&[(v_n.clone()).i()?]) {
        v_c = t4639;
        v_p = py::minmax_key(&[Py::list((py::range_iter(&[(v_c.clone()).i()?, (v_n.clone()).i()?])).into_iter().collect::<Vec<Py>>())], &{ let v_A = v_A.clone(); let v_M = v_M.clone(); let v_b = v_b.clone(); let v_c = v_c.clone(); let v_n = v_n.clone(); let v_scale = v_scale.clone(); py::func(move |a: &[Py]| -> R {
            #[allow(unused_mut, unused_variables, unused_assignments)]
{
            let mut v_r: Py = a.get(0).cloned().unwrap_or(Py::None);
            return Ok(py::abs(&py::getitem(&py::getitem(&v_M.clone(), &v_r.clone())?, &v_c.clone())?)?);
            #[allow(unreachable_code)]
            Ok(Py::None)
        }
        }) }, true)?;
        let _ = f__need({ let t4640 = py::abs(&py::getitem(&py::getitem(&v_M.clone(), &v_p.clone())?, &v_c.clone())?)?; Py::Bool({ let t4641 = py::mul(&Py::Float(1e-12), &v_scale.clone())?; if py::gt(&t4640, &t4641)? { true } else { false } }) }, Py::str("the linear system is singular (the columns are linearly dependent)"))?;
        let t4642 = (Py::tuple(vec![py::getitem(&v_M.clone(), &v_p.clone())?, py::getitem(&v_M.clone(), &v_c.clone())?])).items()?;
        if t4642.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        py::setitem(&v_M.clone(), &v_c.clone(), t4642[0].clone())?;
        py::setitem(&v_M.clone(), &v_p.clone(), t4642[1].clone())?;
        for t4643 in py::range_iter(&[(py::add(&v_c.clone(), &Py::Int(1i64))?).i()?, (v_n.clone()).i()?]) {
            v_r = t4643;
            v_f = py::truediv(&py::getitem(&py::getitem(&v_M.clone(), &v_r.clone())?, &v_c.clone())?, &py::getitem(&py::getitem(&v_M.clone(), &v_c.clone())?, &v_c.clone())?)?;
            if ({ let t4644 = v_f.clone(); Py::Bool({ let t4645 = Py::Int(0i64); if py::ne(&t4644, &t4645) { true } else { false } }) }).truthy() {
                for t4646 in py::range_iter(&[(v_c.clone()).i()?, (py::add(&v_n.clone(), &Py::Int(1i64))?).i()?]) {
                    v_k = t4646;
                    let t4647 = py::getitem(&v_M.clone(), &v_r.clone())?; let t4648 = v_k.clone();
                    py::setitem(&t4647, &t4648, py::sub(&py::getitem(&t4647, &t4648)?, &py::mul(&v_f.clone(), &py::getitem(&py::getitem(&v_M.clone(), &v_c.clone())?, &v_k.clone())?)?)?)?;
                }
            }
        }
    }
    v_x = py::mul(&Py::list(vec![Py::Float(0.0)]), &v_n.clone())?;
    for t4649 in py::iter(&(py::reversed(&Py::list((py::range_iter(&[(v_n.clone()).i()?])).into_iter().collect::<Vec<Py>>()))?))? {
        v_r = t4649;
        py::setitem(&v_x.clone(), &v_r.clone(), py::truediv(&py::sub(&py::getitem(&py::getitem(&v_M.clone(), &v_r.clone())?, &v_n.clone())?, &py::fsum(&{ let mut out = vec![]; for t4650 in py::range_iter(&[(py::add(&v_r.clone(), &Py::Int(1i64))?).i()?, (v_n.clone()).i()?]) { let mut v_k: Py; 
v_k = t4650;
out.push(py::mul(&py::getitem(&py::getitem(&v_M.clone(), &v_r.clone())?, &v_k.clone())?, &py::getitem(&v_x.clone(), &v_k.clone())?)?); } Py::list(out) })?)?, &py::getitem(&py::getitem(&v_M.clone(), &v_r.clone())?, &v_r.clone())?)?)?;
    }
    return Ok(v_x.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__jacobi(mut v_A: Py) -> R {
    let mut v_V: Py = Py::None;
    let mut v__: Py = Py::None;
    let mut v_a: Py = Py::None;
    let mut v_akp: Py = Py::None;
    let mut v_akq: Py = Py::None;
    let mut v_apk: Py = Py::None;
    let mut v_aqk: Py = Py::None;
    let mut v_c: Py = Py::None;
    let mut v_k: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_off: Py = Py::None;
    let mut v_p: Py = Py::None;
    let mut v_q: Py = Py::None;
    let mut v_s: Py = Py::None;
    let mut v_scale: Py = Py::None;
    let mut v_t: Py = Py::None;
    let mut v_theta: Py = Py::None;
    let mut v_vkp: Py = Py::None;
    let mut v_vkq: Py = Py::None;
    v_n = py::len(&v_A.clone())?;
    v_a = { let mut out = vec![]; for t6016 in py::iter(&(v_A.clone()))? { let mut v_r: Py; 
v_r = t6016;
out.push(Py::list((v_r.clone()).items()?)); } Py::list(out) };
    v_V = { let mut out = vec![]; for t6017 in py::range_iter(&[(v_n.clone()).i()?]) { let mut v_i: Py; 
v_i = t6017;
out.push({ let mut out = vec![]; for t6018 in py::range_iter(&[(v_n.clone()).i()?]) { let mut v_j: Py; 
v_j = t6018;
out.push((if ({ let t6019 = v_i.clone(); Py::Bool({ let t6020 = v_j.clone(); if py::eq(&t6019, &t6020) { true } else { false } }) }).truthy() { Py::Float(1.0) } else { Py::Float(0.0) })); } Py::list(out) }); } Py::list(out) };
    for t6021 in py::range_iter(&[(Py::Int(100i64)).i()?]) {
        v__ = t6021;
        v_off = py::fsum(&{ let mut out = vec![]; for t6022 in py::range_iter(&[(v_n.clone()).i()?]) { let mut v_i: Py; 
v_i = t6022;
for t6023 in py::range_iter(&[(v_n.clone()).i()?]) { let mut v_j: Py; 
v_j = t6023;
if ({ let t6024 = v_i.clone(); Py::Bool({ let t6025 = v_j.clone(); if py::ne(&t6024, &t6025) { true } else { false } }) }).truthy() { out.push(py::pow(&py::getitem(&py::getitem(&v_a.clone(), &v_i.clone())?, &v_j.clone())?, &Py::Int(2i64))?); }}} Py::list(out) })?;
        v_scale = py::fsum(&{ let mut out = vec![]; for t6026 in py::range_iter(&[(v_n.clone()).i()?]) { let mut v_i: Py; 
v_i = t6026;
out.push(py::pow(&py::getitem(&py::getitem(&v_a.clone(), &v_i.clone())?, &v_i.clone())?, &Py::Int(2i64))?); } Py::list(out) })?;
        if ({ let t6027 = v_off.clone(); Py::Bool({ let t6028 = py::mul(&Py::Float(1e-34), &py::minmax(&[v_scale.clone(), Py::Float(1e-300)], true)?)?; if py::le(&t6027, &t6028)? { true } else { false } }) }).truthy() {
            break;
        }
        for t6029 in py::range_iter(&[(v_n.clone()).i()?]) {
            v_p = t6029;
            for t6030 in py::range_iter(&[(py::add(&v_p.clone(), &Py::Int(1i64))?).i()?, (v_n.clone()).i()?]) {
                v_q = t6030;
                if ({ let t6031 = py::getitem(&py::getitem(&v_a.clone(), &v_p.clone())?, &v_q.clone())?; Py::Bool({ let t6032 = Py::Int(0i64); if py::eq(&t6031, &t6032) { true } else { false } }) }).truthy() {
                    continue;
                }
                v_theta = py::truediv(&py::sub(&py::getitem(&py::getitem(&v_a.clone(), &v_q.clone())?, &v_q.clone())?, &py::getitem(&py::getitem(&v_a.clone(), &v_p.clone())?, &v_p.clone())?)?, &py::mul(&Py::Int(2i64), &py::getitem(&py::getitem(&v_a.clone(), &v_p.clone())?, &v_q.clone())?)?)?;
                v_t = py::truediv(&py::copysign(&Py::Float(1.0), &v_theta.clone())?, &py::add(&py::abs(&v_theta.clone())?, &py::math1("sqrt", &py::add(&py::mul(&v_theta.clone(), &v_theta.clone())?, &Py::Int(1i64))?)?)?)?;
                v_c = py::truediv(&Py::Int(1i64), &py::math1("sqrt", &py::add(&py::mul(&v_t.clone(), &v_t.clone())?, &Py::Int(1i64))?)?)?;
                v_s = py::mul(&v_t.clone(), &v_c.clone())?;
                for t6033 in py::range_iter(&[(v_n.clone()).i()?]) {
                    v_k = t6033;
                    let t6034 = (Py::tuple(vec![py::getitem(&py::getitem(&v_a.clone(), &v_k.clone())?, &v_p.clone())?, py::getitem(&py::getitem(&v_a.clone(), &v_k.clone())?, &v_q.clone())?])).items()?;
                    if t6034.len() != 2 { return Err(PyErr::Value("unpack".into())); }
                    v_akp = t6034[0].clone();
                    v_akq = t6034[1].clone();
                    let t6035 = (Py::tuple(vec![py::sub(&py::mul(&v_c.clone(), &v_akp.clone())?, &py::mul(&v_s.clone(), &v_akq.clone())?)?, py::add(&py::mul(&v_s.clone(), &v_akp.clone())?, &py::mul(&v_c.clone(), &v_akq.clone())?)?])).items()?;
                    if t6035.len() != 2 { return Err(PyErr::Value("unpack".into())); }
                    py::setitem(&py::getitem(&v_a.clone(), &v_k.clone())?, &v_p.clone(), t6035[0].clone())?;
                    py::setitem(&py::getitem(&v_a.clone(), &v_k.clone())?, &v_q.clone(), t6035[1].clone())?;
                }
                for t6036 in py::range_iter(&[(v_n.clone()).i()?]) {
                    v_k = t6036;
                    let t6037 = (Py::tuple(vec![py::getitem(&py::getitem(&v_a.clone(), &v_p.clone())?, &v_k.clone())?, py::getitem(&py::getitem(&v_a.clone(), &v_q.clone())?, &v_k.clone())?])).items()?;
                    if t6037.len() != 2 { return Err(PyErr::Value("unpack".into())); }
                    v_apk = t6037[0].clone();
                    v_aqk = t6037[1].clone();
                    let t6038 = (Py::tuple(vec![py::sub(&py::mul(&v_c.clone(), &v_apk.clone())?, &py::mul(&v_s.clone(), &v_aqk.clone())?)?, py::add(&py::mul(&v_s.clone(), &v_apk.clone())?, &py::mul(&v_c.clone(), &v_aqk.clone())?)?])).items()?;
                    if t6038.len() != 2 { return Err(PyErr::Value("unpack".into())); }
                    py::setitem(&py::getitem(&v_a.clone(), &v_p.clone())?, &v_k.clone(), t6038[0].clone())?;
                    py::setitem(&py::getitem(&v_a.clone(), &v_q.clone())?, &v_k.clone(), t6038[1].clone())?;
                }
                for t6039 in py::range_iter(&[(v_n.clone()).i()?]) {
                    v_k = t6039;
                    let t6040 = (Py::tuple(vec![py::getitem(&py::getitem(&v_V.clone(), &v_k.clone())?, &v_p.clone())?, py::getitem(&py::getitem(&v_V.clone(), &v_k.clone())?, &v_q.clone())?])).items()?;
                    if t6040.len() != 2 { return Err(PyErr::Value("unpack".into())); }
                    v_vkp = t6040[0].clone();
                    v_vkq = t6040[1].clone();
                    let t6041 = (Py::tuple(vec![py::sub(&py::mul(&v_c.clone(), &v_vkp.clone())?, &py::mul(&v_s.clone(), &v_vkq.clone())?)?, py::add(&py::mul(&v_s.clone(), &v_vkp.clone())?, &py::mul(&v_c.clone(), &v_vkq.clone())?)?])).items()?;
                    if t6041.len() != 2 { return Err(PyErr::Value("unpack".into())); }
                    py::setitem(&py::getitem(&v_V.clone(), &v_k.clone())?, &v_p.clone(), t6041[0].clone())?;
                    py::setitem(&py::getitem(&v_V.clone(), &v_k.clone())?, &v_q.clone(), t6041[1].clone())?;
                }
            }
        }
    }
    return Ok(Py::tuple(vec![{ let mut out = vec![]; for t6042 in py::range_iter(&[(v_n.clone()).i()?]) { let mut v_i: Py; 
v_i = t6042;
out.push(py::getitem(&py::getitem(&v_a.clone(), &v_i.clone())?, &v_i.clone())?); } Py::list(out) }, v_V.clone()]));
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__km_inputs(mut v_X: Py, mut v_C: Py) -> R {
    let mut v_C0: Py = Py::None;
    let mut v_M: Py = Py::None;
    let t3731 = (Py::tuple(vec![f__mat(Py::str("X"), v_X.clone(), Py::Int(1i64), Py::Int(1i64))?, f__mat(Py::str("C"), v_C.clone(), Py::Int(1i64), Py::Int(1i64))?])).items()?;
    if t3731.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_M = t3731[0].clone();
    v_C0 = t3731[1].clone();
    let _ = f__need({ let t3732 = py::len(&py::getitem(&v_C0.clone(), &Py::Int(0i64))?)?; Py::Bool({ let t3733 = py::len(&py::getitem(&v_M.clone(), &Py::Int(0i64))?)?; if py::eq(&t3732, &t3733) { true } else { false } }) }, Py::str("centroids and points must have the same number of coordinates"))?;
    let _ = f__need({ let t3734 = Py::Int(2i64); Py::Bool({ let t3735 = py::len(&v_C0.clone())?; if py::le(&t3734, &t3735)? { { let t3736 = py::len(&v_M.clone())?; if py::le(&t3735, &t3736)? { true } else { false } } } else { false } }) }, Py::str("need at least 2 centroids and no more centroids than points"))?;
    return Ok(Py::tuple(vec![{ let mut out = vec![]; for t3737 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t3737;
out.push({ let mut out = vec![]; for t3738 in py::iter(&(v_r.clone()))? { let mut v_v: Py; 
v_v = t3738;
out.push(f__F(v_v.clone())?); } Py::list(out) }); } Py::list(out) }, { let mut out = vec![]; for t3739 in py::iter(&(v_C0.clone()))? { let mut v_r: Py; 
v_r = t3739;
out.push({ let mut out = vec![]; for t3740 in py::iter(&(v_r.clone()))? { let mut v_v: Py; 
v_v = t3740;
out.push(f__F(v_v.clone())?); } Py::list(out) }); } Py::list(out) }]));
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__knn_core(mut v_X: Py, mut v_y: Py, mut v_x: Py, mut v_k: Py, mut v_metric: Py) -> R {
    let mut v_M: Py = Py::None;
    let mut v_Q: Py = Py::None;
    let mut v_d: Py = Py::None;
    let mut v_dist: Py = Py::None;
    let mut v_nb: Py = Py::None;
    let mut v_order: Py = Py::None;
    let mut v_q: Py = Py::None;
    let mut v_yv: Py = Py::None;
    v_M = f__mat(Py::str("X"), v_X.clone(), Py::Int(1i64), Py::Int(1i64))?;
    v_yv = f__vec(Py::str("y"), v_y.clone(), Py::Int(1i64))?;
    v_q = f__vec(Py::str("x"), v_x.clone(), Py::Int(1i64))?;
    let _ = f__need({ let t3923 = py::len(&v_yv.clone())?; Py::Bool({ let t3924 = py::len(&v_M.clone())?; if py::eq(&t3923, &t3924) { true } else { false } }) }, Py::str("y needs one value per row of X"))?;
    let _ = f__need({ let t3925 = py::len(&v_q.clone())?; Py::Bool({ let t3926 = py::len(&py::getitem(&v_M.clone(), &Py::Int(0i64))?)?; if py::eq(&t3925, &t3926) { true } else { false } }) }, Py::str("the query x needs one coordinate per column of X"))?;
    let _ = f__need({ let t3930 = Py::Bool(matches!(v_k.clone(), Py::Int(_) | Py::Big(_) | Py::Bool(_))); if !t3930.truthy() { t3930 } else { { let t3927 = Py::Int(1i64); Py::Bool({ let t3928 = v_k.clone(); if py::le(&t3927, &t3928)? { { let t3929 = py::len(&v_M.clone())?; if py::le(&t3928, &t3929)? { true } else { false } } } else { false } }) } } }, Py::str("k must be between 1 and the number of training points"))?;
    v_Q = { let mut out = vec![]; for t3931 in py::iter(&(v_q.clone()))? { let mut v_v: Py; 
v_v = t3931;
out.push(f__F(v_v.clone())?); } Py::list(out) };
    if ({ let t3932 = v_metric.clone(); Py::Bool({ let t3933 = Py::str("euclidean"); if py::eq(&t3932, &t3933) { true } else { false } }) }).truthy() {
        v_d = { let mut out = vec![]; for t3934 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t3934;
out.push(py::sum(&{ let mut out = vec![]; for t3935 in py::iter(&(py::zip(&[v_r.clone(), v_Q.clone()])?))? { let mut v_a: Py; let mut v_b: Py; 
let t3936 = (t3935).items()?;
if t3936.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t3936[0].clone();
v_b = t3936[1].clone();
out.push(py::pow(&py::sub(&f__F(v_a.clone())?, &v_b.clone())?, &Py::Int(2i64))?); } Py::list(out) }, None)?); } Py::list(out) };
    } else {
        v_d = { let mut out = vec![]; for t3937 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t3937;
out.push(py::sum(&{ let mut out = vec![]; for t3938 in py::iter(&(py::zip(&[v_r.clone(), v_Q.clone()])?))? { let mut v_a: Py; let mut v_b: Py; 
let t3939 = (t3938).items()?;
if t3939.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t3939[0].clone();
v_b = t3939[1].clone();
out.push(py::abs(&py::sub(&f__F(v_a.clone())?, &v_b.clone())?)?); } Py::list(out) }, None)?); } Py::list(out) };
    }
    v_order = py::sorted_key(&Py::list((py::range_iter(&[(py::len(&v_M.clone())?).i()?])).into_iter().collect::<Vec<Py>>()), &{ let v_M = v_M.clone(); let v_Q = v_Q.clone(); let v_X = v_X.clone(); let v_d = v_d.clone(); let v_k = v_k.clone(); let v_metric = v_metric.clone(); let v_q = v_q.clone(); let v_x = v_x.clone(); let v_y = v_y.clone(); let v_yv = v_yv.clone(); py::func(move |a: &[Py]| -> R {
            #[allow(unused_mut, unused_variables, unused_assignments)]
{
            let mut v_i: Py = a.get(0).cloned().unwrap_or(Py::None);
            return Ok(Py::tuple(vec![py::getitem(&v_d.clone(), &v_i.clone())?, v_i.clone()]));
            #[allow(unreachable_code)]
            Ok(Py::None)
        }
        }) }, (Py::Bool(false)).truthy())?;
    if ({ let t3940 = v_k.clone(); Py::Bool({ let t3941 = py::len(&v_M.clone())?; if py::lt(&t3940, &t3941)? { true } else { false } }) }).truthy() {
        let _ = f__need({ let t3942 = py::getitem(&v_d.clone(), &py::getitem(&v_order.clone(), &py::sub(&v_k.clone(), &Py::Int(1i64))?)?)?; Py::Bool({ let t3943 = py::getitem(&v_d.clone(), &py::getitem(&v_order.clone(), &v_k.clone())?)?; if py::ne(&t3942, &t3943) { true } else { false } }) }, Py::str("the k-th and (k+1)-th nearest neighbours are at the same distance (the neighbour set is ambiguous)"))?;
    }
    v_nb = py::slice(&v_order.clone(), None, Some((v_k.clone()).i()?), None)?;
    v_dist = { let mut out = vec![]; for t3944 in py::iter(&(v_nb.clone()))? { let mut v_i: Py; 
v_i = t3944;
out.push((if ({ let t3945 = v_metric.clone(); Py::Bool({ let t3946 = Py::str("euclidean"); if py::eq(&t3945, &t3946) { true } else { false } }) }).truthy() { py::math1("sqrt", &py::getitem(&v_d.clone(), &v_i.clone())?)? } else { py::float(&py::getitem(&v_d.clone(), &v_i.clone())?)? })); } Py::list(out) };
    return Ok(Py::tuple(vec![v_nb.clone(), v_dist.clone(), v_yv.clone(), v_d.clone()]));
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__labels(mut v_name: Py, mut v_v: Py, mut v_lo: Py) -> R {
    let _ = f__need({ let t7819 = Py::Bool(matches!(v_v.clone(), Py::List(_) | Py::Tuple(_))); if !t7819.truthy() { t7819 } else { { let t7817 = py::len(&v_v.clone())?; Py::Bool({ let t7818 = v_lo.clone(); if py::ge(&t7817, &t7818)? { true } else { false } }) } } }, Py::str(&[py::format(&v_name.clone(), "")?, " needs at least ".to_string(), py::format(&v_lo.clone(), "")?, " labels".to_string()].concat()))?;
    return Ok({ let mut out = vec![]; for t7820 in py::iter(&(v_v.clone()))? { let mut v_x: Py; 
v_x = t7820;
out.push(f__whole(Py::str(&["every label in ".to_string(), py::format(&v_name.clone(), "")?].concat()), v_x.clone())?); } Py::list(out) });
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__lin_loss(mut v_M: Py, mut v_yv: Py, mut v_wv: Py, mut v_b: Py) -> R {
    let mut v_Bf: Py = Py::None;
    let mut v_Mf: Py = Py::None;
    let mut v_Wf: Py = Py::None;
    let mut v_Yf: Py = Py::None;
    let mut v_gb: Py = Py::None;
    let mut v_gw: Py = Py::None;
    let mut v_loss: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_res: Py = Py::None;
    v_Mf = { let mut out = vec![]; for t5312 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t5312;
out.push({ let mut out = vec![]; for t5313 in py::iter(&(v_r.clone()))? { let mut v_v: Py; 
v_v = t5313;
out.push(f__F(v_v.clone())?); } Py::list(out) }); } Py::list(out) };
    let t5316 = (Py::tuple(vec![{ let mut out = vec![]; for t5314 in py::iter(&(v_wv.clone()))? { let mut v_v: Py; 
v_v = t5314;
out.push(f__F(v_v.clone())?); } Py::list(out) }, f__F(v_b.clone())?, { let mut out = vec![]; for t5315 in py::iter(&(v_yv.clone()))? { let mut v_v: Py; 
v_v = t5315;
out.push(f__F(v_v.clone())?); } Py::list(out) }])).items()?;
    if t5316.len() != 3 { return Err(PyErr::Value("unpack".into())); }
    v_Wf = t5316[0].clone();
    v_Bf = t5316[1].clone();
    v_Yf = t5316[2].clone();
    v_res = { let mut out = vec![]; for t5317 in py::iter(&(py::zip(&[v_Mf.clone(), v_Yf.clone()])?))? { let mut v_r: Py; let mut v_t: Py; 
let t5318 = (t5317).items()?;
if t5318.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_r = t5318[0].clone();
v_t = t5318[1].clone();
out.push(py::sub(&py::add(&py::sum(&{ let mut out = vec![]; for t5319 in py::iter(&(py::zip(&[v_r.clone(), v_Wf.clone()])?))? { let mut v_a: Py; let mut v_c: Py; 
let t5320 = (t5319).items()?;
if t5320.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t5320[0].clone();
v_c = t5320[1].clone();
out.push(py::mul(&v_a.clone(), &v_c.clone())?); } Py::list(out) }, None)?, &v_Bf.clone())?, &v_t.clone())?); } Py::list(out) };
    v_n = py::len(&v_res.clone())?;
    v_loss = py::truediv(&py::sum(&{ let mut out = vec![]; for t5321 in py::iter(&(v_res.clone()))? { let mut v_v: Py; 
v_v = t5321;
out.push(py::mul(&v_v.clone(), &v_v.clone())?); } Py::list(out) }, None)?, &v_n.clone())?;
    v_gw = { let mut out = vec![]; for t5322 in py::range_iter(&[(py::len(&v_Wf.clone())?).i()?]) { let mut v_j: Py; 
v_j = t5322;
out.push(py::truediv(&py::mul(&Py::Int(2i64), &py::sum(&{ let mut out = vec![]; for t5323 in py::range_iter(&[(v_n.clone()).i()?]) { let mut v_i: Py; 
v_i = t5323;
out.push(py::mul(&py::getitem(&v_res.clone(), &v_i.clone())?, &py::getitem(&py::getitem(&v_Mf.clone(), &v_i.clone())?, &v_j.clone())?)?); } Py::list(out) }, None)?)?, &v_n.clone())?); } Py::list(out) };
    v_gb = py::truediv(&py::mul(&Py::Int(2i64), &py::sum(&v_res.clone(), None)?)?, &v_n.clone())?;
    return Ok(Py::tuple(vec![v_loss.clone(), v_gw.clone(), v_gb.clone()]));
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__linear_parts(mut v_X: Py, mut v_y: Py, mut v_w: Py, mut v_b: Py) -> R {
    let mut v_M: Py = Py::None;
    let mut v_wv: Py = Py::None;
    let mut v_yv: Py = Py::None;
    v_M = f__mat(Py::str("X"), v_X.clone(), Py::Int(1i64), Py::Int(1i64))?;
    let t5306 = (Py::tuple(vec![f__vec(Py::str("y"), v_y.clone(), Py::Int(1i64))?, f__vec(Py::str("w"), v_w.clone(), Py::Int(1i64))?])).items()?;
    if t5306.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_yv = t5306[0].clone();
    v_wv = t5306[1].clone();
    let _ = f__need({ let t5307 = py::len(&v_yv.clone())?; Py::Bool({ let t5308 = py::len(&v_M.clone())?; if py::eq(&t5307, &t5308) { true } else { false } }) }, Py::str("y needs one value per row of X"))?;
    let _ = f__need({ let t5309 = py::len(&v_wv.clone())?; Py::Bool({ let t5310 = py::len(&py::getitem(&v_M.clone(), &Py::Int(0i64))?)?; if py::eq(&t5309, &t5310) { true } else { false } }) }, Py::str("w needs one weight per column of X"))?;
    let _ = f__fin(Py::str("b"), Py::tuple(vec![v_b.clone()]))?;
    return Ok(Py::tuple(vec![v_M.clone(), v_yv.clone(), v_wv.clone()]));
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__log_sigmoid(mut v_z: Py) -> R {
    return Ok((if ({ let t7363 = v_z.clone(); Py::Bool({ let t7364 = Py::Int(0i64); if py::ge(&t7363, &t7364)? { true } else { false } }) }).truthy() { py::neg(&py::math1("log1p", &py::math1("exp", &py::neg(&v_z.clone())?)?)?)? } else { py::sub(&v_z.clone(), &py::math1("log1p", &py::math1("exp", &v_z.clone())?)?)? }));
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__log_softmax(mut v_z: Py) -> R {
    let mut v_m: Py = Py::None;
    let mut v_s: Py = Py::None;
    v_m = py::minmax(&[v_z.clone()], true)?;
    v_s = py::math1("log", &py::fsum(&{ let mut out = vec![]; for t4013 in py::iter(&(v_z.clone()))? { let mut v_v: Py; 
v_v = t4013;
out.push(py::math1("exp", &py::sub(&v_v.clone(), &v_m.clone())?)?); } Py::list(out) })?)?;
    return Ok({ let mut out = vec![]; for t4014 in py::iter(&(v_z.clone()))? { let mut v_v: Py; 
v_v = t4014;
out.push(py::sub(&py::sub(&v_v.clone(), &v_m.clone())?, &v_s.clone())?); } Py::list(out) });
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__logit_obj(mut v_M: Py, mut v_yv: Py, mut v_w: Py, mut v_b: Py, mut v_C: Py) -> R {
    let mut v_z: Py = Py::None;
    v_z = { let mut out = vec![]; for t4684 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t4684;
out.push(py::add(&py::fsum(&{ let mut out = vec![]; for t4685 in py::iter(&(py::zip(&[v_r.clone(), v_w.clone()])?))? { let mut v_a: Py; let mut v_c: Py; 
let t4686 = (t4685).items()?;
if t4686.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t4686[0].clone();
v_c = t4686[1].clone();
out.push(py::mul(&v_a.clone(), &v_c.clone())?); } Py::list(out) })?, &v_b.clone())?); } Py::list(out) };
    return Ok(py::add(&py::mul(&Py::Float(0.5), &py::fsum(&{ let mut out = vec![]; for t4687 in py::iter(&(v_w.clone()))? { let mut v_c: Py; 
v_c = t4687;
out.push(py::mul(&v_c.clone(), &v_c.clone())?); } Py::list(out) })?)?, &py::mul(&v_C.clone(), &py::fsum(&{ let mut out = vec![]; for t4688 in py::iter(&(py::zip(&[v_z.clone(), v_yv.clone()])?))? { let mut v_t: Py; let mut v_v: Py; 
let t4689 = (t4688).items()?;
if t4689.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_v = t4689[0].clone();
v_t = t4689[1].clone();
out.push(py::add(&py::sub(&py::minmax(&[v_v.clone(), Py::Float(0.0)], true)?, &py::mul(&v_v.clone(), &v_t.clone())?)?, &py::math1("log1p", &py::math1("exp", &py::neg(&py::abs(&v_v.clone())?)?)?)?)?); } Py::list(out) })?)?)?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__logsumexp(mut v_z: Py) -> R {
    let mut v_m: Py = Py::None;
    v_m = py::minmax(&[v_z.clone()], true)?;
    return Ok(py::add(&v_m.clone(), &py::math1("log", &py::fsum(&{ let mut out = vec![]; for t7506 in py::iter(&(v_z.clone()))? { let mut v_v: Py; 
v_v = t7506;
out.push(py::math1("exp", &py::sub(&v_v.clone(), &v_m.clone())?)?); } Py::list(out) })?)?)?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__mat(mut v_name: Py, mut v_M: Py, mut v_rows: Py, mut v_cols: Py) -> R {
    let mut v_r: Py = Py::None;
    let mut v_x: Py = Py::None;
    let _ = f__need({ let t7828 = Py::Bool(matches!(v_M.clone(), Py::List(_) | Py::Tuple(_))); if !t7828.truthy() { t7828 } else { { let t7826 = py::len(&v_M.clone())?; Py::Bool({ let t7827 = v_rows.clone(); if py::ge(&t7826, &t7827)? { true } else { false } }) } } }, Py::str(&[py::format(&v_name.clone(), "")?, " needs at least ".to_string(), py::format(&v_rows.clone(), "")?, " rows".to_string()].concat()))?;
    let _ = f__need({ let t7832 = Py::Bool(({ let mut out = vec![]; for t7829 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t7829;
out.push(Py::Bool(matches!(v_r.clone(), Py::List(_) | Py::Tuple(_)))); } Py::list(out) }).items()?.iter().all(|x| x.truthy())); if !t7832.truthy() { t7832 } else { { let t7830 = py::len(&py::getitem(&v_M.clone(), &Py::Int(0i64))?)?; Py::Bool({ let t7831 = v_cols.clone(); if py::ge(&t7830, &t7831)? { true } else { false } }) } } }, Py::str(&[py::format(&v_name.clone(), "")?, " needs at least ".to_string(), py::format(&v_cols.clone(), "")?, " columns".to_string()].concat()))?;
    let _ = f__need(Py::Bool(({ let mut out = vec![]; for t7833 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t7833;
out.push({ let t7834 = py::len(&v_r.clone())?; Py::Bool({ let t7835 = py::len(&py::getitem(&v_M.clone(), &Py::Int(0i64))?)?; if py::eq(&t7834, &t7835) { true } else { false } }) }); } Py::list(out) }).items()?.iter().all(|x| x.truthy())), Py::str(&[py::format(&v_name.clone(), "")?, " must be rectangular".to_string()].concat()))?;
    for t7836 in py::iter(&(v_M.clone()))? {
        v_r = t7836;
        for t7837 in py::iter(&(v_r.clone()))? {
            v_x = t7837;
            let _ = f__fin(Py::str(&["every entry of ".to_string(), py::format(&v_name.clone(), "")?].concat()), Py::tuple(vec![v_x.clone()]))?;
        }
    }
    return Ok({ let mut out = vec![]; for t7838 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t7838;
out.push({ let mut out = vec![]; for t7839 in py::iter(&(v_r.clone()))? { let mut v_x: Py; 
v_x = t7839;
out.push(py::float(&v_x.clone())?); } Py::list(out) }); } Py::list(out) });
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__median(mut v_vals: Py) -> R {
    let mut v_n: Py = Py::None;
    let mut v_s: Py = Py::None;
    v_s = py::sorted(&v_vals.clone(), (Py::Bool(false)).truthy())?;
    v_n = py::len(&v_s.clone())?;
    return Ok((if (py::modulo(&v_n.clone(), &Py::Int(2i64))?).truthy() { py::getitem(&v_s.clone(), &py::floordiv(&v_n.clone(), &Py::Int(2i64))?)? } else { py::truediv(&py::add(&py::getitem(&v_s.clone(), &py::sub(&py::floordiv(&v_n.clone(), &Py::Int(2i64))?, &Py::Int(1i64))?)?, &py::getitem(&v_s.clone(), &py::floordiv(&v_n.clone(), &Py::Int(2i64))?)?)?, &Py::Int(2i64))? }));
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__mi_counts(mut v_M: Py, mut v_n: Py) -> R {
    let mut v_cols: Py = Py::None;
    let mut v_rows: Py = Py::None;
    v_rows = { let mut out = vec![]; for t5867 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t5867;
out.push(py::sum(&v_r.clone(), None)?); } Py::list(out) };
    v_cols = { let mut out = vec![]; for t5868 in py::range_iter(&[(py::len(&py::getitem(&v_M.clone(), &Py::Int(0i64))?)?).i()?]) { let mut v_j: Py; 
v_j = t5868;
out.push(py::sum(&{ let mut out = vec![]; for t5869 in py::range_iter(&[(py::len(&v_M.clone())?).i()?]) { let mut v_i: Py; 
v_i = t5869;
out.push(py::getitem(&py::getitem(&v_M.clone(), &v_i.clone())?, &v_j.clone())?); } Py::list(out) }, None)?); } Py::list(out) };
    return Ok(py::fsum(&{ let mut out = vec![]; for t5870 in py::range_iter(&[(py::len(&v_M.clone())?).i()?]) { let mut v_i: Py; 
v_i = t5870;
for t5871 in py::range_iter(&[(py::len(&py::getitem(&v_M.clone(), &Py::Int(0i64))?)?).i()?]) { let mut v_j: Py; 
v_j = t5871;
if ({ let t5872 = py::getitem(&py::getitem(&v_M.clone(), &v_i.clone())?, &v_j.clone())?; Py::Bool({ let t5873 = Py::Int(0i64); if py::gt(&t5872, &t5873)? { true } else { false } }) }).truthy() { out.push(py::mul(&py::truediv(&py::getitem(&py::getitem(&v_M.clone(), &v_i.clone())?, &v_j.clone())?, &v_n.clone())?, &py::math1("log", &py::truediv(&py::mul(&v_n.clone(), &py::getitem(&py::getitem(&v_M.clone(), &v_i.clone())?, &v_j.clone())?)?, &py::mul(&py::getitem(&v_rows.clone(), &v_i.clone())?, &py::getitem(&v_cols.clone(), &v_j.clone())?)?)?)?)?); }}} Py::list(out) })?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__nb_data(mut v_X: Py, mut v_y: Py, mut v_x: Py) -> R {
    let mut v_M: Py = Py::None;
    let mut v_cls: Py = Py::None;
    let mut v_q: Py = Py::None;
    let mut v_yl: Py = Py::None;
    v_M = f__mat(Py::str("X"), v_X.clone(), Py::Int(1i64), Py::Int(1i64))?;
    v_yl = f__labels(Py::str("y"), v_y.clone(), Py::Int(1i64))?;
    v_q = f__vec(Py::str("x"), v_x.clone(), Py::Int(1i64))?;
    let _ = f__need({ let t5632 = py::len(&v_yl.clone())?; Py::Bool({ let t5633 = py::len(&v_M.clone())?; if py::eq(&t5632, &t5633) { true } else { false } }) }, Py::str("y needs one label per row of X"))?;
    let _ = f__need({ let t5634 = py::len(&v_q.clone())?; Py::Bool({ let t5635 = py::len(&py::getitem(&v_M.clone(), &Py::Int(0i64))?)?; if py::eq(&t5634, &t5635) { true } else { false } }) }, Py::str("x needs one value per column of X"))?;
    v_cls = py::sorted(&py::set(&v_yl.clone())?, (Py::Bool(false)).truthy())?;
    let _ = f__need({ let t5636 = py::len(&v_cls.clone())?; Py::Bool({ let t5637 = Py::Int(2i64); if py::ge(&t5636, &t5637)? { true } else { false } }) }, Py::str("naive Bayes needs at least two classes"))?;
    return Ok(Py::tuple(vec![v_M.clone(), v_yl.clone(), v_q.clone(), v_cls.clone()]));
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__need(mut v_ok: Py, mut v_msg: Py) -> R {
    if (py::not(&v_ok.clone())).truthy() {
        return Err(py::value_error(py::to_str(&v_msg.clone())));
    }
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__net_inputs(mut v_x: Py, mut v_W1: Py, mut v_b1: Py, mut v_W2: Py, mut v_b2: Py) -> R {
    let mut v_A: Py = Py::None;
    let mut v_B: Py = Py::None;
    let mut v_c1: Py = Py::None;
    let mut v_c2: Py = Py::None;
    let mut v_xv: Py = Py::None;
    v_xv = f__vec(Py::str("x"), v_x.clone(), Py::Int(1i64))?;
    let t2068 = (Py::tuple(vec![f__mat(Py::str("W1"), v_W1.clone(), Py::Int(1i64), Py::Int(1i64))?, f__mat(Py::str("W2"), v_W2.clone(), Py::Int(1i64), Py::Int(1i64))?])).items()?;
    if t2068.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_A = t2068[0].clone();
    v_B = t2068[1].clone();
    let t2069 = (Py::tuple(vec![f__vec(Py::str("b1"), v_b1.clone(), Py::Int(1i64))?, f__vec(Py::str("b2"), v_b2.clone(), Py::Int(1i64))?])).items()?;
    if t2069.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_c1 = t2069[0].clone();
    v_c2 = t2069[1].clone();
    let _ = f__need({ let t2070 = py::len(&py::getitem(&v_A.clone(), &Py::Int(0i64))?)?; Py::Bool({ let t2071 = py::len(&v_xv.clone())?; if py::eq(&t2070, &t2071) { true } else { false } }) }, Py::str("W1 needs one column per input (shape hidden x inputs)"))?;
    let _ = f__need({ let t2072 = py::len(&v_c1.clone())?; Py::Bool({ let t2073 = py::len(&v_A.clone())?; if py::eq(&t2072, &t2073) { true } else { false } }) }, Py::str("b1 needs one entry per row of W1"))?;
    let _ = f__need({ let t2074 = py::len(&py::getitem(&v_B.clone(), &Py::Int(0i64))?)?; Py::Bool({ let t2075 = py::len(&v_A.clone())?; if py::eq(&t2074, &t2075) { true } else { false } }) }, Py::str("W2 needs one column per hidden unit (shape outputs x hidden)"))?;
    let _ = f__need({ let t2076 = py::len(&v_c2.clone())?; Py::Bool({ let t2077 = py::len(&v_B.clone())?; if py::eq(&t2076, &t2077) { true } else { false } }) }, Py::str("b2 needs one entry per row of W2"))?;
    return Ok(Py::tuple(vec![v_xv.clone(), v_A.clone(), v_c1.clone(), v_B.clone(), v_c2.clone()]));
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__ols_exact(mut v_rows: Py, mut v_ys: Py) -> R {
    let mut v_A: Py = Py::None;
    let mut v_G: Py = Py::None;
    let mut v_beta: Py = Py::None;
    let mut v_d: Py = Py::None;
    let mut v_h: Py = Py::None;
    v_d = py::len(&py::getitem(&v_rows.clone(), &Py::Int(0i64))?)?;
    v_A = { let mut out = vec![]; for t4850 in py::iter(&(v_rows.clone()))? { let mut v_r: Py; 
v_r = t4850;
out.push(py::add(&Py::list(vec![py::fraction(&[Py::Int(1i64)])?]), &{ let mut out = vec![]; for t4851 in py::iter(&(v_r.clone()))? { let mut v_v: Py; 
v_v = t4851;
out.push(f__F(v_v.clone())?); } Py::list(out) })?); } Py::list(out) };
    v_G = { let mut out = vec![]; for t4852 in py::range_iter(&[(py::add(&v_d.clone(), &Py::Int(1i64))?).i()?]) { let mut v_i: Py; 
v_i = t4852;
out.push({ let mut out = vec![]; for t4853 in py::range_iter(&[(py::add(&v_d.clone(), &Py::Int(1i64))?).i()?]) { let mut v_j: Py; 
v_j = t4853;
out.push(py::sum(&{ let mut out = vec![]; for t4854 in py::iter(&(v_A.clone()))? { let mut v_a: Py; 
v_a = t4854;
out.push(py::mul(&py::getitem(&v_a.clone(), &v_i.clone())?, &py::getitem(&v_a.clone(), &v_j.clone())?)?); } Py::list(out) }, None)?); } Py::list(out) }); } Py::list(out) };
    v_h = { let mut out = vec![]; for t4855 in py::range_iter(&[(py::add(&v_d.clone(), &Py::Int(1i64))?).i()?]) { let mut v_i: Py; 
v_i = t4855;
out.push(py::sum(&{ let mut out = vec![]; for t4856 in py::iter(&(py::zip(&[v_A.clone(), v_ys.clone()])?))? { let mut v_a: Py; let mut v_y: Py; 
let t4857 = (t4856).items()?;
if t4857.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t4857[0].clone();
v_y = t4857[1].clone();
out.push(py::mul(&py::getitem(&v_a.clone(), &v_i.clone())?, &f__F(v_y.clone())?)?); } Py::list(out) }, None)?); } Py::list(out) };
    v_beta = f__solve_exact(v_G.clone(), v_h.clone())?;
    return Ok(Py::tuple(vec![py::slice(&v_beta.clone(), Some((Py::Int(1i64)).i()?), None, None)?, py::getitem(&v_beta.clone(), &Py::Int(0i64))?]));
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__opt_vecs(mut v_named: Py) -> R {
    let mut v_n: Py = Py::None;
    let mut v_vals: Py = Py::None;
    v_vals = { let mut out = vec![]; for t7269 in py::iter(&(py::method(&v_named.clone(), "items", &[])?))? { let mut v_k: Py; let mut v_v: Py; 
let t7270 = (t7269).items()?;
if t7270.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_k = t7270[0].clone();
v_v = t7270[1].clone();
out.push((v_k.clone(), f__vec(v_k.clone(), v_v.clone(), Py::Int(1i64))?)); } Py::dict(out) };
    v_n = py::set(&{ let mut out = vec![]; for t7279 in py::iter(&(py::method(&v_vals.clone(), "values", &[])?))? { let mut v_v: Py; 
v_v = t7279;
out.push(py::len(&v_v.clone())?); } Py::list(out) })?;
    let _ = f__need({ let t7280 = py::len(&v_n.clone())?; Py::Bool({ let t7281 = Py::Int(1i64); if py::eq(&t7280, &t7281) { true } else { false } }) }, py::add(&py::add(&Py::str("the vectors ("), &py::method(&Py::str(", "), "join", &[v_named.clone()])?)?, &Py::str(") must have the same length"))?)?;
    return Ok(v_vals.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__out(mut v_kw: Py) -> R {
    return Ok({ let mut out = vec![]; for t8173 in py::iter(&(py::method(&v_kw.clone(), "items", &[])?))? { let mut v_k: Py; let mut v_v: Py; 
let t8174 = (t8173).items()?;
if t8174.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_k = t8174[0].clone();
v_v = t8174[1].clone();
out.push((v_k.clone(), f__r(v_v.clone())?)); } Py::dict(out) });
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__pair(mut v_y_true: Py, mut v_y_pred: Py, mut v_lo: Py) -> R {
    let mut v_yp: Py = Py::None;
    let mut v_yt: Py = Py::None;
    let t8072 = (Py::tuple(vec![f__vec(Py::str("y_true"), v_y_true.clone(), v_lo.clone())?, f__vec(Py::str("y_pred"), v_y_pred.clone(), v_lo.clone())?])).items()?;
    if t8072.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_yt = t8072[0].clone();
    v_yp = t8072[1].clone();
    let _ = f__same_len(v_yt.clone(), v_yp.clone(), Py::str("y_true"), Py::str("y_pred"))?;
    return Ok(Py::tuple(vec![v_yt.clone(), v_yp.clone()]));
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__pairs(mut v_a: Py, mut v_b: Py) -> R {
    let mut v_M: Py = Py::None;
    let mut v_c2: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_sa: Py = Py::None;
    let mut v_sb: Py = Py::None;
    let mut v_sij: Py = Py::None;
    v_M = f__contingency(v_a.clone(), v_b.clone())?;
    v_n = py::len(&v_a.clone())?;
    v_c2 = { let v_M = v_M.clone(); let v_a = v_a.clone(); let v_b = v_b.clone(); let v_n = v_n.clone(); py::func(move |a: &[Py]| -> R {
            #[allow(unused_mut, unused_variables, unused_assignments)]
{
            let mut v_v: Py = a.get(0).cloned().unwrap_or(Py::None);
            return Ok(py::floordiv(&py::mul(&v_v.clone(), &py::sub(&v_v.clone(), &Py::Int(1i64))?)?, &Py::Int(2i64))?);
            #[allow(unreachable_code)]
            Ok(Py::None)
        }
        }) };
    v_sij = py::sum(&{ let mut out = vec![]; for t6554 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t6554;
for t6555 in py::iter(&(v_r.clone()))? { let mut v_v: Py; 
v_v = t6555;
out.push(py::call(&v_c2.clone(), &[v_v.clone()])?); }} Py::list(out) }, None)?;
    v_sa = py::sum(&{ let mut out = vec![]; for t6556 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t6556;
out.push(py::call(&v_c2.clone(), &[py::sum(&v_r.clone(), None)?])?); } Py::list(out) }, None)?;
    v_sb = py::sum(&{ let mut out = vec![]; for t6557 in py::range_iter(&[(py::len(&py::getitem(&v_M.clone(), &Py::Int(0i64))?)?).i()?]) { let mut v_j: Py; 
v_j = t6557;
out.push(py::call(&v_c2.clone(), &[py::sum(&{ let mut out = vec![]; for t6558 in py::range_iter(&[(py::len(&v_M.clone())?).i()?]) { let mut v_i: Py; 
v_i = t6558;
out.push(py::getitem(&py::getitem(&v_M.clone(), &v_i.clone())?, &v_j.clone())?); } Py::list(out) }, None)?])?); } Py::list(out) }, None)?;
    return Ok(Py::tuple(vec![v_sij.clone(), v_sa.clone(), v_sb.clone(), py::call(&v_c2.clone(), &[v_n.clone()])?]));
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__per_class(mut v_yt: Py, mut v_yp: Py) -> R {
    let mut v_F: Py = Py::None;
    let mut v_P: Py = Py::None;
    let mut v_R: Py = Py::None;
    let mut v_S: Py = Py::None;
    let mut v_ap: Py = Py::None;
    let mut v_c: Py = Py::None;
    let mut v_cls: Py = Py::None;
    let mut v_f: Py = Py::None;
    let mut v_p: Py = Py::None;
    let mut v_pp: Py = Py::None;
    let mut v_r: Py = Py::None;
    let mut v_tp: Py = Py::None;
    v_cls = f__classes(Py::tuple(vec![v_yt.clone(), v_yp.clone()]))?;
    let t6402 = (Py::tuple(vec![Py::list(vec![]), Py::list(vec![]), Py::list(vec![]), Py::list(vec![])])).items()?;
    if t6402.len() != 4 { return Err(PyErr::Value("unpack".into())); }
    v_P = t6402[0].clone();
    v_R = t6402[1].clone();
    v_F = t6402[2].clone();
    v_S = t6402[3].clone();
    for t6403 in py::iter(&(v_cls.clone()))? {
        v_c = t6403;
        v_tp = py::sum(&{ let mut out = vec![]; for t6404 in py::iter(&(py::zip(&[v_yt.clone(), v_yp.clone()])?))? { let mut v_a: Py; let mut v_b: Py; 
let t6405 = (t6404).items()?;
if t6405.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t6405[0].clone();
v_b = t6405[1].clone();
if ({ let t6410 = { let t6406 = v_a.clone(); Py::Bool({ let t6407 = v_c.clone(); if py::eq(&t6406, &t6407) { true } else { false } }) }; if !t6410.truthy() { t6410 } else { { let t6408 = v_b.clone(); Py::Bool({ let t6409 = v_c.clone(); if py::eq(&t6408, &t6409) { true } else { false } }) } } }).truthy() { out.push(Py::Int(1i64)); }} Py::list(out) }, None)?;
        v_pp = py::sum(&{ let mut out = vec![]; for t6411 in py::iter(&(v_yp.clone()))? { let mut v_b: Py; 
v_b = t6411;
if ({ let t6412 = v_b.clone(); Py::Bool({ let t6413 = v_c.clone(); if py::eq(&t6412, &t6413) { true } else { false } }) }).truthy() { out.push(Py::Int(1i64)); }} Py::list(out) }, None)?;
        v_ap = py::sum(&{ let mut out = vec![]; for t6414 in py::iter(&(v_yt.clone()))? { let mut v_a: Py; 
v_a = t6414;
if ({ let t6415 = v_a.clone(); Py::Bool({ let t6416 = v_c.clone(); if py::eq(&t6415, &t6416) { true } else { false } }) }).truthy() { out.push(Py::Int(1i64)); }} Py::list(out) }, None)?;
        v_p = (if (v_pp.clone()).truthy() { py::fraction(&[v_tp.clone(), v_pp.clone()])? } else { py::fraction(&[Py::Int(0i64)])? });
        v_r = (if (v_ap.clone()).truthy() { py::fraction(&[v_tp.clone(), v_ap.clone()])? } else { py::fraction(&[Py::Int(0i64)])? });
        v_f = (if (py::add(&v_p.clone(), &v_r.clone())?).truthy() { py::truediv(&py::mul(&py::mul(&Py::Int(2i64), &v_p.clone())?, &v_r.clone())?, &py::add(&v_p.clone(), &v_r.clone())?)? } else { py::fraction(&[Py::Int(0i64)])? });
        let _ = Py::tuple(vec![py::method(&v_P.clone(), "append", &[v_p.clone()])?, py::method(&v_R.clone(), "append", &[v_r.clone()])?, py::method(&v_F.clone(), "append", &[v_f.clone()])?, py::method(&v_S.clone(), "append", &[v_ap.clone()])?]);
    }
    return Ok(Py::tuple(vec![v_cls.clone(), v_P.clone(), v_R.clone(), v_F.clone(), v_S.clone()]));
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__percentile(mut v_sorted_vals: Py, mut v_p: Py) -> R {
    let mut v_h: Py = Py::None;
    let mut v_hi: Py = Py::None;
    let mut v_lo: Py = Py::None;
    let mut v_n: Py = Py::None;
    v_n = py::len(&v_sorted_vals.clone())?;
    v_h = py::mul(&py::sub(&v_n.clone(), &Py::Int(1i64))?, &v_p.clone())?;
    v_lo = py::int(&py::math1("floor", &v_h.clone())?)?;
    v_hi = py::minmax(&[py::add(&v_lo.clone(), &Py::Int(1i64))?, py::sub(&v_n.clone(), &Py::Int(1i64))?], false)?;
    return Ok(py::add(&py::getitem(&v_sorted_vals.clone(), &v_lo.clone())?, &py::mul(&py::sub(&v_h.clone(), &v_lo.clone())?, &py::sub(&py::getitem(&v_sorted_vals.clone(), &v_hi.clone())?, &py::getitem(&v_sorted_vals.clone(), &v_lo.clone())?)?)?)?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__points(mut v_X: Py, mut v_labels: Py) -> R {
    let mut v_M: Py = Py::None;
    let mut v_lab: Py = Py::None;
    v_M = f__mat(Py::str("X"), v_X.clone(), Py::Int(1i64), Py::Int(1i64))?;
    v_lab = f__labels(Py::str("labels"), v_labels.clone(), Py::Int(1i64))?;
    let _ = f__need({ let t7417 = py::len(&v_lab.clone())?; Py::Bool({ let t7418 = py::len(&v_M.clone())?; if py::eq(&t7417, &t7418) { true } else { false } }) }, Py::str("labels needs one label per row of X"))?;
    return Ok(Py::tuple(vec![v_M.clone(), v_lab.clone()]));
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__posterior(mut v_cls: Py, mut v_jll: Py) -> R {
    let mut v_best: Py = Py::None;
    let mut v_m: Py = Py::None;
    let mut v_post: Py = Py::None;
    let mut v_s: Py = Py::None;
    v_m = py::minmax(&[v_jll.clone()], true)?;
    v_s = py::math1("log", &py::fsum(&{ let mut out = vec![]; for t5657 in py::iter(&(v_jll.clone()))? { let mut v_v: Py; 
v_v = t5657;
out.push(py::math1("exp", &py::sub(&v_v.clone(), &v_m.clone())?)?); } Py::list(out) })?)?;
    v_post = { let mut out = vec![]; for t5658 in py::iter(&(v_jll.clone()))? { let mut v_v: Py; 
v_v = t5658;
out.push(py::math1("exp", &py::sub(&py::sub(&v_v.clone(), &v_m.clone())?, &v_s.clone())?)?); } Py::list(out) };
    v_best = py::minmax_key(&[Py::list((py::range_iter(&[(py::len(&v_cls.clone())?).i()?])).into_iter().collect::<Vec<Py>>())], &{ let v_cls = v_cls.clone(); let v_jll = v_jll.clone(); let v_m = v_m.clone(); let v_post = v_post.clone(); let v_s = v_s.clone(); py::func(move |a: &[Py]| -> R {
            #[allow(unused_mut, unused_variables, unused_assignments)]
{
            let mut v_i: Py = a.get(0).cloned().unwrap_or(Py::None);
            return Ok(py::getitem(&v_jll.clone(), &v_i.clone())?);
            #[allow(unreachable_code)]
            Ok(Py::None)
        }
        }) }, true)?;
    let _ = f__need({ let t5659 = py::sum(&{ let mut out = vec![]; for t5661 in py::iter(&(v_post.clone()))? { let mut v_v: Py; 
v_v = t5661;
if ({ let t5662 = v_v.clone(); Py::Bool({ let t5663 = py::mul(&py::getitem(&v_post.clone(), &v_best.clone())?, &py::sub(&Py::Int(1i64), &Py::Float(1e-09))?)?; if py::ge(&t5662, &t5663)? { true } else { false } }) }).truthy() { out.push(Py::Int(1i64)); }} Py::list(out) }, None)?; Py::Bool({ let t5660 = Py::Int(1i64); if py::eq(&t5659, &t5660) { true } else { false } }) }, Py::str("two classes are tied (equal posteriors to within rounding)"))?;
    return Ok(f__out(Py::dict(vec![(Py::str("prediction"), py::getitem(&v_cls.clone(), &v_best.clone())?), (Py::str("classes"), v_cls.clone()), (Py::str("posteriors"), v_post.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__r(mut v_x: Py) -> R {
    if ({ let t8178 = Py::Bool(matches!(v_x.clone(), Py::Bool(_))); if t8178.truthy() { t8178 } else { { let t8177 = Py::Bool(matches!(v_x.clone(), Py::Str(_))); if t8177.truthy() { t8177 } else { { let t8175 = v_x.clone(); Py::Bool({ let t8176 = Py::None; if py::is(&t8175, &t8176) { true } else { false } }) } } } } }).truthy() {
        return Ok(v_x.clone());
    }
    if (Py::Bool(matches!(v_x.clone(), Py::Int(_) | Py::Big(_) | Py::Bool(_)))).truthy() {
        return Ok(v_x.clone());
    }
    if (Py::Bool(matches!(v_x.clone(), Py::Frac(_)))).truthy() {
        v_x = py::float(&v_x.clone())?;
    }
    if (Py::Bool(matches!(v_x.clone(), Py::List(_) | Py::Tuple(_)))).truthy() {
        return Ok({ let mut out = vec![]; for t8179 in py::iter(&(v_x.clone()))? { let mut v_v: Py; 
v_v = t8179;
out.push(f__r(v_v.clone())?); } Py::list(out) });
    }
    if ({ let t8182 = { let t8180 = v_x.clone(); Py::Bool({ let t8181 = Py::Int(0i64); if py::eq(&t8180, &t8181) { true } else { false } }) }; if t8182.truthy() { t8182 } else { py::not(&py::math1("isfinite", &v_x.clone())?) } }).truthy() {
        return Ok((if ({ let t8183 = v_x.clone(); Py::Bool({ let t8184 = Py::Int(0i64); if py::eq(&t8183, &t8184) { true } else { false } }) }).truthy() { Py::Float(0.0) } else { v_x.clone() }));
    }
    return Ok(py::add(&py::float(&Py::str(&[py::format(&v_x.clone(), ".12g")?].concat()))?, &Py::Float(0.0))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__rel(mut v_a: Py, mut v_b: Py) -> R {
    let mut v_d: Py = Py::None;
    if ({ let t8194 = Py::Bool(matches!(v_a.clone(), Py::List(_) | Py::Tuple(_))); if t8194.truthy() { t8194 } else { Py::Bool(matches!(v_b.clone(), Py::List(_) | Py::Tuple(_))) } }).truthy() {
        if ({ let t8198 = py::not(&{ let t8195 = Py::Bool(matches!(v_a.clone(), Py::List(_) | Py::Tuple(_))); if !t8195.truthy() { t8195 } else { Py::Bool(matches!(v_b.clone(), Py::List(_) | Py::Tuple(_))) } }); if t8198.truthy() { t8198 } else { { let t8196 = py::len(&v_a.clone())?; Py::Bool({ let t8197 = py::len(&v_b.clone())?; if py::ne(&t8196, &t8197) { true } else { false } }) } } }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        return Ok(py::minmax(&[py::add(&{ let mut out = vec![]; for t8199 in py::iter(&(py::zip(&[v_a.clone(), v_b.clone()])?))? { let mut v_x: Py; let mut v_y: Py; 
let t8200 = (t8199).items()?;
if t8200.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_x = t8200[0].clone();
v_y = t8200[1].clone();
out.push(f__rel(v_x.clone(), v_y.clone())?); } Py::list(out) }, &Py::list(vec![Py::Float(0.0)]))?], true)?);
    }
    if ({ let t8201 = Py::Bool(matches!(v_a.clone(), Py::Str(_) | Py::Bool(_))); if t8201.truthy() { t8201 } else { Py::Bool(matches!(v_b.clone(), Py::Str(_) | Py::Bool(_))) } }).truthy() {
        return Ok((if ({ let t8202 = v_a.clone(); Py::Bool({ let t8203 = v_b.clone(); if py::eq(&t8202, &t8203) { true } else { false } }) }).truthy() { Py::Float(0.0) } else { Py::Float(f64::INFINITY) }));
    }
    let t8205: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            let t8204 = (Py::tuple(vec![py::float(&v_a.clone())?, py::float(&v_b.clone())?])).items()?;
            if t8204.len() != 2 { return Err(PyErr::Value("unpack".into())); }
            v_a = t8204[0].clone();
            v_b = t8204[1].clone();
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t8205 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if ((matches!(e, PyErr::Type(_)) || matches!(&e, PyErr::Other(c) if c.as_str() == "TypeError"))) || (matches!(e, PyErr::Value(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    if (py::not(&{ let t8206 = py::math1("isfinite", &v_a.clone())?; if !t8206.truthy() { t8206 } else { py::math1("isfinite", &v_b.clone())? } })).truthy() {
        return Ok((if ({ let t8207 = v_a.clone(); Py::Bool({ let t8208 = v_b.clone(); if py::eq(&t8207, &t8208) { true } else { false } }) }).truthy() { Py::Float(0.0) } else { Py::Float(f64::INFINITY) }));
    }
    v_d = py::abs(&py::sub(&v_a.clone(), &v_b.clone())?)?;
    if ({ let t8209 = v_d.clone(); Py::Bool({ let t8210 = Py::Int(0i64); if py::eq(&t8209, &t8210) { true } else { false } }) }).truthy() {
        return Ok(Py::Float(0.0));
    }
    if ({ let t8215 = { let t8211 = py::abs(&v_a.clone())?; Py::Bool({ let t8212 = Py::Float(1e-15); if py::lt(&t8211, &t8212)? { true } else { false } }) }; if !t8215.truthy() { t8215 } else { { let t8213 = py::abs(&v_b.clone())?; Py::Bool({ let t8214 = Py::Float(1e-15); if py::lt(&t8213, &t8214)? { true } else { false } }) } } }).truthy() {
        return Ok(Py::Float(0.0));
    }
    return Ok(py::truediv(&v_d.clone(), &py::minmax(&[py::abs(&v_a.clone())?, py::abs(&v_b.clone())?], true)?)?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__rel01(mut v_name: Py, mut v_v: Py) -> R {
    let mut v_r: Py = Py::None;
    v_r = f__labels(v_name.clone(), v_v.clone(), Py::Int(1i64))?;
    let _ = f__need(Py::Bool(({ let mut out = vec![]; for t6335 in py::iter(&(v_r.clone()))? { let mut v_x: Py; 
v_x = t6335;
out.push({ let t6336 = v_x.clone(); Py::Bool({ let t6337 = Py::tuple(vec![Py::Int(0i64), Py::Int(1i64)]); if py::contains(&t6337, &t6336)? { true } else { false } }) }); } Py::list(out) }).items()?.iter().all(|x| x.truthy())), Py::str(&["every relevance in ".to_string(), py::format(&v_name.clone(), "")?, " must be 0 or 1".to_string()].concat()))?;
    return Ok(v_r.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__same_len(mut v_a: Py, mut v_b: Py, mut v_na: Py, mut v_nb: Py) -> R {
    let _ = f__need({ let t8073 = py::len(&v_a.clone())?; Py::Bool({ let t8074 = py::len(&v_b.clone())?; if py::eq(&t8073, &t8074) { true } else { false } }) }, Py::str(&[py::format(&v_na.clone(), "")?, " and ".to_string(), py::format(&v_nb.clone(), "")?, " must have the same length".to_string()].concat()))?;
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__sigmoid(mut v_z: Py) -> R {
    let mut v_e: Py = Py::None;
    if ({ let t7361 = v_z.clone(); Py::Bool({ let t7362 = Py::Int(0i64); if py::ge(&t7361, &t7362)? { true } else { false } }) }).truthy() {
        return Ok(py::truediv(&Py::Float(1.0), &py::add(&Py::Float(1.0), &py::math1("exp", &py::neg(&v_z.clone())?)?)?)?);
    }
    v_e = py::math1("exp", &v_z.clone())?;
    return Ok(py::truediv(&v_e.clone(), &py::add(&Py::Float(1.0), &v_e.clone())?)?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__sign_fix(mut v_v: Py) -> R {
    let mut v_j: Py = Py::None;
    v_j = py::minmax_key(&[Py::list((py::range_iter(&[(py::len(&v_v.clone())?).i()?])).into_iter().collect::<Vec<Py>>())], &{ let v_v = v_v.clone(); py::func(move |a: &[Py]| -> R {
            #[allow(unused_mut, unused_variables, unused_assignments)]
{
            let mut v_i: Py = a.get(0).cloned().unwrap_or(Py::None);
            return Ok(Py::tuple(vec![py::abs(&py::getitem(&v_v.clone(), &v_i.clone())?)?, py::neg(&v_i.clone())?]));
            #[allow(unreachable_code)]
            Ok(Py::None)
        }
        }) }, true)?;
    return Ok((if ({ let t6051 = py::getitem(&v_v.clone(), &v_j.clone())?; Py::Bool({ let t6052 = Py::Int(0i64); if py::lt(&t6051, &t6052)? { true } else { false } }) }).truthy() { { let mut out = vec![]; for t6053 in py::iter(&(v_v.clone()))? { let mut v_x: Py; 
v_x = t6053;
out.push(py::neg(&v_x.clone())?); } Py::list(out) } } else { Py::list((v_v.clone()).items()?) }));
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__single_check(mut v_a: Py, mut v_b: Py) -> R {
    let _ = f__need({ let t5866 = { let t5862 = py::len(&py::set(&v_a.clone())?)?; Py::Bool({ let t5863 = Py::Int(2i64); if py::ge(&t5862, &t5863)? { true } else { false } }) }; if !t5866.truthy() { t5866 } else { { let t5864 = py::len(&py::set(&v_b.clone())?)?; Py::Bool({ let t5865 = Py::Int(2i64); if py::ge(&t5864, &t5865)? { true } else { false } }) } } }, Py::str("each labelling must have at least two clusters"))?;
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__softmax(mut v_z: Py) -> R {
    let mut v_e: Py = Py::None;
    let mut v_m: Py = Py::None;
    let mut v_s: Py = Py::None;
    v_m = py::minmax(&[v_z.clone()], true)?;
    v_e = { let mut out = vec![]; for t7546 in py::iter(&(v_z.clone()))? { let mut v_v: Py; 
v_v = t7546;
out.push(py::math1("exp", &py::sub(&v_v.clone(), &v_m.clone())?)?); } Py::list(out) };
    v_s = py::fsum(&v_e.clone())?;
    return Ok({ let mut out = vec![]; for t7547 in py::iter(&(v_e.clone()))? { let mut v_v: Py; 
v_v = t7547;
out.push(py::truediv(&v_v.clone(), &v_s.clone())?); } Py::list(out) });
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__solve_exact(mut v_A: Py, mut v_b: Py) -> R {
    let mut v_M: Py = Py::None;
    let mut v_c: Py = Py::None;
    let mut v_f: Py = Py::None;
    let mut v_k: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_p: Py = Py::None;
    let mut v_r: Py = Py::None;
    v_n = py::len(&v_A.clone())?;
    v_M = { let mut out = vec![]; for t6847 in py::iter(&(py::enumerate(&v_A.clone(), 0)?))? { let mut v_i: Py; let mut v_row: Py; 
let t6848 = (t6847).items()?;
if t6848.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_i = t6848[0].clone();
v_row = t6848[1].clone();
out.push(py::add(&{ let mut out = vec![]; for t6849 in py::iter(&(v_row.clone()))? { let mut v_v: Py; 
v_v = t6849;
out.push(py::fraction(&[v_v.clone()])?); } Py::list(out) }, &Py::list(vec![py::fraction(&[py::getitem(&v_b.clone(), &v_i.clone())?])?]))?); } Py::list(out) };
    for t6850 in py::range_iter(&[(v_n.clone()).i()?]) {
        v_c = t6850;
        v_p = (match ('g6851: { let mut out = vec![]; for t6852 in py::range_iter(&[(v_c.clone()).i()?, (v_n.clone()).i()?]) { let mut v_r: Py; 
v_r = t6852;
if ({ let t6853 = py::getitem(&py::getitem(&v_M.clone(), &v_r.clone())?, &v_c.clone())?; Py::Bool({ let t6854 = Py::Int(0i64); if py::ne(&t6853, &t6854) { true } else { false } }) }).truthy() { out.push(v_r.clone()); break 'g6851 Py::list(out); }} Py::list(out) }).items()?.into_iter().next() { Some(x) => x, None => Py::None });
        let _ = f__need({ let t6855 = v_p.clone(); Py::Bool({ let t6856 = Py::None; if !py::is(&t6855, &t6856) { true } else { false } }) }, Py::str("the linear system is singular (the columns are linearly dependent)"))?;
        let t6857 = (Py::tuple(vec![py::getitem(&v_M.clone(), &v_p.clone())?, py::getitem(&v_M.clone(), &v_c.clone())?])).items()?;
        if t6857.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        py::setitem(&v_M.clone(), &v_c.clone(), t6857[0].clone())?;
        py::setitem(&v_M.clone(), &v_p.clone(), t6857[1].clone())?;
        for t6858 in py::range_iter(&[(v_n.clone()).i()?]) {
            v_r = t6858;
            if ({ let t6863 = { let t6859 = v_r.clone(); Py::Bool({ let t6860 = v_c.clone(); if py::ne(&t6859, &t6860) { true } else { false } }) }; if !t6863.truthy() { t6863 } else { { let t6861 = py::getitem(&py::getitem(&v_M.clone(), &v_r.clone())?, &v_c.clone())?; Py::Bool({ let t6862 = Py::Int(0i64); if py::ne(&t6861, &t6862) { true } else { false } }) } } }).truthy() {
                v_f = py::truediv(&py::getitem(&py::getitem(&v_M.clone(), &v_r.clone())?, &v_c.clone())?, &py::getitem(&py::getitem(&v_M.clone(), &v_c.clone())?, &v_c.clone())?)?;
                for t6864 in py::range_iter(&[(v_c.clone()).i()?, (py::add(&v_n.clone(), &Py::Int(1i64))?).i()?]) {
                    v_k = t6864;
                    let t6865 = py::getitem(&v_M.clone(), &v_r.clone())?; let t6866 = v_k.clone();
                    py::setitem(&t6865, &t6866, py::sub(&py::getitem(&t6865, &t6866)?, &py::mul(&v_f.clone(), &py::getitem(&py::getitem(&v_M.clone(), &v_c.clone())?, &v_k.clone())?)?)?)?;
                }
            }
        }
    }
    return Ok({ let mut out = vec![]; for t6867 in py::range_iter(&[(v_n.clone()).i()?]) { let mut v_i: Py; 
v_i = t6867;
out.push(py::truediv(&py::getitem(&py::getitem(&v_M.clone(), &v_i.clone())?, &v_n.clone())?, &py::getitem(&py::getitem(&v_M.clone(), &v_i.clone())?, &v_i.clone())?)?); } Py::list(out) });
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__sqd(mut v_a: Py, mut v_b: Py) -> R {
    return Ok(py::fsum(&{ let mut out = vec![]; for t7425 in py::iter(&(py::zip(&[v_a.clone(), v_b.clone()])?))? { let mut v_u: Py; let mut v_v: Py; 
let t7426 = (t7425).items()?;
if t7426.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_u = t7426[0].clone();
v_v = t7426[1].clone();
out.push(py::pow(&py::sub(&v_u.clone(), &v_v.clone())?, &Py::Int(2i64))?); } Py::list(out) })?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__sqd_exact(mut v_a: Py, mut v_b: Py) -> R {
    return Ok(py::sum(&{ let mut out = vec![]; for t3745 in py::iter(&(py::zip(&[v_a.clone(), v_b.clone()])?))? { let mut v_u: Py; let mut v_v: Py; 
let t3746 = (t3745).items()?;
if t3746.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_u = t3746[0].clone();
v_v = t3746[1].clone();
out.push(py::pow(&py::sub(&v_u.clone(), &v_v.clone())?, &Py::Int(2i64))?); } Py::list(out) }, None)?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__stump_X(mut v_X: Py) -> R {
    let mut v_M: Py = Py::None;
    v_M = f__mat(Py::str("X"), v_X.clone(), Py::Int(2i64), Py::Int(1i64))?;
    let _ = f__need(Py::Bool(({ let mut out = vec![]; for t1964 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t1964;
for t1965 in py::iter(&(v_r.clone()))? { let mut v_v: Py; 
v_v = t1965;
out.push({ let t1970 = { let t1966 = v_v.clone(); Py::Bool({ let t1967 = py::int(&v_v.clone())?; if py::eq(&t1966, &t1967) { true } else { false } }) }; if !t1970.truthy() { t1970 } else { { let t1968 = py::abs(&v_v.clone())?; Py::Bool({ let t1969 = Py::Int(10000i64); if py::le(&t1968, &t1969)? { true } else { false } }) } } }); }} Py::list(out) }).items()?.iter().all(|x| x.truthy())), Py::str("features must be whole numbers up to 10000 in size"))?;
    return Ok(v_M.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__stump_scan(mut v_M: Py, mut v_score: Py) -> R {
    let mut v_a: Py = Py::None;
    let mut v_b: Py = Py::None;
    let mut v_best: Py = Py::None;
    let mut v_cands: Py = Py::None;
    let mut v_j: Py = Py::None;
    let mut v_left: Py = Py::None;
    let mut v_right: Py = Py::None;
    let mut v_s: Py = Py::None;
    let mut v_t: Py = Py::None;
    let mut v_top: Py = Py::None;
    let mut v_vals: Py = Py::None;
    let t1983 = (Py::tuple(vec![Py::None, Py::list(vec![])])).items()?;
    if t1983.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_best = t1983[0].clone();
    v_cands = t1983[1].clone();
    for t1984 in py::range_iter(&[(py::len(&py::getitem(&v_M.clone(), &Py::Int(0i64))?)?).i()?]) {
        v_j = t1984;
        v_vals = py::sorted(&py::set(&{ let mut out = vec![]; for t1985 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t1985;
out.push(py::getitem(&v_r.clone(), &v_j.clone())?); } Py::list(out) })?, (Py::Bool(false)).truthy())?;
        for t1986 in py::iter(&(py::zip(&[v_vals.clone(), py::slice(&v_vals.clone(), Some((Py::Int(1i64)).i()?), None, None)?])?))? {
            let t1993 = (t1986).items()?;
            if t1993.len() != 2 { return Err(PyErr::Value("unpack".into())); }
            v_a = t1993[0].clone();
            v_b = t1993[1].clone();
            v_t = py::truediv(&py::add(&v_a.clone(), &v_b.clone())?, &Py::Int(2i64))?;
            v_left = { let mut out = vec![]; for t1987 in py::range_iter(&[(py::len(&v_M.clone())?).i()?]) { let mut v_i: Py; 
v_i = t1987;
if ({ let t1988 = py::getitem(&py::getitem(&v_M.clone(), &v_i.clone())?, &v_j.clone())?; Py::Bool({ let t1989 = v_t.clone(); if py::le(&t1988, &t1989)? { true } else { false } }) }).truthy() { out.push(v_i.clone()); }} Py::list(out) };
            v_right = { let mut out = vec![]; for t1990 in py::range_iter(&[(py::len(&v_M.clone())?).i()?]) { let mut v_i: Py; 
v_i = t1990;
if ({ let t1991 = py::getitem(&py::getitem(&v_M.clone(), &v_i.clone())?, &v_j.clone())?; Py::Bool({ let t1992 = v_t.clone(); if py::gt(&t1991, &t1992)? { true } else { false } }) }).truthy() { out.push(v_i.clone()); }} Py::list(out) };
            v_s = py::call(&v_score.clone(), &[v_left.clone(), v_right.clone()])?;
            let _ = py::method(&v_cands.clone(), "append", &[Py::tuple(vec![v_s.clone(), v_j.clone(), v_t.clone(), v_left.clone(), v_right.clone()])])?;
        }
    }
    let _ = f__need(v_cands.clone(), Py::str("no split is possible: every feature is constant"))?;
    let _ = py::method_sort(&v_cands.clone(), Some(&{ let v_M = v_M.clone(); let v_a = v_a.clone(); let v_b = v_b.clone(); let v_best = v_best.clone(); let v_cands = v_cands.clone(); let v_j = v_j.clone(); let v_left = v_left.clone(); let v_right = v_right.clone(); let v_s = v_s.clone(); let v_score = v_score.clone(); let v_t = v_t.clone(); let v_vals = v_vals.clone(); py::func(move |a: &[Py]| -> R {
            #[allow(unused_mut, unused_variables, unused_assignments)]
{
            let mut v_c: Py = a.get(0).cloned().unwrap_or(Py::None);
            return Ok(py::neg(&py::float(&py::getitem(&v_c.clone(), &Py::Int(0i64))?)?)?);
            #[allow(unreachable_code)]
            Ok(Py::None)
        }
        }) }), false)?;
    v_top = py::getitem(&v_cands.clone(), &Py::Int(0i64))?;
    if ({ let t1994 = py::len(&v_cands.clone())?; Py::Bool({ let t1995 = Py::Int(1i64); if py::gt(&t1994, &t1995)? { true } else { false } }) }).truthy() {
        let t1996 = (Py::tuple(vec![py::float(&py::getitem(&v_top.clone(), &Py::Int(0i64))?)?, py::float(&py::getitem(&py::getitem(&v_cands.clone(), &Py::Int(1i64))?, &Py::Int(0i64))?)?])).items()?;
        if t1996.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_a = t1996[0].clone();
        v_b = t1996[1].clone();
        let _ = f__need({ let t2001 = { let t1997 = py::abs(&py::sub(&v_a.clone(), &v_b.clone())?)?; Py::Bool({ let t1998 = py::mul(&Py::Float(1e-09), &py::minmax(&[py::abs(&v_a.clone())?, Py::Float(1e-300)], true)?)?; if py::gt(&t1997, &t1998)? { true } else { false } }) }; if !t2001.truthy() { t2001 } else { { let t1999 = v_a.clone(); Py::Bool({ let t2000 = v_b.clone(); if py::ne(&t1999, &t2000) { true } else { false } }) } } }, Py::str("two different splits are equally good (the best split is not unique)"))?;
    }
    return Ok(v_top.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__sym_solve(mut v_A: Py, mut v_b: Py) -> R {
    return Ok(f__gauss(v_A.clone(), v_b.clone())?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__update(mut v_F: Py, mut v_lab: Py, mut v_k: Py) -> R {
    let mut v_C: Py = Py::None;
    let mut v_j: Py = Py::None;
    let mut v_rows: Py = Py::None;
    v_C = Py::list(vec![]);
    for t3762 in py::range_iter(&[(v_k.clone()).i()?]) {
        v_j = t3762;
        v_rows = { let mut out = vec![]; for t3763 in py::range_iter(&[(py::len(&v_F.clone())?).i()?]) { let mut v_i: Py; 
v_i = t3763;
if ({ let t3764 = py::getitem(&v_lab.clone(), &v_i.clone())?; Py::Bool({ let t3765 = v_j.clone(); if py::eq(&t3764, &t3765) { true } else { false } }) }).truthy() { out.push(py::getitem(&v_F.clone(), &v_i.clone())?); }} Py::list(out) };
        let _ = f__need(v_rows.clone(), Py::str("a cluster became empty (no point is nearest to one of the centroids)"))?;
        let _ = py::method(&v_C.clone(), "append", &[{ let mut out = vec![]; for t3766 in py::range_iter(&[(py::len(&py::getitem(&v_F.clone(), &Py::Int(0i64))?)?).i()?]) { let mut v_t: Py; 
v_t = t3766;
out.push(py::truediv(&py::sum(&{ let mut out = vec![]; for t3767 in py::iter(&(v_rows.clone()))? { let mut v_r: Py; 
v_r = t3767;
out.push(py::getitem(&v_r.clone(), &v_t.clone())?); } Py::list(out) }, None)?, &py::len(&v_rows.clone())?)?); } Py::list(out) }])?;
    }
    return Ok(v_C.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__vec(mut v_name: Py, mut v_v: Py, mut v_lo: Py) -> R {
    let mut v_x: Py = Py::None;
    let _ = f__need({ let t8066 = Py::Bool(matches!(v_v.clone(), Py::List(_) | Py::Tuple(_))); if !t8066.truthy() { t8066 } else { { let t8064 = py::len(&v_v.clone())?; Py::Bool({ let t8065 = v_lo.clone(); if py::ge(&t8064, &t8065)? { true } else { false } }) } } }, Py::str(&[py::format(&v_name.clone(), "")?, " needs at least ".to_string(), py::format(&v_lo.clone(), "")?, " values".to_string()].concat()))?;
    for t8067 in py::iter(&(v_v.clone()))? {
        v_x = t8067;
        let _ = f__fin(Py::str(&["every value in ".to_string(), py::format(&v_name.clone(), "")?].concat()), Py::tuple(vec![v_x.clone()]))?;
    }
    return Ok({ let mut out = vec![]; for t8071 in py::iter(&(v_v.clone()))? { let mut v_x: Py; 
v_x = t8071;
out.push(py::float(&v_x.clone())?); } Py::list(out) });
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f__whole(mut v_name: Py, mut v_v: Py) -> R {
    let _ = f__fin(v_name.clone(), Py::tuple(vec![v_v.clone()]))?;
    let _ = f__need({ let t7824 = py::float(&v_v.clone())?; Py::Bool({ let t7825 = py::int(&py::float(&v_v.clone())?)?; if py::eq(&t7824, &t7825) { true } else { false } }) }, Py::str(&[py::format(&v_name.clone(), "")?, " must be a whole number".to_string()].concat()))?;
    return Ok(py::int(&py::float(&v_v.clone())?)?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_adam_step(mut v_w: Py, mut v_grad: Py, mut v_m: Py, mut v_v: Py, mut v_t: Py, mut v_lr: Py, mut v_beta1: Py, mut v_beta2: Py, mut v_eps: Py, mut v_weight_decay: Py) -> R {
    let mut v_V: Py = Py::None;
    let mut v_W: Py = Py::None;
    let mut v_bc1: Py = Py::None;
    let mut v_bc2: Py = Py::None;
    let mut v_m2: Py = Py::None;
    let mut v_nm: Py = Py::None;
    let mut v_v2: Py = Py::None;
    let mut v_w2: Py = Py::None;
    let mut v_x: Py = Py::None;
    v_V = f__opt_vecs(Py::dict(vec![(Py::str("w"), v_w.clone()), (Py::str("grad"), v_grad.clone()), (Py::str("m"), v_m.clone()), (Py::str("v"), v_v.clone())]))?;
    let _ = f__need({ let t109 = Py::Bool(matches!(v_t.clone(), Py::Int(_) | Py::Big(_) | Py::Bool(_))); if !t109.truthy() { t109 } else { { let t107 = v_t.clone(); Py::Bool({ let t108 = Py::Int(1i64); if py::ge(&t107, &t108)? { true } else { false } }) } } }, Py::str("t (the step number after this update) must be a whole number >= 1"))?;
    for t110 in py::iter(&(Py::tuple(vec![Py::tuple(vec![Py::str("lr"), v_lr.clone()]), Py::tuple(vec![Py::str("beta1"), v_beta1.clone()]), Py::tuple(vec![Py::str("beta2"), v_beta2.clone()]), Py::tuple(vec![Py::str("eps"), v_eps.clone()])])))? {
        let t111 = (t110).items()?;
        if t111.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_nm = t111[0].clone();
        v_x = t111[1].clone();
        let _ = f__fin(v_nm.clone(), Py::tuple(vec![v_x.clone()]))?;
    }
    let _ = f__need({ let t124 = { let t112 = v_lr.clone(); Py::Bool({ let t113 = Py::Int(0i64); if py::gt(&t112, &t113)? { true } else { false } }) }; if !t124.truthy() { t124 } else { { let t123 = { let t114 = Py::Int(0i64); Py::Bool({ let t115 = v_beta1.clone(); if py::le(&t114, &t115)? { { let t116 = Py::Int(1i64); if py::lt(&t115, &t116)? { true } else { false } } } else { false } }) }; if !t123.truthy() { t123 } else { { let t122 = { let t117 = Py::Int(0i64); Py::Bool({ let t118 = v_beta2.clone(); if py::le(&t117, &t118)? { { let t119 = Py::Int(1i64); if py::lt(&t118, &t119)? { true } else { false } } } else { false } }) }; if !t122.truthy() { t122 } else { { let t120 = v_eps.clone(); Py::Bool({ let t121 = Py::Int(0i64); if py::ge(&t120, &t121)? { true } else { false } }) } } } } } } }, Py::str("need lr > 0, 0 <= beta1, beta2 < 1, eps >= 0"))?;
    let _ = f__need(Py::Bool(({ let mut out = vec![]; for t125 in py::iter(&(py::getitem(&v_V.clone(), &Py::str("v"))?))? { let mut v_x: Py; 
v_x = t125;
out.push({ let t126 = v_x.clone(); Py::Bool({ let t127 = Py::Int(0i64); if py::ge(&t126, &t127)? { true } else { false } }) }); } Py::list(out) }).items()?.iter().all(|x| x.truthy())), Py::str("the second-moment estimate v must be >= 0"))?;
    v_W = Py::list((py::getitem(&v_V.clone(), &Py::str("w"))?).items()?);
    if ({ let t128 = v_weight_decay.clone(); Py::Bool({ let t129 = Py::None; if !py::is(&t128, &t129) { true } else { false } }) }).truthy() {
        v_W = { let mut out = vec![]; for t130 in py::iter(&(v_W.clone()))? { let mut v_x: Py; 
v_x = t130;
out.push(py::mul(&v_x.clone(), &py::sub(&Py::Int(1i64), &py::mul(&v_lr.clone(), &v_weight_decay.clone())?)?)?); } Py::list(out) };
    }
    v_m2 = { let mut out = vec![]; for t131 in py::iter(&(py::zip(&[py::getitem(&v_V.clone(), &Py::str("m"))?, py::getitem(&v_V.clone(), &Py::str("grad"))?])?))? { let mut v_a: Py; let mut v_g: Py; 
let t132 = (t131).items()?;
if t132.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t132[0].clone();
v_g = t132[1].clone();
out.push(py::add(&py::mul(&v_beta1.clone(), &v_a.clone())?, &py::mul(&py::sub(&Py::Int(1i64), &v_beta1.clone())?, &v_g.clone())?)?); } Py::list(out) };
    v_v2 = { let mut out = vec![]; for t133 in py::iter(&(py::zip(&[py::getitem(&v_V.clone(), &Py::str("v"))?, py::getitem(&v_V.clone(), &Py::str("grad"))?])?))? { let mut v_a: Py; let mut v_g: Py; 
let t134 = (t133).items()?;
if t134.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t134[0].clone();
v_g = t134[1].clone();
out.push(py::add(&py::mul(&v_beta2.clone(), &v_a.clone())?, &py::mul(&py::mul(&py::sub(&Py::Int(1i64), &v_beta2.clone())?, &v_g.clone())?, &v_g.clone())?)?); } Py::list(out) };
    let t135 = (Py::tuple(vec![py::sub(&Py::Int(1i64), &py::pow(&v_beta1.clone(), &v_t.clone())?)?, py::sub(&Py::Int(1i64), &py::pow(&v_beta2.clone(), &v_t.clone())?)?])).items()?;
    if t135.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_bc1 = t135[0].clone();
    v_bc2 = t135[1].clone();
    let _ = f__need({ let t140 = { let t136 = v_bc1.clone(); Py::Bool({ let t137 = Py::Int(0i64); if py::gt(&t136, &t137)? { true } else { false } }) }; if !t140.truthy() { t140 } else { { let t138 = v_bc2.clone(); Py::Bool({ let t139 = Py::Int(0i64); if py::gt(&t138, &t139)? { true } else { false } }) } } }, Py::str("the bias corrections are zero (beta = 0 with ... )"))?;
    v_w2 = { let mut out = vec![]; for t141 in py::iter(&(py::zip(&[v_W.clone(), v_m2.clone(), v_v2.clone()])?))? { let mut v_a: Py; let mut v_b: Py; let mut v_x: Py; 
let t142 = (t141).items()?;
if t142.len() != 3 { return Err(PyErr::Value("unpack".into())); }
v_x = t142[0].clone();
v_a = t142[1].clone();
v_b = t142[2].clone();
out.push(py::sub(&v_x.clone(), &py::truediv(&py::mul(&v_lr.clone(), &py::truediv(&v_a.clone(), &v_bc1.clone())?)?, &py::add(&py::math1("sqrt", &py::truediv(&v_b.clone(), &v_bc2.clone())?)?, &v_eps.clone())?)?)?); } Py::list(out) };
    return Ok(f__out(Py::dict(vec![(Py::str("m_new"), v_m2.clone()), (Py::str("v_new"), v_v2.clone()), (Py::str("w_new"), v_w2.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_adjusted_r2(mut v_y_true: Py, mut v_y_pred: Py, mut v_p: Py) -> R {
    let mut v_P: Py = Py::None;
    let mut v_Y: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_r2: Py = Py::None;
    let mut v_sst: Py = Py::None;
    let mut v_ybar: Py = Py::None;
    let mut v_yp: Py = Py::None;
    let mut v_yt: Py = Py::None;
    let t287 = (f__pair(v_y_true.clone(), v_y_pred.clone(), Py::Int(3i64))?).items()?;
    if t287.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_yt = t287[0].clone();
    v_yp = t287[1].clone();
    v_n = py::len(&v_yt.clone())?;
    let _ = f__need({ let t290 = Py::Bool(matches!(v_p.clone(), Py::Int(_) | Py::Big(_) | Py::Bool(_))); if !t290.truthy() { t290 } else { { let t288 = v_p.clone(); Py::Bool({ let t289 = Py::Int(1i64); if py::ge(&t288, &t289)? { true } else { false } }) } } }, Py::str("p (number of predictors) must be a whole number >= 1"))?;
    let _ = f__need({ let t291 = py::sub(&py::sub(&v_n.clone(), &v_p.clone())?, &Py::Int(1i64))?; Py::Bool({ let t292 = Py::Int(0i64); if py::gt(&t291, &t292)? { true } else { false } }) }, Py::str("adjusted R^2 needs more samples than predictors plus one (n - p - 1 > 0)"))?;
    v_r2 = (if (Py::Bool(false)).truthy() { f__F(py::getitem(&f_regression_metrics(v_yt.clone(), v_yp.clone())?, &Py::str("r2"))?)? } else { Py::None });
    let t321 = (Py::tuple(vec![{ let mut out = vec![]; for t319 in py::iter(&(v_yt.clone()))? { let mut v_v: Py; 
v_v = t319;
out.push(f__F(v_v.clone())?); } Py::list(out) }, { let mut out = vec![]; for t320 in py::iter(&(v_yp.clone()))? { let mut v_v: Py; 
v_v = t320;
out.push(f__F(v_v.clone())?); } Py::list(out) }])).items()?;
    if t321.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_Y = t321[0].clone();
    v_P = t321[1].clone();
    v_ybar = py::truediv(&py::sum(&v_Y.clone(), None)?, &v_n.clone())?;
    v_sst = py::sum(&{ let mut out = vec![]; for t322 in py::iter(&(v_Y.clone()))? { let mut v_a: Py; 
v_a = t322;
out.push(py::pow(&py::sub(&v_a.clone(), &v_ybar.clone())?, &Py::Int(2i64))?); } Py::list(out) }, None)?;
    let _ = f__need({ let t323 = v_sst.clone(); Py::Bool({ let t324 = Py::Int(0i64); if py::ne(&t323, &t324) { true } else { false } }) }, Py::str("R^2 is undefined: every true value is the same"))?;
    v_r2 = py::sub(&Py::Int(1i64), &py::truediv(&py::sum(&{ let mut out = vec![]; for t325 in py::iter(&(py::zip(&[v_Y.clone(), v_P.clone()])?))? { let mut v_a: Py; let mut v_b: Py; 
let t326 = (t325).items()?;
if t326.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t326[0].clone();
v_b = t326[1].clone();
out.push(py::pow(&py::sub(&v_a.clone(), &v_b.clone())?, &Py::Int(2i64))?); } Py::list(out) }, None)?, &v_sst.clone())?)?;
    return Ok(f__out(Py::dict(vec![(Py::str("r2"), v_r2.clone()), (Py::str("adjusted_r2"), py::sub(&Py::Int(1i64), &py::mul(&py::sub(&Py::Int(1i64), &v_r2.clone())?, &py::fraction(&[py::sub(&v_n.clone(), &Py::Int(1i64))?, py::sub(&py::sub(&v_n.clone(), &v_p.clone())?, &Py::Int(1i64))?])?)?)?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_ami(mut v_labels_true: Py, mut v_labels_pred: Py, mut v_method: Py) -> R {
    let mut v_M: Py = Py::None;
    let mut v_N: Py = Py::None;
    let mut v_a: Py = Py::None;
    let mut v_ai: Py = Py::None;
    let mut v_b: Py = Py::None;
    let mut v_base: Py = Py::None;
    let mut v_bj: Py = Py::None;
    let mut v_cols: Py = Py::None;
    let mut v_emi: Py = Py::None;
    let mut v_ha: Py = Py::None;
    let mut v_hb: Py = Py::None;
    let mut v_lg: Py = Py::None;
    let mut v_logp: Py = Py::None;
    let mut v_mi: Py = Py::None;
    let mut v_nij: Py = Py::None;
    let mut v_norm: Py = Py::None;
    let mut v_rows: Py = Py::None;
    let mut v_terms: Py = Py::None;
    let t194 = (Py::tuple(vec![f__labels(Py::str("labels_true"), v_labels_true.clone(), Py::Int(1i64))?, f__labels(Py::str("labels_pred"), v_labels_pred.clone(), Py::Int(1i64))?])).items()?;
    if t194.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_a = t194[0].clone();
    v_b = t194[1].clone();
    let _ = f__same_len(v_a.clone(), v_b.clone(), Py::str("labels_true"), Py::str("labels_pred"))?;
    let _ = f__single_check(v_a.clone(), v_b.clone())?;
    v_N = py::len(&v_a.clone())?;
    v_M = f__contingency(v_a.clone(), v_b.clone())?;
    v_rows = { let mut out = vec![]; for t213 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t213;
out.push(py::sum(&v_r.clone(), None)?); } Py::list(out) };
    v_cols = { let mut out = vec![]; for t214 in py::range_iter(&[(py::len(&py::getitem(&v_M.clone(), &Py::Int(0i64))?)?).i()?]) { let mut v_j: Py; 
v_j = t214;
out.push(py::sum(&{ let mut out = vec![]; for t215 in py::range_iter(&[(py::len(&v_M.clone())?).i()?]) { let mut v_i: Py; 
v_i = t215;
out.push(py::getitem(&py::getitem(&v_M.clone(), &v_i.clone())?, &v_j.clone())?); } Py::list(out) }, None)?); } Py::list(out) };
    v_mi = f__mi_counts(v_M.clone(), v_N.clone())?;
    v_lg = py::func(|a: &[Py]| -> R { let x = a.first().cloned().ok_or_else(|| PyErr::Type("lgamma() takes one argument".into()))?; py::math1("lgamma", &x) });
    v_terms = Py::list(vec![]);
    for t223 in py::iter(&(v_rows.clone()))? {
        v_ai = t223;
        for t224 in py::iter(&(v_cols.clone()))? {
            v_bj = t224;
            v_base = py::sub(&py::add(&py::add(&py::add(&py::call(&v_lg.clone(), &[py::add(&v_ai.clone(), &Py::Int(1i64))?])?, &py::call(&v_lg.clone(), &[py::add(&v_bj.clone(), &Py::Int(1i64))?])?)?, &py::call(&v_lg.clone(), &[py::add(&py::sub(&v_N.clone(), &v_ai.clone())?, &Py::Int(1i64))?])?)?, &py::call(&v_lg.clone(), &[py::add(&py::sub(&v_N.clone(), &v_bj.clone())?, &Py::Int(1i64))?])?)?, &py::call(&v_lg.clone(), &[py::add(&v_N.clone(), &Py::Int(1i64))?])?)?;
            for t225 in py::range_iter(&[(py::minmax(&[Py::Int(1i64), py::sub(&py::add(&v_ai.clone(), &v_bj.clone())?, &v_N.clone())?], true)?).i()?, (py::add(&py::minmax(&[v_ai.clone(), v_bj.clone()], false)?, &Py::Int(1i64))?).i()?]) {
                v_nij = t225;
                v_logp = py::sub(&py::sub(&py::sub(&py::sub(&v_base.clone(), &py::call(&v_lg.clone(), &[py::add(&v_nij.clone(), &Py::Int(1i64))?])?)?, &py::call(&v_lg.clone(), &[py::add(&py::sub(&v_ai.clone(), &v_nij.clone())?, &Py::Int(1i64))?])?)?, &py::call(&v_lg.clone(), &[py::add(&py::sub(&v_bj.clone(), &v_nij.clone())?, &Py::Int(1i64))?])?)?, &py::call(&v_lg.clone(), &[py::add(&py::add(&py::sub(&py::sub(&v_N.clone(), &v_ai.clone())?, &v_bj.clone())?, &v_nij.clone())?, &Py::Int(1i64))?])?)?;
                let _ = py::method(&v_terms.clone(), "append", &[py::mul(&py::mul(&py::truediv(&v_nij.clone(), &v_N.clone())?, &py::math1("log", &py::truediv(&py::mul(&v_N.clone(), &v_nij.clone())?, &py::mul(&v_ai.clone(), &v_bj.clone())?)?)?)?, &py::math1("exp", &v_logp.clone())?)?])?;
            }
        }
    }
    v_emi = py::fsum(&v_terms.clone())?;
    let t232 = (Py::tuple(vec![f__Hc(v_rows.clone())?, f__Hc(v_cols.clone())?])).items()?;
    if t232.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_ha = t232[0].clone();
    v_hb = t232[1].clone();
    v_norm = py::getitem(&Py::dict(vec![(Py::str("arithmetic"), py::truediv(&py::add(&v_ha.clone(), &v_hb.clone())?, &Py::Int(2i64))?), (Py::str("geometric"), py::math1("sqrt", &py::mul(&v_ha.clone(), &v_hb.clone())?)?), (Py::str("min"), py::minmax(&[v_ha.clone(), v_hb.clone()], false)?), (Py::str("max"), py::minmax(&[v_ha.clone(), v_hb.clone()], true)?)]), &v_method.clone())?;
    let _ = f__need({ let t233 = py::abs(&py::sub(&v_norm.clone(), &v_emi.clone())?)?; Py::Bool({ let t234 = Py::Float(1e-12); if py::gt(&t233, &t234)? { true } else { false } }) }, Py::str("the adjusted mutual information is undefined (normaliser equals the expected MI)"))?;
    return Ok(f__out(Py::dict(vec![(Py::str("ami"), py::truediv(&py::sub(&v_mi.clone(), &v_emi.clone())?, &py::sub(&v_norm.clone(), &v_emi.clone())?)?), (Py::str("expected_mi"), v_emi.clone()), (Py::str("mutual_information"), v_mi.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_attention(mut v_Q: Py, mut v_K: Py, mut v_V: Py, mut v_causal: Py) -> R {
    let mut v_Km: Py = Py::None;
    let mut v_O: Py = Py::None;
    let mut v_Qm: Py = Py::None;
    let mut v_Vm: Py = Py::None;
    let mut v_Wt: Py = Py::None;
    let mut v_d: Py = Py::None;
    let mut v_i: Py = Py::None;
    let mut v_p: Py = Py::None;
    let mut v_q: Py = Py::None;
    let mut v_s: Py = Py::None;
    let mut v_sc: Py = Py::None;
    let t7209 = (Py::tuple(vec![f__mat(Py::str("Q"), v_Q.clone(), Py::Int(1i64), Py::Int(1i64))?, f__mat(Py::str("K"), v_K.clone(), Py::Int(1i64), Py::Int(1i64))?, f__mat(Py::str("V"), v_V.clone(), Py::Int(1i64), Py::Int(1i64))?])).items()?;
    if t7209.len() != 3 { return Err(PyErr::Value("unpack".into())); }
    v_Qm = t7209[0].clone();
    v_Km = t7209[1].clone();
    v_Vm = t7209[2].clone();
    let _ = f__need({ let t7210 = py::len(&py::getitem(&v_Qm.clone(), &Py::Int(0i64))?)?; Py::Bool({ let t7211 = py::len(&py::getitem(&v_Km.clone(), &Py::Int(0i64))?)?; if py::eq(&t7210, &t7211) { true } else { false } }) }, Py::str("Q and K need the same number of columns (the key dimension)"))?;
    let _ = f__need({ let t7212 = py::len(&v_Km.clone())?; Py::Bool({ let t7213 = py::len(&v_Vm.clone())?; if py::eq(&t7212, &t7213) { true } else { false } }) }, Py::str("K and V need the same number of rows"))?;
    if (v_causal.clone()).truthy() {
        let _ = f__need({ let t7214 = py::len(&v_Qm.clone())?; Py::Bool({ let t7215 = py::len(&v_Km.clone())?; if py::eq(&t7214, &t7215) { true } else { false } }) }, Py::str("a causal mask needs as many queries as keys"))?;
    }
    v_d = py::len(&py::getitem(&v_Qm.clone(), &Py::Int(0i64))?)?;
    v_sc = py::truediv(&Py::Int(1i64), &py::math1("sqrt", &v_d.clone())?)?;
    v_Wt = Py::list(vec![]);
    for t7216 in py::iter(&(py::enumerate(&v_Qm.clone(), 0)?))? {
        let t7222 = (t7216).items()?;
        if t7222.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_i = t7222[0].clone();
        v_q = t7222[1].clone();
        v_s = { let mut out = vec![]; for t7217 in py::iter(&(v_Km.clone()))? { let mut v_k: Py; 
v_k = t7217;
out.push(py::mul(&py::fsum(&{ let mut out = vec![]; for t7218 in py::iter(&(py::zip(&[v_q.clone(), v_k.clone()])?))? { let mut v_a: Py; let mut v_b: Py; 
let t7219 = (t7218).items()?;
if t7219.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t7219[0].clone();
v_b = t7219[1].clone();
out.push(py::mul(&v_a.clone(), &v_b.clone())?); } Py::list(out) })?, &v_sc.clone())?); } Py::list(out) };
        if (v_causal.clone()).truthy() {
            v_s = py::slice(&v_s.clone(), None, Some((py::add(&v_i.clone(), &Py::Int(1i64))?).i()?), None)?;
        }
        v_p = py::add(&f__softmax(v_s.clone())?, &py::mul(&Py::list(vec![Py::Float(0.0)]), &py::sub(&py::len(&v_Km.clone())?, &py::len(&v_s.clone())?)?)?)?;
        let _ = py::method(&v_Wt.clone(), "append", &[v_p.clone()])?;
    }
    v_O = { let mut out = vec![]; for t7223 in py::range_iter(&[(py::len(&v_Qm.clone())?).i()?]) { let mut v_i: Py; 
v_i = t7223;
out.push({ let mut out = vec![]; for t7224 in py::range_iter(&[(py::len(&py::getitem(&v_Vm.clone(), &Py::Int(0i64))?)?).i()?]) { let mut v_c: Py; 
v_c = t7224;
out.push(py::fsum(&{ let mut out = vec![]; for t7225 in py::range_iter(&[(py::len(&v_Km.clone())?).i()?]) { let mut v_j: Py; 
v_j = t7225;
out.push(py::mul(&py::getitem(&py::getitem(&v_Wt.clone(), &v_i.clone())?, &v_j.clone())?, &py::getitem(&py::getitem(&v_Vm.clone(), &v_j.clone())?, &v_c.clone())?)?); } Py::list(out) })?); } Py::list(out) }); } Py::list(out) };
    return Ok(f__out(Py::dict(vec![(Py::str("weights"), v_Wt.clone()), (Py::str("output"), v_O.clone()), (Py::str("output_sum"), py::fsum(&{ let mut out = vec![]; for t7238 in py::iter(&(v_O.clone()))? { let mut v_r: Py; 
v_r = t7238;
for t7239 in py::iter(&(v_r.clone()))? { let mut v_v: Py; 
v_v = t7239;
out.push(v_v.clone()); }} Py::list(out) })?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_average_precision(mut v_y_true: Py, mut v_scores: Py) -> R {
    let mut v_P: Py = Py::None;
    let mut v_Pr: Py = Py::None;
    let mut v_R: Py = Py::None;
    let mut v_ap: Py = Py::None;
    let mut v_fps: Py = Py::None;
    let mut v_prec: Py = Py::None;
    let mut v_rec: Py = Py::None;
    let mut v_s: Py = Py::None;
    let mut v_th: Py = Py::None;
    let mut v_tps: Py = Py::None;
    let mut v_trap: Py = Py::None;
    let mut v_yt: Py = Py::None;
    v_yt = f__binary(Py::str("y_true"), v_y_true.clone(), Py::Int(1i64))?;
    v_s = f__vec(Py::str("scores"), v_scores.clone(), Py::Int(1i64))?;
    let _ = f__same_len(v_yt.clone(), v_s.clone(), Py::str("y_true"), Py::str("scores"))?;
    v_P = py::sum(&v_yt.clone(), None)?;
    let _ = f__need({ let t375 = v_P.clone(); Py::Bool({ let t376 = Py::Int(0i64); if py::gt(&t375, &t376)? { true } else { false } }) }, Py::str("average precision needs at least one positive sample"))?;
    let t386 = (f__curve(v_yt.clone(), v_s.clone())?).items()?;
    if t386.len() != 3 { return Err(PyErr::Value("unpack".into())); }
    v_th = t386[0].clone();
    v_tps = t386[1].clone();
    v_fps = t386[2].clone();
    v_prec = { let mut out = vec![]; for t387 in py::iter(&(py::zip(&[v_tps.clone(), v_fps.clone()])?))? { let mut v_f: Py; let mut v_t: Py; 
let t388 = (t387).items()?;
if t388.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_t = t388[0].clone();
v_f = t388[1].clone();
out.push(py::fraction(&[v_t.clone(), py::add(&v_t.clone(), &v_f.clone())?])?); } Py::list(out) };
    v_rec = { let mut out = vec![]; for t389 in py::iter(&(v_tps.clone()))? { let mut v_t: Py; 
v_t = t389;
out.push(py::fraction(&[v_t.clone(), v_P.clone()])?); } Py::list(out) };
    v_ap = py::sum(&{ let mut out = vec![]; for t390 in py::range_iter(&[(py::len(&v_th.clone())?).i()?]) { let mut v_i: Py; 
v_i = t390;
out.push(py::mul(&py::sub(&py::getitem(&v_rec.clone(), &v_i.clone())?, &(if (v_i.clone()).truthy() { py::getitem(&v_rec.clone(), &py::sub(&v_i.clone(), &Py::Int(1i64))?)? } else { Py::Int(0i64) }))?, &py::getitem(&v_prec.clone(), &v_i.clone())?)?); } Py::list(out) }, None)?;
    v_R = py::add(&Py::list(vec![py::fraction(&[Py::Int(0i64)])?]), &v_rec.clone())?;
    v_Pr = py::add(&Py::list(vec![py::fraction(&[Py::Int(1i64)])?]), &v_prec.clone())?;
    v_trap = py::sum(&{ let mut out = vec![]; for t391 in py::range_iter(&[(py::sub(&py::len(&v_R.clone())?, &Py::Int(1i64))?).i()?]) { let mut v_i: Py; 
v_i = t391;
out.push(py::truediv(&py::mul(&py::sub(&py::getitem(&v_R.clone(), &py::add(&v_i.clone(), &Py::Int(1i64))?)?, &py::getitem(&v_R.clone(), &v_i.clone())?)?, &py::add(&py::getitem(&v_Pr.clone(), &py::add(&v_i.clone(), &Py::Int(1i64))?)?, &py::getitem(&v_Pr.clone(), &v_i.clone())?)?)?, &Py::Int(2i64))?); } Py::list(out) }, None)?;
    return Ok(f__out(Py::dict(vec![(Py::str("average_precision"), v_ap.clone()), (Py::str("pr_auc_trapezoid"), v_trap.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_backprop(mut v_x: Py, mut v_t: Py, mut v_W1: Py, mut v_b1: Py, mut v_activation: Py, mut v_W2: Py, mut v_b2: Py, mut v_loss: Py) -> R {
    let mut v_A: Py = Py::None;
    let mut v_B: Py = Py::None;
    let mut v_L: Py = Py::None;
    let mut v_c1: Py = Py::None;
    let mut v_c2: Py = Py::None;
    let mut v_d1: Py = Py::None;
    let mut v_d2: Py = Py::None;
    let mut v_dW1: Py = Py::None;
    let mut v_dW2: Py = Py::None;
    let mut v_dh: Py = Py::None;
    let mut v_h: Py = Py::None;
    let mut v_lp: Py = Py::None;
    let mut v_o: Py = Py::None;
    let mut v_p: Py = Py::None;
    let mut v_tv: Py = Py::None;
    let mut v_xv: Py = Py::None;
    let mut v_z1: Py = Py::None;
    let mut v_z2: Py = Py::None;
    let t465 = (f__net_inputs(v_x.clone(), v_W1.clone(), v_b1.clone(), v_W2.clone(), v_b2.clone())?).items()?;
    if t465.len() != 5 { return Err(PyErr::Value("unpack".into())); }
    v_xv = t465[0].clone();
    v_A = t465[1].clone();
    v_c1 = t465[2].clone();
    v_B = t465[3].clone();
    v_c2 = t465[4].clone();
    v_tv = f__vec(Py::str("t"), v_t.clone(), Py::Int(1i64))?;
    let _ = f__need({ let t466 = py::len(&v_tv.clone())?; Py::Bool({ let t467 = py::len(&v_B.clone())?; if py::eq(&t466, &t467) { true } else { false } }) }, Py::str("t needs one target per output"))?;
    let t485 = (f__forward(v_xv.clone(), v_A.clone(), v_c1.clone(), v_B.clone(), v_c2.clone(), v_activation.clone())?).items()?;
    if t485.len() != 3 { return Err(PyErr::Value("unpack".into())); }
    v_z1 = t485[0].clone();
    v_h = t485[1].clone();
    v_z2 = t485[2].clone();
    if ({ let t486 = v_activation.clone(); Py::Bool({ let t487 = Py::str("relu"); if py::eq(&t486, &t487) { true } else { false } }) }).truthy() {
        let _ = f__need(Py::Bool(({ let mut out = vec![]; for t488 in py::iter(&(v_z1.clone()))? { let mut v_v: Py; 
v_v = t488;
out.push({ let t489 = v_v.clone(); Py::Bool({ let t490 = Py::Int(0i64); if py::ne(&t489, &t490) { true } else { false } }) }); } Py::list(out) }).items()?.iter().all(|x| x.truthy())), Py::str("a ReLU pre-activation is exactly 0 (its derivative is undefined there)"))?;
    }
    v_o = py::len(&v_z2.clone())?;
    if ({ let t491 = v_loss.clone(); Py::Bool({ let t492 = Py::str("mse"); if py::eq(&t491, &t492) { true } else { false } }) }).truthy() {
        v_L = py::truediv(&py::fsum(&{ let mut out = vec![]; for t493 in py::iter(&(py::zip(&[v_z2.clone(), v_tv.clone()])?))? { let mut v_a: Py; let mut v_b: Py; 
let t494 = (t493).items()?;
if t494.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t494[0].clone();
v_b = t494[1].clone();
out.push(py::pow(&py::sub(&v_a.clone(), &v_b.clone())?, &Py::Int(2i64))?); } Py::list(out) })?, &v_o.clone())?;
        v_d2 = { let mut out = vec![]; for t495 in py::iter(&(py::zip(&[v_z2.clone(), v_tv.clone()])?))? { let mut v_a: Py; let mut v_b: Py; 
let t496 = (t495).items()?;
if t496.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t496[0].clone();
v_b = t496[1].clone();
out.push(py::truediv(&py::mul(&Py::Int(2i64), &py::sub(&v_a.clone(), &v_b.clone())?)?, &v_o.clone())?); } Py::list(out) };
    } else {
        let _ = f__need({ let t497 = v_o.clone(); Py::Bool({ let t498 = Py::Int(2i64); if py::ge(&t497, &t498)? { true } else { false } }) }, Py::str("cross-entropy needs at least two outputs"))?;
        let _ = f__need({ let t504 = Py::Bool(({ let mut out = vec![]; for t499 in py::iter(&(v_tv.clone()))? { let mut v_v: Py; 
v_v = t499;
out.push({ let t500 = v_v.clone(); Py::Bool({ let t501 = Py::Int(0i64); if py::ge(&t500, &t501)? { true } else { false } }) }); } Py::list(out) }).items()?.iter().all(|x| x.truthy())); if !t504.truthy() { t504 } else { { let t502 = py::abs(&py::sub(&py::fsum(&v_tv.clone())?, &Py::Int(1i64))?)?; Py::Bool({ let t503 = Py::Float(1e-09); if py::le(&t502, &t503)? { true } else { false } }) } } }, Py::str("the target t must be a probability distribution for cross-entropy"))?;
        v_lp = f__log_softmax(v_z2.clone())?;
        v_L = py::neg(&py::fsum(&{ let mut out = vec![]; for t507 in py::iter(&(py::zip(&[v_tv.clone(), v_lp.clone()])?))? { let mut v_a: Py; let mut v_b: Py; 
let t508 = (t507).items()?;
if t508.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t508[0].clone();
v_b = t508[1].clone();
out.push(py::mul(&v_a.clone(), &v_b.clone())?); } Py::list(out) })?)?;
        v_p = f__softmax(v_z2.clone())?;
        v_d2 = { let mut out = vec![]; for t511 in py::iter(&(py::zip(&[v_p.clone(), v_tv.clone()])?))? { let mut v_pi: Py; let mut v_ti: Py; 
let t512 = (t511).items()?;
if t512.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_pi = t512[0].clone();
v_ti = t512[1].clone();
out.push(py::sub(&v_pi.clone(), &v_ti.clone())?); } Py::list(out) };
    }
    v_dW2 = { let mut out = vec![]; for t513 in py::iter(&(v_d2.clone()))? { let mut v_d: Py; 
v_d = t513;
out.push({ let mut out = vec![]; for t514 in py::iter(&(v_h.clone()))? { let mut v_hj: Py; 
v_hj = t514;
out.push(py::mul(&v_d.clone(), &v_hj.clone())?); } Py::list(out) }); } Py::list(out) };
    v_dh = { let mut out = vec![]; for t515 in py::range_iter(&[(py::len(&v_h.clone())?).i()?]) { let mut v_j: Py; 
v_j = t515;
out.push(py::fsum(&{ let mut out = vec![]; for t516 in py::range_iter(&[(v_o.clone()).i()?]) { let mut v_k: Py; 
v_k = t516;
out.push(py::mul(&py::getitem(&v_d2.clone(), &v_k.clone())?, &py::getitem(&py::getitem(&v_B.clone(), &v_k.clone())?, &v_j.clone())?)?); } Py::list(out) })?); } Py::list(out) };
    v_d1 = { let mut out = vec![]; for t517 in py::iter(&(py::zip(&[v_dh.clone(), v_z1.clone(), v_h.clone()])?))? { let mut v_a: Py; let mut v_g: Py; let mut v_z: Py; 
let t518 = (t517).items()?;
if t518.len() != 3 { return Err(PyErr::Value("unpack".into())); }
v_g = t518[0].clone();
v_z = t518[1].clone();
v_a = t518[2].clone();
out.push(py::mul(&v_g.clone(), &f__dact(v_activation.clone(), v_z.clone(), v_a.clone())?)?); } Py::list(out) };
    v_dW1 = { let mut out = vec![]; for t525 in py::iter(&(v_d1.clone()))? { let mut v_d: Py; 
v_d = t525;
out.push({ let mut out = vec![]; for t526 in py::iter(&(v_xv.clone()))? { let mut v_xi: Py; 
v_xi = t526;
out.push(py::mul(&v_d.clone(), &v_xi.clone())?); } Py::list(out) }); } Py::list(out) };
    return Ok(f__out(Py::dict(vec![(Py::str("loss"), v_L.clone()), (Py::str("dW1"), v_dW1.clone()), (Py::str("db1"), v_d1.clone()), (Py::str("dW2"), v_dW2.clone()), (Py::str("db2"), v_d2.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_balanced_accuracy(mut v_y_true: Py, mut v_y_pred: Py) -> R {
    let mut v_adj: Py = Py::None;
    let mut v_ba: Py = Py::None;
    let mut v_k: Py = Py::None;
    let mut v_present: Py = Py::None;
    let mut v_rec: Py = Py::None;
    let mut v_yp: Py = Py::None;
    let mut v_yt: Py = Py::None;
    let t577 = (Py::tuple(vec![f__labels(Py::str("y_true"), v_y_true.clone(), Py::Int(1i64))?, f__labels(Py::str("y_pred"), v_y_pred.clone(), Py::Int(1i64))?])).items()?;
    if t577.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_yt = t577[0].clone();
    v_yp = t577[1].clone();
    let _ = f__same_len(v_yt.clone(), v_yp.clone(), Py::str("y_true"), Py::str("y_pred"))?;
    v_present = py::sorted(&py::set(&v_yt.clone())?, (Py::Bool(false)).truthy())?;
    v_rec = { let mut out = vec![]; for t580 in py::iter(&(v_present.clone()))? { let mut v_c: Py; 
v_c = t580;
out.push(py::fraction(&[py::sum(&{ let mut out = vec![]; for t581 in py::iter(&(py::zip(&[v_yt.clone(), v_yp.clone()])?))? { let mut v_a: Py; let mut v_b: Py; 
let t582 = (t581).items()?;
if t582.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t582[0].clone();
v_b = t582[1].clone();
if ({ let t587 = { let t583 = v_a.clone(); Py::Bool({ let t584 = v_c.clone(); if py::eq(&t583, &t584) { true } else { false } }) }; if !t587.truthy() { t587 } else { { let t585 = v_b.clone(); Py::Bool({ let t586 = v_c.clone(); if py::eq(&t585, &t586) { true } else { false } }) } } }).truthy() { out.push(Py::Int(1i64)); }} Py::list(out) }, None)?, py::method(&v_yt.clone(), "count", &[v_c.clone()])?])?); } Py::list(out) };
    v_k = py::len(&v_present.clone())?;
    v_ba = py::truediv(&py::sum(&v_rec.clone(), None)?, &v_k.clone())?;
    let _ = f__need({ let t588 = v_k.clone(); Py::Bool({ let t589 = Py::Int(2i64); if py::ge(&t588, &t589)? { true } else { false } }) }, Py::str("the adjusted balanced accuracy needs at least two true classes"))?;
    v_adj = py::truediv(&py::sub(&v_ba.clone(), &py::fraction(&[Py::Int(1i64), v_k.clone()])?)?, &py::sub(&Py::Int(1i64), &py::fraction(&[Py::Int(1i64), v_k.clone()])?)?)?;
    return Ok(f__out(Py::dict(vec![(Py::str("balanced_accuracy"), v_ba.clone()), (Py::str("adjusted"), v_adj.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_batchnorm(mut v_X: Py, mut v_gamma: Py, mut v_beta: Py, mut v_eps: Py) -> R {
    let mut v_D: Py = Py::None;
    let mut v_F: Py = Py::None;
    let mut v_M: Py = Py::None;
    let mut v_Y: Py = Py::None;
    let mut v_b: Py = Py::None;
    let mut v_g: Py = Py::None;
    let mut v_mean: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_sd: Py = Py::None;
    let mut v_var: Py = Py::None;
    v_M = f__mat(Py::str("X"), v_X.clone(), Py::Int(2i64), Py::Int(1i64))?;
    let t705 = (Py::tuple(vec![f__vec(Py::str("gamma"), v_gamma.clone(), Py::Int(1i64))?, f__vec(Py::str("beta"), v_beta.clone(), Py::Int(1i64))?])).items()?;
    if t705.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_g = t705[0].clone();
    v_b = t705[1].clone();
    v_D = py::len(&py::getitem(&v_M.clone(), &Py::Int(0i64))?)?;
    let _ = f__need({ let t710 = { let t706 = py::len(&v_g.clone())?; Py::Bool({ let t707 = v_D.clone(); if py::eq(&t706, &t707) { true } else { false } }) }; if !t710.truthy() { t710 } else { { let t708 = py::len(&v_b.clone())?; Py::Bool({ let t709 = v_D.clone(); if py::eq(&t708, &t709) { true } else { false } }) } } }, Py::str("gamma and beta need one entry per column (feature) of X"))?;
    let _ = f__fin(Py::str("eps"), Py::tuple(vec![v_eps.clone()]))?;
    let _ = f__need({ let t711 = v_eps.clone(); Py::Bool({ let t712 = Py::Int(0i64); if py::ge(&t711, &t712)? { true } else { false } }) }, Py::str("eps must be >= 0"))?;
    v_n = py::len(&v_M.clone())?;
    v_F = { let mut out = vec![]; for t713 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t713;
out.push({ let mut out = vec![]; for t714 in py::iter(&(v_r.clone()))? { let mut v_v: Py; 
v_v = t714;
out.push(f__F(v_v.clone())?); } Py::list(out) }); } Py::list(out) };
    v_mean = { let mut out = vec![]; for t715 in py::range_iter(&[(v_D.clone()).i()?]) { let mut v_j: Py; 
v_j = t715;
out.push(py::truediv(&py::sum(&{ let mut out = vec![]; for t716 in py::iter(&(v_F.clone()))? { let mut v_r: Py; 
v_r = t716;
out.push(py::getitem(&v_r.clone(), &v_j.clone())?); } Py::list(out) }, None)?, &v_n.clone())?); } Py::list(out) };
    v_var = { let mut out = vec![]; for t717 in py::range_iter(&[(v_D.clone()).i()?]) { let mut v_j: Py; 
v_j = t717;
out.push(py::truediv(&py::sum(&{ let mut out = vec![]; for t718 in py::iter(&(v_F.clone()))? { let mut v_r: Py; 
v_r = t718;
out.push(py::pow(&py::sub(&py::getitem(&v_r.clone(), &v_j.clone())?, &py::getitem(&v_mean.clone(), &v_j.clone())?)?, &Py::Int(2i64))?); } Py::list(out) }, None)?, &v_n.clone())?); } Py::list(out) };
    let _ = f__need(Py::Bool(({ let mut out = vec![]; for t719 in py::iter(&(v_var.clone()))? { let mut v_v: Py; 
v_v = t719;
out.push({ let t720 = py::add(&v_v.clone(), &f__F(v_eps.clone())?)?; Py::Bool({ let t721 = Py::Int(0i64); if py::gt(&t720, &t721)? { true } else { false } }) }); } Py::list(out) }).items()?.iter().all(|x| x.truthy())), Py::str("a feature has zero variance and eps = 0"))?;
    v_sd = { let mut out = vec![]; for t722 in py::iter(&(v_var.clone()))? { let mut v_v: Py; 
v_v = t722;
out.push(py::math1("sqrt", &py::float(&py::add(&v_v.clone(), &f__F(v_eps.clone())?)?)?)?); } Py::list(out) };
    v_Y = { let mut out = vec![]; for t723 in py::iter(&(v_F.clone()))? { let mut v_r: Py; 
v_r = t723;
out.push({ let mut out = vec![]; for t724 in py::range_iter(&[(v_D.clone()).i()?]) { let mut v_j: Py; 
v_j = t724;
out.push(py::add(&py::truediv(&py::mul(&py::getitem(&v_g.clone(), &v_j.clone())?, &py::float(&py::sub(&py::getitem(&v_r.clone(), &v_j.clone())?, &py::getitem(&v_mean.clone(), &v_j.clone())?)?)?)?, &py::getitem(&v_sd.clone(), &v_j.clone())?)?, &py::getitem(&v_b.clone(), &v_j.clone())?)?); } Py::list(out) }); } Py::list(out) };
    return Ok(f__out(Py::dict(vec![(Py::str("output"), v_Y.clone()), (Py::str("batch_mean"), v_mean.clone()), (Py::str("batch_var"), v_var.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_bce_logits(mut v_y: Py, mut v_z: Py) -> R {
    let mut v_loss: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_yv: Py = Py::None;
    let mut v_zv: Py = Py::None;
    let t774 = (Py::tuple(vec![f__vec(Py::str("y"), v_y.clone(), Py::Int(1i64))?, f__vec(Py::str("z"), v_z.clone(), Py::Int(1i64))?])).items()?;
    if t774.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_yv = t774[0].clone();
    v_zv = t774[1].clone();
    let _ = f__same_len(v_yv.clone(), v_zv.clone(), Py::str("y"), Py::str("z"))?;
    let _ = f__need(Py::Bool(({ let mut out = vec![]; for t777 in py::iter(&(v_yv.clone()))? { let mut v_v: Py; 
v_v = t777;
out.push({ let t778 = Py::Int(0i64); Py::Bool({ let t779 = v_v.clone(); if py::le(&t778, &t779)? { { let t780 = Py::Int(1i64); if py::le(&t779, &t780)? { true } else { false } } } else { false } }) }); } Py::list(out) }).items()?.iter().all(|x| x.truthy())), Py::str("targets y must lie between 0 and 1"))?;
    v_n = py::len(&v_yv.clone())?;
    v_loss = py::truediv(&py::fsum(&{ let mut out = vec![]; for t781 in py::iter(&(py::zip(&[v_yv.clone(), v_zv.clone()])?))? { let mut v_a: Py; let mut v_b: Py; 
let t782 = (t781).items()?;
if t782.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t782[0].clone();
v_b = t782[1].clone();
out.push(py::add(&py::sub(&py::minmax(&[v_b.clone(), Py::Float(0.0)], true)?, &py::mul(&v_b.clone(), &v_a.clone())?)?, &py::math1("log1p", &py::math1("exp", &py::neg(&py::abs(&v_b.clone())?)?)?)?)?); } Py::list(out) })?, &v_n.clone())?;
    return Ok(f__out(Py::dict(vec![(Py::str("loss"), v_loss.clone()), (Py::str("gradient"), { let mut out = vec![]; for t795 in py::iter(&(py::zip(&[v_yv.clone(), v_zv.clone()])?))? { let mut v_a: Py; let mut v_b: Py; 
let t796 = (t795).items()?;
if t796.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t796[0].clone();
v_b = t796[1].clone();
out.push(py::truediv(&py::sub(&f__sigmoid(v_b.clone())?, &v_a.clone())?, &v_n.clone())?); } Py::list(out) })]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_bernoulli_nb(mut v_X: Py, mut v_y: Py, mut v_x: Py, mut v_alpha: Py) -> R {
    let mut v_M: Py = Py::None;
    let mut v_c: Py = Py::None;
    let mut v_cls: Py = Py::None;
    let mut v_d: Py = Py::None;
    let mut v_j: Py = Py::None;
    let mut v_jll: Py = Py::None;
    let mut v_lp: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_p: Py = Py::None;
    let mut v_q: Py = Py::None;
    let mut v_rows: Py = Py::None;
    let mut v_yl: Py = Py::None;
    let t862 = (f__nb_data(v_X.clone(), v_y.clone(), v_x.clone())?).items()?;
    if t862.len() != 4 { return Err(PyErr::Value("unpack".into())); }
    v_M = t862[0].clone();
    v_yl = t862[1].clone();
    v_q = t862[2].clone();
    v_cls = t862[3].clone();
    let _ = f__fin(Py::str("alpha"), Py::tuple(vec![v_alpha.clone()]))?;
    let _ = f__need({ let t863 = v_alpha.clone(); Py::Bool({ let t864 = Py::Int(0i64); if py::gt(&t863, &t864)? { true } else { false } }) }, Py::str("alpha (additive smoothing) must be positive"))?;
    let _ = f__need({ let t872 = Py::Bool(({ let mut out = vec![]; for t865 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t865;
for t866 in py::iter(&(v_r.clone()))? { let mut v_v: Py; 
v_v = t866;
out.push({ let t867 = v_v.clone(); Py::Bool({ let t868 = Py::tuple(vec![Py::Float(0.0), Py::Float(1.0)]); if py::contains(&t868, &t867)? { true } else { false } }) }); }} Py::list(out) }).items()?.iter().all(|x| x.truthy())); if !t872.truthy() { t872 } else { Py::Bool(({ let mut out = vec![]; for t869 in py::iter(&(v_q.clone()))? { let mut v_v: Py; 
v_v = t869;
out.push({ let t870 = v_v.clone(); Py::Bool({ let t871 = Py::tuple(vec![Py::Float(0.0), Py::Float(1.0)]); if py::contains(&t871, &t870)? { true } else { false } }) }); } Py::list(out) }).items()?.iter().all(|x| x.truthy())) } }, Py::str("features must be 0 or 1"))?;
    let t873 = (Py::tuple(vec![py::len(&v_yl.clone())?, py::len(&v_q.clone())?])).items()?;
    if t873.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_n = t873[0].clone();
    v_d = t873[1].clone();
    v_jll = Py::list(vec![]);
    for t874 in py::iter(&(v_cls.clone()))? {
        v_c = t874;
        v_rows = { let mut out = vec![]; for t875 in py::range_iter(&[(v_n.clone()).i()?]) { let mut v_i: Py; 
v_i = t875;
if ({ let t876 = py::getitem(&v_yl.clone(), &v_i.clone())?; Py::Bool({ let t877 = v_c.clone(); if py::eq(&t876, &t877) { true } else { false } }) }).truthy() { out.push(py::getitem(&v_M.clone(), &v_i.clone())?); }} Py::list(out) };
        v_lp = py::math1("log", &py::truediv(&py::len(&v_rows.clone())?, &v_n.clone())?)?;
        for t878 in py::range_iter(&[(v_d.clone()).i()?]) {
            v_j = t878;
            v_p = py::truediv(&py::add(&py::sum(&{ let mut out = vec![]; for t879 in py::iter(&(v_rows.clone()))? { let mut v_r: Py; 
v_r = t879;
out.push(py::getitem(&v_r.clone(), &v_j.clone())?); } Py::list(out) }, None)?, &v_alpha.clone())?, &py::add(&py::len(&v_rows.clone())?, &py::mul(&Py::Int(2i64), &v_alpha.clone())?)?)?;
            v_lp = py::iadd(&v_lp, &(if ({ let t880 = py::getitem(&v_q.clone(), &v_j.clone())?; Py::Bool({ let t881 = Py::Int(1i64); if py::eq(&t880, &t881) { true } else { false } }) }).truthy() { py::math1("log", &v_p.clone())? } else { py::math1("log1p", &py::neg(&v_p.clone())?)? }))?;
        }
        let _ = py::method(&v_jll.clone(), "append", &[v_lp.clone()])?;
    }
    return Ok(f__posterior(v_cls.clone(), v_jll.clone())?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_bias_variance(mut v_P: Py, mut v_f: Py) -> R {
    let mut v_F: Py = Py::None;
    let mut v_M: Py = Py::None;
    let mut v_R: Py = Py::None;
    let mut v_b2: Py = Py::None;
    let mut v_fv: Py = Py::None;
    let mut v_m: Py = Py::None;
    let mut v_mean: Py = Py::None;
    let mut v_mse: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_var: Py = Py::None;
    v_M = f__mat(Py::str("P"), v_P.clone(), Py::Int(2i64), Py::Int(1i64))?;
    v_fv = f__vec(Py::str("f"), v_f.clone(), Py::Int(1i64))?;
    let _ = f__need({ let t952 = py::len(&v_fv.clone())?; Py::Bool({ let t953 = py::len(&py::getitem(&v_M.clone(), &Py::Int(0i64))?)?; if py::eq(&t952, &t953) { true } else { false } }) }, Py::str("f needs one true value per column of P"))?;
    v_R = { let mut out = vec![]; for t954 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t954;
out.push({ let mut out = vec![]; for t955 in py::iter(&(v_r.clone()))? { let mut v_v: Py; 
v_v = t955;
out.push(f__F(v_v.clone())?); } Py::list(out) }); } Py::list(out) };
    let t956 = (Py::tuple(vec![py::len(&v_R.clone())?, py::len(&v_fv.clone())?])).items()?;
    if t956.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_m = t956[0].clone();
    v_n = t956[1].clone();
    v_F = { let mut out = vec![]; for t957 in py::iter(&(v_fv.clone()))? { let mut v_v: Py; 
v_v = t957;
out.push(f__F(v_v.clone())?); } Py::list(out) };
    v_mean = { let mut out = vec![]; for t958 in py::range_iter(&[(v_n.clone()).i()?]) { let mut v_j: Py; 
v_j = t958;
out.push(py::truediv(&py::sum(&{ let mut out = vec![]; for t959 in py::iter(&(v_R.clone()))? { let mut v_r: Py; 
v_r = t959;
out.push(py::getitem(&v_r.clone(), &v_j.clone())?); } Py::list(out) }, None)?, &v_m.clone())?); } Py::list(out) };
    v_b2 = py::truediv(&py::sum(&{ let mut out = vec![]; for t960 in py::range_iter(&[(v_n.clone()).i()?]) { let mut v_j: Py; 
v_j = t960;
out.push(py::pow(&py::sub(&py::getitem(&v_mean.clone(), &v_j.clone())?, &py::getitem(&v_F.clone(), &v_j.clone())?)?, &Py::Int(2i64))?); } Py::list(out) }, None)?, &v_n.clone())?;
    v_var = py::truediv(&py::sum(&{ let mut out = vec![]; for t961 in py::range_iter(&[(v_n.clone()).i()?]) { let mut v_j: Py; 
v_j = t961;
out.push(py::truediv(&py::sum(&{ let mut out = vec![]; for t962 in py::iter(&(v_R.clone()))? { let mut v_r: Py; 
v_r = t962;
out.push(py::pow(&py::sub(&py::getitem(&v_r.clone(), &v_j.clone())?, &py::getitem(&v_mean.clone(), &v_j.clone())?)?, &Py::Int(2i64))?); } Py::list(out) }, None)?, &v_m.clone())?); } Py::list(out) }, None)?, &v_n.clone())?;
    v_mse = py::truediv(&py::sum(&{ let mut out = vec![]; for t963 in py::range_iter(&[(v_n.clone()).i()?]) { let mut v_j: Py; 
v_j = t963;
out.push(py::truediv(&py::sum(&{ let mut out = vec![]; for t964 in py::iter(&(v_R.clone()))? { let mut v_r: Py; 
v_r = t964;
out.push(py::pow(&py::sub(&py::getitem(&v_r.clone(), &v_j.clone())?, &py::getitem(&v_F.clone(), &v_j.clone())?)?, &Py::Int(2i64))?); } Py::list(out) }, None)?, &v_m.clone())?); } Py::list(out) }, None)?, &v_n.clone())?;
    return Ok(f__out(Py::dict(vec![(Py::str("bias_squared"), v_b2.clone()), (Py::str("variance"), v_var.clone()), (Py::str("expected_mse"), v_mse.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_binary_confusion(mut v_y_true: Py, mut v_y_pred: Py) -> R {
    let mut v_fn: Py = Py::None;
    let mut v_fp: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_tn: Py = Py::None;
    let mut v_tp: Py = Py::None;
    let mut v_yp: Py = Py::None;
    let mut v_yt: Py = Py::None;
    let t1018 = (Py::tuple(vec![f__binary(Py::str("y_true"), v_y_true.clone(), Py::Int(1i64))?, f__binary(Py::str("y_pred"), v_y_pred.clone(), Py::Int(1i64))?])).items()?;
    if t1018.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_yt = t1018[0].clone();
    v_yp = t1018[1].clone();
    let _ = f__same_len(v_yt.clone(), v_yp.clone(), Py::str("y_true"), Py::str("y_pred"))?;
    let t1049 = (f__counts(v_yt.clone(), v_yp.clone())?).items()?;
    if t1049.len() != 4 { return Err(PyErr::Value("unpack".into())); }
    v_tp = t1049[0].clone();
    v_fp = t1049[1].clone();
    v_fn = t1049[2].clone();
    v_tn = t1049[3].clone();
    v_n = py::len(&v_yt.clone())?;
    return Ok(f__out(Py::dict(vec![(Py::str("TP"), v_tp.clone()), (Py::str("FP"), v_fp.clone()), (Py::str("FN"), v_fn.clone()), (Py::str("TN"), v_tn.clone()), (Py::str("accuracy"), py::fraction(&[py::add(&v_tp.clone(), &v_tn.clone())?, v_n.clone()])?), (Py::str("error_rate"), py::fraction(&[py::add(&v_fp.clone(), &v_fn.clone())?, v_n.clone()])?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_binary_from_counts(mut v_TP: Py, mut v_FP: Py, mut v_FN: Py, mut v_TN: Py) -> R {
    let mut v_P: Py = Py::None;
    let mut v_R: Py = Py::None;
    let mut v_S: Py = Py::None;
    let mut v_f1: Py = Py::None;
    let mut v_n: Py = Py::None;
    let _ = f__check_counts(v_TP.clone(), v_FP.clone(), v_FN.clone(), v_TN.clone())?;
    let _ = f__need({ let t1096 = py::add(&v_TP.clone(), &v_FP.clone())?; Py::Bool({ let t1097 = Py::Int(0i64); if py::gt(&t1096, &t1097)? { true } else { false } }) }, Py::str("precision is undefined: there are no predicted positives (TP + FP = 0)"))?;
    let _ = f__need({ let t1098 = py::add(&v_TP.clone(), &v_FN.clone())?; Py::Bool({ let t1099 = Py::Int(0i64); if py::gt(&t1098, &t1099)? { true } else { false } }) }, Py::str("recall is undefined: there are no actual positives (TP + FN = 0)"))?;
    let _ = f__need({ let t1100 = py::add(&v_TN.clone(), &v_FP.clone())?; Py::Bool({ let t1101 = Py::Int(0i64); if py::gt(&t1100, &t1101)? { true } else { false } }) }, Py::str("specificity is undefined: there are no actual negatives (TN + FP = 0)"))?;
    let _ = f__need({ let t1102 = py::add(&v_TN.clone(), &v_FN.clone())?; Py::Bool({ let t1103 = Py::Int(0i64); if py::gt(&t1102, &t1103)? { true } else { false } }) }, Py::str("the negative predictive value is undefined: there are no predicted negatives (TN + FN = 0)"))?;
    v_n = py::add(&py::add(&py::add(&v_TP.clone(), &v_FP.clone())?, &v_FN.clone())?, &v_TN.clone())?;
    let t1104 = (Py::tuple(vec![py::fraction(&[v_TP.clone(), py::add(&v_TP.clone(), &v_FP.clone())?])?, py::fraction(&[v_TP.clone(), py::add(&v_TP.clone(), &v_FN.clone())?])?])).items()?;
    if t1104.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_P = t1104[0].clone();
    v_R = t1104[1].clone();
    v_S = py::fraction(&[v_TN.clone(), py::add(&v_TN.clone(), &v_FP.clone())?])?;
    v_f1 = py::fraction(&[py::mul(&Py::Int(2i64), &v_TP.clone())?, py::add(&py::add(&py::mul(&Py::Int(2i64), &v_TP.clone())?, &v_FP.clone())?, &v_FN.clone())?])?;
    return Ok(f__out(Py::dict(vec![(Py::str("accuracy"), py::fraction(&[py::add(&v_TP.clone(), &v_TN.clone())?, v_n.clone()])?), (Py::str("precision"), v_P.clone()), (Py::str("recall"), v_R.clone()), (Py::str("specificity"), v_S.clone()), (Py::str("f1"), v_f1.clone()), (Py::str("npv"), py::fraction(&[v_TN.clone(), py::add(&v_TN.clone(), &v_FN.clone())?])?), (Py::str("balanced_accuracy"), py::truediv(&py::add(&v_R.clone(), &v_S.clone())?, &Py::Int(2i64))?), (Py::str("fpr"), py::sub(&Py::Int(1i64), &v_S.clone())?), (Py::str("fnr"), py::sub(&Py::Int(1i64), &v_R.clone())?), (Py::str("jaccard"), py::fraction(&[v_TP.clone(), py::add(&py::add(&v_TP.clone(), &v_FP.clone())?, &v_FN.clone())?])?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_brier(mut v_y_true: Py, mut v_p: Py) -> R {
    let mut v_pv: Py = Py::None;
    let mut v_yt: Py = Py::None;
    v_yt = f__binary(Py::str("y_true"), v_y_true.clone(), Py::Int(1i64))?;
    v_pv = f__vec(Py::str("p"), v_p.clone(), Py::Int(1i64))?;
    let _ = f__same_len(v_yt.clone(), v_pv.clone(), Py::str("y_true"), Py::str("p"))?;
    let _ = f__need(Py::Bool(({ let mut out = vec![]; for t1165 in py::iter(&(v_pv.clone()))? { let mut v_q: Py; 
v_q = t1165;
out.push({ let t1166 = Py::Int(0i64); Py::Bool({ let t1167 = v_q.clone(); if py::le(&t1166, &t1167)? { { let t1168 = Py::Int(1i64); if py::le(&t1167, &t1168)? { true } else { false } } } else { false } }) }); } Py::list(out) }).items()?.iter().all(|x| x.truthy())), Py::str("every probability must lie between 0 and 1"))?;
    return Ok(f__out(Py::dict(vec![(Py::str("brier"), py::truediv(&py::sum(&{ let mut out = vec![]; for t1181 in py::iter(&(py::zip(&[v_yt.clone(), v_pv.clone()])?))? { let mut v_q: Py; let mut v_y: Py; 
let t1182 = (t1181).items()?;
if t1182.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_y = t1182[0].clone();
v_q = t1182[1].clone();
out.push(py::pow(&py::sub(&f__F(v_q.clone())?, &v_y.clone())?, &Py::Int(2i64))?); } Py::list(out) }, None)?, &py::len(&v_yt.clone())?)?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_calinski_harabasz(mut v_X: Py, mut v_labels: Py) -> R {
    let mut v_B: Py = Py::None;
    let mut v_F: Py = Py::None;
    let mut v_M: Py = Py::None;
    let mut v_W: Py = Py::None;
    let mut v_c: Py = Py::None;
    let mut v_cen: Py = Py::None;
    let mut v_d: Py = Py::None;
    let mut v_k: Py = Py::None;
    let mut v_ks: Py = Py::None;
    let mut v_lab: Py = Py::None;
    let mut v_mean: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_rows: Py = Py::None;
    let t1237 = (f__points(v_X.clone(), v_labels.clone())?).items()?;
    if t1237.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_M = t1237[0].clone();
    v_lab = t1237[1].clone();
    v_n = py::len(&v_M.clone())?;
    v_ks = py::sorted(&py::set(&v_lab.clone())?, (Py::Bool(false)).truthy())?;
    v_k = py::len(&v_ks.clone())?;
    let _ = f__need({ let t1238 = Py::Int(2i64); Py::Bool({ let t1239 = v_k.clone(); if py::le(&t1238, &t1239)? { { let t1240 = py::sub(&v_n.clone(), &Py::Int(1i64))?; if py::le(&t1239, &t1240)? { true } else { false } } } else { false } }) }, Py::str("the Calinski-Harabasz index needs between 2 and n - 1 clusters"))?;
    v_F = { let mut out = vec![]; for t1241 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t1241;
out.push({ let mut out = vec![]; for t1242 in py::iter(&(v_r.clone()))? { let mut v_v: Py; 
v_v = t1242;
out.push(f__F(v_v.clone())?); } Py::list(out) }); } Py::list(out) };
    v_d = py::len(&py::getitem(&v_F.clone(), &Py::Int(0i64))?)?;
    v_mean = { let mut out = vec![]; for t1243 in py::range_iter(&[(v_d.clone()).i()?]) { let mut v_j: Py; 
v_j = t1243;
out.push(py::truediv(&py::sum(&{ let mut out = vec![]; for t1244 in py::iter(&(v_F.clone()))? { let mut v_r: Py; 
v_r = t1244;
out.push(py::getitem(&v_r.clone(), &v_j.clone())?); } Py::list(out) }, None)?, &v_n.clone())?); } Py::list(out) };
    let t1245 = py::fraction(&[Py::Int(0i64)])?;
    v_B = t1245.clone();
    v_W = t1245.clone();
    for t1246 in py::iter(&(v_ks.clone()))? {
        v_c = t1246;
        v_rows = { let mut out = vec![]; for t1247 in py::range_iter(&[(v_n.clone()).i()?]) { let mut v_i: Py; 
v_i = t1247;
if ({ let t1248 = py::getitem(&v_lab.clone(), &v_i.clone())?; Py::Bool({ let t1249 = v_c.clone(); if py::eq(&t1248, &t1249) { true } else { false } }) }).truthy() { out.push(py::getitem(&v_F.clone(), &v_i.clone())?); }} Py::list(out) };
        v_cen = { let mut out = vec![]; for t1250 in py::range_iter(&[(v_d.clone()).i()?]) { let mut v_j: Py; 
v_j = t1250;
out.push(py::truediv(&py::sum(&{ let mut out = vec![]; for t1251 in py::iter(&(v_rows.clone()))? { let mut v_r: Py; 
v_r = t1251;
out.push(py::getitem(&v_r.clone(), &v_j.clone())?); } Py::list(out) }, None)?, &py::len(&v_rows.clone())?)?); } Py::list(out) };
        v_B = py::iadd(&v_B, &py::mul(&py::len(&v_rows.clone())?, &py::sum(&{ let mut out = vec![]; for t1252 in py::range_iter(&[(v_d.clone()).i()?]) { let mut v_j: Py; 
v_j = t1252;
out.push(py::pow(&py::sub(&py::getitem(&v_cen.clone(), &v_j.clone())?, &py::getitem(&v_mean.clone(), &v_j.clone())?)?, &Py::Int(2i64))?); } Py::list(out) }, None)?)?)?;
        v_W = py::iadd(&v_W, &py::sum(&{ let mut out = vec![]; for t1253 in py::iter(&(v_rows.clone()))? { let mut v_r: Py; 
v_r = t1253;
out.push(py::sum(&{ let mut out = vec![]; for t1254 in py::range_iter(&[(v_d.clone()).i()?]) { let mut v_j: Py; 
v_j = t1254;
out.push(py::pow(&py::sub(&py::getitem(&v_r.clone(), &v_j.clone())?, &py::getitem(&v_cen.clone(), &v_j.clone())?)?, &Py::Int(2i64))?); } Py::list(out) }, None)?); } Py::list(out) }, None)?)?;
    }
    let _ = f__need({ let t1255 = v_W.clone(); Py::Bool({ let t1256 = Py::Int(0i64); if py::gt(&t1255, &t1256)? { true } else { false } }) }, Py::str("the within-cluster dispersion is 0"))?;
    return Ok(f__out(Py::dict(vec![(Py::str("calinski_harabasz"), py::mul(&py::truediv(&v_B.clone(), &v_W.clone())?, &py::fraction(&[py::sub(&v_n.clone(), &v_k.clone())?, py::sub(&v_k.clone(), &Py::Int(1i64))?])?)?), (Py::str("between"), v_B.clone()), (Py::str("within"), v_W.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_ce_logits(mut v_z: Py, mut v_y: Py) -> R {
    let mut v_g: Py = Py::None;
    let mut v_p: Py = Py::None;
    let mut v_zv: Py = Py::None;
    v_zv = f__vec(Py::str("z"), v_z.clone(), Py::Int(2i64))?;
    let _ = f__need({ let t1701 = Py::Bool(matches!(v_y.clone(), Py::Int(_) | Py::Big(_) | Py::Bool(_))); if !t1701.truthy() { t1701 } else { { let t1698 = Py::Int(0i64); Py::Bool({ let t1699 = v_y.clone(); if py::le(&t1698, &t1699)? { { let t1700 = py::len(&v_zv.clone())?; if py::lt(&t1699, &t1700)? { true } else { false } } } else { false } }) } } }, Py::str("y must be the index of the true class (0 to K-1)"))?;
    v_p = f__softmax(v_zv.clone())?;
    v_g = Py::list((v_p.clone()).items()?);
    py::setitem(&v_g.clone(), &v_y.clone(), py::neg(&py::fsum(&{ let mut out = vec![]; for t1704 in py::range_iter(&[(py::len(&v_p.clone())?).i()?]) { let mut v_j: Py; 
v_j = t1704;
if ({ let t1705 = v_j.clone(); Py::Bool({ let t1706 = v_y.clone(); if py::ne(&t1705, &t1706) { true } else { false } }) }).truthy() { out.push(py::getitem(&v_p.clone(), &v_j.clone())?); }} Py::list(out) })?)?)?;
    return Ok(f__out(Py::dict(vec![(Py::str("loss"), py::neg(&py::getitem(&f__log_softmax(v_zv.clone())?, &v_y.clone())?)?), (Py::str("probability"), py::getitem(&v_p.clone(), &v_y.clone())?), (Py::str("gradient"), v_g.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_class_weights(mut v_y: Py) -> R {
    let mut v_cls: Py = Py::None;
    let mut v_k: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_yl: Py = Py::None;
    v_yl = f__labels(Py::str("y"), v_y.clone(), Py::Int(1i64))?;
    v_cls = py::sorted(&py::set(&v_yl.clone())?, (Py::Bool(false)).truthy())?;
    let t640 = (Py::tuple(vec![py::len(&v_yl.clone())?, py::len(&v_cls.clone())?])).items()?;
    if t640.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_n = t640[0].clone();
    v_k = t640[1].clone();
    return Ok(f__out(Py::dict(vec![(Py::str("classes"), v_cls.clone()), (Py::str("weights"), { let mut out = vec![]; for t653 in py::iter(&(v_cls.clone()))? { let mut v_c: Py; 
v_c = t653;
out.push(py::fraction(&[v_n.clone(), py::mul(&v_k.clone(), &py::method(&v_yl.clone(), "count", &[v_c.clone()])?)?])?); } Py::list(out) })]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_cohen_kappa(mut v_y1: Py, mut v_y2: Py, mut v_weights: Py) -> R {
    let mut v_O: Py = Py::None;
    let mut v_a: Py = Py::None;
    let mut v_b: Py = Py::None;
    let mut v_cls: Py = Py::None;
    let mut v_cols: Py = Py::None;
    let mut v_exp: Py = Py::None;
    let mut v_idx: Py = Py::None;
    let mut v_k: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_obs: Py = Py::None;
    let mut v_rows: Py = Py::None;
    let mut v_u: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_w: Py = Py::None;
    let t1377 = (Py::tuple(vec![f__labels(Py::str("y1"), v_y1.clone(), Py::Int(1i64))?, f__labels(Py::str("y2"), v_y2.clone(), Py::Int(1i64))?])).items()?;
    if t1377.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_a = t1377[0].clone();
    v_b = t1377[1].clone();
    let _ = f__same_len(v_a.clone(), v_b.clone(), Py::str("y1"), Py::str("y2"))?;
    v_cls = f__classes(Py::tuple(vec![v_a.clone(), v_b.clone()]))?;
    v_k = py::len(&v_cls.clone())?;
    let _ = f__need({ let t1381 = v_k.clone(); Py::Bool({ let t1382 = Py::Int(2i64); if py::ge(&t1381, &t1382)? { true } else { false } }) }, Py::str("kappa needs at least two classes"))?;
    v_idx = { let mut out = vec![]; for t1383 in py::iter(&(py::enumerate(&v_cls.clone(), 0)?))? { let mut v_c: Py; let mut v_i: Py; 
let t1384 = (t1383).items()?;
if t1384.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_i = t1384[0].clone();
v_c = t1384[1].clone();
out.push((v_c.clone(), v_i.clone())); } Py::dict(out) };
    v_n = py::len(&v_a.clone())?;
    v_O = { let mut out = vec![]; for t1385 in py::range_iter(&[(v_k.clone()).i()?]) { let mut v__: Py; 
v__ = t1385;
out.push(py::mul(&Py::list(vec![Py::Int(0i64)]), &v_k.clone())?); } Py::list(out) };
    for t1386 in py::iter(&(py::zip(&[v_a.clone(), v_b.clone()])?))? {
        let t1389 = (t1386).items()?;
        if t1389.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_u = t1389[0].clone();
        v_v = t1389[1].clone();
        let t1387 = py::getitem(&v_O.clone(), &py::getitem(&v_idx.clone(), &v_u.clone())?)?; let t1388 = py::getitem(&v_idx.clone(), &v_v.clone())?;
        py::setitem(&t1387, &t1388, py::iadd(&py::getitem(&t1387, &t1388)?, &Py::Int(1i64))?)?;
    }
    v_rows = { let mut out = vec![]; for t1390 in py::iter(&(v_O.clone()))? { let mut v_r: Py; 
v_r = t1390;
out.push(py::sum(&v_r.clone(), None)?); } Py::list(out) };
    v_cols = { let mut out = vec![]; for t1391 in py::range_iter(&[(v_k.clone()).i()?]) { let mut v_j: Py; 
v_j = t1391;
out.push(py::sum(&{ let mut out = vec![]; for t1392 in py::range_iter(&[(v_k.clone()).i()?]) { let mut v_i: Py; 
v_i = t1392;
out.push(py::getitem(&py::getitem(&v_O.clone(), &v_i.clone())?, &v_j.clone())?); } Py::list(out) }, None)?); } Py::list(out) };
    v_w = { let v_O = v_O.clone(); let v_a = v_a.clone(); let v_b = v_b.clone(); let v_cls = v_cls.clone(); let v_cols = v_cols.clone(); let v_idx = v_idx.clone(); let v_k = v_k.clone(); let v_n = v_n.clone(); let v_rows = v_rows.clone(); let v_u = v_u.clone(); let v_v = v_v.clone(); let v_w = v_w.clone(); let v_weights = v_weights.clone(); let v_y1 = v_y1.clone(); let v_y2 = v_y2.clone(); py::func(move |a: &[Py]| -> R {
            #[allow(unused_mut, unused_variables, unused_assignments)]
{
            let mut v_i: Py = a.get(0).cloned().unwrap_or(Py::None);
            let mut v_j: Py = a.get(1).cloned().unwrap_or(Py::None);
            if ({ let t1393 = v_weights.clone(); Py::Bool({ let t1394 = Py::str("none"); if py::eq(&t1393, &t1394) { true } else { false } }) }).truthy() {
                return Ok((if ({ let t1395 = v_i.clone(); Py::Bool({ let t1396 = v_j.clone(); if py::eq(&t1395, &t1396) { true } else { false } }) }).truthy() { Py::Int(0i64) } else { Py::Int(1i64) }));
            }
            return Ok((if ({ let t1397 = v_weights.clone(); Py::Bool({ let t1398 = Py::str("linear"); if py::eq(&t1397, &t1398) { true } else { false } }) }).truthy() { py::abs(&py::sub(&v_i.clone(), &v_j.clone())?)? } else { py::pow(&py::sub(&v_i.clone(), &v_j.clone())?, &Py::Int(2i64))? }));
            #[allow(unreachable_code)]
            Ok(Py::None)
        }
        }) };
    v_obs = py::sum(&{ let mut out = vec![]; for t1399 in py::range_iter(&[(v_k.clone()).i()?]) { let mut v_i: Py; 
v_i = t1399;
for t1400 in py::range_iter(&[(v_k.clone()).i()?]) { let mut v_j: Py; 
v_j = t1400;
out.push(py::mul(&py::call(&v_w.clone(), &[v_i.clone(), v_j.clone()])?, &py::getitem(&py::getitem(&v_O.clone(), &v_i.clone())?, &v_j.clone())?)?); }} Py::list(out) }, None)?;
    v_exp = py::sum(&{ let mut out = vec![]; for t1401 in py::range_iter(&[(v_k.clone()).i()?]) { let mut v_i: Py; 
v_i = t1401;
for t1402 in py::range_iter(&[(v_k.clone()).i()?]) { let mut v_j: Py; 
v_j = t1402;
out.push(py::mul(&py::call(&v_w.clone(), &[v_i.clone(), v_j.clone()])?, &py::fraction(&[py::mul(&py::getitem(&v_rows.clone(), &v_i.clone())?, &py::getitem(&v_cols.clone(), &v_j.clone())?)?, v_n.clone()])?)?); }} Py::list(out) }, None)?;
    let _ = f__need({ let t1403 = v_exp.clone(); Py::Bool({ let t1404 = Py::Int(0i64); if py::ne(&t1403, &t1404) { true } else { false } }) }, Py::str("kappa is undefined: the expected disagreement is 0 (both raters use one class)"))?;
    return Ok(f__out(Py::dict(vec![(Py::str("kappa"), py::sub(&Py::Int(1i64), &py::truediv(&py::fraction(&[v_obs.clone()])?, &v_exp.clone())?)?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_conv2d(mut v_image: Py, mut v_kernel: Py, mut v_stride: Py, mut v_padding: Py) -> R {
    let mut v_H: Py = Py::None;
    let mut v_I: Py = Py::None;
    let mut v_K: Py = Py::None;
    let mut v_KF: Py = Py::None;
    let mut v_P: Py = Py::None;
    let mut v_W: Py = Py::None;
    let mut v_i: Py = Py::None;
    let mut v_j: Py = Py::None;
    let mut v_kh: Py = Py::None;
    let mut v_kw: Py = Py::None;
    let mut v_oh: Py = Py::None;
    let mut v_out: Py = Py::None;
    let mut v_ow: Py = Py::None;
    let t1463 = (Py::tuple(vec![f__mat(Py::str("image"), v_image.clone(), Py::Int(1i64), Py::Int(1i64))?, f__mat(Py::str("kernel"), v_kernel.clone(), Py::Int(1i64), Py::Int(1i64))?])).items()?;
    if t1463.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_I = t1463[0].clone();
    v_K = t1463[1].clone();
    let _ = f__need({ let t1466 = Py::Bool(matches!(v_stride.clone(), Py::Int(_) | Py::Big(_) | Py::Bool(_))); if !t1466.truthy() { t1466 } else { { let t1464 = v_stride.clone(); Py::Bool({ let t1465 = Py::Int(1i64); if py::ge(&t1464, &t1465)? { true } else { false } }) } } }, Py::str("stride must be a whole number >= 1"))?;
    let _ = f__need({ let t1469 = Py::Bool(matches!(v_padding.clone(), Py::Int(_) | Py::Big(_) | Py::Bool(_))); if !t1469.truthy() { t1469 } else { { let t1467 = v_padding.clone(); Py::Bool({ let t1468 = Py::Int(0i64); if py::ge(&t1467, &t1468)? { true } else { false } }) } } }, Py::str("padding must be a whole number >= 0"))?;
    let _ = f__need({ let t1470 = v_padding.clone(); Py::Bool({ let t1471 = Py::Int(100i64); if py::le(&t1470, &t1471)? { true } else { false } }) }, Py::str("padding must be at most 100"))?;
    let t1472 = (Py::tuple(vec![py::len(&v_I.clone())?, py::len(&py::getitem(&v_I.clone(), &Py::Int(0i64))?)?, py::len(&v_K.clone())?, py::len(&py::getitem(&v_K.clone(), &Py::Int(0i64))?)?])).items()?;
    if t1472.len() != 4 { return Err(PyErr::Value("unpack".into())); }
    v_H = t1472[0].clone();
    v_W = t1472[1].clone();
    v_kh = t1472[2].clone();
    v_kw = t1472[3].clone();
    let _ = f__need({ let t1477 = { let t1473 = py::add(&v_H.clone(), &py::mul(&Py::Int(2i64), &v_padding.clone())?)?; Py::Bool({ let t1474 = v_kh.clone(); if py::ge(&t1473, &t1474)? { true } else { false } }) }; if !t1477.truthy() { t1477 } else { { let t1475 = py::add(&v_W.clone(), &py::mul(&Py::Int(2i64), &v_padding.clone())?)?; Py::Bool({ let t1476 = v_kw.clone(); if py::ge(&t1475, &t1476)? { true } else { false } }) } } }, Py::str("the kernel is larger than the padded image"))?;
    v_P = { let mut out = vec![]; for t1478 in py::range_iter(&[(py::add(&v_H.clone(), &py::mul(&Py::Int(2i64), &v_padding.clone())?)?).i()?]) { let mut v__: Py; 
v__ = t1478;
out.push(py::mul(&Py::list(vec![py::fraction(&[Py::Int(0i64)])?]), &py::add(&v_W.clone(), &py::mul(&Py::Int(2i64), &v_padding.clone())?)?)?); } Py::list(out) };
    for t1479 in py::range_iter(&[(v_H.clone()).i()?]) {
        v_i = t1479;
        for t1480 in py::range_iter(&[(v_W.clone()).i()?]) {
            v_j = t1480;
            py::setitem(&py::getitem(&v_P.clone(), &py::add(&v_i.clone(), &v_padding.clone())?)?, &py::add(&v_j.clone(), &v_padding.clone())?, f__F(py::getitem(&py::getitem(&v_I.clone(), &v_i.clone())?, &v_j.clone())?)?)?;
        }
    }
    v_KF = { let mut out = vec![]; for t1481 in py::iter(&(v_K.clone()))? { let mut v_r: Py; 
v_r = t1481;
out.push({ let mut out = vec![]; for t1482 in py::iter(&(v_r.clone()))? { let mut v_v: Py; 
v_v = t1482;
out.push(f__F(v_v.clone())?); } Py::list(out) }); } Py::list(out) };
    let t1483 = (Py::tuple(vec![py::add(&py::floordiv(&py::sub(&py::add(&v_H.clone(), &py::mul(&Py::Int(2i64), &v_padding.clone())?)?, &v_kh.clone())?, &v_stride.clone())?, &Py::Int(1i64))?, py::add(&py::floordiv(&py::sub(&py::add(&v_W.clone(), &py::mul(&Py::Int(2i64), &v_padding.clone())?)?, &v_kw.clone())?, &v_stride.clone())?, &Py::Int(1i64))?])).items()?;
    if t1483.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_oh = t1483[0].clone();
    v_ow = t1483[1].clone();
    v_out = { let mut out = vec![]; for t1484 in py::range_iter(&[(v_oh.clone()).i()?]) { let mut v_i: Py; 
v_i = t1484;
out.push({ let mut out = vec![]; for t1485 in py::range_iter(&[(v_ow.clone()).i()?]) { let mut v_j: Py; 
v_j = t1485;
out.push(py::sum(&{ let mut out = vec![]; for t1486 in py::range_iter(&[(v_kh.clone()).i()?]) { let mut v_a: Py; 
v_a = t1486;
for t1487 in py::range_iter(&[(v_kw.clone()).i()?]) { let mut v_b: Py; 
v_b = t1487;
out.push(py::mul(&py::getitem(&py::getitem(&v_P.clone(), &py::add(&py::mul(&v_i.clone(), &v_stride.clone())?, &v_a.clone())?)?, &py::add(&py::mul(&v_j.clone(), &v_stride.clone())?, &v_b.clone())?)?, &py::getitem(&py::getitem(&v_KF.clone(), &v_a.clone())?, &v_b.clone())?)?); }} Py::list(out) }, None)?); } Py::list(out) }); } Py::list(out) };
    return Ok(f__out(Py::dict(vec![(Py::str("output"), v_out.clone()), (Py::str("output_sum"), py::sum(&{ let mut out = vec![]; for t1500 in py::iter(&(v_out.clone()))? { let mut v_r: Py; 
v_r = t1500;
out.push(py::sum(&v_r.clone(), None)?); } Py::list(out) }, None)?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_conv_out(mut v_n: Py, mut v_kernel: Py, mut v_stride: Py, mut v_padding: Py, mut v_dilation: Py) -> R {
    let mut v_eff: Py = Py::None;
    let mut v_lo: Py = Py::None;
    let mut v_nm: Py = Py::None;
    let mut v_v: Py = Py::None;
    for t1530 in py::iter(&(Py::tuple(vec![Py::tuple(vec![Py::str("n"), v_n.clone(), Py::Int(1i64)]), Py::tuple(vec![Py::str("kernel"), v_kernel.clone(), Py::Int(1i64)]), Py::tuple(vec![Py::str("stride"), v_stride.clone(), Py::Int(1i64)]), Py::tuple(vec![Py::str("padding"), v_padding.clone(), Py::Int(0i64)]), Py::tuple(vec![Py::str("dilation"), v_dilation.clone(), Py::Int(1i64)])])))? {
        let t1534 = (t1530).items()?;
        if t1534.len() != 3 { return Err(PyErr::Value("unpack".into())); }
        v_nm = t1534[0].clone();
        v_v = t1534[1].clone();
        v_lo = t1534[2].clone();
        let _ = f__need({ let t1533 = Py::Bool(matches!(v_v.clone(), Py::Int(_) | Py::Big(_) | Py::Bool(_))); if !t1533.truthy() { t1533 } else { { let t1531 = v_v.clone(); Py::Bool({ let t1532 = v_lo.clone(); if py::ge(&t1531, &t1532)? { true } else { false } }) } } }, Py::str(&[py::format(&v_nm.clone(), "")?, " must be a whole number >= ".to_string(), py::format(&v_lo.clone(), "")?].concat()))?;
    }
    v_eff = py::add(&py::mul(&v_dilation.clone(), &py::sub(&v_kernel.clone(), &Py::Int(1i64))?)?, &Py::Int(1i64))?;
    let _ = f__need({ let t1535 = py::add(&v_n.clone(), &py::mul(&Py::Int(2i64), &v_padding.clone())?)?; Py::Bool({ let t1536 = v_eff.clone(); if py::ge(&t1535, &t1536)? { true } else { false } }) }, Py::str("the (dilated) kernel is larger than the padded input"))?;
    return Ok(f__out(Py::dict(vec![(Py::str("output_size"), py::add(&py::floordiv(&py::sub(&py::add(&v_n.clone(), &py::mul(&Py::Int(2i64), &v_padding.clone())?)?, &v_eff.clone())?, &v_stride.clone())?, &Py::Int(1i64))?), (Py::str("effective_kernel"), v_eff.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_conv_params(mut v_in_channels: Py, mut v_out_channels: Py, mut v_kernel_h: Py, mut v_kernel_w: Py, mut v_groups: Py, mut v_bias: Py) -> R {
    let mut v_nm: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_w: Py = Py::None;
    for t1578 in py::iter(&(Py::tuple(vec![Py::tuple(vec![Py::str("in_channels"), v_in_channels.clone()]), Py::tuple(vec![Py::str("out_channels"), v_out_channels.clone()]), Py::tuple(vec![Py::str("kernel_h"), v_kernel_h.clone()]), Py::tuple(vec![Py::str("kernel_w"), v_kernel_w.clone()]), Py::tuple(vec![Py::str("groups"), v_groups.clone()])])))? {
        let t1582 = (t1578).items()?;
        if t1582.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_nm = t1582[0].clone();
        v_v = t1582[1].clone();
        let _ = f__need({ let t1581 = Py::Bool(matches!(v_v.clone(), Py::Int(_) | Py::Big(_) | Py::Bool(_))); if !t1581.truthy() { t1581 } else { { let t1579 = v_v.clone(); Py::Bool({ let t1580 = Py::Int(1i64); if py::ge(&t1579, &t1580)? { true } else { false } }) } } }, Py::str(&[py::format(&v_nm.clone(), "")?, " must be a whole number >= 1".to_string()].concat()))?;
    }
    let _ = f__need({ let t1587 = { let t1583 = py::modulo(&v_in_channels.clone(), &v_groups.clone())?; Py::Bool({ let t1584 = Py::Int(0i64); if py::eq(&t1583, &t1584) { true } else { false } }) }; if !t1587.truthy() { t1587 } else { { let t1585 = py::modulo(&v_out_channels.clone(), &v_groups.clone())?; Py::Bool({ let t1586 = Py::Int(0i64); if py::eq(&t1585, &t1586) { true } else { false } }) } } }, Py::str("in_channels and out_channels must both be divisible by groups"))?;
    v_w = py::mul(&py::mul(&py::mul(&v_out_channels.clone(), &py::floordiv(&v_in_channels.clone(), &v_groups.clone())?)?, &v_kernel_h.clone())?, &v_kernel_w.clone())?;
    return Ok(f__out(Py::dict(vec![(Py::str("weights"), v_w.clone()), (Py::str("biases"), (if (v_bias.clone()).truthy() { v_out_channels.clone() } else { Py::Int(0i64) })), (Py::str("total"), py::add(&v_w.clone(), &(if (v_bias.clone()).truthy() { v_out_channels.clone() } else { Py::Int(0i64) }))?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_cosine_lr(mut v_lr_max: Py, mut v_eta_min: Py, mut v_T_max: Py, mut v_epoch: Py) -> R {
    let _ = Py::tuple(vec![f__fin(Py::str("lr_max"), Py::tuple(vec![v_lr_max.clone()]))?, f__fin(Py::str("eta_min"), Py::tuple(vec![v_eta_min.clone()]))?]);
    let _ = f__need({ let t1637 = { let t1632 = v_lr_max.clone(); Py::Bool({ let t1633 = Py::Int(0i64); if py::gt(&t1632, &t1633)? { true } else { false } }) }; if !t1637.truthy() { t1637 } else { { let t1634 = Py::Int(0i64); Py::Bool({ let t1635 = v_eta_min.clone(); if py::le(&t1634, &t1635)? { { let t1636 = v_lr_max.clone(); if py::lt(&t1635, &t1636)? { true } else { false } } } else { false } }) } } }, Py::str("need lr_max > 0 and 0 <= eta_min < lr_max"))?;
    let _ = f__need({ let t1646 = Py::Bool(matches!(v_T_max.clone(), Py::Int(_) | Py::Big(_) | Py::Bool(_))); if !t1646.truthy() { t1646 } else { { let t1645 = { let t1638 = Py::Int(1i64); Py::Bool({ let t1639 = v_T_max.clone(); if py::le(&t1638, &t1639)? { { let t1640 = Py::Int(1000i64); if py::le(&t1639, &t1640)? { true } else { false } } } else { false } }) }; if !t1645.truthy() { t1645 } else { { let t1644 = Py::Bool(matches!(v_epoch.clone(), Py::Int(_) | Py::Big(_) | Py::Bool(_))); if !t1644.truthy() { t1644 } else { { let t1641 = Py::Int(0i64); Py::Bool({ let t1642 = v_epoch.clone(); if py::le(&t1641, &t1642)? { { let t1643 = v_T_max.clone(); if py::le(&t1642, &t1643)? { true } else { false } } } else { false } }) } } } } } } }, Py::str("1 <= T_max <= 1000 and 0 <= epoch <= T_max (whole numbers)"))?;
    if ({ let t1647 = v_epoch.clone(); Py::Bool({ let t1648 = v_T_max.clone(); if py::eq(&t1647, &t1648) { true } else { false } }) }).truthy() {
        return Ok(f__out(Py::dict(vec![(Py::str("lr"), v_eta_min.clone())]))?);
    }
    return Ok(f__out(Py::dict(vec![(Py::str("lr"), py::add(&v_eta_min.clone(), &py::truediv(&py::mul(&py::sub(&v_lr_max.clone(), &v_eta_min.clone())?, &py::add(&Py::Int(1i64), &py::math1("cos", &py::truediv(&py::mul(&Py::Float(std::f64::consts::PI), &v_epoch.clone())?, &v_T_max.clone())?)?)?)?, &Py::Int(2i64))?)?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_davies_bouldin(mut v_X: Py, mut v_labels: Py) -> R {
    let mut v_C: Py = Py::None;
    let mut v_M: Py = Py::None;
    let mut v_S: Py = Py::None;
    let mut v_d: Py = Py::None;
    let mut v_k: Py = Py::None;
    let mut v_ks: Py = Py::None;
    let mut v_lab: Py = Py::None;
    let mut v_m: Py = Py::None;
    let mut v_r: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t1775 = (f__points(v_X.clone(), v_labels.clone())?).items()?;
    if t1775.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_M = t1775[0].clone();
    v_lab = t1775[1].clone();
    let t1782 = (f__centroids(v_M.clone(), v_lab.clone())?).items()?;
    if t1782.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_ks = t1782[0].clone();
    v_C = t1782[1].clone();
    let _ = f__need({ let t1783 = Py::Int(2i64); Py::Bool({ let t1784 = py::len(&v_ks.clone())?; if py::le(&t1783, &t1784)? { { let t1785 = py::sub(&py::len(&v_M.clone())?, &Py::Int(1i64))?; if py::le(&t1784, &t1785)? { true } else { false } } } else { false } }) }, Py::str("the Davies-Bouldin index needs between 2 and n - 1 clusters"))?;
    v_S = { let mut out = vec![]; for t1786 in py::iter(&(v_ks.clone()))? { let mut v_k: Py; 
v_k = t1786;
out.push((v_k.clone(), py::truediv(&py::fsum(&{ let mut out = vec![]; for t1787 in py::range_iter(&[(py::len(&v_M.clone())?).i()?]) { let mut v_i: Py; 
v_i = t1787;
if ({ let t1788 = py::getitem(&v_lab.clone(), &v_i.clone())?; Py::Bool({ let t1789 = v_k.clone(); if py::eq(&t1788, &t1789) { true } else { false } }) }).truthy() { out.push(py::math1("sqrt", &f__sqd(py::getitem(&v_M.clone(), &v_i.clone())?, py::getitem(&v_C.clone(), &v_k.clone())?)?)?); }} Py::list(out) })?, &py::method(&v_lab.clone(), "count", &[v_k.clone()])?)?)); } Py::dict(out) };
    v_worst = Py::list(vec![]);
    for t1792 in py::iter(&(v_ks.clone()))? {
        v_k = t1792;
        v_r = Py::list(vec![]);
        for t1793 in py::iter(&(v_ks.clone()))? {
            v_m = t1793;
            if ({ let t1794 = v_m.clone(); Py::Bool({ let t1795 = v_k.clone(); if py::ne(&t1794, &t1795) { true } else { false } }) }).truthy() {
                v_d = py::math1("sqrt", &f__sqd(py::getitem(&v_C.clone(), &v_k.clone())?, py::getitem(&v_C.clone(), &v_m.clone())?)?)?;
                let _ = f__need({ let t1796 = v_d.clone(); Py::Bool({ let t1797 = Py::Int(0i64); if py::gt(&t1796, &t1797)? { true } else { false } }) }, Py::str("two clusters have the same centroid"))?;
                let _ = py::method(&v_r.clone(), "append", &[py::truediv(&py::add(&py::getitem(&v_S.clone(), &v_k.clone())?, &py::getitem(&v_S.clone(), &v_m.clone())?)?, &v_d.clone())?])?;
            }
        }
        let _ = py::method(&v_worst.clone(), "append", &[py::minmax(&[v_r.clone()], true)?])?;
    }
    return Ok(f__out(Py::dict(vec![(Py::str("davies_bouldin"), py::truediv(&py::fsum(&v_worst.clone())?, &py::len(&v_ks.clone())?)?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_dense_forward(mut v_x: Py, mut v_W1: Py, mut v_b1: Py, mut v_activation: Py, mut v_W2: Py, mut v_b2: Py, mut v_output: Py) -> R {
    let mut v_A: Py = Py::None;
    let mut v_B: Py = Py::None;
    let mut v__: Py = Py::None;
    let mut v_c1: Py = Py::None;
    let mut v_c2: Py = Py::None;
    let mut v_h: Py = Py::None;
    let mut v_xv: Py = Py::None;
    let mut v_y: Py = Py::None;
    let mut v_z2: Py = Py::None;
    let t2078 = (f__net_inputs(v_x.clone(), v_W1.clone(), v_b1.clone(), v_W2.clone(), v_b2.clone())?).items()?;
    if t2078.len() != 5 { return Err(PyErr::Value("unpack".into())); }
    v_xv = t2078[0].clone();
    v_A = t2078[1].clone();
    v_c1 = t2078[2].clone();
    v_B = t2078[3].clone();
    v_c2 = t2078[4].clone();
    let t2096 = (f__forward(v_xv.clone(), v_A.clone(), v_c1.clone(), v_B.clone(), v_c2.clone(), v_activation.clone())?).items()?;
    if t2096.len() != 3 { return Err(PyErr::Value("unpack".into())); }
    v__ = t2096[0].clone();
    v_h = t2096[1].clone();
    v_z2 = t2096[2].clone();
    if ({ let t2097 = v_output.clone(); Py::Bool({ let t2098 = Py::str("sigmoid"); if py::eq(&t2097, &t2098) { true } else { false } }) }).truthy() {
        v_y = { let mut out = vec![]; for t2099 in py::iter(&(v_z2.clone()))? { let mut v_v: Py; 
v_v = t2099;
out.push(f__sigmoid(v_v.clone())?); } Py::list(out) };
    } else {
        if ({ let t2100 = v_output.clone(); Py::Bool({ let t2101 = Py::str("softmax"); if py::eq(&t2100, &t2101) { true } else { false } }) }).truthy() {
            v_y = f__softmax(v_z2.clone())?;
        } else {
            v_y = v_z2.clone();
        }
    }
    return Ok(f__out(Py::dict(vec![(Py::str("hidden"), v_h.clone()), (Py::str("output"), v_y.clone()), (Py::str("max_output"), py::minmax(&[v_y.clone()], true)?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_dense_params(mut v_sizes: Py, mut v_bias: Py) -> R {
    let mut v_per: Py = Py::None;
    let mut v_s: Py = Py::None;
    v_s = f__labels(Py::str("sizes"), v_sizes.clone(), Py::Int(2i64))?;
    let _ = f__need(Py::Bool(({ let mut out = vec![]; for t2154 in py::iter(&(v_s.clone()))? { let mut v_v: Py; 
v_v = t2154;
out.push({ let t2155 = v_v.clone(); Py::Bool({ let t2156 = Py::Int(1i64); if py::ge(&t2155, &t2156)? { true } else { false } }) }); } Py::list(out) }).items()?.iter().all(|x| x.truthy())), Py::str("layer sizes must be positive"))?;
    v_per = { let mut out = vec![]; for t2157 in py::iter(&(py::zip(&[v_s.clone(), py::slice(&v_s.clone(), Some((Py::Int(1i64)).i()?), None, None)?])?))? { let mut v_a: Py; let mut v_b: Py; 
let t2158 = (t2157).items()?;
if t2158.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t2158[0].clone();
v_b = t2158[1].clone();
out.push(py::add(&py::mul(&v_a.clone(), &v_b.clone())?, &(if (v_bias.clone()).truthy() { v_b.clone() } else { Py::Int(0i64) }))?); } Py::list(out) };
    return Ok(f__out(Py::dict(vec![(Py::str("per_layer"), v_per.clone()), (Py::str("total"), py::sum(&v_per.clone(), None)?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_distances(mut v_x: Py, mut v_z: Py) -> R {
    let mut v_a: Py = Py::None;
    let mut v_b: Py = Py::None;
    let mut v_cos: Py = Py::None;
    let mut v_na: Py = Py::None;
    let mut v_nb: Py = Py::None;
    let t2208 = (Py::tuple(vec![f__vec(Py::str("x"), v_x.clone(), Py::Int(1i64))?, f__vec(Py::str("z"), v_z.clone(), Py::Int(1i64))?])).items()?;
    if t2208.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_a = t2208[0].clone();
    v_b = t2208[1].clone();
    let _ = f__same_len(v_a.clone(), v_b.clone(), Py::str("x"), Py::str("z"))?;
    v_na = py::math1("sqrt", &py::fsum(&{ let mut out = vec![]; for t2211 in py::iter(&(v_a.clone()))? { let mut v_v: Py; 
v_v = t2211;
out.push(py::mul(&v_v.clone(), &v_v.clone())?); } Py::list(out) })?)?;
    v_nb = py::math1("sqrt", &py::fsum(&{ let mut out = vec![]; for t2212 in py::iter(&(v_b.clone()))? { let mut v_v: Py; 
v_v = t2212;
out.push(py::mul(&v_v.clone(), &v_v.clone())?); } Py::list(out) })?)?;
    let _ = f__need({ let t2217 = { let t2213 = v_na.clone(); Py::Bool({ let t2214 = Py::Int(0i64); if py::gt(&t2213, &t2214)? { true } else { false } }) }; if !t2217.truthy() { t2217 } else { { let t2215 = v_nb.clone(); Py::Bool({ let t2216 = Py::Int(0i64); if py::gt(&t2215, &t2216)? { true } else { false } }) } } }, Py::str("cosine similarity needs two non-zero vectors"))?;
    v_cos = py::truediv(&py::fsum(&{ let mut out = vec![]; for t2218 in py::iter(&(py::zip(&[v_a.clone(), v_b.clone()])?))? { let mut v_u: Py; let mut v_v: Py; 
let t2219 = (t2218).items()?;
if t2219.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_u = t2219[0].clone();
v_v = t2219[1].clone();
out.push(py::mul(&v_u.clone(), &v_v.clone())?); } Py::list(out) })?, &py::mul(&v_na.clone(), &v_nb.clone())?)?;
    return Ok(f__out(Py::dict(vec![(Py::str("euclidean"), py::math1("sqrt", &py::fsum(&{ let mut out = vec![]; for t2232 in py::iter(&(py::zip(&[v_a.clone(), v_b.clone()])?))? { let mut v_u: Py; let mut v_v: Py; 
let t2233 = (t2232).items()?;
if t2233.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_u = t2233[0].clone();
v_v = t2233[1].clone();
out.push(py::pow(&py::sub(&v_u.clone(), &v_v.clone())?, &Py::Int(2i64))?); } Py::list(out) })?)?), (Py::str("manhattan"), py::fsum(&{ let mut out = vec![]; for t2234 in py::iter(&(py::zip(&[v_a.clone(), v_b.clone()])?))? { let mut v_u: Py; let mut v_v: Py; 
let t2235 = (t2234).items()?;
if t2235.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_u = t2235[0].clone();
v_v = t2235[1].clone();
out.push(py::abs(&py::sub(&v_u.clone(), &v_v.clone())?)?); } Py::list(out) })?), (Py::str("chebyshev"), py::minmax(&[{ let mut out = vec![]; for t2236 in py::iter(&(py::zip(&[v_a.clone(), v_b.clone()])?))? { let mut v_u: Py; let mut v_v: Py; 
let t2237 = (t2236).items()?;
if t2237.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_u = t2237[0].clone();
v_v = t2237[1].clone();
out.push(py::abs(&py::sub(&v_u.clone(), &v_v.clone())?)?); } Py::list(out) }], true)?), (Py::str("cosine_similarity"), v_cos.clone()), (Py::str("cosine_distance"), py::sub(&Py::Int(1i64), &v_cos.clone())?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_ece(mut v_y_true: Py, mut v_p: Py, mut v_n_bins: Py) -> R {
    let mut v_acc: Py = Py::None;
    let mut v_b: Py = Py::None;
    let mut v_bins: Py = Py::None;
    let mut v_conf: Py = Py::None;
    let mut v_gaps: Py = Py::None;
    let mut v_items: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_pv: Py = Py::None;
    let mut v_q: Py = Py::None;
    let mut v_y: Py = Py::None;
    let mut v_yt: Py = Py::None;
    v_yt = f__binary(Py::str("y_true"), v_y_true.clone(), Py::Int(1i64))?;
    v_pv = f__vec(Py::str("p"), v_p.clone(), Py::Int(1i64))?;
    let _ = f__same_len(v_yt.clone(), v_pv.clone(), Py::str("y_true"), Py::str("p"))?;
    let _ = f__need(Py::Bool(({ let mut out = vec![]; for t2345 in py::iter(&(v_pv.clone()))? { let mut v_q: Py; 
v_q = t2345;
out.push({ let t2346 = Py::Int(0i64); Py::Bool({ let t2347 = v_q.clone(); if py::le(&t2346, &t2347)? { { let t2348 = Py::Int(1i64); if py::le(&t2347, &t2348)? { true } else { false } } } else { false } }) }); } Py::list(out) }).items()?.iter().all(|x| x.truthy())), Py::str("probabilities must lie between 0 and 1"))?;
    let _ = f__need({ let t2349 = Py::Int(1i64); Py::Bool({ let t2350 = v_n_bins.clone(); if py::le(&t2349, &t2350)? { { let t2351 = Py::Int(100i64); if py::le(&t2350, &t2351)? { true } else { false } } } else { false } }) }, Py::str("n_bins must be between 1 and 100"))?;
    v_bins = Py::dict(vec![]);
    for t2352 in py::iter(&(py::zip(&[v_yt.clone(), v_pv.clone()])?))? {
        let t2353 = (t2352).items()?;
        if t2353.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_y = t2353[0].clone();
        v_q = t2353[1].clone();
        v_b = py::minmax(&[py::int(&py::math1("floor", &py::mul(&f__F(v_q.clone())?, &v_n_bins.clone())?)?)?, py::sub(&v_n_bins.clone(), &Py::Int(1i64))?], false)?;
        let _ = py::method(&py::method(&v_bins.clone(), "setdefault", &[v_b.clone(), Py::list(vec![])])?, "append", &[Py::tuple(vec![v_y.clone(), f__F(v_q.clone())?])])?;
    }
    v_n = py::len(&v_yt.clone())?;
    v_gaps = Py::list(vec![]);
    for t2354 in py::iter(&(py::sorted(&py::method(&v_bins.clone(), "items", &[])?, (Py::Bool(false)).truthy())?))? {
        let t2359 = (t2354).items()?;
        if t2359.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_b = t2359[0].clone();
        v_items = t2359[1].clone();
        v_acc = py::fraction(&[py::sum(&{ let mut out = vec![]; for t2355 in py::iter(&(v_items.clone()))? { let mut v__: Py; let mut v_y: Py; 
let t2356 = (t2355).items()?;
if t2356.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_y = t2356[0].clone();
v__ = t2356[1].clone();
out.push(v_y.clone()); } Py::list(out) }, None)?, py::len(&v_items.clone())?])?;
        v_conf = py::truediv(&py::sum(&{ let mut out = vec![]; for t2357 in py::iter(&(v_items.clone()))? { let mut v__: Py; let mut v_q: Py; 
let t2358 = (t2357).items()?;
if t2358.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v__ = t2358[0].clone();
v_q = t2358[1].clone();
out.push(v_q.clone()); } Py::list(out) }, None)?, &py::len(&v_items.clone())?)?;
        let _ = py::method(&v_gaps.clone(), "append", &[Py::tuple(vec![py::fraction(&[py::len(&v_items.clone())?, v_n.clone()])?, py::abs(&py::sub(&v_acc.clone(), &v_conf.clone())?)?])])?;
    }
    return Ok(f__out(Py::dict(vec![(Py::str("ece"), py::sum(&{ let mut out = vec![]; for t2372 in py::iter(&(v_gaps.clone()))? { let mut v_g: Py; let mut v_w: Py; 
let t2373 = (t2372).items()?;
if t2373.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_w = t2373[0].clone();
v_g = t2373[1].clone();
out.push(py::mul(&v_w.clone(), &v_g.clone())?); } Py::list(out) }, None)?), (Py::str("mce"), py::minmax(&[{ let mut out = vec![]; for t2374 in py::iter(&(v_gaps.clone()))? { let mut v__: Py; let mut v_g: Py; 
let t2375 = (t2374).items()?;
if t2375.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v__ = t2375[0].clone();
v_g = t2375[1].clone();
out.push(v_g.clone()); } Py::list(out) }], true)?), (Py::str("n_nonempty_bins"), py::len(&v_gaps.clone())?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_entropy(mut v_p: Py, mut v_unit: Py) -> R {
    let mut v_L: Py = Py::None;
    let mut v_P: Py = Py::None;
    let mut v_h: Py = Py::None;
    let mut v_k: Py = Py::None;
    v_P = f__vec(Py::str("p"), v_p.clone(), Py::Int(1i64))?;
    let _ = f__need(Py::Bool(({ let mut out = vec![]; for t2275 in py::iter(&(v_P.clone()))? { let mut v_v: Py; 
v_v = t2275;
out.push({ let t2276 = v_v.clone(); Py::Bool({ let t2277 = Py::Int(0i64); if py::ge(&t2276, &t2277)? { true } else { false } }) }); } Py::list(out) }).items()?.iter().all(|x| x.truthy())), Py::str("probabilities must be >= 0"))?;
    let _ = f__need({ let t2278 = py::abs(&py::sub(&py::fsum(&v_P.clone())?, &Py::Int(1i64))?)?; Py::Bool({ let t2279 = Py::Float(1e-09); if py::le(&t2278, &t2279)? { true } else { false } }) }, Py::str("the probabilities must sum to 1"))?;
    v_h = f__H(v_P.clone())?;
    v_k = py::len(&v_P.clone())?;
    v_L = f__LN(v_unit.clone())?;
    return Ok(f__out(Py::dict(vec![(Py::str("entropy"), py::truediv(&v_h.clone(), &v_L.clone())?), (Py::str("perplexity"), py::math1("exp", &v_h.clone())?), (Py::str("max_entropy"), py::truediv(&py::math1("log", &v_k.clone())?, &v_L.clone())?), (Py::str("normalized_entropy"), (if ({ let t2295 = v_k.clone(); Py::Bool({ let t2296 = Py::Int(1i64); if py::gt(&t2295, &t2296)? { true } else { false } }) }).truthy() { py::truediv(&v_h.clone(), &py::math1("log", &v_k.clone())?)? } else { Py::Float(0.0) }))]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_fbeta(mut v_TP: Py, mut v_FP: Py, mut v_FN: Py, mut v_beta: Py) -> R {
    let mut v_P: Py = Py::None;
    let mut v_R: Py = Py::None;
    let mut v_b2: Py = Py::None;
    let _ = f__check_counts(v_TP.clone(), v_FP.clone(), v_FN.clone(), Py::Int(0i64))?;
    let _ = f__fin(Py::str("beta"), Py::tuple(vec![v_beta.clone()]))?;
    let _ = f__need({ let t2413 = v_beta.clone(); Py::Bool({ let t2414 = Py::Int(0i64); if py::gt(&t2413, &t2414)? { true } else { false } }) }, Py::str("beta must be positive"))?;
    let _ = f__need({ let t2419 = { let t2415 = py::add(&v_TP.clone(), &v_FP.clone())?; Py::Bool({ let t2416 = Py::Int(0i64); if py::gt(&t2415, &t2416)? { true } else { false } }) }; if !t2419.truthy() { t2419 } else { { let t2417 = py::add(&v_TP.clone(), &v_FN.clone())?; Py::Bool({ let t2418 = Py::Int(0i64); if py::gt(&t2417, &t2418)? { true } else { false } }) } } }, Py::str("precision or recall is undefined (no predicted or no actual positives)"))?;
    let _ = f__need({ let t2420 = v_TP.clone(); Py::Bool({ let t2421 = Py::Int(0i64); if py::gt(&t2420, &t2421)? { true } else { false } }) }, Py::str("precision and recall are both 0: the F-beta score is undefined (0/0)"))?;
    v_b2 = py::pow(&f__F(v_beta.clone())?, &Py::Int(2i64))?;
    let t2422 = (Py::tuple(vec![py::fraction(&[v_TP.clone(), py::add(&v_TP.clone(), &v_FP.clone())?])?, py::fraction(&[v_TP.clone(), py::add(&v_TP.clone(), &v_FN.clone())?])?])).items()?;
    if t2422.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_P = t2422[0].clone();
    v_R = t2422[1].clone();
    return Ok(f__out(Py::dict(vec![(Py::str("precision"), v_P.clone()), (Py::str("recall"), v_R.clone()), (Py::str("fbeta"), py::truediv(&py::mul(&py::mul(&py::add(&Py::Int(1i64), &v_b2.clone())?, &v_P.clone())?, &v_R.clone())?, &py::add(&py::mul(&v_b2.clone(), &v_P.clone())?, &v_R.clone())?)?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_focal(mut v_y: Py, mut v_z: Py, mut v_gamma: Py, mut v_alpha: Py) -> R {
    let mut v_G: Py = Py::None;
    let mut v_L: Py = Py::None;
    let mut v_at: Py = Py::None;
    let mut v_logpt: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_pt: Py = Py::None;
    let mut v_q: Py = Py::None;
    let mut v_s: Py = Py::None;
    let mut v_t: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_w: Py = Py::None;
    let mut v_yv: Py = Py::None;
    let mut v_zv: Py = Py::None;
    v_yv = f__binary(Py::str("y"), v_y.clone(), Py::Int(1i64))?;
    v_zv = f__vec(Py::str("z"), v_z.clone(), Py::Int(1i64))?;
    let _ = f__same_len(v_yv.clone(), v_zv.clone(), Py::str("y"), Py::str("z"))?;
    let _ = Py::tuple(vec![f__fin(Py::str("gamma"), Py::tuple(vec![v_gamma.clone()]))?, f__fin(Py::str("alpha"), Py::tuple(vec![v_alpha.clone()]))?]);
    let _ = f__need({ let t2483 = v_gamma.clone(); Py::Bool({ let t2484 = Py::Int(0i64); if py::ge(&t2483, &t2484)? { true } else { false } }) }, Py::str("gamma must be >= 0"))?;
    let _ = f__need({ let t2485 = Py::Int(0i64); Py::Bool({ let t2486 = v_alpha.clone(); if py::le(&t2485, &t2486)? { { let t2487 = Py::Int(1i64); if py::le(&t2486, &t2487)? { true } else { false } } } else { false } }) }, Py::str("alpha must lie between 0 and 1"))?;
    v_n = py::len(&v_yv.clone())?;
    let t2488 = (Py::tuple(vec![Py::list(vec![]), Py::list(vec![])])).items()?;
    if t2488.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_L = t2488[0].clone();
    v_G = t2488[1].clone();
    for t2489 in py::iter(&(py::zip(&[v_yv.clone(), v_zv.clone()])?))? {
        let t2498 = (t2489).items()?;
        if t2498.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_t = t2498[0].clone();
        v_v = t2498[1].clone();
        v_s = (if ({ let t2490 = v_t.clone(); Py::Bool({ let t2491 = Py::Int(1i64); if py::eq(&t2490, &t2491) { true } else { false } }) }).truthy() { Py::Int(1i64) } else { py::neg(&Py::Int(1i64))? });
        v_logpt = f__log_sigmoid(py::mul(&v_s.clone(), &v_v.clone())?)?;
        v_pt = f__sigmoid(py::mul(&v_s.clone(), &v_v.clone())?)?;
        v_q = f__sigmoid(py::mul(&py::neg(&v_s.clone())?, &v_v.clone())?)?;
        v_at = (if ({ let t2496 = v_t.clone(); Py::Bool({ let t2497 = Py::Int(1i64); if py::eq(&t2496, &t2497) { true } else { false } }) }).truthy() { v_alpha.clone() } else { py::sub(&Py::Int(1i64), &v_alpha.clone())? });
        v_w = py::pow(&v_q.clone(), &v_gamma.clone())?;
        let _ = py::method(&v_L.clone(), "append", &[py::mul(&py::mul(&py::neg(&v_at.clone())?, &v_w.clone())?, &v_logpt.clone())?])?;
        let _ = py::method(&v_G.clone(), "append", &[py::truediv(&py::mul(&py::mul(&py::mul(&v_s.clone(), &v_at.clone())?, &v_w.clone())?, &py::sub(&py::mul(&py::mul(&v_gamma.clone(), &v_pt.clone())?, &v_logpt.clone())?, &v_q.clone())?)?, &v_n.clone())?])?;
    }
    return Ok(f__out(Py::dict(vec![(Py::str("loss"), py::truediv(&py::fsum(&v_L.clone())?, &v_n.clone())?), (Py::str("gradient"), v_G.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_gaussian_nb(mut v_X: Py, mut v_y: Py, mut v_x: Py) -> R {
    let mut v_M: Py = Py::None;
    let mut v_c: Py = Py::None;
    let mut v_cls: Py = Py::None;
    let mut v_col: Py = Py::None;
    let mut v_j: Py = Py::None;
    let mut v_jll: Py = Py::None;
    let mut v_lp: Py = Py::None;
    let mut v_mu: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_q: Py = Py::None;
    let mut v_rows: Py = Py::None;
    let mut v_var: Py = Py::None;
    let mut v_yl: Py = Py::None;
    let t2574 = (f__nb_data(v_X.clone(), v_y.clone(), v_x.clone())?).items()?;
    if t2574.len() != 4 { return Err(PyErr::Value("unpack".into())); }
    v_M = t2574[0].clone();
    v_yl = t2574[1].clone();
    v_q = t2574[2].clone();
    v_cls = t2574[3].clone();
    v_n = py::len(&v_yl.clone())?;
    v_jll = Py::list(vec![]);
    for t2575 in py::iter(&(v_cls.clone()))? {
        v_c = t2575;
        v_rows = { let mut out = vec![]; for t2576 in py::range_iter(&[(v_n.clone()).i()?]) { let mut v_i: Py; 
v_i = t2576;
if ({ let t2577 = py::getitem(&v_yl.clone(), &v_i.clone())?; Py::Bool({ let t2578 = v_c.clone(); if py::eq(&t2577, &t2578) { true } else { false } }) }).truthy() { out.push(py::getitem(&v_M.clone(), &v_i.clone())?); }} Py::list(out) };
        v_lp = py::math1("log", &py::truediv(&py::len(&v_rows.clone())?, &v_n.clone())?)?;
        for t2579 in py::range_iter(&[(py::len(&v_q.clone())?).i()?]) {
            v_j = t2579;
            v_col = { let mut out = vec![]; for t2580 in py::iter(&(v_rows.clone()))? { let mut v_r: Py; 
v_r = t2580;
out.push(f__F(py::getitem(&v_r.clone(), &v_j.clone())?)?); } Py::list(out) };
            v_mu = py::truediv(&py::sum(&v_col.clone(), None)?, &py::len(&v_col.clone())?)?;
            v_var = py::truediv(&py::sum(&{ let mut out = vec![]; for t2581 in py::iter(&(v_col.clone()))? { let mut v_v: Py; 
v_v = t2581;
out.push(py::pow(&py::sub(&v_v.clone(), &v_mu.clone())?, &Py::Int(2i64))?); } Py::list(out) }, None)?, &py::len(&v_col.clone())?)?;
            let _ = f__need({ let t2582 = v_var.clone(); Py::Bool({ let t2583 = Py::Int(0i64); if py::gt(&t2582, &t2583)? { true } else { false } }) }, Py::str("a feature has zero variance within a class (the Gaussian likelihood is undefined)"))?;
            v_var = py::float(&v_var.clone())?;
            v_lp = py::iadd(&v_lp, &py::sub(&py::mul(&py::neg(&Py::Float(0.5))?, &py::math1("log", &py::mul(&py::mul(&Py::Int(2i64), &Py::Float(std::f64::consts::PI))?, &v_var.clone())?)?)?, &py::truediv(&py::float(&py::pow(&py::sub(&f__F(py::getitem(&v_q.clone(), &v_j.clone())?)?, &v_mu.clone())?, &Py::Int(2i64))?)?, &py::mul(&Py::Int(2i64), &v_var.clone())?)?)?)?;
        }
        let _ = py::method(&v_jll.clone(), "append", &[v_lp.clone()])?;
    }
    return Ok(f__posterior(v_cls.clone(), v_jll.clone())?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_gd_step_linear(mut v_X: Py, mut v_y: Py, mut v_w: Py, mut v_b: Py, mut v_lr: Py) -> R {
    let mut v_L: Py = Py::None;
    let mut v_M: Py = Py::None;
    let mut v_Mf: Py = Py::None;
    let mut v_Yf: Py = Py::None;
    let mut v_b2: Py = Py::None;
    let mut v_gb: Py = Py::None;
    let mut v_gw: Py = Py::None;
    let mut v_loss: Py = Py::None;
    let mut v_res: Py = Py::None;
    let mut v_w2: Py = Py::None;
    let mut v_wv: Py = Py::None;
    let mut v_yv: Py = Py::None;
    let t2720 = (f__linear_parts(v_X.clone(), v_y.clone(), v_w.clone(), v_b.clone())?).items()?;
    if t2720.len() != 3 { return Err(PyErr::Value("unpack".into())); }
    v_M = t2720[0].clone();
    v_yv = t2720[1].clone();
    v_wv = t2720[2].clone();
    let _ = f__fin(Py::str("lr"), Py::tuple(vec![v_lr.clone()]))?;
    let _ = f__need({ let t2721 = v_lr.clone(); Py::Bool({ let t2722 = Py::Int(0i64); if py::gt(&t2721, &t2722)? { true } else { false } }) }, Py::str("the learning rate must be positive"))?;
    let t2735 = (f__lin_loss(v_M.clone(), v_yv.clone(), v_wv.clone(), v_b.clone())?).items()?;
    if t2735.len() != 3 { return Err(PyErr::Value("unpack".into())); }
    v_loss = t2735[0].clone();
    v_gw = t2735[1].clone();
    v_gb = t2735[2].clone();
    v_L = f__F(v_lr.clone())?;
    v_w2 = { let mut out = vec![]; for t2736 in py::iter(&(py::zip(&[v_wv.clone(), v_gw.clone()])?))? { let mut v_c: Py; let mut v_g: Py; 
let t2737 = (t2736).items()?;
if t2737.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_c = t2737[0].clone();
v_g = t2737[1].clone();
out.push(py::sub(&f__F(v_c.clone())?, &py::mul(&v_L.clone(), &v_g.clone())?)?); } Py::list(out) };
    v_b2 = py::sub(&f__F(v_b.clone())?, &py::mul(&v_L.clone(), &v_gb.clone())?)?;
    let t2741 = (Py::tuple(vec![{ let mut out = vec![]; for t2738 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t2738;
out.push({ let mut out = vec![]; for t2739 in py::iter(&(v_r.clone()))? { let mut v_v: Py; 
v_v = t2739;
out.push(f__F(v_v.clone())?); } Py::list(out) }); } Py::list(out) }, { let mut out = vec![]; for t2740 in py::iter(&(v_yv.clone()))? { let mut v_v: Py; 
v_v = t2740;
out.push(f__F(v_v.clone())?); } Py::list(out) }])).items()?;
    if t2741.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_Mf = t2741[0].clone();
    v_Yf = t2741[1].clone();
    v_res = { let mut out = vec![]; for t2742 in py::iter(&(py::zip(&[v_Mf.clone(), v_Yf.clone()])?))? { let mut v_r: Py; let mut v_t: Py; 
let t2743 = (t2742).items()?;
if t2743.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_r = t2743[0].clone();
v_t = t2743[1].clone();
out.push(py::sub(&py::add(&py::sum(&{ let mut out = vec![]; for t2744 in py::iter(&(py::zip(&[v_r.clone(), v_w2.clone()])?))? { let mut v_a: Py; let mut v_c: Py; 
let t2745 = (t2744).items()?;
if t2745.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t2745[0].clone();
v_c = t2745[1].clone();
out.push(py::mul(&v_a.clone(), &v_c.clone())?); } Py::list(out) }, None)?, &v_b2.clone())?, &v_t.clone())?); } Py::list(out) };
    return Ok(f__out(Py::dict(vec![(Py::str("w_new"), v_w2.clone()), (Py::str("b_new"), v_b2.clone()), (Py::str("loss_before"), v_loss.clone()), (Py::str("loss_after"), py::truediv(&py::sum(&{ let mut out = vec![]; for t2758 in py::iter(&(v_res.clone()))? { let mut v_v: Py; 
v_v = t2758;
out.push(py::mul(&v_v.clone(), &v_v.clone())?); } Py::list(out) }, None)?, &py::len(&v_res.clone())?)?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_gini_split(mut v_y: Py, mut v_g: Py) -> R {
    let mut v_child: Py = Py::None;
    let mut v_gini: Py = Py::None;
    let mut v_gl: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_parent: Py = Py::None;
    let mut v_yl: Py = Py::None;
    let t2641 = (Py::tuple(vec![f__labels(Py::str("y"), v_y.clone(), Py::Int(1i64))?, f__labels(Py::str("g"), v_g.clone(), Py::Int(1i64))?])).items()?;
    if t2641.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_yl = t2641[0].clone();
    v_gl = t2641[1].clone();
    let _ = f__same_len(v_yl.clone(), v_gl.clone(), Py::str("y"), Py::str("g"))?;
    let _ = f__need({ let t2644 = py::len(&py::set(&v_gl.clone())?)?; Py::Bool({ let t2645 = Py::Int(2i64); if py::ge(&t2644, &t2645)? { true } else { false } }) }, Py::str("a split needs at least two groups"))?;
    v_gini = { let v_g = v_g.clone(); let v_gini = v_gini.clone(); let v_gl = v_gl.clone(); let v_y = v_y.clone(); let v_yl = v_yl.clone(); py::func(move |a: &[Py]| -> R {
            #[allow(unused_mut, unused_variables, unused_assignments)]
{
            let mut v_lab: Py = a.get(0).cloned().unwrap_or(Py::None);
            let mut v_n: Py = Py::None;
            v_n = py::len(&v_lab.clone())?;
            return Ok(py::sub(&Py::Int(1i64), &py::sum(&{ let mut out = vec![]; for t2646 in py::iter(&(py::set(&v_lab.clone())?))? { let mut v_c: Py; 
v_c = t2646;
out.push(py::pow(&py::fraction(&[py::method(&v_lab.clone(), "count", &[v_c.clone()])?, v_n.clone()])?, &Py::Int(2i64))?); } Py::list(out) }, None)?)?);
            #[allow(unreachable_code)]
            Ok(Py::None)
        }
        }) };
    v_n = py::len(&v_yl.clone())?;
    v_parent = py::call(&v_gini.clone(), &[v_yl.clone()])?;
    v_child = py::sum(&{ let mut out = vec![]; for t2647 in py::iter(&(py::set(&v_gl.clone())?))? { let mut v_v: Py; 
v_v = t2647;
out.push(py::mul(&py::fraction(&[py::method(&v_gl.clone(), "count", &[v_v.clone()])?, v_n.clone()])?, &py::call(&v_gini.clone(), &[{ let mut out = vec![]; for t2648 in py::iter(&(py::zip(&[v_yl.clone(), v_gl.clone()])?))? { let mut v_a: Py; let mut v_b: Py; 
let t2649 = (t2648).items()?;
if t2649.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t2649[0].clone();
v_b = t2649[1].clone();
if ({ let t2650 = v_b.clone(); Py::Bool({ let t2651 = v_v.clone(); if py::eq(&t2650, &t2651) { true } else { false } }) }).truthy() { out.push(v_a.clone()); }} Py::list(out) }])?)?); } Py::list(out) }, None)?;
    return Ok(f__out(Py::dict(vec![(Py::str("gini_parent"), v_parent.clone()), (Py::str("gini_children"), v_child.clone()), (Py::str("gini_decrease"), py::sub(&v_parent.clone(), &v_child.clone())?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_hamming(mut v_Y_true: Py, mut v_Y_pred: Py) -> R {
    let mut v_A: Py = Py::None;
    let mut v_B: Py = Py::None;
    let mut v_L: Py = Py::None;
    let mut v_M: Py = Py::None;
    let mut v_exact: Py = Py::None;
    let mut v_miss: Py = Py::None;
    let mut v_n: Py = Py::None;
    let t2805 = (Py::tuple(vec![f__mat(Py::str("Y_true"), v_Y_true.clone(), Py::Int(1i64), Py::Int(1i64))?, f__mat(Py::str("Y_pred"), v_Y_pred.clone(), Py::Int(1i64), Py::Int(1i64))?])).items()?;
    if t2805.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_A = t2805[0].clone();
    v_B = t2805[1].clone();
    let _ = f__need({ let t2810 = { let t2806 = py::len(&v_A.clone())?; Py::Bool({ let t2807 = py::len(&v_B.clone())?; if py::eq(&t2806, &t2807) { true } else { false } }) }; if !t2810.truthy() { t2810 } else { { let t2808 = py::len(&py::getitem(&v_A.clone(), &Py::Int(0i64))?)?; Py::Bool({ let t2809 = py::len(&py::getitem(&v_B.clone(), &Py::Int(0i64))?)?; if py::eq(&t2808, &t2809) { true } else { false } }) } } }, Py::str("Y_true and Y_pred must have the same shape"))?;
    for t2811 in py::iter(&(Py::tuple(vec![v_A.clone(), v_B.clone()])))? {
        v_M = t2811;
        let _ = f__need(Py::Bool(({ let mut out = vec![]; for t2812 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t2812;
for t2813 in py::iter(&(v_r.clone()))? { let mut v_v: Py; 
v_v = t2813;
out.push({ let t2814 = v_v.clone(); Py::Bool({ let t2815 = Py::tuple(vec![Py::Float(0.0), Py::Float(1.0)]); if py::contains(&t2815, &t2814)? { true } else { false } }) }); }} Py::list(out) }).items()?.iter().all(|x| x.truthy())), Py::str("entries must be 0 or 1"))?;
    }
    let t2816 = (Py::tuple(vec![py::len(&v_A.clone())?, py::len(&py::getitem(&v_A.clone(), &Py::Int(0i64))?)?])).items()?;
    if t2816.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_n = t2816[0].clone();
    v_L = t2816[1].clone();
    v_miss = py::sum(&{ let mut out = vec![]; for t2817 in py::range_iter(&[(v_n.clone()).i()?]) { let mut v_i: Py; 
v_i = t2817;
for t2818 in py::range_iter(&[(v_L.clone()).i()?]) { let mut v_j: Py; 
v_j = t2818;
if ({ let t2819 = py::getitem(&py::getitem(&v_A.clone(), &v_i.clone())?, &v_j.clone())?; Py::Bool({ let t2820 = py::getitem(&py::getitem(&v_B.clone(), &v_i.clone())?, &v_j.clone())?; if py::ne(&t2819, &t2820) { true } else { false } }) }).truthy() { out.push(Py::Int(1i64)); }}} Py::list(out) }, None)?;
    v_exact = py::sum(&{ let mut out = vec![]; for t2821 in py::range_iter(&[(v_n.clone()).i()?]) { let mut v_i: Py; 
v_i = t2821;
if ({ let t2822 = py::getitem(&v_A.clone(), &v_i.clone())?; Py::Bool({ let t2823 = py::getitem(&v_B.clone(), &v_i.clone())?; if py::eq(&t2822, &t2823) { true } else { false } }) }).truthy() { out.push(Py::Int(1i64)); }} Py::list(out) }, None)?;
    return Ok(f__out(Py::dict(vec![(Py::str("hamming_loss"), py::fraction(&[v_miss.clone(), py::mul(&v_n.clone(), &v_L.clone())?])?), (Py::str("subset_accuracy"), py::fraction(&[v_exact.clone(), v_n.clone()])?), (Py::str("n_mismatches"), v_miss.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_hcv(mut v_labels_true: Py, mut v_labels_pred: Py, mut v_beta: Py) -> R {
    let mut v_a: Py = Py::None;
    let mut v_b: Py = Py::None;
    let mut v_c: Py = Py::None;
    let mut v_h: Py = Py::None;
    let mut v_hc: Py = Py::None;
    let mut v_hk: Py = Py::None;
    let mut v_mi: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_v: Py = Py::None;
    let t3033 = (Py::tuple(vec![f__labels(Py::str("labels_true"), v_labels_true.clone(), Py::Int(1i64))?, f__labels(Py::str("labels_pred"), v_labels_pred.clone(), Py::Int(1i64))?])).items()?;
    if t3033.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_a = t3033[0].clone();
    v_b = t3033[1].clone();
    let _ = f__same_len(v_a.clone(), v_b.clone(), Py::str("labels_true"), Py::str("labels_pred"))?;
    let _ = f__single_check(v_a.clone(), v_b.clone())?;
    let _ = f__fin(Py::str("beta"), Py::tuple(vec![v_beta.clone()]))?;
    let _ = f__need({ let t3041 = v_beta.clone(); Py::Bool({ let t3042 = Py::Int(0i64); if py::gt(&t3041, &t3042)? { true } else { false } }) }, Py::str("beta must be positive"))?;
    v_n = py::len(&v_a.clone())?;
    v_mi = f__mi_counts(f__contingency(v_a.clone(), v_b.clone())?, v_n.clone())?;
    v_hc = f__Hc({ let mut out = vec![]; for t3067 in py::iter(&(py::set(&v_a.clone())?))? { let mut v_v: Py; 
v_v = t3067;
out.push(py::method(&v_a.clone(), "count", &[v_v.clone()])?); } Py::list(out) })?;
    v_hk = f__Hc({ let mut out = vec![]; for t3068 in py::iter(&(py::set(&v_b.clone())?))? { let mut v_v: Py; 
v_v = t3068;
out.push(py::method(&v_b.clone(), "count", &[v_v.clone()])?); } Py::list(out) })?;
    let t3069 = (Py::tuple(vec![py::truediv(&v_mi.clone(), &v_hc.clone())?, py::truediv(&v_mi.clone(), &v_hk.clone())?])).items()?;
    if t3069.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_h = t3069[0].clone();
    v_c = t3069[1].clone();
    v_v = (if ({ let t3070 = py::add(&v_h.clone(), &v_c.clone())?; Py::Bool({ let t3071 = Py::Int(0i64); if py::gt(&t3070, &t3071)? { true } else { false } }) }).truthy() { py::truediv(&py::mul(&py::mul(&py::add(&Py::Int(1i64), &v_beta.clone())?, &v_h.clone())?, &v_c.clone())?, &py::add(&py::mul(&v_beta.clone(), &v_h.clone())?, &v_c.clone())?)? } else { Py::Float(0.0) });
    return Ok(f__out(Py::dict(vec![(Py::str("homogeneity"), v_h.clone()), (Py::str("completeness"), v_c.clone()), (Py::str("v_measure"), v_v.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_hinge(mut v_y_true: Py, mut v_scores: Py) -> R {
    let mut v_m: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_s: Py = Py::None;
    let mut v_yt: Py = Py::None;
    v_yt = f__labels(Py::str("y_true"), v_y_true.clone(), Py::Int(1i64))?;
    let _ = f__need(Py::Bool(({ let mut out = vec![]; for t2965 in py::iter(&(v_yt.clone()))? { let mut v_v: Py; 
v_v = t2965;
out.push({ let t2966 = v_v.clone(); Py::Bool({ let t2967 = Py::tuple(vec![py::neg(&Py::Int(1i64))?, Py::Int(1i64)]); if py::contains(&t2967, &t2966)? { true } else { false } }) }); } Py::list(out) }).items()?.iter().all(|x| x.truthy())), Py::str("labels must be -1 or +1"))?;
    v_s = f__vec(Py::str("scores"), v_scores.clone(), Py::Int(1i64))?;
    let _ = f__same_len(v_yt.clone(), v_s.clone(), Py::str("y_true"), Py::str("scores"))?;
    let _ = f__need({ let t2975 = py::len(&py::set(&v_yt.clone())?)?; Py::Bool({ let t2976 = Py::Int(2i64); if py::eq(&t2975, &t2976) { true } else { false } }) }, Py::str("both classes (-1 and +1) must occur"))?;
    v_m = { let mut out = vec![]; for t2977 in py::iter(&(py::zip(&[v_yt.clone(), v_s.clone()])?))? { let mut v_v: Py; let mut v_y: Py; 
let t2978 = (t2977).items()?;
if t2978.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_y = t2978[0].clone();
v_v = t2978[1].clone();
out.push(py::minmax(&[py::fraction(&[Py::Int(0i64)])?, py::sub(&Py::Int(1i64), &py::mul(&v_y.clone(), &f__F(v_v.clone())?)?)?], true)?); } Py::list(out) };
    v_n = py::len(&v_m.clone())?;
    return Ok(f__out(Py::dict(vec![(Py::str("hinge"), py::truediv(&py::sum(&v_m.clone(), None)?, &v_n.clone())?), (Py::str("squared_hinge"), py::truediv(&py::sum(&{ let mut out = vec![]; for t2991 in py::iter(&(v_m.clone()))? { let mut v_v: Py; 
v_v = t2991;
out.push(py::mul(&v_v.clone(), &v_v.clone())?); } Py::list(out) }, None)?, &v_n.clone())?), (Py::str("n_violations"), py::sum(&{ let mut out = vec![]; for t2992 in py::iter(&(v_m.clone()))? { let mut v_v: Py; 
v_v = t2992;
if ({ let t2993 = v_v.clone(); Py::Bool({ let t2994 = Py::Int(0i64); if py::gt(&t2993, &t2994)? { true } else { false } }) }).truthy() { out.push(Py::Int(1i64)); }} Py::list(out) }, None)?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_huber(mut v_y_true: Py, mut v_y_pred: Py, mut v_delta: Py) -> R {
    let mut v_H: Py = Py::None;
    let mut v_a: Py = Py::None;
    let mut v_b: Py = Py::None;
    let mut v_d: Py = Py::None;
    let mut v_h: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_pseudo: Py = Py::None;
    let mut v_r: Py = Py::None;
    let mut v_x2: Py = Py::None;
    let mut v_yp: Py = Py::None;
    let mut v_yt: Py = Py::None;
    let t3124 = (f__pair(v_y_true.clone(), v_y_pred.clone(), Py::Int(1i64))?).items()?;
    if t3124.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_yt = t3124[0].clone();
    v_yp = t3124[1].clone();
    let _ = f__fin(Py::str("delta"), Py::tuple(vec![v_delta.clone()]))?;
    let _ = f__need({ let t3125 = v_delta.clone(); Py::Bool({ let t3126 = Py::Int(0i64); if py::gt(&t3125, &t3126)? { true } else { false } }) }, Py::str("delta must be positive"))?;
    v_d = f__F(v_delta.clone())?;
    let t3127 = (Py::tuple(vec![Py::list(vec![]), Py::list(vec![])])).items()?;
    if t3127.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_h = t3127[0].clone();
    v_pseudo = t3127[1].clone();
    for t3128 in py::iter(&(py::zip(&[v_yt.clone(), v_yp.clone()])?))? {
        let t3131 = (t3128).items()?;
        if t3131.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_a = t3131[0].clone();
        v_b = t3131[1].clone();
        v_r = py::sub(&f__F(v_a.clone())?, &f__F(v_b.clone())?)?;
        let _ = py::method(&v_h.clone(), "append", &[(if ({ let t3129 = py::abs(&v_r.clone())?; Py::Bool({ let t3130 = v_d.clone(); if py::le(&t3129, &t3130)? { true } else { false } }) }).truthy() { py::truediv(&py::mul(&v_r.clone(), &v_r.clone())?, &Py::Int(2i64))? } else { py::mul(&v_d.clone(), &py::sub(&py::abs(&v_r.clone())?, &py::truediv(&v_d.clone(), &Py::Int(2i64))?)?)? })])?;
        v_x2 = py::float(&py::truediv(&py::mul(&v_r.clone(), &v_r.clone())?, &py::mul(&v_d.clone(), &v_d.clone())?)?)?;
        let _ = py::method(&v_pseudo.clone(), "append", &[py::truediv(&py::mul(&py::mul(&v_delta.clone(), &v_delta.clone())?, &v_x2.clone())?, &py::add(&py::math1("sqrt", &py::add(&Py::Int(1i64), &v_x2.clone())?)?, &Py::Int(1i64))?)?])?;
    }
    v_n = py::len(&v_yt.clone())?;
    v_H = py::truediv(&py::sum(&v_h.clone(), None)?, &v_n.clone())?;
    return Ok(f__out(Py::dict(vec![(Py::str("huber"), v_H.clone()), (Py::str("smooth_l1"), py::truediv(&v_H.clone(), &v_d.clone())?), (Py::str("pseudo_huber"), py::truediv(&py::fsum(&v_pseudo.clone())?, &v_n.clone())?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_impurity_counts(mut v_counts: Py) -> R {
    let mut v_c: Py = Py::None;
    let mut v_gini: Py = Py::None;
    let mut v_n: Py = Py::None;
    v_c = f__labels(Py::str("counts"), v_counts.clone(), Py::Int(1i64))?;
    let _ = f__need({ let t3187 = Py::Bool(({ let mut out = vec![]; for t3182 in py::iter(&(v_c.clone()))? { let mut v_v: Py; 
v_v = t3182;
out.push({ let t3183 = v_v.clone(); Py::Bool({ let t3184 = Py::Int(0i64); if py::ge(&t3183, &t3184)? { true } else { false } }) }); } Py::list(out) }).items()?.iter().all(|x| x.truthy())); if !t3187.truthy() { t3187 } else { { let t3185 = py::sum(&v_c.clone(), None)?; Py::Bool({ let t3186 = Py::Int(0i64); if py::gt(&t3185, &t3186)? { true } else { false } }) } } }, Py::str("counts must be whole numbers >= 0 with a positive total"))?;
    v_n = py::sum(&v_c.clone(), None)?;
    v_gini = py::sub(&Py::Int(1i64), &py::sum(&{ let mut out = vec![]; for t3188 in py::iter(&(v_c.clone()))? { let mut v_v: Py; 
v_v = t3188;
out.push(py::pow(&py::fraction(&[v_v.clone(), v_n.clone()])?, &Py::Int(2i64))?); } Py::list(out) }, None)?)?;
    return Ok(f__out(Py::dict(vec![(Py::str("entropy_bits"), py::truediv(&f__Hc(v_c.clone())?, &py::math1("log", &Py::Float(2.0))?)?), (Py::str("gini"), v_gini.clone()), (Py::str("misclassification"), py::sub(&Py::Int(1i64), &py::fraction(&[py::minmax(&[v_c.clone()], true)?, v_n.clone()])?)?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_information_gain(mut v_y: Py, mut v_x: Py) -> R {
    let mut v_cond: Py = Py::None;
    let mut v_hy: Py = Py::None;
    let mut v_ig: Py = Py::None;
    let mut v_ln2: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_parts: Py = Py::None;
    let mut v_split: Py = Py::None;
    let mut v_sub: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_xl: Py = Py::None;
    let mut v_yl: Py = Py::None;
    let t3245 = (Py::tuple(vec![f__labels(Py::str("y"), v_y.clone(), Py::Int(1i64))?, f__labels(Py::str("x"), v_x.clone(), Py::Int(1i64))?])).items()?;
    if t3245.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_yl = t3245[0].clone();
    v_xl = t3245[1].clone();
    let _ = f__same_len(v_yl.clone(), v_xl.clone(), Py::str("y"), Py::str("x"))?;
    v_n = py::len(&v_yl.clone())?;
    v_hy = f__Hc({ let mut out = vec![]; for t3254 in py::iter(&(py::set(&v_yl.clone())?))? { let mut v_v: Py; 
v_v = t3254;
out.push(py::method(&v_yl.clone(), "count", &[v_v.clone()])?); } Py::list(out) })?;
    v_cond = Py::Float(0.0);
    v_parts = Py::list(vec![]);
    for t3255 in py::iter(&(py::sorted(&py::set(&v_xl.clone())?, (Py::Bool(false)).truthy())?))? {
        v_v = t3255;
        v_sub = { let mut out = vec![]; for t3256 in py::iter(&(py::zip(&[v_yl.clone(), v_xl.clone()])?))? { let mut v_a: Py; let mut v_b: Py; 
let t3257 = (t3256).items()?;
if t3257.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t3257[0].clone();
v_b = t3257[1].clone();
if ({ let t3258 = v_b.clone(); Py::Bool({ let t3259 = v_v.clone(); if py::eq(&t3258, &t3259) { true } else { false } }) }).truthy() { out.push(v_a.clone()); }} Py::list(out) };
        let _ = py::method(&v_parts.clone(), "append", &[py::mul(&py::truediv(&py::len(&v_sub.clone())?, &v_n.clone())?, &f__Hc({ let mut out = vec![]; for t3260 in py::iter(&(py::set(&v_sub.clone())?))? { let mut v_c: Py; 
v_c = t3260;
out.push(py::method(&v_sub.clone(), "count", &[v_c.clone()])?); } Py::list(out) })?)?])?;
    }
    v_cond = py::fsum(&v_parts.clone())?;
    v_split = f__Hc({ let mut out = vec![]; for t3261 in py::iter(&(py::set(&v_xl.clone())?))? { let mut v_v: Py; 
v_v = t3261;
out.push(py::method(&v_xl.clone(), "count", &[v_v.clone()])?); } Py::list(out) })?;
    let _ = f__need({ let t3262 = v_split.clone(); Py::Bool({ let t3263 = Py::Int(0i64); if py::gt(&t3262, &t3263)? { true } else { false } }) }, Py::str("the gain ratio is undefined when the feature takes a single value"))?;
    v_ln2 = py::math1("log", &Py::Float(2.0))?;
    v_ig = py::sub(&v_hy.clone(), &v_cond.clone())?;
    return Ok(f__out(Py::dict(vec![(Py::str("parent_entropy"), py::truediv(&v_hy.clone(), &v_ln2.clone())?), (Py::str("conditional_entropy"), py::truediv(&v_cond.clone(), &v_ln2.clone())?), (Py::str("information_gain"), py::truediv(&v_ig.clone(), &v_ln2.clone())?), (Py::str("split_info"), py::truediv(&v_split.clone(), &v_ln2.clone())?), (Py::str("gain_ratio"), py::truediv(&v_ig.clone(), &v_split.clone())?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_joint_entropy(mut v_P: Py, mut v_unit: Py) -> R {
    let mut v_L: Py = Py::None;
    let mut v_M: Py = Py::None;
    let mut v_hx: Py = Py::None;
    let mut v_hxy: Py = Py::None;
    let mut v_hy: Py = Py::None;
    let mut v_mi: Py = Py::None;
    let mut v_px: Py = Py::None;
    let mut v_py: Py = Py::None;
    v_M = f__mat(Py::str("P"), v_P.clone(), Py::Int(1i64), Py::Int(1i64))?;
    let _ = f__need(Py::Bool(({ let mut out = vec![]; for t3322 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t3322;
for t3323 in py::iter(&(v_r.clone()))? { let mut v_v: Py; 
v_v = t3323;
out.push({ let t3324 = v_v.clone(); Py::Bool({ let t3325 = Py::Int(0i64); if py::ge(&t3324, &t3325)? { true } else { false } }) }); }} Py::list(out) }).items()?.iter().all(|x| x.truthy())), Py::str("joint probabilities must be >= 0"))?;
    let _ = f__need({ let t3326 = py::abs(&py::sub(&py::fsum(&{ let mut out = vec![]; for t3328 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t3328;
for t3329 in py::iter(&(v_r.clone()))? { let mut v_v: Py; 
v_v = t3329;
out.push(v_v.clone()); }} Py::list(out) })?, &Py::Int(1i64))?)?; Py::Bool({ let t3327 = Py::Float(1e-09); if py::le(&t3326, &t3327)? { true } else { false } }) }, Py::str("the joint probabilities must sum to 1"))?;
    v_px = { let mut out = vec![]; for t3330 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t3330;
out.push(py::fsum(&v_r.clone())?); } Py::list(out) };
    v_py = { let mut out = vec![]; for t3331 in py::range_iter(&[(py::len(&py::getitem(&v_M.clone(), &Py::Int(0i64))?)?).i()?]) { let mut v_j: Py; 
v_j = t3331;
out.push(py::fsum(&{ let mut out = vec![]; for t3332 in py::range_iter(&[(py::len(&v_M.clone())?).i()?]) { let mut v_i: Py; 
v_i = t3332;
out.push(py::getitem(&py::getitem(&v_M.clone(), &v_i.clone())?, &v_j.clone())?); } Py::list(out) })?); } Py::list(out) };
    v_L = f__LN(v_unit.clone())?;
    v_hxy = f__H({ let mut out = vec![]; for t3336 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t3336;
for t3337 in py::iter(&(v_r.clone()))? { let mut v_v: Py; 
v_v = t3337;
out.push(v_v.clone()); }} Py::list(out) })?;
    let t3338 = (Py::tuple(vec![f__H(v_px.clone())?, f__H(v_py.clone())?])).items()?;
    if t3338.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_hx = t3338[0].clone();
    v_hy = t3338[1].clone();
    v_mi = py::fsum(&{ let mut out = vec![]; for t3339 in py::range_iter(&[(py::len(&v_M.clone())?).i()?]) { let mut v_i: Py; 
v_i = t3339;
for t3340 in py::range_iter(&[(py::len(&py::getitem(&v_M.clone(), &Py::Int(0i64))?)?).i()?]) { let mut v_j: Py; 
v_j = t3340;
if ({ let t3341 = py::getitem(&py::getitem(&v_M.clone(), &v_i.clone())?, &v_j.clone())?; Py::Bool({ let t3342 = Py::Int(0i64); if py::gt(&t3341, &t3342)? { true } else { false } }) }).truthy() { out.push(py::mul(&py::getitem(&py::getitem(&v_M.clone(), &v_i.clone())?, &v_j.clone())?, &py::math1("log", &py::truediv(&py::getitem(&py::getitem(&v_M.clone(), &v_i.clone())?, &v_j.clone())?, &py::mul(&py::getitem(&v_px.clone(), &v_i.clone())?, &py::getitem(&v_py.clone(), &v_j.clone())?)?)?)?)?); }}} Py::list(out) })?;
    return Ok(f__out(Py::dict(vec![(Py::str("H_XY"), py::truediv(&v_hxy.clone(), &v_L.clone())?), (Py::str("H_X"), py::truediv(&v_hx.clone(), &v_L.clone())?), (Py::str("H_Y"), py::truediv(&v_hy.clone(), &v_L.clone())?), (Py::str("H_Y_given_X"), py::truediv(&py::sub(&v_hxy.clone(), &v_hx.clone())?, &v_L.clone())?), (Py::str("H_X_given_Y"), py::truediv(&py::sub(&v_hxy.clone(), &v_hy.clone())?, &v_L.clone())?), (Py::str("mutual_information"), py::truediv(&v_mi.clone(), &v_L.clone())?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_kfold_cv_ols(mut v_X: Py, mut v_y: Py, mut v_k: Py) -> R {
    let mut v_M: Py = Py::None;
    let mut v_b: Py = Py::None;
    let mut v_d: Py = Py::None;
    let mut v_err: Py = Py::None;
    let mut v_mses: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_s: Py = Py::None;
    let mut v_sizes: Py = Py::None;
    let mut v_start: Py = Py::None;
    let mut v_te: Py = Py::None;
    let mut v_tr: Py = Py::None;
    let mut v_w: Py = Py::None;
    let mut v_yv: Py = Py::None;
    v_M = f__mat(Py::str("X"), v_X.clone(), Py::Int(1i64), Py::Int(1i64))?;
    v_yv = f__vec(Py::str("y"), v_y.clone(), Py::Int(1i64))?;
    let t3406 = (Py::tuple(vec![py::len(&v_M.clone())?, py::len(&py::getitem(&v_M.clone(), &Py::Int(0i64))?)?])).items()?;
    if t3406.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_n = t3406[0].clone();
    v_d = t3406[1].clone();
    let _ = f__need({ let t3407 = py::len(&v_yv.clone())?; Py::Bool({ let t3408 = v_n.clone(); if py::eq(&t3407, &t3408) { true } else { false } }) }, Py::str("y needs one value per row of X"))?;
    let _ = f__need({ let t3412 = Py::Bool(matches!(v_k.clone(), Py::Int(_) | Py::Big(_) | Py::Bool(_))); if !t3412.truthy() { t3412 } else { { let t3409 = Py::Int(2i64); Py::Bool({ let t3410 = v_k.clone(); if py::le(&t3409, &t3410)? { { let t3411 = v_n.clone(); if py::le(&t3410, &t3411)? { true } else { false } } } else { false } }) } } }, Py::str("k must be between 2 and n"))?;
    v_sizes = py::getitem(&f_kfold_sizes(v_n.clone(), v_k.clone())?, &Py::str("test_sizes"))?;
    let _ = f__need({ let t3435 = py::sub(&v_n.clone(), &py::minmax(&[v_sizes.clone()], true)?)?; Py::Bool({ let t3436 = py::add(&v_d.clone(), &Py::Int(2i64))?; if py::ge(&t3435, &t3436)? { true } else { false } }) }, Py::str("every training set needs at least d + 2 samples"))?;
    let t3437 = (Py::tuple(vec![Py::list(vec![]), Py::Int(0i64)])).items()?;
    if t3437.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_mses = t3437[0].clone();
    v_start = t3437[1].clone();
    for t3438 in py::iter(&(v_sizes.clone()))? {
        v_s = t3438;
        v_te = Py::list((Py::list((py::range_iter(&[(v_start.clone()).i()?, (py::add(&v_start.clone(), &v_s.clone())?).i()?])).into_iter().collect::<Vec<Py>>())).items()?);
        v_tr = { let mut out = vec![]; for t3439 in py::range_iter(&[(v_n.clone()).i()?]) { let mut v_i: Py; 
v_i = t3439;
if ({ let t3444 = { let t3440 = v_i.clone(); Py::Bool({ let t3441 = v_start.clone(); if py::lt(&t3440, &t3441)? { true } else { false } }) }; if t3444.truthy() { t3444 } else { { let t3442 = v_i.clone(); Py::Bool({ let t3443 = py::add(&v_start.clone(), &v_s.clone())?; if py::ge(&t3442, &t3443)? { true } else { false } }) } } }).truthy() { out.push(v_i.clone()); }} Py::list(out) };
        let t3476 = (f__ols_exact({ let mut out = vec![]; for t3474 in py::iter(&(v_tr.clone()))? { let mut v_i: Py; 
v_i = t3474;
out.push(py::getitem(&v_M.clone(), &v_i.clone())?); } Py::list(out) }, { let mut out = vec![]; for t3475 in py::iter(&(v_tr.clone()))? { let mut v_i: Py; 
v_i = t3475;
out.push(py::getitem(&v_yv.clone(), &v_i.clone())?); } Py::list(out) })?).items()?;
        if t3476.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_w = t3476[0].clone();
        v_b = t3476[1].clone();
        v_err = { let mut out = vec![]; for t3477 in py::iter(&(v_te.clone()))? { let mut v_i: Py; 
v_i = t3477;
out.push(py::sub(&py::add(&py::sum(&{ let mut out = vec![]; for t3478 in py::iter(&(py::zip(&[v_w.clone(), py::getitem(&v_M.clone(), &v_i.clone())?])?))? { let mut v_v: Py; let mut v_wj: Py; 
let t3479 = (t3478).items()?;
if t3479.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_wj = t3479[0].clone();
v_v = t3479[1].clone();
out.push(py::mul(&v_wj.clone(), &f__F(v_v.clone())?)?); } Py::list(out) }, None)?, &v_b.clone())?, &f__F(py::getitem(&v_yv.clone(), &v_i.clone())?)?)?); } Py::list(out) };
        let _ = py::method(&v_mses.clone(), "append", &[py::truediv(&py::sum(&{ let mut out = vec![]; for t3480 in py::iter(&(v_err.clone()))? { let mut v_e: Py; 
v_e = t3480;
out.push(py::mul(&v_e.clone(), &v_e.clone())?); } Py::list(out) }, None)?, &v_s.clone())?])?;
        v_start = py::iadd(&v_start, &v_s.clone())?;
    }
    return Ok(f__out(Py::dict(vec![(Py::str("fold_mse"), v_mses.clone()), (Py::str("mean_mse"), py::truediv(&py::sum(&v_mses.clone(), None)?, &v_k.clone())?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_kfold_sizes(mut v_n: Py, mut v_k: Py) -> R {
    let mut v_base: Py = Py::None;
    let mut v_extra: Py = Py::None;
    let mut v_test: Py = Py::None;
    let _ = f__need({ let t3514 = Py::Bool(matches!(v_n.clone(), Py::Int(_) | Py::Big(_) | Py::Bool(_))); if !t3514.truthy() { t3514 } else { { let t3513 = Py::Bool(matches!(v_k.clone(), Py::Int(_) | Py::Big(_) | Py::Bool(_))); if !t3513.truthy() { t3513 } else { { let t3510 = Py::Int(2i64); Py::Bool({ let t3511 = v_k.clone(); if py::le(&t3510, &t3511)? { { let t3512 = v_n.clone(); if py::le(&t3511, &t3512)? { true } else { false } } } else { false } }) } } } } }, Py::str("k must be at least 2 and at most n"))?;
    let t3515 = (Py::tuple(vec![py::floordiv(&v_n.clone(), &v_k.clone())?, py::modulo(&v_n.clone(), &v_k.clone())?])).items()?;
    if t3515.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_base = t3515[0].clone();
    v_extra = t3515[1].clone();
    v_test = { let mut out = vec![]; for t3516 in py::range_iter(&[(v_k.clone()).i()?]) { let mut v_i: Py; 
v_i = t3516;
out.push((if ({ let t3517 = v_i.clone(); Py::Bool({ let t3518 = v_extra.clone(); if py::lt(&t3517, &t3518)? { true } else { false } }) }).truthy() { py::add(&v_base.clone(), &Py::Int(1i64))? } else { v_base.clone() })); } Py::list(out) };
    return Ok(f__out(Py::dict(vec![(Py::str("test_sizes"), v_test.clone()), (Py::str("train_sizes"), { let mut out = vec![]; for t3531 in py::iter(&(v_test.clone()))? { let mut v_t: Py; 
v_t = t3531;
out.push(py::sub(&v_n.clone(), &v_t.clone())?); } Py::list(out) }), (Py::str("first_test_fold"), Py::list(vec![Py::Int(0i64), py::sub(&py::getitem(&v_test.clone(), &Py::Int(0i64))?, &Py::Int(1i64))?]))]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_kl_discrete(mut v_p: Py, mut v_q: Py, mut v_unit: Py) -> R {
    let mut v_L: Py = Py::None;
    let mut v_M: Py = Py::None;
    let mut v_P: Py = Py::None;
    let mut v_Q: Py = Py::None;
    let mut v_kl: Py = Py::None;
    let t3574 = (Py::tuple(vec![f__dist(Py::str("p"), v_p.clone())?, f__dist(Py::str("q"), v_q.clone())?])).items()?;
    if t3574.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_P = t3574[0].clone();
    v_Q = t3574[1].clone();
    let _ = f__same_len(v_P.clone(), v_Q.clone(), Py::str("p"), Py::str("q"))?;
    v_L = f__LN(v_unit.clone())?;
    v_kl = { let v_L = v_L.clone(); let v_P = v_P.clone(); let v_Q = v_Q.clone(); let v_p = v_p.clone(); let v_q = v_q.clone(); let v_unit = v_unit.clone(); py::func(move |a: &[Py]| -> R {
            #[allow(unused_mut, unused_variables, unused_assignments)]
{
            let mut v_a: Py = a.get(0).cloned().unwrap_or(Py::None);
            let mut v_b: Py = a.get(1).cloned().unwrap_or(Py::None);
            return Ok(py::truediv(&py::fsum(&{ let mut out = vec![]; for t3577 in py::iter(&(py::zip(&[v_a.clone(), v_b.clone()])?))? { let mut v_x: Py; let mut v_y: Py; 
let t3578 = (t3577).items()?;
if t3578.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_x = t3578[0].clone();
v_y = t3578[1].clone();
out.push(py::mul(&v_x.clone(), &py::math1("log", &py::truediv(&v_x.clone(), &v_y.clone())?)?)?); } Py::list(out) })?, &v_L.clone())?);
            #[allow(unreachable_code)]
            Ok(Py::None)
        }
        }) };
    v_M = { let mut out = vec![]; for t3579 in py::iter(&(py::zip(&[v_P.clone(), v_Q.clone()])?))? { let mut v_x: Py; let mut v_y: Py; 
let t3580 = (t3579).items()?;
if t3580.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_x = t3580[0].clone();
v_y = t3580[1].clone();
out.push(py::truediv(&py::add(&v_x.clone(), &v_y.clone())?, &Py::Int(2i64))?); } Py::list(out) };
    return Ok(f__out(Py::dict(vec![(Py::str("kl_pq"), py::call(&v_kl.clone(), &[v_P.clone(), v_Q.clone()])?), (Py::str("kl_qp"), py::call(&v_kl.clone(), &[v_Q.clone(), v_P.clone()])?), (Py::str("js_divergence"), py::truediv(&py::add(&py::call(&v_kl.clone(), &[v_P.clone(), v_M.clone()])?, &py::call(&v_kl.clone(), &[v_Q.clone(), v_M.clone()])?)?, &Py::Int(2i64))?), (Py::str("cross_entropy"), py::truediv(&py::neg(&py::fsum(&{ let mut out = vec![]; for t3593 in py::iter(&(py::zip(&[v_P.clone(), v_Q.clone()])?))? { let mut v_x: Py; let mut v_y: Py; 
let t3594 = (t3593).items()?;
if t3594.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_x = t3594[0].clone();
v_y = t3594[1].clone();
out.push(py::mul(&v_x.clone(), &py::math1("log", &v_y.clone())?)?); } Py::list(out) })?)?, &v_L.clone())?), (Py::str("entropy_p"), py::truediv(&py::neg(&py::fsum(&{ let mut out = vec![]; for t3595 in py::iter(&(v_P.clone()))? { let mut v_x: Py; 
v_x = t3595;
out.push(py::mul(&v_x.clone(), &py::math1("log", &v_x.clone())?)?); } Py::list(out) })?)?, &v_L.clone())?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_kmeans(mut v_X: Py, mut v_C: Py) -> R {
    let mut v_F: Py = Py::None;
    let mut v__: Py = Py::None;
    let mut v_cen: Py = Py::None;
    let mut v_inertia: Py = Py::None;
    let mut v_k: Py = Py::None;
    let mut v_lab: Py = Py::None;
    let mut v_new: Py = Py::None;
    let t3652 = (f__km_inputs(v_X.clone(), v_C.clone())?).items()?;
    if t3652.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_F = t3652[0].clone();
    v_cen = t3652[1].clone();
    v_k = py::len(&v_cen.clone())?;
    let t3660 = (f__assign(v_F.clone(), v_cen.clone())?).items()?;
    if t3660.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_lab = t3660[0].clone();
    v__ = t3660[1].clone();
    for t3661 in py::range_iter(&[(Py::Int(100i64)).i()?]) {
        v__ = t3661;
        v_cen = f__update(v_F.clone(), v_lab.clone(), v_k.clone())?;
        let t3668 = (f__assign(v_F.clone(), v_cen.clone())?).items()?;
        if t3668.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_new = t3668[0].clone();
        v_inertia = t3668[1].clone();
        if ({ let t3669 = v_new.clone(); Py::Bool({ let t3670 = v_lab.clone(); if py::eq(&t3669, &t3670) { true } else { false } }) }).truthy() {
            return Ok(f__out(Py::dict(vec![(Py::str("centroids"), { let mut out = vec![]; for t3683 in py::iter(&(v_cen.clone()))? { let mut v_r: Py; 
v_r = t3683;
out.push({ let mut out = vec![]; for t3684 in py::iter(&(v_r.clone()))? { let mut v_v: Py; 
v_v = t3684;
out.push(py::float(&v_v.clone())?); } Py::list(out) }); } Py::list(out) }), (Py::str("labels"), v_lab.clone()), (Py::str("inertia"), v_inertia.clone())]))?);
        }
        v_lab = v_new.clone();
    }
    return Err(py::value_error(py::to_str(&Py::str("k-means did not converge in 100 iterations"))));
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_kmeans_step(mut v_X: Py, mut v_C: Py) -> R {
    let mut v_C0: Py = Py::None;
    let mut v_F: Py = Py::None;
    let mut v_inertia: Py = Py::None;
    let mut v_lab: Py = Py::None;
    let t3741 = (f__km_inputs(v_X.clone(), v_C.clone())?).items()?;
    if t3741.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_F = t3741[0].clone();
    v_C0 = t3741[1].clone();
    let t3749 = (f__assign(v_F.clone(), v_C0.clone())?).items()?;
    if t3749.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_lab = t3749[0].clone();
    v_inertia = t3749[1].clone();
    return Ok(f__out(Py::dict(vec![(Py::str("labels"), v_lab.clone()), (Py::str("new_centroids"), { let mut out = vec![]; for t3768 in py::iter(&(f__update(v_F.clone(), v_lab.clone(), py::len(&v_C0.clone())?)?))? { let mut v_r: Py; 
v_r = t3768;
out.push({ let mut out = vec![]; for t3769 in py::iter(&(v_r.clone()))? { let mut v_v: Py; 
v_v = t3769;
out.push(py::float(&v_v.clone())?); } Py::list(out) }); } Py::list(out) }), (Py::str("inertia"), v_inertia.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_knn_classify(mut v_X: Py, mut v_y: Py, mut v_x: Py, mut v_k: Py, mut v_metric: Py) -> R {
    let mut v__: Py = Py::None;
    let mut v_counts: Py = Py::None;
    let mut v_dist: Py = Py::None;
    let mut v_lab: Py = Py::None;
    let mut v_nb: Py = Py::None;
    let mut v_top: Py = Py::None;
    let mut v_winners: Py = Py::None;
    let mut v_yv: Py = Py::None;
    let t3845 = (f__knn_core(v_X.clone(), v_y.clone(), v_x.clone(), v_k.clone(), v_metric.clone())?).items()?;
    if t3845.len() != 4 { return Err(PyErr::Value("unpack".into())); }
    v_nb = t3845[0].clone();
    v_dist = t3845[1].clone();
    v_yv = t3845[2].clone();
    v__ = t3845[3].clone();
    v_lab = { let mut out = vec![]; for t3846 in py::iter(&(v_nb.clone()))? { let mut v_i: Py; 
v_i = t3846;
out.push(f__whole(Py::str("every label in y"), py::getitem(&v_yv.clone(), &v_i.clone())?)?); } Py::list(out) };
    let _ = f__labels(Py::str("y"), v_yv.clone(), Py::Int(1i64))?;
    v_counts = { let mut out = vec![]; for t3853 in py::iter(&(py::set(&v_lab.clone())?))? { let mut v_c: Py; 
v_c = t3853;
out.push((v_c.clone(), py::method(&v_lab.clone(), "count", &[v_c.clone()])?)); } Py::dict(out) };
    v_top = py::minmax(&[py::method(&v_counts.clone(), "values", &[])?], true)?;
    v_winners = { let mut out = vec![]; for t3854 in py::iter(&(py::method(&v_counts.clone(), "items", &[])?))? { let mut v_c: Py; let mut v_v: Py; 
let t3855 = (t3854).items()?;
if t3855.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_c = t3855[0].clone();
v_v = t3855[1].clone();
if ({ let t3856 = v_v.clone(); Py::Bool({ let t3857 = v_top.clone(); if py::eq(&t3856, &t3857) { true } else { false } }) }).truthy() { out.push(v_c.clone()); }} Py::list(out) };
    let _ = f__need({ let t3858 = py::len(&v_winners.clone())?; Py::Bool({ let t3859 = Py::Int(1i64); if py::eq(&t3858, &t3859) { true } else { false } }) }, Py::str("the vote among the k neighbours is tied"))?;
    return Ok(f__out(Py::dict(vec![(Py::str("prediction"), py::getitem(&v_winners.clone(), &Py::Int(0i64))?), (Py::str("votes"), v_top.clone()), (Py::str("neighbors"), py::sorted(&v_nb.clone(), (Py::Bool(false)).truthy())?), (Py::str("neighbor_distances"), v_dist.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_knn_regress(mut v_X: Py, mut v_y: Py, mut v_x: Py, mut v_k: Py, mut v_metric: Py, mut v_weights: Py) -> R {
    let mut v__: Py = Py::None;
    let mut v_dist: Py = Py::None;
    let mut v_nb: Py = Py::None;
    let mut v_pred: Py = Py::None;
    let mut v_w: Py = Py::None;
    let mut v_yv: Py = Py::None;
    let t3947 = (f__knn_core(v_X.clone(), v_y.clone(), v_x.clone(), v_k.clone(), v_metric.clone())?).items()?;
    if t3947.len() != 4 { return Err(PyErr::Value("unpack".into())); }
    v_nb = t3947[0].clone();
    v_dist = t3947[1].clone();
    v_yv = t3947[2].clone();
    v__ = t3947[3].clone();
    if ({ let t3948 = v_weights.clone(); Py::Bool({ let t3949 = Py::str("uniform"); if py::eq(&t3948, &t3949) { true } else { false } }) }).truthy() {
        v_pred = py::truediv(&py::sum(&{ let mut out = vec![]; for t3950 in py::iter(&(v_nb.clone()))? { let mut v_i: Py; 
v_i = t3950;
out.push(f__F(py::getitem(&v_yv.clone(), &v_i.clone())?)?); } Py::list(out) }, None)?, &v_k.clone())?;
    } else {
        let _ = f__need(Py::Bool(({ let mut out = vec![]; for t3951 in py::iter(&(v_dist.clone()))? { let mut v_v: Py; 
v_v = t3951;
out.push({ let t3952 = v_v.clone(); Py::Bool({ let t3953 = Py::Int(0i64); if py::gt(&t3952, &t3953)? { true } else { false } }) }); } Py::list(out) }).items()?.iter().all(|x| x.truthy())), Py::str("distance weighting is undefined: the query coincides with a training point"))?;
        v_w = { let mut out = vec![]; for t3954 in py::iter(&(v_dist.clone()))? { let mut v_v: Py; 
v_v = t3954;
out.push(py::truediv(&Py::Int(1i64), &v_v.clone())?); } Py::list(out) };
        v_pred = py::truediv(&py::fsum(&{ let mut out = vec![]; for t3955 in py::iter(&(py::zip(&[v_w.clone(), v_nb.clone()])?))? { let mut v_i: Py; let mut v_wi: Py; 
let t3956 = (t3955).items()?;
if t3956.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_wi = t3956[0].clone();
v_i = t3956[1].clone();
out.push(py::mul(&v_wi.clone(), &py::getitem(&v_yv.clone(), &v_i.clone())?)?); } Py::list(out) })?, &py::fsum(&v_w.clone())?)?;
    }
    return Ok(f__out(Py::dict(vec![(Py::str("prediction"), v_pred.clone()), (Py::str("neighbors"), py::sorted(&v_nb.clone(), (Py::Bool(false)).truthy())?), (Py::str("neighbor_distances"), v_dist.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_label_smoothing_ce(mut v_z: Py, mut v_y: Py, mut v_epsilon: Py) -> R {
    let mut v_K: Py = Py::None;
    let mut v_g: Py = Py::None;
    let mut v_logp: Py = Py::None;
    let mut v_p: Py = Py::None;
    let mut v_q: Py = Py::None;
    let mut v_zv: Py = Py::None;
    v_zv = f__vec(Py::str("z"), v_z.clone(), Py::Int(2i64))?;
    v_K = py::len(&v_zv.clone())?;
    let _ = f__need({ let t4009 = Py::Bool(matches!(v_y.clone(), Py::Int(_) | Py::Big(_) | Py::Bool(_))); if !t4009.truthy() { t4009 } else { { let t4006 = Py::Int(0i64); Py::Bool({ let t4007 = v_y.clone(); if py::le(&t4006, &t4007)? { { let t4008 = v_K.clone(); if py::lt(&t4007, &t4008)? { true } else { false } } } else { false } }) } } }, Py::str("y must be the index of the true class (0 to K-1)"))?;
    let _ = f__fin(Py::str("epsilon"), Py::tuple(vec![v_epsilon.clone()]))?;
    let _ = f__need({ let t4010 = Py::Int(0i64); Py::Bool({ let t4011 = v_epsilon.clone(); if py::le(&t4010, &t4011)? { { let t4012 = Py::Int(1i64); if py::lt(&t4011, &t4012)? { true } else { false } } } else { false } }) }, Py::str("epsilon must lie in [0, 1)"))?;
    v_logp = f__log_softmax(v_zv.clone())?;
    v_q = { let mut out = vec![]; for t4015 in py::range_iter(&[(v_K.clone()).i()?]) { let mut v_j: Py; 
v_j = t4015;
out.push(py::add(&py::truediv(&v_epsilon.clone(), &v_K.clone())?, &(if ({ let t4016 = v_j.clone(); Py::Bool({ let t4017 = v_y.clone(); if py::eq(&t4016, &t4017) { true } else { false } }) }).truthy() { py::sub(&Py::Int(1i64), &v_epsilon.clone())? } else { Py::Float(0.0) }))?); } Py::list(out) };
    v_p = f__softmax(v_zv.clone())?;
    v_g = { let mut out = vec![]; for t4020 in py::range_iter(&[(v_K.clone()).i()?]) { let mut v_j: Py; 
v_j = t4020;
out.push(py::sub(&py::getitem(&v_p.clone(), &v_j.clone())?, &py::getitem(&v_q.clone(), &v_j.clone())?)?); } Py::list(out) };
    return Ok(f__out(Py::dict(vec![(Py::str("loss"), py::neg(&py::fsum(&{ let mut out = vec![]; for t4033 in py::iter(&(py::zip(&[v_q.clone(), v_logp.clone()])?))? { let mut v_a: Py; let mut v_b: Py; 
let t4034 = (t4033).items()?;
if t4034.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t4034[0].clone();
v_b = t4034[1].clone();
out.push(py::mul(&v_a.clone(), &v_b.clone())?); } Py::list(out) })?)?), (Py::str("gradient"), v_g.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_lasso(mut v_X: Py, mut v_y: Py, mut v_alpha: Py) -> R {
    let mut v_A: Py = Py::None;
    let mut v_F: Py = Py::None;
    let mut v_G: Py = Py::None;
    let mut v_M: Py = Py::None;
    let mut v_Xc: Py = Py::None;
    let mut v_XcF: Py = Py::None;
    let mut v_Y: Py = Py::None;
    let mut v__: Py = Py::None;
    let mut v_b: Py = Py::None;
    let mut v_c: Py = Py::None;
    let mut v_d: Py = Py::None;
    let mut v_delta: Py = Py::None;
    let mut v_h: Py = Py::None;
    let mut v_i: Py = Py::None;
    let mut v_j: Py = Py::None;
    let mut v_lam: Py = Py::None;
    let mut v_lamF: Py = Py::None;
    let mut v_mx: Py = Py::None;
    let mut v_mxF: Py = Py::None;
    let mut v_my: Py = Py::None;
    let mut v_myF: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_new: Py = Py::None;
    let mut v_nrm: Py = Py::None;
    let mut v_res: Py = Py::None;
    let mut v_resF: Py = Py::None;
    let mut v_rho: Py = Py::None;
    let mut v_s: Py = Py::None;
    let mut v_sol: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_w: Py = Py::None;
    let mut v_wF: Py = Py::None;
    let mut v_yc: Py = Py::None;
    let mut v_ycF: Py = Py::None;
    let mut v_yv: Py = Py::None;
    v_M = f__mat(Py::str("X"), v_X.clone(), Py::Int(3i64), Py::Int(1i64))?;
    v_yv = f__vec(Py::str("y"), v_y.clone(), Py::Int(1i64))?;
    let t4086 = (Py::tuple(vec![py::len(&v_M.clone())?, py::len(&py::getitem(&v_M.clone(), &Py::Int(0i64))?)?])).items()?;
    if t4086.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_n = t4086[0].clone();
    v_d = t4086[1].clone();
    let _ = f__need({ let t4087 = py::len(&v_yv.clone())?; Py::Bool({ let t4088 = v_n.clone(); if py::eq(&t4087, &t4088) { true } else { false } }) }, Py::str("y needs one value per row of X"))?;
    let _ = f__fin(Py::str("alpha"), Py::tuple(vec![v_alpha.clone()]))?;
    let _ = f__need({ let t4089 = v_alpha.clone(); Py::Bool({ let t4090 = Py::Int(0i64); if py::gt(&t4089, &t4090)? { true } else { false } }) }, Py::str("alpha must be positive"))?;
    v_mx = { let mut out = vec![]; for t4091 in py::range_iter(&[(v_d.clone()).i()?]) { let mut v_j: Py; 
v_j = t4091;
out.push(py::truediv(&py::fsum(&{ let mut out = vec![]; for t4092 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t4092;
out.push(py::getitem(&v_r.clone(), &v_j.clone())?); } Py::list(out) })?, &v_n.clone())?); } Py::list(out) };
    v_my = py::truediv(&py::fsum(&v_yv.clone())?, &v_n.clone())?;
    v_Xc = { let mut out = vec![]; for t4093 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t4093;
out.push({ let mut out = vec![]; for t4094 in py::range_iter(&[(v_d.clone()).i()?]) { let mut v_j: Py; 
v_j = t4094;
out.push(py::sub(&py::getitem(&v_r.clone(), &v_j.clone())?, &py::getitem(&v_mx.clone(), &v_j.clone())?)?); } Py::list(out) }); } Py::list(out) };
    v_yc = { let mut out = vec![]; for t4095 in py::iter(&(v_yv.clone()))? { let mut v_v: Py; 
v_v = t4095;
out.push(py::sub(&v_v.clone(), &v_my.clone())?); } Py::list(out) };
    v_nrm = { let mut out = vec![]; for t4096 in py::range_iter(&[(v_d.clone()).i()?]) { let mut v_j: Py; 
v_j = t4096;
out.push(py::fsum(&{ let mut out = vec![]; for t4097 in py::iter(&(v_Xc.clone()))? { let mut v_r: Py; 
v_r = t4097;
out.push(py::pow(&py::getitem(&v_r.clone(), &v_j.clone())?, &Py::Int(2i64))?); } Py::list(out) })?); } Py::list(out) };
    let _ = f__need(Py::Bool(({ let mut out = vec![]; for t4098 in py::iter(&(v_nrm.clone()))? { let mut v_v: Py; 
v_v = t4098;
out.push({ let t4099 = v_v.clone(); Py::Bool({ let t4100 = Py::Int(0i64); if py::gt(&t4099, &t4100)? { true } else { false } }) }); } Py::list(out) }).items()?.iter().all(|x| x.truthy())), Py::str("a feature is constant"))?;
    v_w = py::mul(&Py::list(vec![Py::Float(0.0)]), &v_d.clone())?;
    v_res = Py::list((v_yc.clone()).items()?);
    v_lam = py::mul(&v_n.clone(), &v_alpha.clone())?;
    for t4101 in py::range_iter(&[(Py::Int(20000i64)).i()?]) {
        v__ = t4101;
        v_delta = Py::Float(0.0);
        for t4102 in py::range_iter(&[(v_d.clone()).i()?]) {
            v_j = t4102;
            v_rho = py::fsum(&{ let mut out = vec![]; for t4103 in py::range_iter(&[(v_n.clone()).i()?]) { let mut v_i: Py; 
v_i = t4103;
out.push(py::mul(&py::getitem(&py::getitem(&v_Xc.clone(), &v_i.clone())?, &v_j.clone())?, &py::add(&py::getitem(&v_res.clone(), &v_i.clone())?, &py::mul(&py::getitem(&py::getitem(&v_Xc.clone(), &v_i.clone())?, &v_j.clone())?, &py::getitem(&v_w.clone(), &v_j.clone())?)?)?)?); } Py::list(out) })?;
            v_new = py::truediv(&py::copysign(&py::minmax(&[py::sub(&py::abs(&v_rho.clone())?, &v_lam.clone())?, Py::Float(0.0)], true)?, &v_rho.clone())?, &py::getitem(&v_nrm.clone(), &v_j.clone())?)?;
            if ({ let t4104 = v_new.clone(); Py::Bool({ let t4105 = py::getitem(&v_w.clone(), &v_j.clone())?; if py::ne(&t4104, &t4105) { true } else { false } }) }).truthy() {
                for t4106 in py::range_iter(&[(v_n.clone()).i()?]) {
                    v_i = t4106;
                    let t4107 = v_res.clone(); let t4108 = v_i.clone();
                    py::setitem(&t4107, &t4108, py::sub(&py::getitem(&t4107, &t4108)?, &py::mul(&py::getitem(&py::getitem(&v_Xc.clone(), &v_i.clone())?, &v_j.clone())?, &py::sub(&v_new.clone(), &py::getitem(&v_w.clone(), &v_j.clone())?)?)?)?)?;
                }
                v_delta = py::minmax(&[v_delta.clone(), py::abs(&py::sub(&v_new.clone(), &py::getitem(&v_w.clone(), &v_j.clone())?)?)?], true)?;
                py::setitem(&v_w.clone(), &v_j.clone(), v_new.clone())?;
            }
        }
        if ({ let t4109 = v_delta.clone(); Py::Bool({ let t4110 = py::mul(&Py::Float(1e-13), &py::minmax(&[Py::Float(1.0), py::minmax(&[{ let mut out = vec![]; for t4111 in py::iter(&(v_w.clone()))? { let mut v_v: Py; 
v_v = t4111;
out.push(py::abs(&v_v.clone())?); } Py::list(out) }], true)?], true)?)?; if py::le(&t4109, &t4110)? { true } else { false } }) }).truthy() {
            break;
        }
    }
    v_A = { let mut out = vec![]; for t4112 in py::range_iter(&[(v_d.clone()).i()?]) { let mut v_j: Py; 
v_j = t4112;
if ({ let t4113 = py::getitem(&v_w.clone(), &v_j.clone())?; Py::Bool({ let t4114 = Py::Int(0i64); if py::ne(&t4113, &t4114) { true } else { false } }) }).truthy() { out.push(v_j.clone()); }} Py::list(out) };
    v_s = { let mut out = vec![]; for t4115 in py::iter(&(v_A.clone()))? { let mut v_j: Py; 
v_j = t4115;
out.push((v_j.clone(), (if ({ let t4116 = py::getitem(&v_w.clone(), &v_j.clone())?; Py::Bool({ let t4117 = Py::Int(0i64); if py::gt(&t4116, &t4117)? { true } else { false } }) }).truthy() { Py::Int(1i64) } else { py::neg(&Py::Int(1i64))? }))); } Py::dict(out) };
    v_F = { let mut out = vec![]; for t4118 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t4118;
out.push({ let mut out = vec![]; for t4119 in py::iter(&(v_r.clone()))? { let mut v_v: Py; 
v_v = t4119;
out.push(f__F(v_v.clone())?); } Py::list(out) }); } Py::list(out) };
    v_Y = { let mut out = vec![]; for t4120 in py::iter(&(v_yv.clone()))? { let mut v_v: Py; 
v_v = t4120;
out.push(f__F(v_v.clone())?); } Py::list(out) };
    v_mxF = { let mut out = vec![]; for t4121 in py::range_iter(&[(v_d.clone()).i()?]) { let mut v_j: Py; 
v_j = t4121;
out.push(py::truediv(&py::sum(&{ let mut out = vec![]; for t4122 in py::iter(&(v_F.clone()))? { let mut v_r: Py; 
v_r = t4122;
out.push(py::getitem(&v_r.clone(), &v_j.clone())?); } Py::list(out) }, None)?, &v_n.clone())?); } Py::list(out) };
    v_myF = py::truediv(&py::sum(&v_Y.clone(), None)?, &v_n.clone())?;
    v_XcF = { let mut out = vec![]; for t4123 in py::iter(&(v_F.clone()))? { let mut v_r: Py; 
v_r = t4123;
out.push({ let mut out = vec![]; for t4124 in py::range_iter(&[(v_d.clone()).i()?]) { let mut v_j: Py; 
v_j = t4124;
out.push(py::sub(&py::getitem(&v_r.clone(), &v_j.clone())?, &py::getitem(&v_mxF.clone(), &v_j.clone())?)?); } Py::list(out) }); } Py::list(out) };
    v_ycF = { let mut out = vec![]; for t4125 in py::iter(&(v_Y.clone()))? { let mut v_t: Py; 
v_t = t4125;
out.push(py::sub(&v_t.clone(), &v_myF.clone())?); } Py::list(out) };
    v_lamF = py::mul(&v_n.clone(), &f__F(v_alpha.clone())?)?;
    v_wF = py::mul(&Py::list(vec![py::fraction(&[Py::Int(0i64)])?]), &v_d.clone())?;
    if (v_A.clone()).truthy() {
        v_G = { let mut out = vec![]; for t4126 in py::iter(&(v_A.clone()))? { let mut v_i: Py; 
v_i = t4126;
out.push({ let mut out = vec![]; for t4127 in py::iter(&(v_A.clone()))? { let mut v_j: Py; 
v_j = t4127;
out.push(py::sum(&{ let mut out = vec![]; for t4128 in py::iter(&(v_XcF.clone()))? { let mut v_r: Py; 
v_r = t4128;
out.push(py::mul(&py::getitem(&v_r.clone(), &v_i.clone())?, &py::getitem(&v_r.clone(), &v_j.clone())?)?); } Py::list(out) }, None)?); } Py::list(out) }); } Py::list(out) };
        v_h = { let mut out = vec![]; for t4129 in py::iter(&(v_A.clone()))? { let mut v_i: Py; 
v_i = t4129;
out.push(py::sub(&py::sum(&{ let mut out = vec![]; for t4130 in py::iter(&(py::zip(&[v_XcF.clone(), v_ycF.clone()])?))? { let mut v_r: Py; let mut v_t: Py; 
let t4131 = (t4130).items()?;
if t4131.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_r = t4131[0].clone();
v_t = t4131[1].clone();
out.push(py::mul(&py::getitem(&v_r.clone(), &v_i.clone())?, &v_t.clone())?); } Py::list(out) }, None)?, &py::mul(&v_lamF.clone(), &py::getitem(&v_s.clone(), &v_i.clone())?)?)?); } Py::list(out) };
        v_sol = f__solve_exact(v_G.clone(), v_h.clone())?;
        for t4153 in py::iter(&(py::zip(&[v_A.clone(), v_sol.clone()])?))? {
            let t4163 = (t4153).items()?;
            if t4163.len() != 2 { return Err(PyErr::Value("unpack".into())); }
            v_j = t4163[0].clone();
            v_v = t4163[1].clone();
            let _ = f__need({ let t4162 = { let t4154 = v_v.clone(); Py::Bool({ let t4155 = Py::Int(0i64); if py::ne(&t4154, &t4155) { true } else { false } }) }; if !t4162.truthy() { t4162 } else { { let t4156 = { let t4158 = v_v.clone(); Py::Bool({ let t4159 = Py::Int(0i64); if py::gt(&t4158, &t4159)? { true } else { false } }) }; Py::Bool({ let t4157 = { let t4160 = py::getitem(&v_s.clone(), &v_j.clone())?; Py::Bool({ let t4161 = Py::Int(0i64); if py::gt(&t4160, &t4161)? { true } else { false } }) }; if py::eq(&t4156, &t4157) { true } else { false } }) } } }, Py::str("the lasso solution is degenerate (a coefficient sits exactly at zero)"))?;
            py::setitem(&v_wF.clone(), &v_j.clone(), v_v.clone())?;
        }
    }
    v_resF = { let mut out = vec![]; for t4164 in py::iter(&(py::zip(&[v_XcF.clone(), v_ycF.clone()])?))? { let mut v_r: Py; let mut v_t: Py; 
let t4165 = (t4164).items()?;
if t4165.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_r = t4165[0].clone();
v_t = t4165[1].clone();
out.push(py::sub(&v_t.clone(), &py::sum(&{ let mut out = vec![]; for t4166 in py::range_iter(&[(v_d.clone()).i()?]) { let mut v_j: Py; 
v_j = t4166;
out.push(py::mul(&py::getitem(&v_r.clone(), &v_j.clone())?, &py::getitem(&v_wF.clone(), &v_j.clone())?)?); } Py::list(out) }, None)?)?); } Py::list(out) };
    for t4167 in py::range_iter(&[(v_d.clone()).i()?]) {
        v_j = t4167;
        if ({ let t4168 = v_j.clone(); Py::Bool({ let t4169 = v_s.clone(); if !py::contains(&t4169, &t4168)? { true } else { false } }) }).truthy() {
            v_c = py::abs(&py::sum(&{ let mut out = vec![]; for t4170 in py::iter(&(py::zip(&[v_XcF.clone(), v_resF.clone()])?))? { let mut v_e: Py; let mut v_r: Py; 
let t4171 = (t4170).items()?;
if t4171.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_r = t4171[0].clone();
v_e = t4171[1].clone();
out.push(py::mul(&py::getitem(&v_r.clone(), &v_j.clone())?, &v_e.clone())?); } Py::list(out) }, None)?)?;
            let _ = f__need({ let t4172 = v_c.clone(); Py::Bool({ let t4173 = py::mul(&v_lamF.clone(), &py::sub(&Py::Int(1i64), &py::fraction(&[Py::Int(1i64), py::pow(&Py::Int(10i64), &Py::Int(9i64))?])?)?)?; if py::lt(&t4172, &t4173)? { true } else { false } }) }, Py::str("the lasso solution is degenerate (an inactive feature is exactly at the threshold)"))?;
        }
    }
    v_b = py::sub(&v_myF.clone(), &py::sum(&{ let mut out = vec![]; for t4174 in py::iter(&(py::zip(&[v_mxF.clone(), v_wF.clone()])?))? { let mut v_a: Py; let mut v_c: Py; 
let t4175 = (t4174).items()?;
if t4175.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t4175[0].clone();
v_c = t4175[1].clone();
out.push(py::mul(&v_a.clone(), &v_c.clone())?); } Py::list(out) }, None)?)?;
    return Ok(f__out(Py::dict(vec![(Py::str("w"), v_wF.clone()), (Py::str("b"), v_b.clone()), (Py::str("n_nonzero"), py::len(&v_A.clone())?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_linkage_heights(mut v_X: Py, mut v_method: Py) -> R {
    let mut v_D: Py = Py::None;
    let mut v_M: Py = Py::None;
    let mut v_a: Py = Py::None;
    let mut v_active: Py = Py::None;
    let mut v_b: Py = Py::None;
    let mut v_c: Py = Py::None;
    let mut v_dac: Py = Py::None;
    let mut v_dbc: Py = Py::None;
    let mut v_h: Py = Py::None;
    let mut v_heights: Py = Py::None;
    let mut v_i: Py = Py::None;
    let mut v_j: Py = Py::None;
    let mut v_key: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_na: Py = Py::None;
    let mut v_nb: Py = Py::None;
    let mut v_nc: Py = Py::None;
    let mut v_nxt: Py = Py::None;
    let mut v_pairs: Py = Py::None;
    let mut v_size: Py = Py::None;
    let mut v_v: Py = Py::None;
    v_M = f__mat(Py::str("X"), v_X.clone(), Py::Int(2i64), Py::Int(1i64))?;
    v_n = py::len(&v_M.clone())?;
    let _ = f__need({ let t2882 = v_n.clone(); Py::Bool({ let t2883 = Py::Int(40i64); if py::le(&t2882, &t2883)? { true } else { false } }) }, Py::str("at most 40 points"))?;
    v_D = Py::dict(vec![]);
    for t2884 in py::range_iter(&[(v_n.clone()).i()?]) {
        v_i = t2884;
        for t2885 in py::range_iter(&[(py::add(&v_i.clone(), &Py::Int(1i64))?).i()?, (v_n.clone()).i()?]) {
            v_j = t2885;
            py::setitem(&v_D.clone(), &Py::tuple(vec![v_i.clone(), v_j.clone()]), py::math1("sqrt", &f__sqd(py::getitem(&v_M.clone(), &v_i.clone())?, py::getitem(&v_M.clone(), &v_j.clone())?)?)?)?;
        }
    }
    v_size = { let mut out = vec![]; for t2888 in py::range_iter(&[(v_n.clone()).i()?]) { let mut v_i: Py; 
v_i = t2888;
out.push((v_i.clone(), Py::Int(1i64))); } Py::dict(out) };
    v_active = py::set(&Py::list((py::range_iter(&[(v_n.clone()).i()?])).into_iter().collect::<Vec<Py>>()))?;
    v_nxt = v_n.clone();
    v_heights = Py::list(vec![]);
    v_key = { let v_D = v_D.clone(); let v_M = v_M.clone(); let v_X = v_X.clone(); let v_active = v_active.clone(); let v_heights = v_heights.clone(); let v_i = v_i.clone(); let v_j = v_j.clone(); let v_method = v_method.clone(); let v_n = v_n.clone(); let v_nxt = v_nxt.clone(); let v_size = v_size.clone(); py::func(move |a: &[Py]| -> R {
            #[allow(unused_mut, unused_variables, unused_assignments)]
{
            let mut v_a: Py = a.get(0).cloned().unwrap_or(Py::None);
            let mut v_b: Py = a.get(1).cloned().unwrap_or(Py::None);
            return Ok((if ({ let t2889 = v_a.clone(); Py::Bool({ let t2890 = v_b.clone(); if py::lt(&t2889, &t2890)? { true } else { false } }) }).truthy() { Py::tuple(vec![v_a.clone(), v_b.clone()]) } else { Py::tuple(vec![v_b.clone(), v_a.clone()]) }));
            #[allow(unreachable_code)]
            Ok(Py::None)
        }
        }) };
    while ({ let t2913 = py::len(&v_active.clone())?; Py::Bool({ let t2914 = Py::Int(1i64); if py::gt(&t2913, &t2914)? { true } else { false } }) }).truthy() {
        v_pairs = py::sorted(&{ let mut out = vec![]; for t2891 in py::iter(&(v_active.clone()))? { let mut v_a: Py; 
v_a = t2891;
for t2892 in py::iter(&(v_active.clone()))? { let mut v_b: Py; 
v_b = t2892;
if ({ let t2893 = v_a.clone(); Py::Bool({ let t2894 = v_b.clone(); if py::lt(&t2893, &t2894)? { true } else { false } }) }).truthy() { out.push(Py::tuple(vec![py::getitem(&v_D.clone(), &py::call(&v_key.clone(), &[v_a.clone(), v_b.clone()])?)?, py::call(&v_key.clone(), &[v_a.clone(), v_b.clone()])?])); }}} Py::list(out) }, (Py::Bool(false)).truthy())?;
        let t2895 = (py::getitem(&v_pairs.clone(), &Py::Int(0i64))?).items()?;
        if t2895.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_h = t2895[0].clone();
        let t2896 = (t2895[1].clone()).items()?;
        if t2896.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_a = t2896[0].clone();
        v_b = t2896[1].clone();
        if ({ let t2901 = { let t2897 = py::len(&v_pairs.clone())?; Py::Bool({ let t2898 = Py::Int(1i64); if py::gt(&t2897, &t2898)? { true } else { false } }) }; if !t2901.truthy() { t2901 } else { { let t2899 = v_method.clone(); Py::Bool({ let t2900 = Py::str("single"); if py::ne(&t2899, &t2900) { true } else { false } }) } } }).truthy() {
            let _ = f__need({ let t2902 = py::sub(&py::getitem(&py::getitem(&v_pairs.clone(), &Py::Int(1i64))?, &Py::Int(0i64))?, &v_h.clone())?; Py::Bool({ let t2903 = py::mul(&Py::Float(1e-12), &py::minmax(&[v_h.clone(), Py::Float(1e-300)], true)?)?; if py::gt(&t2902, &t2903)? { true } else { false } }) }, Py::str("two merges are tied at the same distance (the order is ambiguous)"))?;
        }
        let _ = py::method(&v_heights.clone(), "append", &[v_h.clone()])?;
        let t2904 = (Py::tuple(vec![py::getitem(&v_size.clone(), &v_a.clone())?, py::getitem(&v_size.clone(), &v_b.clone())?])).items()?;
        if t2904.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_na = t2904[0].clone();
        v_nb = t2904[1].clone();
        for t2905 in py::iter(&(py::sub(&v_active.clone(), &py::set(&Py::list(vec![v_a.clone(), v_b.clone()]))?)?))? {
            v_c = t2905;
            let t2906 = (Py::tuple(vec![py::getitem(&v_D.clone(), &py::call(&v_key.clone(), &[v_a.clone(), v_c.clone()])?)?, py::getitem(&v_D.clone(), &py::call(&v_key.clone(), &[v_b.clone(), v_c.clone()])?)?])).items()?;
            if t2906.len() != 2 { return Err(PyErr::Value("unpack".into())); }
            v_dac = t2906[0].clone();
            v_dbc = t2906[1].clone();
            if ({ let t2907 = v_method.clone(); Py::Bool({ let t2908 = Py::str("single"); if py::eq(&t2907, &t2908) { true } else { false } }) }).truthy() {
                v_v = py::minmax(&[v_dac.clone(), v_dbc.clone()], false)?;
            } else {
                if ({ let t2909 = v_method.clone(); Py::Bool({ let t2910 = Py::str("complete"); if py::eq(&t2909, &t2910) { true } else { false } }) }).truthy() {
                    v_v = py::minmax(&[v_dac.clone(), v_dbc.clone()], true)?;
                } else {
                    if ({ let t2911 = v_method.clone(); Py::Bool({ let t2912 = Py::str("average"); if py::eq(&t2911, &t2912) { true } else { false } }) }).truthy() {
                        v_v = py::truediv(&py::add(&py::mul(&v_na.clone(), &v_dac.clone())?, &py::mul(&v_nb.clone(), &v_dbc.clone())?)?, &py::add(&v_na.clone(), &v_nb.clone())?)?;
                    } else {
                        v_nc = py::getitem(&v_size.clone(), &v_c.clone())?;
                        v_v = py::math1("sqrt", &py::truediv(&py::sub(&py::add(&py::mul(&py::mul(&py::add(&v_na.clone(), &v_nc.clone())?, &v_dac.clone())?, &v_dac.clone())?, &py::mul(&py::mul(&py::add(&v_nb.clone(), &v_nc.clone())?, &v_dbc.clone())?, &v_dbc.clone())?)?, &py::mul(&py::mul(&v_nc.clone(), &v_h.clone())?, &v_h.clone())?)?, &py::add(&py::add(&v_na.clone(), &v_nb.clone())?, &v_nc.clone())?)?)?;
                    }
                }
            }
            py::setitem(&v_D.clone(), &py::call(&v_key.clone(), &[v_nxt.clone(), v_c.clone()])?, v_v.clone())?;
        }
        v_active = py::sub(&v_active, &py::set(&Py::list(vec![v_a.clone(), v_b.clone()]))?)?;
        let _ = py::method(&v_active.clone(), "add", &[v_nxt.clone()])?;
        py::setitem(&v_size.clone(), &v_nxt.clone(), py::add(&v_na.clone(), &v_nb.clone())?)?;
        v_nxt = py::iadd(&v_nxt, &Py::Int(1i64))?;
    }
    return Ok(f__out(Py::dict(vec![(Py::str("heights"), v_heights.clone()), (Py::str("final_height"), py::getitem(&v_heights.clone(), &py::neg(&Py::Int(1i64))?)?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_log_loss_binary(mut v_y_true: Py, mut v_p: Py) -> R {
    let mut v_ll: Py = Py::None;
    let mut v_pv: Py = Py::None;
    let mut v_q: Py = Py::None;
    let mut v_yt: Py = Py::None;
    v_yt = f__binary(Py::str("y_true"), v_y_true.clone(), Py::Int(1i64))?;
    v_pv = f__vec(Py::str("p"), v_p.clone(), Py::Int(1i64))?;
    let _ = f__same_len(v_yt.clone(), v_pv.clone(), Py::str("y_true"), Py::str("p"))?;
    for t4311 in py::iter(&(v_pv.clone()))? {
        v_q = t4311;
        let _ = f__need({ let t4312 = Py::Int(0i64); Py::Bool({ let t4313 = v_q.clone(); if py::lt(&t4312, &t4313)? { { let t4314 = Py::Int(1i64); if py::lt(&t4313, &t4314)? { true } else { false } } } else { false } }) }, Py::str("every probability must lie strictly between 0 and 1"))?;
    }
    v_ll = py::truediv(&py::neg(&py::fsum(&{ let mut out = vec![]; for t4315 in py::iter(&(py::zip(&[v_yt.clone(), v_pv.clone()])?))? { let mut v_q: Py; let mut v_y: Py; 
let t4316 = (t4315).items()?;
if t4316.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_y = t4316[0].clone();
v_q = t4316[1].clone();
out.push((if ({ let t4317 = v_y.clone(); Py::Bool({ let t4318 = Py::Int(1i64); if py::eq(&t4317, &t4318) { true } else { false } }) }).truthy() { py::math1("log", &v_q.clone())? } else { py::math1("log1p", &py::neg(&v_q.clone())?)? })); } Py::list(out) })?)?, &py::len(&v_yt.clone())?)?;
    return Ok(f__out(Py::dict(vec![(Py::str("log_loss"), v_ll.clone()), (Py::str("log_loss_bits"), py::truediv(&v_ll.clone(), &py::math1("log", &Py::Float(2.0))?)?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_log_loss_multiclass(mut v_y_true: Py, mut v_P: Py) -> R {
    let mut v_M: Py = Py::None;
    let mut v_k: Py = Py::None;
    let mut v_ll: Py = Py::None;
    let mut v_row: Py = Py::None;
    let mut v_tot: Py = Py::None;
    let mut v_y: Py = Py::None;
    let mut v_yt: Py = Py::None;
    v_yt = f__labels(Py::str("y_true"), v_y_true.clone(), Py::Int(1i64))?;
    v_M = f__mat(Py::str("P"), v_P.clone(), Py::Int(1i64), Py::Int(2i64))?;
    let _ = f__same_len(v_yt.clone(), v_M.clone(), Py::str("y_true"), Py::str("the rows of P"))?;
    v_k = py::len(&py::getitem(&v_M.clone(), &Py::Int(0i64))?)?;
    v_tot = Py::list(vec![]);
    for t4385 in py::iter(&(py::zip(&[v_yt.clone(), v_M.clone()])?))? {
        let t4397 = (t4385).items()?;
        if t4397.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_y = t4397[0].clone();
        v_row = t4397[1].clone();
        let _ = f__need({ let t4386 = Py::Int(0i64); Py::Bool({ let t4387 = v_y.clone(); if py::le(&t4386, &t4387)? { { let t4388 = v_k.clone(); if py::lt(&t4387, &t4388)? { true } else { false } } } else { false } }) }, Py::str("every label must be a column index of P (0 to K-1)"))?;
        let _ = f__need(Py::Bool(({ let mut out = vec![]; for t4389 in py::iter(&(v_row.clone()))? { let mut v_q: Py; 
v_q = t4389;
out.push({ let t4390 = Py::Int(0i64); Py::Bool({ let t4391 = v_q.clone(); if py::le(&t4390, &t4391)? { { let t4392 = Py::Int(1i64); if py::le(&t4391, &t4392)? { true } else { false } } } else { false } }) }); } Py::list(out) }).items()?.iter().all(|x| x.truthy())), Py::str("probabilities must lie between 0 and 1"))?;
        let _ = f__need({ let t4393 = py::abs(&py::sub(&py::fsum(&v_row.clone())?, &Py::Int(1i64))?)?; Py::Bool({ let t4394 = Py::Float(1e-09); if py::le(&t4393, &t4394)? { true } else { false } }) }, Py::str("every row of P must sum to 1"))?;
        let _ = f__need({ let t4395 = py::getitem(&v_row.clone(), &v_y.clone())?; Py::Bool({ let t4396 = Py::Int(0i64); if py::gt(&t4395, &t4396)? { true } else { false } }) }, Py::str("the probability of the true class must be positive"))?;
        let _ = py::method(&v_tot.clone(), "append", &[py::math1("log", &py::getitem(&v_row.clone(), &v_y.clone())?)?])?;
    }
    v_ll = py::truediv(&py::neg(&py::fsum(&v_tot.clone())?)?, &py::len(&v_yt.clone())?)?;
    return Ok(f__out(Py::dict(vec![(Py::str("log_loss"), v_ll.clone()), (Py::str("perplexity"), py::math1("exp", &v_ll.clone())?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_log_softmax(mut v_z: Py) -> R {
    let mut v_m: Py = Py::None;
    let mut v_s: Py = Py::None;
    let mut v_zv: Py = Py::None;
    v_zv = f__vec(Py::str("z"), v_z.clone(), Py::Int(2i64))?;
    v_m = py::minmax(&[v_zv.clone()], true)?;
    v_s = py::math1("log", &py::fsum(&{ let mut out = vec![]; for t4447 in py::iter(&(v_zv.clone()))? { let mut v_v: Py; 
v_v = t4447;
out.push(py::math1("exp", &py::sub(&v_v.clone(), &v_m.clone())?)?); } Py::list(out) })?)?;
    return Ok(f__out(Py::dict(vec![(Py::str("log_probabilities"), { let mut out = vec![]; for t4460 in py::iter(&(v_zv.clone()))? { let mut v_v: Py; 
v_v = t4460;
out.push(py::sub(&py::sub(&v_v.clone(), &v_m.clone())?, &v_s.clone())?); } Py::list(out) }), (Py::str("log_sum_exp"), py::add(&v_m.clone(), &v_s.clone())?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_logistic_fit(mut v_X: Py, mut v_y: Py, mut v_C: Py) -> R {
    let mut v_H: Py = Py::None;
    let mut v_M: Py = Py::None;
    let mut v_acc: Py = Py::None;
    let mut v_d: Py = Py::None;
    let mut v_g: Py = Py::None;
    let mut v_gnorm: Py = Py::None;
    let mut v_it: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_p: Py = Py::None;
    let mut v_r_: Py = Py::None;
    let mut v_s: Py = Py::None;
    let mut v_step: Py = Py::None;
    let mut v_th: Py = Py::None;
    let mut v_yv: Py = Py::None;
    let mut v_z: Py = Py::None;
    v_M = f__mat(Py::str("X"), v_X.clone(), Py::Int(2i64), Py::Int(1i64))?;
    v_yv = f__vec(Py::str("y"), v_y.clone(), Py::Int(1i64))?;
    let _ = f__need({ let t4598 = py::len(&v_yv.clone())?; Py::Bool({ let t4599 = py::len(&v_M.clone())?; if py::eq(&t4598, &t4599) { true } else { false } }) }, Py::str("y needs one label per row of X"))?;
    let _ = f__need({ let t4605 = Py::Bool(({ let mut out = vec![]; for t4600 in py::iter(&(v_yv.clone()))? { let mut v_v: Py; 
v_v = t4600;
out.push({ let t4601 = v_v.clone(); Py::Bool({ let t4602 = Py::tuple(vec![Py::Float(0.0), Py::Float(1.0)]); if py::contains(&t4602, &t4601)? { true } else { false } }) }); } Py::list(out) }).items()?.iter().all(|x| x.truthy())); if !t4605.truthy() { t4605 } else { { let t4603 = py::len(&py::set(&v_yv.clone())?)?; Py::Bool({ let t4604 = Py::Int(2i64); if py::eq(&t4603, &t4604) { true } else { false } }) } } }, Py::str("labels must be 0 and 1, both present"))?;
    let _ = f__fin(Py::str("C"), Py::tuple(vec![v_C.clone()]))?;
    let _ = f__need({ let t4606 = v_C.clone(); Py::Bool({ let t4607 = Py::Int(0i64); if py::gt(&t4606, &t4607)? { true } else { false } }) }, Py::str("C (inverse regularisation strength) must be positive"))?;
    let t4608 = (Py::tuple(vec![py::len(&v_M.clone())?, py::len(&py::getitem(&v_M.clone(), &Py::Int(0i64))?)?])).items()?;
    if t4608.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_n = t4608[0].clone();
    v_d = t4608[1].clone();
    v_th = py::mul(&Py::list(vec![Py::Float(0.0)]), &py::add(&v_d.clone(), &Py::Int(1i64))?)?;
    let mut t4609 = false;
    for t4610 in py::range_iter(&[(Py::Int(200i64)).i()?]) {
        v_it = t4610;
        v_z = { let mut out = vec![]; for t4611 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t4611;
out.push(py::add(&py::fsum(&{ let mut out = vec![]; for t4612 in py::iter(&(py::zip(&[v_r.clone(), py::slice(&v_th.clone(), None, Some((v_d.clone()).i()?), None)?])?))? { let mut v_a: Py; let mut v_c: Py; 
let t4613 = (t4612).items()?;
if t4613.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t4613[0].clone();
v_c = t4613[1].clone();
out.push(py::mul(&v_a.clone(), &v_c.clone())?); } Py::list(out) })?, &py::getitem(&v_th.clone(), &v_d.clone())?)?); } Py::list(out) };
        v_p = { let mut out = vec![]; for t4614 in py::iter(&(v_z.clone()))? { let mut v_v: Py; 
v_v = t4614;
out.push(f__sigmoid(v_v.clone())?); } Py::list(out) };
        v_r_ = { let mut out = vec![]; for t4617 in py::iter(&(py::zip(&[v_p.clone(), v_yv.clone()])?))? { let mut v_pi: Py; let mut v_t: Py; 
let t4618 = (t4617).items()?;
if t4618.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_pi = t4618[0].clone();
v_t = t4618[1].clone();
out.push(py::sub(&v_pi.clone(), &v_t.clone())?); } Py::list(out) };
        v_g = py::add(&{ let mut out = vec![]; for t4619 in py::range_iter(&[(v_d.clone()).i()?]) { let mut v_j: Py; 
v_j = t4619;
out.push(py::add(&py::getitem(&v_th.clone(), &v_j.clone())?, &py::mul(&v_C.clone(), &py::fsum(&{ let mut out = vec![]; for t4620 in py::range_iter(&[(v_n.clone()).i()?]) { let mut v_i: Py; 
v_i = t4620;
out.push(py::mul(&py::getitem(&v_r_.clone(), &v_i.clone())?, &py::getitem(&py::getitem(&v_M.clone(), &v_i.clone())?, &v_j.clone())?)?); } Py::list(out) })?)?)?); } Py::list(out) }, &Py::list(vec![py::mul(&v_C.clone(), &py::fsum(&v_r_.clone())?)?]))?;
        v_s = { let mut out = vec![]; for t4621 in py::iter(&(v_p.clone()))? { let mut v_pi: Py; 
v_pi = t4621;
out.push(py::mul(&v_pi.clone(), &py::sub(&Py::Int(1i64), &v_pi.clone())?)?); } Py::list(out) };
        v_H = { let mut out = vec![]; for t4622 in py::range_iter(&[(py::add(&v_d.clone(), &Py::Int(1i64))?).i()?]) { let mut v_i: Py; 
v_i = t4622;
out.push({ let mut out = vec![]; for t4623 in py::range_iter(&[(py::add(&v_d.clone(), &Py::Int(1i64))?).i()?]) { let mut v_j: Py; 
v_j = t4623;
out.push(py::add(&(if ({ let t4628 = { let t4624 = v_i.clone(); Py::Bool({ let t4625 = v_j.clone(); if py::eq(&t4624, &t4625) { true } else { false } }) }; if !t4628.truthy() { t4628 } else { { let t4626 = v_i.clone(); Py::Bool({ let t4627 = v_d.clone(); if py::lt(&t4626, &t4627)? { true } else { false } }) } } }).truthy() { Py::Float(1.0) } else { Py::Float(0.0) }), &py::mul(&v_C.clone(), &py::fsum(&{ let mut out = vec![]; for t4629 in py::range_iter(&[(v_n.clone()).i()?]) { let mut v_t: Py; 
v_t = t4629;
out.push(py::mul(&py::mul(&py::getitem(&v_s.clone(), &v_t.clone())?, &(if ({ let t4630 = v_i.clone(); Py::Bool({ let t4631 = v_d.clone(); if py::lt(&t4630, &t4631)? { true } else { false } }) }).truthy() { py::getitem(&py::getitem(&v_M.clone(), &v_t.clone())?, &v_i.clone())? } else { Py::Float(1.0) }))?, &(if ({ let t4632 = v_j.clone(); Py::Bool({ let t4633 = v_d.clone(); if py::lt(&t4632, &t4633)? { true } else { false } }) }).truthy() { py::getitem(&py::getitem(&v_M.clone(), &v_t.clone())?, &v_j.clone())? } else { Py::Float(1.0) }))?); } Py::list(out) })?)?)?); } Py::list(out) }); } Py::list(out) };
        v_step = f__sym_solve(v_H.clone(), v_g.clone())?;
        v_th = { let mut out = vec![]; for t4651 in py::iter(&(py::zip(&[v_th.clone(), v_step.clone()])?))? { let mut v_a: Py; let mut v_b: Py; 
let t4652 = (t4651).items()?;
if t4652.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t4652[0].clone();
v_b = t4652[1].clone();
out.push(py::sub(&v_a.clone(), &v_b.clone())?); } Py::list(out) };
        v_gnorm = py::minmax(&[{ let mut out = vec![]; for t4653 in py::iter(&(v_g.clone()))? { let mut v_v: Py; 
v_v = t4653;
out.push(py::abs(&v_v.clone())?); } Py::list(out) }], true)?;
        if ({ let t4660 = { let t4654 = py::minmax(&[{ let mut out = vec![]; for t4656 in py::iter(&(v_step.clone()))? { let mut v_v: Py; 
v_v = t4656;
out.push(py::abs(&v_v.clone())?); } Py::list(out) }], true)?; Py::Bool({ let t4655 = py::mul(&Py::Float(1e-15), &py::minmax(&[Py::Float(1.0), py::minmax(&[{ let mut out = vec![]; for t4657 in py::iter(&(v_th.clone()))? { let mut v_v: Py; 
v_v = t4657;
out.push(py::abs(&v_v.clone())?); } Py::list(out) }], true)?], true)?)?; if py::le(&t4654, &t4655)? { true } else { false } }) }; if !t4660.truthy() { t4660 } else { { let t4658 = v_it.clone(); Py::Bool({ let t4659 = Py::Int(2i64); if py::gt(&t4658, &t4659)? { true } else { false } }) } } }).truthy() {
            { t4609 = true; break; }
        }
    }
    if !t4609 {
        return Err(py::value_error(py::to_str(&Py::str("Newton's method did not converge"))));
    }
    v_z = { let mut out = vec![]; for t4661 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t4661;
out.push(py::add(&py::fsum(&{ let mut out = vec![]; for t4662 in py::iter(&(py::zip(&[v_r.clone(), py::slice(&v_th.clone(), None, Some((v_d.clone()).i()?), None)?])?))? { let mut v_a: Py; let mut v_c: Py; 
let t4663 = (t4662).items()?;
if t4663.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t4663[0].clone();
v_c = t4663[1].clone();
out.push(py::mul(&v_a.clone(), &v_c.clone())?); } Py::list(out) })?, &py::getitem(&v_th.clone(), &v_d.clone())?)?); } Py::list(out) };
    v_acc = py::fraction(&[py::sum(&{ let mut out = vec![]; for t4664 in py::iter(&(py::zip(&[v_z.clone(), v_yv.clone()])?))? { let mut v_t: Py; let mut v_v: Py; 
let t4665 = (t4664).items()?;
if t4665.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_v = t4665[0].clone();
v_t = t4665[1].clone();
if ({ let t4666 = { let t4668 = v_v.clone(); Py::Bool({ let t4669 = Py::Int(0i64); if py::gt(&t4668, &t4669)? { true } else { false } }) }; Py::Bool({ let t4667 = { let t4670 = v_t.clone(); Py::Bool({ let t4671 = Py::Int(1i64); if py::eq(&t4670, &t4671) { true } else { false } }) }; if py::eq(&t4666, &t4667) { true } else { false } }) }).truthy() { out.push(Py::Int(1i64)); }} Py::list(out) }, None)?, v_n.clone()])?;
    return Ok(f__out(Py::dict(vec![(Py::str("w"), py::slice(&v_th.clone(), None, Some((v_d.clone()).i()?), None)?), (Py::str("b"), py::getitem(&v_th.clone(), &v_d.clone())?), (Py::str("objective"), f__logit_obj(v_M.clone(), v_yv.clone(), py::slice(&v_th.clone(), None, Some((v_d.clone()).i()?), None)?, py::getitem(&v_th.clone(), &v_d.clone())?, v_C.clone())?), (Py::str("train_accuracy"), v_acc.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_logistic_gradient(mut v_X: Py, mut v_y: Py, mut v_w: Py, mut v_b: Py, mut v_lam: Py) -> R {
    let mut v_M: Py = Py::None;
    let mut v_gw: Py = Py::None;
    let mut v_loss: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_res: Py = Py::None;
    let mut v_wv: Py = Py::None;
    let mut v_yv: Py = Py::None;
    let mut v_z: Py = Py::None;
    let t4517 = (f__linear_parts(v_X.clone(), v_y.clone(), v_w.clone(), v_b.clone())?).items()?;
    if t4517.len() != 3 { return Err(PyErr::Value("unpack".into())); }
    v_M = t4517[0].clone();
    v_yv = t4517[1].clone();
    v_wv = t4517[2].clone();
    let _ = f__need(Py::Bool(({ let mut out = vec![]; for t4518 in py::iter(&(v_yv.clone()))? { let mut v_v: Py; 
v_v = t4518;
out.push({ let t4519 = v_v.clone(); Py::Bool({ let t4520 = Py::tuple(vec![Py::Float(0.0), Py::Float(1.0)]); if py::contains(&t4520, &t4519)? { true } else { false } }) }); } Py::list(out) }).items()?.iter().all(|x| x.truthy())), Py::str("labels y must be 0 or 1"))?;
    let _ = f__fin(Py::str("lam"), Py::tuple(vec![v_lam.clone()]))?;
    let _ = f__need({ let t4521 = v_lam.clone(); Py::Bool({ let t4522 = Py::Int(0i64); if py::ge(&t4521, &t4522)? { true } else { false } }) }, Py::str("the L2 strength lam must be >= 0"))?;
    v_n = py::len(&v_yv.clone())?;
    v_z = { let mut out = vec![]; for t4523 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t4523;
out.push(py::add(&py::fsum(&{ let mut out = vec![]; for t4524 in py::iter(&(py::zip(&[v_r.clone(), v_wv.clone()])?))? { let mut v_a: Py; let mut v_c: Py; 
let t4525 = (t4524).items()?;
if t4525.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t4525[0].clone();
v_c = t4525[1].clone();
out.push(py::mul(&v_a.clone(), &v_c.clone())?); } Py::list(out) })?, &v_b.clone())?); } Py::list(out) };
    v_loss = py::add(&py::truediv(&py::fsum(&{ let mut out = vec![]; for t4526 in py::iter(&(py::zip(&[v_z.clone(), v_yv.clone()])?))? { let mut v_t: Py; let mut v_v: Py; 
let t4527 = (t4526).items()?;
if t4527.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_v = t4527[0].clone();
v_t = t4527[1].clone();
out.push(py::add(&py::sub(&py::minmax(&[v_v.clone(), Py::Float(0.0)], true)?, &py::mul(&v_v.clone(), &v_t.clone())?)?, &py::math1("log1p", &py::math1("exp", &py::neg(&py::abs(&v_v.clone())?)?)?)?)?); } Py::list(out) })?, &v_n.clone())?, &py::mul(&py::truediv(&v_lam.clone(), &Py::Int(2i64))?, &py::fsum(&{ let mut out = vec![]; for t4528 in py::iter(&(v_wv.clone()))? { let mut v_c: Py; 
v_c = t4528;
out.push(py::mul(&v_c.clone(), &v_c.clone())?); } Py::list(out) })?)?)?;
    v_res = { let mut out = vec![]; for t4529 in py::iter(&(py::zip(&[v_z.clone(), v_yv.clone()])?))? { let mut v_t: Py; let mut v_v: Py; 
let t4530 = (t4529).items()?;
if t4530.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_v = t4530[0].clone();
v_t = t4530[1].clone();
out.push(py::sub(&f__sigmoid(v_v.clone())?, &v_t.clone())?); } Py::list(out) };
    v_gw = { let mut out = vec![]; for t4533 in py::range_iter(&[(py::len(&v_wv.clone())?).i()?]) { let mut v_j: Py; 
v_j = t4533;
out.push(py::add(&py::truediv(&py::fsum(&{ let mut out = vec![]; for t4534 in py::range_iter(&[(v_n.clone()).i()?]) { let mut v_i: Py; 
v_i = t4534;
out.push(py::mul(&py::getitem(&v_res.clone(), &v_i.clone())?, &py::getitem(&py::getitem(&v_M.clone(), &v_i.clone())?, &v_j.clone())?)?); } Py::list(out) })?, &v_n.clone())?, &py::mul(&v_lam.clone(), &py::getitem(&v_wv.clone(), &v_j.clone())?)?)?); } Py::list(out) };
    return Ok(f__out(Py::dict(vec![(Py::str("loss"), v_loss.clone()), (Py::str("grad_w"), v_gw.clone()), (Py::str("grad_b"), py::truediv(&py::fsum(&v_res.clone())?, &v_n.clone())?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_logistic_predict(mut v_w: Py, mut v_b: Py, mut v_X: Py) -> R {
    let mut v_M: Py = Py::None;
    let mut v_wv: Py = Py::None;
    let mut v_z: Py = Py::None;
    v_wv = f__vec(Py::str("w"), v_w.clone(), Py::Int(1i64))?;
    let _ = f__fin(Py::str("b"), Py::tuple(vec![v_b.clone()]))?;
    v_M = f__mat(Py::str("X"), v_X.clone(), Py::Int(1i64), Py::Int(1i64))?;
    let _ = f__need({ let t4741 = py::len(&py::getitem(&v_M.clone(), &Py::Int(0i64))?)?; Py::Bool({ let t4742 = py::len(&v_wv.clone())?; if py::eq(&t4741, &t4742) { true } else { false } }) }, Py::str("w needs one weight per column of X"))?;
    v_z = { let mut out = vec![]; for t4743 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t4743;
out.push(py::add(&py::fsum(&{ let mut out = vec![]; for t4744 in py::iter(&(py::zip(&[v_r.clone(), v_wv.clone()])?))? { let mut v_a: Py; let mut v_c: Py; 
let t4745 = (t4744).items()?;
if t4745.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t4745[0].clone();
v_c = t4745[1].clone();
out.push(py::mul(&v_a.clone(), &v_c.clone())?); } Py::list(out) })?, &v_b.clone())?); } Py::list(out) };
    return Ok(f__out(Py::dict(vec![(Py::str("probabilities"), { let mut out = vec![]; for t4758 in py::iter(&(v_z.clone()))? { let mut v_v: Py; 
v_v = t4758;
out.push(f__sigmoid(v_v.clone())?); } Py::list(out) }), (Py::str("classes"), { let mut out = vec![]; for t4761 in py::iter(&(v_z.clone()))? { let mut v_v: Py; 
v_v = t4761;
out.push((if ({ let t4762 = v_v.clone(); Py::Bool({ let t4763 = Py::Int(0i64); if py::gt(&t4762, &t4763)? { true } else { false } }) }).truthy() { Py::Int(1i64) } else { Py::Int(0i64) })); } Py::list(out) }), (Py::str("logits"), v_z.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_loocv_ols(mut v_X: Py, mut v_y: Py) -> R {
    let mut v_A: Py = Py::None;
    let mut v_G: Py = Py::None;
    let mut v_Ginv: Py = Py::None;
    let mut v_M: Py = Py::None;
    let mut v_a: Py = Py::None;
    let mut v_b: Py = Py::None;
    let mut v_d: Py = Py::None;
    let mut v_e: Py = Py::None;
    let mut v_h: Py = Py::None;
    let mut v_i: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_press: Py = Py::None;
    let mut v_w: Py = Py::None;
    let mut v_yv: Py = Py::None;
    v_M = f__mat(Py::str("X"), v_X.clone(), Py::Int(1i64), Py::Int(1i64))?;
    v_yv = f__vec(Py::str("y"), v_y.clone(), Py::Int(1i64))?;
    let t4815 = (Py::tuple(vec![py::len(&v_M.clone())?, py::len(&py::getitem(&v_M.clone(), &Py::Int(0i64))?)?])).items()?;
    if t4815.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_n = t4815[0].clone();
    v_d = t4815[1].clone();
    let _ = f__need({ let t4816 = py::len(&v_yv.clone())?; Py::Bool({ let t4817 = v_n.clone(); if py::eq(&t4816, &t4817) { true } else { false } }) }, Py::str("y needs one value per row of X"))?;
    let _ = f__need({ let t4818 = v_n.clone(); Py::Bool({ let t4819 = py::add(&v_d.clone(), &Py::Int(3i64))?; if py::ge(&t4818, &t4819)? { true } else { false } }) }, Py::str("LOOCV needs at least d + 3 samples"))?;
    v_A = { let mut out = vec![]; for t4820 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t4820;
out.push(py::add(&Py::list(vec![py::fraction(&[Py::Int(1i64)])?]), &{ let mut out = vec![]; for t4821 in py::iter(&(v_r.clone()))? { let mut v_v: Py; 
v_v = t4821;
out.push(f__F(v_v.clone())?); } Py::list(out) })?); } Py::list(out) };
    v_G = { let mut out = vec![]; for t4822 in py::range_iter(&[(py::add(&v_d.clone(), &Py::Int(1i64))?).i()?]) { let mut v_i: Py; 
v_i = t4822;
out.push({ let mut out = vec![]; for t4823 in py::range_iter(&[(py::add(&v_d.clone(), &Py::Int(1i64))?).i()?]) { let mut v_j: Py; 
v_j = t4823;
out.push(py::sum(&{ let mut out = vec![]; for t4824 in py::iter(&(v_A.clone()))? { let mut v_a: Py; 
v_a = t4824;
out.push(py::mul(&py::getitem(&v_a.clone(), &v_i.clone())?, &py::getitem(&v_a.clone(), &v_j.clone())?)?); } Py::list(out) }, None)?); } Py::list(out) }); } Py::list(out) };
    v_Ginv = { let mut out = vec![]; for t4825 in py::range_iter(&[(py::add(&v_d.clone(), &Py::Int(1i64))?).i()?]) { let mut v_j: Py; 
v_j = t4825;
out.push(f__solve_exact(v_G.clone(), { let mut out = vec![]; for t4847 in py::range_iter(&[(py::add(&v_d.clone(), &Py::Int(1i64))?).i()?]) { let mut v_i: Py; 
v_i = t4847;
out.push(py::fraction(&[py::int(&{ let t4848 = v_i.clone(); Py::Bool({ let t4849 = v_j.clone(); if py::eq(&t4848, &t4849) { true } else { false } }) })?])?); } Py::list(out) })?); } Py::list(out) };
    let t4858 = (f__ols_exact(v_M.clone(), v_yv.clone())?).items()?;
    if t4858.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_w = t4858[0].clone();
    v_b = t4858[1].clone();
    v_press = py::fraction(&[Py::Int(0i64)])?;
    for t4859 in py::range_iter(&[(v_n.clone()).i()?]) {
        v_i = t4859;
        v_a = py::getitem(&v_A.clone(), &v_i.clone())?;
        v_h = py::sum(&{ let mut out = vec![]; for t4860 in py::range_iter(&[(py::add(&v_d.clone(), &Py::Int(1i64))?).i()?]) { let mut v_p: Py; 
v_p = t4860;
for t4861 in py::range_iter(&[(py::add(&v_d.clone(), &Py::Int(1i64))?).i()?]) { let mut v_q: Py; 
v_q = t4861;
out.push(py::mul(&py::mul(&py::getitem(&v_a.clone(), &v_p.clone())?, &py::getitem(&py::getitem(&v_Ginv.clone(), &v_q.clone())?, &v_p.clone())?)?, &py::getitem(&v_a.clone(), &v_q.clone())?)?); }} Py::list(out) }, None)?;
        let _ = f__need({ let t4862 = v_h.clone(); Py::Bool({ let t4863 = Py::Int(1i64); if py::ne(&t4862, &t4863) { true } else { false } }) }, Py::str("a sample has leverage 1: leaving it out makes the fit undetermined"))?;
        v_e = py::sub(&py::add(&py::sum(&{ let mut out = vec![]; for t4864 in py::iter(&(py::zip(&[v_w.clone(), py::getitem(&v_M.clone(), &v_i.clone())?])?))? { let mut v_v: Py; let mut v_wj: Py; 
let t4865 = (t4864).items()?;
if t4865.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_wj = t4865[0].clone();
v_v = t4865[1].clone();
out.push(py::mul(&v_wj.clone(), &f__F(v_v.clone())?)?); } Py::list(out) }, None)?, &v_b.clone())?, &f__F(py::getitem(&v_yv.clone(), &v_i.clone())?)?)?;
        v_press = py::iadd(&v_press, &py::pow(&py::truediv(&v_e.clone(), &py::sub(&Py::Int(1i64), &v_h.clone())?)?, &Py::Int(2i64))?)?;
    }
    return Ok(f__out(Py::dict(vec![(Py::str("press"), v_press.clone()), (Py::str("loocv_mse"), py::truediv(&v_press.clone(), &v_n.clone())?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_mape(mut v_y_true: Py, mut v_y_pred: Py) -> R {
    let mut v_m: Py = Py::None;
    let mut v_yp: Py = Py::None;
    let mut v_yt: Py = Py::None;
    let t4918 = (f__pair(v_y_true.clone(), v_y_pred.clone(), Py::Int(1i64))?).items()?;
    if t4918.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_yt = t4918[0].clone();
    v_yp = t4918[1].clone();
    let _ = f__need(Py::Bool(({ let mut out = vec![]; for t4919 in py::iter(&(v_yt.clone()))? { let mut v_v: Py; 
v_v = t4919;
out.push({ let t4920 = v_v.clone(); Py::Bool({ let t4921 = Py::Int(0i64); if py::ne(&t4920, &t4921) { true } else { false } }) }); } Py::list(out) }).items()?.iter().all(|x| x.truthy())), Py::str("MAPE is undefined when a true value is 0"))?;
    v_m = py::truediv(&py::sum(&{ let mut out = vec![]; for t4922 in py::iter(&(py::zip(&[v_yt.clone(), v_yp.clone()])?))? { let mut v_a: Py; let mut v_b: Py; 
let t4923 = (t4922).items()?;
if t4923.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t4923[0].clone();
v_b = t4923[1].clone();
out.push(py::truediv(&py::abs(&py::sub(&f__F(v_a.clone())?, &f__F(v_b.clone())?)?)?, &py::abs(&f__F(v_a.clone())?)?)?); } Py::list(out) }, None)?, &py::len(&v_yt.clone())?)?;
    return Ok(f__out(Py::dict(vec![(Py::str("mape"), v_m.clone()), (Py::str("mape_percent"), py::mul(&Py::Int(100i64), &v_m.clone())?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_mcc(mut v_TP: Py, mut v_FP: Py, mut v_FN: Py, mut v_TN: Py) -> R {
    let mut v_a: Py = Py::None;
    let mut v_b: Py = Py::None;
    let mut v_c: Py = Py::None;
    let mut v_d: Py = Py::None;
    let mut v_den2: Py = Py::None;
    let mut v_num: Py = Py::None;
    let _ = f__check_counts(v_TP.clone(), v_FP.clone(), v_FN.clone(), v_TN.clone())?;
    let t4970 = (Py::tuple(vec![py::add(&v_TP.clone(), &v_FP.clone())?, py::add(&v_TP.clone(), &v_FN.clone())?, py::add(&v_TN.clone(), &v_FP.clone())?, py::add(&v_TN.clone(), &v_FN.clone())?])).items()?;
    if t4970.len() != 4 { return Err(PyErr::Value("unpack".into())); }
    v_a = t4970[0].clone();
    v_b = t4970[1].clone();
    v_c = t4970[2].clone();
    v_d = t4970[3].clone();
    let _ = f__need({ let t4971 = py::mul(&py::mul(&py::mul(&v_a.clone(), &v_b.clone())?, &v_c.clone())?, &v_d.clone())?; Py::Bool({ let t4972 = Py::Int(0i64); if py::gt(&t4971, &t4972)? { true } else { false } }) }, Py::str("the MCC is undefined when a row or column of the confusion matrix is empty"))?;
    v_num = py::sub(&py::mul(&v_TP.clone(), &v_TN.clone())?, &py::mul(&v_FP.clone(), &v_FN.clone())?)?;
    v_den2 = py::mul(&py::mul(&py::mul(&v_a.clone(), &v_b.clone())?, &v_c.clone())?, &v_d.clone())?;
    return Ok(f__out(Py::dict(vec![(Py::str("mcc"), (if (v_num.clone()).truthy() { py::truediv(&v_num.clone(), &py::math1("sqrt", &v_den2.clone())?)? } else { Py::Float(0.0) }))]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_mean_ap(mut v_R: Py) -> R {
    let mut v_M: Py = Py::None;
    let mut v_aps: Py = Py::None;
    let mut v_hits: Py = Py::None;
    let mut v_i: Py = Py::None;
    let mut v_nrel: Py = Py::None;
    let mut v_r: Py = Py::None;
    let mut v_row: Py = Py::None;
    let mut v_s: Py = Py::None;
    let mut v_v: Py = Py::None;
    v_M = f__mat(Py::str("R"), v_R.clone(), Py::Int(1i64), Py::Int(1i64))?;
    v_aps = Py::list(vec![]);
    for t5031 in py::iter(&(v_M.clone()))? {
        v_row = t5031;
        v_r = f__rel01(Py::str("each row of R"), v_row.clone())?;
        v_nrel = py::sum(&v_r.clone(), None)?;
        let _ = f__need({ let t5041 = v_nrel.clone(); Py::Bool({ let t5042 = Py::Int(0i64); if py::gt(&t5041, &t5042)? { true } else { false } }) }, Py::str("a query has no relevant item (its average precision is undefined)"))?;
        let t5043 = (Py::tuple(vec![Py::Int(0i64), py::fraction(&[Py::Int(0i64)])?])).items()?;
        if t5043.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_hits = t5043[0].clone();
        v_s = t5043[1].clone();
        for t5044 in py::iter(&(py::enumerate(&v_r.clone(), (Py::Int(1i64)).i()?)?))? {
            let t5045 = (t5044).items()?;
            if t5045.len() != 2 { return Err(PyErr::Value("unpack".into())); }
            v_i = t5045[0].clone();
            v_v = t5045[1].clone();
            if (v_v.clone()).truthy() {
                v_hits = py::iadd(&v_hits, &Py::Int(1i64))?;
                v_s = py::iadd(&v_s, &py::fraction(&[v_hits.clone(), v_i.clone()])?)?;
            }
        }
        let _ = py::method(&v_aps.clone(), "append", &[py::truediv(&v_s.clone(), &v_nrel.clone())?])?;
    }
    return Ok(f__out(Py::dict(vec![(Py::str("average_precision"), v_aps.clone()), (Py::str("map"), py::truediv(&py::sum(&v_aps.clone(), None)?, &py::len(&v_aps.clone())?)?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_minmax_scaler(mut v_X: Py) -> R {
    let mut v_F: Py = Py::None;
    let mut v_M: Py = Py::None;
    let mut v_S: Py = Py::None;
    let mut v_d: Py = Py::None;
    let mut v_hi: Py = Py::None;
    let mut v_lo: Py = Py::None;
    let mut v_n: Py = Py::None;
    v_M = f__mat(Py::str("X"), v_X.clone(), Py::Int(1i64), Py::Int(1i64))?;
    let t5232 = (Py::tuple(vec![py::len(&v_M.clone())?, py::len(&py::getitem(&v_M.clone(), &Py::Int(0i64))?)?])).items()?;
    if t5232.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_n = t5232[0].clone();
    v_d = t5232[1].clone();
    v_F = { let mut out = vec![]; for t5233 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t5233;
out.push({ let mut out = vec![]; for t5234 in py::iter(&(v_r.clone()))? { let mut v_v: Py; 
v_v = t5234;
out.push(f__F(v_v.clone())?); } Py::list(out) }); } Py::list(out) };
    v_lo = { let mut out = vec![]; for t5235 in py::range_iter(&[(v_d.clone()).i()?]) { let mut v_j: Py; 
v_j = t5235;
out.push(py::minmax(&[{ let mut out = vec![]; for t5236 in py::iter(&(v_F.clone()))? { let mut v_r: Py; 
v_r = t5236;
out.push(py::getitem(&v_r.clone(), &v_j.clone())?); } Py::list(out) }], false)?); } Py::list(out) };
    v_hi = { let mut out = vec![]; for t5237 in py::range_iter(&[(v_d.clone()).i()?]) { let mut v_j: Py; 
v_j = t5237;
out.push(py::minmax(&[{ let mut out = vec![]; for t5238 in py::iter(&(v_F.clone()))? { let mut v_r: Py; 
v_r = t5238;
out.push(py::getitem(&v_r.clone(), &v_j.clone())?); } Py::list(out) }], true)?); } Py::list(out) };
    v_S = { let mut out = vec![]; for t5239 in py::iter(&(v_F.clone()))? { let mut v_r: Py; 
v_r = t5239;
out.push({ let mut out = vec![]; for t5240 in py::range_iter(&[(v_d.clone()).i()?]) { let mut v_j: Py; 
v_j = t5240;
out.push((if ({ let t5241 = py::getitem(&v_hi.clone(), &v_j.clone())?; Py::Bool({ let t5242 = py::getitem(&v_lo.clone(), &v_j.clone())?; if py::gt(&t5241, &t5242)? { true } else { false } }) }).truthy() { py::truediv(&py::sub(&py::getitem(&v_r.clone(), &v_j.clone())?, &py::getitem(&v_lo.clone(), &v_j.clone())?)?, &py::sub(&py::getitem(&v_hi.clone(), &v_j.clone())?, &py::getitem(&v_lo.clone(), &v_j.clone())?)?)? } else { py::fraction(&[Py::Int(0i64)])? })); } Py::list(out) }); } Py::list(out) };
    return Ok(f__out(Py::dict(vec![(Py::str("mins"), v_lo.clone()), (Py::str("maxs"), v_hi.clone()), (Py::str("scaled"), v_S.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_momentum_step(mut v_w: Py, mut v_grad: Py, mut v_buf: Py, mut v_lr: Py, mut v_momentum: Py, mut v_nesterov: Py, mut v_weight_decay: Py) -> R {
    let mut v_B: Py = Py::None;
    let mut v_B2: Py = Py::None;
    let mut v_G: Py = Py::None;
    let mut v_L: Py = Py::None;
    let mut v_V: Py = Py::None;
    let mut v_W: Py = Py::None;
    let mut v_d: Py = Py::None;
    let mut v_mu: Py = Py::None;
    let mut v_nm: Py = Py::None;
    let mut v_step: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_wd: Py = Py::None;
    v_V = f__opt_vecs(Py::dict(vec![(Py::str("w"), v_w.clone()), (Py::str("grad"), v_grad.clone()), (Py::str("buf"), v_buf.clone())]))?;
    for t7282 in py::iter(&(Py::tuple(vec![Py::tuple(vec![Py::str("lr"), v_lr.clone()]), Py::tuple(vec![Py::str("momentum"), v_momentum.clone()]), Py::tuple(vec![Py::str("weight_decay"), v_weight_decay.clone()])])))? {
        let t7283 = (t7282).items()?;
        if t7283.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_nm = t7283[0].clone();
        v_v = t7283[1].clone();
        let _ = f__fin(v_nm.clone(), Py::tuple(vec![v_v.clone()]))?;
    }
    let _ = f__need({ let t7292 = { let t7284 = v_lr.clone(); Py::Bool({ let t7285 = Py::Int(0i64); if py::gt(&t7284, &t7285)? { true } else { false } }) }; if !t7292.truthy() { t7292 } else { { let t7291 = { let t7286 = Py::Int(0i64); Py::Bool({ let t7287 = v_momentum.clone(); if py::le(&t7286, &t7287)? { { let t7288 = Py::Int(1i64); if py::lt(&t7287, &t7288)? { true } else { false } } } else { false } }) }; if !t7291.truthy() { t7291 } else { { let t7289 = v_weight_decay.clone(); Py::Bool({ let t7290 = Py::Int(0i64); if py::ge(&t7289, &t7290)? { true } else { false } }) } } } } }, Py::str("need lr > 0, 0 <= momentum < 1, weight_decay >= 0"))?;
    let t7295 = ({ let mut out = vec![]; for t7293 in py::iter(&(Py::tuple(vec![Py::str("w"), Py::str("grad"), Py::str("buf")])))? { let mut v_k: Py; 
v_k = t7293;
out.push({ let mut out = vec![]; for t7294 in py::iter(&(py::getitem(&v_V.clone(), &v_k.clone())?))? { let mut v_x: Py; 
v_x = t7294;
out.push(f__F(v_x.clone())?); } Py::list(out) }); } Py::list(out) }).items()?;
    if t7295.len() != 3 { return Err(PyErr::Value("unpack".into())); }
    v_W = t7295[0].clone();
    v_G = t7295[1].clone();
    v_B = t7295[2].clone();
    let t7296 = (Py::tuple(vec![f__F(v_lr.clone())?, f__F(v_momentum.clone())?, f__F(v_weight_decay.clone())?])).items()?;
    if t7296.len() != 3 { return Err(PyErr::Value("unpack".into())); }
    v_L = t7296[0].clone();
    v_mu = t7296[1].clone();
    v_wd = t7296[2].clone();
    v_d = { let mut out = vec![]; for t7297 in py::iter(&(py::zip(&[v_G.clone(), v_W.clone()])?))? { let mut v_g: Py; let mut v_x: Py; 
let t7298 = (t7297).items()?;
if t7298.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_g = t7298[0].clone();
v_x = t7298[1].clone();
out.push(py::add(&v_g.clone(), &py::mul(&v_wd.clone(), &v_x.clone())?)?); } Py::list(out) };
    v_B2 = { let mut out = vec![]; for t7299 in py::iter(&(py::zip(&[v_B.clone(), v_d.clone()])?))? { let mut v_b: Py; let mut v_di: Py; 
let t7300 = (t7299).items()?;
if t7300.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_b = t7300[0].clone();
v_di = t7300[1].clone();
out.push(py::add(&py::mul(&v_mu.clone(), &v_b.clone())?, &v_di.clone())?); } Py::list(out) };
    v_step = (if (v_nesterov.clone()).truthy() { { let mut out = vec![]; for t7301 in py::iter(&(py::zip(&[v_d.clone(), v_B2.clone()])?))? { let mut v_b: Py; let mut v_di: Py; 
let t7302 = (t7301).items()?;
if t7302.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_di = t7302[0].clone();
v_b = t7302[1].clone();
out.push(py::add(&v_di.clone(), &py::mul(&v_mu.clone(), &v_b.clone())?)?); } Py::list(out) } } else { v_B2.clone() });
    return Ok(f__out(Py::dict(vec![(Py::str("buf_new"), v_B2.clone()), (Py::str("w_new"), { let mut out = vec![]; for t7315 in py::iter(&(py::zip(&[v_W.clone(), v_step.clone()])?))? { let mut v_s: Py; let mut v_x: Py; 
let t7316 = (t7315).items()?;
if t7316.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_x = t7316[0].clone();
v_s = t7316[1].clone();
out.push(py::sub(&v_x.clone(), &py::mul(&v_L.clone(), &v_s.clone())?)?); } Py::list(out) })]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_mrr(mut v_R: Py) -> R {
    let mut v_M: Py = Py::None;
    let mut v_r: Py = Py::None;
    let mut v_row: Py = Py::None;
    let mut v_rr: Py = Py::None;
    v_M = f__mat(Py::str("R"), v_R.clone(), Py::Int(1i64), Py::Int(1i64))?;
    v_rr = Py::list(vec![]);
    for t5104 in py::iter(&(v_M.clone()))? {
        v_row = t5104;
        v_r = f__rel01(Py::str("each row of R"), v_row.clone())?;
        let _ = py::method(&v_rr.clone(), "append", &[(if ({ let t5114 = Py::Int(1i64); Py::Bool({ let t5115 = v_r.clone(); if py::contains(&t5115, &t5114)? { true } else { false } }) }).truthy() { py::fraction(&[Py::Int(1i64), py::add(&py::method(&v_r.clone(), "index", &[Py::Int(1i64)])?, &Py::Int(1i64))?])? } else { py::fraction(&[Py::Int(0i64)])? })])?;
    }
    return Ok(f__out(Py::dict(vec![(Py::str("reciprocal_ranks"), v_rr.clone()), (Py::str("mrr"), py::truediv(&py::sum(&v_rr.clone(), None)?, &py::len(&v_rr.clone())?)?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_mse_gradient(mut v_X: Py, mut v_y: Py, mut v_w: Py, mut v_b: Py) -> R {
    let mut v_M: Py = Py::None;
    let mut v_gb: Py = Py::None;
    let mut v_gw: Py = Py::None;
    let mut v_loss: Py = Py::None;
    let mut v_wv: Py = Py::None;
    let mut v_yv: Py = Py::None;
    let t5311 = (f__linear_parts(v_X.clone(), v_y.clone(), v_w.clone(), v_b.clone())?).items()?;
    if t5311.len() != 3 { return Err(PyErr::Value("unpack".into())); }
    v_M = t5311[0].clone();
    v_yv = t5311[1].clone();
    v_wv = t5311[2].clone();
    let t5324 = (f__lin_loss(v_M.clone(), v_yv.clone(), v_wv.clone(), v_b.clone())?).items()?;
    if t5324.len() != 3 { return Err(PyErr::Value("unpack".into())); }
    v_loss = t5324[0].clone();
    v_gw = t5324[1].clone();
    v_gb = t5324[2].clone();
    return Ok(f__out(Py::dict(vec![(Py::str("loss"), v_loss.clone()), (Py::str("grad_w"), v_gw.clone()), (Py::str("grad_b"), v_gb.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_msle(mut v_y_true: Py, mut v_y_pred: Py) -> R {
    let mut v_m: Py = Py::None;
    let mut v_yp: Py = Py::None;
    let mut v_yt: Py = Py::None;
    let t5168 = (f__pair(v_y_true.clone(), v_y_pred.clone(), Py::Int(1i64))?).items()?;
    if t5168.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_yt = t5168[0].clone();
    v_yp = t5168[1].clone();
    let _ = f__need(Py::Bool(({ let mut out = vec![]; for t5169 in py::iter(&(py::add(&v_yt.clone(), &v_yp.clone())?))? { let mut v_v: Py; 
v_v = t5169;
out.push({ let t5170 = v_v.clone(); Py::Bool({ let t5171 = Py::Int(0i64); if py::ge(&t5170, &t5171)? { true } else { false } }) }); } Py::list(out) }).items()?.iter().all(|x| x.truthy())), Py::str("the squared log error needs non-negative values"))?;
    v_m = py::truediv(&py::fsum(&{ let mut out = vec![]; for t5172 in py::iter(&(py::zip(&[v_yt.clone(), v_yp.clone()])?))? { let mut v_a: Py; let mut v_b: Py; 
let t5173 = (t5172).items()?;
if t5173.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t5173[0].clone();
v_b = t5173[1].clone();
out.push(py::pow(&py::sub(&py::math1("log1p", &v_a.clone())?, &py::math1("log1p", &v_b.clone())?)?, &Py::Int(2i64))?); } Py::list(out) })?, &py::len(&v_yt.clone())?)?;
    return Ok(f__out(Py::dict(vec![(Py::str("msle"), v_m.clone()), (Py::str("rmsle"), py::math1("sqrt", &v_m.clone())?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_multiclass_confusion(mut v_y_true: Py, mut v_y_pred: Py) -> R {
    let mut v_M: Py = Py::None;
    let mut v_a: Py = Py::None;
    let mut v_b: Py = Py::None;
    let mut v_cls: Py = Py::None;
    let mut v_correct: Py = Py::None;
    let mut v_idx: Py = Py::None;
    let mut v_yp: Py = Py::None;
    let mut v_yt: Py = Py::None;
    let t5375 = (Py::tuple(vec![f__labels(Py::str("y_true"), v_y_true.clone(), Py::Int(1i64))?, f__labels(Py::str("y_pred"), v_y_pred.clone(), Py::Int(1i64))?])).items()?;
    if t5375.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_yt = t5375[0].clone();
    v_yp = t5375[1].clone();
    let _ = f__same_len(v_yt.clone(), v_yp.clone(), Py::str("y_true"), Py::str("y_pred"))?;
    v_cls = f__classes(Py::tuple(vec![v_yt.clone(), v_yp.clone()]))?;
    v_idx = { let mut out = vec![]; for t5379 in py::iter(&(py::enumerate(&v_cls.clone(), 0)?))? { let mut v_c: Py; let mut v_i: Py; 
let t5380 = (t5379).items()?;
if t5380.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_i = t5380[0].clone();
v_c = t5380[1].clone();
out.push((v_c.clone(), v_i.clone())); } Py::dict(out) };
    v_M = { let mut out = vec![]; for t5381 in py::iter(&(v_cls.clone()))? { let mut v__: Py; 
v__ = t5381;
out.push(py::mul(&Py::list(vec![Py::Int(0i64)]), &py::len(&v_cls.clone())?)?); } Py::list(out) };
    for t5382 in py::iter(&(py::zip(&[v_yt.clone(), v_yp.clone()])?))? {
        let t5385 = (t5382).items()?;
        if t5385.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_a = t5385[0].clone();
        v_b = t5385[1].clone();
        let t5383 = py::getitem(&v_M.clone(), &py::getitem(&v_idx.clone(), &v_a.clone())?)?; let t5384 = py::getitem(&v_idx.clone(), &v_b.clone())?;
        py::setitem(&t5383, &t5384, py::iadd(&py::getitem(&t5383, &t5384)?, &Py::Int(1i64))?)?;
    }
    v_correct = py::sum(&{ let mut out = vec![]; for t5386 in py::range_iter(&[(py::len(&v_cls.clone())?).i()?]) { let mut v_i: Py; 
v_i = t5386;
out.push(py::getitem(&py::getitem(&v_M.clone(), &v_i.clone())?, &v_i.clone())?); } Py::list(out) }, None)?;
    return Ok(f__out(Py::dict(vec![(Py::str("classes"), v_cls.clone()), (Py::str("matrix"), v_M.clone()), (Py::str("n_correct"), v_correct.clone()), (Py::str("accuracy"), py::fraction(&[v_correct.clone(), py::len(&v_yt.clone())?])?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_multiclass_hinge(mut v_y_true: Py, mut v_D: Py) -> R {
    let mut v_K: Py = Py::None;
    let mut v_M: Py = Py::None;
    let mut v_cs: Py = Py::None;
    let mut v_fy: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_others: Py = Py::None;
    let mut v_row: Py = Py::None;
    let mut v_ww: Py = Py::None;
    let mut v_y: Py = Py::None;
    let mut v_yt: Py = Py::None;
    v_yt = f__labels(Py::str("y_true"), v_y_true.clone(), Py::Int(1i64))?;
    v_M = f__mat(Py::str("D"), v_D.clone(), Py::Int(1i64), Py::Int(3i64))?;
    let _ = f__same_len(v_yt.clone(), v_M.clone(), Py::str("y_true"), Py::str("the rows of D"))?;
    v_K = py::len(&py::getitem(&v_M.clone(), &Py::Int(0i64))?)?;
    let t5453 = (Py::tuple(vec![Py::list(vec![]), Py::list(vec![])])).items()?;
    if t5453.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_cs = t5453[0].clone();
    v_ww = t5453[1].clone();
    for t5454 in py::iter(&(py::zip(&[v_yt.clone(), v_M.clone()])?))? {
        let t5462 = (t5454).items()?;
        if t5462.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_y = t5462[0].clone();
        v_row = t5462[1].clone();
        let _ = f__need({ let t5455 = Py::Int(0i64); Py::Bool({ let t5456 = v_y.clone(); if py::le(&t5455, &t5456)? { { let t5457 = v_K.clone(); if py::lt(&t5456, &t5457)? { true } else { false } } } else { false } }) }, Py::str("every label must be a column index of D (0 to K-1)"))?;
        v_fy = f__F(py::getitem(&v_row.clone(), &v_y.clone())?)?;
        v_others = { let mut out = vec![]; for t5458 in py::range_iter(&[(v_K.clone()).i()?]) { let mut v_j: Py; 
v_j = t5458;
if ({ let t5459 = v_j.clone(); Py::Bool({ let t5460 = v_y.clone(); if py::ne(&t5459, &t5460) { true } else { false } }) }).truthy() { out.push(f__F(py::getitem(&v_row.clone(), &v_j.clone())?)?); }} Py::list(out) };
        let _ = py::method(&v_cs.clone(), "append", &[py::minmax(&[py::fraction(&[Py::Int(0i64)])?, py::add(&py::sub(&Py::Int(1i64), &v_fy.clone())?, &py::minmax(&[v_others.clone()], true)?)?], true)?])?;
        let _ = py::method(&v_ww.clone(), "append", &[py::truediv(&py::sum(&{ let mut out = vec![]; for t5461 in py::iter(&(v_others.clone()))? { let mut v_o: Py; 
v_o = t5461;
out.push(py::minmax(&[py::fraction(&[Py::Int(0i64)])?, py::add(&py::sub(&Py::Int(1i64), &v_fy.clone())?, &v_o.clone())?], true)?); } Py::list(out) }, None)?, &v_K.clone())?])?;
    }
    v_n = py::len(&v_yt.clone())?;
    return Ok(f__out(Py::dict(vec![(Py::str("crammer_singer"), py::truediv(&py::sum(&v_cs.clone(), None)?, &v_n.clone())?), (Py::str("weston_watkins"), py::truediv(&py::sum(&v_ww.clone(), None)?, &v_n.clone())?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_multihead(mut v_Q: Py, mut v_K: Py, mut v_V: Py, mut v_heads: Py) -> R {
    let mut v_D: Py = Py::None;
    let mut v_Km: Py = Py::None;
    let mut v_Qm: Py = Py::None;
    let mut v_Vm: Py = Py::None;
    let mut v_dh: Py = Py::None;
    let mut v_h: Py = Py::None;
    let mut v_i: Py = Py::None;
    let mut v_o: Py = Py::None;
    let mut v_out: Py = Py::None;
    let mut v_sl: Py = Py::None;
    let t5521 = (Py::tuple(vec![f__mat(Py::str("Q"), v_Q.clone(), Py::Int(1i64), Py::Int(1i64))?, f__mat(Py::str("K"), v_K.clone(), Py::Int(1i64), Py::Int(1i64))?, f__mat(Py::str("V"), v_V.clone(), Py::Int(1i64), Py::Int(1i64))?])).items()?;
    if t5521.len() != 3 { return Err(PyErr::Value("unpack".into())); }
    v_Qm = t5521[0].clone();
    v_Km = t5521[1].clone();
    v_Vm = t5521[2].clone();
    v_D = py::len(&py::getitem(&v_Qm.clone(), &Py::Int(0i64))?)?;
    let _ = f__need({ let t5526 = { let t5522 = py::len(&py::getitem(&v_Km.clone(), &Py::Int(0i64))?)?; Py::Bool({ let t5523 = v_D.clone(); if py::eq(&t5522, &t5523) { true } else { false } }) }; if !t5526.truthy() { t5526 } else { { let t5524 = py::len(&py::getitem(&v_Vm.clone(), &Py::Int(0i64))?)?; Py::Bool({ let t5525 = v_D.clone(); if py::eq(&t5524, &t5525) { true } else { false } }) } } }, Py::str("Q, K and V need the same width (the model dimension)"))?;
    let _ = f__need({ let t5527 = py::len(&v_Km.clone())?; Py::Bool({ let t5528 = py::len(&v_Vm.clone())?; if py::eq(&t5527, &t5528) { true } else { false } }) }, Py::str("K and V need the same number of rows"))?;
    let _ = f__need({ let t5534 = Py::Bool(matches!(v_heads.clone(), Py::Int(_) | Py::Big(_) | Py::Bool(_))); if !t5534.truthy() { t5534 } else { { let t5533 = { let t5529 = v_heads.clone(); Py::Bool({ let t5530 = Py::Int(1i64); if py::ge(&t5529, &t5530)? { true } else { false } }) }; if !t5533.truthy() { t5533 } else { { let t5531 = py::modulo(&v_D.clone(), &v_heads.clone())?; Py::Bool({ let t5532 = Py::Int(0i64); if py::eq(&t5531, &t5532) { true } else { false } }) } } } } }, Py::str("the model dimension must be divisible by the number of heads"))?;
    v_dh = py::floordiv(&v_D.clone(), &v_heads.clone())?;
    v_out = { let mut out = vec![]; for t5535 in py::iter(&(v_Qm.clone()))? { let mut v__: Py; 
v__ = t5535;
out.push(py::mul(&Py::list(vec![Py::Float(0.0)]), &v_D.clone())?); } Py::list(out) };
    for t5536 in py::range_iter(&[(v_heads.clone()).i()?]) {
        v_h = t5536;
        v_sl = Py::tuple(vec![py::mul(&v_h.clone(), &v_dh.clone())?, py::mul(&py::add(&v_h.clone(), &Py::Int(1i64))?, &v_dh.clone())?, Py::None]);
        v_o = py::getitem(&f_attention({ let mut out = vec![]; for t5568 in py::iter(&(v_Qm.clone()))? { let mut v_r: Py; 
v_r = t5568;
out.push(py::slice_by(&v_r.clone(), &v_sl.clone())?); } Py::list(out) }, { let mut out = vec![]; for t5569 in py::iter(&(v_Km.clone()))? { let mut v_r: Py; 
v_r = t5569;
out.push(py::slice_by(&v_r.clone(), &v_sl.clone())?); } Py::list(out) }, { let mut out = vec![]; for t5570 in py::iter(&(v_Vm.clone()))? { let mut v_r: Py; 
v_r = t5570;
out.push(py::slice_by(&v_r.clone(), &v_sl.clone())?); } Py::list(out) }, Py::Bool(false))?, &Py::str("output"))?;
        for t5571 in py::range_iter(&[(py::len(&v_Qm.clone())?).i()?]) {
            v_i = t5571;
            let t5572 = py::getitem(&v_o.clone(), &v_i.clone())?;
            py::setslice(&py::getitem(&v_out.clone(), &v_i.clone())?, Some((py::mul(&v_h.clone(), &v_dh.clone())?).i()?), Some((py::mul(&py::add(&v_h.clone(), &Py::Int(1i64))?, &v_dh.clone())?).i()?), &t5572)?;
        }
    }
    return Ok(f__out(Py::dict(vec![(Py::str("output"), v_out.clone()), (Py::str("output_sum"), py::fsum(&{ let mut out = vec![]; for t5573 in py::iter(&(v_out.clone()))? { let mut v_r: Py; 
v_r = t5573;
for t5574 in py::iter(&(v_r.clone()))? { let mut v_v: Py; 
v_v = t5574;
out.push(v_v.clone()); }} Py::list(out) })?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_multinomial_nb(mut v_X: Py, mut v_y: Py, mut v_x: Py, mut v_alpha: Py) -> R {
    let mut v_M: Py = Py::None;
    let mut v_c: Py = Py::None;
    let mut v_cls: Py = Py::None;
    let mut v_d: Py = Py::None;
    let mut v_den: Py = Py::None;
    let mut v_jll: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_q: Py = Py::None;
    let mut v_rows: Py = Py::None;
    let mut v_tot: Py = Py::None;
    let mut v_yl: Py = Py::None;
    let t5638 = (f__nb_data(v_X.clone(), v_y.clone(), v_x.clone())?).items()?;
    if t5638.len() != 4 { return Err(PyErr::Value("unpack".into())); }
    v_M = t5638[0].clone();
    v_yl = t5638[1].clone();
    v_q = t5638[2].clone();
    v_cls = t5638[3].clone();
    let _ = f__fin(Py::str("alpha"), Py::tuple(vec![v_alpha.clone()]))?;
    let _ = f__need({ let t5639 = v_alpha.clone(); Py::Bool({ let t5640 = Py::Int(0i64); if py::gt(&t5639, &t5640)? { true } else { false } }) }, Py::str("alpha (additive smoothing) must be positive"))?;
    let _ = f__need({ let t5648 = Py::Bool(({ let mut out = vec![]; for t5641 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t5641;
for t5642 in py::iter(&(v_r.clone()))? { let mut v_v: Py; 
v_v = t5642;
out.push({ let t5643 = v_v.clone(); Py::Bool({ let t5644 = Py::Int(0i64); if py::ge(&t5643, &t5644)? { true } else { false } }) }); }} Py::list(out) }).items()?.iter().all(|x| x.truthy())); if !t5648.truthy() { t5648 } else { Py::Bool(({ let mut out = vec![]; for t5645 in py::iter(&(v_q.clone()))? { let mut v_v: Py; 
v_v = t5645;
out.push({ let t5646 = v_v.clone(); Py::Bool({ let t5647 = Py::Int(0i64); if py::ge(&t5646, &t5647)? { true } else { false } }) }); } Py::list(out) }).items()?.iter().all(|x| x.truthy())) } }, Py::str("counts must be >= 0"))?;
    let t5649 = (Py::tuple(vec![py::len(&v_yl.clone())?, py::len(&v_q.clone())?])).items()?;
    if t5649.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_n = t5649[0].clone();
    v_d = t5649[1].clone();
    v_jll = Py::list(vec![]);
    for t5650 in py::iter(&(v_cls.clone()))? {
        v_c = t5650;
        v_rows = { let mut out = vec![]; for t5651 in py::range_iter(&[(v_n.clone()).i()?]) { let mut v_i: Py; 
v_i = t5651;
if ({ let t5652 = py::getitem(&v_yl.clone(), &v_i.clone())?; Py::Bool({ let t5653 = v_c.clone(); if py::eq(&t5652, &t5653) { true } else { false } }) }).truthy() { out.push(py::getitem(&v_M.clone(), &v_i.clone())?); }} Py::list(out) };
        v_tot = { let mut out = vec![]; for t5654 in py::range_iter(&[(v_d.clone()).i()?]) { let mut v_j: Py; 
v_j = t5654;
out.push(py::fsum(&{ let mut out = vec![]; for t5655 in py::iter(&(v_rows.clone()))? { let mut v_r: Py; 
v_r = t5655;
out.push(py::getitem(&v_r.clone(), &v_j.clone())?); } Py::list(out) })?); } Py::list(out) };
        v_den = py::add(&py::fsum(&v_tot.clone())?, &py::mul(&v_alpha.clone(), &v_d.clone())?)?;
        let _ = py::method(&v_jll.clone(), "append", &[py::add(&py::math1("log", &py::truediv(&py::len(&v_rows.clone())?, &v_n.clone())?)?, &py::fsum(&{ let mut out = vec![]; for t5656 in py::range_iter(&[(v_d.clone()).i()?]) { let mut v_j: Py; 
v_j = t5656;
out.push(py::mul(&py::getitem(&v_q.clone(), &v_j.clone())?, &py::math1("log", &py::truediv(&py::add(&py::getitem(&v_tot.clone(), &v_j.clone())?, &v_alpha.clone())?, &v_den.clone())?)?)?); } Py::list(out) })?)?])?;
    }
    return Ok(f__posterior(v_cls.clone(), v_jll.clone())?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_mutual_info_labels(mut v_x: Py, mut v_y: Py, mut v_unit: Py) -> R {
    let mut v_L: Py = Py::None;
    let mut v_M: Py = Py::None;
    let mut v_a: Py = Py::None;
    let mut v_b: Py = Py::None;
    let mut v_hx: Py = Py::None;
    let mut v_hy: Py = Py::None;
    let mut v_mi: Py = Py::None;
    let mut v_n: Py = Py::None;
    let t5714 = (Py::tuple(vec![f__labels(Py::str("x"), v_x.clone(), Py::Int(1i64))?, f__labels(Py::str("y"), v_y.clone(), Py::Int(1i64))?])).items()?;
    if t5714.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_a = t5714[0].clone();
    v_b = t5714[1].clone();
    let _ = f__same_len(v_a.clone(), v_b.clone(), Py::str("x"), Py::str("y"))?;
    v_n = py::len(&v_a.clone())?;
    v_M = f__contingency(v_a.clone(), v_b.clone())?;
    v_L = f__LN(v_unit.clone())?;
    v_hx = f__Hc({ let mut out = vec![]; for t5734 in py::iter(&(py::set(&v_a.clone())?))? { let mut v_v: Py; 
v_v = t5734;
out.push(py::method(&v_a.clone(), "count", &[v_v.clone()])?); } Py::list(out) })?;
    v_hy = f__Hc({ let mut out = vec![]; for t5735 in py::iter(&(py::set(&v_b.clone())?))? { let mut v_v: Py; 
v_v = t5735;
out.push(py::method(&v_b.clone(), "count", &[v_v.clone()])?); } Py::list(out) })?;
    v_mi = f__mi_counts(v_M.clone(), v_n.clone())?;
    return Ok(f__out(Py::dict(vec![(Py::str("mutual_information"), py::truediv(&v_mi.clone(), &v_L.clone())?), (Py::str("H_x"), py::truediv(&v_hx.clone(), &v_L.clone())?), (Py::str("H_y"), py::truediv(&v_hy.clone(), &v_L.clone())?), (Py::str("H_x_given_y"), py::truediv(&py::sub(&v_hx.clone(), &v_mi.clone())?, &v_L.clone())?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_ndcg(mut v_rel: Py, mut v_k: Py, mut v_gain: Py) -> R {
    let mut v_dcg: Py = Py::None;
    let mut v_disc: Py = Py::None;
    let mut v_g: Py = Py::None;
    let mut v_idcg: Py = Py::None;
    let mut v_ideal: Py = Py::None;
    let mut v_r: Py = Py::None;
    v_r = f__vec(Py::str("rel"), v_rel.clone(), Py::Int(1i64))?;
    let _ = f__need(Py::Bool(({ let mut out = vec![]; for t5792 in py::iter(&(v_r.clone()))? { let mut v_v: Py; 
v_v = t5792;
out.push({ let t5793 = v_v.clone(); Py::Bool({ let t5794 = Py::Int(0i64); if py::ge(&t5793, &t5794)? { true } else { false } }) }); } Py::list(out) }).items()?.iter().all(|x| x.truthy())), Py::str("relevance grades must be >= 0"))?;
    let _ = f__need({ let t5798 = Py::Bool(matches!(v_k.clone(), Py::Int(_) | Py::Big(_) | Py::Bool(_))); if !t5798.truthy() { t5798 } else { { let t5795 = Py::Int(1i64); Py::Bool({ let t5796 = v_k.clone(); if py::le(&t5795, &t5796)? { { let t5797 = py::len(&v_r.clone())?; if py::le(&t5796, &t5797)? { true } else { false } } } else { false } }) } } }, Py::str("k must be between 1 and the length of the list"))?;
    v_g = { let mut out = vec![]; for t5799 in py::iter(&(v_r.clone()))? { let mut v_v: Py; 
v_v = t5799;
out.push((if ({ let t5800 = v_gain.clone(); Py::Bool({ let t5801 = Py::str("exponential"); if py::eq(&t5800, &t5801) { true } else { false } }) }).truthy() { py::sub(&py::pow(&Py::Float(2.0), &v_v.clone())?, &Py::Int(1i64))? } else { v_v.clone() })); } Py::list(out) };
    v_disc = { let mut out = vec![]; for t5802 in py::range_iter(&[(py::len(&v_r.clone())?).i()?]) { let mut v_i: Py; 
v_i = t5802;
out.push(py::truediv(&Py::Int(1i64), &py::math1("log2", &py::add(&v_i.clone(), &Py::Int(2i64))?)?)?); } Py::list(out) };
    v_dcg = py::fsum(&{ let mut out = vec![]; for t5803 in py::iter(&(py::zip(&[py::slice(&v_g.clone(), None, Some((v_k.clone()).i()?), None)?, v_disc.clone()])?))? { let mut v_di: Py; let mut v_gi: Py; 
let t5804 = (t5803).items()?;
if t5804.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_gi = t5804[0].clone();
v_di = t5804[1].clone();
out.push(py::mul(&v_gi.clone(), &v_di.clone())?); } Py::list(out) })?;
    v_ideal = py::sorted(&v_g.clone(), (Py::Bool(true)).truthy())?;
    v_idcg = py::fsum(&{ let mut out = vec![]; for t5805 in py::iter(&(py::zip(&[py::slice(&v_ideal.clone(), None, Some((v_k.clone()).i()?), None)?, v_disc.clone()])?))? { let mut v_di: Py; let mut v_gi: Py; 
let t5806 = (t5805).items()?;
if t5806.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_gi = t5806[0].clone();
v_di = t5806[1].clone();
out.push(py::mul(&v_gi.clone(), &v_di.clone())?); } Py::list(out) })?;
    let _ = f__need({ let t5807 = v_idcg.clone(); Py::Bool({ let t5808 = Py::Int(0i64); if py::gt(&t5807, &t5808)? { true } else { false } }) }, Py::str("NDCG is undefined: no relevant item"))?;
    return Ok(f__out(Py::dict(vec![(Py::str("dcg"), v_dcg.clone()), (Py::str("idcg"), v_idcg.clone()), (Py::str("ndcg"), py::truediv(&v_dcg.clone(), &v_idcg.clone())?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_nmi(mut v_labels_true: Py, mut v_labels_pred: Py, mut v_method: Py) -> R {
    let mut v_a: Py = Py::None;
    let mut v_b: Py = Py::None;
    let mut v_ha: Py = Py::None;
    let mut v_hb: Py = Py::None;
    let mut v_mi: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_norm: Py = Py::None;
    let t5859 = (Py::tuple(vec![f__labels(Py::str("labels_true"), v_labels_true.clone(), Py::Int(1i64))?, f__labels(Py::str("labels_pred"), v_labels_pred.clone(), Py::Int(1i64))?])).items()?;
    if t5859.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_a = t5859[0].clone();
    v_b = t5859[1].clone();
    let _ = f__same_len(v_a.clone(), v_b.clone(), Py::str("labels_true"), Py::str("labels_pred"))?;
    let _ = f__single_check(v_a.clone(), v_b.clone())?;
    v_n = py::len(&v_a.clone())?;
    v_mi = f__mi_counts(f__contingency(v_a.clone(), v_b.clone())?, v_n.clone())?;
    v_ha = f__Hc({ let mut out = vec![]; for t5891 in py::iter(&(py::set(&v_a.clone())?))? { let mut v_v: Py; 
v_v = t5891;
out.push(py::method(&v_a.clone(), "count", &[v_v.clone()])?); } Py::list(out) })?;
    v_hb = f__Hc({ let mut out = vec![]; for t5892 in py::iter(&(py::set(&v_b.clone())?))? { let mut v_v: Py; 
v_v = t5892;
out.push(py::method(&v_b.clone(), "count", &[v_v.clone()])?); } Py::list(out) })?;
    v_norm = py::getitem(&Py::dict(vec![(Py::str("arithmetic"), py::truediv(&py::add(&v_ha.clone(), &v_hb.clone())?, &Py::Int(2i64))?), (Py::str("geometric"), py::math1("sqrt", &py::mul(&v_ha.clone(), &v_hb.clone())?)?), (Py::str("min"), py::minmax(&[v_ha.clone(), v_hb.clone()], false)?), (Py::str("max"), py::minmax(&[v_ha.clone(), v_hb.clone()], true)?)]), &v_method.clone())?;
    return Ok(f__out(Py::dict(vec![(Py::str("nmi"), py::truediv(&v_mi.clone(), &v_norm.clone())?), (Py::str("mutual_information"), v_mi.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_one_hot(mut v_categories: Py) -> R {
    let mut v_cats: Py = Py::None;
    let mut v_idx: Py = Py::None;
    let mut v_items: Py = Py::None;
    v_items = f__cats(v_categories.clone())?;
    v_cats = py::sorted(&py::set(&v_items.clone())?, (Py::Bool(false)).truthy())?;
    v_idx = { let mut out = vec![]; for t5938 in py::iter(&(py::enumerate(&v_cats.clone(), 0)?))? { let mut v_c: Py; let mut v_i: Py; 
let t5939 = (t5938).items()?;
if t5939.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_i = t5939[0].clone();
v_c = t5939[1].clone();
out.push((v_c.clone(), v_i.clone())); } Py::dict(out) };
    return Ok(f__out(Py::dict(vec![(Py::str("categories"), v_cats.clone()), (Py::str("counts"), { let mut out = vec![]; for t5952 in py::iter(&(v_cats.clone()))? { let mut v_c: Py; 
v_c = t5952;
out.push(py::method(&v_items.clone(), "count", &[v_c.clone()])?); } Py::list(out) }), (Py::str("codes"), { let mut out = vec![]; for t5953 in py::iter(&(v_items.clone()))? { let mut v_c: Py; 
v_c = t5953;
out.push(py::getitem(&v_idx.clone(), &v_c.clone())?); } Py::list(out) }), (Py::str("one_hot"), { let mut out = vec![]; for t5954 in py::iter(&(v_items.clone()))? { let mut v_c: Py; 
v_c = t5954;
out.push({ let mut out = vec![]; for t5955 in py::range_iter(&[(py::len(&v_cats.clone())?).i()?]) { let mut v_j: Py; 
v_j = t5955;
out.push((if ({ let t5956 = py::getitem(&v_idx.clone(), &v_c.clone())?; Py::Bool({ let t5957 = v_j.clone(); if py::eq(&t5956, &t5957) { true } else { false } }) }).truthy() { Py::Int(1i64) } else { Py::Int(0i64) })); } Py::list(out) }); } Py::list(out) })]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_pca(mut v_X: Py) -> R {
    let mut v_Cv: Py = Py::None;
    let mut v_F: Py = Py::None;
    let mut v_M: Py = Py::None;
    let mut v_V: Py = Py::None;
    let mut v_comp: Py = Py::None;
    let mut v_d: Py = Py::None;
    let mut v_lam: Py = Py::None;
    let mut v_mean: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_order: Py = Py::None;
    let mut v_scores: Py = Py::None;
    let mut v_tot: Py = Py::None;
    let mut v_vals: Py = Py::None;
    v_M = f__mat(Py::str("X"), v_X.clone(), Py::Int(3i64), Py::Int(1i64))?;
    let t6004 = (Py::tuple(vec![py::len(&v_M.clone())?, py::len(&py::getitem(&v_M.clone(), &Py::Int(0i64))?)?])).items()?;
    if t6004.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_n = t6004[0].clone();
    v_d = t6004[1].clone();
    let _ = f__need({ let t6005 = v_n.clone(); Py::Bool({ let t6006 = v_d.clone(); if py::gt(&t6005, &t6006)? { true } else { false } }) }, Py::str("PCA here needs more samples than features"))?;
    let _ = f__need({ let t6007 = v_d.clone(); Py::Bool({ let t6008 = Py::Int(8i64); if py::le(&t6007, &t6008)? { true } else { false } }) }, Py::str("at most 8 features"))?;
    v_F = { let mut out = vec![]; for t6009 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t6009;
out.push({ let mut out = vec![]; for t6010 in py::iter(&(v_r.clone()))? { let mut v_v: Py; 
v_v = t6010;
out.push(f__F(v_v.clone())?); } Py::list(out) }); } Py::list(out) };
    v_mean = { let mut out = vec![]; for t6011 in py::range_iter(&[(v_d.clone()).i()?]) { let mut v_j: Py; 
v_j = t6011;
out.push(py::truediv(&py::sum(&{ let mut out = vec![]; for t6012 in py::iter(&(v_F.clone()))? { let mut v_r: Py; 
v_r = t6012;
out.push(py::getitem(&v_r.clone(), &v_j.clone())?); } Py::list(out) }, None)?, &v_n.clone())?); } Py::list(out) };
    v_Cv = { let mut out = vec![]; for t6013 in py::range_iter(&[(v_d.clone()).i()?]) { let mut v_i: Py; 
v_i = t6013;
out.push({ let mut out = vec![]; for t6014 in py::range_iter(&[(v_d.clone()).i()?]) { let mut v_j: Py; 
v_j = t6014;
out.push(py::float(&py::truediv(&py::sum(&{ let mut out = vec![]; for t6015 in py::iter(&(v_F.clone()))? { let mut v_r: Py; 
v_r = t6015;
out.push(py::mul(&py::sub(&py::getitem(&v_r.clone(), &v_i.clone())?, &py::getitem(&v_mean.clone(), &v_i.clone())?)?, &py::sub(&py::getitem(&v_r.clone(), &v_j.clone())?, &py::getitem(&v_mean.clone(), &v_j.clone())?)?)?); } Py::list(out) }, None)?, &py::sub(&v_n.clone(), &Py::Int(1i64))?)?)?); } Py::list(out) }); } Py::list(out) };
    let t6043 = (f__jacobi(v_Cv.clone())?).items()?;
    if t6043.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_vals = t6043[0].clone();
    v_V = t6043[1].clone();
    v_order = py::sorted_key(&Py::list((py::range_iter(&[(v_d.clone()).i()?])).into_iter().collect::<Vec<Py>>()), &{ let v_Cv = v_Cv.clone(); let v_F = v_F.clone(); let v_M = v_M.clone(); let v_V = v_V.clone(); let v_X = v_X.clone(); let v_d = v_d.clone(); let v_mean = v_mean.clone(); let v_n = v_n.clone(); let v_vals = v_vals.clone(); py::func(move |a: &[Py]| -> R {
            #[allow(unused_mut, unused_variables, unused_assignments)]
{
            let mut v_i: Py = a.get(0).cloned().unwrap_or(Py::None);
            return Ok(py::neg(&py::getitem(&v_vals.clone(), &v_i.clone())?)?);
            #[allow(unreachable_code)]
            Ok(Py::None)
        }
        }) }, (Py::Bool(false)).truthy())?;
    v_lam = { let mut out = vec![]; for t6044 in py::iter(&(v_order.clone()))? { let mut v_i: Py; 
v_i = t6044;
out.push(py::getitem(&v_vals.clone(), &v_i.clone())?); } Py::list(out) };
    let _ = f__need({ let t6045 = py::getitem(&v_lam.clone(), &py::neg(&Py::Int(1i64))?)?; Py::Bool({ let t6046 = py::mul(&Py::Float(1e-10), &py::getitem(&v_lam.clone(), &Py::Int(0i64))?)?; if py::gt(&t6045, &t6046)? { true } else { false } }) }, Py::str("the data are rank-deficient: a component's variance is zero to rounding"))?;
    if ({ let t6047 = v_d.clone(); Py::Bool({ let t6048 = Py::Int(1i64); if py::gt(&t6047, &t6048)? { true } else { false } }) }).truthy() {
        let _ = f__need({ let t6049 = py::sub(&py::getitem(&v_lam.clone(), &Py::Int(0i64))?, &py::getitem(&v_lam.clone(), &Py::Int(1i64))?)?; Py::Bool({ let t6050 = py::mul(&Py::Float(1e-06), &py::getitem(&v_lam.clone(), &Py::Int(0i64))?)?; if py::gt(&t6049, &t6050)? { true } else { false } }) }, Py::str("the two largest variances are equal: the first component is not unique"))?;
    }
    v_comp = f__sign_fix({ let mut out = vec![]; for t6054 in py::range_iter(&[(v_d.clone()).i()?]) { let mut v_k: Py; 
v_k = t6054;
out.push(py::getitem(&py::getitem(&v_V.clone(), &v_k.clone())?, &py::getitem(&v_order.clone(), &Py::Int(0i64))?)?); } Py::list(out) })?;
    v_tot = py::fsum(&v_lam.clone())?;
    v_scores = { let mut out = vec![]; for t6055 in py::iter(&(v_F.clone()))? { let mut v_r: Py; 
v_r = t6055;
out.push(py::fsum(&{ let mut out = vec![]; for t6056 in py::range_iter(&[(v_d.clone()).i()?]) { let mut v_j: Py; 
v_j = t6056;
out.push(py::mul(&py::float(&py::sub(&py::getitem(&v_r.clone(), &v_j.clone())?, &py::getitem(&v_mean.clone(), &v_j.clone())?)?)?, &py::getitem(&v_comp.clone(), &v_j.clone())?)?); } Py::list(out) })?); } Py::list(out) };
    return Ok(f__out(Py::dict(vec![(Py::str("explained_variance"), v_lam.clone()), (Py::str("explained_variance_ratio"), { let mut out = vec![]; for t6069 in py::iter(&(v_lam.clone()))? { let mut v_v: Py; 
v_v = t6069;
out.push(py::truediv(&v_v.clone(), &v_tot.clone())?); } Py::list(out) }), (Py::str("component_1"), v_comp.clone()), (Py::str("scores_1"), v_scores.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_per_class_report(mut v_y_true: Py, mut v_y_pred: Py) -> R {
    let mut v_F: Py = Py::None;
    let mut v_P: Py = Py::None;
    let mut v_R: Py = Py::None;
    let mut v_S: Py = Py::None;
    let mut v_cls: Py = Py::None;
    let mut v_yp: Py = Py::None;
    let mut v_yt: Py = Py::None;
    let t1307 = (Py::tuple(vec![f__labels(Py::str("y_true"), v_y_true.clone(), Py::Int(1i64))?, f__labels(Py::str("y_pred"), v_y_pred.clone(), Py::Int(1i64))?])).items()?;
    if t1307.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_yt = t1307[0].clone();
    v_yp = t1307[1].clone();
    let _ = f__same_len(v_yt.clone(), v_yp.clone(), Py::str("y_true"), Py::str("y_pred"))?;
    let t1326 = (f__per_class(v_yt.clone(), v_yp.clone())?).items()?;
    if t1326.len() != 5 { return Err(PyErr::Value("unpack".into())); }
    v_cls = t1326[0].clone();
    v_P = t1326[1].clone();
    v_R = t1326[2].clone();
    v_F = t1326[3].clone();
    v_S = t1326[4].clone();
    return Ok(f__out(Py::dict(vec![(Py::str("classes"), v_cls.clone()), (Py::str("precision"), v_P.clone()), (Py::str("recall"), v_R.clone()), (Py::str("f1"), v_F.clone()), (Py::str("support"), v_S.clone()), (Py::str("macro_f1"), py::truediv(&py::sum(&v_F.clone(), None)?, &py::len(&v_F.clone())?)?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_perceptron_epoch(mut v_X: Py, mut v_y: Py, mut v_w: Py, mut v_b: Py, mut v_lr: Py) -> R {
    let mut v_B: Py = Py::None;
    let mut v_L: Py = Py::None;
    let mut v_M: Py = Py::None;
    let mut v_f: Py = Py::None;
    let mut v_mistakes: Py = Py::None;
    let mut v_r: Py = Py::None;
    let mut v_t: Py = Py::None;
    let mut v_wv: Py = Py::None;
    let mut v_yl: Py = Py::None;
    v_M = f__mat(Py::str("X"), v_X.clone(), Py::Int(1i64), Py::Int(1i64))?;
    v_yl = f__labels(Py::str("y"), v_y.clone(), Py::Int(1i64))?;
    let _ = f__need({ let t6122 = py::len(&v_yl.clone())?; Py::Bool({ let t6123 = py::len(&v_M.clone())?; if py::eq(&t6122, &t6123) { true } else { false } }) }, Py::str("y needs one label per row of X"))?;
    let _ = f__need({ let t6129 = Py::Bool(({ let mut out = vec![]; for t6124 in py::iter(&(v_yl.clone()))? { let mut v_v: Py; 
v_v = t6124;
out.push({ let t6125 = v_v.clone(); Py::Bool({ let t6126 = Py::tuple(vec![py::neg(&Py::Int(1i64))?, Py::Int(1i64)]); if py::contains(&t6126, &t6125)? { true } else { false } }) }); } Py::list(out) }).items()?.iter().all(|x| x.truthy())); if !t6129.truthy() { t6129 } else { { let t6127 = py::len(&py::set(&v_yl.clone())?)?; Py::Bool({ let t6128 = Py::Int(2i64); if py::eq(&t6127, &t6128) { true } else { false } }) } } }, Py::str("labels must be -1 and +1, both present"))?;
    v_wv = { let mut out = vec![]; for t6135 in py::iter(&(f__vec(Py::str("w"), v_w.clone(), Py::Int(1i64))?))? { let mut v_v: Py; 
v_v = t6135;
out.push(f__F(v_v.clone())?); } Py::list(out) };
    let _ = f__need({ let t6136 = py::len(&v_wv.clone())?; Py::Bool({ let t6137 = py::len(&py::getitem(&v_M.clone(), &Py::Int(0i64))?)?; if py::eq(&t6136, &t6137) { true } else { false } }) }, Py::str("w needs one weight per column of X"))?;
    let _ = Py::tuple(vec![f__fin(Py::str("b"), Py::tuple(vec![v_b.clone()]))?, f__fin(Py::str("lr"), Py::tuple(vec![v_lr.clone()]))?]);
    let _ = f__need({ let t6138 = v_lr.clone(); Py::Bool({ let t6139 = Py::Int(0i64); if py::gt(&t6138, &t6139)? { true } else { false } }) }, Py::str("the learning rate must be positive"))?;
    let t6140 = (Py::tuple(vec![f__F(v_b.clone())?, f__F(v_lr.clone())?])).items()?;
    if t6140.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_B = t6140[0].clone();
    v_L = t6140[1].clone();
    v_mistakes = Py::Int(0i64);
    for t6141 in py::iter(&(py::zip(&[v_M.clone(), v_yl.clone()])?))? {
        let t6148 = (t6141).items()?;
        if t6148.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_r = t6148[0].clone();
        v_t = t6148[1].clone();
        v_f = py::add(&py::sum(&{ let mut out = vec![]; for t6142 in py::iter(&(py::zip(&[v_wv.clone(), v_r.clone()])?))? { let mut v_a: Py; let mut v_c: Py; 
let t6143 = (t6142).items()?;
if t6143.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t6143[0].clone();
v_c = t6143[1].clone();
out.push(py::mul(&v_a.clone(), &f__F(v_c.clone())?)?); } Py::list(out) }, None)?, &v_B.clone())?;
        if ({ let t6144 = py::mul(&v_t.clone(), &v_f.clone())?; Py::Bool({ let t6145 = Py::Int(0i64); if py::le(&t6144, &t6145)? { true } else { false } }) }).truthy() {
            v_wv = { let mut out = vec![]; for t6146 in py::iter(&(py::zip(&[v_wv.clone(), v_r.clone()])?))? { let mut v_a: Py; let mut v_c: Py; 
let t6147 = (t6146).items()?;
if t6147.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t6147[0].clone();
v_c = t6147[1].clone();
out.push(py::add(&v_a.clone(), &py::mul(&py::mul(&v_L.clone(), &v_t.clone())?, &f__F(v_c.clone())?)?)?); } Py::list(out) };
            v_B = py::iadd(&v_B, &py::mul(&v_L.clone(), &v_t.clone())?)?;
            v_mistakes = py::iadd(&v_mistakes, &Py::Int(1i64))?;
        }
    }
    return Ok(f__out(Py::dict(vec![(Py::str("w_new"), v_wv.clone()), (Py::str("b_new"), v_B.clone()), (Py::str("n_updates"), v_mistakes.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_pinball(mut v_y_true: Py, mut v_y_pred: Py, mut v_alpha: Py) -> R {
    let mut v_a: Py = Py::None;
    let mut v_tot: Py = Py::None;
    let mut v_yp: Py = Py::None;
    let mut v_yt: Py = Py::None;
    let t6484 = (f__pair(v_y_true.clone(), v_y_pred.clone(), Py::Int(1i64))?).items()?;
    if t6484.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_yt = t6484[0].clone();
    v_yp = t6484[1].clone();
    let _ = f__fin(Py::str("alpha"), Py::tuple(vec![v_alpha.clone()]))?;
    let _ = f__need({ let t6485 = Py::Int(0i64); Py::Bool({ let t6486 = v_alpha.clone(); if py::lt(&t6485, &t6486)? { { let t6487 = Py::Int(1i64); if py::lt(&t6486, &t6487)? { true } else { false } } } else { false } }) }, Py::str("alpha (the quantile level) must lie strictly between 0 and 1"))?;
    v_a = f__F(v_alpha.clone())?;
    v_tot = py::sum(&{ let mut out = vec![]; for t6488 in py::iter(&(py::zip(&[v_yt.clone(), v_yp.clone()])?))? { let mut v_u: Py; let mut v_v: Py; 
let t6489 = (t6488).items()?;
if t6489.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_u = t6489[0].clone();
v_v = t6489[1].clone();
out.push(py::minmax(&[py::mul(&v_a.clone(), &py::sub(&f__F(v_u.clone())?, &f__F(v_v.clone())?)?)?, py::mul(&py::sub(&v_a.clone(), &Py::Int(1i64))?, &py::sub(&f__F(v_u.clone())?, &f__F(v_v.clone())?)?)?], true)?); } Py::list(out) }, None)?;
    return Ok(f__out(Py::dict(vec![(Py::str("pinball"), py::truediv(&v_tot.clone(), &py::len(&v_yt.clone())?)?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_poly_kernel(mut v_x: Py, mut v_z: Py, mut v_gamma: Py, mut v_coef0: Py, mut v_degree: Py) -> R {
    let mut v_a: Py = Py::None;
    let mut v_b: Py = Py::None;
    let t6198 = (Py::tuple(vec![f__vec(Py::str("x"), v_x.clone(), Py::Int(1i64))?, f__vec(Py::str("z"), v_z.clone(), Py::Int(1i64))?])).items()?;
    if t6198.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_a = t6198[0].clone();
    v_b = t6198[1].clone();
    let _ = f__same_len(v_a.clone(), v_b.clone(), Py::str("x"), Py::str("z"))?;
    let _ = Py::tuple(vec![f__fin(Py::str("gamma"), Py::tuple(vec![v_gamma.clone()]))?, f__fin(Py::str("coef0"), Py::tuple(vec![v_coef0.clone()]))?]);
    let _ = f__need({ let t6204 = Py::Bool(matches!(v_degree.clone(), Py::Int(_) | Py::Big(_) | Py::Bool(_))); if !t6204.truthy() { t6204 } else { { let t6201 = Py::Int(1i64); Py::Bool({ let t6202 = v_degree.clone(); if py::le(&t6201, &t6202)? { { let t6203 = Py::Int(10i64); if py::le(&t6202, &t6203)? { true } else { false } } } else { false } }) } } }, Py::str("degree must be a whole number from 1 to 10"))?;
    return Ok(f__out(Py::dict(vec![(Py::str("kernel"), py::pow(&py::add(&py::mul(&f__F(v_gamma.clone())?, &py::sum(&{ let mut out = vec![]; for t6217 in py::iter(&(py::zip(&[v_a.clone(), v_b.clone()])?))? { let mut v_u: Py; let mut v_v: Py; 
let t6218 = (t6217).items()?;
if t6218.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_u = t6218[0].clone();
v_v = t6218[1].clone();
out.push(py::mul(&f__F(v_u.clone())?, &f__F(v_v.clone())?)?); } Py::list(out) }, None)?)?, &f__F(v_coef0.clone())?)?, &v_degree.clone())?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_pooling(mut v_X: Py, mut v_size: Py, mut v_stride: Py) -> R {
    let mut v_H: Py = Py::None;
    let mut v_M: Py = Py::None;
    let mut v_W: Py = Py::None;
    let mut v_oh: Py = Py::None;
    let mut v_ow: Py = Py::None;
    let mut v_win: Py = Py::None;
    v_M = f__mat(Py::str("X"), v_X.clone(), Py::Int(1i64), Py::Int(1i64))?;
    let _ = f__need({ let t6271 = Py::Bool(matches!(v_size.clone(), Py::Int(_) | Py::Big(_) | Py::Bool(_))); if !t6271.truthy() { t6271 } else { { let t6270 = { let t6265 = v_size.clone(); Py::Bool({ let t6266 = Py::Int(1i64); if py::ge(&t6265, &t6266)? { true } else { false } }) }; if !t6270.truthy() { t6270 } else { { let t6269 = Py::Bool(matches!(v_stride.clone(), Py::Int(_) | Py::Big(_) | Py::Bool(_))); if !t6269.truthy() { t6269 } else { { let t6267 = v_stride.clone(); Py::Bool({ let t6268 = Py::Int(1i64); if py::ge(&t6267, &t6268)? { true } else { false } }) } } } } } } }, Py::str("size and stride must be whole numbers >= 1"))?;
    let t6272 = (Py::tuple(vec![py::len(&v_M.clone())?, py::len(&py::getitem(&v_M.clone(), &Py::Int(0i64))?)?])).items()?;
    if t6272.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_H = t6272[0].clone();
    v_W = t6272[1].clone();
    let _ = f__need({ let t6277 = { let t6273 = v_H.clone(); Py::Bool({ let t6274 = v_size.clone(); if py::ge(&t6273, &t6274)? { true } else { false } }) }; if !t6277.truthy() { t6277 } else { { let t6275 = v_W.clone(); Py::Bool({ let t6276 = v_size.clone(); if py::ge(&t6275, &t6276)? { true } else { false } }) } } }, Py::str("the pooling window is larger than the input"))?;
    let t6278 = (Py::tuple(vec![py::add(&py::floordiv(&py::sub(&v_H.clone(), &v_size.clone())?, &v_stride.clone())?, &Py::Int(1i64))?, py::add(&py::floordiv(&py::sub(&v_W.clone(), &v_size.clone())?, &v_stride.clone())?, &Py::Int(1i64))?])).items()?;
    if t6278.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_oh = t6278[0].clone();
    v_ow = t6278[1].clone();
    v_win = { let v_H = v_H.clone(); let v_M = v_M.clone(); let v_W = v_W.clone(); let v_X = v_X.clone(); let v_oh = v_oh.clone(); let v_ow = v_ow.clone(); let v_size = v_size.clone(); let v_stride = v_stride.clone(); py::func(move |a: &[Py]| -> R {
            #[allow(unused_mut, unused_variables, unused_assignments)]
{
            let mut v_i: Py = a.get(0).cloned().unwrap_or(Py::None);
            let mut v_j: Py = a.get(1).cloned().unwrap_or(Py::None);
            return Ok({ let mut out = vec![]; for t6279 in py::range_iter(&[(v_size.clone()).i()?]) { let mut v_a: Py; 
v_a = t6279;
for t6280 in py::range_iter(&[(v_size.clone()).i()?]) { let mut v_b: Py; 
v_b = t6280;
out.push(f__F(py::getitem(&py::getitem(&v_M.clone(), &py::add(&py::mul(&v_i.clone(), &v_stride.clone())?, &v_a.clone())?)?, &py::add(&py::mul(&v_j.clone(), &v_stride.clone())?, &v_b.clone())?)?)?); }} Py::list(out) });
            #[allow(unreachable_code)]
            Ok(Py::None)
        }
        }) };
    return Ok(f__out(Py::dict(vec![(Py::str("max_pool"), { let mut out = vec![]; for t6293 in py::range_iter(&[(v_oh.clone()).i()?]) { let mut v_i: Py; 
v_i = t6293;
out.push({ let mut out = vec![]; for t6294 in py::range_iter(&[(v_ow.clone()).i()?]) { let mut v_j: Py; 
v_j = t6294;
out.push(py::minmax(&[py::call(&v_win.clone(), &[v_i.clone(), v_j.clone()])?], true)?); } Py::list(out) }); } Py::list(out) }), (Py::str("avg_pool"), { let mut out = vec![]; for t6295 in py::range_iter(&[(v_oh.clone()).i()?]) { let mut v_i: Py; 
v_i = t6295;
out.push({ let mut out = vec![]; for t6296 in py::range_iter(&[(v_ow.clone()).i()?]) { let mut v_j: Py; 
v_j = t6296;
out.push(py::truediv(&py::sum(&py::call(&v_win.clone(), &[v_i.clone(), v_j.clone()])?, None)?, &py::mul(&v_size.clone(), &v_size.clone())?)?); } Py::list(out) }); } Py::list(out) })]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_prec_rec_at_k(mut v_rel: Py, mut v_n_relevant: Py, mut v_k: Py) -> R {
    let mut v_hits: Py = Py::None;
    let mut v_r: Py = Py::None;
    v_r = f__rel01(Py::str("rel"), v_rel.clone())?;
    let _ = f__need({ let t6341 = Py::Bool(matches!(v_k.clone(), Py::Int(_) | Py::Big(_) | Py::Bool(_))); if !t6341.truthy() { t6341 } else { { let t6338 = Py::Int(1i64); Py::Bool({ let t6339 = v_k.clone(); if py::le(&t6338, &t6339)? { { let t6340 = py::len(&v_r.clone())?; if py::le(&t6339, &t6340)? { true } else { false } } } else { false } }) } } }, Py::str("k must be between 1 and the length of the ranked list"))?;
    v_hits = py::sum(&py::slice(&v_r.clone(), None, Some((v_k.clone()).i()?), None)?, None)?;
    let _ = f__need({ let t6347 = Py::Bool(matches!(v_n_relevant.clone(), Py::Int(_) | Py::Big(_) | Py::Bool(_))); if !t6347.truthy() { t6347 } else { { let t6346 = { let t6342 = v_n_relevant.clone(); Py::Bool({ let t6343 = py::sum(&v_r.clone(), None)?; if py::ge(&t6342, &t6343)? { true } else { false } }) }; if !t6346.truthy() { t6346 } else { { let t6344 = v_n_relevant.clone(); Py::Bool({ let t6345 = Py::Int(1i64); if py::ge(&t6344, &t6345)? { true } else { false } }) } } } } }, Py::str("n_relevant must be at least the number of relevant items in the list and at least 1"))?;
    return Ok(f__out(Py::dict(vec![(Py::str("precision_at_k"), py::fraction(&[v_hits.clone(), v_k.clone()])?), (Py::str("recall_at_k"), py::fraction(&[v_hits.clone(), v_n_relevant.clone()])?), (Py::str("hits"), v_hits.clone()), (Py::str("hit_rate"), (if (v_hits.clone()).truthy() { Py::Int(1i64) } else { Py::Int(0i64) }))]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_prf_averaged(mut v_y_true: Py, mut v_y_pred: Py, mut v_average: Py) -> R {
    let mut v_F: Py = Py::None;
    let mut v_P: Py = Py::None;
    let mut v_R: Py = Py::None;
    let mut v_S: Py = Py::None;
    let mut v_acc: Py = Py::None;
    let mut v_cls: Py = Py::None;
    let mut v_correct: Py = Py::None;
    let mut v_k: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_yp: Py = Py::None;
    let mut v_yt: Py = Py::None;
    let t6398 = (Py::tuple(vec![f__labels(Py::str("y_true"), v_y_true.clone(), Py::Int(1i64))?, f__labels(Py::str("y_pred"), v_y_pred.clone(), Py::Int(1i64))?])).items()?;
    if t6398.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_yt = t6398[0].clone();
    v_yp = t6398[1].clone();
    let _ = f__same_len(v_yt.clone(), v_yp.clone(), Py::str("y_true"), Py::str("y_pred"))?;
    let t6417 = (f__per_class(v_yt.clone(), v_yp.clone())?).items()?;
    if t6417.len() != 5 { return Err(PyErr::Value("unpack".into())); }
    v_cls = t6417[0].clone();
    v_P = t6417[1].clone();
    v_R = t6417[2].clone();
    v_F = t6417[3].clone();
    v_S = t6417[4].clone();
    if ({ let t6418 = v_average.clone(); Py::Bool({ let t6419 = Py::str("macro"); if py::eq(&t6418, &t6419) { true } else { false } }) }).truthy() {
        v_k = py::len(&v_cls.clone())?;
        return Ok(f__out(Py::dict(vec![(Py::str("precision"), py::truediv(&py::sum(&v_P.clone(), None)?, &v_k.clone())?), (Py::str("recall"), py::truediv(&py::sum(&v_R.clone(), None)?, &v_k.clone())?), (Py::str("f1"), py::truediv(&py::sum(&v_F.clone(), None)?, &v_k.clone())?)]))?);
    }
    if ({ let t6432 = v_average.clone(); Py::Bool({ let t6433 = Py::str("weighted"); if py::eq(&t6432, &t6433) { true } else { false } }) }).truthy() {
        v_n = py::sum(&v_S.clone(), None)?;
        return Ok(f__out(Py::dict(vec![(Py::str("precision"), py::truediv(&py::sum(&{ let mut out = vec![]; for t6434 in py::iter(&(py::zip(&[v_P.clone(), v_S.clone()])?))? { let mut v_p: Py; let mut v_s: Py; 
let t6435 = (t6434).items()?;
if t6435.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_p = t6435[0].clone();
v_s = t6435[1].clone();
out.push(py::mul(&v_p.clone(), &v_s.clone())?); } Py::list(out) }, None)?, &v_n.clone())?), (Py::str("recall"), py::truediv(&py::sum(&{ let mut out = vec![]; for t6436 in py::iter(&(py::zip(&[v_R.clone(), v_S.clone()])?))? { let mut v_r: Py; let mut v_s: Py; 
let t6437 = (t6436).items()?;
if t6437.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_r = t6437[0].clone();
v_s = t6437[1].clone();
out.push(py::mul(&v_r.clone(), &v_s.clone())?); } Py::list(out) }, None)?, &v_n.clone())?), (Py::str("f1"), py::truediv(&py::sum(&{ let mut out = vec![]; for t6438 in py::iter(&(py::zip(&[v_F.clone(), v_S.clone()])?))? { let mut v_f: Py; let mut v_s: Py; 
let t6439 = (t6438).items()?;
if t6439.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_f = t6439[0].clone();
v_s = t6439[1].clone();
out.push(py::mul(&v_f.clone(), &v_s.clone())?); } Py::list(out) }, None)?, &v_n.clone())?)]))?);
    }
    v_correct = py::sum(&{ let mut out = vec![]; for t6440 in py::iter(&(py::zip(&[v_yt.clone(), v_yp.clone()])?))? { let mut v_a: Py; let mut v_b: Py; 
let t6441 = (t6440).items()?;
if t6441.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t6441[0].clone();
v_b = t6441[1].clone();
if ({ let t6442 = v_a.clone(); Py::Bool({ let t6443 = v_b.clone(); if py::eq(&t6442, &t6443) { true } else { false } }) }).truthy() { out.push(Py::Int(1i64)); }} Py::list(out) }, None)?;
    v_acc = py::fraction(&[v_correct.clone(), py::len(&v_yt.clone())?])?;
    return Ok(f__out(Py::dict(vec![(Py::str("precision"), v_acc.clone()), (Py::str("recall"), v_acc.clone()), (Py::str("f1"), v_acc.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_rand_index(mut v_labels_true: Py, mut v_labels_pred: Py) -> R {
    let mut v_a: Py = Py::None;
    let mut v_b: Py = Py::None;
    let mut v_exp: Py = Py::None;
    let mut v_mx: Py = Py::None;
    let mut v_ri: Py = Py::None;
    let mut v_sa: Py = Py::None;
    let mut v_sb: Py = Py::None;
    let mut v_sij: Py = Py::None;
    let mut v_tot: Py = Py::None;
    let t6540 = (Py::tuple(vec![f__labels(Py::str("labels_true"), v_labels_true.clone(), Py::Int(2i64))?, f__labels(Py::str("labels_pred"), v_labels_pred.clone(), Py::Int(2i64))?])).items()?;
    if t6540.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_a = t6540[0].clone();
    v_b = t6540[1].clone();
    let _ = f__same_len(v_a.clone(), v_b.clone(), Py::str("labels_true"), Py::str("labels_pred"))?;
    let t6559 = (f__pairs(v_a.clone(), v_b.clone())?).items()?;
    if t6559.len() != 4 { return Err(PyErr::Value("unpack".into())); }
    v_sij = t6559[0].clone();
    v_sa = t6559[1].clone();
    v_sb = t6559[2].clone();
    v_tot = t6559[3].clone();
    v_ri = py::fraction(&[py::sub(&py::sub(&py::add(&v_tot.clone(), &py::mul(&Py::Int(2i64), &v_sij.clone())?)?, &v_sa.clone())?, &v_sb.clone())?, v_tot.clone()])?;
    v_exp = py::fraction(&[py::mul(&v_sa.clone(), &v_sb.clone())?, v_tot.clone()])?;
    v_mx = py::fraction(&[py::add(&v_sa.clone(), &v_sb.clone())?, Py::Int(2i64)])?;
    let _ = f__need({ let t6560 = v_mx.clone(); Py::Bool({ let t6561 = v_exp.clone(); if py::ne(&t6560, &t6561) { true } else { false } }) }, Py::str("the adjusted Rand index is undefined (both labellings trivial)"))?;
    let _ = f__need({ let t6566 = { let t6562 = v_sa.clone(); Py::Bool({ let t6563 = Py::Int(0i64); if py::gt(&t6562, &t6563)? { true } else { false } }) }; if !t6566.truthy() { t6566 } else { { let t6564 = v_sb.clone(); Py::Bool({ let t6565 = Py::Int(0i64); if py::gt(&t6564, &t6565)? { true } else { false } }) } } }, Py::str("the Fowlkes-Mallows index needs at least one pair inside a cluster in each labelling"))?;
    return Ok(f__out(Py::dict(vec![(Py::str("rand_index"), v_ri.clone()), (Py::str("adjusted_rand_index"), py::truediv(&py::sub(&v_sij.clone(), &v_exp.clone())?, &py::sub(&v_mx.clone(), &v_exp.clone())?)?), (Py::str("fowlkes_mallows"), py::truediv(&v_sij.clone(), &py::math1("sqrt", &py::mul(&v_sa.clone(), &v_sb.clone())?)?)?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_rbf_kernel(mut v_X: Py, mut v_gamma: Py) -> R {
    let mut v_K: Py = Py::None;
    let mut v_M: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_off: Py = Py::None;
    v_M = f__mat(Py::str("X"), v_X.clone(), Py::Int(2i64), Py::Int(1i64))?;
    let _ = f__fin(Py::str("gamma"), Py::tuple(vec![v_gamma.clone()]))?;
    let _ = f__need({ let t6625 = v_gamma.clone(); Py::Bool({ let t6626 = Py::Int(0i64); if py::gt(&t6625, &t6626)? { true } else { false } }) }, Py::str("gamma must be positive"))?;
    v_n = py::len(&v_M.clone())?;
    v_K = { let mut out = vec![]; for t6627 in py::range_iter(&[(v_n.clone()).i()?]) { let mut v_i: Py; 
v_i = t6627;
out.push({ let mut out = vec![]; for t6628 in py::range_iter(&[(v_n.clone()).i()?]) { let mut v_j: Py; 
v_j = t6628;
out.push(py::math1("exp", &py::mul(&py::neg(&v_gamma.clone())?, &py::fsum(&{ let mut out = vec![]; for t6629 in py::iter(&(py::zip(&[py::getitem(&v_M.clone(), &v_i.clone())?, py::getitem(&v_M.clone(), &v_j.clone())?])?))? { let mut v_a: Py; let mut v_b: Py; 
let t6630 = (t6629).items()?;
if t6630.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t6630[0].clone();
v_b = t6630[1].clone();
out.push(py::pow(&py::sub(&v_a.clone(), &v_b.clone())?, &Py::Int(2i64))?); } Py::list(out) })?)?)?); } Py::list(out) }); } Py::list(out) };
    v_off = { let mut out = vec![]; for t6631 in py::range_iter(&[(v_n.clone()).i()?]) { let mut v_i: Py; 
v_i = t6631;
for t6632 in py::range_iter(&[(v_n.clone()).i()?]) { let mut v_j: Py; 
v_j = t6632;
if ({ let t6633 = v_i.clone(); Py::Bool({ let t6634 = v_j.clone(); if py::ne(&t6633, &t6634) { true } else { false } }) }).truthy() { out.push(py::getitem(&py::getitem(&v_K.clone(), &v_i.clone())?, &v_j.clone())?); }}} Py::list(out) };
    return Ok(f__out(Py::dict(vec![(Py::str("K"), v_K.clone()), (Py::str("mean_offdiagonal"), py::truediv(&py::fsum(&v_off.clone())?, &py::len(&v_off.clone())?)?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_receptive_field(mut v_kernels: Py, mut v_strides: Py, mut v_dilations: Py) -> R {
    let mut v_d: Py = Py::None;
    let mut v_dd: Py = Py::None;
    let mut v_jump: Py = Py::None;
    let mut v_k: Py = Py::None;
    let mut v_kk: Py = Py::None;
    let mut v_rf: Py = Py::None;
    let mut v_s: Py = Py::None;
    let mut v_ss: Py = Py::None;
    v_k = f__labels(Py::str("kernels"), v_kernels.clone(), Py::Int(1i64))?;
    v_s = f__labels(Py::str("strides"), v_strides.clone(), Py::Int(1i64))?;
    v_d = f__labels(Py::str("dilations"), v_dilations.clone(), Py::Int(1i64))?;
    let _ = f__need({ let t6685 = py::len(&v_k.clone())?; Py::Bool({ let t6686 = py::len(&v_s.clone())?; if py::eq(&t6685, &t6686) { { let t6687 = py::len(&v_d.clone())?; if py::eq(&t6686, &t6687) { true } else { false } } } else { false } }) }, Py::str("kernels, strides and dilations need one entry per layer"))?;
    let _ = f__need(Py::Bool(({ let mut out = vec![]; for t6688 in py::iter(&(py::add(&py::add(&v_k.clone(), &v_s.clone())?, &v_d.clone())?))? { let mut v_v: Py; 
v_v = t6688;
out.push({ let t6689 = v_v.clone(); Py::Bool({ let t6690 = Py::Int(1i64); if py::ge(&t6689, &t6690)? { true } else { false } }) }); } Py::list(out) }).items()?.iter().all(|x| x.truthy())), Py::str("kernel sizes, strides and dilations must be >= 1"))?;
    let _ = f__need({ let t6691 = py::len(&v_k.clone())?; Py::Bool({ let t6692 = Py::Int(20i64); if py::le(&t6691, &t6692)? { true } else { false } }) }, Py::str("at most 20 layers"))?;
    let t6693 = (Py::tuple(vec![Py::Int(1i64), Py::Int(1i64)])).items()?;
    if t6693.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_rf = t6693[0].clone();
    v_jump = t6693[1].clone();
    for t6694 in py::iter(&(py::zip(&[v_k.clone(), v_s.clone(), v_d.clone()])?))? {
        let t6695 = (t6694).items()?;
        if t6695.len() != 3 { return Err(PyErr::Value("unpack".into())); }
        v_kk = t6695[0].clone();
        v_ss = t6695[1].clone();
        v_dd = t6695[2].clone();
        v_rf = py::iadd(&v_rf, &py::mul(&py::mul(&py::sub(&v_kk.clone(), &Py::Int(1i64))?, &v_dd.clone())?, &v_jump.clone())?)?;
        v_jump = py::mul(&v_jump, &v_ss.clone())?;
    }
    let _ = f__need({ let t6696 = v_rf.clone(); Py::Bool({ let t6697 = Py::Int(20000i64); if py::le(&t6696, &t6697)? { true } else { false } }) }, Py::str("the receptive field is too large (more than 20000)"))?;
    return Ok(f__out(Py::dict(vec![(Py::str("receptive_field"), v_rf.clone()), (Py::str("jump"), v_jump.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_regression_metrics(mut v_y_true: Py, mut v_y_pred: Py) -> R {
    let mut v_P: Py = Py::None;
    let mut v_Y: Py = Py::None;
    let mut v_e: Py = Py::None;
    let mut v_ebar: Py = Py::None;
    let mut v_mse: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_sse: Py = Py::None;
    let mut v_sst: Py = Py::None;
    let mut v_ybar: Py = Py::None;
    let mut v_yp: Py = Py::None;
    let mut v_yt: Py = Py::None;
    let t6750 = (f__pair(v_y_true.clone(), v_y_pred.clone(), Py::Int(2i64))?).items()?;
    if t6750.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_yt = t6750[0].clone();
    v_yp = t6750[1].clone();
    let t6753 = (Py::tuple(vec![{ let mut out = vec![]; for t6751 in py::iter(&(v_yt.clone()))? { let mut v_v: Py; 
v_v = t6751;
out.push(f__F(v_v.clone())?); } Py::list(out) }, { let mut out = vec![]; for t6752 in py::iter(&(v_yp.clone()))? { let mut v_v: Py; 
v_v = t6752;
out.push(f__F(v_v.clone())?); } Py::list(out) }])).items()?;
    if t6753.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_Y = t6753[0].clone();
    v_P = t6753[1].clone();
    v_n = py::len(&v_Y.clone())?;
    v_e = { let mut out = vec![]; for t6754 in py::iter(&(py::zip(&[v_Y.clone(), v_P.clone()])?))? { let mut v_a: Py; let mut v_b: Py; 
let t6755 = (t6754).items()?;
if t6755.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t6755[0].clone();
v_b = t6755[1].clone();
out.push(py::sub(&v_a.clone(), &v_b.clone())?); } Py::list(out) };
    v_ybar = py::truediv(&py::sum(&v_Y.clone(), None)?, &v_n.clone())?;
    v_sst = py::sum(&{ let mut out = vec![]; for t6756 in py::iter(&(v_Y.clone()))? { let mut v_a: Py; 
v_a = t6756;
out.push(py::pow(&py::sub(&v_a.clone(), &v_ybar.clone())?, &Py::Int(2i64))?); } Py::list(out) }, None)?;
    let _ = f__need({ let t6757 = v_sst.clone(); Py::Bool({ let t6758 = Py::Int(0i64); if py::ne(&t6757, &t6758) { true } else { false } }) }, Py::str("R^2 is undefined: every true value is the same"))?;
    v_sse = py::sum(&{ let mut out = vec![]; for t6759 in py::iter(&(v_e.clone()))? { let mut v_v: Py; 
v_v = t6759;
out.push(py::mul(&v_v.clone(), &v_v.clone())?); } Py::list(out) }, None)?;
    v_ebar = py::truediv(&py::sum(&v_e.clone(), None)?, &v_n.clone())?;
    v_mse = py::truediv(&v_sse.clone(), &v_n.clone())?;
    return Ok(f__out(Py::dict(vec![(Py::str("mae"), py::truediv(&py::sum(&{ let mut out = vec![]; for t6772 in py::iter(&(v_e.clone()))? { let mut v_v: Py; 
v_v = t6772;
out.push(py::abs(&v_v.clone())?); } Py::list(out) }, None)?, &v_n.clone())?), (Py::str("mse"), v_mse.clone()), (Py::str("rmse"), py::math1("sqrt", &v_mse.clone())?), (Py::str("r2"), py::sub(&Py::Int(1i64), &py::truediv(&v_sse.clone(), &v_sst.clone())?)?), (Py::str("explained_variance"), py::sub(&Py::Int(1i64), &py::truediv(&py::sum(&{ let mut out = vec![]; for t6773 in py::iter(&(v_e.clone()))? { let mut v_v: Py; 
v_v = t6773;
out.push(py::pow(&py::sub(&v_v.clone(), &v_ebar.clone())?, &Py::Int(2i64))?); } Py::list(out) }, None)?, &v_sst.clone())?)?), (Py::str("median_ae"), f__median({ let mut out = vec![]; for t6774 in py::iter(&(v_e.clone()))? { let mut v_v: Py; 
v_v = t6774;
out.push(py::abs(&v_v.clone())?); } Py::list(out) })?), (Py::str("max_error"), py::minmax(&[{ let mut out = vec![]; for t6775 in py::iter(&(v_e.clone()))? { let mut v_v: Py; 
v_v = t6775;
out.push(py::abs(&v_v.clone())?); } Py::list(out) }], true)?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_ridge(mut v_X: Py, mut v_y: Py, mut v_alpha: Py) -> R {
    let mut v_F: Py = Py::None;
    let mut v_G: Py = Py::None;
    let mut v_M: Py = Py::None;
    let mut v_Xc: Py = Py::None;
    let mut v_Y: Py = Py::None;
    let mut v_b: Py = Py::None;
    let mut v_d: Py = Py::None;
    let mut v_h: Py = Py::None;
    let mut v_mse: Py = Py::None;
    let mut v_mx: Py = Py::None;
    let mut v_my: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_w: Py = Py::None;
    let mut v_yv: Py = Py::None;
    v_M = f__mat(Py::str("X"), v_X.clone(), Py::Int(2i64), Py::Int(1i64))?;
    v_yv = f__vec(Py::str("y"), v_y.clone(), Py::Int(1i64))?;
    let t6827 = (Py::tuple(vec![py::len(&v_M.clone())?, py::len(&py::getitem(&v_M.clone(), &Py::Int(0i64))?)?])).items()?;
    if t6827.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_n = t6827[0].clone();
    v_d = t6827[1].clone();
    let _ = f__need({ let t6828 = py::len(&v_yv.clone())?; Py::Bool({ let t6829 = v_n.clone(); if py::eq(&t6828, &t6829) { true } else { false } }) }, Py::str("y needs one value per row of X"))?;
    let _ = f__fin(Py::str("alpha"), Py::tuple(vec![v_alpha.clone()]))?;
    let _ = f__need({ let t6830 = v_alpha.clone(); Py::Bool({ let t6831 = Py::Int(0i64); if py::ge(&t6830, &t6831)? { true } else { false } }) }, Py::str("alpha must be >= 0"))?;
    v_F = { let mut out = vec![]; for t6832 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t6832;
out.push({ let mut out = vec![]; for t6833 in py::iter(&(v_r.clone()))? { let mut v_v: Py; 
v_v = t6833;
out.push(f__F(v_v.clone())?); } Py::list(out) }); } Py::list(out) };
    v_Y = { let mut out = vec![]; for t6834 in py::iter(&(v_yv.clone()))? { let mut v_v: Py; 
v_v = t6834;
out.push(f__F(v_v.clone())?); } Py::list(out) };
    v_mx = { let mut out = vec![]; for t6835 in py::range_iter(&[(v_d.clone()).i()?]) { let mut v_j: Py; 
v_j = t6835;
out.push(py::truediv(&py::sum(&{ let mut out = vec![]; for t6836 in py::iter(&(v_F.clone()))? { let mut v_r: Py; 
v_r = t6836;
out.push(py::getitem(&v_r.clone(), &v_j.clone())?); } Py::list(out) }, None)?, &v_n.clone())?); } Py::list(out) };
    v_my = py::truediv(&py::sum(&v_Y.clone(), None)?, &v_n.clone())?;
    v_Xc = { let mut out = vec![]; for t6837 in py::iter(&(v_F.clone()))? { let mut v_r: Py; 
v_r = t6837;
out.push({ let mut out = vec![]; for t6838 in py::range_iter(&[(v_d.clone()).i()?]) { let mut v_j: Py; 
v_j = t6838;
out.push(py::sub(&py::getitem(&v_r.clone(), &v_j.clone())?, &py::getitem(&v_mx.clone(), &v_j.clone())?)?); } Py::list(out) }); } Py::list(out) };
    v_G = { let mut out = vec![]; for t6839 in py::range_iter(&[(v_d.clone()).i()?]) { let mut v_i: Py; 
v_i = t6839;
out.push({ let mut out = vec![]; for t6840 in py::range_iter(&[(v_d.clone()).i()?]) { let mut v_j: Py; 
v_j = t6840;
out.push(py::add(&py::sum(&{ let mut out = vec![]; for t6841 in py::iter(&(v_Xc.clone()))? { let mut v_r: Py; 
v_r = t6841;
out.push(py::mul(&py::getitem(&v_r.clone(), &v_i.clone())?, &py::getitem(&v_r.clone(), &v_j.clone())?)?); } Py::list(out) }, None)?, &(if ({ let t6842 = v_i.clone(); Py::Bool({ let t6843 = v_j.clone(); if py::eq(&t6842, &t6843) { true } else { false } }) }).truthy() { f__F(v_alpha.clone())? } else { Py::Int(0i64) }))?); } Py::list(out) }); } Py::list(out) };
    v_h = { let mut out = vec![]; for t6844 in py::range_iter(&[(v_d.clone()).i()?]) { let mut v_i: Py; 
v_i = t6844;
out.push(py::sum(&{ let mut out = vec![]; for t6845 in py::iter(&(py::zip(&[v_Xc.clone(), v_Y.clone()])?))? { let mut v_r: Py; let mut v_t: Py; 
let t6846 = (t6845).items()?;
if t6846.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_r = t6846[0].clone();
v_t = t6846[1].clone();
out.push(py::mul(&py::getitem(&v_r.clone(), &v_i.clone())?, &py::sub(&v_t.clone(), &v_my.clone())?)?); } Py::list(out) }, None)?); } Py::list(out) };
    v_w = f__solve_exact(v_G.clone(), v_h.clone())?;
    v_b = py::sub(&v_my.clone(), &py::sum(&{ let mut out = vec![]; for t6868 in py::iter(&(py::zip(&[v_mx.clone(), v_w.clone()])?))? { let mut v_a: Py; let mut v_c: Py; 
let t6869 = (t6868).items()?;
if t6869.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t6869[0].clone();
v_c = t6869[1].clone();
out.push(py::mul(&v_a.clone(), &v_c.clone())?); } Py::list(out) }, None)?)?;
    v_mse = py::truediv(&py::sum(&{ let mut out = vec![]; for t6870 in py::iter(&(py::zip(&[v_F.clone(), v_Y.clone()])?))? { let mut v_r: Py; let mut v_t: Py; 
let t6871 = (t6870).items()?;
if t6871.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_r = t6871[0].clone();
v_t = t6871[1].clone();
out.push(py::pow(&py::sub(&py::add(&py::sum(&{ let mut out = vec![]; for t6872 in py::iter(&(py::zip(&[v_r.clone(), v_w.clone()])?))? { let mut v_a: Py; let mut v_c: Py; 
let t6873 = (t6872).items()?;
if t6873.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t6873[0].clone();
v_c = t6873[1].clone();
out.push(py::mul(&v_a.clone(), &v_c.clone())?); } Py::list(out) }, None)?, &v_b.clone())?, &v_t.clone())?, &Py::Int(2i64))?); } Py::list(out) }, None)?, &v_n.clone())?;
    return Ok(f__out(Py::dict(vec![(Py::str("w"), v_w.clone()), (Py::str("b"), v_b.clone()), (Py::str("train_mse"), v_mse.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_rnn_params(mut v_cell: Py, mut v_input_size: Py, mut v_hidden_size: Py, mut v_num_layers: Py, mut v_bidirectional: Py) -> R {
    let mut v_G: Py = Py::None;
    let mut v_dirs: Py = Py::None;
    let mut v_inp: Py = Py::None;
    let mut v_layer: Py = Py::None;
    let mut v_per: Py = Py::None;
    let _ = f__need(Py::Bool(({ let mut out = vec![]; for t6915 in py::iter(&(Py::tuple(vec![v_input_size.clone(), v_hidden_size.clone(), v_num_layers.clone()])))? { let mut v_v: Py; 
v_v = t6915;
out.push({ let t6918 = Py::Bool(matches!(v_v.clone(), Py::Int(_) | Py::Big(_) | Py::Bool(_))); if !t6918.truthy() { t6918 } else { { let t6916 = v_v.clone(); Py::Bool({ let t6917 = Py::Int(1i64); if py::ge(&t6916, &t6917)? { true } else { false } }) } } }); } Py::list(out) }).items()?.iter().all(|x| x.truthy())), Py::str("sizes must be whole numbers >= 1"))?;
    let _ = f__need({ let t6926 = { let t6919 = v_num_layers.clone(); Py::Bool({ let t6920 = Py::Int(200i64); if py::le(&t6919, &t6920)? { true } else { false } }) }; if !t6926.truthy() { t6926 } else { { let t6925 = { let t6921 = v_input_size.clone(); Py::Bool({ let t6922 = Py::Int(1000000i64); if py::le(&t6921, &t6922)? { true } else { false } }) }; if !t6925.truthy() { t6925 } else { { let t6923 = v_hidden_size.clone(); Py::Bool({ let t6924 = Py::Int(65536i64); if py::le(&t6923, &t6924)? { true } else { false } }) } } } } }, Py::str("num_layers must be at most 200, input_size at most 1000000 and hidden_size at most 65536"))?;
    v_G = py::getitem(&Py::dict(vec![(Py::str("lstm"), Py::Int(4i64)), (Py::str("gru"), Py::Int(3i64)), (Py::str("rnn"), Py::Int(1i64))]), &v_cell.clone())?;
    v_dirs = (if (v_bidirectional.clone()).truthy() { Py::Int(2i64) } else { Py::Int(1i64) });
    v_per = Py::list(vec![]);
    for t6927 in py::range_iter(&[(v_num_layers.clone()).i()?]) {
        v_layer = t6927;
        v_inp = (if ({ let t6928 = v_layer.clone(); Py::Bool({ let t6929 = Py::Int(0i64); if py::eq(&t6928, &t6929) { true } else { false } }) }).truthy() { v_input_size.clone() } else { py::mul(&v_hidden_size.clone(), &v_dirs.clone())? });
        let _ = py::method(&v_per.clone(), "append", &[py::mul(&v_dirs.clone(), &py::add(&py::mul(&py::mul(&v_G.clone(), &v_hidden_size.clone())?, &py::add(&v_inp.clone(), &v_hidden_size.clone())?)?, &py::mul(&py::mul(&Py::Int(2i64), &v_G.clone())?, &v_hidden_size.clone())?)?)?])?;
    }
    return Ok(f__out(Py::dict(vec![(Py::str("per_layer"), v_per.clone()), (Py::str("total"), py::sum(&v_per.clone(), None)?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_robust_scaler(mut v_X: Py) -> R {
    let mut v_F: Py = Py::None;
    let mut v_M: Py = Py::None;
    let mut v_S: Py = Py::None;
    let mut v_col: Py = Py::None;
    let mut v_d: Py = Py::None;
    let mut v_iqr: Py = Py::None;
    let mut v_j: Py = Py::None;
    let mut v_med: Py = Py::None;
    v_M = f__mat(Py::str("X"), v_X.clone(), Py::Int(1i64), Py::Int(1i64))?;
    v_d = py::len(&py::getitem(&v_M.clone(), &Py::Int(0i64))?)?;
    v_F = { let mut out = vec![]; for t6988 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t6988;
out.push({ let mut out = vec![]; for t6989 in py::iter(&(v_r.clone()))? { let mut v_v: Py; 
v_v = t6989;
out.push(f__F(v_v.clone())?); } Py::list(out) }); } Py::list(out) };
    let t6990 = (Py::tuple(vec![Py::list(vec![]), Py::list(vec![])])).items()?;
    if t6990.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_med = t6990[0].clone();
    v_iqr = t6990[1].clone();
    for t6991 in py::range_iter(&[(v_d.clone()).i()?]) {
        v_j = t6991;
        v_col = py::sorted(&{ let mut out = vec![]; for t6992 in py::iter(&(v_F.clone()))? { let mut v_r: Py; 
v_r = t6992;
out.push(py::getitem(&v_r.clone(), &v_j.clone())?); } Py::list(out) }, (Py::Bool(false)).truthy())?;
        let _ = py::method(&v_med.clone(), "append", &[f__percentile(v_col.clone(), py::fraction(&[Py::Int(1i64), Py::Int(2i64)])?)?])?;
        let _ = py::method(&v_iqr.clone(), "append", &[py::sub(&f__percentile(v_col.clone(), py::fraction(&[Py::Int(3i64), Py::Int(4i64)])?)?, &f__percentile(v_col.clone(), py::fraction(&[Py::Int(1i64), Py::Int(4i64)])?)?)?])?;
    }
    v_S = { let mut out = vec![]; for t6993 in py::iter(&(v_F.clone()))? { let mut v_r: Py; 
v_r = t6993;
out.push({ let mut out = vec![]; for t6994 in py::range_iter(&[(v_d.clone()).i()?]) { let mut v_j: Py; 
v_j = t6994;
out.push((if ({ let t6995 = py::getitem(&v_iqr.clone(), &v_j.clone())?; Py::Bool({ let t6996 = Py::Int(0i64); if py::ne(&t6995, &t6996) { true } else { false } }) }).truthy() { py::truediv(&py::sub(&py::getitem(&v_r.clone(), &v_j.clone())?, &py::getitem(&v_med.clone(), &v_j.clone())?)?, &py::getitem(&v_iqr.clone(), &v_j.clone())?)? } else { py::sub(&py::getitem(&v_r.clone(), &v_j.clone())?, &py::getitem(&v_med.clone(), &v_j.clone())?)? })); } Py::list(out) }); } Py::list(out) };
    return Ok(f__out(Py::dict(vec![(Py::str("medians"), v_med.clone()), (Py::str("iqr"), v_iqr.clone()), (Py::str("scaled"), v_S.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_roc_auc(mut v_y_true: Py, mut v_scores: Py) -> R {
    let mut v_a: Py = Py::None;
    let mut v_auc: Py = Py::None;
    let mut v_b: Py = Py::None;
    let mut v_neg: Py = Py::None;
    let mut v_pos: Py = Py::None;
    let mut v_s: Py = Py::None;
    let mut v_wins: Py = Py::None;
    let mut v_yt: Py = Py::None;
    v_yt = f__binary(Py::str("y_true"), v_y_true.clone(), Py::Int(1i64))?;
    v_s = f__vec(Py::str("scores"), v_scores.clone(), Py::Int(1i64))?;
    let _ = f__same_len(v_yt.clone(), v_s.clone(), Py::str("y_true"), Py::str("scores"))?;
    v_pos = { let mut out = vec![]; for t7057 in py::iter(&(py::zip(&[v_yt.clone(), v_s.clone()])?))? { let mut v_v: Py; let mut v_y: Py; 
let t7058 = (t7057).items()?;
if t7058.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_y = t7058[0].clone();
v_v = t7058[1].clone();
if ({ let t7059 = v_y.clone(); Py::Bool({ let t7060 = Py::Int(1i64); if py::eq(&t7059, &t7060) { true } else { false } }) }).truthy() { out.push(v_v.clone()); }} Py::list(out) };
    v_neg = { let mut out = vec![]; for t7061 in py::iter(&(py::zip(&[v_yt.clone(), v_s.clone()])?))? { let mut v_v: Py; let mut v_y: Py; 
let t7062 = (t7061).items()?;
if t7062.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_y = t7062[0].clone();
v_v = t7062[1].clone();
if ({ let t7063 = v_y.clone(); Py::Bool({ let t7064 = Py::Int(0i64); if py::eq(&t7063, &t7064) { true } else { false } }) }).truthy() { out.push(v_v.clone()); }} Py::list(out) };
    let _ = f__need({ let t7065 = v_pos.clone(); if !t7065.truthy() { t7065 } else { v_neg.clone() } }, Py::str("ROC AUC needs at least one positive and one negative sample"))?;
    v_wins = py::fraction(&[Py::Int(0i64)])?;
    for t7066 in py::iter(&(v_pos.clone()))? {
        v_a = t7066;
        for t7067 in py::iter(&(v_neg.clone()))? {
            v_b = t7067;
            v_wins = py::iadd(&v_wins, &(if ({ let t7068 = v_a.clone(); Py::Bool({ let t7069 = v_b.clone(); if py::gt(&t7068, &t7069)? { true } else { false } }) }).truthy() { Py::Int(1i64) } else { (if ({ let t7070 = v_a.clone(); Py::Bool({ let t7071 = v_b.clone(); if py::eq(&t7070, &t7071) { true } else { false } }) }).truthy() { py::fraction(&[Py::Int(1i64), Py::Int(2i64)])? } else { Py::Int(0i64) }) }))?;
        }
    }
    v_auc = py::truediv(&v_wins.clone(), &py::mul(&py::len(&v_pos.clone())?, &py::len(&v_neg.clone())?)?)?;
    return Ok(f__out(Py::dict(vec![(Py::str("auc"), v_auc.clone()), (Py::str("gini"), py::sub(&py::mul(&Py::Int(2i64), &v_auc.clone())?, &Py::Int(1i64))?), (Py::str("n_pos"), py::len(&v_pos.clone())?), (Py::str("n_neg"), py::len(&v_neg.clone())?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_roc_curve(mut v_y_true: Py, mut v_scores: Py) -> R {
    let mut v_N: Py = Py::None;
    let mut v_P: Py = Py::None;
    let mut v_auc: Py = Py::None;
    let mut v_fpr: Py = Py::None;
    let mut v_fps: Py = Py::None;
    let mut v_s: Py = Py::None;
    let mut v_th: Py = Py::None;
    let mut v_tpr: Py = Py::None;
    let mut v_tps: Py = Py::None;
    let mut v_yt: Py = Py::None;
    v_yt = f__binary(Py::str("y_true"), v_y_true.clone(), Py::Int(1i64))?;
    v_s = f__vec(Py::str("scores"), v_scores.clone(), Py::Int(1i64))?;
    let _ = f__same_len(v_yt.clone(), v_s.clone(), Py::str("y_true"), Py::str("scores"))?;
    let t7132 = (Py::tuple(vec![py::sum(&v_yt.clone(), None)?, py::sub(&py::len(&v_yt.clone())?, &py::sum(&v_yt.clone(), None)?)?])).items()?;
    if t7132.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_P = t7132[0].clone();
    v_N = t7132[1].clone();
    let _ = f__need({ let t7137 = { let t7133 = v_P.clone(); Py::Bool({ let t7134 = Py::Int(0i64); if py::gt(&t7133, &t7134)? { true } else { false } }) }; if !t7137.truthy() { t7137 } else { { let t7135 = v_N.clone(); Py::Bool({ let t7136 = Py::Int(0i64); if py::gt(&t7135, &t7136)? { true } else { false } }) } } }, Py::str("the ROC curve needs at least one positive and one negative sample"))?;
    let t7147 = (f__curve(v_yt.clone(), v_s.clone())?).items()?;
    if t7147.len() != 3 { return Err(PyErr::Value("unpack".into())); }
    v_th = t7147[0].clone();
    v_tps = t7147[1].clone();
    v_fps = t7147[2].clone();
    v_fpr = py::add(&Py::list(vec![py::fraction(&[Py::Int(0i64)])?]), &{ let mut out = vec![]; for t7148 in py::iter(&(v_fps.clone()))? { let mut v_f: Py; 
v_f = t7148;
out.push(py::fraction(&[v_f.clone(), v_N.clone()])?); } Py::list(out) })?;
    v_tpr = py::add(&Py::list(vec![py::fraction(&[Py::Int(0i64)])?]), &{ let mut out = vec![]; for t7149 in py::iter(&(v_tps.clone()))? { let mut v_t: Py; 
v_t = t7149;
out.push(py::fraction(&[v_t.clone(), v_P.clone()])?); } Py::list(out) })?;
    v_auc = py::sum(&{ let mut out = vec![]; for t7150 in py::range_iter(&[(py::sub(&py::len(&v_fpr.clone())?, &Py::Int(1i64))?).i()?]) { let mut v_i: Py; 
v_i = t7150;
out.push(py::truediv(&py::mul(&py::sub(&py::getitem(&v_fpr.clone(), &py::add(&v_i.clone(), &Py::Int(1i64))?)?, &py::getitem(&v_fpr.clone(), &v_i.clone())?)?, &py::add(&py::getitem(&v_tpr.clone(), &py::add(&v_i.clone(), &Py::Int(1i64))?)?, &py::getitem(&v_tpr.clone(), &v_i.clone())?)?)?, &Py::Int(2i64))?); } Py::list(out) }, None)?;
    return Ok(f__out(Py::dict(vec![(Py::str("thresholds"), v_th.clone()), (Py::str("fpr"), v_fpr.clone()), (Py::str("tpr"), v_tpr.clone()), (Py::str("auc"), v_auc.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_sigmoid(mut v_z: Py) -> R {
    let mut v_e: Py = Py::None;
    let _ = f__fin(Py::str("z"), Py::tuple(vec![v_z.clone()]))?;
    v_e = py::math1("exp", &py::neg(&py::abs(&v_z.clone())?)?)?;
    return Ok(f__out(Py::dict(vec![(Py::str("sigmoid"), f__sigmoid(v_z.clone())?), (Py::str("derivative"), py::truediv(&v_e.clone(), &py::pow(&py::add(&Py::Int(1i64), &v_e.clone())?, &Py::Int(2i64))?)?), (Py::str("log_sigmoid"), f__log_sigmoid(v_z.clone())?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_silhouette(mut v_X: Py, mut v_labels: Py) -> R {
    let mut v_D: Py = Py::None;
    let mut v_M: Py = Py::None;
    let mut v_a: Py = Py::None;
    let mut v_b: Py = Py::None;
    let mut v_i: Py = Py::None;
    let mut v_ks: Py = Py::None;
    let mut v_lab: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_own: Py = Py::None;
    let mut v_s: Py = Py::None;
    let mut v_v: Py = Py::None;
    let t7419 = (f__points(v_X.clone(), v_labels.clone())?).items()?;
    if t7419.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_M = t7419[0].clone();
    v_lab = t7419[1].clone();
    v_n = py::len(&v_M.clone())?;
    v_ks = py::sorted(&py::set(&v_lab.clone())?, (Py::Bool(false)).truthy())?;
    let _ = f__need({ let t7420 = Py::Int(2i64); Py::Bool({ let t7421 = py::len(&v_ks.clone())?; if py::le(&t7420, &t7421)? { { let t7422 = py::sub(&v_n.clone(), &Py::Int(1i64))?; if py::le(&t7421, &t7422)? { true } else { false } } } else { false } }) }, Py::str("the silhouette needs between 2 and n - 1 clusters"))?;
    v_D = { let mut out = vec![]; for t7423 in py::range_iter(&[(v_n.clone()).i()?]) { let mut v_i: Py; 
v_i = t7423;
out.push({ let mut out = vec![]; for t7424 in py::range_iter(&[(v_n.clone()).i()?]) { let mut v_j: Py; 
v_j = t7424;
out.push(py::math1("sqrt", &f__sqd(py::getitem(&v_M.clone(), &v_i.clone())?, py::getitem(&v_M.clone(), &v_j.clone())?)?)?); } Py::list(out) }); } Py::list(out) };
    v_s = Py::list(vec![]);
    for t7427 in py::range_iter(&[(v_n.clone()).i()?]) {
        v_i = t7427;
        v_own = { let mut out = vec![]; for t7428 in py::range_iter(&[(v_n.clone()).i()?]) { let mut v_j: Py; 
v_j = t7428;
if ({ let t7433 = { let t7429 = py::getitem(&v_lab.clone(), &v_j.clone())?; Py::Bool({ let t7430 = py::getitem(&v_lab.clone(), &v_i.clone())?; if py::eq(&t7429, &t7430) { true } else { false } }) }; if !t7433.truthy() { t7433 } else { { let t7431 = v_j.clone(); Py::Bool({ let t7432 = v_i.clone(); if py::ne(&t7431, &t7432) { true } else { false } }) } } }).truthy() { out.push(v_j.clone()); }} Py::list(out) };
        if (py::not(&v_own.clone())).truthy() {
            let _ = py::method(&v_s.clone(), "append", &[Py::Float(0.0)])?;
            continue;
        }
        v_a = py::truediv(&py::fsum(&{ let mut out = vec![]; for t7434 in py::iter(&(v_own.clone()))? { let mut v_j: Py; 
v_j = t7434;
out.push(py::getitem(&py::getitem(&v_D.clone(), &v_i.clone())?, &v_j.clone())?); } Py::list(out) })?, &py::len(&v_own.clone())?)?;
        v_b = py::minmax(&[{ let mut out = vec![]; for t7435 in py::iter(&(v_ks.clone()))? { let mut v_k: Py; 
v_k = t7435;
if ({ let t7436 = v_k.clone(); Py::Bool({ let t7437 = py::getitem(&v_lab.clone(), &v_i.clone())?; if py::ne(&t7436, &t7437) { true } else { false } }) }).truthy() { out.push(py::truediv(&py::fsum(&{ let mut out = vec![]; for t7438 in py::range_iter(&[(v_n.clone()).i()?]) { let mut v_j: Py; 
v_j = t7438;
if ({ let t7439 = py::getitem(&v_lab.clone(), &v_j.clone())?; Py::Bool({ let t7440 = v_k.clone(); if py::eq(&t7439, &t7440) { true } else { false } }) }).truthy() { out.push(py::getitem(&py::getitem(&v_D.clone(), &v_i.clone())?, &v_j.clone())?); }} Py::list(out) })?, &py::method(&v_lab.clone(), "count", &[v_k.clone()])?)?); }} Py::list(out) }], false)?;
        v_v = py::truediv(&py::sub(&v_b.clone(), &v_a.clone())?, &py::minmax(&[v_a.clone(), v_b.clone()], true)?)?;
        let _ = py::method(&v_s.clone(), "append", &[(if ({ let t7441 = py::abs(&v_v.clone())?; Py::Bool({ let t7442 = Py::Float(1e-12); if py::lt(&t7441, &t7442)? { true } else { false } }) }).truthy() { Py::Float(0.0) } else { v_v.clone() })])?;
    }
    return Ok(f__out(Py::dict(vec![(Py::str("silhouette"), py::truediv(&py::fsum(&v_s.clone())?, &v_n.clone())?), (Py::str("samples"), v_s.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_softmax(mut v_z: Py) -> R {
    let mut v_zv: Py = Py::None;
    v_zv = f__vec(Py::str("z"), v_z.clone(), Py::Int(2i64))?;
    return Ok(f__out(Py::dict(vec![(Py::str("probabilities"), f__softmax(v_zv.clone())?), (Py::str("log_sum_exp"), f__logsumexp(v_zv.clone())?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_softmax_temperature(mut v_z: Py, mut v_T: Py) -> R {
    let mut v_p: Py = Py::None;
    let mut v_zv: Py = Py::None;
    v_zv = f__vec(Py::str("z"), v_z.clone(), Py::Int(2i64))?;
    let _ = f__fin(Py::str("T"), Py::tuple(vec![v_T.clone()]))?;
    let _ = f__need({ let t7544 = v_T.clone(); Py::Bool({ let t7545 = Py::Int(0i64); if py::gt(&t7544, &t7545)? { true } else { false } }) }, Py::str("the temperature T must be positive"))?;
    v_p = f__softmax({ let mut out = vec![]; for t7548 in py::iter(&(v_zv.clone()))? { let mut v_v: Py; 
v_v = t7548;
out.push(py::truediv(&v_v.clone(), &v_T.clone())?); } Py::list(out) })?;
    return Ok(f__out(Py::dict(vec![(Py::str("probabilities"), v_p.clone()), (Py::str("entropy"), py::neg(&py::fsum(&{ let mut out = vec![]; for t7561 in py::iter(&(v_p.clone()))? { let mut v_q: Py; 
v_q = t7561;
if ({ let t7562 = v_q.clone(); Py::Bool({ let t7563 = Py::Int(0i64); if py::gt(&t7562, &t7563)? { true } else { false } }) }).truthy() { out.push(py::mul(&v_q.clone(), &py::math1("log", &v_q.clone())?)?); }} Py::list(out) })?)?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_split_sizes(mut v_n: Py, mut v_test_size: Py) -> R {
    let mut v_n_test: Py = Py::None;
    let mut v_n_train: Py = Py::None;
    let mut v_t: Py = Py::None;
    let _ = f__need({ let t7900 = Py::Bool(matches!(v_n.clone(), Py::Int(_) | Py::Big(_) | Py::Bool(_))); if !t7900.truthy() { t7900 } else { { let t7898 = v_n.clone(); Py::Bool({ let t7899 = Py::Int(2i64); if py::ge(&t7898, &t7899)? { true } else { false } }) } } }, Py::str("n must be a whole number >= 2"))?;
    let _ = f__fin(Py::str("test_size"), Py::tuple(vec![v_test_size.clone()]))?;
    let _ = f__need({ let t7904 = Py::Int(0i64); Py::Bool({ let t7905 = v_test_size.clone(); if py::lt(&t7904, &t7905)? { { let t7906 = Py::Int(1i64); if py::lt(&t7905, &t7906)? { true } else { false } } } else { false } }) }, Py::str("test_size must be a fraction strictly between 0 and 1"))?;
    v_t = py::fraction(&[Py::str(&py::repr(&py::float(&v_test_size.clone())?))])?;
    v_n_test = py::math1("ceil", &py::mul(&v_t.clone(), &v_n.clone())?)?;
    v_n_train = py::sub(&v_n.clone(), &v_n_test.clone())?;
    let _ = f__need({ let t7907 = v_n_train.clone(); Py::Bool({ let t7908 = Py::Int(1i64); if py::ge(&t7907, &t7908)? { true } else { false } }) }, Py::str("the training set would be empty"))?;
    return Ok(f__out(Py::dict(vec![(Py::str("n_test"), v_n_test.clone()), (Py::str("n_train"), v_n_train.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_standard_scaler(mut v_X: Py) -> R {
    let mut v_F: Py = Py::None;
    let mut v_M: Py = Py::None;
    let mut v_Z: Py = Py::None;
    let mut v_d: Py = Py::None;
    let mut v_mean: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_sd: Py = Py::None;
    let mut v_var: Py = Py::None;
    v_M = f__mat(Py::str("X"), v_X.clone(), Py::Int(1i64), Py::Int(1i64))?;
    let t7610 = (Py::tuple(vec![py::len(&v_M.clone())?, py::len(&py::getitem(&v_M.clone(), &Py::Int(0i64))?)?])).items()?;
    if t7610.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_n = t7610[0].clone();
    v_d = t7610[1].clone();
    v_F = { let mut out = vec![]; for t7611 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t7611;
out.push({ let mut out = vec![]; for t7612 in py::iter(&(v_r.clone()))? { let mut v_v: Py; 
v_v = t7612;
out.push(f__F(v_v.clone())?); } Py::list(out) }); } Py::list(out) };
    v_mean = { let mut out = vec![]; for t7613 in py::range_iter(&[(v_d.clone()).i()?]) { let mut v_j: Py; 
v_j = t7613;
out.push(py::truediv(&py::sum(&{ let mut out = vec![]; for t7614 in py::iter(&(v_F.clone()))? { let mut v_r: Py; 
v_r = t7614;
out.push(py::getitem(&v_r.clone(), &v_j.clone())?); } Py::list(out) }, None)?, &v_n.clone())?); } Py::list(out) };
    v_var = { let mut out = vec![]; for t7615 in py::range_iter(&[(v_d.clone()).i()?]) { let mut v_j: Py; 
v_j = t7615;
out.push(py::truediv(&py::sum(&{ let mut out = vec![]; for t7616 in py::iter(&(v_F.clone()))? { let mut v_r: Py; 
v_r = t7616;
out.push(py::pow(&py::sub(&py::getitem(&v_r.clone(), &v_j.clone())?, &py::getitem(&v_mean.clone(), &v_j.clone())?)?, &Py::Int(2i64))?); } Py::list(out) }, None)?, &v_n.clone())?); } Py::list(out) };
    v_sd = { let mut out = vec![]; for t7617 in py::iter(&(v_var.clone()))? { let mut v_v: Py; 
v_v = t7617;
out.push(py::math1("sqrt", &v_v.clone())?); } Py::list(out) };
    v_Z = { let mut out = vec![]; for t7618 in py::iter(&(v_F.clone()))? { let mut v_r: Py; 
v_r = t7618;
out.push({ let mut out = vec![]; for t7619 in py::range_iter(&[(v_d.clone()).i()?]) { let mut v_j: Py; 
v_j = t7619;
out.push((if ({ let t7620 = py::getitem(&v_var.clone(), &v_j.clone())?; Py::Bool({ let t7621 = Py::Int(0i64); if py::gt(&t7620, &t7621)? { true } else { false } }) }).truthy() { py::truediv(&py::float(&py::sub(&py::getitem(&v_r.clone(), &v_j.clone())?, &py::getitem(&v_mean.clone(), &v_j.clone())?)?)?, &py::getitem(&v_sd.clone(), &v_j.clone())?)? } else { Py::Float(0.0) })); } Py::list(out) }); } Py::list(out) };
    return Ok(f__out(Py::dict(vec![(Py::str("means"), v_mean.clone()), (Py::str("stds"), v_sd.clone()), (Py::str("scaled"), v_Z.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_step_lr(mut v_lr0: Py, mut v_step_size: Py, mut v_gamma: Py, mut v_epoch: Py) -> R {
    let mut v_k: Py = Py::None;
    let _ = Py::tuple(vec![f__fin(Py::str("lr0"), Py::tuple(vec![v_lr0.clone()]))?, f__fin(Py::str("gamma"), Py::tuple(vec![v_gamma.clone()]))?]);
    let _ = f__need({ let t7671 = { let t7666 = v_lr0.clone(); Py::Bool({ let t7667 = Py::Int(0i64); if py::gt(&t7666, &t7667)? { true } else { false } }) }; if !t7671.truthy() { t7671 } else { { let t7668 = Py::Int(0i64); Py::Bool({ let t7669 = v_gamma.clone(); if py::lt(&t7668, &t7669)? { { let t7670 = Py::Int(1i64); if py::le(&t7669, &t7670)? { true } else { false } } } else { false } }) } } }, Py::str("need lr0 > 0 and 0 < gamma <= 1"))?;
    let _ = f__need({ let t7679 = Py::Bool(matches!(v_step_size.clone(), Py::Int(_) | Py::Big(_) | Py::Bool(_))); if !t7679.truthy() { t7679 } else { { let t7678 = { let t7672 = v_step_size.clone(); Py::Bool({ let t7673 = Py::Int(1i64); if py::ge(&t7672, &t7673)? { true } else { false } }) }; if !t7678.truthy() { t7678 } else { { let t7677 = Py::Bool(matches!(v_epoch.clone(), Py::Int(_) | Py::Big(_) | Py::Bool(_))); if !t7677.truthy() { t7677 } else { { let t7674 = Py::Int(0i64); Py::Bool({ let t7675 = v_epoch.clone(); if py::le(&t7674, &t7675)? { { let t7676 = Py::Int(1000i64); if py::le(&t7675, &t7676)? { true } else { false } } } else { false } }) } } } } } } }, Py::str("step_size >= 1 and 0 <= epoch <= 1000 (whole numbers)"))?;
    v_k = py::floordiv(&v_epoch.clone(), &v_step_size.clone())?;
    return Ok(f__out(Py::dict(vec![(Py::str("lr"), py::mul(&v_lr0.clone(), &py::pow(&v_gamma.clone(), &v_k.clone())?)?), (Py::str("decays"), v_k.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_stump_classify(mut v_X: Py, mut v_y: Py, mut v_criterion: Py) -> R {
    let mut v_L: Py = Py::None;
    let mut v_M: Py = Py::None;
    let mut v_R: Py = Py::None;
    let mut v_imp: Py = Py::None;
    let mut v_j: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_parent: Py = Py::None;
    let mut v_s: Py = Py::None;
    let mut v_t: Py = Py::None;
    let mut v_yl: Py = Py::None;
    v_M = f__stump_X(v_X.clone())?;
    v_yl = f__labels(Py::str("y"), v_y.clone(), Py::Int(1i64))?;
    let _ = f__need({ let t1869 = py::len(&v_yl.clone())?; Py::Bool({ let t1870 = py::len(&v_M.clone())?; if py::eq(&t1869, &t1870) { true } else { false } }) }, Py::str("y needs one label per row of X"))?;
    let _ = f__need({ let t1871 = py::len(&py::set(&v_yl.clone())?)?; Py::Bool({ let t1872 = Py::Int(2i64); if py::ge(&t1871, &t1872)? { true } else { false } }) }, Py::str("the node is pure: there is nothing to split"))?;
    v_imp = { let v_M = v_M.clone(); let v_X = v_X.clone(); let v_criterion = v_criterion.clone(); let v_imp = v_imp.clone(); let v_y = v_y.clone(); let v_yl = v_yl.clone(); py::func(move |a: &[Py]| -> R {
            #[allow(unused_mut, unused_variables, unused_assignments)]
{
            let mut v_idx: Py = a.get(0).cloned().unwrap_or(Py::None);
            let mut v_lab: Py = Py::None;
            let mut v_n: Py = Py::None;
            v_lab = { let mut out = vec![]; for t1873 in py::iter(&(v_idx.clone()))? { let mut v_i: Py; 
v_i = t1873;
out.push(py::getitem(&v_yl.clone(), &v_i.clone())?); } Py::list(out) };
            v_n = py::len(&v_lab.clone())?;
            if ({ let t1874 = v_criterion.clone(); Py::Bool({ let t1875 = Py::str("gini"); if py::eq(&t1874, &t1875) { true } else { false } }) }).truthy() {
                return Ok(py::sub(&Py::Int(1i64), &py::sum(&{ let mut out = vec![]; for t1876 in py::iter(&(py::set(&v_lab.clone())?))? { let mut v_c: Py; 
v_c = t1876;
out.push(py::pow(&py::fraction(&[py::method(&v_lab.clone(), "count", &[v_c.clone()])?, v_n.clone()])?, &Py::Int(2i64))?); } Py::list(out) }, None)?)?);
            }
            return Ok(py::truediv(&f__Hc({ let mut out = vec![]; for t1883 in py::iter(&(py::set(&v_lab.clone())?))? { let mut v_c: Py; 
v_c = t1883;
out.push(py::method(&v_lab.clone(), "count", &[v_c.clone()])?); } Py::list(out) })?, &py::math1("log", &Py::Float(2.0))?)?);
            #[allow(unreachable_code)]
            Ok(Py::None)
        }
        }) };
    v_n = py::len(&v_yl.clone())?;
    v_parent = py::call(&v_imp.clone(), &[Py::list((py::range_iter(&[(v_n.clone()).i()?])).into_iter().collect::<Vec<Py>>())])?;
    let t1905 = (f__stump_scan(v_M.clone(), { let v_M = v_M.clone(); let v_X = v_X.clone(); let v_criterion = v_criterion.clone(); let v_imp = v_imp.clone(); let v_n = v_n.clone(); let v_parent = v_parent.clone(); let v_y = v_y.clone(); let v_yl = v_yl.clone(); py::func(move |a: &[Py]| -> R {
            #[allow(unused_mut, unused_variables, unused_assignments)]
{
            let mut v_l: Py = a.get(0).cloned().unwrap_or(Py::None);
            let mut v_r: Py = a.get(1).cloned().unwrap_or(Py::None);
            return Ok((if ({ let t1903 = v_criterion.clone(); Py::Bool({ let t1904 = Py::str("gini"); if py::eq(&t1903, &t1904) { true } else { false } }) }).truthy() { py::sub(&v_parent.clone(), &py::add(&py::mul(&py::fraction(&[py::len(&v_l.clone())?, v_n.clone()])?, &py::call(&v_imp.clone(), &[v_l.clone()])?)?, &py::mul(&py::fraction(&[py::len(&v_r.clone())?, v_n.clone()])?, &py::call(&v_imp.clone(), &[v_r.clone()])?)?)?)? } else { py::sub(&v_parent.clone(), &py::add(&py::mul(&py::truediv(&py::len(&v_l.clone())?, &v_n.clone())?, &py::call(&v_imp.clone(), &[v_l.clone()])?)?, &py::mul(&py::truediv(&py::len(&v_r.clone())?, &v_n.clone())?, &py::call(&v_imp.clone(), &[v_r.clone()])?)?)?)? }));
            #[allow(unreachable_code)]
            Ok(Py::None)
        }
        }) })?).items()?;
    if t1905.len() != 5 { return Err(PyErr::Value("unpack".into())); }
    v_s = t1905[0].clone();
    v_j = t1905[1].clone();
    v_t = t1905[2].clone();
    v_L = t1905[3].clone();
    v_R = t1905[4].clone();
    return Ok(f__out(Py::dict(vec![(Py::str("feature"), v_j.clone()), (Py::str("threshold"), v_t.clone()), (Py::str("impurity_decrease"), v_s.clone()), (Py::str("n_left"), py::len(&v_L.clone())?), (Py::str("n_right"), py::len(&v_R.clone())?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_stump_regress(mut v_X: Py, mut v_y: Py) -> R {
    let mut v_L: Py = Py::None;
    let mut v_M: Py = Py::None;
    let mut v_R: Py = Py::None;
    let mut v_Y: Py = Py::None;
    let mut v_j: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_parent: Py = Py::None;
    let mut v_s: Py = Py::None;
    let mut v_sse: Py = Py::None;
    let mut v_t: Py = Py::None;
    let mut v_yv: Py = Py::None;
    v_M = f__stump_X(v_X.clone())?;
    v_yv = f__vec(Py::str("y"), v_y.clone(), Py::Int(1i64))?;
    let _ = f__need({ let t1976 = py::len(&v_yv.clone())?; Py::Bool({ let t1977 = py::len(&v_M.clone())?; if py::eq(&t1976, &t1977) { true } else { false } }) }, Py::str("y needs one value per row of X"))?;
    v_Y = { let mut out = vec![]; for t1978 in py::iter(&(v_yv.clone()))? { let mut v_v: Py; 
v_v = t1978;
out.push(f__F(v_v.clone())?); } Py::list(out) };
    v_n = py::len(&v_Y.clone())?;
    v_sse = { let v_M = v_M.clone(); let v_X = v_X.clone(); let v_Y = v_Y.clone(); let v_n = v_n.clone(); let v_sse = v_sse.clone(); let v_y = v_y.clone(); let v_yv = v_yv.clone(); py::func(move |a: &[Py]| -> R {
            #[allow(unused_mut, unused_variables, unused_assignments)]
{
            let mut v_idx: Py = a.get(0).cloned().unwrap_or(Py::None);
            let mut v_m: Py = Py::None;
            let mut v_vals: Py = Py::None;
            v_vals = { let mut out = vec![]; for t1979 in py::iter(&(v_idx.clone()))? { let mut v_i: Py; 
v_i = t1979;
out.push(py::getitem(&v_Y.clone(), &v_i.clone())?); } Py::list(out) };
            v_m = py::truediv(&py::sum(&v_vals.clone(), None)?, &py::len(&v_vals.clone())?)?;
            return Ok(py::sum(&{ let mut out = vec![]; for t1980 in py::iter(&(v_vals.clone()))? { let mut v_v: Py; 
v_v = t1980;
out.push(py::pow(&py::sub(&v_v.clone(), &v_m.clone())?, &Py::Int(2i64))?); } Py::list(out) }, None)?);
            #[allow(unreachable_code)]
            Ok(Py::None)
        }
        }) };
    v_parent = py::call(&v_sse.clone(), &[Py::list((py::range_iter(&[(v_n.clone()).i()?])).into_iter().collect::<Vec<Py>>())])?;
    let _ = f__need({ let t1981 = v_parent.clone(); Py::Bool({ let t1982 = Py::Int(0i64); if py::gt(&t1981, &t1982)? { true } else { false } }) }, Py::str("every target is the same: there is nothing to split"))?;
    let t2002 = (f__stump_scan(v_M.clone(), { let v_M = v_M.clone(); let v_X = v_X.clone(); let v_Y = v_Y.clone(); let v_n = v_n.clone(); let v_parent = v_parent.clone(); let v_sse = v_sse.clone(); let v_y = v_y.clone(); let v_yv = v_yv.clone(); py::func(move |a: &[Py]| -> R {
            #[allow(unused_mut, unused_variables, unused_assignments)]
{
            let mut v_l: Py = a.get(0).cloned().unwrap_or(Py::None);
            let mut v_r: Py = a.get(1).cloned().unwrap_or(Py::None);
            return Ok(py::truediv(&py::sub(&py::sub(&v_parent.clone(), &py::call(&v_sse.clone(), &[v_l.clone()])?)?, &py::call(&v_sse.clone(), &[v_r.clone()])?)?, &v_n.clone())?);
            #[allow(unreachable_code)]
            Ok(Py::None)
        }
        }) })?).items()?;
    if t2002.len() != 5 { return Err(PyErr::Value("unpack".into())); }
    v_s = t2002[0].clone();
    v_j = t2002[1].clone();
    v_t = t2002[2].clone();
    v_L = t2002[3].clone();
    v_R = t2002[4].clone();
    return Ok(f__out(Py::dict(vec![(Py::str("feature"), v_j.clone()), (Py::str("threshold"), v_t.clone()), (Py::str("mse_decrease"), v_s.clone()), (Py::str("left_mean"), py::truediv(&py::sum(&{ let mut out = vec![]; for t2015 in py::iter(&(v_L.clone()))? { let mut v_i: Py; 
v_i = t2015;
out.push(py::getitem(&v_Y.clone(), &v_i.clone())?); } Py::list(out) }, None)?, &py::len(&v_L.clone())?)?), (Py::str("right_mean"), py::truediv(&py::sum(&{ let mut out = vec![]; for t2016 in py::iter(&(v_R.clone()))? { let mut v_i: Py; 
v_i = t2016;
out.push(py::getitem(&v_Y.clone(), &v_i.clone())?); } Py::list(out) }, None)?, &py::len(&v_R.clone())?)?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_svm_decision(mut v_w: Py, mut v_b: Py, mut v_X: Py) -> R {
    let mut v_M: Py = Py::None;
    let mut v_f: Py = Py::None;
    let mut v_nw: Py = Py::None;
    let mut v_wv: Py = Py::None;
    v_wv = f__vec(Py::str("w"), v_w.clone(), Py::Int(1i64))?;
    let _ = f__fin(Py::str("b"), Py::tuple(vec![v_b.clone()]))?;
    v_M = f__mat(Py::str("X"), v_X.clone(), Py::Int(1i64), Py::Int(1i64))?;
    let _ = f__need({ let t4239 = py::len(&py::getitem(&v_M.clone(), &Py::Int(0i64))?)?; Py::Bool({ let t4240 = py::len(&v_wv.clone())?; if py::eq(&t4239, &t4240) { true } else { false } }) }, Py::str("w needs one weight per column of X"))?;
    v_nw = py::math1("sqrt", &py::fsum(&{ let mut out = vec![]; for t4241 in py::iter(&(v_wv.clone()))? { let mut v_v: Py; 
v_v = t4241;
out.push(py::mul(&v_v.clone(), &v_v.clone())?); } Py::list(out) })?)?;
    let _ = f__need({ let t4242 = v_nw.clone(); Py::Bool({ let t4243 = Py::Int(0i64); if py::gt(&t4242, &t4243)? { true } else { false } }) }, Py::str("w must not be the zero vector"))?;
    v_f = { let mut out = vec![]; for t4244 in py::iter(&(v_M.clone()))? { let mut v_r: Py; 
v_r = t4244;
out.push(py::add(&py::sum(&{ let mut out = vec![]; for t4245 in py::iter(&(py::zip(&[v_r.clone(), v_wv.clone()])?))? { let mut v_a: Py; let mut v_c: Py; 
let t4246 = (t4245).items()?;
if t4246.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_a = t4246[0].clone();
v_c = t4246[1].clone();
out.push(py::mul(&f__F(v_a.clone())?, &f__F(v_c.clone())?)?); } Py::list(out) }, None)?, &f__F(v_b.clone())?)?); } Py::list(out) };
    return Ok(f__out(Py::dict(vec![(Py::str("decision"), v_f.clone()), (Py::str("classes"), { let mut out = vec![]; for t4259 in py::iter(&(v_f.clone()))? { let mut v_v: Py; 
v_v = t4259;
out.push((if ({ let t4260 = v_v.clone(); Py::Bool({ let t4261 = Py::Int(0i64); if py::gt(&t4260, &t4261)? { true } else { false } }) }).truthy() { Py::Int(1i64) } else { py::neg(&Py::Int(1i64))? })); } Py::list(out) }), (Py::str("margin_width"), py::truediv(&Py::Int(2i64), &v_nw.clone())?), (Py::str("distances"), { let mut out = vec![]; for t4262 in py::iter(&(v_f.clone()))? { let mut v_v: Py; 
v_v = t4262;
out.push(py::truediv(&py::abs(&py::float(&v_v.clone())?)?, &v_nw.clone())?); } Py::list(out) })]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_threshold_metrics(mut v_y_true: Py, mut v_scores: Py, mut v_threshold: Py) -> R {
    let mut v_fn: Py = Py::None;
    let mut v_fp: Py = Py::None;
    let mut v_s: Py = Py::None;
    let mut v_tn: Py = Py::None;
    let mut v_tp: Py = Py::None;
    let mut v_yp: Py = Py::None;
    let mut v_yt: Py = Py::None;
    v_yt = f__binary(Py::str("y_true"), v_y_true.clone(), Py::Int(1i64))?;
    v_s = f__vec(Py::str("scores"), v_scores.clone(), Py::Int(1i64))?;
    let _ = f__same_len(v_yt.clone(), v_s.clone(), Py::str("y_true"), Py::str("scores"))?;
    let _ = f__fin(Py::str("threshold"), Py::tuple(vec![v_threshold.clone()]))?;
    v_yp = { let mut out = vec![]; for t7740 in py::iter(&(v_s.clone()))? { let mut v_v: Py; 
v_v = t7740;
out.push((if ({ let t7741 = v_v.clone(); Py::Bool({ let t7742 = v_threshold.clone(); if py::ge(&t7741, &t7742)? { true } else { false } }) }).truthy() { Py::Int(1i64) } else { Py::Int(0i64) })); } Py::list(out) };
    let t7771 = (f__counts(v_yt.clone(), v_yp.clone())?).items()?;
    if t7771.len() != 4 { return Err(PyErr::Value("unpack".into())); }
    v_tp = t7771[0].clone();
    v_fp = t7771[1].clone();
    v_fn = t7771[2].clone();
    v_tn = t7771[3].clone();
    let _ = f__need({ let t7772 = py::add(&v_tp.clone(), &v_fp.clone())?; Py::Bool({ let t7773 = Py::Int(0i64); if py::gt(&t7772, &t7773)? { true } else { false } }) }, Py::str("no sample reaches the threshold: precision is undefined"))?;
    let _ = f__need({ let t7774 = py::add(&v_tp.clone(), &v_fn.clone())?; Py::Bool({ let t7775 = Py::Int(0i64); if py::gt(&t7774, &t7775)? { true } else { false } }) }, Py::str("there are no actual positives: recall is undefined"))?;
    return Ok(f__out(Py::dict(vec![(Py::str("TP"), v_tp.clone()), (Py::str("FP"), v_fp.clone()), (Py::str("FN"), v_fn.clone()), (Py::str("TN"), v_tn.clone()), (Py::str("precision"), py::fraction(&[v_tp.clone(), py::add(&v_tp.clone(), &v_fp.clone())?])?), (Py::str("recall"), py::fraction(&[v_tp.clone(), py::add(&v_tp.clone(), &v_fn.clone())?])?), (Py::str("accuracy"), py::fraction(&[py::add(&v_tp.clone(), &v_tn.clone())?, py::len(&v_yt.clone())?])?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_top_k_accuracy(mut v_y_true: Py, mut v_S: Py, mut v_k: Py) -> R {
    let mut v_K: Py = Py::None;
    let mut v_M: Py = Py::None;
    let mut v_hits: Py = Py::None;
    let mut v_rank: Py = Py::None;
    let mut v_row: Py = Py::None;
    let mut v_y: Py = Py::None;
    let mut v_yt: Py = Py::None;
    v_yt = f__labels(Py::str("y_true"), v_y_true.clone(), Py::Int(1i64))?;
    v_M = f__mat(Py::str("S"), v_S.clone(), Py::Int(1i64), Py::Int(3i64))?;
    let _ = f__same_len(v_yt.clone(), v_M.clone(), Py::str("y_true"), Py::str("the rows of S"))?;
    v_K = py::len(&py::getitem(&v_M.clone(), &Py::Int(0i64))?)?;
    let _ = f__need({ let t7842 = Py::Int(1i64); Py::Bool({ let t7843 = v_k.clone(); if py::le(&t7842, &t7843)? { { let t7844 = v_K.clone(); if py::lt(&t7843, &t7844)? { true } else { false } } } else { false } }) }, Py::str("k must be at least 1 and less than the number of classes"))?;
    v_hits = Py::Int(0i64);
    for t7845 in py::iter(&(py::zip(&[v_yt.clone(), v_M.clone()])?))? {
        let t7856 = (t7845).items()?;
        if t7856.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_y = t7856[0].clone();
        v_row = t7856[1].clone();
        let _ = f__need({ let t7846 = Py::Int(0i64); Py::Bool({ let t7847 = v_y.clone(); if py::le(&t7846, &t7847)? { { let t7848 = v_K.clone(); if py::lt(&t7847, &t7848)? { true } else { false } } } else { false } }) }, Py::str("every label must be a column index of S (0 to K-1)"))?;
        let _ = f__need({ let t7849 = py::len(&py::set(&v_row.clone())?)?; Py::Bool({ let t7850 = v_K.clone(); if py::eq(&t7849, &t7850) { true } else { false } }) }, Py::str("scores within a row must be distinct (a tie makes the top k ambiguous)"))?;
        v_rank = py::sum(&{ let mut out = vec![]; for t7851 in py::iter(&(v_row.clone()))? { let mut v_v: Py; 
v_v = t7851;
if ({ let t7852 = v_v.clone(); Py::Bool({ let t7853 = py::getitem(&v_row.clone(), &v_y.clone())?; if py::gt(&t7852, &t7853)? { true } else { false } }) }).truthy() { out.push(Py::Int(1i64)); }} Py::list(out) }, None)?;
        v_hits = py::iadd(&v_hits, &{ let t7854 = v_rank.clone(); Py::Bool({ let t7855 = v_k.clone(); if py::lt(&t7854, &t7855)? { true } else { false } }) })?;
    }
    return Ok(f__out(Py::dict(vec![(Py::str("top_k_accuracy"), py::fraction(&[v_hits.clone(), py::len(&v_yt.clone())?])?), (Py::str("n_hits"), v_hits.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_transformer_params(mut v_d_model: Py, mut v_n_heads: Py, mut v_d_ff: Py, mut v_n_layers: Py) -> R {
    let mut v_attn: Py = Py::None;
    let mut v_ff: Py = Py::None;
    let mut v_norm: Py = Py::None;
    let mut v_per: Py = Py::None;
    let _ = f__need(Py::Bool(({ let mut out = vec![]; for t7950 in py::iter(&(Py::tuple(vec![v_d_model.clone(), v_n_heads.clone(), v_d_ff.clone(), v_n_layers.clone()])))? { let mut v_v: Py; 
v_v = t7950;
out.push({ let t7953 = Py::Bool(matches!(v_v.clone(), Py::Int(_) | Py::Big(_) | Py::Bool(_))); if !t7953.truthy() { t7953 } else { { let t7951 = v_v.clone(); Py::Bool({ let t7952 = Py::Int(1i64); if py::ge(&t7951, &t7952)? { true } else { false } }) } } }); } Py::list(out) }).items()?.iter().all(|x| x.truthy())), Py::str("all sizes must be whole numbers >= 1"))?;
    let _ = f__need({ let t7954 = py::modulo(&v_d_model.clone(), &v_n_heads.clone())?; Py::Bool({ let t7955 = Py::Int(0i64); if py::eq(&t7954, &t7955) { true } else { false } }) }, Py::str("d_model must be divisible by n_heads"))?;
    let _ = f__need({ let t7963 = { let t7956 = v_d_model.clone(); Py::Bool({ let t7957 = Py::Int(8192i64); if py::le(&t7956, &t7957)? { true } else { false } }) }; if !t7963.truthy() { t7963 } else { { let t7962 = { let t7958 = v_d_ff.clone(); Py::Bool({ let t7959 = Py::Int(32768i64); if py::le(&t7958, &t7959)? { true } else { false } }) }; if !t7962.truthy() { t7962 } else { { let t7960 = v_n_layers.clone(); Py::Bool({ let t7961 = Py::Int(200i64); if py::le(&t7960, &t7961)? { true } else { false } }) } } } } }, Py::str("sizes too large"))?;
    v_attn = py::add(&py::mul(&py::mul(&Py::Int(4i64), &v_d_model.clone())?, &v_d_model.clone())?, &py::mul(&Py::Int(4i64), &v_d_model.clone())?)?;
    v_ff = py::add(&py::add(&py::mul(&py::mul(&Py::Int(2i64), &v_d_model.clone())?, &v_d_ff.clone())?, &v_d_ff.clone())?, &v_d_model.clone())?;
    v_norm = py::mul(&Py::Int(4i64), &v_d_model.clone())?;
    v_per = py::add(&py::add(&v_attn.clone(), &v_ff.clone())?, &v_norm.clone())?;
    return Ok(f__out(Py::dict(vec![(Py::str("attention"), v_attn.clone()), (Py::str("feed_forward"), v_ff.clone()), (Py::str("layer_norms"), v_norm.clone()), (Py::str("per_layer"), v_per.clone()), (Py::str("total"), py::mul(&v_per.clone(), &v_n_layers.clone())?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_triplet(mut v_anchor: Py, mut v_positive: Py, mut v_negative: Py, mut v_margin: Py) -> R {
    let mut v_a: Py = Py::None;
    let mut v_dan: Py = Py::None;
    let mut v_dap: Py = Py::None;
    let mut v_n: Py = Py::None;
    let mut v_p: Py = Py::None;
    let mut v_v: Py = Py::None;
    let t8013 = (Py::tuple(vec![f__vec(Py::str("anchor"), v_anchor.clone(), Py::Int(1i64))?, f__vec(Py::str("positive"), v_positive.clone(), Py::Int(1i64))?, f__vec(Py::str("negative"), v_negative.clone(), Py::Int(1i64))?])).items()?;
    if t8013.len() != 3 { return Err(PyErr::Value("unpack".into())); }
    v_a = t8013[0].clone();
    v_p = t8013[1].clone();
    v_n = t8013[2].clone();
    let _ = f__need({ let t8014 = py::len(&v_a.clone())?; Py::Bool({ let t8015 = py::len(&v_p.clone())?; if py::eq(&t8014, &t8015) { { let t8016 = py::len(&v_n.clone())?; if py::eq(&t8015, &t8016) { true } else { false } } } else { false } }) }, Py::str("anchor, positive and negative must have the same length"))?;
    let _ = f__fin(Py::str("margin"), Py::tuple(vec![v_margin.clone()]))?;
    let _ = f__need({ let t8017 = v_margin.clone(); Py::Bool({ let t8018 = Py::Int(0i64); if py::ge(&t8017, &t8018)? { true } else { false } }) }, Py::str("the margin must be >= 0"))?;
    v_dap = py::math1("sqrt", &py::fsum(&{ let mut out = vec![]; for t8019 in py::iter(&(py::zip(&[v_a.clone(), v_p.clone()])?))? { let mut v_x: Py; let mut v_y: Py; 
let t8020 = (t8019).items()?;
if t8020.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_x = t8020[0].clone();
v_y = t8020[1].clone();
out.push(py::pow(&py::sub(&v_x.clone(), &v_y.clone())?, &Py::Int(2i64))?); } Py::list(out) })?)?;
    v_dan = py::math1("sqrt", &py::fsum(&{ let mut out = vec![]; for t8021 in py::iter(&(py::zip(&[v_a.clone(), v_n.clone()])?))? { let mut v_x: Py; let mut v_y: Py; 
let t8022 = (t8021).items()?;
if t8022.len() != 2 { return Err(PyErr::Value("unpack".into())); }
v_x = t8022[0].clone();
v_y = t8022[1].clone();
out.push(py::pow(&py::sub(&v_x.clone(), &v_y.clone())?, &Py::Int(2i64))?); } Py::list(out) })?)?;
    v_v = py::add(&py::sub(&v_dap.clone(), &v_dan.clone())?, &v_margin.clone())?;
    return Ok(f__out(Py::dict(vec![(Py::str("loss"), py::minmax(&[Py::Float(0.0), v_v.clone()], true)?), (Py::str("d_ap"), v_dap.clone()), (Py::str("d_an"), v_dan.clone())]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_tweedie(mut v_y_true: Py, mut v_y_pred: Py, mut v_power: Py) -> R {
    let mut v_dev: Py = Py::None;
    let mut v_m: Py = Py::None;
    let mut v_p: Py = Py::None;
    let mut v_y: Py = Py::None;
    let mut v_yp: Py = Py::None;
    let mut v_yt: Py = Py::None;
    let t8075 = (f__pair(v_y_true.clone(), v_y_pred.clone(), Py::Int(1i64))?).items()?;
    if t8075.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_yt = t8075[0].clone();
    v_yp = t8075[1].clone();
    let _ = f__fin(Py::str("power"), Py::tuple(vec![v_power.clone()]))?;
    let _ = f__need({ let t8080 = { let t8076 = v_power.clone(); Py::Bool({ let t8077 = Py::Int(0i64); if py::eq(&t8076, &t8077) { true } else { false } }) }; if t8080.truthy() { t8080 } else { { let t8078 = v_power.clone(); Py::Bool({ let t8079 = Py::Int(1i64); if py::ge(&t8078, &t8079)? { true } else { false } }) } } }, Py::str("power must be 0 (normal) or at least 1 (1 Poisson, 2 gamma, 3 inverse Gaussian)"))?;
    if ({ let t8081 = v_power.clone(); Py::Bool({ let t8082 = Py::Int(0i64); if py::gt(&t8081, &t8082)? { true } else { false } }) }).truthy() {
        let _ = f__need(Py::Bool(({ let mut out = vec![]; for t8083 in py::iter(&(v_yp.clone()))? { let mut v_m: Py; 
v_m = t8083;
out.push({ let t8084 = v_m.clone(); Py::Bool({ let t8085 = Py::Int(0i64); if py::gt(&t8084, &t8085)? { true } else { false } }) }); } Py::list(out) }).items()?.iter().all(|x| x.truthy())), Py::str("predictions must be positive for power > 0"))?;
    }
    if ({ let t8086 = Py::Int(1i64); Py::Bool({ let t8087 = v_power.clone(); if py::le(&t8086, &t8087)? { { let t8088 = Py::Int(2i64); if py::lt(&t8087, &t8088)? { true } else { false } } } else { false } }) }).truthy() {
        let _ = f__need(Py::Bool(({ let mut out = vec![]; for t8089 in py::iter(&(v_yt.clone()))? { let mut v_y: Py; 
v_y = t8089;
out.push({ let t8090 = v_y.clone(); Py::Bool({ let t8091 = Py::Int(0i64); if py::ge(&t8090, &t8091)? { true } else { false } }) }); } Py::list(out) }).items()?.iter().all(|x| x.truthy())), Py::str("true values must be non-negative for 1 <= power < 2"))?;
    }
    if ({ let t8092 = v_power.clone(); Py::Bool({ let t8093 = Py::Int(2i64); if py::ge(&t8092, &t8093)? { true } else { false } }) }).truthy() {
        let _ = f__need(Py::Bool(({ let mut out = vec![]; for t8094 in py::iter(&(v_yt.clone()))? { let mut v_y: Py; 
v_y = t8094;
out.push({ let t8095 = v_y.clone(); Py::Bool({ let t8096 = Py::Int(0i64); if py::gt(&t8095, &t8096)? { true } else { false } }) }); } Py::list(out) }).items()?.iter().all(|x| x.truthy())), Py::str("true values must be positive for power >= 2"))?;
    }
    v_p = v_power.clone();
    v_dev = Py::list(vec![]);
    for t8097 in py::iter(&(py::zip(&[v_yt.clone(), v_yp.clone()])?))? {
        let t8106 = (t8097).items()?;
        if t8106.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_y = t8106[0].clone();
        v_m = t8106[1].clone();
        if ({ let t8098 = v_p.clone(); Py::Bool({ let t8099 = Py::Int(0i64); if py::eq(&t8098, &t8099) { true } else { false } }) }).truthy() {
            let _ = py::method(&v_dev.clone(), "append", &[py::pow(&py::sub(&v_y.clone(), &v_m.clone())?, &Py::Int(2i64))?])?;
        } else {
            if ({ let t8100 = v_p.clone(); Py::Bool({ let t8101 = Py::Int(1i64); if py::eq(&t8100, &t8101) { true } else { false } }) }).truthy() {
                let _ = py::method(&v_dev.clone(), "append", &[py::mul(&Py::Int(2i64), &py::add(&py::sub(&(if ({ let t8102 = v_y.clone(); Py::Bool({ let t8103 = Py::Int(0i64); if py::gt(&t8102, &t8103)? { true } else { false } }) }).truthy() { py::mul(&v_y.clone(), &py::math1("log", &py::truediv(&v_y.clone(), &v_m.clone())?)?)? } else { Py::Float(0.0) }), &v_y.clone())?, &v_m.clone())?)?])?;
            } else {
                if ({ let t8104 = v_p.clone(); Py::Bool({ let t8105 = Py::Int(2i64); if py::eq(&t8104, &t8105) { true } else { false } }) }).truthy() {
                    let _ = py::method(&v_dev.clone(), "append", &[py::mul(&Py::Int(2i64), &py::sub(&py::add(&py::math1("log", &py::truediv(&v_m.clone(), &v_y.clone())?)?, &py::truediv(&v_y.clone(), &v_m.clone())?)?, &Py::Int(1i64))?)?])?;
                } else {
                    let _ = py::method(&v_dev.clone(), "append", &[py::mul(&Py::Int(2i64), &py::add(&py::sub(&py::truediv(&py::pow(&py::minmax(&[v_y.clone(), Py::Float(0.0)], true)?, &py::sub(&Py::Int(2i64), &v_p.clone())?)?, &py::mul(&py::sub(&Py::Int(1i64), &v_p.clone())?, &py::sub(&Py::Int(2i64), &v_p.clone())?)?)?, &py::truediv(&py::mul(&v_y.clone(), &py::pow(&v_m.clone(), &py::sub(&Py::Int(1i64), &v_p.clone())?)?)?, &py::sub(&Py::Int(1i64), &v_p.clone())?)?)?, &py::truediv(&py::pow(&v_m.clone(), &py::sub(&Py::Int(2i64), &v_p.clone())?)?, &py::sub(&Py::Int(2i64), &v_p.clone())?)?)?)?])?;
                }
            }
        }
    }
    return Ok(f__out(Py::dict(vec![(Py::str("deviance"), py::truediv(&py::fsum(&v_dev.clone())?, &py::len(&v_dev.clone())?)?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn f_warmup_cosine(mut v_lr_max: Py, mut v_start_factor: Py, mut v_warmup: Py, mut v_total: Py, mut v_eta_min: Py, mut v_step: Py) -> R {
    let mut v_T: Py = Py::None;
    let mut v_j: Py = Py::None;
    let _ = Py::tuple(vec![f__fin(Py::str("lr_max"), Py::tuple(vec![v_lr_max.clone()]))?, f__fin(Py::str("start_factor"), Py::tuple(vec![v_start_factor.clone()]))?, f__fin(Py::str("eta_min"), Py::tuple(vec![v_eta_min.clone()]))?]);
    let _ = f__need({ let t8160 = { let t8151 = v_lr_max.clone(); Py::Bool({ let t8152 = Py::Int(0i64); if py::gt(&t8151, &t8152)? { true } else { false } }) }; if !t8160.truthy() { t8160 } else { { let t8159 = { let t8153 = Py::Int(0i64); Py::Bool({ let t8154 = v_start_factor.clone(); if py::lt(&t8153, &t8154)? { { let t8155 = Py::Int(1i64); if py::le(&t8154, &t8155)? { true } else { false } } } else { false } }) }; if !t8159.truthy() { t8159 } else { { let t8156 = Py::Int(0i64); Py::Bool({ let t8157 = v_eta_min.clone(); if py::le(&t8156, &t8157)? { { let t8158 = v_lr_max.clone(); if py::lt(&t8157, &t8158)? { true } else { false } } } else { false } }) } } } } }, Py::str("need lr_max > 0, 0 < start_factor <= 1, 0 <= eta_min < lr_max"))?;
    let _ = f__need({ let t8170 = Py::Bool(({ let mut out = vec![]; for t8161 in py::iter(&(Py::tuple(vec![v_warmup.clone(), v_total.clone(), v_step.clone()])))? { let mut v_v: Py; 
v_v = t8161;
out.push(Py::Bool(matches!(v_v.clone(), Py::Int(_) | Py::Big(_) | Py::Bool(_)))); } Py::list(out) }).items()?.iter().all(|x| x.truthy())); if !t8170.truthy() { t8170 } else { { let t8169 = { let t8162 = Py::Int(1i64); Py::Bool({ let t8163 = v_warmup.clone(); if py::le(&t8162, &t8163)? { { let t8164 = v_total.clone(); if py::lt(&t8163, &t8164)? { { let t8165 = Py::Int(1000i64); if py::le(&t8164, &t8165)? { true } else { false } } } else { false } } } else { false } }) }; if !t8169.truthy() { t8169 } else { { let t8166 = Py::Int(0i64); Py::Bool({ let t8167 = v_step.clone(); if py::le(&t8166, &t8167)? { { let t8168 = v_total.clone(); if py::le(&t8167, &t8168)? { true } else { false } } } else { false } }) } } } } }, Py::str("need whole numbers 1 <= warmup < total <= 1000 and 0 <= step <= total"))?;
    if ({ let t8171 = v_step.clone(); Py::Bool({ let t8172 = v_warmup.clone(); if py::lt(&t8171, &t8172)? { true } else { false } }) }).truthy() {
        return Ok(f__out(Py::dict(vec![(Py::str("lr"), py::mul(&v_lr_max.clone(), &py::add(&v_start_factor.clone(), &py::truediv(&py::mul(&py::sub(&Py::Int(1i64), &v_start_factor.clone())?, &v_step.clone())?, &v_warmup.clone())?)?)?)]))?);
    }
    let t8185 = (Py::tuple(vec![py::sub(&v_step.clone(), &v_warmup.clone())?, py::sub(&v_total.clone(), &v_warmup.clone())?])).items()?;
    if t8185.len() != 2 { return Err(PyErr::Value("unpack".into())); }
    v_j = t8185[0].clone();
    v_T = t8185[1].clone();
    if ({ let t8186 = v_j.clone(); Py::Bool({ let t8187 = v_T.clone(); if py::eq(&t8186, &t8187) { true } else { false } }) }).truthy() {
        return Ok(f__out(Py::dict(vec![(Py::str("lr"), v_eta_min.clone())]))?);
    }
    return Ok(f__out(Py::dict(vec![(Py::str("lr"), py::add(&v_eta_min.clone(), &py::truediv(&py::mul(&py::sub(&v_lr_max.clone(), &v_eta_min.clone())?, &py::add(&Py::Int(1i64), &py::math1("cos", &py::truediv(&py::mul(&Py::Float(std::f64::consts::PI), &v_j.clone())?, &v_T.clone())?)?)?)?, &Py::Int(2i64))?)?)]))?);
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_1062(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t1064: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t1063 = v_inputs.clone(); f_binary_confusion(py::getitem(&t1063, &Py::str("y_true"))?, py::getitem(&t1063, &Py::str("y_pred"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t1064 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t1065 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t1090 = (t1065).items()?;
        if t1090.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t1090[0].clone();
        v_v = t1090[1].clone();
        if ({ let t1066 = v_k.clone(); Py::Bool({ let t1067 = v_outputs.clone(); if !py::contains(&t1067, &t1066)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_1117(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t1119: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t1118 = v_inputs.clone(); f_binary_from_counts(py::getitem(&t1118, &Py::str("TP"))?, py::getitem(&t1118, &Py::str("FP"))?, py::getitem(&t1118, &Py::str("FN"))?, py::getitem(&t1118, &Py::str("TN"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t1119 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t1120 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t1145 = (t1120).items()?;
        if t1145.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t1145[0].clone();
        v_v = t1145[1].clone();
        if ({ let t1121 = v_k.clone(); Py::Bool({ let t1122 = v_outputs.clone(); if !py::contains(&t1122, &t1121)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_1183(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t1185: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t1184 = v_inputs.clone(); f_brier(py::getitem(&t1184, &Py::str("y_true"))?, py::getitem(&t1184, &Py::str("p"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t1185 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t1186 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t1211 = (t1186).items()?;
        if t1211.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t1211[0].clone();
        v_v = t1211[1].clone();
        if ({ let t1187 = v_k.clone(); Py::Bool({ let t1188 = v_outputs.clone(); if !py::contains(&t1188, &t1187)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_1269(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t1271: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t1270 = v_inputs.clone(); f_calinski_harabasz(py::getitem(&t1270, &Py::str("X"))?, py::getitem(&t1270, &Py::str("labels"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t1271 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t1272 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t1297 = (t1272).items()?;
        if t1297.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t1297[0].clone();
        v_v = t1297[1].clone();
        if ({ let t1273 = v_k.clone(); Py::Bool({ let t1274 = v_outputs.clone(); if !py::contains(&t1274, &t1273)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_1339(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t1341: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t1340 = v_inputs.clone(); f_per_class_report(py::getitem(&t1340, &Py::str("y_true"))?, py::getitem(&t1340, &Py::str("y_pred"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t1341 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t1342 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t1367 = (t1342).items()?;
        if t1367.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t1367[0].clone();
        v_v = t1367[1].clone();
        if ({ let t1343 = v_k.clone(); Py::Bool({ let t1344 = v_outputs.clone(); if !py::contains(&t1344, &t1343)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_1417(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t1419: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t1418 = v_inputs.clone(); f_cohen_kappa(py::getitem(&t1418, &Py::str("y1"))?, py::getitem(&t1418, &Py::str("y2"))?, py::getitem(&t1418, &Py::str("weights"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t1419 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t1420 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t1445 = (t1420).items()?;
        if t1445.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t1445[0].clone();
        v_v = t1445[1].clone();
        if ({ let t1421 = v_k.clone(); Py::Bool({ let t1422 = v_outputs.clone(); if !py::contains(&t1422, &t1421)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_1501(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t1503: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t1502 = v_inputs.clone(); f_conv2d(py::getitem(&t1502, &Py::str("image"))?, py::getitem(&t1502, &Py::str("kernel"))?, py::getitem(&t1502, &Py::str("stride"))?, py::getitem(&t1502, &Py::str("padding"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t1503 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t1504 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t1529 = (t1504).items()?;
        if t1529.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t1529[0].clone();
        v_v = t1529[1].clone();
        if ({ let t1505 = v_k.clone(); Py::Bool({ let t1506 = v_outputs.clone(); if !py::contains(&t1506, &t1505)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_1549(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t1551: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t1550 = v_inputs.clone(); f_conv_out(py::getitem(&t1550, &Py::str("n"))?, py::getitem(&t1550, &Py::str("kernel"))?, py::getitem(&t1550, &Py::str("stride"))?, py::getitem(&t1550, &Py::str("padding"))?, py::getitem(&t1550, &Py::str("dilation"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t1551 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t1552 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t1577 = (t1552).items()?;
        if t1577.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t1577[0].clone();
        v_v = t1577[1].clone();
        if ({ let t1553 = v_k.clone(); Py::Bool({ let t1554 = v_outputs.clone(); if !py::contains(&t1554, &t1553)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_156(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t158: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t157 = v_inputs.clone(); s_91(py::getitem(&t157, &Py::str("w"))?, py::getitem(&t157, &Py::str("grad"))?, py::getitem(&t157, &Py::str("m"))?, py::getitem(&t157, &Py::str("v"))?, py::getitem(&t157, &Py::str("t"))?, py::getitem(&t157, &Py::str("lr"))?, py::getitem(&t157, &Py::str("beta1"))?, py::getitem(&t157, &Py::str("beta2"))?, py::getitem(&t157, &Py::str("eps"))?, py::getitem(&t157, &Py::str("weight_decay"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t158 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t159 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t184 = (t159).items()?;
        if t184.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t184[0].clone();
        v_v = t184[1].clone();
        if ({ let t160 = v_k.clone(); Py::Bool({ let t161 = v_outputs.clone(); if !py::contains(&t161, &t160)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_1600(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t1602: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t1601 = v_inputs.clone(); f_conv_params(py::getitem(&t1601, &Py::str("in_channels"))?, py::getitem(&t1601, &Py::str("out_channels"))?, py::getitem(&t1601, &Py::str("kernel_h"))?, py::getitem(&t1601, &Py::str("kernel_w"))?, py::getitem(&t1601, &Py::str("groups"))?, py::getitem(&t1601, &Py::str("bias"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t1602 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t1603 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t1628 = (t1603).items()?;
        if t1628.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t1628[0].clone();
        v_v = t1628[1].clone();
        if ({ let t1604 = v_k.clone(); Py::Bool({ let t1605 = v_outputs.clone(); if !py::contains(&t1605, &t1604)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_1661(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t1663: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t1662 = v_inputs.clone(); f_cosine_lr(py::getitem(&t1662, &Py::str("lr_max"))?, py::getitem(&t1662, &Py::str("eta_min"))?, py::getitem(&t1662, &Py::str("T_max"))?, py::getitem(&t1662, &Py::str("epoch"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t1663 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t1664 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t1689 = (t1664).items()?;
        if t1689.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t1689[0].clone();
        v_v = t1689[1].clone();
        if ({ let t1665 = v_k.clone(); Py::Bool({ let t1666 = v_outputs.clone(); if !py::contains(&t1666, &t1665)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_1721(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t1723: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t1722 = v_inputs.clone(); f_ce_logits(py::getitem(&t1722, &Py::str("z"))?, py::getitem(&t1722, &Py::str("y"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t1723 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t1724 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t1749 = (t1724).items()?;
        if t1749.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t1749[0].clone();
        v_v = t1749[1].clone();
        if ({ let t1725 = v_k.clone(); Py::Bool({ let t1726 = v_outputs.clone(); if !py::contains(&t1726, &t1725)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_1810(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t1812: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t1811 = v_inputs.clone(); f_davies_bouldin(py::getitem(&t1811, &Py::str("X"))?, py::getitem(&t1811, &Py::str("labels"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t1812 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t1813 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t1838 = (t1813).items()?;
        if t1838.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t1838[0].clone();
        v_v = t1838[1].clone();
        if ({ let t1814 = v_k.clone(); Py::Bool({ let t1815 = v_outputs.clone(); if !py::contains(&t1815, &t1814)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_1918(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t1920: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t1919 = v_inputs.clone(); f_stump_classify(py::getitem(&t1919, &Py::str("X"))?, py::getitem(&t1919, &Py::str("y"))?, py::getitem(&t1919, &Py::str("criterion"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t1920 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t1921 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t1946 = (t1921).items()?;
        if t1946.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t1946[0].clone();
        v_v = t1946[1].clone();
        if ({ let t1922 = v_k.clone(); Py::Bool({ let t1923 = v_outputs.clone(); if !py::contains(&t1923, &t1922)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_2017(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t2019: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t2018 = v_inputs.clone(); f_stump_regress(py::getitem(&t2018, &Py::str("X"))?, py::getitem(&t2018, &Py::str("y"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t2019 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t2020 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t2045 = (t2020).items()?;
        if t2045.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t2045[0].clone();
        v_v = t2045[1].clone();
        if ({ let t2021 = v_k.clone(); Py::Bool({ let t2022 = v_outputs.clone(); if !py::contains(&t2022, &t2021)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_2116(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t2118: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t2117 = v_inputs.clone(); f_dense_forward(py::getitem(&t2117, &Py::str("x"))?, py::getitem(&t2117, &Py::str("W1"))?, py::getitem(&t2117, &Py::str("b1"))?, py::getitem(&t2117, &Py::str("activation"))?, py::getitem(&t2117, &Py::str("W2"))?, py::getitem(&t2117, &Py::str("b2"))?, py::getitem(&t2117, &Py::str("output"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t2118 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t2119 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t2144 = (t2119).items()?;
        if t2144.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t2144[0].clone();
        v_v = t2144[1].clone();
        if ({ let t2120 = v_k.clone(); Py::Bool({ let t2121 = v_outputs.clone(); if !py::contains(&t2121, &t2120)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_2171(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t2173: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t2172 = v_inputs.clone(); f_dense_params(py::getitem(&t2172, &Py::str("sizes"))?, py::getitem(&t2172, &Py::str("bias"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t2173 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t2174 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t2199 = (t2174).items()?;
        if t2199.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t2199[0].clone();
        v_v = t2199[1].clone();
        if ({ let t2175 = v_k.clone(); Py::Bool({ let t2176 = v_outputs.clone(); if !py::contains(&t2176, &t2175)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_2238(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t2240: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t2239 = v_inputs.clone(); f_distances(py::getitem(&t2239, &Py::str("x"))?, py::getitem(&t2239, &Py::str("z"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t2240 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t2241 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t2266 = (t2241).items()?;
        if t2266.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t2266[0].clone();
        v_v = t2266[1].clone();
        if ({ let t2242 = v_k.clone(); Py::Bool({ let t2243 = v_outputs.clone(); if !py::contains(&t2243, &t2242)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_2297(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t2299: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t2298 = v_inputs.clone(); f_entropy(py::getitem(&t2298, &Py::str("p"))?, py::getitem(&t2298, &Py::str("unit"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t2299 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t2300 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t2325 = (t2300).items()?;
        if t2325.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t2325[0].clone();
        v_v = t2325[1].clone();
        if ({ let t2301 = v_k.clone(); Py::Bool({ let t2302 = v_outputs.clone(); if !py::contains(&t2302, &t2301)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_2376(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t2378: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t2377 = v_inputs.clone(); f_ece(py::getitem(&t2377, &Py::str("y_true"))?, py::getitem(&t2377, &Py::str("p"))?, py::getitem(&t2377, &Py::str("n_bins"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t2378 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t2379 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t2404 = (t2379).items()?;
        if t2404.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t2404[0].clone();
        v_v = t2404[1].clone();
        if ({ let t2380 = v_k.clone(); Py::Bool({ let t2381 = v_outputs.clone(); if !py::contains(&t2381, &t2380)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_2435(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t2437: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t2436 = v_inputs.clone(); f_fbeta(py::getitem(&t2436, &Py::str("TP"))?, py::getitem(&t2436, &Py::str("FP"))?, py::getitem(&t2436, &Py::str("FN"))?, py::getitem(&t2436, &Py::str("beta"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t2437 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t2438 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t2463 = (t2438).items()?;
        if t2463.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t2463[0].clone();
        v_v = t2463[1].clone();
        if ({ let t2439 = v_k.clone(); Py::Bool({ let t2440 = v_outputs.clone(); if !py::contains(&t2440, &t2439)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_247(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t249: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t248 = v_inputs.clone(); f_ami(py::getitem(&t248, &Py::str("labels_true"))?, py::getitem(&t248, &Py::str("labels_pred"))?, py::getitem(&t248, &Py::str("method"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t249 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t250 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t275 = (t250).items()?;
        if t275.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t275[0].clone();
        v_v = t275[1].clone();
        if ({ let t251 = v_k.clone(); Py::Bool({ let t252 = v_outputs.clone(); if !py::contains(&t252, &t251)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_2511(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t2513: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t2512 = v_inputs.clone(); f_focal(py::getitem(&t2512, &Py::str("y"))?, py::getitem(&t2512, &Py::str("z"))?, py::getitem(&t2512, &Py::str("gamma"))?, py::getitem(&t2512, &Py::str("alpha"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t2513 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t2514 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t2539 = (t2514).items()?;
        if t2539.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t2539[0].clone();
        v_v = t2539[1].clone();
        if ({ let t2515 = v_k.clone(); Py::Bool({ let t2516 = v_outputs.clone(); if !py::contains(&t2516, &t2515)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_2603(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t2605: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t2604 = v_inputs.clone(); f_gaussian_nb(py::getitem(&t2604, &Py::str("X"))?, py::getitem(&t2604, &Py::str("y"))?, py::getitem(&t2604, &Py::str("x"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t2605 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t2606 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t2631 = (t2606).items()?;
        if t2631.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t2631[0].clone();
        v_v = t2631[1].clone();
        if ({ let t2607 = v_k.clone(); Py::Bool({ let t2608 = v_outputs.clone(); if !py::contains(&t2608, &t2607)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_2664(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t2666: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t2665 = v_inputs.clone(); f_gini_split(py::getitem(&t2665, &Py::str("y"))?, py::getitem(&t2665, &Py::str("g"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t2666 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t2667 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t2692 = (t2667).items()?;
        if t2692.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t2692[0].clone();
        v_v = t2692[1].clone();
        if ({ let t2668 = v_k.clone(); Py::Bool({ let t2669 = v_outputs.clone(); if !py::contains(&t2669, &t2668)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_2759(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t2761: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t2760 = v_inputs.clone(); f_gd_step_linear(py::getitem(&t2760, &Py::str("X"))?, py::getitem(&t2760, &Py::str("y"))?, py::getitem(&t2760, &Py::str("w"))?, py::getitem(&t2760, &Py::str("b"))?, py::getitem(&t2760, &Py::str("lr"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t2761 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t2762 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t2787 = (t2762).items()?;
        if t2787.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t2787[0].clone();
        v_v = t2787[1].clone();
        if ({ let t2763 = v_k.clone(); Py::Bool({ let t2764 = v_outputs.clone(); if !py::contains(&t2764, &t2763)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_2836(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t2838: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t2837 = v_inputs.clone(); f_hamming(py::getitem(&t2837, &Py::str("Y_true"))?, py::getitem(&t2837, &Py::str("Y_pred"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t2838 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t2839 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t2864 = (t2839).items()?;
        if t2864.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t2864[0].clone();
        v_v = t2864[1].clone();
        if ({ let t2840 = v_k.clone(); Py::Bool({ let t2841 = v_outputs.clone(); if !py::contains(&t2841, &t2840)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_2927(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t2929: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t2928 = v_inputs.clone(); f_linkage_heights(py::getitem(&t2928, &Py::str("X"))?, py::getitem(&t2928, &Py::str("method"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t2929 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t2930 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t2955 = (t2930).items()?;
        if t2955.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t2955[0].clone();
        v_v = t2955[1].clone();
        if ({ let t2931 = v_k.clone(); Py::Bool({ let t2932 = v_outputs.clone(); if !py::contains(&t2932, &t2931)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_2995(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t2997: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t2996 = v_inputs.clone(); f_hinge(py::getitem(&t2996, &Py::str("y_true"))?, py::getitem(&t2996, &Py::str("scores"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t2997 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t2998 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t3023 = (t2998).items()?;
        if t3023.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t3023[0].clone();
        v_v = t3023[1].clone();
        if ({ let t2999 = v_k.clone(); Py::Bool({ let t3000 = v_outputs.clone(); if !py::contains(&t3000, &t2999)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_3084(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t3086: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t3085 = v_inputs.clone(); f_hcv(py::getitem(&t3085, &Py::str("labels_true"))?, py::getitem(&t3085, &Py::str("labels_pred"))?, py::getitem(&t3085, &Py::str("beta"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t3086 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t3087 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t3112 = (t3087).items()?;
        if t3112.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t3112[0].clone();
        v_v = t3112[1].clone();
        if ({ let t3088 = v_k.clone(); Py::Bool({ let t3089 = v_outputs.clone(); if !py::contains(&t3089, &t3088)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_3144(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t3146: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t3145 = v_inputs.clone(); f_huber(py::getitem(&t3145, &Py::str("y_true"))?, py::getitem(&t3145, &Py::str("y_pred"))?, py::getitem(&t3145, &Py::str("delta"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t3146 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t3147 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t3172 = (t3147).items()?;
        if t3172.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t3172[0].clone();
        v_v = t3172[1].clone();
        if ({ let t3148 = v_k.clone(); Py::Bool({ let t3149 = v_outputs.clone(); if !py::contains(&t3149, &t3148)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_3207(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t3209: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t3208 = v_inputs.clone(); f_impurity_counts(py::getitem(&t3208, &Py::str("counts"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t3209 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t3210 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t3235 = (t3210).items()?;
        if t3235.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t3235[0].clone();
        v_v = t3235[1].clone();
        if ({ let t3211 = v_k.clone(); Py::Bool({ let t3212 = v_outputs.clone(); if !py::contains(&t3212, &t3211)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_327(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t329: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t328 = v_inputs.clone(); f_adjusted_r2(py::getitem(&t328, &Py::str("y_true"))?, py::getitem(&t328, &Py::str("y_pred"))?, py::getitem(&t328, &Py::str("p"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t329 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t330 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t355 = (t330).items()?;
        if t355.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t355[0].clone();
        v_v = t355[1].clone();
        if ({ let t331 = v_k.clone(); Py::Bool({ let t332 = v_outputs.clone(); if !py::contains(&t332, &t331)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_3276(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t3278: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t3277 = v_inputs.clone(); f_information_gain(py::getitem(&t3277, &Py::str("y"))?, py::getitem(&t3277, &Py::str("x"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t3278 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t3279 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t3304 = (t3279).items()?;
        if t3304.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t3304[0].clone();
        v_v = t3304[1].clone();
        if ({ let t3280 = v_k.clone(); Py::Bool({ let t3281 = v_outputs.clone(); if !py::contains(&t3281, &t3280)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_3355(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t3357: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t3356 = v_inputs.clone(); f_joint_entropy(py::getitem(&t3356, &Py::str("P"))?, py::getitem(&t3356, &Py::str("unit"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t3357 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t3358 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t3383 = (t3358).items()?;
        if t3383.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t3383[0].clone();
        v_v = t3383[1].clone();
        if ({ let t3359 = v_k.clone(); Py::Bool({ let t3360 = v_outputs.clone(); if !py::contains(&t3360, &t3359)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_3481(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t3483: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t3482 = v_inputs.clone(); f_kfold_cv_ols(py::getitem(&t3482, &Py::str("X"))?, py::getitem(&t3482, &Py::str("y"))?, py::getitem(&t3482, &Py::str("k"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t3483 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t3484 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t3509 = (t3484).items()?;
        if t3509.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t3509[0].clone();
        v_v = t3509[1].clone();
        if ({ let t3485 = v_k.clone(); Py::Bool({ let t3486 = v_outputs.clone(); if !py::contains(&t3486, &t3485)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_3532(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t3534: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t3533 = v_inputs.clone(); f_kfold_sizes(py::getitem(&t3533, &Py::str("n"))?, py::getitem(&t3533, &Py::str("k"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t3534 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t3535 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t3560 = (t3535).items()?;
        if t3560.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t3560[0].clone();
        v_v = t3560[1].clone();
        if ({ let t3536 = v_k.clone(); Py::Bool({ let t3537 = v_outputs.clone(); if !py::contains(&t3537, &t3536)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_3596(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t3598: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t3597 = v_inputs.clone(); f_kl_discrete(py::getitem(&t3597, &Py::str("p"))?, py::getitem(&t3597, &Py::str("q"))?, py::getitem(&t3597, &Py::str("unit"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t3598 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t3599 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t3624 = (t3599).items()?;
        if t3624.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t3624[0].clone();
        v_v = t3624[1].clone();
        if ({ let t3600 = v_k.clone(); Py::Bool({ let t3601 = v_outputs.clone(); if !py::contains(&t3601, &t3600)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_3685(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t3687: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t3686 = v_inputs.clone(); f_kmeans(py::getitem(&t3686, &Py::str("X"))?, py::getitem(&t3686, &Py::str("C"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t3687 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t3688 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t3713 = (t3688).items()?;
        if t3713.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t3713[0].clone();
        v_v = t3713[1].clone();
        if ({ let t3689 = v_k.clone(); Py::Bool({ let t3690 = v_outputs.clone(); if !py::contains(&t3690, &t3689)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_3770(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t3772: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t3771 = v_inputs.clone(); f_kmeans_step(py::getitem(&t3771, &Py::str("X"))?, py::getitem(&t3771, &Py::str("C"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t3772 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t3773 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t3798 = (t3773).items()?;
        if t3798.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t3798[0].clone();
        v_v = t3798[1].clone();
        if ({ let t3774 = v_k.clone(); Py::Bool({ let t3775 = v_outputs.clone(); if !py::contains(&t3775, &t3774)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_3872(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t3874: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t3873 = v_inputs.clone(); f_knn_classify(py::getitem(&t3873, &Py::str("X"))?, py::getitem(&t3873, &Py::str("y"))?, py::getitem(&t3873, &Py::str("x"))?, py::getitem(&t3873, &Py::str("k"))?, py::getitem(&t3873, &Py::str("metric"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t3874 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t3875 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t3900 = (t3875).items()?;
        if t3900.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t3900[0].clone();
        v_v = t3900[1].clone();
        if ({ let t3876 = v_k.clone(); Py::Bool({ let t3877 = v_outputs.clone(); if !py::contains(&t3877, &t3876)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_3969(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t3971: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t3970 = v_inputs.clone(); f_knn_regress(py::getitem(&t3970, &Py::str("X"))?, py::getitem(&t3970, &Py::str("y"))?, py::getitem(&t3970, &Py::str("x"))?, py::getitem(&t3970, &Py::str("k"))?, py::getitem(&t3970, &Py::str("metric"))?, py::getitem(&t3970, &Py::str("weights"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t3971 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t3972 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t3997 = (t3972).items()?;
        if t3997.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t3997[0].clone();
        v_v = t3997[1].clone();
        if ({ let t3973 = v_k.clone(); Py::Bool({ let t3974 = v_outputs.clone(); if !py::contains(&t3974, &t3973)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_4035(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t4037: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t4036 = v_inputs.clone(); f_label_smoothing_ce(py::getitem(&t4036, &Py::str("z"))?, py::getitem(&t4036, &Py::str("y"))?, py::getitem(&t4036, &Py::str("epsilon"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t4037 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t4038 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t4063 = (t4038).items()?;
        if t4063.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t4063[0].clone();
        v_v = t4063[1].clone();
        if ({ let t4039 = v_k.clone(); Py::Bool({ let t4040 = v_outputs.clone(); if !py::contains(&t4040, &t4039)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_404(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t406: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t405 = v_inputs.clone(); f_average_precision(py::getitem(&t405, &Py::str("y_true"))?, py::getitem(&t405, &Py::str("scores"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t406 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t407 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t432 = (t407).items()?;
        if t432.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t432[0].clone();
        v_v = t432[1].clone();
        if ({ let t408 = v_k.clone(); Py::Bool({ let t409 = v_outputs.clone(); if !py::contains(&t409, &t408)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_4188(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t4190: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t4189 = v_inputs.clone(); f_lasso(py::getitem(&t4189, &Py::str("X"))?, py::getitem(&t4189, &Py::str("y"))?, py::getitem(&t4189, &Py::str("alpha"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t4190 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t4191 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t4216 = (t4191).items()?;
        if t4216.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t4216[0].clone();
        v_v = t4216[1].clone();
        if ({ let t4192 = v_k.clone(); Py::Bool({ let t4193 = v_outputs.clone(); if !py::contains(&t4193, &t4192)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_4263(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t4265: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t4264 = v_inputs.clone(); f_svm_decision(py::getitem(&t4264, &Py::str("w"))?, py::getitem(&t4264, &Py::str("b"))?, py::getitem(&t4264, &Py::str("X"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t4265 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t4266 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t4291 = (t4266).items()?;
        if t4291.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t4291[0].clone();
        v_v = t4291[1].clone();
        if ({ let t4267 = v_k.clone(); Py::Bool({ let t4268 = v_outputs.clone(); if !py::contains(&t4268, &t4267)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_4331(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t4333: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t4332 = v_inputs.clone(); f_log_loss_binary(py::getitem(&t4332, &Py::str("y_true"))?, py::getitem(&t4332, &Py::str("p"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t4333 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t4334 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t4359 = (t4334).items()?;
        if t4359.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t4359[0].clone();
        v_v = t4359[1].clone();
        if ({ let t4335 = v_k.clone(); Py::Bool({ let t4336 = v_outputs.clone(); if !py::contains(&t4336, &t4335)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_4410(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t4412: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t4411 = v_inputs.clone(); f_log_loss_multiclass(py::getitem(&t4411, &Py::str("y_true"))?, py::getitem(&t4411, &Py::str("P"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t4412 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t4413 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t4438 = (t4413).items()?;
        if t4438.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t4438[0].clone();
        v_v = t4438[1].clone();
        if ({ let t4414 = v_k.clone(); Py::Bool({ let t4415 = v_outputs.clone(); if !py::contains(&t4415, &t4414)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_4461(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t4463: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t4462 = v_inputs.clone(); f_log_softmax(py::getitem(&t4462, &Py::str("z"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t4463 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t4464 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t4489 = (t4464).items()?;
        if t4489.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t4489[0].clone();
        v_v = t4489[1].clone();
        if ({ let t4465 = v_k.clone(); Py::Bool({ let t4466 = v_outputs.clone(); if !py::contains(&t4466, &t4465)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_4547(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t4549: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t4548 = v_inputs.clone(); f_logistic_gradient(py::getitem(&t4548, &Py::str("X"))?, py::getitem(&t4548, &Py::str("y"))?, py::getitem(&t4548, &Py::str("w"))?, py::getitem(&t4548, &Py::str("b"))?, py::getitem(&t4548, &Py::str("lam"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t4549 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t4550 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t4575 = (t4550).items()?;
        if t4575.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t4575[0].clone();
        v_v = t4575[1].clone();
        if ({ let t4551 = v_k.clone(); Py::Bool({ let t4552 = v_outputs.clone(); if !py::contains(&t4552, &t4551)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_4690(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t4692: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t4691 = v_inputs.clone(); f_logistic_fit(py::getitem(&t4691, &Py::str("X"))?, py::getitem(&t4691, &Py::str("y"))?, py::getitem(&t4691, &Py::str("C"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t4692 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t4693 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t4718 = (t4693).items()?;
        if t4718.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t4718[0].clone();
        v_v = t4718[1].clone();
        if ({ let t4694 = v_k.clone(); Py::Bool({ let t4695 = v_outputs.clone(); if !py::contains(&t4695, &t4694)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_4764(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t4766: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t4765 = v_inputs.clone(); f_logistic_predict(py::getitem(&t4765, &Py::str("w"))?, py::getitem(&t4765, &Py::str("b"))?, py::getitem(&t4765, &Py::str("X"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t4766 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t4767 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t4792 = (t4767).items()?;
        if t4792.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t4792[0].clone();
        v_v = t4792[1].clone();
        if ({ let t4768 = v_k.clone(); Py::Bool({ let t4769 = v_outputs.clone(); if !py::contains(&t4769, &t4768)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_4878(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t4880: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t4879 = v_inputs.clone(); f_loocv_ols(py::getitem(&t4879, &Py::str("X"))?, py::getitem(&t4879, &Py::str("y"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t4880 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t4881 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t4906 = (t4881).items()?;
        if t4906.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t4906[0].clone();
        v_v = t4906[1].clone();
        if ({ let t4882 = v_k.clone(); Py::Bool({ let t4883 = v_outputs.clone(); if !py::contains(&t4883, &t4882)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_4936(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t4938: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t4937 = v_inputs.clone(); f_mape(py::getitem(&t4937, &Py::str("y_true"))?, py::getitem(&t4937, &Py::str("y_pred"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t4938 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t4939 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t4964 = (t4939).items()?;
        if t4964.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t4964[0].clone();
        v_v = t4964[1].clone();
        if ({ let t4940 = v_k.clone(); Py::Bool({ let t4941 = v_outputs.clone(); if !py::contains(&t4941, &t4940)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_4985(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t4987: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t4986 = v_inputs.clone(); f_mcc(py::getitem(&t4986, &Py::str("TP"))?, py::getitem(&t4986, &Py::str("FP"))?, py::getitem(&t4986, &Py::str("FN"))?, py::getitem(&t4986, &Py::str("TN"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t4987 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t4988 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t5013 = (t4988).items()?;
        if t5013.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t5013[0].clone();
        v_v = t5013[1].clone();
        if ({ let t4989 = v_k.clone(); Py::Bool({ let t4990 = v_outputs.clone(); if !py::contains(&t4990, &t4989)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_5058(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t5060: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t5059 = v_inputs.clone(); f_mean_ap(py::getitem(&t5059, &Py::str("R"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t5060 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t5061 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t5086 = (t5061).items()?;
        if t5086.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t5086[0].clone();
        v_v = t5086[1].clone();
        if ({ let t5062 = v_k.clone(); Py::Bool({ let t5063 = v_outputs.clone(); if !py::contains(&t5063, &t5062)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_5128(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t5130: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t5129 = v_inputs.clone(); f_mrr(py::getitem(&t5129, &Py::str("R"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t5130 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t5131 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t5156 = (t5131).items()?;
        if t5156.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t5156[0].clone();
        v_v = t5156[1].clone();
        if ({ let t5132 = v_k.clone(); Py::Bool({ let t5133 = v_outputs.clone(); if !py::contains(&t5133, &t5132)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_5186(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t5188: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t5187 = v_inputs.clone(); f_msle(py::getitem(&t5187, &Py::str("y_true"))?, py::getitem(&t5187, &Py::str("y_pred"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t5188 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t5189 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t5214 = (t5189).items()?;
        if t5214.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t5214[0].clone();
        v_v = t5214[1].clone();
        if ({ let t5190 = v_k.clone(); Py::Bool({ let t5191 = v_outputs.clone(); if !py::contains(&t5191, &t5190)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_5255(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t5257: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t5256 = v_inputs.clone(); f_minmax_scaler(py::getitem(&t5256, &Py::str("X"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t5257 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t5258 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t5283 = (t5258).items()?;
        if t5283.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t5283[0].clone();
        v_v = t5283[1].clone();
        if ({ let t5259 = v_k.clone(); Py::Bool({ let t5260 = v_outputs.clone(); if !py::contains(&t5260, &t5259)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_5337(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t5339: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t5338 = v_inputs.clone(); f_mse_gradient(py::getitem(&t5338, &Py::str("X"))?, py::getitem(&t5338, &Py::str("y"))?, py::getitem(&t5338, &Py::str("w"))?, py::getitem(&t5338, &Py::str("b"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t5339 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t5340 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t5365 = (t5340).items()?;
        if t5365.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t5365[0].clone();
        v_v = t5365[1].clone();
        if ({ let t5341 = v_k.clone(); Py::Bool({ let t5342 = v_outputs.clone(); if !py::contains(&t5342, &t5341)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_539(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t541: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t540 = v_inputs.clone(); f_backprop(py::getitem(&t540, &Py::str("x"))?, py::getitem(&t540, &Py::str("t"))?, py::getitem(&t540, &Py::str("W1"))?, py::getitem(&t540, &Py::str("b1"))?, py::getitem(&t540, &Py::str("activation"))?, py::getitem(&t540, &Py::str("W2"))?, py::getitem(&t540, &Py::str("b2"))?, py::getitem(&t540, &Py::str("loss"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t541 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t542 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t567 = (t542).items()?;
        if t567.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t567[0].clone();
        v_v = t567[1].clone();
        if ({ let t543 = v_k.clone(); Py::Bool({ let t544 = v_outputs.clone(); if !py::contains(&t544, &t543)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_5399(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t5401: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t5400 = v_inputs.clone(); f_multiclass_confusion(py::getitem(&t5400, &Py::str("y_true"))?, py::getitem(&t5400, &Py::str("y_pred"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t5401 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t5402 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t5427 = (t5402).items()?;
        if t5427.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t5427[0].clone();
        v_v = t5427[1].clone();
        if ({ let t5403 = v_k.clone(); Py::Bool({ let t5404 = v_outputs.clone(); if !py::contains(&t5404, &t5403)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_5475(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t5477: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t5476 = v_inputs.clone(); f_multiclass_hinge(py::getitem(&t5476, &Py::str("y_true"))?, py::getitem(&t5476, &Py::str("D"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t5477 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t5478 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t5503 = (t5478).items()?;
        if t5503.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t5503[0].clone();
        v_v = t5503[1].clone();
        if ({ let t5479 = v_k.clone(); Py::Bool({ let t5480 = v_outputs.clone(); if !py::contains(&t5480, &t5479)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_5575(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t5577: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t5576 = v_inputs.clone(); f_multihead(py::getitem(&t5576, &Py::str("Q"))?, py::getitem(&t5576, &Py::str("K"))?, py::getitem(&t5576, &Py::str("V"))?, py::getitem(&t5576, &Py::str("heads"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t5577 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t5578 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t5603 = (t5578).items()?;
        if t5603.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t5603[0].clone();
        v_v = t5603[1].clone();
        if ({ let t5579 = v_k.clone(); Py::Bool({ let t5580 = v_outputs.clone(); if !py::contains(&t5580, &t5579)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_5676(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t5678: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t5677 = v_inputs.clone(); f_multinomial_nb(py::getitem(&t5677, &Py::str("X"))?, py::getitem(&t5677, &Py::str("y"))?, py::getitem(&t5677, &Py::str("x"))?, py::getitem(&t5677, &Py::str("alpha"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t5678 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t5679 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t5704 = (t5679).items()?;
        if t5704.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t5704[0].clone();
        v_v = t5704[1].clone();
        if ({ let t5680 = v_k.clone(); Py::Bool({ let t5681 = v_outputs.clone(); if !py::contains(&t5681, &t5680)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_5755(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t5757: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t5756 = v_inputs.clone(); f_mutual_info_labels(py::getitem(&t5756, &Py::str("x"))?, py::getitem(&t5756, &Py::str("y"))?, py::getitem(&t5756, &Py::str("unit"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t5757 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t5758 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t5783 = (t5758).items()?;
        if t5783.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t5783[0].clone();
        v_v = t5783[1].clone();
        if ({ let t5759 = v_k.clone(); Py::Bool({ let t5760 = v_outputs.clone(); if !py::contains(&t5760, &t5759)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_5821(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t5823: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t5822 = v_inputs.clone(); f_ndcg(py::getitem(&t5822, &Py::str("rel"))?, py::getitem(&t5822, &Py::str("k"))?, py::getitem(&t5822, &Py::str("gain"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t5823 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t5824 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t5849 = (t5824).items()?;
        if t5849.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t5849[0].clone();
        v_v = t5849[1].clone();
        if ({ let t5825 = v_k.clone(); Py::Bool({ let t5826 = v_outputs.clone(); if !py::contains(&t5826, &t5825)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_5905(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t5907: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t5906 = v_inputs.clone(); f_nmi(py::getitem(&t5906, &Py::str("labels_true"))?, py::getitem(&t5906, &Py::str("labels_pred"))?, py::getitem(&t5906, &Py::str("method"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t5907 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t5908 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t5933 = (t5908).items()?;
        if t5933.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t5933[0].clone();
        v_v = t5933[1].clone();
        if ({ let t5909 = v_k.clone(); Py::Bool({ let t5910 = v_outputs.clone(); if !py::contains(&t5910, &t5909)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_5958(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t5960: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t5959 = v_inputs.clone(); f_one_hot(py::getitem(&t5959, &Py::str("categories"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t5960 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t5961 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t5986 = (t5961).items()?;
        if t5986.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t5986[0].clone();
        v_v = t5986[1].clone();
        if ({ let t5962 = v_k.clone(); Py::Bool({ let t5963 = v_outputs.clone(); if !py::contains(&t5963, &t5962)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_602(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t604: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t603 = v_inputs.clone(); f_balanced_accuracy(py::getitem(&t603, &Py::str("y_true"))?, py::getitem(&t603, &Py::str("y_pred"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t604 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t605 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t630 = (t605).items()?;
        if t630.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t630[0].clone();
        v_v = t630[1].clone();
        if ({ let t606 = v_k.clone(); Py::Bool({ let t607 = v_outputs.clone(); if !py::contains(&t607, &t606)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_6070(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t6072: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t6071 = v_inputs.clone(); f_pca(py::getitem(&t6071, &Py::str("X"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t6072 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t6073 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t6098 = (t6073).items()?;
        if t6098.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t6098[0].clone();
        v_v = t6098[1].clone();
        if ({ let t6074 = v_k.clone(); Py::Bool({ let t6075 = v_outputs.clone(); if !py::contains(&t6075, &t6074)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_6161(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t6163: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t6162 = v_inputs.clone(); f_perceptron_epoch(py::getitem(&t6162, &Py::str("X"))?, py::getitem(&t6162, &Py::str("y"))?, py::getitem(&t6162, &Py::str("w"))?, py::getitem(&t6162, &Py::str("b"))?, py::getitem(&t6162, &Py::str("lr"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t6163 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t6164 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t6189 = (t6164).items()?;
        if t6189.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t6189[0].clone();
        v_v = t6189[1].clone();
        if ({ let t6165 = v_k.clone(); Py::Bool({ let t6166 = v_outputs.clone(); if !py::contains(&t6166, &t6165)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_62(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t64: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t63 = v_inputs.clone(); f_adam_step(py::getitem(&t63, &Py::str("w"))?, py::getitem(&t63, &Py::str("grad"))?, py::getitem(&t63, &Py::str("m"))?, py::getitem(&t63, &Py::str("v"))?, py::getitem(&t63, &Py::str("t"))?, py::getitem(&t63, &Py::str("lr"))?, py::getitem(&t63, &Py::str("beta1"))?, py::getitem(&t63, &Py::str("beta2"))?, py::getitem(&t63, &Py::str("eps"))?, py::method(&t63, "get", &[Py::str("weight_decay"), Py::None])?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t64 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t65 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t90 = (t65).items()?;
        if t90.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t90[0].clone();
        v_v = t90[1].clone();
        if ({ let t66 = v_k.clone(); Py::Bool({ let t67 = v_outputs.clone(); if !py::contains(&t67, &t66)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_6219(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t6221: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t6220 = v_inputs.clone(); f_poly_kernel(py::getitem(&t6220, &Py::str("x"))?, py::getitem(&t6220, &Py::str("z"))?, py::getitem(&t6220, &Py::str("gamma"))?, py::getitem(&t6220, &Py::str("coef0"))?, py::getitem(&t6220, &Py::str("degree"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t6221 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t6222 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t6247 = (t6222).items()?;
        if t6247.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t6247[0].clone();
        v_v = t6247[1].clone();
        if ({ let t6223 = v_k.clone(); Py::Bool({ let t6224 = v_outputs.clone(); if !py::contains(&t6224, &t6223)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_6297(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t6299: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t6298 = v_inputs.clone(); f_pooling(py::getitem(&t6298, &Py::str("X"))?, py::getitem(&t6298, &Py::str("size"))?, py::getitem(&t6298, &Py::str("stride"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t6299 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t6300 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t6325 = (t6300).items()?;
        if t6325.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t6325[0].clone();
        v_v = t6325[1].clone();
        if ({ let t6301 = v_k.clone(); Py::Bool({ let t6302 = v_outputs.clone(); if !py::contains(&t6302, &t6301)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_6360(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t6362: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t6361 = v_inputs.clone(); f_prec_rec_at_k(py::getitem(&t6361, &Py::str("rel"))?, py::getitem(&t6361, &Py::str("n_relevant"))?, py::getitem(&t6361, &Py::str("k"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t6362 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t6363 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t6388 = (t6363).items()?;
        if t6388.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t6388[0].clone();
        v_v = t6388[1].clone();
        if ({ let t6364 = v_k.clone(); Py::Bool({ let t6365 = v_outputs.clone(); if !py::contains(&t6365, &t6364)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_6444(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t6446: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t6445 = v_inputs.clone(); f_prf_averaged(py::getitem(&t6445, &Py::str("y_true"))?, py::getitem(&t6445, &Py::str("y_pred"))?, py::getitem(&t6445, &Py::str("average"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t6446 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t6447 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t6472 = (t6447).items()?;
        if t6472.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t6472[0].clone();
        v_v = t6472[1].clone();
        if ({ let t6448 = v_k.clone(); Py::Bool({ let t6449 = v_outputs.clone(); if !py::contains(&t6449, &t6448)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_6502(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t6504: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t6503 = v_inputs.clone(); f_pinball(py::getitem(&t6503, &Py::str("y_true"))?, py::getitem(&t6503, &Py::str("y_pred"))?, py::getitem(&t6503, &Py::str("alpha"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t6504 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t6505 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t6530 = (t6505).items()?;
        if t6530.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t6530[0].clone();
        v_v = t6530[1].clone();
        if ({ let t6506 = v_k.clone(); Py::Bool({ let t6507 = v_outputs.clone(); if !py::contains(&t6507, &t6506)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_654(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t656: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t655 = v_inputs.clone(); f_class_weights(py::getitem(&t655, &Py::str("y"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t656 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t657 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t682 = (t657).items()?;
        if t682.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t682[0].clone();
        v_v = t682[1].clone();
        if ({ let t658 = v_k.clone(); Py::Bool({ let t659 = v_outputs.clone(); if !py::contains(&t659, &t658)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_6579(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t6581: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t6580 = v_inputs.clone(); f_rand_index(py::getitem(&t6580, &Py::str("labels_true"))?, py::getitem(&t6580, &Py::str("labels_pred"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t6581 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t6582 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t6607 = (t6582).items()?;
        if t6607.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t6607[0].clone();
        v_v = t6607[1].clone();
        if ({ let t6583 = v_k.clone(); Py::Bool({ let t6584 = v_outputs.clone(); if !py::contains(&t6584, &t6583)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_6647(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t6649: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t6648 = v_inputs.clone(); f_rbf_kernel(py::getitem(&t6648, &Py::str("X"))?, py::getitem(&t6648, &Py::str("gamma"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t6649 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t6650 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t6675 = (t6650).items()?;
        if t6675.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t6675[0].clone();
        v_v = t6675[1].clone();
        if ({ let t6651 = v_k.clone(); Py::Bool({ let t6652 = v_outputs.clone(); if !py::contains(&t6652, &t6651)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_6710(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t6712: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t6711 = v_inputs.clone(); f_receptive_field(py::getitem(&t6711, &Py::str("kernels"))?, py::getitem(&t6711, &Py::str("strides"))?, py::getitem(&t6711, &Py::str("dilations"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t6712 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t6713 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t6738 = (t6713).items()?;
        if t6738.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t6738[0].clone();
        v_v = t6738[1].clone();
        if ({ let t6714 = v_k.clone(); Py::Bool({ let t6715 = v_outputs.clone(); if !py::contains(&t6715, &t6714)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_6776(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t6778: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t6777 = v_inputs.clone(); f_regression_metrics(py::getitem(&t6777, &Py::str("y_true"))?, py::getitem(&t6777, &Py::str("y_pred"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t6778 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t6779 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t6804 = (t6779).items()?;
        if t6804.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t6804[0].clone();
        v_v = t6804[1].clone();
        if ({ let t6780 = v_k.clone(); Py::Bool({ let t6781 = v_outputs.clone(); if !py::contains(&t6781, &t6780)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_6886(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t6888: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t6887 = v_inputs.clone(); f_ridge(py::getitem(&t6887, &Py::str("X"))?, py::getitem(&t6887, &Py::str("y"))?, py::getitem(&t6887, &Py::str("alpha"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t6888 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t6889 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t6914 = (t6889).items()?;
        if t6914.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t6914[0].clone();
        v_v = t6914[1].clone();
        if ({ let t6890 = v_k.clone(); Py::Bool({ let t6891 = v_outputs.clone(); if !py::contains(&t6891, &t6890)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_6942(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t6944: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t6943 = v_inputs.clone(); f_rnn_params(py::getitem(&t6943, &Py::str("cell"))?, py::getitem(&t6943, &Py::str("input_size"))?, py::getitem(&t6943, &Py::str("hidden_size"))?, py::getitem(&t6943, &Py::str("num_layers"))?, py::getitem(&t6943, &Py::str("bidirectional"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t6944 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t6945 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t6970 = (t6945).items()?;
        if t6970.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t6970[0].clone();
        v_v = t6970[1].clone();
        if ({ let t6946 = v_k.clone(); Py::Bool({ let t6947 = v_outputs.clone(); if !py::contains(&t6947, &t6946)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_7009(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t7011: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t7010 = v_inputs.clone(); f_robust_scaler(py::getitem(&t7010, &Py::str("X"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t7011 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t7012 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t7037 = (t7012).items()?;
        if t7037.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t7037[0].clone();
        v_v = t7037[1].clone();
        if ({ let t7013 = v_k.clone(); Py::Bool({ let t7014 = v_outputs.clone(); if !py::contains(&t7014, &t7013)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_7084(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t7086: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t7085 = v_inputs.clone(); f_roc_auc(py::getitem(&t7085, &Py::str("y_true"))?, py::getitem(&t7085, &Py::str("scores"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t7086 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t7087 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t7112 = (t7087).items()?;
        if t7112.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t7112[0].clone();
        v_v = t7112[1].clone();
        if ({ let t7088 = v_k.clone(); Py::Bool({ let t7089 = v_outputs.clone(); if !py::contains(&t7089, &t7088)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_7163(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t7165: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t7164 = v_inputs.clone(); f_roc_curve(py::getitem(&t7164, &Py::str("y_true"))?, py::getitem(&t7164, &Py::str("scores"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t7165 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t7166 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t7191 = (t7166).items()?;
        if t7191.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t7191[0].clone();
        v_v = t7191[1].clone();
        if ({ let t7167 = v_k.clone(); Py::Bool({ let t7168 = v_outputs.clone(); if !py::contains(&t7168, &t7167)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_7240(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t7242: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t7241 = v_inputs.clone(); f_attention(py::getitem(&t7241, &Py::str("Q"))?, py::getitem(&t7241, &Py::str("K"))?, py::getitem(&t7241, &Py::str("V"))?, py::getitem(&t7241, &Py::str("causal"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t7242 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t7243 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t7268 = (t7243).items()?;
        if t7268.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t7268[0].clone();
        v_v = t7268[1].clone();
        if ({ let t7244 = v_k.clone(); Py::Bool({ let t7245 = v_outputs.clone(); if !py::contains(&t7245, &t7244)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_7317(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t7319: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t7318 = v_inputs.clone(); f_momentum_step(py::getitem(&t7318, &Py::str("w"))?, py::getitem(&t7318, &Py::str("grad"))?, py::getitem(&t7318, &Py::str("buf"))?, py::getitem(&t7318, &Py::str("lr"))?, py::getitem(&t7318, &Py::str("momentum"))?, py::getitem(&t7318, &Py::str("nesterov"))?, py::getitem(&t7318, &Py::str("weight_decay"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t7319 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t7320 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t7345 = (t7320).items()?;
        if t7345.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t7345[0].clone();
        v_v = t7345[1].clone();
        if ({ let t7321 = v_k.clone(); Py::Bool({ let t7322 = v_outputs.clone(); if !py::contains(&t7322, &t7321)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_7365(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t7367: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t7366 = v_inputs.clone(); f_sigmoid(py::getitem(&t7366, &Py::str("z"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t7367 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t7368 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t7393 = (t7368).items()?;
        if t7393.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t7393[0].clone();
        v_v = t7393[1].clone();
        if ({ let t7369 = v_k.clone(); Py::Bool({ let t7370 = v_outputs.clone(); if !py::contains(&t7370, &t7369)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_737(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t739: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t738 = v_inputs.clone(); f_batchnorm(py::getitem(&t738, &Py::str("X"))?, py::getitem(&t738, &Py::str("gamma"))?, py::getitem(&t738, &Py::str("beta"))?, py::getitem(&t738, &Py::str("eps"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t739 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t740 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t765 = (t740).items()?;
        if t765.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t765[0].clone();
        v_v = t765[1].clone();
        if ({ let t741 = v_k.clone(); Py::Bool({ let t742 = v_outputs.clone(); if !py::contains(&t742, &t741)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_7455(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t7457: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t7456 = v_inputs.clone(); f_silhouette(py::getitem(&t7456, &Py::str("X"))?, py::getitem(&t7456, &Py::str("labels"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t7457 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t7458 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t7483 = (t7458).items()?;
        if t7483.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t7483[0].clone();
        v_v = t7483[1].clone();
        if ({ let t7459 = v_k.clone(); Py::Bool({ let t7460 = v_outputs.clone(); if !py::contains(&t7460, &t7459)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_7507(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t7509: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t7508 = v_inputs.clone(); f_softmax(py::getitem(&t7508, &Py::str("z"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t7509 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t7510 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t7535 = (t7510).items()?;
        if t7535.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t7535[0].clone();
        v_v = t7535[1].clone();
        if ({ let t7511 = v_k.clone(); Py::Bool({ let t7512 = v_outputs.clone(); if !py::contains(&t7512, &t7511)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_7564(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t7566: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t7565 = v_inputs.clone(); f_softmax_temperature(py::getitem(&t7565, &Py::str("z"))?, py::getitem(&t7565, &Py::str("T"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t7566 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t7567 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t7592 = (t7567).items()?;
        if t7592.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t7592[0].clone();
        v_v = t7592[1].clone();
        if ({ let t7568 = v_k.clone(); Py::Bool({ let t7569 = v_outputs.clone(); if !py::contains(&t7569, &t7568)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_7634(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t7636: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t7635 = v_inputs.clone(); f_standard_scaler(py::getitem(&t7635, &Py::str("X"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t7636 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t7637 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t7662 = (t7637).items()?;
        if t7662.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t7662[0].clone();
        v_v = t7662[1].clone();
        if ({ let t7638 = v_k.clone(); Py::Bool({ let t7639 = v_outputs.clone(); if !py::contains(&t7639, &t7638)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_7692(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t7694: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t7693 = v_inputs.clone(); f_step_lr(py::getitem(&t7693, &Py::str("lr0"))?, py::getitem(&t7693, &Py::str("step_size"))?, py::getitem(&t7693, &Py::str("gamma"))?, py::getitem(&t7693, &Py::str("epoch"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t7694 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t7695 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t7720 = (t7695).items()?;
        if t7720.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t7720[0].clone();
        v_v = t7720[1].clone();
        if ({ let t7696 = v_k.clone(); Py::Bool({ let t7697 = v_outputs.clone(); if !py::contains(&t7697, &t7696)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_7788(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t7790: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t7789 = v_inputs.clone(); f_threshold_metrics(py::getitem(&t7789, &Py::str("y_true"))?, py::getitem(&t7789, &Py::str("scores"))?, py::getitem(&t7789, &Py::str("threshold"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t7790 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t7791 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t7816 = (t7791).items()?;
        if t7816.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t7816[0].clone();
        v_v = t7816[1].clone();
        if ({ let t7792 = v_k.clone(); Py::Bool({ let t7793 = v_outputs.clone(); if !py::contains(&t7793, &t7792)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_7869(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t7871: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t7870 = v_inputs.clone(); f_top_k_accuracy(py::getitem(&t7870, &Py::str("y_true"))?, py::getitem(&t7870, &Py::str("S"))?, py::getitem(&t7870, &Py::str("k"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t7871 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t7872 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t7897 = (t7872).items()?;
        if t7897.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t7897[0].clone();
        v_v = t7897[1].clone();
        if ({ let t7873 = v_k.clone(); Py::Bool({ let t7874 = v_outputs.clone(); if !py::contains(&t7874, &t7873)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_7921(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t7923: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t7922 = v_inputs.clone(); f_split_sizes(py::getitem(&t7922, &Py::str("n"))?, py::getitem(&t7922, &Py::str("test_size"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t7923 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t7924 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t7949 = (t7924).items()?;
        if t7949.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t7949[0].clone();
        v_v = t7949[1].clone();
        if ({ let t7925 = v_k.clone(); Py::Bool({ let t7926 = v_outputs.clone(); if !py::contains(&t7926, &t7925)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_7976(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t7978: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t7977 = v_inputs.clone(); f_transformer_params(py::getitem(&t7977, &Py::str("d_model"))?, py::getitem(&t7977, &Py::str("n_heads"))?, py::getitem(&t7977, &Py::str("d_ff"))?, py::getitem(&t7977, &Py::str("n_layers"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t7978 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t7979 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t8004 = (t7979).items()?;
        if t8004.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t8004[0].clone();
        v_v = t8004[1].clone();
        if ({ let t7980 = v_k.clone(); Py::Bool({ let t7981 = v_outputs.clone(); if !py::contains(&t7981, &t7980)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_799(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t801: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t800 = v_inputs.clone(); f_bce_logits(py::getitem(&t800, &Py::str("y"))?, py::getitem(&t800, &Py::str("z"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t801 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t802 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t827 = (t802).items()?;
        if t827.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t827[0].clone();
        v_v = t827[1].clone();
        if ({ let t803 = v_k.clone(); Py::Bool({ let t804 = v_outputs.clone(); if !py::contains(&t804, &t803)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_8035(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t8037: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t8036 = v_inputs.clone(); f_triplet(py::getitem(&t8036, &Py::str("anchor"))?, py::getitem(&t8036, &Py::str("positive"))?, py::getitem(&t8036, &Py::str("negative"))?, py::getitem(&t8036, &Py::str("margin"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t8037 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t8038 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t8063 = (t8038).items()?;
        if t8063.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t8063[0].clone();
        v_v = t8063[1].clone();
        if ({ let t8039 = v_k.clone(); Py::Bool({ let t8040 = v_outputs.clone(); if !py::contains(&t8040, &t8039)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_8119(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t8121: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t8120 = v_inputs.clone(); f_tweedie(py::getitem(&t8120, &Py::str("y_true"))?, py::getitem(&t8120, &Py::str("y_pred"))?, py::getitem(&t8120, &Py::str("power"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t8121 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t8122 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t8147 = (t8122).items()?;
        if t8147.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t8147[0].clone();
        v_v = t8147[1].clone();
        if ({ let t8123 = v_k.clone(); Py::Bool({ let t8124 = v_outputs.clone(); if !py::contains(&t8124, &t8123)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_8188(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t8190: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t8189 = v_inputs.clone(); f_warmup_cosine(py::getitem(&t8189, &Py::str("lr_max"))?, py::getitem(&t8189, &Py::str("start_factor"))?, py::getitem(&t8189, &Py::str("warmup"))?, py::getitem(&t8189, &Py::str("total"))?, py::getitem(&t8189, &Py::str("eta_min"))?, py::getitem(&t8189, &Py::str("step"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t8190 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t8191 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t8216 = (t8191).items()?;
        if t8216.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t8216[0].clone();
        v_v = t8216[1].clone();
        if ({ let t8192 = v_k.clone(); Py::Bool({ let t8193 = v_outputs.clone(); if !py::contains(&t8193, &t8192)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_901(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t903: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t902 = v_inputs.clone(); f_bernoulli_nb(py::getitem(&t902, &Py::str("X"))?, py::getitem(&t902, &Py::str("y"))?, py::getitem(&t902, &Py::str("x"))?, py::getitem(&t902, &Py::str("alpha"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t903 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t904 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t929 = (t904).items()?;
        if t929.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t929[0].clone();
        v_v = t929[1].clone();
        if ({ let t905 = v_k.clone(); Py::Bool({ let t906 = v_outputs.clone(); if !py::contains(&t906, &t905)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_91(mut v_w: Py, mut v_grad: Py, mut v_m: Py, mut v_v: Py, mut v_t: Py, mut v_lr: Py, mut v_beta1: Py, mut v_beta2: Py, mut v_eps: Py, mut v_weight_decay: Py) -> R {
    return Ok({ let t155 = f__need({ let t92 = v_weight_decay.clone(); Py::Bool({ let t93 = Py::Int(0i64); if py::ge(&t92, &t93)? { true } else { false } }) }, Py::str("weight_decay must be >= 0"))?; if t155.truthy() { t155 } else { f_adam_step(v_w.clone(), v_grad.clone(), v_m.clone(), v_v.clone(), v_t.clone(), v_lr.clone(), v_beta1.clone(), v_beta2.clone(), v_eps.clone(), v_weight_decay.clone())? } });
    #[allow(unreachable_code)]
    Ok(Py::None)
}

#[allow(unused_mut, unused_variables, unused_assignments, non_snake_case, unreachable_code)]
fn s_977(mut v_inputs: Py, mut v_outputs: Py) -> R {
    let mut v_k: Py = Py::None;
    let mut v_v: Py = Py::None;
    let mut v_want: Py = Py::None;
    let mut v_worst: Py = Py::None;
    let t979: Result<Option<Py>, PyErr> = (|| -> Result<Option<Py>, PyErr> {
            v_want = { let t978 = v_inputs.clone(); f_bias_variance(py::getitem(&t978, &Py::str("P"))?, py::getitem(&t978, &Py::str("f"))?)? };
            #[allow(unreachable_code)]
            Ok(None)
        })();
    match t979 {
        Ok(Some(r)) => { return Ok(r); }
        Ok(None) => {}
        Err(e) if (!matches!(e, PyErr::Unsupported(_))) => {
            return Ok(Py::Float(f64::INFINITY));
        }
        Err(e) => return Err(e),
    }
    v_worst = Py::Float(0.0);
    for t980 in py::iter(&(py::method(&v_want.clone(), "items", &[])?))? {
        let t1005 = (t980).items()?;
        if t1005.len() != 2 { return Err(PyErr::Value("unpack".into())); }
        v_k = t1005[0].clone();
        v_v = t1005[1].clone();
        if ({ let t981 = v_k.clone(); Py::Bool({ let t982 = v_outputs.clone(); if !py::contains(&t982, &t981)? { true } else { false } }) }).truthy() {
            return Ok(Py::Float(f64::INFINITY));
        }
        v_worst = py::minmax(&[v_worst.clone(), f__rel(py::getitem(&v_outputs.clone(), &v_k.clone())?, v_v.clone())?], true)?;
    }
    return Ok(v_worst.clone());
    #[allow(unreachable_code)]
    Ok(Py::None)
}
