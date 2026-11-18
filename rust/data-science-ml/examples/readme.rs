//! Data Science and Machine Learning: 15 calculations through the Rust library (the same engine as the desktop app, the
//! command line and the browser). Run from the rust/ folder:
//!     cargo run --release --example readme -p data-science-ml-ai999x
//! Inputs are text, read like a form field ("2 m", "0.5", "x^3 - 3x"); a JSON list stays a list.

use serde_json::{json, Value};

/// One calculation: the request in, its outcome and every value with its unit out.
fn show(request: Value) {
    let answer: Value = serde_json::from_str(&data_science_ml_ai999x::solve_json(&request.to_string())).unwrap();
    println!("{}", answer["outcome"].as_str().unwrap_or(""));
    for v in answer["values"].as_array().into_iter().flatten() {
        let unit = v[2].as_str().unwrap_or("");
        let unit = if unit.is_empty() || unit == "1" { String::new() } else { format!(" {unit}") };
        println!("{} = {}{}", v[0].as_str().unwrap_or(""), v[1], unit);
    }
}

fn main() {
    show(json!({"key": "ml_hierarchical_clustering", "inputs": {"X": [[0, 0], [0, 1], [4, 0], [4, 1.5], [10, 10]], "method": "single"}}));
    println!();
    show(json!({"key": "ml_average_precision", "inputs": {"y_true": [0, 1, 1, 0, 1], "scores": [0.1, 0.9, 0.8, 0.3, 0.2]}}));
    println!();
    show(json!({"key": "ml_adjusted_mutual_information", "inputs": {"labels_true": [0, 0, 1, 1, 2, 2], "labels_pred": [0, 0, 1, 1, 1, 1], "method": "arithmetic"}}));
    println!();
    show(json!({"key": "ml_pca", "inputs": {"X": [[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3], [2.3, 2.7], [2, 1.6], [1, 1.1], [1.5, 1.6], [1.1, 0.9]]}}));
    println!();
    show(json!({"key": "ml_logistic_loss_gradient", "inputs": {"X": [[1, 2], [2, 1], [3, 3]], "y": [0, 1, 1], "w": [0.5, -0.25], "b": "0.1", "lam": "0"}}));
    println!();
    show(json!({"key": "ml_gini_split", "inputs": {"y": [0, 0, 0, 1, 1, 1, 1, 0], "g": [0, 0, 0, 0, 1, 1, 1, 1]}}));
    println!();
    show(json!({"key": "ml_lasso_regression", "inputs": {"X": [[1, 0.5], [2, 1.5], [3, 1], [4, 3], [5, 2], [6, 3.5]], "y": [1.1, 2.3, 2.9, 4.2, 5.1, 5.8], "alpha": "0.1"}}));
    println!();
    show(json!({"key": "ml_bce_with_logits", "inputs": {"y": [1, 0, 1], "z": [2, -1, -0.5]}}));
    println!();
    show(json!({"key": "ml_bias_variance_decomposition", "inputs": {"P": [[1, 2], [3, 2.5], [2, 3.5]], "f": [1.5, 2.5]}}));
    println!();
    show(json!({"key": "ml_conv2d_forward", "inputs": {"image": [[1, 2, 3, 0], [0, 1, 2, 3], [3, 0, 1, 2], [2, 3, 0, 1]], "kernel": [[2, 0, 1], [0, 1, 2], [1, 0, 2]], "stride": "1", "padding": "0"}}));
    println!();
    show(json!({"key": "ml_adam_step", "inputs": {"w": [1, -2], "grad": [0.5, -1], "m": [0, 0], "v": [0, 0], "t": "1", "lr": "0.001", "beta1": "0.9", "beta2": "0.999", "eps": "1e-08"}}));
    println!();
    show(json!({"key": "ml_one_hot_encoding", "inputs": {"categories": "red, green, red, blue, green, red"}}));
    println!();
    show(json!({"key": "ml_mean_average_precision", "inputs": {"R": [[1, 0, 1, 0, 0], [0, 1, 0, 0, 1]]}}));
    println!();
    show(json!({"key": "ml_adjusted_r2", "inputs": {"y_true": [10, 20, 30, 40, 50], "y_pred": [12, 18, 33, 37, 52], "p": "1"}}));
    println!();
    show(json!({"key": "ml_bernoulli_naive_bayes", "inputs": {"X": [[1, 0, 1], [1, 1, 0], [0, 1, 1], [0, 0, 1], [1, 1, 1]], "y": [0, 0, 1, 1, 0], "x": [1, 0, 0], "alpha": "1"}}));
}
