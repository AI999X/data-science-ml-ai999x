//! Data Science and ML's own live figures (ax_engine::figkit), drawn for the visitor's inputs from what the model
//! computes: loss curves and schedules from runs of the model on one sample or one step, optimiser paths from its
//! steps fed back into it, ROC and precision-recall curves from its own threshold runs, clusters, decision regions,
//! confusion matrices, networks with their weights and gradients. A figure that cannot be drawn for the inputs is
//! left out (docs/FIGURE_PARITY.md).

use crate::figures_b as b;
use crate::figures_c as c;
use crate::plot::{self as pl, list, Plot, BAR, DOT, LINE};
use ax_engine::figkit::{self as k, num, Solved};
use ax_engine::figures::Figure;
use ax_engine::{Answer, Input, Val};
use serde_json::Value;

/// The figures for one solved request (the dispatcher ax_engine calls: Pack::figures).
pub fn live(s: &Solved) -> Vec<Figure> {
    let v: Vec<Option<Figure>> = match s.key {
        "ml_bce_with_logits" | "ml_brier_score" | "ml_focal_loss" | "ml_hinge_loss" | "ml_huber_loss"
        | "ml_log_loss_binary" | "ml_quantile_loss" | "ml_tweedie_deviance" => vec![loss(s)],
        "ml_cosine_lr_schedule" | "ml_step_lr_schedule" | "ml_warmup_cosine_lr_schedule" => vec![schedule(s)],
        "ml_adam_step" | "ml_adamw_step" | "ml_sgd_momentum_step" => vec![opt_path(s)],
        "ml_lasso_regression" => vec![lasso_path(s)],
        "ml_ndcg_at_k" | "ml_precision_recall_at_k" => vec![over_k(s)],
        "ml_receptive_field" => vec![growth(s)],
        "ml_gradient_descent_step_linear" | "ml_knn_regress" | "ml_logistic_regression_fit" | "ml_ridge_regression" => {
            vec![fit(s)]
        }
        "ml_roc_auc" => vec![roc(s)],
        "ml_roc_curve" => vec![roc(s), b::thresholds(s)],
        "ml_average_precision" => vec![pr(s)],
        "ml_expected_calibration_error" => vec![reliability(s)],
        "ml_logistic_loss_gradient" | "ml_mse_gradient" => vec![landscape(s)],
        "ml_train_test_split_sizes" => vec![split_field(s), c::sizes(s)],
        _ => {
            let mut v = b::live(s);
            v.extend(c::live(s));
            v
        }
    };
    v.into_iter().flatten().map(|f| with_answer(s, f)).collect()
}

/// The answer's first number (its value, else the asked output's first entry, else its first number), stated in every
/// figure: a figure whose text does not hold it gets "Returned: ..." at the end of its caption.
fn with_answer(s: &Solved, mut f: Figure) -> Figure {
    let target = s.a.target.clone().or(s.a.solved_for.clone()).unwrap_or_default();
    let first = |v: &serde_json::Value| -> Option<f64> {
        match v {
            serde_json::Value::Number(n) => n.as_f64(),
            serde_json::Value::Array(a) => a.first().and_then(|x| x.as_f64()),
            _ => None,
        }
    };
    let (name, value, shown) = if let Some(v) = s.a.value.filter(|v| v.is_finite()) {
        (target.clone(), v, num(v))
    } else if let Some(e) = s.a.values.iter().find(|e| e.0 == target && first(&e.1).is_some()) {
        let v = first(&e.1).unwrap();
        (e.0.clone(), v, match &e.1 { serde_json::Value::Array(a) => format!("({})", a.iter().take(8)
            .map(|x| x.as_f64().map(num).unwrap_or_default()).collect::<Vec<_>>().join(", ")), _ => num(v) })
    } else if let Some(e) = s.a.values.iter().find(|e| e.1.as_f64().is_some()) {
        let v = e.1.as_f64().unwrap();
        (e.0.clone(), v, num(v))
    } else if let Some(e) = s.a.values.iter().find(|e| e.1.is_array() && first(&e.1).is_some()) {
        let v = first(&e.1).unwrap();
        let a = e.1.as_array().unwrap();
        (e.0.clone(), v, format!("({})", a.iter().take(8).map(|x| x.as_f64().map(num).unwrap_or_default()).collect::<Vec<_>>().join(", ")))
    } else {
        return f;
    };
    if !value.is_finite() || f.svg.contains(&num(value)) {
        return f;
    }
    let add = format!(" Returned: {} = {}.", name.replace('_', " "), shown);
    let esc = add.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;");
    // before the visitor's inputs, which end every caption
    if f.caption.contains(" Your inputs:") && f.svg.contains(" Your inputs:") {
        f.caption = f.caption.replacen(" Your inputs:", &format!("{add} Your inputs:"), 1);
        f.svg = f.svg.replacen(" Your inputs:", &format!("{esc} Your inputs:"), 1);
    } else {
        f.caption.push_str(&add);
        f.svg = f.svg.replacen("</desc>", &format!("{esc}</desc>"), 1);
    }
    f
}

// ------------------------------------------------------------------------------------------------ helpers
pub fn fl(x: f64) -> Input {
    Input::Float(x)
}
pub fn it(x: i64) -> Input {
    Input::Int(x)
}
pub fn tx(x: &str) -> Input {
    Input::Text(x.to_string())
}
/// A list as an input ("[1, 2, 3]").
pub fn tl(v: &[f64]) -> Input {
    Input::Text(format!("[{}]", pl::text_list(v)))
}
/// A matrix as an input.
pub fn tm(m: &[Vec<f64>]) -> Input {
    Input::Text(pl::text_mat(m))
}
/// The kit's charts with their heading kept inside the canvas.
pub fn hbars(labels: &[String], values: &[f64], hl: &[usize], xlab: &str, ylab: &str, header: &str) -> String {
    k::bars(labels, values, hl, xlab, ylab, &pl::clip(header, 44))
}
pub fn hheat(m: &[Vec<f64>], rows: &[String], cols: &[String], header: &str) -> String {
    k::heat(m, rows, cols, &pl::clip(header, 44))
}
#[allow(clippy::too_many_arguments)]
pub fn hfield(xs: &[f64], ys: &[f64], vals: &[Vec<Option<f64>>], mine: (f64, f64), xlab: &str, ylab: &str, header: &str,
              log: (bool, bool)) -> String {
    k::field(xs, ys, vals, mine, xlab, ylab, &pl::clip(header, 30), log)
}
#[allow(clippy::too_many_arguments)]
pub fn hdecision(xs: &[f64], ys: &[f64], cells: &[(usize, usize, String)], names: &[String], mine: (f64, f64), xlab: &str,
                 ylab: &str, header: &str) -> String {
    k::decision(xs, ys, cells, names, mine, xlab, ylab, &pl::clip(header, 30))
}
pub fn htable(head: &[String], rows: &[Vec<String>], mark: Option<usize>, header: &str) -> (String, f64) {
    k::table(head, rows, mark, &pl::clip(header, 52))
}
pub fn o1(s: &Solved, key: &str, set: &[(&str, Input)], out: &str) -> Option<f64> {
    let a = s.solve_other(key, set)?;
    k::num_of(&a, out)
}
pub fn ans(a: &Answer, sym: &str) -> Option<f64> {
    k::num_of(a, sym)
}
/// A number input (real or whole) as read.
pub fn nin(s: &Solved, sym: &str) -> Option<f64> {
    s.inp(sym).or_else(|| match s.val(sym)? {
        Val::Int(i) => Some(i as f64),
        Val::Num(x) if x.is_finite() => Some(x),
        Val::Bool(b) => Some(if b { 1.0 } else { 0.0 }),
        _ => None,
    })
}
/// A matrix output (a list of lists of numbers).
pub fn mat_of(a: &Answer, sym: &str) -> Option<Vec<Vec<f64>>> {
    let v = a.values.iter().find(|x| x.0 == sym)?.1.as_array()?;
    let mut out = vec![];
    for r in v {
        let row: Vec<f64> = match r {
            Value::Array(xs) => xs.iter().map(|x| x.as_f64()).collect::<Option<Vec<_>>>()?,
            Value::Number(n) => vec![n.as_f64()?],
            _ => return None,
        };
        if row.iter().any(|x| !x.is_finite()) {
            return None;
        }
        out.push(row);
    }
    (!out.is_empty()).then_some(out)
}
pub fn mat_out(s: &Solved, sym: &str) -> Option<Vec<Vec<f64>>> {
    mat_of(s.a, sym)
}
/// A list output of text or numbers as words.
pub fn words_of(s: &Solved, sym: &str) -> Option<Vec<String>> {
    let v = s.outj(sym)?.as_array()?;
    Some(v.iter().map(|x| match x {
        Value::String(t) => t.clone(),
        Value::Number(n) => num(n.as_f64().unwrap_or(f64::NAN)),
        other => other.to_string(),
    }).collect())
}

/// Curves (a legend when several) with optional marked points (x, y, label), in the generic frame.
pub fn curves(series: &[(String, Vec<(f64, f64)>)], marks: &[(f64, f64, String)], xlab: &str, ylab: &str, header: &str,
              log_y: bool) -> Option<String> {
    let xs: Vec<f64> = series.iter().flat_map(|s| s.1.iter().map(|p| p.0)).chain(marks.iter().map(|m| m.0)).collect();
    let ys: Vec<f64> = series.iter().flat_map(|s| s.1.iter().map(|p| p.1)).chain(marks.iter().map(|m| m.1)).collect();
    if xs.len() < 2 {
        return None;
    }
    let (x0, x1) = pl::minmax(&xs);
    let (y0, y1) = pl::minmax(&ys);
    let ya = if log_y && y0 > 0.0 { pl::Ax::log(y0, y1, pl::Y1, pl::Y0) } else { pl::Ax::new(y0, y1, pl::Y1, pl::Y0, true) };
    let mut p = Plot::axes(pl::Ax::new(x0, x1, pl::X0, pl::X1, false), ya);
    p.clip_on();
    for (i, (_, pts)) in series.iter().enumerate() {
        p.path(pts, LINE[i % 4]);
        if pts.len() <= 12 {
            for q in pts {
                p.dot(q.0, q.1, 3.0, DOT[i % 4]);
            }
        }
    }
    p.unclip();
    for (x, y, l) in marks {
        p.dot(*x, *y, 5.5, "axp-m");
        if !l.is_empty() {
            let (px, py) = (p.x(*x), p.y(*y));
            p.place(px + 3.0, (py - 9.0).max(pl::Y0 + 14.0), 15.0, 3, "axp-mt", l);
        }
    }
    if series.len() > 1 {
        let items: Vec<(String, &str)> = series.iter().enumerate().map(|(i, s)| (s.0.clone(), BAR[i % 4])).collect();
        p.legend_row(&items);
    }
    p.frame(xlab, ylab, header);
    Some(p.body())
}

// ------------------------------------------------------------------------------------------------ losses
/// The loss of one sample over a range, each point one run of the model on that sample alone.
fn loss(s: &Solved) -> Option<Figure> {
    let key = s.key;
    let one = |set: &[(&str, Input)], out: &str| -> Option<f64> { s.with_input(set).and_then(|a| ans(&a, out)) };
    let mut series = vec![];
    let mut marks = vec![];
    let (title, xlab, ylab, caption, header);
    match key {
        "ml_bce_with_logits" | "ml_log_loss_binary" | "ml_brier_score" => {
            let logit = key == "ml_bce_with_logits";
            let (ysym, xsym, out) = match key {
                "ml_bce_with_logits" => ("y", "z", "loss"),
                "ml_log_loss_binary" => ("y_true", "p", "log_loss"),
                _ => ("y_true", "p", "brier"),
            };
            let (yv, xv) = (s.vec(ysym)?, s.vec(xsym)?);
            if yv.len() != xv.len() || yv.len() > 30 {
                return None;
            }
            let grid: Vec<f64> = if logit { k::span(-5.0, 5.0, 61, false) } else if key == "ml_brier_score" {
                k::span(0.0, 1.0, 41, false)
            } else {
                k::span(0.02, 0.98, 49, false)
            };
            for lab in [1.0, 0.0] {
                let pts: Vec<(f64, f64)> = grid.iter().filter_map(|g| one(&[(ysym, tl(&[lab])), (xsym, tl(&[*g]))], out).map(|v| (*g, v))).collect();
                series.push((format!("{ysym} = {}", num(lab)), pts));
            }
            for (a, z) in yv.iter().zip(&xv) {
                if let Some(v) = one(&[(ysym, tl(&[*a])), (xsym, tl(&[*z]))], out) {
                    marks.push((*z, v, String::new()));
                }
            }
            let tot = s.out(out)?;
            title = match key {
                "ml_bce_with_logits" => "Binary cross-entropy against the logit",
                "ml_log_loss_binary" => "Log loss against the predicted probability",
                _ => "Brier score against the predicted probability",
            };
            xlab = if logit { "logit z" } else { "predicted probability p" };
            ylab = "loss of one sample";
            header = format!("mean = {}", num(tot));
            caption = format!("The loss of one sample as its {} runs over the axis, for {ysym} = 1 and {ysym} = 0 ({} runs each). \
                               Marked: the visitor's samples ({ysym} {}, {xsym} {}), one run each; their mean is the returned \
                               value {}.", if logit { "logit" } else { "predicted probability" }, grid.len(), list(&yv, 10),
                              list(&xv, 10), num(tot));
        }
        "ml_focal_loss" => {
            let (yv, zv, g0) = (s.vec("y")?, s.vec("z")?, s.inp("gamma")?);
            if yv.len() != zv.len() {
                return None;
            }
            let grid = k::span(-4.0, 4.0, 41, false);
            let mut gs = vec![0.0, 1.0, 2.0, 5.0];
            if !gs.contains(&g0) {
                gs.push(g0);
            }
            for g in gs {
                let pts: Vec<(f64, f64)> = grid.iter().filter_map(|z| one(&[("y", tl(&[1.0])), ("z", tl(&[*z])), ("gamma", fl(g))], "loss").map(|v| (*z, v))).collect();
                series.push((format!("γ = {}", num(g)), pts));
            }
            let tot = s.out("loss")?;
            title = "Focal loss: down-weighting easy samples";
            xlab = "logit z of a positive sample";
            ylab = "focal loss";
            header = format!("visitor's loss {} (γ = {})", num(tot), num(g0));
            caption = format!("Focal loss of one positive sample (y = 1) as its logit z runs from -4 to 4, for several focusing \
                               parameters γ with α = {} (41 runs each): larger γ silences the well-classified samples. The \
                               visitor's samples give the returned loss {}.", num(s.inp("alpha")?), num(tot));
        }
        "ml_hinge_loss" => {
            let (yv, sc) = (s.vec("y_true")?, s.vec("scores")?);
            if yv.len() != sc.len() {
                return None;
            }
            let grid = k::span(-2.0, 3.0, 51, false);
            let mut h = vec![];
            let mut q = vec![];
            for m in &grid {
                if let Some(a) = s.with_input(&[("y_true", tl(&[1.0, -1.0])), ("scores", tl(&[*m, -*m]))]) {
                    if let (Some(x), Some(y)) = (ans(&a, "hinge"), ans(&a, "squared_hinge")) {
                        h.push((*m, x));
                        q.push((*m, y));
                    }
                }
            }
            series.push(("hinge".into(), h));
            series.push(("squared hinge".into(), q));
            for (a, b) in yv.iter().zip(&sc) {
                if let Some(v) = one(&[("y_true", tl(&[*a, -*a])), ("scores", tl(&[*b, -*b]))], "hinge") {
                    marks.push((a * b, v, String::new()));
                }
            }
            title = "Hinge loss against the margin";
            xlab = "signed margin y·f(x)";
            ylab = "loss of one sample";
            header = format!("hinge {}, squared {}", num(s.out("hinge")?), num(s.out("squared_hinge")?));
            caption = format!("Hinge and squared hinge loss of one sample against its signed margin from -2 to 3 (51 runs each; \
                               each run holds a +1 and a -1 sample at the same margin, since both classes must occur). Marked: \
                               the visitor's samples (labels {}, scores {}) at their margins; returned means: hinge {}, squared \
                               hinge {}.", list(&yv, 10), list(&sc, 10), num(s.out("hinge")?), num(s.out("squared_hinge")?));
        }
        "ml_huber_loss" | "ml_quantile_loss" => {
            let huber = key == "ml_huber_loss";
            let grid = if huber { k::span(-3.0, 3.0, 61, false) } else { k::span(-2.0, 2.0, 41, false) };
            if huber {
                let (mut a1, mut a2) = (vec![], vec![]);
                for r in &grid {
                    if let Some(a) = s.with_input(&[("y_true", tl(&[*r])), ("y_pred", tl(&[0.0]))]) {
                        if let (Some(x), Some(y)) = (ans(&a, "huber"), ans(&a, "pseudo_huber")) {
                            a1.push((*r, x));
                            a2.push((*r, y));
                        }
                    }
                }
                series.push(("Huber".into(), a1));
                series.push(("pseudo-Huber".into(), a2));
                header = format!("δ = {}: Huber {}", num(s.inp("delta")?), num(s.out("huber")?));
                caption = format!("Huber and pseudo-Huber loss of one sample as the residual y - ŷ runs from -3 to 3 (δ = {}, 61 \
                                   runs each): quadratic inside ±δ, linear outside. The visitor's samples give Huber {}, smooth L1 \
                                   {}, pseudo-Huber {}.", num(s.inp("delta")?), num(s.out("huber")?), num(s.out("smooth_l1")?),
                                  num(s.out("pseudo_huber")?));
                title = "Huber loss: quadratic near zero, linear beyond δ";
            } else {
                let a0 = s.inp("alpha")?;
                let mut al = vec![0.1, 0.5, 0.9];
                if !al.iter().any(|x| (x - a0).abs() < 1e-12) {
                    al.push(a0);
                }
                for a in al {
                    let pts: Vec<(f64, f64)> = grid.iter().filter_map(|r| one(&[("y_true", tl(&[*r])), ("y_pred", tl(&[0.0])), ("alpha", fl(a))], "pinball").map(|v| (*r, v))).collect();
                    series.push((format!("α = {}", num(a)), pts));
                }
                header = format!("α = {}: pinball {}", num(a0), num(s.out("pinball")?));
                caption = format!("Pinball (quantile) loss of one sample against the residual y - ŷ from -2 to 2, for several α (41 \
                                   runs each): under-prediction costs α per unit, over-prediction 1 - α. The visitor's samples \
                                   (α = {}) give the returned {}.", num(a0), num(s.out("pinball")?));
                title = "Pinball loss for several quantiles";
            }
            xlab = "residual y - ŷ";
            ylab = "loss of one sample";
        }
        _ => {
            // Tweedie deviance of one sample of true value y as the prediction runs
            let (yv, p0) = (s.vec("y_true")?, s.inp("power")?);
            let y0 = yv.iter().cloned().find(|v| *v > 0.0)?;
            let grid = k::span(0.25 * y0, 3.0 * y0, 40, false);
            let mut ps = vec![0.0, 1.0, 1.5, 2.0];
            if !ps.contains(&p0) {
                ps.push(p0);
            }
            for pw in ps {
                let pts: Vec<(f64, f64)> = grid.iter().filter_map(|g| one(&[("y_true", tl(&[y0])), ("y_pred", tl(&[*g])), ("power", fl(pw))], "deviance").map(|v| (*g, v))).collect();
                series.push((format!("power {}", num(pw)), pts));
            }
            title = "Tweedie deviance for several powers";
            xlab = "prediction ŷ";
            ylab = "deviance of one sample";
            header = format!("y = {}; visitor's deviance {}", num(y0), num(s.out("deviance")?));
            caption = format!("Tweedie deviance of one sample with true value {} as the prediction runs from {} to {}, for powers 0 \
                               (squared error), 1 (Poisson), 1.5 and 2 (gamma), 40 runs each: all vanish at ŷ = y. The visitor's \
                               samples (power {}) give the returned {}.", num(y0), num(0.25 * y0), num(3.0 * y0), num(p0),
                              num(s.out("deviance")?));
        }
    }
    series.retain(|x| x.1.len() >= 2);
    let body = curves(&series, &marks, xlab, ylab, &header, false)?;
    Some(s.chart("loss", title, &caption, &body))
}

// ------------------------------------------------------------------------------------------------ schedules
fn schedule(s: &Solved) -> Option<Figure> {
    let lr0 = s.out("lr")?;
    let mut series = vec![];
    let (title, caption, header, xsym, x0, logy);
    match s.key {
        "ml_cosine_lr_schedule" => {
            let (lmax, emin, t) = (s.inp("lr_max")?, s.inp("eta_min")?, nin(s, "T_max")?);
            let e0 = nin(s, "epoch")?;
            if !(t >= 1.0) || t > 100_000.0 {
                return None;
            }
            let es: Vec<f64> = (0..=50).map(|i| (t * i as f64 / 50.0).round()).collect();
            let other = if emin == 0.0 { 0.2 * lmax } else { 0.0 };
            for em in [emin, other] {
                let pts: Vec<(f64, f64)> = es.iter().filter_map(|e| s.with_input(&[("epoch", it(*e as i64)), ("eta_min", fl(em))])
                    .and_then(|a| ans(&a, "lr")).map(|v| (*e, v))).collect();
                series.push((format!("η_min = {}", num(em)), pts));
            }
            title = "Cosine learning-rate schedule";
            caption = format!("Cosine annealing from lr_max = {} over T_max = {} epochs down to η_min = {} and {} (51 runs each). \
                               Marked: the visitor's epoch {}, lr = {}.", num(lmax), num(t), num(emin), num(other), num(e0), num(lr0));
            header = format!("epoch {}: lr = {}", num(e0), num(lr0));
            xsym = "epoch";
            x0 = e0;
            logy = false;
        }
        "ml_step_lr_schedule" => {
            let (l0, step, g) = (s.inp("lr0")?, nin(s, "step_size")?, s.inp("gamma")?);
            let e0 = nin(s, "epoch")?;
            if !(step >= 1.0) {
                return None;
            }
            let top = (e0 * 1.5).max(step * 3.5).max(10.0).round();
            let es: Vec<f64> = (0..=80).map(|i| (top * i as f64 / 80.0).round()).collect();
            let other = if (g - 0.5).abs() < 1e-12 { 0.1 } else { 0.5 };
            for gg in [g, other] {
                let pts: Vec<(f64, f64)> = es.iter().filter_map(|e| s.with_input(&[("epoch", it(*e as i64)), ("gamma", fl(gg))])
                    .and_then(|a| ans(&a, "lr")).map(|v| (*e, v))).collect();
                series.push((format!("γ = {}", num(gg)), pts));
            }
            title = "Step learning-rate schedule";
            caption = format!("Step decay from lr0 = {}, multiplied by γ every {} epochs, for γ = {} and {} over {} epochs (81 runs \
                               each). Marked: the visitor's epoch {}, lr = {} after {} decays.", num(l0), num(step), num(g),
                              num(other), num(top), num(e0), num(lr0), num(s.out("decays")?));
            header = format!("epoch {}: lr = {}", num(e0), num(lr0));
            xsym = "epoch";
            x0 = e0;
            logy = series.iter().all(|x| x.1.iter().all(|p| p.1 > 0.0));
        }
        _ => {
            let (lmax, w, total) = (s.inp("lr_max")?, nin(s, "warmup")?, nin(s, "total")?);
            let st = nin(s, "step")?;
            if !(total >= 2.0) || total > 100_000.0 {
                return None;
            }
            let ss: Vec<f64> = (0..=60).map(|i| (total * i as f64 / 60.0).round()).collect();
            let other = if w * 2.5 < total { (w * 2.5).round() } else { (w / 2.0).round() };
            for ww in [w, other] {
                let pts: Vec<(f64, f64)> = ss.iter().filter_map(|e| s.with_input(&[("step", it(*e as i64)), ("warmup", it(ww as i64))])
                    .and_then(|a| ans(&a, "lr")).map(|v| (*e, v))).collect();
                series.push((format!("warm-up {} steps", num(ww)), pts));
            }
            title = "Warm-up then cosine decay";
            caption = format!("Linear warm-up from {} × lr_max to lr_max = {}, then cosine decay to {} at step {}, for warm-ups of \
                               {} and {} steps (61 runs each). Marked: the visitor's step {}, lr = {}.",
                              num(s.inp("start_factor")?), num(lmax), num(s.inp("eta_min")?), num(total), num(w), num(other),
                              num(st), num(lr0));
            header = format!("step {}: lr = {}", num(st), num(lr0));
            xsym = "step";
            x0 = st;
            logy = false;
        }
    }
    series.retain(|x| x.1.len() >= 2);
    let body = curves(&series, &[(x0, lr0, String::new())], xsym, "learning rate", &header, logy)?;
    Some(s.chart("schedule", title, &caption, &body))
}

// ------------------------------------------------------------------------------------------------ optimiser paths
/// The bowl L(w) = ½(w₁² + 10 w₂²) chosen for the figure: its gradient, fed to every step of the model.
fn bowl_grad(w: &[f64]) -> Vec<f64> {
    vec![w[0], 10.0 * w[1]]
}

fn opt_path(s: &Solved) -> Option<Figure> {
    let w0 = s.vec("w")?;
    if w0.len() != 2 {
        return None;
    }
    let steps = 40;
    let (variants, vname, state): (Vec<f64>, &str, Vec<&str>) = match s.key {
        "ml_adam_step" => {
            let lr = s.inp("lr")?;
            (vec![lr, lr * 10.0], "lr", vec!["m", "v"])
        }
        "ml_adamw_step" => (vec![0.0, s.inp("weight_decay")?], "weight_decay", vec!["m", "v"]),
        _ => {
            let mut m = vec![0.0, 0.5, 0.9];
            let m0 = s.inp("momentum")?;
            if !m.iter().any(|x| (x - m0).abs() < 1e-12) {
                m.push(m0);
            }
            (m, "momentum", vec!["buf"])
        }
    };
    let mut paths = vec![];
    for var in &variants {
        let mut w = w0.clone();
        let mut st: Vec<Vec<f64>> = state.iter().map(|_| vec![0.0, 0.0]).collect();
        let mut t = nin(s, "t").unwrap_or(1.0) as i64;
        let mut pts = vec![(w[0], w[1])];
        for _ in 0..steps {
            let g = bowl_grad(&w);
            let mut set: Vec<(&str, Input)> = vec![("w", tl(&w)), ("grad", tl(&g)), (vname, fl(*var))];
            for (i, n) in state.iter().enumerate() {
                set.push((n, tl(&st[i])));
            }
            if s.key != "ml_sgd_momentum_step" {
                set.push(("t", it(t)));
            }
            let Some(a) = s.with_input(&set) else { break };
            let Some(wn) = k::list_of(&a, "w_new") else { break };
            for (i, n) in state.iter().enumerate() {
                if let Some(v) = k::list_of(&a, &format!("{n}_new")) {
                    st[i] = v;
                }
            }
            if wn.len() != 2 {
                break;
            }
            w = wn;
            t += 1;
            pts.push((w[0], w[1]));
        }
        if pts.len() >= 3 {
            paths.push((format!("{vname} = {}", num(*var)), pts));
        }
    }
    if paths.is_empty() {
        return None;
    }
    let mut xs: Vec<f64> = paths.iter().flat_map(|p| p.1.iter().map(|q| q.0)).collect();
    let mut ys: Vec<f64> = paths.iter().flat_map(|p| p.1.iter().map(|q| q.1)).collect();
    xs.extend([0.0, w0[0].abs(), -w0[0].abs()]);
    ys.extend([0.0, w0[1].abs(), -w0[1].abs()]);
    let mut p = Plot::equal(&xs, &ys);
    p.clip_on();
    // level sets of the chosen bowl through the start and inside it
    let l0 = 0.5 * (w0[0] * w0[0] + 10.0 * w0[1] * w0[1]);
    for f in [0.05, 0.2, 0.5, 1.0] {
        let l = l0 * f;
        let ell: Vec<(f64, f64)> = (0..=72).map(|i| {
            let th = std::f64::consts::TAU * i as f64 / 72.0;
            ((2.0 * l).sqrt() * th.cos(), (2.0 * l / 10.0).sqrt() * th.sin())
        }).collect();
        p.path(&ell, "axp-g");
    }
    for (i, (_, pts)) in paths.iter().enumerate() {
        p.path(pts, LINE[i % 3]);
        for q in pts.iter().step_by(2) {
            p.dot(q.0, q.1, 2.6, DOT[i % 3]);
        }
    }
    p.unclip();
    p.dot(w0[0], w0[1], 6.0, "axp-m");
    p.dot(0.0, 0.0, 4.0, "axp-kd");
    let items: Vec<(String, &str)> = paths.iter().enumerate().map(|(i, x)| (x.0.clone(), BAR[i % 3])).collect();
    p.legend_row(&items);
    p.frame("w₁", "w₂", &format!("{} steps from w = ({}, {})", steps, num(w0[0]), num(w0[1])));
    let ends = paths.iter().map(|x| {
        let e = x.1[x.1.len() - 1];
        format!("{}: ({}, {})", x.0, num(e.0), num(e.1))
    }).collect::<Vec<_>>().join("; ");
    let name = match s.key { "ml_adam_step" => "Adam", "ml_adamw_step" => "AdamW", _ => "SGD with momentum" };
    Some(s.chart("path", &format!("{name} steps across a stretched bowl"),
                 &format!("{steps} {name} steps from the visitor's start w = ({}, {}) on the loss L(w) = ½(w₁² + 10 w₂²) chosen \
                           for this figure (its gradient (w₁, 10 w₂) is passed to every step; grey ellipses are its level sets), \
                           for {vname} = {}. Each dot is the w the model returned, its returned state fed to the next step. \
                           Ends: {ends}.", num(w0[0]), num(w0[1]),
                          variants.iter().map(|v| num(*v)).collect::<Vec<_>>().join(", ")), &p.body()))
}

fn lasso_path(s: &Solved) -> Option<Figure> {
    let a0 = s.inp("alpha")?;
    let w0 = s.out_list("w")?;
    let alphas = k::span(0.001, (a0 * 50.0).max(5.0), 7, true);
    let mut feats: Vec<Vec<(f64, f64)>> = vec![vec![]; w0.len()];
    for a in &alphas {
        if let Some(r) = s.with(&[("alpha", *a)]) {
            if let Some(w) = k::list_of(&r, "w") {
                for (j, v) in w.iter().enumerate().take(feats.len()) {
                    feats[j].push((*a, *v));
                }
            }
        }
    }
    if feats.iter().any(|f| f.len() < 2) || feats.len() > 8 {
        return None;
    }
    let series: Vec<(String, Vec<(f64, f64)>)> = feats.into_iter().enumerate().map(|(j, f)| (format!("w{}", j + 1), f)).collect();
    let xs: Vec<f64> = series.iter().flat_map(|x| x.1.iter().map(|p| p.0)).collect();
    let ys: Vec<f64> = series.iter().flat_map(|x| x.1.iter().map(|p| p.1)).chain(w0.iter().cloned()).collect();
    let (x0, x1) = pl::minmax(&xs);
    let mut p = Plot::axes(pl::Ax::log(x0, x1, pl::X0, pl::X1), pl::Ax::new(pl::minmax(&ys).0.min(0.0), pl::minmax(&ys).1.max(0.0), pl::Y1, pl::Y0, true));
    p.hline(0.0, "axp-c4", true);
    for (i, (_, pts)) in series.iter().enumerate() {
        p.path(pts, LINE[i % 4]);
    }
    p.vline(a0, "axp-c4", true);
    for v in &w0 {
        p.dot(a0, *v, 5.0, "axp-m");
    }
    let items: Vec<(String, &str)> = series.iter().enumerate().map(|(i, x)| (x.0.clone(), BAR[i % 4])).collect();
    p.legend_row(&items);
    p.frame("penalty α (logarithmic)", "coefficient", &format!("α = {}: {} nonzero", num(a0), num(s.out("n_nonzero")?)));
    Some(s.chart("path", "Lasso path: coefficients against the penalty",
                 &format!("Lasso coefficients for the stated X and y as the penalty α runs from {} to {} (7 runs): weights reach \
                           exactly zero one after another. Marked: the visitor's α = {}, returned w = ({}), b = {}, {} nonzero.",
                          num(x0), num(x1), num(a0), list(&w0, 8), num(s.out("b")?), num(s.out("n_nonzero")?)), &p.body()))
}

// ------------------------------------------------------------------------------------------------ over k, growth
fn over_k(s: &Solved) -> Option<Figure> {
    let rel = s.vec("rel")?;
    let n = rel.len();
    if n < 1 || n > 60 {
        return None;
    }
    let k0 = nin(s, "k")?;
    let mut series = vec![];
    let (title, caption, header);
    if s.key == "ml_ndcg_at_k" {
        let g0 = s.txt("gain")?;
        for g in ["linear", "exponential"] {
            let pts: Vec<(f64, f64)> = (1..=n).filter_map(|kk| s.with_input(&[("k", it(kk as i64)), ("gain", tx(g))])
                .and_then(|a| ans(&a, "ndcg")).map(|v| (kk as f64, v))).collect();
            series.push((format!("{g} gain"), pts));
        }
        let nd = s.out("ndcg")?;
        title = "NDCG at each cut-off";
        header = format!("k = {}, {g0} gain: NDCG = {}", num(k0), num(nd));
        caption = format!("NDCG at each cut-off k for the ranked relevance grades [{}], linear and exponential gain, one run per \
                           point. Visitor's example (k = {}, {g0} gain): DCG {}, ideal {}, NDCG {}.", list(&rel, 16), num(k0),
                          num(s.out("dcg")?), num(s.out("idcg")?), num(nd));
    } else {
        let (mut pr, mut rc) = (vec![], vec![]);
        for kk in 1..=n {
            if let Some(a) = s.with_input(&[("k", it(kk as i64))]) {
                if let (Some(x), Some(y)) = (ans(&a, "precision_at_k"), ans(&a, "recall_at_k")) {
                    pr.push((kk as f64, x));
                    rc.push((kk as f64, y));
                }
            }
        }
        series.push(("precision@k".to_string(), pr));
        series.push(("recall@k".to_string(), rc));
        title = "Precision and recall at k";
        header = format!("k = {}: precision {}, recall {}", num(k0), num(s.out("precision_at_k")?), num(s.out("recall_at_k")?));
        caption = format!("Precision and recall among the top k of the ranked relevance list [{}] ({} relevant items in all), one \
                           run per k. Visitor's k = {}: precision {}, recall {}, {} hits.", list(&rel, 16),
                          num(nin(s, "n_relevant")?), num(k0), num(s.out("precision_at_k")?), num(s.out("recall_at_k")?),
                          num(s.out("hits")?));
    }
    series.retain(|x| x.1.len() >= 2);
    let y0 = series.first()?.1.iter().find(|p| p.0 == k0).map(|p| p.1)?;
    let body = curves(&series, &[(k0, y0, String::new())], "cut-off k", "score", &header, false)?;
    Some(s.chart("over-k", title, &caption, &body))
}

fn growth(s: &Solved) -> Option<Figure> {
    let ks = s.vec("kernels")?;
    let rf = s.out("receptive_field")?;
    let kern = *ks.first()?;
    let depth = ks.len();
    let mut series = vec![];
    for (name, mode) in [("stride 1", 0), ("stride 2", 1), ("dilation doubling", 2)] {
        let mut pts = vec![];
        for d in 1..=8usize {
            let kv = vec![kern; d];
            let sv: Vec<f64> = (0..d).map(|_| if mode == 1 { 2.0 } else { 1.0 }).collect();
            let dv: Vec<f64> = (0..d).map(|i| if mode == 2 { 2f64.powi(i as i32) } else { 1.0 }).collect();
            if let Some(v) = s.with_input(&[("kernels", tl(&kv)), ("strides", tl(&sv)), ("dilations", tl(&dv))]).and_then(|a| ans(&a, "receptive_field")) {
                pts.push((d as f64, v));
            }
        }
        series.push((name.to_string(), pts));
    }
    series.retain(|x| x.1.len() >= 2);
    let body = curves(&series, &[(depth as f64, rf, format!("{depth} layers: {}", num(rf)))], "number of layers",
                      "receptive field", &format!("receptive field {} (jump {})", num(rf), num(s.out("jump")?)), false)?;
    Some(s.chart("growth", "Receptive field growing with depth",
                 &format!("Receptive field of a stack of 1 to 8 convolutions with {}-wide kernels: stride 1, stride 2, and dilations \
                           doubling per layer; one run per point. Marked: the visitor's {depth} layers (kernels {}, strides {}, \
                           dilations {}), receptive field {} (jump {}).", num(kern), list(&ks, 8), list(&s.vec("strides")?, 8),
                          list(&s.vec("dilations")?, 8), num(rf), num(s.out("jump")?)), &body))
}

// ------------------------------------------------------------------------------------------------ fits through 1-D data
fn fit(s: &Solved) -> Option<Figure> {
    let xm = s.mat("X")?;
    let y = s.vec("y")?;
    if xm.iter().any(|r| r.len() != 1) || xm.len() != y.len() || y.len() > 200 {
        return None;
    }
    let x: Vec<f64> = xm.iter().map(|r| r[0]).collect();
    let (lo, hi) = pl::minmax(&x);
    let pad = (hi - lo).max(1e-9) * 0.1;
    let mut series = vec![];
    let mut marks = vec![];
    let (title, caption, header, ylab);
    match s.key {
        "ml_gradient_descent_step_linear" => {
            let (mut w, mut b) = (s.vec("w")?, s.inp("b")?);
            if w.len() != 1 {
                return None;
            }
            let mut losses = vec![];
            for i in 0..5 {
                let a = s.with_input(&[("w", tl(&w)), ("b", fl(b))])?;
                let wn = k::list_of(&a, "w_new")?;
                let bn = ans(&a, "b_new")?;
                losses.push(ans(&a, "loss_after")?);
                w = wn;
                b = bn;
                series.push((format!("step {}", i + 1), vec![(lo - pad, w[0] * (lo - pad) + b), (hi + pad, w[0] * (hi + pad) + b)]));
            }
            title = "Gradient descent pulling a line onto the data";
            caption = format!("Five gradient-descent steps for a line through x = {}, y = {} from w = {}, b = {} (lr = {}), each a \
                               run of the model from the w, b the previous run returned; MSE after each step: {}.", list(&x, 10),
                              list(&y, 10), list(&s.vec("w")?, 2), num(s.inp("b")?), num(s.inp("lr")?), list(&losses, 5));
            header = format!("MSE {} after 5 steps", num(*losses.last()?));
            ylab = "y";
        }
        "ml_ridge_regression" => {
            let a0 = s.inp("alpha")?;
            let mut al = vec![0.01, 1.0, 10.0, 100.0];
            if !al.iter().any(|v| (v - a0).abs() < 1e-12) {
                al.push(a0);
            }
            let mut parts = vec![];
            for a in al {
                let r = s.with(&[("alpha", a)])?;
                let (w, b) = (k::list_of(&r, "w")?, ans(&r, "b")?);
                parts.push(format!("α = {}: w = {}, b = {}", num(a), num(w[0]), num(b)));
                series.push((format!("α = {}", num(a)), vec![(lo - pad, w[0] * (lo - pad) + b), (hi + pad, w[0] * (hi + pad) + b)]));
            }
            title = "Ridge regression: slope shrinking with the penalty";
            caption = format!("Ridge lines through x = {}, y = {} for several penalties, one run each ({}): the slope shrinks toward \
                               zero as α grows.", list(&x, 10), list(&y, 10), parts.join("; "));
            header = format!("visitor's α = {}: train MSE {}", num(a0), num(s.out("train_mse")?));
            ylab = "y";
        }
        "ml_knn_regress" => {
            let q = s.vec("x")?;
            let yh = s.out("prediction")?;
            let grid = k::span(lo - pad, hi + pad, 41, false);
            for wt in ["uniform", "distance"] {
                let pts: Vec<(f64, f64)> = grid.iter().filter_map(|g| s.with_input(&[("x", tl(&[*g])), ("weights", tx(wt))])
                    .and_then(|a| ans(&a, "prediction")).map(|v| (*g, v))).collect();
                series.push((format!("{wt} weights"), pts));
            }
            marks.push((*q.first()?, yh, format!("ŷ = {}", num(yh))));
            title = "k-nearest-neighbour regression";
            caption = format!("k-nearest-neighbour regression (k = {}) through the stated points x = {}, y = {}: the prediction at 41 \
                               query points for uniform and distance weights (one run each). Marked: the visitor's x = {}, ŷ = {}.",
                              num(nin(s, "k")?), list(&x, 10), list(&y, 10), num(q[0]), num(yh));
            header = format!("k = {}, {}", num(nin(s, "k")?), s.txt("metric")?);
            ylab = "y";
        }
        _ => {
            // logistic regression: the predicted probability from the model's own predict capability, one run
            let (w, b) = (s.out_list("w")?, s.out("b")?);
            if w.len() != 1 {
                return None;
            }
            let grid = k::span(lo - pad, hi + pad, 81, false);
            let gm: Vec<Vec<f64>> = grid.iter().map(|g| vec![*g]).collect();
            let a = s.solve_other("ml_logistic_regression_predict", &[("w", tl(&w)), ("b", fl(b)), ("X", tm(&gm))])?;
            let pr = k::list_of(&a, "probabilities")?;
            series.push(("P(y = 1)".to_string(), grid.iter().cloned().zip(pr).collect()));
            let xb = -b / w[0];
            if xb.is_finite() && xb >= lo - pad && xb <= hi + pad {
                marks.push((xb, 0.5, format!("boundary x = {}", num(xb))));
            }
            title = "Logistic regression: fitted probability and decision boundary";
            caption = format!("Logistic regression fitted by the model to x = {} with labels {} (C = {}): weight {}, intercept {}, \
                               training accuracy {}; the curve is the model's predicted probability at 81 points (one run of its \
                               predict capability); marked: the decision boundary where P = 0.5.", list(&x, 10), list(&y, 10),
                              num(s.inp("C")?), num(w[0]), num(b), num(s.out("train_accuracy")?));
            header = format!("w = {}, b = {}", num(w[0]), num(b));
            ylab = "y, P(y = 1)";
        }
    }
    let xs: Vec<f64> = x.iter().cloned().chain([lo - pad, hi + pad]).collect();
    let ys: Vec<f64> = y.iter().cloned().chain(series.iter().flat_map(|s| s.1.iter().map(|p| p.1))).collect();
    let mut p = Plot::new(pl::minmax(&xs), pl::minmax(&ys));
    p.clip_on();
    for (i, (_, pts)) in series.iter().enumerate() {
        p.path(pts, if series.len() > 4 { LINE[i % 4] } else { LINE[i % 4] });
    }
    p.unclip();
    for (a, b2) in x.iter().zip(&y) {
        p.dot(*a, *b2, 4.5, "axp-kd");
    }
    for (mx, my, l) in &marks {
        p.dot(*mx, *my, 6.0, "axp-m");
        let (px, py) = (p.x(*mx), p.y(*my));
        p.place(px, py - 10.0, -15.0, 3, "axp-mt", l);
    }
    if series.len() > 1 {
        let items: Vec<(String, &str)> = series.iter().enumerate().map(|(i, x)| (x.0.clone(), BAR[i % 4])).collect();
        p.legend_row(&items);
    }
    p.frame("x", ylab, &header);
    Some(s.chart("fit", title, &caption, &p.body()))
}

// ------------------------------------------------------------------------------------------------ ROC, PR, reliability
fn roc(s: &Solved) -> Option<Figure> {
    let (y, sc) = (s.vec("y_true")?, s.vec("scores")?);
    let a = if s.key == "ml_roc_curve" { s.a.clone() } else {
        s.solve_other("ml_roc_curve", &[("y_true", tl(&y)), ("scores", tl(&sc))])?
    };
    let (fpr, tpr) = (k::list_of(&a, "fpr")?, k::list_of(&a, "tpr")?);
    if fpr.len() != tpr.len() || fpr.len() < 2 {
        return None;
    }
    let auc = s.out("auc")?;
    let pts: Vec<(f64, f64)> = fpr.iter().cloned().zip(tpr.iter().cloned()).collect();
    let mut p = Plot::new((0.0, 1.0), (0.0, 1.0));
    let mut poly = pts.clone();
    poly.push((1.0, 0.0));
    poly.push((0.0, 0.0));
    p.fill(&poly, "axp-s");
    p.dseg((0.0, 0.0), (1.0, 1.0), "axp-c4");
    p.path(&pts, "axp-c");
    for q in &pts {
        p.dot(q.0, q.1, 3.5, "axp-d");
    }
    p.frame("false positive rate", "true positive rate", &format!("AUC = {}", num(auc)));
    let extra = if s.key == "ml_roc_auc" {
        format!(" (Gini {}; {} positives, {} negatives)", num(s.out("gini")?), num(s.out("n_pos")?), num(s.out("n_neg")?))
    } else {
        String::new()
    };
    Some(s.chart("roc", if s.key == "ml_roc_auc" { "Area under the ROC curve" } else { "ROC curve of a scored classifier" },
                 &format!("The ROC curve of labels {} with scores {} (points from the model's ROC capability): true against false \
                           positive rate at each threshold; the shaded area is the returned AUC {}{extra}.", list(&y, 12),
                          list(&sc, 12), num(auc)), &p.body()))
}

fn pr(s: &Solved) -> Option<Figure> {
    let (y, sc) = (s.vec("y_true")?, s.vec("scores")?);
    if y.len() != sc.len() || y.len() > 100 {
        return None;
    }
    let mut th = sc.clone();
    th.sort_by(|a, b| b.partial_cmp(a).unwrap());
    th.dedup();
    let mut pts = vec![];
    for t in &th {
        if let Some(a) = s.solve_other("ml_threshold_metrics", &[("y_true", tl(&y)), ("scores", tl(&sc)), ("threshold", fl(*t))]) {
            if let (Some(pp), Some(rr)) = (ans(&a, "precision"), ans(&a, "recall")) {
                pts.push((rr, pp));
            }
        }
    }
    if pts.len() < 2 {
        return None;
    }
    let (ap, tr) = (s.out("average_precision")?, s.out("pr_auc_trapezoid")?);
    let mut p = Plot::new((0.0, 1.0), (0.0, 1.0));
    let mut prev_r = 0.0;
    for q in &pts {
        if q.0 > prev_r {
            p.rect((prev_r, 0.0), (q.0, q.1), "axp-s");
        }
        prev_r = prev_r.max(q.0);
    }
    p.dpath(&pts, "axp-c4");
    for q in &pts {
        p.dot(q.0, q.1, 4.0, "axp-d");
    }
    p.frame("recall", "precision", &format!("average precision = {}", num(ap)));
    Some(s.chart("pr", "Precision-recall curve and average precision",
                 &format!("Precision-recall points of labels {} with scores {}, one per score used as the threshold (from the model's \
                           threshold metrics); the shaded steps are the returned average precision {}, the dashed path its \
                           trapezoid area {}.", list(&y, 12), list(&sc, 12), num(ap), num(tr)), &p.body()))
}

fn reliability(s: &Solved) -> Option<Figure> {
    let (y, pv, nb) = (s.vec("y_true")?, s.vec("p")?, nin(s, "n_bins")?);
    if y.len() != pv.len() || !(nb >= 1.0) || nb > 50.0 {
        return None;
    }
    let nb = nb as usize;
    let mut bins = vec![(0.0, 0.0, 0usize); nb];
    for (t, q) in y.iter().zip(&pv) {
        let i = ((q * nb as f64).floor() as usize).min(nb - 1);
        bins[i].0 += q;
        bins[i].1 += t;
        bins[i].2 += 1;
    }
    let mut p = Plot::new((0.0, 1.0), (0.0, 1.0));
    p.dseg((0.0, 0.0), (1.0, 1.0), "axp-c4");
    let w = 1.0 / nb as f64;
    let mut pts = vec![];
    for (i, (sp, sy, n)) in bins.iter().enumerate() {
        if *n == 0 {
            continue;
        }
        let (mp, fr) = (sp / *n as f64, sy / *n as f64);
        p.rect((i as f64 * w + w * 0.1, 0.0), ((i + 1) as f64 * w - w * 0.1, fr), "axp-b4");
        pts.push((mp, fr));
    }
    p.path(&pts, "axp-c");
    for q in &pts {
        p.dot(q.0, q.1, 4.5, "axp-d");
    }
    let (ece, mce) = (s.out("ece")?, s.out("mce")?);
    p.frame("mean predicted probability", "frequency of class 1", &format!("ECE = {}, MCE = {}", num(ece), num(mce)));
    Some(s.chart("reliability", "Reliability diagram",
                 &format!("Reliability diagram of the stated predictions p = {} for labels {} in {nb} equal-width bins: the observed \
                           frequency of class 1 (bars) and the mean prediction of each bin (dots), tallied from the stated lists; \
                           the dashed diagonal is perfect calibration. Returned: expected calibration error {}, maximum {}, {} \
                           non-empty bins.", list(&pv, 12), list(&y, 12), num(ece), num(mce), num(s.out("n_nonempty_bins")?)),
                 &p.body()))
}

// ------------------------------------------------------------------------------------------------ fields
/// The loss over two weights around the visitor's, one run of the model per cell, with the returned gradient.
fn landscape(s: &Solved) -> Option<Figure> {
    let w = s.vec("w")?;
    if w.len() != 2 {
        return None;
    }
    let (l0, g) = (s.out("loss")?, s.out_list("grad_w")?);
    let n = 15;
    let xs = k::span(w[0] - 1.5, w[0] + 1.5, n, false);
    let ys = k::span(w[1] - 1.5, w[1] + 1.5, n, false);
    let mut vals = vec![vec![None; n]; n];
    for (i, a) in xs.iter().enumerate() {
        for (j, b) in ys.iter().enumerate() {
            vals[i][j] = s.with_input(&[("w", tl(&[*a, *b]))]).and_then(|r| ans(&r, "loss"));
        }
    }
    let mut body = hfield(&xs, &ys, &vals, (w[0], w[1]), "w₁", "w₂", &format!("loss {} at the visitor's w", num(l0)), (false, false));
    if body.is_empty() {
        return None;
    }
    // the returned negative gradient as an arrow 0.8 long (the field's axes span xs and ys exactly)
    let gn = (g[0] * g[0] + g[1] * g[1]).sqrt();
    if gn > 0.0 {
        let px = |v: f64| pl::X0 + (v - xs[0]) / (xs[n - 1] - xs[0]) * (pl::X1 - pl::X0);
        let py = |v: f64| pl::Y1 - (v - ys[0]) / (ys[n - 1] - ys[0]) * (pl::Y1 - pl::Y0);
        let e = (w[0] - 0.8 * g[0] / gn, w[1] - 0.8 * g[1] / gn);
        let mut c = Plot::new((0.0, 1.0), (0.0, 1.0));
        c.xa = pl::Ax::new(xs[0], xs[n - 1], pl::X0, pl::X1, false);
        c.ya = pl::Ax::new(ys[0], ys[n - 1], pl::Y1, pl::Y0, false);
        let _ = (px, py);
        c.arrow((w[0], w[1]), e, "axp-k");
        body.push_str(&c.body());
    }
    let what = if s.key == "ml_mse_gradient" { "mean squared error" } else { "mean logistic loss" };
    Some(s.chart("landscape", if s.key == "ml_mse_gradient" { "Squared-error landscape and its gradient" } else {
                     "Logistic loss landscape and its gradient" },
                 &format!("The {what} over weights (w₁, w₂) within ±1.5 of the visitor's w = ({}, {}), other inputs as given ({} runs, \
                           darker = higher). Marked: the visitor's w, returned loss {}; the arrow points along the returned \
                           negative gradient -({}), drawn 0.8 long.", num(w[0]), num(w[1]), n * n, num(l0), list(&g, 2)), &body))
}

fn split_field(s: &Solved) -> Option<Figure> {
    let (n0, f0) = (nin(s, "n")?, s.inp("test_size")?);
    let fs = k::span(0.02, 0.98, 21, false);
    let ns: Vec<f64> = k::span(1.0, (n0 * 2.0).max(20.0), 15, true).iter().map(|v| v.round()).collect();
    let mut vals = vec![vec![None; ns.len()]; fs.len()];
    for (i, f) in fs.iter().enumerate() {
        for (j, n) in ns.iter().enumerate() {
            vals[i][j] = s.with_input(&[("test_size", fl(*f)), ("n", it(*n as i64))]).and_then(|a| ans(&a, "n_test"));
        }
    }
    let body = hfield(&fs, &ns, &vals, (f0, n0), "test fraction", "number of samples n (logarithmic)",
                        &format!("n_test = {}", num(s.out("n_test")?)), (false, true));
    if body.is_empty() {
        return None;
    }
    Some(s.chart("field", "Test-set size over the test fraction and the number of samples",
                 &format!("n_test over the test fraction from 0.02 to 0.98 and n from 1 to {}: each of the 21 × 15 cells is one run of \
                           the model, shaded by its value; blank cells are inputs the capability refuses. Marked: the visitor's \
                           n = {}, test_size = {}: n_test = {}, n_train = {}.", num(ns[ns.len() - 1]), num(n0), num(f0),
                          num(s.out("n_test")?), num(s.out("n_train")?)), &body))
}
