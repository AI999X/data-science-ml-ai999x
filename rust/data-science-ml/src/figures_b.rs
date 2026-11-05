#![allow(unused_imports)]
//! Data Science and ML's own live figures, continued: points in the plane (clusters, decision regions and boundaries,
//! neighbours, principal components, distances), splits, folds, parity plots and spreads.

use crate::figures::{hbars, hheat, hfield, hdecision, htable, ans, fl, it, mat_out, nin, tl, tm, tx};
use crate::plot::{self as pl, list, Plot, BAR, DOT, LINE, X0, X1, Y0, Y1};
use ax_engine::figkit::{self as k, num, Solved};
use ax_engine::figures::Figure;

pub fn live(s: &Solved) -> Vec<Option<Figure>> {
    match s.key {
        "ml_calinski_harabasz_index" | "ml_davies_bouldin_index" => vec![clusters(s)],
        "ml_silhouette_score" => vec![clusters(s), silhouette(s)],
        "ml_kmeans" => vec![clusters(s), kmeans_regions(s)],
        "ml_kmeans_step" => vec![kmeans_step(s)],
        "ml_knn_classify" => vec![neighbours(s), knn_regions(s)],
        "ml_linear_svm_decision" => vec![svm(s)],
        "ml_logistic_regression_predict" => vec![probability(s)],
        "ml_perceptron_epoch" => vec![perceptron(s)],
        "ml_distance_metrics" | "ml_triplet_loss" => vec![geometry(s)],
        "ml_pca" => vec![pca(s)],
        "ml_decision_stump_classification" | "ml_decision_stump_regression" => vec![split(s)],
        "ml_bias_variance_decomposition" => vec![spread(s)],
        "ml_adjusted_r2" => vec![parity(s), penalty(s)],
        "ml_regression_metrics" => vec![parity(s)],
        "ml_kfold_cv_linear_regression" => vec![kfold_data(s), kfold_folds(s)],
        "ml_kfold_sizes" => vec![fold_grid(s)],
        "ml_loocv_linear_regression" => vec![loocv(s)],
        "ml_gaussian_naive_bayes" => vec![features(s)],
        _ => vec![],
    }
}

/// The class names of a label list in sorted order.
fn classes(v: &[f64]) -> Vec<f64> {
    let mut c = v.to_vec();
    c.sort_by(|a, b| a.partial_cmp(b).unwrap());
    c.dedup();
    c
}

/// Points of a 2-column matrix (None for any other shape or more than 400 points).
fn plane_pts(m: &[Vec<f64>]) -> Option<Vec<(f64, f64)>> {
    (m.len() <= 400 && m.iter().all(|r| r.len() == 2)).then(|| m.iter().map(|r| (r[0], r[1])).collect())
}

/// Points coloured by label in a plane with equal scales; `extra`: further points to keep in view.
fn labelled(pts: &[(f64, f64)], lab: &[f64], extra: &[(f64, f64)]) -> (Plot, Vec<f64>) {
    let xs: Vec<f64> = pts.iter().chain(extra).map(|p| p.0).collect();
    let ys: Vec<f64> = pts.iter().chain(extra).map(|p| p.1).collect();
    let mut p = Plot::equal(&xs, &ys);
    let cl = classes(lab);
    for (q, l) in pts.iter().zip(lab) {
        let i = cl.iter().position(|c| c == l).unwrap_or(0);
        p.dot(q.0, q.1, 5.0, DOT[i % 3]);
    }
    (p, cl)
}

fn class_legend(p: &mut Plot, cl: &[f64], what: &str) {
    let items: Vec<(String, &str)> = cl.iter().take(4).enumerate().map(|(i, c)| (format!("{what} {}", num(*c)), BAR[i % 3])).collect();
    p.legend_row(&items);
}

fn means(pts: &[(f64, f64)], lab: &[f64], cl: &[f64]) -> Vec<(f64, f64)> {
    cl.iter().map(|c| {
        let g: Vec<&(f64, f64)> = pts.iter().zip(lab).filter(|x| x.1 == c).map(|x| x.0).collect();
        let n = g.len().max(1) as f64;
        (g.iter().map(|q| q.0).sum::<f64>() / n, g.iter().map(|q| q.1).sum::<f64>() / n)
    }).collect()
}

/// Clusters: points by label with their means (from the stated points) and the spread the index measures.
fn clusters(s: &Solved) -> Option<Figure> {
    let pts = plane_pts(&s.mat("X")?)?;
    let lab = if s.key == "ml_kmeans" { s.out_list("labels")? } else { s.vec("labels")? };
    if lab.len() != pts.len() {
        return None;
    }
    let cl = classes(&lab);
    let ms = means(&pts, &lab, &cl);
    let mut extra = ms.clone();
    if s.key == "ml_kmeans" {
        extra.extend(plane_pts(&s.mat("C")?)?);
    }
    let (mut p, _) = labelled(&pts, &lab, &extra);
    let (title, caption, header);
    match s.key {
        "ml_kmeans" => {
            let c0 = plane_pts(&s.mat("C")?)?;
            let cf = plane_pts(&mat_out(s, "centroids")?)?;
            for (a, b) in c0.iter().zip(&cf) {
                p.arrow(*a, *b, "axp-k");
                p.dot(a.0, a.1, 4.0, "axp-d4");
            }
            for (i, c) in cf.iter().enumerate() {
                p.dot(c.0, c.1, 7.0, "axp-kd");
                p.label(c.0, c.1, 9.0, -8.0, "start", "axp-vt", &format!("c{} ({}, {})", i + 1, num(c.0), num(c.1)));
            }
            title = "k-means clusters and their centroids";
            header = format!("inertia {}", num(s.out("inertia")?));
            caption = format!("k-means on {} points from the starting centres (grey): the model's final centroids (black, labelled; \
                               arrows from the starting centres), the points coloured by the returned labels, inertia {}.",
                              pts.len(), num(s.out("inertia")?));
        }
        "ml_davies_bouldin_index" => {
            for (i, m) in ms.iter().enumerate() {
                let g: Vec<&(f64, f64)> = pts.iter().zip(&lab).filter(|x| *x.1 == cl[i]).map(|x| x.0).collect();
                let d = g.iter().map(|q| ((q.0 - m.0).powi(2) + (q.1 - m.1).powi(2)).sqrt()).sum::<f64>() / g.len().max(1) as f64;
                p.ring(m.0, m.1, d, "axp-c4", true);
                p.dot(m.0, m.1, 6.0, "axp-kd");
            }
            for w in ms.windows(2) {
                p.dseg(w[0], w[1], "axp-k");
            }
            title = "Cluster spread against separation";
            header = format!("Davies-Bouldin {}", num(s.out("davies_bouldin")?));
            caption = format!("{} stated points coloured by their labels; black dots: cluster means, dashed circles: each cluster's \
                               mean distance to its mean, dashed line: the distance between the means (all from the stated points, \
                               the ratio the index is built from). Returned Davies-Bouldin index {} (lower is better).", pts.len(),
                              num(s.out("davies_bouldin")?));
        }
        "ml_silhouette_score" => {
            let sm = s.out_list("samples")?;
            for (q, v) in pts.iter().zip(&sm) {
                p.label(q.0, q.1, 7.0, -7.0, "start", "axp-vt", &num(*v));
            }
            title = "Points labelled with their silhouettes";
            header = format!("mean silhouette {}", num(s.out("silhouette")?));
            caption = format!("The {} stated points coloured by label, each labelled with its returned silhouette; mean {}.", pts.len(),
                              num(s.out("silhouette")?));
        }
        _ => {
            for (i, m) in ms.iter().enumerate() {
                for (q, l) in pts.iter().zip(&lab) {
                    if *l == cl[i] {
                        p.seg(*q, *m, "axp-g");
                    }
                }
                p.dot(m.0, m.1, 6.0, "axp-kd");
            }
            title = "Clusters judged by the Calinski-Harabasz index";
            header = format!("Calinski-Harabasz {}", num(s.out("calinski_harabasz")?));
            caption = format!("{} stated points with their cluster labels; black dots: each cluster's mean (from the stated points), \
                               grey: the within-cluster spread. Returned: between-cluster dispersion {}, within {}, index {}.",
                              pts.len(), num(s.out("between")?), num(s.out("within")?), num(s.out("calinski_harabasz")?));
        }
    }
    class_legend(&mut p, &cl, "label");
    p.frame("x₁", "x₂", &header);
    Some(s.chart("clusters", title, &caption, &p.body()))
}

fn silhouette(s: &Solved) -> Option<Figure> {
    let sm = s.out_list("samples")?;
    let lab = s.vec("labels")?;
    if sm.len() != lab.len() || sm.len() > 60 {
        return None;
    }
    let cl = classes(&lab);
    let mut order: Vec<usize> = (0..sm.len()).collect();
    order.sort_by(|a, b| {
        let ca = cl.iter().position(|c| *c == lab[*a]).unwrap();
        let cb = cl.iter().position(|c| *c == lab[*b]).unwrap();
        ca.cmp(&cb).then(sm[*b].partial_cmp(&sm[*a]).unwrap())
    });
    let mean = s.out("silhouette")?;
    let n = sm.len() as f64;
    let mut p = Plot::axes(pl::Ax::new(pl::minmax(&sm).0.min(0.0), pl::minmax(&sm).1.max(0.0), X0, X1, true),
                           pl::Ax::new(0.4, n + 0.6, Y0 + 10.0, Y1, false));
    p.vline(0.0, "axp-c4", false);
    for (r, i) in order.iter().enumerate() {
        let y = r as f64 + 1.0;
        let c = cl.iter().position(|c| *c == lab[*i]).unwrap();
        p.rect((0.0, y - 0.38), (sm[*i], y + 0.38), BAR[c % 3]);
        if n <= 24.0 {
            p.text(X0 - 6.0, p.y(y) + 5.0, "end", "axp-lt", &format!("point {}", i));
        }
    }
    p.vline(mean, "axp-c3", true);
    p.text(p.x(mean) + 5.0, Y1 - 6.0, "start", "axp-mt", &format!("mean {}", num(mean)));
    class_legend(&mut p, &cl, "cluster");
    p.frame_with("silhouette", "", "", true, false);
    Some(s.chart("silhouette", "Silhouette of each point",
                 &format!("The silhouette of each of the {} stated points (returned: {}), grouped by cluster label and sorted; the \
                           dashed line is the returned mean silhouette {}.", sm.len(), list(&sm, 12), num(mean)), &p.body()))
}

/// A grid over the plane of the points, `nx` × `ny` cell centres.
fn grid(pts: &[(f64, f64)], nx: usize, ny: usize) -> (Vec<f64>, Vec<f64>) {
    let (x0, x1) = pl::minmax(&pts.iter().map(|p| p.0).collect::<Vec<_>>());
    let (y0, y1) = pl::minmax(&pts.iter().map(|p| p.1).collect::<Vec<_>>());
    let (dx, dy) = (((x1 - x0) * 0.15).max(0.5), ((y1 - y0) * 0.15).max(0.5));
    (k::span(x0 - dx, x1 + dx, nx, false), k::span(y0 - dy, y1 + dy, ny, false))
}

fn kmeans_regions(s: &Solved) -> Option<Figure> {
    let pts = plane_pts(&s.mat("X")?)?;
    let cf = mat_out(s, "centroids")?;
    let (xs, ys) = grid(&pts, 30, 16);
    let cells: Vec<Vec<f64>> = xs.iter().flat_map(|x| ys.iter().map(move |y| vec![*x, *y])).collect();
    // one run of the model's assignment step over every cell centre
    let a = s.solve_other("ml_kmeans_step", &[("X", tm(&cells)), ("C", tm(&cf))])?;
    let lab = k::list_of(&a, "labels")?;
    if lab.len() != cells.len() {
        return None;
    }
    let names: Vec<String> = (0..cf.len()).map(|i| format!("centroid {}", i + 1)).collect();
    let mut cl = vec![];
    for (i, _) in xs.iter().enumerate() {
        for (j, _) in ys.iter().enumerate() {
            let l = lab[i * ys.len() + j] as usize;
            cl.push((i, j, names.get(l).cloned().unwrap_or_default()));
        }
    }
    let first = cf.first()?;
    let mut body = hdecision(&xs, &ys, &cl, &names, (first[0], first[1]), "x₁", "x₂", &format!("inertia {}", num(s.out("inertia")?)));
    // the points and centroids on top, on the decision chart's axes (its cells span xs and ys)
    let mut p = Plot::new((0.0, 1.0), (0.0, 1.0));
    // the decision chart's axes: its cell centres' range, padded as the kit pads it
    p.xa = pl::Ax::new(xs[0], xs[xs.len() - 1], X0, X1, true);
    p.ya = pl::Ax::new(ys[0], ys[ys.len() - 1], Y1, Y0, true);
    for q in &pts {
        p.dot(q.0, q.1, 3.5, "axp-kd");
    }
    for (i, c) in cf.iter().enumerate() {
        if c.len() == 2 {
            p.dot(c[0], c[1], 7.0, "axp-m");
            p.label(c[0], c[1], 9.0, -8.0, "start", "axp-mt", &format!("c{}", i + 1));
        }
    }
    body.push_str(&p.body());
    Some(s.chart("regions", "k-means: the plane split by the final centroids",
                 &format!("k-means on the {} stated points: the returned centroids (marked) and the plane split by nearest centroid, \
                           each of {} cells assigned by one run of the model's assignment step. Inertia {}.", pts.len(), cells.len(),
                          num(s.out("inertia")?)), &body))
}

fn kmeans_step(s: &Solved) -> Option<Figure> {
    let pts = plane_pts(&s.mat("X")?)?;
    let c0 = plane_pts(&s.mat("C")?)?;
    let cn = plane_pts(&mat_out(s, "new_centroids")?)?;
    let lab = s.out_list("labels")?;
    if lab.len() != pts.len() {
        return None;
    }
    let mut extra = c0.clone();
    extra.extend(cn.iter().cloned());
    let (mut p, cl) = labelled(&pts, &lab, &extra);
    for (q, l) in pts.iter().zip(&lab) {
        if let Some(c) = c0.get(*l as usize) {
            p.seg(*q, *c, "axp-g");
        }
    }
    for (a, b) in c0.iter().zip(&cn) {
        p.dot(a.0, a.1, 6.0, "axp-d4");
        p.arrow(*a, *b, "axp-k");
        p.dot(b.0, b.1, 7.0, "axp-kd");
    }
    class_legend(&mut p, &cl, "label");
    p.frame("x₁", "x₂", &format!("inertia before the move {}", num(s.out("inertia")?)));
    Some(s.chart("step", "One k-means step: assign, then move",
                 &format!("One k-means step on the {} stated points: each point joined to its nearest current centre (grey; returned \
                           labels {}); arrows move each centre to the mean of its points, the returned new centroids (black). \
                           Inertia before the move {}.", pts.len(), list(&lab, 12), num(s.out("inertia")?)), &p.body()))
}

fn neighbours(s: &Solved) -> Option<Figure> {
    let pts = plane_pts(&s.mat("X")?)?;
    let lab = s.vec("y")?;
    let q = s.vec("x")?;
    if q.len() != 2 || lab.len() != pts.len() {
        return None;
    }
    let nb = s.out_list("neighbors")?;
    let d = s.out_list("neighbor_distances")?;
    let r = pl::minmax(&d).1;
    let extra = vec![(q[0] - r, q[1] - r), (q[0] + r, q[1] + r)];
    let (mut p, cl) = labelled(&pts, &lab, &extra);
    for i in &nb {
        if let Some(t) = pts.get(*i as usize) {
            p.dseg((q[0], q[1]), *t, "axp-k");
        }
    }
    if s.txt("metric").as_deref() == Some("euclidean") {
        p.ring(q[0], q[1], r, "axp-c4", true);
    }
    p.dot(q[0], q[1], 7.0, "axp-m");
    let pred = s.out("prediction")?;
    p.label(q[0], q[1], 10.0, 18.0, "start", "axp-mt", &format!("→ class {}", num(pred)));
    class_legend(&mut p, &cl, "class");
    p.frame("x₁", "x₂", &format!("k = {}: {} of {} votes", num(nin(s, "k")?), num(s.out("votes")?), num(nin(s, "k")?)));
    Some(s.chart("neighbours", "k-nearest neighbours of a query point",
                 &format!("k-nearest-neighbour classification of the query point ({}, {}) among {} labelled points (k = {}, {} \
                           distance): the model's nearest neighbours are points {} (0-based) at distances {} (dashed lines); the \
                           prediction is class {} with {} votes.", num(q[0]), num(q[1]), pts.len(), num(nin(s, "k")?),
                          s.txt("metric")?, list(&nb, 10), list(&d, 10), num(pred), num(s.out("votes")?)), &p.body()))
}

fn knn_regions(s: &Solved) -> Option<Figure> {
    let pts = plane_pts(&s.mat("X")?)?;
    let q = s.vec("x")?;
    if q.len() != 2 {
        return None;
    }
    let mut all = pts.clone();
    all.push((q[0], q[1]));
    let (xs, ys) = grid(&all, 10, 7);
    let lab = s.vec("y")?;
    let cl = classes(&lab);
    let names: Vec<String> = cl.iter().map(|c| format!("class {}", num(*c))).collect();
    let mut cells = vec![];
    let mut refused = 0;
    for (i, x) in xs.iter().enumerate() {
        for (j, y) in ys.iter().enumerate() {
            match s.with_input(&[("x", tl(&[*x, *y]))]).and_then(|a| ans(&a, "prediction")) {
                Some(v) => cells.push((i, j, format!("class {}", num(v)))),
                None => refused += 1,
            }
        }
    }
    if cells.len() < 10 {
        return None;
    }
    let mut body = hdecision(&xs, &ys, &cells, &names, (q[0], q[1]), "x₁", "x₂",
                               &format!("k = {}, {}", num(nin(s, "k")?), s.txt("metric")?));
    let mut p = Plot::new((0.0, 1.0), (0.0, 1.0));
    p.xa = pl::Ax::new(xs[0], xs[xs.len() - 1], X0, X1, true);
    p.ya = pl::Ax::new(ys[0], ys[ys.len() - 1], Y1, Y0, true);
    for (pt, l) in pts.iter().zip(&lab) {
        let i = cl.iter().position(|c| c == l).unwrap_or(0);
        p.dot(pt.0, pt.1, 4.5, DOT[i % 3]);
    }
    body.push_str(&p.body());
    Some(s.chart("regions", "k-nearest-neighbour decision regions",
                 &format!("The k-nearest-neighbour vote over the plane: each of {} cells is coloured by the class the model predicts at \
                           its centre, from the {} stated labelled points; {} cells the model refuses (a tie) are left blank. \
                           Marked: the visitor's query ({}, {}) → class {}.", xs.len() * ys.len(), pts.len(), refused, num(q[0]),
                          num(q[1]), num(s.out("prediction")?)), &body))
}

/// The line w·x + b = c in the plot's range (None for w = 0).
fn line_pts(w: &[f64], b: f64, c: f64, p: &Plot) -> Option<[(f64, f64); 2]> {
    let (x0, x1, y0, y1) = (p.xa.lo, p.xa.hi, p.ya.lo, p.ya.hi);
    if w[1].abs() > 1e-12 {
        Some([(x0, (c - b - w[0] * x0) / w[1]), (x1, (c - b - w[0] * x1) / w[1])])
    } else if w[0].abs() > 1e-12 {
        let x = (c - b) / w[0];
        Some([(x, y0), (x, y1)])
    } else {
        None
    }
}

fn svm(s: &Solved) -> Option<Figure> {
    let pts = plane_pts(&s.mat("X")?)?;
    let w = s.vec("w")?;
    let b = s.inp("b")?;
    if w.len() != 2 {
        return None;
    }
    let (dec, cls, dist) = (s.out_list("decision")?, s.out_list("classes")?, s.out_list("distances")?);
    let mw = s.out("margin_width")?;
    let (mut p, cl) = labelled(&pts, &cls, &[]);
    p.clip_on();
    for (c, class) in [(0.0, "axp-c3"), (1.0, "axp-c4"), (-1.0, "axp-c4")] {
        if let Some([a, e]) = line_pts(&w, b, c, &p) {
            if c == 0.0 { p.seg(a, e, class) } else { p.dseg(a, e, class) }
        }
    }
    p.unclip();
    for (i, q) in pts.iter().enumerate().take(24) {
        if let (Some(d), Some(t)) = (dec.get(i), dist.get(i)) {
            p.label(q.0, q.1, 8.0, -7.0, "start", "axp-vt", &format!("{} | {}", num(*d), num(*t)));
        }
    }
    class_legend(&mut p, &cl, "class");
    p.frame("x₁", "x₂", &format!("margin width {}", num(mw)));
    Some(s.chart("boundary", "Linear SVM: boundary, margins and decisions",
                 &format!("The linear SVM w = ({}, {}), b = {}: decision line and the margins w·x + b = ±1 (drawn from the stated w, \
                           b; returned margin width {}). The stated points are coloured by the returned class ({}) and labelled with \
                           the returned decision value | distance to the line.", num(w[0]), num(w[1]), num(b), num(mw),
                          list(&cls, 12)), &p.body()))
}

/// The predicted probability over the plane, every cell from one run of the model on all cell centres.
fn probability(s: &Solved) -> Option<Figure> {
    let pts = plane_pts(&s.mat("X")?)?;
    let w = s.vec("w")?;
    let b = s.inp("b")?;
    if w.len() != 2 {
        return None;
    }
    let (pr, cls) = (s.out_list("probabilities")?, s.out_list("classes")?);
    let (xs, ys) = grid(&pts, 30, 25);
    let cells: Vec<Vec<f64>> = xs.iter().flat_map(|x| ys.iter().map(move |y| vec![*x, *y])).collect();
    let a = s.with_input(&[("X", tm(&cells))])?;
    let all = k::list_of(&a, "probabilities")?;
    if all.len() != cells.len() {
        return None;
    }
    let vals: Vec<Vec<Option<f64>>> = (0..xs.len()).map(|i| (0..ys.len()).map(|j| Some(all[i * ys.len() + j])).collect()).collect();
    let mut body = hfield(&xs, &ys, &vals, pts[0], "x₁", "x₂", "P(class 1)", (false, false));
    if body.is_empty() {
        return None;
    }
    let mut p = Plot::new((0.0, 1.0), (0.0, 1.0));
    p.xa = pl::Ax::new(xs[0], xs[xs.len() - 1], X0, X1, false);
    p.ya = pl::Ax::new(ys[0], ys[ys.len() - 1], Y1, Y0, false);
    p.clip_on();
    if let Some([u, v]) = line_pts(&w, b, 0.0, &p) {
        p.dseg(u, v, "axp-k");
    }
    p.unclip();
    for (i, q) in pts.iter().enumerate().take(12) {
        p.dot(q.0, q.1, 5.0, if cls.get(i) == Some(&1.0) { "axp-d" } else { "axp-d3" });
        p.label(q.0, q.1, 8.0, -7.0, "start", "axp-vt", &num(*pr.get(i)?));
    }
    body.push_str(&p.body());
    Some(s.chart("probability", "Logistic regression: probability over the plane",
                 &format!("Predicted probability of class 1 for w = ({}, {}), b = {} over the plane, {} cells from one run of the model \
                           (darker = higher); dashed: the boundary w·x + b = 0 (from the stated w, b). Marked: the stated points with \
                           their returned probabilities ({}) and classes ({}).", num(w[0]), num(w[1]), num(b), cells.len(),
                          list(&pr, 10), list(&cls, 10)), &body))
}

fn perceptron(s: &Solved) -> Option<Figure> {
    let pts = plane_pts(&s.mat("X")?)?;
    let lab = s.vec("y")?;
    let (mut w, mut b) = (s.vec("w")?, s.inp("b")?);
    if w.len() != 2 || lab.len() != pts.len() {
        return None;
    }
    let mut lines = vec![];
    for _ in 0..4 {
        let a = s.with_input(&[("w", tl(&w)), ("b", fl(b))])?;
        w = k::list_of(&a, "w_new")?;
        b = ans(&a, "b_new")?;
        lines.push((w.clone(), b, ans(&a, "n_updates")?));
    }
    let (mut p, cl) = labelled(&pts, &lab, &[]);
    p.clip_on();
    for (i, (ww, bb, _)) in lines.iter().enumerate() {
        if let Some([a, e]) = line_pts(ww, *bb, 0.0, &p) {
            if i + 1 == lines.len() { p.seg(a, e, "axp-c3") } else { p.dseg(a, e, LINE[(i + 1) % 4]) }
        }
    }
    p.unclip();
    class_legend(&mut p, &cl, "label");
    let words = lines.iter().enumerate().map(|(i, (ww, bb, u))| format!("epoch {}: w = ({}, {}), b = {}, {} updates", i + 1,
                                                                        num(ww[0]), num(ww[1]), num(*bb), num(*u)))
        .collect::<Vec<_>>().join("; ");
    p.frame("x₁", "x₂", &format!("after 4 epochs: w = ({}, {}), b = {}", num(w[0]), num(w[1]), num(b)));
    Some(s.chart("boundary", "Perceptron boundary epoch by epoch",
                 &format!("Perceptron epochs on the {} stated points (labels {}, start w = ({}), b = {}, lr = {}): each epoch is one run \
                           of the model from the w, b the previous run returned; the line w·x + b = 0 after each (last one solid): \
                           {words}.", pts.len(), list(&lab, 12), list(&s.vec("w")?, 2), num(s.inp("b")?), num(s.inp("lr")?)),
                 &p.body()))
}

fn geometry(s: &Solved) -> Option<Figure> {
    if s.key == "ml_distance_metrics" {
        let (x, z) = (s.vec("x")?, s.vec("z")?);
        if x.len() < 2 || x.len() != z.len() {
            return None;
        }
        let (a, b) = ((x[0], x[1]), (z[0], z[1]));
        let mut p = Plot::equal(&[a.0, b.0, 0.0], &[a.1, b.1, 0.0]);
        p.dseg((0.0, 0.0), a, "axp-g");
        p.dseg((0.0, 0.0), b, "axp-g");
        p.seg(a, (b.0, a.1), "axp-c2");
        p.seg((b.0, a.1), b, "axp-c2");
        p.seg(a, b, "axp-c");
        p.dot(a.0, a.1, 6.0, "axp-d");
        p.dot(b.0, b.1, 6.0, "axp-d3");
        p.label(a.0, a.1, -8.0, -8.0, "end", "axp-bt", "x");
        p.label(b.0, b.1, 8.0, -8.0, "start", "axp-bt", "z");
        let (eu, ma, ch, cs) = (s.out("euclidean")?, s.out("manhattan")?, s.out("chebyshev")?, s.out("cosine_similarity")?);
        p.legend_row(&[(format!("euclidean {}", num(eu)), "axp-b"), (format!("legs (manhattan {})", num(ma)), "axp-b2")]);
        p.frame("coordinate 1", "coordinate 2", &format!("chebyshev {}, cosine {}", num(ch), num(cs)));
        return Some(s.chart("geometry", "Distances between two points",
                            &format!("The distances the model returns between x = ({}) and z = ({}) ({} dimensions; drawn in the \
                                      first two, the legs those of the first two coordinates): straight line {}, city-block path {}, \
                                      largest coordinate gap {}, cosine similarity {} (distance {}).", list(&x, 6), list(&z, 6),
                                     x.len(), num(eu), num(ma), num(ch), num(cs), num(s.out("cosine_distance")?)), &p.body()));
    }
    let (a, po, ne) = (s.vec("anchor")?, s.vec("positive")?, s.vec("negative")?);
    if a.len() != 2 || po.len() != 2 || ne.len() != 2 {
        return None;
    }
    let (dap, dan, l, m) = (s.out("d_ap")?, s.out("d_an")?, s.out("loss")?, s.inp("margin")?);
    let r = dap + m;
    let mut p = Plot::equal(&[a[0] - r, a[0] + r, po[0], ne[0]], &[a[1] - r, a[1] + r, po[1], ne[1]]);
    p.ring(a[0], a[1], dap, "axp-c2", true);
    p.ring(a[0], a[1], r, "axp-c", true);
    p.seg((a[0], a[1]), (po[0], po[1]), "axp-c2");
    p.seg((a[0], a[1]), (ne[0], ne[1]), "axp-c3");
    for (q, n, c) in [(&a, "anchor", "axp-kd"), (&po, "positive", "axp-d2"), (&ne, "negative", "axp-d3")] {
        p.dot(q[0], q[1], 6.0, c);
        p.label(q[0], q[1], 8.0, -8.0, "start", "axp-bt", n);
    }
    p.frame("coordinate 1", "coordinate 2", &format!("loss {}", num(l)));
    Some(s.chart("geometry", "Triplet loss: the negative must clear the margin",
                 &format!("Anchor ({}), positive ({}) and negative ({}) with margin {}: the returned distances d(a, p) = {} (inner \
                           circle) and d(a, n) = {}. The loss is zero only if the negative lies outside the outer circle (radius \
                           d(a, p) + margin); the returned loss is {}.", list(&a, 2), list(&po, 2), list(&ne, 2), num(m), num(dap),
                          num(dan), num(l)), &p.body()))
}

fn pca(s: &Solved) -> Option<Figure> {
    let pts = plane_pts(&s.mat("X")?)?;
    let (c1, sc) = (s.out_list("component_1")?, s.out_list("scores_1")?);
    let (ev, er) = (s.out_list("explained_variance")?, s.out_list("explained_variance_ratio")?);
    if c1.len() != 2 || sc.len() != pts.len() || ev.len() < 2 {
        return None;
    }
    let n = pts.len() as f64;
    let m = (pts.iter().map(|p| p.0).sum::<f64>() / n, pts.iter().map(|p| p.1).sum::<f64>() / n);
    let proj: Vec<(f64, f64)> = sc.iter().map(|t| (m.0 + t * c1[0], m.1 + t * c1[1])).collect();
    let (lo, hi) = pl::minmax(&sc);
    let mut extra = proj.clone();
    extra.push(m);
    let mut p = Plot::equal(&pts.iter().chain(&extra).map(|q| q.0).collect::<Vec<_>>(), &pts.iter().chain(&extra).map(|q| q.1).collect::<Vec<_>>());
    p.seg((m.0 + lo * c1[0], m.1 + lo * c1[1]), (m.0 + hi * c1[0], m.1 + hi * c1[1]), "axp-c3");
    for (q, r) in pts.iter().zip(&proj) {
        p.seg(*q, *r, "axp-g");
        p.dot(r.0, r.1, 3.0, "axp-d3");
        p.dot(q.0, q.1, 4.5, "axp-d");
    }
    let sd2 = 2.0 * ev[1].max(0.0).sqrt();
    p.arrow(m, (m.0 - c1[1] * sd2, m.1 + c1[0] * sd2), "axp-k");
    p.dot(m.0, m.1, 5.0, "axp-kd");
    p.frame("x₁", "x₂", &format!("first component: {} % of the variance", num(er[0] * 100.0)));
    Some(s.chart("pca", "Principal components: projection onto the first component",
                 &format!("Principal components of {} points: the model's first component ({}, {}) through the mean of the points \
                           explains {} % of the variance (variance {} against {}); each point is joined to its projection at the \
                           score the model returns; the short arrow is two standard deviations ({}) along the second component.",
                          pts.len(), num(c1[0]), num(c1[1]), num(er[0] * 100.0), num(ev[0]), num(ev[1]), num(sd2)), &p.body()))
}

fn split(s: &Solved) -> Option<Figure> {
    let xm = s.mat("X")?;
    let y = s.vec("y")?;
    let (f, t) = (s.out("feature")?, s.out("threshold")?);
    if xm.len() != y.len() || xm.len() > 400 {
        return None;
    }
    if s.key == "ml_decision_stump_regression" {
        if xm.iter().any(|r| r.len() != 1) {
            return None;
        }
        let x: Vec<f64> = xm.iter().map(|r| r[0]).collect();
        let (lm, rm) = (s.out("left_mean")?, s.out("right_mean")?);
        let (lo, hi) = pl::minmax(&x);
        let mut p = Plot::new((lo, hi), pl::minmax(&y.iter().cloned().chain([lm, rm]).collect::<Vec<_>>()));
        p.seg((p.xa.lo, lm), (t, lm), "axp-c3");
        p.seg((t, rm), (p.xa.hi, rm), "axp-c3");
        p.dseg((t, lm), (t, rm), "axp-c3");
        p.vline(t, "axp-c4", true);
        for (a, b) in x.iter().zip(&y) {
            p.dot(*a, *b, 4.5, "axp-d");
        }
        p.place(p.x(t), Y0 + 16.0, 15.0, 2, "axp-mt", &format!("x ≤ {}", num(t)));
        p.frame("x", "y", &format!("MSE decrease {}", num(s.out("mse_decrease")?)));
        return Some(s.chart("split", "Regression stump: a step through the data",
                            &format!("The best single split of x = {} for y = {}: the model returns threshold {}, left mean {}, right \
                                      mean {} and a decrease of the squared error by {}; the stump predicts the mean of each side.",
                                     list(&x, 12), list(&y, 12), num(t), num(lm), num(rm), num(s.out("mse_decrease")?)), &p.body()));
    }
    let pts = plane_pts(&xm)?;
    let (mut p, cl) = labelled(&pts, &y, &[]);
    if f == 0.0 { p.vline(t, "axp-c3", false) } else { p.hline(t, "axp-c3", false) }
    let lab = format!("x{} ≤ {}", f as i64 + 1, num(t));
    if f == 0.0 { p.place(p.x(t), Y0 + 16.0, 15.0, 2, "axp-mt", &lab) } else { p.text(X0 + 6.0, p.y(t) - 6.0, "start", "axp-mt", &lab) }
    class_legend(&mut p, &cl, "class");
    p.frame("x₁", "x₂", &format!("impurity decrease {}", num(s.out("impurity_decrease")?)));
    Some(s.chart("split", "Decision stump: the best single split",
                 &format!("The best single split of the {} stated points (labels {}, {} criterion): the model returns feature {} \
                           (0-based), threshold {}, {} points left and {} right, impurity decrease {}.", pts.len(), list(&y, 12),
                          s.txt("criterion")?, num(f), num(t), num(s.out("n_left")?), num(s.out("n_right")?),
                          num(s.out("impurity_decrease")?)), &p.body()))
}

fn spread(s: &Solved) -> Option<Figure> {
    let pm = s.mat("P")?;
    let f = s.vec("f")?;
    if pm.iter().any(|r| r.len() != f.len()) || f.len() > 30 {
        return None;
    }
    let n = f.len() as f64;
    let all: Vec<f64> = pm.iter().flatten().cloned().chain(f.iter().cloned()).collect();
    let mut p = Plot::new((0.5, n + 0.5), pl::minmax(&all));
    for j in 0..f.len() {
        let c = j as f64 + 1.0;
        let col: Vec<f64> = pm.iter().map(|r| r[j]).collect();
        let m = col.iter().sum::<f64>() / col.len() as f64;
        for (i, v) in col.iter().enumerate() {
            p.dot(c - 0.12 + 0.24 * i as f64 / (col.len().max(2) - 1) as f64, *v, 4.0, "axp-d3");
        }
        p.seg((c - 0.25, m), (c + 0.25, m), "axp-c3");
        p.seg((c - 0.25, f[j]), (c + 0.25, f[j]), "axp-c");
        p.dseg((c + 0.3, m), (c + 0.3, f[j]), "axp-k");
    }
    p.legend_row(&[("truth f".into(), "axp-b"), ("mean prediction".into(), "axp-b3")]);
    let (b2, v, e) = (s.out("bias_squared")?, s.out("variance")?, s.out("expected_mse")?);
    p.frame("test point", "prediction", &format!("bias² {} + variance {} = {}", num(b2), num(v), num(e)));
    Some(s.chart("spread", "Bias and variance of several models' predictions",
                 &format!("Predictions of {} models (rows of P) at {} test points with true values f = {}: their spread is the \
                           variance, their mean's distance from the truth (dashed) the bias. Returned: bias² {}, variance {}, \
                           expected MSE {}.", pm.len(), f.len(), list(&f, 10), num(b2), num(v), num(e)), &p.body()))
}

fn parity(s: &Solved) -> Option<Figure> {
    let (t, pr) = (s.vec("y_true")?, s.vec("y_pred")?);
    if t.len() != pr.len() || t.len() > 400 {
        return None;
    }
    let all: Vec<f64> = t.iter().chain(&pr).cloned().collect();
    let (lo, hi) = pl::minmax(&all);
    let mut p = Plot::new((lo, hi), (lo, hi));
    p.dseg((p.xa.lo, p.xa.lo), (p.xa.hi, p.xa.hi), "axp-c4");
    for (a, b) in t.iter().zip(&pr) {
        p.seg((*a, *a), (*a, *b), "axp-c3");
        p.dot(*a, *b, 4.5, "axp-d");
    }
    let r2 = s.out("r2")?;
    let header = if s.key == "ml_adjusted_r2" { format!("R² {}, adjusted {}", num(r2), num(s.out("adjusted_r2")?)) } else {
        format!("R² {}, RMSE {}", num(r2), num(s.out("rmse")?))
    };
    p.frame("true value", "predicted value", &header);
    Some(s.chart("parity", "Predictions against truth",
                 &format!("Predicted against true values (y_true {}, y_pred {}); each vertical segment is a residual, the dashed line a \
                           perfect prediction. Returned: {header}.", list(&t, 10), list(&pr, 10)), &p.body()))
}

fn penalty(s: &Solved) -> Option<Figure> {
    let p0 = nin(s, "p")?;
    let n = s.vec("y_true")?.len() as i64;
    let mut labels = vec![];
    let mut vals = vec![];
    let mut hl = vec![];
    for pp in 1..=(n - 2).min(8) {
        if let Some(v) = s.with_input(&[("p", it(pp))]).and_then(|a| ans(&a, "adjusted_r2")) {
            if pp as f64 == p0 {
                hl.push(labels.len());
            }
            labels.push(format!("p = {pp}"));
            vals.push(v);
        }
    }
    if vals.len() < 2 {
        return None;
    }
    let body = hbars(&labels, &vals, &hl, "number of predictors", "adjusted R²", &format!("R² = {}, n = {n}", num(s.out("r2")?)));
    Some(s.chart("penalty", "Adjusted R² falls as predictors are added",
                 &format!("Adjusted R² of the same predictions (n = {n} samples, R² = {}) charged for p = 1 to {} predictors, one run \
                           each: {}; the visitor's p = {} in the first colour.", num(s.out("r2")?), labels.len(), list(&vals, 8),
                          num(p0)), &body))
}

/// 1-D regression data in consecutive folds of the model's fold sizes.
fn kfold_data(s: &Solved) -> Option<Figure> {
    let xm = s.mat("X")?;
    let y = s.vec("y")?;
    let kk = nin(s, "k")?;
    if xm.iter().any(|r| r.len() != 1) || xm.len() != y.len() || y.len() > 400 {
        return None;
    }
    let a = s.solve_other("ml_kfold_sizes", &[("n", it(y.len() as i64)), ("k", it(kk as i64))])?;
    let sizes = k::list_of(&a, "test_sizes")?;
    let x: Vec<f64> = xm.iter().map(|r| r[0]).collect();
    let mut p = Plot::new(pl::minmax(&x), pl::minmax(&y));
    let mut start = 0usize;
    for (f, sz) in sizes.iter().enumerate() {
        for i in start..(start + *sz as usize).min(x.len()) {
            p.dot(x[i], y[i], 5.0, DOT[f % 4]);
        }
        start += *sz as usize;
    }
    let items: Vec<(String, &str)> = (0..sizes.len()).take(5).map(|f| (format!("fold {}", f + 1), BAR[f % 4])).collect();
    p.legend_row(&items);
    p.frame("x", "y", &format!("{} folds of sizes {}", sizes.len(), list(&sizes, 8)));
    Some(s.chart("data", "Data split into cross-validation folds",
                 &format!("The stated data x = {}, y = {} coloured by test fold, the folds being consecutive blocks of the sizes the \
                           model's fold-size capability returns ({}); each fold's points are predicted by a line fitted to the \
                           others.", list(&x, 12), list(&y, 12), list(&sizes, 8)), &p.body()))
}

fn kfold_folds(s: &Solved) -> Option<Figure> {
    let m = s.out_list("fold_mse")?;
    let mean = s.out("mean_mse")?;
    let labels: Vec<String> = (1..=m.len()).map(|i| format!("fold {i}")).collect();
    let mut body = hbars(&labels, &m, &[], "fold", "test MSE", &format!("mean MSE {}", num(mean)));
    // the mean as a dashed line across the bars' axis: bars start at 0, so the line is drawn on the same scale
    let (lo, hi) = pl::minmax(&m.iter().cloned().chain([0.0]).collect::<Vec<_>>());
    let ya = pl::Ax::new(lo.min(0.0), hi.max(0.0), Y1, Y0, true);
    let y = ya.px(mean);
    body.push_str(&format!("<path class=\"axp-c3\" stroke-dasharray=\"5 4\" d=\"M{X0},{y:.1} L{X1},{y:.1}\"/>"));
    Some(s.chart("folds", "Test error of each cross-validation fold",
                 &format!("{}-fold cross-validation of a straight-line fit: the returned test MSE of each fold ({}) and their mean {} \
                           (dashed).", m.len(), list(&m, 10), num(mean)), &body))
}

fn fold_grid(s: &Solved) -> Option<Figure> {
    let (n, kk) = (nin(s, "n")? as usize, nin(s, "k")? as usize);
    let ts = s.out_list("test_sizes")?;
    if n > 80 || kk > 20 || ts.len() != kk {
        return None;
    }
    let cw = (400.0 / n as f64).min(30.0);
    let rh = 22.0_f64.min(220.0 / kk as f64);
    let x0 = 100.0;
    let mut c = ax_engine::figkit::Canvas::new(70.0 + rh * kk as f64 + 20.0);
    let mut start = 0usize;
    for (f, sz) in ts.iter().enumerate() {
        let y = 50.0 + rh * f as f64;
        c.name(x0 - 8.0, y + rh * 0.7, "end", &format!("fold {}", f + 1));
        for i in 0..n {
            let test = i >= start && i < start + *sz as usize;
            c.raw(&format!("<rect class=\"{}\" x=\"{:.1}\" y=\"{y:.1}\" width=\"{:.1}\" height=\"{:.1}\"/>",
                           if test { "axp-b" } else { "axp-b4" }, x0 + cw * i as f64, cw - 1.5, rh - 3.0));
        }
        start += *sz as usize;
    }
    c.bold(x0, 30.0, "start", &format!("{n} samples, {kk} folds: test sizes {}", list(&ts, 10)));
    let body = c.body().to_string();
    Some(s.figure("folds", "k-fold split of the samples",
                  &format!("{n} samples split into {kk} folds: each row is one fold, its test samples dark, training samples light, \
                            the test blocks consecutive with the returned sizes [{}] (train [{}]).", list(&ts, 10),
                           list(&s.out_list("train_sizes")?, 10)), &body, c.h))
}

fn loocv(s: &Solved) -> Option<Figure> {
    let xm = s.mat("X")?;
    let y = s.vec("y")?;
    let n = y.len();
    let lo = s.out("loocv_mse")?;
    let mut labels = vec![];
    let mut vals = vec![];
    for kk in [2usize, 3] {
        if kk > n || labels.iter().any(|l: &String| *l == format!("{kk}-fold")) {
            continue;
        }
        if let Some(v) = s.solve_other("ml_kfold_cv_linear_regression", &[("X", tm(&xm)), ("y", tl(&y)), ("k", it(kk as i64))])
            .and_then(|a| ans(&a, "mean_mse")) {
            labels.push(format!("{kk}-fold"));
            vals.push(v);
        }
    }
    labels.push("leave-one-out".into());
    vals.push(lo);
    let hl = vec![vals.len() - 1];
    let body = hbars(&labels, &vals, &hl, "method", "mean test MSE", &format!("LOOCV MSE {}", num(lo)));
    Some(s.chart("comparison", "Leave-one-out against k-fold error",
                 &format!("Cross-validated error of a line fitted to the stated data: k-fold mean MSE from the model's k-fold capability \
                           ({}) and the returned leave-one-out MSE {} (PRESS {}); leave-one-out is {n}-fold cross-validation.",
                          labels.iter().zip(&vals).map(|(l, v)| format!("{l} {}", num(*v))).collect::<Vec<_>>().join(", "), num(lo),
                          num(s.out("press")?)), &body))
}

/// Gaussian naive Bayes: each feature on its own axis, both classes' training values and the query.
fn features(s: &Solved) -> Option<Figure> {
    let xm = s.mat("X")?;
    let y = s.vec("y")?;
    let q = s.vec("x")?;
    let nf = q.len();
    if xm.len() != y.len() || nf == 0 || nf > 6 || xm.iter().any(|r| r.len() != nf) {
        return None;
    }
    let cl = classes(&y);
    let post = s.out_list("posteriors")?;
    let mut c = ax_engine::figkit::Canvas::new(70.0 + 56.0 * nf as f64);
    let (xa, xb) = (130.0, 490.0);
    for j in 0..nf {
        let col: Vec<f64> = xm.iter().map(|r| r[j]).chain([q[j]]).collect();
        let (lo, hi) = pl::minmax(&col);
        let span = (hi - lo).max(1e-9);
        let px = |v: f64| xa + (v - lo) / span * (xb - xa);
        let yb = 60.0 + 56.0 * j as f64;
        c.line((xa, yb), (xb, yb));
        c.name(xa - 10.0, yb + 4.0, "end", &format!("feature {}", j + 1));
        c.value(xa, yb + 20.0, "middle", &num(lo));
        c.value(xb, yb + 20.0, "middle", &num(hi));
        for (r, l) in xm.iter().zip(&y) {
            let i = cl.iter().position(|cc| cc == l).unwrap_or(0);
            c.raw(&format!("<circle class=\"{}\" cx=\"{:.1}\" cy=\"{:.1}\" r=\"4.5\"/>", DOT[i % 3], px(r[j]), yb - 8.0 + 8.0 * i as f64));
        }
        c.raw(&format!("<path class=\"axp-c3\" d=\"M{:.1},{:.1} L{:.1},{:.1}\"/>", px(q[j]), yb - 18.0, px(q[j]), yb + 10.0));
    }
    c.bold(16.0, 26.0, "start", &format!("query ({}): prediction class {}", list(&q, 6), num(s.out("prediction")?)));
    let body = c.body().to_string();
    Some(s.figure("features", "Gaussian naive Bayes: each feature of both classes",
                  &format!("The stated training rows split by class (y = {}; one colour per class), one axis per feature, with the query \
                            x = ({}) as the vertical line. Returned posteriors: {}; prediction {}.", list(&y, 12), list(&q, 6),
                           cl.iter().zip(&post).map(|(c, p)| format!("class {} {}", num(*c), num(*p))).collect::<Vec<_>>().join(", "),
                           num(s.out("prediction")?)), &body, c.h))
}

/// The ROC thresholds along the score axis, each with its (false, true) positive rate.
pub fn thresholds(s: &Solved) -> Option<Figure> {
    let (y, sc) = (s.vec("y_true")?, s.vec("scores")?);
    let (th, fpr, tpr) = (s.out_list("thresholds")?, s.out_list("fpr")?, s.out_list("tpr")?);
    if y.len() != sc.len() || th.len() > 30 || fpr.len() < th.len() + 1 {
        return None;
    }
    let mut p = Plot::axes(pl::Ax::new(pl::minmax(&sc).0, pl::minmax(&sc).1, X0, X1, true), pl::Ax::new(0.0, 1.0, Y1, Y0, false));
    for (i, t) in th.iter().enumerate() {
        p.vline(*t, "axp-c4", true);
        p.place(p.x(*t), Y0 + 16.0, 15.0, 6, "axp-vt", &format!("({}, {})", num(fpr[i + 1]), num(tpr[i + 1])));
    }
    for (i, (a, l)) in sc.iter().zip(&y).enumerate() {
        p.dot(*a, if *l > 0.5 { 0.35 } else { 0.2 } + 0.02 * (i % 3) as f64, 6.0, if *l > 0.5 { "axp-d" } else { "axp-d3" });
    }
    p.legend_row(&[("label 1".into(), "axp-b"), ("label 0".into(), "axp-b3")]);
    p.frame_with("score (labels: false and true positive rate at or above the threshold)", "", &format!("AUC {}", num(s.out("auc")?)), true, false);
    let _ = (BAR, LINE, tx);
    Some(s.chart("thresholds", "ROC thresholds along the score axis",
                 &format!("The stated scores {} on one axis, coloured by label ({}); dashed: the returned thresholds {}, each labelled \
                           with the (false, true) positive rate the model returns when everything at or above it is called \
                           positive; AUC {}.", list(&sc, 12), list(&y, 12), list(&th, 12), num(s.out("auc")?)), &p.body()))
}
