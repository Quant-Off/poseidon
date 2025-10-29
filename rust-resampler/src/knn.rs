use ndarray::Array2;
use rayon::prelude::*;

pub fn knn(shape: &Array2<f64>, k: usize) -> Vec<Vec<usize>> {
    let n = shape.nrows();
    (0..n)
        .into_par_iter()
        .map(|i| {
            let query = shape.row(i);
            let mut dists: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let point = shape.row(j);
                    let diff = &query - &point;
                    let dist = diff.dot(&diff).sqrt();
                    (j, dist)
                })
                .collect();
            dists.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            dists.into_iter().take(k).map(|(j, _)| j).collect()
        })
        .collect()
}
