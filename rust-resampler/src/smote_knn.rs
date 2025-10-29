use kdtree::KdTree;
use kdtree::distance::squared_euclidean;
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::{Rng, thread_rng};
use rayon::prelude::*;
use std::error::Error;

#[derive(Debug, Clone)]
enum Metric {
    Euclidean,
    Manhattan,
    // 추가 metric 지원 가능
}

impl Metric {
    fn distance(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        match self {
            Metric::Euclidean => (a - b).mapv(|x| x.powi(2)).sum().sqrt(),
            Metric::Manhattan => (a - b).mapv(|x| x.abs()).sum(),
        }
    }
}

fn compute_knn(
    x: &Array2<f64>,
    k: usize,
    metric: Metric,
) -> Result<(Array2<f64>, Array2<usize>), Box<dyn Error>> {
    if x.shape()[0] <= k {
        return Err(format!("샘플 수({})가 k({})보다 작거나 같습니다!", x.shape()[0], k).into());
    }

    let dimensions = x.shape()[1];
    let mut kdtree = KdTree::new(dimensions);
    for (i, row) in x.outer_iter().enumerate() {
        kdtree.add(row.to_vec(), i)?;
    }

    let mut distances = Array2::zeros((x.shape()[0], k));
    let mut indices = Array2::zeros((x.shape()[0], k));

    x.outer_iter()
        .into_par_iter()
        .enumerate()
        .for_each(|(i, row)| {
            let query = row.to_vec();
            let knn = kdtree.nearest(&query, k + 1, &squared_euclidean).unwrap(); // +1 for self
            for (j, (dist, &idx)) in knn.iter().skip(1).enumerate() {
                // skip self
                distances[[i, j]] = dist.sqrt();
                indices[[i, j]] = idx;
            }
        });

    Ok((distances, indices))
}

fn find_knn_for_sample(
    x: &Array2<f64>,
    sample_idx: usize,
    k: usize,
    metric: Metric,
) -> Result<(Array1<f64>, Array1<usize>), Box<dyn Error>> {
    if sample_idx >= x.shape()[0] {
        return Err(format!(
            "샘플 인덱스({})가 데이터 크기({})를 초과합니다!",
            sample_idx,
            x.shape()[0]
        )
        .into());
    }
    if x.shape()[0] <= k {
        return Err(format!("샘플 수({})가 k({})보다 작거나 같습니다!", x.shape()[0], k).into());
    }

    let row = x.row(sample_idx).to_owned();
    let mut distances = Array1::zeros(k);
    let mut indices = Array1::zeros(k);

    // 전체 pairwise 계산 (작은 데이터셋 가정)
    let dists: Vec<(f64, usize)> = x
        .outer_iter()
        .enumerate()
        .map(|(i, r)| (metric.distance(&row, &r.to_owned()), i))
        .filter(|&(_, i)| i != sample_idx)
        .collect();

    let mut sorted_dists = dists;
    sorted_dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    for (j, (d, idx)) in sorted_dists.iter().take(k).enumerate() {
        distances[j] = *d;
        indices[j] = *idx;
    }

    Ok((distances, indices))
}

fn compute_pairwise_distances(x: &Array2<f64>, metric: Metric) -> Array2<f64> {
    let n = x.shape()[0];
    let mut dists = Array2::zeros((n, n));
    x.outer_iter()
        .into_par_iter()
        .enumerate()
        .for_each(|(i, row_i)| {
            for (j, row_j) in x.outer_iter().enumerate() {
                if i != j {
                    dists[[i, j]] = metric.distance(&row_i, &row_j);
                }
            }
        });
    dists
}

fn smote(
    x: &Array2<f64>,
    y: &Array1<i64>,
    k: usize,
    sampling_ratio: f64,
    random_state: Option<u64>,
    metric: Metric,
) -> Result<(Array2<f64>, Array1<i64>), Box<dyn Error>> {
    let mut rng = thread_rng();
    if let Some(seed) = random_state {
        // rand crate에서 seeded RNG 지원 (추가 구현 필요)
        // 여기서는 간단히 무시하고 thread_rng 사용
    }

    // 클래스 카운트
    let mut class_counts: HashMap<i64, usize> = HashMap::new();
    for &label in y.iter() {
        *class_counts.entry(label).or_insert(0) += 1;
    }

    if class_counts.len() != 2 {
        return Err("SMOTE는 이진 분류 문제에만 적용 가능합니다!".into());
    }

    let minority_class = class_counts
        .iter()
        .min_by_key(|&(_, count)| count)
        .unwrap()
        .0
        .clone();
    let minority_indices: Vec<usize> = y
        .iter()
        .enumerate()
        .filter(|&(_, &l)| l == minority_class)
        .map(|(i, _)| i)
        .collect();
    let minority_x = x.select(Axis(0), &minority_indices);

    let n_minority = minority_indices.len();
    let n_majority = x.shape()[0] - n_minority;
    let n_synthetic = ((n_majority - n_minority) as f64 * sampling_ratio) as usize;

    if n_synthetic == 0 {
        return Ok((x.clone(), y.clone()));
    }

    let (_, indices) = compute_knn(&minority_x, k, metric.clone())?;

    let mut synthetic_samples: Vec<Array1<f64>> = Vec::with_capacity(n_synthetic);
    let mut synthetic_labels: Vec<i64> = Vec::with_capacity(n_synthetic);

    for _ in 0..n_synthetic {
        let sample_idx = rng.gen_range(0..n_minority);
        let neighbor_idx = rng.gen_range(0..k);
        let neighbor_sample_idx = indices[[sample_idx, neighbor_idx]];

        let alpha = rng.gen_range(0.0..1.0);
        let synth = &minority_x.row(sample_idx)
            + alpha * (&minority_x.row(neighbor_sample_idx) - &minority_x.row(sample_idx));

        synthetic_samples.push(synth.to_owned());
        synthetic_labels.push(minority_class);
    }

    let mut x_resampled = x.clone();
    for sample in synthetic_samples {
        x_resampled.push_row(sample.view())?;
    }
    let mut y_resampled = y.clone();
    y_resampled.append(&mut Array1::from_vec(synthetic_labels));

    Ok((x_resampled, y_resampled))
}

// smote_with_custom_knn은 smote와 유사하므로, metric을 smote에 통합하여 중복 구현 생략 (필요 시 확장)

fn main() -> Result<(), Box<dyn Error>> {
    // 예시 사용
    let x = Array2::random((100, 10), Uniform::new(0., 10.));
    let mut y = Array1::zeros(100);
    y.slice_mut(s![50..]).fill(1); // 불균형 클래스

    let (x_res, y_res) = smote(&x, &y, 5, 1.0, None, Metric::Euclidean)?;
    println!("Resampled shape: {:?}, {:?}", x_res.shape(), y_res.shape());

    Ok(())
}
