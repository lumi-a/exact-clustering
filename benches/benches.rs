use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::prelude::*;
use price_of_hierarchy::{Cluster, Clustering, ContinuousKMeansCluster, DiscreteCluster};

pub fn discrete_k_median(c: &mut Criterion) {
    let points: Vec<Array1<f64>> = (0..4)
        .flat_map(|x| (0..4).map(move |y| array![f64::from(x), 1.3 * f64::from(y)]))
        .collect();
    let distances = points
        .iter()
        .map(|p| {
            points
                .iter()
                .map(|q| (p - q).iter().map(|x| x.abs()).sum())
                .collect()
        })
        .collect::<Vec<_>>();

    for k in 1..=9 {
        c.bench_function(&format!("grid 4×4, k={k}, discrete k-median"), |b| {
            b.iter(|| DiscreteCluster::optimal_clustering(black_box(&distances), black_box(k)));
        });
    }
}

pub fn continuous_k_means(c: &mut Criterion) {
    let points: Vec<Array1<f64>> = (0..4)
        .flat_map(|x| (0..4).map(move |y| array![f64::from(x), 1.3 * f64::from(y)]))
        .collect();

    for k in 1..=9 {
        c.bench_function(&format!("grid 4×4, k={k}, continuous k-means"), |b| {
            b.iter(|| {
                ContinuousKMeansCluster::optimal_clustering(black_box(&points), black_box(k))
            });
        });
    }
}

pub fn hierarchies(c: &mut Criterion) {
    let points: Vec<Array1<f64>> = (0..3)
        .flat_map(|x| (0..4).map(move |y| array![f64::from(x), 1.3 * f64::from(y)]))
        .collect();
    let distances = points
        .iter()
        .map(|p| {
            points
                .iter()
                .map(|q| (p - q).iter().map(|x| x.abs()).sum())
                .collect()
        })
        .collect::<Vec<_>>();

    c.bench_function("grid 3×4, discrete k-median hierarchy", |b| {
        b.iter(|| Clustering::<DiscreteCluster>::optimal_hierarchy(black_box(&distances)));
    });
    c.bench_function("grid 3×4, continuous k-means hierarchy", |b| {
        b.iter(|| Clustering::<ContinuousKMeansCluster>::optimal_hierarchy(black_box(&points)));
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = discrete_k_median, continuous_k_means, hierarchies
);
criterion_main!(benches);
