use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::prelude::*;
use pathfinding::num_traits::ToPrimitive;
use price_of_hierarchy::{Cost, Discrete, KMeans, Points};
use rustc_hash::FxHashMap;

fn grid(x: usize, y: usize) -> Points {
    /// (1+√3)/2
    const PERTURBATION: f64 = 1.366_025_403_784_438_6;
    Array2::from_shape_vec(
        (2, x * y),
        (0..x)
            .flat_map(|x| {
                (0..y).map(move |y| [x.to_f64().unwrap(), PERTURBATION * y.to_f64().unwrap()])
            })
            .flatten()
            .collect(),
    )
    .unwrap()
}

pub fn optimal_clustering(c: &mut Criterion) {
    let discrete = Discrete::median_from_points(&grid(4, 5));
    for k in 2..=5 {
        c.bench_function(&format!("4×5 grid, discrete k-median, opt k={k}"), |b| {
            b.iter(|| discrete.optimal_clustering(k, &mut black_box(FxHashMap::default())));
        });
    }
    let kmeans = KMeans::new(grid(4, 5));
    for k in 2..=5 {
        c.bench_function(&format!("4×5 grid, continuous k-means, opt k={k}"), |b| {
            b.iter(|| kmeans.optimal_clustering(k, &mut black_box(FxHashMap::default())));
        });
    }
}

pub fn hierarchies(c: &mut Criterion) {
    let discrete = Discrete::median_from_points(&grid(4, 5));
    let kmeans = KMeans::new(grid(4, 5));

    c.bench_function("4×5 grid, discrete k-median hierarchy", |b| {
        b.iter(|| {
            black_box(discrete.clone()).price_of_hierarchy();
        });
    });
    c.bench_function("4×5 grid, continuous k-means hierarchy", |b| {
        b.iter(|| {
            black_box(kmeans.clone()).price_of_hierarchy();
        });
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = optimal_clustering, hierarchies
);
criterion_main!(benches);
