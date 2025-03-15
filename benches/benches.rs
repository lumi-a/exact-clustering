//! Benchmarks for the `price_of_hierarchy` crate.

#![allow(
    unused_results,
    reason = "Criterion's `.bench_function` returns results we won't handle."
)]
#![allow(
    missing_docs,
    reason = "Doc is not needed for benchmarks. Also, criterion-macros create functions I can't document."
)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::prelude::*;
use price_of_hierarchy::{Cost as _, Discrete, KMeans, Points};

/// Create a 2d-grid of `width`×`height` points.
fn grid(width: u32, height: u32) -> Points {
    /// (1+√3)/2
    const PERTURBATION: f64 = 1.366_025_403_784_438_6;

    (0..width)
        .flat_map(|x| (0..height).map(move |y| array![f64::from(x), PERTURBATION * f64::from(y)]))
        .collect()
}

pub fn optimal_clustering(c: &mut Criterion) {
    let grid = grid(4, 4);
    c.bench_function(
        &format!("4×4 grid, discrete k-median, opt for k=1..={}", grid.len()),
        |b| {
            b.iter(|| {
                (1..=grid.len()).for_each(|k| {
                    black_box(Discrete::median_from_points(&grid).unwrap()).optimal_clustering(k);
                });
            });
        },
    );
    c.bench_function(
        &format!("4×4 grid, continuous k-means, opt for k=1..={}", grid.len()),
        |b| {
            b.iter(|| {
                (1..=grid.len()).for_each(|k| {
                    black_box(KMeans::new(&grid).unwrap()).optimal_clustering(k);
                });
            });
        },
    );
}

pub fn hierarchies(c: &mut Criterion) {
    let grid = grid(4, 4);

    c.bench_function("4×4 grid, discrete k-median hierarchy", |b| {
        b.iter(|| {
            let _: f64 = black_box(Discrete::median_from_points(&grid).unwrap())
                .price_of_hierarchy()
                .0;
        });
    });
    c.bench_function("4×4 grid, continuous k-means hierarchy", |b| {
        b.iter(|| {
            let _: f64 = black_box(KMeans::new(&grid).unwrap())
                .price_of_hierarchy()
                .0;
        });
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = optimal_clustering, hierarchies
);
criterion_main!(benches);
