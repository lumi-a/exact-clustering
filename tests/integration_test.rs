#![allow(missing_docs, reason = "Not part of the API.")]
#![allow(
    clippy::indexing_slicing,
    reason = "Panicking indexing is fine in tests."
)]
#![allow(
    clippy::tests_outside_test_module,
    reason = "This is an integration-test. This is a false-positive by clippy, see https://github.com/rust-lang/rust-clippy/issues/11024"
)]

use core::iter;

use exact_clustering::*;
use itertools::Itertools as _;
use ndarray::{array, Array1};

fn clustering_from_iterators<I, J>(it: I) -> Clustering
where
    I: IntoIterator<Item = J>,
    J: IntoIterator<Item = usize>,
{
    it.into_iter().map(cluster_from_iterator).collect()
}

fn square_grid() -> Vec<Point> {
    // Looks like this:
    //
    //  ::  ::
    //
    //  ::  ::
    //
    vec![
        array![0.0, 0.0],
        array![1.0, 0.0],
        array![0.0, 1.0],
        array![1.0, 1.0],
        array![0.0, 4.0],
        array![1.0, 4.0],
        array![0.0, 5.0],
        array![1.0, 5.0],
        array![4.0, 0.0],
        array![5.0, 0.0],
        array![4.0, 1.0],
        array![5.0, 1.0],
        array![4.0, 4.0],
        array![5.0, 4.0],
        array![4.0, 5.0],
        array![5.0, 5.0],
    ]
}

fn weighted_square() -> Vec<WeightedPoint> {
    // Looks like this:
    //
    //  ∙  ∙
    //
    //  ⬤  ●

    vec![
        (4.0, array![0.0, 0.0]),
        (2.0, array![1.0, 0.0]),
        (1.0, array![0.0, 1.0]),
        (1.0, array![1.0, 1.0]),
    ]
}

fn triangle_grid() -> Vec<Point> {
    // Looks like this:
    //
    //  .
    //  .  .
    //                  .
    //
    //  :.              .     .
    //
    vec![
        array![0.0, 0.0],
        array![0.0, 1.0],
        array![2.0, 0.0],
        array![0.0, 16.0],
        array![0.0, 20.0],
        array![8.0, 16.0],
        array![32.0, 0.0],
        array![32.0, 12.0],
        array![48.0, 0.0],
    ]
}

fn get_high_kmeans_price_of_greedy_instance(d: u8) -> Vec<WeightedPoint> {
    // Construct the example from https://arxiv.org/abs/1907.05094v1, section 4,
    // in `d` dimensions.

    // The paper used:
    //  {-1, -(√2 - 1), (√2 - 1), 1},
    // As the first coordinates. To politely convince the greedy-hierarchy to pick
    // the worst-possible clustering, move the two points in the middle a bit closer
    // together:
    iter::once(vec![
        -1.0,
        -(2.0_f64.sqrt() - 1.0 - 1e-8),
        2.0_f64.sqrt() - 1.0 - 1e-8,
        1.0,
    ])
    .chain((2..=d).map(|i| {
        // Also move these closer together:
        let z = (3.0_f64.powi(i32::from(i) - 2) / 2.0_f64.powi(i32::from(i) - 1)).sqrt() - 1e-8;
        vec![z, -z]
    }))
    .multi_cartesian_product()
    .map(|v| {
        (
            // The paper puts the weights as ∞ and 1.0, but we don't allow weights to be infinite
            // to avoid ambiguity when multiple points in the same cluster have infinite weight.
            #[expect(clippy::float_cmp, reason = "This comparison sholud be exact.")]
            if v[0].abs() == 1.0 { 1e16 } else { 1.0 },
            Array1::from_vec(v),
        )
    })
    .sorted_unstable_by(|(a, _): &WeightedPoint, (b, _)| b.total_cmp(a))
    .collect()
}

#[test]
fn high_kmeans_price_of_greedy() {
    // TODO: Consider also including k=4 here, if it allows the test to pass quickly enough
    // after future optimizations.
    for d in 1..=3 {
        let weighted_points = get_high_kmeans_price_of_greedy_instance(d);
        let mut kmeans = WeightedKMeans::new(&weighted_points)
            .expect("Creating WeightedKMeans should not fail.");

        let optimal_clusterings = kmeans.optimal_clusterings();
        let greedy_clusterings = kmeans.greedy_hierarchy();
        let optimal_cost = optimal_clusterings[2_usize.pow(u32::from(d))].0;
        let greedy_cost = greedy_clusterings[2_usize.pow(u32::from(d))].0;

        let expected_optimal_cost = 2.0_f64.powi(i32::from(d)) * (2.0 - 2.0_f64.sqrt()).powi(2);
        let expected_greedy_cost = 4.0_f64.mul_add(
            3.0_f64.powi(i32::from(d) - 1),
            2.0_f64.powi(i32::from(d) - 1) * (2.0 - 2.0_f64.sqrt()).powi(2),
        ) - 2.0_f64.powi(i32::from(d));

        assert!(optimal_cost - 1e-4 < expected_optimal_cost);
        assert!(optimal_cost + 1e-4 > expected_optimal_cost);

        assert!(greedy_cost - 1e-4 < expected_greedy_cost);
        assert!(greedy_cost + 1e-4 > expected_greedy_cost);
    }
}

#[test]
#[expect(clippy::float_cmp, reason = "These comparisons should be exact.")]
fn cost() {
    let grid = square_grid();
    let mut kmedian = Discrete::kmedian(&grid).expect("Creating discrete should not fail.");
    let mut discrete_kmeans = Discrete::kmeans(&grid).expect("Creating discrete should not fail.");
    let mut kmeans = KMeans::new(&grid).expect("Creating kmeans should not fail.");
    for i in [0, 4, 8, 12] {
        let cluster = cluster_from_iterator(i..(i + 4));
        assert_eq!(kmedian.cost(cluster), 4.0);
        assert_eq!(discrete_kmeans.cost(cluster), 4.0);
        assert_eq!(kmeans.cost(cluster), 4.0 * 0.5);
    }
}

#[test]
#[expect(clippy::float_cmp, reason = "These comparisons should be exact.")]
fn weighted_cost() {
    let square = weighted_square();
    let mut kmedian =
        Discrete::weighted_kmedian(&square).expect("Creating kmedian should not fail.");
    let mut discrete_kmeans =
        Discrete::weighted_kmeans(&square).expect("Creating discrete kmeans should not fail.");
    let mut kmeans = WeightedKMeans::new(&square).expect("Creating kmeans should not fail.");

    let full_cluster = cluster_from_iterator(0..4);
    assert_eq!(kmedian.cost(full_cluster), 5.0);
    assert_eq!(discrete_kmeans.cost(full_cluster), 5.0);
    // Centroid should be at (0.375, 0.25)
    assert_eq!(kmeans.cost(full_cluster), 3.375);
}

#[test]
fn optimal_discrete_k_median_clustering() {
    let expected_clusters: Clustering = clustering_from_iterators([0..4, 4..8, 8..12, 12..16]);
    let mut discrete =
        Discrete::kmedian(&square_grid()).expect("Creating discrete should not fail.");

    let optimals = discrete.optimal_clusterings();
    let (score, clusters) = &optimals[4];
    assert_eq!(
        *clusters, expected_clusters,
        "Clusters should match expected clusters."
    );
    #[expect(clippy::float_cmp, reason = "This comparison should be exact.")]
    {
        assert_eq!(
            *score,
            4.0 * (1.0 + 2.0 + 1.0), // 4 clusters, with 4 points each
            "Score should exactly match expected score."
        );
    }
}

#[test]
fn optimal_discrete_k_means_clustering() {
    let expected_clusters: Clustering = clustering_from_iterators([0..4, 4..8, 8..12, 12..16]);
    let mut discrete =
        Discrete::kmeans(&square_grid()).expect("Creating discrete should not fail.");

    let (score, clusters) = &discrete.optimal_clusterings()[4];
    assert_eq!(
        *clusters, expected_clusters,
        "Clusters should match expected clusters."
    );
    #[expect(clippy::float_cmp, reason = "This comparison should be exact.")]
    {
        assert_eq!(
            *score,
            4.0 * (1.0 + 2.0 + 1.0), // 4 clusters, with 4 points each
            "Score should exactly match expected score."
        );
    }
}

#[test]
fn optimal_continuous_k_means_clustering() {
    let expected_clusters: Clustering = clustering_from_iterators([0..4, 4..8, 8..12, 12..16]);
    let mut kmeans = KMeans::new(&square_grid()).expect("Creating kmeans should not fail.");

    let (score, clusters) = &kmeans.optimal_clusterings()[4];

    assert_eq!(
        *clusters, expected_clusters,
        "Clusters should match expected clusters."
    );
    let expected_score = 8.0; // 4 * (4/2)
    assert!(
        (*score - expected_score).abs() < 1e-12,
        "Score should be close to the expected score"
    );
}

#[test]
fn optimal_discrete_k_median_hierarchy() {
    let triangle_grid = triangle_grid();
    let mut discrete =
        Discrete::kmedian(&triangle_grid).expect("Creating discrete should not fail.");
    assert_eq!(discrete.num_points(), triangle_grid.len());
    let (score, hierarchy) = discrete.price_of_hierarchy();
    assert_eq!(hierarchy.len(), triangle_grid.len() + 1);

    let expected_hierarchy: Vec<Clustering> = [
        vec![],
        vec![0..=8],
        vec![0..=5, 6..=8],
        vec![0..=2, 3..=5, 6..=8],
        vec![0..=2, 3..=5, 6..=7, 8..=8],
        vec![0..=2, 3..=5, 6..=6, 7..=7, 8..=8],
        vec![0..=2, 3..=4, 5..=5, 6..=6, 7..=7, 8..=8],
        vec![0..=2, 3..=3, 4..=4, 5..=5, 6..=6, 7..=7, 8..=8],
        vec![0..=1, 2..=2, 3..=3, 4..=4, 5..=5, 6..=6, 7..=7, 8..=8],
        vec![
            0..=0,
            1..=1,
            2..=2,
            3..=3,
            4..=4,
            5..=5,
            6..=6,
            7..=7,
            8..=8,
        ],
    ]
    .map(clustering_from_iterators)
    .to_vec();

    for (level, expected_level) in hierarchy.iter().zip(expected_hierarchy.iter()) {
        assert_eq!(
            level, expected_level,
            "Hierarchy-level should match expected hierarchy-level."
        );
    }

    #[expect(clippy::float_cmp, reason = "This comparison should be exact.")]
    {
        assert_eq!(
            score, 1.0,
            "Each level of the hierarchy should be an optimal clustering."
        );
    }
}

#[test]
fn suboptimal_discrete_k_median_hierarchy() {
    // Points like this:
    //
    // ..    .     .     ..
    //

    let points = [
        array![0.0],
        array![1e-9],
        array![(3.0_f64.sqrt() - 1.0) / 2.0],
        array![(3.0 - 3.0_f64.sqrt()) / 2.0],
        array![1.0 + 1e-9],
        array![1.0 + 3e-9],
    ];
    let mut discrete = Discrete::kmedian(&points).expect("Creating discrete should not fail.");
    assert_eq!(discrete.num_points(), points.len());
    let (score, hierarchy) = discrete.price_of_hierarchy();
    assert_eq!(hierarchy.len(), points.len() + 1);
    assert!(
        (score - (1.0 + 3.0_f64.sqrt()) / 2.0).abs() <= 1e-3,
        "Score {score} should be close to 1.366."
    );

    let expected_hierarchy: Vec<Clustering> = [
        vec![],
        vec![0..=5],
        vec![0..=2, 3..=5],
        vec![0..=2, 3..=3, 4..=5],
        vec![0..=1, 2..=2, 3..=3, 4..=5],
        vec![0..=1, 2..=2, 3..=3, 4..=4, 5..=5],
        vec![0..=0, 1..=1, 2..=2, 3..=3, 4..=4, 5..=5],
    ]
    .map(clustering_from_iterators)
    .to_vec();

    for (level, expected_level) in hierarchy.iter().zip(expected_hierarchy.iter()) {
        assert_eq!(
            level, expected_level,
            "Hierarchy-level should match expected hierarchy-level."
        );
    }
}

#[test]
fn suboptimal_weighted_discrete_k_median() {
    // Points like this:
    //
    // o    .   .    o
    let weighted_points = [
        (1e6, array![0.0]),
        (1.0, array![1.0 + 1e-6]),
        (1.0, array![2.0 - 1e-6]),
        (1e6, array![3.0]),
    ];

    let mut discrete =
        Discrete::weighted_kmedian(&weighted_points).expect("Creating discrete should not fail.");
    assert_eq!(discrete.num_points(), weighted_points.len());

    let (greedy_score, greedy_hierarchy) = discrete.price_of_greedy();
    assert_eq!(greedy_hierarchy.len(), weighted_points.len() + 1);
    assert!(((1.5 - 1e-5)..1.5).contains(&greedy_score));

    let (hierarchy_score, optimal_hierarchy) = discrete.price_of_hierarchy();
    assert_eq!(optimal_hierarchy.len(), weighted_points.len() + 1);
    assert!((1.0..(1.0 + 1e-5)).contains(&hierarchy_score));
}

#[test]
fn negative_weighted_instances() {
    #[expect(
        unused_results,
        reason = "We only care about the weights being accepted, not the results."
    )]
    for nonnegatively_weighted_point in [(1.0, array![0.0]), (1e100, array![0.0])] {
        let extended = [nonnegatively_weighted_point];
        Discrete::weighted_kmedian(&extended).expect("These weights should be accepted");
        Discrete::weighted_kmeans(&extended).expect("These weights should be accepted");
        WeightedKMeans::new(&extended).expect("These weights should be accepted");
    }

    for negatively_weighted_point in [
        (-1.0, array![0.0]),
        (-0.0, array![0.0]),
        (0.0, array![0.0]),
        (f64::NAN, array![0.0]),
        (f64::INFINITY, array![0.0]),
        (f64::NEG_INFINITY, array![0.0]),
    ] {
        let extended = [
            (1.0, array![0.0]),
            negatively_weighted_point,
            (1.0, array![0.0]),
        ];
        assert_eq!(
            Discrete::weighted_kmedian(&extended).err(),
            Some(Error::BadWeight(1))
        );
        assert_eq!(
            Discrete::weighted_kmeans(&extended).err(),
            Some(Error::BadWeight(1))
        );
        assert_eq!(
            WeightedKMeans::new(&extended).err(),
            Some(Error::BadWeight(1))
        );
    }
}

#[test]
fn empty_instances() {
    assert_eq!(Discrete::kmedian(&[]).err(), Some(Error::EmptyPoints));
    assert_eq!(Discrete::kmeans(&[]).err(), Some(Error::EmptyPoints));
    assert_eq!(
        Discrete::weighted_kmedian(&[]).err(),
        Some(Error::EmptyPoints)
    );
    assert_eq!(
        Discrete::weighted_kmeans(&[]).err(),
        Some(Error::EmptyPoints)
    );
    assert_eq!(KMeans::new(&[]).err(), Some(Error::EmptyPoints));
    assert_eq!(WeightedKMeans::new(&[]).err(), Some(Error::EmptyPoints));
}

#[test]
#[expect(clippy::float_cmp, reason = "This should be exact.")]
fn singleton_instances() {
    fn correct_clustering<C: Cost>(maybe_problem: Result<C, Error>) {
        let mut problem = maybe_problem.expect("Creating problem should not fail.");
        assert_eq!(problem.num_points(), 1);

        let (score, clusters) = &problem.approximate_clusterings()[1];
        assert_eq!(*score, 0.0);
        assert_eq!(
            clusters
                .iter()
                .map(|x| x.iter().collect_vec())
                .collect_vec(),
            vec![vec![0]]
        );
    }

    let high_dimensional_singleton = [Array1::from_iter((0..256).map(f64::from))];
    let high_dimensional_weighted_singleton = [(1.0, Array1::from_iter((0..256).map(f64::from)))];

    correct_clustering(Discrete::kmedian(&high_dimensional_singleton));
    correct_clustering(Discrete::kmeans(&high_dimensional_singleton));
    correct_clustering(Discrete::weighted_kmedian(
        &high_dimensional_weighted_singleton,
    ));
    correct_clustering(Discrete::weighted_kmeans(
        &high_dimensional_weighted_singleton,
    ));
    correct_clustering(KMeans::new(&high_dimensional_singleton));
    correct_clustering(WeightedKMeans::new(&high_dimensional_weighted_singleton));
}

#[test]
#[expect(clippy::float_cmp, reason = "These comparisons should be exact.")]
#[expect(clippy::cast_precision_loss, reason = "The casts here are safe.")]
#[expect(clippy::as_conversions, reason = "The casts here are safe.")]
fn linear_exponential_hierarchy() {
    fn correct_hierarchy(i: usize, (score, hierarchy): (f64, Vec<Clustering>)) {
        assert_eq!(score, 1.0);

        for (k, level) in hierarchy.iter().enumerate() {
            // We expect it to look like {.}{.}{.}{..........}
            let mut expected_level = vec![];
            if k > 0 {
                for j in 0..(k - 1) {
                    expected_level.push(vec![j]);
                }
                expected_level.push(((k - 1)..i).collect_vec());
            }
            assert_eq!(
                level
                    .iter()
                    .sorted()
                    .map(|cluster| cluster.iter().collect_vec())
                    .collect_vec(),
                expected_level
            );
        }
    }

    let mut points = vec![];
    for i in 1..=MAX_POINT_COUNT {
        points.push(array![(-(i as f64)).exp()]);
        assert_eq!(points.len(), i);

        let mut kmedian = Discrete::kmedian(&points).expect("Creating discrete should not fail.");

        correct_hierarchy(i, kmedian.price_of_greedy());
        correct_hierarchy(i, kmedian.price_of_hierarchy());
    }
    points.push(array![((MAX_POINT_COUNT + 1) as f64).exp()]);
    assert_eq!(
        Discrete::kmedian(&points).err(),
        Some(Error::TooManyPoints(MAX_POINT_COUNT + 1))
    );
}
