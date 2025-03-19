#![allow(missing_docs, reason = "Docs aren't be necessary for tests.")]
#![allow(
    clippy::tests_outside_test_module,
    reason = "This is an integration-test. This is a false-positive by clippy, see https://github.com/rust-lang/rust-clippy/issues/11024"
)]

use itertools::Itertools as _;
use ndarray::{array, Array1};
use price_of_hierarchy::*;

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

#[test]
#[expect(clippy::float_cmp, reason = "These comparisons should be exact.")]
fn cost_calculation() {
    let grid = square_grid();
    let mut median = Discrete::kmedian(&grid).expect("Creating discrete should not fail.");
    let mut discrete_kmeans = Discrete::kmeans(&grid).expect("Creating discrete should not fail.");
    let mut kmeans = KMeans::new(&grid).expect("Creating kmeans should not fail.");
    for i in [0, 4, 8, 12] {
        let cluster = cluster_from_iterator(i..(i + 4));
        assert_eq!(median.cost(cluster), 4.0);
        assert_eq!(discrete_kmeans.cost(cluster), 4.0);
        assert_eq!(kmeans.cost(cluster), 4.0 * 0.5);
    }
}

#[test]
fn optimal_discrete_k_median_clustering() {
    let expected_clusters: Clustering = clustering_from_iterators([0..4, 4..8, 8..12, 12..16]);
    let mut discrete =
        Discrete::kmedian(&square_grid()).expect("Creating discrete should not fail.");

    let (score, clusters) = discrete.optimal_clustering(4);
    assert_eq!(
        clusters, expected_clusters,
        "Clusters should match expected clusters."
    );
    #[expect(clippy::float_cmp, reason = "This comparison should be exact.")]
    {
        assert_eq!(
            score,
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

    let (score, clusters) = discrete.optimal_clustering(4);
    assert_eq!(
        clusters, expected_clusters,
        "Clusters should match expected clusters."
    );
    #[expect(clippy::float_cmp, reason = "This comparison should be exact.")]
    {
        assert_eq!(
            score,
            4.0 * (1.0 + 2.0 + 1.0), // 4 clusters, with 4 points each
            "Score should exactly match expected score."
        );
    }
}

#[test]
fn optimal_continuous_k_means_clustering() {
    let expected_clusters: Clustering = clustering_from_iterators([0..4, 4..8, 8..12, 12..16]);
    let mut kmeans = KMeans::new(&square_grid()).expect("Creating kmeans should not fail.");

    let (score, clusters) = kmeans.optimal_clustering(4);

    assert_eq!(
        clusters, expected_clusters,
        "Clusters should match expected clusters."
    );
    let expected_score = 8.0; // 4 * (4/2)
    assert!(
        (score - expected_score).abs() < 1e-12,
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

    let points = vec![
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

    assert!(
        (score - (1.0 + 3.0_f64.sqrt()) / 2.0).abs() <= 1e-3,
        "Score should be close to 1.366."
    );
}

#[test]
fn empty_instances() {
    assert_eq!(Discrete::kmedian(&[]), Err(Error::EmptyPoints));
    assert_eq!(Discrete::kmeans(&[]), Err(Error::EmptyPoints));
    assert_eq!(KMeans::new(&[]), Err(Error::EmptyPoints));
}

#[test]
#[expect(clippy::float_cmp, reason = "This should be exact.")]
fn singleton_instances() {
    fn correct_clustering<C: Cost>(problem: &mut C) {
        assert_eq!(problem.num_points(), 1);

        let approximate = problem.approximate_clustering(1);
        assert_eq!(approximate.0, 0.0);
        assert_eq!(
            approximate
                .1
                .iter()
                .map(|x| x.iter().collect_vec())
                .collect_vec(),
            vec![vec![0]]
        );
    }

    let high_dimensional_singleton = vec![Array1::from_iter((0..256).map(f64::from))];
    let mut singleton_discrete_kmedian =
        Discrete::kmedian(&high_dimensional_singleton).expect("Creating discrete should not fail.");
    let mut singleton_discrete_kmeans =
        Discrete::kmeans(&high_dimensional_singleton).expect("Creating discrete should not fail.");
    let mut singleton_continuous_kmeans =
        KMeans::new(&high_dimensional_singleton).expect("Creating kmeans should not fail.");

    correct_clustering(&mut singleton_discrete_kmedian);
    correct_clustering(&mut singleton_discrete_kmeans);
    correct_clustering(&mut singleton_continuous_kmeans);
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
        Discrete::kmedian(&points),
        Err(Error::TooManyPoints(MAX_POINT_COUNT + 1))
    );
}
