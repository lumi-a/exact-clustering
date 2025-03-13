//! Finding optimal clusterings and hierarchical clusterings.
//! TODO: Better crate-doc.

use core::f64;
use ndarray::{Array1, Array2, ArrayView1};
use pathfinding::{num_traits::Zero, prelude::dijkstra};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::{
    collections::{BinaryHeap, HashSet},
    hash::Hash,
};

// TODO: If we just use a u32 or something, could we use a Vec instead of a HashMap for calculating cluster-costs?
//       Almost all cluster-costs will be calculated anyway. That's like 100Mb of memory for 20 points?
/// The storage-medium for representing clusters. We'll hopefully never have to calculate clusters on more than 32
/// points, so 32 bits is enough for now.
type Storage = u32;
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Cluster(Storage);
impl Cluster {
    /// Create a new empty cluster.
    fn empty() -> Self {
        Self(0)
    }

    /// Create a new cluster containing a single point.
    fn singleton(i: usize) -> Self {
        Self(1 << i)
    }

    /// Insert a point into the cluster.
    ///
    /// TODO: Benchmark whether we should just return the mutated self instead of taking a mutref to self.
    fn insert(&mut self, i: usize) {
        self.0 |= 1 << i;
    }

    /// Check if a point is in the cluster.
    fn is_set(self, i: usize) -> bool {
        (self.0 & (1 << i)) != 0
    }

    /// The number of points in the cluster.
    fn count_set_bits(self) -> u32 {
        self.0.count_ones()
    }

    /// Iterate over the points in the cluster.
    fn iter_set_bits(self) -> impl Iterator<Item = usize> {
        let mut bits = self.0;
        (0..(Storage::BITS as usize)).filter(move |_| {
            let is_set = (bits & 1) == 1;
            bits >>= 1;
            is_set
        })
    }

    /// Merge this cluster with another one.
    fn union_with(&mut self, other: Self) {
        // TODO: Since they're both disjoint, this is the same as addition. Benchmark, please?
        self.0 |= other.0;
    }

    /// Create a cluster from an iterator of points.
    fn from_iterator(it: impl IntoIterator<Item = usize>) -> Cluster {
        let mut bits = 0 as Storage;
        for i in it {
            bits |= 1 << i;
        }
        Self(bits)
    }
}
impl std::fmt::Display for Cluster {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut result = String::with_capacity(Storage::BITS as usize);
        let mut bits = self.0;
        for _ in 0..Storage::BITS {
            if (bits & 1) == 1 {
                result.push('#');
            } else {
                result.push('.');
            }
            bits >>= 1;
        }
        write!(f, "{result}")
    }
}

/// A partition of a set of points into disjoint clusters.
type Clustering = HashSet<Cluster>;

/// A square-matrix where entry Distances[i][j] corresponds to the distance between points i and j.
type Distances = Array2<f64>;

/// A rectangular matrix whose column-vectors are the coordinate-vectors of the points.
pub type Points = Array2<f64>;

#[derive(Clone, Debug)]
/// A helper-struct for efficiently merging clusters, used in finding optimal hierarchies.
struct ClusteringNodeMergeMultiple {
    /// The clusters. We use a [`SmallVec`] because this will allocate frequently. The smallvec
    /// must remain sorted so that two Nodes with the same clusters are recognised as
    /// equal.
    ///
    /// We could also use some set (`OrderSet`, [`BTreeSet`], or even union-find) datastructure,
    /// as long as it implements `Hash`.
    ///
    /// TODO: Benchmark those three datastructures.
    clusters: SmallVec<[Cluster; 6]>,
    /// The total cost of the clustering. We keep track of this to efficiently recalculate
    /// costs after merging.
    ///
    /// TODO: Try not keeping track of it, instead having [`Self::get_all_merges`] return a delta,
    /// and using that delta in Dijkstra.
    cost: f64,
}
// Only consider `clusters` in equality-checks, costs should be near-equal anyway.
impl PartialEq for ClusteringNodeMergeMultiple {
    fn eq(&self, other: &Self) -> bool {
        self.clusters == other.clusters
    }
}
impl Eq for ClusteringNodeMergeMultiple {}
impl Hash for ClusteringNodeMergeMultiple {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.clusters.hash(state);
    }
}
impl ClusteringNodeMergeMultiple {
    /// Return all possible pairs of merges of the current clusters.
    ///
    /// TODO: Try returning an iterator instead of a vec.
    fn get_all_merges<C: Cost + ?Sized>(&self, data: &C, costs: &mut Costs) -> Vec<Self> {
        let mut nodes = Vec::with_capacity(self.clusters.len() * (self.clusters.len() - 1) / 2);

        // TODO: Use nodes.extend instead of the inner call.
        debug_assert!(self.clusters.is_sorted());
        for i in 0..(self.clusters.len() - 1) {
            // Split off cluster_i.
            let (cluster_i, clusters_minus_i) = {
                let mut clusters_minus_i = self.clusters.clone();
                // This must *not* be a swap_remove, to preserve order.
                let cluster_i = clusters_minus_i.remove(i);
                (cluster_i, clusters_minus_i)
            };
            let cost_minus_i = self.cost - data.cost(cluster_i, costs);
            // Index `i` is gone now, so the lower bound is still `i`
            for j in i..clusters_minus_i.len() {
                nodes.push({
                    let mut new_clusters = clusters_minus_i.clone();
                    let cluster_j = unsafe { new_clusters.get_unchecked_mut(i) };
                    let mut new_cost = cost_minus_i - data.cost(*cluster_j, costs);
                    cluster_j.union_with(cluster_i);
                    new_cost += data.cost(*cluster_j, costs);
                    Self {
                        clusters: new_clusters,
                        cost: new_cost,
                    }
                });
            }
        }
        debug_assert!(nodes
            .iter()
            .all(|n| (n.clusters.len() == self.clusters.len() - 1)));
        debug_assert!(nodes.iter().all(|n| (n.clusters.is_sorted())));
        debug_assert!(nodes.iter().all(|n| {
            let mut ugh = vec![0; data.num_points()];
            for cluster in &n.clusters {
                for i in cluster.iter_set_bits() {
                    ugh[i] += 1;
                }
            }
            ugh.iter().all(|&x| x == 1)
        }));
        nodes
    }
    fn new_singletons(num_points: usize) -> Self {
        let mut clusters = SmallVec::default();
        for i in 0..num_points {
            clusters.push(Cluster::singleton(i));
        }
        debug_assert!(clusters.is_sorted());
        Self {
            clusters,
            cost: 0.0,
        }
    }
    fn to_clustering(self) -> Clustering {
        self.clusters.into_iter().collect()
    }
}
#[derive(Debug, PartialEq, PartialOrd, Clone, Copy)]
pub struct MaxRatio(f64);
impl MaxRatio {
    fn new(float: f64) -> Self {
        debug_assert!(float.is_normal());
        debug_assert!(float >= 1.0 - 1e-6);
        MaxRatio(float)
    }
}
impl Eq for MaxRatio {} // I always check for NaNs.
impl Ord for MaxRatio {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // This technically constitutes a logic-error, as this doesn't fully agree
        // with partial_cmp (see total_cmp's docs), but since I always
        // expect them to be normal and larger-ish than 1.0, this won't be an issue.
        self.0.total_cmp(&other.0)
    }
}
impl std::ops::Add for MaxRatio {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        MaxRatio(self.0.max(other.0))
    }
}
impl Zero for MaxRatio {
    fn zero() -> Self {
        MaxRatio(f64::NEG_INFINITY)
    }
    fn is_zero(&self) -> bool {
        self.0 == f64::NEG_INFINITY
    }
}

type Costs = FxHashMap<Cluster, f64>;

pub trait Cost {
    fn calculate_cost(&self, cluster: Cluster) -> f64;
    fn num_points(&self) -> usize;
    fn approximate_clustering(&self, k: usize, costs: &mut Costs) -> (f64, Clustering) {
        // TODO: Benchmark this against just picking clusters at random.
        // self.greedy_hierarchy(costs).swap_remove(self.num_points() - k)
        let mut clusters = vec![Cluster::empty(); k];
        for i in 0..self.num_points() {
            clusters[i % k].insert(i);
        }
        let clustering: Clustering = clusters.into_iter().collect();
        let cost = self.total_cost(&clustering, costs);
        (cost, clustering)
    }
    fn cost(&self, cluster: Cluster, costs: &mut Costs) -> f64 {
        *costs
            .entry(cluster)
            .or_insert_with(|| Self::calculate_cost(self, cluster))
    }
    fn total_cost(&self, clustering: &Clustering, costs: &mut Costs) -> f64 {
        clustering
            .iter()
            .map(|cluster| self.cost(*cluster, costs))
            .sum()
    }
    fn optimal_clustering(&self, k: usize, costs: &mut Costs) -> Option<(f64, Clustering)> {
        #[derive(Clone, Debug)]
        struct ClusteringNodeMergeSingle {
            clusters: SmallVec<[Cluster; 6]>,
            cost: f64,
            next_to_add: usize, // TODO: Try not storing this
        }
        impl PartialEq for ClusteringNodeMergeSingle {
            fn eq(&self, other: &Self) -> bool {
                self.clusters == other.clusters
            }
        }
        impl Eq for ClusteringNodeMergeSingle {}
        impl Ord for ClusteringNodeMergeSingle {
            // Order them by highest cost first
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                // TODO: Benchmark if using partial_cmp is faster
                other
                    .cost
                    .total_cmp(&self.cost)
                    .then_with(|| self.clusters.cmp(&other.clusters))
            }
        }
        impl PartialOrd for ClusteringNodeMergeSingle {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        let num_points = self.num_points();
        let mut to_see: BinaryHeap<ClusteringNodeMergeSingle> = BinaryHeap::new();
        to_see.push(ClusteringNodeMergeSingle {
            clusters: Default::default(),
            cost: 0.0,
            next_to_add: 0,
        });

        let (approximate_clustering_cost, approximate_clustering) =
            self.approximate_clustering(k, costs);
        debug_assert_eq!(approximate_clustering.len(), k);
        let mut min_cost = approximate_clustering_cost;
        while let Some(clustering_node) = to_see.pop() {
            if clustering_node.clusters.len() == k && clustering_node.next_to_add == num_points {
                return Some((
                    clustering_node.cost,
                    clustering_node.clusters.into_iter().collect(),
                ));
            }
            if clustering_node.next_to_add < num_points {
                // TODO: Can we reduce this code-duplication somehow?
                for cluster_ix in 0..clustering_node.clusters.len() {
                    let new_clustering_node = {
                        let mut new_clustering_node = clustering_node.clone();
                        let cluster_to_edit =
                            unsafe { new_clustering_node.clusters.get_unchecked_mut(cluster_ix) };
                        new_clustering_node.cost -= Self::cost(self, *cluster_to_edit, costs);
                        cluster_to_edit.insert(new_clustering_node.next_to_add);
                        new_clustering_node.cost += Self::cost(self, *cluster_to_edit, costs);
                        new_clustering_node.next_to_add += 1;
                        debug_assert_eq!(
                            new_clustering_node.clusters[cluster_ix].count_set_bits(),
                            clustering_node.clusters[cluster_ix].count_set_bits() + 1
                        );
                        new_clustering_node
                    };
                    if new_clustering_node.cost < min_cost {
                        if new_clustering_node.clusters.len() == k
                            && new_clustering_node.next_to_add == num_points
                        {
                            min_cost = new_clustering_node.cost;
                        }
                        to_see.push(new_clustering_node);
                    }
                }
                if clustering_node.clusters.len() < k {
                    let new_clustering_node = {
                        let mut clustering_node = clustering_node.clone();
                        clustering_node
                            .clusters
                            .push(Cluster::singleton(clustering_node.next_to_add));
                        clustering_node.next_to_add += 1;
                        clustering_node
                    };
                    if new_clustering_node.cost < min_cost {
                        if new_clustering_node.clusters.len() == k
                            && new_clustering_node.next_to_add == num_points
                        {
                            min_cost = new_clustering_node.cost;
                        }
                        to_see.push(new_clustering_node);
                    }
                }
            }
        }
        // This can only happen due to floating-point-rounding-errors, or
        // if the approximate_clustering_cost was off.
        Some((approximate_clustering_cost, approximate_clustering))
    }

    #[must_use]
    fn price_of_hierarchy(&self) -> (f64, Vec<Clustering>) {
        let num_points = self.num_points();
        let mut costs = FxHashMap::default();
        let opt_for_fixed_k: Vec<f64> = std::iter::once(0.0)
            .chain((1..=num_points).map(|k| self.optimal_clustering(k, &mut costs).unwrap().0))
            .collect();

        let initial_clustering = ClusteringNodeMergeMultiple::new_singletons(num_points);
        let solution = dijkstra(
            &initial_clustering,
            |clustering| {
                // TODO: Is collecting this into a vector really necessary here?
                let opt_cost = opt_for_fixed_k[clustering.clusters.len() - 1];
                let successors = clustering.get_all_merges(self, &mut costs).into_iter().map(
                    move |new_clustering| {
                        let ratio = MaxRatio::new(new_clustering.cost / opt_cost);
                        (new_clustering, ratio)
                    },
                );
                debug_assert!(successors
                    .clone()
                    .all(|(x, _)| x.clusters.len() == clustering.clusters.len() - 1));

                successors
            },
            |clustering| clustering.clusters.len() == 1,
        )
        .expect("Dijkstra should find a solution.");
        let vec_clustering: Vec<Clustering> = solution
            .0
            .into_iter()
            .map(ClusteringNodeMergeMultiple::to_clustering)
            .collect();

        (solution.1 .0, vec_clustering)
    }

    #[must_use]
    fn greedy_hierarchy(&self, costs: &mut Costs) -> Vec<(f64, Clustering)> {
        let num_points = self.num_points();

        let mut clustering = ClusteringNodeMergeMultiple::new_singletons(num_points);
        let mut solution: Vec<(f64, Clustering)> = vec![(0.0, clustering.clone().to_clustering())];
        // TODO: We could just put costs into self.
        // Might be kinda bad for benchmarks? But no, not really, just blackbox the struct-creation.
        // We could also add a method "approximate clusters for all k" or something, which'd be useful for
        // k-median but not k-means.
        while clustering.clusters.len() > 1 {
            let best_merge = clustering
                .get_all_merges(self, costs)
                .into_iter()
                .min_by(|a, b| a.cost.total_cmp(&b.cost))
                .expect("There should always be a possible merge");
            solution.push((best_merge.cost, best_merge.clone().to_clustering()));
            clustering = best_merge;
        }

        solution
    }

    #[must_use]
    fn cost_of_greedy(&self) -> (f64, Vec<Clustering>) {
        let mut max_ratio = MaxRatio::zero();
        let mut costs = FxHashMap::default();
        let greedy_hierarchy = self.greedy_hierarchy(&mut costs);
        for (cost, clustering) in &greedy_hierarchy {
            let ratio = MaxRatio::new(
                cost / self
                    .optimal_clustering(clustering.len(), &mut costs)
                    .unwrap()
                    .0,
            );
            max_ratio = max_ratio + ratio;
        }

        let hierarchy = greedy_hierarchy.into_iter().map(|x| x.1).collect();
        (max_ratio.0, hierarchy)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Discrete(Distances);
impl Cost for Discrete {
    fn num_points(&self) -> usize {
        self.0.ncols()
    }
    fn calculate_cost(&self, cluster: Cluster) -> f64 {
        // TODO: Instead of using iter_set_bits, should you iterate over usize and check `is_set` instead?
        cluster
            .iter_set_bits()
            .map(|center_candidate_ix| {
                let center_candidate_row = self.0.row(center_candidate_ix);
                cluster
                    .iter_set_bits()
                    .map(|i| center_candidate_row[i])
                    .sum()
            })
            .min_by(f64::total_cmp)
            .unwrap_or(0.0)
    }
}

fn distances_from_points_with_metric(
    points: &Points,
    metric: impl Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64,
) -> Distances {
    let n = points.ncols();
    Array2::from_shape_fn((n, n), |(i, j)| {
        let p = points.column(i);
        let q = points.column(j);
        metric(&p, &q)
    })
}
fn distances_from_points_with_norm(
    points: &Points,
    norm: impl Fn(Array1<f64>) -> f64,
) -> Distances {
    distances_from_points_with_metric(points, |p, q| norm(p - q))
}
fn distances_from_points_with_element_norm(
    points: &Points,
    elementnorm: impl Fn(f64) -> f64,
) -> Distances {
    distances_from_points_with_norm(points, |p| p.map(|x| elementnorm(*x)).sum())
}
fn squared_euclidean_distances_from_points(points: &Points) -> Distances {
    distances_from_points_with_element_norm(points, |x| x.powi(2))
}
fn taxicab_distances_from_points(points: &Points) -> Distances {
    distances_from_points_with_element_norm(points, f64::abs)
}

impl Discrete {
    #[must_use]
    pub fn squared_euclidean_from_points(points: &Points) -> Self {
        Discrete(squared_euclidean_distances_from_points(points))
    }
    #[must_use]
    pub fn median_from_points(points: &Points) -> Self {
        Discrete(taxicab_distances_from_points(points))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct KMeans(Points);
impl Cost for KMeans {
    fn num_points(&self) -> usize {
        self.0.ncols()
    }
    fn calculate_cost(&self, cluster: Cluster) -> f64 {
        // TODO: Check we don't divide by 0
        let mut center = ndarray::Array1::zeros(self.0.nrows());
        for point_ix in cluster.iter_set_bits() {
            center += &self.0.column(point_ix);
        }
        center /= f64::from(cluster.count_set_bits());
        // TODO: Try using self.0.columns.select instead, and returning 0 with map instead of using filter_map.
        self.0
            .columns()
            .into_iter()
            .enumerate()
            .filter(|&(ix, _p)| cluster.is_set(ix))
            .map(|(_ix, p)| (&p - &center).map(|x| x.powi(2)).sum())
            .sum()
    }
    fn approximate_clustering(&self, k: usize, costs: &mut Costs) -> (f64, Clustering) {
        use clustering::kmeans;

        let max_iter = 1000; // TODO: Benchmark this?
        let samples: Vec<Vec<f64>> = self
            .0
            .columns()
            .into_iter()
            .map(|x| x.into_iter().copied().collect())
            .collect();
        let clustering = kmeans(k, &samples, max_iter);
        let mut clusters = vec![Cluster::empty(); k];
        for (point_ix, cluster_ix) in clustering.membership.iter().enumerate() {
            clusters[*cluster_ix].insert(point_ix); // TODO: Unsafe
        }
        let clustering: Clustering = clusters.into_iter().collect();
        (self.total_cost(&clustering, costs), clustering)
    }
}
impl KMeans {
    #[must_use]
    pub fn new(points: Points) -> Self {
        KMeans(points)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn square_grid() -> Points {
        // Looks like this:
        //
        //  ::  ::
        //
        //  ::  ::
        //
        array![
            [0, 1, 0, 1, 0, 1, 0, 1, 4, 5, 4, 5, 4, 5, 4, 5],
            [0, 0, 1, 1, 4, 4, 5, 5, 0, 0, 1, 1, 4, 4, 5, 5],
        ]
        .mapv(f64::from)
    }

    fn triangle_grid() -> Points {
        // Looks like this:
        //
        //  .
        //  .  .
        //                  .
        //
        //  :.              .     .
        //
        array![
            [0, 0, 2, 0, 0, 8, 32, 32, 48],
            [0, 1, 0, 16, 20, 16, 0, 12, 0]
        ]
        .mapv(f64::from)
    }

    fn clustering_from_iterators(
        it: impl IntoIterator<Item = impl IntoIterator<Item = usize>>,
    ) -> Clustering {
        it.into_iter().map(Cluster::from_iterator).collect()
    }

    #[test]
    fn cost_calculation() {
        let grid = square_grid();
        let median = Discrete::median_from_points(&grid);
        let discrete_kmeans = Discrete::squared_euclidean_from_points(&grid);
        let kmeans = KMeans::new(grid);
        for i in [0, 4, 8, 12] {
            let cluster = Cluster::from_iterator(i..(i + 4));
            assert_eq!(median.calculate_cost(cluster), 4.0);
            assert_eq!(discrete_kmeans.calculate_cost(cluster), 4.0);
            assert_eq!(kmeans.calculate_cost(cluster), 4.0 * 0.5);
        }
    }

    #[test]
    fn optimal_discrete_k_median_clustering() {
        let expected_clusters: Clustering = clustering_from_iterators([0..4, 4..8, 8..12, 12..16]);
        let discrete = Discrete::median_from_points(&square_grid());

        let (score, clusters) = discrete
            .optimal_clustering(4, &mut FxHashMap::default())
            .expect("Calculating the optimal clustering should not fail.");
        assert_eq!(
            clusters, expected_clusters,
            "Clusters should match expected clusters."
        );
        #[allow(clippy::float_cmp, reason = "This comparison should be exact.")]
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
        let discrete = Discrete::squared_euclidean_from_points(&square_grid());

        let (score, clusters) = discrete
            .optimal_clustering(4, &mut FxHashMap::default())
            .expect("Calculating the optimal clustering should not fail.");
        assert_eq!(
            clusters, expected_clusters,
            "Clusters should match expected clusters."
        );
        #[allow(clippy::float_cmp, reason = "This comparison should be exact.")]
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
        let discrete = KMeans::new(square_grid());

        let (score, clusters) = discrete
            .optimal_clustering(4, &mut FxHashMap::default())
            .expect("Calculating the optimal clustering should not fail.");

        assert_eq!(
            clusters, expected_clusters,
            "Clusters should match expected clusters."
        );
        assert!(
            (score - (4.0 * (4.0 * 0.5))).abs() < 1e-12,
            "Score should be close to the expected score"
        );
    }

    #[test]
    fn optimal_discrete_k_median_hierarchy() {
        let triangle_grid = triangle_grid();
        let discrete = Discrete::median_from_points(&triangle_grid);
        let (score, hierarchy) = discrete.price_of_hierarchy();
        assert_eq!(hierarchy.len(), triangle_grid.columns().into_iter().len());

        let expected_hierarchy: Vec<Clustering> = [
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
            vec![0..=1, 2..=2, 3..=3, 4..=4, 5..=5, 6..=6, 7..=7, 8..=8],
            vec![0..=2, 3..=3, 4..=4, 5..=5, 6..=6, 7..=7, 8..=8],
            vec![0..=2, 3..=4, 5..=5, 6..=6, 7..=7, 8..=8],
            vec![0..=2, 3..=5, 6..=6, 7..=7, 8..=8],
            vec![0..=2, 3..=5, 6..=7, 8..=8],
            vec![0..=2, 3..=5, 6..=8],
            vec![0..=5, 6..=8],
            vec![0..=8],
        ]
        .map(clustering_from_iterators)
        .to_vec();

        for (level, expected_level) in hierarchy.iter().zip(expected_hierarchy.iter()) {
            assert_eq!(
                level, expected_level,
                "Hierarchy-level should match expected hierarchy-level."
            );
        }

        #[allow(clippy::float_cmp, reason = "This comparison should be exact.")]
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

        let points: Points = array![[
            0.0,
            1e-9,
            (3.0f64.sqrt() - 1.0) / 2.0,
            (3.0 - 3.0f64.sqrt()) / 2.0,
            1.0 + 1e-9,
            1.0 + 3e-9
        ]];
        let discrete = Discrete::median_from_points(&points);
        let (score, hierarchy) = discrete.price_of_hierarchy();
        assert_eq!(hierarchy.len(), points.columns().into_iter().len());

        let expected_hierarchy: Vec<Clustering> = [
            vec![0..=0, 1..=1, 2..=2, 3..=3, 4..=4, 5..=5],
            vec![0..=1, 2..=2, 3..=3, 4..=4, 5..=5],
            vec![0..=1, 2..=2, 3..=3, 4..=5],
            vec![0..=2, 3..=3, 4..=5],
            vec![0..=2, 3..=5],
            vec![0..=5],
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
            (score - (1.0 + 3.0f64.sqrt()) / 2.0).abs() <= 1e-3,
            "Score should be close to 1.366."
        );
    }
}
