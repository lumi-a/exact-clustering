//! Find optimal [clusterings](https://en.wikipedia.org/wiki/Cluster_analysis) and
//! [hierarchical clusterings](https://en.wikipedia.org/wiki/Hierarchical_clustering) on up to 32 points.
//! If you only need approximate clusterings, there are excellent
//! [other crates](https://www.arewelearningyet.com/clustering/) available that run significantly faster.
//!
//! To create discrete (weighted or unweighted) clustering-problems, see [`Discrete`].
//! For continuous kmeans-clustering, see [`KMeans`] and [`WeightedKMeans`].
//! If you'd like to solve other clustering-problems, implement the [`Cost`]-trait (and feel free to submit
//! a pull-request!), or submit an issue on GitHub.
//!
//! Among others, the [`Cost`]-trait allows you to calculate:
//! - Optimal clusterings using [`Cost::optimal_clusterings`]
//! - Optimal hierarchical clusterings using [`Cost::price_of_hierarchy`]
//! - Greedy hierarchical clusterings using [`Cost::price_of_greedy`]
//!
//! # Example
//!
//! ```
//! use ndarray::prelude::*;
//! use std::collections::BTreeSet;
//! use exact_clustering::{Cost as _, Discrete, KMeans};
//!
//! // Set of 2d-points looking like ⠥
//! // This has a uniqe optimal 2-clustering for all three problems.
//! let points = vec![
//!     array![0.0, 0.0],
//!     array![1.0, 0.0],
//!     array![0.0, 2.0],
//! ];
//! // Instances are mutable to allow caching cluster-costs
//! let mut discrete_kmedian = Discrete::kmedian(&points).unwrap();
//! // All optimal clusterings are calculated at once to permit some speedups.
//! let (cost, clusters) = &discrete_kmedian.optimal_clusterings()[2];
//!
//! assert_eq!(*cost, 1.0);
//! // Each cluster in the returned clustering is a set of point-indices:
//! assert_eq!(
//!     BTreeSet::from([BTreeSet::from([0, 1]), BTreeSet::from([2])]),
//!     clusters
//!         .iter()
//!         .cloned()
//!         .map(BTreeSet::from_iter)
//!         .collect(),
//! );
//!
//! let price_of_hierarchy = discrete_kmedian.price_of_hierarchy().0;
//! assert_eq!(price_of_hierarchy, 1.0);
//!
//! let price_of_greedy = discrete_kmedian.price_of_greedy().0;
//! assert_eq!(price_of_greedy, 1.0);
//! ```

#![expect(
    clippy::missing_errors_doc,
    reason = "The Error-Enum is sparse and documented."
)]

use core::hash::{self, Hash};
use core::{cmp, ops};
use core::{f64, fmt, iter};
use ndarray::Array1;
use pathfinding::{num_traits::Zero, prelude::dijkstra};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::collections::BinaryHeap;

/// The storage-medium for representing clusters compactly via a bitset.
///
/// We'll hopefully never have to calculate clusters on more than 32 points, so 32 bits is enough for now.
///
/// If you are not on a 32-bit-or-above-platform, this will cause issues with indices.
#[cfg(not(target_pointer_width = "16"))]
pub type Storage = u32;

/// The maximum number of points we can cluster before we overflow [`Storage`].
#[expect(
    clippy::as_conversions,
    reason = "`Storage::BITS` will always fit into a `usize`."
)]
pub const MAX_POINT_COUNT: usize = Storage::BITS as usize;

/// A compact representation of a cluster of points, using a bitset.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Cluster(Storage);
impl Cluster {
    /// Create a new empty cluster containing no points.
    const fn empty() -> Self {
        Self(0)
    }

    /// Create a new cluster containing a single point.
    const fn singleton(point_ix: usize) -> Self {
        Self(1 << point_ix)
    }

    /// Insert a point into the cluster.
    fn insert(&mut self, point_ix: usize) {
        let point = 1 << point_ix;
        debug_assert!(
            (point & self.0) == 0,
            "Throughout the entire implementation, we should never to add the same point twice."
        );
        self.0 |= point;
    }

    /// Remove a point from the cluster.
    fn remove(&mut self, point_ix: usize) {
        let point = 1 << point_ix;
        debug_assert!(
            (point & self.0) != 0,
            "Throughout the entire implementation, we should never remove a non-existing point."
        );
        self.0 &= !point;
    }

    /// Check whether the set contains a point-index.
    #[must_use]
    #[inline]
    pub const fn contains(self, point_ix: usize) -> bool {
        (self.0 & (1 << point_ix)) != 0
    }

    /// Return the number of points in the cluster.
    #[must_use]
    #[inline]
    pub const fn len(self) -> Storage {
        self.0.count_ones()
    }

    /// Return whether the cluster contains no points.
    #[must_use]
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Return an iterator over the point-indices in the cluster.
    #[inline]
    #[must_use]
    pub const fn iter(self) -> ClusterIter {
        ClusterIter(self.0)
    }

    /// Merge this cluster with another one.
    fn union_with(&mut self, other: Self) {
        debug_assert!(
            self.0 & other.0 == 0,
            "Troughout the entire implementation, we should never be merging intersecting clusters."
        );
        self.0 |= other.0;
    }
}

/// An iterator over the points in a cluster.
#[derive(Debug, Clone)]
pub struct ClusterIter(Storage);
impl Iterator for ClusterIter {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.0 == 0 {
            None
        } else {
            #[expect(
                clippy::as_conversions,
                reason = "I assume `usize` is at least `Storage`."
            )]
            let ix = self.0.trailing_zeros() as usize;
            self.0 &= self.0 - 1;
            Some(ix)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        #[expect(
            clippy::as_conversions,
            reason = "I assume `usize` is at least `Storage`."
        )]
        let count = self.0.count_ones() as usize;
        (count, Some(count))
    }
}

impl IntoIterator for Cluster {
    type Item = usize;
    type IntoIter = ClusterIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        ClusterIter(self.0)
    }
}

impl fmt::Display for Cluster {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        #[expect(
            clippy::as_conversions,
            reason = "I assume `usize` is at least `Storage`."
        )]
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
pub type Clustering = FxHashSet<Cluster>;

/// Entry `Distances[i][j]` corresponds to the distance between points `i` and `j`.
type Distances = Vec<Vec<f64>>;

/// A single point.
pub type Point = Array1<f64>;
/// A weighted point. The distance between two weighted points `d((w0, p0), (w1, p1))` is `w1 * d(p0, p1)`.
pub type WeightedPoint = (f64, Array1<f64>);

#[derive(Clone, Debug)]
/// A helper-struct for efficiently merging clusters, used in finding optimal hierarchies.
struct ClusteringNodeMergeMultiple {
    /// The clusters. We use a [`SmallVec`] because this will allocate frequently. The smallvec
    /// must remain sorted so that two Nodes with the same clusters are recognised as
    /// equal.
    ///
    /// TODO: Once [`generic_const_exprs`](https://github.com/rust-lang/rust/issues/76560) is stable,
    /// try using an array with dynamic dispatch instead. Consider hiding this behind a feature-gate
    /// to reduce compile-times.
    clusters: SmallVec<[Cluster; 6]>,
    /// The total cost of the clustering. We keep track of this to efficiently recalculate
    /// costs after merging.
    ///
    /// TODO: Try not keeping track of it, instead having [`ClusteringNodeMergeMultiple::get_all_merges`]
    /// return a delta, and using that delta in Dijkstra.
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
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.clusters.hash(state);
    }
}
impl ClusteringNodeMergeMultiple {
    /// Return all possible pairs of merges of the current clusters.
    ///
    /// If performance is at a premium, you can avoid the vec-allocation by inlining the loops at the
    /// callsite, but that increases code-duplication (this method is required in [`Cost::price_of_hierarchy`]
    /// and [`Cost::price_of_greedy`]) and prevents unit-testing. I wasn't able to return
    /// a lifetimed iterator because it'd move costs into a closure. The performance-gain was about 5% on benchmarks.
    #[must_use]
    #[inline]
    fn get_all_merges<C: Cost + ?Sized>(&self, data: &mut C) -> Vec<Self> {
        debug_assert!(
            self.clusters.is_sorted(),
            "The clusters should always be sorted, to prevent duplicates."
        );

        #[expect(
            clippy::integer_division,
            reason = "At least one of the factors is always even."
        )]
        let mut nodes = Vec::with_capacity(self.clusters.len() * (self.clusters.len() - 1) / 2);
        for i in 0..(self.clusters.len() - 1) {
            // Split off cluster_i.
            let (cluster_i, clusters_minus_i) = {
                let mut clusters_minus_i = self.clusters.clone();
                // This must *not* be a swap_remove, to preserve order.
                let cluster_i = clusters_minus_i.remove(i);
                (cluster_i, clusters_minus_i)
            };
            let cost_minus_i = self.cost - data.cost(cluster_i);
            // Index `i` is gone now, so the lower bound is still `i`
            nodes.extend((i..clusters_minus_i.len()).map(|j| {
                let mut new_clusters = clusters_minus_i.clone();
                // SAFETY:
                // `j` is less than `clusters_minus_i.len()`, and `new_clusters` is a clone of
                // `clusters_minus_i`, so it's a valid index.
                let cluster_j = unsafe { new_clusters.get_unchecked_mut(j) };
                let mut new_cost = cost_minus_i - data.cost(*cluster_j);
                cluster_j.union_with(cluster_i);
                new_cost += data.cost(*cluster_j);

                debug_assert!(new_clusters.len() == self.clusters.len() - 1, "We should have merged two clusters, which should have reduced the number of clusters by exactly one.");
                debug_assert!(new_clusters.is_sorted(), "The clusters should always be sorted, to prevent duplicates.");
                debug_assert!({
                    (0..data.num_points()).all(|point_ix| new_clusters.iter().filter(|cluster| cluster.contains(point_ix)).count()==1)
                },"The clusters should always cover every point exactly once.");
                Self {
                    clusters: new_clusters,
                    cost: new_cost,
                }
            }));
        }
        nodes
    }

    /// Change `self` to be locally optimal, i.e.: For every point, try moving that point
    /// to a different cluster and check if it decreases the cost, repeating this until moving points
    /// no longer decreases the cost.
    fn optimise_locally<C: Cost + ?Sized>(&mut self, data: &mut C) {
        let mut found_improvement = || {
            #[expect(
                clippy::indexing_slicing,
                reason = "These are safe, we just use indices to avoid borrow-issues."
            )]
            for source_cluster_ix in 0..self.clusters.len() {
                let source_cluster = self.clusters[source_cluster_ix];
                for point_ix in source_cluster {
                    let mut updated_source_cluster = source_cluster;
                    updated_source_cluster.remove(point_ix);
                    let source_costdelta =
                        data.cost(updated_source_cluster) - data.cost(source_cluster);

                    for target_cluster_ix in
                        (0..self.clusters.len()).filter(|ix| *ix != source_cluster_ix)
                    {
                        let target_cluster = self.clusters[target_cluster_ix];

                        let mut updated_target_cluster = target_cluster;
                        updated_target_cluster.insert(point_ix);
                        let costdelta = source_costdelta + data.cost(updated_target_cluster)
                            - data.cost(target_cluster);
                        if costdelta < 0.0 {
                            // Keep the clusters in order:
                            if updated_source_cluster.cmp(&updated_target_cluster)
                                == source_cluster_ix.cmp(&target_cluster_ix)
                            {
                                self.clusters[source_cluster_ix] = updated_source_cluster;
                                self.clusters[target_cluster_ix] = updated_target_cluster;
                            } else {
                                self.clusters[source_cluster_ix] = updated_target_cluster;
                                self.clusters[target_cluster_ix] = updated_source_cluster;
                            }

                            self.cost += costdelta;
                            return true;
                        }
                    }
                }
            }
            false
        };

        while found_improvement() {}

        self.clusters.sort();

        debug_assert!(
            {
                (0..data.num_points()).all(|point_ix| {
                    self.clusters
                        .iter()
                        .filter(|cluster| cluster.contains(point_ix))
                        .count()
                        == 1
                })
            },
            "The clusters should always cover every point exactly once."
        );
    }

    /// Create a new node with `num_points` singleton-clusters.
    #[inline]
    fn new_singletons(num_points: usize) -> Self {
        let mut clusters = SmallVec::default();
        for i in 0..num_points {
            clusters.push(Cluster::singleton(i));
        }
        debug_assert!(
            clusters.is_sorted(),
            "The clusters should always be sorted, to prevent duplicates."
        );
        Self {
            clusters,
            cost: 0.0,
        }
    }

    /// Convert to a [`Clustering`].
    #[inline]
    fn into_clustering(self) -> Clustering {
        self.clusters.into_iter().collect()
    }
}

#[derive(Clone, Debug)]
/// A node in the merge-heap. Unlike [`ClusteringNodeMergeMultiple`], this only allows for
/// merging of a cluster with a singleton-cluster.
struct ClusteringNodeMergeSingle {
    /// The nonempty clusters already created. May include singletons.
    /// Will always contain at most `k` clusters.
    ///
    /// TODO: Once [`generic_const_exprs`](https://github.com/rust-lang/rust/issues/76560) is stable,
    /// try using an array with dynamic dispatch instead. Consider hiding this behind a feature-gate
    /// to reduce compile-times.
    clusters: SmallVec<[Cluster; 6]>,
    /// The cost of the clustering represented by [`Self::clusters`]. This should
    /// always be nearly-equal to [`Cost::total_cost`] of [`Self::clusters`].
    cost: f64,
    /// The next point to add. This point, and implicitly every point after it, are
    /// singleton clusters. It's more efficient to track them this way, because
    /// - It means we can store fewer clusters in [`Self::clusters`].
    /// - It ensures we enumerate the clusterings less redundantly
    ///
    /// TODO: Try not storing this, but instead doing best-first-search level-wise?
    /// We could also use the significantly smaller u8 here.
    next_to_add: usize,
}
impl PartialEq for ClusteringNodeMergeSingle {
    fn eq(&self, other: &Self) -> bool {
        self.clusters == other.clusters
    }
}
impl Eq for ClusteringNodeMergeSingle {}
impl Hash for ClusteringNodeMergeSingle {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.clusters.hash(state);
    }
}
impl Ord for ClusteringNodeMergeSingle {
    // Order them by highest cost first
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        other
            .cost
            .total_cmp(&self.cost)
            .then_with(|| self.clusters.cmp(&other.clusters))
    }
}
impl PartialOrd for ClusteringNodeMergeSingle {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl ClusteringNodeMergeSingle {
    /// Returns all possible next nodes from the current node by merging [`Self::next_to_add`] into
    /// existing clusters, and creating a new singleton-cluster for it if [`Self::clusters`] has
    /// fewer than `k` clusters.
    #[inline]
    fn get_next_nodes<'a, C: Cost + ?Sized>(
        &'a self,
        data: &'a mut C,
        k: usize,
    ) -> impl Iterator<Item = Self> + use<'a, C> {
        (0..self.clusters.len())
            .map(|cluster_ix| {
                let mut new_clustering_node = self.clone();
                // SAFETY:
                // `cluster_ix` is less than `self.clusters.len()`, and `new_clustering_node.clusters` is
                // a clone of `self.clusters`, so `cluster_ix` is in bounds.
                let cluster_to_edit =
                    unsafe { new_clustering_node.clusters.get_unchecked_mut(cluster_ix) };
                new_clustering_node.cost -= data.cost(*cluster_to_edit);
                cluster_to_edit.insert(new_clustering_node.next_to_add);
                new_clustering_node.cost += data.cost(*cluster_to_edit);
                new_clustering_node.next_to_add += 1;
                new_clustering_node
            })
            .chain((self.clusters.len() < k).then(|| {
                let mut clustering_node = self.clone();
                clustering_node
                    .clusters
                    .push(Cluster::singleton(clustering_node.next_to_add));
                clustering_node.next_to_add += 1;
                clustering_node
            }))
    }

    /// Create a new empty [`ClusteringNodeMergeSingle`], i.e. implicitly every node is in a singleton-cluster.
    fn empty() -> Self {
        Self {
            clusters: SmallVec::default(),
            cost: 0.0,
            next_to_add: 0,
        }
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
/// A struct for tracking hierarchy-cost in Dijkstra.
///
/// The cost of a hierarchy-level is the cost of its clustering over the cost of the best-possible-clustering on that level.
/// The cost of a hierarchy is the maximum of the costs of all its levels, so addition between two [`MaxRatio`]s
/// is the maximum of the two.
struct MaxRatio(f64);
impl MaxRatio {
    /// Create a new [`MaxRatio`] from a hierarchy cost and an optimal cost.
    #[inline]
    fn new(hierarchy_cost: f64, opt_cost: f64) -> Self {
        debug_assert!(
            hierarchy_cost.is_finite(),
            "hierarchy_cost {hierarchy_cost} should be finite."
        );
        debug_assert!(
            opt_cost.is_finite(),
            "opt_cost {opt_cost} should be finite."
        );
        debug_assert!(
            opt_cost >= 0.0,
            "opt_cost {opt_cost} should be non-negative."
        );
        debug_assert!(
            hierarchy_cost >= 0.0,
            "hierarchy_cost {hierarchy_cost} should be non-negative"
        );
        debug_assert!(
            hierarchy_cost >= opt_cost - 1e-9,
            "hierarchy_cost {hierarchy_cost} should be at least opt_cost {opt_cost}"
        );
        Self(if opt_cost.is_zero() {
            if hierarchy_cost.is_zero() {
                1.0
            } else {
                f64::INFINITY
            }
        } else {
            hierarchy_cost / opt_cost
        })
    }
}
impl Eq for MaxRatio {} // The max-ratios should always be finite.
impl Ord for MaxRatio {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}
impl PartialOrd for MaxRatio {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl ops::Add for MaxRatio {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0.max(rhs.0))
    }
}
impl Zero for MaxRatio {
    fn zero() -> Self {
        Self(1.0)
    }
    #[expect(clippy::float_cmp, reason = "This should be exact.")]
    fn is_zero(&self) -> bool {
        self.0 == 1.0
    }
}

/// A map from clusters to costs, used for memoization.
type Costs = FxHashMap<Cluster, f64>;

/// A trait for cost-functions for a class of clustering-problems.
///
/// TODO: Specify contract for trait-implementors.
pub trait Cost {
    /// Return the total number of points that must be clustered.
    /// This must never exceed [`MAX_POINT_COUNT`].
    fn num_points(&self) -> usize;

    /// Quickly calculate a not-necessarily-optimal clustering, used for speeding up the search
    /// for an optimal clustering by pruning the search-tree earlier.
    ///
    /// For the returned vector, `vec[k]` must be a tuple containing the approximate clustering
    /// for level `k`, along with the [`total_cost`](Cost::total_cost) of that clustering.
    /// Here, `vec[0]` can have an arbitrary score (usually `0.0`) and arbitrary clustering (usually empty).
    #[inline]
    fn approximate_clusterings(&mut self) -> Vec<(f64, Clustering)> {
        // This is similar to `greedy_hierarchy`, but with local search after each merge.
        // TODO: Can we reduce code-duplication here?
        let num_points = self.num_points();

        let mut clustering = ClusteringNodeMergeMultiple::new_singletons(num_points);
        let mut solution: Vec<(f64, Clustering)> =
            vec![(0.0, clustering.clone().into_clustering())];

        while clustering.clusters.len() > 1 {
            let mut best_merge = clustering
                .get_all_merges(self)
                .into_iter()
                .min_by(|a, b| a.cost.total_cmp(&b.cost))
                .expect("There should always be a possible merge");
            best_merge.optimise_locally(self);

            solution.push((best_merge.cost, best_merge.clone().into_clustering()));
            clustering = best_merge;
        }

        solution.push((0.0, Clustering::default()));
        solution.reverse();
        solution
    }

    /// Return the cost of a cluster. This can be memoized via data in `self`.
    ///
    /// The `cluster` will never contain an index higher than `self.num_points()-1` and will
    /// never be empty.
    fn cost(&mut self, cluster: Cluster) -> f64;

    /// Return the total cost of a clustering.
    #[inline]
    fn total_cost(&mut self, clustering: &Clustering) -> f64 {
        clustering.iter().map(|cluster| self.cost(*cluster)).sum()
    }

    /// Return an optimal `k`-clustering for every `0 ≤ k ≤ self.num_points()`.
    ///
    /// For the returned vector, `vec[k]` must be a tuple containing the optimal clustering
    /// for level `k`, along with the [`total_cost`](Cost::total_cost) of that clustering.
    /// Here, `vec[0]` can have an arbitrary score (usually `0.0`) and arbitrary clustering (usually empty).
    #[inline]
    fn optimal_clusterings(&mut self) -> Vec<(f64, Clustering)> {
        let num_points = self.num_points();
        let mut results = Vec::with_capacity(num_points);

        // TODO: Could we instead use some good A* heuristic? Then ordering the points by weight might also
        // be redundant.
        for (k, (approximate_cost, approximate_clustering)) in
            self.approximate_clusterings().into_iter().enumerate()
        {
            results.push((|| {
                debug_assert_eq!(
                    approximate_clustering.len(),
                    k,
                    "The approximate clustering on level {k} should have exactly {k} clusters."
                );
                let mut min_cost = approximate_cost;

                let mut to_see: BinaryHeap<ClusteringNodeMergeSingle> = BinaryHeap::new();
                to_see.push(ClusteringNodeMergeSingle::empty());

                while let Some(clustering_node) = to_see.pop() {
                    if clustering_node.clusters.len() == k
                        && clustering_node.next_to_add == num_points
                    {
                        return (
                            clustering_node.cost,
                            clustering_node.clusters.into_iter().collect(),
                        );
                    }
                    if clustering_node.next_to_add < num_points {
                        for new_clustering_node in clustering_node.get_next_nodes(self, k) {
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
                (approximate_cost, approximate_clustering)
            })());
        }
        results
    }

    /// Return the price-of-hierarchy of the clustering-problem.
    ///
    /// A hierarchical clustering is a set of nested clusterings, one for each possible value of k.
    /// The cost-ratio of level `k` in the hierarchy is its [total cost](`Cost::total_cost`) divided by the cost of
    /// an [optimal `k`-clustering](`Cost::optimal_clusterings`).
    ///
    /// The cost-ratio of the hierarchy is the maximum of the cost-ratios across all its levels.
    /// The price-of-hierarchy is the lowest-possible cost-ratio across all hierarchical clusterings.
    ///
    /// For the returned vector, `vec[k]` is the cluster for level `k`, defaulting to the empty clustering
    /// for `k==0`. Note that the algorithm constructs this hierarchy in reverse, starting with every
    /// point in a singleton-cluster.
    #[must_use]
    #[inline]
    fn price_of_hierarchy(&mut self) -> (f64, Vec<Clustering>) {
        let num_points = self.num_points();
        let opt_for_fixed_k: Vec<f64> = self
            .optimal_clusterings()
            .into_iter()
            .map(|(cost, _)| cost)
            .collect();

        let (price_of_greedy, greedy_hierarchy) = self.price_of_greedy();
        let mut min_hierarchy_price = MaxRatio(price_of_greedy);
        let initial_clustering = ClusteringNodeMergeMultiple::new_singletons(num_points);
        // TODO: If we ever decide to inline dijkstra, we should also have a workhorse-variable for collecting the
        // get_all_merges results, unless inlining dijkstra makes allocations entirely obsolete due to inline
        // iterators.
        // TODO: If we ever decide to inline dijkstra, benchmark running `retain` on all nodes, discarding those
        // whose cost is below the new `min_hierarchy_price`.
        // TODO: Could we instead use some good A* heuristic? Then ordering the points by weight might also
        // be redundant.
        dijkstra(
            &initial_clustering,
            |clustering| {
                let opt_cost =
                    // SAFETY:
                    // We'll never have more than `num_points` clusters, and `opt_for_fixed_k` can index
                    // up to `num_points`.
                    *unsafe { opt_for_fixed_k.get_unchecked(clustering.clusters.len()) };
                clustering
                    .get_all_merges(self)
                    .into_iter()
                    .filter_map(move |new_clustering| {
                        let ratio = MaxRatio::new(new_clustering.cost, opt_cost);
                        (ratio < min_hierarchy_price).then(|| {
                            if new_clustering.clusters.len() == 1 {
                                min_hierarchy_price = ratio;
                            }
                            (new_clustering, ratio)
                        })
                    })
            },
            |clustering| clustering.clusters.len() == 1,
        )
        .map_or_else(
            || (price_of_greedy, greedy_hierarchy),
            |(path, cost)| {
                (
                    cost.0,
                    iter::once(Clustering::default())
                        .chain(
                            path.into_iter()
                                .rev()
                                .map(ClusteringNodeMergeMultiple::into_clustering),
                        )
                        .collect(),
                )
            },
        )
    }

    #[must_use]
    #[inline]
    /// The greedy-hierarchy calculates a hierarchical clustering by starting with
    /// each point in a singleton-cluster, and then repeatedly merging those clusters whose
    /// merging yields the smallest increase in cost.
    ///
    /// For the returned vector, `vec[k]` is a tuple containing the greedy clustering
    /// for level `k`, along with the [`total_cost`](Cost::total_cost) of that clustering.
    ///
    /// Here, `vec[0]` has a score of `0.0` and an empty clustering. Note that the clusterings
    /// are constructed in reverse, as we start with every point in a singleton-cluster.
    fn greedy_hierarchy(&mut self) -> Vec<(f64, Clustering)> {
        let num_points = self.num_points();

        let mut clustering = ClusteringNodeMergeMultiple::new_singletons(num_points);
        let mut solution: Vec<(f64, Clustering)> =
            vec![(0.0, clustering.clone().into_clustering())];

        while clustering.clusters.len() > 1 {
            let best_merge = clustering
                .get_all_merges(self)
                .into_iter()
                .min_by(|a, b| a.cost.total_cmp(&b.cost))
                .expect("There should always be a possible merge");
            solution.push((best_merge.cost, best_merge.clone().into_clustering()));
            clustering = best_merge;
        }

        solution.push((0.0, Clustering::default()));
        solution.reverse();
        solution
    }

    /// Return the cost-ratio and the hierarchy of a greedy hierarchical clustering.
    /// See [`Cost::price_of_hierarchy`] for information about the cost-ratio of a hierarchical clustering,
    /// and the returned hierarchy.
    #[must_use]
    #[inline]
    fn price_of_greedy(&mut self) -> (f64, Vec<Clustering>) {
        let mut max_ratio = MaxRatio::zero();
        let greedy_hierarchy = self.greedy_hierarchy();
        // TODO: Calculation of optimal_clusterings can be sped up by feeding the
        // greedy-hierarchy into it as a starting-point, perhaps? Though currently approximate-clusterings
        // uses a better local-search algorithm.
        let opt_for_fixed_k: Vec<f64> = self
            .optimal_clusterings()
            .into_iter()
            .map(|(cost, _)| cost)
            .collect();

        // Skip the first (empty) level
        for (cost, clustering) in greedy_hierarchy.iter().skip(1) {
            let opt_cost = opt_for_fixed_k
                .get(clustering.len())
                .expect("opt_for_fixed_k should have an entry for this number of clusters.");
            let ratio = MaxRatio::new(*cost, *opt_cost);
            max_ratio = max_ratio + ratio;
        }

        let hierarchy = greedy_hierarchy.into_iter().map(|x| x.1).collect();
        (max_ratio.0, hierarchy)
    }
}

/// A clustering-problem where each center must be one of the points that are to be clustered.
///
/// The cost is supplied using a distance-function between points. The cost of a cluster, given a center,
/// is the sum of the distances between the center and all points in the cluster. The cost of a cluster
/// will always be calculated by choosing the center yielding the smallest cost.
#[derive(Clone, Debug, PartialEq)]
pub struct Discrete {
    /// The distances between the points.
    distances: Distances,
    /// A cache for storing already-calculated costs of clusters.
    costs: Costs,
}
impl Discrete {
    /// Create a discrete `k`-means clustering instance from a given vector of points.
    #[inline]
    pub fn kmeans(points: &[Point]) -> Result<Self, Error> {
        let verified_points = verify_points(points)?;
        Ok(Self {
            distances: distances_from_points_with_element_norm(verified_points, |x| x.powi(2)),
            costs: Costs::default(),
        })
    }

    /// Create a discrete `k`-median clustering instance from a given vector of points.
    #[inline]
    pub fn kmedian(points: &[Point]) -> Result<Self, Error> {
        let verified_points = verify_points(points)?;
        Ok(Self {
            distances: distances_from_points_with_element_norm(verified_points, f64::abs),
            costs: Costs::default(),
        })
    }

    /// Create a discrete weighted `k`-means clustering instance from a given vector of weighted points.
    /// If all your points have the same weight, use [`Discrete::kmeans`] instead.
    ///
    /// The distance between a weighted point `(w, p)` and the center `(v, c)` is the
    /// squared [euclidean-distance](https://en.wikipedia.org/wiki/Euclidean_norm) between `c` and `p`,
    /// multiplied by `w`.
    /// For instance, the center of the cluster {`(1, [0,0])`, `(2, [3,0])`} will be `(2, [3,0])`, because the cost
    /// of choosing that center is `9`, whereas the cost of choosing `(1, [0,0])` as a center is `18`.
    #[inline]
    pub fn weighted_kmeans(weighted_points: &[WeightedPoint]) -> Result<Self, Error> {
        let verified_weighted_points = verify_weighted_points(weighted_points)?;
        Ok(Self {
            distances: distances_from_weighted_points_with_element_norm(
                verified_weighted_points,
                |x| x.powi(2),
            ),
            costs: Costs::default(),
        })
    }

    /// Create a discrete weighted `k`-median clustering instance from a given vector of weighted points.
    /// If all your points have the same weight, use [`Discrete::kmedian`] instead.
    ///
    /// The distance between a weighted point `(w, p)` and the center `(v, c)` is the
    /// [taxicab-distance](https://en.wikipedia.org/wiki/Taxicab_geometry) between `c` and `p`, multiplied by `w`.
    /// For instance, the center of the cluster {`(1, [0,0])`, `(2, [3,0])`} will be `(2, [3,0])`, because the cost
    /// of choosing that center is `3`, whereas the cost of choosing `(1, [0,0])` as a center is `6`.
    #[inline]
    pub fn weighted_kmedian(weighted_points: &[WeightedPoint]) -> Result<Self, Error> {
        let verified_weighted_points = verify_weighted_points(weighted_points)?;
        Ok(Self {
            distances: distances_from_weighted_points_with_element_norm(
                verified_weighted_points,
                f64::abs,
            ),
            costs: Costs::default(),
        })
    }
}
impl Cost for Discrete {
    // TODO: Could we achieve a faster optimal-clusterings-impl for Discrete
    // by not searching for clusterings but for centroids?
    #[inline]
    fn num_points(&self) -> usize {
        self.distances.len()
    }
    #[inline]
    fn cost(&mut self, cluster: Cluster) -> f64 {
        *self.costs.entry(cluster).or_insert_with(|| {
            cluster
                .iter()
                .map(|center_candidate_ix| {
                    let center_candidate_row =
                        // SAFETY:
                        // [`Cost::cost`] promises that `cluster` will never contain an index
                        // higher than `self.num_points()-1`. Because `self.num_points()` is
                        // the length of `self.distances`, this bound is safe.
                        unsafe { self.distances.get_unchecked(center_candidate_ix) };
                    cluster
                        .iter()
                        // SAFETY:
                        // Similar to the above safety-comment, and noting that `center_candidate_row`
                        // has length `self.num_points()`, as well.
                        .map(|ix| *unsafe { center_candidate_row.get_unchecked(ix) })
                        .sum()
                })
                .min_by(f64::total_cmp)
                .unwrap_or(0.0)
        })
    }
}

/// Create [`Distances`] from Points using a distance-function. This function must be non-negative, but need not
/// be symmetric or satisfy the triangle-inequality.
fn distances_from_points_with_distance_function<T>(
    points: &[T],
    distance_function: impl Fn(&T, &T) -> f64,
) -> Distances {
    points
        .iter()
        .map(|p| points.iter().map(|q| distance_function(p, q)).collect())
        .collect()
}

/// Create [`Distances`] from Points using a function that will be applied to each coordinate of the difference
/// between two points, and then summed up. The function must be non-negative, but need not be symmetric or
/// satisfy the triangle-inequality.
fn distances_from_points_with_element_norm(
    points: &[Point],
    elementnorm: impl Fn(f64) -> f64,
) -> Distances {
    distances_from_points_with_distance_function(points, |p, q| {
        (p - q).map(|x| elementnorm(*x)).sum()
    })
}

/// Create [`Distances`] from weighted points using a function that will be applied to each coordinate of the
/// difference between two points, summed up, and then multiplied by the second point's weight.
/// The function must be non-negative, but need not be symmetric or satisfy the triangle-inequality. The weights
/// must be non-negative.
fn distances_from_weighted_points_with_element_norm(
    points: &[WeightedPoint],
    elementnorm: impl Fn(f64) -> f64,
) -> Distances {
    distances_from_points_with_distance_function(points, |p, q| {
        q.0 * (&p.1 - &q.1).map(|x| elementnorm(*x)).sum()
    })
}

/// An error-type for creating clustering-problems.
#[derive(Debug, PartialEq, Eq)]
#[expect(
    clippy::exhaustive_enums,
    reason = "Extending this enum should be a breaking change."
)]
pub enum Error {
    /// No points were supplied.
    EmptyPoints,
    /// The number of points in the problem is too large. It must not exceed [`MAX_POINT_COUNT`].
    TooManyPoints(usize),
    /// Two points (specified by their indices in the points-vec) have different dimensions.
    ShapeMismatch(usize, usize),
    /// A point's (specified by its index in the points-vec) weight is non-finite or non-positive.
    ///
    /// Positive infinity is not allowed to avoid degenerate cases for multiple +∞-points in the same cluster:
    /// If we have `{(+∞, x), (+∞, y)}` with `x!=y`, then we can reasonably set the cost to +∞.
    /// But if `x==y`, should the cost still be +∞, or should it be 0? `+∞ * 0.0` is NaN.
    BadWeight(usize),
}

impl fmt::Display for Error {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let msg = match *self {
            Self::EmptyPoints => "no points supplied".to_owned(),
            Self::TooManyPoints(pointcount) => {
                format!("can cluster at most {MAX_POINT_COUNT} points, but got {pointcount}")
            }
            Self::ShapeMismatch(ix1, ix2) => {
                format!("points {ix1} and {ix2} have different dimensions",)
            }
            Self::BadWeight(ix) => {
                format!("point {ix} doesn't have a finite and positive weight",)
            }
        };
        f.write_str(&msg)
    }
}

#[expect(
    clippy::absolute_paths,
    reason = "Not worth bringing into scope for one use."
)]
impl core::error::Error for Error {}

/// Check whether a set of points is valid for clustering.
fn verify_points(points: &[Point]) -> Result<&[Point], Error> {
    let point_count = points.len();
    if point_count > MAX_POINT_COUNT {
        return Err(Error::TooManyPoints(point_count));
    }

    let first_point = points.first().ok_or(Error::EmptyPoints)?;
    let first_dim = first_point.raw_dim();

    if let Some(ix) = points.iter().position(|p| p.raw_dim() != first_dim) {
        return Err(Error::ShapeMismatch(0, ix));
    }

    Ok(points)
}

/// Check whether a set of weighted points is valid for clustering.
fn verify_weighted_points(weighted_points: &[WeightedPoint]) -> Result<&[WeightedPoint], Error> {
    let point_count = weighted_points.len();
    if point_count > MAX_POINT_COUNT {
        return Err(Error::TooManyPoints(point_count));
    }

    let first_point = weighted_points.first().ok_or(Error::EmptyPoints)?;
    let first_dim = first_point.1.raw_dim();

    if let Some(ix) = weighted_points
        .iter()
        .position(|p| p.1.raw_dim() != first_dim)
    {
        return Err(Error::ShapeMismatch(0, ix));
    }

    if let Some(ix) = weighted_points
        .iter()
        .position(|p| !p.0.is_finite() || p.0 <= 0.0)
    {
        return Err(Error::BadWeight(ix));
    }

    Ok(weighted_points)
}

/// A clustering-problem where each center can be any point in the metric space. The metric
/// space is the same space the points live in.
///
/// The cost of a cluster, given a center, is the sum of the squared Euclidean distances between
/// the center and all points in the cluster.
/// The center is automatically calculated to minimise the cost, which turns out to simply be the
/// average of all point-positions in the cluster.
#[derive(Clone, Debug, PartialEq)]
pub struct KMeans {
    /// The points to be clustered.
    points: Vec<Point>,
    /// A cache for storing already-calculated costs of clusters.
    costs: Costs,
}
impl Cost for KMeans {
    #[inline]
    fn num_points(&self) -> usize {
        self.points.len()
    }
    #[inline]
    fn cost(&mut self, cluster: Cluster) -> f64 {
        *self.costs.entry(cluster).or_insert_with(|| {
            let first_point_dimensions =
                // SAFETY:
                // [`verify_points`] ensures that we always have at least one point.
                unsafe { self.points.first().unwrap_unchecked() }.raw_dim();
            let mut center = Array1::zeros(first_point_dimensions);

            // For some reason, this is 30% faster than a for-loop.
            cluster
                .iter()
                // SAFETY:
                // [`Cost::cost`] promises us that this index is in-bounds.
                .for_each(|i| center += unsafe { self.points.get_unchecked(i) });

            // We never divide by 0 here.
            center /= f64::from(cluster.len());
            cluster
                .iter()
                .map(|i| {
                    // SAFETY:
                    // [`Cost::cost`] promises us that this index is in-bounds.
                    let p = unsafe { self.points.get_unchecked(i) };
                    (p - &center).map(|x| x.powi(2)).sum()
                })
                .sum()
        })
    }
    #[inline]
    fn approximate_clusterings(&mut self) -> Vec<(f64, Clustering)> {
        use clustering::kmeans;
        let mut results = Vec::with_capacity(self.num_points() + 1);
        results.push((0.0, Clustering::default()));
        let max_iter = 1000;
        let samples: Vec<Vec<f64>> = self
            .points
            .iter()
            .map(|x| x.into_iter().copied().collect())
            .collect();
        results.extend((1..=self.num_points()).map(|k| {
            let kmeans_clustering = kmeans(k, &samples, max_iter);
            let mut clusters = vec![Cluster::empty(); k];
            for (point_ix, cluster_ix) in kmeans_clustering.membership.iter().enumerate() {
                clusters
                    .get_mut(*cluster_ix)
                    .expect("Cluster index out of range")
                    .insert(point_ix);
            }
            let clustering: Clustering = clusters.into_iter().collect();
            (self.total_cost(&clustering), clustering)
        }));
        results
    }
}
impl KMeans {
    /// Return a new `k`-means clustering instance from a given vector of points.
    #[inline]
    pub fn new(points: &[Point]) -> Result<Self, Error> {
        let verified_points = verify_points(points)?;
        Ok(Self {
            points: verified_points.to_vec(),
            costs: Costs::default(),
        })
    }
}

/// A weighted clustering-problem where each center can be any point in the metric space. The metric
/// space is the same space the weighted points live in.
///
/// The cost of a cluster, given a center, is the sum of the squared Euclidean distances between
/// the center and each point in the cluster, multiplied by the point's weight.
/// The center is automatically calculated to minimise the cost, which turns out to simply be the
/// weighted average of all point-positions in the cluster.
#[derive(Clone, Debug, PartialEq)]
pub struct WeightedKMeans {
    /// The points to be clustered.
    weighted_points: Vec<WeightedPoint>,
    /// A cache for storing already-calculated costs of clusters.
    costs: Costs,
}
impl Cost for WeightedKMeans {
    #[inline]
    fn num_points(&self) -> usize {
        self.weighted_points.len()
    }
    #[inline]
    fn cost(&mut self, cluster: Cluster) -> f64 {
        *self.costs.entry(cluster).or_insert_with(|| {
            let mut total_weight = 0.0;
            let first_point_dimensions =
                // SAFETY:
                // [`verify_points`] ensures that we always have at least one point.
                unsafe { self.weighted_points.first().unwrap_unchecked() }.1.raw_dim();
            let mut center: Array1<f64> = Array1::zeros(first_point_dimensions);

            // For some reason, this is 30% faster than a for-loop.
            // TODO: If this is hot, benchmark changes in assignments (e.g. assign let weight = weighted_point.0 first)
            cluster.iter().for_each(|i| {
                // SAFETY:
                // [`Cost::cost`] promises us that this index is in-bounds.
                let weighted_point = unsafe { self.weighted_points.get_unchecked(i) };
                total_weight += weighted_point.0;
                center += &(&weighted_point.1 * weighted_point.0);
            });

            // Because `cluster` is never empty, and weights are always positive,
            // we never divide by 0 here.
            center /= total_weight;

            cluster
                .iter()
                .map(|i| {
                    // SAFETY:
                    // [`Cost::cost`] promises us that this index is in-bounds.
                    let weighted_point = unsafe { self.weighted_points.get_unchecked(i) };
                    weighted_point.0 * (&weighted_point.1 - &center).map(|x| x.powi(2)).sum()
                })
                .sum()
        })
    }
}
impl WeightedKMeans {
    /// Return a new `k`-means clustering instance from a given vector of points.
    ///
    /// The algorithm runs significantly faster if you sort the points by weight first.
    ///
    /// TODO: Do so internally instead of at callsite. This probably requires better return-values
    /// in the API. Afterwards, rework the `get_high_kmeans_price_of_greedy_instance` function in
    /// integration-tests to not sort the points on its own.
    #[inline]
    pub fn new(weighted_points: &[WeightedPoint]) -> Result<Self, Error> {
        let verified_weighted_points = verify_weighted_points(weighted_points)?;
        Ok(Self {
            weighted_points: verified_weighted_points.to_vec(),
            costs: Costs::default(),
        })
    }
}

/// Return a cluster from an iterator of point-indices.
///
/// TODO: This method only exists due to a malnourished API. An API-improvement should make
/// it obsolete.
#[inline]
pub fn cluster_from_iterator<I: IntoIterator<Item = usize>>(it: I) -> Cluster {
    let mut cluster = Cluster::empty();
    for i in it {
        cluster.insert(i);
    }
    cluster
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::SQRT_2;
    use itertools::Itertools as _;
    use ndarray::array;
    use smallvec::smallvec;
    use std::panic::catch_unwind;

    #[test]
    #[should_panic(
        expected = "Throughout the entire implementation, we should never to add the same point twice."
    )]
    fn cluster_double_insert() {
        let mut cluster = Cluster::singleton(7);
        cluster.insert(7);
    }

    #[test]
    #[should_panic(
        expected = "Troughout the entire implementation, we should never be merging intersecting clusters."
    )]
    fn cluster_intersecting_merge() {
        let mut cluster7 = Cluster::singleton(7);
        let mut cluster9 = Cluster::singleton(7);
        cluster7.insert(8);
        cluster9.insert(8);
        cluster7.union_with(cluster9);
    }

    #[test]
    fn cluster() {
        for i in 0..8 {
            let cluster = Cluster::singleton(i);
            assert!(!cluster.is_empty());
            assert_eq!(cluster.len(), 1);
            assert_eq!(cluster.iter().collect_vec(), vec![i]);
            for j in 0..8 {
                assert_eq!(cluster.contains(j), j == i);
                let cluster2 = {
                    let mut cluster2 = cluster;
                    if i != j {
                        cluster2.insert(j);
                    }
                    assert!(!cluster2.is_empty());
                    cluster2
                };
                assert!(!cluster2.is_empty());
                assert_eq!(cluster2.len(), if i == j { 1 } else { 2 });
                assert_eq!(
                    cluster2.iter().collect_vec(),
                    match i.cmp(&j) {
                        cmp::Ordering::Less => vec![i, j],
                        cmp::Ordering::Equal => vec![i],
                        cmp::Ordering::Greater => vec![j, i],
                    }
                );
            }
        }
        let mut cluster_div_3 = Cluster::empty();
        let mut cluster_div_5 = Cluster::empty();
        assert!(cluster_div_3.is_empty());
        assert!(cluster_div_5.is_empty());
        // Only go up to 14, we don't want any intersections between the two.
        for i in 1..=14 {
            if i % 3 == 0 {
                cluster_div_3.insert(i);
                assert!(!cluster_div_3.is_empty());
            }
            if i % 5 == 0 {
                cluster_div_5.insert(i);
                assert!(!cluster_div_5.is_empty());
            }
        }
        assert_eq!(cluster_div_3.iter().collect_vec(), vec![3, 6, 9, 12]);
        assert_eq!(cluster_div_5.iter().collect_vec(), vec![5, 10]);
        let merged = {
            let mut merged = cluster_div_3;
            merged.union_with(cluster_div_5);
            merged
        };
        assert_eq!(merged.iter().collect_vec(), vec![3, 5, 6, 9, 10, 12]);

        assert_eq!(merged.to_string(), "...#.##..##.#...................");
    }

    #[expect(clippy::float_cmp, reason = "This should be exact.")]
    #[expect(
        clippy::assertions_on_result_states,
        reason = "We'd like to catch the errors."
    )]
    #[test]
    fn max_ratio() {
        assert_eq!(MaxRatio::new(3.0, 1.5).0, 2.0);
        assert_eq!(MaxRatio::new(SQRT_2, SQRT_2).0, 1.0);
        assert_eq!(MaxRatio::new(SQRT_2, 0.0).0, f64::INFINITY);
        assert_eq!(MaxRatio::new(SQRT_2, -0.0).0, f64::INFINITY);
        assert_eq!(MaxRatio::new(0.0, 0.0).0, 1.0);
        assert_eq!(MaxRatio::new(-0.0, 0.0).0, 1.0);
        assert_eq!(MaxRatio::new(0.0, -0.0).0, 1.0);
        assert_eq!(MaxRatio::new(-0.0, -0.0).0, 1.0);
        assert!(catch_unwind(|| MaxRatio::new(1.0 - 1e-3, 1.0)).is_err());
        assert!(catch_unwind(|| MaxRatio::new(1.0 - 1e-12, 1.0)).is_ok());
        assert!(catch_unwind(|| MaxRatio::new(0.0 - 1e-12, 0.0)).is_err());
        assert!(catch_unwind(|| MaxRatio::new(f64::INFINITY, 1.0)).is_err());
        assert!(catch_unwind(|| MaxRatio::new(f64::NAN, 1.0)).is_err());
        assert!(catch_unwind(|| MaxRatio::new(f64::NEG_INFINITY, 1.0)).is_err());
        assert!(catch_unwind(|| MaxRatio::new(1.0, f64::INFINITY)).is_err());
        assert!(catch_unwind(|| MaxRatio::new(1.0, f64::NAN)).is_err());
        assert!(catch_unwind(|| MaxRatio::new(1.0, f64::NEG_INFINITY)).is_err());
        assert!(catch_unwind(|| MaxRatio::new(1.0, 0.0)).is_ok());
        assert!(catch_unwind(|| MaxRatio::new(1.0, -1e-12)).is_err());
    }

    macro_rules! clusterings {
        ( $( [ $( [ $( $num:expr ),* ] ),* ] ),* $(,)? ) => {
            [
                $(
                    vec![
                        $(
                            cluster_from_iterator([$( $num ),*]),
                        )*
                    ],
                )*
            ]
        }
    }

    #[test]
    fn node_merge_multiple() {
        fn clusters_are_correct(
            expected_clusterings: &[Vec<Cluster>],
            nodes: &[ClusteringNodeMergeMultiple],
        ) {
            let actual = nodes.iter().map(|x| x.clusters.to_vec()).collect_vec();
            assert_eq!(
                expected_clusterings, actual,
                "Clustering should match expected clustering. Maybe the order of returned Clusters has changed?"
            );
        }
        let mut discrete = Discrete::kmeans(&[array![0.0], array![1.0], array![2.0], array![3.0]])
            .expect("Creating discrete should not fail.");
        let mut update_nodes = |nodes: &mut Vec<ClusteringNodeMergeMultiple>| {
            *nodes = nodes
                .iter()
                .flat_map(|n| n.get_all_merges(&mut discrete))
                .collect();
        };
        let mut nodes = vec![ClusteringNodeMergeMultiple::new_singletons(4)];
        let expected_init_clusters = smallvec![
            Cluster::singleton(0),
            Cluster::singleton(1),
            Cluster::singleton(2),
            Cluster::singleton(3)
        ];
        assert_eq!(
            nodes,
            vec![ClusteringNodeMergeMultiple {
                clusters: expected_init_clusters,
                cost: f64::NAN,
            }],
            "Testing nodes for equality should only depend on clusters, not on their cost."
        );
        clusters_are_correct(&clusterings![[[0], [1], [2], [3]]], &nodes);

        update_nodes(&mut nodes);
        clusters_are_correct(
            &clusterings![
                [[0, 1], [2], [3]],
                [[1], [0, 2], [3]],
                [[1], [2], [0, 3]],
                [[0], [1, 2], [3]],
                [[0], [2], [1, 3]],
                [[0], [1], [2, 3]],
            ],
            &nodes,
        );

        update_nodes(&mut nodes);
        clusters_are_correct(
            &clusterings![
                [[0, 1, 2], [3]],
                [[2], [0, 1, 3]],
                [[0, 1], [2, 3]],
                [[1, 0, 2], [3]],
                [[0, 2], [1, 3]],
                [[1], [0, 2, 3]],
                [[1, 2], [0, 3]],
                [[2], [1, 0, 3]],
                [[1], [2, 0, 3]],
                [[0, 1, 2], [3]],
                [[1, 2], [0, 3]],
                [[0], [1, 2, 3]],
                [[0, 2], [1, 3]],
                [[2], [0, 1, 3]],
                [[0], [2, 1, 3]],
                [[0, 1], [2, 3]],
                [[1], [0, 2, 3]],
                [[0], [1, 2, 3]],
            ],
            &nodes,
        );

        update_nodes(&mut nodes);
        clusters_are_correct(&vec![vec![Cluster(15)]; 18], &nodes);
    }

    #[test]
    #[should_panic(expected = "The clusters should always be sorted, to prevent duplicates.")]
    fn unsorted_node_merge_multiple() {
        let unsorted = ClusteringNodeMergeMultiple {
            clusters: smallvec![Cluster(1), Cluster(0)],
            cost: 0.0,
        };
        let mut small_discrete = Discrete::kmedian(&[array![0.0], array![1.0]])
            .expect("Creating discrete should not fail.");
        let _: Vec<_> = unsorted
            .get_all_merges(&mut small_discrete) // This should fail.
            .into_iter()
            .collect_vec();
    }

    #[test]
    fn node_merge_single() {
        fn clusters_are_correct(
            expected_clusterings: &[Vec<Cluster>],
            nodes: &[ClusteringNodeMergeSingle],
        ) {
            let actual = nodes.iter().map(|x| x.clusters.to_vec()).collect_vec();
            assert_eq!(
                expected_clusterings, actual,
                "Clustering should match expected clustering. Maybe the order of returned Clusters has changed?"
            );
        }
        let mut discrete = Discrete::kmeans(&[array![0.0], array![1.0], array![2.0], array![3.0]])
            .expect("Creating discrete should not fail.");
        let mut update_nodes = |nodes: &mut Vec<ClusteringNodeMergeSingle>| {
            *nodes = nodes
                .iter()
                .flat_map(|n| n.get_next_nodes(&mut discrete, 3).collect_vec())
                .collect();
        };
        let mut nodes = vec![ClusteringNodeMergeSingle::empty()];
        clusters_are_correct(&clusterings![[]], &nodes);

        update_nodes(&mut nodes);
        clusters_are_correct(&clusterings![[[0]]], &nodes);

        update_nodes(&mut nodes);
        clusters_are_correct(&clusterings![[[0, 1]], [[0], [1]]], &nodes);

        update_nodes(&mut nodes);
        clusters_are_correct(
            &clusterings![
                [[0, 1, 2]],
                [[0, 1], [2]],
                [[0, 2], [1]],
                [[0], [1, 2]],
                [[0], [1], [2]],
            ],
            &nodes,
        );

        update_nodes(&mut nodes);
        clusters_are_correct(
            &clusterings![
                [[0, 1, 2, 3]],
                [[0, 1, 2], [3]],
                [[0, 1, 3], [2]],
                [[0, 1], [2, 3]],
                [[0, 1], [2], [3]],
                [[0, 2, 3], [1]],
                [[0, 2], [1, 3]],
                [[0, 2], [1], [3]],
                [[0, 3], [1, 2]],
                [[0], [1, 2, 3]],
                [[0], [1, 2], [3]],
                [[0, 3], [1], [2]],
                [[0], [1, 3], [2]],
                [[0], [1], [2, 3]],
                // Notice that [[0],[1],[2],[3]] is not in this list.
            ],
            &nodes,
        );
    }
}
