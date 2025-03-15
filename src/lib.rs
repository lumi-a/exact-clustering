//! Finding optimal clusterings and hierarchical clusterings.
//! TODO: Better crate-doc.

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
#[cfg(not(target_pointer_width = "32"))]
pub type Storage = u32;

/// A compact representation of a cluster of points, using a bitset.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
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
#[derive(Clone, Debug)]
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

/// A rectangular matrix whose column-vectors are the coordinate-vectors of the points.
pub type Points = Vec<Array1<f64>>;

#[derive(Clone, Debug)]
/// A helper-struct for efficiently merging clusters, used in finding optimal hierarchies.
struct ClusteringNodeMergeMultiple {
    /// The clusters. We use a [`SmallVec`] because this will allocate frequently. The smallvec
    /// must remain sorted so that two Nodes with the same clusters are recognised as
    /// equal.
    ///
    /// We could also use some set (`OrderSet`, `BTreeSet`) datastructure,
    /// as long as it implements `Hash`.
    ///
    /// TODO: Benchmark those three datastructures, and also benchmark the smallvec-size.
    clusters: SmallVec<[Cluster; 6]>,
    /// The total cost of the clustering. We keep track of this to efficiently recalculate
    /// costs after merging.
    ///
    /// TODO: Try not keeping track of it, instead having [`ClusteringNodeMergeMultiple::get_all_merges`] return a delta,
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
        Self(f64::NEG_INFINITY)
    }
    fn is_zero(&self) -> bool {
        self.0 == f64::NEG_INFINITY
    }
}

/// A map from clusters to costs, used for memoization.
type Costs = FxHashMap<Cluster, f64>;

/// A trait for cost-functions for a class of clustering-problems.
pub trait Cost {
    /// Return the total number of points that must be clustered.
    fn num_points(&self) -> usize;

    /// Quickly calculate a not-necessarily-optimal clustering, used for speeding up the search
    /// for an optimal clustering by pruning the search-tree earlier.
    #[inline]
    fn approximate_clustering(&mut self, k: usize) -> (f64, Clustering) {
        // TODO: Use greedy-hierarchy instead. Needs new api.
        let mut clusters = vec![Cluster::empty(); k];
        for i in 0..self.num_points() {
            clusters[i % k].insert(i);
        }
        let clustering: Clustering = clusters.into_iter().collect();
        let cost = self.total_cost(&clustering);
        (cost, clustering)
    }

    /// Return the cost of a cluster. This should be memoized via data in `self`.
    fn cost(&mut self, cluster: Cluster) -> f64;

    /// Return the total cost of a clustering.
    #[inline]
    fn total_cost(&mut self, clustering: &Clustering) -> f64 {
        clustering.iter().map(|cluster| self.cost(*cluster)).sum()
    }

    /// Return an optimal `k`-clustering.
    ///
    /// TODO: Information about return-types.
    #[inline]
    fn optimal_clustering(&mut self, k: usize) -> (f64, Clustering) {
        let num_points = self.num_points();
        let (approximate_clustering_cost, approximate_clustering) = self.approximate_clustering(k);
        debug_assert_eq!(
            approximate_clustering.len(),
            k,
            "The approximate clustering on level {k} should have exactly {k} clusters."
        );
        let mut min_cost = approximate_clustering_cost;

        let mut to_see: BinaryHeap<ClusteringNodeMergeSingle> = BinaryHeap::new();
        to_see.push(ClusteringNodeMergeSingle::empty());

        while let Some(clustering_node) = to_see.pop() {
            if clustering_node.clusters.len() == k && clustering_node.next_to_add == num_points {
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
        (approximate_clustering_cost, approximate_clustering)
    }

    /// Return the price-of-hierarchy of the clustering-problem.
    ///
    /// A hierarchical clustering is a set of nested clusterings, one for each possible value of k.
    /// The cost-ratio of level `k` in the hierarchy is its [total cost](`Cost::total_cost`) divided by the cost of
    /// an [optimal `k`-clustering](`Cost::optimal_clustering`).
    ///
    /// The cost-ratio of the hierarchy is the maximum of the cost-ratios across all its levels.
    ///
    /// The price-of-hierarchy is the lowest-possible cost-ratio across all hierarchical clusterings.
    ///
    /// TODO: Information about return-types.
    #[must_use]
    #[inline]
    fn price_of_hierarchy(&mut self) -> (f64, Vec<Clustering>) {
        let num_points = self.num_points();
        let opt_for_fixed_k: Vec<f64> = iter::once(0.0)
            .chain((1..=num_points).map(|k| self.optimal_clustering(k).0))
            .collect();

        let (price_of_greedy, greedy_hierarchy) = self.price_of_greedy();
        let mut min_hierarchy_price = MaxRatio(price_of_greedy);
        let initial_clustering = ClusteringNodeMergeMultiple::new_singletons(num_points);
        dijkstra(
            &initial_clustering,
            |clustering| {
                let opt_cost =
                    *unsafe { opt_for_fixed_k.get_unchecked(clustering.clusters.len() - 1) };
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
                    path.into_iter()
                        .map(ClusteringNodeMergeMultiple::into_clustering)
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
    /// TODO: Information about return-types.
    fn greedy_hierarchy(&mut self) -> Vec<(f64, Clustering)> {
        let num_points = self.num_points();

        let mut clustering = ClusteringNodeMergeMultiple::new_singletons(num_points);
        let mut solution: Vec<(f64, Clustering)> =
            vec![(0.0, clustering.clone().into_clustering())];
        // TODO: Add a method "approximate clusters for all k" or something, which'd be useful for
        // k-median but not k-means.
        while clustering.clusters.len() > 1 {
            let best_merge = clustering
                .get_all_merges(self)
                .into_iter()
                .min_by(|a, b| a.cost.total_cmp(&b.cost))
                .expect("There should always be a possible merge");
            solution.push((best_merge.cost, best_merge.clone().into_clustering()));
            clustering = best_merge;
        }

        solution
    }

    /// Return the cost-ratio of a greedy hierarchical clustering. See [`Cost::price_of_hierarchy`] for
    /// information about the cost-ratio of a hierarchical clustering.
    ///
    /// TODO: Information about return-types.
    #[must_use]
    #[inline]
    fn price_of_greedy(&mut self) -> (f64, Vec<Clustering>) {
        let mut max_ratio = MaxRatio::zero();
        let greedy_hierarchy = self.greedy_hierarchy();
        for (cost, clustering) in &greedy_hierarchy {
            let opt_cost = self.optimal_clustering(clustering.len()).0;
            let ratio = MaxRatio::new(*cost, opt_cost);
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
    distances: Distances,
    costs: Costs,
}
impl Cost for Discrete {
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
                        unsafe { self.distances.get_unchecked(center_candidate_ix) };
                    cluster
                        .iter()
                        .map(|ix| *unsafe { center_candidate_row.get_unchecked(ix) })
                        .sum()
                })
                .min_by(f64::total_cmp)
                .unwrap_or(0.0)
        })
    }
}

/// Create [`Distances`] from [`Points`] using a a distance-function [`Points`] × [`Points`] -> [`f64`].
///
/// This will usually be a metric, but the only part of the code that assumes non-negativity is [`MaxRatio::new`],
/// symmetry and triangle-inequality are not assumed anywhere, I believe.
fn distances_from_points_with_distfun(
    points: &Points,
    metric: impl Fn(&Array1<f64>, &Array1<f64>) -> f64,
) -> Distances {
    points
        .iter()
        .map(|p| points.iter().map(|q| metric(p, q)).collect())
        .collect()
}
/// Create [`Distances`] from [`Points`] using a norm [`Points`] -> [`f64`].
///
/// [`MaxRatio::new`] assumes non-negativity. I believe all other norm-properties may be violated without error.
fn distances_from_points_with_norm(
    points: &Points,
    norm: impl Fn(Array1<f64>) -> f64,
) -> Distances {
    distances_from_points_with_distfun(points, |p, q| norm(p - q))
}
/// Create [`Distances`] from [`Points`] using a function that will be applied to each coordinate of the difference
/// between two points, and then summed up.
///
/// Only [`MaxRatio::new`] assumes non-negativity, I believe.
fn distances_from_points_with_element_norm(
    points: &Points,
    elementnorm: impl Fn(f64) -> f64,
) -> Distances {
    distances_from_points_with_norm(points, |p| p.map(|x| elementnorm(*x)).sum())
}
/// Create [`Distances`] from [`Points`] using the squared [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_norm).
fn squared_euclidean_distances_from_points(points: &Points) -> Distances {
    distances_from_points_with_element_norm(points, |x| x.powi(2))
}
/// Create [`Distances`] from [`Points`] using the [taxicab distance](https://en.wikipedia.org/wiki/Taxicab_geometry).
fn taxicab_distances_from_points(points: &Points) -> Distances {
    distances_from_points_with_element_norm(points, f64::abs)
}

/// An error-type for creating clustering-problems.
#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    /// The number of points in the problem is too large. It must not exceed [`Storage::BITS`].
    TooManyPoints(usize),
    /// Two points (specified by their indices in the points-vec) have different dimensions.
    ShapeMismatch(usize, usize),
}

impl fmt::Display for Error {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let msg = &match self {
            Self::TooManyPoints(pointcount) => format!(
                "Can cluster at most {} points, but {pointcount} were supplied.",
                Storage::BITS
            ),
            Self::ShapeMismatch(ix1, ix2) => {
                format!("Points {ix1} and {ix2} have different dimensions.",)
            }
        };
        f.write_str(msg)
    }
}

#[expect(
    clippy::absolute_paths,
    reason = "Not worth bringing into scope for one use."
)]
impl core::error::Error for Error {}

/// Check whether a set of points is valid for clustering.
fn verify_points(points: &Points) -> Result<&Points, Error> {
    points.first().map_or(Ok(points), |first_point| {
        let first_dim = first_point.raw_dim();
        points
            .iter()
            .position(|p| p.raw_dim() != first_dim)
            .map_or(Ok(points), |ix| Err(Error::ShapeMismatch(0, ix)))
    })
}

impl Discrete {
    /// Create a discrete clustering-instance given the distances between points.
    #[must_use]
    fn new_with_distances(distances: Distances) -> Self {
        Self {
            distances,
            costs: Costs::default(),
        }
    }
    /// Create a discrete `k`-means clustering instance from a given vector of points.
    ///
    /// # Errors
    /// Returns an [`Error`] if too many points are supplied or if the dimensions of the points don't match.
    #[inline]
    pub fn squared_euclidean_from_points(points: &Points) -> Result<Self, Error> {
        let verified_points = verify_points(points)?;
        Ok(Self::new_with_distances(
            squared_euclidean_distances_from_points(verified_points),
        ))
    }
    /// Create a discrete `k`-median clustering instance from a given vector of points.
    ///
    /// # Errors
    /// Returns an [`Error`] if too many points are supplied or if the dimensions of the points don't match.
    #[inline]
    pub fn median_from_points(points: &Points) -> Result<Self, Error> {
        let verified_points = verify_points(points)?;
        Ok(Self::new_with_distances(taxicab_distances_from_points(
            verified_points,
        )))
    }
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
    points: Points,
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
            let mut center = Array1::zeros(self.points[0].raw_dim());
            for i in cluster.iter() {
                center += unsafe { self.points.get_unchecked(i) };
            }
            // TODO: Check we don't divide by 0
            center /= f64::from(cluster.len());
            cluster
                .iter()
                .map(|i| {
                    let p = unsafe { self.points.get_unchecked(i) };
                    (p - &center).map(|x| x.powi(2)).sum()
                })
                .sum()
        })
    }
    #[inline]
    fn approximate_clustering(&mut self, k: usize) -> (f64, Clustering) {
        use clustering::kmeans;

        let max_iter = 1000; // TODO: Benchmark this? Maybe there's some way to print the number of iterations
        let samples: Vec<Vec<f64>> = self
            .points
            .iter()
            .map(|x| x.into_iter().copied().collect())
            .collect();
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
    }
}
impl KMeans {
    /// Return a new `k`-means clustering instance from a given vector of points.
    ///
    /// # Errors
    /// Returns an [`Error`] if too many points are supplied or if the dimensions of the points don't match.
    #[inline]
    pub fn new(points: &Points) -> Result<Self, Error> {
        let verified_points = verify_points(points)?;
        Ok(Self {
            points: verified_points.clone(),
            costs: Costs::default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::SQRT_2;
    use ndarray::array;
    use std::f64;

    fn square_grid() -> Points {
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

    fn triangle_grid() -> Points {
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

    fn cluster_from_iterator(it: impl IntoIterator<Item = usize>) -> Cluster {
        let mut cluster = Cluster::empty();
        for i in it {
            cluster.insert(i);
        }
        cluster
    }

    fn clustering_from_iterators(
        it: impl IntoIterator<Item = impl IntoIterator<Item = usize>>,
    ) -> Clustering {
        it.into_iter().map(cluster_from_iterator).collect()
    }

    #[test]
    #[should_panic(
        expected = "Throughout the entire implementation, we should never to add the same point twice."
    )]
    fn test_cluster_double_insert() {
        let mut cluster = Cluster::singleton(7);
        cluster.insert(7);
    }

    #[test]
    #[should_panic(
        expected = "Troughout the entire implementation, we should never be merging intersecting clusters."
    )]
    fn test_cluster_intersecting_merge() {
        let mut cluster7 = Cluster::singleton(7);
        let mut cluster9 = Cluster::singleton(7);
        cluster7.insert(8);
        cluster9.insert(8);
        cluster7.union_with(cluster9);
    }

    #[test]
    fn test_cluster() {
        for i in 0..8 {
            let cluster = Cluster::singleton(i);
            assert!(!cluster.is_empty());
            assert_eq!(cluster.len(), 1);
            assert_eq!(cluster.iter().collect::<Vec<_>>(), vec![i]);
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
                    cluster2.iter().collect::<Vec<_>>(),
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
        assert_eq!(cluster_div_3.iter().collect::<Vec<_>>(), vec![3, 6, 9, 12]);
        assert_eq!(cluster_div_5.iter().collect::<Vec<_>>(), vec![5, 10]);
        let merged = {
            let mut merged = cluster_div_3;
            merged.union_with(cluster_div_5);
            merged
        };
        assert_eq!(merged.iter().collect::<Vec<_>>(), vec![3, 5, 6, 9, 10, 12]);

        assert_eq!(merged.to_string(), "...#.##..##.#...................");
    }

    #[expect(clippy::float_cmp, reason = "This should be exact.")]
    #[expect(
        clippy::assertions_on_result_states,
        reason = "We'd like to catch the errors."
    )]
    #[test]
    fn test_max_ratio() {
        use std::panic::catch_unwind;

        assert_eq!(MaxRatio::new(3.0, 1.5).0, 2.0);
        assert_eq!(MaxRatio::new(SQRT_2, SQRT_2).0, 1.0);
        assert_eq!(MaxRatio::new(SQRT_2, 0.0).0, f64::INFINITY);
        assert_eq!(MaxRatio::new(SQRT_2, -0.0).0, f64::INFINITY);
        assert_eq!(MaxRatio::new(0.0, 0.0).0, 1.0);
        assert_eq!(MaxRatio::new(-0.0, 0.0).0, 1.0);
        assert_eq!(MaxRatio::new(0.0, -0.0).0, 1.0);
        assert_eq!(MaxRatio::new(-0.0, -0.0).0, 1.0);
        assert!(catch_unwind(|| MaxRatio::new(1.0 - 1e-12, 1.0)).is_ok());
        assert!(catch_unwind(|| MaxRatio::new(f64::INFINITY, 1.0)).is_err());
        assert!(catch_unwind(|| MaxRatio::new(f64::NAN, 1.0)).is_err());
        assert!(catch_unwind(|| MaxRatio::new(f64::NEG_INFINITY, 1.0)).is_err());
        assert!(catch_unwind(|| MaxRatio::new(1.0, f64::INFINITY)).is_err());
        assert!(catch_unwind(|| MaxRatio::new(1.0, f64::NAN)).is_err());
        assert!(catch_unwind(|| MaxRatio::new(1.0, f64::NEG_INFINITY)).is_err());
        assert!(catch_unwind(|| MaxRatio::new(1.0, 0.0)).is_ok());
        assert!(catch_unwind(|| MaxRatio::new(1.0, -1e-12)).is_err());
    }

    #[test]
    fn test_node_merge_multiple() {
        use smallvec::smallvec;

        fn clusters_are_correct(
            expected_clusterings: &[Vec<Cluster>],
            nodes: &[ClusteringNodeMergeMultiple],
        ) {
            let actual = nodes
                .iter()
                .map(|x| x.clusters.to_vec())
                .collect::<Vec<_>>();
            assert_eq!(
                expected_clusterings, actual,
                "Clustering should match expected clustering. Maybe the order of returned Clusters has changed?"
            );
        }
        let mut discrete = Discrete::squared_euclidean_from_points(&vec![
            array![0.0],
            array![1.0],
            array![2.0],
            array![3.0],
        ])
        .expect("Creating discrete should not fail.");
        let mut update_nodes = |nodes: &mut Vec<ClusteringNodeMergeMultiple>| {
            *nodes = nodes
                .iter()
                .flat_map(|n| n.get_all_merges(&mut discrete))
                .collect();
        };
        let mut nodes = vec![ClusteringNodeMergeMultiple::new_singletons(4)];
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
        clusters_are_correct(
            &vec![vec![cluster_from_iterator(vec![0, 1, 2, 3])]; 18],
            &nodes,
        );
    }

    #[test]
    fn test_node_merge_single() {
        fn clusters_are_correct(
            expected_clusterings: &[Vec<Cluster>],
            nodes: &[ClusteringNodeMergeSingle],
        ) {
            let actual = nodes
                .iter()
                .map(|x| x.clusters.to_vec())
                .collect::<Vec<_>>();
            assert_eq!(
                expected_clusterings, actual,
                "Clustering should match expected clustering. Maybe the order of returned Clusters has changed?"
            );
        }
        let mut discrete = Discrete::squared_euclidean_from_points(&vec![
            array![0.0],
            array![1.0],
            array![2.0],
            array![3.0],
        ])
        .expect("Creating discrete should not fail.");
        let mut update_nodes = |nodes: &mut Vec<ClusteringNodeMergeSingle>| {
            *nodes = nodes
                .iter()
                .flat_map(|n| n.get_next_nodes(&mut discrete, 3).collect::<Vec<_>>())
                .collect();
        };
        let mut nodes = vec![ClusteringNodeMergeSingle::empty()];
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

    #[test]
    #[expect(clippy::float_cmp, reason = "These comparisons should be exact.")]
    fn cost_calculation() {
        let grid = square_grid();
        let mut median =
            Discrete::median_from_points(&grid).expect("Creating discrete should not fail.");
        let mut discrete_kmeans = Discrete::squared_euclidean_from_points(&grid)
            .expect("Creating discrete should not fail.");
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
        let mut discrete = Discrete::median_from_points(&square_grid())
            .expect("Creating discrete should not fail.");

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
        let mut discrete = Discrete::squared_euclidean_from_points(&square_grid())
            .expect("Creating discrete should not fail.");

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
        let mut discrete = Discrete::median_from_points(&triangle_grid)
            .expect("Creating discrete should not fail.");
        assert_eq!(discrete.num_points(), triangle_grid.len());
        let (score, hierarchy) = discrete.price_of_hierarchy();
        assert_eq!(hierarchy.len(), triangle_grid.len());

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

        let points: Points = vec![
            array![0.0],
            array![1e-9],
            array![(3.0_f64.sqrt() - 1.0) / 2.0],
            array![(3.0 - 3.0_f64.sqrt()) / 2.0],
            array![1.0 + 1e-9],
            array![1.0 + 3e-9],
        ];
        let mut discrete =
            Discrete::median_from_points(&points).expect("Creating discrete should not fail.");
        assert_eq!(discrete.num_points(), points.len());
        let (score, hierarchy) = discrete.price_of_hierarchy();
        assert_eq!(hierarchy.len(), points.len());

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
            (score - (1.0 + 3.0_f64.sqrt()) / 2.0).abs() <= 1e-3,
            "Score should be close to 1.366."
        );
    }
}
