//! Finding optimal clusterings and hierarchical clusterings.
//! TODO: Better crate-doc.

use bit_set::BitSet;
use ndarray::Array1;
use ordered_float::OrderedFloat;
use ordermap::OrderSet;
use pathfinding::{directed::astar::astar, num_traits::Zero, prelude::dijkstra};
use rustc_hash::{FxHashMap, FxHashSet};
use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashSet},
    hash::Hash,
};

type Distances = Vec<Vec<f64>>;

// TODO: Should we have Data in the signature? Would it be acceptable to store a reference+lifetime
// in each struct-instance?
pub trait Cluster {
    type Data;
    fn cluster_size(&self) -> usize;
    fn num_points(data: &Self::Data) -> usize;
    fn calculate_cost(&self, data: &Self::Data) -> f64;
    fn new_singleton(i: usize) -> Self;
    fn merge_with(&mut self, other: &Self);
    fn optimal_clustering(data: &Self::Data, k: usize) -> Result<(f64, HashSet<Self>), grb::Error>
    where
        Self: Sized;
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DiscreteCluster(BitSet<usize>);
impl Cluster for DiscreteCluster {
    type Data = Distances;
    fn cluster_size(&self) -> usize {
        self.0.len()
    }
    fn num_points(data: &Self::Data) -> usize {
        data.len()
    }
    fn calculate_cost(&self, data: &Self::Data) -> f64 {
        self.0
            .iter()
            .map(|center_candidate| {
                let center_distances = &data[center_candidate];
                self.0.iter().map(|i| center_distances[i]).sum()
            })
            .min_by(f64::total_cmp)
            .unwrap_or(0.0)
    }
    fn new_singleton(i: usize) -> Self {
        let mut bv: BitSet<usize> = Default::default();
        bv.insert(i);
        Self(bv)
    }
    fn merge_with(&mut self, other: &Self) {
        self.0.union_with(&other.0);
    }
    fn optimal_clustering(
        distances: &Self::Data,
        k: usize,
    ) -> Result<(f64, HashSet<Self>), grb::Error> {
        use grb::{expr::LinExpr, prelude::*};

        let num_points = distances.len();

        let mut env = Env::empty()?;
        env.set(param::OutputFlag, 0)?;
        env.set(param::LogToConsole, 0)?;
        let mut model = Model::with_env("clustering", &env.start()?)?;

        // var_point_is_assigned_to[i][j] is true iff points[i] has center points[j]
        let var_point_is_assigned_to: Vec<Vec<Var>> = (0..num_points)
            .map(|_| (0..num_points).map(|_| add_binvar!(model)).collect())
            .collect::<Result<_, _>>()?;

        // var_point_is_a_center is true iff points[i] is a center
        let var_point_is_a_center: Vec<Var> = (0..num_points)
            .map(|_| add_binvar!(model))
            .collect::<Result<_, _>>()?;

        let mut objective = LinExpr::new();
        for (i, point_i_is_assigned_to) in var_point_is_assigned_to.iter().enumerate() {
            for (j, assigned_to_center) in point_i_is_assigned_to.iter().enumerate() {
                objective.add_term(distances[i][j], *assigned_to_center);
            }
        }

        model.set_objective(objective, Minimize)?;

        model.add_constr(
            "exactly_k_centers",
            c!(var_point_is_a_center.iter().grb_sum() == k),
        )?;

        for (i, point_i_is_assigned_to) in var_point_is_assigned_to.iter().enumerate() {
            let constraint_name = format!("{i}_has_exactly_one_center");
            let constraint = c!(point_i_is_assigned_to.iter().grb_sum() == 1);
            model.add_constr(&constraint_name, constraint)?;
        }

        for (i, point_i_is_a_center) in var_point_is_a_center.iter().enumerate() {
            let constraint_name = format!("if_{i}_has_points_assigned_to_it_it_should_be_a_center");
            let constraint = {
                let mut assignments_to_center_i = LinExpr::new();
                for point_has_center in &var_point_is_assigned_to {
                    assignments_to_center_i.add_term(1.0, point_has_center[i]);
                }
                c!(assignments_to_center_i <= *point_i_is_a_center * num_points)
            };
            model.add_constr(&constraint_name, constraint)?;
        }

        model.optimize()?;

        let opt_value: f64 = model.get_attr(attr::ObjVal)?;
        let mut clusters: Vec<BitSet<usize>> = vec![BitSet::default(); num_points];
        // Immediately terminate the function if any errors are encountered
        for (point_ix, assigned_to) in var_point_is_assigned_to.iter().enumerate() {
            for (cluster_ix, assigned) in assigned_to.iter().enumerate() {
                if model.get_obj_attr(attr::X, assigned)? > 0.5 {
                    clusters[cluster_ix].insert(point_ix);
                }
            }
        }
        let clusters: HashSet<Self> = clusters
            .into_iter()
            .filter(|cluster| !cluster.is_empty())
            .map(Self)
            .collect();

        // Verify objective function
        #[cfg(debug_assertions)]
        {
            let objective_value_from_centers: f64 = clusters
                .iter()
                .map(|cluster| cluster.calculate_cost(distances))
                .sum();
            debug_assert!((opt_value - objective_value_from_centers).abs() < 1e-6, "The objective calculated from the chunk-centers ({objective_value_from_centers}), should not deviate too much from the objective gurobi calculated ({opt_value}).");
        }

        Ok((opt_value, clusters))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ContinuousKMeansCluster(BitSet<usize>);
impl Cluster for ContinuousKMeansCluster {
    type Data = Vec<Array1<f64>>;
    fn cluster_size(&self) -> usize {
        self.0.len()
    }
    fn num_points(data: &Self::Data) -> usize {
        data.len()
    }
    fn calculate_cost(&self, data: &Self::Data) -> f64 {
        let mut center = Array1::zeros(data[0].raw_dim());
        for ix in self.0.iter() {
            center += &data[ix];
        }
        center /= self.0.len() as f64;
        self.0
            .iter()
            .map(|ix| (&center - &data[ix]).map(|x| x.powi(2)).sum())
            .sum()
    }
    fn new_singleton(i: usize) -> Self {
        let mut bv: BitSet<usize> = BitSet::default();
        bv.insert(i);
        Self(bv)
    }
    fn merge_with(&mut self, other: &Self) {
        self.0.union_with(&other.0);
    }
    fn optimal_clustering(
        points: &Self::Data,
        k: usize,
    ) -> Result<(f64, HashSet<Self>), grb::Error> {
        use clustering::kmeans;

        let max_iter = 1000; // TODO: Benchmark this?
        let (kmeans_cost, kmeans_clusters): (f64, HashSet<Self>) = {
            let clusters: HashSet<Self> = {
                let samples: Vec<Vec<f64>> = points
                    .iter()
                    .map(|x| x.clone().into_iter().collect())
                    .collect();
                let clustering = kmeans(k, &samples, max_iter);
                let mut clusters = vec![BitSet::<usize>::default(); k];
                for (point_ix, cluster_ix) in clustering.membership.iter().enumerate() {
                    clusters[*cluster_ix].insert(point_ix);
                }
                clusters.into_iter().map(Self).collect()
            };
            let cost = clusters.iter().map(|x| x.calculate_cost(&points)).sum();
            (cost, clusters)
        };

        type Cluster = BitSet<usize>;
        #[derive(Clone, Debug)]
        struct ClusteringNode {
            cost: f64,
            clusters: Vec<BitSet<usize>>,
            next_to_add: usize,
        }
        impl PartialEq for ClusteringNode {
            fn eq(&self, other: &Self) -> bool {
                (self.cost == other.cost) & (self.clusters == other.clusters)
            }
        }
        impl Eq for ClusteringNode {}
        // Order them by highest cost first
        impl Ord for ClusteringNode {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                // TODO: Check if using partial_cmp is faster
                other
                    .cost
                    .total_cmp(&self.cost)
                    .then_with(|| self.clusters.cmp(&other.clusters))
            }
        }
        impl PartialOrd for ClusteringNode {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        type Costs = FxHashMap<Cluster, f64>;
        impl ClusteringNode {
            fn join<'a>(
                &'a self,
                costs: &'a mut Costs,
                points: &'a Vec<Array1<f64>>,
                k: usize,
            ) -> impl Iterator<Item = Self> + use<'a> {
                (0..self.clusters.len())
                    .map(move |cluster_ix| {
                        let mut clustering_node = self.clone();
                        let cluster_to_edit =
                            unsafe { clustering_node.clusters.get_unchecked_mut(cluster_ix) };
                        clustering_node.cost -=
                            unsafe { costs.get(cluster_to_edit).unwrap_unchecked() };
                        cluster_to_edit.insert(clustering_node.next_to_add);
                        clustering_node.cost += *costs
                            .entry(cluster_to_edit.clone())
                            .or_insert_with_key(|cluster| {
                                let mut center = Array1::zeros(points[0].raw_dim());
                                for ix in cluster {
                                    center += &points[ix];
                                }
                                center /= cluster.len() as f64;
                                cluster
                                    .iter()
                                    .map(|ix| (&center - &points[ix]).map(|x| x.powi(2)).sum())
                                    .sum()
                            });
                        clustering_node.next_to_add += 1;
                        clustering_node
                    })
                    .chain((self.clusters.len() < k).then(|| {
                        let mut clustering_node = self.clone();
                        let mut singleton: BitSet<usize> = BitSet::default();
                        singleton.insert(clustering_node.next_to_add);
                        clustering_node.clusters.push(singleton);
                        clustering_node.next_to_add += 1;
                        clustering_node
                    }))
            }
            fn empty() -> Self {
                Self {
                    clusters: Vec::new(),
                    cost: 0.0,
                    next_to_add: 0,
                }
            }
        }

        let mut to_see: BinaryHeap<ClusteringNode> = BinaryHeap::new();
        to_see.push(ClusteringNode::empty());
        let mut costs: FxHashMap<Cluster, f64> = FxHashMap::default();
        let empty: BitSet<usize> = BitSet::default();
        costs.insert(empty, 0.0);
        for i in 0..points.len() {
            let mut singleton: BitSet<usize> = BitSet::default();
            singleton.insert(i);
            costs.insert(singleton, 0.0);
        }

        let mut min_cost = kmeans_cost;
        while let Some(clustering_node) = to_see.pop() {
            if clustering_node.clusters.len() == k && clustering_node.next_to_add == points.len() {
                return Ok((
                    clustering_node.cost,
                    clustering_node
                        .clusters
                        .into_iter()
                        .map(ContinuousKMeansCluster)
                        .collect(),
                ));
            }
            for successor_clustering in clustering_node.join(&mut costs, &points, k) {
                if successor_clustering.cost < min_cost {
                    if successor_clustering.clusters.len() == k
                        && successor_clustering.next_to_add == points.len()
                    {
                        min_cost = successor_clustering.cost;
                    }
                    to_see.push(successor_clustering);
                }
            }
        }
        // This can only happen due to floating-point-rounding-errors.
        Ok((kmeans_cost, kmeans_clusters))
    }
}

type Set<T> = OrderSet<T, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;

#[derive(Clone, Debug)]
pub struct Clustering<C> {
    // At all times, it must hold that clusters[i].calculate_cost() == cluster_costs[i], and
    // cost = cluster_costs.sum().
    // TODO: Enforce this somehow.
    clusters: Set<C>,
    cluster_costs: Vec<f64>,
    cost: f64,
}
impl<C> Clustering<C>
where
    C: Cluster + Hash + Eq,
{
    fn new(num_points: usize) -> Self {
        Self {
            clusters: (0..num_points).map(C::new_singleton).collect(),
            cluster_costs: vec![0.0; num_points],
            cost: 0.0,
        }
    }
}
impl<C> Clustering<C>
where
    C: Cluster + Hash + Eq + Clone,
{
    fn get_all_merges<'a>(&'a self, data: &'a C::Data) -> impl Iterator<Item = Self> + use<'a, C> {
        // Iterate over the late element first. We'll require that i < j,
        // but for the merge, we do want to see cluster_j (for the union)
        // and we won't need to see cluster_i (because it's cloned and
        // modified in-place).
        (0..self.clusters.len())
            .flat_map(move |j| (0..j).map(move |i| self.merge_clusters(data, j, i)))
    }

    fn merge_clusters(&self, data: &<C as Cluster>::Data, j: usize, i: usize) -> Clustering<C> {
        let cluster_j = &self.clusters[j];
        let mut clusters = self.clusters.clone();
        let mut cluster_costs = self.cluster_costs.clone();
        let mut cost = self.cost;

        // Deduct cluster_costs[i] and cluster_costs[j] from cost, we'll re-add the new cost later
        cost -= cluster_costs[i] + cluster_costs[j];

        // Merge clusters, do evil hack to update clusters[i], as seen in
        // https://github.com/indexmap-rs/indexmap/issues/362#issuecomment-2495107460
        let mut merged_cluster = clusters.swap_remove_index(i).unwrap();
        merged_cluster.merge_with(cluster_j);
        let (inserted_index, was_new) = clusters.insert_full(merged_cluster);
        debug_assert!(was_new);
        clusters.swap_indices(i, inserted_index);
        // `clusters` should now be the same, including orders of elements, except clusters[i] is updated.
        clusters.swap_remove_index(j);

        // Update cluster_costs
        // TODO: For continuous k-means, we could store the centroids of each cluster, the
        // merged centroid should then be (size_a * centroid_a + size_b * centroid_b) / (size_a + size_b),
        // right?
        let cluster_cost_i = clusters[i].calculate_cost(data);
        cluster_costs[i] = cluster_cost_i;
        cluster_costs.swap_remove(j);

        // Re-add the new cluster_cost to cost
        cost += cluster_cost_i;

        Self {
            clusters,
            cluster_costs,
            cost,
        }
    }

    #[must_use]
    pub fn optimal_hierarchy(data: &C::Data) -> (f64, Vec<Clustering<C>>) {
        let num_points = C::num_points(data);
        let opt_for_fixed_k: Vec<f64> = std::iter::once(0.0)
            .chain((1..=num_points).map(|k| C::optimal_clustering(data, k).unwrap().0))
            .collect();

        let initial_clustering = Clustering::new(num_points);
        let solution = dijkstra(
            &initial_clustering,
            |clustering| {
                // TODO: Is collecting this into a vector really necessary here?
                let opt_cost = opt_for_fixed_k[clustering.clusters.len() - 1];
                let vec: Vec<_> = clustering
                    .get_all_merges(data)
                    .map(|new_clustering| {
                        let ratio = MaxRatio::new(new_clustering.cost / opt_cost);
                        (new_clustering, ratio)
                    })
                    .collect();
                vec
            },
            |clustering| clustering.clusters.len() == 1,
        )
        .unwrap();

        (solution.1 .0 .0, solution.0)
    }

    #[must_use]
    pub fn greedy_hierarchy(data: &C::Data) -> (f64, Vec<Clustering<C>>) {
        let num_points = C::num_points(data);

        let mut clustering = Clustering::new(num_points);
        let mut solution: Vec<Clustering<C>> = vec![clustering.clone()];
        let mut highest_ratio: MaxRatio = MaxRatio::zero();
        while clustering.clusters.len() > 1 {
            let best_merge = clustering
                .get_all_merges(data)
                .min_by_key(|clustering| OrderedFloat(clustering.cost))
                .expect("There should always be a possible merge");
            clustering = best_merge;
            let optimal_clustering = C::optimal_clustering(data, clustering.clusters.len())
                .unwrap()
                .0;
            highest_ratio = highest_ratio + MaxRatio::new(clustering.cost / optimal_clustering);
            solution.push(clustering.clone());
        }

        (highest_ratio.0 .0, solution)
    }
}
// Only consider self.clusters in equality-checks
impl<C> PartialEq for Clustering<C>
where
    C: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.clusters == other.clusters
    }
}
impl<C> Eq for Clustering<C> where C: Eq {}
impl<C> Hash for Clustering<C>
where
    C: Hash,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.clusters.hash(state);
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct MaxRatio(OrderedFloat<f64>);
impl MaxRatio {
    fn new(float: f64) -> Self {
        MaxRatio(OrderedFloat(float))
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
        MaxRatio(OrderedFloat(f64::NEG_INFINITY))
    }
    fn is_zero(&self) -> bool {
        self.0 == OrderedFloat(f64::NEG_INFINITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn get_grid() -> Vec<Array1<f64>> {
        // Grid that looks like this:
        //
        //   ::  ::
        //
        //   ::  ::
        //
        [0, 3]
            .iter()
            .flat_map(|big_x| {
                [0, 3].iter().flat_map(move |big_y| {
                    [0, 1].iter().flat_map(move |small_x| {
                        [0, 1].iter().map(move |small_y| {
                            array![f64::from(big_x + small_x), f64::from(big_y + small_y)]
                        })
                    })
                })
            })
            .collect()
    }

    fn get_distances<F>(points: &Vec<Array1<f64>>, element_function: &F) -> Distances
    where
        F: Fn(f64) -> f64,
    {
        points
            .iter()
            .map(|p| {
                points
                    .iter()
                    .map(|q| (p - q).into_iter().map(element_function).sum())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn optimal_discrete_k_median_clustering() {
        let distances = get_distances(&get_grid(), &|x| x.abs());
        let expected_clusters: HashSet<DiscreteCluster> = [0..4, 4..8, 8..12, 12..16]
            .into_iter()
            .map(|x| DiscreteCluster(x.collect()))
            .collect();

        let (score, clusters) = DiscreteCluster::optimal_clustering(&distances, 4)
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
        let distances = get_distances(&get_grid(), &|x| x.powi(2));
        let expected_clusters: HashSet<DiscreteCluster> = [0..4, 4..8, 8..12, 12..16]
            .into_iter()
            .map(|x| DiscreteCluster(x.collect()))
            .collect();

        let (score, clusters) = DiscreteCluster::optimal_clustering(&distances, 4)
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
    fn optimal_simple_continuous_k_means_clustering() {
        let points = vec![
            array![0.0, 0.0],
            array![0.0, 2.0],
            array![3.0, 0.0],
            array![3.0, 2.0],
        ];
        let (score, clusters) = ContinuousKMeansCluster::optimal_clustering(&points, 2)
            .expect("Calculating the optimal clustering should not fail.");
        assert_eq!(clusters.len(), 2);
        let expected_clusters: HashSet<ContinuousKMeansCluster> = [
            ContinuousKMeansCluster((0..=1).collect()),
            ContinuousKMeansCluster((2..=3).collect()),
        ]
        .into_iter()
        .collect();

        assert_eq!(
            clusters, expected_clusters,
            "Clusters should match expected clusters."
        );

        assert!(
            (score - 4.0).abs() < 1e-6,
            "Score should be close to the expected score"
        );
    }

    #[test]
    fn optimal_discrete_k_means_hierarchy() {
        // Points like this:
        //
        //  .
        //  .  .
        //                  .
        //
        //  :.              .     .
        //

        let triangle = [array![0.0, 0.0], array![0.0, 1.0], array![1.5, 0.0]];
        let points: Vec<Array1<f64>> = [
            triangle.clone(),
            triangle.clone().map(|v| v * 2.0 + array![0.0, 4.0]),
            triangle.clone().map(|v| v * 4.0 + array![0.0, 16.0]),
        ]
        .into_iter()
        .flatten()
        .collect();
        let distances = get_distances(&points, &|x| x.abs());

        let (score, hierarchy) = Clustering::<DiscreteCluster>::optimal_hierarchy(&distances);
        assert_eq!(hierarchy.len(), points.len());

        let expected_hierarchy: Vec<HashSet<DiscreteCluster>> = [
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
        .into_iter()
        .map(|v| {
            v.into_iter()
                .map(|x| DiscreteCluster(x.collect()))
                .collect()
        })
        .collect::<Vec<_>>();

        for (clustering, expected_clustering) in hierarchy.into_iter().zip(expected_hierarchy) {
            let clusterset: HashSet<DiscreteCluster> = clustering.clusters.into_iter().collect();
            assert_eq!(clusterset, expected_clustering);
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
    fn suboptimal_discrete_k_means_hierarchy() {
        // Points like this:
        //
        // ..    .     .     ..
        //

        let points: Vec<Array1<f64>> = vec![
            array![0.0],
            array![1e-9],
            array![(3.0f64.sqrt() - 1.0) / 2.0],
            array![(3.0 - 3.0f64.sqrt()) / 2.0],
            array![1.0],
            array![1.0 + 2e-9],
        ];
        let distances = get_distances(&points, &|x| x.abs());

        let (score, hierarchy) = Clustering::<DiscreteCluster>::optimal_hierarchy(&distances);
        assert_eq!(hierarchy.len(), points.len());

        let expected_hierarchy: Vec<HashSet<DiscreteCluster>> = [
            vec![0..=0, 1..=1, 2..=2, 3..=3, 4..=4, 5..=5],
            vec![0..=1, 2..=2, 3..=3, 4..=4, 5..=5],
            vec![0..=1, 2..=2, 3..=3, 4..=5],
            vec![0..=2, 3..=3, 4..=5],
            vec![0..=2, 3..=5],
            vec![0..=5],
        ]
        .into_iter()
        .map(|v| {
            v.into_iter()
                .map(|x| DiscreteCluster(x.collect()))
                .collect()
        })
        .collect::<Vec<_>>();

        for (clustering, expected_clustering) in hierarchy.into_iter().zip(expected_hierarchy) {
            let clusterset: HashSet<DiscreteCluster> = clustering.clusters.into_iter().collect();
            assert_eq!(clusterset, expected_clustering);
        }
        assert!(
            (score - (1.0 + 3.0f64.sqrt()) / 2.0).abs() <= 1e-3,
            "Score should be close to 1.366."
        );
    }
}
