//! Finding optimal clusterings and hierarchical clusterings.
//! TODO: Better crate-doc.

use bit_set::BitSet;
use grb::{
    expr::LinExpr,
    prelude::*,
};
use ndarray::Array1;
use ordered_float::OrderedFloat;
use ordermap::OrderSet;
use pathfinding::{directed::astar::astar, num_traits::Zero};
use std::{collections::HashSet, hash::Hash};

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
        self.0.iter().fold(f64::MAX, |acc, j| {
            acc.min(self.0.iter().map(|k| data[k][j]).sum())
        })
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
        let mut clusters: Vec<BitSet<usize>> = vec![Default::default(); num_points];
        // Immediately terminate the function if any errors are encountered
        for (point_ix, assigned_to) in var_point_is_assigned_to.iter().enumerate() {
            for (cluster_ix, assigned) in assigned_to.iter().enumerate() {
                if model.get_obj_attr(attr::X, assigned)? > 0.5 {
                    clusters[cluster_ix].insert(point_ix);
                }
            }
        }
        let clusters = HashSet::from_iter(clusters.into_iter().map(|cluster| Self(cluster)));

        // Verify objective function
        #[cfg(debug_assertions)]
        {
            let objective_value_from_centers: f64 = clusters
                .iter()
                .map(|cluster| cluster.calculate_cost(distances))
                .sum();
            debug_assert!((opt_value - objective_value_from_centers).abs() < 1e-6, "The objective, calculated from the chunk-centers, should not deviate too much from the objective gurobi calculated.");
        }

        Ok((opt_value, clusters))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
            center = center + data[ix].clone();
        }
        center /= self.0.len() as f64;
        self.0
            .iter()
            .map(|ix| (center.clone() - data[ix].clone()).map(|x| x.powi(2)).sum())
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
        let num_points = points.len();
        let dim = points[0].len();
        let max_squared_dist: f64 = points.iter().enumerate().fold(0.0, |acc, (i, p)| {
            points.iter().skip(i + 1).fold(acc, |acc, q| {
                acc.max((p - q).iter().map(|x| x.powi(2)).sum())
            })
        });

        let mut env = Env::empty()?;
        env.set(param::OutputFlag, 0)?;
        env.set(param::LogToConsole, 0)?;
        let mut model = Model::with_env("clustering", &env.start()?)?;

        // point_is_assigned_to_cluster[i][c] is true iff points[i] is in cluster c
        let point_is_assigned_to_cluster: Vec<Vec<Var>> = (0..num_points)
            .map(|_| (0..k).map(|_| add_binvar!(model)).collect())
            .collect::<Result<_, _>>()?;
        // cluster_center[c][j] is the j-th coordinate of the center of cluster c
        let cluster_center: Vec<Vec<Var>> = (0..k)
            .map(|_| (0..dim).map(|_| add_ctsvar!(model)).collect())
            .collect::<Result<_, _>>()?;
        // cluster_point_radius_squared[c][i] denotes the distance between points[i] and cluster_center[c]
        let cluster_point_radius_squared: Vec<Vec<Var>> = (0..k)
            .map(|_| {
                (0..num_points)
                    .map(|_| add_ctsvar!(model, bounds: 0.0..))
                    .collect()
            })
            .collect::<Result<_, _>>()?;

        model.set_objective(
            cluster_point_radius_squared
                .iter()
                .map(|xs| xs.grb_sum())
                .grb_sum(),
            Minimize,
        )?;

        for (cluster_ix, radii_squared) in cluster_point_radius_squared.iter().enumerate() {
            for (point_ix, radius_squared) in radii_squared.iter().enumerate() {
                let constraint_name = format!(
                    "cluster_{cluster_ix}_has_a_large_enough_radius_to_contain_point_{point_ix}"
                );

                let distance_squared = points[point_ix]
                    .iter()
                    .enumerate()
                    .map(|(coordinate_ix, point_coordinate)| {
                        // (a-b)^2 = a^2 - 2ab + b^2
                        let a = *point_coordinate;
                        let b = cluster_center[cluster_ix][coordinate_ix];
                        a * a - 2.0 * a * b + b * b
                    })
                    .grb_sum();
                let constraint = c!(distance_squared
                    <= *radius_squared
                        + max_squared_dist
                            * (1.0 - point_is_assigned_to_cluster[point_ix][cluster_ix]));
                model.add_qconstr(&constraint_name, constraint)?;
            }
        }

        for (point_ix, is_assigned_to_cluster) in point_is_assigned_to_cluster.iter().enumerate() {
            let constraint_name = format!("{point_ix}_is_assigned_to_exactly_one_cluster");
            let constraint = c!(is_assigned_to_cluster.iter().grb_sum() == 1);
            model.add_constr(&constraint_name, constraint)?;
        }

        // TODO: Try setting start-values using kmeans++
        model.optimize()?;

        let opt_value: f64 = model.get_attr(attr::ObjVal)?;
        let mut clusters: Vec<BitSet<usize>> = vec![BitSet::default(); num_points];
        // Immediately terminate the function if any errors are encountered
        for (point_ix, assigned_to) in point_is_assigned_to_cluster.iter().enumerate() {
            for (cluster_ix, assigned) in assigned_to.iter().enumerate() {
                if model.get_obj_attr(attr::X, assigned)? > 0.5 {
                    clusters[cluster_ix].insert(point_ix);
                }
            }
        }
        let clusters: HashSet<Self> = clusters.into_iter().map(Self).collect();

        // Verify objective function
        #[cfg(debug_assertions)]
        {
            let objective_value_from_centers: f64 = clusters
                .iter()
                .map(|cluster| cluster.calculate_cost(points))
                .sum();
            debug_assert!((opt_value - objective_value_from_centers).abs() < 1e-4, "The objective, calculated from the chunk-centers, should not deviate too much from the objective gurobi calculated.");
        }

        Ok((opt_value, clusters))
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
        self.clusters
            .iter()
            .enumerate()
            .flat_map(move |(j, cluster_j)| {
                (0..j).map(move |i| {
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
                })
            })
    }

    #[must_use]
    pub fn optimal_hierarchy(data: &C::Data) -> (Vec<Clustering<C>>, f64) {
        let num_points = C::num_points(data);
        let opt_for_fixed_k: Vec<f64> = std::iter::once(0.0)
            .chain((1..=num_points).map(|k| C::optimal_clustering(data, k).unwrap().0))
            .collect();

        let initial_clustering = Clustering::new(num_points);
        let solution = astar(
            &initial_clustering,
            |clustering| {
                // TODO: Is collecting this into a vector really necessary here?
                let neighbors: Vec<(Clustering<C>, MaxRatio)> = clustering
                    .get_all_merges(data)
                    .map(|clustering| {
                        // This can be optimized slightly by implementing an own version of astar
                        // and using hierarchies instead of calling clustering.clusters.len() every time
                        let ratio = MaxRatio::new(
                            clustering.cost / opt_for_fixed_k[clustering.clusters.len()],
                        );
                        (clustering, ratio)
                    })
                    .collect();
                neighbors
            },
            // TODO: This is a terrible heuristic. Instead of taking the maximum of
            |clustering| {
                MaxRatio::new(clustering.cost / opt_for_fixed_k[clustering.clusters.len()])
            },
            |clustering| clustering.clusters.len() == 1,
        )
        .unwrap();

        (solution.0, solution.1 .0 .0)
    }

    #[must_use]
    pub fn greedy_hierarchy(data: &C::Data) -> (Vec<Clustering<C>>, f64) {
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

        (solution, highest_ratio.0 .0)
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
