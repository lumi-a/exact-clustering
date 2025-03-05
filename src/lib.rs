//! Finding optimal clusterings and hierarchical clusterings.
//! TODO: Better crate-doc.

use std::error::Error;

use bit_set::BitSet;
use good_lp::{
    constraint, default_solver, variable, variables, Expression, ResolutionError, Solution,
    SolverModel, Variable,
};
use ordered_float::OrderedFloat;
use ordermap::OrderSet;
use pathfinding::{directed::astar::astar, num_traits::Zero};

type Distances = Vec<Vec<f64>>;

pub fn optimal_discrete_clustering(
    distances: &Distances,
    k: usize,
) -> Result<(f64, Vec<usize>), ResolutionError> {
    let num_points = distances.len();
    let mut problem = variables!();

    // var_point_is_assigned_to[i][j] is true iff points[i] has center points[j]
    let var_point_is_assigned_to: Vec<Vec<Variable>> = (0..num_points)
        .map(|_| {
            (0..num_points)
                .map(|_| problem.add(variable().binary()))
                .collect()
        })
        .collect();
    // var_point_is_a_center is true iff points[i] is a center
    let var_point_is_a_center: Vec<Variable> = (0..num_points)
        .map(|_| problem.add(variable().binary()))
        .collect();

    let objective: Expression = var_point_is_assigned_to
        .iter()
        .enumerate()
        .map(|(i, vars_assigned_to)| {
            vars_assigned_to
                .iter()
                .enumerate()
                .map(|(j, var_assigned_to)| distances[i][j] * *var_assigned_to)
                .sum::<Expression>()
        })
        .sum();
    let mut model = problem
        .minimise(objective.clone())
        .using(default_solver)
        .with(constraint!(
            // We have exactly k centers
            var_point_is_a_center.iter().sum::<Expression>() == k as i32
        ));
    // Every point is assigned to exactly one center
    for vars_assigned_to in &var_point_is_assigned_to {
        model = model.with(constraint!(
            vars_assigned_to.iter().sum::<Expression>() == 1
        ));
    }
    // If a point i has points assigned to it, it should be a center
    for (i, center) in var_point_is_a_center.iter().enumerate() {
        // Sum over all points j assigned to center i.
        let assignments_to_center_i: Expression = (0..num_points)
            .map(|j| var_point_is_assigned_to[j][i])
            .sum();
        model = model.with(constraint!(
            assignments_to_center_i <= *center * (num_points as f64)
        ));
    }

    let solution = model.solve()?;

    let opt_value = solution.eval(&objective);
    let centers = var_point_is_a_center
        .iter()
        .enumerate()
        .filter_map(|(i, point)| {
            if solution.value(*point) == 1.0 {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    Ok((opt_value, centers))
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Cluster(BitSet<usize>);
impl Cluster {
    fn calculate_cost(&self, distances: &Distances) -> f64 {
        self.0.iter().fold(f64::MAX, |acc, j| {
            acc.min(self.0.iter().map(|k| distances[k][j]).sum())
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
}

type Set<T> = OrderSet<T, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;

#[derive(Clone, Debug)]
pub struct Clustering {
    // At all times, it must hold that clusters[i].calculate_cost() == cluster_costs[i], and
    // cost = cluster_costs.sum().
    // TODO: Enforce this somehow.
    clusters: Set<Cluster>,
    cluster_costs: Vec<f64>,
    cost: f64,
}
impl Clustering {
    fn new(num_points: usize) -> Self {
        Self {
            clusters: (0..num_points).map(Cluster::new_singleton).collect(),
            cluster_costs: vec![0.0; num_points],
            cost: 0.0,
        }
    }

    fn get_all_merges<'a>(
        &'a self,
        distances: &'a Distances,
    ) -> impl Iterator<Item = Self> + use<'a> {
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
                    let cluster_cost_i = clusters[i].calculate_cost(distances);
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
}
// Only consider self.clusters in equality-checks
impl PartialEq for Clustering {
    fn eq(&self, other: &Self) -> bool {
        self.clusters == other.clusters
    }
}
impl Eq for Clustering {}
impl std::hash::Hash for Clustering {
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

#[must_use]
pub fn optimal_hierarchy(distances: &Distances) -> (Vec<Clustering>, f64) {
    let num_points = distances.len();
    let opt_for_fixed_k: Vec<f64> = std::iter::once(0.0)
        .chain((1..=num_points).map(|k| optimal_discrete_clustering(distances, k).unwrap().0))
        .collect();

    let initial_clustering = Clustering::new(num_points);
    let solution = astar(
        &initial_clustering,
        |clustering| {
            // TODO: Is collecting this into a vector really necessary here?
            let neighbors: Vec<(Clustering, MaxRatio)> = clustering
                .get_all_merges(distances)
                .map(|clustering| {
                    // This can be optimized slightly by implementing an own version of astar
                    // and using hierarchies instead of calling clustering.clusters.len() every time
                    let ratio =
                        MaxRatio::new(clustering.cost / opt_for_fixed_k[clustering.clusters.len()]);
                    (clustering, ratio)
                })
                .collect();
            neighbors
        },
        // TODO: This is a terrible heuristic. Instead of taking the maximum of
        |clustering| MaxRatio::new(clustering.cost / opt_for_fixed_k[clustering.clusters.len()]),
        |clustering| clustering.clusters.len() == 1,
    )
    .unwrap();

    (solution.0, solution.1 .0 .0)
}

#[must_use]
pub fn greedy_hierarchy(distances: &Distances) -> (Vec<Clustering>, f64) {
    let num_points = distances.len();

    let mut clustering = Clustering::new(num_points);
    let mut solution: Vec<Clustering> = vec![clustering.clone()];
    let mut highest_ratio: MaxRatio = MaxRatio::zero();
    while clustering.clusters.len() > 1 {
        let best_merge = clustering
            .get_all_merges(distances)
            .min_by_key(|clustering| OrderedFloat(clustering.cost))
            .expect("There should always be a possible merge");
        clustering = best_merge;
        let optimal_clustering = optimal_discrete_clustering(distances, clustering.clusters.len())
            .unwrap()
            .0;
        highest_ratio = highest_ratio + MaxRatio::new(clustering.cost / optimal_clustering);
        solution.push(clustering.clone());
    }

    (solution, highest_ratio.0 .0)
}
