//! Finding instances with a high cost-of-hierarchy.
//! TODO: Better doc.

use ::rand::distributions::Uniform;
use ndarray::Array;
use ndarray_rand::{rand, RandomExt};
use price_of_hierarchy::{Cost, Discrete};

const NUM_POINTS: usize = 8;
const DIMENSION: usize = 2;

fn main() {
    let points = Array::random((2, 24), Uniform::new(0.0, 1.0));
    println!(
        "{}",
        Discrete::median_from_points(&points).price_of_hierarchy().0
    );
    /*
    type Point = ndarray::Array1<f64>;
    type Population = [Point; NUM_POINTS];

    get_score(&[
        array![-0.4332835992338185, 0.1812396142149292],
        array![2.8755997103542925, 6.412788278518556],
        array![-1.4249310222898859, 1.7171915366089017],
        array![-1.4144065311858454, 1.7644583139955183],
        array![-0.35555264364671574, -1.4887623316716254],
        array![-4.4855887746900365, 0.12340344297463382],
        array![-0.14248514389973613, -3.8332493986219505],
        array![-4.5758122596668205, 0.1580379508633737],
    ]);

    /// Calculate the score of a population
    fn get_score(points: &Population) -> f64 {
        let distances: Vec<Vec<f64>> = points
            .iter()
            .map(|p| {
                points
                    .iter()
                    .map(|q| (p - q).iter().map(|x| x.abs()).sum())
                    .collect()
            })
            .collect();
        println!(
            "{}",
            Clustering::<ContinuousKMeansCluster>::optimal_hierarchy(&points.to_vec()).0
        );
        println!(
            "{}",
            Clustering::<ContinuousKMeansCluster>::greedy_hierarchy(&points.to_vec()).0
        );
        println!(
            "{}",
            Clustering::<DiscreteCluster>::optimal_hierarchy(&distances.to_vec()).0
        );
        0.1
    }
    return;

    /// Return a mutated population
    fn mutate_population(pop: &Population, rng: &mut ThreadRng, temperature: f64) -> Population {
        pop.clone().map(|p: Point| {
            let offset: Array1<f64> =
                temperature * Array::random_using(DIMENSION, StandardNormal, rng);
            p + offset
        })
    }

    let mut rng = rand::thread_rng();
    let num_populations = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let mut populations: Vec<(f64, Population)> = (0..num_populations)
        .map(|_| {
            let population = [0; NUM_POINTS]
                .map(|_| Array::random_using(DIMENSION, Uniform::new(0.0, 1.0), &mut rng));
            (get_score(&population), population)
        })
        .collect();

    /// Calculate the temperature of a given generation
    fn get_temperature(generation: u32) -> f64 {
        f64::exp(-(f64::from(generation) / 2000.0))
    }

    let generations_between_replacements = 1500;
    let mut generation: u32 = 0;
    loop {
        let current_generation = generation;
        populations.par_iter_mut().for_each(|(score, population)| {
            let mut rng = rand::thread_rng();
            for gen in current_generation..(current_generation + generations_between_replacements) {
                let mutated_population: Population =
                    mutate_population(population, &mut rng, get_temperature(gen));
                let mutated_score = get_score(&mutated_population);
                if mutated_score > *score {
                    *score = mutated_score;
                    *population = mutated_population;
                }
            }
        });
        generation += generations_between_replacements;

        // Print populations
        populations.sort_by_key(|p| OrderedFloat(p.0));
        let scoresvec: Vec<String> = populations.iter().map(|p| format!("{:.2}", p.0)).collect();
        let populationstrings: Vec<String> = populations
            .iter()
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .expect("The population should have constant size, which implies it should be nonempty")
            .1
            .iter()
            .map(|point| format!("{point}"))
            .collect();
        println!(
            "Generation {generation}, temp {:e}, scores {}, best population:\n[{}]\n",
            get_temperature(generation),
            scoresvec.join(", "),
            populationstrings.join(", ")
        );

        // Reset populations, replacing the worst half with the best half
        #[allow(
            clippy::indexing_slicing,
            reason = "I *promise* `populations` will always have size `num_populations`"
        )]
        {
            let better_half = populations[num_populations / 2..].to_vec();
            populations[0..num_populations / 2].clone_from_slice(&better_half);
        }
    }
    */
}
