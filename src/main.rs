//! Finding instances with a high cost-of-hierarchy.
//! TODO: Better doc.

use core::f64;

use ordered_float::OrderedFloat;
use price_of_hierarchy::optimal_hierarchy;
use rand::{rngs::ThreadRng, Rng};
use rayon::prelude::*;

fn main() {
    type Point = (f64,);
    type Population = [Point; 8];

    /// Calculate the score of a population
    fn get_score(points: &Population) -> f64 {
        let population: Vec<Point> = points.to_vec();

        let metric = |(a0,): &Point, (b0,): &Point| (a0 - b0).powi(2);
        // TODO: Inefficient, we could cut this down to about half the computations
        let distances: Vec<Vec<f64>> = population
            .iter()
            .map(|p| population.iter().map(|q| metric(p, q)).collect())
            .collect();

        optimal_hierarchy(&distances).1
    }

    /// Return a mutated population
    fn mutate_population(pop: &Population, rng: &mut ThreadRng, temperature: f64) -> Population {
        // TODO: Use normal distribution instead?
        let mutate = |(p0,): Point| ((p0 + (rng.random::<f64>() - 0.5) * temperature),);
        pop.map(mutate)
    }

    /// Calculate the temperature of a given generation
    fn get_temperature(generation: u32) -> f64 {
        f64::exp(-(f64::from(generation) / 250.0))
    }

    let mut rng = rand::rng();
    let num_populations = 3 * std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let mut populations: Vec<(f64, Population)> = (0..num_populations)
        .map(|_| {
            let population = rng.random();
            (get_score(&population), population)
        })
        .collect();

    let generations_between_replacements = 250;
    let mut generation: u32 = 0;
    loop {
        let current_generation = generation;
        populations.par_iter_mut().for_each(|(score, population)| {
            let mut rng = rand::rng();
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
        let populationstrings = populations
            .iter()
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .expect("The population should have constant size, which implies it should be nonempty")
            .1
            .map(|(p0,)| format!("({p0:.32},)"));
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
}
