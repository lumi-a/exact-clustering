//! Finding instances with a high cost-of-hierarchy.
//! TODO: Better doc.

use core::iter::repeat_with;
use core::num::NonZero;
use std::thread::available_parallelism;

use ndarray::{Array, Array1};
use ndarray_rand::{rand, rand_distr::StandardNormal, RandomExt as _};
use price_of_hierarchy::{Cost as _, KMeans, Points};
use rand::{distributions::Uniform, seq::index::sample, Rng as _};
use rayon::prelude::*;

/// The number of points
const NUM_POINTS: usize = 8;
/// The dimension of each point
const DIMENSION: usize = 1;

fn main() -> ! {
    /// Calculate the score of a given population
    fn get_score(points: &Points) -> f64 {
        KMeans::new(points)
            .map(|mut x| x.price_of_greedy().0)
            .unwrap_or(0.0)
    }
    /// Return a mutated population
    fn mutate(points: &Points, rng: &mut rand::rngs::ThreadRng, temperature: f64) -> Points {
        let mut new_points = points.clone();
        let numper_of_points_to_change = rng.gen_range(1..=points.len());
        let chosen_points = sample(rng, points.len(), numper_of_points_to_change).into_vec();

        for point_ix in chosen_points {
            let noise: Array1<f64> = temperature
                * Array::random_using(new_points[point_ix].raw_dim(), StandardNormal, rng);
            new_points[point_ix] += &noise;
        }
        new_points
    }
    /// Calculate the temperature of a given generation
    fn get_temperature(generation: u32) -> f64 {
        f64::exp(-(f64::from(generation) * 1e-4))
    }
    const GENERATIONS_BETWEEN_PRINTS: u32 = 1000;

    let num_populations = 10 * available_parallelism().map(NonZero::get).unwrap_or(1);
    let mut instances: Vec<(f64, Points)> = repeat_with(|| {
        let points: Points = repeat_with(|| {
            Array::random_using(DIMENSION, Uniform::new(0.0, 1.0), &mut rand::thread_rng())
        })
        .take(NUM_POINTS)
        .collect();
        (get_score(&points), points)
    })
    .take(num_populations)
    .collect();

    let mut generation: u32 = 0;
    loop {
        let current_generation = generation;
        instances.par_iter_mut().for_each(|(score, population)| {
            for gen in current_generation..(current_generation + GENERATIONS_BETWEEN_PRINTS) {
                let mut rng = rand::thread_rng();
                let mutant = mutate(population, &mut rng, get_temperature(gen));
                let mutated_score = get_score(&mutant);
                if mutated_score > *score {
                    *score = mutated_score;
                    *population = mutant;
                }
            }
        });
        generation += GENERATIONS_BETWEEN_PRINTS;

        // Print populations
        instances.sort_by(|a, b| a.0.total_cmp(&b.0));
        for instance in &instances {
            println!(
                "{}\n[{}]\n",
                instance.0,
                instance
                    .1
                    .iter()
                    .map(|p| p.to_string().replace('[', "(").replace(']', ")"))
                    .collect::<Vec<String>>()
                    .join(",")
            );
        }
        println!(
            "Generation {generation}, temperature {:e}\n----------------\n",
            get_temperature(generation)
        );

        // Reset populations, replacing the worst half with the best half
        /*
        #[expect(
            clippy::indexing_slicing,
            reason = "I *promise* `populations` will always have size `num_populations`"
        )]
        {
            let better_half = populations[num_populations / 2..].to_vec();
            populations[0..num_populations / 2].clone_from_slice(&better_half);
        }
        */
    }
}
