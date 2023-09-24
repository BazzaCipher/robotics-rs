#![allow(dead_code)] use criterion::measurement;
// TODO: remove this
use nalgebra::{allocator::Allocator, Const, DefaultAllocator, Dim, OMatrix, OVector, RealField};
use rand::distributions::Distribution;
use rand::Rng;
use rand_distr::{Standard, StandardNormal};
use rustc_hash::FxHashMap;

use crate::localization::bayesian_filter::{BayesianFilter, BayesianFilterKnownCorrespondences};
use crate::models::measurement::MeasurementModel;
use crate::models::motion::MotionModel;
use crate::utils::mvn::MultiVariateNormal;
use crate::utils::state::GaussianState;

pub enum ResamplingScheme {
    IID,
    Stratified,
    Systematic,
}

/// Trait that generalises the particle filter to respond with the particles
/// Useful for when using multiple different particle-based filters (MCL)
pub trait ParticleFilter<T: RealField + Copy, S: Dim, Z: Dim, U: Dim>:
    BayesianFilter<T, S, Z, U>
where
    DefaultAllocator: Allocator<T, S>
        + Allocator<T, S, S>
        + Allocator<T, Z, Z>
        + Allocator<T, U>
        + Allocator<T, Z>,
{
    type Particle;

    fn particles(&self) -> &Vec<Self::Particle>;
    fn particles_mut(&mut self) -> &mut Vec<Self::Particle>;
}

/// Trait that generalises the particle filter to respond with the particles
/// Useful for when using multiple different particle-based filters (MCL)
pub trait ParticleFilterKnownCorrespondences<T: RealField + Copy, S: Dim, Z: Dim, U: Dim>:
    BayesianFilterKnownCorrespondences<T, S, Z, U>
where
    DefaultAllocator: Allocator<T, S>
        + Allocator<T, S, S>
        + Allocator<T, Z, Z>
        + Allocator<T, U>
        + Allocator<T, Z>,
{
    fn particles(&self) -> &Vec<OVector<T, S>>;
    fn particles_mut(&mut self) -> &mut Vec<OVector<T, S>>;
}

/// S : State Size, Z: Observation Size, U: Input Size
pub struct GeneralParticleFilter<T: RealField, S: Dim, Z: Dim, U: Dim>
where
    DefaultAllocator: Allocator<T, S> + Allocator<T, S, S> + Allocator<T, Z, Z>,
{
    r: OMatrix<T, S, S>,
    q: OMatrix<T, Z, Z>,
    measurement_model: Box<dyn MeasurementModel<T, S, Z> + Send>,
    motion_model: Box<dyn MotionModel<T, S, Z, U> + Send>,
    pub particules: Vec<OVector<T, S>>,
    resampling_scheme: ResamplingScheme,
}

impl<T: RealField + Copy, S: Dim, Z: Dim, U: Dim> GeneralParticleFilter<T, S, Z, U>
where
    StandardNormal: Distribution<T>,
    Standard: Distribution<T>,
    DefaultAllocator: Allocator<T, S>
        + Allocator<T, S, S>
        + Allocator<T, Z, Z>
        + Allocator<T, Const<1>, S>
        + Allocator<T, Const<1>, Z>,
{
    pub fn new(
        r: OMatrix<T, S, S>,
        q: OMatrix<T, Z, Z>,
        measurement_model: Box<dyn MeasurementModel<T, S, Z> + Send>,
        motion_model: Box<dyn MotionModel<T, S, Z, U> + Send>,
        initial_state: GaussianState<T, S>,
        num_particules: usize,
        resampling_scheme: ResamplingScheme,
    ) -> GeneralParticleFilter<T, S, Z, U> {
        let mvn = MultiVariateNormal::new(&initial_state.x, &r).unwrap();
        let mut particules = Vec::with_capacity(num_particules);
        for _ in 0..num_particules {
            particules.push(mvn.sample());
        }

        GeneralParticleFilter {
            r,
            q,
            measurement_model,
            motion_model,
            particules,
            resampling_scheme,
        }
    }
}

impl<T: RealField + Copy, S: Dim, Z: Dim, U: Dim> BayesianFilter<T, S, Z, U>
    for GeneralParticleFilter<T, S, Z, U>
where
    DefaultAllocator: Allocator<T, S>
        + Allocator<T, U>
        + Allocator<T, Z>
        + Allocator<T, S, S>
        + Allocator<T, Z, Z>
        + Allocator<T, Z, S>
        + Allocator<T, S, U>
        + Allocator<T, U, U>
        + Allocator<T, Const<1>, S>
        + Allocator<T, Const<1>, Z>,
    Standard: Distribution<T>,
    StandardNormal: Distribution<T>,
{
    fn update_estimate(&mut self, u: Option<OVector<T, U>>, z: Option<Vec<OVector<T, Z>>>, dt: T) {
        // Add positional noise to model
        let shape = self.particules[0].shape_generic();
        let mvn =
            MultiVariateNormal::new(&OMatrix::zeros_generic(shape.0, shape.1), &self.r).unwrap();

        // With the motion model, predict the particles next position assuming
        // there is little to no noise
        if let Some(control) = u {
            self.particules = self
                .particules
                .iter()
                .map(|p| self.motion_model.prediction(p, &control, dt) + mvn.sample())
                .collect();
        }

        // Predicts the location of the particles based on the landmarks
        if let Some(measurements) = z {
            let mut weights = vec![T::one(); self.particules.len()];

            for measurement in measurements {
                let shape = measurement.shape_generic();
                let mvn =
                    MultiVariateNormal::new(&OMatrix::zeros_generic(shape.0, shape.1), &self.q)
                        .unwrap();

                for (i, particule) in self.particules.iter().enumerate() {
                    let z_pred = self.measurement_model.prediction(particule, None);
                    let error = &measurement - &z_pred;
                    let pdf = mvn.pdf(&error);
                    weights[i] *= pdf;
                }
            }

            self.particules = match self.resampling_scheme {
                ResamplingScheme::IID => resampling_sort(&self.particules, &weights),
                ResamplingScheme::Stratified => resampling_stratified(&self.particules, &weights),
                ResamplingScheme::Systematic => resampling_systematic(&self.particules, &weights),
            };
        }
    }

    fn gaussian_estimate(&self) -> GaussianState<T, S> {
        gaussian_estimate(&self.particules)
    }
}

impl<T: RealField + Copy, S: Dim, Z: Dim, U: Dim> ParticleFilter<T, S, Z, U>
    for GeneralParticleFilter<T, S, Z, U>
where
    DefaultAllocator: Allocator<T, S>
        + Allocator<T, Z>
        + Allocator<T, U>
        + Allocator<T, S, S>
        + Allocator<T, Z, Z>
        + Allocator<T, Z, S>
        + Allocator<T, S, U>
        + Allocator<T, U, U>
        + Allocator<T, Const<1>, S>
        + Allocator<T, Const<1>, Z>,
    Standard: Distribution<T>,
    StandardNormal: Distribution<T>,
{
    type Particle = OVector<T, S>;

    fn particles(&self) -> &Vec<OVector<T, S>> {
        &self.particules
    }
    fn particles_mut(&mut self) -> &mut Vec<OVector<T, S>> {
        &mut self.particules
    }
}

/// S : State Size, Z: Observation Size, U: Input Size
pub struct GeneralParticleFilterKnownCorrespondences<T: RealField, S: Dim, Z: Dim, U: Dim>
where
    DefaultAllocator: Allocator<T, S> + Allocator<T, S, S> + Allocator<T, Z, Z>,
{
    q: OMatrix<T, Z, Z>,
    landmarks: FxHashMap<u32, OVector<T, S>>,
    measurement_model: Box<dyn MeasurementModel<T, S, Z> + Send>,
    motion_model: Box<dyn MotionModel<T, S, Z, U> + Send>,
    pub particules: Vec<OVector<T, S>>,
}

impl<T: RealField + Copy, S: Dim, Z: Dim, U: Dim>
    GeneralParticleFilterKnownCorrespondences<T, S, Z, U>
where
    StandardNormal: Distribution<T>,
    Standard: Distribution<T>,
    DefaultAllocator:
        Allocator<T, S> + Allocator<T, S, S> + Allocator<T, Z, Z> + Allocator<T, Const<1>, S>,
{
    pub fn new(
        initial_noise: OMatrix<T, S, S>,
        q: OMatrix<T, Z, Z>,
        landmarks: FxHashMap<u32, OVector<T, S>>,
        measurement_model: Box<dyn MeasurementModel<T, S, Z> + Send>,
        motion_model: Box<dyn MotionModel<T, S, Z, U> + Send>,
        initial_state: GaussianState<T, S>,
        num_particules: usize,
    ) -> GeneralParticleFilterKnownCorrespondences<T, S, Z, U> {
        let mvn = MultiVariateNormal::new(&initial_state.x, &initial_noise).unwrap();
        let mut particules = Vec::with_capacity(num_particules);
        for _ in 0..num_particules {
            particules.push(mvn.sample());
        }

        GeneralParticleFilterKnownCorrespondences {
            q,
            landmarks,
            measurement_model,
            motion_model,
            particules,
        }
    }
}

impl<T: RealField + Copy, S: Dim, Z: Dim, U: Dim> BayesianFilterKnownCorrespondences<T, S, Z, U>
    for GeneralParticleFilterKnownCorrespondences<T, S, Z, U>
where
    StandardNormal: Distribution<T>,
    Standard: Distribution<T>,
    DefaultAllocator: Allocator<T, S>
        + Allocator<T, U>
        + Allocator<T, Z>
        + Allocator<T, S, S>
        + Allocator<T, Z, Z>
        + Allocator<T, Z, S>
        + Allocator<T, S, U>
        + Allocator<T, U, U>
        + Allocator<T, Const<1>, S>
        + Allocator<T, Const<1>, Z>,
{
    fn update_estimate(
        &mut self,
        control: Option<OVector<T, U>>,
        measurements: Option<Vec<(u32, OVector<T, Z>)>>,
        dt: T,
    ) {
        // Predicts new belief state from the past beliefs with the motion model
        if let Some(u) = control {
            self.particules = self
                .particules
                .iter()
                .map(|p| self.motion_model.sample(p, &u, dt))
                .collect();
        }

        // Samples the particles by predicting with the measurement model
        if let Some(measurements) = measurements {
            let mut weights = vec![T::one(); self.particules.len()];
            let shape = measurements[0].1.shape_generic();
            let mvn = MultiVariateNormal::new(&OMatrix::zeros_generic(shape.0, shape.1), &self.q)
                .unwrap();

            // Taking each landmark, approximate posterior with marginals
            for (id, z) in measurements
                .iter()
                .filter(|(id, _)| self.landmarks.contains_key(id))
            {
                let landmark = self.landmarks.get(id);
                for (i, particule) in self.particules.iter().enumerate() {
                    // Prediction of the landmark position at particle position
                    let z_pred = self.measurement_model.prediction(particule, landmark);
                    let error = z - z_pred;
                    let pdf = mvn.pdf(&error);
                    // Multiplying weights by each marginal
                    weights[i] *= pdf;
                }
            }
            self.particules = resampling(&self.particules, &weights);
            // self.particules = resampling_sort(&self.particules, weights);
        }
    }

    fn gaussian_estimate(&self) -> GaussianState<T, S> {
        gaussian_estimate(&self.particules)
    }
}

impl<T: RealField + Copy, S: Dim, Z: Dim, U: Dim> ParticleFilterKnownCorrespondences<T, S, Z, U>
    for GeneralParticleFilterKnownCorrespondences<T, S, Z, U>
where
    DefaultAllocator: Allocator<T, S>
        + Allocator<T, Z>
        + Allocator<T, U>
        + Allocator<T, S, S>
        + Allocator<T, Z, Z>
        + Allocator<T, U, U>
        + Allocator<T, S, U>
        + Allocator<T, Z, S>
        + Allocator<T, Const<1>, S>
        + Allocator<T, Const<1>, Z>,
    Standard: Distribution<T>,
    StandardNormal: Distribution<T>,
{
    fn particles(&self) -> &Vec<OVector<T, S>> {
        &self.particules
    }
    fn particles_mut(&mut self) -> &mut Vec<OVector<T, S>> {
        &mut self.particules
    }
}

/// Struct that contains the determinants of a given particle
#[derive(Debug, Clone)]
pub struct FastParticle<T, S>
where
    T: RealField + Copy,
    S: Dim,
    DefaultAllocator: Allocator<T, S>,
{
    pose: OVector<T, S>,
    // Features are mean, covariance, and probabilistic estimate (logarithmic)
    features: Vec<OVector<OVector<T, Const<2>>, Const<3>>>,
}

/// Simple implementation of FastSLAM v1 with EKF landmark estimation
/// TODO: Optimise data structure and implementation details
pub struct FastSlam1<T: RealField + Copy, S: Dim, Z: Dim, U: Dim>
where
    DefaultAllocator: Allocator<T, S> + Allocator<T, S, S> + Allocator<T, Z, Z>,
{
    state_noise: OMatrix<T, S, S>,
    control_noise: OMatrix<T, Z, Z>,
    measurement_model: Box<dyn MeasurementModel<T, S, Z> + Send>,
    motion_model: Box<dyn MotionModel<T, S, Z, U> + Send>,
    pub particules: Vec<FastParticle<T, S>>,
    resampling_scheme: ResamplingScheme,
}

impl<T: RealField + Copy, S: Dim, Z: Dim, U: Dim> FastSlam1<T, S, Z, U>
where
    DefaultAllocator: Allocator<T, S>
        + Allocator<T, Z>
        + Allocator<T, U>
        + Allocator<T, S, S>
        + Allocator<T, Z, Z>
        + Allocator<T, Const<1>, S>,
    StandardNormal: Distribution<T>,
{
    pub fn new(
        state_noise: OMatrix<T, S, S>,
        control_noise: OMatrix<T, Z, Z>,
        measurement_model: Box<dyn MeasurementModel<T, S, Z> + Send>,
        motion_model: Box<dyn MotionModel<T, S, Z, U> + Send>,
        initial_state: GaussianState<T, S>,
        num_particules: usize,
        resampling_scheme: ResamplingScheme,
    ) -> FastSlam1<T, S, Z, U> {
        let mvn = MultiVariateNormal::new(&initial_state.x, &state_noise).unwrap();
        let mut particules = Vec::with_capacity(num_particules);
        for _ in 0..num_particules {
            particules.push(FastParticle {
                pose: mvn.sample(),
                features: Vec::with_capacity(50),
            });
        }

        FastSlam1 {
            state_noise,
            control_noise,
            measurement_model,
            motion_model,
            particules,
            resampling_scheme,
        }
    }
}

impl<T: RealField + Copy, S: Dim, Z: Dim, U: Dim> BayesianFilter<T, S, Z, U>
    for FastSlam1<T, S, Z, U>
where
    DefaultAllocator: Allocator<T, S>
        + Allocator<T, Z>
        + Allocator<T, U>
        + Allocator<T, S, S>
        + Allocator<T, Z, Z>
        + Allocator<T, Const<1>, S>,
{
    fn update_estimate(
        &mut self,
        u: Option<OVector<T, U>>,
        z: Option<Vec<OVector<T, Z>>>,
        dt: T,
    ) {
        // Note that we do sequential updates for all z; TODO: Make efficient
        //  Generate positional noise
        let shape = self.particles()[0].pose.shape_generic();
        let mvn =
            MultiVariantNormal::new(&Omatrix::zeros_generic(shape.0, shape.1), &self.state_noise).unwrap();

        // With the motion model, use the pose of the particle with the control
        // to predict the next position
        let Some(control) = u else {
            return
        }

        self.particles_mut()
            .iter_mut()
            .map(|(pose, features)| {
                // Sample new pose
                let x_pred = self.motion_model.prediction(p.pose, &control, dt) + mvn.sample();
                if z.is_none() { return };
                let Some(measurements) = z;

                for feature in measurements {
                    // Calculates the probability of observing feature
                    // p (z_t | etc)
                    let [m, c, i] = feature;
                    let G_theta = self.measurement_model.jacobian(x_pred, Some(m));
                    let z = self.measurement_model.prediction(x_pred, Some(m)) + G_theta * ();
                }
            .collect()

        ()
    }
    fn gaussian_estimate(&self) -> GaussianState<T, S> {
        unimplemented!()
    }
}

impl<T: RealField + Copy, S: Dim, Z: Dim, U: Dim> ParticleFilter<T, S, Z, U>
    for FastSlam1<T, S, Z, U>
    where
    DefaultAllocator: Allocator<T, S>
        + Allocator<T, Z>
        + Allocator<T, U>
        + Allocator<T, S, S>
        + Allocator<T, Z, Z>
        + Allocator<T, Const<1>, S>,
{
    type Particle = FastParticle<T, S>;

    fn particles(&self) -> &Vec<Self::Particle> { &self.particules }
    fn particles_mut(&mut self) -> &mut Vec<Self::Particle> { &mut self.particules }
}

fn gaussian_estimate<T: RealField + Copy, S: Dim>(
    particules: &[OVector<T, S>],
) -> GaussianState<T, S>
where
    DefaultAllocator: Allocator<T, S> + Allocator<T, S, S> + Allocator<T, Const<1>, S>,
{
    let shape = particules[0].shape_generic();
    let x = particules
        .iter()
        .fold(OMatrix::zeros_generic(shape.0, shape.1), |a, b| a + b)
        / T::from_usize(particules.len()).unwrap();
    let cov = particules
        .iter()
        .map(|p| p - &x)
        .map(|dx| &dx * dx.transpose())
        .fold(OMatrix::zeros_generic(shape.0, shape.0), |a, b| a + b)
        / T::from_usize(particules.len()).unwrap();
    GaussianState { x, cov }
}

fn resampling<T: RealField + Copy, S: Dim>(
    particules: &Vec<OVector<T, S>>,
    weights: &[T],
) -> Vec<OVector<T, S>>
where
    DefaultAllocator: Allocator<T, S>,
    Standard: Distribution<T>,
{
    let cum_weight: Vec<T> = weights
        .iter()
        .scan(T::zero(), |state, &x| {
            *state += x;
            Some(*state)
        })
        .collect();
    let weight_tot = *cum_weight.last().unwrap();

    // sampling
    let mut rng = rand::thread_rng();
    (0..particules.len())
        .map(|_| {
            let rng_nb = rng.gen::<T>() * weight_tot;
            for (i, p) in particules.iter().enumerate() {
                if (&cum_weight)[i] > rng_nb {
                    return p.clone();
                }
            }
            unreachable!()
        })
        .collect()
}

fn resampling_sort<T: RealField + Copy, S: Dim>(
    particules: &Vec<OVector<T, S>>,
    weights: &[T],
) -> Vec<OVector<T, S>>
where
    DefaultAllocator: Allocator<T, S>,
    Standard: Distribution<T>,
{
    let total_weight: T = weights.iter().fold(T::zero(), |a, b| a + *b);
    let mut rng = rand::thread_rng();
    let mut draws: Vec<T> = (0..particules.len())
        .map(|_| rng.gen::<T>() * total_weight)
        .collect();
    resample(&mut draws, total_weight, particules, weights)
}

fn resampling_stratified<T: RealField + Copy, S: Dim>(
    particules: &Vec<OVector<T, S>>,
    weights: &[T],
) -> Vec<OVector<T, S>>
where
    DefaultAllocator: Allocator<T, S>,
    Standard: Distribution<T>,
{
    let total_weight: T = weights.iter().fold(T::zero(), |a, b| a + *b);
    let mut rng = rand::thread_rng();
    let mut draws: Vec<T> = (0..particules.len())
        .map(|i| {
            (T::from_usize(i).unwrap() + rng.gen::<T>()) / T::from_usize(particules.len()).unwrap()
                * total_weight
        })
        .collect();
    resample(&mut draws, total_weight, particules, weights)
}

fn resampling_systematic<T: RealField + Copy, S: Dim>(
    particules: &Vec<OVector<T, S>>,
    weights: &[T],
) -> Vec<OVector<T, S>>
where
    DefaultAllocator: Allocator<T, S>,
    Standard: Distribution<T>,
{
    let total_weight: T = weights.iter().fold(T::zero(), |a, b| a + *b);
    let mut rng = rand::thread_rng();
    let draw = rng.gen::<T>();
    let mut draws: Vec<T> = (0..particules.len())
        .map(|i| {
            (T::from_usize(i).unwrap() + draw) / T::from_usize(particules.len()).unwrap()
                * total_weight
        })
        .collect();
    resample(&mut draws, total_weight, particules, weights)
}

fn resample<T: RealField + Copy, S: Dim>(
    draws: &mut [T],
    total_weight: T,
    particules: &Vec<OVector<T, S>>,
    weights: &[T],
) -> Vec<OVector<T, S>>
where
    DefaultAllocator: Allocator<T, S>,
    Standard: Distribution<T>,
{
    draws.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let mut index = 0;
    let mut cum_weight = draws[0];
    (0..particules.len())
        .map(|i| {
            while cum_weight < draws[i] {
                if index == particules.len() - 1 {
                    // weird precision edge case
                    cum_weight = total_weight;
                    break;
                } else {
                    cum_weight += weights[index];
                    index += 1;
                }
            }
            particules[index].clone()
        })
        .collect()
}
