use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector, RealField};

use crate::utils::state::GaussianState;

/// Positional Filter based on the assumption of conditional independence
/// T: Used for units in math  - Odometry input such as angular velocity
/// S: Dimension of robot pose - In 2d will be (x, y, theta) = 3
/// Z: Dimension of environmental data - Individual beams etc.,
///                                     (Range, bearing) = 2,
/// U: Dimension of control data - With IMU (forward, angular velocity) = 2
///
/// This trait maintains an interface for using any kind of Bayesian-based
/// filter with two functions, update_estimate and gaussian_estimate.
/// The robot's belief can be represented in any way deemed efficacious
pub trait BayesianFilter<T: RealField, S: Dim, Z: Dim, U: Dim>
where
    DefaultAllocator: Allocator<T, S> + Allocator<T, U> + Allocator<T, Z> + Allocator<T, S, S>,
{
    /// Use the current control and environmental data inputs to generate the
    /// next robot pose x_n with the state transition distribution
    /// p(x_t|x_{t-1},u) and measurement distribution p(z_t|x_t, etc.)
    fn update_estimate(
        &mut self,
        control: Option<OVector<T, U>>,
        measurements: Option<Vec<OVector<T, Z>>>,
        dt: T,
    );

    /// Generate the best guess according to the current state of the robot's
    /// belief state. Not guaranteed to be idempotent.
    fn gaussian_estimate(&self) -> GaussianState<T, S>;
}

pub trait BayesianFilterKnownCorrespondences<T: RealField, S: Dim, Z: Dim, U: Dim>
where
    DefaultAllocator: Allocator<T, S> + Allocator<T, U> + Allocator<T, Z> + Allocator<T, S, S>,
{
    fn update_estimate(
        &mut self,
        control: Option<OVector<T, U>>,
        measurements: Option<Vec<(u32, OVector<T, Z>)>>,
        dt: T,
    );

    fn gaussian_estimate(&self) -> GaussianState<T, S>;
}
