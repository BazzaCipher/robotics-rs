#![allow(non_snake_case)]
use nalgebra::{RealField, SMatrix, SVector};
use std::collections::HashMap;

use crate::models::measurement::MeasurementModel;
use crate::models::motion::MotionModel;
use crate::utils::state::GaussianStateStatic;

/// S : State Size, Z: Observation Size, U: Input Size
pub struct ExtendedKalmanFilter<T: RealField, const S: usize, const Z: usize, const U: usize> {
    R: SMatrix<T, S, S>,
    Q: SMatrix<T, Z, Z>,
    measurement_model: Box<dyn MeasurementModel<T, S, Z>>,
    motion_model: Box<dyn MotionModel<T, S, Z, U>>,
}

impl<T: RealField, const S: usize, const Z: usize, const U: usize>
    ExtendedKalmanFilter<T, S, Z, U>
{
    pub fn new(
        R: SMatrix<T, S, S>,
        Q: SMatrix<T, Z, Z>,
        measurement_model: Box<dyn MeasurementModel<T, S, Z>>,
        motion_model: Box<dyn MotionModel<T, S, Z, U>>,
    ) -> ExtendedKalmanFilter<T, S, Z, U> {
        ExtendedKalmanFilter {
            R,
            Q,
            measurement_model,
            motion_model,
        }
    }

    pub fn estimate(
        &self,
        // model: &impl ExtendedKalmanFilterModel<T, S, Z, U>,
        estimate: &GaussianStateStatic<T, S>,
        u: &SVector<T, U>,
        z: &SVector<T, Z>,
        dt: T,
    ) -> GaussianStateStatic<T, S> {
        // predict
        let G = self
            .motion_model
            .jacobian_wrt_state(&estimate.x, u, dt.clone());
        let x_pred = self.motion_model.prediction(&estimate.x, u, dt);
        let p_pred = &G * &estimate.P * G.transpose() + &self.R;

        // update
        let H = self.measurement_model.jacobian(&x_pred, None);
        let z_pred = self.measurement_model.prediction(&x_pred, None);

        let s = &H * &p_pred * H.transpose() + &self.Q;
        let kalman_gain = &p_pred * H.transpose() * s.try_inverse().unwrap();
        let x_est = &x_pred + &kalman_gain * (z - z_pred);
        let p_est = (SMatrix::<T, S, S>::identity() - kalman_gain * H) * &p_pred;
        GaussianStateStatic { x: x_est, P: p_est }
    }
}

/// S : State Size, Z: Observation Size, U: Input Size
pub struct ExtendedKalmanFilterKnownCorrespondences<
    T: RealField,
    const S: usize,
    const Z: usize,
    const U: usize,
> {
    R: SMatrix<T, S, S>,
    Q: SMatrix<T, Z, Z>,
    landmarks: HashMap<i32, SVector<T, Z>>,
    measurement_model: Box<dyn MeasurementModel<T, S, Z>>,
    motion_model: Box<dyn MotionModel<T, S, Z, U>>,
}

impl<T: RealField, const S: usize, const Z: usize, const U: usize>
    ExtendedKalmanFilterKnownCorrespondences<T, S, Z, U>
{
    pub fn new(
        R: SMatrix<T, S, S>,
        Q: SMatrix<T, Z, Z>,
        measurement_model: Box<dyn MeasurementModel<T, S, Z>>,
        motion_model: Box<dyn MotionModel<T, S, Z, U>>,
    ) -> ExtendedKalmanFilterKnownCorrespondences<T, S, Z, U> {
        ExtendedKalmanFilterKnownCorrespondences {
            Q,
            R,
            landmarks: HashMap::new(),
            measurement_model,
            motion_model,
        }
    }

    pub fn estimate(
        &self,
        estimate: &GaussianStateStatic<T, S>,
        u: &SVector<T, U>,
        z_vec: &[(i32, SVector<T, Z>)],
        dt: T,
    ) -> GaussianStateStatic<T, S> {
        // predict
        let G = self
            .motion_model
            .jacobian_wrt_state(&estimate.x, u, dt.clone());

        // fixed version
        let mut x_est = self.motion_model.prediction(&estimate.x, u, dt);
        let mut p_est = &G * &estimate.P * G.transpose() + &self.R;

        // version with adjustable R
        // let V = model.jacobian_motion_model_wrt_input(&estimate.x, u, dt.clone());
        // let M = model.cov_control_model(u, dt.clone());
        // let mut p_est = &G * &estimate.P * G.transpose() + &V * M * V.transpose();

        // update / correction step
        for (id, z) in z_vec
            .iter()
            .filter(|(id, _v)| self.landmarks.contains_key(id))
        {
            let landmark = self.landmarks.get(id).unwrap();

            let z_pred = self.measurement_model.prediction(&x_est, Some(landmark));
            let H = self.measurement_model.jacobian(&x_est, Some(landmark));
            let s = &H * &p_est * H.transpose() + &self.Q;
            let kalman_gain = &p_est * H.transpose() * s.try_inverse().unwrap();
            x_est += &kalman_gain * (z - z_pred);
            p_est = (SMatrix::<T, S, S>::identity() - kalman_gain * H) * &p_est
        }
        GaussianStateStatic { x: x_est, P: p_est }
    }
}

#[cfg(test)]
mod tests {
    extern crate test;
    use crate::localization::extended_kalman_filter::ExtendedKalmanFilter;
    use crate::models::measurement::SimpleProblemMeasurementModel;
    use crate::models::motion::SimpleProblemMotionModel;
    use crate::utils::deg2rad;
    use crate::utils::state::GaussianStateStatic as GaussianState;
    use nalgebra::{Matrix4, Vector2, Vector4};
    use test::{black_box, Bencher};

    #[bench]
    fn ekf(b: &mut Bencher) {
        // setup ukf
        let q = Matrix4::<f32>::from_diagonal(&Vector4::new(0.1, 0.1, deg2rad(1.0), 1.0));
        let r = nalgebra::Matrix2::identity();
        let motion_model = Box::new(SimpleProblemMotionModel {});
        let measurement_model = Box::new(SimpleProblemMeasurementModel {});
        let ekf = ExtendedKalmanFilter::<f32, 4, 2, 2>::new(q, r, measurement_model, motion_model);

        let dt = 0.1;
        let u: Vector2<f32> = Default::default();
        let kalman_state = GaussianState {
            x: Vector4::<f32>::new(0., 0., 0., 0.),
            P: Matrix4::<f32>::identity(),
        };
        let z: Vector2<f32> = Default::default();

        b.iter(|| {
            black_box(ekf.estimate(&kalman_state, &u, &z, dt));
        });
    }
}
