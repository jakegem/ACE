extern crate ndarray;
extern crate ndarray_linalg;


pub mod filters {
    use ndarray_linalg::Inverse;
    use ndarray::{ArrayBase, Array2, Dim, OwnedRepr};

    pub struct KalmanFilter {
        state: Array2<f64>,
        state_covariance: Array2<f64>,
        process_model: Array2<f64>,
        process_noise: Array2<f64>,
        observation_model: Array2<f64>,
        observation_noise: Array2<f64>,
    }

    impl KalmanFilter {
        pub fn new(
            state: Array2<f64>,
            state_covariance: Array2<f64>,
            process_model: Array2<f64>,
            process_noise: Array2<f64>,
            observation_model: Array2<f64>,
            observation_noise: Array2<f64>,
        ) -> Self {
            KalmanFilter {
                state,
                state_covariance,
                process_model,
                process_noise,
                observation_model,
                observation_noise,
            }
        }

        pub fn predict(&mut self) {
            self.state = self.process_model.dot(&self.state);
            self.state_covariance = self.process_model.dot(&self.state_covariance).dot(&self.process_model.t()) + &self.process_noise;

        }

        pub fn update(&mut self, observation: Array2<f64>) {
            let innovation: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = 
                observation - self.observation_model.dot(&self.state);
            let innovation_covariance: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = 
                self.observation_model.dot(&self.state_covariance).dot(&self.observation_model.t()) + &self.observation_noise;
            let kalman_gain: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = 
                self.state_covariance.dot(&self.observation_model.t()).dot(&innovation_covariance.inv().unwrap());
            self.state = &self.state + kalman_gain.dot(&innovation);
            self.state_covariance = (Array2::<f64>::eye(self.state_covariance.shape()[0]) - kalman_gain.dot(&self.observation_model)).dot(&self.state_covariance);
        }

        pub fn get_state(&self) -> Array2<f64> {
            return self.state.clone();
        }

        pub fn get_covariance(&self) -> Array2<f64> {
            return self.state_covariance.clone();
        }
    }

}