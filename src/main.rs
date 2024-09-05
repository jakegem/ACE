extern crate plotters;
extern crate rand;

use plotters::prelude::*;
use rand::Rng;
use std::f64::consts::PI;
use ndarray::{array, ArrayBase, Dim, OwnedRepr};

use ace::filters::KalmanFilter;

fn main() {
    let p: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = array![[1.0, 0.0], [0.0, 1.0]];
    let f: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = array![[1.0, 1.0], [0.0, 1.0]];
    let x: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = array![[0.0], [0.0]];
    let q: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = array![[1e-4, 0.0], [0.0, 1e-4]];
    let h: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = array![[1.0, 0.0]];
    let r: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = array![[0.01]];
    let mut kf: KalmanFilter = KalmanFilter::new(x, p, f, q, h, r);

    
    let rate = 60.0; // Hz
    let duration = 5.0; // seconds
    let num_measurements = (rate * duration) as usize;
    let dt = 1.0 / rate;

    // Initial conditions
    let mut position = 0.0;
    let initial_velocity = 1.0;
    let noise_std_dev = 0.1;

    // Generate measurements
    let mut rng = rand::thread_rng();
    let mut measurements = Vec::with_capacity(num_measurements);
    for i in 0..num_measurements {
        let t = i as f64 * dt;
        let velocity = initial_velocity + (2.0 * PI * t).sin();
        position += velocity * dt;
        let noisy_measurement = position + rng.gen_range(-noise_std_dev..noise_std_dev);
        measurements.push(noisy_measurement);
    }
    
    let mut predictions: Vec<f64> = Vec::new();
    let mut upper_bounds: Vec<f64> = Vec::with_capacity(num_measurements);
    let mut lower_bounds: Vec<f64> = Vec::with_capacity(num_measurements);

    for &measurement in &measurements {
        kf.predict();
        predictions.push(kf.get_state()[[0, 0]]);
        
        let state_covariance = kf.get_covariance();
        let position_variance = state_covariance[(0, 0)];
        let sigma = position_variance.sqrt();
        
        upper_bounds.push(kf.get_state()[[0, 0]] + 3.0 * sigma);
        lower_bounds.push(kf.get_state()[[0, 0]] - 3.0 * sigma);
        println!("sigma: {}", sigma);

        kf.update(array![[measurement]]);
    }
    
    // Plotting
    let root_area = BitMapBackend::new("kalman_filter.png", (1600, 1200)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root_area)
        .caption("Kalman Filter Results", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..measurements.len(), 0.0..6.0)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    let x_data: Vec<usize> = (0..measurements.len()).collect();
    let y_measurements: Vec<f64> = measurements.into_iter().collect();
    let y_predictions: Vec<f64> = predictions.into_iter().collect();
    let y_upper_bounds: Vec<f64> = upper_bounds.iter().cloned().collect();
    let y_lower_bounds: Vec<f64> = lower_bounds.iter().cloned().collect();

    chart
    .draw_series(LineSeries::new(
        x_data.iter().zip(y_measurements.iter()).map(|(&x, &y)| (x, y)),
        &RED,
    ))
    .unwrap()
    .label("Measurements")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .draw_series(LineSeries::new(
            x_data.iter().zip(y_predictions.iter()).map(|(&x, &y)| (x, y)),
            &BLUE,
        ))
        .unwrap()
        .label("Predictions")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .draw_series(LineSeries::new(
            x_data.iter().zip(y_upper_bounds.iter()).map(|(&x, &y)| (x, y)),
            &GREEN,
        ))
        .unwrap()
        .label("Upper 3-Sigma Bound")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    chart
        .draw_series(LineSeries::new(
            x_data.iter().zip(y_lower_bounds.iter()).map(|(&x, &y)| (x, y)),
            &GREEN,
        ))
        .unwrap()
        .label("Lower 3-Sigma Bound")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();
}
