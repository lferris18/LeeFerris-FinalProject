use std::error::Error;
mod functions; 

use functions::read_csv;
use functions::normalize;
use functions::linear_regression;
use functions::coefficient_of_determination;
use functions::mean_absolute_error;
use functions::mean_squared_error;

fn main() -> Result<(), Box<dyn Error>> {
    let file_path = "cleaned_df.csv";

    let (bedrooms, prices) = read_csv(file_path)?;

    println!("Filtered Data: {} rows", bedrooms.len());
    if bedrooms.is_empty() || prices.is_empty() {
        eprintln!("No valid data available for regression.");
        return Ok(());
    }

    let (normalized_bedrooms, bedrooms_mean, bedrooms_std) = normalize(&bedrooms);
    let (normalized_prices, prices_mean, prices_std) = normalize(&prices);

    let learning_rate = 0.01;
    let iterations = 10000;

    let (slope_norm, intercept_norm) = linear_regression(
        &normalized_bedrooms,
        &normalized_prices,
        learning_rate,
        iterations,
    );

    let slope = slope_norm * prices_std / bedrooms_std;
    let intercept = intercept_norm * prices_std + prices_mean - slope * bedrooms_mean;

    println!("Linear Regression Result:");
    println!("Slope (m): {}", slope);
    println!("Intercept (b): {}", intercept);

    let r_squared = coefficient_of_determination(&bedrooms, &prices, slope, intercept);
    println!("Coefficient of Determination (R²): {}", r_squared);

    let mse = mean_squared_error(&bedrooms, &prices, slope, intercept);
    println!("Mean Squared Error (MSE): {}", mse);

    let mae = mean_absolute_error(&bedrooms, &prices, slope, intercept);
    println!("Mean Absolute Error (MAE): {}", mae);

    Ok(())
}

#[test]
    fn test_normalize() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (normalized, mean, std) = normalize(&data);

        let expected_mean = 3.0;
        let epsilon = 1e-5;
        assert!((mean - expected_mean).abs() < epsilon);

        let expected_std = 1.414213562;
        assert!((std - expected_std).abs() < epsilon);

        let normalized_mean: f64 = normalized.iter().sum::<f64>() / normalized.len() as f64;
        let normalized_std: f64 = (normalized.iter().map(|&x| (x - normalized_mean).powi(2)).sum::<f64>() / normalized.len() as f64).sqrt();
        assert!((normalized_mean).abs() < epsilon);
        assert!((normalized_std - 1.0).abs() < epsilon);
    }

#[test]
fn test_linear_regression() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![3.0, 5.0, 7.0, 9.0, 11.0];
    
    let (m, b) = linear_regression(&x, &y, 0.01, 10000);
    
    let expected_slope = 2.0;
    let expected_intercept = 1.0;
    

    let epsilon = 1e-2; 
    
    assert!((m - expected_slope).abs() < epsilon, "Expected slope: {}, but got: {}", expected_slope, m);
    assert!((b - expected_intercept).abs() < epsilon, "Expected intercept: {}, but got: {}", expected_intercept, b);
}
    
#[test]
fn test_coefficient_of_determination() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let m = 2.0;
    let b = 0.0;

    let r_squared = coefficient_of_determination(&x, &y, m, b);

    assert!((r_squared - 1.0).abs() < 1e-5);
}

#[test]
fn test_mean_squared_error() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let m = 2.0;
    let b = 0.0;

    let mse = mean_squared_error(&x, &y, m, b);

    assert!(mse.abs() < 1e-5);
}

#[test]
fn test_mean_absolute_error() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let m = 2.0;
    let b = 0.0;

    let mae = mean_absolute_error(&x, &y, m, b);

    assert!(mae.abs() < 1e-5);
}