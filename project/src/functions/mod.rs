use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

//My first function reads over a csv file and will take the 5th column (# of bedrooms)
//and the 13the column (Listed Price). It will skip any row that doesn't have 13 columns.
//It iterates over every line and makes sure that there is a value in the 4th and 13th
//columns. If not, then it will skip over that row, as it will not contribute. It also makes
//sure that both of the values in those columns are of floating point type. If not, it will 
//also skip over those rows.
pub fn read_csv(file_path: &str) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let mut bedrooms = Vec::new();
    let mut prices = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        if i == 0 {
            continue;
        }
        let line = line?;
        let cols: Vec<&str> = line.split(',').collect();

        if cols.len() < 14 {
            eprintln!("Skipping row {}: Not enough columns", i + 1);
            continue;
        }

        let bedroom = cols[5].trim();
        let price = cols[13].trim();

        if bedroom.is_empty() || price.is_empty() {
            eprintln!("Skipping row {}: Empty values in columns", i + 1);
            continue;
        }

        match (bedroom.parse::<f64>(), price.parse::<f64>()) {
            (Ok(b), Ok(p)) => {
                bedrooms.push(b);
                prices.push(p);
            }
            _ => {
                eprintln!("Skipping row {}: Invalid numeric values", i + 1);
                continue;
            }
        }
    }

    Ok((bedrooms, prices))
}

//My next function normalizes the data to have a mean of 0 and a standard deviation of 1.
pub fn normalize(data: &Vec<f64>) -> (Vec<f64>, f64, f64) {
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let std = (data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64).sqrt();

    let normalized_data = data.iter().map(|&x| (x - mean) / std).collect();
    (normalized_data, mean, std)
}

//My linear regression function starts by setting the slope and intercept at 0. It then finds the
//gradients of the slope and intercept to determine how they should change to minimize the error.
pub fn linear_regression(
    x: &Vec<f64>,
    y: &Vec<f64>,
    learning_rate: f64,
    iterations: usize,
) -> (f64, f64) {
    let mut m = 0.0;
    let mut b = 0.0;
    let n = x.len() as f64;

    for _ in 0..iterations {
        let mut m_gradient = 0.0;
        let mut b_gradient = 0.0;

        for (xi, yi) in x.iter().zip(y.iter()) {
            let prediction = m * xi + b;
            m_gradient += -2.0 * xi * (yi - prediction);
            b_gradient += -2.0 * (yi - prediction);
        }

        m -= learning_rate * (m_gradient / n);
        b -= learning_rate * (b_gradient / n);
    }

    (m, b)
}

//my function for the coefficient of determination calculates it by subtracting  
//the residual sum of the squares over the total sum of squares from 1
pub fn coefficient_of_determination(x: &Vec<f64>, y: &Vec<f64>, m: f64, b: f64) -> f64 {
    let y_mean = y.iter().sum::<f64>() / y.len() as f64;

    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| (yi - (m * xi + b)).powi(2)).sum();

    1.0 - (ss_res / ss_tot)
}

//my next function calculates the MSE, which evaluates the accuracy of the linear regression.
//It is calculated through a closure that iterates over the data points and calculates the squared
//difference in the predicted value and the actual value. If the result is low, then there is little error, 
//if it is high, then there is lots of error
pub fn mean_squared_error(x: &Vec<f64>, y: &Vec<f64>, m: f64, b: f64) -> f64 {
    let n = x.len() as f64;
    x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (yi - (m * xi + b)).powi(2))
        .sum::<f64>()
        / n
}

//My final function calculates the MAE. It takes the absolute value of the difference. This is more resistant to outliers
//than the MSE
pub fn mean_absolute_error(x: &Vec<f64>, y: &Vec<f64>, m: f64, b: f64) -> f64 {
    let n = x.len() as f64;
    x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (yi - (m * xi + b)).abs())
        .sum::<f64>()
        / n
}