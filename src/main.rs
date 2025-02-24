use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    optim::{Adam, AdamConfig},
    tensor::{backend::Backend, Data, Shape, Tensor},
};
use burn_ndarray::NdArray;
use rand::Rng;
use textplots::{Chart, Plot, Shape as PlotShape}; // Fix conflict with burn::Shape

type BackendImpl = NdArray<f32>; // Define the backend for computations

#[derive(Module, Debug, Clone)]
struct LinearRegression {
    linear: Linear<BackendImpl>,
}

impl LinearRegression {
    fn new() -> Self {
        let device = <BackendImpl as Backend>::Device::default();
        Self {
            linear: LinearConfig::new(1, 1).init(&device), // Initialize with device
        }
    }

    fn forward(&self, x: Tensor<BackendImpl, 2>) -> Tensor<BackendImpl, 2> {
        self.linear.forward(x)
    }
}

// Function to generate synthetic (x, y) data
fn generate_data(samples: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = rand::thread_rng();
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();

    for _ in 0..samples {
        let x = rng.gen_range(-10.0..10.0);
        let noise: f32 = rng.gen_range(-1.0..1.0);
        let y = 2.0 * x + 1.0 + noise; // Linear relation with noise
        x_data.push(x);
        y_data.push(y);
    }
    (x_data, y_data)
}

fn main() {
    let samples = 100;
    let (x_train, y_train) = generate_data(samples);

    // Convert data to tensors
    let x_tensor = Tensor::<BackendImpl, 2>::from_data(
        Data::from(x_train.iter().map(|x| vec![*x]).collect::<Vec<_>>()),
        Shape::new([samples, 1]), // Define tensor shape
    );

    let y_tensor = Tensor::<BackendImpl, 2>::from_data(
        Data::from(y_train.iter().map(|y| vec![*y]).collect::<Vec<_>>()),
        Shape::new([samples, 1]), // Define tensor shape
    );

    let mut model = LinearRegression::new(); // Create model
    let mut optimizer = AdamConfig::new().init(); // Initialize optimizer

    let epochs = 100;
    for epoch in 0..epochs {
        let y_pred = model.forward(x_tensor.clone());
        let loss = (y_pred - y_tensor.clone())
            .powf_scalar(2.0)
            .mean(); // Compute Mean Squared Error

        optimizer.update_module(&mut model, loss.backward()); // Update model weights

        if epoch % 10 == 0 {
            println!(
                "Epoch {}: Loss = {:?}",
                epoch,
                loss.to_data().convert::<f32>().to_vec() // Convert loss to Vec<f32>
            );
        }
    }

    // Test the trained model
    let test_x: Vec<f32> = (-100..100).map(|x| x as f32 / 10.0).collect();
    let test_x_tensor = Tensor::<BackendImpl, 2>::from_data(
        Data::from(test_x.iter().map(|x| vec![*x]).collect::<Vec<_>>()),
        Shape::new([test_x.len(), 1]), // Define shape
    );
    let test_y_tensor = model.forward(test_x_tensor);

    // Extract data for plotting
    let test_y = test_y_tensor.to_data().convert::<f32>().to_vec();

    // Plot the regression line
    println!("\nLinear Regression Fit:");
    let points: Vec<(f32, f32)> = test_x.into_iter().zip(test_y.into_iter()).collect();
    Chart::new(100, 30, -10.0, 10.0)
        .lineplot(&PlotShape::Lines(&points)) // Corrected textplots Shape usage
        .display();
}
