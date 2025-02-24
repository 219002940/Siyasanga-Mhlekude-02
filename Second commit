# Linear Regression with Burn in Rust

## Introduction
This project implements a simple linear regression model using the [Burn](https://github.com/burn-rs/burn) deep learning framework in Rust. The goal is to train a model that predicts values based on a linear relationship between input and output data. However, I faced challenges running the code successfully, and this document outlines the setup process, my approach, encountered issues, and reflections.

## Setup and Running the Code
### Prerequisites
- Install [Rust](https://www.rust-lang.org/tools/install)
- Install [Cargo](https://doc.rust-lang.org/cargo/)
- Add the required dependencies to `Cargo.toml`:
  ```toml
  [dependencies]
  burn = "0.16"
  burn-ndarray = "0.16"
  rand = "0.8"
  textplots = "0.8"
  ```

### Steps to Run the Project
1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd <project-directory>
   ```
2. Compile and run the code:
   ```sh
   cargo run
   ```

### Issues Faced
Despite setting up the dependencies correctly, the code failed to run due to:
- **Shape and Data Handling Errors:**
  - The Burn library's `Shape` module was causing type mismatches.
  - `Data::from()` was not correctly converting `Vec<Vec<f32>>` into a tensor-compatible format.
- **Optimizer Issues:**
  - The Adam optimizer required correct initialization and updating syntax.
- **Tensor Operations and Backend Compatibility:**
  - Backend operations in Burn differ from other frameworks, and correctly defining tensor operations was challenging.

## Approach
1. **Data Generation**
   - Used `rand` to generate synthetic linear data with noise.
   - Converted data into tensors with explicitly defined shapes.
2. **Model Implementation**
   - Built a simple `LinearRegression` struct with a `Linear` layer.
   - Forward pass mapped input `x` to output `y`.
3. **Training Loop**
   - Used mean squared error as the loss function.
   - Updated the model using Adam optimizer.
   - Printed loss values every 10 epochs.
4. **Evaluation and Visualization**
   - Predicted values were plotted using `textplots`.

## Results and Evaluation
Since I could not successfully run the code, no numerical results or plots were generated. However, based on theoretical expectations, a well-trained model should have predicted values close to `y = 2x + 1` with minimal error.

## Reflection on Learning Process
### AI and Documentation Assistance
- I heavily relied on AI tools, official Burn documentation, and Rust forums.
- AI helped correct syntax errors, but running the code required deep debugging.

### Lessons Learned
- **Understanding Rust Type System**: Rust’s strict type system required precise handling of data structures.
- **Deep Learning with Burn**: Unlike PyTorch or TensorFlow, Burn’s API demands a strong understanding of backends and tensor operations.
- **Debugging in Rust**: Error messages were helpful, but fixing type mismatches required careful reading of documentation.

### Next Steps
- Investigate Burn’s backend compatibility and tensor operations further.
- Try alternative Rust ML frameworks like `tch-rs` (PyTorch bindings for Rust).
- Seek help from the Burn community for debugging support.



