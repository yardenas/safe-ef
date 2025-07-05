# Safe-EF: Error Feedback with Safety Constraints for Federated Learning ⚡

**Efficient and Safe Federated Learning!**

Federated learning (FL) enables decentralized training across multiple devices while preserving data privacy. However, communication bottlenecks severely hinder its efficiency due to the high-dimensional nature of model updates. While contractive compressors (e.g., Top-$K$) can alleviate this issue, they often degrade performance without proper handling. **Safe-EF** tackles this challenge by integrating **error feedback (EF) with safety constraints**, ensuring both efficiency and robustness in federated learning.

### Key Idea 🌐
- **Communication-Efficient Learning**: Employs contractive compression (e.g., Top-$K$) to reduce communication costs.
- **Error Feedback for Stability**: Mitigates the adverse effects of compression to maintain convergence speed and accuracy.
- **Safety Constraints**: Ensures feasible updates, even in non-smooth convex settings, making it practical for real-world applications.
- **Scalability & Robustness**: Extends to stochastic settings and large-scale federated learning setups.


## Requirements 🛠

- **Python** • Version 3.10+
- **pip** • Python package installer

## Installation 📝

Get started with Safe-EF in just a few steps:

### Using pip

1. Clone the repository:
   ```bash
   git clone https://github.com/anon/safe-ef.git
   cd safe-ef
   ```
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -e .
   ```

### Using Poetry

1. Clone the repository:
   ```bash
   git clone https://github.com/anon/safe-ef.git
   cd safe-ef
   ```
2. Install dependencies and create a virtual environment with Poetry:
   ```bash
   poetry install
   ```
3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

## Usage 🔧

Run the training script with:
```bash
python train_brax.py --help
```
This will display all available options and configurations.

Check out our ICML paper:
```
@inproceedings{islamovsafe,
  title={Safe-EF: Error Feedback for Non-smooth Constrained Optimization},
  author={Islamov, Rustem and As, Yarden and Fatkhullin, Ilyas},
  booktitle={Forty-second International Conference on Machine Learning}
}
```
