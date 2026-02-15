# ROM Mock FEA POC

This repository demonstrates a **Reduced Order Model (ROM)** workflow for Finite Element Analysis (FEA) using Python.

## Workflow
1.  **Data Generation**: `MockFEASolver` (using PyVista/Numpy) simulates a cantilever beam (Displacement/Stress) based on Geometry and Load parameters.
2.  **Training**: A Neural Network (`MLPRegressor`) learns the mapping from Parameters -> Field Results.
3.  **Visualization**: `PyVista` visualizes the "Ground Truth" vs "ROM Prediction" side-by-side.

## Setup

Project is managed with `uv`.

```bash
uv sync
```

## Usage

Run the full workflow:
```bash
uv run python main.py
```

Arguments:
- `--generate`: Force data generation.
- `--train`: Force training.
- `--visualize`: Run visualization.
- `--samples N`: Number of samples (default 50).

## Project Structure
- `src/data_generator.py`: Synthetic FEA solver.
- `src/generate_dataset.py`: Batch generator.
- `src/rom_model.py`: ML Training logic.
- `src/visualizer.py`: Prediction and 3D Plotting.
