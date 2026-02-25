import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import qmc
import itertools
import os

from src.data_generator import MockFEASolver

def generate_dataset(
    n_samples=100,
    output_dir="mock_data",
    sampling="taguchi",      # "random" | "lhs" | "sobol" | "taguchi"
    taguchi_levels=5,     # Only used if sampling="taguchi"
    seed=42
):
    output_dir = os.path.join(output_dir, sampling)

    if not os.path.exists(output_dir) or len(os.listdir(output_dir)) == 0:
        print("Generating Data...")
    else:
        print("Data found. Skipping generation")
        return
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    solver = MockFEASolver(output_dir=output_dir)
    print(f"Generating {n_samples} samples in '{output_dir}' using {sampling} sampling...")

    # Parameter ranges (min, max)
    param_ranges = {
        "length": (5.0, 20.0),
        "width": (1.0, 3.0),
        "depth": (1.0, 3.0),
        "load": (-500.0, 500.0)
    }

    param_names = list(param_ranges.keys())
    bounds = np.array([param_ranges[p] for p in param_names])
    dim = len(param_names)

    # -------------------------------------------------
    # Sampling strategies
    # -------------------------------------------------
    if sampling == "random":
        samples_unit = np.random.rand(n_samples, dim)

    elif sampling == "lhs":
        sampler = qmc.LatinHypercube(d=dim, seed=seed)
        samples_unit = sampler.random(n=n_samples)

    elif sampling == "sobol":
        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
        m = int(np.ceil(np.log2(n_samples)))
        samples_unit = sampler.random_base2(m=m)
        samples_unit = samples_unit[:n_samples]

    elif sampling == "taguchi":
        levels = taguchi_levels
        level_values = []

        for low, high in bounds:
            level_values.append(np.linspace(low, high, levels))

        grid = list(itertools.product(*level_values))
        samples_scaled = np.array(grid)

        if len(samples_scaled) > n_samples:
            samples_scaled = samples_scaled[:n_samples]

        samples_unit = None

    else:
        raise ValueError("Unknown sampling method.")

    # Scale to physical range (except taguchi which is already scaled)
    if sampling in ["random", "lhs", "sobol"]:
        lower = bounds[:, 0]
        upper = bounds[:, 1]
        samples_scaled = qmc.scale(samples_unit, lower, upper)

    # -------------------------------------------------
    # Save parameter table to CSV
    # -------------------------------------------------
    df = pd.DataFrame(samples_scaled, columns=param_names)
    df.insert(0, "sample_id", np.arange(len(df)))

    csv_path = os.path.join(output_dir, "design_table.csv")
    df.to_csv(csv_path, index=False)

    print(f"Design table saved to: {csv_path}")

    # -------------------------------------------------
    # Run solver
    # -------------------------------------------------
    for i, sample in tqdm(df.iterrows(), total=len(df)):
        solver.solve(
            length=sample["length"],
            width=sample["width"],
            depth=sample["depth"],
            load=sample["load"],
            sample_id=int(sample["sample_id"])
        )

    print("Dataset generation complete.")

if __name__ == "__main__":
    generate_dataset()
