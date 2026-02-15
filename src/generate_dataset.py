import numpy as np
import os
from src.data_generator import MockFEASolver
from tqdm import tqdm

def generate_dataset(n_samples=100, output_dir="mock_data"):
    solver = MockFEASolver(output_dir=output_dir)
    print(f"Generating {n_samples} samples in '{output_dir}'...")
    
    # Ranges
    # (min, max)
    l_range = (5.0, 20.0)
    w_range = (1.0, 3.0)
    d_range = (1.0, 3.0)
    load_range = (100.0, 500.0) # Increased load for steel
    
    for i in tqdm(range(n_samples)):
        l = np.random.uniform(*l_range)
        w = np.random.uniform(*w_range)
        d = np.random.uniform(*d_range)
        load = np.random.uniform(*load_range)
        
        solver.solve(length=l, width=w, depth=d, load=load, sample_id=i)
        
    print("Dataset generation complete.")

if __name__ == "__main__":
    generate_dataset()
