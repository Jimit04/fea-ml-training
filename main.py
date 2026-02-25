"""Command-line entry point for the FEA Reduced-Order Model workflow.

Orchestrates three pipeline stages — data generation, model training, and
visualisation — which can be invoked individually or as a complete end-to-end
pipeline via the CLI flags.
"""

import os
import argparse


def main():
    """Parse CLI arguments and run the requested pipeline stages.

    Stages
    ------
    1. **Generate** — Create synthetic FEA samples using ``MockFEASolver``.
    2. **Train**    — Train a TensorFlow/Keras ROM (MLP or GCN).
    3. **Visualize** — Render predicted vs. ground-truth fields with PyVista.

    If no stage flags are provided, all three stages run sequentially.
    """
    parser = argparse.ArgumentParser(description="FEA ROM Workflow POC")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic data")
    parser.add_argument("--train", action="store_true", help="Train ROM model")
    parser.add_argument("--visualize", action="store_true", help="Visualize results")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples to generate")
    parser.add_argument("--sampling", type=str, default="lhc", choices=["random", "lhs", "sobol", "taguchi"], help="Sampling strategy for data generation")
    parser.add_argument("--screenshot", type=str, default=None, help="Save screenshot to file instead of showing window")
    parser.add_argument("--model", type=str, default="gcn", choices=["mlp", "gcn"], help="Model type to train")
    
    args = parser.parse_args()
    
    # If no stage flags given, run the full pipeline
    if not (args.generate or args.train or args.visualize):
        print("No action specified. Running full workflow: Generate -> Train -> Visualize")
        run_full = True
    else:
        run_full = False
        
    # Stage 1: Generate data
    if args.generate or run_full:
        from src.generate_dataset import generate_dataset
        generate_dataset(n_samples=args.samples, sampling=args.sampling)

            
    # Stage 2: Train model
    if args.train or run_full:
        from src.rom_model import ROMTrainer
        print(f"Training Model ({args.model})...")
        trainer = ROMTrainer(model_type=args.model, data_dir=os.path.join("mock_data", args.sampling), model_dir=os.path.join("models", args.sampling))
        trainer.train()
            
    # Stage 3: Visualise predictions vs ground truth
    if args.visualize or run_full:
        from src.visualizer import ROMVisualizer
        print("Launching Visualizer...")
        viz = ROMVisualizer(model_dir=os.path.join("models", args.sampling))
        # Static demo with fixed parameters
        l, w, d, load = 12.5, 1.5, 1.5, 250.0
        print(f"Predicting for L={l}, W={w}, Depth={d}, Load={load}")
        viz.predict_and_plot(l, w, d, load, screenshot=args.screenshot)

if __name__ == "__main__":
    main()
