import os
import argparse
from src.generate_dataset import generate_dataset
from src.rom_model import ROMTrainer
from src.visualizer import ROMVisualizer

def main():
    parser = argparse.ArgumentParser(description="FEA ROM Workflow POC")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic data")
    parser.add_argument("--train", action="store_true", help="Train ROM model")
    parser.add_argument("--visualize", action="store_true", help="Visualize results")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples to generate")
    parser.add_argument("--screenshot", type=str, default=None, help="Save screenshot to file instead of showing window")
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "gcn"], help="Model type to train")
    
    args = parser.parse_args()
    
    # Defaults: if nothing specified, run full pipeline if needed
    if not (args.generate or args.train or args.visualize):
        print("No action specified. Running full workflow: Generate -> Train -> Visualize")
        run_full = True
    else:
        run_full = False
        
    # 1. Generate Data
    if args.generate or run_full:
        if not os.path.exists("mock_data") or len(os.listdir("mock_data")) == 0:
            print("Generating Data...")
            generate_dataset(n_samples=args.samples)
        else:
            print("Data found. Skipping generation (use --generate to force).")
            
    # 2. Train Model
    if args.train or run_full:
        print(f"Training Model ({args.model})...")
        trainer = ROMTrainer(model_type=args.model)
        trainer.train()
            
    # 3. Visualize
    if args.visualize or run_full:
        print("Launching Visualizer...")
        viz = ROMVisualizer()
        # Interactive loop or static demo?
        # Let's do a static demo with random params
        l, w, d, load = 20.0, 3.0, 1.0, 100.0 
        print(f"Predicting for L={l}, W={w}, Depth={d}, Load={load}")
        viz.predict_and_plot(l, w, d, load, screenshot=args.screenshot)

if __name__ == "__main__":
    main()
