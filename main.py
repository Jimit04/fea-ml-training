#%%
from pathlib import Path

data_dir = Path("mock_data")
model_dir = Path("models")
model_type = "transformer"
# samplings=["random", "lhs", "sobol", "taguchi"]
samplings=["random"]

#%%
# Generate datasets for each sampling strategy
from src.generate_dataset import generate_dataset
for sampling in samplings:
    print(f"Generating dataset for {sampling} sampling...")
    generate_dataset(n_samples=600, sampling=sampling, 
                     output_dir=data_dir/sampling, taguchi_levels=4)

#%%
# Visualise input data as 3D point cloud for one of the datasets
# from src.input_visualizer import PointCloudVisualizer
# for sampling in samplings:
#     print(f"Visualizing input data for {sampling} sampling...")
#     viz=PointCloudVisualizer(csv_path=data_dir/sampling/"design_table.csv")
#     viz.load_data()
#     viz.plot(symmetric_color_scale=True)

#%%
# Train models for each sampling strategy
from src.rom_model import ROMTrainer
for sampling in samplings:
    print(f"Training model for {sampling} sampling...")
    trainer = ROMTrainer(model_type=model_type, data_dir=data_dir/sampling, model_dir=model_dir/model_type/sampling)
    trainer.train()

#%%
# Visualise predictions vs ground truth for each model
from src.visualizer import ROMVisualizer
Path(f"screenshots").mkdir(exist_ok=True)
for sampling in samplings:
    print(f"Visualizing model for {sampling} sampling...")
    viz = ROMVisualizer(model_dir=model_dir/model_type/sampling)
    l, w, d, load = 180, 25, 12, 400.0
    viz.predict_and_plot(l, w, d, load, screenshot=Path(f"screenshots/predictions_{sampling}.png"))

# %%
# Play here with different input parameters to see how predictions change
