import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class PointCloudVisualizer:
    """
    Visualizes CSV-based parametric data as a 3D point cloud.
    
    Required columns:
        - length
        - width
        - depth
        - load
    """

    REQUIRED_COLUMNS = ["length", "width", "depth", "load"]

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None

    def load_data(self):
        """Load CSV file and validate required columns."""
        self.df = pd.read_csv(self.csv_path)

        missing = [col for col in self.REQUIRED_COLUMNS if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def plot(self, symmetric_color_scale=False):
        """
        Plot 3D point cloud.
        
        Parameters
        ----------
        symmetric_color_scale : bool
            If True, color scale will be symmetric about zero.
            Useful when 'load' contains positive and negative values.
        """
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        x = self.df["length"]
        y = self.df["width"]
        z = self.df["depth"]
        c = self.df["load"]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        if symmetric_color_scale:
            max_abs = np.max(np.abs(c))
            sc = ax.scatter(x, y, z, c=c, vmin=-max_abs, vmax=max_abs)
        else:
            sc = ax.scatter(x, y, z, c=c)

        ax.set_xlabel("Length")
        ax.set_ylabel("Width")
        ax.set_zlabel("Depth")

        cbar = plt.colorbar(sc)
        cbar.set_label("Load")

        plt.title(f"3D Point Cloud for {self.csv_path}")
        plt.show()