import numpy as np
import os
import glob
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score

class ROMTrainer:
    def __init__(self, data_dir="mock_data", model_dir="models", model_type="mlp"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.model_type = model_type.lower()
        os.makedirs(self.model_dir, exist_ok=True)
        
    def _get_model(self):
        """Factory to return the requested model instance."""
        if self.model_type == "mlp":
            return MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)
        elif self.model_type == "linear":
            return LinearRegression()
        elif self.model_type == "ridge":
            return Ridge()
        elif self.model_type == "lasso":
            return Lasso()
        elif self.model_type == "random_forest":
            return RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        elif self.model_type == "knn":
            return KNeighborsRegressor(n_neighbors=5)
        elif self.model_type == "svr":
            # SVR doesn't support multi-output naturally, wrap it
            return MultiOutputRegressor(SVR(kernel='rbf'))
        elif self.model_type == "gradient_boosting":
            # GB doesn't support multi-output naturally, wrap it
            return MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def load_data(self):
        print("Loading data...")
        param_files = sorted(glob.glob(os.path.join(self.data_dir, "*_params.npy")))
        disp_files = sorted(glob.glob(os.path.join(self.data_dir, "*_disp.npy")))
        stress_files = sorted(glob.glob(os.path.join(self.data_dir, "*_stress.npy")))
        
        if not param_files:
            raise FileNotFoundError("No data found. Run generate_dataset.py first.")
            
        X = []
        Y_disp = []
        Y_stress = []
        
        for p_f, d_f, s_f in zip(param_files, disp_files, stress_files):
            X.append(np.load(p_f))
            Y_disp.append(np.load(d_f).flatten()) # Flatten for basic regression
            Y_stress.append(np.load(s_f).flatten())
            
        return np.array(X), np.array(Y_disp), np.array(Y_stress)

    def train(self):
        print(f"Training with model: {self.model_type.upper()}")
        X, Y_disp, Y_stress = self.load_data()
        
        # Concatenate targets for single model or train separate?
        # Let's train separate models for proper scaling
        print(f"Data shape: X={X.shape}, Y_disp={Y_disp.shape}")
        
        # 1. Train Displacement Model
        print("Training Displacement Model...")
        X_train, X_test, y_train, y_test = train_test_split(X, Y_disp, test_size=0.2, random_state=42)
        
        scaler_x = StandardScaler()
        X_train_s = scaler_x.fit_transform(X_train)
        X_test_s = scaler_x.transform(X_test)
        
        # Get Model
        model_disp = self._get_model()
        model_disp.fit(X_train_s, y_train)
        
        score = model_disp.score(X_test_s, y_test)
        print(f"Displacement Model R2: {score}")
        
        # Save
        joblib.dump(model_disp, os.path.join(self.model_dir, "rom_disp.pkl"))
        joblib.dump(scaler_x, os.path.join(self.model_dir, "scaler_x.pkl"))
        
        # 2. Train Stress Model
        # Using same X scaler
        print("Training Stress Model...")
        y_stress_train, y_stress_test = train_test_split(Y_stress, test_size=0.2, random_state=42)
        
        model_stress = self._get_model()
        model_stress.fit(X_train_s, y_stress_train)
        print(f"Stress Model R2: {model_stress.score(X_test_s, y_stress_test)}")
        
        joblib.dump(model_stress, os.path.join(self.model_dir, "rom_stress.pkl"))
        
        print("Training complete.")

if __name__ == "__main__":
    trainer = ROMTrainer()
    trainer.train()
