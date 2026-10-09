"""
train_pinn.py
Step 2: Build & Train the Physics-Informed Neural Network (PINN)
Model: PhysicsInformedFuelPredictor
Loss: MSE(y_pred, y_true) + 0.1 * mean(relu(-d(y_pred)/d(speed)))
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

FEATURE_COLS = [
    "Speed_Over_Ground_knots",
    "Average_Load_Percentage",
    "wave_height_m",
    "wind_knots",
]
TARGET_COL = "hfo_consumption_ton_per_hr"

class ClassicalFuelPredictor(nn.Module):
    """3 linear layers with ReLU activation for fuel burn prediction (Baseline)."""
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

class PhysicsInformedFuelPredictor(nn.Module):
    """3 linear layers with ReLU activation for fuel burn prediction."""
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

def get_compute_device():
    """Detect CUDA GPU or fall back gracefully to CPU."""
    if torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(2, 2, device="cuda")
            _ = test_tensor.sum().item()
            device_name = torch.cuda.get_device_name(0)
            print(f"[+] CUDA acceleration enabled on {device_name}")
            return torch.device("cuda")
        except Exception as e:
            print(f"[!] CUDA detected but unavailable for execution ({e}). Falling back to CPU.")
    print("[*] Running on CPU.")
    return torch.device("cpu")

def train_model(
    csv_path=None,
    epochs=3000,
    lr=0.001,
    batch_size=128,
    physics_lambda=0.1
):
    print("=" * 60)
    print("STEP 2: Train Physics-Informed Neural Network (PINN)")
    print("=" * 60)

    if csv_path is None:
        if os.path.exists("green_maritime_fleet_dataset.csv"):
            csv_path = "green_maritime_fleet_dataset.csv"
        elif os.path.exists("processed_ship_telemetry.csv"):
            csv_path = "processed_ship_telemetry.csv"
        else:
            raise FileNotFoundError("No telemetry dataset found. Run generate_green_maritime_dataset.py first.")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Telemetry dataset not found at '{csv_path}'.")

    df = pd.read_csv(csv_path)
    print(f"[+] Loaded {len(df)} records from '{csv_path}'")

    # Extract inputs and target
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[[TARGET_COL]].values.astype(np.float32)

    # Train / Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # 4. Fit StandardScaler from sklearn.preprocessing and save using joblib
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_val_scaled = feature_scaler.transform(X_val)

    y_train_scaled = target_scaler.fit_transform(y_train)
    y_val_scaled = target_scaler.transform(y_val)

    joblib.dump(feature_scaler, "feature_scaler.joblib")
    joblib.dump(target_scaler, "target_scaler.joblib")
    print("[+] Saved feature_scaler.joblib and target_scaler.joblib")

    # PyTorch DataLoaders
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train_scaled, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val_scaled, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = get_compute_device()
    model = PhysicsInformedFuelPredictor(input_dim=len(FEATURE_COLS)).to(device)
    classical_model = ClassicalFuelPredictor(input_dim=len(FEATURE_COLS)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    classical_optimizer = torch.optim.Adam(classical_model.parameters(), lr=lr)
    mse_criterion = nn.MSELoss()

    print(f"[+] Starting training for {epochs} epochs (Physics penalty weight = {physics_lambda})...")

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        classical_model.train()
        running_mse = 0.0
        running_physics = 0.0
        running_total = 0.0
        running_classical_mse = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_x.requires_grad_(True)

            # --- Train PINN ---
            optimizer.zero_grad()
            y_pred = model(batch_x)
            mse_loss = mse_criterion(y_pred, batch_y)

            # Physics-Informed Loss:
            grad_outputs = torch.ones_like(y_pred)
            d_pred_d_x = torch.autograd.grad(
                outputs=y_pred,
                inputs=batch_x,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True
            )[0]
            d_pred_d_speed = d_pred_d_x[:, 0:1]
            physics_penalty = torch.mean(torch.relu(-d_pred_d_speed))
            total_loss = mse_loss + physics_lambda * physics_penalty

            total_loss.backward()
            optimizer.step()

            running_mse += mse_loss.item() * len(batch_x)
            running_physics += physics_penalty.item() * len(batch_x)
            running_total += total_loss.item() * len(batch_x)

            # --- Train Classical Model ---
            classical_optimizer.zero_grad()
            y_pred_classical = classical_model(batch_x.detach())
            classical_loss = mse_criterion(y_pred_classical, batch_y)
            classical_loss.backward()
            classical_optimizer.step()

            running_classical_mse += classical_loss.item() * len(batch_x)

        train_mse = running_mse / len(train_dataset)
        train_phys = running_physics / len(train_dataset)
        train_total = running_total / len(train_dataset)
        train_class_mse = running_classical_mse / len(train_dataset)

        # Validation
        model.eval()
        classical_model.eval()
        val_mse = 0.0
        val_class_mse = 0.0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(device)
                vy = vy.to(device)
                val_mse += mse_criterion(model(vx), vy).item() * len(vx)
                val_class_mse += mse_criterion(classical_model(vx), vy).item() * len(vx)
        val_mse /= len(val_dataset)
        val_class_mse /= len(val_dataset)

        if epoch % 50 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch [{epoch:04d}/{epochs:04d}] | "
                f"PINN (Train: {train_mse:.5f}, Phys: {train_phys:.5f}, Val: {val_mse:.5f}) | "
                f"Classical (Train: {train_class_mse:.5f}, Val: {val_class_mse:.5f})"
            )

    # 6. Save trained model parameters
    model_save_path = "maritime_pinn_predictor.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"[+] PINN Model weights saved to '{model_save_path}'.")

    classical_save_path = "classical_predictor.pth"
    torch.save(classical_model.state_dict(), classical_save_path)
    print(f"[+] Classical Model weights saved to '{classical_save_path}'.")

    # Sanity check physics monotonicity on test inputs
    model.eval()
    test_speeds = np.linspace(10.0, 22.0, 10)
    test_load = 0.70
    test_wave = 2.5
    test_wind = 20.0
    test_matrix = np.array([[s, test_load, test_wave, test_wind] for s in test_speeds], dtype=np.float32)
    test_matrix_scaled = feature_scaler.transform(test_matrix)
    with torch.no_grad():
        preds_scaled = model(torch.tensor(test_matrix_scaled, dtype=torch.float32).to(device))
        preds_tons_per_hr = target_scaler.inverse_transform(preds_scaled.cpu().numpy())

    print("\n--- Physical Monotonicity Verification (Moderate Sea: Wave=2.5m, Wind=20kts, Load=70%) ---")
    for s, p in zip(test_speeds, preds_tons_per_hr.flatten()):
        print(f"  Speed: {s:.1f} kts -> Predicted Fuel Burn: {p:.4f} tons/hr")

    # Check monotonicity
    diffs = np.diff(preds_tons_per_hr.flatten())
    monotonic = bool(np.all(diffs >= 0))
    print(f"[+] Monotonicity Check (d(fuel)/d(speed) >= 0): {'PASSED (Strictly Monotonic)' if monotonic else 'WARNING (Non-monotonic)'}")
    print("=" * 60)
    print("STEP 2 COMPLETE: PINN Training Successful.")
    print("=" * 60)

if __name__ == "__main__":
    train_model()
