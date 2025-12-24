"""
Extract calibrated Streamlit demo presets directly from
the cleaned galaxy CSV used to train the PINN.
"""

import numpy as np
import pandas as pd
import torch
import joblib
import json

from models import PINNJoint
from flow_utils import build_conditional_maf

# ----------------------
# CONFIG
# ----------------------
DEVICE = "cpu"
INPUT_DIM = 10
CONTEXT_DIM = 64
N_FLOW_SAMPLES = 64

CSV_PATH = "mpa_dr7_clean.csv"   # <-- CHANGE THIS to your actual file

FEATURES = [
    "u", "g", "r", "i", "z",
    "g-r", "u-g", "r-i",
    "M_r", "redshift"
]

# ----------------------
# Load models & scaler
# ----------------------
print("Loading scaler...")
scaler = joblib.load("scaler.joblib")

print("Loading PINN joint model...")
joint = PINNJoint(INPUT_DIM, context_dim=CONTEXT_DIM).to(DEVICE)
sd = torch.load("pinn_stageC_joint_final.pth", map_location=DEVICE)
sd = {k: v for k, v in sd.items() if not k.startswith("flow.")}
joint.load_state_dict(sd, strict=False)
joint.eval()

print("Loading conditional flow...")
flow = build_conditional_maf(
    context_dim=CONTEXT_DIM,
    n_blocks=4,
    hidden_features=64
).to(DEVICE)
flow.load_state_dict(
    torch.load("pinn_stageC_flow_final.pth", map_location=DEVICE)
)
flow.eval()

# ----------------------
# Load dataset
# ----------------------
print(f"Loading dataset: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

X_raw = df[FEATURES].values
X_scaled = scaler.transform(X_raw)

# ----------------------
# Compute quenching probability
# ----------------------
def compute_q(x_scaled):
    with torch.no_grad():
        x_t = torch.tensor(x_scaled[None, :], dtype=torch.float32)
        ctx = joint(x_t)["context"]
        q = flow.sample(N_FLOW_SAMPLES, context=ctx)
        q = q.squeeze().cpu().numpy()
        q = np.clip(q, 0.0, 1.0)
    return float(q.mean())

print("Computing quenching probabilities...")
Q_vals = np.array([compute_q(X_scaled[i]) for i in range(len(X_scaled))])

print(
    f"Q distribution: min={Q_vals.min():.2f}, "
    f"median={np.median(Q_vals):.2f}, "
    f"max={Q_vals.max():.2f}"
)

# ----------------------
# Define physical regimes
# ----------------------
sf_idx   = np.where(Q_vals < 0.2)[0]
tran_idx = np.where((Q_vals > 0.4) & (Q_vals < 0.6))[0]
q_idx    = np.where(Q_vals > 0.8)[0]

assert len(sf_idx) > 0
assert len(tran_idx) > 0
assert len(q_idx) > 0

# ----------------------
# Pick representative examples
# ----------------------
def pick(indices):
    qs = Q_vals[indices]
    target = np.median(qs)
    return indices[np.argmin(np.abs(qs - target))]

i_sf   = pick(sf_idx)
i_tran = pick(tran_idx)
i_q    = pick(q_idx)

# ----------------------
# Build Streamlit presets
# ----------------------
def preset(i):
    return {
        "u": float(X_raw[i, 0]),
        "g": float(X_raw[i, 1]),
        "r": float(X_raw[i, 2]),
        "i": float(X_raw[i, 3]),
        "z": float(X_raw[i, 4]),
        "zred": float(X_raw[i, 9]),
    }

print("\n==============================")
print("CALIBRATED STREAMLIT PRESETS")
print("==============================\n")

print("Star-forming galaxy:")
print(preset(i_sf), f"Q ≈ {Q_vals[i_sf]:.2f}\n")

print("Transition galaxy:")
print(preset(i_tran), f"Q ≈ {Q_vals[i_tran]:.2f}\n")

print("Quenched galaxy:")
print(preset(i_q), f"Q ≈ {Q_vals[i_q]:.2f}\n")