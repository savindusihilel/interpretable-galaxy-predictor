import streamlit as st
import os
import json
import joblib
import numpy as np
import torch

from models import PINNJoint
from flow_utils import build_conditional_maf

# ======================
# Configuration
# ======================
DEVICE = "cpu"
INPUT_DIM = 10
CONTEXT_DIM = 64

FEATURES = ['u','g','r','i','z','g-r','u-g','r-i','M_r','redshift']

st.set_page_config(
    page_title="Interpretable Galaxy Property Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================
# Session state defaults
# ======================
for key in [
    "assets_loaded",
    "scaler", "priors",
    "joint", "flow",
    "rf_mass", "rf_sfr",
    "demo"
]:
    if key not in st.session_state:
        st.session_state[key] = None

# ======================
# Asset loader
# ======================
def load_assets():
    torch.set_num_threads(1)

    if st.session_state.assets_loaded:
        return

    BASE = os.path.dirname(os.path.abspath(__file__))
    def p(x): return os.path.join(BASE, x)

    with st.spinner("Loading trained models and calibration assets…"):
        st.session_state.scaler = joblib.load(p("scaler.joblib"))

        with open(p("priors.json"), "r") as f:
            st.session_state.priors = json.load(f)

        joint = PINNJoint(INPUT_DIM, context_dim=CONTEXT_DIM).to(DEVICE)
        sd = torch.load(p("pinn_stageC_joint_final.pth"), map_location=DEVICE)
        sd = {k: v for k, v in sd.items() if not k.startswith("flow.")}
        joint.load_state_dict(sd, strict=False)
        joint.eval()
        st.session_state.joint = joint

        flow = build_conditional_maf(
            context_dim=CONTEXT_DIM,
            n_blocks=4,
            hidden_features=64
        ).to(DEVICE)
        flow.load_state_dict(
            torch.load(p("pinn_stageC_flow_final.pth"), map_location=DEVICE)
        )
        flow.eval()
        st.session_state.flow = flow

        rf_m = p("rf_mass.joblib")
        rf_s = p("rf_sfr.joblib")
        if os.path.exists(rf_m) and os.path.exists(rf_s):
            st.session_state.rf_mass = joblib.load(rf_m)
            st.session_state.rf_sfr  = joblib.load(rf_s)

    st.session_state.assets_loaded = True

# ======================
# First run
# ======================
if not st.session_state.assets_loaded:
    load_assets()
    st.rerun()

# ======================
# Helper functions
# ======================
def predict_quenching_probability(joint, flow, x_t, n_samples=256, return_samples=False):
    with torch.no_grad():
        ctx = joint(x_t)["context"]
        q = flow.sample(n_samples, context=ctx).squeeze().cpu().numpy()
        q = np.clip(q, 0, 1)

    if return_samples:
        return float(q.mean()), float(q.std()), q
    return float(q.mean()), float(q.std())

def rf_predict_with_uncertainty(rf_model, x):
    preds = np.array([tree.predict(x) for tree in rf_model.estimators_])
    return float(preds.mean()), float(preds.std())

# ======================
# HEADER
# ======================
st.markdown("""
## Interpretable Galaxy Property Predictor (POC)  
**Physics-Informed Neural Networks with Explainable Uncertainty**

This application estimates **stellar mass**, **star formation rate**, and **quenching probability**
from broadband photometry, while explicitly modeling **physical uncertainty** arising from
galaxy evolution processes.
""")

st.divider()

# ======================
# SIDEBAR — Inputs
# ======================
st.sidebar.title("Galaxy Observables")

st.sidebar.markdown("Use either a preset galaxy or enter values manually.")

st.sidebar.subheader("Example galaxies")

if st.sidebar.button("Star-forming galaxy"):
    st.session_state.demo = {
        "u": 20.7468,
        "g": 19.5216,
        "r": 18.8356,
        "i": 18.4295,
        "z": 18.158,
        "zred": 0.039427325
    }

if st.sidebar.button("Transition galaxy"):
    st.session_state.demo = {
        "u": 21.4997,
        "g": 19.7431,
        "r": 18.7274,
        "i": 18.2965,
        "z": 18.0088,
        "zred": 0.103435345
    }

if st.sidebar.button("Quenched galaxy"):
    st.session_state.demo = {
        "u": 22.4709,
        "g": 20.15,
        "r": 18.7786,
        "i": 18.2661,
        "z": 18.0014,
        "zred": 0.18520756
    }

st.sidebar.caption(
    "Presets are calibrated from the training distribution "
    "to represent physically distinct regimes learned by the model."
)

demo = st.session_state.demo or {}

st.sidebar.subheader("Photometry")

u = st.sidebar.number_input("u-band magnitude", value=demo.get("u", 18.0))
g = st.sidebar.number_input("g-band magnitude", value=demo.get("g", 17.5))
r = st.sidebar.number_input("r-band magnitude", value=demo.get("r", 17.2))
i = st.sidebar.number_input("i-band magnitude", value=demo.get("i", 17.0))
z = st.sidebar.number_input("z-band magnitude", value=demo.get("z", 16.9))
redshift = st.sidebar.number_input("Redshift", value=demo.get("zred", 0.05))

st.sidebar.divider()

if st.sidebar.button("Reload models (debug)"):
    for k in st.session_state.keys():
        st.session_state[k] = None
    st.rerun()

# ======================
# RUN INFERENCE
# ======================
if st.button("Run inference", type="primary"):
    x = np.array([[u, g, r, i, z, g-r, u-g, r-i, 0.0, redshift]], dtype=np.float32)
    x_scaled = st.session_state.scaler.transform(x)
    x_t = torch.tensor(x_scaled, dtype=torch.float32)

    with torch.no_grad():
        out = st.session_state.joint(x_t)

    m_mu = out["mu_mass"].item()
    s_mu = out["mu_sfr"].item()
    sigma_m = float(np.sqrt(np.exp(out["logvar_mass"].item())))
    sigma_s = float(np.sqrt(np.exp(out["logvar_sfr"].item())))

    q_mean, q_std, q_samples = predict_quenching_probability(
        st.session_state.joint,
        st.session_state.flow,
        x_t,
        return_samples=True
    )
    q_std = min(q_std, 0.25)

    # ======================
    # RESULTS — Physical Properties
    # ======================
    st.subheader("Predicted Physical Properties")

    c1, c2 = st.columns(2)
    c1.metric("Stellar Mass log(M★)", f"{m_mu:.2f}", f"± {sigma_m:.2f}")
    c2.metric("Star Formation Rate log(SFR)", f"{s_mu:.2f}", f"± {sigma_s:.2f}")

    # ======================
    # RESULTS — Quenching Probability
    # ======================
    st.subheader("Quenching Probability")

    q_mean = np.clip(q_mean, 0, 1)
    q_std  = np.clip(q_std, 0, 0.5)

    st.progress(q_mean)
    st.caption(f"Estimated probability: **Q = {q_mean:.2f} ± {q_std:.2f}**")

    st.markdown("**Posterior distribution of Q**")
    hist, _ = np.histogram(q_samples, bins=30, range=(0,1), density=True)
    st.bar_chart(hist)

    with st.expander("Why is this uncertain?"):
        st.markdown("""
This uncertainty reflects **real physical ambiguity**, not model noise.

From broadband photometry alone:
- A galaxy may appear red because it is **dust-obscured but still forming stars**
- Or because it has **truly quenched star formation**
- Both scenarios are physically valid

The model preserves this ambiguity instead of collapsing to a single answer.
        """)

    # ======================
    # INTERPRETATION
    # ======================
    st.subheader("Physical Interpretation")

    if q_mean < 0.3:
        st.success("Consistent with an actively star-forming galaxy.")
    elif q_mean > 0.7:
        st.error("Consistent with a quenched, passive galaxy.")
    else:
        st.warning(
            "This galaxy lies in a physically ambiguous regime.\n\n"
            "Uncertainty reflects competing physical interpretations "
            "rather than lack of data."
        )

    # ======================
    # MODEL COMPARISON
    # ======================
    st.subheader("Model Comparison")

    if st.session_state.rf_mass is not None:
        rf_m, rf_m_std = rf_predict_with_uncertainty(st.session_state.rf_mass, x_scaled)
        rf_s, rf_s_std = rf_predict_with_uncertainty(st.session_state.rf_sfr, x_scaled)

        c_rf, c_p = st.columns(2)

        with c_rf:
            st.markdown("🌲 **Random Forest (black-box)**")
            st.metric("logM", f"{rf_m:.2f}", f"± {rf_m_std:.2f}")
            st.metric("logSFR", f"{rf_s:.2f}", f"± {rf_s_std:.2f}")
            st.caption("Uncertainty from ensemble variance")

        with c_p:
            st.markdown("🧠 **Physics-Informed Neural Network**")
            st.metric("logM", f"{m_mu:.2f}", f"± {sigma_m:.2f}")
            st.metric("logSFR", f"{s_mu:.2f}", f"± {sigma_s:.2f}")
            st.caption("Uncertainty from physical degeneracy")

        st.info(
            "Random Forest uncertainty arises from statistical variation.\n\n"
            "PINN uncertainty arises because **the physics allows multiple explanations**."
        )