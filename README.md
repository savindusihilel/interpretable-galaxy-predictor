# Interpretable Galaxy Property Predictor (POC)

**Physics-Informed Neural Networks with Explainable Uncertainty**

This project presents a proof-of-concept system that predicts:

- Stellar mass (log M★)
- Star formation rate (log SFR)
- Quenching probability (Q)

from broadband photometry using a **Physics-Informed Neural Network (PINN)** combined with a **conditional normalizing flow**.

Unlike standard black-box regressors, this framework produces **physically meaningful uncertainty**, reflecting genuine degeneracies in galaxy evolution rather than purely statistical noise.

---

## Key Features

- Physics-informed constraints on galaxy scaling relations
- Probabilistic quenching model using conditional normalizing flows
- Comparison with Random Forest baselines
- Interactive Streamlit demo with posterior visualization
- Designed for interpretability and scientific reasoning

---

## Demo

The Streamlit application allows users to:
- Select physically calibrated galaxy presets
- Enter custom photometric measurements
- Visualize posterior distributions for quenching probability
- Compare black-box ML vs physics-informed uncertainty

---

## Repository Structure

├── app.py # Streamlit application  
├── models.py # PINN architecture  
├── flow_utils.py # Conditional normalizing flow  
├── priors.json # Learned astrophysical priors  
├── scaler.joblib # Feature scaler (not committed)  
├── rf_mass.joblib # Random Forest baseline (optional)  
├── rf_sfr.joblib # Random Forest baseline (optional)  

> Trained model weights are excluded from version control.

---

## Scientific Motivation

Galaxy properties such as mass, star formation rate, and quenching state are often **degenerate** when inferred from photometry alone.

This work demonstrates how:
- Uncertainty can be physically interpretable
- Ambiguity can be informative
- Physics-informed learning improves trustworthiness

---

## Author

Savindu Sihilel  
Department of Computer Science
Sri Lanka Institute of Information Technology (SLIIT)
Malabe, Sri Lanka LK