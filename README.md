# GWR Causal Inference

A Streamlit web app that evaluates **geographically weighted regression (GWR)** as a causal inference model, comparing it against causal forest and propensity score matching on spatial datasets.

Built as part of a research project investigating whether GWR can estimate treatment effects in spatially structured observational data.

---

## What it does

1. **Upload** any spatial dataset as a CSV
2. **Configure** which columns represent the outcome, coordinates, and covariates
3. **Generate** semi-synthetic counterfactuals using the paper's data generating process (DGP), which simulates a spatially confounded treatment assignment and known potential outcomes
4. **Run** all three models and compare them on:
   - **ATE error** — how accurately each model estimates the average treatment effect
   - **PEHE** — how accurately it estimates individual-level treatment effects (lower is better)
5. **Predict** outcomes or treatment effects for new observations using the best model

---

## Background

The fundamental problem of causal inference is that we can only ever observe one potential outcome per observation — the one that actually occurred. To evaluate a causal model, we need to know both.

This app addresses that by taking a real spatial dataset and applying a semi-synthetic DGP (Heiss [2022], adapted by S. Hawkes) that:

- Constructs a spatially structured confounder `U` from the outcome and coordinates
- Assigns treatment probabilistically via a logistic model involving covariates, spatial terms, and `U`
- Generates both potential outcomes `Y0` and `Y1` with a known treatment effect `τ = 0.8`

Because the original data is spatial and the treatment assignment and confounding are both spatial, the resulting dataset preserves the spatial structure that GWR is designed to exploit.

---

## Models

| Model | Approach |
|---|---|
| **GWR** | Fits a local linear regression at each point, weighted by spatial proximity. Estimates ITE by intervening on the treatment variable and comparing predictions. |
| **Causal forest** | Fits separate random forests for treated and untreated groups, estimates ITE as the difference in predictions. |
| **Propensity score** | Estimates treatment probability via logistic regression, uses IPW for ATE and nearest-neighbour matching for ITE. |

---

## Getting started

**Requirements:** Python 3.9+

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Run the app:**

```bash
streamlit run app.py
```

The app will open in your browser automatically at `http://localhost:8501`.

---

## Dataset format

Your CSV should have at least four columns:

| Column type | Description |
|---|---|
| Outcome | The variable you want to analyse (e.g. crime rate, house price) |
| Longitude | East-west coordinate |
| Latitude | North-south coordinate |
| Covariates | One or more explanatory variables |

Column order does not matter — you assign roles in the app. The app has been tested on US county-level datasets with 2,000–5,000 observations and 6–13 covariates.

**Example datasets used in the paper:** Crime, Depression, HIV, Housing, Mental Health — each containing rates of their respective outcome across US regions with geographic covariates.

---

## Project structure

```
├── app.py              # Streamlit interface
├── preprocess.py       # Semi-synthetic data generation (DGP)
├── data.py             # Data loading and splitting for GWR
├── GWR.py              # GCV bandwidth selection
├── GWR_utilities.py    # Kernel weights, cache build, distance matrix
├── spatial.py          # Outlier removal, heteroskedastic adjustment
├── causal.py           # GWR causal statistics (ATE error, PEHE)
├── causal_forest.py    # Causal forest model
├── propensity.py       # Propensity score model
└── requirements.txt
```

---

## How the DGP works

Given a raw spatial dataset with outcome `y`, coordinates `(lon, lat)`, and covariates `x1, x2, ..., xp`:

1. All variables are standardised
2. A spatial confounder is constructed:

$$U_i = 0.45 y_i + 0.20 \sin(1.5 \cdot \text{lat}_i)\cos(1.2 \cdot \text{lon}_i) + \eta_i, \quad \eta_i \sim \mathcal{N}(0, 0.6)$$

3. Treatment is assigned via a logistic model:

$$\text{logit}(p_i) = 0.90 y_i + 0.35 x_1 + 0.25 x_2 - 0.20 x_3 + 0.25 x_1 x_2 + 0.35 \sin(1.1 \cdot \text{lat}_i) - 0.25 \cos(1.0 \cdot \text{lon}_i) + 0.65 U_i$$

4. Potential outcomes are generated with a known treatment effect `τ = 0.8`:

$$f_i = y_i + 0.30 x_1 + 0.20 x_2^2 + 0.15 x_3 + 0.25 \sin(1.2 \cdot \text{lat}_i)\cos(0.8 \cdot \text{lon}_i) + 0.35 U_i$$

$$Y_i^0 = f_i + \varepsilon_i, \quad Y_i^1 = f_i + \tau + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, 0.6)$$

---

## Metrics

**ATE error** — absolute difference between the estimated and true average treatment effect:

$$\text{ATE error} = |\widehat{\text{ATE}} - \text{ATE}|, \quad \text{ATE} = \mathbb{E}[Y^1 - Y^0] = \tau = 0.8$$

**PEHE** — precision in estimation of heterogeneous effects, measuring individual-level accuracy:

$$\text{PEHE} = \frac{1}{n} \sum_{i=1}^{n} \left(\hat{\tau}_i - \tau_i\right)^2$$

---

## Reference

Heiss, A. (2022). *Causal inference with observational data and unobserved confounding*. Used as the basis for the semi-synthetic DGP. Adapted for spatial data by S. Hawkes.
