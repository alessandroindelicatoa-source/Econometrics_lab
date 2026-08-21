# Econometrics Lab

A modular Streamlit application for applied econometrics, visual analysis, robustness checks, causal models and fuzzy indicators.

## Main modules

- Data import and transformations
- Dedicated Graph Studio
- OLS/WLS with classical, HC0-HC3 or clustered covariance; Logit, Probit, Poisson, Negative Binomial, ZIP, Ordered models, Quantile Regression
- Panel FE/RE/Pooled/First Differences
- IV/2SLS and Difference-in-Differences
- ADF/KPSS, ARIMA and VAR
- Diagnostics, influence, VIF, automated specification curves
- Scenario prediction playground
- Interpretation Lab with model-aware substantive interpretation and diagnostic recommendations
- Fuzzy Likert indices and TOPSIS
- Excel, Word, PDF and interactive Plotly HTML export
- Python, R, Stata and Gretl code generator

## Graph Studio

- Histograms
- Box and violin plots
- Scatter plots with OLS/LOWESS trends
- Means by group with 95% confidence intervals
- Pearson/Spearman/Kendall heatmaps
- Scatter matrices
- Time and panel trajectories
- Missingness
- Coefficient/forest plots with confidence intervals
- Actual vs predicted
- Residual vs fitted
- Residual distributions
- Q-Q plots
- ROC/AUC
- Marginal effects
- DiD trends
- Saved-model coefficient comparison

## Local install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload all files preserving the folder structure.
3. Select `app.py` as the main file.
4. Deploy.

No API keys are required.

## Bundled demo data

- `data/panel_causal_demo.csv`
- `data/time_series_demo.csv`

## Architecture

```text
app.py
econometrics_lab/
  data_manager.py
  model_engine.py
  diagnostics.py
  plot_factory.py
  fuzzy.py
  exporting.py
  codegen.py
  utils.py
data/
.streamlit/
requirements.txt
```
