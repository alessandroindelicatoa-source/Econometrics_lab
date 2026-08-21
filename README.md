# Econometrics Lab V2

**Interactive Econometric Research Environment**

V2 reorganises the project around a professional research workflow instead of a long menu of estimators.

## Seven workspaces

1. **Workspace** — project overview, dataset intelligence and recent models
2. **Data** — import, profile, missing data, transformations and filters
3. **Explore** — distributions, relationships, groups, correlations, panel/time plots and data quality
4. **Model Studio** — unified cross-sectional, panel, causal and time-series model builder
5. **Research Lab** — model health, robustness battery, specification curves, event studies, model comparison and fuzzy methods
6. **Simulator** — two-profile prediction and counterfactual comparison
7. **Report** — regression tables, Excel/Word/PDF export and Python/R/Stata/Gretl code

## Econometric models currently wired into the UI

### Cross-sectional
- OLS
- WLS with an explicit weight variable
- Linear Probability Model
- Logit
- Probit
- Complementary Log-Log
- Poisson
- Negative Binomial
- Zero-Inflated Poisson
- Ordered Logit
- Ordered Probit
- Quantile Regression
- Classical, HC0-HC3 and clustered covariance where supported

### Panel
- Fixed Effects
- Random Effects
- Pooled OLS
- First Differences

### Causal
- IV / 2SLS
- Difference-in-Differences
- Generic Event Study

### Time series
- ADF
- KPSS
- ARIMA
- VAR

### Fuzzy / multicriteria
- Likert → TFN → defuzzified index
- TOPSIS

## Research Lab

The V2 research layer adds:

- model-health dashboard
- Breusch-Pagan, White, RESET, Jarque-Bera, Durbin-Watson, Breusch-Godfrey
- VIF
- Cook's-distance influence screening
- robustness battery:
  - classical baseline
  - HC1
  - HC3
  - clustered covariance
  - fixed-effect controls
  - 1% winsorisation
  - influential-observation exclusion
- automated specification curve
- event-study plot with confidence intervals
- saved-model coefficient comparison

## Publication graphics

The sidebar controls a global graphics theme and target size:

- Minimal
- Economics journal
- APA
- Presentation
- Dark presentation

Sizes:
- Single column
- Double column
- 16:9
- Square

All relevant plots can be exported as interactive HTML and, when Kaleido is available, PNG/SVG/PDF.

## Deployment

Upload the **contents** of this folder to the root of a GitHub repository.

The root should contain:

```text
app.py
requirements.txt
README.md
DEPLOY.md
.streamlit/
data/
econometrics_lab/
```

In Streamlit Community Cloud choose `app.py` as the main file.

No external API key is required.

The bundled panel demo contains `relative_time = year - 2022` for immediate Event Study testing.
