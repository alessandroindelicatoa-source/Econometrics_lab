from __future__ import annotations
import numpy as np
import pandas as pd

def _sig_phrase(p):
    if pd.isna(p):
        return "with no reported p-value"
    if p < .01:
        return f"highly statistically significant (p={p:.3g})"
    if p < .05:
        return f"statistically significant at 5% (p={p:.3g})"
    if p < .10:
        return f"marginally significant at 10% (p={p:.3g})"
    return f"not statistically significant at conventional levels (p={p:.3g})"

def interpret_model(model, focal=None):
    lines = []
    lines.append(f"### {model.name}")
    lines.append(f"**Model:** {model.family}.  **Specification:** `{model.formula}`.")
    if model.metrics:
        bits=[]
        for k in ["N","R²","Adj. R²","Pseudo R²","AIC","BIC"]:
            if k in model.metrics and np.isfinite(model.metrics[k]):
                v=model.metrics[k]
                bits.append(f"{k} = {v:.3f}" if k!="N" else f"N = {int(v):,}")
        if bits:
            lines.append("**Fit:** " + "; ".join(bits) + ".")

    tab=model.coef_table.copy()
    if focal is None:
        candidates=tab[~tab["term"].str.lower().isin(["intercept","const"])]
        if len(candidates):
            focal=candidates.iloc[0]["term"]

    if focal is not None:
        row=tab[tab["term"]==focal]
        if len(row):
            r=row.iloc[0]
            b=float(r["coef"]); p=float(r["p_value"]) if pd.notna(r["p_value"]) else np.nan
            ci=f"[{r['ci_low']:.4g}, {r['ci_high']:.4g}]"
            if model.family in ("OLS","WLS","Linear Probability Model","Quantile Regression","Fixed Effects","Random Effects","Pooled OLS","First Differences"):
                lines.append(
                    f"**Focal result — `{focal}`:** the estimated coefficient is **{b:.4g}** "
                    f"(95% CI {ci}), {_sig_phrase(p)}. Holding the remaining regressors fixed, "
                    f"a one-unit increase in `{focal}` is associated with a {b:.4g}-unit change in the dependent variable."
                )
            elif model.family in ("Poisson","Negative Binomial","Zero-Inflated Poisson"):
                pct=(np.exp(b)-1)*100
                lines.append(
                    f"**Focal result — `{focal}`:** β = **{b:.4g}** (95% CI {ci}), {_sig_phrase(p)}. "
                    f"For the count component, exp(β) implies an approximate **{pct:.2f}%** multiplicative change in the expected count per one-unit increase, ceteris paribus."
                )
            elif model.family in ("Logit","Probit","Cloglog"):
                me=None
                if model.marginal_effects is not None:
                    m=model.marginal_effects
                    rr=m[m["term"]==focal]
                    if len(rr):
                        effect_col=next((c for c in m.columns if "dy/dx" in str(c)),None)
                        if effect_col:
                            me=float(rr.iloc[0][effect_col])
                if me is not None:
                    lines.append(
                        f"**Focal result — `{focal}`:** the average marginal effect is **{me:.4f}**, "
                        f"equivalent to about **{me*100:.2f} percentage points** for a one-unit change, averaged over the observed sample. "
                        f"The underlying latent-index coefficient is β = {b:.4g}, {_sig_phrase(p)}."
                    )
                else:
                    lines.append(
                        f"**Focal result — `{focal}`:** β = **{b:.4g}**, {_sig_phrase(p)}. "
                        "For nonlinear probability models the coefficient itself is not a percentage-point effect; use marginal effects for substantive interpretation."
                    )
            elif model.family=="Difference-in-Differences":
                lines.append(
                    f"**Focal result — `{focal}`:** β = **{b:.4g}** (95% CI {ci}), {_sig_phrase(p)}. "
                    "If this is the treatment × post interaction, it is the DiD treatment-effect estimate, conditional on the identifying assumptions including parallel trends."
                )
            elif model.family=="IV / 2SLS":
                lines.append(
                    f"**Focal result — `{focal}`:** the second-stage coefficient is **{b:.4g}** (95% CI {ci}), {_sig_phrase(p)}. "
                    "A causal interpretation additionally requires instrument relevance, exclusion and the other IV assumptions."
                )
            else:
                lines.append(f"**Focal result — `{focal}`:** β = **{b:.4g}** (95% CI {ci}), {_sig_phrase(p)}.")

    if model.family=="OLS":
        lines.append("**Recommended checks:** functional form/RESET, heteroskedasticity, influential observations, multicollinearity and residual dependence where relevant.")
    elif model.family in ("Logit","Probit","Cloglog"):
        lines.append("**Recommended checks:** marginal effects, calibration, ROC/AUC, separation, influential observations and sensitivity to specification.")
    elif model.family in ("Fixed Effects","Random Effects","Pooled OLS","First Differences"):
        lines.append("**Recommended checks:** serial/cross-sectional dependence, clustered covariance, time effects, within variation and FE–RE specification.")
    elif model.family=="Difference-in-Differences":
        lines.append("**Recommended checks:** pre-trends/event-study evidence, anticipation, treatment timing, composition changes and clustering level.")
    elif model.family=="IV / 2SLS":
        lines.append("**Recommended checks:** first-stage strength, over-identification when applicable, weak-IV robust inference and exclusion-restriction plausibility.")
    elif model.family.startswith("ARIMA") or model.family.startswith("VAR"):
        lines.append("**Recommended checks:** stationarity, residual autocorrelation, lag-order sensitivity and stability.")

    lines.append("> Interpretation is statistical, not automatic proof of causality. Causal language depends on the research design and identifying assumptions.")
    return "\n\n".join(lines)
