from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.discrete.count_model import ZeroInflatedPoisson
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from patsy import dmatrices
from .utils import build_formula, significance_stars

@dataclass
class ModelResult:
    name: str
    family: str
    formula: str
    coef_table: pd.DataFrame
    metrics: dict
    fitted: pd.Series | None = None
    residuals: pd.Series | None = None
    marginal_effects: pd.DataFrame | None = None
    result: object | None = None
    y_name: str | None = None
    notes: list[str] = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def _coef_table(result):
    params = pd.Series(result.params)
    bse = pd.Series(result.bse, index=params.index)
    pvals = pd.Series(result.pvalues, index=params.index)
    ci = result.conf_int()
    if not isinstance(ci, pd.DataFrame):
        ci = pd.DataFrame(ci, index=params.index)
    stat = getattr(result, "tvalues", getattr(result, "zvalues", np.repeat(np.nan,len(params))))
    tab = pd.DataFrame({
        "term": params.index.astype(str),
        "coef": params.values,
        "std_err": bse.values,
        "stat": np.asarray(stat),
        "p_value": pvals.values,
        "ci_low": ci.iloc[:,0].values,
        "ci_high": ci.iloc[:,1].values,
    })
    tab["sig"] = tab["p_value"].apply(significance_stars)
    return tab

def _metrics(result):
    attrs = {
        "N":"nobs","R²":"rsquared","Adj. R²":"rsquared_adj","Pseudo R²":"prsquared",
        "AIC":"aic","BIC":"bic","LogLik":"llf"
    }
    out = {}
    for label, attr in attrs.items():
        if hasattr(result, attr):
            try: out[label] = float(getattr(result, attr))
            except Exception: pass
    return out

def _fit_cov(result, cov):
    if cov == "Classical": return result.fit()
    if cov in ("HC0","HC1","HC2","HC3"): return result.fit(cov_type=cov)
    return result.fit()

def fit_cross_section(df, name, family, y, x, categoricals=None, interactions=None,
                      cov="Classical", quantile=.5, weights=None, cluster=None):
    formula = build_formula(y, x, categoricals, interactions)

    def fit_with_cov(model, discrete=False):
        if cov == "Classical":
            return model.fit(disp=False) if discrete else model.fit()
        if cov == "Cluster":
            if not cluster:
                raise ValueError("Select a cluster variable.")
            base = model.fit(disp=False) if discrete else model.fit()
            groups = df.loc[base.model.data.row_labels, cluster]
            # Refit directly with clustered covariance where supported
            try:
                return model.fit(disp=False, cov_type="cluster", cov_kwds={"groups": groups}) if discrete else model.fit(
                    cov_type="cluster", cov_kwds={"groups": groups})
            except TypeError:
                return base.get_robustcov_results(cov_type="cluster", groups=groups)
        return model.fit(disp=False, cov_type=cov) if discrete else model.fit(cov_type=cov)

    if family == "OLS":
        result = fit_with_cov(smf.ols(formula, data=df, missing="drop"))
    elif family == "WLS":
        if not weights:
            raise ValueError("WLS requires a weight variable.")
        model_df = df.copy()
        model = smf.wls(formula, data=model_df, weights=model_df[weights], missing="drop")
        result = fit_with_cov(model)
    elif family == "Logit":
        result = fit_with_cov(smf.logit(formula, data=df, missing="drop"), discrete=True)
    elif family == "Probit":
        result = fit_with_cov(smf.probit(formula, data=df, missing="drop"), discrete=True)
    elif family == "Cloglog":
        model = smf.glm(formula, data=df, family=sm.families.Binomial(link=sm.families.links.CLogLog()), missing="drop")
        if cov == "Classical":
            result = model.fit()
        elif cov == "Cluster":
            base = model.fit()
            groups = df.loc[base.model.data.row_labels, cluster]
            result = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
        else:
            result = model.fit(cov_type=cov)
    elif family == "Poisson":
        model = smf.glm(formula, data=df, family=sm.families.Poisson(), missing="drop")
        if cov == "Classical":
            result = model.fit()
        elif cov == "Cluster":
            base = model.fit()
            groups = df.loc[base.model.data.row_labels, cluster]
            result = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
        else:
            result = model.fit(cov_type=cov)
    elif family == "Negative Binomial":
        model = smf.glm(formula, data=df, family=sm.families.NegativeBinomial(), missing="drop")
        if cov == "Classical":
            result = model.fit()
        elif cov == "Cluster":
            base = model.fit()
            groups = df.loc[base.model.data.row_labels, cluster]
            result = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
        else:
            result = model.fit(cov_type=cov)
    elif family == "Quantile Regression":
        result = smf.quantreg(formula, data=df, missing="drop").fit(q=quantile, max_iter=5000)
    else:
        raise ValueError(f"Model not implemented here: {family}")

    me = None
    if family in ("Logit","Probit","Cloglog"):
        try:
            me = result.get_margeff(at="overall").summary_frame().reset_index().rename(columns={"index":"term"})
        except Exception:
            pass
    fitted = pd.Series(np.asarray(result.predict()), index=result.model.data.row_labels, name="fitted")
    resid = None
    try:
        resid = pd.Series(np.asarray(result.resid), index=result.model.data.row_labels, name="residual")
    except Exception:
        pass
    notes = []
    if family == "WLS":
        notes.append(f"Weights: {weights}")
    if cov == "Cluster":
        notes.append(f"Clustered covariance by: {cluster}")
    return ModelResult(name, family, formula, _coef_table(result), _metrics(result), fitted, resid, me, result, y, notes)

def fit_ordered(df, name, y, x, distr="logit"):
    use = df[[y]+x].dropna().copy()
    yv = use[y]
    if not pd.api.types.is_numeric_dtype(yv):
        yv = pd.Categorical(yv, ordered=True).codes
    else:
        cats = sorted(pd.Series(yv).unique())
        yv = pd.Categorical(yv, categories=cats, ordered=True).codes
    X = use[x].astype(float)
    model = OrderedModel(yv, X, distr=distr)
    result = model.fit(method="bfgs", disp=False)
    return ModelResult(name, f"Ordered {distr.title()}", f"{y} ~ " + " + ".join(x),
                       _coef_table(result), _metrics(result), None, None, None, result, y)

def fit_zero_inflated_poisson(df, name, y, x):
    formula = build_formula(y, x)
    yy, XX = dmatrices(formula, df, return_type="dataframe", NA_action="drop")
    model = ZeroInflatedPoisson(yy.iloc[:,0], XX, exog_infl=XX, inflation="logit")
    result = model.fit(method="bfgs", maxiter=300, disp=0)
    return ModelResult(name, "Zero-Inflated Poisson", formula, _coef_table(result), _metrics(result),
                       pd.Series(result.predict(), index=XX.index), None, None, result, y)

def fit_panel(df, name, family, y, x, entity, time, time_effects=False, cov="robust"):
    from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS, FirstDifferenceOLS
    use = df[[entity,time,y]+x].dropna().copy().set_index([entity,time]).sort_index()
    Y = use[y].astype(float)
    X = use[x].astype(float)
    formula = f"{y} ~ " + " + ".join(x)
    if family == "Fixed Effects":
        model = PanelOLS(Y, X, entity_effects=True, time_effects=time_effects, drop_absorbed=True, check_rank=False)
    elif family == "Random Effects":
        model = RandomEffects(Y, sm.add_constant(X))
    elif family == "Pooled OLS":
        model = PooledOLS(Y, sm.add_constant(X))
    elif family == "First Differences":
        model = FirstDifferenceOLS(Y, X)
    else:
        raise ValueError(family)
    if cov == "clustered":
        result = model.fit(cov_type="clustered", cluster_entity=True)
    elif cov == "robust":
        result = model.fit(cov_type="robust")
    else:
        result = model.fit(cov_type="unadjusted")
    params = pd.Series(result.params)
    bse = pd.Series(result.std_errors)
    pvals = pd.Series(result.pvalues)
    ci = result.conf_int()
    tab = pd.DataFrame({"term":params.index.astype(str),"coef":params.values,"std_err":bse.values,
                        "stat":np.asarray(result.tstats),"p_value":pvals.values,
                        "ci_low":ci.iloc[:,0].values,"ci_high":ci.iloc[:,1].values})
    tab["sig"] = tab["p_value"].apply(significance_stars)
    metrics = {"N":float(result.nobs)}
    for label, attr in [("R²", "rsquared"),("Within R²","rsquared_within"),("Between R²","rsquared_between")]:
        if hasattr(result, attr):
            try: metrics[label] = float(getattr(result,attr))
            except Exception: pass
    try:
        fitted = pd.Series(np.asarray(result.fitted_values).ravel(), index=use.index, name="fitted")
        resid = pd.Series(np.asarray(result.resids).ravel(), index=use.index, name="residual")
    except Exception:
        fitted=resid=None
    return ModelResult(name, family, formula, tab, metrics, fitted, resid, None, result, y)

def fit_iv(df, name, y, exog, endog, instruments, cov="robust"):
    from linearmodels.iv import IV2SLS
    exog_part = " + ".join(exog)
    rhs = "1" + (f" + {exog_part}" if exog_part else "")
    formula = f"{y} ~ {rhs} + [{' + '.join(endog)} ~ {' + '.join(instruments)}]"
    model = IV2SLS.from_formula(formula, data=df)
    result = model.fit(cov_type=cov)
    params=pd.Series(result.params); bse=pd.Series(result.std_errors); pvals=pd.Series(result.pvalues); ci=result.conf_int()
    tab=pd.DataFrame({"term":params.index.astype(str),"coef":params.values,"std_err":bse.values,
                      "stat":np.asarray(result.tstats),"p_value":pvals.values,
                      "ci_low":ci.iloc[:,0].values,"ci_high":ci.iloc[:,1].values})
    tab["sig"]=tab["p_value"].apply(significance_stars)
    metrics={"N":float(result.nobs),"R²":float(result.rsquared)}
    return ModelResult(name,"IV / 2SLS",formula,tab,metrics,None,None,None,result,y)

def fit_did(df, name, y, treat, post, controls=None, unit=None, time=None, add_fe=False, cov="HC3"):
    controls = controls or []
    rhs = [treat, post, f"{treat}:{post}"] + controls
    if add_fe and unit:
        rhs.append(f"C({unit})")
    if add_fe and time:
        rhs.append(f"C({time})")
    formula = f"{y} ~ " + " + ".join(rhs)
    mod = smf.ols(formula, data=df, missing="drop")
    base = mod.fit()

    if cov == "Cluster unit" and unit:
        groups = df.loc[base.model.data.row_labels, unit]
        result = base.get_robustcov_results(cov_type="cluster", groups=groups)
        names = base.model.exog_names
        params = np.asarray(result.params)
        bse = np.asarray(result.bse)
        pvals = np.asarray(result.pvalues)
        stat = np.asarray(result.tvalues)
        ci = np.asarray(result.conf_int())
        tab = pd.DataFrame({
            "term": names,
            "coef": params,
            "std_err": bse,
            "stat": stat,
            "p_value": pvals,
            "ci_low": ci[:,0],
            "ci_high": ci[:,1],
        })
        tab["sig"] = tab["p_value"].apply(significance_stars)
    elif cov in ("HC0","HC1","HC2","HC3"):
        result = mod.fit(cov_type=cov)
        tab = _coef_table(result)
    else:
        result = base
        tab = _coef_table(result)

    fitted = pd.Series(np.asarray(result.predict()), index=base.model.data.row_labels, name="fitted")
    resid = pd.Series(np.asarray(result.resid), index=base.model.data.row_labels, name="residual")
    notes = []
    if cov == "Cluster unit" and unit:
        notes.append(f"Clustered covariance by: {unit}")
    return ModelResult(
        name,"Difference-in-Differences",formula,tab,_metrics(result),
        fitted,resid,None,result,y,notes
    )

def fit_arima(df, name, y, p=1,d=0,q=0, exog=None):
    cols=[y]+(exog or [])
    use=df[cols].dropna().copy()
    X=use[exog] if exog else None
    result=ARIMA(use[y], exog=X, order=(p,d,q)).fit()
    fitted=pd.Series(result.fittedvalues,index=use.index)
    resid=pd.Series(result.resid,index=use.index)
    return ModelResult(name,f"ARIMA({p},{d},{q})",f"{y} ~ ARIMA({p},{d},{q})",
                       _coef_table(result),_metrics(result),fitted,resid,None,result,y)

def fit_var(df, name, variables, lags=1):
    use=df[variables].dropna().astype(float)
    result=VAR(use).fit(lags)
    params=result.params.stack()
    tab=params.rename("coef").reset_index()
    tab.columns=["equation_term","equation","coef"]
    tab["term"]=tab["equation"].astype(str)+" : "+tab["equation_term"].astype(str)
    tab["std_err"]=np.nan; tab["stat"]=np.nan; tab["p_value"]=np.nan; tab["ci_low"]=np.nan;tab["ci_high"]=np.nan;tab["sig"]=""
    metrics={"N":float(result.nobs),"AIC":float(result.aic),"BIC":float(result.bic)}
    return ModelResult(name,f"VAR({lags})","VAR: "+", ".join(variables),tab[["term","coef","std_err","stat","p_value","ci_low","ci_high","sig"]],
                       metrics,None,None,None,result,variables[0])
