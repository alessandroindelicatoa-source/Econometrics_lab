from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from .model_engine import fit_cross_section
from .diagnostics import ols_diagnostics, vif_table, influence_table

def health_check(model):
    rows=[]
    fam=model.family
    if fam in ("OLS","Linear Probability Model"):
        try:
            d=ols_diagnostics(model)
            for test,label in [
                ("Breusch–Pagan","Heteroskedasticity"),
                ("Ramsey RESET","Functional form"),
                ("Jarque–Bera","Residual normality"),
                ("Breusch–Godfrey","Serial correlation"),
            ]:
                r=d[d["test"]==test]
                if len(r) and pd.notna(r.iloc[0]["p_value"]):
                    p=float(r.iloc[0]["p_value"])
                    status="Good" if p>=.05 else "Warning"
                    rows.append([label,status,f"{test}: p={p:.3g}"])
            try:
                v=vif_table(model)
                mx=float(v["VIF"].max()) if len(v) else np.nan
                status="Good" if mx<5 else "Warning" if mx<10 else "High"
                rows.append(["Multicollinearity",status,f"max VIF={mx:.2f}"])
            except Exception:
                rows.append(["Multicollinearity","N/A","VIF unavailable"])
            try:
                inf=influence_table(model)
                cd=inf["cooks_d"] if "cooks_d" in inf.columns else pd.Series(dtype=float)
                n=max(1,int(model.metrics.get("N",len(cd))))
                k=int((cd>4/n).sum()) if len(cd) else 0
                rows.append(["Influential cases","Good" if k==0 else "Warning",f"{k} observations with Cook's D > 4/N"])
            except Exception:
                pass
        except Exception:
            rows.append(["Classical diagnostics","N/A","Not available for this fitted result"])
    elif fam in ("Logit","Probit","Cloglog"):
        rows += [
            ["Probability scale","Good","Use marginal effects for substantive interpretation"],
            ["Classification","Check","Inspect ROC/AUC and calibration"],
            ["Specification","Check","Test alternative controls, nonlinearities and interactions"],
        ]
    elif fam=="Difference-in-Differences":
        rows += [
            ["Parallel trends","Check","Inspect pre-treatment trends / event study"],
            ["Inference","Check","Cluster at the treatment-assignment level when appropriate"],
            ["Anticipation","Check","Consider whether treatment could affect pre-period behaviour"],
        ]
    elif fam=="IV / 2SLS":
        rows += [
            ["Instrument relevance","Check","Inspect first-stage strength"],
            ["Exclusion restriction","Check","Requires substantive justification"],
            ["Weak instruments","Check","Consider weak-IV robust inference"],
        ]
    elif fam in ("Fixed Effects","Random Effects","Pooled OLS","First Differences"):
        rows += [
            ["Panel dependence","Check","Use an appropriate clustered/robust covariance"],
            ["Time effects","Check","Assess common shocks and time fixed effects"],
            ["Specification","Check","Compare within/between identifying variation"],
        ]
    else:
        rows += [["Model diagnostics","Check","Use model-specific diagnostic tests and sensitivity analysis"]]
    return pd.DataFrame(rows,columns=["dimension","status","detail"])

def robustness_ols(df,y,focal,controls=None,cluster=None,fe_vars=None,winsor_vars=None,exclude_influential=False):
    controls=controls or []
    fe_vars=fe_vars or []
    winsor_vars=winsor_vars or []
    rows=[]
    models=[]

    def collect(label,m):
        r=m.coef_table[m.coef_table["term"]==focal]
        if len(r):
            q=r.iloc[0]
            rows.append({
                "specification":label,
                "coef":q["coef"],
                "std_err":q["std_err"],
                "p_value":q["p_value"],
                "ci_low":q["ci_low"],
                "ci_high":q["ci_high"],
                "N":m.metrics.get("N",np.nan),
            })
            models.append(m)

    base=fit_cross_section(df,"Baseline","OLS",y,[focal]+controls,cov="Classical")
    collect("Baseline",base)
    collect("HC3",fit_cross_section(df,"HC3","OLS",y,[focal]+controls,cov="HC3"))
    collect("HC1",fit_cross_section(df,"HC1","OLS",y,[focal]+controls,cov="HC1"))

    if cluster:
        collect(
            f"Cluster: {cluster}",
            fit_cross_section(df,f"Cluster {cluster}","OLS",y,[focal]+controls,cov="Cluster",cluster=cluster)
        )

    if fe_vars:
        x=[focal]+controls+fe_vars
        collect(
            "Fixed-effect controls: "+", ".join(fe_vars),
            fit_cross_section(df,"FE controls","OLS",y,x,categoricals=fe_vars,cov="HC3")
        )

    if winsor_vars:
        d=df.copy()
        for c in winsor_vars:
            lo,hi=d[c].quantile([.01,.99])
            d[c]=d[c].clip(lo,hi)
        collect("Winsorized 1%",fit_cross_section(d,"Winsorized","OLS",y,[focal]+controls,cov="HC3"))

    if exclude_influential:
        try:
            infl=base.result.get_influence().summary_frame()
            keep=infl["cooks_d"] <= 4/len(infl)
            d=df.loc[infl.index[keep]].copy()
            collect("Exclude influential",fit_cross_section(d,"No influential","OLS",y,[focal]+controls,cov="HC3"))
        except Exception:
            pass

    return pd.DataFrame(rows),models

def event_study(df,y,treat,relative_time,controls=None,unit=None,time=None,reference=-1,cluster=True):
    controls=controls or []
    use=df.copy()

    periods=[
        p for p in sorted(pd.Series(use[relative_time].dropna().unique()).tolist())
        if p != reference
    ]

    es_cols=[]
    period_map={}
    for i,p in enumerate(periods):
        col=f"__es_{i}"
        use[col]=((use[relative_time]==p) & (use[treat]==1)).astype(int)
        es_cols.append(col)
        period_map[col]=p

    rhs=es_cols + controls
    if unit:
        rhs.append(f"C({unit})")
    if time:
        rhs.append(f"C({time})")

    formula=f"{y} ~ " + " + ".join(rhs)
    model=smf.ols(formula,data=use,missing="drop")
    base=model.fit()

    if cluster and unit:
        groups=use.loc[base.model.data.row_labels,unit]
        result=model.fit(cov_type="cluster",cov_kwds={"groups":groups})
    else:
        result=model.fit(cov_type="HC3")

    ci=result.conf_int()
    rows=[]
    for col in es_cols:
        if col not in result.params.index:
            continue
        rows.append({
            "period":period_map[col],
            "coef":float(result.params[col]),
            "std_err":float(result.bse[col]),
            "p_value":float(result.pvalues[col]),
            "ci_low":float(ci.loc[col,0]),
            "ci_high":float(ci.loc[col,1]),
            "term":f"{treat} × 1({relative_time}={period_map[col]})",
        })

    out=pd.DataFrame(rows)
    if len(out):
        try:
            out=out.sort_values("period")
        except Exception:
            pass

    display_formula=(
        f"{y} ~ event-time × {treat} [reference={reference}]"
        + (f" + {', '.join(controls)}" if controls else "")
        + (f" + unit FE({unit})" if unit else "")
        + (f" + time FE({time})" if time else "")
    )
    return out,display_formula,result
