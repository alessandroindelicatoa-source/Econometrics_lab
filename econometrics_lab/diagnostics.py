from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, linear_reset, acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor

def ols_diagnostics(model_result):
    r = model_result.result
    out=[]
    resid=np.asarray(r.resid)
    exog=np.asarray(r.model.exog)
    try:
        lm,lmp,_,_=het_breuschpagan(resid,exog); out.append(["Breusch–Pagan",lm,lmp,"Homoscedasticity"])
    except Exception: pass
    try:
        lm,lmp,_,_=het_white(resid,exog); out.append(["White",lm,lmp,"Homoscedasticity"])
    except Exception: pass
    try:
        jb,jbp,_,_=jarque_bera(resid); out.append(["Jarque–Bera",jb,jbp,"Normal residuals"])
    except Exception: pass
    try:
        out.append(["Durbin–Watson",durbin_watson(resid),np.nan,"≈2 suggests no AR(1)"])
    except Exception: pass
    try:
        bg=acorr_breusch_godfrey(r,nlags=min(4,max(1,len(resid)//20))); out.append(["Breusch–Godfrey",bg[0],bg[1],"No serial correlation"])
    except Exception: pass
    try:
        rr=linear_reset(r,power=2,use_f=True); out.append(["Ramsey RESET",float(rr.fvalue),float(rr.pvalue),"Correct functional form"])
    except Exception: pass
    return pd.DataFrame(out,columns=["test","statistic","p_value","null_hypothesis"])

def vif_table(model_result):
    r=model_result.result
    X=pd.DataFrame(r.model.exog, columns=r.model.exog_names)
    rows=[]
    for i,c in enumerate(X.columns):
        if c.lower() in ("intercept","const"): continue
        try: v=variance_inflation_factor(X.values,i)
        except Exception: v=np.nan
        rows.append([c,v])
    return pd.DataFrame(rows,columns=["term","VIF"]).sort_values("VIF",ascending=False)

def influence_table(model_result):
    r=model_result.result
    sf=r.get_influence().summary_frame()
    cols=[c for c in ["standard_resid","student_resid","cooks_d","hat_diag","dffits_internal"] if c in sf.columns]
    return sf[cols].sort_values("cooks_d" if "cooks_d" in cols else cols[0],ascending=False)
