from __future__ import annotations
import math
import numpy as np
import pandas as pd
import streamlit as st

def inject_css():
    st.markdown("""
    <style>
    :root{
      --ink:#142b38; --muted:#647986; --line:#e2eaee; --soft:#f5f8fa;
      --teal:#0f6c78; --teal2:#17899a; --blue:#315b7d; --amber:#b77b18;
    }
    .block-container{padding-top:1.35rem;max-width:1480px;padding-bottom:4rem}
    [data-testid="stSidebar"]{border-right:1px solid var(--line);background:#fbfcfd}
    [data-testid="stSidebar"] .stRadio label{font-size:.97rem}
    h1,h2,h3{color:var(--ink)}
    .el-hero{
      padding:1.7rem 1.85rem;border:1px solid var(--line);border-radius:20px;
      background:linear-gradient(130deg,#f4fbfc 0%,#ffffff 55%,#f5f8fb 100%);
      margin-bottom:1.2rem;
    }
    .el-hero h1{margin:0 0 .25rem 0;font-size:2.45rem;letter-spacing:-.04em}
    .el-hero p{margin:0;color:var(--muted);font-size:1.05rem}
    .eyebrow{font-size:.78rem;letter-spacing:.12em;text-transform:uppercase;color:var(--teal);font-weight:700}
    .card{
      padding:1rem 1.05rem;border:1px solid var(--line);border-radius:16px;background:#fff;min-height:120px;
    }
    .card strong{color:var(--ink)}
    .small{font-size:.86rem;color:var(--muted)}
    .formula{
      font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
      padding:.8rem 1rem;background:#f7f9fb;border:1px solid var(--line);border-radius:12px;
      overflow:auto;color:#24404f
    }
    .pill{
      display:inline-block;padding:.22rem .58rem;border-radius:999px;background:#eaf5f6;
      border:1px solid #cfe8ea;color:#185f68;font-size:.78rem;font-weight:650;margin-right:.25rem
    }
    .health-good{color:#19764d;font-weight:700}.health-warn{color:#9a6700;font-weight:700}
    .health-bad{color:#b42318;font-weight:700}.health-na{color:#667085;font-weight:700}
    div[data-testid="stMetric"]{
      border:1px solid var(--line);border-radius:14px;padding:.55rem .8rem;background:white
    }
    div[data-testid="stMetricValue"]{font-size:1.35rem}
    .stTabs [data-baseweb="tab-list"]{gap:.35rem}
    .stTabs [data-baseweb="tab"]{border-radius:10px 10px 0 0;padding:.5rem .85rem}
    </style>
    """, unsafe_allow_html=True)

def hero(title, subtitle, eyebrow="Interactive Econometric Research Environment"):
    st.markdown(
        f'<div class="el-hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True
    )

def dataset_fingerprint(df: pd.DataFrame) -> dict:
    nums=df.select_dtypes(include=np.number).columns.tolist()
    cats=[c for c in df.columns if c not in nums]
    binary=[c for c in nums if set(pd.Series(df[c].dropna().unique()).tolist()).issubset({0,1}) and df[c].nunique(dropna=True)<=2]
    low_card=[c for c in df.columns if 2 < df[c].nunique(dropna=True) <= 10]
    missing=float(df.isna().mean().mean()*100) if len(df) else 0
    duplicate=int(df.duplicated().sum())
    return {
        "rows":len(df),"cols":df.shape[1],"numeric":len(nums),"categorical":len(cats),
        "binary":binary,"low_card":low_card,"missing_pct":missing,"duplicates":duplicate
    }

def candidate_panel_pairs(df: pd.DataFrame):
    names=df.columns.tolist()
    id_candidates=[c for c in names if c.lower() in ("id","entity","individual","person","firm","country","region_id","panel_id") or c.lower().endswith("_id")]
    time_candidates=[c for c in names if c.lower() in ("year","time","date","wave","month","quarter","period","t")]
    return id_candidates,time_candidates

def recommend_models(df: pd.DataFrame, y: str):
    s=df[y].dropna()
    nuniq=s.nunique()
    rec=[]
    if pd.api.types.is_numeric_dtype(s):
        vals=set(pd.Series(s.unique()).tolist())
        if nuniq==2 and vals.issubset({0,1}):
            rec += [("Logit","Strong fit for a binary outcome"),("Probit","Alternative latent-index probability model"),("LPM","Simple linear probability benchmark")]
        elif pd.api.types.is_integer_dtype(s) and s.min()>=0 and nuniq>2 and s.quantile(.9)<50:
            rec += [("Poisson","Natural starting point for counts"),("Negative Binomial","Useful when counts are overdispersed"),("OLS","Benchmark if the outcome is approximately continuous")]
        elif 3 <= nuniq <= 10:
            rec += [("Ordered Logit","Useful if numeric codes represent ordered categories"),("OLS","Often used as a transparent benchmark"),("Quantile Regression","Useful for heterogeneous conditional effects")]
        else:
            rec += [("OLS","Default model for a continuous outcome"),("Quantile Regression","Robust distributional perspective")]
    else:
        if nuniq==2:
            rec += [("Logit","Binary outcome"),("Probit","Binary outcome")]
        elif nuniq<=10:
            rec += [("Ordered Logit","If categories have a meaningful order")]
    ids,times=candidate_panel_pairs(df)
    if ids and times:
        rec.append(("Panel FE / RE",f"Repeated-data structure may be available: {ids[0]} × {times[0]}"))
    return rec

def compact_metrics(model):
    preferred=["N","R²","Adj. R²","Pseudo R²","Within R²","AIC","BIC","LogLik"]
    return [(k,model.metrics[k]) for k in preferred if k in model.metrics and pd.notna(model.metrics[k])]

def model_header(model):
    st.markdown(f"### {model.name}")
    st.markdown(f'<div class="formula">{model.formula}</div>',unsafe_allow_html=True)
    vals=compact_metrics(model)
    if vals:
        cols=st.columns(min(5,len(vals)))
        for i,(k,v) in enumerate(vals[:5]):
            if k=="N":
                cols[i].metric(k,f"{int(v):,}")
            else:
                cols[i].metric(k,f"{float(v):.4g}")

def comparison_table(models):
    if not models:
        return pd.DataFrame()
    all_terms=[]
    for m in models:
        all_terms += [t for t in m.coef_table["term"].astype(str) if t.lower() not in ("intercept","const")]
    terms=list(dict.fromkeys(all_terms))
    out=pd.DataFrame(index=terms)
    for m in models:
        vals={}
        for _,r in m.coef_table.iterrows():
            term=str(r["term"])
            if term.lower() in ("intercept","const"): continue
            p=r.get("p_value",np.nan)
            stars="***" if pd.notna(p) and p<.01 else "**" if pd.notna(p) and p<.05 else "*" if pd.notna(p) and p<.10 else ""
            vals[term]=f'{r["coef"]:.4g}{stars}\n({r["std_err"]:.3g})'
        out[m.name]=pd.Series(vals)
    fit_rows={}
    for label in ["N","R²","Adj. R²","Pseudo R²","AIC","BIC"]:
        fit_rows[label]={m.name:(f"{int(m.metrics[label]):,}" if label=="N" and label in m.metrics else
                                  f"{m.metrics[label]:.4g}" if label in m.metrics else "") for m in models}
    fit=pd.DataFrame(fit_rows).T
    return pd.concat([out,fit],axis=0)

def assistant_recommendation(df, y=None):
    lines=[]
    fp=dataset_fingerprint(df)
    lines.append(f"Dataset: **{fp['rows']:,} observations**, **{fp['cols']} variables**, average missingness **{fp['missing_pct']:.2f}%**.")
    if y:
        rec=recommend_models(df,y)
        lines.append(f"For **`{y}`**, the most plausible starting points are:")
        for name,why in rec[:4]:
            lines.append(f"- **{name}** — {why}.")
    ids,times=candidate_panel_pairs(df)
    if ids and times:
        lines.append(f"Possible panel identifiers detected: **{', '.join(ids[:3])}**; possible time variables: **{', '.join(times[:3])}**.")
    if fp["duplicates"]>0:
        lines.append(f"There are **{fp['duplicates']:,} duplicated rows**; verify whether these are genuine repeated records.")
    if fp["missing_pct"]>5:
        lines.append("Missingness is non-trivial; inspect patterns before relying on complete-case estimation.")
    lines.append("Model choice should follow the research design and identifying assumptions, not only the observed variable type.")
    return "\n\n".join(lines)
