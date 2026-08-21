import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.io as pio

from econometrics_lab.data_manager import load_bytes, apply_transform, missing_summary, handle_missing
from econometrics_lab.utils import dataframe_profile, numeric_columns, categorical_columns
from econometrics_lab.model_engine import fit_cross_section, fit_ordered, fit_zero_inflated_poisson, fit_panel, fit_iv, fit_did, fit_arima, fit_var
from econometrics_lab.diagnostics import ols_diagnostics, vif_table, influence_table
from econometrics_lab.plot_factory import *
from econometrics_lab.fuzzy import fuzzy_index, topsis
from econometrics_lab.exporting import models_excel, docx_report, pdf_report
from econometrics_lab.codegen import generate_code
from econometrics_lab.interpretation import interpret_model

APP_DIR=Path(__file__).parent
st.set_page_config(page_title="Econometrics Lab",page_icon="📊",layout="wide",initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container{padding-top:1.25rem;max-width:1500px}
.hero{padding:1.35rem 1.5rem;border-radius:18px;background:linear-gradient(120deg,#e8f4f5,#f7fbfc);border:1px solid #dcebee;margin-bottom:1rem}
.hero h1{margin:0;font-size:2.2rem}.muted{opacity:.68}
[data-testid="stSidebar"]{border-right:1px solid #e6edf0}
</style>
""",unsafe_allow_html=True)

if "df" not in st.session_state: st.session_state.df=None
if "source_name" not in st.session_state: st.session_state.source_name=None
if "models" not in st.session_state: st.session_state.models=[]
if "column_map" not in st.session_state: st.session_state.column_map={}

def set_df(df,name="dataset",mapping=None):
    st.session_state.df=df;st.session_state.source_name=name;st.session_state.column_map=mapping or {}

def need_data():
    if st.session_state.df is None:
        st.warning("Load a dataset first in **DATA → Import**.");st.stop()
    return st.session_state.df

def add_model(m):
    st.session_state.models.append(m);st.success(f"Saved model: **{m.name}**")

def fig_download(fig,key):
    html=pio.to_html(fig,include_plotlyjs="cdn",full_html=True).encode()
    st.download_button("Download interactive graph (.html)",html,file_name=f"{key}.html",mime="text/html",key=f"dl_{key}")

with st.sidebar:
    st.markdown("## 📊 ECONOMETRICS LAB")
    st.caption("Research-grade econometric playground")
    page=st.radio("Workspace",[
        "🏠 Home","📁 Data","📊 Graphs","📈 Cross-sectional models","🧱 Panel data","🎯 IV & Causal","⏱ Time series",
        "🧪 Diagnostics & Robustness","🎮 Model Playground","🧠 Interpretation","🔷 Fuzzy Lab","📚 Results & Export","💻 Code Generator",
    ],label_visibility="collapsed")
    st.divider()
    if st.session_state.df is not None:
        d=st.session_state.df;st.caption(f"**Data:** {st.session_state.source_name}");st.caption(f"{len(d):,} rows · {d.shape[1]} variables")
    st.caption(f"Saved models: {len(st.session_state.models)}")

if page=="🏠 Home":
    st.markdown("""<div class="hero"><h1>Econometrics Lab</h1><p>Build, estimate, stress-test, visualise and export econometric models in one interactive workspace.</p></div>""",unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4);c1.metric("Model families","20+");c2.metric("Graph modules","15+");c3.metric("Code targets","4");c4.metric("Workflow","Data → Model → Lab")
    st.subheader("What you can do")
    st.markdown("""
**Data engineering.** Import CSV, Excel, Stata, SPSS and Parquet; inspect missingness; transform, lag, difference, standardise and winsorise variables.

**Econometrics.** OLS/WLS, Logit, Probit, Poisson, Negative Binomial, Ordered models, Quantile Regression, FE/RE/Pooled/FD panel models, IV/2SLS, DiD, ARIMA and VAR.

**Graphs.** A dedicated visual laboratory plus model-aware figures: coefficient plots, confidence intervals, residuals, Q–Q, actual-vs-predicted, ROC, marginal effects, DiD trends and model comparison.

**Research lab.** Diagnostics, specification curves, scenario simulation, model storage/comparison, fuzzy indicators, TOPSIS and reproducible code.
""")
    st.info("Start with **📁 Data**. Two synthetic research datasets are bundled with the project.")

elif page=="📁 Data":
    st.title("📁 Data");tabs=st.tabs(["Import","Preview & profile","Missing data","Transform","Filter"])
    with tabs[0]:
        c1,c2=st.columns([1,1])
        with c1:
            st.subheader("Upload");up=st.file_uploader("CSV, Excel, Stata, SPSS or Parquet",type=["csv","xlsx","xls","dta","sav","parquet"])
            if up is not None:
                try:
                    df,mapping=load_bytes(up.getvalue(),up.name);set_df(df,up.name,mapping);st.success(f"Loaded {len(df):,} rows.")
                except Exception as e: st.error(str(e))
        with c2:
            st.subheader("Built-in demos");demo=st.selectbox("Dataset",["Panel / causal demo","Time-series demo"])
            if st.button("Load demo",type="primary"):
                fn="panel_causal_demo.csv" if demo.startswith("Panel") else "time_series_demo.csv";df=pd.read_csv(APP_DIR/"data"/fn);set_df(df,fn,{c:c for c in df.columns});st.success(f"Loaded {fn}.")
    with tabs[1]:
        df=need_data();st.dataframe(df.head(200),use_container_width=True);st.subheader("Variable profile");st.dataframe(dataframe_profile(df),use_container_width=True,hide_index=True)
        st.download_button("Download current data (CSV)",df.to_csv(index=False).encode(),file_name="econometrics_lab_data.csv",mime="text/csv")
    with tabs[2]:
        df=need_data();st.dataframe(missing_summary(df),use_container_width=True,hide_index=True)
        mode=st.selectbox("Treatment",["Drop rows","Mean","Median","Mode","Forward fill","Backward fill"]);cols=st.multiselect("Variables",df.columns.tolist(),default=df.columns.tolist())
        if st.button("Apply missing-data treatment"): set_df(handle_missing(df,mode,cols),st.session_state.source_name,st.session_state.column_map);st.success("Applied.")
    with tabs[3]:
        df=need_data();nums=numeric_columns(df);op=st.selectbox("Transformation",["Log","Log(1+x)","Square","Standardize","Difference","Lag","Winsorize","Interaction","Dummy threshold"]);cols=st.multiselect("Variables",nums);new=st.text_input("New variable name (optional)");val=None
        if op=="Lag": val=st.number_input("Lag",1,50,1)
        elif op=="Winsorize": val=st.number_input("Tail proportion",.001,.20,.01,.001)
        elif op=="Dummy threshold": val=st.number_input("Threshold",value=0.0)
        if st.button("Create transformation"):
            try: set_df(apply_transform(df,op,cols,new or None,val),st.session_state.source_name,st.session_state.column_map);st.success("Variable(s) created.")
            except Exception as e: st.error(str(e))
    with tabs[4]:
        df=need_data();st.caption("Use a pandas query, e.g. `age >= 30 and immigrant == 1`.");q=st.text_input("Filter expression")
        if st.button("Apply filter") and q:
            try: set_df(df.query(q),st.session_state.source_name,st.session_state.column_map);st.success("Filter applied.")
            except Exception as e: st.error(str(e))

elif page=="📊 Graphs":
    df=need_data();st.title("📊 Graph Studio");st.caption("Exploratory, descriptive and model-aware graphics in one place.")
    tabs=st.tabs(["Distributions","Relationships","Groups","Correlation","Time / Panel","Missingness","Model graphs","Graph gallery"]);nums=numeric_columns(df)
    with tabs[0]:
        kind=st.radio("Chart",["Histogram","Box","Violin"],horizontal=True);y=st.selectbox("Variable",nums,key="g_dist_y");color=st.selectbox("Colour/group",["None"]+df.columns.tolist(),key="g_dist_c");color=None if color=="None" else color
        if kind=="Histogram":
            bins=st.slider("Bins",5,100,30);fig=histogram(df,y,color,bins)
        else:
            group=st.selectbox("X/group",["None"]+df.columns.tolist(),key="g_dist_x");group=None if group=="None" else group;fig=box_violin(df,y,group,color,kind)
        st.plotly_chart(fig,width="stretch");fig_download(fig,"distribution")
    with tabs[1]:
        if len(nums)>=2:
            c1,c2=st.columns(2);x=c1.selectbox("X",nums,key="relx");y=c2.selectbox("Y",nums,index=min(1,len(nums)-1),key="rely");color=st.selectbox("Colour",["None"]+df.columns.tolist(),key="relc");color=None if color=="None" else color
            trend=st.selectbox("Trend",["None","OLS","LOWESS"]);trend=None if trend=="None" else trend;fig=scatter(df,x,y,color,None,trend);st.plotly_chart(fig,width="stretch");fig_download(fig,"relationship")
    with tabs[2]:
        y=st.selectbox("Outcome",nums,key="groupy");group=st.selectbox("Group",df.columns.tolist(),key="groupx");fig=group_means(df,y,group);st.plotly_chart(fig,width="stretch");fig_download(fig,"group_means")
    with tabs[3]:
        vars_=st.multiselect("Variables",nums,default=nums[:min(8,len(nums))],key="corrvars");method=st.radio("Correlation",["pearson","spearman","kendall"],horizontal=True)
        if len(vars_)>=2:
            fig=correlation_heatmap(df,vars_,method);st.plotly_chart(fig,width="stretch");fig_download(fig,"correlation")
    with tabs[4]:
        x=st.selectbox("Time/index",df.columns.tolist(),key="timex");y=st.selectbox("Y",nums,key="timey");group=st.selectbox("Panel/group (optional)",["None"]+df.columns.tolist(),key="timeg");group=None if group=="None" else group
        fig=line_plot(df,x,y,group);st.plotly_chart(fig,width="stretch");fig_download(fig,"time_panel")
    with tabs[5]:
        fig=missingness(df);st.plotly_chart(fig,width="stretch");fig_download(fig,"missingness")
    with tabs[6]:
        if not st.session_state.models: st.info("Estimate and save at least one model.")
        else:
            names=[m.name for m in st.session_state.models];mn=st.selectbox("Model",names,key="gm_model");m=next(x for x in st.session_state.models if x.name==mn);choices=["Coefficients"]
            if m.fitted is not None and m.residuals is not None: choices += ["Actual vs predicted","Residuals vs fitted","Residual histogram","Q–Q"]
            if m.family in ("Logit","Probit"): choices += ["ROC"]
            if m.marginal_effects is not None: choices += ["Marginal effects"]
            chart=st.selectbox("Model graph",choices)
            try:
                if chart=="Coefficients": fig=coefficient_plot(m)
                elif chart=="Actual vs predicted": fig=actual_predicted(df,m)
                elif chart=="Residuals vs fitted": fig=residual_fitted(m)
                elif chart=="Residual histogram": fig=residual_hist(m)
                elif chart=="Q–Q": fig=qq_plot(m)
                elif chart=="ROC": fig=roc_plot(df,m)
                else: fig=marginal_effects_plot(m)
                st.plotly_chart(fig,width="stretch");fig_download(fig,"model_graph")
            except Exception as e: st.error(str(e))
    with tabs[7]:
        st.markdown("""
**Included visual families**

- Distribution: histogram, marginal box plots, box and violin.
- Relationships: scatter, OLS trend, LOWESS and grouped scatter.
- Structure: Pearson/Spearman/Kendall heatmaps and scatter matrices.
- Groups: means with 95% confidence intervals.
- Panel/time: trajectories by unit or group.
- Data quality: missingness charts.
- Models: coefficient forest plots, actual vs predicted, residual diagnostics, Q–Q, ROC/AUC and marginal effects.
- Causal: DiD group trends.
- Comparison: the same coefficient across saved specifications.
""")
        if len(nums)>=2 and len(nums)<=10 and st.button("Build scatter-matrix"):
            fig=scatter_matrix(df,nums[:8]);st.plotly_chart(fig,width="stretch")

elif page=="📈 Cross-sectional models":
    df=need_data();st.title("📈 Cross-sectional models");family=st.selectbox("Model",["OLS","WLS","Logit","Probit","Poisson","Negative Binomial","Zero-Inflated Poisson","Ordered Logit","Ordered Probit","Quantile Regression"]);y=st.selectbox("Dependent variable",df.columns.tolist());x=st.multiselect("Explanatory variables",[c for c in df.columns if c!=y]);name=st.text_input("Model name",value=f"{family} — {y}")
    cats=[];interactions=[];cov="Classical";q=.5;weights=None;cluster=None
    if family in ("OLS","WLS","Logit","Probit","Poisson","Negative Binomial"):
        cats=st.multiselect("Treat as categorical",[c for c in x if df[c].nunique(dropna=True)<30])
        with st.expander("Interactions"):
            a=st.selectbox("Variable A",["None"]+x,key="inta");b=st.selectbox("Variable B",["None"]+x,key="intb")
            if a!="None" and b!="None" and a!=b: interactions=[(a,b)]
        cov=st.selectbox("Covariance",["Classical","HC0","HC1","HC2","HC3","Cluster"],index=4 if family in ("OLS","WLS") else 0)
        if cov=="Cluster":
            cluster=st.selectbox("Cluster variable",df.columns.tolist())
        if family=="WLS":
            positive=[c for c in numeric_columns(df) if c!=y]
            weights=st.selectbox("Weight variable",positive)
    elif family=="Quantile Regression": q=st.slider("Quantile",.05,.95,.50,.05)
    if st.button("Estimate & save",type="primary",disabled=not bool(x)):
        try:
            if family=="Zero-Inflated Poisson": m=fit_zero_inflated_poisson(df,name,y,x)
            elif family in ("Ordered Logit","Ordered Probit"): m=fit_ordered(df,name,y,x,"logit" if family.endswith("Logit") else "probit")
            else: m=fit_cross_section(df,name,family,y,x,cats,interactions,cov,q,weights,cluster)
            add_model(m);st.dataframe(m.coef_table,use_container_width=True,hide_index=True);st.json(m.metrics);st.plotly_chart(coefficient_plot(m),width="stretch")
        except Exception as e: st.exception(e)

elif page=="🧱 Panel data":
    df=need_data();nums=numeric_columns(df);st.title("🧱 Panel data");family=st.selectbox("Estimator",["Fixed Effects","Random Effects","Pooled OLS","First Differences"]);c1,c2=st.columns(2);entity=c1.selectbox("Entity ID",df.columns.tolist());time=c2.selectbox("Time",df.columns.tolist());y=st.selectbox("Dependent variable",nums);x=st.multiselect("Regressors",[c for c in nums if c!=y]);time_fe=st.checkbox("Time fixed effects",value=True,disabled=family!="Fixed Effects");cov=st.selectbox("Covariance",["robust","clustered","unadjusted"]);name=st.text_input("Model name",f"{family} — {y}")
    if st.button("Estimate panel model",type="primary",disabled=not bool(x)):
        try: m=fit_panel(df,name,family,y,x,entity,time,time_fe,cov);add_model(m);st.dataframe(m.coef_table,use_container_width=True,hide_index=True);st.json(m.metrics);st.plotly_chart(coefficient_plot(m),width="stretch")
        except Exception as e: st.exception(e)

elif page=="🎯 IV & Causal":
    df=need_data();nums=numeric_columns(df);st.title("🎯 IV & Causal");tabs=st.tabs(["IV / 2SLS","Difference-in-Differences"])
    with tabs[0]:
        y=st.selectbox("Outcome",nums,key="ivy");endog=st.multiselect("Endogenous regressor(s)",[c for c in nums if c!=y],key="ivendog");inst=st.multiselect("Instrument(s)",[c for c in nums if c!=y and c not in endog],key="ivinst");exog=st.multiselect("Exogenous controls",[c for c in nums if c!=y and c not in endog and c not in inst],key="ivexog");name=st.text_input("Model name",f"2SLS — {y}",key="ivname")
        if st.button("Estimate 2SLS",disabled=not(endog and inst)):
            try: m=fit_iv(df,name,y,exog,endog,inst);add_model(m);st.dataframe(m.coef_table,use_container_width=True,hide_index=True)
            except Exception as e: st.exception(e)
    with tabs[1]:
        y=st.selectbox("Outcome",nums,key="didy");treat=st.selectbox("Treatment indicator",nums,key="didt");post=st.selectbox("Post indicator",nums,key="didp");controls=st.multiselect("Controls",[c for c in nums if c not in [y,treat,post]],key="didc");add_fe=st.checkbox("Add unit/time fixed effects");unit=st.selectbox("Unit",["None"]+df.columns.tolist(),key="didu");unit=None if unit=="None" else unit;time=st.selectbox("Time",["None"]+df.columns.tolist(),key="didtime");time=None if time=="None" else time;cov=st.selectbox("SE",["HC3","HC2","HC1","HC0","Cluster unit"]);name=st.text_input("Model name",f"DiD — {y}",key="didname")
        if st.button("Estimate DiD"):
            try:
                m=fit_did(df,name,y,treat,post,controls,unit,time,add_fe,cov);add_model(m);st.dataframe(m.coef_table,use_container_width=True,hide_index=True)
                if time: st.plotly_chart(did_trends(df,y,treat,time),width="stretch")
            except Exception as e: st.exception(e)

elif page=="⏱ Time series":
    df=need_data();nums=numeric_columns(df);st.title("⏱ Time series");tabs=st.tabs(["Stationarity","ARIMA","VAR"])
    with tabs[0]:
        from statsmodels.tsa.stattools import adfuller,kpss
        y=st.selectbox("Series",nums,key="tsy");s=df[y].dropna()
        if st.button("Run ADF & KPSS"):
            try:
                adf=adfuller(s,autolag="AIC");kp=kpss(s,regression="c",nlags="auto")
                st.dataframe(pd.DataFrame([["ADF",adf[0],adf[1],"Unit root"],["KPSS",kp[0],kp[1],"Stationary"]],columns=["test","statistic","p_value","null_hypothesis"]),hide_index=True,use_container_width=True)
            except Exception as e: st.error(str(e))
    with tabs[1]:
        y=st.selectbox("Series",nums,key="arimay");c1,c2,c3=st.columns(3);p=c1.number_input("p",0,10,1);d=c2.number_input("d",0,3,0);q=c3.number_input("q",0,10,1);exog=st.multiselect("Exogenous regressors",[c for c in nums if c!=y],key="arimax");name=st.text_input("Model name",f"ARIMA({p},{d},{q}) — {y}",key="ariman")
        if st.button("Estimate ARIMA"):
            try:
                m=fit_arima(df,name,y,p,d,q,exog);add_model(m);st.dataframe(m.coef_table,use_container_width=True,hide_index=True)
                if m.fitted is not None:
                    plot=pd.DataFrame({"actual":df.loc[m.fitted.index,y],"fitted":m.fitted});st.plotly_chart(style(px.line(plot,y=["actual","fitted"]),"ARIMA fit"),width="stretch")
            except Exception as e: st.exception(e)
    with tabs[2]:
        vars_=st.multiselect("Endogenous variables",nums,default=nums[:min(2,len(nums))],key="varvars");lags=st.number_input("Lags",1,12,1);name=st.text_input("Model name",f"VAR({lags})",key="varname")
        if st.button("Estimate VAR",disabled=len(vars_)<2):
            try: m=fit_var(df,name,vars_,lags);add_model(m);st.dataframe(m.coef_table,use_container_width=True,hide_index=True)
            except Exception as e: st.exception(e)

elif page=="🧪 Diagnostics & Robustness":
    df=need_data();st.title("🧪 Diagnostics & Robustness")
    if not st.session_state.models: st.info("Estimate models first.");st.stop()
    mn=st.selectbox("Model",[x.name for x in st.session_state.models]);model=next(x for x in st.session_state.models if x.name==mn);tabs=st.tabs(["Diagnostics","Influence","Specification curve","Compare saved models"])
    with tabs[0]:
        if model.family=="OLS":
            st.dataframe(ols_diagnostics(model),use_container_width=True,hide_index=True);st.subheader("Variance Inflation Factors");st.dataframe(vif_table(model),use_container_width=True,hide_index=True)
        else: st.info("Classical OLS diagnostics are shown for OLS models.")
        if model.residuals is not None:
            c1,c2=st.columns(2);c1.plotly_chart(residual_fitted(model),width="stretch");c2.plotly_chart(qq_plot(model),width="stretch")
    with tabs[1]:
        if model.family=="OLS":
            try: st.dataframe(influence_table(model).head(100),use_container_width=True)
            except Exception as e: st.error(str(e))
        else: st.info("Influence diagnostics currently target OLS.")
    with tabs[2]:
        st.markdown("### Automated specification curve");nums=numeric_columns(df);y=st.selectbox("Outcome",nums,key="scy");focal=st.selectbox("Focal variable",[c for c in nums if c!=y],key="scf");candidate=st.multiselect("Candidate controls",[c for c in nums if c not in [y,focal]],key="scc");max_controls=st.slider("Maximum controls per specification",0,min(5,len(candidate)),min(3,len(candidate)))
        if st.button("Run specification curve"):
            from itertools import combinations
            rows=[];combos=[]
            for k in range(max_controls+1): combos.extend(list(combinations(candidate,k)))
            for i,ctrl in enumerate(combos[:100],1):
                try:
                    mm=fit_cross_section(df,f"Spec {i}","OLS",y,[focal]+list(ctrl),cov="HC3");rr=mm.coef_table[mm.coef_table.term==focal].iloc[0];rows.append({"spec":i,"controls":", ".join(ctrl) or "None","coef":rr.coef,"low":rr.ci_low,"high":rr.ci_high,"p":rr.p_value})
                except Exception: pass
            res=pd.DataFrame(rows);st.dataframe(res,use_container_width=True,hide_index=True)
            if len(res):
                import plotly.graph_objects as go
                fig=go.Figure(go.Scatter(x=res["spec"],y=res["coef"],mode="markers",error_y=dict(type="data",symmetric=False,array=res["high"]-res["coef"],arrayminus=res["coef"]-res["low"])));fig.add_hline(y=0,line_dash="dash");fig.update_xaxes(title="Specification");fig.update_yaxes(title=f"Coefficient: {focal}");st.plotly_chart(style(fig,"Specification curve"),width="stretch")
    with tabs[3]:
        models=st.session_state.models;common=set(models[0].coef_table.term)
        for mm in models[1:]: common &= set(mm.coef_table.term)
        common=sorted(common-set(["Intercept","const"]))
        if common:
            term=st.selectbox("Common coefficient",common);st.plotly_chart(model_comparison(models,term),width="stretch")
        st.dataframe(pd.DataFrame([{"model":x.name,"family":x.family,"formula":x.formula,**x.metrics} for x in models]),use_container_width=True,hide_index=True)

elif page=="🎮 Model Playground":
    df=need_data();st.title("🎮 Model Playground")
    if not st.session_state.models: st.info("Save a model with a prediction function first.");st.stop()
    usable=[m for m in st.session_state.models if m.result is not None and hasattr(m.result,"predict") and m.family in ("OLS","WLS","Logit","Probit","Poisson","Negative Binomial")]
    if not usable: st.info("Playground currently supports formula-based cross-sectional models.");st.stop()
    mn=st.selectbox("Model",[x.name for x in usable]);model=next(x for x in usable if x.name==mn);st.markdown(f"**{model.formula}**");vars_in=[c for c in df.columns if c in model.formula and c!=model.y_name];scenario={};cols=st.columns(2)
    for i,c in enumerate(vars_in):
        with cols[i%2]:
            if pd.api.types.is_numeric_dtype(df[c]):
                lo=float(df[c].quantile(.01));hi=float(df[c].quantile(.99));med=float(df[c].median())
                scenario[c]=st.slider(c,lo,hi,med,key=f"play_{c}") if np.isfinite(lo) and np.isfinite(hi) and lo<hi else med
            else:
                vals=df[c].dropna().astype(str).unique().tolist();scenario[c]=st.selectbox(c,vals,key=f"play_{c}")
    if scenario:
        new=pd.DataFrame([scenario])
        try: pred=np.asarray(model.result.predict(new)).ravel()[0];st.metric("Predicted outcome / probability",f"{pred:.4f}");st.dataframe(new,use_container_width=True,hide_index=True)
        except Exception as e: st.error(f"Prediction unavailable: {e}")

elif page=="🧠 Interpretation":
    st.title("🧠 Interpretation Lab")
    st.caption("Statistically aware interpretation without requiring an external AI API.")
    if not st.session_state.models:
        st.info("Estimate and save at least one model.");st.stop()
    mn=st.selectbox("Model",[m.name for m in st.session_state.models],key="interp_model")
    model=next(m for m in st.session_state.models if m.name==mn)
    terms=model.coef_table["term"].astype(str).tolist()
    nonconst=[t for t in terms if t.lower() not in ("intercept","const")]
    focal=st.selectbox("Focal coefficient",nonconst or terms,key="interp_term")
    st.markdown(interpret_model(model,focal))
    st.subheader("Coefficient table")
    st.dataframe(model.coef_table,use_container_width=True,hide_index=True)
    if model.marginal_effects is not None:
        st.subheader("Average marginal effects")
        st.dataframe(model.marginal_effects,use_container_width=True,hide_index=True)

elif page=="🔷 Fuzzy Lab":
    df=need_data();nums=numeric_columns(df);st.title("🔷 Fuzzy Lab");tabs=st.tabs(["Likert → fuzzy index","TOPSIS"])
    with tabs[0]:
        items=st.multiselect("Likert items (expected scale 1–5)",nums,key="fitems");method=st.radio("Defuzzification",["centroid","weighted"],horizontal=True);new=st.text_input("New index name","fuzzy_index")
        if st.button("Create fuzzy index",disabled=not bool(items)):
            try: out=df.copy();out[new]=fuzzy_index(out,items,method=method);set_df(out,st.session_state.source_name,st.session_state.column_map);st.success(f"Created `{new}`.");st.dataframe(out[[new]].describe(),use_container_width=True)
            except Exception as e: st.exception(e)
    with tabs[1]:
        criteria=st.multiselect("Criteria",nums,key="topcrit")
        if criteria:
            benefits={c:st.checkbox(f"{c}: higher is better",value=True,key=f"benef_{c}") for c in criteria};new=st.text_input("Score name","topsis_score")
            if st.button("Compute TOPSIS"):
                try: out=df.copy();out[new]=topsis(out,criteria,benefit=[benefits[c] for c in criteria]);set_df(out,st.session_state.source_name,st.session_state.column_map);st.success(f"Created `{new}`.");st.dataframe(out[[new]].describe(),use_container_width=True)
                except Exception as e: st.exception(e)

elif page=="📚 Results & Export":
    st.title("📚 Results & Export");models=st.session_state.models
    if not models: st.info("No models saved.");st.stop()
    st.dataframe(pd.DataFrame([{"model":m.name,"family":m.family,"formula":m.formula,**m.metrics} for m in models]),use_container_width=True,hide_index=True)
    selected=st.multiselect("Models for report",[m.name for m in models],default=[m.name for m in models]);chosen=[m for m in models if m.name in selected];c1,c2,c3=st.columns(3)
    if chosen:
        c1.download_button("Excel workbook",models_excel(chosen),"econometrics_lab_results.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        c2.download_button("Word report",docx_report(chosen),"econometrics_lab_report.docx","application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        c3.download_button("PDF report",pdf_report(chosen),"econometrics_lab_report.pdf","application/pdf")
    st.subheader("Manage saved models");delete=st.selectbox("Delete model",["None"]+[m.name for m in models])
    if st.button("Delete") and delete!="None": st.session_state.models=[m for m in models if m.name!=delete];st.rerun()
    if st.button("Clear all models"): st.session_state.models=[];st.rerun()

elif page=="💻 Code Generator":
    df=need_data();st.title("💻 Reproducible Code Generator");family=st.selectbox("Model",["OLS","Logit","Probit","Poisson"]);y=st.selectbox("Dependent variable",df.columns.tolist());x=st.multiselect("Regressors",[c for c in df.columns if c!=y]);language=st.radio("Language",["Python","R","Stata","Gretl"],horizontal=True);code=generate_code(language,family,y,x);st.code(code,language={"Python":"python","R":"r","Stata":"stata","Gretl":"text"}[language])
