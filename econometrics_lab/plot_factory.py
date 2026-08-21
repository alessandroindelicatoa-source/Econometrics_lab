from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

def style(fig, title=None):
    fig.update_layout(title=title,template="plotly_white",margin=dict(l=20,r=20,t=55,b=20),legend_title_text="",hovermode="closest")
    return fig

def histogram(df,var,color=None,bins=30,marginal="box"):
    return style(px.histogram(df,x=var,color=color,nbins=bins,marginal=marginal,opacity=.78),f"Distribution of {var}")

def box_violin(df,y,x=None,color=None,kind="Box"):
    fig=px.box(df,x=x,y=y,color=color,points="outliers") if kind=="Box" else px.violin(df,x=x,y=y,color=color,box=True,points="outliers")
    return style(fig,f"{kind}: {y}")

def scatter(df,x,y,color=None,size=None,trendline=None):
    kwargs={}
    if trendline=="OLS": kwargs["trendline"]="ols"
    elif trendline=="LOWESS": kwargs["trendline"]="lowess"
    return style(px.scatter(df,x=x,y=y,color=color,size=size,opacity=.72,**kwargs),f"{y} vs {x}")

def correlation_heatmap(df,variables,method="pearson"):
    corr=df[variables].corr(method=method)
    return style(px.imshow(corr,text_auto=".2f",aspect="auto",zmin=-1,zmax=1,color_continuous_scale="RdBu_r"),f"Correlation matrix ({method})")

def scatter_matrix(df,variables,color=None):
    fig=px.scatter_matrix(df,dimensions=variables,color=color)
    fig.update_traces(diagonal_visible=False,showupperhalf=False)
    return style(fig,"Scatter matrix")

def line_plot(df,x,y,color=None):
    return style(px.line(df.sort_values(x),x=x,y=y,color=color,markers=True),f"{y} over {x}")

def group_means(df,y,group):
    g=df.groupby(group,dropna=False)[y].agg(["mean","std","count"]).reset_index()
    g["se"]=g["std"]/np.sqrt(g["count"].clip(lower=1)); g["ci95"]=1.96*g["se"]
    fig=go.Figure(go.Bar(x=g[group].astype(str),y=g["mean"],error_y=dict(type="data",array=g["ci95"])))
    fig.update_xaxes(title=group);fig.update_yaxes(title=f"Mean {y}")
    return style(fig,f"Mean {y} by {group} (95% CI)")

def missingness(df):
    vals=(df.isna().mean()*100).sort_values(ascending=True)
    fig=go.Figure(go.Bar(x=vals.values,y=vals.index,orientation="h"));fig.update_xaxes(title="Missing (%)")
    return style(fig,"Missing data by variable")

def coefficient_plot(model_result, include_intercept=False):
    tab=model_result.coef_table.copy()
    if not include_intercept: tab=tab[~tab["term"].str.lower().isin(["intercept","const"])]
    fig=go.Figure(go.Scatter(x=tab["coef"],y=tab["term"],mode="markers",error_x=dict(type="data",symmetric=False,array=tab["ci_high"]-tab["coef"],arrayminus=tab["coef"]-tab["ci_low"])))
    fig.add_vline(x=0,line_dash="dash");fig.update_xaxes(title="Estimate (95% CI)")
    return style(fig,f"Coefficient plot — {model_result.name}")

def actual_predicted(df,model_result):
    idx=model_result.fitted.index; y=df.loc[idx,model_result.y_name]
    fig=px.scatter(x=y,y=model_result.fitted,labels={"x":"Actual","y":"Predicted"})
    lo=float(np.nanmin([y.min(),model_result.fitted.min()])); hi=float(np.nanmax([y.max(),model_result.fitted.max()]))
    fig.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode="lines",name="45°"))
    return style(fig,f"Actual vs predicted — {model_result.name}")

def residual_fitted(model_result):
    fig=px.scatter(x=model_result.fitted,y=model_result.residuals,labels={"x":"Fitted","y":"Residual"});fig.add_hline(y=0,line_dash="dash")
    return style(fig,f"Residuals vs fitted — {model_result.name}")

def residual_hist(model_result):
    return style(px.histogram(x=model_result.residuals,nbins=35,marginal="box",labels={"x":"Residual"}),f"Residual distribution — {model_result.name}")

def qq_plot(model_result):
    r=np.asarray(model_result.residuals.dropna()); osm,osr=stats.probplot(r,dist="norm",fit=False); slope,intercept=np.polyfit(osm,osr,1)
    fig=go.Figure();fig.add_trace(go.Scatter(x=osm,y=osr,mode="markers",name="Residuals"));fig.add_trace(go.Scatter(x=osm,y=slope*np.asarray(osm)+intercept,mode="lines",name="Reference"))
    fig.update_xaxes(title="Theoretical quantiles");fig.update_yaxes(title="Sample quantiles")
    return style(fig,f"Q–Q plot — {model_result.name}")

def roc_plot(df,model_result):
    from sklearn.metrics import roc_curve,auc
    idx=model_result.fitted.index; y=np.asarray(df.loc[idx,model_result.y_name]); p=np.asarray(model_result.fitted)
    fpr,tpr,_=roc_curve(y,p); a=auc(fpr,tpr)
    fig=go.Figure();fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name=f"AUC={a:.3f}"));fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random"))
    fig.update_xaxes(title="False positive rate",range=[0,1]);fig.update_yaxes(title="True positive rate",range=[0,1])
    return style(fig,f"ROC — {model_result.name}")

def marginal_effects_plot(model_result):
    m=model_result.marginal_effects.copy()
    effect_col=next((c for c in m.columns if "dy/dx" in str(c)),None); se_col=next((c for c in m.columns if "Std. Err." in str(c)),None)
    if effect_col is None: raise ValueError("No marginal-effects column found.")
    fig=go.Figure(go.Scatter(x=m[effect_col],y=m["term"],mode="markers",error_x=dict(type="data",array=1.96*m[se_col]) if se_col else None))
    fig.add_vline(x=0,line_dash="dash");fig.update_xaxes(title="Average marginal effect (95% CI)")
    return style(fig,f"Marginal effects — {model_result.name}")

def model_comparison(models, term):
    rows=[]
    for m in models:
        t=m.coef_table[m.coef_table["term"]==term]
        if len(t):
            r=t.iloc[0];rows.append({"model":m.name,"coef":r.coef,"low":r.ci_low,"high":r.ci_high})
    d=pd.DataFrame(rows)
    fig=go.Figure(go.Scatter(x=d["coef"],y=d["model"],mode="markers",error_x=dict(type="data",symmetric=False,array=d["high"]-d["coef"],arrayminus=d["coef"]-d["low"])))
    fig.add_vline(x=0,line_dash="dash");fig.update_xaxes(title=f"Coefficient: {term} (95% CI)")
    return style(fig,"Model comparison")

def did_trends(df,y,treat,time):
    g=df.groupby([time,treat])[y].mean().reset_index()
    return style(px.line(g,x=time,y=y,color=treat,markers=True),f"DiD trends: {y}")
