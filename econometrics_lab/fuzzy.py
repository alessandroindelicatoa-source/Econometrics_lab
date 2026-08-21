from __future__ import annotations
import numpy as np
import pandas as pd

DEFAULT_TFN = {1:(0.00,0.00,0.25),2:(0.00,0.25,0.50),3:(0.25,0.50,0.75),4:(0.50,0.75,1.00),5:(0.75,1.00,1.00)}

def fuzzify_likert(series, mapping=None):
    vals=series.map(mapping or DEFAULT_TFN)
    return pd.DataFrame(vals.tolist(),index=series.index,columns=["l","m","u"])

def defuzzify(tfn_df, method="centroid"):
    if method=="weighted": return (tfn_df["l"]+2*tfn_df["m"]+tfn_df["u"])/4
    return (tfn_df["l"]+tfn_df["m"]+tfn_df["u"])/3

def fuzzy_index(df, items, weights=None, mapping=None, method="centroid"):
    weights=np.asarray(weights if weights is not None else np.ones(len(items)),dtype=float);weights=weights/weights.sum()
    scores=[defuzzify(fuzzify_likert(df[item],mapping),method).rename(item) for item in items]
    return (pd.concat(scores,axis=1)*weights).sum(axis=1)

def topsis(df, criteria, weights=None, benefit=None):
    X=df[criteria].astype(float).copy(); weights=np.asarray(weights if weights is not None else np.ones(len(criteria)),dtype=float);weights=weights/weights.sum()
    benefit=benefit or [True]*len(criteria); denom=np.sqrt((X**2).sum(axis=0)).replace(0,np.nan); V=(X/denom)*weights
    best=[];worst=[]
    for j,c in enumerate(criteria):
        best.append(V[c].max() if benefit[j] else V[c].min());worst.append(V[c].min() if benefit[j] else V[c].max())
    best=np.asarray(best);worst=np.asarray(worst);db=np.sqrt(((V-best)**2).sum(axis=1));dw=np.sqrt(((V-worst)**2).sum(axis=1))
    return dw/(db+dw)
