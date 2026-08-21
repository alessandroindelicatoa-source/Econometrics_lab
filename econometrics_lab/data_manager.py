from __future__ import annotations
from io import BytesIO
import numpy as np
import pandas as pd
from .utils import clean_columns

def load_bytes(data: bytes, filename: str, sheet_name=0):
    name = filename.lower()
    bio = BytesIO(data)
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(bio)
        except UnicodeDecodeError:
            bio.seek(0)
            df = pd.read_csv(bio, encoding="latin1")
    elif name.endswith((".xlsx",".xls")):
        df = pd.read_excel(bio, sheet_name=sheet_name)
    elif name.endswith(".dta"):
        df = pd.read_stata(bio)
    elif name.endswith(".sav"):
        df = pd.read_spss(bio)
    elif name.endswith(".parquet"):
        df = pd.read_parquet(bio)
    else:
        raise ValueError("Unsupported file format.")
    return clean_columns(df)

def apply_transform(df, operation, columns, new_name=None, value=None):
    out = df.copy()
    if operation == "Log":
        for c in columns:
            out[new_name or f"log_{c}"] = np.where(out[c] > 0, np.log(out[c]), np.nan)
    elif operation == "Log(1+x)":
        for c in columns:
            out[new_name or f"log1p_{c}"] = np.where(out[c] >= 0, np.log1p(out[c]), np.nan)
    elif operation == "Square":
        for c in columns:
            out[new_name or f"{c}_sq"] = out[c]**2
    elif operation == "Standardize":
        for c in columns:
            sd = out[c].std()
            out[new_name or f"z_{c}"] = (out[c]-out[c].mean())/sd if sd else np.nan
    elif operation == "Difference":
        for c in columns:
            out[new_name or f"d_{c}"] = out[c].diff()
    elif operation == "Lag":
        lag = int(value or 1)
        for c in columns:
            out[new_name or f"l{lag}_{c}"] = out[c].shift(lag)
    elif operation == "Winsorize":
        p = float(value or .01)
        for c in columns:
            lo, hi = out[c].quantile([p,1-p])
            out[new_name or f"w_{c}"] = out[c].clip(lo,hi)
    elif operation == "Interaction":
        if len(columns) != 2:
            raise ValueError("Select exactly two variables.")
        out[new_name or f"{columns[0]}_x_{columns[1]}"] = out[columns[0]] * out[columns[1]]
    elif operation == "Dummy threshold":
        if len(columns) != 1:
            raise ValueError("Select one variable.")
        out[new_name or f"d_{columns[0]}"] = (out[columns[0]] >= float(value)).astype(int)
    return out

def missing_summary(df):
    m = pd.DataFrame({
        "variable": df.columns,
        "missing": df.isna().sum().values,
        "missing_%": (df.isna().mean()*100).round(2).values,
        "unique": [df[c].nunique(dropna=True) for c in df.columns],
    })
    return m.sort_values("missing_%", ascending=False)

def handle_missing(df, mode, columns=None):
    out = df.copy()
    columns = columns or out.columns.tolist()
    if mode == "Drop rows":
        return out.dropna(subset=columns)
    if mode == "Mean":
        for c in columns:
            if pd.api.types.is_numeric_dtype(out[c]): out[c] = out[c].fillna(out[c].mean())
    if mode == "Median":
        for c in columns:
            if pd.api.types.is_numeric_dtype(out[c]): out[c] = out[c].fillna(out[c].median())
    if mode == "Mode":
        for c in columns:
            md = out[c].mode(dropna=True)
            if len(md): out[c] = out[c].fillna(md.iloc[0])
    if mode == "Forward fill":
        out[columns] = out[columns].ffill()
    if mode == "Backward fill":
        out[columns] = out[columns].bfill()
    return out
