from __future__ import annotations
import re
import numpy as np
import pandas as pd

def clean_name(name: str) -> str:
    s = str(name).strip().lower()
    replacements = {
        "á":"a","à":"a","ä":"a","â":"a","ã":"a",
        "é":"e","è":"e","ë":"e","ê":"e",
        "í":"i","ì":"i","ï":"i","î":"i",
        "ó":"o","ò":"o","ö":"o","ô":"o","õ":"o",
        "ú":"u","ù":"u","ü":"u","û":"u",
        "ñ":"n","ç":"c"
    }
    for a,b in replacements.items():
        s = s.replace(a,b)
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "var"
    if s[0].isdigit():
        s = "v_" + s
    return s

def clean_columns(df: pd.DataFrame):
    mapping = {}
    used = set()
    new_cols = []
    for c in df.columns:
        base = clean_name(c)
        candidate = base
        i = 2
        while candidate in used:
            candidate = f"{base}_{i}"
            i += 1
        used.add(candidate)
        mapping[str(c)] = candidate
        new_cols.append(candidate)
    out = df.copy()
    out.columns = new_cols
    return out, mapping

def numeric_columns(df: pd.DataFrame):
    return df.select_dtypes(include=np.number).columns.tolist()

def categorical_columns(df: pd.DataFrame):
    return [c for c in df.columns if c not in numeric_columns(df)]

def formula_term(var: str, categoricals: list[str] | None = None) -> str:
    categoricals = categoricals or []
    return f"C({var})" if var in categoricals else var

def build_formula(y: str, x: list[str], categoricals=None, interactions=None, intercept=True):
    categoricals = categoricals or []
    interactions = interactions or []
    terms = [formula_term(v, categoricals) for v in x]
    for a,b in interactions:
        terms.append(f"{formula_term(a,categoricals)}:{formula_term(b,categoricals)}")
    rhs = " + ".join(terms) if terms else "1"
    if not intercept:
        rhs = "0 + " + rhs
    return f"{y} ~ {rhs}"

def significance_stars(p):
    try:
        if p < .01: return "***"
        if p < .05: return "**"
        if p < .10: return "*"
    except Exception:
        pass
    return ""

def dataframe_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        row = {
            "variable": c,
            "dtype": str(s.dtype),
            "n": int(s.notna().sum()),
            "missing": int(s.isna().sum()),
            "missing_%": round(float(s.isna().mean()*100), 2),
            "unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            row.update({
                "mean": s.mean(),
                "sd": s.std(),
                "min": s.min(),
                "median": s.median(),
                "max": s.max()
            })
        rows.append(row)
    return pd.DataFrame(rows)
