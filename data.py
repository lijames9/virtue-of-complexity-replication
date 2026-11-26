import numpy as np
import pandas as pd

def load_crsp(filepath):
    '''
    Loads CRSP data into Pandas dataframe.

    csv filepath --> Pandas df
    '''
    df = pd.read_csv(filepath, parse_dates=["mthcaldt"])
    df["ym"] = df["mthcaldt"].dt.to_period("M")

    return df.set_index("ym").drop(columns="mthcaldt")

def load_pred(filepath, sheet = "Monthly"):
    '''
    Loads Goyal-Welch predictor data into Pandas dataframe.

    excel filepath --> Pandas df
    '''
    df = pd.read_excel(filepath, sheet_name=sheet)
    df["ym"] = pd.PeriodIndex(df["yyyymm"].astype(str), freq="M")

    return df.set_index("ym").drop(columns="yyyymm")

def build_predictors(gw, crsp):
    """
    Return a DataFrame of the 15 raw predictors G_t and the excess return.
    """

    df = crsp.join(gw, how="inner")
    out = pd.DataFrame(index=df.index)
    out["dp"] = np.log(df["D12"]) - np.log(df["Index"])
    out["dy"] = np.log(df["D12"]) - np.log(df["Index"].shift(1))
    out["ep"] = np.log(df["E12"]) - np.log(df["Index"])
    out["de"] = np.log(df["D12"]) - np.log(df["E12"])
    out["tms"] = df["lty"] - df["tbl"]
    out["dfy"] = df["BAA"] - df["AAA"]
    out["dfr"] = df["corpr"] - df["ltr"]

    for c in ["svar", "b/m", "ntis", "tbl", "lty", "ltr", "infl"]:
        out[c] = df[c]
    out["exret"] = df["vwretd"] - df["Rfree"]
    out["mkt_lag"] = out["exret"] # month-t return, used to predict t+1
    return out

def clean(df, start="1930-01", end="2020-12", vol_window=12, min_obs=36):
    '''
    Standardizes predictors and returns, aligns forecast timing, and trims the sample.

    Returns:
    G : pd.DataFrame
        Standardized predictors known at end of month t (15 columns).
    R : pd.Series
        Standardized excess return for month t+1, aligned to G's index.
    '''
    preds = [c for c in df.columns if c not in ("exret", "mkt_lag")]

    # Step 2a: return / trailing 12-month uncentered vol (months t-12..t-1, excludes t)
    R = df["exret"]
    vol = np.sqrt((R**2).rolling(vol_window).mean().shift(1))
    R_std = R / vol

    # Step 2b: predictors / expanding-window std (through month t, min 36 obs)
    G = df[preds] / df[preds].expanding(min_periods=min_obs).std()
    G["mkt_lag"] = R_std  # standardized month-t return as the 15th predictor

    # Step 3: align G_t with R_{t+1}
    data = G.assign(R_next=R_std.shift(-1)).dropna()

    # Step 4: trim to sample period
    data = data.loc[start:end]
    return data.drop(columns="R_next"), data["R_next"]

def get_data(crsp_path="CRSP_Stock_Index_Data.csv",
             gw_path="PredictorData2025.xlsx",
             start="1930-01", end="2020-12"):
    '''
    Full pipeline: load, build predictors, standardize, align.

    Returns
    -------
    G : pd.DataFrame   (months x 15) predictors at t
    R : pd.Series      standardized excess return at t+1
    '''
    crsp = load_crsp(crsp_path)
    gw = load_pred(gw_path)
    return clean(build_predictors(gw, crsp), start=start, end=end)