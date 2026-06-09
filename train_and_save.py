"""
================================================================================
train_and_save.py — Run this ONCE on your machine to pre-train and save models
================================================================================
Usage:
    pip install -r requirements.txt
    python train_and_save.py

This saves trained models to a folder called 'models/'
The Streamlit app will load from there instantly instead of retraining.
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import json, joblib, os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from xgboost import XGBRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==============================================================================
# CONSTANTS (must match app.py)
# ==============================================================================

WFP_PATH   = "wfp_food_prices_nga.csv"
MACRO_PATH = "final_dataset.csv"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

TARGET_COMMODITIES = {
    "Maize (white)":  "maize_white",
    "Maize (yellow)": "maize_yellow",
    "Rice (imported)":"rice_imported",
    "Rice (local)":   "rice_local",
    "Beans (red)":    "beans_red",
    "Beans (white)":  "beans_white",
}
UNIT_DIVISOR = {
    "KG":1.0,"100 KG":100.0,"50 KG":50.0,
    "2.5 KG":2.5,"2.7 KG":2.7,"2.8 KG":2.8,
}
FX_DATA = {
    "2018-01":306.1,"2018-02":306.1,"2018-03":306.1,"2018-04":306.1,
    "2018-05":306.1,"2018-06":306.1,"2018-07":306.1,"2018-08":306.1,
    "2018-09":306.1,"2018-10":306.1,"2018-11":306.3,"2018-12":306.9,
    "2019-01":306.9,"2019-02":306.9,"2019-03":306.9,"2019-04":306.9,
    "2019-05":306.9,"2019-06":306.9,"2019-07":306.9,"2019-08":306.9,
    "2019-09":306.9,"2019-10":306.9,"2019-11":306.9,"2019-12":306.9,
    "2020-01":306.9,"2020-02":306.9,"2020-03":360.0,"2020-04":386.0,
    "2020-05":386.0,"2020-06":381.0,"2020-07":381.0,"2020-08":381.0,
    "2020-09":381.0,"2020-10":379.0,"2020-11":379.0,"2020-12":394.0,
    "2021-01":394.0,"2021-02":394.0,"2021-03":407.0,"2021-04":407.0,
    "2021-05":407.0,"2021-06":411.0,"2021-07":411.0,"2021-08":411.0,
    "2021-09":411.0,"2021-10":414.0,"2021-11":414.0,"2021-12":414.0,
    "2022-01":415.0,"2022-02":416.0,"2022-03":416.0,"2022-04":416.0,
    "2022-05":419.0,"2022-06":422.0,"2022-07":422.0,"2022-08":426.0,
    "2022-09":435.0,"2022-10":442.0,"2022-11":447.0,"2022-12":448.0,
    "2023-01":461.0,"2023-02":462.0,"2023-03":463.0,"2023-04":464.0,
    "2023-05":465.0,"2023-06":769.0,"2023-07":800.0,"2023-08":899.0,
    "2023-09":950.0,"2023-10":990.0,"2023-11":1050.0,"2023-12":1100.0,
    "2024-01":1490.0,
}
STATE_ZONE = {
    "Abia":"South East","Anambra":"South East","Ebonyi":"South East",
    "Enugu":"South East","Imo":"South East",
    "Akwa Ibom":"South South","Bayelsa":"South South","Cross River":"South South",
    "Delta":"South South","Edo":"South South","Rivers":"South South",
    "Ekiti":"South West","Lagos":"South West","Ogun":"South West",
    "Ondo":"South West","Osun":"South West","Oyo":"South West",
    "Adamawa":"North East","Bauchi":"North East","Borno":"North East",
    "Gombe":"North East","Taraba":"North East","Yobe":"North East",
    "Jigawa":"North West","Kaduna":"North West","Kano":"North West",
    "Katsina":"North West","Kebbi":"North West","Sokoto":"North West",
    "Zamfara":"North West",
    "Benue":"North Central","FCT":"North Central","Kogi":"North Central",
    "Kwara":"North Central","Nasarawa":"North Central","Niger":"North Central",
    "Plateau":"North Central",
}
ZONE_MULTIPLIER = {
    "North East":1.08,"North West":1.06,"North Central":1.02,
    "South South":0.99,"South West":0.98,"South East":0.97,
}
KEPT_COMMODITIES = {
    "maize_white":"Maize (White)",
    "maize_yellow":"Maize (Yellow)",
    "beans_red":"Beans (Red)",
    "beans_white":"Beans (White)",
}
TEST_SIZE  = 12
DATE_START = "2018-01-01"
DATE_END   = "2024-01-31"
N_TRIALS   = 60   # increase for better accuracy, decrease for speed


# ==============================================================================
# DATA PIPELINE
# ==============================================================================

def prepare_data():
    print("[1/4] Loading and preparing data...")
    df_wfp   = pd.read_csv(WFP_PATH);   df_wfp["date"]   = pd.to_datetime(df_wfp["date"])
    df_macro = pd.read_csv(MACRO_PATH); df_macro["date"] = pd.to_datetime(df_macro["date"])

    df_wfp = df_wfp[df_wfp["commodity"].isin(TARGET_COMMODITIES.keys())].copy()
    df_wfp["price_per_kg"] = df_wfp.apply(
        lambda r: r["price"] / UNIT_DIVISOR.get(r["unit"], np.nan), axis=1)
    df_wfp = df_wfp[df_wfp["price_per_kg"].notna()]

    df_wfp["year_month"]      = df_wfp["date"].dt.to_period("M")
    df_wfp["commodity_clean"] = df_wfp["commodity"].map(TARGET_COMMODITIES)
    df_macro["year_month"]    = df_macro["date"].dt.to_period("M")

    agg = df_wfp.groupby(
        ["year_month","admin1","market","latitude","longitude","commodity_clean"]
    )["price_per_kg"].mean().reset_index()
    wide = agg.pivot_table(
        index=["year_month","admin1","market","latitude","longitude"],
        columns="commodity_clean", values="price_per_kg", aggfunc="mean"
    ).reset_index()
    wide.columns.name = None

    macro_cols = [c for c in ["cpi","fuel_price","temperature","rainfall"] if c in df_macro.columns]
    df = wide.merge(df_macro[["year_month"]+macro_cols], on="year_month", how="inner")
    df["date"] = df["year_month"].dt.to_timestamp()
    df = df.drop(columns=["year_month"])

    commodity_cols = list(TARGET_COMMODITIES.values())
    mask = pd.Series([True]*len(df), index=df.index)
    for col in commodity_cols+[c for c in ["cpi","fuel_price","rainfall"] if c in df.columns]:
        Q1,Q3 = df[col].quantile(0.25),df[col].quantile(0.75); IQR=Q3-Q1
        mask &= df[col].isna()|df[col].between(Q1-3*IQR,Q3+3*IQR)
    df = df[mask].reset_index(drop=True)

    df = df.sort_values(["market","date"]).reset_index(drop=True)
    for col in commodity_cols:
        df[col] = df.groupby("market")[col].transform(lambda x: x.ffill().bfill())
        if df[col].isnull().any():
            df[col] = df[col].fillna(df.groupby("date")[col].transform("median"))
        df[col] = df[col].fillna(df[col].median())

    fx_df = pd.DataFrame([{"date":pd.to_datetime(k+"-01"),"usd_ngn":v} for k,v in FX_DATA.items()])
    df = df.merge(fx_df, on="date", how="left")
    df["usd_ngn"] = df["usd_ngn"].ffill().bfill()
    df["zone"]      = df["admin1"].map(STATE_ZONE).fillna("National")
    df["state_cpi"] = df["cpi"] * df["zone"].map(ZONE_MULTIPLIER).fillna(1.0)

    df = df[(df["date"]>=DATE_START)&(df["date"]<=DATE_END)].copy()
    df = df.sort_values(["market","date"]).reset_index(drop=True)

    feat_cols = [c for c in ["cpi","state_cpi","fuel_price","temperature","rainfall","usd_ngn"] if c in df.columns]
    for c in feat_cols: df[f"{c}_raw"] = df[c]
    df[feat_cols] = MinMaxScaler().fit_transform(df[feat_cols])

    print(f"    Data ready: {df.shape[0]:,} rows | {df['date'].min().date()} → {df['date'].max().date()}")
    return df


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================

def get_base_features(df):
    return [c for c in ["cpi","state_cpi","fuel_price","temperature","rainfall","usd_ngn"] if c in df.columns]

def build_features(df, col):
    bf  = get_base_features(df)
    agg = {c:pd.NamedAgg(column=c,aggfunc="mean") for c in bf}
    agg["price"] = pd.NamedAgg(column=col,aggfunc="mean")
    feat = df.groupby("date").agg(**agg).reset_index().sort_values("date").reset_index(drop=True)
    feat["month"]=feat["date"].dt.month.astype(float)
    feat["year"] =feat["date"].dt.year.astype(float)
    for lag in [1,2,3,6,12]: feat[f"lag{lag}"]=feat["price"].shift(lag)
    feat["roll3"]  =feat["price"].shift(1).rolling(3).mean()
    feat["roll6"]  =feat["price"].shift(1).rolling(6).mean()
    feat["roll12"] =feat["price"].shift(1).rolling(12).mean()
    feat["fx_mom3"]=feat["usd_ngn"].diff(3)
    feat=feat.dropna().reset_index(drop=True)
    num=[c for c in feat.columns if c!="date"]; feat[num]=feat[num].astype(float)
    lag_c=[c for c in feat.columns if "lag" in c or "roll" in c]
    feat[lag_c]=MinMaxScaler().fit_transform(feat[lag_c])
    feat[["month","year"]]=MinMaxScaler().fit_transform(feat[["month","year"]])
    return feat

def get_xcols(feat):
    lr=["lag1","lag2","lag3","lag6","lag12","roll3","roll6","roll12","fx_mom3","arima_resid"]
    return [c for c in get_base_features(feat)+["month","year"]+lr if c in feat.columns]

def safe_eval(yt, yp):
    yt,yp=np.array(yt,dtype=float),np.array(yp,dtype=float)
    mask=np.isfinite(yt)&np.isfinite(yp)&(yt>0)
    if mask.sum()<2: return dict(MAE=np.nan,RMSE=np.nan,R2=np.nan,MAPE=np.nan,accuracy=np.nan)
    yt,yp=yt[mask],yp[mask]
    mape=float(np.mean(np.abs((yt-yp)/yt))*100)
    return dict(MAE=round(float(mean_absolute_error(yt,yp)),2),
                RMSE=round(float(np.sqrt(mean_squared_error(yt,yp))),2),
                R2=round(float(r2_score(yt,yp)),4),
                MAPE=round(mape,2),accuracy=round(100-mape,2))


# ==============================================================================
# TRAINING
# ==============================================================================

def train_and_save(df):
    print(f"[2/4] Training hybrid models ({N_TRIALS} Optuna trials per commodity)...")
    all_metrics = {}

    for col, label in KEPT_COMMODITIES.items():
        print(f"\n  → {label}")
        ts = (df.groupby("date")[col].mean()
                .reset_index().set_index("date")
                .asfreq("MS",method="ffill")[col])
        ts_tr, ts_te = ts.iloc[:-TEST_SIZE], ts.iloc[-TEST_SIZE:]

        # ARIMA
        adf_p=adfuller(ts_tr.dropna())[1]; d=1 if adf_p>=0.05 else 0
        try:
            am       = ARIMA(ts_tr,order=(2,d,2)).fit()
            ar_fit   = am.fittedvalues
            ar_pred  = am.forecast(steps=TEST_SIZE).values
            ar_resid = (ts_tr-ar_fit).fillna(0)
        except:
            am=None; ar_pred=np.full(TEST_SIZE,float(ts_tr.mean()))
            ar_resid=pd.Series(np.zeros(len(ts_tr)),index=ts_tr.index)

        # Build features + inject ARIMA residuals
        feat=build_features(df,col)
        rd=ar_resid.reset_index(); rd.columns=["date","arima_resid"]
        rd["date"]=pd.to_datetime(rd["date"])
        feat=feat.merge(rd,on="date",how="left")
        feat["arima_resid"]=feat["arima_resid"].fillna(0).astype(float)
        xcols=get_xcols(feat)
        X,y=feat[xcols].astype(float),feat["price"].astype(float)
        sp=len(feat)-TEST_SIZE

        # Optuna tuning
        def objective(trial):
            p=dict(n_estimators=trial.suggest_int("n_estimators",200,800),
                   learning_rate=trial.suggest_float("learning_rate",0.005,0.1,log=True),
                   max_depth=trial.suggest_int("max_depth",3,6),
                   subsample=trial.suggest_float("subsample",0.6,1.0),
                   colsample_bytree=trial.suggest_float("colsample_bytree",0.5,1.0),
                   min_child_weight=trial.suggest_int("min_child_weight",1,5),
                   reg_alpha=trial.suggest_float("reg_alpha",1e-4,1.0,log=True),
                   random_state=42,verbosity=0)
            m=XGBRegressor(**p)
            m.fit(X.iloc[:sp],y.iloc[:sp],eval_set=[(X.iloc[sp:],y.iloc[sp:])],verbose=False)
            return mean_absolute_error(y.iloc[sp:],m.predict(X.iloc[sp:]))

        study=optuna.create_study(direction="minimize")
        study.optimize(objective,n_trials=N_TRIALS,show_progress_bar=False)
        best_p={**study.best_params,"random_state":42,"verbosity":0}

        # Eval on test set
        xgb_test=XGBRegressor(**best_p)
        xgb_test.fit(X.iloc[:sp],y.iloc[:sp],
                     eval_set=[(X.iloc[sp:],y.iloc[sp:])],verbose=False)
        xp=xgb_test.predict(X.iloc[sp:])
        ar_m=safe_eval(ts_te.values,ar_pred)
        xg_m=safe_eval(y.iloc[sp:].values,xp)

        if (not np.isnan(ar_m["MAPE"]) and not np.isnan(xg_m["MAPE"])
                and ar_m["MAPE"]>0 and xg_m["MAPE"]>0):
            w_ar=(1/ar_m["MAPE"])/(1/ar_m["MAPE"]+1/xg_m["MAPE"]); w_xg=1-w_ar
        else: w_ar,w_xg=0.4,0.6

        n=min(len(ar_pred),len(xp))
        hy=w_ar*ar_pred[:n]+w_xg*xp[:n]
        hy_m=safe_eval(ts_te.values[:n],hy)

        print(f"    ARIMA {ar_m['accuracy']:.1f}%  |  XGBoost {xg_m['accuracy']:.1f}%  |  Hybrid {hy_m['accuracy']:.1f}%")

        # Retrain on ALL data
        xgb_full=XGBRegressor(**best_p)
        xgb_full.fit(X.astype(float),y.astype(float),verbose=False)

        # Save model artefacts
        model_bundle = dict(
            xgb       = xgb_full,
            arima     = am,
            xcols     = xcols,
            last_row  = feat.iloc[-1][xcols].astype(float).to_dict(),
            w_ar      = w_ar,
            w_xg      = w_xg,
            label     = label,
            mae       = hy_m["MAE"],
            last_ts   = ts,
            actual    = ts_te.values.tolist(),
            hybrid_pred=hy.tolist(),
            test_dates = [str(d) for d in ts_te.index],
        )
        joblib.dump(model_bundle, MODELS_DIR / f"{col}.joblib")

        all_metrics[col] = dict(
            label=label, arima=ar_m, xgb=xg_m, hybrid=hy_m,
            w_ar=round(w_ar,3), w_xg=round(w_xg,3)
        )

    # Save metrics separately (small JSON — fast to load)
    with open(MODELS_DIR/"metrics.json","w") as f:
        json.dump(all_metrics,f,indent=2,default=str)

    print(f"\n[3/4] Models saved to '{MODELS_DIR}/' folder")
    return all_metrics


# ==============================================================================
# SUMMARY
# ==============================================================================

def print_summary(metrics):
    print("\n[4/4] Training complete!\n")
    print(f"{'Commodity':<20} {'Accuracy':>10} {'MAPE':>8} {'MAE (₦/kg)':>12}  {'Weights':>16}")
    print("─"*72)
    for col, m in metrics.items():
        h = m["hybrid"]
        print(f"  {m['label']:<18} {h['accuracy']:>9.2f}% {h['MAPE']:>7.2f}% "
              f"{h['MAE']:>11,.0f}  AR={m['w_ar']:.2f} XGB={m['w_xg']:.2f}")

    print(f"""
Files saved in models/ folder:
  maize_white.joblib
  maize_yellow.joblib
  beans_red.joblib
  beans_white.joblib
  metrics.json

Next step:
  Upload the 'models/' folder to your GitHub repo,
  then deploy app.py on Streamlit — it will load
  instantly without retraining.
""")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("="*60)
    print("  Nigeria Food Price — Model Training Script")
    print("="*60)
    df      = prepare_data()
    metrics = train_and_save(df)
    print_summary(metrics)
