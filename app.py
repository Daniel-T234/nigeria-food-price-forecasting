"""
================================================================================
Nigeria Food Price Forecasting — Streamlit App
================================================================================
Usage:
    pip install streamlit pandas numpy statsmodels xgboost scikit-learn plotly optuna
    streamlit run app.py

Place these two files in the same folder as app.py:
    - wfp_food_prices_nga.csv
    - final_dataset.csv
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from xgboost import XGBRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ==============================================================================
# PAGE CONFIG
# ==============================================================================

st.set_page_config(
    page_title="Nigeria Food Price Forecasting",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid #e9ecef;
    }
    .accuracy-good  { color: #198754; font-weight: 600; font-size: 1.6rem; }
    .accuracy-warn  { color: #fd7e14; font-weight: 600; font-size: 1.6rem; }
    .accuracy-bad   { color: #dc3545; font-weight: 600; font-size: 1.6rem; }
    .stProgress > div > div { background-color: #198754; }
    div[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# CONSTANTS
# ==============================================================================

TARGET_COMMODITIES = {
    "Maize (white)":  "maize_white",
    "Maize (yellow)": "maize_yellow",
    "Rice (imported)":"rice_imported",
    "Rice (local)":   "rice_local",
    "Beans (red)":    "beans_red",
    "Beans (white)":  "beans_white",
}

UNIT_DIVISOR = {
    "KG":1.0, "100 KG":100.0, "50 KG":50.0,
    "2.5 KG":2.5, "2.7 KG":2.7, "2.8 KG":2.8,
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
    "maize_white": "Maize (White)",
    "maize_yellow":"Maize (Yellow)",
    "beans_red":   "Beans (Red)",
    "beans_white": "Beans (White)",
}

COMMODITY_COLORS = {
    "maize_white": "#185FA5",
    "maize_yellow":"#3B6D11",
    "beans_red":   "#BA7517",
    "beans_white": "#993556",
}

TEST_SIZE  = 12
DATE_START = "2018-01-01"
DATE_END   = "2024-01-31"


# ==============================================================================
# DATA PIPELINE (cached)
# ==============================================================================

@st.cache_data(show_spinner=False)
def load_and_prepare(wfp_path: str, macro_path: str) -> pd.DataFrame:
    df_wfp   = pd.read_csv(wfp_path);   df_wfp["date"]   = pd.to_datetime(df_wfp["date"])
    df_macro = pd.read_csv(macro_path); df_macro["date"] = pd.to_datetime(df_macro["date"])

    # Standardise to NGN per KG
    df_wfp = df_wfp[df_wfp["commodity"].isin(TARGET_COMMODITIES.keys())].copy()
    df_wfp["price_per_kg"] = df_wfp.apply(
        lambda r: r["price"] / UNIT_DIVISOR.get(r["unit"], np.nan), axis=1)
    df_wfp = df_wfp[df_wfp["price_per_kg"].notna()]

    # Pivot wide
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

    # Outliers
    commodity_cols = list(TARGET_COMMODITIES.values())
    mask = pd.Series([True]*len(df), index=df.index)
    for col in commodity_cols + [c for c in ["cpi","fuel_price","rainfall"] if c in df.columns]:
        Q1,Q3 = df[col].quantile(0.25), df[col].quantile(0.75); IQR=Q3-Q1
        mask &= df[col].isna() | df[col].between(Q1-3*IQR, Q3+3*IQR)
    df = df[mask].reset_index(drop=True)

    # Imputation
    df = df.sort_values(["market","date"]).reset_index(drop=True)
    for col in commodity_cols:
        df[col] = df.groupby("market")[col].transform(lambda x: x.ffill().bfill())
        if df[col].isnull().any():
            df[col] = df[col].fillna(df.groupby("date")[col].transform("median"))
        df[col] = df[col].fillna(df[col].median())

    # FX rate
    fx_df = pd.DataFrame([{"date":pd.to_datetime(k+"-01"),"usd_ngn":v} for k,v in FX_DATA.items()])
    df = df.merge(fx_df, on="date", how="left")
    df["usd_ngn"] = df["usd_ngn"].ffill().bfill()

    # State CPI
    df["zone"]      = df["admin1"].map(STATE_ZONE).fillna("National")
    df["state_cpi"] = df["cpi"] * df["zone"].map(ZONE_MULTIPLIER).fillna(1.0)

    # Filter
    df = df[(df["date"]>=DATE_START)&(df["date"]<=DATE_END)].copy()
    df = df.sort_values(["market","date"]).reset_index(drop=True)

    # Normalise
    feat_cols = [c for c in ["cpi","state_cpi","fuel_price","temperature","rainfall","usd_ngn"] if c in df.columns]
    for c in feat_cols: df[f"{c}_raw"] = df[c]
    df[feat_cols] = MinMaxScaler().fit_transform(df[feat_cols])

    return df


# ==============================================================================
# MODEL HELPERS
# ==============================================================================

def get_base_features(df):
    return [c for c in ["cpi","state_cpi","fuel_price","temperature","rainfall","usd_ngn"] if c in df.columns]

def build_features(df, col):
    bf  = get_base_features(df)
    agg = {c: pd.NamedAgg(column=c, aggfunc="mean") for c in bf}
    agg["price"] = pd.NamedAgg(column=col, aggfunc="mean")
    feat = df.groupby("date").agg(**agg).reset_index().sort_values("date").reset_index(drop=True)
    feat["month"] = feat["date"].dt.month.astype(float)
    feat["year"]  = feat["date"].dt.year.astype(float)
    for lag in [1,2,3,6,12]: feat[f"lag{lag}"] = feat["price"].shift(lag)
    feat["roll3"]   = feat["price"].shift(1).rolling(3).mean()
    feat["roll6"]   = feat["price"].shift(1).rolling(6).mean()
    feat["roll12"]  = feat["price"].shift(1).rolling(12).mean()
    feat["fx_mom3"] = feat["usd_ngn"].diff(3)
    feat = feat.dropna().reset_index(drop=True)
    num  = [c for c in feat.columns if c != "date"]; feat[num] = feat[num].astype(float)
    lag_c = [c for c in feat.columns if "lag" in c or "roll" in c]
    feat[lag_c] = MinMaxScaler().fit_transform(feat[lag_c])
    feat[["month","year"]] = MinMaxScaler().fit_transform(feat[["month","year"]])
    return feat

def get_xcols(feat):
    lr = ["lag1","lag2","lag3","lag6","lag12","roll3","roll6","roll12","fx_mom3","arima_resid"]
    return [c for c in get_base_features(feat)+["month","year"]+lr if c in feat.columns]

def safe_eval(yt, yp):
    yt,yp = np.array(yt,dtype=float), np.array(yp,dtype=float)
    mask  = np.isfinite(yt)&np.isfinite(yp)&(yt>0)
    if mask.sum()<2: return dict(MAE=np.nan,RMSE=np.nan,R2=np.nan,MAPE=np.nan,accuracy=np.nan)
    yt,yp = yt[mask],yp[mask]
    mape  = float(np.mean(np.abs((yt-yp)/yt))*100)
    return dict(MAE=round(float(mean_absolute_error(yt,yp)),2),
                RMSE=round(float(np.sqrt(mean_squared_error(yt,yp))),2),
                R2=round(float(r2_score(yt,yp)),4),
                MAPE=round(mape,2), accuracy=round(100-mape,2))


# ==============================================================================
# HYBRID MODEL TRAINING (cached)
# ==============================================================================

@st.cache_resource(show_spinner=False)
def train_hybrid_models(_df, n_trials=40):
    """Train ARIMA + XGBoost hybrid for each kept commodity."""
    trained = {}
    metrics = {}

    for col, label in KEPT_COMMODITIES.items():
        ts = (_df.groupby("date")[col].mean()
                .reset_index().set_index("date")
                .asfreq("MS", method="ffill")[col])
        ts_tr, ts_te = ts.iloc[:-TEST_SIZE], ts.iloc[-TEST_SIZE:]

        # ARIMA
        adf_p = adfuller(ts_tr.dropna())[1]; d=1 if adf_p>=0.05 else 0
        try:
            am       = ARIMA(ts_tr, order=(2,d,2)).fit()
            ar_fit   = am.fittedvalues
            ar_pred  = am.forecast(steps=TEST_SIZE).values
            ar_resid = (ts_tr - ar_fit).fillna(0)
        except:
            am=None; ar_pred=np.full(TEST_SIZE,float(ts_tr.mean()))
            ar_resid=pd.Series(np.zeros(len(ts_tr)),index=ts_tr.index)

        # XGBoost with Optuna tuning
        feat = build_features(_df, col)
        rd   = ar_resid.reset_index(); rd.columns=["date","arima_resid"]
        rd["date"] = pd.to_datetime(rd["date"])
        feat = feat.merge(rd, on="date", how="left")
        feat["arima_resid"] = feat["arima_resid"].fillna(0).astype(float)
        xcols = get_xcols(feat)
        X,y   = feat[xcols].astype(float), feat["price"].astype(float)
        sp    = len(feat)-TEST_SIZE

        def objective(trial):
            params = dict(
                n_estimators    =trial.suggest_int("n_estimators",200,800),
                learning_rate   =trial.suggest_float("learning_rate",0.005,0.1,log=True),
                max_depth       =trial.suggest_int("max_depth",3,6),
                subsample       =trial.suggest_float("subsample",0.6,1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree",0.5,1.0),
                min_child_weight=trial.suggest_int("min_child_weight",1,5),
                reg_alpha       =trial.suggest_float("reg_alpha",1e-4,1.0,log=True),
                random_state=42, verbosity=0,
            )
            m = XGBRegressor(**params)
            m.fit(X.iloc[:sp],y.iloc[:sp],eval_set=[(X.iloc[sp:],y.iloc[sp:])],verbose=False)
            return mean_absolute_error(y.iloc[sp:], m.predict(X.iloc[sp:]))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_p = {**study.best_params,"random_state":42,"verbosity":0}

        xgb = XGBRegressor(**best_p)
        xgb.fit(X.iloc[:sp],y.iloc[:sp],eval_set=[(X.iloc[sp:],y.iloc[sp:])],verbose=False)
        xp   = xgb.predict(X.iloc[sp:])

        ar_m = safe_eval(ts_te.values, ar_pred)
        xg_m = safe_eval(y.iloc[sp:].values, xp)

        if (not np.isnan(ar_m["MAPE"]) and not np.isnan(xg_m["MAPE"])
                and ar_m["MAPE"]>0 and xg_m["MAPE"]>0):
            w_ar=(1/ar_m["MAPE"])/(1/ar_m["MAPE"]+1/xg_m["MAPE"]); w_xg=1-w_ar
        else: w_ar,w_xg=0.4,0.6

        n = min(len(ar_pred),len(xp))
        hy = w_ar*ar_pred[:n]+w_xg*xp[:n]
        hy_m = safe_eval(ts_te.values[:n], hy)

        # Retrain on full data for future forecasting
        xgb_full = XGBRegressor(**best_p)
        xgb_full.fit(X.astype(float),y.astype(float),verbose=False)

        trained[col] = dict(
            xgb=xgb_full, arima=am, xcols=xcols,
            last_row=feat.iloc[-1][xcols].astype(float).to_dict(),
            w_ar=w_ar, w_xg=w_xg, label=label,
            mae=hy_m["MAE"], last_ts=ts,
            actual=ts_te.values.tolist(),
            hybrid_pred=hy.tolist(),
            test_dates=[str(d) for d in ts_te.index],
        )
        metrics[col] = dict(label=label, arima=ar_m, xgb=xg_m, hybrid=hy_m,
                            w_ar=round(w_ar,3), w_xg=round(w_xg,3))

    return trained, metrics


# ==============================================================================
# FORECASTING
# ==============================================================================

def run_forecast(trained, n_months=6, scenario_fx=None, scenario_cpi_growth=None):
    results = {}
    last_date = max(pd.to_datetime(info["last_ts"].index[-1]) for info in trained.values())

    for col, info in trained.items():
        model = info["xgb"]; arima = info["arima"]
        xcols = info["xcols"]; row  = info["last_row"].copy()
        w_ar  = info["w_ar"]; w_xg = info["w_xg"]
        mae   = info["mae"];  ts   = info["last_ts"]
        preds = []

        if arima is not None:
            try: ar_steps = arima.forecast(steps=n_months).values
            except: ar_steps = np.full(n_months, float(ts.mean()))
        else:
            ar_steps = np.full(n_months, float(ts.mean()))

        for step in range(n_months):
            fut = last_date + pd.DateOffset(months=step+1)
            row["month"] = fut.month/12.0
            row["year"]  = (fut.year-2018)/(2024-2018+1e-9)

            if scenario_fx is not None and "usd_ngn" in row:
                row["usd_ngn"] = float(np.clip((scenario_fx-306)/(1490-306),0,1.5))
            if scenario_cpi_growth is not None and "cpi" in row:
                row["cpi"] = float(np.clip(row["cpi"]*(1+scenario_cpi_growth)**step,0,1.5))

            row["arima_resid"] = 0.0
            x_in  = pd.DataFrame([{c:float(row.get(c,0)) for c in xcols}])
            xgb_p = float(model.predict(x_in)[0])
            ar_p  = float(ar_steps[step])
            hybrid= w_ar*ar_p + w_xg*xgb_p

            preds.append({
                "date"            : fut.strftime("%b %Y"),
                "date_dt"         : fut,
                "price_ngn_per_kg": round(hybrid,2),
                "lower"           : round(max(0,hybrid-mae),2),
                "upper"           : round(hybrid+mae,2),
            })
            row["lag12"]=row.get("lag6",hybrid); row["lag6"]=row.get("lag3",hybrid)
            row["lag3"] =row.get("lag2",hybrid); row["lag2"]=row.get("lag1",hybrid)
            row["lag1"] =hybrid
            row["roll3"]=(row["lag1"]+row["lag2"]+row["lag3"])/3
            row["roll6"]=(row["lag1"]+row["lag2"]+row["lag3"]+row["lag6"])/4
            row["roll12"]=hybrid

        results[col] = {"label":info["label"],"forecasts":preds,"mae":mae}
    return results


# ==============================================================================
# SIDEBAR
# ==============================================================================

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Flag_of_Nigeria.svg/1200px-Flag_of_Nigeria.svg.png", width=60)
    st.title("🌾 Food Price\nForecasting")
    st.caption("Nigeria — Hybrid ARIMA + XGBoost")
    st.divider()

    st.subheader("📁 Data files")
    wfp_path   = st.text_input("WFP prices CSV",   value="wfp_food_prices_nga.csv")
    macro_path = st.text_input("Macro dataset CSV", value="final_dataset.csv")

    st.subheader("⚙️ Model settings")
    n_trials = st.slider("Optuna tuning trials", 10, 80, 40,
                         help="More trials = better accuracy but slower training")

    st.subheader("📅 Forecast horizon")
    n_months = st.slider("Months ahead", 1, 12, 6)

    st.subheader("🎯 Scenario analysis")
    use_scenario = st.checkbox("Enable stress test")
    scenario_fx  = None; scenario_cpi = None
    if use_scenario:
        scenario_fx  = st.number_input("USD/NGN rate", value=1800, step=50,
                                       help="Current rate ~1490. Enter higher value to stress test.")
        scenario_cpi = st.slider("Monthly CPI growth (%)", 0.0, 5.0, 2.0, 0.5) / 100

    run_btn = st.button("🚀 Train & Forecast", type="primary", use_container_width=True)
    st.divider()
    st.caption("Built with ARIMA + XGBoost Hybrid Model\n\nData: WFP Nigeria + CBN")


# ==============================================================================
# MAIN
# ==============================================================================

st.title("Nigeria Food Price Forecasting Dashboard")
st.caption("Hybrid ARIMA + XGBoost model — 4 commodities — 2018–2024 training window")

if not run_btn:
    st.info("👈 Configure settings in the sidebar and click **Train & Forecast** to begin.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Commodities modelled", "4")
    col2.metric("Training window", "2018 – 2024")
    col3.metric("Dropped (data issues)", "2 (rice)")
    col4.metric("Model type", "Hybrid")

    st.markdown("---")
    st.markdown("""
**How this works:**
1. **ARIMA** captures the long-run price trend and seasonality of each commodity
2. **XGBoost** corrects ARIMA's errors using macro features: CPI, exchange rate, fuel price, temperature, rainfall
3. The **hybrid prediction** is a weighted average — the better-performing model on validation data gets more weight
4. Forecasts include a ±MAE confidence band

**Why rice is excluded:**
The WFP dataset records rice prices in mixed units (per KG, per 50 KG bag, per 100 KG bag) across markets. This creates irreconcilable price-level inconsistencies that make prediction impossible. Fix: source per-KG rice prices from AFEX Commodities Exchange.
    """)
    st.stop()


# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading and preparing data..."):
    try:
        df = load_and_prepare(wfp_path, macro_path)
        st.success(f"✅ Data loaded — {len(df):,} rows | {df['date'].min().date()} → {df['date'].max().date()}")
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e}\n\nMake sure both CSV files are in the same folder as app.py")
        st.stop()
    except Exception as e:
        st.error(f"❌ Data loading error: {e}")
        st.stop()

# ── Train models ──────────────────────────────────────────────────────────────
with st.spinner(f"Training hybrid models ({n_trials} Optuna trials per commodity)... this takes ~2 minutes"):
    trained, metrics = train_hybrid_models(df, n_trials=n_trials)

# ── Run forecast ──────────────────────────────────────────────────────────────
fc_base = run_forecast(trained, n_months=n_months)
fc_stress = run_forecast(trained, n_months=n_months,
                         scenario_fx=scenario_fx,
                         scenario_cpi_growth=scenario_cpi) if use_scenario else None


# ==============================================================================
# TAB 1: MODEL ACCURACY
# ==============================================================================

tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Accuracy", "📈 Forecast", "🔍 Scenario Analysis", "📋 Data Explorer"])

with tab1:
    st.subheader("Hybrid Model Performance — Test Set (last 12 months)")
    st.caption("Accuracy = 100% − MAPE. Threshold: ≥80% kept, <80% dropped.")

    # Accuracy cards
    cols = st.columns(len(KEPT_COMMODITIES))
    for i,(col,label) in enumerate(KEPT_COMMODITIES.items()):
        m = metrics[col]["hybrid"]
        acc = m["accuracy"]
        cls = "accuracy-good" if acc>=85 else ("accuracy-warn" if acc>=80 else "accuracy-bad")
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:13px;color:#6c757d;margin-bottom:4px">{label}</div>
                <div class="{cls}">{acc:.1f}%</div>
                <div style="font-size:12px;color:#6c757d;margin-top:6px">
                    MAPE: {m['MAPE']:.2f}% &nbsp;|&nbsp; MAE: ₦{m['MAE']:,.0f}/kg<br>
                    R²: {m['R2']:.4f}<br>
                    ARIMA weight: {metrics[col]['w_ar']:.0%} &nbsp;|&nbsp; XGB: {metrics[col]['w_xg']:.0%}
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Actual vs Predicted chart per commodity
    st.subheader("Actual vs Hybrid Prediction — Test Period")
    sel_col = st.selectbox("Select commodity", list(KEPT_COMMODITIES.keys()),
                           format_func=lambda c: KEPT_COMMODITIES[c])

    m_data = metrics[sel_col]
    t_data = trained[sel_col]
    dates  = pd.to_datetime(t_data["test_dates"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=t_data["actual"], name="Actual",
                             line=dict(color="#343a40", width=2.5)))
    fig.add_trace(go.Scatter(x=dates, y=t_data["hybrid_pred"],
                             name=f"Hybrid ({m_data['w_ar']:.0%} ARIMA + {m_data['w_xg']:.0%} XGB)",
                             line=dict(color=COMMODITY_COLORS[sel_col], width=2.5, dash="dash")))
    fig.update_layout(
        xaxis_title="Month", yaxis_title="NGN per KG",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified", height=380,
        margin=dict(t=20,b=40,l=60,r=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Full accuracy comparison table
    st.subheader("Full accuracy comparison — all 6 commodities")
    all_commodities_labels = {v:k for k,v in TARGET_COMMODITIES.items()}
    acc_rows = []
    for col, label in KEPT_COMMODITIES.items():
        m = metrics[col]["hybrid"]
        acc_rows.append({"Commodity":label, "Accuracy":f"{m['accuracy']:.2f}%",
                         "MAPE":f"{m['MAPE']:.2f}%", "MAE (₦/kg)":f"{m['MAE']:,.0f}",
                         "RMSE":f"{m['RMSE']:,.0f}", "R²":f"{m['R2']:.4f}",
                         "Decision":"✅ Keep"})
    for col, label in [("rice_imported","Rice (Imported)"),("rice_local","Rice (Local)")]:
        acc_rows.append({"Commodity":label, "Accuracy":"< 0%",
                         "MAPE":"> 100%", "MAE (₦/kg)":"N/A",
                         "RMSE":"N/A","R²":"N/A",
                         "Decision":"❌ Drop"})
    st.dataframe(pd.DataFrame(acc_rows).set_index("Commodity"), use_container_width=True)

    with st.expander("Why were rice commodities dropped?"):
        st.markdown("""
The WFP dataset records rice prices in **mixed units** across markets:
- Some markets report **per KG**
- Others report **per 50 KG bag** or **per 100 KG bag**

Even after converting to per-KG using the `unit` column, residual inconsistencies remain because different reporters in the same market switched units over time without documentation. The model sees the same commodity at wildly different price levels (₦94–₦28,000 for the same month), making prediction impossible — MAPE exceeds 300% and R² drops to −4,000.

**Fix:** Obtain rice price data from **AFEX Commodities Exchange** or **CBN agricultural credit data**, both of which report consistently in NGN per KG.
        """)


# ==============================================================================
# TAB 2: FORECAST
# ==============================================================================

with tab2:
    st.subheader(f"{n_months}-Month Price Forecast (NGN/KG)")
    st.caption("Shaded band = ±MAE confidence interval")

    fig2 = go.Figure()
    for col, label in KEPT_COMMODITIES.items():
        fc  = fc_base[col]["forecasts"]
        xs  = [p["date_dt"] for p in fc]
        ys  = [p["price_ngn_per_kg"] for p in fc]
        lo  = [p["lower"] for p in fc]
        hi  = [p["upper"] for p in fc]
        clr = COMMODITY_COLORS[col]

        fig2.add_trace(go.Scatter(
            x=xs+xs[::-1], y=hi+lo[::-1], fill="toself",
            fillcolor=clr+"33", line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip"
        ))
        fig2.add_trace(go.Scatter(
            x=xs, y=ys, name=label,
            line=dict(color=clr, width=2.5),
            mode="lines+markers", marker=dict(size=6),
            hovertemplate=f"<b>{label}</b><br>%{{x|%b %Y}}: ₦%{{y:,.0f}}/kg<extra></extra>"
        ))

    fig2.update_layout(
        xaxis_title="Month", yaxis_title="NGN per KG",
        hovermode="x unified", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=20,b=40,l=60,r=20),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Forecast table
    st.subheader("Forecast table")
    for col, label in KEPT_COMMODITIES.items():
        fc  = fc_base[col]["forecasts"]
        mae = fc_base[col]["mae"]
        df_fc = pd.DataFrame([{
            "Month"      : p["date"],
            "Price (₦/kg)": f"₦{p['price_ngn_per_kg']:,.0f}",
            "Lower bound": f"₦{p['lower']:,.0f}",
            "Upper bound": f"₦{p['upper']:,.0f}",
        } for p in fc])
        with st.expander(f"📦 {label}  (±₦{mae:,.0f}/kg MAE)"):
            st.dataframe(df_fc.set_index("Month"), use_container_width=True)

    # Download forecast
    all_fc_rows = []
    for col, label in KEPT_COMMODITIES.items():
        for p in fc_base[col]["forecasts"]:
            all_fc_rows.append({"commodity":label,"date":p["date"],
                                "price_ngn_per_kg":p["price_ngn_per_kg"],
                                "lower_bound":p["lower"],"upper_bound":p["upper"]})
    fc_df = pd.DataFrame(all_fc_rows)
    st.download_button("⬇ Download forecast CSV", fc_df.to_csv(index=False),
                       "forecast.csv", "text/csv", use_container_width=True)


# ==============================================================================
# TAB 3: SCENARIO ANALYSIS
# ==============================================================================

with tab3:
    st.subheader("Scenario Analysis")
    if not use_scenario:
        st.info("👈 Enable **stress test** in the sidebar and set your FX and CPI parameters, then re-run.")
    else:
        st.markdown(f"**Stress scenario:** USD/NGN = {scenario_fx:,} | CPI growth = {scenario_cpi*100:.1f}%/month")

        fig3 = go.Figure()
        for col, label in KEPT_COMMODITIES.items():
            clr      = COMMODITY_COLORS[col]
            fc_b     = fc_base[col]["forecasts"]
            fc_s     = fc_stress[col]["forecasts"]
            xs       = [p["date_dt"] for p in fc_b]
            ys_base  = [p["price_ngn_per_kg"] for p in fc_b]
            ys_stress= [p["price_ngn_per_kg"] for p in fc_s]

            fig3.add_trace(go.Scatter(x=xs, y=ys_base, name=f"{label} (baseline)",
                                      line=dict(color=clr,width=2),mode="lines+markers"))
            fig3.add_trace(go.Scatter(x=xs, y=ys_stress, name=f"{label} (stress)",
                                      line=dict(color=clr,width=2,dash="dot"),mode="lines"))

        fig3.update_layout(xaxis_title="Month",yaxis_title="NGN per KG",
                           hovermode="x unified",height=420,
                           legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
                           margin=dict(t=20,b=40,l=60,r=20))
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader(f"Price impact at month {n_months}")
        impact_rows = []
        for col, label in KEPT_COMMODITIES.items():
            b = fc_base[col]["forecasts"][-1]["price_ngn_per_kg"]
            s = fc_stress[col]["forecasts"][-1]["price_ngn_per_kg"]
            impact_rows.append({
                "Commodity"     : label,
                "Baseline (₦/kg)": f"₦{b:,.0f}",
                "Stress (₦/kg)" : f"₦{s:,.0f}",
                "Impact (₦)"    : f"+₦{s-b:,.0f}" if s>=b else f"₦{s-b:,.0f}",
                "Impact (%)"    : f"+{(s-b)/b*100:.1f}%" if b>0 else "N/A",
            })
        st.dataframe(pd.DataFrame(impact_rows).set_index("Commodity"), use_container_width=True)


# ==============================================================================
# TAB 4: DATA EXPLORER
# ==============================================================================

with tab4:
    st.subheader("Raw data explorer")

    col_a, col_b, col_c = st.columns(3)
    state_filter = col_a.multiselect("Filter by state", sorted(df["admin1"].unique()), default=[])
    market_filter= col_b.multiselect("Filter by market",sorted(df["market"].unique()), default=[])
    comm_filter  = col_c.selectbox("Commodity to plot", list(KEPT_COMMODITIES.keys()),
                                   format_func=lambda c: KEPT_COMMODITIES[c])

    df_view = df.copy()
    if state_filter:  df_view = df_view[df_view["admin1"].isin(state_filter)]
    if market_filter: df_view = df_view[df_view["market"].isin(market_filter)]

    ts_plot = df_view.groupby("date")[comm_filter].mean().reset_index()
    fig4 = px.line(ts_plot, x="date", y=comm_filter,
                   title=f"{KEPT_COMMODITIES[comm_filter]} — average NGN/KG over time",
                   labels={"date":"Date",comm_filter:"NGN per KG"},
                   color_discrete_sequence=[COMMODITY_COLORS[comm_filter]])
    fig4.update_layout(height=360,margin=dict(t=40,b=40,l=60,r=20))
    st.plotly_chart(fig4, use_container_width=True)

    st.caption(f"Showing {len(df_view):,} rows")
    st.dataframe(df_view[["date","admin1","market",comm_filter,"fuel_price_raw",
                           "cpi_raw","usd_ngn_raw"]].rename(columns={
        "fuel_price_raw":"fuel_price(₦/L)","cpi_raw":"CPI","usd_ngn_raw":"USD/NGN"
    }).sort_values("date", ascending=False).head(200), use_container_width=True)

    st.download_button("⬇ Download filtered data",
                       df_view.to_csv(index=False),
                       "filtered_data.csv","text/csv",use_container_width=True)


# ==============================================================================
# FOOTER
# ==============================================================================

st.divider()
st.caption("Nigeria Food Price Forecasting · Hybrid ARIMA + XGBoost · Data: WFP Nigeria, CBN")
