"""
AquaSakhi — ML Pipeline (SVM-Based)
Team ACE IT | WitchHunt 2026 | Climate Action Theme

Groundwater Scarcity Prediction using:
  - SVR  (Support Vector Regression)     → predict groundwater level (next 6 months)
  - SVC  (Support Vector Classification) → classify district as Low / Moderate / High risk
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import os, warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
os.makedirs("data", exist_ok=True)

# ════════════════════════════════════════════════════════════
# STEP 1 — Synthetic Dataset Generation
# ════════════════════════════════════════════════════════════
DISTRICTS_CONFIG = [
    # (name,           base_level_m, base_rain_mm, trend_m_per_yr)
    ("Guntur",         4.2,  72,  -0.8),
    ("Kurnool",        3.8,  58,  -1.1),
    ("Anantapur",      3.1,  45,  -1.4),
    ("Kadapa",         6.5,  88,  -0.4),
    ("Nellore",        7.2,  96,  -0.2),
    ("Vizianagaram",  10.1, 142,   0.3),
    ("Srikakulam",    11.4, 160,   0.6),
    ("West Godavari",  9.8, 138,   0.1),
    ("East Godavari",  8.1, 118,  -0.3),
    ("Krishna",        6.9,  94,  -0.5),
]

def generate_district_data(name, base_level, base_rain, trend, n=84):
    """Generate 84 months (7 years) of synthetic monthly groundwater data."""
    months = pd.date_range("2018-01", periods=n, freq="MS")
    t = np.arange(n)
    seasonal   = np.sin(t * 2 * np.pi / 12) * 1.2
    long_trend = trend * t / 12
    gw_level   = base_level + long_trend + seasonal + np.random.normal(0, 0.3, n)
    rainfall   = np.maximum(0, base_rain
                            + np.maximum(0, np.sin(t * 2 * np.pi / 12 + 1) * 80)
                            + np.random.normal(0, 15, n))
    temperature = 28 + 6 * np.sin(t * 2 * np.pi / 12) + np.random.normal(0, 1, n)
    land_use    = np.linspace(0.3, 0.6, n) + np.random.normal(0, 0.02, n)
    return pd.DataFrame({
        "date":                months,
        "district":            name,
        "groundwater_level_m": gw_level,
        "rainfall_mm":         rainfall,
        "temperature_c":       temperature,
        "land_use_index":      land_use,
    })

print("=" * 57)
print("  AquaSakhi — SVM-Based Water Scarcity Prediction Pipeline")
print("=" * 57)
print("\n[1/5] Generating synthetic dataset...")
df = pd.concat([generate_district_data(*d) for d in DISTRICTS_CONFIG], ignore_index=True)
df.to_csv("data/groundwater_dataset.csv", index=False)
print(f"      ✓ {len(df)} records | {df['district'].nunique()} districts | saved to data/groundwater_dataset.csv")


# ════════════════════════════════════════════════════════════
# STEP 2 — Feature Engineering
# ════════════════════════════════════════════════════════════
print("\n[2/5] Engineering features...")

def engineer_features(df):
    df = df.copy().sort_values(["district", "date"]).reset_index(drop=True)
    gw   = df.groupby("district")["groundwater_level_m"]
    rain = df.groupby("district")["rainfall_mm"]

    # Lag features (past groundwater observations)
    df["gw_lag1"]  = gw.shift(1)
    df["gw_lag2"]  = gw.shift(2)
    df["gw_lag3"]  = gw.shift(3)

    # Rolling averages (trend context)
    df["gw_roll3"]   = gw.transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["gw_roll6"]   = gw.transform(lambda x: x.rolling(6, min_periods=1).mean())
    df["rain_roll3"] = rain.transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["rain_roll6"] = rain.transform(lambda x: x.rolling(6, min_periods=1).mean())

    # Domain indicators
    df["rain_anomaly"]      = df["rainfall_mm"] - df["rain_roll6"]
    df["recharge_idx"]      = df["rainfall_mm"] / (df["temperature_c"] + 1)
    df["extraction_stress"] = df["land_use_index"] * df["gw_roll3"]

    # Calendar features
    df["month"]  = pd.to_datetime(df["date"]).dt.month
    df["season"] = df["month"].map(
        lambda m: "monsoon" if m in [6,7,8,9]
        else ("winter" if m in [11,12,1,2] else "summer")
    )
    df = pd.get_dummies(df, columns=["season"], drop_first=False)
    return df.dropna()

df_feat = engineer_features(df)

FEATURES = [
    "gw_lag1", "gw_lag2", "gw_lag3",
    "gw_roll3", "gw_roll6",
    "rainfall_mm", "rain_roll3", "rain_roll6", "rain_anomaly",
    "recharge_idx", "extraction_stress",
    "temperature_c", "land_use_index", "month",
    "season_monsoon", "season_summer", "season_winter",
]
TARGET = "groundwater_level_m"

def label_risk(level):
    """Threshold-based risk classification."""
    if level <= 5.0:   return "High"
    elif level <= 8.0: return "Moderate"
    else:              return "Low"

df_feat["risk_zone"] = df_feat[TARGET].apply(label_risk)

X     = df_feat[FEATURES].values
y_reg = df_feat[TARGET].values
y_clf = df_feat["risk_zone"].values

X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# Feature scaling — mandatory for SVM (distance-based algorithm)
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"      ✓ {len(FEATURES)} features | Train={len(X_train)} | Test={len(X_test)}")


# ════════════════════════════════════════════════════════════
# STEP 3 — SVR: Groundwater Level Prediction
# ════════════════════════════════════════════════════════════
print("\n[3/5] Training SVR (Support Vector Regression)...")
print("      Kernel=RBF  |  C=100  |  epsilon=0.1  |  gamma=scale")

svr = SVR(kernel="rbf", C=100, epsilon=0.1, gamma="scale")
svr.fit(X_train_s, yr_train)
svr_pred = svr.predict(X_test_s)

svr_rmse = np.sqrt(mean_squared_error(yr_test, svr_pred))
svr_mae  = mean_absolute_error(yr_test, svr_pred)
svr_r2   = r2_score(yr_test, svr_pred)
cv_r2    = cross_val_score(svr, X_train_s, yr_train, cv=5, scoring="r2")

print(f"\n      ── SVR Performance ──────────────────────")
print(f"      RMSE        : {svr_rmse:.4f} m")
print(f"      MAE         : {svr_mae:.4f} m")
print(f"      R² (test)   : {svr_r2:.4f}")
print(f"      R² (5-fold) : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")


# ════════════════════════════════════════════════════════════
# STEP 4 — SVC: Risk Zone Classification
# ════════════════════════════════════════════════════════════
print("\n[4/5] Training SVC (Support Vector Classification)...")
print("      Kernel=RBF  |  C=10  |  gamma=scale  |  class_weight=balanced")

svc = SVC(kernel="rbf", C=10, gamma="scale",
          class_weight="balanced", probability=True, random_state=42)
svc.fit(X_train_s, yc_train)
svc_pred  = svc.predict(X_test_s)
svc_proba = svc.predict_proba(X_test_s)

print("\n      ── SVC Classification Report ────────────")
print(classification_report(yc_test, svc_pred,
      target_names=["High", "Moderate", "Low"], zero_division=0))

print("      Confusion Matrix  (rows=actual, cols=predicted):")
cm = confusion_matrix(yc_test, svc_pred, labels=["High", "Moderate", "Low"])
print(f"                    High  Moderate  Low")
for label, row in zip(["High    ", "Moderate", "Low     "], cm):
    print(f"      {label}   {str(row[0]):>4}  {str(row[1]):>8}  {str(row[2]):>3}")


# ════════════════════════════════════════════════════════════
# STEP 5 — Save Predictions
# ════════════════════════════════════════════════════════════
print("\n[5/5] Saving predictions...")
results = (df_feat.iloc[-len(X_test):]
           [["date", "district", TARGET]]
           .copy().reset_index(drop=True))
results["svr_predicted_level_m"] = svr_pred.round(3)
results["svc_risk_zone"]         = svc_pred
results["risk_confidence_pct"]   = (svc_proba.max(axis=1) * 100).round(1)
results.to_csv("data/predictions.csv", index=False)
print("      ✓ Saved: data/predictions.csv")

# District-level summary
summary = results.groupby("district").agg(
    avg_actual    = (TARGET,                  "mean"),
    avg_predicted = ("svr_predicted_level_m", "mean"),
    risk_zone     = ("svc_risk_zone",         lambda x: x.mode()[0]),
    confidence    = ("risk_confidence_pct",   "mean"),
).reset_index()

print("\n── District Prediction Summary ──────────────────────────────────")
print(f"{'District':<18} {'Actual':>8} {'Predicted':>10} {'Risk':>10} {'Conf%':>7}")
print("-" * 60)
for _, r in summary.iterrows():
    print(f"{r['district']:<18} {r['avg_actual']:>8.2f}m"
          f" {r['avg_predicted']:>9.2f}m {r['risk_zone']:>10} {r['confidence']:>6.1f}%")

print("\n✅ Pipeline complete.")
print("   Open index.html in your browser to view the dashboard.\n")
