import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

print("🇮🇳 Starting All-India Meteorological AI Training (HydroCast-AI Engine)...")

np.random.seed(42)
n_samples = 15000

# 1. All-India Meteorological Subdivisions & Micro-Climates
INDIAN_REGIONS = [
    {"name": "Central India (MP, Chhattisgarh)", "base_rain": 18, "regime": 0, "cape": 1100, "weight": 0.25},
    {"name": "Western Ghats & Konkan (Maharashtra, Goa, Coastal Karnataka)", "base_rain": 65, "regime": 3, "cape": 1600, "weight": 0.20},
    {"name": "Bay of Bengal Track (Odisha, West Bengal, AP)", "base_rain": 55, "regime": 2, "cape": 2200, "weight": 0.20},
    {"name": "Northwest India (Rajasthan, Punjab, Haryana)", "base_rain": 8, "regime": 1, "cape": 600, "weight": 0.15},
    {"name": "Northeast Hills (Assam, Meghalaya)", "base_rain": 50, "regime": 3, "cape": 1800, "weight": 0.10},
    {"name": "Western Himalayas (HP, Uttarakhand, J&K)", "base_rain": 25, "regime": 5, "cape": 750, "weight": 0.10}
]

region_choices = np.random.choice(len(INDIAN_REGIONS), size=n_samples, p=[r["weight"] for r in INDIAN_REGIONS])

records = []
for i in range(n_samples):
    reg = INDIAN_REGIONS[region_choices[i]]
    
    temp = np.random.normal(30, 4.0)
    humidity = np.clip(np.random.normal(76, 14), 30, 100)
    wind_speed = np.clip(np.random.normal(16, 7), 2, 60)
    pressure = np.random.normal(1006, 6)
    cape = max(100.0, np.random.normal(reg["cape"], 350))
    dem = np.random.uniform(10, 1800)
    
    # Physics systematic bias injection
    raw_nwp = np.maximum(0, np.random.exponential(reg["base_rain"]))
    
    # Orographic underprediction, break overprediction, convective underestimation
    bias = (
        0.35 * raw_nwp * (reg["regime"] == 3) - 
        0.45 * raw_nwp * (reg["regime"] == 1) + 
        0.003 * cape + 
        np.random.normal(0, 3.5)
    )
    observed_rain = np.maximum(0, raw_nwp + bias)
    error = observed_rain - raw_nwp
    
    records.append({
        'temp': temp, 'humidity': humidity, 'wind_speed': wind_speed,
        'pressure': pressure, 'cape': cape, 'elevation': dem,
        'raw_nwp': raw_nwp, 'observed_rain': observed_rain,
        'error': error, 'regime': reg["regime"]
    })

df = pd.DataFrame(records)
df.to_csv("all_india_patterns.csv", index=False)
print("💾 Created & Saved 15,000 All-India atmospheric patterns to 'all_india_patterns.csv'!")

# 2. Train Models
features = ['temp', 'humidity', 'wind_speed', 'pressure', 'cape', 'elevation', 'raw_nwp']
X = df[features]
y_regime = df['regime']
y_error = df['error']
y_heavy = (df['observed_rain'] >= 64.5).astype(int)

# Classifier 1: Soft Synoptic Regime
print("🌀 Training All-India Regime Classifier...")
clf_regime = lgb.LGBMClassifier(n_estimators=80, learning_rate=0.08, verbose=-1, random_state=42)
clf_regime.fit(X, y_regime)

reg_probs = clf_regime.predict_proba(X)
X_ext = np.hstack([X, reg_probs])

# Regressors 2: Quantiles (P50, P90, P95)
print("⚖️ Training Quantile Repair Regressors (P50, P90, P95)...")
q_models = {}
for q in [0.50, 0.75, 0.90, 0.95]:
    model = lgb.LGBMRegressor(objective='quantile', alpha=q, n_estimators=75, learning_rate=0.08, verbose=-1, random_state=42)
    model.fit(X_ext, y_error)
    q_models[q] = model

# Classifier 3: Extreme Heavy Rain Exceedance (>100mm)
print("🚨 Training Extreme Rainfall Exceedance Model...")
clf_heavy = lgb.LGBMClassifier(n_estimators=75, learning_rate=0.08, verbose=-1, random_state=42)
clf_heavy.fit(X_ext, (df['observed_rain'] >= 100.0).astype(int))

with open("rain_repair_model.pkl", "wb") as f:
    pickle.dump({
        'regime_model': clf_regime,
        'quantile_models': q_models,
        'heavy_model': clf_heavy
    }, f)

print("✅ SUCCESS: All-India Model Package Saved to 'rain_repair_model.pkl'!")