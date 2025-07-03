#!/usr/bin/env python3
"""
final_model.py

Drops the target “price” from the predictors, then:
  • Fits HC3-robust OLS
  • Removes top-5 Cook’s-D outliers and refits
  • Runs a train/test split + RidgeCV (no NumPy)
"""

import math
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# 1) Load processed features
df = pd.read_csv("data/processed/car_price_features.csv")

# 2) Separate target & drop ID, log_price, and the raw price column
y = df["log_price"].astype(float)
X = df.drop(columns=["car_ID", "log_price", "price"])

# 3) Coerce booleans & everything else to numeric (errors→NaN)
for col in X.columns:
    if X[col].dtype == "bool":
        X[col] = X[col].astype(int)
    X[col] = pd.to_numeric(X[col], errors="coerce")

# 4) Drop any columns still containing NaNs
bad = [c for c in X.columns if X[c].isnull().any()]
if bad:
    print("Dropping non-numeric columns:", bad)
    X = X.drop(columns=bad)

# 5) Add intercept for Statsmodels
X_const = sm.add_constant(X)

# 6) HC3-robust OLS on full data
ols = sm.OLS(y, X_const).fit(cov_type="HC3")
print("\n=== HC3-Robust OLS on Full Data ===")
print(ols.summary())

# 7) Identify top-5 Cook’s distance outliers
infl = OLSInfluence(ols)
cooks_d = infl.cooks_distance[0]
# sort indices by Cook’s D, take the last 5
sorted_idx = sorted(range(len(cooks_d)), key=lambda i: cooks_d[i])
top5 = sorted_idx[-5:]
print("\nTop-5 influential rows:", top5)

# 8) Refit plain OLS after dropping those rows
df2 = df.drop(index=top5)
y2  = df2["log_price"].astype(float)
X2  = df2.drop(columns=["car_ID","log_price","price"])

for col in X2.columns:
    if X2[col].dtype == "bool":
        X2[col] = X2[col].astype(int)
    X2[col] = pd.to_numeric(X2[col], errors="coerce")

bad2 = [c for c in X2.columns if X2[c].isnull().any()]
X2 = X2.drop(columns=bad2)

X2_const = sm.add_constant(X2)
ols2 = sm.OLS(y2, X2_const).fit()
print("\n=== OLS after Dropping Top-5 Points ===")
print(f"Adj-R² = {ols2.rsquared_adj:.3f}")
print(ols2.summary())

# 9) Train/test split + RidgeCV
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build 50 log-spaced alphas between 1e-3 and 1e3 (pure Python)
alphas = []
for k in range(50):
    exp = -3 + 6 * k / 49
    alphas.append(10 ** exp)

ridge_pipe = make_pipeline(
    StandardScaler(),
    RidgeCV(alphas=alphas, store_cv_results=True)
)
ridge_pipe.fit(X_train, y_train)

def report(name, y_true, y_pred):
    r2  = ridge_pipe.score(X_train, y_train) if name=="train" else ridge_pipe.score(X_test, y_test)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    print(f"{name.title():5s} → R² = {r2:.3f}, RMSE(log) = {rmse:.3f}")

best_alpha = ridge_pipe.named_steps["ridgecv"].alpha_
print(f"\n=== RidgeCV (best α = {best_alpha:.4g}) ===")
report("train", y_train, ridge_pipe.predict(X_train))
report("test",  y_test,  ridge_pipe.predict(X_test))
