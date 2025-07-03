#!/usr/bin/env python3
"""
refine_model.py—with train/test split, mild WLS, and out‐of‐sample evaluation
"""

import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ─── 1) LOAD & PREP ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/car_price_features.csv")

# target & features
y = df["log_price"].astype(float)
X = df.drop(columns=["car_ID", "log_price", "price"])

# bool→int & ensure numeric
for c in X.columns:
    if X[c].dtype == "bool":
        X[c] = X[c].astype(int)
    X[c] = pd.to_numeric(X[c], errors="coerce")

# drop any cols with nulls
X = X.drop(columns=[c for c in X if X[c].isnull().any()])

# add intercept
X = sm.add_constant(X)

# train/test split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows")

# ─── 2) MULTICOLLINEARITY: VIF DROPPING on TRAIN ────────────────────────────────
def drop_high_vif(df, thresh=10.0):
    Xv = df.copy()
    while True:
        vifs = {
            col: variance_inflation_factor(Xv.values, i)
            for i, col in enumerate(Xv.columns)
        }
        vifs.pop("const", None)
        worst = max(vifs, key=vifs.get)
        if vifs[worst] > thresh:
            print(f"Dropping '{worst}' (VIF={vifs[worst]:.1f})")
            Xv = Xv.drop(columns=[worst])
        else:
            break
    return Xv

X_train_vif = drop_high_vif(X_train, thresh=10.0)
print("Features kept after VIF:", X_train_vif.columns.tolist())

# restrict test set to same cols
X_test_vif = X_test[X_train_vif.columns]

# rename for clarity
X_train, X_test = X_train_vif, X_test_vif

# ─── 3) HC3‐ROBUST OLS & DIAGNOSTICS on TRAIN ─────────────────────────────────
ols = sm.OLS(y_train, X_train).fit(cov_type="HC3")
print("\n=== HC3‐Robust OLS Summary (train) ===")
print(ols.summary())

def plot_diagnostics(model, title_suffix=""):
    resid  = model.resid
    fitted = model.fittedvalues

    # Residual vs Fitted
    plt.figure()
    sns.scatterplot(x=fitted, y=resid, alpha=0.6)
    plt.axhline(0, linestyle="--", color="red")
    plt.title(f"Residuals vs Fitted {title_suffix}")
    plt.xlabel("Fitted"); plt.ylabel("Residual")
    plt.show()

    # Normal Q–Q
    plt.figure()
    sm.qqplot(resid, line="45", fit=True)
    plt.title(f"Normal Q–Q {title_suffix}")
    plt.show()

    # Scale–Location
    std_resid = resid / resid.std()
    sqrt_abs  = std_resid.abs().map(math.sqrt)
    plt.figure()
    sns.scatterplot(x=fitted, y=sqrt_abs, alpha=0.6)
    plt.title(f"Scale–Location {title_suffix}")
    plt.xlabel("Fitted"); plt.ylabel("√|Std Residual|")
    plt.show()

    # Cook's distance
    cooks, _ = OLSInfluence(model).cooks_distance
    plt.figure()
    plt.stem(cooks, markerfmt=",")
    plt.axhline(4/len(cooks), linestyle="--", color="red")
    plt.title(f"Cook's Distance {title_suffix}")
    plt.show()

plot_diagnostics(ols, "(OLS train)")

# ─── 4) MILD WLS & DIAGNOSTICS on TRAIN ────────────────────────────────────────
# use weights = 1 / fitted (clipped to avoid inf)
eps   = 1e-3
wts   = 1.0 / ols.fittedvalues.clip(lower=eps)
wls   = sm.WLS(y_train, X_train, weights=wts).fit(cov_type="HC3")
print("\n=== WLS (1/fitted) Summary (train) ===")
print(wls.summary())
plot_diagnostics(wls, "(WLS train)")

# ─── 5) OUT‐OF‐SAMPLE EVALUATION ────────────────────────────────────────────────
for name, model in [("OLS", ols), ("WLS", wls)]:
    y_pred = model.predict(X_test)
    r2      = r2_score(y_test, y_pred)
    rmse    = math.sqrt(((y_test - y_pred) ** 2).mean())
    print(f"\n{name} on test set → R² = {r2:.3f}, RMSE(log) = {rmse:.3f}")
