#!/usr/bin/env python3
"""
diagnose_linear.py

Loads processed features, refits via statsmodels, and runs assumption diagnostics.
"""

import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan

# 1) Locate and load the engineered features
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
feat_path = os.path.join(root_dir, "data", "processed", "car_price_features.csv")
df = pd.read_csv(feat_path)
print(f"Loaded data from {feat_path!r} with shape {df.shape}")

# 2) Prepare X and y
X = df.drop(columns=["price", "log_price", "car_ID"])
y = df["log_price"].astype(float)

# 3) Refit with statsmodels OLS for inference
X_sm = sm.add_constant(X).astype(float)
ols_sm = sm.OLS(y, X_sm).fit()
print("\n=== OLS Summary ===")
print(ols_sm.summary())

# 4) Extract residuals and fitted values
resid  = ols_sm.resid
fitted = ols_sm.fittedvalues

# 5) Residuals vs Fitted
plt.figure()
sns.scatterplot(x=fitted, y=resid, alpha=0.6)
plt.axhline(0, color="red", ls="--")
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")
plt.tight_layout()
plt.show()

# 6) Normal Q–Q plot
plt.figure()
sm.qqplot(resid, line="45", fit=True)
plt.title("Normal Q–Q")
plt.tight_layout()
plt.show()

# 7) Scale–Location plot
std_resid = resid / resid.std()
sqrt_abs = std_resid.abs().map(math.sqrt)
plt.figure()
sns.scatterplot(x=fitted, y=sqrt_abs, alpha=0.6)
plt.xlabel("Fitted")
plt.ylabel("√|Standardized Residuals|")
plt.title("Scale–Location")
plt.tight_layout()
plt.show()

# 8) Cook’s distance
influence = ols_sm.get_influence()
cooks, _  = influence.cooks_distance
plt.figure()
plt.stem(range(len(cooks)), cooks, markerfmt=",")
plt.axhline(4/len(cooks), color="red", ls="--")
plt.xlabel("Observation")
plt.ylabel("Cook's distance")
plt.title("Cook's Distance")
plt.tight_layout()
plt.show()

# 9) Durbin–Watson test for autocorrelation
dw = durbin_watson(resid)
print(f"\nDurbin–Watson statistic: {dw:.3f} (≈2 indicates no autocorrelation)")

# 10) Breusch–Pagan test for heteroscedasticity
bp_test = het_breuschpagan(resid, ols_sm.model.exog)
labels  = ["Lagrange multiplier", "p-value", "f-value", "f-p-value"]
print("\nBreusch–Pagan test results:")
print(dict(zip(labels, bp_test)))

# 11) Tidy coefficient table (top 10 by |t-stat|)
coef_table = ols_sm.summary2().tables[1]
coef_table["|t|"] = coef_table["t"].abs()
print("\nTop 10 coefficients by |t-stat|:")
print(coef_table.sort_values("|t|", ascending=False).head(10))
