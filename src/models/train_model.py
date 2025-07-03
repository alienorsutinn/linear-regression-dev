import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score

# 1) Load engineered features
df = pd.read_csv("../data/processed/car_price_features.csv")

# 2) Prepare X, y (we predict log_price)
X = df.drop(columns=["price", "log_price", "car_ID"])
y = df["log_price"]

# 3) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# 4) Pipeline helper
def make_pipeline(model):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg", model)
    ])

# 5) Ordinary Least Squares
ols = make_pipeline(LinearRegression())
ols.fit(X_train, y_train)
y_pred_ols = ols.predict(X_test)
rmse_ols = np.sqrt(mean_squared_error(y_test, y_pred_ols))
r2_ols   = r2_score(y_test, y_pred_ols)
print(f"\nOLS → RMSE(log): {rmse_ols:.3f}, R²: {r2_ols:.3f}")

# 6) Ridge regression with built-in CV for alpha
alphas = np.logspace(-3, 3, 50)
ridge = make_pipeline(RidgeCV(alphas=alphas, cv=5))
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge   = r2_score(y_test, y_pred_ridge)
best_alpha = ridge.named_steps["reg"].alpha_
print(f"Ridge → α={best_alpha:.4f}, RMSE(log): {rmse_ridge:.3f}, R²: {r2_ridge:.3f}")

# 7) Lasso regression with CV for alpha
lasso = make_pipeline(LassoCV(alphas=alphas, cv=5, max_iter=5000))
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
r2_lasso   = r2_score(y_test, y_pred_lasso)
best_alpha_l = lasso.named_steps["reg"].alpha_
print(f"Lasso → α={best_alpha_l:.4f}, RMSE(log): {rmse_lasso:.3f}, R²: {r2_lasso:.3f}")

# 8) Cross‐validation comparison
print("\n5-Fold CV (neg RMSE of log_price):")
for name, model in [("OLS", ols), ("Ridge", ridge), ("Lasso", lasso)]:
    scores = cross_val_score(
        model, X, y,
        scoring="neg_root_mean_squared_error",
        cv=5, n_jobs=-1
    )
    print(f"{name}: {-scores.mean():.3f} ± {scores.std():.3f}")

# 9) Inspect coefficients (Ridge as example)
coef = pd.Series(
    ridge.named_steps["reg"].coef_,
    index=X.columns
).sort_values(key=abs, ascending=False).head(15)
print("\nTop 15 coefficients (by absolute value) in Ridge model:")
print(coef)

# 10) Back‐transform and report price-space RMSE
for label, pred in [("OLS", y_pred_ols), ("Ridge", y_pred_ridge), ("Lasso", y_pred_lasso)]:
    rmse_price = np.sqrt(mean_squared_error(np.exp(y_test), np.exp(pred)))
    print(f"{label} → RMSE(price): ${rmse_price:,.0f}")
