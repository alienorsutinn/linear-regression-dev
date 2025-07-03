refined_model.PY RESULTS SHOW:

# Car Price Regression — README Notes

## Overview  
We fit a linear regression to **predict `log_price`** (the natural-log of the car’s sale price) from a parsimonious set of features after aggressively removing collinearity.

---

## Data & Split  
- **Total observations:** 205 cars  
- **Training set:** 164 cars  
- **Test set:**  41 cars  

---

## Preprocessing  
1. **Feature pruning via VIF:** iteratively drop raw dimensions & engine specs with VIF > 10  
2. **Convert booleans → ints**, enforce numeric dtypes  
3. **Add constant** column for intercept  

---

## Modeling  

### 1. HC3-Robust OLS (train)  
- **R² ≈ 0.928** → explains ~92.8 % of variance in log(price)  
- **Key significant predictors:**  
  - Physical dimensions: *carwidth*, *carheight*  
  - Mechanical specs: *fuelsystem*, *boreratio*, *stroke*, *peakrpm*  
  - Brand dummies (e.g. *brand_toyota*, *brand_peugeot*, …)  
  - Drivetrain: *drivewheel_fwd*, *enginelocation_rear*  
  - Interaction: *hp_x_size*  

### 2. Weighted Least Squares (weights = 1/fitted)  
- **R² ≈ 0.926** on train → nearly identical to OLS, so heteroskedasticity is mild  

---

## Out-of-Sample Performance (test set)  
| Model | R² (test) | RMSE (log-price) | ≈ multiplicative error |
|:------|:---------:|:----------------:|:---------------------:|
| OLS   |   0.889   |      0.174       |    ±19 % in dollars   |
| WLS   |   0.889   |      0.174       |    ±19 % in dollars   |

---

## Diagnostics Recap  
- **Residuals vs Fitted:** no major non-linearity; slight funnel pattern  
- **Normal Q–Q:** residuals roughly Gaussian after log-transform  
- **Scale–Location:** WLS narrows spread of standardized residuals  
- **Cook’s Distance:** no single point dominates after collinearity pruning  

---

## What Are We Predicting?  
\[
  \text{log\_price} = \ln(\text{price in USD})
\]  
- **Why log?**  
  1. Stabilizes variance (prices are right-skewed)  
  2. Turns multiplicative errors into additive ones  

- **Back to dollars:**  
  \[
    \widehat{\text{price}} = \exp\!\bigl(\widehat{\ln(\text{price})}\bigr).
  \]

---

## Takeaways & Next Steps  
1. **Model:** log(price) ≃ linear combo of size, specs, body style, drivetrain, and brand  
2. **Collinearity:** raw size/engine variables were too inter-correlated; VIF dropping retained the cleanest signals  
3. **Generalization:** test R²≈0.89 shows strong out-of-sample performance  
4. **Reporting:** always exponentiate predictions back into dollar units  

### Ideas for further refinement  
- **PCA “size” index:** collapse length/width/height into a single score to further reduce multicollinearity  
- **“Luxury” submodel:** build a separate model for the handful of outliers (top-5 Cook’s distance) to improve high-end accuracy  
- **Robust losses:** try Huber or quantile regression, or milder WLS weights, if heteroskedasticity remains a concern  
