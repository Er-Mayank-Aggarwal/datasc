"""04 - Regression analysis on diabetes datasets

Part A: scikit-learn Diabetes dataset (442 patients, disease progression - linear regression)
Part B: PIMA Indians Diabetes dataset (UCI/Kaggle - linear + logistic regression)

pip install pandas numpy scikit-learn statsmodels matplotlib seaborn
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score, roc_curve)

# =====================================================================
# PART A - Diabetes (regression target: disease progression after 1 year)
# =====================================================================
print("=" * 60, "\nPART A: Diabetes dataset (sklearn)\n", "=" * 60)
data = load_diabetes(as_frame=True)
df = data.frame
X, y = df.drop(columns="target"), df["target"]
print(df.head()); print(df.describe().T)

# --- Simple linear regression (one feature: BMI) ---
Xb = X[["bmi"]]
Xtr, Xte, ytr, yte = train_test_split(Xb, y, test_size=0.2, random_state=42)
slr = LinearRegression().fit(Xtr, ytr)
pred = slr.predict(Xte)
print(f"\nSimple LR (bmi): slope={slr.coef_[0]:.2f}, intercept={slr.intercept_:.2f}, "
      f"R2={r2_score(yte, pred):.3f}, RMSE={np.sqrt(mean_squared_error(yte, pred)):.2f}")

plt.scatter(Xte, yte, alpha=.7, label="actual")
plt.plot(Xte.sort_values("bmi"), slr.predict(Xte.sort_values("bmi")), "r", label="fit")
plt.xlabel("BMI (standardised)"); plt.ylabel("Disease progression")
plt.title("Simple linear regression"); plt.legend(); plt.show()

# --- Multiple linear regression ---
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
mlr = LinearRegression().fit(Xtr, ytr)
pred = mlr.predict(Xte)
print(f"\nMultiple LR: R2={r2_score(yte, pred):.3f}, "
      f"MAE={mean_absolute_error(yte, pred):.2f}, "
      f"RMSE={np.sqrt(mean_squared_error(yte, pred)):.2f}")
print(pd.Series(mlr.coef_, index=X.columns).sort_values())
print("5-fold CV R2:", cross_val_score(LinearRegression(), X, y, cv=5, scoring="r2").round(3))

# --- Regularised models ---
for name, model in [("Ridge", Ridge(alpha=1.0)), ("Lasso", Lasso(alpha=0.1))]:
    model.fit(Xtr, ytr)
    print(f"{name}: R2={r2_score(yte, model.predict(Xte)):.3f}")

# --- Statsmodels: full statistical summary (p-values, CIs, F-stat) ---
ols = sm.OLS(y, sm.add_constant(X)).fit()
print(ols.summary())

# --- Residual diagnostics ---
resid = yte - pred
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].scatter(pred, resid, alpha=.7); ax[0].axhline(0, color="r")
ax[0].set(title="Residuals vs predicted", xlabel="Predicted", ylabel="Residual")
sns.histplot(resid, kde=True, ax=ax[1]); ax[1].set_title("Residual distribution")
sm.qqplot(resid, line="s", ax=ax[2]); ax[2].set_title("Q-Q plot")
plt.tight_layout(); plt.show()

ax = plt.figure(figsize=(5, 5)).gca()
ax.scatter(yte, pred, alpha=.7)
ax.plot([yte.min(), yte.max()], [yte.min(), yte.max()], "r--")
ax.set(xlabel="Actual", ylabel="Predicted", title="Actual vs predicted"); plt.show()

# =====================================================================
# PART B - PIMA Indians Diabetes
# =====================================================================
print("\n", "=" * 60, "\nPART B: PIMA Indians Diabetes\n", "=" * 60)
cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
URL = ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
       "pima-indians-diabetes.data.csv")
try:
    pima = pd.read_csv(URL, header=None, names=cols)
except Exception as e:
    print("Download failed:", e, "\nSave 'diabetes.csv' (Kaggle PIMA) locally instead.")
    pima = pd.read_csv("diabetes.csv")        # Kaggle file has same column names

print(pima.shape); print(pima.describe().T)

# In PIMA, zeros in these columns really mean "missing"
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
pima[zero_cols] = pima[zero_cols].replace(0, np.nan)
print("Missing:\n", pima.isnull().sum())
pima[zero_cols] = pima[zero_cols].fillna(pima[zero_cols].median())

plt.figure(figsize=(8, 6))
sns.heatmap(pima.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("PIMA correlation"); plt.show()

# --- B1: Linear regression - predict Glucose from other features ---
Xp = pima.drop(columns=["Glucose", "Outcome"])
yp = pima["Glucose"]
Xtr, Xte, ytr, yte = train_test_split(Xp, yp, test_size=0.2, random_state=42)
lr = LinearRegression().fit(Xtr, ytr)
print(f"\nPredict Glucose: R2={r2_score(yte, lr.predict(Xte)):.3f}, "
      f"RMSE={np.sqrt(mean_squared_error(yte, lr.predict(Xte))):.2f}")
print(sm.OLS(yp, sm.add_constant(Xp)).fit().summary())

# --- B2: Logistic regression - predict diabetes (Outcome) ---
Xc, yc = pima.drop(columns="Outcome"), pima["Outcome"]
Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)
scaler = StandardScaler().fit(Xtr)
clf = LogisticRegression(max_iter=1000).fit(scaler.transform(Xtr), ytr)
yhat = clf.predict(scaler.transform(Xte))
proba = clf.predict_proba(scaler.transform(Xte))[:, 1]

print(f"\nLogistic regression accuracy: {accuracy_score(yte, yhat):.3f}, "
      f"AUC: {roc_auc_score(yte, proba):.3f}")
print("Confusion matrix:\n", confusion_matrix(yte, yhat))
print(classification_report(yte, yhat))
print("Odds ratios:\n", pd.Series(np.exp(clf.coef_[0]), index=Xc.columns).sort_values())

logit = sm.Logit(yc, sm.add_constant(Xc)).fit(disp=0)
print(logit.summary())

fpr, tpr, _ = roc_curve(yte, proba)
plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(yte, proba):.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC - PIMA logistic regression"); plt.legend(); plt.show()
