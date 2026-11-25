# Regression on the Diabetes and PIMA Indians datasets
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

# ---------- Part A: Linear regression (Diabetes dataset) ----------
data = load_diabetes(as_frame=True)
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("R2 score:", r2_score(y_test, pred))
print("MSE:", mean_squared_error(y_test, pred))
print("Coefficients:\n", pd.Series(model.coef_, index=X.columns))

# ---------- Part B: PIMA Indians dataset ----------
cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigree", "Age", "Outcome"]
url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
       "pima-indians-diabetes.data.csv")
pima = pd.read_csv(url, header=None, names=cols)   # or pd.read_csv("diabetes.csv")
print(pima.head())

X = pima.drop(columns="Outcome")
y = pima["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic regression (predicts diabetes: 0 or 1)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, clf.predict(X_test)))
