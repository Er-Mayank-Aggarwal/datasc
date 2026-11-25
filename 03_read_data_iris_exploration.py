"""03 - Reading from text files, Excel and the web + exploring the Iris dataset

pip install pandas numpy openpyxl scikit-learn seaborn matplotlib scipy
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

cols = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

# =====================================================================
# 1. READING FROM THE WEB (UCI repository) - with offline fallback
# =====================================================================
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
try:
    iris = pd.read_csv(URL, header=None, names=cols)
    print("Loaded from the web (UCI).")
except Exception as e:
    print("Web read failed (", e, ") -> using scikit-learn's copy.")
    d = load_iris(as_frame=True)
    iris = d.frame.copy()
    iris["species"] = d.target_names[d.target]
    iris.columns = cols

iris = iris.dropna()

# =====================================================================
# 2. READING FROM TEXT FILES
# =====================================================================
iris.to_csv("iris.csv", index=False)
iris.to_csv("iris.txt", sep="\t", index=False)

df_csv = pd.read_csv("iris.csv")
df_txt = pd.read_csv("iris.txt", sep="\t")                  # tab separated
df_txt2 = pd.read_table("iris.txt")                         # same thing
arr = np.genfromtxt("iris.csv", delimiter=",", skip_header=1,
                    usecols=(0, 1, 2, 3))                  # numeric to NumPy
print("CSV shape:", df_csv.shape, "| TXT shape:", df_txt.shape,
      "| NumPy:", arr.shape)

# plain Python reading
with open("iris.txt") as f:
    for i, line in enumerate(f):
        if i < 3:
            print(line.strip())

# =====================================================================
# 3. READING FROM EXCEL
# =====================================================================
with pd.ExcelWriter("iris.xlsx") as writer:
    iris.to_excel(writer, sheet_name="iris", index=False)
    iris.describe().to_excel(writer, sheet_name="summary")

df_xl = pd.read_excel("iris.xlsx", sheet_name="iris")
print("Excel sheets:", pd.ExcelFile("iris.xlsx").sheet_names)
print("Excel shape:", df_xl.shape)
# Useful read_excel options: header=, usecols="A:D", skiprows=, nrows=

# =====================================================================
# 4. EXPLORING THE DATA
# =====================================================================
print("\n--- Basic look ---")
print(iris.head()); print(iris.tail(3))
print("Shape:", iris.shape)
print("Columns:", iris.columns.tolist())
iris.info()
print(iris.dtypes)

print("\n--- Summary statistics ---")
print(iris.describe())
print(iris.describe(include="all"))
print("Missing values:\n", iris.isnull().sum())
print("Duplicate rows:", iris.duplicated().sum())
print("Unique species:", iris["species"].unique())
print(iris["species"].value_counts())

print("\n--- Group-wise analysis ---")
print(iris.groupby("species").mean(numeric_only=True))
print(iris.groupby("species").agg(["mean", "std", "min", "max"]).T)
print(iris.groupby("species")["petal_length"].describe())

print("\n--- Measures of shape & relation ---")
num = iris.drop(columns="species")
print("Skewness:\n", num.skew())
print("Kurtosis:\n", num.kurt())
print("Correlation:\n", num.corr())
print("Covariance:\n", num.cov())
print("Quantiles:\n", num.quantile([0.25, 0.5, 0.75]))

print("\n--- Filtering / sorting ---")
print(iris[iris["petal_length"] > 5].head())
print(iris.query("species == 'Iris-setosa' and sepal_width > 3.5")
      if "Iris-setosa" in iris["species"].values else "")
print(iris.sort_values("sepal_length", ascending=False).head())
iris["petal_area"] = iris["petal_length"] * iris["petal_width"]
iris["size_class"] = pd.cut(iris["sepal_length"], bins=3,
                            labels=["small", "medium", "large"])
print(iris["size_class"].value_counts())
print(pd.crosstab(iris["species"], iris["size_class"]))

# Outlier detection with IQR
Q1, Q3 = num.quantile(0.25), num.quantile(0.75)
IQR = Q3 - Q1
outliers = ((num < (Q1 - 1.5 * IQR)) | (num > (Q3 + 1.5 * IQR))).sum()
print("Outliers per column (IQR rule):\n", outliers)

# Hypothesis test: do species differ in petal length?
from scipy import stats
groups = [g["petal_length"].values for _, g in iris.groupby("species")]
f, p = stats.f_oneway(*groups)
print(f"\nANOVA petal_length across species: F={f:.2f}, p={p:.3g}")

# =====================================================================
# 5. QUICK VISUAL EXPLORATION
# =====================================================================
sns.pairplot(iris, hue="species")
plt.suptitle("Iris pairplot", y=1.02)
plt.show()

iris.drop(columns=["petal_area"]).select_dtypes("number").hist(figsize=(8, 6), bins=15)
plt.tight_layout(); plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(data=iris, x="species", y="petal_length")
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(num.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation heatmap"); plt.show()
