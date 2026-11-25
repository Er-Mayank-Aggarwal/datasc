# Reading text, Excel and web data + exploring Iris
import pandas as pd
from sklearn.datasets import load_iris

# Load Iris (built in) and save it in different formats
iris = load_iris(as_frame=True).frame
iris["species"] = iris["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
iris = iris.drop(columns="target")

iris.to_csv("iris.csv", index=False)
iris.to_csv("iris.txt", sep="\t", index=False)
iris.to_excel("iris.xlsx", index=False)      # needs: pip install openpyxl

# Read from text file, CSV and Excel
df_txt = pd.read_csv("iris.txt", sep="\t")
df_csv = pd.read_csv("iris.csv")
df_xlsx = pd.read_excel("iris.xlsx")
print("Rows read:", len(df_txt), len(df_csv), len(df_xlsx))

# Read from the web
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
try:
    web = pd.read_csv(url, header=None)
    print("Web data:\n", web.head())
except Exception:
    print("No internet - skipping web download")

# Explore
df = df_csv
print(df.head())
print(df.shape)
print(df.columns)
df.info()
print(df.describe())
print(df["species"].value_counts())
print(df.isnull().sum())
print(df.groupby("species").mean())
print(df.corr(numeric_only=True))
