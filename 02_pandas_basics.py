"""02 - Basic pandas operations"""
import numpy as np
import pandas as pd

# ---------- Series ----------
s = pd.Series([10, 20, 30, 40], index=["a", "b", "c", "d"])
print(s, "\n", s["b"], s[s > 15].values)

# ---------- DataFrame creation ----------
df = pd.DataFrame({
    "Name": ["Asha", "Ravi", "Meena", "Karan", "Sita", "Vikram"],
    "Age": [23, 31, 28, np.nan, 35, 29],
    "City": ["Jodhpur", "Jaipur", "Jodhpur", "Delhi", "Jaipur", "Delhi"],
    "Salary": [40000, 62000, 51000, 58000, 75000, np.nan],
})
print(df)

# ---------- Inspecting ----------
print(df.head(3)); print(df.tail(2))
print(df.shape, list(df.columns), df.index)
df.info()
print(df.describe())
print(df.dtypes)

# ---------- Selecting ----------
print(df["Name"])
print(df[["Name", "City"]])
print(df.loc[0:2, ["Name", "Age"]])     # label based
print(df.iloc[1:4, 0:3])                # position based
print(df[df["City"] == "Jodhpur"])
print(df[(df["Age"] > 25) & (df["Salary"] > 50000)])
print(df.query("City in ['Delhi','Jaipur'] and Salary > 55000"))

# ---------- Missing values ----------
print("Missing per column:\n", df.isnull().sum())
filled = df.copy()
filled["Age"] = filled["Age"].fillna(filled["Age"].mean())
filled["Salary"] = filled["Salary"].fillna(filled["Salary"].median())
print(filled)
print("Drop NA rows:\n", df.dropna())

# ---------- Adding / modifying / dropping ----------
filled["Bonus"] = filled["Salary"] * 0.10
filled["Senior"] = np.where(filled["Age"] > 30, "Yes", "No")
filled["Name_upper"] = filled["Name"].str.upper()
filled = filled.rename(columns={"Name_upper": "NAME"})
filled = filled.drop(columns=["NAME"])
print(filled)

# ---------- Sorting ----------
print(filled.sort_values("Salary", ascending=False))
print(filled.sort_values(["City", "Age"]))

# ---------- Group by & aggregation ----------
print(filled.groupby("City")["Salary"].mean())
print(filled.groupby("City").agg(avg_salary=("Salary", "mean"),
                                 max_age=("Age", "max"),
                                 count=("Name", "count")))
print(pd.pivot_table(filled, values="Salary", index="City",
                     columns="Senior", aggfunc="mean"))
print(filled["City"].value_counts())
print(pd.crosstab(filled["City"], filled["Senior"]))

# ---------- apply / map ----------
filled["Tax"] = filled["Salary"].apply(lambda x: x * 0.2 if x > 50000 else x * 0.1)
filled["CityCode"] = filled["City"].map({"Jodhpur": 1, "Jaipur": 2, "Delhi": 3})
print(filled)

# ---------- Merge / join / concat ----------
dept = pd.DataFrame({"Name": ["Asha", "Ravi", "Meena", "Sita"],
                     "Dept": ["HR", "IT", "IT", "Finance"]})
print(pd.merge(filled, dept, on="Name", how="left"))
print(pd.merge(filled, dept, on="Name", how="inner"))
print(pd.concat([filled.head(2), filled.tail(2)]))

# ---------- Duplicates ----------
dup = pd.concat([df, df.iloc[[0]]])
print("Duplicates:", dup.duplicated().sum())
print(dup.drop_duplicates().shape)

# ---------- Dates ----------
dates = pd.date_range("2024-01-01", periods=6, freq="D")
ts = pd.DataFrame({"Date": dates, "Sales": np.random.randint(100, 500, 6)})
ts["Month"] = ts["Date"].dt.month_name()
ts["Weekday"] = ts["Date"].dt.day_name()
ts = ts.set_index("Date")
print(ts)
print(ts["Sales"].rolling(3).mean())

# ---------- Correlation ----------
print(filled[["Age", "Salary", "Bonus"]].corr())

# ---------- Save ----------
filled.to_csv("employees.csv", index=False)
print(pd.read_csv("employees.csv").head())
