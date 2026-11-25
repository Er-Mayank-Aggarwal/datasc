# Basic pandas operations
import pandas as pd

df = pd.DataFrame({
    "Name": ["Asha", "Ravi", "Meena", "Karan"],
    "Age": [23, 31, 28, None],
    "City": ["Jodhpur", "Jaipur", "Jodhpur", "Delhi"],
    "Salary": [40000, 62000, 51000, 58000],
})
print(df)

# Look at the data
print(df.head(2))
print(df.shape)
print(df.describe())

# Select columns and rows
print(df["Name"])
print(df[["Name", "City"]])
print(df.iloc[0])            # first row

# Filter
print(df[df["Salary"] > 50000])
print(df[df["City"] == "Jodhpur"])

# Missing values
print(df.isnull().sum())
df["Age"] = df["Age"].fillna(df["Age"].mean())

# New column
df["Bonus"] = df["Salary"] * 0.1
print(df)

# Sort
print(df.sort_values("Salary", ascending=False))

# Group by
print(df.groupby("City")["Salary"].mean())

# Save and read CSV
df.to_csv("people.csv", index=False)
print(pd.read_csv("people.csv"))
