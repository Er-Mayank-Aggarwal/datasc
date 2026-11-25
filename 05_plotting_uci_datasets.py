"""05 - Plotting functions on UCI datasets
Normal curves, density & contour plots, correlation & scatter plots,
histograms and 3-D plots.

Datasets: Iris (UCI) and Wine (UCI) via scikit-learn (works offline).
pip install numpy pandas matplotlib seaborn scipy scikit-learn
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_iris, load_wine
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sns.set_style("whitegrid")

iris = load_iris(as_frame=True).frame
iris["species"] = iris["target"].map(dict(enumerate(load_iris().target_names)))
iris.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in iris.columns]
wine = load_wine(as_frame=True).frame

# =====================================================================
# 1. HISTOGRAMS
# =====================================================================
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].hist(iris["sepal_length"], bins=15, color="steelblue", edgecolor="k")
ax[0].set(title="Basic histogram", xlabel="Sepal length")
for sp, g in iris.groupby("species"):
    ax[1].hist(g["petal_length"], bins=12, alpha=.6, label=sp)
ax[1].set(title="Overlaid by species", xlabel="Petal length"); ax[1].legend()
ax[2].hist(wine["alcohol"], bins=20, density=True, cumulative=True,
           histtype="step", color="darkred")
ax[2].set(title="Cumulative (normalised)", xlabel="Alcohol")
plt.tight_layout(); plt.show()

iris.drop(columns="target").hist(figsize=(8, 6), bins=15)
plt.suptitle("All Iris features"); plt.tight_layout(); plt.show()

# =====================================================================
# 2. NORMAL CURVES (fit + theoretical)
# =====================================================================
x = iris["sepal_length"]
mu, sigma = stats.norm.fit(x)
grid = np.linspace(x.min() - 1, x.max() + 1, 300)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].hist(x, bins=15, density=True, alpha=.6, edgecolor="k")
ax[0].plot(grid, stats.norm.pdf(grid, mu, sigma), "r", lw=2)
ax[0].set(title=f"Normal fit: mu={mu:.2f}, sigma={sigma:.2f}")

xs = np.linspace(-10, 10, 400)
for m, s in [(0, 1), (0, 2), (2, 1), (-3, 0.7)]:
    ax[1].plot(xs, stats.norm.pdf(xs, m, s), label=f"mu={m}, sd={s}")
ax[1].set_title("Normal curves, various parameters"); ax[1].legend()

xs = np.linspace(-4, 4, 400)
ax[2].plot(xs, stats.norm.pdf(xs), "k")
for lo, hi, c in [(-1, 1, "green"), (-2, 2, "orange"), (-3, 3, "red")]:
    ax[2].fill_between(xs, stats.norm.pdf(xs), where=(xs >= lo) & (xs <= hi),
                       alpha=.25, color=c, label=f"+/-{hi} sd")
ax[2].set_title("68-95-99.7 rule"); ax[2].legend()
plt.tight_layout(); plt.show()

# Q-Q plot to check normality
stats.probplot(x, dist="norm", plot=plt)
plt.title("Q-Q plot: sepal length"); plt.show()
print("Shapiro-Wilk:", stats.shapiro(x))

# =====================================================================
# 3. DENSITY PLOTS
# =====================================================================
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.kdeplot(data=iris, x="petal_length", hue="species", fill=True, ax=ax[0])
ax[0].set_title("KDE by species")
sns.kdeplot(wine["alcohol"], fill=True, ax=ax[1], bw_adjust=0.5)
sns.kdeplot(wine["alcohol"], fill=False, ax=ax[1], bw_adjust=2, color="r")
ax[1].set_title("Bandwidth effect (wine alcohol)")
sns.violinplot(data=iris, x="species", y="sepal_width", ax=ax[2])
ax[2].set_title("Violin plot")
plt.tight_layout(); plt.show()

# =====================================================================
# 4. 2-D DENSITY & CONTOUR PLOTS
# =====================================================================
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
sns.kdeplot(data=iris, x="sepal_length", y="petal_length", fill=True,
            cmap="viridis", thresh=0.02, ax=ax[0])
ax[0].set_title("2-D KDE (filled)")

sns.kdeplot(data=iris, x="sepal_length", y="petal_length",
            hue="species", levels=5, ax=ax[1])
ax[1].set_title("Contours by species")

hb = ax[2].hexbin(wine["alcohol"], wine["flavanoids"], gridsize=15, cmap="Blues")
plt.colorbar(hb, ax=ax[2]); ax[2].set_title("Hexbin density (wine)")
plt.tight_layout(); plt.show()

# Contour plot of a function, with filled contours
X, Y = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
Z = np.exp(-(X ** 2 + Y ** 2)) + 0.5 * np.exp(-((X - 1.5) ** 2 + (Y + 1) ** 2))
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
cs = ax[0].contour(X, Y, Z, levels=12, cmap="RdYlBu")
ax[0].clabel(cs, inline=True, fontsize=8); ax[0].set_title("contour()")
cf = ax[1].contourf(X, Y, Z, levels=20, cmap="magma")
plt.colorbar(cf, ax=ax[1]); ax[1].set_title("contourf()")
plt.tight_layout(); plt.show()

# =====================================================================
# 5. SCATTER PLOTS
# =====================================================================
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].scatter(iris["sepal_length"], iris["petal_length"], c=iris["target"],
              cmap="viridis", s=40, edgecolor="k")
ax[0].set(title="Scatter coloured by class", xlabel="Sepal length", ylabel="Petal length")

sc = ax[1].scatter(wine["alcohol"], wine["color_intensity"],
                   s=wine["proline"] / 10, c=wine["flavanoids"], cmap="plasma", alpha=.7)
plt.colorbar(sc, ax=ax[1], label="flavanoids")
ax[1].set(title="Bubble chart (size=proline)", xlabel="Alcohol", ylabel="Colour intensity")

sns.regplot(data=iris, x="sepal_length", y="petal_length", ax=ax[2])
ax[2].set_title("Scatter + regression line")
plt.tight_layout(); plt.show()

sns.pairplot(iris.drop(columns="target"), hue="species", corner=True)
plt.show()
sns.jointplot(data=iris, x="sepal_length", y="petal_length", hue="species", kind="scatter")
plt.show()

# =====================================================================
# 6. CORRELATION PLOTS
# =====================================================================
corr = iris.drop(columns=["target", "species"]).corr()
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax[0])
ax[0].set_title("Iris correlation heatmap")

wcorr = wine.drop(columns="target").corr()
mask = np.triu(np.ones_like(wcorr, dtype=bool))
sns.heatmap(wcorr, mask=mask, cmap="vlag", center=0, square=True,
            cbar_kws={"shrink": .7}, ax=ax[1])
ax[1].set_title("Wine correlation (lower triangle)")
plt.tight_layout(); plt.show()

sns.clustermap(wcorr, cmap="vlag", center=0, figsize=(8, 8))
plt.show()

# =====================================================================
# 7. OTHER USEFUL PLOTS
# =====================================================================
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.boxplot(data=iris, x="species", y="sepal_length", ax=ax[0])
sns.barplot(data=iris, x="species", y="petal_width", errorbar="sd", ax=ax[1])
iris["species"].value_counts().plot.pie(autopct="%1.0f%%", ax=ax[2])
ax[2].set_ylabel("")
plt.tight_layout(); plt.show()

# =====================================================================
# 8. 3-D PLOTS
# =====================================================================
fig = plt.figure(figsize=(18, 5))

# 3-D scatter (Iris)
ax = fig.add_subplot(1, 3, 1, projection="3d")
p = ax.scatter(iris["sepal_length"], iris["sepal_width"], iris["petal_length"],
               c=iris["target"], cmap="viridis", s=40)
ax.set(xlabel="Sepal length", ylabel="Sepal width", zlabel="Petal length",
       title="3-D scatter (Iris)")

# 3-D surface
X, Y = np.meshgrid(np.linspace(-3, 3, 60), np.linspace(-3, 3, 60))
Z = np.sin(np.sqrt(X ** 2 + Y ** 2))
ax = fig.add_subplot(1, 3, 2, projection="3d")
ax.plot_surface(X, Y, Z, cmap="coolwarm", edgecolor="none")
ax.set_title("3-D surface")

# 3-D wireframe + contour projection
ax = fig.add_subplot(1, 3, 3, projection="3d")
ax.plot_wireframe(X, Y, Z, color="gray", linewidth=.5)
ax.contour(X, Y, Z, zdir="z", offset=-1.5, cmap="viridis")
ax.set_zlim(-1.5, 1); ax.set_title("Wireframe + contour")
plt.tight_layout(); plt.show()

# 3-D bar chart: mean features by class
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(projection="3d")
means = iris.groupby("species")[["sepal_length", "sepal_width",
                                 "petal_length", "petal_width"]].mean()
for i, (sp, row) in enumerate(means.iterrows()):
    ax.bar(np.arange(4), row.values, zs=i, zdir="y", alpha=.8)
ax.set_yticks(range(3)); ax.set_yticklabels(means.index)
ax.set_xticks(range(4)); ax.set_xticklabels(["SL", "SW", "PL", "PW"])
ax.set_title("3-D bar chart of feature means")
plt.show()

# 3-D density of two Iris variables (histogram2d as bars)
hist, xe, ye = np.histogram2d(iris["sepal_length"], iris["petal_length"], bins=8)
xpos, ypos = np.meshgrid(xe[:-1], ye[:-1], indexing="ij")
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(projection="3d")
ax.bar3d(xpos.ravel(), ypos.ravel(), 0, 0.2, 0.3, hist.ravel(), shade=True)
ax.set(xlabel="Sepal length", ylabel="Petal length", zlabel="Count",
       title="3-D histogram")
plt.show()
