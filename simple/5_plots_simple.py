# Plotting on UCI datasets (Iris and Wine)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.datasets import load_iris, load_wine

iris = load_iris(as_frame=True).frame
wine = load_wine(as_frame=True).frame

# 1. Histogram
plt.hist(iris["sepal length (cm)"], bins=15)
plt.title("Histogram"); plt.xlabel("Sepal length"); plt.show()

# 2. Normal curve on top of a histogram
data = iris["sepal length (cm)"]
mu, sigma = data.mean(), data.std()
x = np.linspace(data.min(), data.max(), 100)
plt.hist(data, bins=15, density=True)
plt.plot(x, norm.pdf(x, mu, sigma), "r")
plt.title("Normal curve"); plt.show()

# 3. Density plot
sns.kdeplot(iris["petal length (cm)"], fill=True)
plt.title("Density plot"); plt.show()

# 4. Contour plot
xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
zz = np.exp(-(xx**2 + yy**2))
plt.contourf(xx, yy, zz)
plt.colorbar(); plt.title("Contour plot"); plt.show()

# 5. Scatter plot
plt.scatter(iris["sepal length (cm)"], iris["petal length (cm)"], c=iris["target"])
plt.xlabel("Sepal length"); plt.ylabel("Petal length")
plt.title("Scatter plot"); plt.show()

# 6. Correlation heatmap
sns.heatmap(iris.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation"); plt.show()

# 7. 3D plot
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(iris["sepal length (cm)"], iris["sepal width (cm)"],
           iris["petal length (cm)"], c=iris["target"])
ax.set_xlabel("Sepal length"); ax.set_ylabel("Sepal width"); ax.set_zlabel("Petal length")
plt.title("3D scatter"); plt.show()
