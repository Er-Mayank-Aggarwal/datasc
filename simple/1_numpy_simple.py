# Basic NumPy operations
import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array([[1, 2, 3], [4, 5, 6]])
print("a =", a)
print("b =\n", b)
print("shape:", b.shape)

# Special arrays
print(np.zeros((2, 3)))
print(np.ones((2, 2)))
print(np.arange(0, 10, 2))
print(np.linspace(0, 1, 5))

# Math
x = np.array([10, 20, 30])
y = np.array([1, 2, 3])
print("add:", x + y)
print("multiply:", x * y)
print("square:", x ** 2)

# Statistics
print("sum:", a.sum())
print("mean:", a.mean())
print("max:", a.max())
print("std:", a.std())

# Indexing and slicing
print("first two:", a[:2])
print("b[1, 2] =", b[1, 2])
print("greater than 2:", a[a > 2])

# Reshape and transpose
print(np.arange(6).reshape(2, 3))
print(b.T)

# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(A @ B)

# Random numbers
np.random.seed(0)
print(np.random.rand(3))
print(np.random.randint(1, 10, 5))
