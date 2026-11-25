"""01 - Basic NumPy operations"""
import numpy as np

# ---------- Creating arrays ----------
a = np.array([1, 2, 3, 4, 5])
b = np.array([[1, 2, 3], [4, 5, 6]])
print("1-D array:", a)
print("2-D array:\n", b)
print("Shape, ndim, size, dtype:", b.shape, b.ndim, b.size, b.dtype)

zeros = np.zeros((2, 3))
ones = np.ones((3, 2))
eye = np.eye(3)
rng = np.arange(0, 10, 2)
lin = np.linspace(0, 1, 5)
full = np.full((2, 2), 7)
print("zeros:\n", zeros, "\nones:\n", ones, "\neye:\n", eye)
print("arange:", rng, "| linspace:", lin, "\nfull:\n", full)

# ---------- Random numbers ----------
np.random.seed(42)
r1 = np.random.rand(3, 3)            # uniform [0,1)
r2 = np.random.randn(3, 3)           # standard normal
r3 = np.random.randint(1, 100, size=(2, 4))
print("rand:\n", r1, "\nrandn:\n", r2, "\nrandint:\n", r3)

# ---------- Indexing & slicing ----------
m = np.arange(1, 17).reshape(4, 4)
print("Matrix:\n", m)
print("Element [1,2]:", m[1, 2])
print("First row:", m[0], "| Last column:", m[:, -1])
print("Sub-matrix [1:3, 1:3]:\n", m[1:3, 1:3])
print("Elements > 10:", m[m > 10])           # boolean masking

# ---------- Element-wise arithmetic ----------
x = np.array([10, 20, 30])
y = np.array([1, 2, 3])
print("x+y:", x + y, "| x-y:", x - y, "| x*y:", x * y, "| x/y:", x / y)
print("x**2:", x ** 2, "| sqrt:", np.sqrt(x), "| exp(y):", np.exp(y))

# ---------- Broadcasting ----------
print("Broadcast add:\n", m + np.array([100, 200, 300, 400]))

# ---------- Aggregations ----------
print("sum:", m.sum(), "| mean:", m.mean(), "| std:", m.std(), "| var:", m.var())
print("min/max:", m.min(), m.max(), "| argmin/argmax:", m.argmin(), m.argmax())
print("Column sums:", m.sum(axis=0), "| Row means:", m.mean(axis=1))
print("Median:", np.median(m), "| 75th percentile:", np.percentile(m, 75))
print("Cumulative sum:", np.cumsum(x))

# ---------- Reshaping & stacking ----------
v = np.arange(12)
print("reshape(3,4):\n", v.reshape(3, 4))
print("flatten:", m.flatten()[:6], "...")
print("transpose:\n", b.T)
print("vstack:\n", np.vstack([x, y]))
print("hstack:", np.hstack([x, y]))
print("concatenate axis=1:\n", np.concatenate([b, b], axis=1))
print("split:", np.split(np.arange(6), 3))

# ---------- Linear algebra ----------
A = np.array([[2, 1], [1, 3]])
B = np.array([[1, 0], [4, 5]])
print("Dot product:\n", np.dot(A, B))
print("A @ B:\n", A @ B)
print("Determinant:", np.linalg.det(A))
print("Inverse:\n", np.linalg.inv(A))
vals, vecs = np.linalg.eig(A)
print("Eigenvalues:", vals, "\nEigenvectors:\n", vecs)
print("Solve Ax = [1,2]:", np.linalg.solve(A, np.array([1, 2])))

# ---------- Sorting, searching, unique ----------
d = np.array([5, 3, 8, 1, 9, 3])
print("sort:", np.sort(d), "| argsort:", np.argsort(d))
print("unique:", np.unique(d), "| where d>4:", np.where(d > 4))
print("np.where(cond, a, b):", np.where(d > 4, "big", "small"))

# ---------- Statistics ----------
data1 = np.random.randn(100)
data2 = 2 * data1 + np.random.randn(100) * 0.5
print("Correlation coef:\n", np.corrcoef(data1, data2))

# ---------- Save / load ----------
np.save("array.npy", m)
print("Loaded:\n", np.load("array.npy"))
np.savetxt("array.txt", m, fmt="%d", delimiter=",")
print("Loaded txt:\n", np.loadtxt("array.txt", delimiter=","))
