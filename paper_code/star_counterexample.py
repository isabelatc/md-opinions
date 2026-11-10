"""Counterexample to Euclidean monotonicity (Theorem 1(iii)): star graph, 6 leaves, weights 0.99."""
import numpy as np

k, w = 6, 0.99
n = k + 1
A = np.zeros((n, n)); A[0, 1:] = w; A[1:, 0] = w
D = np.diag(A.sum(1)); L = D - A; I = np.eye(n)
s = np.array([-0.98, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16])
zs = np.linalg.solve(L + I, s)
M = np.linalg.solve(D + I, A)
print("||M||_2 = %.3f, rho(M) = %.3f, rho_max = %.3f" % (np.linalg.norm(M, 2), max(abs(np.linalg.eigvals(M))), (k * w) / (1 + k * w)))
print("z* =", np.round(zs, 4))
z = s.copy()
for t in range(4):
    e = z - zs
    print(t, "euclid %.4f  weighted %.4f  sup %.4f" % (np.linalg.norm(e), np.sqrt(e @ (D + I) @ e), abs(e).max()))
    z = np.linalg.solve(D + I, s + A @ z)
