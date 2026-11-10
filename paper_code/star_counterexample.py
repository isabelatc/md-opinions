"""Counterexample to Euclidean monotonicity (Theorem 1(iii)): star graph, 6 leaves, weights 0.99."""
import numpy as np

# ===== Main Function =====

def star_counterexample(k: int=6, w: float=0.99, steps: int=4):
  """
  Prints the distance to equilibrium during the first <steps> rounds of the social process on a star
  graph with <k> leaves and weights <w>, using the Euclidean, (D + I)-weighted and sup norms.
  """
  # Star graph with the center as node 0
  n = k + 1
  A = np.zeros((n, n))
  A[0, 1:] = w
  A[1:, 0] = w
  D = np.diag(A.sum(1))
  L = D - A
  I = np.eye(n)

  # Innate opinions, equilibrium and update matrix
  s = np.array([-0.98, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16])
  z_eq = np.linalg.solve(L + I, s)
  M = np.linalg.solve(D + I, A)
  print(f"||M||_2 = {np.linalg.norm(M, 2):.3f}, rho(M) = {max(abs(np.linalg.eigvals(M))):.3f}, "
        f"rho_max = {(k * w) / (1 + k * w):.3f}")
  print("z* =", np.round(z_eq, 4))

  # Distance to equilibrium in each norm
  z = s.copy()
  for t in range(steps):
    e = z - z_eq
    print(t, f"euclid {np.linalg.norm(e):.4f}  weighted {np.sqrt(e @ (D + I) @ e):.4f}  sup {abs(e).max():.4f}")
    z = np.linalg.solve(D + I, s + A @ z)


if __name__ == "__main__":
  star_counterexample()
