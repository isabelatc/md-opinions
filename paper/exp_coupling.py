"""Experiments 2 and 3 rerun with controlled cross-topic coupling alpha.

For every network the graph, innate opinions and same-topic weights are fixed, and the
cross-topic blocks are set to A[a,b] = alpha * (A[a,a] + A[b,b]) / 2 on the same edges.
alpha = 0 gives independent topics; the thesis setting corresponds to alpha ~ 0.003.

Usage (from the repository root):
  python paper/exp_coupling.py e2 [networks_per_case]
  python paper/exp_coupling.py e3 [networks]     (thesis optimizer)
  python paper/exp_coupling.py e3m [networks]    (monotone optimizer with backtracking)
Results are written to paper/results/.
"""
import os
import sys
import time
import random
import json
import numpy as np

SEED = 20261028
_shared_rng = np.random.Generator(np.random.PCG64(SEED))
_default_rng = np.random.default_rng
np.random.default_rng = lambda *args, **kwargs: _shared_rng  # make the repo's unseeded calls reproducible
random.seed(SEED)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scipy.sparse as sp
from md_network import MDNetwork, get_vector_topic
from utils import random_parameters
from exp2_init_conds import EXP2_generate_probs, EXP2_generate_means
from exp3_min_dsg import EXP3_create_nws, EXP3_gradient_A, EXP3_project_A

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)


def couple(A, n, m, alpha):
    """Returns a copy of A whose cross-topic blocks are alpha times the mean same-topic weights."""
    A = A.toarray() if sp.issparse(A) else A.copy()
    for a in range(m):
        for b in range(m):
            if a != b:
                Aaa = A[a * n:(a + 1) * n, a * n:(a + 1) * n]
                Abb = A[b * n:(b + 1) * n, b * n:(b + 1) * n]
                A[a * n:(a + 1) * n, b * n:(b + 1) * n] = alpha * (Aaa + Abb) / 2
    return A


def full_metrics(nw, z=None):
    """Global P, D, I, mean energy M, plus per-topic P, D, I and per-topic means."""
    m, n = nw.m, sum(nw.sizes)
    if z is None:
        z = nw.get_z_eq()
    P, D, I = nw.get_metrics(z=z)
    mu = np.array([get_vector_topic(z, a, m).mean() for a in range(m)])
    M = n / m * np.sum(mu ** 2)
    return dict(P=P.mean(), D=D.mean(), I=I.mean(), M=M, Pt=P, Dt=D, It=I, mu=mu)


# ---------------------------------------------------------------- Experiment 2
ALPHAS_E2 = [0.0, 0.1, 0.25, 0.5, 1.0]


def run_e2(per_case=100):
    mean_ranges = [[0.1, 0.3], [0.7, 0.9]]
    mu_ranges, k_ranges = [[0.05, 0.2], [0.25, 0.4]], [[20, 45], [55, 80]]
    rows = []
    case_id = 0
    for mu_c in range(2):
        for k_c in range(2):
            for c, t in [(0, 0), (1, 0), (1, 1)]:
                me = {(0, 0): "ME1", (1, 0): "ME2", (1, 1): "ME3"}[(c, t)]
                pr = f"PR{2 * mu_c + k_c + 1}"
                cc = int(_shared_rng.choice([0, 1])) if t == 1 else c  # contested topic for ME3
                case_id += 1
                t0 = time.time()
                for i in range(per_case):
                    while True:
                        try:
                            sizes, probs, means, std, m, _ = random_parameters(
                                c=[2, 2], p_func=EXP2_generate_probs, p_args=(mu_ranges[mu_c], k_ranges[k_c]),
                                mu_func=EXP2_generate_means, mu_args=(cc, t, *mean_ranges), max_m=2)
                            nw = MDNetwork(sizes=sizes, probs=probs, means=means, std=std, m=m)
                            break
                        except ValueError:
                            pass
                    n = sum(sizes)
                    s = nw.s
                    mu_s = np.array([get_vector_topic(s, a, 2).mean() for a in range(2)])
                    A0 = nw.A.toarray()
                    for alpha in ALPHAS_E2:
                        nw.update_MDN(sp.csr_matrix(couple(A0, n, 2, alpha)))
                        r = full_metrics(nw)
                        P1d, D1d, I1d = nw.get_metrics_1D()
                        rows.append(dict(case=f"{pr}/{me}", PR=pr, ME=me, net=i, alpha=alpha, n=n,
                                         P=r["P"], D=r["D"], I=r["I"], M=r["M"],
                                         M_s=n / 2 * np.sum(mu_s ** 2), Ps=float(np.mean([np.var(get_vector_topic(s, a, 2)) * n for a in range(2)])),
                                         P1=r["Pt"][0], P2=r["Pt"][1], D1=r["Dt"][0], D2=r["Dt"][1],
                                         I1=r["It"][0], I2=r["It"][1], mu1=r["mu"][0], mu2=r["mu"][1],
                                         P1_iso=P1d[0], P2_iso=P1d[1], D1_iso=D1d[0], D2_iso=D1d[1]))
                print(f"[E2] case {case_id}/12 {pr}/{me} done in {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(OUT, "e2_coupling.json"), "w") as f:
        json.dump(rows, f, default=float)


# ---------------------------------------------------------------- Experiment 3
ALPHAS_E3 = [0.0, 0.25, 0.5, 1.0]


def recommend(nw, lr_factor=0.5, epsilon=0.2, max_iters=100, tol=1e-3, reg=False, gamma=0.0, monotone=False):
    """Projected-gradient recommender of exp3_min_dsg.EXP3_min_disagreement, recording mean energy.

    monotone=False reproduces the thesis optimizer (full step, absolute tolerance on D).
    monotone=True halves the step until the objective f = D (+ gamma ||A||_F^2) decreases, stops when
    no step decreases it, and uses a relative tolerance |f_prev - f_new| < tol * f_prev.
    """
    if monotone:
        return _recommend_monotone(nw, lr_factor, epsilon, max_iters, tol, reg, gamma)
    A_init = nw.A.toarray()
    z = nw.get_z_eq()
    traj = [full_metrics(nw, z)]
    D_prev = traj[-1]["D"]
    it = 0
    for it in range(1, max_iters + 1):
        grad = EXP3_gradient_A(nw, z, reg=reg, gamma=gamma)
        A_prev = nw.A.toarray()
        lr = lr_factor * np.linalg.norm(A_prev, "fro") / np.linalg.norm(grad, "fro")
        A_proj = EXP3_project_A(nw, A_prev - lr * grad, A_init, epsilon)
        nw.update_MDN(sp.csr_matrix(A_proj))
        z = nw.get_z_eq()
        traj.append(full_metrics(nw, z))
        if abs(D_prev - traj[-1]["D"]) < tol:
            break
        D_prev = traj[-1]["D"]
    nw.update_MDN(sp.csr_matrix(A_init))
    keys = ["P", "D", "I", "M"]
    out = {k: [float(t[k]) for t in traj] for k in keys}
    out["Pt_init"] = traj[0]["Pt"].tolist()
    out["Pt_final"] = traj[-1]["Pt"].tolist()
    out["iters"] = it
    return out


def _recommend_monotone(nw, lr_factor, epsilon, max_iters, rtol, reg, gamma, max_halvings=12):
    A_init = nw.A.toarray()
    obj = lambda r: r["D"] + (gamma * np.linalg.norm(nw.A.toarray(), "fro") ** 2 if reg else 0.0)
    z = nw.get_z_eq()
    traj = [full_metrics(nw, z)]
    f_prev = obj(traj[-1])
    it = 0
    for it in range(1, max_iters + 1):
        grad = EXP3_gradient_A(nw, z, reg=reg, gamma=gamma)
        A_prev = nw.A.toarray()
        lr = lr_factor * np.linalg.norm(A_prev, "fro") / np.linalg.norm(grad, "fro")
        accepted = False
        for _ in range(max_halvings):
            A_proj = EXP3_project_A(nw, A_prev - lr * grad, A_init, epsilon)
            nw.update_MDN(sp.csr_matrix(A_proj))
            z_try = nw.get_z_eq()
            r_try = full_metrics(nw, z_try)
            f_try = obj(r_try)
            if f_try < f_prev:
                accepted = True
                break
            lr /= 2
        if not accepted:
            nw.update_MDN(sp.csr_matrix(A_prev))
            break
        z = z_try
        traj.append(r_try)
        if f_prev - f_try < rtol * f_prev:
            break
        f_prev = f_try
    nw.update_MDN(sp.csr_matrix(A_init))
    keys = ["P", "D", "I", "M"]
    out = {k: [float(t[k]) for t in traj] for k in keys}
    out["Pt_init"] = traj[0]["Pt"].tolist()
    out["Pt_final"] = traj[-1]["Pt"].tolist()
    out["iters"] = it
    return out


def run_e3(num=15, monotone=False):
    nws = EXP3_create_nws(num)
    results = []
    for k, nw in enumerate(nws):
        n, m = sum(nw.sizes), nw.m
        A0 = nw.A.toarray()
        t0 = time.time()
        for alpha in ALPHAS_E3:
            nw.update_MDN(sp.csr_matrix(couple(A0, n, m, alpha)))
            tol = 1e-4 if monotone else 1e-3
            nr = recommend(nw, tol=tol, monotone=monotone)
            rr = recommend(nw, reg=True, gamma=0.2, tol=tol, monotone=monotone)
            results.append(dict(net=k, n=int(n), m=int(m), alpha=alpha, NR=nr, RR=rr))
        nw.update_MDN(sp.csr_matrix(A0))
        print(f"[E3] network {k + 1}/{num} (n={n}, m={m}) done in {time.time() - t0:.0f}s", flush=True)
        fname = "e3_coupling_monotone.json" if monotone else "e3_coupling.json"
        with open(os.path.join(OUT, fname), "w") as f:
            json.dump(results, f, default=float)


if __name__ == "__main__":
    which = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if which == "e2":
        run_e2(count or 100)
    elif which == "e3":
        run_e3(count or 15)
    elif which == "e3m":
        run_e3(count or 15, monotone=True)
