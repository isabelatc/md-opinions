"""Numerical illustration of the mode decomposition theorem (homogeneous coupling A = C_alpha kron W).

Generates figures/coupling_sweep.pdf. Two-community SBM, n = 300, p_in = 0.1, p_out = 0.01,
edge weights U(0,1); innate opinions per topic drawn around community means U(-1,1), sd 0.3.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(2026)
SIZES, PIN, POUT, SD, NETS = [150, 150], 0.10, 0.01, 0.3, 30
ALPHAS = np.linspace(0, 1, 11)
MS = [2, 3, 5]
COL = {2: "#2a78d6", 3: "#eb6834", 5: "#1baf7a"}


def sbm_weights():
    n = sum(SIZES)
    lab = np.repeat(np.arange(len(SIZES)), SIZES)
    P = np.where(lab[:, None] == lab[None, :], PIN, POUT)
    W = np.triu((rng.random((n, n)) < P) * rng.uniform(0, 1, (n, n)), 1)
    return W + W.T, lab


def innate(lab, m):
    mu = rng.uniform(-1, 1, (len(SIZES), m))
    return np.clip(rng.normal(mu[lab].T, SD), -1, 1)  # shape (m, n)


def metrics(W, S, alpha, neutral=False):
    m, n = S.shape
    if neutral:  # budget-neutral coupling: rows of C sum to 1, total influence fixed
        C = (1 - alpha) * np.eye(m) + alpha / (m - 1) * (np.ones((m, m)) - np.eye(m))
    else:
        C = (1 - alpha) * np.eye(m) + alpha * np.ones((m, m))
    A = np.kron(C, W)
    L = np.diag(A.sum(1)) - A
    s = S.ravel()
    z = np.linalg.solve(L + np.eye(m * n), s)
    Z = z.reshape(m, n)
    Zc = Z - Z.mean(1, keepdims=True)
    P = (Zc ** 2).sum() / m
    D = z @ L @ z / m
    I = ((z - s) ** 2).sum() / m
    U = Z.mean(0)  # common mode
    Pu = ((U - U.mean()) ** 2).sum()
    mean_spread = Z.mean(1).std()
    return P, D, I, Pu, P - Pu, mean_spread


res = {m: np.zeros((NETS, len(ALPHAS), 6)) for m in MS}
neu = {m: np.zeros((NETS, len(ALPHAS), 6)) for m in MS}
for k in range(NETS):
    W, lab = sbm_weights()
    for m in MS:
        S = innate(lab, m)
        for j, a in enumerate(ALPHAS):
            res[m][k, j] = metrics(W, S, a)
            neu[m][k, j] = metrics(W, S, a, neutral=True)

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "font.family": "serif",
                     "mathtext.fontset": "cm"})
fig, ax = plt.subplots(1, 4, figsize=(7.16, 1.85), constrained_layout=True)


def band(axis, x, Y, color, label, ls="-"):
    mean = Y.mean(0)
    lo, hi = np.percentile(Y, [10, 90], axis=0)
    axis.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
    axis.plot(x, mean, color=color, lw=1.6, ls=ls, label=label)


for m in MS:
    R = res[m]
    band(ax[0], ALPHAS, R[:, :, 0] / R[:, :1, 0], COL[m], f"$m={m}$")
    band(ax[2], ALPHAS, R[:, :, 5] / R[:, :1, 5], COL[m], f"$m={m}$")
    N = neu[m]
    band(ax[3], ALPHAS, N[:, :, 0] / N[:, :1, 0], COL[m], f"$m={m}$")
R = res[3]
band(ax[1], ALPHAS, R[:, :, 3] / R[:, :1, 0], "#0b0b0b", "common")
band(ax[1], ALPHAS, R[:, :, 4] / R[:, :1, 0], "#0b0b0b", "contrast", ls="--")
ax[1].text(0.55, 0.17, "common", ha="left", fontsize=7, color="#0b0b0b")
ax[1].text(0.08, 0.62, "contrast", ha="left", fontsize=7, color="#0b0b0b")

titles = ["(a) Polarization $\\mathcal{P}_{z^*}$", "(b) Modes of $\\mathcal{P}_{z^*}$ ($m=3$)",
          "(c) Spread of topic means", "(d) Budget-neutral"]
for a_, t in zip(ax, titles):
    a_.set_title(t, fontsize=7.5)
    a_.set_xlabel("cross-topic coupling $\\alpha$")
    a_.grid(axis="y", color="#e6e5e0", lw=0.5)
    a_.set_ylim(bottom=0)
ax[0].set_ylabel("relative to $\\alpha=0$")
ax[0].legend(frameon=False, fontsize=7, loc="upper right")
fig.savefig("figures/coupling_sweep.pdf")

for m in MS:
    R = res[m].mean(0)
    Nn = neu[m].mean(0)
    print(m, "neutral P ratio a=1: %.3f D %.3f" % (Nn[-1, 0] / Nn[0, 0], Nn[-1, 1] / Nn[0, 1]))
    print(m, "P ratio a=1: %.3f  D ratio: %.3f  I ratio: %.3f  spread ratio: %.3f" %
          (R[-1, 0] / R[0, 0], R[-1, 1] / R[0, 1], R[-1, 2] / R[0, 2], R[-1, 5] / R[0, 5]))
