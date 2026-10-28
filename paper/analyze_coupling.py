"""Summaries and figures for the coupled reruns of Experiments 2 and 3.

Usage: python paper/analyze_coupling.py <figure_output_dir>
Reads the JSON files in paper/results/ written by paper/exp_coupling.py.
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIG = sys.argv[1] if len(sys.argv) > 1 else RES

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "font.family": "serif", "mathtext.fontset": "cm"})


def style(ax, title, xlabel=r"cross-topic coupling $\alpha$"):
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", color=GRID, lw=0.5)


# ------------------------------------------------------------------ Reddit network
def reddit(variant=""):
    """Column-width figure: recommender's relative change of D and P versus alpha on Reddit.

    variant="" uses the thesis optimizer, variant="_mono" the monotone one.
    """
    res = json.load(open(os.path.join(RES, "reddit.json")))
    alphas = [r["alpha"] for r in res]
    pct = lambda d, k: 100 * (d[k][-1] / d[k][0] - 1)
    print("== Reddit (variant %r) ==" % (variant or "thesis"))
    for r in res:
        print(r["alpha"], " ".join(f"{v}: dD {pct(r[v + variant], 'D'):.1f}% dP {pct(r[v + variant], 'P'):.1f}% "
                                   f"dI {pct(r[v + variant], 'I'):.1f}% iters {r[v + variant]['iters']}" for v in ["NR", "RR"]))
    with plt.rc_context({"font.size": 7, "axes.titlesize": 7, "legend.fontsize": 7}):
        fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.65), constrained_layout=True)
        for j, v in enumerate(["NR", "RR"]):
            for k, met in enumerate(["D", "P"]):
                ax[k].plot(alphas, [pct(r[v + variant], met) for r in res], color=SERIES[j], lw=1.3,
                           marker="o", ms=3, label=v)
        for k, t in enumerate([r"(a) $\Delta\mathcal{D}_{z^*}$ (%)", r"(b) $\Delta\mathcal{P}_{z^*}$ (%)"]):
            ax[k].set_title(t)
            ax[k].set_xlabel(r"cross-topic coupling $\alpha$")
            ax[k].axhline(0, color=INK2, lw=0.6)
            ax[k].grid(axis="y", color=GRID, lw=0.5)
            ax[k].set_xticks(alphas)
            ax[k].tick_params(labelsize=6.5)
        ax[1].legend(frameon=False)
        fig.savefig(os.path.join(FIG, "reddit_alpha.pdf"))


# ------------------------------------------------- Experiment 2, original weak coupling
def e2weak():
    """Column-width figures for E2 with the thesis generator (weak cross-topic weights)."""
    rows = json.load(open(os.path.join(RES, "e2_weak.json")))
    prs, mes = ["PR1", "PR2", "PR3", "PR4"], ["ME1", "ME2", "ME3"]
    small = {"font.size": 7, "axes.titlesize": 7, "legend.fontsize": 7}

    with plt.rc_context(small):
        # Mean equilibrium metrics by initial-condition case
        fig, ax = plt.subplots(1, 3, figsize=(3.5, 1.75), constrained_layout=True)
        names = {"P": r"Polarization $\mathcal{P}_{z^*}$", "D": r"Disagreement $\mathcal{D}_{z^*}$",
                 "I": r"Internal conflict $\mathcal{I}_{z^*}$"}
        for k, met in enumerate(["P", "D", "I"]):
            for j, pr in enumerate(prs):
                y = [np.mean([r[met] for r in rows if r["PR"] == pr and r["ME"] == me]) for me in mes]
                ax[k].plot(range(3), y, color=SERIES[j], lw=1.3, marker="o", ms=3, label=pr)
            ax[k].set_xticks(range(3), mes)
            ax[k].set_title(names[met])
            ax[k].grid(axis="y", color=GRID, lw=0.5)
            ax[k].tick_params(labelsize=6.5)
        ax[0].set_ylabel("mean over 100 networks")
        fig.legend(*ax[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, frameon=False)
        fig.savefig(os.path.join(FIG, "e2_initial_conditions.pdf"))

        # Coupled vs topic-isolated equilibrium, per network and topic (all cases)
        fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.75), constrained_layout=True)
        pairs = [(r[f"P{a}"], r[f"P{a}_iso"]) for r in rows for a in (1, 2)]
        x, y = np.array(pairs).T
        ax[0].scatter(x, y, s=4, color=SERIES[0], alpha=0.35, lw=0)
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax[0].plot(lim, lim, color=INK2, lw=0.6, ls="--")
        ax[0].set_xscale("log"); ax[0].set_yscale("log")
        ax[0].set_title("Per-topic polarization")
        ax[0].set_xlabel("coupled ($m=2$)"); ax[0].set_ylabel("topic-isolated")
        pairs = [(r[f"D{a}"], r[f"D{a}_iso"]) for r in rows for a in (1, 2)]
        x, y = np.array(pairs).T
        ax[1].scatter(x, y, s=4, color=SERIES[1], alpha=0.35, lw=0)
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax[1].plot(lim, lim, color=INK2, lw=0.6, ls="--")
        ax[1].set_xlim(-6, 22); ax[1].set_ylim(-1, 22)
        ax[1].axvline(0, color=INK2, lw=0.5)
        ax[1].set_title("Per-topic disagreement")
        ax[1].set_xlabel("coupled ($m=2$)")
        for a_ in ax:
            a_.tick_params(labelsize=6.5)
        fig.savefig(os.path.join(FIG, "e2_isolated.pdf"))

    # Topic 1 vs topic 2 per metric (appendix, full width)
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.1), constrained_layout=True)
    for k, met in enumerate(["P", "D", "I"]):
        for j, me in enumerate(mes):
            sel = [r for r in rows if r["ME"] == me]
            ax[k].scatter([r[met + "1"] for r in sel], [r[met + "2"] for r in sel], s=5, alpha=0.4,
                          lw=0, color=SERIES[j], label=me)
        lo, hi = ax[k].get_xlim()
        ax[k].plot([lo, hi], [lo, hi], color=INK2, lw=0.6, ls="--")
        ax[k].set_title(names[met], fontsize=8)
        ax[k].set_xlabel("topic 1"); ax[k].grid(color=GRID, lw=0.5)
    ax[0].set_ylabel("topic 2")
    ax[0].legend(frameon=False, markerscale=2)
    fig.savefig(os.path.join(FIG, "e2_topics.pdf"))


# ------------------------------------------------------------------ Experiment 2
def e2():
    rows = json.load(open(os.path.join(RES, "e2_coupling.json")))
    alphas = sorted({r["alpha"] for r in rows})
    prs, mes = ["PR1", "PR2", "PR3", "PR4"], ["ME1", "ME2", "ME3"]
    key = lambda r: (r["case"], r["net"])
    base = {key(r): r for r in rows if r["alpha"] == 0.0}

    def ratio(metric, pr=None, me=None):
        out = []
        for a in alphas:
            v = [r[metric] / base[key(r)][metric] for r in rows if r["alpha"] == a
                 and (pr is None or r["PR"] == pr) and (me is None or r["ME"] == me) and base[key(r)][metric] > 0]
            out.append((np.median(v), np.percentile(v, 25), np.percentile(v, 75)))
        return np.array(out)

    def asym(pr, me, metric="P"):
        out = []
        for a in alphas:
            v = [abs(r[metric + "1"] - r[metric + "2"]) / (r[metric + "1"] + r[metric + "2"]) for r in rows
                 if r["alpha"] == a and r["PR"] == pr and r["ME"] == me and r[metric + "1"] + r[metric + "2"] > 0]
            out.append(np.median(v))
        return np.array(out)

    print("== E2: median ratio to alpha=0 at alpha=1 (by PR, all ME) ==")
    for pr in prs:
        print(pr, "P %.3f  D %.3f  I %.3f" % tuple(ratio(m, pr=pr)[-1, 0] for m in ["P", "D", "I"]))
    print("== E2: by ME at alpha=1 ==")
    for me in mes:
        print(me, "P %.3f  D %.3f  I %.3f" % tuple(ratio(m, me=me)[-1, 0] for m in ["P", "D", "I"]))
    print("== E2: ME3 polarization asymmetry |P1-P2|/(P1+P2) by alpha ==")
    for pr in prs:
        print(pr, np.round(asym(pr, "ME3"), 3))
    print("== E2: ME1/ME2 polarization asymmetry at alpha=0 and 1 ==")
    for me in ["ME1", "ME2"]:
        print(me, [np.round(asym(pr, me)[[0, -1]], 3).tolist() for pr in prs])
    print("== E2: mean energy change (M_s - M_z*)/P_s, median by alpha ==")
    for a in alphas:
        v = [(r["M_s"] - r["M"]) / r["Ps"] for r in rows if r["alpha"] == a and r["Ps"] > 0]
        print(a, "%.4f" % np.median(v), "IQR %.4f %.4f" % tuple(np.percentile(v, [25, 75])))
    print("== E2: topic-mean gap |mu1-mu2| median by alpha (ME3) ==")
    for a in alphas:
        v = [abs(r["mu1"] - r["mu2"]) for r in rows if r["alpha"] == a and r["ME"] == "ME3"]
        print(a, "%.4f" % np.median(v))
    print("== E2: per-topic D isolated/coupled, median by alpha ==")
    for a in alphas:
        v = [r["D1_iso"] / r["D1"] for r in rows if r["alpha"] == a and r["D1"] > 0]
        neg = np.mean([min(r["D1"], r["D2"]) < 0 for r in rows if r["alpha"] == a])
        print(a, "median %.3f" % np.median(v), "share of networks with a negative per-topic D: %.3f" % neg)

    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.0), constrained_layout=True)
    labels = {"P": r"polarization $\mathcal{P}$", "D": r"disagreement $\mathcal{D}$", "I": r"internal conflict $\mathcal{I}$"}
    for j, met in enumerate(["P", "D", "I"]):
        R = ratio(met)
        ax[0].fill_between(alphas, R[:, 1], R[:, 2], color=SERIES[j], alpha=0.15, lw=0)
        ax[0].plot(alphas, R[:, 0], color=SERIES[j], lw=1.6, marker="o", ms=3, label=labels[met])
    for j, pr in enumerate(prs):
        ax[1].plot(alphas, asym(pr, "ME3"), color=SERIES[j], lw=1.6, marker="o", ms=3, label=pr)
    neg = [np.mean([min(r["D1"], r["D2"]) < 0 for r in rows if r["alpha"] == a]) for a in alphas]
    ax[2].plot(alphas, neg, color=INK, lw=1.6, marker="o", ms=3)
    style(ax[0], "(a) Global metrics, all cases")
    style(ax[1], "(b) Topic asymmetry of polarization (ME3)")
    style(ax[2], "(c) Negative per-topic $\\mathcal{D}$")
    ax[0].set_ylabel(r"median ratio to $\alpha=0$")
    ax[1].set_ylabel(r"$|\mathcal{P}^{[1]}-\mathcal{P}^{[2]}|\,/\,(\mathcal{P}^{[1]}+\mathcal{P}^{[2]})$")
    ax[2].set_ylabel("share of networks")
    ax[0].set_ylim(0, 1.25); ax[1].set_ylim(0, 1); ax[2].set_ylim(0, 1)
    ax[0].legend(frameon=False, fontsize=7, loc="lower left")
    ax[1].legend(frameon=False, fontsize=7, loc="upper right", ncol=2)
    fig.savefig(os.path.join(FIG, "e2_coupling.pdf"))


# ------------------------------------------------------------------ Experiment 3
def e3(fname="e3_coupling_monotone.json", figname="e3_coupling.pdf"):
    res = json.load(open(os.path.join(RES, fname)))
    alphas = sorted({r["alpha"] for r in res})
    pct = lambda d, k: 100 * (d[k][-1] / d[k][0] - 1)

    print("== E3: networks", len({r["net"] for r in res}), "==")
    summary = {}
    for v in ["NR", "RR"]:
        for a in alphas:
            sel = [r[v] for r in res if r["alpha"] == a]
            dP = np.array([pct(d, "P") for d in sel]); dD = np.array([pct(d, "D") for d in sel])
            summary[(v, a)] = (dP, dD)
            # budget split: changes of P, 2D, I, M (they sum to zero), normalised by released budget -2*dD
            parts = np.array([[d["P"][-1] - d["P"][0], 2 * (d["D"][-1] - d["D"][0]), d["I"][-1] - d["I"][0],
                               d["M"][-1] - d["M"][0]] for d in sel])
            resid = np.abs(parts.sum(1)).max()
            moved = parts[:, 1] < 0  # runs that released disagreement
            rel = parts[moved][:, [0, 2, 3]] / (-parts[moved][:, [1]])
            print(f"{v} alpha={a}: dD% median {np.median(dD):.1f} (all<0: {np.all(dD < 0)}); "
                  f"dP% median {np.median(dP):.1f} [IQR {np.percentile(dP, 25):.1f},{np.percentile(dP, 75):.1f}] "
                  f"max {dP.max():.1f}; share dP>0 {np.mean(dP > 0):.2f}; "
                  f"budget to P/I/M (median) {np.round(np.median(rel, 0), 2).tolist()}; max residual {resid:.1e}")
    # per-network paired comparison alpha=0 vs alpha=1
    for v in ["NR", "RR"]:
        p0, p1 = summary[(v, 0.0)][0], summary[(v, 1.0)][0]
        print(f"{v}: networks where dP(alpha=1) < dP(alpha=0): {np.sum(p1 < p0)}/{len(p0)}")
    # topic concentration of polarization increases at alpha=0
    conc = []
    for r in res:
        if r["alpha"] == 0.0:
            d = r["NR"]; inc = np.array(d["Pt_final"]) - np.array(d["Pt_init"])
            if inc.sum() > 0 and r["m"] > 1:
                conc.append(inc.max() / np.abs(inc).sum())
    if conc:
        print("alpha=0 NR, networks with a P increase: share of |dP| in top topic, median %.2f (n=%d)" % (np.median(conc), len(conc)))

    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.0), constrained_layout=True)
    off = {"NR": -0.03, "RR": 0.03}
    col = {"NR": SERIES[0], "RR": SERIES[1]}
    for v in ["NR", "RR"]:
        for k in range(2):
            med = [np.median(summary[(v, a)][k]) for a in alphas]
            for a in alphas:
                y = summary[(v, a)][k]
                ax[k].scatter(np.full(len(y), a + off[v]), y, s=9, color=col[v], alpha=0.35, lw=0)
            ax[k].plot(np.array(alphas) + off[v], med, color=col[v], lw=1.6, marker="o", ms=4, label=v)
    for k, t in enumerate([r"(a) $\Delta\mathcal{P}_{z^*}$ (%)", r"(b) $\Delta\mathcal{D}_{z^*}$ (%)"]):
        style(ax[k], t)
        ax[k].axhline(0, color=INK2, lw=0.6)
    ax[0].set_ylim(-70, 175)
    ax[0].legend(frameon=False, fontsize=7)
    # budget split for NR
    width = 0.18
    labels = [r"$\Delta\mathcal{P}$", r"$\Delta\mathcal{I}$", r"$\Delta\mathcal{M}$"]
    for j, lab in enumerate(labels):
        vals = []
        for a in alphas:
            sel = [r["NR"] for r in res if r["alpha"] == a]
            parts = np.array([[d["P"][-1] - d["P"][0], 2 * (d["D"][-1] - d["D"][0]), d["I"][-1] - d["I"][0],
                               d["M"][-1] - d["M"][0]] for d in sel])
            moved = parts[:, 1] < 0
            vals.append(np.median(parts[moved][:, [0, 2, 3]][:, j] / (-parts[moved][:, 1])))
        ax[2].bar(np.arange(len(alphas)) + (j - 1) * width, vals, width=width - 0.02, color=SERIES[j], label=lab)
    ax[2].set_xticks(range(len(alphas)), [str(a) for a in alphas])
    ax[2].axhline(0, color=INK2, lw=0.6)
    style(ax[2], "(c) Released budget (NR, median)")
    ax[2].set_ylabel(r"share of $-2\Delta\mathcal{D}$")
    ax[2].set_ylim(-0.3, 1.5)
    ax[2].legend(frameon=False, fontsize=7, ncol=3, loc="upper right")
    fig.savefig(os.path.join(FIG, figname))


if __name__ == "__main__":
    for part in sys.argv[2:] or ["e2weak", "e2", "e3"]:  # e3 = monotone optimizer; e3thesis = thesis optimizer
        {"reddit": reddit, "reddit_mono": lambda: reddit("_mono"), "e2weak": e2weak, "e2": e2, "e3": e3, "e3thesis": lambda: e3("e3_coupling.json", "e3_coupling_thesis_optimizer.pdf")}[part]()
