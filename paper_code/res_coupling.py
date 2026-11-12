"""Summaries and figures for the coupled reruns of Experiments 2 and 3.

Usage: python paper_code/res_coupling.py <figure_output_dir> [parts]
Reads the JSON files in paper_code/results/ written by paper_code/exp_coupling.py.
"""
import os
import sys
import json
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIG = sys.argv[1] if len(sys.argv) > 1 else RES

# Enable LaTeX rendering for plots, with sizes for the final print size (IEEE column = 3.5 in)
mpl.rcParams.update({
  "text.usetex": True,
  "font.family": "serif",
  "font.serif": ["Computer Modern Roman"],
  "axes.unicode_minus": False,
  "font.size": 8,
  "axes.titlesize": 8,
  "axes.labelsize": 8,
  "xtick.labelsize": 7,
  "ytick.labelsize": 7,
  "legend.fontsize": 7,
  "axes.linewidth": 0.6,
  "xtick.major.width": 0.6,
  "ytick.major.width": 0.6,
  "xtick.major.size": 2.5,
  "ytick.major.size": 2.5,
  "axes.titlepad": 4,
  "axes.labelpad": 2,
  "legend.frameon": True,
  "legend.edgecolor": "#bbbbbb",
  "legend.framealpha": 0.9,
  "patch.linewidth": 0.5,
  "legend.handlelength": 1.6,
  "legend.borderaxespad": 0.3
})

# ===== Labels =====

MET_LABELS = {
  "P":    r"$\mathcal{P}_{z^*}$",
  "D":    r"$\mathcal{D}_{z^*}$",
  "I":    r"$\mathcal{I}_{z^*}$"
}

M_EXTRA_LABELS = {
  "dD":   r"$\Delta\mathcal{D}_{z^*}$",
  "dP":   r"$\Delta\mathcal{P}_{z^*}$",
  "dI":   r"$\Delta\mathcal{I}_{z^*}$"
}

MET_NAMES = {
  "P":    "Polarization",
  "D":    "Disagreement",
  "I":    "Internal Conflict"
}

COND_LABELS = {
  "NR": "Non-Regularized",
  "RR":  "Regularized"
}

# ===== Colors =====

# Thesis hues (purple, magenta, orange, green), adjusted for print and color-blind legibility
COLORS_4 = ["#6547C7", "#CC1F63", "#EA9418", "#3E7E3E"]
COLORS_2 = [COLORS_4[0], COLORS_4[2]]
C_1 = "#333333"

# ===== Constants =====

L_ARGS = {
  "color": "#9a9a9a",
  "linestyle": "--",
  "linewidth": 0.6,
  "zorder": 0
}

GRID_ARGS = {
  "color": "#e6e6e6",
  "linewidth": 0.4,
  "zorder": 0
}

P_ARGS = {
  "linewidth": 1.1,
  "marker": "o",
  "markersize": 3.5,
  "markeredgecolor": "white",
  "markeredgewidth": 0.4
}

S_ARGS = {
  "s": 5,
  "alpha": 0.45,
  "linewidths": 0,
  "rasterized": True
}

COLUMN, FULL = 3.5, 7.16

# ===== Plot Helpers =====

def make_fig(
    nrows: int=1, ncols: int=1,
    width: float=COLUMN, height: float=1.7,
    p_args: dict={}
    ) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
  """
  Creates a figure with specific parameters (sizes in inches, at print size). The spacing between
  plots can be changed with <p_args> (w_pad, h_pad, wspace, hspace).
  """
  f, axs = plt.subplots(nrows, ncols, figsize=(width, height), constrained_layout=True)
  f.set_constrained_layout_pads(**(dict(w_pad=0.02, h_pad=0.02, wspace=0.04) | p_args))
  return f, axs


def set_labels(ax: mpl.axes.Axes, title: str, xlabel: str, ylabel: str, grid: bool=True, pad: float=None):
  """
  Add formatted labels to a plot, and a light horizontal grid. <pad> is the space below the title.
  """
  ax.set_title(title, pad=pad)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  if grid:
    ax.grid(axis="y", **GRID_ARGS)
    ax.set_axisbelow(True)


def conds_handles(extra: list=[]) -> list:
  """
  Creates legend handles for non-regularized and regularized cases (EXP3).
  """
  m_args = {k: v for k, v in P_ARGS.items() if k != "linewidth"}
  handles = [Line2D([0], [0], color=COLORS_2[c], label=con, **m_args) for c, con in enumerate(COND_LABELS)]
  return handles + extra


def save_plot(filename: str):
  """
  Saves a plot into a file and closes it.
  """
  plt.savefig(filename, bbox_inches="tight", pad_inches=0.02, dpi=600)
  plt.close()


def get_pct_change(data: dict, metric: str) -> float:
  """
  Returns the difference in percentage between the first and last values of <metric>.
  """
  return 100 * (data[metric][-1] / data[metric][0] - 1)


# ===== Reddit Network (EXP3) =====

def EXP3_res_reddit(variant: str=""):
  """
  Prints the change of D, P and I caused by the recommendation algorithm on the Reddit network, and
  plots the change of D, P and I versus alpha.
  If variant is "", it uses the thesis optimizer (reddit_alpha.pdf), and if it is "_mono", the
  monotone one (reddit_alpha_mono.pdf).
  """
  res = json.load(open(os.path.join(RES, "exp3_reddit.json")))
  alphas = [r["alpha"] for r in res]
  pct = get_pct_change

  # Summary for each alpha
  print(f"== Reddit (variant {(variant or 'thesis')!r}) ==")
  for r in res:
    summary = []
    for v in ["NR", "RR"]:
      data = r[v + variant]
      summary.append(f"{v}: dD {pct(data, 'D'):.1f}% dP {pct(data, 'P'):.1f}% "
                     f"dI {pct(data, 'I'):.1f}% iters {data['iters']}")
    print(r["alpha"], " ".join(summary))

  # Figure
  f, axs = make_fig(ncols=3, height=1.6, p_args=dict(wspace=0.12))

  for i, (ax, met) in enumerate(zip(axs, ["D", "P", "I"])):
    for j, con in enumerate(COND_LABELS):
      diff = [pct(r[con + variant], met) for r in res]
      ax.plot(alphas, diff, color=COLORS_2[j], label=con, **(P_ARGS | {"linewidth": 0.7}))

    # Zero line, fewer alpha ticks (narrow plots) and labels
    ax.axhline(0, **L_ARGS)
    ax.set_xticks([0, 0.5, 1], [0, 0.5, 1])
    set_labels(ax, MET_NAMES[met], "$\\alpha$", f"{M_EXTRA_LABELS['d' + met]} (\\%)")

  # Shared legend below the plots
  f.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
  save_plot(os.path.join(FIG, f"reddit_alpha{variant}.pdf"))


# ===== Experiment 2, Original Weak Coupling =====

def EXP2_res_weak():
  """
  Plots the figures for EXP2 with the thesis generator (weak cross-topic weights): averaged metrics
  by case, coupled vs. topic-isolated metrics, and topic 1 vs. topic 2 (appendix).
  """
  rows = json.load(open(os.path.join(RES, "exp2_weak.json")))
  prs, mes = ["PR1", "PR2", "PR3", "PR4"], ["ME1", "ME2", "ME3"]

  # 1) Averaged metrics for each case of initial conditions
  f, axs = make_fig(ncols=3, height=1.8, p_args=dict(h_pad=0.04))

  for i, (ax, met) in enumerate(zip(axs, MET_LABELS)):
    means = [[np.mean([r[met] for r in rows if r["PR"] == pr and r["ME"] == me]) for me in mes]
             for pr in prs]
    for j in range(4):
      ax.plot(range(3), means[j], color=COLORS_4[j], label=f"PR{j + 1}", linestyle="none", **P_ARGS)
    # Labels, with the ME cases as x ticks
    set_labels(ax, MET_NAMES[met], "", f"Mean of {MET_LABELS[met]}")
    ax.set_xticks(range(3), mes)
    ax.set_xlim(-0.25, 2.25)

  # Shared legend below the plots
  handles, labels = axs[0].get_legend_handles_labels()
  f.legend(
      handles, labels, loc="outside lower center", ncol=4,
      columnspacing=2.0, handlelength=0.6, handletextpad=0.3
      )
  save_plot(os.path.join(FIG, "e2_initial_conditions.pdf"))

  # 2) Coupled (m = 2) vs. topic-isolated (m = 1) metrics, per network and topic
  f, axs = make_fig(ncols=2, height=1.75)

  for i, (ax, met) in enumerate(zip(axs, ["P", "D"])):
    for t in range(2):
      x_data = [r[f"{met}{t + 1}"] for r in rows]
      y_data = [r[f"{met}{t + 1}_iso"] for r in rows]
      ax.scatter(x_data, y_data, color=COLORS_2[t], label=f"Topic {t + 1}", **S_ARGS)

    set_labels(
        ax,
        f"Per-Topic {MET_NAMES[met]}",
        f"{MET_LABELS[met]} ($m = 2$)",
        f"{MET_LABELS[met]} ($m = 1$)",
        grid=False
        )

    # Polarization in log scale, disagreement around zero (it can be negative per topic)
    if met == "P":
      ax.set_xscale("log")
      ax.set_yscale("log")
    else:
      ax.set_xlim(-6, 22)
      ax.set_ylim(-1, 22)
      ax.axvline(0, **L_ARGS)

    # Diagonal (x = y) across the whole plot, and a small legend in an empty corner
    ax.axline((1, 1), (10, 10), **L_ARGS)
    ax.legend(
        loc="upper left" if met == "P" else "lower right", fontsize=6,
        markerscale=1.2, handlelength=0.8, handletextpad=0.1
        )

  save_plot(os.path.join(FIG, "e2_isolated.pdf"))

  # 3) Topic 1 vs. topic 2, one row per metric and one column per ME case (same limits in each row)
  me_names = {"ME1": "ME1 (Agree on Both Topics)", "ME2": "ME2 (Disagree on Both Topics)",
              "ME3": "ME3 (Disagree on One Topic)"}
  f, axs = make_fig(nrows=3, ncols=3, width=FULL, height=5.8, p_args=dict(h_pad=0.08, hspace=0.08))

  for i, met in enumerate(MET_LABELS):
    t1 = [r[met + "1"] for r in rows]
    t2 = [r[met + "2"] for r in rows]
    t_min, t_max = np.min(t1 + t2), np.max(t1 + t2)
    tol = (t_max - t_min) * 0.05
    lims = (t_min - tol, t_max + tol)

    for j, me in enumerate(mes):
      ax = axs[i, j]
      sel = [r for r in rows if r["ME"] == me]
      ax.scatter([r[met + "1"] for r in sel], [r[met + "2"] for r in sel], color=COLORS_4[j], **S_ARGS)
      ax.plot(lims, lims, **L_ARGS)

      # Titles only on the first row, y labels and ticks only on the first column
      set_labels(
          ax, me_names[me] if i == 0 else "",
          f"{MET_LABELS[met]}$^{{[1]}}$",
          f"{MET_LABELS[met]}$^{{[2]}}$" if j == 0 else "",
          grid=False, pad=10
          )
      ax.set_xlim(lims)
      ax.set_ylim(lims)
      if j > 0:
        ax.tick_params(labelleft=False)

  save_plot(os.path.join(FIG, "e2_topics.pdf"))


# ===== Experiment 2 =====

def EXP2_res_coupling():
  """
  Prints the summaries of EXP2 with controlled coupling (ratios to alpha = 0, polarization asymmetry
  between topics, mean energy, topic-mean gap and negative per-topic D), and plots them.
  """
  rows = json.load(open(os.path.join(RES, "exp2_coupling.json")))
  alphas = sorted({r["alpha"] for r in rows})
  prs, mes = ["PR1", "PR2", "PR3", "PR4"], ["ME1", "ME2", "ME3"]
  key = lambda r: (r["case"], r["net"])
  base = {key(r): r for r in rows if r["alpha"] == 0.0}

  def ratio(metric: str, pr: str=None, me: str=None) -> np.ndarray:
    """
    Returns the median and quartiles of <metric> relative to alpha = 0, for every alpha.
    """
    out = []
    for a in alphas:
      v = [r[metric] / base[key(r)][metric] for r in rows if r["alpha"] == a
           and (pr is None or r["PR"] == pr) and (me is None or r["ME"] == me) and base[key(r)][metric] > 0]
      out.append((np.median(v), np.percentile(v, 25), np.percentile(v, 75)))
    return np.array(out)

  def asym(pr: str, me: str, metric: str="P") -> np.ndarray:
    """
    Returns the median of |X1 - X2| / (X1 + X2) for <metric> X, for every alpha.
    """
    out = []
    for a in alphas:
      v = [abs(r[metric + "1"] - r[metric + "2"]) / (r[metric + "1"] + r[metric + "2"]) for r in rows
           if r["alpha"] == a and r["PR"] == pr and r["ME"] == me and r[metric + "1"] + r[metric + "2"] > 0]
      out.append(np.median(v))
    return np.array(out)

  # Ratios to alpha = 0 at alpha = 1
  print("== E2: median ratio to alpha=0 at alpha=1 (by PR, all ME) ==")
  for pr in prs:
    P_r, D_r, I_r = [ratio(met, pr=pr)[-1, 0] for met in ["P", "D", "I"]]
    print(pr, f"P {P_r:.3f}  D {D_r:.3f}  I {I_r:.3f}")
  print("== E2: by ME at alpha=1 ==")
  for me in mes:
    P_r, D_r, I_r = [ratio(met, me=me)[-1, 0] for met in ["P", "D", "I"]]
    print(me, f"P {P_r:.3f}  D {D_r:.3f}  I {I_r:.3f}")

  # Polarization asymmetry between topics
  print("== E2: ME3 polarization asymmetry |P1-P2|/(P1+P2) by alpha ==")
  for pr in prs:
    print(pr, np.round(asym(pr, "ME3"), 3))
  print("== E2: ME1/ME2 polarization asymmetry at alpha=0 and 1 ==")
  for me in ["ME1", "ME2"]:
    print(me, [np.round(asym(pr, me)[[0, -1]], 3).tolist() for pr in prs])

  # Mean energy and topic means
  print("== E2: mean energy change (M_s - M_z*)/P_s, median by alpha ==")
  for a in alphas:
    v = [(r["M_s"] - r["M"]) / r["Ps"] for r in rows if r["alpha"] == a and r["Ps"] > 0]
    q1, q3 = np.percentile(v, [25, 75])
    print(a, f"{np.median(v):.4f}", f"IQR {q1:.4f} {q3:.4f}")
  print("== E2: topic-mean gap |mu1-mu2| median by alpha (ME3) ==")
  for a in alphas:
    v = [abs(r["mu1"] - r["mu2"]) for r in rows if r["alpha"] == a and r["ME"] == "ME3"]
    print(a, f"{np.median(v):.4f}")

  # Per-topic disagreement
  print("== E2: per-topic D isolated/coupled, median by alpha ==")
  for a in alphas:
    v = [r["D1_iso"] / r["D1"] for r in rows if r["alpha"] == a and r["D1"] > 0]
    neg = np.mean([min(r["D1"], r["D2"]) < 0 for r in rows if r["alpha"] == a])
    print(a, f"median {np.median(v):.3f}", f"share of networks with a negative per-topic D: {neg:.3f}")

  # Figure
  f, axs = make_fig(ncols=3, width=FULL, height=1.95, p_args=dict(wspace=0.08))

  # (a) Global metrics relative to alpha = 0, with interquartile ranges
  for j, met in enumerate(MET_LABELS):
    R = ratio(met)
    axs[0].fill_between(alphas, R[:, 1], R[:, 2], color=COLORS_4[j], alpha=0.15, lw=0)
    axs[0].plot(alphas, R[:, 0], color=COLORS_4[j], label=MET_LABELS[met], **P_ARGS)
  axs[0].axhline(1, **L_ARGS)
  set_labels(axs[0], "(a) Global Metrics", "$\\alpha$", "Median Ratio to $\\alpha = 0$", pad=10)
  axs[0].set_ylim(0, 1.25)
  axs[0].legend(loc="lower left")

  # (b) Polarization asymmetry between topics (ME3)
  for j, pr in enumerate(prs):
    axs[1].plot(alphas, asym(pr, "ME3"), color=COLORS_4[j], label=pr, **P_ARGS)
  set_labels(
      axs[1], "(b) Topic Asymmetry of Polarization (ME3)", "$\\alpha$",
      r"$|\mathcal{P}^{[1]}-\mathcal{P}^{[2]}|\,/\,(\mathcal{P}^{[1]}+\mathcal{P}^{[2]})$",
      pad=10
      )
  axs[1].set_ylim(0, 1)
  axs[1].legend(
      loc="upper center", ncol=4, columnspacing=0.6, handlelength=1.0,
      handletextpad=0.4, borderaxespad=0.5
      )

  # (c) Networks with a negative per-topic disagreement
  neg = [np.mean([min(r["D1"], r["D2"]) < 0 for r in rows if r["alpha"] == a]) for a in alphas]
  axs[2].plot(alphas, neg, color=C_1, **P_ARGS)
  set_labels(axs[2], "(c) Negative Per-Topic Disagreement", "$\\alpha$", "Share of Networks", pad=10)
  axs[2].set_ylim(0, 1)

  # Same alpha ticks for all plots
  for ax in axs:
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], [0, 0.25, 0.5, 0.75, 1])
  save_plot(os.path.join(FIG, "e2_coupling.pdf"))


# ===== Experiment 3 =====

def EXP3_res_coupling(fname: str="exp3_coupling_monotone.json", figname: str="e3_coupling.pdf"):
  """
  Prints the summaries of EXP3 with controlled coupling (change of P and D, and how the released
  disagreement is split into P, I and M), and plots the change of P and D for every alpha (values
  outside the shared y range are clipped and marked).
  If fname is the thesis optimizer's results, figname should be e3_coupling_thesis_optimizer.pdf.
  """
  res = json.load(open(os.path.join(RES, fname)))
  alphas = sorted({r["alpha"] for r in res})
  pct = get_pct_change

  # Summary for each process and alpha
  print("== E3: networks", len({r["net"] for r in res}), "==")
  summary = {}
  for v in ["NR", "RR"]:
    for a in alphas:
      sel = [r[v] for r in res if r["alpha"] == a]
      dP = np.array([pct(d, "P") for d in sel])
      dD = np.array([pct(d, "D") for d in sel])
      summary[(v, a)] = (dP, dD)

      # Budget split: changes of P, 2D, I, M (they sum to zero), normalized by released budget -2*dD
      parts = np.array([[d["P"][-1] - d["P"][0], 2 * (d["D"][-1] - d["D"][0]), d["I"][-1] - d["I"][0],
                         d["M"][-1] - d["M"][0]] for d in sel])
      resid = np.abs(parts.sum(1)).max()
      moved = parts[:, 1] < 0  # runs that released disagreement
      rel = parts[moved][:, [0, 2, 3]] / (-parts[moved][:, [1]])
      print(f"{v} alpha={a}: dD% median {np.median(dD):.1f} (all<0: {np.all(dD < 0)}); "
            f"dP% median {np.median(dP):.1f} [IQR {np.percentile(dP, 25):.1f},{np.percentile(dP, 75):.1f}] "
            f"max {dP.max():.1f}; share dP>0 {np.mean(dP > 0):.2f}; "
            f"budget to P/I/M (median) {np.round(np.median(rel, 0), 2).tolist()}; max residual {resid:.1e}")

  # Per-network paired comparison of alpha = 0 and alpha = 1
  for v in ["NR", "RR"]:
    p0, p1 = summary[(v, 0.0)][0], summary[(v, 1.0)][0]
    print(f"{v}: networks where dP(alpha=1) < dP(alpha=0): {np.sum(p1 < p0)}/{len(p0)}")

  # Topic concentration of polarization increases at alpha = 0
  conc = []
  for r in res:
    if r["alpha"] == 0.0:
      d = r["NR"]
      inc = np.array(d["Pt_final"]) - np.array(d["Pt_init"])
      if inc.sum() > 0 and r["m"] > 1:
        conc.append(inc.max() / np.abs(inc).sum())
  if conc:
    print(f"alpha=0 NR, networks with a P increase: share of |dP| in top topic, "
          f"median {np.median(conc):.2f} (n={len(conc)})")

  # Figure
  f, axs = make_fig(ncols=2, width=5.4, height=1.9, p_args=dict(wspace=0.08))
  off = {"NR": -0.03, "RR": 0.03}
  y_lim = (-70, 70)  # same scale for both panels, values outside are clipped and marked

  # Change of polarization and disagreement per network (dots) and medians (lines)
  for j, con in enumerate(COND_LABELS):
    for k in range(2):
      med = [np.median(summary[(con, a)][k]) for a in alphas]
      for a in alphas:
        y = summary[(con, a)][k]
        x = a + off[con]
        inside = (y >= y_lim[0]) & (y <= y_lim[1])
        axs[k].scatter(np.full(inside.sum(), x), y[inside], color=COLORS_2[j], s=7, alpha=0.35, linewidths=0)

        # Clipped values: a triangle on the border and their values
        for out, y_mark, mark in [(y[y > y_lim[1]], y_lim[1] - 5 - 10 * j, "^"),
                                  (y[y < y_lim[0]], y_lim[0] + 5 + 10 * j, "v")]:
          if len(out):
            axs[k].plot(x, y_mark, marker=mark, color=COLORS_2[j], markersize=3.5, linestyle="none")
            values = ", ".join(f"{v:.0f}" for v in sorted(out, reverse=True))
            axs[k].text(x + 0.025, y_mark, f"{values}\\%", fontsize=5, color="#555555", va="center", ha="left")
      axs[k].plot(np.array(alphas) + off[con], med, color=COLORS_2[j], **P_ARGS)

  # Labels, shared y range and one legend per plot
  for k, met in enumerate(["P", "D"]):
    axs[k].axhline(0, **L_ARGS)
    axs[k].set_xticks(alphas, alphas)
    axs[k].set_ylim(y_lim)
    set_labels(axs[k], f"({'ab'[k]}) {MET_NAMES[met]}", "$\\alpha$", f"{M_EXTRA_LABELS['d' + met]} (\\%)")
  for ax in axs:
    ax.legend(handles=conds_handles(), loc="upper right")
  save_plot(os.path.join(FIG, figname))


# ===== Command Line =====

if __name__ == "__main__":
  # e3 uses the monotone optimizer, and e3thesis the thesis optimizer
  parts = {
      "reddit": EXP3_res_reddit,
      "reddit_mono": lambda: EXP3_res_reddit("_mono"),
      "e2weak": EXP2_res_weak,
      "e2": EXP2_res_coupling,
      "e3": EXP3_res_coupling,
      "e3thesis": lambda: EXP3_res_coupling("exp3_coupling.json", "e3_coupling_thesis_optimizer.pdf")
      }
  for part in sys.argv[2:] or ["e2weak", "e2", "e3"]:
    parts[part]()
