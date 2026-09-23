"""Re-render the two figures whose artwork carries notation, with b split
into k (timestep band) and q (bit-width) to match the UPCON text.
Derived from ../make_col_figs.py; writes to upcon/figdir/.
"""
"""Rebuild every figure at SINGLE-COLUMN aspect.

The previous figures were wide and short: in a two-column layout each one had
to be set across the full text width, which forces a page break and leaves the
rest of that page empty.  Five of eleven pages held nothing but a caption.

These are drawn tall and narrow (3.35 in wide, the column measure) so each
figure sits inside one column and body text flows past it on the other side.
Content is unchanged -- every number is read from the same CSVs.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
import vizstyle as V

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # paper_final
FIG = os.path.join(HERE, "upcon", "figdir")
os.makedirs(FIG, exist_ok=True)

COL = 3.35                      # column measure in inches
V.apply(6.6)

GROUPS = ["down_early", "down_late", "mid", "up_all"]
BANDS = ["b0_detail", "b1_mid_lo", "b2_mid_hi", "b3_noise"]
BLAB = ["0-.25", ".25-.5", ".5-.75", ".75-1"]


def grid(df, bits):
    """D[g,b] as a 4x4 array for one bit-width."""
    d = df[df.bits == bits]
    M = np.zeros((len(GROUPS), len(BANDS)))
    for i, g in enumerate(GROUPS):
        for j, b in enumerate(BANDS):
            r = d[(d.layer_group == g) & (d.timestep_band == b)]
            M[i, j] = r.damage_rms.iloc[0]
    return M


def main_effects(M):
    """Two-way main-effects fit in log space (the separable null)."""
    L = np.log(np.clip(M, 1e-12, None))
    mu = L.mean()
    a = L.mean(axis=1) - mu
    c = L.mean(axis=0) - mu
    return np.exp(mu + a[:, None] + c[None, :])


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("  wrote", name, "%.2f x %.2f in" % tuple(fig.get_size_inches()))


def panel_tag(ax, s, dx=-0.16, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=7.6,
            fontweight="bold", color=V.INK, va="bottom", ha="left")


# --------------------------------------------------------------------------
# Figure 1 -- the probe, drawn as a schematic
# --------------------------------------------------------------------------
def fig_probe():
    fig = plt.figure(figsize=(COL, 2.95))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.02, 1.0], hspace=0.18)

    # (a) paired sampling from one noise draw
    ax = fig.add_subplot(gs[0])
    V.bare(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0.1, 4.6)

    ax.add_patch(FancyBboxPatch((0.08, 1.92), 1.18, 0.80,
                                boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc="#f2f1ec", ec=V.MUTED, lw=0.6))
    ax.text(0.67, 2.32, "same\nnoise", ha="center", va="center",
            fontsize=6.0, color=V.INK, linespacing=1.15)

    for row, (y, lab, col) in enumerate(
            [(3.50, "reference: FP32 at every layer, every step", V.INK2),
             (1.12, "probe: one cell at q bits, the rest FP32",
              V.SERIES["down_early"])]):
        ax.annotate("", xy=(1.42, y), xytext=(1.26, 2.32),
                    arrowprops=dict(arrowstyle="-", lw=0.7, color=V.MUTED,
                                    connectionstyle="arc3,rad=%.2f" % (
                                        -0.22 if row == 0 else 0.22)))
        ax.text(1.42, y + 0.44, lab, ha="left", va="bottom", fontsize=5.8,
                color=col)
        for k in range(4):
            x = 1.42 + k * 1.72
            on = (row == 1 and k == 2)
            ax.add_patch(Rectangle((x, y - 0.30), 1.52, 0.60,
                                   fc=col if on else "#f2f1ec",
                                   ec=col if on else V.MUTED,
                                   lw=0.9 if on else 0.55, zorder=2))
            ax.text(x + 0.76, y, "q bits" if on else "32",
                    ha="center", va="center", fontsize=5.6,
                    color="white" if on else V.MUTED, zorder=3)
            if k < 3:
                ax.annotate("", xy=(x + 1.70, y), xytext=(x + 1.56, y),
                            arrowprops=dict(arrowstyle="-|>", lw=0.6,
                                            color=V.MUTED, mutation_scale=5))

    ax.plot([8.50, 8.72, 8.72, 8.50], [3.50, 3.50, 1.12, 1.12],
            lw=0.7, color=V.INK, solid_joinstyle="miter", clip_on=False)
    ax.text(8.86, 2.31, "D", ha="left", va="center", fontsize=7.4,
            style="italic", color=V.INK)
    ax.text(8.86, 1.88, "paired\nRMS", ha="left", va="center", fontsize=5.4,
            color=V.INK2, linespacing=1.15)
    ax.annotate("", xy=(8.30, 4.34), xytext=(1.42, 4.34),
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color=V.GRID,
                                mutation_scale=6))
    ax.text(1.42, 4.40, "denoising trajectory, detail end to noise end",
            fontsize=5.6, color=V.MUTED, va="bottom")
    panel_tag(ax, "a", dx=-0.01, dy=0.99)

    # (b) the cell lattice
    ax = fig.add_subplot(gs[1])
    V.bare(ax)
    ax.set_xlim(-1.35, 4.10)
    ax.set_ylim(-1.05, 4.62)
    for i in range(4):
        for j in range(4):
            on = (i == 0 and j == 2)
            ax.add_patch(Rectangle((j + 0.04, 3 - i + 0.04), 0.92, 0.92,
                                   fc=V.SERIES["down_early"] if on else "#f2f1ec",
                                   ec=V.SERIES["down_early"] if on else V.GRID,
                                   lw=0.9 if on else 0.6))
    ax.annotate("one cell (g, k)", xy=(2.72, 4.02), xytext=(4.06, 4.46),
                fontsize=5.9, color=V.SERIES["down_early"], ha="right",
                va="bottom",
                arrowprops=dict(arrowstyle="-", lw=0.6,
                                color=V.SERIES["down_early"]))
    for i, g in enumerate(GROUPS):
        ax.text(-0.10, 3 - i + 0.5, V.GLAB[g], ha="right", va="center",
                fontsize=5.8, color=V.INK2)
    for j, b in enumerate(BLAB):
        ax.text(j + 0.5, -0.14, b, ha="center", va="top", fontsize=5.5,
                color=V.INK2)
    ax.text(2.0, -0.60, "timestep band  t/T", ha="center", va="top",
            fontsize=6.0, color=V.INK2)
    ax.text(-1.32, 2.0, "layer group", rotation=90, ha="left", va="center",
            fontsize=6.0, color=V.INK2)
    panel_tag(ax, "b", dx=-0.005, dy=0.99)
    save(fig, "f1_probe.png")


# --------------------------------------------------------------------------
# Figure 2 -- the two regimes
# --------------------------------------------------------------------------
def fig_regimes(cif, cel, an):
    fig = plt.figure(figsize=(COL, 4.9))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.95], hspace=0.78)

    for k, (df, bits, ttl, note) in enumerate([
            (cif, 1, "CIFAR-10 DDPM, W1", "profiles cross"),
            (cel, 1, "CelebA-HQ 256, W1", "profiles run parallel")]):
        ax = fig.add_subplot(gs[k])
        M = grid(df, bits)
        # direct labels live INSIDE the axes, so the saved figure is never
        # wider than the column measure
        order = np.argsort(-M[:, 3])
        placed = []
        for i in order:
            ax.plot(range(4), M[i], "-o", ms=2.6, lw=1.15,
                    color=V.SERIES[GROUPS[i]], clip_on=False, zorder=3)
            ly = np.log10(M[i, 3])
            for q in placed:                       # keep labels from colliding
                if abs(ly - q) < 0.115:
                    ly = q - 0.115
            placed.append(ly)
            ax.annotate(V.GLAB[GROUPS[i]], xy=(3.08, 10 ** ly),
                        fontsize=5.7, color=V.SERIES[GROUPS[i]], va="center")
        ax.set_yscale("log")
        ax.set_xticks(range(4))
        ax.set_xticklabels(BLAB, fontsize=5.7)
        ax.set_xlim(-0.10, 4.02)
        ax.tick_params(labelsize=5.7)
        ax.grid(axis="y", color=V.GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylabel("damage (RMS)", fontsize=6.2)
        ax.set_title("%s  --  %s" % (ttl, note), fontsize=6.4,
                     color=V.INK, loc="left", pad=3)
        if k == 1:
            ax.set_xlabel("detail end  ->  noise end", fontsize=6.2)
        panel_tag(ax, "ab"[k])

    # (c) ANOVA split
    ax = fig.add_subplot(gs[2])
    rows = [("cifar", b) for b in (1, 2, 4, 8)] + [("celebahq", b) for b in (1, 4)]
    labs, sl, st, si = [], [], [], []
    for m, b in rows:
        r = an[(an.model == m) & (an.bits == b)].iloc[0]
        labs.append("W%d" % b)
        sl.append(r.share_layer * 100)
        st.append(r.share_timestep * 100)
        si.append(r.share_interaction * 100)
    y = np.arange(len(rows))[::-1]
    ax.barh(y, sl, color="#dcdbd4", height=0.62, label="layer")
    ax.barh(y, st, left=sl, color="#9a9992", height=0.62, label="timestep")
    ax.barh(y, si, left=np.array(sl) + np.array(st),
            color=V.ACCENT, height=0.62, label="layer x timestep")
    for k in range(len(rows)):
        ax.text(sl[k] + st[k] + si[k] + 1.6, y[k], "%.0f" % si[k],
                va="center", fontsize=5.6, color=V.ACCENT)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=5.8)
    ax.set_xlim(0, 110)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(labelsize=5.7)
    ax.set_xlabel("share of log-damage variance (%)", fontsize=6.2)
    ax.legend(fontsize=5.4, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.48, 1.30), handlelength=0.8,
              columnspacing=0.7, handletextpad=0.35)
    ax.spines["left"].set_visible(False)
    ax.text(-0.155, 0.80, "CIFAR-10", transform=ax.transAxes, rotation=90,
            fontsize=5.7, color=V.INK2, va="center", ha="right")
    ax.text(-0.155, 0.17, "CelebA", transform=ax.transAxes, rotation=90,
            fontsize=5.7, color=V.INK2, va="center", ha="right")
    panel_tag(ax, "c", dx=-0.22, dy=1.30)
    save(fig, "f3_regimes.png")


# --------------------------------------------------------------------------
# Figure 3 -- measured grid, separable fit, interaction residual
# --------------------------------------------------------------------------
def fig_grid(cif):
    M = grid(cif, 1)
    F = main_effects(M)
    R = M / F

    fig = plt.figure(figsize=(COL, 3.9))
    gs = fig.add_gridspec(3, 1, hspace=0.50)
    lo, hi = np.log10(M.min()), np.log10(M.max())

    for k, (A, ttl, div) in enumerate([
            (M, "measured  D[g,k]", False),
            (F, "separable fit  exp(mu + a_g + c_k)", False),
            (R, "residual  measured / fit", True)]):
        ax = fig.add_subplot(gs[k])
        if div:
            v = np.max(np.abs(np.log(R)))
            im = ax.imshow(np.log(R), cmap=V.DIVERGING, aspect="auto",
                           vmin=-v, vmax=v)
        else:
            im = ax.imshow(np.log10(A), cmap="cividis", aspect="auto",
                           vmin=lo, vmax=hi)
        for i in range(4):
            for j in range(4):
                val = R[i, j] if div else A[i, j]
                txt = ("%.1fx" % val) if div else (
                    "%.3f" % val if val >= 0.001 else "%.0e" % val)
                if div:
                    fc = V.INK if abs(np.log(val)) < 0.55 * v else "white"
                else:
                    fc = "white" if np.log10(val) < (lo + hi) / 2 else V.INK
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=5.2, color=fc)
        ax.set_xticks(range(4))
        ax.set_xticklabels(BLAB if k == 2 else [""] * 4, fontsize=5.6)
        ax.set_yticks(range(4))
        ax.set_yticklabels([V.GLAB[g] for g in GROUPS], fontsize=5.6)
        ax.tick_params(length=0)
        ax.set_title(ttl, fontsize=6.3, loc="left", pad=3, color=V.INK)
        if k == 2:
            ax.set_xlabel("timestep band  t/T", fontsize=6.2)
        panel_tag(ax, "abc"[k], dx=-0.21, dy=1.02)
    save(fig, "f2_grid.png")


# --------------------------------------------------------------------------
# Figure 4 -- composition
# --------------------------------------------------------------------------
def fig_comp(c):
    """Three panels, because the text makes three claims.

    An earlier column-format redraw kept only two and drew only the power fit.
    The prose reports BOTH fits and BOTH Spearman correlations, so a figure
    showing one of each leaves assertions in the text unsupported by the plot.
    """
    from scipy.stats import spearmanr
    x = c.predicted_sum_rms.values
    y = c.measured_rms.values
    n = c.n_cells.values
    r = y / x
    beta, la = np.polyfit(np.log(x), np.log(y), 1)
    r2p = np.corrcoef(np.log(x), np.log(y))[0, 1] ** 2
    slope, icept = np.polyfit(x, y, 1)
    r2l = 1 - ((y - (slope * x + icept)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rho_m, p_m = spearmanr(r, x)
    rho_n, p_n = spearmanr(r, n)

    fig = plt.figure(figsize=(COL, 4.85))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.30, 1.0, 1.0], hspace=0.72)

    # (a) measured against additive prediction, with BOTH fits
    ax = fig.add_subplot(gs[0])
    g = np.logspace(np.log10(x.min()) - 0.12, np.log10(x.max()) + 0.08, 80)
    ax.plot(g, g, lw=0.8, ls=(0, (3, 2)), color=V.MUTED, zorder=2)
    ax.plot(g, np.exp(la) * g ** beta, lw=1.2,
            color=V.SERIES["down_early"], zorder=4)
    ax.plot(g, slope * g + icept, lw=1.2, color=V.ACCENT, zorder=4)
    ax.scatter(x, y, s=5 + 3.0 * n, facecolor=V.SERIES["down_early"],
               alpha=0.5, edgecolor="none", zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(y.min() * 0.55, y.max() * 2.4)
    ax.tick_params(labelsize=5.7)
    ax.set_xlabel("additive prediction, sum of single-cell damage", fontsize=6.2)
    ax.set_ylabel("measured damage (RMS)", fontsize=6.2)
    ax.grid(color=V.GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.text(0.03, 0.97, "power fit  measured $\\propto$ predicted$^{%.2f}$\n"
                        "$R^2$ = %.2f" % (beta, r2p),
            transform=ax.transAxes, fontsize=5.7,
            color=V.SERIES["down_early"], va="top", linespacing=1.3)
    ax.text(0.03, 0.76, "linear fit  slope %.2f, $R^2$ = %.2f\n"
                        "intercept %.3f" % (slope, r2l, icept),
            transform=ax.transAxes, fontsize=5.7, color=V.ACCENT, va="top",
            linespacing=1.3)
    ax.text(0.17, 0.40, "additive", transform=ax.transAxes, fontsize=5.7,
            color=V.MUTED, ha="left", va="bottom", rotation=29)
    ax.text(0.97, 0.05, "area = cells quantized", transform=ax.transAxes,
            fontsize=5.5, color=V.INK2, ha="right")
    panel_tag(ax, "a")

    # (b) the deficit against the size of the prediction
    ax = fig.add_subplot(gs[1])
    ax.axhline(1.0, lw=0.8, ls=(0, (3, 2)), color=V.MUTED, zorder=2)
    ax.scatter(x, r, s=5 + 3.0 * n, facecolor=V.SERIES["down_early"],
               alpha=0.5, edgecolor="none", zorder=4)
    lx = np.log10(x)
    m2, c2 = np.polyfit(lx, r, 1)
    gx = np.linspace(lx.min(), lx.max(), 40)
    ax.plot(10 ** gx, m2 * gx + c2, lw=1.1, color=V.ACCENT, zorder=5)
    ax.set_xscale("log")
    ax.set_ylim(0.30, 1.12)
    ax.tick_params(labelsize=5.7)
    ax.set_xlabel("additive prediction, sum of single-cell damage", fontsize=6.2)
    ax.set_ylabel("measured / predicted", fontsize=6.2)
    ax.grid(color=V.GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.text(0.03, 0.94, "additive", transform=ax.transAxes, fontsize=5.7,
            color=V.MUTED, va="top")
    ax.text(0.97, 0.08, r"Spearman $\rho$ = %.2f,  $p$ = %.3f" % (rho_m, p_m),
            transform=ax.transAxes, fontsize=5.7, color=V.INK2, ha="right")
    panel_tag(ax, "b", dy=1.13)

    # (c) the deficit against how many cells were quantized together
    ax = fig.add_subplot(gs[2])
    ax.axhline(1.0, lw=0.8, ls=(0, (3, 2)), color=V.MUTED, zorder=2)
    ax.scatter(n, r, s=9, facecolor=V.SERIES["down_early"], alpha=0.55,
               edgecolor="none", zorder=4)
    med = pd.Series(r).groupby(pd.Series(n)).median()
    ax.plot(med.index, med.values, lw=1.1, color=V.ACCENT, zorder=5)
    ax.set_ylim(0.30, 1.12)
    ax.tick_params(labelsize=5.7)
    ax.set_xlabel("cells quantized together", fontsize=6.2)
    ax.set_ylabel("measured / predicted", fontsize=6.2)
    ax.grid(color=V.GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.text(0.03, 0.94, "additive", transform=ax.transAxes, fontsize=5.7,
            color=V.MUTED, va="top")
    ax.text(0.97, 0.08, r"Spearman $\rho$ = %.2f,  $p$ = %.3f" % (rho_n, p_n),
            transform=ax.transAxes, fontsize=5.7, color=V.INK2, ha="right")
    panel_tag(ax, "c", dy=1.13)
    save(fig, "f4_composition.png")


# --------------------------------------------------------------------------
# Figure 5 -- the proxy is not the objective
# --------------------------------------------------------------------------
def fig_proxy(al, fd, w8):
    fig = plt.figure(figsize=(COL, 4.65))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.06], hspace=0.85)

    # (a) the proxy
    ax = fig.add_subplot(gs[0])
    for arm in ("measured", "separable"):
        d = al[al.allocator == arm].groupby("budget_avg_bits").measured_rms.mean()
        ax.plot(d.index, d.values, "-o", ms=2.8, lw=1.15, color=V.ARM[arm],
                clip_on=False, zorder=3)
        ax.annotate(arm, xy=(6.18, d.values[-1]), fontsize=5.7,
                    color=V.ARM[arm], va="center")
    ax.set_yscale("log")
    ax.set_xlim(1.8, 8.3)
    ax.set_xticks([2, 3, 4, 6])
    ax.tick_params(labelsize=5.7)
    ax.set_ylabel("paired RMS", fontsize=6.2)
    ax.set_xlabel("average bit budget B", fontsize=6.2)
    ax.set_title("the proxy the allocators minimise", fontsize=6.4,
                 loc="left", pad=3, color=V.INK)
    ax.grid(color=V.GRID, lw=0.5)
    ax.set_axisbelow(True)
    panel_tag(ax, "a")

    # (b) the objective
    ax = fig.add_subplot(gs[1])
    ax.axhspan(250, 470, color="#f4f3ee", zorder=0)
    ax.text(2.05, 435, "no recognisable images", fontsize=5.4,
            color=V.MUTED, ha="left", va="center", zorder=1)
    for arm in ("measured", "separable"):
        d = fd[fd.arm == arm].groupby("budget_avg_bits").fid.agg(["mean", "std"])
        ax.errorbar(d.index, d["mean"], yerr=d["std"], fmt="-o", ms=2.8,
                    lw=1.15, elinewidth=0.7, capsize=1.4, color=V.ARM[arm],
                    zorder=3)
    ax.annotate("measured", xy=(6.18, 140), fontsize=5.7,
                color=V.ARM["measured"], va="center")
    ax.annotate("separable", xy=(6.18, 72), fontsize=5.7,
                color=V.ARM["separable"], va="center")
    fp = fd[fd.arm == "fp32"].fid.mean()
    ax.axhline(fp, lw=0.8, ls=(0, (3, 2)), color=V.INK2, zorder=2)
    ax.text(2.02, fp + 26, "FP32  %.1f" % fp, fontsize=5.4, color=V.INK2)
    ax.plot([4.85, 4.85], [w8, 385.1], lw=0.9, color=V.MUTED, zorder=4)
    # W8 (30.7) and the FP32 line (28.4) nearly coincide, so the marker
    # legitimately sits on the line; the label is lifted clear of it
    for yv, dy, tag in ((385.1, 0, "W4"), (w8, 34, "W8")):
        ax.scatter([4.85], [yv], s=11, marker="s", color=V.MUTED, zorder=5)
        ax.text(4.73, yv + dy, tag, fontsize=5.4, color=V.MUTED, ha="right",
                va="center")
    # the word "uniform" has nowhere to sit that is not on top of a curve;
    # the W4/W8 tags plus the caption identify the bracket unambiguously
    ax.set_xlim(1.8, 8.3)
    ax.set_ylim(-10, 470)
    ax.set_xticks([2, 3, 4, 6])
    ax.tick_params(labelsize=5.7)
    ax.set_ylabel("FID", fontsize=6.2)
    ax.set_xlabel("average bit budget B", fontsize=6.2)
    ax.set_title("the objective it was meant to stand in for", fontsize=6.4,
                 loc="left", pad=3, color=V.INK)
    ax.grid(color=V.GRID, lw=0.5)
    ax.set_axisbelow(True)
    panel_tag(ax, "b")

    # (c) agreement plane, both axes as percentage improvement of the measured
    # allocator over the separable one.  The x values are exactly the 28.9 /
    # 47.9 / 12.9 % quoted in the text, so the panel and the prose carry the
    # same numbers instead of the reader having to convert a log ratio.
    ax = fig.add_subplot(gs[2])
    pts = []
    for bud in (2.0, 4.0, 6.0):
        pm = al[(al.allocator == "measured") &
                (al.budget_avg_bits == bud)].measured_rms.mean()
        ps = al[(al.allocator == "separable") &
                (al.budget_avg_bits == bud)].measured_rms.mean()
        fm = fd[(fd.arm == "measured") & (fd.budget_avg_bits == bud)].fid.mean()
        fs = fd[(fd.arm == "separable") & (fd.budget_avg_bits == bud)].fid.mean()
        pts.append((100.0 * (ps - pm) / ps, 100.0 * (fs - fm) / fs,
                    min(fm, fs) < 250, int(bud)))

    ax.axhspan(-150, 0, color="#fdeceb", zorder=0)
    ax.axhline(0, lw=0.8, color=V.INK2, zorder=2)
    for dx, dy, live, bud in pts:
        ax.scatter([dx], [dy], s=30,
                   facecolor=V.ACCENT if live else "white",
                   edgecolor=V.ACCENT, lw=1.0, zorder=4)
        ax.annotate("B=%d" % bud, xy=(dx, dy), xytext=(dx + 1.2, dy - 14),
                    fontsize=5.9, color=V.INK2, style="italic")
    ax.scatter([], [], s=30, facecolor=V.ACCENT, edgecolor=V.ACCENT, lw=1.0,
               label="FID comparison meaningful")
    ax.scatter([], [], s=30, facecolor="white", edgecolor=V.ACCENT, lw=1.0,
               label="both arms above FID 250")
    ax.legend(fontsize=5.4, frameon=False, loc="upper left",
              bbox_to_anchor=(-0.02, 1.05), handletextpad=0.3,
              borderpad=0.1, labelspacing=0.35)
    ax.set_xlim(0, 57)
    ax.set_ylim(-150, 62)
    ax.tick_params(labelsize=5.7)
    ax.set_xlabel("proxy improvement of measured\nover separable (%)",
                  fontsize=6.2, linespacing=1.25)
    ax.set_ylabel("FID improvement (%)", fontsize=6.2)
    ax.text(0.97, 0.045, "proxy better, FID worse", transform=ax.transAxes,
            fontsize=5.9, color=V.ACCENT, ha="right", va="bottom")
    panel_tag(ax, "c", dx=-0.24, dy=1.10)
    save(fig, "f5_proxy.png")


if __name__ == "__main__":
    cif = pd.read_csv(os.path.join(HERE, "grid_cifar.csv"))
    cel = pd.read_csv(os.path.join(HERE, "grid_celebahq.csv"))
    an = pd.read_csv(os.path.join(HERE, "anova_split.csv"))
    comp = pd.read_csv(os.path.join(HERE, "composition_trials.csv"))
    al = pd.read_csv(os.path.join(HERE, "allocation_paired_rms.csv"))
    fd = pd.read_csv(os.path.join(HERE, "fid_results.csv"))
    w8 = np.mean([json.loads(l)["fid"] for l in
                  open(os.path.join(HERE, "fid_uniform_w8.jsonl"))])
    print("uniform W8 FID %.2f" % w8)
    fig_probe()
    fig_grid(cif)
