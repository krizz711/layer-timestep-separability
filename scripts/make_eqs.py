"""Render the paper's display equations to PNG.

No LaTeX toolchain exists on this host, so the equations are set with
matplotlib's mathtext and embedded as images.  Each is rendered at a known
point size and dpi, so the builder can place it at TRUE SIZE next to 9.3 pt
body text rather than scaling it to fit and leaving the maths a different
size from the prose around it.

Writes fig/eqN.png and fig/eqs.json (natural width and height in points).
"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "fig")
os.makedirs(FIG, exist_ok=True)

PT = 9.6            # match the body measure
DPI = 600
INK = "#1a1a19"

# the body is Times; STIX is the Times-metric-compatible maths font, so the
# equations do not read as a different typeface from the prose around them
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"

EQS = {
    # --- the quantizer -----------------------------------------------------
    1: r"$Q_b(w)\ =\ s_c\,\mathrm{clip}\!\left(\mathrm{round}"
       r"\!\left(w/s_c\right),\ -2^{\,b-1},\ 2^{\,b-1}-1\right)$",
    2: r"$s_c\ =\ \mathrm{arg\,min}_{\,s>0}\ \left\|\,W_c-"
       r"Q_b(W_c;\,s)\,\right\|_2^{\,2}$",
    3: r"$Q_1(W_c)\ =\ \alpha_c\,\mathrm{sign}(W_c),"
       r"\qquad \alpha_c=\frac{1}{n_c}\sum_{w\,\in\,W_c}|w|$",
    # --- the probe ---------------------------------------------------------
    4: r"$b(g,t)\ =\ b\ \ \mathrm{if}\ \ g=g_0\ \ \mathrm{and}"
       r"\ \ \tau_0\leq t/T<\tau_1,\ \ \mathrm{else}\ \ 32$",
    5: r"$D(g,b)\ =\ \sqrt{\dfrac{1}{NP}\sum_{i=1}^{N}"
       r"\left\|\,x_i-\hat{x}_i\,\right\|_2^{2}}$",
    # --- the decomposition -------------------------------------------------
    6: r"$\log D[g,b]\ =\ \mu+\alpha_g+\gamma_b+\varepsilon_{gb}$",
    7: r"$\mathrm{SS}_{\mathrm{tot}}=\mathrm{SS}_{\mathrm{layer}}+"
       r"\mathrm{SS}_{\mathrm{time}}+\mathrm{SS}_{\mathrm{int}},"
       r"\quad \eta^2_{\mathrm{int}}=\dfrac{\mathrm{SS}_{\mathrm{int}}}"
       r"{\mathrm{SS}_{\mathrm{tot}}}$",
    # --- composition and allocation ---------------------------------------
    8: r"$\hat{D}_{\mathrm{meas}}\ \approx\ c\left(\sum_{k}D_k\right)"
       r"^{\beta},\qquad \beta=0.93$",
    9: r"$\min_{\{b_{gb}\}}\ \sum_{g,b}D[g,b_{gb}]\quad\mathrm{s.t.}"
       r"\quad \frac{1}{|G||B|}\sum_{g,b}b_{gb}\ \leq\ B$",
}


def render(n, src):
    fig = plt.figure(figsize=(0.01, 0.01))
    t = fig.text(0, 0, src, fontsize=PT, color=INK)
    path = os.path.join(FIG, "eq%d.png" % n)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.012,
                transparent=False, facecolor="white")
    bb = t.get_window_extent(fig.canvas.get_renderer())
    plt.close(fig)
    import struct
    d = open(path, "rb").read(33)
    w, h = struct.unpack(">II", d[16:24])
    # px -> points at the dpi it was rendered with: this is TRUE SIZE
    return {"w": w * 72.0 / DPI, "h": h * 72.0 / DPI, "px": [w, h]}


if __name__ == "__main__":
    out = {}
    for n in sorted(EQS):
        out[str(n)] = render(n, EQS[n])
        print("  eq%d  %6.1f x %5.1f pt" % (n, out[str(n)]["w"], out[str(n)]["h"]))
    json.dump(out, open(os.path.join(FIG, "eqs.json"), "w"), indent=1)
    wide = [n for n in out if out[n]["w"] > 232]
    print("over column measure (232 pt):", wide or "none")
