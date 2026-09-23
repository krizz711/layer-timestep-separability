"""Render the display equations for the UPCON build.

Same renderer as ../make_eqs.py, with the notation disentangled.  The earlier
set overloaded b as BOTH the timestep-band index and the bit-width, so
"quantize group g to b bits inside band b" used one symbol for two quantities,
and (9) indexed cells by b while also assigning b as their bit-width.  B was
likewise both the set of bands and the average-bit budget.  Here:

    g  layer group        k  timestep band (set K)
    q  bit-width          B  average-bit budget

Output goes to upcon/eqfig/ so the original paper's equations are untouched.
"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "eqfig")
os.makedirs(FIG, exist_ok=True)

PT = 10.0           # match the 10 pt body of the IEEE template
DPI = 600
INK = "#1a1a19"

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"

EQS = {
    # --- the quantizer -----------------------------------------------------
    1: r"$Q_q(w)\ =\ s_c\,\mathrm{clip}\!\left(\mathrm{round}"
       r"\!\left(w/s_c\right),\ -2^{\,q-1},\ 2^{\,q-1}-1\right)$",
    2: r"$s_c\ =\ \mathrm{arg\,min}_{\,s>0}\ \left\|\,W_c-"
       r"Q_q(W_c;\,s)\,\right\|_2^{\,2}$",
    3: r"$Q_1(W_c)\ =\ \alpha_c\,\mathrm{sign}(W_c),"
       r"\qquad \alpha_c=\frac{1}{n_c}\sum_{w\,\in\,W_c}|w|$",
    # --- the probe ---------------------------------------------------------
    4: r"$q(g,t)\ =\ q_0\ \ \mathrm{if}\ \ g=g_0\ \ \mathrm{and}"
       r"\ \ \tau_0\leq t/T<\tau_1,\ \ \mathrm{else}\ \ 32$",
    5: r"$D(g,k)\ =\ \sqrt{\dfrac{1}{NP}\sum_{i=1}^{N}"
       r"\left\|\,x_i-\hat{x}_i\,\right\|_2^{2}}$",
    # --- the decomposition -------------------------------------------------
    6: r"$\log D[g,k]\ =\ \mu+\alpha_g+\gamma_k+\varepsilon_{gk}$",
    7: r"$\mathrm{SS}_{\mathrm{tot}}=\mathrm{SS}_{\mathrm{layer}}+"
       r"\mathrm{SS}_{\mathrm{time}}+\mathrm{SS}_{\mathrm{int}},"
       r"\quad \eta^2_{\mathrm{int}}=\dfrac{\mathrm{SS}_{\mathrm{int}}}"
       r"{\mathrm{SS}_{\mathrm{tot}}}$",
    # --- composition and allocation ---------------------------------------
    8: r"$\hat{D}_{\mathrm{meas}}\ \approx\ c\left(\sum_{j}D_j\right)"
       r"^{\beta},\qquad \beta=0.93$",
    9: r"$\min_{\{q_{gk}\}}\ \sum_{g,k}D[g,q_{gk}]\quad\mathrm{s.t.}"
       r"\quad \frac{1}{|G||K|}\sum_{g,k}q_{gk}\ \leq\ B$",
}


def render(n, src):
    fig = plt.figure(figsize=(0.01, 0.01))
    t = fig.text(0, 0, src, fontsize=PT, color=INK)
    path = os.path.join(FIG, "eq%d.png" % n)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.012,
                transparent=False, facecolor="white")
    t.get_window_extent(fig.canvas.get_renderer())
    plt.close(fig)
    import struct
    d = open(path, "rb").read(33)
    w, h = struct.unpack(">II", d[16:24])
    return {"w": w * 72.0 / DPI, "h": h * 72.0 / DPI, "px": [w, h]}


if __name__ == "__main__":
    out = {}
    for n in sorted(EQS):
        out[str(n)] = render(n, EQS[n])
        print("  eq%d  %6.1f x %5.1f pt" % (n, out[str(n)]["w"], out[str(n)]["h"]))
    json.dump(out, open(os.path.join(FIG, "eqs.json"), "w"), indent=1)
    # the column measure is 3.5 in less the equation-number gutter
    wide = [n for n in out if out[n]["w"] > 224]
    print("over column measure (224 pt):", wide or "none")
