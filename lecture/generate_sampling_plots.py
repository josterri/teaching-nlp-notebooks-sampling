"""Charts of sampling-lecture.tex: schematic charts for parts 1 to 5, charts of
GPT-2 small's numbers for the Worked Example part. Every number the deck prints
in that part is printed here first, and every one-line computation on a slide
is recomputed from the operands as the slide prints them.

    ~/venvs/py313/bin/python -B lecture/generate_sampling_plots.py   # images/*.pdf, *.png
"""
import pathlib
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, "/Users/joerg/.claude/skills/house-lecture/scripts")
from house_figures import C, save, setup  # noqa: E402

OUT = HERE / "images"
OUT.mkdir(exist_ok=True)


def out(stem):
    return str(OUT / stem)


def schematic(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def show(ok, line):
    """Print a number of the deck; stop when a slide's computation does not match."""
    assert ok, line
    print(line)


# ---------------------------------------------------------------- schematic charts

def zipf(n, s=1.1):
    p = 1 / np.arange(1, n + 1) ** s
    return p / p.sum()


def softmax_np(z, T=1.0):
    e = np.exp((z - z.max()) / T)
    return e / e.sum()


Z8 = np.array([2.2, 1.4, 0.9, 0.4, 0.0, -0.5, -0.9, -1.4])   # schematic scores, two of them negative

# scores, probabilities
fig, (a1, a2) = setup(width=0.48, aspect=0.62, ncols=2)
x = np.arange(len(Z8))
a1.bar(x, Z8, color=C["blue"], width=0.7)
a1.axhline(0, color=C["gray"], lw=0.8)
a1.set_title("scores $z_i$")
a2.bar(x, softmax_np(Z8), color=C["blue"], width=0.7)
a2.set_title("probabilities $p_i$")
for a in (a1, a2):
    schematic(a)
    a.set_xlabel("tokens")
fig.suptitle("Schematic: softmax of the scores", color=C["purple"], fontweight="bold",
             fontsize=a1.title.get_fontsize())
fig.tight_layout()
save(fig, out("softmax_scores"))

# greedy token
P10 = softmax_np(np.array([2.0, 1.5, 1.2, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.0]))
fig, ax = setup(width=0.48, aspect=0.7)
x = np.arange(10)
ax.bar(x, P10, color=[C["orange"]] + [C["blue"]] * 9, width=0.7)
ax.annotate("greedy token", (0.35, P10[0] * 0.9), (2.2, P10[0] * 0.8), color=C["orange"],
            arrowprops=dict(arrowstyle="->", color=C["orange"]))
schematic(ax)
ax.set_xlabel("tokens, likeliest first")
ax.set_ylabel("probability $p_i$")
ax.set_title("Schematic: greedy token in orange")
save(fig, out("greedy_bars"))

# probability of a repeated phrase: question, answer
copies = np.arange(1, 7)
rise = 1 / (1 + np.exp(-(copies - 3.6) * 1.5))
for stem, full in (("repeat_q", False), ("repeat_a", True)):
    fig, ax = setup(width=0.48, aspect=0.7)
    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(0, 1.05)
    if full:
        ax.plot(copies, rise, "-o", color=C["orange"])
    else:
        ax.plot(copies[:1], rise[:1], "o", color=C["blue"])
        ax.text(4.2, 0.45, "?", fontsize=24, color=C["gray"], ha="center")
    schematic(ax)
    ax.set_xlabel("copies of the phrase in the text")
    ax.set_ylabel("probability of the next copy")
    ax.set_title("Schematic: one phrase, repeated")
    save(fig, out(stem))

# p^(T) at three temperatures
Z12 = np.array([2.0, 1.4, 1.0, 0.7, 0.5, 0.3, 0.15, 0.0, -0.15, -0.3, -0.4, -0.5])
fig, ax = setup(width=0.48, aspect=0.7)
x = np.arange(len(Z12))
for T, col, lab in ((0.5, C["blue"], "$T < 1$"), (1.0, C["green"], "$T = 1$"), (3.0, C["red"], "$T > 1$")):
    p = softmax_np(Z12, T)
    ax.plot(x, p, "-o", color=col, ms=4)
    ax.text(x[0] - 0.3, p[0], lab, color="#333333", va="center", ha="right")
ax.set_xlim(-2.4, len(Z12) - 0.5)
schematic(ax)
ax.set_xlabel("tokens, likeliest first")
ax.set_ylabel("probability $p_i^{(T)}$")
ax.set_title("Schematic: three temperatures")
save(fig, out("temperature_three"))

# limits: question (T = 1), answer (T -> 0, T -> infinity)
fig, (a1, a2) = setup(width=0.48, aspect=0.7, ncols=2)
for a, lab in ((a1, "$T \\to 0$: ?"), (a2, "$T \\to \\infty$: ?")):
    a.bar(x, softmax_np(Z12), color=C["blue"], width=0.7)
    a.set_ylim(0, 1.0)
    a.text(len(x) / 2 - 0.5, 0.6, "?", fontsize=20, color=C["gray"], ha="center")
    a.set_title(lab)
    schematic(a)
    a.set_xlabel("tokens")
fig.suptitle("Schematic: $p^{(T)}$ at $T = 1$", color=C["purple"], fontweight="bold",
             fontsize=a1.title.get_fontsize())
fig.tight_layout()
save(fig, out("limits_q"))

fig, (a1, a2) = setup(width=0.48, aspect=0.7, ncols=2)
a1.bar(x, np.r_[1.0, np.zeros(len(x) - 1)], color=[C["orange"]] + [C["blue"]] * (len(x) - 1), width=0.7)
a1.set_title("$T \\to 0$")
a2.bar(x, np.full(len(x), 1 / len(x)), color=C["blue"], width=0.7)
a2.set_ylim(0, 1.0)
a2.text(len(x) / 2 - 0.5, 1 / len(x) + 0.05, "$1/|V|$ each", ha="center", color="#333333")
a2.set_title("$T \\to \\infty$")
for a in (a1, a2):
    schematic(a)
    a.set_xlabel("tokens")
fig.suptitle("Schematic: two limits of $p^{(T)}$", color=C["purple"], fontweight="bold",
             fontsize=a1.title.get_fontsize())
fig.tight_layout()
save(fig, out("limits_a"))

# entropy against T
zz = np.log(zipf(400, 1.3)) * 2.5
Ts = np.linspace(0.05, 6, 300)
H = [-(lambda p: (p * np.log(p)).sum())(softmax_np(zz, T)) for T in Ts]
fig, ax = setup(width=0.48, aspect=0.7)
ax.plot(Ts, H, color=C["blue"])
ax.axhline(np.log(len(zz)), color=C["gray"], ls="--", lw=1)
ax.text(6, np.log(len(zz)) * 1.02, "$\\log |V|$, uniform", ha="right", va="bottom", color="#333333")
ax.plot([0.05], [H[0]], "o", color=C["orange"])
ax.annotate("$H = 0$: greedy decoding", (0.05, H[0]), (3.3, 0.1 * np.log(len(zz))), color="#333333",
            arrowprops=dict(arrowstyle="->", color=C["gray"]))
ax.set_ylim(0, np.log(len(zz)) * 1.15)
schematic(ax)
ax.set_xlabel("temperature $T$")
ax.set_ylabel("entropy $H$")
ax.set_title("Schematic: entropy against $T$")
save(fig, out("entropy_T"))

# top-k cut
P14 = zipf(14, 0.9)
k = 4
fig, ax = setup(width=0.48, aspect=0.7)
x = np.arange(14)
ax.bar(x, P14, color=[C["blue"]] * k + [C["gray"]] * (14 - k), width=0.7)
ax.axvline(k - 0.5, color=C["gray"], ls="--", lw=1)
ax.text(k / 2 - 0.5, P14[0] * 1.05, "$k$ kept", ha="center", color="#333333")
ax.text(k + 4.5, P14[3] * 0.9, "cut", ha="center", color="#333333")
ax.set_ylim(0, P14[0] * 1.25)
schematic(ax)
ax.set_xlabel("tokens, likeliest first")
ax.set_ylabel("probability $p_i$")
ax.set_title("Schematic: top-$k$, kept tokens in blue")
save(fig, out("topk_cut"))

# top-p: cumulative probability, threshold pi
P30 = zipf(30, 1.0)
cum = np.cumsum(P30)
pi = 0.7
ks = int(np.searchsorted(cum, pi)) + 1
fig, ax = setup(width=0.48, aspect=0.7)
x = np.arange(1, 31)
ax.plot(x[:ks], cum[:ks], "o", color=C["blue"], ms=4)
ax.plot(x[ks:], cum[ks:], "o", color=C["gray"], ms=4)
ax.axhline(pi, color=C["gray"], ls="--", lw=1)
ax.axvline(ks + 0.5, color=C["gray"], ls=":", lw=1)
ax.text(30, pi - 0.03, "$\\pi$", ha="right", va="top", color="#333333")
ax.text(ks + 1, 0.12, "$k^*(\\pi)$ kept", color="#333333")
ax.set_ylim(0, 1.05)
schematic(ax)
ax.set_xlabel("tokens, likeliest first")
ax.set_ylabel("cumulative probability")
ax.set_title("Schematic: top-$p$, kept tokens in blue")
save(fig, out("topp_cum"))

# a peaked, a flat distribution: question; with top-k line, top-p kept set: answer
peaked = softmax_np(np.array([4.0, 1.6, 0.6, 0.3, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4]))
flat = softmax_np(np.linspace(0.45, 0.0, 10))
kk, pp = 3, 0.8
for stem, marks in (("twodist_q", False), ("twodist_a", True)):
    fig, axs = setup(width=0.48, aspect=0.7, ncols=2)
    for a, p, name in zip(axs, (peaked, flat), ("peaked", "flat")):
        x = np.arange(len(p))
        n = int(np.searchsorted(np.cumsum(p), pp)) + 1
        cols = [C["blue"]] * n + [C["gray"]] * (len(p) - n) if marks else [C["blue"]] * len(p)
        a.bar(x, p, color=cols, width=0.7)
        if marks:
            a.axvline(kk - 0.5, color=C["orange"], ls="--", lw=1.5)
        a.set_ylim(0, 0.85)
        schematic(a)
        a.set_title(name)
        a.set_xlabel("tokens")
    title = "Schematic: dashed $k$; blue top-$p$ set" if marks else "Schematic: two distributions"
    fig.suptitle(title, color=C["purple"], fontweight="bold", fontsize=axs[0].title.get_fontsize())
    fig.tight_layout()
    save(fig, out(stem))

# ---------------------------------------------------------------- GPT-2 small, Worked Example

with plt.rc_context():                        # the notebooks' chart style stays out of the deck's charts
    exec((ROOT / "src" / "setup.py").read_text())
src3 = (ROOT / "src" / "03_beam_search.py").read_text()
for name in ("two_step_tree", "log_softmax_steps", "beam_search"):
    exec(re.search(rf"^(?:@functools.*\n)?def {name}.*?return .*?\n", src3, re.S | re.M).group(0))

print("\n== Next tokens after the Austen prompt")
z = logits(AUSTEN)
p = softmax(z)
order = np.argsort(-p)
for r, i in enumerate(order[:10], 1):
    print(f"{r:2d} {word(i):6s} z = {z[i]:.3f}  p = {p[i]:.4f}")
top5 = p[order[:5]].sum()
kstar = int(np.searchsorted(np.cumsum(p[order]), 0.9)) + 1
show(kstar == (top_p_filter(p, 0.9) > 0).sum(), f"top-p 0.9 after Austen: {kstar:,} tokens")
show(True, f"five likeliest tokens: mass {top5:.3f}; rest {1 - top5:.3f}")

fig, ax = setup(width=0.48, aspect=0.75)
names = [word(i) for i in order[:10]][::-1]
vals = p[order[:10]][::-1]
ax.barh(range(10), vals, color=[C["blue"]] * 9 + [C["orange"]], height=0.7)
ax.set_yticks(range(10), names)
ax.set_xlabel("probability $p_i$")
ax.grid(axis="y", visible=False)
ax.text(vals[-1] + 0.004, 9, f"{vals[-1]:.3f}", va="center", color="#333333")
ax.set_title("Ten likeliest tokens, $T = 1$")
save(fig, out("austen_top10"))

print("\n== Temperature after the Austen prompt")
zt, zr, za = (round(float(z[i]), 3) for i in order[:3])
gap = round(zt - zr, 3)
show(abs(gap - 1.573) < 1e-9, f"z(the) = {zt}, z(there) = {zr}, z(a) = {za}; gap {gap:.3f}")
for T, shown in ((0.5, "23.2"), (1.0, "4.82"), (2.0, "2.20")):
    val = np.exp(gap / T)
    show(f"{val:.3g}" == shown or f"{val:.2f}" == shown or f"{val:.1f}" == shown,
         f"T = {T}: e^(1.573/T) = {shown}")
for T in (0.5, 1.0, 2.0):
    pT = softmax(z, T)
    HT = -(pT[pT > 0] * np.log(pT[pT > 0])).sum()
    print(f"T = {T}: p(the) = {pT[order[0]]:.3f}, H = {HT:.2f} nats")
print(f"log|V| = {np.log(V):.2f}")

print("\n== Top-k, top-p after three prompts")
cums = {}
for name, prompt in (("peaked", PEAKED), ("Austen", AUSTEN), ("flat", FLAT)):
    q = softmax(logits(prompt))
    o = np.argsort(-q)
    cums[name] = np.cumsum(q[o])
    ks = int(np.searchsorted(cums[name], 0.9)) + 1
    print(f"{name:7s} p_max = {q[o[0]]:.3f} ({word(o[0])}), M_5 = {cums[name][4]:.3f}, "
          f"k*(0.9) = {ks:,}, KL top-5 = {-np.log(cums[name][4]):.3f}")
show(f"{0.033 / 0.112:.3f}" == "0.295", "flat, top-5: 0.033 / 0.112 = 0.295")
show(f"{-np.log(0.112):.2f}" == "2.19", "flat, top-5: -log 0.112 = 2.19 nats")

fig, ax = setup(width=0.48, aspect=0.75)
for (name, c), col, lab in zip(cums.items(), (C["blue"], C["green"], C["red"]),
                               ("Prime Minister", "Austen", "I went to the")):
    r = np.arange(1, len(c) + 1)
    ax.plot(r, c, color=col, lw=1.6)
    ks = int(np.searchsorted(c, 0.9)) + 1
    ax.plot([ks], [c[ks - 1]], "o", color=col, ms=5, label=f"{lab}: {ks:,}")
ax.axhline(0.9, color=C["gray"], ls="--", lw=1)
ax.axvline(5, color=C["orange"], ls=":", lw=1.5)
ax.text(5.6, 0.62, "$k = 5$", color="#333333")
ax.legend(loc="lower right", title="tokens to 0.9", frameon=False)
ax.set_xscale("log")
ax.set_xlim(1, V)
ax.set_ylim(0, 1.08)
ax.set_xlabel("tokens, likeliest first (log scale)")
ax.set_ylabel("cumulative probability")
ax.set_title("Tokens to reach 0.9")
save(fig, out("cumulative_three"))

print("\n== Two-step tree after the Austen prompt")
tree = two_step_tree(AUSTEN)
for a, pa, kids in tree:
    for b, pb in kids:
        shown = round(float(f"{pa:.4g}") * float(f"{pb:.4g}"), 4)   # the slide prints 4 significant digits
        show(abs(shown - round(pa * pb, 4)) < 1e-12,
             f"{word(a):6s} {pa:.4g} x {word(b):7s} {pb:.4g} = {pa * pb:.4f}")
greedy2, best2 = tree[0][1] * tree[0][2][0][1], tree[1][1] * tree[1][2][0][1]
show(f"{0.0159 / 0.0027:.1f}" == f"{best2 / greedy2:.1f}" == "5.9", "0.0159 / 0.0027 = 5.9")

print("\n== Greedy loop: probability of each copy of the sentence")
toks, probs = generate(AUSTEN, 54)[0]                     # greedy decoding: T = 0, no draw
copy = tok(" The United States is a nation of immigrants.").input_ids
first = tok(" the United States is a nation of immigrants.").input_ids
starts = [i for i in range(len(toks) - len(copy) + 1) if toks[i:i + len(copy)] in (copy, first)]
P_copy = np.array([np.prod(probs[i:i + len(copy)]) for i in starts])
for n, (i, P) in enumerate(zip(starts, P_copy), 1):
    print(f"copy {n}: tokens {i + 1}-{i + len(copy)}, P = {P:.4g}, p(first token) = {probs[i]:.3f}")
show(len(P_copy) == 6 and np.all(np.diff(P_copy) > 0), "six copies, P rises with every copy")
show(f"{0.6087 / 5.268e-7:.3g}" == "1.16e+06", "0.6087 / 5.268e-7 = 1.16 x 10^6")

fig, ax = setup(width=0.48, aspect=0.75)
ax.plot(np.arange(1, 7), P_copy, "-o", color=C["blue"])
ax.set_yscale("log")
ax.set_xticks(range(1, 7))
ax.set_xlabel("copy of the sentence")
ax.set_ylabel("probability of the copy")
ax.text(1.25, P_copy[0], "$5.268 \\times 10^{-7}$", color="#333333", va="center")
ax.text(6, P_copy[5] / 4, f"{P_copy[5]:.4f}", color="#333333", ha="center", va="top")
ax.set_title("Six copies under greedy decoding")
save(fig, out("loop_copies"))

print("\n== Twenty tokens after the Austen prompt")
for B in (1, 2, 5):
    t, lp = beam_search(AUSTEN, B, 20)
    print(f"beam B = {B}: log P = {lp:.2f}  {tok.decode(t)!r}")
for T in (0.7, 1.5):  # owner's presets: 0.7 below 1, 1.5 above 1; the draws show sampled text, draw 0
    (t, pr), = generate(AUSTEN, 20, T=T, draw=0)
    print(f"T = {T}, draw 0: log P = {np.log(pr).sum():.2f}  {tok.decode(t)!r}")

print("\n== Listing, exercise")
zf = logits(FLAT)
q1, q5, qp = next_token_dist(zf, T=1.0), next_token_dist(zf, T=1.0, k=5), next_token_dist(zf, T=1.0, top_p=0.9)
print(f"flat: argmax {word(q1.argmax())}, p.max {q1.max():.4f}; top-5 max {q5.max():.4f}, "
      f"{(q5 > 0).sum()} kept; top-p 0.9: {(qp > 0).sum()} kept")
for T in (0.5, 0.7, 1.0, 1.5):  # distributions at four temperatures, no draw
    r = next_token_dist(z, T=T, top_p=0.9)
    print(f"Austen, T = {T}, top-p 0.9: {(r > 0).sum():,} kept, max {r.max():.3f}")
