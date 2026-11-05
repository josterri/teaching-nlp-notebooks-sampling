# @title Setup: GPT-2 small and the decoding rules (run this cell first)
import functools, html, inspect, subprocess, sys, warnings

warnings.filterwarnings("ignore")
try:
    import transformers  # preinstalled in Colab
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers"], check=True)

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import HTML, display
from huggingface_hub.utils import logging as hub_logging
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.utils import logging as hf_logging

hub_logging.set_verbosity_error()
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()
torch.set_grad_enabled(False)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading GPT-2 small; on the first run, 548 MB are downloaded ...")
tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(DEVICE)
V = len(tok)  # 50,257 tokens

AUSTEN = "It is a truth universally acknowledged that"
PEAKED = "The Prime Minister of the United"
FLAT = "I went to the"
BANK = "The central bank decided to"
PROMPTS = [AUSTEN, PEAKED, FLAT, BANK]

BLUE, ORANGE, AQUA, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#c9c8c2", "#52514e"
plt.rcParams.update({
    "figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e6e5e0", "axes.axisbelow": True,
    "axes.titlelocation": "left", "axes.titlesize": 9, "legend.frameon": False,
})


def forward(seqs):
    """GPT-2's scores at every position of each token list in `seqs` (all of one length), float64 on the CPU."""
    return model(torch.tensor(seqs, device=DEVICE)).logits.double().cpu()


@functools.lru_cache(maxsize=128)
def logits(text):
    """GPT-2's score z_i for every token i that may follow `text`."""
    z = forward([tok(text).input_ids])[0, -1].numpy()
    z.setflags(write=False)
    return z


def softmax(z, T=1.0):
    """Probabilities from the scores z at temperature T. T = 0 puts all mass on the largest score."""
    if T == 0:
        p = np.zeros(len(z))
        p[np.argmax(z)] = 1.0
        return p
    e = np.exp((np.asarray(z) - np.max(z)) / T)
    return e / e.sum()


def top_k_filter(p, k):
    """Keep the k likeliest tokens (ties at the boundary kept), rescale them to sum to 1."""
    q = np.where(p >= np.sort(p)[-k], p, 0.0)
    return q / q.sum()


def top_p_filter(p, top_p):
    """Keep the smallest set of likeliest tokens whose probabilities add up to at least top_p."""
    order = np.argsort(p)                      # least likely first
    drop = np.cumsum(p[order]) <= 1 - top_p    # the tail holding at most 1 - top_p
    drop[-1] = False                           # the likeliest token always stays
    q = p.copy()
    q[order[drop]] = 0.0
    return q / q.sum()


def min_p_filter(p, p_base):
    """Keep every token whose probability is at least p_base times the largest probability."""
    q = np.where(p >= p_base * p.max(), p, 0.0)
    return q / q.sum()


def penalise(z, seen, theta):
    """Repetition penalty: scores of tokens already in the text, divided by theta if positive, times theta if negative."""
    z = np.array(z)
    s = z[seen]
    z[seen] = np.where(s < 0, s * theta, s / theta)
    return z


def next_token_dist(z, T=0.0, k=0, top_p=1.0, min_p=0.0):
    """The distribution the next token is drawn from: temperature, then top-k, top-p, min-p (k = 0: no top-k)."""
    p = softmax(z, T)
    if k:
        p = top_k_filter(p, k)
    if top_p < 1:
        p = top_p_filter(p, top_p)
    if min_p > 0:
        p = min_p_filter(p, min_p)
    return p


def generate(text, n_tokens=30, T=0.0, k=0, top_p=1.0, min_p=0.0, theta=1.0, n=1, draw=0):
    """n continuations of `text`, token by token. Each is (tokens, GPT-2's probability at T = 1 of each token)."""
    rng = np.random.default_rng(draw)
    start = tok(text).input_ids
    seqs = [list(start) for _ in range(n)]
    probs = [[] for _ in range(n)]
    for _ in range(n_tokens):
        Z = forward(seqs)[:, -1].numpy()
        for s in range(n):
            z = penalise(Z[s], np.unique(seqs[s]), theta) if theta != 1 else Z[s]
            t = int(rng.choice(V, p=next_token_dist(z, T, k, top_p, min_p)))
            probs[s].append(softmax(Z[s])[t])
            seqs[s].append(t)
    out = []
    for seq, pr in zip(seqs, probs):
        new = seq[len(start):]
        end = new.index(tok.eos_token_id) + 1 if tok.eos_token_id in new else len(new)  # stop at <|endoftext|>
        out.append((new[:end], pr[:end]))
    return out


def word(i):
    """Token i as text, without GPT-2's leading space."""
    s = tok.decode([int(i)])
    return s.strip() or repr(s)


def shade(p):
    """Background colour for a token of probability p: white near 1, dark orange at 0.001 and below."""
    level = np.clip(-np.log10(max(p, 1e-12)) / 3, 0, 1)
    r, g, b, _ = plt.get_cmap("Oranges")(0.8 * level)
    return f"rgb({r * 255:.0f},{g * 255:.0f},{b * 255:.0f})"


def show_text(text, tokens, probs, label=""):
    """The prompt in grey, then each new token on a background set by GPT-2's probability of it."""
    spans = "".join(
        f'<span title="probability {p:.3g}" style="background:{shade(p)};color:#0b0b0b">'
        f'{html.escape(tok.decode([t])).replace(chr(10), "↵")}</span>'
        for t, p in zip(tokens, probs))
    head = f'<b style="color:#52514e">{html.escape(label)}</b> ' if label else ""
    display(HTML(
        f'<div style="font-family:Menlo,monospace;font-size:12px;line-height:1.8;background:#fff;'
        f'color:#0b0b0b;padding:4px 8px;margin:2px 0;border-left:3px solid {GREY}">'
        f'{head}<span style="color:#8a8983">{html.escape(text)}</span>{spans}</div>'))


def colour_key():
    cells = "".join(f'<span style="background:{shade(p)};color:#0b0b0b;padding:0 6px">{p:g}</span> '
                    for p in (1, 0.1, 0.01, 0.001))
    display(HTML(f'<div style="font-size:11px;color:#52514e;background:#fff;padding:2px 6px">'
                 f'background: GPT-2\'s probability of each new token at T = 1, before any rule: {cells}</div>'))


def sentences(text, n=3, n_tokens=30, label="", **rule):
    """Print n continuations of `text` under one decoding rule."""
    for tokens, probs in generate(text, n_tokens, n=n, **rule):
        show_text(text, tokens, probs, label)


def bars(ax, p, keep=None, n=20, color=BLUE, order=None, label=None):
    """Bar chart of the n likeliest tokens; tokens outside `keep` in grey."""
    top = np.argsort(-p)[:n] if order is None else order
    colours = [color if keep is None or keep[i] else GREY for i in top]
    ax.bar(np.arange(len(top)), p[top], width=0.7, color=colours, label=label)
    ax.set_xticks(np.arange(len(top)), [word(i) for i in top], rotation=55, ha="right")
    ax.set_ylabel("probability")
    ax.set_xlabel("next token")
    return top


def slider(lo, hi, step, value, name=""):
    kind = widgets.IntSlider if isinstance(step, int) else widgets.FloatSlider
    return kind(min=lo, max=hi, step=step, value=value, description=name, continuous_update=False)


def prompt_box(value=AUSTEN):
    """A text box with the four prompts as suggestions; any other text works too (press Enter)."""
    return widgets.Combobox(value=value, options=PROMPTS, ensure_option=False, description="prompt",
                            continuous_update=False, layout=widgets.Layout(width="90%"))


def explore(f, **controls):
    """Sliders for the arguments of f; the output is redrawn when a slider is released."""
    missing = set(inspect.signature(f).parameters) - set(controls)
    assert not missing, f"no control for {missing}"
    @functools.wraps(f)
    def g(**kwargs):
        if "prompt" in kwargs:
            kwargs["prompt"] = kwargs["prompt"].strip() or AUSTEN  # an empty box falls back to the Austen prompt
        f(**kwargs)
    widgets.interact(g, **controls)


print(f"GPT-2 small loaded: {sum(x.numel() for x in model.parameters()):,} parameters, {V:,} tokens.")
