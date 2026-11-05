# %% [markdown]
# # Beam Search: The Most Probable Continuation
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/josterri/teaching-nlp-notebooks-sampling/blob/main/03_beam_search.ipynb)
#
# After `It is a truth universally acknowledged that`, GPT-2 small's likeliest next token is `the`, with probability 0.182. Its likeliest continuation of two tokens is `there is`, with probability 0.0159: 5.9 times as probable as `the United` (0.0027), the continuation under greedy decoding.
#
# Run the two setup cells first; on the first run, GPT-2 small (548 MB) is downloaded. In each section, write down an answer to the question, move the sliders, then open the answer. A token is a word or a piece of a word; ↵ is a line break.

# %% include setup.py

# %%
# @title Charts for the sections below (run this cell)
import textwrap


@functools.lru_cache(maxsize=32)
def two_step_tree(prompt, width=3):
    """The `width` likeliest first tokens and, after each, the `width` likeliest second tokens, with probabilities."""
    ids = tok(prompt).input_ids
    p1 = softmax(forward([ids])[0, -1].numpy())
    tree = []
    for a in np.argsort(-p1)[:width]:
        p2 = softmax(forward([ids + [int(a)]])[0, -1].numpy())
        tree.append((int(a), float(p1[a]), [(int(b), float(p2[b])) for b in np.argsort(-p2)[:width]]))
    return tuple(tree)


def tree_picture(prompt, B=None):
    """The two-step tree; B = None marks the greedy path, a width B marks what beam search keeps."""
    tree = two_step_tree(prompt)
    paths = {(a, b): pa * pb for a, pa, kids in tree for b, pb in kids}
    if B is None:
        kept1 = kept2 = set()
        result = (tree[0][0], tree[0][2][0][0])                               # greedy path
    else:
        kept1 = {a for a, _, _ in tree[:B]}                                   # B likeliest first tokens
        kept2 = set(sorted((ab for ab in paths if ab[0] in kept1), key=paths.get, reverse=True)[:B])
        result = max(kept2, key=paths.get)
    mark = ORANGE if B is None else BLUE
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axis("off")
    ax.set_xlim(-0.35, 3.4)
    ax.set_ylim(-0.8, 9.2)

    def box(x, y, text, colour, bold=False):
        ax.text(x, y, text, ha="left" if x > 1.5 else "center", va="center", fontsize=9,
                fontweight="bold" if bold else "normal", color="#0b0b0b",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colour, lw=1.5 if bold else 1))

    for j, (a, pa, kids) in enumerate(tree):
        ya = 8 - (3 * j + 1)
        on = a == result[0]
        c1 = mark if on else (BLUE if a in kept1 else GREY)
        ax.plot([0, 1], [4, ya], color=c1, lw=4 if on else 1.5, zorder=1)
        for m, (b, pb) in enumerate(kids):
            yb = 8 - (3 * j + m)
            on2 = (a, b) == result
            c2 = mark if on2 else (BLUE if (a, b) in kept2 else GREY)
            ax.plot([1, 2], [ya, yb], color=c2, lw=4 if on2 else 1.5, zorder=1)
            box(2.02, yb, f"{word(b)}  {pb:.3f}    path {paths[(a, b)]:.4f}", c2, bold=on2)
        box(1, ya, f"{word(a)}\n{pa:.3f}", c1, bold=on)
    box(0, 4, textwrap.fill(f'"{prompt}"', 16), INK)
    ax.text(1, 8.7, "first token", ha="center", color=INK)
    ax.text(2.02, 8.7, "second token, path probability", ha="left", color=INK)
    if B is None:
        ax.set_title("the 3 likeliest tokens at each step; orange: greedy decoding")
    else:
        ax.set_title(f"beam search, B = {B}: blue kept, grey dropped; thick line: result")
    plt.show()
    return result


def show_tree(prompt=AUSTEN):
    tree_picture(prompt)


def show_beam_tree(prompt=AUSTEN, B=1):
    tree_picture(prompt, B)
    tokens, lp = beam_search(prompt, B, n_tokens=2)
    print(f"beam search over all {V:,} tokens, B = {B}: {tok.decode(tokens)!r}, probability {np.exp(lp):.4f}")


def log_softmax_steps(prompt, tokens):
    """log p of each token of a continuation, from one forward pass over prompt and continuation."""
    ids = tok(prompt).input_ids
    L = torch.log_softmax(forward([ids + list(tokens)])[0], -1).numpy()
    return np.array([L[len(ids) - 1 + t, tok_id] for t, tok_id in enumerate(tokens)])


def beam_search(text, B, n_tokens=20):
    """Beam search of width B: the B partial continuations with the largest log P, extended token by token."""
    start = tok(text).input_ids
    beams = [(start, 0.0)]                                                     # (tokens, log P)
    for _ in range(n_tokens):
        L = torch.log_softmax(forward([s for s, _ in beams])[:, -1], -1).numpy()
        L = L + np.array([[lp] for _, lp in beams])                            # log P of every extension
        best = np.argsort(L, axis=None)[::-1][:B]                              # the B largest of B x 50,257
        beams = [(beams[r][0] + [int(c)], L[r, c]) for r, c in zip(*np.unravel_index(best, L.shape))]
    return beams[0][0][len(start):], float(beams[0][1])


def show_beam(prompt=AUSTEN, B=1, n_tokens=20):
    colour_key()
    for b in sorted({1, B}):
        tokens, lp = beam_search(prompt, b, n_tokens)
        show_text(prompt, tokens, np.exp(log_softmax_steps(prompt, tokens)), label=f"B = {b}, log P = {lp:.2f}")

# %% [markdown]
# ## 1. Two Steps After the Austen Prompt
#
# A continuation of two tokens has the probability of its first token times the probability of its second token after the first: 0.182 × 0.015 = 0.0027 for `the United`. The tree has the three likeliest first tokens and, after each, the three likeliest second tokens. Under greedy decoding, the likeliest token is picked at each step: the orange path.
#
# **Question.** Which of the nine paths has the largest probability, and how many times as probable as the greedy path `the United` is it?

# %%
show_tree(AUSTEN)

# %% [markdown]
# <details><summary><b>Answer</b></summary>
#
# `there is`: 0.038 × 0.422 = 0.0159, 5.9 times the 0.0027 of `the United`. The first token `there` has a small probability, but after it GPT-2's probability is concentrated on `is` (0.422) and `are` (0.398); after `the`, it is spread, and the likeliest second token, `United`, has 0.015. Under greedy decoding only the first step is compared, and `there` is dropped at once.
#
# </details>

# %% [markdown]
# ## 2. Beam Search of Width B
#
# Beam search of width B, step by step:
#
# 1. keep the B partial continuations with the largest probability;
# 2. extend each of them by every token, all 50,257;
# 3. keep the B most probable extensions, and repeat.
#
# B = 1 is greedy decoding. In the tree, kept partial continuations are blue, dropped ones grey, and the result is the thick line. Under the tree: the result of beam search over all 50,257 tokens, the same path as the thick line.
#
# **Question.** What is the smallest B at which the result is `there is`? Try other prompts in the box as well.

# %%
show_beam_tree(AUSTEN, B=1)

# %%
explore(show_beam_tree, prompt=prompt_box(AUSTEN), B=slider(1, 3, 1, 1))

# %% [markdown]
# <details><summary><b>Answer</b></summary>
#
# B = 2. At B = 1, only `the` is kept after the first step, and the result is `the United` (0.0027), the greedy path. At B = 2, `the` and `there` are kept; among their extensions, `there is` (0.0159) and `there are` (0.0150) are the two most probable, and `there is` is the result. B = 3 gives the same result.
#
# </details>

# %% [markdown]
# ## 3. Longer Continuations
#
# Over 20 tokens, the probability of a continuation is the product of 20 token probabilities. Its logarithm, log P, is the sum of the 20 log-probabilities: a negative number, closer to 0 for a more probable continuation. The first line is greedy decoding (B = 1), the second beam search of width B; each token has a background set by its probability.
#
# **Question.** Move B from 1 to 5 at 20 tokens. Does log P grow with every step of B?

# %%
show_beam(AUSTEN, B=5, n_tokens=20)

# %%
explore(show_beam, prompt=prompt_box(AUSTEN), B=slider(1, 5, 1, 1), n_tokens=slider(10, 40, 10, 20))

# %% [markdown]
# <details><summary><b>Answer</b></summary>
#
# No. log P is −27.77 at B = 1, −27.78 at B = 2, −27.34 at B = 3, and −25.14 at B = 4 and 5. At B = 2 the greedy path is dropped at an intermediate step, and the continuation kept in its place has the lower probability at the end. Under greedy decoding the text is the loop `The United States is a nation of immigrants.`; from B = 2 on, it is `there is no such thing as a ...`.
#
# </details>

# %% [markdown]
# ## Beam Search Compared With Greedy Decoding
#
# - The probability of a continuation is the product of its token probabilities: 0.038 × 0.422 = 0.0159 for `there is`.
# - Under greedy decoding (B = 1), each step is decided alone: `the United`, 0.0027.
# - With beam search of width 2, the result is `there is`, 5.9 times as probable.
# - Over 20 tokens, log P is −27.78 at B = 2 against −27.77 at B = 1: a wider beam can end on a less probable continuation.
