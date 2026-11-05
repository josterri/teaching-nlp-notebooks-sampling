#!/usr/bin/env python3
"""Compare the notebooks' decoding rules (src/setup.py, beam_search in src/02) with Hugging Face transformers.

    python tools/check_decoding.py

Per-step distributions: temperature, top-k, top-p, min-p and their chain against the
transformers warpers, on GPT-2 scores after four prompts. Whole continuations: greedy decoding,
greedy decoding with repetition penalty 1.2 and beam search of width 2 to 5 against
model.generate. Exits non-zero on the first mismatch.
"""
import itertools
import pathlib
import re
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
exec((ROOT / "src" / "setup.py").read_text())
src2 = (ROOT / "src" / "02_sampling_mathematics.py").read_text()
exec(re.search(r"^def beam_search.*?return .*?\n", src2, re.S | re.M).group(0))

from transformers.generation.logits_process import (MinPLogitsWarper, RepetitionPenaltyLogitsProcessor,
                                                    TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper)

failures = 0


def report(ok, what):
    global failures
    failures += not ok
    print(("ok    " if ok else "FAIL  ") + what)


def hf_dist(z, T, k, top_p, min_p):
    s = torch.tensor(z, dtype=torch.float64)[None]
    ids = torch.zeros((1, 1), dtype=torch.long)
    for warper in ([TemperatureLogitsWarper(T)] + ([TopKLogitsWarper(k)] if k else [])
                   + ([TopPLogitsWarper(top_p)] if top_p < 1 else []) + ([MinPLogitsWarper(min_p)] if min_p else [])):
        s = warper(ids, s)
    return torch.softmax(s, -1)[0].numpy()


settings = list(itertools.product((0.5, 1.0, 1.5), (0, 5, 50), (1.0, 0.9), (0.0, 0.1)))
for prompt in PROMPTS:
    z = np.array(logits(prompt))
    worst = max(np.abs(next_token_dist(z, *s) - hf_dist(z, *s)).max() for s in settings)
    same = all(((next_token_dist(z, *s) > 0) == (hf_dist(z, *s) > 0)).all() for s in settings)
    report(same and worst < 1e-12, f'{len(settings)} settings (T, k, p, min-p) after "{prompt}": max |difference| {worst:.1e}')

    seen = np.unique(tok(prompt).input_ids)
    hf = RepetitionPenaltyLogitsProcessor(1.2)(torch.tensor([tok(prompt).input_ids]), torch.tensor(z)[None])[0].numpy()
    report(np.allclose(penalise(z, seen, 1.2), hf), f'repetition penalty 1.2 after "{prompt}"')

for prompt in (AUSTEN, BANK):
    ids = torch.tensor([tok(prompt).input_ids])
    hf = model.generate(ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)[0, ids.shape[1]:].tolist()
    report(generate(prompt, 40)[0][0] == hf, f'greedy, 40 tokens after "{prompt}"')
    hf = model.generate(ids, max_new_tokens=40, do_sample=False, repetition_penalty=1.2,
                        pad_token_id=tok.eos_token_id)[0, ids.shape[1]:].tolist()
    report(generate(prompt, 40, theta=1.2)[0][0] == hf, f'greedy with penalty 1.2, 40 tokens after "{prompt}"')
    for B in range(2, 6):
        hf = model.generate(ids, max_new_tokens=20, do_sample=False, num_beams=B, early_stopping=False,
                            pad_token_id=tok.eos_token_id)[0, ids.shape[1]:].tolist()
        report(beam_search(prompt, B, 20)[0] == hf, f'beam search B = {B}, 20 tokens after "{prompt}"')

sys.exit(1 if failures else 0)
