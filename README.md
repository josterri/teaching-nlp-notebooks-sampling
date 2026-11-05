# Sampling the Next Word

After every prompt, GPT-2 small has a probability for each of its 50,257 tokens. Greedy decoding, temperature, top-k and top-p are four rules for picking the next token from them, and each gives different continuations.

| Notebook | Content | Time | Colab |
|---|---|---|---|
| `01_sampling_visual.ipynb` | bar charts, sliders, continuations coloured by probability: greedy decoding, temperature, top-k, top-p | 20 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/josterri/teaching-nlp-notebooks-sampling/blob/main/01_sampling_visual.ipynb) |
| `02_sampling_mathematics.ipynb` | formulas with derivations, each checked on GPT-2's numbers: temperature limits, entropy against T, the divergence of a truncation, the top-p bound, min-p, the repetition penalty, beam search | 60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/josterri/teaching-nlp-notebooks-sampling/blob/main/02_sampling_mathematics.ipynb) |

In Colab, run the setup cell first; GPT-2 small (548 MB) is downloaded on the first run. On a local machine: `pip install -r requirements.txt`, then open a notebook in Jupyter.
