# Experiments

Runnable experiments for the [posttraining](https://github.com/ivanleomk/posttraining) notebook. This repository is separate from the notes and Anki decks, and the notebook includes it as a git submodule at `experiments/`.

## Layout

One folder per experiment:

```text
<experiment-name>/
  README.md    # question, setup, how to run, result
  ...
```

## Clone

From a fresh checkout of the notebook:

```bash
git submodule update --init experiments
```

To clone this repository on its own:

```bash
git clone git@github.com:ivanleomk/posttraining-experiments.git
```

## Experiments

- [GPT-J code SFT: dataset size and training objective](code-sft-data/README.md) — dataset preparation, generated executable tests, NTP/NextLat-style LoRA ablations and evaluation plan.
