# Bigram vs LSTM vs Small Transformer — Next-Character Prediction

Applied NLP midterm project. Three language models, one task, one dataset, one
set of hyperparameters, and an honest look at whether attention was worth it.

> **Status: in progress.** The notebook is complete and verified; the final
> numbers, interpretation, and reflection are filled in from the full training
> run. Sections marked `TO FILL` are not finished yet.

---

## What NLP task did I solve?

**Character-level next-character prediction.** Given a window of preceding
characters, each model outputs a probability distribution over the next
character. Training minimises cross-entropy; generation samples from the
distribution repeatedly, feeding each sampled character back in as context.

Working at the character level rather than the word level keeps the vocabulary
at 65 symbols instead of tens of thousands, which is what makes three models
trainable in a single session. The cost is that the models have to learn
spelling before they can learn anything else — which turns out to be useful,
because you can watch that happen.

## What dataset did I use?

**Tiny Shakespeare** — 1,115,394 characters of dialogue from the plays,
formatted as a speaker name, a colon, then that speaker's lines.

| Property | Value |
| --- | --- |
| Total characters | 1,115,394 |
| Distinct characters (vocabulary) | 65 |
| Train split (first 90%) | 1,003,854 characters |
| Validation split (last 10%) | 111,540 characters |
| Characters in validation but never in training | 0 |

The split is contiguous rather than random, because the training units are
overlapping character windows — a random split would put a window's own
neighbours on the other side of the split and leak the answer.

Two baselines are computed before any model is built, so that a loss number
means something:

| Baseline | Loss (nats/char) |
| --- | --- |
| Uniform guess over 65 characters, `ln(65)` | 4.1744 |
| Character frequency only (unigram entropy) | 3.3128 |

The second is the one that matters. A model scoring between those two numbers
looks like it is learning when all it has picked up is that spaces and `e` are
common.

Details and a re-download command are in [`data/README.md`](data/README.md).

## What three models did I compare?

| | Context available | How earlier information reaches the prediction |
| --- | --- | --- |
| Bigram | 1 character | a lookup in a 65×65 count table |
| LSTM | one 128-character window | compressed into a fixed-size state, updated one step at a time |
| Transformer | one 128-character window | self-attention reads any earlier position directly |

## How does each model work?

### 1. Bigram

Count how often each character follows each other character, then normalise:

```
P(next | current) = count(current, next) / count(current)
```

There is no gradient training — the model is a single counting pass.

**What it uses:** one character. **What it ignores:** everything else — the
character before that, the word being spelled, whether a quote is open, who is
speaking.

**Why its language understanding is limited:** it has no notion of a word,
because it never sees two characters at once. It can learn that `h` often
follows `t`, but not that `t-h-e` is a word.

One real finding here: **smoothing is mandatory, not cosmetic.** Only 1,380 of
the 4,225 possible character pairs occur in training, and 187 validation
positions (0.17%) use a pair that training never saw. Unsmoothed, those give
`-log(0)` and the validation loss is infinite. Add-1 smoothing fixes that, but
it is not free — it assigns a floor probability to characters that essentially
never occur, which is why `$` (a character appearing exactly **once** in 1.1M
characters) shows up in the bigram's generated text.

### 2. LSTM

Reads the window one character at a time, maintaining a hidden state and a
**cell state** updated through three learned gates:

```
c_t = f_t * c_{t-1} + i_t * g_t
h_t = o_t * tanh(c_t)
```

The decisive detail is that the cell state is carried forward by **addition**,
not by a matrix multiplication. With the forget gate near 1, information — and
gradient — passes through a timestep essentially unchanged. A plain RNN has no
such path: a gradient travelling back 60 steps is multiplied by roughly `W_h`
sixty times and vanishes, so the model never receives the signal that would
teach it a long-range rule. The LSTM's additive path is why it can *learn* to
use distant context instead of merely being permitted to.

**Why it still struggles with longer dependencies:** everything it knows about
the preceding 120 characters must fit in one fixed-size vector, contested by
everything else competing for space; distance is survivable but not free; and
it is sequential, so position *t* cannot be computed until *t−1* is.

### 3. Small transformer

A decoder-only transformer with the five required components:

| Component | What it does |
| --- | --- |
| Token embeddings | maps each of the 65 characters to a learned vector |
| Positional embeddings | adds a learned vector per position in the window |
| Causal multi-head self-attention | each position gathers information from earlier positions |
| Feedforward layer | per-position processing of what attention gathered |
| Output layer | projects back to 65 scores, one per possible next character |

Each position emits a **query**, a **key** and a **value**. Queries are scored
against all keys, scaled by `1/sqrt(d_k)`, softmaxed, and used to take a
weighted sum of values:

```
Attention(Q, K, V) = softmax(Q Kᵀ / sqrt(d_k)) V
```

Position 120 therefore reads position 1 in **one** operation, with no
compression into a running state and no decay with distance, and all positions
are computed in parallel.

Two parts of this are not optional:

- **Positional embeddings**, because a weighted sum does not depend on the order
  of what is summed — without them, `"dog"` and `"god"` are the same input.
- **Causal masking**, because without it the model can see the character it is
  being asked to predict. This failure is silent: training loss collapses to
  something impressively low and the model has learned nothing. The notebook
  therefore *proves* the mask holds, by editing a later character and
  confirming that predictions at earlier positions do not move at all
  (measured change: `0.000e+00`), and separately by checking that no attention
  weight above the diagonal is non-zero.

## How the comparison was kept fair

This is the part I spent the most care on, because it is easy to accidentally
measure the wrong thing.

- **One interface.** All three models are `nn.Module`s mapping `(batch, time)`
  character indices to `(batch, time, 65)` scores, so the **same** evaluation
  function and the **same** sampling function are used for all three. No model
  is measured by its own code path.
- **Identical evaluation batches.** `estimate_loss` draws from its own
  generator, re-seeded to the same value every call, so every model is scored
  on exactly the same batches — and a mid-training evaluation does not disturb
  the training batch sequence.
- **Identical training.** The LSTM and the transformer are trained by the same
  function, with the same context length (128), batch size (64), learning rate
  (3e-3), optimizer (AdamW), and number of steps (5000), from the same seed.
- **Matched capacity.** The LSTM's hidden size is 240 rather than a round 256
  specifically so the two trained models land within **1.8%** of each other on
  parameter count (841,905 vs 826,433). Otherwise "the transformer won" could
  just mean "the transformer was bigger".
- **Equal steps, not equal wall-clock time.** Equal steps compares the
  architectures at a fixed number of updates; equal wall-clock time would
  instead reward whichever architecture is faster per step. Since speed is
  itself one of the things being compared, the budget is fixed and the time is
  *reported* as a result.
- **The bigram's asymmetry is stated, not hidden.** It has no epochs, no
  trainable parameters, and its "training time" is a single counting pass.
- **Identical sampling.** Same prompt, seed, length and temperature for all
  three, so differences in the generated text are differences in the models.
- **One unit throughout.** All losses are in nats per character.

The learning rate was chosen by measurement, not habit: in a pilot run, 3e-3
reached a lower validation loss than 1e-3 for **both** trained models, so it is
the better setting on its own merits rather than one that flatters either
architecture.

## Which model performed best?

`TO FILL` — from the full run. The results table is generated from measured
values in [`results/results_table.md`](results/results_table.md), and the raw
numbers are in `results/metrics.json`.

One thing already known from the pilot run and worth reporting rather than
hiding: **the ranking depends on the step budget.** The LSTM is ahead for the
first few thousand steps, reaches its best validation loss, and then begins to
overfit, while the transformer starts slower and is still improving when the
LSTM has turned around. The learning-curve plot in the notebook shows the
crossover directly.

## What did attention improve?

`TO FILL`

## What did I learn from this project?

`TO FILL`

## How did I use AI responsibly?

See [`ai_use_summary.md`](ai_use_summary.md) for the full dated log.

In short: I used Claude to have the architectures explained to me and to draft
the implementations, and I made every design decision myself — which
non-transformer model to use, which dataset, which hardware, and the step
budget, which I chose knowing it determines which model wins. I ran all the
training, read all the generated samples, and wrote all of the interpretation
and reflection. The log records which parts are which.

---

## Repository layout

```
.
├── nlp_midterm.ipynb        the complete project: data, all three models, results
├── README.md
├── ai_use_summary.md        dated log of how AI was used
├── requirements.txt
├── data/
│   ├── input.txt            Tiny Shakespeare (committed, 1.1 MB)
│   └── README.md            dataset statistics and re-download instructions
└── results/
    ├── results_table.md     generated by the notebook from measured values
    ├── metrics.json         full config and results
    └── samples/             generated text from each model
```

## How to run it

### Google Colab (what I used)

```python
!git clone https://github.com/Chirag-Mokashi/nlp-midterm.git
%cd nlp-midterm
```

Then open `nlp_midterm.ipynb` and run all cells. Set the runtime to a GPU
(Runtime → Change runtime type → T4 GPU) before running, since the training
times in the results table are only comparable if all models run on the same
device in one session.

The notebook downloads the dataset itself if `data/input.txt` is missing, so it
also runs from a bare Colab session with no clone.

### Locally

```bash
pip install -r requirements.txt
jupyter notebook nlp_midterm.ipynb
```

The notebook selects CUDA, then Apple MPS, then CPU, whichever is available
first.
