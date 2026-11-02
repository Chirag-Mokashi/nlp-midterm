# AI-Use Summary

> **Status: living document.** I am writing this as I go, not reconstructing it
> at the end. Each entry is dated and records what I asked for, what the AI
> produced, and what I did with it.

**Assistant used:** Claude (Claude Code, Opus 5)
**Project:** character-level next-character prediction — bigram vs LSTM vs transformer
**Dataset:** Tiny Shakespeare

---

## Summary of the division of work

| Area | AI's role | My role |
| --- | --- | --- |
| Explaining the three architectures | Explained mechanisms, trade-offs, and expected behaviour | Asked follow-up questions until I could explain each one without notes |
| Choosing the non-transformer model | Laid out RNN vs LSTM vs feedforward with the consequences of each | **My decision** — I chose LSTM |
| Choosing dataset and hardware | Gave the trade-offs | **My decisions** — Tiny Shakespeare, Colab GPU |
| Experiment design (what makes the comparison fair) | Proposed the constraints | Reviewed and approved them |
| Code | Drafted the implementations | Read every function, ran them, debugged failures |
| Running the experiments | — | **Entirely mine** — I ran all training in Colab |
| Reading generated samples and judging quality | — | **Entirely mine** |
| Results interpretation and reflection | Reviewed my drafts and pushed back where a claim was not supported by a number | **My writing** |

---

## Log

### 2026-11-01 — Project setup and architecture decisions

**What I asked:** to have the whole project explained before writing any code —
specifically what each of the three models actually does, and why attention is
supposed to be an improvement.

**What the AI explained:**

- The shared task: every model outputs a probability distribution over the next
  character, trained with cross-entropy.
- Bigram: a 65x65 count table, one character of context.
- The three allowed non-transformer options and how they differ in *how
  information from far back reaches the current prediction* — a fixed window
  (feedforward) vs a running state that is overwritten each step (RNN) vs a
  gated additive cell state (LSTM).
- Why a vanilla RNN struggles: the gradient has to travel back through one
  matrix multiply per timestep, so it shrinks multiplicatively and the model
  never receives the signal that would teach it a long-range rule. The LSTM's
  cell state is updated by *addition* when the forget gate is near 1, which is
  the path that lets gradients survive.
- Why a transformer removes both limits: self-attention reads any earlier
  position directly in one operation instead of through a compressed running
  state, and all positions are computed in parallel. Positional encoding is
  required precisely because attention is otherwise order-blind.

**My decision:** LSTM for model 2. I picked it over the vanilla RNN because I
want three clearly separated results rather than a middle row I would have to
explain away, and over the feedforward because a fixed window is closer to a
wider bigram than to a genuine context model.

*(To fill in myself: the specific thing about the LSTM cell state that I had
to ask about twice before it made sense.)*

### 2026-11-01 — Hour 1: dataset and setup

**What the AI did:** downloaded Tiny Shakespeare, verified its checksum, and
computed the corpus statistics rather than estimating them — character count,
vocabulary, the 90/10 split sizes, and a check that the validation split
introduces no character the training split has never seen.

It also computed two baselines I had not thought to ask for: the uniform-guess
loss (`ln 65` = 4.1744 nats) and the character-frequency-only loss (3.3128
nats). These matter because they tell me what a *meaningless* result looks
like — a model scoring 3.4 would look like it was learning when it had only
learned which letters are common.

**What I did:** *(to fill in — verify the numbers, confirm the split choice.)*

### 2026-11-01 — Hour 2: bigram

**What the AI did:** implemented the count-based bigram, and measured the
sparsity of the count table instead of asserting that smoothing is a good idea.
Only 1,380 of 4,225 possible character pairs occur in training, and 187
validation positions use a pair training never saw — so an unsmoothed model has
an *infinite* validation loss. That turned smoothing from a line of boilerplate
into something I can explain the need for.

It also cross-checked the bigram loss two ways — once in pure Python over the
whole corpus, once in PyTorch over sampled batches — which agreed to within
0.003 nats.

**Something I noticed in the output myself:** the bigram's generated text
contains a `$`. There is exactly one `$` in the entire 1.1M-character corpus, so
it should essentially never appear. It appears because add-1 smoothing gives it
a floor probability after every character. So smoothing is required *and* it
distorts the model — I would not have thought of that as a trade-off before
seeing it in the sample.

### 2026-11-01 — Hours 3-6: LSTM, transformer, and the decision that shaped the result

**What the AI did:** drafted the LSTM and the transformer, built the shared
training loop both use, and wrote a verification for the causal mask rather
than trusting it — the test edits a character later in the sequence and
confirms predictions at earlier positions do not move (measured change exactly
`0.000e+00`), cross-checked by confirming no attention weight above the diagonal
is non-zero. This matters because a broken causal mask fails *silently*: the
loss looks excellent and the model has learned nothing.

It also set the LSTM's hidden size to 240 rather than 256 so the two trained
models land within 1.8% of each other on parameter count, so that a win cannot
be explained away as one model simply being bigger.

**The decision I had to make, and why it was mine to make.** The AI ran pilot
training runs rather than picking a learning rate by habit, and found something
I had not expected: **which model wins depends on the step budget.** At
`lr=3e-3` the LSTM leads for the first ~3,000 steps, reaches its best
validation loss around step 2,000, and then starts to overfit, while the
transformer starts slower and overtakes somewhere between 3,000 and 4,000 steps.
At `lr=1e-3` the LSTM leads at every checkpoint out to 6,000 and the
transformer never catches up.

The AI explicitly declined to choose the budget for me on the grounds that it
determines the conclusion. I chose **`lr=3e-3`, 5,000 steps**, on the reasoning
that 3e-3 measured better for *both* models than 1e-3 did, so it is the better
hyperparameter on its own merits and not one picked because of who it favours.
I also decided that the crossover gets **reported in the write-up**, with the
full learning curve, rather than being left hidden inside a hyperparameter
choice.

*(To fill in myself: whether I still think that was the right call after seeing
the full run.)*

**What I did:** ran the real training, and all the numbers in the results table
come from my own run, not from the AI's pilot runs on a different machine.

### Next entries

- Hour 7: results table and interpretation
- Hour 8: README and reflection

---

## What I understand without assistance

*(To fill in at the end — I will write, in my own words and without looking
anything up, what each model does and why the transformer wins. If I cannot
write it, I have not finished the project.)*
