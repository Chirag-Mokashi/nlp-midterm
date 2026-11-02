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
| Results interpretation and reflection | Checked each of my claims against the measured numbers, corrected the ones that were wrong, and edited my grammar and phrasing | **My observations, judgments and conclusions** |

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

**What took me longest:** the gate equations — specifically that the cell state
`c_t` is an additive conveyor belt running the length of the sequence, while the
hidden state `h_t` is a filtered view of it that gets read out at each step. It
took two different explanations before that clicked.

### 2026-11-01 — Hour 1: dataset and setup

**What the AI did:** downloaded Tiny Shakespeare, verified its checksum, and
computed the corpus statistics rather than estimating them — character count,
vocabulary, the 90/10 split sizes, and a check that the validation split
introduces no character the training split has never seen.

It also computed two baselines I had not thought to ask for: the uniform-guess
loss (`ln 65` = 4.1744 nats) and the character-frequency-only loss (3.3128
nats). These matter because they tell me what a *meaningless* result looks
like — a model scoring 3.4 would look like it was learning, because it beats a
uniform guess, when in fact it is doing worse than simply predicting the overall
letter frequencies.

**What I did:** I checked the corpus statistics, the vocabulary size and the two
baselines against the notebook's own output rather than taking them on trust. I
also confirmed the decision to use a contiguous 90/10 train/validation split:
with a random split, the overlapping 128-character windows would put a window's
immediate neighbours on the other side of the split and leak the answer.

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
contains a `&`. There are exactly three `&` in the entire 1.1M-character corpus,
so it should essentially never appear. It appears because add-1 smoothing gives
it a floor probability after every character. So smoothing is required *and* it
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

**Having now seen the full run, I think it was the right call.** The crossover
landed between steps 3,500 and 4,000, so a 5,000-step budget sits past the
inflection point rather than stopping early at a point that would have flattered
the LSTM. The final margin is thin — 0.0226 nats — and the LSTM still wins on
best checkpoint, which is why both numbers are reported rather than only the one
the budget happened to produce.

**What I did:** ran the real training, and all the numbers in the results table
come from my own run, not from the AI's pilot runs on a different machine.

### 2026-11-02 — Hours 7-8: interpretation, reflection, and README

**How this part was written, stated plainly.** I read the generated samples,
the results table and the learning-curve plot myself and wrote down what I saw
and what I concluded from it. The AI then did two things with that: it checked
my claims against the measured numbers, and it edited my English. Every
observation, comparison and verdict in §2.3, §3.3, §4.3, §5.5 and §6 is mine;
the grammar and sentence construction were edited with assistance.

**What the check actually caught.** Three of my claims were wrong against my own
results, and they were not phrasing problems:

- I called 1.5048 and 1.5180 *training* losses. They are *validation* losses at
  each model's best checkpoint — the LSTM's training loss was 1.0792.
- In the "what did the transformer improve" answer I wrote that the transformer
  *underperformed* the LSTM on final validation loss. It is the other way round
  (1.5219 against 1.5445), and it contradicted my own answer two paragraphs
  earlier.
- My reflection said the LSTM's weakness was its "relatively short context
  window". Both models were given the identical 128-character window on
  purpose; the LSTM's limitation is that it compresses that window into one
  fixed-size state, not that it sees less of it.

Two more were overclaims rather than errors. I had written that the attention
heads attend to "line formatting and punctuation marks" — the heatmap shows a
vertical stripe at one position, but nothing in the notebook decodes *which*
character sits there, so that was a cause I inferred and stated as an
observation. It now describes what is actually visible. I had also written in
§5.5 that the transformer produced "slightly more cohesive" text while §4.3 said
the two were indistinguishable; the §4.3 reading is the honest one and both now
say so.

**What I decided not to do.** I considered a seed-variance check across several
seeds to see whether the ordering between the two models is stable, and decided
against it — the assignment asks for three models and a comparison, and adding
an experiment nobody asked for is scope, not rigour. The consequence is worth
stating plainly rather than hiding: the 0.023-nat margin between the LSTM and
the transformer is a single-seed result, and I have no evidence about how much
of it is seed noise.

**The one I should have caught** is the claim that the transformer did worse than
the LSTM on final validation loss, because it contradicted the answer I had
given two paragraphs earlier in the same section. Catching it needed no numbers
at all — only reading my own answers in order. I skimmed my own work instead of
reading it. The other two are different: the training-versus-validation mislabel
needs me to remember which column a number came from, and the context-window one
is only wrong because I matched the two windows deliberately, which is a fact
about my own experiment design rather than about LSTMs in general.

### 2026-11-02 — A cold review of the finished submission

**What I asked:** before making the repository public, for the whole submission
to be checked over again from scratch — every number re-derived from the corpus
and from `metrics.json` rather than read back off my own tables, and every claim
in the notebook, the README and this file checked against the executed outputs.

**The one that mattered.** I had written, in four places, that the transformer
"was still improving when the budget ran out." It was not. `metrics.json` records
its best checkpoint at step **4000**, and it got slightly *worse* over the last
thousand steps (1.5180 → 1.5219). I had built a conclusion on that premise — that
a longer run would most likely flip the result — and the premise was not in my
own data. What my data does show is that both models peaked before the budget
ended and that the LSTM's decline afterwards is about ten times steeper (0.0396
nats against 0.0039). That supports the same conclusion and is true, so that is
what the write-up now says.

**Two claims I had attributed to a cause I never measured.** I first explained
the transformer's steadier curve by its residual connections and multi-head
attention, and the LSTM's sharper decline by the fixed 128-character window.
Neither
survives: the window is identical for both models by design, and the transformer
carries **eight** dropout sites to the LSTM's **one**, because `nn.LSTM` applies
dropout only between stacked layers. At the same nominal `p=0.1` the transformer
is simply the more regularised model, which is enough on its own to explain the
smaller train/validation gap. The observation stays; the mechanism is gone, and
the dropout asymmetry is now stated wherever that gap is discussed.

**Smaller things that were wrong.** The opening table in the notebook still said
the LSTM sees "the whole sequence, in principle" against the transformer's "the
whole block" — the same context-window error I had already corrected in §6 and
left standing in the first table a reader meets. My interpretation of the
baselines was backwards in three places: a model scoring between 4.1744 and
3.3128 has *not* "only learned which letters are common," it is doing worse than
that. The README's status banner claimed every number in it came from one
measured run, while two learning-rate pilot figures came from somewhere else
entirely. A reference to §4.1 should have been §4.4, "the preceding 120
characters" should have been 128, and `requirements.txt` was missing the package
needed to follow my own local-install instructions.

**What I now know about the timings that I did not before.** The training times
are `time.perf_counter()` wall-clock with no CUDA synchronisation, and reading
the per-500-step figures back shows the LSTM's chunks rising 19.6% across a
constant workload while the transformer's stay flat. I do not know why, and I
have deliberately not guessed: a plausible-sounding explanation with no
measurement behind it is exactly the kind of claim this review existed to catch.
The 60% figure is now described as what this run measured rather than as a
property of the architectures.

**The pattern across all of it.** Every one of these had been read over by me
more than once. What found them was re-deriving the numbers independently
instead of re-reading the sentences, and the one real error was in the only
claim I had never had a number for.

---

## What I understand without assistance

*Written at the end, in one sitting, without looking anything up.*

- **The task and how it is evaluated.** Next-character prediction trains a model
  to output a probability distribution over the 65 characters in the corpus,
  given the text that precedes it, optimised by minimising cross-entropy and
  measured in nats per character. The baselines are what make a loss number
  mean anything: a uniform guess (ln 65) and the unigram entropy tell you
  whether the model learned structure or only learned which letters are common.

- **Bigram.** One character of context, stored in a 65x65 count matrix. Many
  valid character pairs never occur in a corpus this small, so add-1 smoothing
  is not optional — without it a transition the model never saw gets
  probability zero and the validation loss is infinite.

- **LSTM.** Extends the context by passing the input through a recurrent cell
  with gates, in which the cell state acts as an additive conveyor belt. The
  forget, input and output gates control what is written to it and what is read
  out, and because that path is additive rather than multiplicative the
  gradient does not fade away across many steps. That is what lets the network
  learn spelling, word separation and the `NAME:` tag format.

- **Transformer.** Removes the recurrent compression bottleneck using
  multi-head self-attention, which computes a direct dot-product relationship
  between positions inside a 128-character block instead of passing information
  along one step at a time. Attention is order-agnostic, so learned positional
  embeddings are needed to impose order on the sequence. A lower-triangular
  causal mask is compulsory, so that no position can see the character it is
  being asked to predict.

- **The trade-off between the architectures.** At this scale — 1.1M characters,
  and 842k parameters in the LSTM against 826k in the transformer — the two
  land in much the same place, around 1.52-1.54 nats of validation loss, and the
  LSTM is faster in wall-clock time. What the transformer buys is
  not a better number here but the removal of the recurrent limit, which is
  what makes scaling up and exploiting hardware parallelism possible at all.
