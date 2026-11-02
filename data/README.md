# Dataset

## What it is

**Tiny Shakespeare** — a concatenation of lines from Shakespeare's plays, in the
format `SPEAKER:` followed by that speaker's lines. It is the standard small
corpus for character-level language modelling.

`input.txt` is committed to this repository (1.1 MB), so the notebook runs with
no download step once the repository is cloned. The committed run was executed
in a bare Colab session without the clone, which is why its first cell reports
`downloaded the dataset`; the file fetched there is byte-identical to this one,
as the checksum below confirms.

## Measured properties

| Property | Value |
| --- | --- |
| Total characters | 1,115,394 |
| Distinct characters (vocabulary) | 65 |
| Training characters (first 90%) | 1,003,854 |
| Validation characters (last 10%) | 111,540 |
| Characters appearing in validation but never in training | 0 |

The vocabulary is:

```
\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
```

Two reference points used in the results discussion:

| Baseline | Loss (nats/char) |
| --- | --- |
| Uniform guess over 65 characters — `ln(65)` | 4.1744 |
| Character-frequency-only (unigram entropy) | 3.3128 |

Any model that does not beat 3.3128 has learned nothing beyond which letters
are common.

## Re-downloading it

If `input.txt` is missing:

```bash
curl -o data/input.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Verify the file:

```bash
shasum -a 256 data/input.txt
# 86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed
```

## Licence

The underlying text is Shakespeare, which is public domain.
