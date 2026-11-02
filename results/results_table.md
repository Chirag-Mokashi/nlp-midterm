# Results

Device: `cuda`  |  block size 128, batch 64, lr 0.003, 5000 steps (identical for both trained models)

| Model | Parameters | Train Loss | Val Loss | Best Val | Training Time (s) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Bigram | 0 (counts only) | 2.4516 | 2.4822 | n/a (not trained) | 0.00 | one counting pass, no gradient training; add-1 smoothing |
| LSTM | 841,905 | 1.0792 | 1.5445 | 1.5048 @ 2500 | 104.70 | 2 layers, hidden 240, 5000 steps |
| Transformer | 826,433 | 1.1834 | 1.5219 | 1.5180 @ 4000 | 166.69 | 4 blocks, 4 heads, n_embd 128, 5000 steps |

## Reference baselines

| Baseline | Loss (nats/char) |
| --- | --- |
| Uniform guess over 65 characters | 4.1744 |
| Character frequency only | 3.3128 |
