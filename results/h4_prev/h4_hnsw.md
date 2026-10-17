# h4 memory-reduction stage `hnsw` (synthetic-until-validated; n_seeds=10)

Margin pre-declared: +-0.01 success (strict tier +-0.005); reference = full (unbounded) memory, paired by seed. EQ = TOST p<0.05 (Holm-adjusted over the main-world family; controls use raw p). bytes = final resident estimate / full at D=64; proj384 = same retained set priced at D=384. lat = mean search time vs full (parallel runs, noisy).

## A-zipf  (oracle success ceiling on eval window 0.711)

| cell | success | dSucc vs full [95% CI] | dUseful | bytes | proj384 | items | lat | p(.01) | EQ.01 | EQ.005 | NI.01 | ovr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 0.5643 | +0.0000 [+0.0000,+0.0000] | +0.0000 | 1.000 | 1.000 | 1.000 | 1.00 | 0.0000 | EQ | EQ | NI | 0 |
| value@25% | 0.5717 | +0.0074 [-0.0039,+0.0186] | -0.0627 | 0.248 | 0.248 | 0.248 | 0.49 | 0.3055 | no | no | NI | 0 |
| value-hnsw@25% | 0.5176 | -0.0467 [-0.0583,-0.0351] | -0.2763 | 0.250 | 0.032 | 0.032 | 1.19 | 1.0000 | no | no | no | 0 |
| full-hnsw | 0.5643 | +0.0000 [+0.0000,+0.0000] | +0.0000 | 1.000 | 1.000 | 1.000 | 1.00 | 0.0000 | EQ | EQ | NI | 0 |

