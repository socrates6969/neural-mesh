## Fitted cost model (p50 us = a + b N, single thread, rows from results/h6_gate_grid_*.json)
- flat f32                                     RAM B/vec   1536  p50 @1e4/1e5/1e6 = 466/5136/65836  recall = 1.000/1.000/1.000  model p50(1e7) = 665202 us (EXTRAPOLATED)
- flat block-fp16                              RAM B/vec    816  p50 @1e4/1e5/1e6 = 298/3699/43614  recall = 1.000/1.000/1.000  model p50(1e7) = 439546 us (EXTRAPOLATED)
- int8 + f32 rerank 4x (f32 RAM)               RAM B/vec   1924  p50 @1e4/1e5/1e6 = 150/1588/18114  recall = 1.000/1.000/1.000  model p50(1e7) = 182205 us (EXTRAPOLATED)
- int8 + f32 rerank 4x (f32 disk)              RAM B/vec    388  p50 @1e4/1e5/1e6 = 4425/6145/24334  recall = 1.000/1.000/1.000  model p50(1e7) = 205684 us (EXTRAPOLATED)
- turbovec-4bit + f32 rerank 4x (f32 RAM)      RAM B/vec   1732  p50 @1e4/1e5/1e6 = 241/1107/14038  recall = 1.000/1.000/1.000  model p50(1e7) = 141021 us (EXTRAPOLATED)
- turbovec-4bit + f32 rerank 4x (f32 disk)     RAM B/vec    196  p50 @1e4/1e5/1e6 = 4635/5266/25127  recall = 1.000/1.000/1.000  model p50(1e7) = 216351 us (EXTRAPOLATED)
- turbovec-2bit + f32 rerank 16x (f32 RAM)     RAM B/vec   1636  p50 @1e4/1e5/1e6 = 272/1283/10167  recall = 1.000/1.000/0.972  model p50(1e7) = 99672 us (EXTRAPOLATED)
- turbovec-2bit + f32 rerank 16x (f32 disk)    RAM B/vec    100  p50 @1e4/1e5/1e6 = 17192/20810/35256  recall = 1.000/1.000/0.972  model p50(1e7) = 191569 us (EXTRAPOLATED)
- RaBitQ-1bit + f32 rerank 32x (f32 RAM)       RAM B/vec   1604  p50 @1e4/1e5/1e6 = 613/4564/45531  recall = 1.000/1.000/0.927  model p50(1e7) = 454407 us (EXTRAPOLATED)
- RaBitQ-1bit + f32 rerank 32x (f32 disk)      RAM B/vec     68  p50 @1e4/1e5/1e6 = 34618/39612/108214  recall = 1.000/1.000/0.927  model p50(1e7) = 784063 us (EXTRAPOLATED)

## Rule output: least-compressed backend with RAM <= budget, modelled p50 <= budget, recall >= 0.99 (f32 copy of rerank formats is ON DISK; its fetch time is inside p50)

| N | latency budget | RAM 128 MB | RAM 512 MB | RAM 2 GB |
|---|---|---|---|---|
| 1e3 | 1 ms | flat f32 | flat f32 | flat f32 |
| 1e3 | 5 ms | flat f32 | flat f32 | flat f32 |
| 1e3 | 20 ms | flat f32 | flat f32 | flat f32 |
| 1e4 | 1 ms | flat f32 | flat f32 | flat f32 |
| 1e4 | 5 ms | flat f32 | flat f32 | flat f32 |
| 1e4 | 20 ms | flat f32 | flat f32 | flat f32 |
| 3e4 | 1 ms | flat block-fp16 | flat block-fp16 | flat block-fp16 |
| 3e4 | 5 ms | flat f32 | flat f32 | flat f32 |
| 3e4 | 20 ms | flat f32 | flat f32 | flat f32 |
| 1e5 | 1 ms | NONE (needs ANN graph / more RAM / more latency) | NONE (needs ANN graph / more RAM / more latency) | NONE (needs ANN graph / more RAM / more latency) |
| 1e5 | 5 ms | flat block-fp16 | flat block-fp16 | flat block-fp16 |
| 1e5 | 20 ms | flat block-fp16 | flat f32 | flat f32 |
| 3e5 | 1 ms | NONE (needs ANN graph / more RAM / more latency) | NONE (needs ANN graph / more RAM / more latency) | NONE (needs ANN graph / more RAM / more latency) |
| 3e5 | 5 ms | NONE (needs ANN graph / more RAM / more latency) | turbovec-2bit + f32 rerank 16x (f32 RAM) | turbovec-4bit + f32 rerank 4x (f32 RAM) |
| 3e5 | 20 ms | int8 + f32 rerank 4x (f32 disk) | flat f32 | flat f32 |
| 1e6 | 1 ms | NONE (needs ANN graph / more RAM / more latency) | NONE (needs ANN graph / more RAM / more latency) | NONE (needs ANN graph / more RAM / more latency) |
| 1e6 | 5 ms | NONE (needs ANN graph / more RAM / more latency) | NONE (needs ANN graph / more RAM / more latency) | NONE (needs ANN graph / more RAM / more latency) |
| 1e6 | 20 ms | NONE (needs ANN graph / more RAM / more latency) | NONE (needs ANN graph / more RAM / more latency) | int8 + f32 rerank 4x (f32 RAM) |
| 3e6 (extrap.) | 1 ms | NONE (needs ANN graph / more RAM / more latency) | NONE (needs ANN graph / more RAM / more latency) | NONE (needs ANN graph / more RAM / more latency) |
| 3e6 (extrap.) | 5 ms | NONE (needs ANN graph / more RAM / more latency) | NONE (needs ANN graph / more RAM / more latency) | NONE (needs ANN graph / more RAM / more latency) |
| 3e6 (extrap.) | 20 ms | NONE (needs ANN graph / more RAM / more latency) | NONE (needs ANN graph / more RAM / more latency) | NONE (needs ANN graph / more RAM / more latency) |
| 1e7 (extrap.) | 1 ms | NONE (needs ANN graph / more RAM / more latency) | NONE (needs ANN graph / more RAM / more latency) | NONE (needs ANN graph / more RAM / more latency) |
| 1e7 (extrap.) | 5 ms | NONE (needs ANN graph / more RAM / more latency) | NONE (needs ANN graph / more RAM / more latency) | NONE (needs ANN graph / more RAM / more latency) |
| 1e7 (extrap.) | 20 ms | NONE (needs ANN graph / more RAM / more latency) | NONE (needs ANN graph / more RAM / more latency) | NONE (needs ANN graph / more RAM / more latency) |

## Validation: pick vs. measured grid (N = 1e4, 1e5, 1e6; measured p50/recall/RAM of the picked row; agreement with 'least-compressed measured-feasible')

| N | lat budget | RAM budget | rule pick | measured p50 us | measured recall | feasible by measurement | measured-least-compressed | agree |
|---|---|---|---|---|---|---|---|---|
| 1e4 | 1 ms | 128 MB | flat f32 | 466 | 1.000 | true | flat f32 | true |
| 1e4 | 1 ms | 512 MB | flat f32 | 466 | 1.000 | true | flat f32 | true |
| 1e4 | 1 ms | 2 GB | flat f32 | 466 | 1.000 | true | flat f32 | true |
| 1e4 | 5 ms | 128 MB | flat f32 | 466 | 1.000 | true | flat f32 | true |
| 1e4 | 5 ms | 512 MB | flat f32 | 466 | 1.000 | true | flat f32 | true |
| 1e4 | 5 ms | 2 GB | flat f32 | 466 | 1.000 | true | flat f32 | true |
| 1e4 | 20 ms | 128 MB | flat f32 | 466 | 1.000 | true | flat f32 | true |
| 1e4 | 20 ms | 512 MB | flat f32 | 466 | 1.000 | true | flat f32 | true |
| 1e4 | 20 ms | 2 GB | flat f32 | 466 | 1.000 | true | flat f32 | true |
| 1e5 | 1 ms | 128 MB | NONE | - | - | - | NONE | true |
| 1e5 | 1 ms | 512 MB | NONE | - | - | - | NONE | true |
| 1e5 | 1 ms | 2 GB | NONE | - | - | - | NONE | true |
| 1e5 | 5 ms | 128 MB | flat block-fp16 | 3699 | 1.000 | true | flat block-fp16 | true |
| 1e5 | 5 ms | 512 MB | flat block-fp16 | 3699 | 1.000 | true | flat block-fp16 | true |
| 1e5 | 5 ms | 2 GB | flat block-fp16 | 3699 | 1.000 | true | flat block-fp16 | true |
| 1e5 | 20 ms | 128 MB | flat block-fp16 | 3699 | 1.000 | true | flat block-fp16 | true |
| 1e5 | 20 ms | 512 MB | flat f32 | 5136 | 1.000 | true | flat f32 | true |
| 1e5 | 20 ms | 2 GB | flat f32 | 5136 | 1.000 | true | flat f32 | true |
| 1e6 | 1 ms | 128 MB | NONE | - | - | - | NONE | true |
| 1e6 | 1 ms | 512 MB | NONE | - | - | - | NONE | true |
| 1e6 | 1 ms | 2 GB | NONE | - | - | - | NONE | true |
| 1e6 | 5 ms | 128 MB | NONE | - | - | - | NONE | true |
| 1e6 | 5 ms | 512 MB | NONE | - | - | - | NONE | true |
| 1e6 | 5 ms | 2 GB | NONE | - | - | - | NONE | true |
| 1e6 | 20 ms | 128 MB | NONE | - | - | - | NONE | true |
| 1e6 | 20 ms | 512 MB | NONE | - | - | - | NONE | true |
| 1e6 | 20 ms | 2 GB | int8 + f32 rerank 4x (f32 RAM) | 18114 | 1.000 | true | int8 + f32 rerank 4x (f32 RAM) | true |

Rule vs measured-least-compressed agreement: 27/27 cells
