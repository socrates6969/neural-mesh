
#### synth384-n30000 (n=30000, dim=384, nq=300) - fixed crate, ONE build per cell, p50/p99 = mean of 3 passes
Pareto frontier (recall up, p99 down):
| recall@10 | p50 us | p99 us | M | efc | ef | heap B/vec | build s |
|---|---|---|---|---|---|---|---|
| 0.983 | 141 | 245 | 24 | 200 | 10 | 6697 | 29.8 |
| 0.988 | 147 | 279 | 48 | 100 | 16 | 7268 | 7.7 |
| 0.998 | 177 | 299 | 24 | 200 | 16 | 6697 | 29.8 |
| 1.000 | 184 | 343 | 48 | 100 | 24 | 7268 | 7.7 |

Best recall at fixed p99 budget:
| p99 budget us | best recall | config (M/efc/ef) | p50 | p99 |
|---|---|---|---|---|
| 250 | 0.983 | 24/200/10 | 141 | 245 |
| 400 | 1.000 | 48/100/24 | 184 | 343 |
| 800 | 1.000 | 48/100/24 | 184 | 343 |
| 1500 | 1.000 | 48/100/24 | 184 | 343 |

Minimum p99 at recall target:
| recall >= | min p99 us | config | p50 | recall |
|---|---|---|---|---|
| 0.9 | 245 | 24/200/10 | 141 | 0.983 |
| 0.95 | 245 | 24/200/10 | 141 | 0.983 |
| 0.98 | 245 | 24/200/10 | 141 | 0.983 |
| 0.99 | 299 | 24/200/16 | 177 | 0.998 |
| 0.995 | 299 | 24/200/16 | 177 | 0.998 |

Self-recall (own vector as query, ef=64, sample): M16/efc100: 8/15000 not top-1, 8 absent from top-10; M16/efc200: 0/15000 not top-1, 0 absent from top-10; M16/efc400: 0/15000 not top-1, 0 absent from top-10; M24/efc100: 1/15000 not top-1, 1 absent from top-10; M24/efc200: 0/15000 not top-1, 0 absent from top-10; M24/efc400: 0/15000 not top-1, 0 absent from top-10; M32/efc100: 0/15000 not top-1, 0 absent from top-10; M32/efc200: 0/15000 not top-1, 0 absent from top-10; M32/efc400: 0/15000 not top-1, 0 absent from top-10; M48/efc100: 0/15000 not top-1, 0 absent from top-10; M48/efc200: 0/15000 not top-1, 0 absent from top-10; M48/efc400: 0/15000 not top-1, 0 absent from top-10; M64/efc100: 3447/15000 not top-1, 3447 absent from top-10; M64/efc200: 0/15000 not top-1, 0 absent from top-10; M64/efc400: 0/15000 not top-1, 0 absent from top-10

recall at ef=64, efc=200 by M: M16: 1.000, M24: 1.000, M32: 1.000, M48: 1.000, M64: 1.000

#### real-mini (n=23153, dim=384, nq=300) - fixed crate, ONE build per cell, p50/p99 = mean of 3 passes
Pareto frontier (recall up, p99 down):
| recall@10 | p50 us | p99 us | M | efc | ef | heap B/vec | build s |
|---|---|---|---|---|---|---|---|
| 0.978 | 58 | 105 | 48 | 100 | 10 | 6981 | 2.3 |
| 0.989 | 73 | 145 | 32 | 200 | 10 | 6754 | 6.7 |
| 0.994 | 79 | 158 | 32 | 100 | 16 | 6531 | 2.7 |
| 0.995 | 134 | 218 | 24 | 400 | 16 | 6611 | 20.5 |
| 0.996 | 128 | 219 | 24 | 100 | 24 | 6255 | 3.2 |
| 0.999 | 149 | 241 | 32 | 100 | 32 | 6531 | 2.7 |
| 1.000 | 178 | 303 | 32 | 100 | 48 | 6531 | 2.7 |
| 1.000 | 243 | 425 | 64 | 400 | 48 | 7630 | 17.3 |

Best recall at fixed p99 budget:
| p99 budget us | best recall | config (M/efc/ef) | p50 | p99 |
|---|---|---|---|---|
| 150 | 0.989 | 32/200/10 | 73 | 145 |
| 250 | 0.999 | 32/100/32 | 149 | 241 |
| 400 | 1.000 | 32/100/48 | 178 | 303 |
| 800 | 1.000 | 64/400/48 | 243 | 425 |
| 1500 | 1.000 | 64/400/48 | 243 | 425 |

Minimum p99 at recall target:
| recall >= | min p99 us | config | p50 | recall |
|---|---|---|---|---|
| 0.9 | 105 | 48/100/10 | 58 | 0.978 |
| 0.95 | 105 | 48/100/10 | 58 | 0.978 |
| 0.98 | 145 | 32/200/10 | 73 | 0.989 |
| 0.99 | 158 | 32/100/16 | 79 | 0.994 |
| 0.995 | 219 | 24/100/24 | 128 | 0.996 |

Self-recall (own vector as query, ef=64, sample): M16/efc100: 5/23153 not top-1, 5 absent from top-10; M16/efc200: 0/23153 not top-1, 0 absent from top-10; M16/efc400: 0/23153 not top-1, 0 absent from top-10; M24/efc100: 2/23153 not top-1, 2 absent from top-10; M24/efc200: 0/23153 not top-1, 0 absent from top-10; M24/efc400: 0/23153 not top-1, 0 absent from top-10; M32/efc100: 5/23153 not top-1, 5 absent from top-10; M32/efc200: 0/23153 not top-1, 0 absent from top-10; M32/efc400: 0/23153 not top-1, 0 absent from top-10; M48/efc100: 3/23153 not top-1, 3 absent from top-10; M48/efc200: 0/23153 not top-1, 0 absent from top-10; M48/efc400: 0/23153 not top-1, 0 absent from top-10; M64/efc100: 286/23153 not top-1, 286 absent from top-10; M64/efc200: 0/23153 not top-1, 0 absent from top-10; M64/efc400: 0/23153 not top-1, 0 absent from top-10

recall at ef=64, efc=200 by M: M16: 1.000, M24: 1.000, M32: 0.999, M48: 1.000, M64: 0.999

#### real-bge (n=23453, dim=768, nq=300) - fixed crate, ONE build per cell, p50/p99 = mean of 3 passes
Pareto frontier (recall up, p99 down):
| recall@10 | p50 us | p99 us | M | efc | ef | heap B/vec | build s |
|---|---|---|---|---|---|---|---|
| 0.970 | 81 | 178 | 16 | 100 | 10 | 7399 | 3.4 |
| 0.995 | 106 | 189 | 64 | 400 | 10 | 9184 | 38.3 |
| 0.999 | 143 | 247 | 64 | 400 | 16 | 9184 | 38.3 |
| 0.999 | 139 | 277 | 24 | 100 | 16 | 7804 | 5.1 |
| 0.999 | 196 | 322 | 16 | 400 | 24 | 7661 | 27.3 |
| 1.000 | 217 | 355 | 24 | 100 | 32 | 7804 | 5.1 |

Best recall at fixed p99 budget:
| p99 budget us | best recall | config (M/efc/ef) | p50 | p99 |
|---|---|---|---|---|
| 250 | 0.999 | 64/400/16 | 143 | 247 |
| 400 | 1.000 | 24/100/32 | 217 | 355 |
| 800 | 1.000 | 24/100/32 | 217 | 355 |
| 1500 | 1.000 | 24/100/32 | 217 | 355 |

Minimum p99 at recall target:
| recall >= | min p99 us | config | p50 | recall |
|---|---|---|---|---|
| 0.9 | 178 | 16/100/10 | 81 | 0.970 |
| 0.95 | 178 | 16/100/10 | 81 | 0.970 |
| 0.98 | 189 | 64/400/10 | 106 | 0.995 |
| 0.99 | 189 | 64/400/10 | 106 | 0.995 |
| 0.995 | 247 | 64/400/16 | 143 | 0.999 |

Self-recall (own vector as query, ef=64, sample): M16/efc100: 147/23453 not top-1, 147 absent from top-10; M16/efc200: 31/23453 not top-1, 31 absent from top-10; M16/efc400: 13/23453 not top-1, 13 absent from top-10; M24/efc100: 26/23453 not top-1, 26 absent from top-10; M24/efc200: 10/23453 not top-1, 10 absent from top-10; M24/efc400: 8/23453 not top-1, 8 absent from top-10; M32/efc100: 27/23453 not top-1, 27 absent from top-10; M32/efc200: 4/23453 not top-1, 4 absent from top-10; M32/efc400: 6/23453 not top-1, 6 absent from top-10; M48/efc100: 13/23453 not top-1, 13 absent from top-10; M48/efc200: 1/23453 not top-1, 1 absent from top-10; M48/efc400: 1/23453 not top-1, 1 absent from top-10; M64/efc100: 523/23453 not top-1, 523 absent from top-10; M64/efc200: 0/23453 not top-1, 0 absent from top-10; M64/efc400: 0/23453 not top-1, 0 absent from top-10

recall at ef=64, efc=200 by M: M16: 1.000, M24: 1.000, M32: 1.000, M48: 1.000, M64: 1.000
