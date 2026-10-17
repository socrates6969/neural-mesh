
### real-bge [memory mode] (n=23453, dim=768)

(A) RAM-only 3-D Pareto (RAM B/vec, recall, p99):
| config | RAM B/vec | SSD B/vec | M vec/GB RAM | recall@10 | p50 us | p99 us |
|---|---|---|---|---|---|---|
| turbovec 2-bit | 197 | 0 | 5.08 | 0.931 | 130 | 207 |
| turbovec 3-bit | 389 | 0 | 2.57 | 0.958 | 234 | 395 |
| turbovec 4-bit | 389 | 0 | 2.57 | 0.976 | 659 | 722 |
| rabitq-style 5-bit | 585 | 0 | 1.71 | 0.987 | 1632 | 2072 |
| rabitq-style 6-bit | 681 | 0 | 1.47 | 0.992 | 1677 | 2397 |
| flat int8(per-vector scale) | 772 | 0 | 1.30 | 0.993 | 953 | 1059 |
| flat f16 | 1536 | 0 | 0.65 | 1.000 | 1489 | 1799 |
| flat blockfp16(32,i16+f32 scale) | 1632 | 0 | 0.61 | 1.000 | 1394 | 1717 |
| turbovec 2-bit + rerank4x [RAM] | 3269 | 0 | 0.31 | 1.000 | 170 | 217 |
| HNSW M32 efc100 ef24 [int8 stored] | 5874 | 0 | 0.17 | 0.993 | 95 | 164 |
| HNSW M48 efc200 ef16 [int8 stored] | 6405 | 0 | 0.16 | 0.993 | 75 | 133 |
| HNSW M48 efc200 ef16 [bfp16 stored] | 7268 | 0 | 0.14 | 0.999 | 91 | 172 |

(B) with f32 rerank rows on SSD (cold direct I/O) 3-D Pareto:
| config | RAM B/vec | SSD B/vec | M vec/GB RAM | recall@10 | p50 us | p99 us |
|---|---|---|---|---|---|---|
| turbovec 2-bit | 197 | 0 | 5.08 | 0.931 | 130 | 207 |
| turbovec 2-bit + rerank4x [SSD direct] | 197 | 3072 | 5.08 | 1.000 | 4524 | 6546 |
| turbovec 3-bit | 389 | 0 | 2.57 | 0.958 | 234 | 395 |
| turbovec 3-bit + rerank4x [SSD direct] | 389 | 3072 | 2.57 | 1.000 | 4080 | 5293 |
| turbovec 4-bit | 389 | 0 | 2.57 | 0.976 | 659 | 722 |
| turbovec 4-bit + rerank2x [SSD direct] | 389 | 3072 | 2.57 | 1.000 | 2455 | 2804 |
| rabitq-style 5-bit | 585 | 0 | 1.71 | 0.987 | 1632 | 2072 |
| rabitq-style 6-bit | 681 | 0 | 1.47 | 0.992 | 1677 | 2397 |
| flat int8(per-vector scale) | 772 | 0 | 1.30 | 0.993 | 953 | 1059 |
| flat f16 | 1536 | 0 | 0.65 | 1.000 | 1489 | 1799 |
| flat blockfp16(32,i16+f32 scale) | 1632 | 0 | 0.61 | 1.000 | 1394 | 1717 |
| turbovec 2-bit + rerank4x [RAM] | 3269 | 0 | 0.31 | 1.000 | 170 | 217 |
| HNSW M32 efc100 ef24 [int8 stored] | 5874 | 0 | 0.17 | 0.993 | 95 | 164 |
| HNSW M48 efc200 ef16 [int8 stored] | 6405 | 0 | 0.16 | 0.993 | 75 | 133 |
| HNSW M48 efc200 ef16 [bfp16 stored] | 7268 | 0 | 0.14 | 0.999 | 91 | 172 |

### real-mini [memory mode] (n=23153, dim=384)

(A) RAM-only 3-D Pareto (RAM B/vec, recall, p99):
| config | RAM B/vec | SSD B/vec | M vec/GB RAM | recall@10 | p50 us | p99 us |
|---|---|---|---|---|---|---|
| turbovec 2-bit | 100 | 0 | 9.96 | 0.924 | 84 | 303 |
| rabitq-style 2-bit | 125 | 0 | 7.97 | 0.929 | 447 | 1103 |
| rabitq-style 3-bit | 173 | 0 | 5.76 | 0.959 | 648 | 683 |
| turbovec 3-bit | 196 | 0 | 5.09 | 0.960 | 125 | 152 |
| turbovec 4-bit | 196 | 0 | 5.09 | 0.977 | 341 | 380 |
| rabitq-style 5-bit | 269 | 0 | 3.71 | 0.987 | 954 | 1065 |
| rabitq-style 6-bit | 317 | 0 | 3.15 | 0.992 | 1188 | 2391 |
| flat int8(per-vector scale) | 388 | 0 | 2.58 | 0.997 | 560 | 2584 |
| flat f16 | 768 | 0 | 1.30 | 1.000 | 803 | 2386 |
| flat blockfp16(32,i16+f32 scale) | 816 | 0 | 1.23 | 1.000 | 1070 | 2703 |
| rabitq-style 1-bit + rerank8x [RAM] | 1613 | 0 | 0.62 | 1.000 | 292 | 416 |
| turbovec 2-bit + rerank8x [RAM] | 1636 | 0 | 0.61 | 1.000 | 148 | 206 |
| turbovec 3-bit + rerank2x [RAM] | 1732 | 0 | 0.58 | 1.000 | 140 | 171 |
| HNSW M32 efc100 ef24 [int8 stored] | 5432 | 0 | 0.18 | 0.988 | 103 | 168 |
| HNSW M48 efc200 ef16 [int8 stored] | 6003 | 0 | 0.17 | 0.992 | 75 | 146 |

(B) with f32 rerank rows on SSD (cold direct I/O) 3-D Pareto:
| config | RAM B/vec | SSD B/vec | M vec/GB RAM | recall@10 | p50 us | p99 us |
|---|---|---|---|---|---|---|
| rabitq-style 1-bit + rerank8x [SSD direct] | 77 | 1536 | 12.91 | 1.000 | 6805 | 9336 |
| turbovec 2-bit | 100 | 0 | 9.96 | 0.924 | 84 | 303 |
| rabitq-style 2-bit | 125 | 0 | 7.97 | 0.929 | 447 | 1103 |
| rabitq-style 2-bit + rerank4x [SSD direct] | 125 | 1536 | 7.97 | 1.000 | 4339 | 7649 |
| rabitq-style 3-bit | 173 | 0 | 5.76 | 0.959 | 648 | 683 |
| turbovec 3-bit | 196 | 0 | 5.09 | 0.960 | 125 | 152 |
| turbovec 4-bit | 196 | 0 | 5.09 | 0.977 | 341 | 380 |
| turbovec 4-bit + rerank2x [SSD direct] | 196 | 1536 | 5.09 | 1.000 | 2019 | 2941 |
| rabitq-style 5-bit | 269 | 0 | 3.71 | 0.987 | 954 | 1065 |
| rabitq-style 6-bit | 317 | 0 | 3.15 | 0.992 | 1188 | 2391 |
| flat int8(per-vector scale) | 388 | 0 | 2.58 | 0.997 | 560 | 2584 |
| flat f16 | 768 | 0 | 1.30 | 1.000 | 803 | 2386 |
| flat blockfp16(32,i16+f32 scale) | 816 | 0 | 1.23 | 1.000 | 1070 | 2703 |
| rabitq-style 1-bit + rerank8x [RAM] | 1613 | 0 | 0.62 | 1.000 | 292 | 416 |
| turbovec 2-bit + rerank8x [RAM] | 1636 | 0 | 0.61 | 1.000 | 148 | 206 |
| turbovec 3-bit + rerank2x [RAM] | 1732 | 0 | 0.58 | 1.000 | 140 | 171 |
| HNSW M32 efc100 ef24 [int8 stored] | 5432 | 0 | 0.18 | 0.988 | 103 | 168 |
| HNSW M48 efc200 ef16 [int8 stored] | 6003 | 0 | 0.17 | 0.992 | 75 | 146 |

### synth384-n100000 [memory mode] (n=100000, dim=384)

(A) RAM-only 3-D Pareto (RAM B/vec, recall, p99):
| config | RAM B/vec | SSD B/vec | M vec/GB RAM | recall@10 | p50 us | p99 us |
|---|---|---|---|---|---|---|
| rabitq-style 5-bit | 250 | 0 | 4.00 | 0.908 | 4372 | 6225 |
| rabitq-style 6-bit | 298 | 0 | 3.36 | 0.950 | 4544 | 7496 |
| flat int8(per-vector scale) | 388 | 0 | 2.58 | 0.976 | 2100 | 2235 |
| flat f16 | 768 | 0 | 1.30 | 0.999 | 3004 | 3144 |
| flat blockfp16(32,i16+f32 scale) | 816 | 0 | 1.23 | 1.000 | 2540 | 3240 |
| rabitq-style 1-bit + rerank8x [RAM] | 1594 | 0 | 0.63 | 0.987 | 1003 | 1093 |
| rabitq-style 1-bit + rerank16x [RAM] | 1594 | 0 | 0.63 | 1.000 | 1561 | 1630 |
| turbovec 2-bit + rerank8x [RAM] | 1636 | 0 | 0.61 | 1.000 | 274 | 507 |
| HNSW M32 efc100 ef24 [int8 stored] | 5684 | 0 | 0.18 | 0.927 | 152 | 329 |
| HNSW M32 efc100 ef24 [bfp16 stored] | 6130 | 0 | 0.16 | 0.942 | 109 | 240 |
| HNSW M48 efc200 ef16 [int8 stored] | 6277 | 0 | 0.16 | 0.945 | 95 | 175 |
| HNSW M48 efc200 ef16 [bfp16 stored] | 6707 | 0 | 0.15 | 0.969 | 144 | 285 |

(B) with f32 rerank rows on SSD (cold direct I/O) 3-D Pareto:
| config | RAM B/vec | SSD B/vec | M vec/GB RAM | recall@10 | p50 us | p99 us |
|---|---|---|---|---|---|---|
| rabitq-style 1-bit + rerank8x [SSD direct] | 58 | 1536 | 17.27 | 0.987 | 7202 | 8128 |
| turbovec 2-bit + rerank4x [SSD direct] | 100 | 1536 | 9.99 | 0.957 | 3762 | 4543 |
| turbovec 2-bit + rerank8x [SSD direct] | 100 | 1536 | 9.99 | 1.000 | 6656 | 8205 |
| rabitq-style 2-bit + rerank4x [SSD direct] | 106 | 1536 | 9.44 | 0.958 | 4605 | 6119 |
| rabitq-style 3-bit + rerank4x [SSD direct] | 154 | 1536 | 6.50 | 0.999 | 5116 | 7080 |
| turbovec 3-bit + rerank4x [SSD direct] | 196 | 1536 | 5.10 | 0.998 | 4165 | 4939 |
| turbovec 4-bit + rerank2x [SSD direct] | 196 | 1536 | 5.10 | 0.995 | 2506 | 2926 |
| turbovec 4-bit + rerank4x [SSD direct] | 196 | 1536 | 5.10 | 1.000 | 4415 | 5082 |
| flat int8(per-vector scale) | 388 | 0 | 2.58 | 0.976 | 2100 | 2235 |
| flat f16 | 768 | 0 | 1.30 | 0.999 | 3004 | 3144 |
| flat blockfp16(32,i16+f32 scale) | 816 | 0 | 1.23 | 1.000 | 2540 | 3240 |
| rabitq-style 1-bit + rerank8x [RAM] | 1594 | 0 | 0.63 | 0.987 | 1003 | 1093 |
| rabitq-style 1-bit + rerank16x [RAM] | 1594 | 0 | 0.63 | 1.000 | 1561 | 1630 |
| turbovec 2-bit + rerank8x [RAM] | 1636 | 0 | 0.61 | 1.000 | 274 | 507 |
| HNSW M32 efc100 ef24 [int8 stored] | 5684 | 0 | 0.18 | 0.927 | 152 | 329 |
| HNSW M32 efc100 ef24 [bfp16 stored] | 6130 | 0 | 0.16 | 0.942 | 109 | 240 |
| HNSW M48 efc200 ef16 [int8 stored] | 6277 | 0 | 0.16 | 0.945 | 95 | 175 |
| HNSW M48 efc200 ef16 [bfp16 stored] | 6707 | 0 | 0.15 | 0.969 | 144 | 285 |

### synth384-n100000 [scale mode, streamed] (n=100000, dim=384)

(A) RAM-only 3-D Pareto (RAM B/vec, recall, p99):
| config | RAM B/vec | SSD B/vec | M vec/GB RAM | recall@10 | p50 us | p99 us |
|---|---|---|---|---|---|---|
| flat int8(per-vector scale) | 388 | 0 | 2.58 | 0.974 | 1122 | 1819 |
| flat blockfp16(32,i16+f32 scale) | 816 | 0 | 1.23 | 1.000 | 2705 | 3808 |
| turbovec 4-bit + rerank4x [RAM f32] | 1732 | 0 | 0.58 | 1.000 | 928 | 1470 |
| HNSW M32 efc100 ef32 [int8 stored] | 5703 | 0 | 0.18 | 0.949 | 120 | 235 |
| HNSW M32 efc100 ef64 [int8 stored] | 5703 | 0 | 0.18 | 0.969 | 184 | 381 |
| HNSW M32 efc100 ef256 [int8 stored] | 5703 | 0 | 0.18 | 0.974 | 619 | 972 |
| HNSW M32 efc100 ef32 [bfp16 stored] | 6117 | 0 | 0.16 | 0.989 | 135 | 310 |
| HNSW M32 efc100 ef64 [bfp16 stored] | 6117 | 0 | 0.16 | 0.990 | 208 | 394 |
| HNSW M32 efc100 ef128 [bfp16 stored] | 6117 | 0 | 0.16 | 1.000 | 368 | 576 |
| HNSW M48 efc100 ef32 [f32lib stored] | 7267 | 0 | 0.14 | 0.995 | 192 | 378 |

(B) with f32 rerank rows on SSD (cold direct I/O) 3-D Pareto:
| config | RAM B/vec | SSD B/vec | M vec/GB RAM | recall@10 | p50 us | p99 us |
|---|---|---|---|---|---|---|
| rabitq-style 1-bit + rerank16x [SSD direct] | 58 | 1536 | 17.27 | 1.000 | 16357 | 52625 |
| rabitq-style 2-bit + rerank8x [SSD direct] | 106 | 1536 | 9.44 | 1.000 | 9834 | 51251 |
| turbovec 2-bit + rerank4x [SSD direct] | 134 | 1536 | 7.48 | 0.957 | 3398 | 4171 |
| turbovec 2-bit + rerank16x [SSD direct] | 134 | 1536 | 7.48 | 1.000 | 12546 | 15679 |
| flat int8(per-vector scale) | 388 | 0 | 2.58 | 0.974 | 1122 | 1819 |
| flat int8 + rerank4x [SSD direct] | 388 | 1536 | 2.58 | 1.000 | 4458 | 5829 |
| flat blockfp16(32,i16+f32 scale) | 816 | 0 | 1.23 | 1.000 | 2705 | 3808 |
| turbovec 4-bit + rerank4x [RAM f32] | 1732 | 0 | 0.58 | 1.000 | 928 | 1470 |
| HNSW M32 efc100 ef32 [int8 stored] | 5703 | 0 | 0.18 | 0.949 | 120 | 235 |
| HNSW M32 efc100 ef64 [int8 stored] | 5703 | 0 | 0.18 | 0.969 | 184 | 381 |
| HNSW M32 efc100 ef256 [int8 stored] | 5703 | 0 | 0.18 | 0.974 | 619 | 972 |
| HNSW M32 efc100 ef32 [bfp16 stored] | 6117 | 0 | 0.16 | 0.989 | 135 | 310 |
| HNSW M32 efc100 ef64 [bfp16 stored] | 6117 | 0 | 0.16 | 0.990 | 208 | 394 |
| HNSW M32 efc100 ef128 [bfp16 stored] | 6117 | 0 | 0.16 | 1.000 | 368 | 576 |
| HNSW M48 efc100 ef32 [f32lib stored] | 7267 | 0 | 0.14 | 0.995 | 192 | 378 |

### synth384-n300000 [scale mode, streamed] (n=300000, dim=384)

(A) RAM-only 3-D Pareto (RAM B/vec, recall, p99):
| config | RAM B/vec | SSD B/vec | M vec/GB RAM | recall@10 | p50 us | p99 us |
|---|---|---|---|---|---|---|
| flat int8(per-vector scale) | 388 | 0 | 2.58 | 0.964 | 5681 | 22578 |
| flat blockfp16(32,i16+f32 scale) | 816 | 0 | 1.23 | 1.000 | 9234 | 10755 |
| turbovec 4-bit + rerank4x [RAM f32] | 1732 | 0 | 0.58 | 1.000 | 2944 | 3774 |
| HNSW M32 efc100 ef128 [int8 stored] | 5719 | 0 | 0.17 | 0.944 | 328 | 811 |
| HNSW M32 efc100 ef256 [int8 stored] | 5719 | 0 | 0.17 | 0.958 | 499 | 1192 |
| HNSW M32 efc100 ef128 [bfp16 stored] | 6182 | 0 | 0.16 | 0.957 | 377 | 983 |
| HNSW M32 efc100 ef256 [bfp16 stored] | 6182 | 0 | 0.16 | 0.987 | 545 | 1346 |
| HNSW M32 efc100 ef128 [f32lib stored] | 6879 | 0 | 0.15 | 0.958 | 447 | 1140 |
| HNSW M32 efc100 ef256 [f32lib stored] | 6879 | 0 | 0.15 | 0.993 | 662 | 1601 |
| HNSW M48 efc100 ef64 [f32lib stored] | 7323 | 0 | 0.14 | 0.974 | 332 | 730 |
| HNSW M48 efc100 ef128 [f32lib stored] | 7323 | 0 | 0.14 | 0.990 | 456 | 918 |
| HNSW M48 efc100 ef256 [f32lib stored] | 7323 | 0 | 0.14 | 1.000 | 673 | 1237 |

(B) with f32 rerank rows on SSD (cold direct I/O) 3-D Pareto:
| config | RAM B/vec | SSD B/vec | M vec/GB RAM | recall@10 | p50 us | p99 us |
|---|---|---|---|---|---|---|
| rabitq-style 1-bit + rerank16x [SSD direct] | 54 | 1536 | 18.53 | 0.953 | 15590 | 26680 |
| rabitq-style 2-bit + rerank8x [SSD direct] | 102 | 1536 | 9.81 | 0.961 | 11910 | 13200 |
| turbovec 2-bit + rerank16x [SSD direct] | 116 | 1536 | 8.64 | 0.999 | 14386 | 16945 |
| rabitq-style 3-bit + rerank4x [SSD direct] | 150 | 1536 | 6.67 | 0.979 | 10917 | 12274 |
| rabitq-style 4-bit + rerank4x [SSD direct] | 198 | 1536 | 5.05 | 1.000 | 11310 | 12825 |
| turbovec 4-bit + rerank4x [SSD direct] | 217 | 1536 | 4.61 | 1.000 | 5705 | 7056 |
| flat int8 + rerank4x [SSD direct] | 388 | 1536 | 2.58 | 1.000 | 8768 | 25991 |
| flat blockfp16(32,i16+f32 scale) | 816 | 0 | 1.23 | 1.000 | 9234 | 10755 |
| turbovec 4-bit + rerank4x [RAM f32] | 1732 | 0 | 0.58 | 1.000 | 2944 | 3774 |
| HNSW M32 efc100 ef128 [int8 stored] | 5719 | 0 | 0.17 | 0.944 | 328 | 811 |
| HNSW M32 efc100 ef256 [int8 stored] | 5719 | 0 | 0.17 | 0.958 | 499 | 1192 |
| HNSW M32 efc100 ef128 [bfp16 stored] | 6182 | 0 | 0.16 | 0.957 | 377 | 983 |
| HNSW M32 efc100 ef256 [bfp16 stored] | 6182 | 0 | 0.16 | 0.987 | 545 | 1346 |
| HNSW M32 efc100 ef128 [f32lib stored] | 6879 | 0 | 0.15 | 0.958 | 447 | 1140 |
| HNSW M32 efc100 ef256 [f32lib stored] | 6879 | 0 | 0.15 | 0.993 | 662 | 1601 |
| HNSW M48 efc100 ef64 [f32lib stored] | 7323 | 0 | 0.14 | 0.974 | 332 | 730 |
| HNSW M48 efc100 ef128 [f32lib stored] | 7323 | 0 | 0.14 | 0.990 | 456 | 918 |
| HNSW M48 efc100 ef256 [f32lib stored] | 7323 | 0 | 0.14 | 1.000 | 673 | 1237 |

### synth384-n1000000 [scale mode, streamed] (n=1000000, dim=384)

(A) RAM-only 3-D Pareto (RAM B/vec, recall, p99):
| config | RAM B/vec | SSD B/vec | M vec/GB RAM | recall@10 | p50 us | p99 us |
|---|---|---|---|---|---|---|
| flat int8(per-vector scale) | 388 | 0 | 2.58 | 0.963 | 11637 | 17060 |
| flat blockfp16(32,i16+f32 scale) | 816 | 0 | 1.23 | 1.000 | 32710 | 52483 |
| flat f32 exact (avx2) | 1536 | 0 | 0.65 | 1.000 | 35225 | 43992 |
| turbovec 4-bit + rerank4x [RAM f32] | 1732 | 0 | 0.58 | 0.999 | 10167 | 14924 |

(B) with f32 rerank rows on SSD (cold direct I/O) 3-D Pareto:
| config | RAM B/vec | SSD B/vec | M vec/GB RAM | recall@10 | p50 us | p99 us |
|---|---|---|---|---|---|---|
| turbovec 2-bit + rerank16x [SSD direct] | 107 | 1536 | 9.33 | 0.969 | 18508 | 22966 |
| rabitq-style 4-bit + rerank4x [SSD direct] | 197 | 1536 | 5.09 | 0.997 | 42326 | 66406 |
| turbovec 3-bit + rerank8x [SSD direct] | 207 | 1536 | 4.83 | 0.995 | 16741 | 20111 |
| turbovec 4-bit + rerank4x [SSD direct] | 207 | 1536 | 4.83 | 0.999 | 14626 | 16264 |
| flat int8 + rerank4x [SSD direct] | 388 | 1536 | 2.58 | 1.000 | 15485 | 23980 |
| turbovec 4-bit + rerank4x [RAM f32] | 1732 | 0 | 0.58 | 0.999 | 10167 | 14924 |
