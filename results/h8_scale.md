
#### h8_scale.json n=100000 dim=384
| config | index heap B/vec | RAM B/vec | SSD B/vec | M vec/GB RAM | recall@10 | p50 us | p99 us | build s |
|---|---|---|---|---|---|---|---|---|
| flat blockfp16(32,i16+f32 scale) | 816.0 | 816.0 | 0 | 1.23 | 1.000 | 2705 | 3808 | 0.8 |
| flat int8(per-vector scale) | 388.0 | 388.0 | 0 | 2.58 | 0.974 | 1122 | 1819 | 0.7 |
| flat int8 + rerank4x [SSD direct] | 388.0 | 388.0 | 1536 | 2.58 | 1.000 | 4458 | 5829 | 0.7 |
| flat int8 + rerank4x [SSD buffered] | 388.0 | 388.0 | 1536 | 2.58 | 1.000 | 1324 | 1886 | 0.7 |
| turbovec 2-bit | 133.6 | 133.6 | 0 | 7.48 | 0.553 | 198 | 480 | 0.8 |
| turbovec 2-bit + rerank4x [SSD direct] | 133.6 | 133.6 | 1536 | 7.48 | 0.957 | 3398 | 4171 | 0.8 |
| turbovec 2-bit + rerank4x [SSD buffered] | 133.6 | 133.6 | 1536 | 7.48 | 0.957 | 360 | 504 | 0.8 |
| turbovec 2-bit + rerank16x [SSD direct] | 133.6 | 133.6 | 1536 | 7.48 | 1.000 | 12546 | 15679 | 0.8 |
| turbovec 2-bit + rerank16x [SSD buffered] | 133.6 | 133.6 | 1536 | 7.48 | 1.000 | 1995 | 9149 | 0.8 |
| turbovec 3-bit | 231.6 | 231.6 | 0 | 4.32 | 0.723 | 1797 | 15822 | 2.6 |
| turbovec 3-bit + rerank8x [SSD direct] | 231.6 | 231.6 | 1536 | 4.32 | 1.000 | 10591 | 50043 | 2.6 |
| turbovec 3-bit + rerank8x [SSD buffered] | 231.6 | 231.6 | 1536 | 4.32 | 1.000 | 2554 | 13648 | 2.6 |
| turbovec 4-bit | 231.6 | 231.6 | 0 | 4.32 | 0.848 | 1512 | 20147 | 3.0 |
| turbovec 4-bit + rerank4x [SSD direct] | 231.6 | 231.6 | 1536 | 4.32 | 1.000 | 6014 | 43683 | 3.0 |
| turbovec 4-bit + rerank4x [SSD buffered] | 231.6 | 231.6 | 1536 | 4.32 | 1.000 | 1862 | 9193 | 3.0 |
| rabitq-style 1-bit | 57.9 | 57.9 | 0 | 17.27 | 0.344 | 1094 | 4467 | 2.4 |
| rabitq-style 1-bit + rerank16x [SSD direct] | 57.9 | 57.9 | 1536 | 17.27 | 1.000 | 16357 | 52625 | 2.4 |
| rabitq-style 1-bit + rerank16x [SSD buffered] | 57.9 | 57.9 | 1536 | 17.27 | 1.000 | 2305 | 12260 | 2.4 |
| rabitq-style 2-bit | 105.9 | 105.9 | 0 | 9.44 | 0.557 | 1946 | 6047 | 3.9 |
| rabitq-style 2-bit + rerank8x [SSD direct] | 105.9 | 105.9 | 1536 | 9.44 | 1.000 | 9834 | 51251 | 3.9 |
| rabitq-style 2-bit + rerank8x [SSD buffered] | 105.9 | 105.9 | 1536 | 9.44 | 1.000 | 2643 | 17354 | 3.9 |
| rabitq-style 3-bit | 153.9 | 153.9 | 0 | 6.50 | 0.708 | 2771 | 10070 | 3.7 |
| rabitq-style 3-bit + rerank4x [SSD direct] | 153.9 | 153.9 | 1536 | 6.50 | 1.000 | 6811 | 49162 | 3.7 |
| rabitq-style 3-bit + rerank4x [SSD buffered] | 153.9 | 153.9 | 1536 | 6.50 | 1.000 | 3149 | 19184 | 3.7 |
| rabitq-style 4-bit | 201.9 | 201.9 | 0 | 4.95 | 0.838 | 3566 | 20902 | 3.5 |
| rabitq-style 4-bit + rerank4x [SSD direct] | 201.9 | 201.9 | 1536 | 4.95 | 1.000 | 7730 | 51676 | 3.5 |
| rabitq-style 4-bit + rerank4x [SSD buffered] | 201.9 | 201.9 | 1536 | 4.95 | 1.000 | 3960 | 21132 | 3.5 |
| HNSW M16 efc200 ef32 [f32lib stored] | 6167.7 | 6167.7 | 0 | 0.16 | 0.919 | 205 | 449 | 50.0 |
| HNSW M16 efc200 ef64 [f32lib stored] | 6167.7 | 6167.7 | 0 | 0.16 | 0.985 | 319 | 685 | 50.0 |
| HNSW M16 efc200 ef128 [f32lib stored] | 6167.7 | 6167.7 | 0 | 0.16 | 1.000 | 489 | 883 | 50.0 |
| HNSW M16 efc200 ef256 [f32lib stored] | 6167.7 | 6167.7 | 0 | 0.16 | 1.000 | 855 | 1465 | 50.0 |
| HNSW M32 efc100 ef32 [f32lib stored] | 6898.1 | 6898.1 | 0 | 0.14 | 0.934 | 166 | 357 | 24.9 |
| HNSW M32 efc100 ef64 [f32lib stored] | 6898.1 | 6898.1 | 0 | 0.14 | 0.985 | 263 | 521 | 24.9 |
| HNSW M32 efc100 ef128 [f32lib stored] | 6898.1 | 6898.1 | 0 | 0.14 | 1.000 | 451 | 821 | 24.9 |
| HNSW M32 efc100 ef256 [f32lib stored] | 6898.1 | 6898.1 | 0 | 0.14 | 1.000 | 742 | 1155 | 24.9 |
| HNSW M48 efc100 ef24 [f32lib stored] | 7266.9 | 7266.9 | 0 | 0.14 | 0.982 | 166 | 323 | 23.3 |
| HNSW M48 efc100 ef32 [f32lib stored] | 7266.9 | 7266.9 | 0 | 0.14 | 0.995 | 192 | 378 | 23.3 |
| HNSW M48 efc100 ef64 [f32lib stored] | 7266.9 | 7266.9 | 0 | 0.14 | 1.000 | 302 | 599 | 23.3 |
| HNSW M48 efc100 ef128 [f32lib stored] | 7266.9 | 7266.9 | 0 | 0.14 | 1.000 | 504 | 769 | 23.3 |
| HNSW M48 efc100 ef256 [f32lib stored] | 7266.9 | 7266.9 | 0 | 0.14 | 1.000 | 849 | 1186 | 23.3 |
| HNSW M32 efc100 ef32 [bfp16 stored] | 6116.6 | 6116.6 | 0 | 0.16 | 0.989 | 135 | 310 | 22.5 |
| HNSW M32 efc100 ef64 [bfp16 stored] | 6116.6 | 6116.6 | 0 | 0.16 | 0.990 | 208 | 394 | 22.5 |
| HNSW M32 efc100 ef128 [bfp16 stored] | 6116.6 | 6116.6 | 0 | 0.16 | 1.000 | 368 | 576 | 22.5 |
| HNSW M32 efc100 ef256 [bfp16 stored] | 6116.6 | 6116.6 | 0 | 0.16 | 1.000 | 702 | 1335 | 22.5 |
| HNSW M32 efc100 ef32 [int8 stored] | 5703.2 | 5703.2 | 0 | 0.18 | 0.949 | 120 | 235 | 16.0 |
| HNSW M32 efc100 ef64 [int8 stored] | 5703.2 | 5703.2 | 0 | 0.18 | 0.969 | 184 | 381 | 16.0 |
| HNSW M32 efc100 ef128 [int8 stored] | 5703.2 | 5703.2 | 0 | 0.18 | 0.969 | 324 | 520 | 16.0 |
| HNSW M32 efc100 ef256 [int8 stored] | 5703.2 | 5703.2 | 0 | 0.18 | 0.974 | 619 | 972 | 16.0 |
| flat f32 exact (avx2) | 1536.0 | 1536.0 | 0 | 0.65 | 1.000 | 3852 | 5912 | 0.0 |
| turbovec 4-bit + rerank4x [RAM f32] | 196.1 | 1732.1 | 0 | 0.58 | 1.000 | 928 | 1470 | 0.1 |

#### h8_scale.json n=300000 dim=384
| config | index heap B/vec | RAM B/vec | SSD B/vec | M vec/GB RAM | recall@10 | p50 us | p99 us | build s |
|---|---|---|---|---|---|---|---|---|
| flat blockfp16(32,i16+f32 scale) | 816.0 | 816.0 | 0 | 1.23 | 1.000 | 9234 | 10755 | 2.3 |
| flat int8(per-vector scale) | 388.0 | 388.0 | 0 | 2.58 | 0.964 | 5681 | 22578 | 2.3 |
| flat int8 + rerank4x [SSD direct] | 388.0 | 388.0 | 1536 | 2.58 | 1.000 | 8768 | 25991 | 2.3 |
| flat int8 + rerank4x [SSD buffered] | 388.0 | 388.0 | 1536 | 2.58 | 1.000 | 3744 | 6067 | 2.3 |
| turbovec 2-bit | 115.7 | 115.7 | 0 | 8.64 | 0.461 | 1839 | 2875 | 2.3 |
| turbovec 2-bit + rerank4x [SSD direct] | 115.7 | 115.7 | 1536 | 8.64 | 0.854 | 5553 | 10330 | 2.3 |
| turbovec 2-bit + rerank4x [SSD buffered] | 115.7 | 115.7 | 1536 | 8.64 | 0.854 | 2390 | 4163 | 2.3 |
| turbovec 2-bit + rerank16x [SSD direct] | 115.7 | 115.7 | 1536 | 8.64 | 0.999 | 14386 | 16945 | 2.3 |
| turbovec 2-bit + rerank16x [SSD buffered] | 115.7 | 115.7 | 1536 | 8.64 | 0.999 | 2878 | 4209 | 2.3 |
| turbovec 3-bit | 216.7 | 216.7 | 0 | 4.61 | 0.651 | 3151 | 4328 | 2.6 |
| turbovec 3-bit + rerank8x [SSD direct] | 216.7 | 216.7 | 1536 | 4.61 | 1.000 | 12985 | 44763 | 2.6 |
| turbovec 3-bit + rerank8x [SSD buffered] | 216.7 | 216.7 | 1536 | 4.61 | 1.000 | 3663 | 5011 | 2.6 |
| turbovec 4-bit | 216.7 | 216.7 | 0 | 4.61 | 0.803 | 2647 | 3145 | 2.4 |
| turbovec 4-bit + rerank4x [SSD direct] | 216.7 | 216.7 | 1536 | 4.61 | 1.000 | 5705 | 7056 | 2.4 |
| turbovec 4-bit + rerank4x [SSD buffered] | 216.7 | 216.7 | 1536 | 4.61 | 1.000 | 2803 | 3839 | 2.4 |
| rabitq-style 1-bit | 54.0 | 54.0 | 0 | 18.53 | 0.227 | 3273 | 15688 | 3.5 |
| rabitq-style 1-bit + rerank16x [SSD direct] | 54.0 | 54.0 | 1536 | 18.53 | 0.953 | 15590 | 26680 | 3.5 |
| rabitq-style 1-bit + rerank16x [SSD buffered] | 54.0 | 54.0 | 1536 | 18.53 | 0.953 | 3569 | 4738 | 3.5 |
| rabitq-style 2-bit | 102.0 | 102.0 | 0 | 9.81 | 0.466 | 4201 | 5138 | 8.1 |
| rabitq-style 2-bit + rerank8x [SSD direct] | 102.0 | 102.0 | 1536 | 9.81 | 0.961 | 11910 | 13200 | 8.1 |
| rabitq-style 2-bit + rerank8x [SSD buffered] | 102.0 | 102.0 | 1536 | 9.81 | 0.961 | 5309 | 7089 | 8.1 |
| rabitq-style 3-bit | 150.0 | 150.0 | 0 | 6.67 | 0.655 | 7064 | 8875 | 6.8 |
| rabitq-style 3-bit + rerank4x [SSD direct] | 150.0 | 150.0 | 1536 | 6.67 | 0.979 | 10917 | 12274 | 6.8 |
| rabitq-style 3-bit + rerank4x [SSD buffered] | 150.0 | 150.0 | 1536 | 6.67 | 0.979 | 6586 | 8130 | 6.8 |
| rabitq-style 4-bit | 198.0 | 198.0 | 0 | 5.05 | 0.794 | 8073 | 9234 | 5.4 |
| rabitq-style 4-bit + rerank4x [SSD direct] | 198.0 | 198.0 | 1536 | 5.05 | 1.000 | 11310 | 12825 | 5.4 |
| rabitq-style 4-bit + rerank4x [SSD buffered] | 198.0 | 198.0 | 1536 | 5.05 | 1.000 | 8410 | 9202 | 5.4 |
| HNSW M16 efc200 ef32 [f32lib stored] | 6234.2 | 6234.2 | 0 | 0.16 | 0.752 | 196 | 445 | 130.4 |
| HNSW M16 efc200 ef64 [f32lib stored] | 6234.2 | 6234.2 | 0 | 0.16 | 0.807 | 292 | 671 | 130.4 |
| HNSW M16 efc200 ef128 [f32lib stored] | 6234.2 | 6234.2 | 0 | 0.16 | 0.832 | 420 | 1023 | 130.4 |
| HNSW M16 efc200 ef256 [f32lib stored] | 6234.2 | 6234.2 | 0 | 0.16 | 0.891 | 682 | 1952 | 130.4 |
| HNSW M32 efc100 ef32 [f32lib stored] | 6879.2 | 6879.2 | 0 | 0.15 | 0.791 | 232 | 546 | 71.3 |
| HNSW M32 efc100 ef64 [f32lib stored] | 6879.2 | 6879.2 | 0 | 0.15 | 0.868 | 332 | 708 | 71.3 |
| HNSW M32 efc100 ef128 [f32lib stored] | 6879.2 | 6879.2 | 0 | 0.15 | 0.958 | 447 | 1140 | 71.3 |
| HNSW M32 efc100 ef256 [f32lib stored] | 6879.2 | 6879.2 | 0 | 0.15 | 0.993 | 662 | 1601 | 71.3 |
| HNSW M48 efc100 ef24 [f32lib stored] | 7323.4 | 7323.4 | 0 | 0.14 | 0.877 | 256 | 597 | 69.2 |
| HNSW M48 efc100 ef32 [f32lib stored] | 7323.4 | 7323.4 | 0 | 0.14 | 0.897 | 238 | 552 | 69.2 |
| HNSW M48 efc100 ef64 [f32lib stored] | 7323.4 | 7323.4 | 0 | 0.14 | 0.974 | 332 | 730 | 69.2 |
| HNSW M48 efc100 ef128 [f32lib stored] | 7323.4 | 7323.4 | 0 | 0.14 | 0.990 | 456 | 918 | 69.2 |
| HNSW M48 efc100 ef256 [f32lib stored] | 7323.4 | 7323.4 | 0 | 0.14 | 1.000 | 673 | 1237 | 69.2 |
| HNSW M32 efc100 ef32 [bfp16 stored] | 6182.2 | 6182.2 | 0 | 0.16 | 0.804 | 203 | 455 | 57.4 |
| HNSW M32 efc100 ef64 [bfp16 stored] | 6182.2 | 6182.2 | 0 | 0.16 | 0.908 | 312 | 881 | 57.4 |
| HNSW M32 efc100 ef128 [bfp16 stored] | 6182.2 | 6182.2 | 0 | 0.16 | 0.957 | 377 | 983 | 57.4 |
| HNSW M32 efc100 ef256 [bfp16 stored] | 6182.2 | 6182.2 | 0 | 0.16 | 0.987 | 545 | 1346 | 57.4 |
| HNSW M32 efc100 ef32 [int8 stored] | 5719.0 | 5719.0 | 0 | 0.17 | 0.848 | 155 | 396 | 51.3 |
| HNSW M32 efc100 ef64 [int8 stored] | 5719.0 | 5719.0 | 0 | 0.17 | 0.897 | 231 | 575 | 51.3 |
| HNSW M32 efc100 ef128 [int8 stored] | 5719.0 | 5719.0 | 0 | 0.17 | 0.944 | 328 | 811 | 51.3 |
| HNSW M32 efc100 ef256 [int8 stored] | 5719.0 | 5719.0 | 0 | 0.17 | 0.958 | 499 | 1192 | 51.3 |
| flat f32 exact (avx2) | 1536.0 | 1536.0 | 0 | 0.65 | 1.000 | 10862 | 15548 | 0.0 |
| turbovec 4-bit + rerank4x [RAM f32] | 196.0 | 1732.0 | 0 | 0.58 | 1.000 | 2944 | 3774 | 0.3 |

#### h8_scale.json n=1000000 dim=384
| config | index heap B/vec | RAM B/vec | SSD B/vec | M vec/GB RAM | recall@10 | p50 us | p99 us | build s |
|---|---|---|---|---|---|---|---|---|
| flat blockfp16(32,i16+f32 scale) | 816.0 | 816.0 | 0 | 1.23 | 1.000 | 32710 | 52483 | 10.4 |
| flat int8(per-vector scale) | 388.0 | 388.0 | 0 | 2.58 | 0.963 | 11637 | 17060 | 9.9 |
| flat int8 + rerank4x [SSD direct] | 388.0 | 388.0 | 1536 | 2.58 | 1.000 | 15485 | 23980 | 9.9 |
| flat int8 + rerank4x [SSD buffered] | 388.0 | 388.0 | 1536 | 2.58 | 1.000 | 11956 | 19591 | 9.9 |
| turbovec 2-bit | 107.2 | 107.2 | 0 | 9.33 | 0.361 | 5832 | 7537 | 7.5 |
| turbovec 2-bit + rerank4x [SSD direct] | 107.2 | 107.2 | 1536 | 9.33 | 0.717 | 9197 | 11600 | 7.5 |
| turbovec 2-bit + rerank4x [SSD buffered] | 107.2 | 107.2 | 1536 | 9.33 | 0.717 | 6297 | 8243 | 7.5 |
| turbovec 2-bit + rerank16x [SSD direct] | 107.2 | 107.2 | 1536 | 9.33 | 0.969 | 18508 | 22966 | 7.5 |
| turbovec 2-bit + rerank16x [SSD buffered] | 107.2 | 107.2 | 1536 | 9.33 | 0.969 | 7904 | 15477 | 7.5 |
| turbovec 3-bit | 207.0 | 207.0 | 0 | 4.83 | 0.606 | 9601 | 11478 | 8.3 |
| turbovec 3-bit + rerank8x [SSD direct] | 207.0 | 207.0 | 1536 | 4.83 | 0.995 | 16741 | 20111 | 8.3 |
| turbovec 3-bit + rerank8x [SSD buffered] | 207.0 | 207.0 | 1536 | 4.83 | 0.995 | 10565 | 21491 | 8.3 |
| turbovec 4-bit | 207.0 | 207.0 | 0 | 4.83 | 0.772 | 11788 | 19106 | 9.8 |
| turbovec 4-bit + rerank4x [SSD direct] | 207.0 | 207.0 | 1536 | 4.83 | 0.999 | 14626 | 16264 | 9.8 |
| turbovec 4-bit + rerank4x [SSD buffered] | 207.0 | 207.0 | 1536 | 4.83 | 0.999 | 11672 | 14931 | 9.8 |
| rabitq-style 1-bit | 52.6 | 52.6 | 0 | 19.02 | 0.139 | 11317 | 12077 | 15.9 |
| rabitq-style 1-bit + rerank16x [SSD direct] | 52.6 | 52.6 | 1536 | 19.02 | 0.711 | 22099 | 25348 | 15.9 |
| rabitq-style 1-bit + rerank16x [SSD buffered] | 52.6 | 52.6 | 1536 | 19.02 | 0.711 | 8288 | 10635 | 15.9 |
| rabitq-style 2-bit | 100.6 | 100.6 | 0 | 9.94 | 0.343 | 18437 | 20894 | 29.1 |
| rabitq-style 2-bit + rerank8x [SSD direct] | 100.6 | 100.6 | 1536 | 9.94 | 0.865 | 25910 | 52337 | 29.1 |
| rabitq-style 2-bit + rerank8x [SSD buffered] | 100.6 | 100.6 | 1536 | 9.94 | 0.865 | 14608 | 17882 | 29.1 |
| rabitq-style 3-bit | 148.6 | 148.6 | 0 | 6.73 | 0.587 | 26280 | 29539 | 18.4 |
| rabitq-style 3-bit + rerank4x [SSD direct] | 148.6 | 148.6 | 1536 | 6.73 | 0.955 | 31649 | 37748 | 18.4 |
| rabitq-style 3-bit + rerank4x [SSD buffered] | 148.6 | 148.6 | 1536 | 6.73 | 0.955 | 30093 | 57559 | 18.4 |
| rabitq-style 4-bit | 196.6 | 196.6 | 0 | 5.09 | 0.755 | 32638 | 48419 | 25.2 |
| rabitq-style 4-bit + rerank4x [SSD direct] | 196.6 | 196.6 | 1536 | 5.09 | 0.997 | 42326 | 66406 | 25.2 |
| rabitq-style 4-bit + rerank4x [SSD buffered] | 196.6 | 196.6 | 1536 | 5.09 | 0.997 | 38657 | 40561 | 25.2 |
| HNSW M16 efc200 ef128 [f32lib stored] SKIPPED (projected 6.23 GB > cap 2.5 GB or free RAM) | - | - | - | - | - | - | - | - |
| HNSW M32 efc100 ef32 [f32lib stored] SKIPPED (projected 6.88 GB > cap 2.5 GB or free RAM) | - | - | - | - | - | - | - | - |
| HNSW M48 efc100 ef24 [f32lib stored] SKIPPED (projected 7.32 GB > cap 2.5 GB or free RAM) | - | - | - | - | - | - | - | - |
| HNSW M32 efc100 ef32 [bfp16 stored] SKIPPED (projected 6.18 GB > cap 2.5 GB or free RAM) | - | - | - | - | - | - | - | - |
| HNSW M32 efc100 ef32 [int8 stored] SKIPPED (projected 5.72 GB > cap 2.5 GB or free RAM) | - | - | - | - | - | - | - | - |
| flat f32 exact (avx2) | 1536.0 | 1536.0 | 0 | 0.65 | 1.000 | 35225 | 43992 | 0.0 |
| turbovec 4-bit + rerank4x [RAM f32] | 196.0 | 1732.0 | 0 | 0.58 | 0.999 | 10167 | 14924 | 1.3 |
