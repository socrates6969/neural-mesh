
#### scale N=10000 dim=384 (peak WS 65 MB, WS before 22 MB)
| config | nominal B/vec | heap B/vec (index) | RAM B/vec incl. f32 for rerank | M vectors/GB (RAM) | recall@10 | p50 us | p99 us | build s |
|---|---|---|---|---|---|---|---|---|
| flat-f32 exact (avx2) | 1536.0 | 1536.0 | 1536.0 | 0.7 | 1.000 | 497.5 | 6130.3 | 0.0 |
| flat-f32 exact (lib sse2 kernel) | 1536.0 | 1536.0 | 1536.0 | 0.7 | 1.000 | 536.0 | 6110.4 | 0.0 |
| flat-int8 scan | 388.0 | 388.0 | 388.0 | 2.6 | 0.997 | 134.9 | 245.7 | 0.0 |
| flat-int8 scan + f32 rerank 10x | 388.0 | 388.0 | 1924.0 | 0.5 | 1.000 | 185.1 | 298.4 | 0.0 |
| turbovec-2bit (1 thread) | 100.0 | 100.9 | 100.9 | 9.9 | 0.888 | 53.1 | 67.6 | 0.0 |
| turbovec-2bit + f32 rerank 4x (1 thread) | 100.0 | 100.9 | 1636.9 | 0.6 | 0.998 | 80.5 | 122.8 | 0.0 |
| turbovec-2bit + f32 rerank 16x (1 thread) | 100.0 | 100.9 | 1636.9 | 0.6 | 1.000 | 235.1 | 711.3 | 0.0 |
| turbovec-4bit (1 thread) | 196.0 | 197.1 | 197.1 | 5.1 | 0.965 | 183.6 | 512.1 | 0.1 |
| turbovec-4bit + f32 rerank 4x (1 thread) | 196.0 | 197.1 | 1733.1 | 0.6 | 1.000 | 212.1 | 3813.8 | 0.0 |
| turbovec-4bit + f32 rerank 16x (1 thread) | 196.0 | 197.1 | 1733.1 | 0.6 | 1.000 | 372.3 | 12968.5 | 0.0 |
| pq-lite m=48 + f32 rerank 30x | 87.3 | 100.4 | 1636.4 | 0.6 | 1.000 | 560.5 | 5937.4 | 4.2 |
| hnsw M16 efc200 ef32 (lib kernel) | - | 6480.0 | 6480.0 | 0.15 | 0.976 | 381.6 | 8259.9 | 21.2 |
| hnsw M16 efc200 ef64 (lib kernel) | - | 6480.0 | 6480.0 | 0.15 | 0.979 | 633.8 | 5261.7 | 21.2 |
| hnsw M16 efc200 ef128 (lib kernel) | - | 6480.0 | 6480.0 | 0.15 | 0.980 | 2103.3 | 13220.7 | 21.2 |
| hnsw M16 efc200 ef256 (lib kernel) | - | 6480.0 | 6480.0 | 0.15 | 0.980 | 6247.0 | 24059.4 | 21.2 |
| hnsw M16 efc200 ef64 (avx2 kernel) | - | 6468.3 | 6468.3 | 0.15 | 0.982 | 581.2 | 6290.0 | 11.4 |
| hnsw M16 efc200 ef128 (avx2 kernel) | - | 6468.3 | 6468.3 | 0.15 | 0.983 | 1016.2 | 11160.1 | 11.4 |
 setting number of points 50000 
 setting number of points 100000 
 setting number of points 50000 
 setting number of points 100000 
 setting number of points 50000 
 setting number of points 100000 
 setting number of points 50000 
 setting number of points 100000 

#### scale N=100000 dim=384 (peak WS 542 MB, WS before 157 MB)
| config | nominal B/vec | heap B/vec (index) | RAM B/vec incl. f32 for rerank | M vectors/GB (RAM) | recall@10 | p50 us | p99 us | build s |
|---|---|---|---|---|---|---|---|---|
| flat-f32 exact (avx2) | 1536.0 | 1536.0 | 1536.0 | 0.7 | 1.000 | 9869.7 | 25182.1 | 0.0 |
| flat-f32 exact (lib sse2 kernel) | 1536.0 | 1536.0 | 1536.0 | 0.7 | 1.000 | 14619.6 | 31770.1 | 0.0 |
| flat-int8 scan | 388.0 | 388.0 | 388.0 | 2.6 | 0.974 | 5479.0 | 13085.2 | 0.5 |
| flat-int8 scan + f32 rerank 10x | 388.0 | 388.0 | 1924.0 | 0.5 | 1.000 | 5993.5 | 13124.0 | 0.0 |
| turbovec-2bit (1 thread) | 100.0 | 868.1 | 868.1 | 1.2 | 0.553 | 1124.5 | 5118.5 | 0.3 |
| turbovec-2bit + f32 rerank 4x (1 thread) | 100.0 | 868.1 | 2404.1 | 0.4 | 0.957 | 996.4 | 3213.7 | 0.0 |
| turbovec-2bit + f32 rerank 16x (1 thread) | 100.0 | 868.1 | 2404.1 | 0.4 | 1.000 | 2067.4 | 4741.3 | 0.0 |
| turbovec-4bit (1 thread) | 196.0 | 964.1 | 964.1 | 1.0 | 0.848 | 1970.6 | 3254.9 | 0.3 |
| turbovec-4bit + f32 rerank 4x (1 thread) | 196.0 | 964.1 | 2500.1 | 0.4 | 1.000 | 2003.7 | 4511.1 | 0.0 |
| turbovec-4bit + f32 rerank 16x (1 thread) | 196.0 | 964.1 | 2500.1 | 0.4 | 1.000 | 2652.7 | 4325.4 | 0.0 |
| turbovec-4bit + f32 rerank 4x (global pool, 2 threads) | 196.0 | 964.1 | 2500.1 | 0.4 | 1.000 | 3286.2 | 7363.9 | 0.0 |
| pq-lite m=48 + f32 rerank 30x | 51.9 | 53.2 | 1589.2 | 0.6 | 1.000 | 5518.8 | 17484.9 | 12.4 |
| hnsw M16 efc200 ef32 (lib kernel) | - | 6276.1 | 6276.1 | 0.16 | 0.924 | 337.5 | 2845.7 | 116.5 |
| hnsw M16 efc200 ef64 (lib kernel) | - | 6276.1 | 6276.1 | 0.16 | 0.959 | 541.8 | 2734.2 | 116.5 |
| hnsw M16 efc200 ef128 (lib kernel) | - | 6276.1 | 6276.1 | 0.16 | 0.984 | 963.4 | 3343.0 | 116.5 |
| hnsw M16 efc200 ef256 (lib kernel) | - | 6276.1 | 6276.1 | 0.16 | 0.989 | 1591.4 | 5182.2 | 116.5 |
| hnsw M16 efc200 ef64 (avx2 kernel) | - | 6153.4 | 6153.4 | 0.16 | 0.964 | 399.2 | 1076.2 | 87.5 |
| hnsw M16 efc200 ef128 (avx2 kernel) | - | 6153.4 | 6153.4 | 0.16 | 0.989 | 677.5 | 1324.2 | 87.5 |

#### scale N=1000000 dim=384 (peak WS 1916 MB, WS before 1480 MB)
| config | nominal B/vec | heap B/vec (index) | RAM B/vec incl. f32 for rerank | M vectors/GB (RAM) | recall@10 | p50 us | p99 us | build s |
|---|---|---|---|---|---|---|---|---|
| flat-f32 exact (avx2) | 1536.0 | 1536.0 | 1536.0 | 0.7 | 1.000 | 49763.3 | 68975.1 | 0.0 |
| flat-int8 scan | 388.0 | 388.0 | 388.0 | 2.6 | 0.963 | 15437.1 | 19363.8 | 1.3 |
| flat-int8 scan + f32 rerank 10x | 388.0 | 388.0 | 1924.0 | 0.5 | 1.000 | 15423.0 | 20758.2 | 0.0 |
| turbovec-2bit (1 thread) | 100.0 | 188.7 | 188.7 | 5.3 | 0.361 | 8432.4 | 13536.3 | 1.5 |
| turbovec-2bit + f32 rerank 4x (1 thread) | 100.0 | 188.7 | 1724.7 | 0.6 | 0.717 | 7777.3 | 10593.3 | 0.0 |
| turbovec-2bit + f32 rerank 16x (1 thread) | 100.0 | 188.7 | 1724.7 | 0.6 | 0.969 | 8257.1 | 13665.1 | 0.0 |
| turbovec-4bit (1 thread) | 196.0 | 296.1 | 296.1 | 3.4 | 0.772 | 11457.0 | 14131.4 | 1.6 |
| turbovec-4bit + f32 rerank 4x (1 thread) | 196.0 | 296.1 | 1832.1 | 0.5 | 0.999 | 11249.3 | 13671.3 | 0.0 |
| turbovec-4bit + f32 rerank 16x (1 thread) | 196.0 | 296.1 | 1832.1 | 0.5 | 1.000 | 11691.0 | 14942.3 | 0.0 |
| turbovec-4bit + f32 rerank 4x (global pool, 2 threads) | 196.0 | 296.1 | 1832.1 | 0.5 | 0.999 | 6643.2 | 10688.5 | 0.0 |
hnsw skipped: "N=1000000 > --hnsw-max 100000"
