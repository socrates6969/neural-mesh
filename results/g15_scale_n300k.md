 setting number of points 50000 
 setting number of points 100000 
 setting number of points 150000 
 setting number of points 200000 
 setting number of points 250000 
 setting number of points 300000 
 setting number of points 50000 
 setting number of points 100000 
 setting number of points 150000 
 setting number of points 200000 
 setting number of points 250000 
 setting number of points 300000 

#### scale N=300000 dim=384 (peak WS 1602 MB, WS before 447 MB)
| config | nominal B/vec | heap B/vec (index) | RAM B/vec incl. f32 for rerank | M vectors/GB (RAM) | recall@10 | p50 us | p99 us | build s |
|---|---|---|---|---|---|---|---|---|
| flat-f32 exact (avx2) | 1536.0 | 1536.0 | 1536.0 | 0.7 | 1.000 | 13074.7 | 18072.5 | 0.0 |
| flat-int8 scan | 388.0 | 388.0 | 388.0 | 2.6 | 0.964 | 4124.1 | 4830.9 | 0.3 |
| flat-int8 scan + f32 rerank 10x | 388.0 | 388.0 | 1924.0 | 0.5 | 1.000 | 4446.3 | 5265.0 | 0.0 |
| turbovec-2bit (1 thread) | 100.0 | 115.6 | 115.6 | 8.6 | 0.461 | 2137.2 | 2449.8 | 0.3 |
| turbovec-2bit + f32 rerank 4x (1 thread) | 100.0 | 115.6 | 1651.6 | 0.6 | 0.854 | 2158.9 | 2456.3 | 0.0 |
| turbovec-2bit + f32 rerank 16x (1 thread) | 100.0 | 115.6 | 1651.6 | 0.6 | 0.999 | 2376.9 | 2783.3 | 0.0 |
| turbovec-4bit (1 thread) | 196.0 | 216.6 | 216.6 | 4.6 | 0.803 | 3329.9 | 4185.9 | 0.5 |
| turbovec-4bit + f32 rerank 4x (1 thread) | 196.0 | 216.6 | 1752.6 | 0.6 | 1.000 | 3341.1 | 4630.0 | 0.0 |
| turbovec-4bit + f32 rerank 16x (1 thread) | 196.0 | 216.6 | 1752.6 | 0.6 | 1.000 | 3570.0 | 4081.2 | 0.0 |
| turbovec-4bit + f32 rerank 4x (global pool, 2 threads) | 196.0 | 216.6 | 1752.6 | 0.6 | 1.000 | 2165.0 | 2945.2 | 0.0 |
| hnsw M16 efc200 ef32 (lib kernel) | - | 6231.0 | 6231.0 | 0.16 | 0.782 | 332.8 | 746.4 | 188.1 |
| hnsw M16 efc200 ef64 (lib kernel) | - | 6231.0 | 6231.0 | 0.16 | 0.841 | 493.1 | 1264.8 | 188.1 |
| hnsw M16 efc200 ef128 (lib kernel) | - | 6231.0 | 6231.0 | 0.16 | 0.882 | 756.9 | 1808.2 | 188.1 |
| hnsw M16 efc200 ef256 (lib kernel) | - | 6231.0 | 6231.0 | 0.16 | 0.923 | 1184.9 | 3135.2 | 188.1 |
| hnsw M16 efc200 ef64 (avx2 kernel) | - | 6244.0 | 6244.0 | 0.16 | 0.808 | 484.2 | 1193.7 | 193.0 |
| hnsw M16 efc200 ef128 (avx2 kernel) | - | 6244.0 | 6244.0 | 0.16 | 0.846 | 713.3 | 1765.2 | 193.0 |
