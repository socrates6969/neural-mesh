
#### scale N=10000 dim=384 (peak WS 66 MB, WS before 22 MB)
| config | nominal B/vec | heap B/vec (index) | RAM B/vec incl. f32 for rerank | M vectors/GB (RAM) | recall@10 | p50 us | p99 us | build s |
|---|---|---|---|---|---|---|---|---|
| flat-f32 exact (avx2) | 1536.0 | 1536.0 | 1536.0 | 0.7 | 1.000 | 435.7 | 821.0 | 0.0 |
| flat-f32 exact (lib sse2 kernel) | 1536.0 | 1536.0 | 1536.0 | 0.7 | 1.000 | 469.4 | 635.5 | 0.0 |
| flat-int8 scan | 388.0 | 388.0 | 388.0 | 2.6 | 0.997 | 123.1 | 178.8 | 0.0 |
| flat-int8 scan + f32 rerank 10x | 388.0 | 388.0 | 1924.0 | 0.5 | 1.000 | 131.4 | 219.9 | 0.0 |
| turbovec-2bit (1 thread) | 100.0 | 479.2 | 479.2 | 2.1 | 0.888 | 35.1 | 44.7 | 0.0 |
| turbovec-2bit + f32 rerank 4x (1 thread) | 100.0 | 479.2 | 2015.2 | 0.5 | 0.998 | 53.7 | 72.2 | 0.0 |
| turbovec-2bit + f32 rerank 16x (1 thread) | 100.0 | 479.2 | 2015.2 | 0.5 | 1.000 | 170.9 | 228.0 | 0.0 |
| turbovec-4bit (1 thread) | 196.0 | 636.5 | 636.5 | 1.6 | 0.965 | 153.8 | 233.5 | 0.0 |
| turbovec-4bit + f32 rerank 4x (1 thread) | 196.0 | 636.5 | 2172.5 | 0.5 | 1.000 | 172.7 | 244.3 | 0.0 |
| turbovec-4bit + f32 rerank 16x (1 thread) | 196.0 | 636.5 | 2172.5 | 0.5 | 1.000 | 277.0 | 403.6 | 0.0 |
| pq-lite m=48 + f32 rerank 30x | 87.3 | 100.4 | 1636.4 | 0.6 | 1.000 | 482.0 | 664.9 | 2.2 |
| hnsw M16 efc200 ef32 (lib kernel) | - | 6508.2 | 6508.2 | 0.15 | 0.982 | 265.5 | 348.9 | 5.6 |
| hnsw M16 efc200 ef64 (lib kernel) | - | 6508.2 | 6508.2 | 0.15 | 0.985 | 481.2 | 641.0 | 5.6 |
| hnsw M16 efc200 ef128 (lib kernel) | - | 6508.2 | 6508.2 | 0.15 | 0.985 | 799.8 | 1000.1 | 5.6 |
| hnsw M16 efc200 ef256 (lib kernel) | - | 6508.2 | 6508.2 | 0.15 | 0.985 | 1675.2 | 2036.9 | 5.6 |
| hnsw M16 efc200 ef64 (avx2 kernel) | - | 6506.5 | 6506.5 | 0.15 | 0.982 | 399.9 | 506.3 | 5.0 |
| hnsw M16 efc200 ef128 (avx2 kernel) | - | 6506.5 | 6506.5 | 0.15 | 0.983 | 700.5 | 821.3 | 5.0 |
 setting number of points 50000 
 setting number of points 100000 
 setting number of points 50000 
 setting number of points 100000 
 setting number of points 50000 
 setting number of points 100000 
 setting number of points 50000 
 setting number of points 100000 

#### scale N=100000 dim=384 (peak WS 531 MB, WS before 159 MB)
| config | nominal B/vec | heap B/vec (index) | RAM B/vec incl. f32 for rerank | M vectors/GB (RAM) | recall@10 | p50 us | p99 us | build s |
|---|---|---|---|---|---|---|---|---|
| flat-f32 exact (avx2) | 1536.0 | 1536.0 | 1536.0 | 0.7 | 1.000 | 4502.9 | 5824.7 | 0.0 |
| flat-f32 exact (lib sse2 kernel) | 1536.0 | 1536.0 | 1536.0 | 0.7 | 1.000 | 4793.9 | 6233.7 | 0.0 |
| flat-int8 scan | 388.0 | 388.0 | 388.0 | 2.6 | 0.974 | 1487.1 | 1834.3 | 0.1 |
| flat-int8 scan + f32 rerank 10x | 388.0 | 388.0 | 1924.0 | 0.5 | 1.000 | 1499.5 | 1890.7 | 0.0 |
| turbovec-2bit (1 thread) | 100.0 | 132.7 | 132.7 | 7.5 | 0.553 | 592.5 | 1529.4 | 0.1 |
| turbovec-2bit + f32 rerank 4x (1 thread) | 100.0 | 132.7 | 1668.7 | 0.6 | 0.957 | 359.5 | 532.7 | 0.0 |
| turbovec-2bit + f32 rerank 16x (1 thread) | 100.0 | 132.7 | 1668.7 | 0.6 | 1.000 | 739.8 | 1354.7 | 0.0 |
| turbovec-4bit (1 thread) | 196.0 | 229.7 | 229.7 | 4.4 | 0.848 | 1030.0 | 1398.3 | 0.1 |
| turbovec-4bit + f32 rerank 4x (1 thread) | 196.0 | 229.7 | 1765.7 | 0.6 | 1.000 | 1103.2 | 1534.2 | 0.0 |
| turbovec-4bit + f32 rerank 16x (1 thread) | 196.0 | 229.7 | 1765.7 | 0.6 | 1.000 | 1227.3 | 1419.0 | 0.0 |
| turbovec-4bit + f32 rerank 4x (global pool, 2 threads) | 196.0 | 229.7 | 1765.7 | 0.6 | 1.000 | 560.6 | 750.5 | 0.0 |
| pq-lite m=48 + f32 rerank 30x | 51.9 | 53.2 | 1589.2 | 0.6 | 1.000 | 3281.9 | 3780.8 | 6.3 |
| hnsw M16 efc200 ef32 (lib kernel) | - | 6210.9 | 6210.9 | 0.16 | 0.950 | 275.5 | 618.8 | 56.5 |
| hnsw M16 efc200 ef64 (lib kernel) | - | 6210.9 | 6210.9 | 0.16 | 0.978 | 429.2 | 969.2 | 56.5 |
| hnsw M16 efc200 ef128 (lib kernel) | - | 6210.9 | 6210.9 | 0.16 | 0.990 | 753.1 | 1411.4 | 56.5 |
| hnsw M16 efc200 ef256 (lib kernel) | - | 6210.9 | 6210.9 | 0.16 | 0.990 | 1147.6 | 1875.3 | 56.5 |
| hnsw M16 efc200 ef64 (avx2 kernel) | - | 6176.6 | 6176.6 | 0.16 | 0.969 | 354.3 | 728.1 | 50.6 |
| hnsw M16 efc200 ef128 (avx2 kernel) | - | 6176.6 | 6176.6 | 0.16 | 0.984 | 684.0 | 1249.7 | 50.6 |

#### scale N=1000000 dim=384 (peak WS 1856 MB, WS before 1482 MB)
| config | nominal B/vec | heap B/vec (index) | RAM B/vec incl. f32 for rerank | M vectors/GB (RAM) | recall@10 | p50 us | p99 us | build s |
|---|---|---|---|---|---|---|---|---|
| flat-f32 exact (avx2) | 1536.0 | 1536.0 | 1536.0 | 0.7 | 1.000 | 45429.5 | 64558.8 | 0.0 |
| flat-int8 scan | 388.0 | 388.0 | 388.0 | 2.6 | 0.963 | 14682.5 | 17798.1 | 1.2 |
| flat-int8 scan + f32 rerank 10x | 388.0 | 388.0 | 1924.0 | 0.5 | 1.000 | 14589.9 | 17344.1 | 0.0 |
| turbovec-2bit (1 thread) | 100.0 | 107.7 | 107.7 | 9.3 | 0.361 | 7043.6 | 8381.7 | 1.0 |
| turbovec-2bit + f32 rerank 4x (1 thread) | 100.0 | 107.7 | 1643.7 | 0.6 | 0.717 | 7139.9 | 8637.0 | 0.0 |
| turbovec-2bit + f32 rerank 16x (1 thread) | 100.0 | 107.7 | 1643.7 | 0.6 | 0.969 | 8035.2 | 9998.3 | 0.0 |
| turbovec-4bit (1 thread) | 196.0 | 208.1 | 208.1 | 4.8 | 0.772 | 11421.4 | 13135.2 | 1.4 |
| turbovec-4bit + f32 rerank 4x (1 thread) | 196.0 | 208.1 | 1744.1 | 0.6 | 0.999 | 10843.4 | 12169.7 | 0.0 |
| turbovec-4bit + f32 rerank 16x (1 thread) | 196.0 | 208.1 | 1744.1 | 0.6 | 1.000 | 11290.9 | 12764.0 | 0.0 |
| turbovec-4bit + f32 rerank 4x (global pool, 2 threads) | 196.0 | 208.1 | 1744.1 | 0.6 | 0.999 | 6312.3 | 8240.7 | 0.0 |
hnsw skipped: "N=1000000 > --hnsw-max 100000"
