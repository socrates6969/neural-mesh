 setting number of points 50000 
 setting number of points 100000 

#### concurrency (n=100000, dim=384, 4s windows x3)
| index | scenario | threads | qps (mean of reps) [min-max] | p50 us | p99 us | writer ops/s |
|---|---|---|---|---|---|---|
| hnsw M16 efc200 ef128 | read-only, shared Arc, no lock | 1 | 1229 [1129-1299] | 758 | 1746 | 0 |
| hnsw M16 efc200 ef128 | read-only, shared Arc, no lock | 2 | 2402 [2125-2623] | 754 | 2142 | 0 |
| hnsw M16 efc200 ef128 | read-only, shared Arc, no lock | 4 | 4367 [3790-4919] | 761 | 3054 | 0 |
| hnsw M16 efc200 ef128 | readers + 1 concurrent insert thread (insert_slice on &self, no external lock) | 1 | 1270 [1218-1312] | 725 | 1798 | 752 |
| hnsw M16 efc200 ef128 | readers + 1 concurrent insert thread (insert_slice on &self, no external lock) | 2 | 2497 [2389-2618] | 718 | 2068 | 762 |
| turbovec-4bit + f32 rerank 4x, RwLock, per-reader private 1-thread rayon pool | read-only | 1 | 799 [660-898] | 1187 | 2807 | 0 |
| turbovec-4bit + f32 rerank 4x, RwLock, per-reader private 1-thread rayon pool | read-only | 2 | 1767 [1745-1805] | 1035 | 2528 | 0 |
| turbovec-4bit + f32 rerank 4x, RwLock, per-reader private 1-thread rayon pool | read-only | 4 | 2204 [1138-3722] | 2211 | 5048 | 0 |
| turbovec-4bit + rerank 4x, RwLock | 2 readers + writer adding 64 vecs/50ms (added vectors are NOT in the f32 rerank store: rerank skips ids >= n) | 2 | 647 [451-782] | 2728 | 11973 | 1138 |
| flat-f32 avx2 | read-only | 1 | 195 [167-224] | 4901 | 10058 | 0 |
| flat-f32 avx2 | read-only | 2 | 222 [203-260] | 7635 | 42092 | 0 |
| flat-f32 avx2 | read-only | 4 | 284 [270-303] | 10918 | 50703 | 0 |
