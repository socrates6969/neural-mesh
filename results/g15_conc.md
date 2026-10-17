 setting number of points 50000 
 setting number of points 100000 

#### concurrency (n=100000, dim=384, 4s windows x3)
| index | scenario | threads | qps (mean of reps) [min-max] | p50 us | p99 us | writer ops/s |
|---|---|---|---|---|---|---|
| hnsw M16 efc200 ef128 | read-only, shared Arc, no lock | 1 | 1122 [1033-1182] | 859 | 1712 | 0 |
| hnsw M16 efc200 ef128 | read-only, shared Arc, no lock | 2 | 2302 [2079-2481] | 832 | 1668 | 0 |
| hnsw M16 efc200 ef128 | read-only, shared Arc, no lock | 4 | 4673 [4343-4865] | 811 | 1641 | 0 |
| hnsw M16 efc200 ef128 | readers + 1 concurrent insert thread (insert_slice on &self, no external lock) | 1 | 1148 [1098-1200] | 836 | 1620 | 693 |
| hnsw M16 efc200 ef128 | readers + 1 concurrent insert thread (insert_slice on &self, no external lock) | 2 | 2273 [2203-2380] | 832 | 1675 | 697 |
| turbovec-4bit + f32 rerank 4x, RwLock, per-reader private 1-thread rayon pool | read-only | 1 | 685 [676-697] | 1414 | 2409 | 0 |
| turbovec-4bit + f32 rerank 4x, RwLock, per-reader private 1-thread rayon pool | read-only | 2 | 1492 [1409-1548] | 1183 | 3232 | 0 |
| turbovec-4bit + f32 rerank 4x, RwLock, per-reader private 1-thread rayon pool | read-only | 4 | 2780 [2340-3263] | 1091 | 6944 | 0 |
| turbovec-4bit + rerank 4x, RwLock | 2 readers + writer adding 64 vecs/50ms (added vectors are NOT in the f32 rerank store: rerank skips ids >= n) | 2 | 1077 [641-1603] | 1303 | 18106 | 1087 |
| flat-f32 avx2 | read-only | 1 | 178 [164-195] | 5421 | 8601 | 0 |
| flat-f32 avx2 | read-only | 2 | 300 [275-327] | 6292 | 12324 | 0 |
| flat-f32 avx2 | read-only | 4 | 419 [396-448] | 8129 | 33597 | 0 |
