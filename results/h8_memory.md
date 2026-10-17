
#### h8_memory_real-bge.json n=23453 dim=768
| config | index heap B/vec | RAM B/vec | SSD B/vec | M vec/GB RAM | recall@10 | p50 us | p99 us | build s |
|---|---|---|---|---|---|---|---|---|
| flat f32 exact (avx2) | 3072.0 | 3072.0 | 0 | 0.33 | 1.000 | 1973 | 2940 | 0.0 |
| flat f16 | 1536.0 | 1536.0 | 0 | 0.65 | 1.000 | 1489 | 1799 | 0.0 |
| flat blockfp16(32,i16+f32 scale) | 1632.0 | 1632.0 | 0 | 0.61 | 1.000 | 1394 | 1717 | 0.1 |
| flat int8(per-vector scale) | 772.0 | 772.0 | 0 | 1.30 | 0.993 | 953 | 1059 | 0.1 |
| flat int8 + rerank4x [RAM] | 772.0 | 3844.0 | 0 | 0.26 | 1.000 | 987 | 1184 | 0.1 |
| flat int8 + rerank4x [SSD buffered] | 772.0 | 772.0 | 3072 | 1.30 | 1.000 | 1203 | 1458 | 0.1 |
| flat int8 + rerank4x [SSD direct] | 772.0 | 772.0 | 3072 | 1.30 | 1.000 | 4949 | 14200 | 0.1 |
| turbovec 2-bit | 196.7 | 196.7 | 0 | 5.08 | 0.931 | 130 | 207 | 0.1 |
| turbovec 2-bit + rerank4x [RAM] | 196.7 | 3268.7 | 0 | 0.31 | 1.000 | 170 | 217 | 0.1 |
| turbovec 2-bit + rerank4x [SSD buffered] | 196.7 | 196.7 | 3072 | 5.08 | 1.000 | 430 | 856 | 0.1 |
| turbovec 2-bit + rerank4x [SSD direct] | 196.7 | 196.7 | 3072 | 5.08 | 1.000 | 4524 | 6546 | 0.1 |
| turbovec 2-bit + rerank8x [RAM] | 196.7 | 3268.7 | 0 | 0.31 | 1.000 | 383 | 1243 | 0.1 |
| turbovec 2-bit + rerank8x [SSD buffered] | 196.7 | 196.7 | 3072 | 5.08 | 1.000 | 780 | 1365 | 0.1 |
| turbovec 2-bit + rerank8x [SSD direct] | 196.7 | 196.7 | 3072 | 5.08 | 1.000 | 8385 | 11926 | 0.1 |
| turbovec 2-bit + rerank16x [RAM] | 196.7 | 3268.7 | 0 | 0.31 | 1.000 | 404 | 538 | 0.1 |
| turbovec 2-bit + rerank16x [SSD buffered] | 196.7 | 196.7 | 3072 | 5.08 | 1.000 | 1278 | 3909 | 0.1 |
| turbovec 2-bit + rerank16x [SSD direct] | 196.7 | 196.7 | 3072 | 5.08 | 1.000 | 15624 | 24601 | 0.1 |
| turbovec 3-bit | 388.7 | 388.7 | 0 | 2.57 | 0.958 | 234 | 395 | 0.2 |
| turbovec 3-bit + rerank2x [RAM] | 388.7 | 3460.7 | 0 | 0.29 | 1.000 | 296 | 514 | 0.2 |
| turbovec 3-bit + rerank4x [RAM] | 388.7 | 3460.7 | 0 | 0.29 | 1.000 | 592 | 770 | 0.2 |
| turbovec 3-bit + rerank4x [SSD buffered] | 388.7 | 388.7 | 3072 | 2.57 | 1.000 | 773 | 1008 | 0.2 |
| turbovec 3-bit + rerank4x [SSD direct] | 388.7 | 388.7 | 3072 | 2.57 | 1.000 | 4080 | 5293 | 0.2 |
| turbovec 3-bit + rerank8x [RAM] | 388.7 | 3460.7 | 0 | 0.29 | 1.000 | 426 | 627 | 0.2 |
| turbovec 3-bit + rerank8x [SSD buffered] | 388.7 | 388.7 | 3072 | 2.57 | 1.000 | 924 | 1094 | 0.2 |
| turbovec 3-bit + rerank8x [SSD direct] | 388.7 | 388.7 | 3072 | 2.57 | 1.000 | 7368 | 8162 | 0.2 |
| turbovec 4-bit | 388.7 | 388.7 | 0 | 2.57 | 0.976 | 659 | 722 | 0.2 |
| turbovec 4-bit + rerank2x [RAM] | 388.7 | 3460.7 | 0 | 0.29 | 1.000 | 680 | 769 | 0.2 |
| turbovec 4-bit + rerank2x [SSD buffered] | 388.7 | 388.7 | 3072 | 2.57 | 1.000 | 802 | 938 | 0.2 |
| turbovec 4-bit + rerank2x [SSD direct] | 388.7 | 388.7 | 3072 | 2.57 | 1.000 | 2455 | 2804 | 0.2 |
| turbovec 4-bit + rerank4x [RAM] | 388.7 | 3460.7 | 0 | 0.29 | 1.000 | 699 | 1267 | 0.2 |
| turbovec 4-bit + rerank4x [SSD buffered] | 388.7 | 388.7 | 3072 | 2.57 | 1.000 | 871 | 1063 | 0.2 |
| turbovec 4-bit + rerank4x [SSD direct] | 388.7 | 388.7 | 3072 | 2.57 | 1.000 | 4067 | 6034 | 0.2 |
| turbovec 4-bit + rerank8x [RAM] | 388.7 | 3460.7 | 0 | 0.29 | 1.000 | 747 | 828 | 0.2 |
| turbovec 4-bit + rerank8x [SSD buffered] | 388.7 | 388.7 | 3072 | 2.57 | 1.000 | 1079 | 1227 | 0.2 |
| turbovec 4-bit + rerank8x [SSD direct] | 388.7 | 388.7 | 3072 | 2.57 | 1.000 | 7040 | 8789 | 0.2 |
| rabitq-style 1-bit | 200.6 | 200.6 | 0 | 4.99 | 0.869 | 463 | 554 | 0.7 |
| rabitq-style 1-bit + rerank8x [RAM] | 200.6 | 3272.6 | 0 | 0.31 | 1.000 | 372 | 553 | 0.7 |
| rabitq-style 1-bit + rerank8x [SSD buffered] | 200.6 | 200.6 | 3072 | 4.99 | 1.000 | 762 | 1007 | 0.7 |
| rabitq-style 1-bit + rerank8x [SSD direct] | 200.6 | 200.6 | 3072 | 4.99 | 1.000 | 6107 | 7264 | 0.7 |
| rabitq-style 1-bit + rerank16x [RAM] | 200.6 | 3272.6 | 0 | 0.31 | 1.000 | 507 | 599 | 0.7 |
| rabitq-style 2-bit | 296.6 | 296.6 | 0 | 3.37 | 0.923 | 822 | 855 | 1.2 |
| rabitq-style 2-bit + rerank2x [RAM] | 296.6 | 3368.6 | 0 | 0.30 | 0.999 | 670 | 869 | 1.2 |
| rabitq-style 2-bit + rerank4x [RAM] | 296.6 | 3368.6 | 0 | 0.30 | 1.000 | 843 | 880 | 1.2 |
| rabitq-style 2-bit + rerank4x [SSD buffered] | 296.6 | 296.6 | 3072 | 3.37 | 1.000 | 882 | 1134 | 1.2 |
| rabitq-style 2-bit + rerank4x [SSD direct] | 296.6 | 296.6 | 3072 | 3.37 | 1.000 | 4555 | 7563 | 1.2 |
| rabitq-style 2-bit + rerank8x [RAM] | 296.6 | 3368.6 | 0 | 0.30 | 1.000 | 881 | 937 | 1.2 |
| rabitq-style 3-bit | 392.6 | 392.6 | 0 | 2.55 | 0.955 | 899 | 1180 | 1.3 |
| rabitq-style 3-bit + rerank2x [RAM] | 392.6 | 3464.6 | 0 | 0.29 | 1.000 | 1035 | 1271 | 1.3 |
| rabitq-style 3-bit + rerank4x [RAM] | 392.6 | 3464.6 | 0 | 0.29 | 1.000 | 922 | 1265 | 1.3 |
| rabitq-style 3-bit + rerank4x [SSD buffered] | 392.6 | 392.6 | 3072 | 2.55 | 1.000 | 1110 | 1482 | 1.3 |
| rabitq-style 3-bit + rerank4x [SSD direct] | 392.6 | 392.6 | 3072 | 2.55 | 1.000 | 3788 | 4715 | 1.3 |
| rabitq-style 3-bit + rerank8x [RAM] | 392.6 | 3464.6 | 0 | 0.29 | 1.000 | 1066 | 1248 | 1.3 |
| rabitq-style 4-bit | 488.6 | 488.6 | 0 | 2.05 | 0.975 | 1343 | 1631 | 1.0 |
| rabitq-style 4-bit + rerank2x [RAM] | 488.6 | 3560.6 | 0 | 0.28 | 1.000 | 1151 | 1647 | 1.0 |
| rabitq-style 4-bit + rerank4x [RAM] | 488.6 | 3560.6 | 0 | 0.28 | 1.000 | 1445 | 2927 | 1.0 |
| rabitq-style 4-bit + rerank4x [SSD buffered] | 488.6 | 488.6 | 3072 | 2.05 | 1.000 | 1656 | 1933 | 1.0 |
| rabitq-style 4-bit + rerank4x [SSD direct] | 488.6 | 488.6 | 3072 | 2.05 | 1.000 | 4259 | 5138 | 1.0 |
| rabitq-style 4-bit + rerank8x [RAM] | 488.6 | 3560.6 | 0 | 0.28 | 1.000 | 1212 | 2134 | 1.0 |
| rabitq-style 5-bit | 584.6 | 584.6 | 0 | 1.71 | 0.987 | 1632 | 2072 | 1.1 |
| rabitq-style 6-bit | 680.6 | 680.6 | 0 | 1.47 | 0.992 | 1677 | 2397 | 1.1 |
| HNSW M32 efc100 ef24 [f32lib stored] | 8164.6 | 8164.6 | 0 | 0.12 | 0.996 | 162 | 257 | 3.4 |
| HNSW M32 efc100 ef24 [bfp16 stored] | 6696.3 | 6696.3 | 0 | 0.15 | 0.996 | 128 | 222 | 2.5 |
| HNSW M32 efc100 ef24 [int8 stored] | 5873.9 | 5873.9 | 0 | 0.17 | 0.993 | 95 | 164 | 2.1 |
| HNSW M16 efc100 ef24 [f32lib stored] | 7416.2 | 7416.2 | 0 | 0.13 | 0.999 | 142 | 238 | 3.4 |
| HNSW M16 efc100 ef24 [bfp16 stored] | 5986.2 | 5986.2 | 0 | 0.17 | 0.999 | 144 | 227 | 2.5 |
| HNSW M16 efc100 ef24 [int8 stored] | 5111.4 | 5111.4 | 0 | 0.20 | 0.980 | 99 | 223 | 2.0 |
| HNSW M48 efc200 ef16 [f32lib stored] | 8697.6 | 8697.6 | 0 | 0.11 | 0.999 | 154 | 240 | 10.0 |
| HNSW M48 efc200 ef16 [bfp16 stored] | 7268.2 | 7268.2 | 0 | 0.14 | 0.999 | 91 | 172 | 5.9 |
| HNSW M48 efc200 ef16 [int8 stored] | 6404.9 | 6404.9 | 0 | 0.16 | 0.993 | 75 | 133 | 3.9 |

#### h8_memory_real-mini.json n=23153 dim=384
| config | index heap B/vec | RAM B/vec | SSD B/vec | M vec/GB RAM | recall@10 | p50 us | p99 us | build s |
|---|---|---|---|---|---|---|---|---|
| flat f32 exact (avx2) | 1536.0 | 1536.0 | 0 | 0.65 | 1.000 | 1151 | 3413 | 0.0 |
| flat f16 | 768.0 | 768.0 | 0 | 1.30 | 1.000 | 803 | 2386 | 0.0 |
| flat blockfp16(32,i16+f32 scale) | 816.0 | 816.0 | 0 | 1.23 | 1.000 | 1070 | 2703 | 0.0 |
| flat int8(per-vector scale) | 388.0 | 388.0 | 0 | 2.58 | 0.997 | 560 | 2584 | 0.0 |
| flat int8 + rerank4x [RAM] | 388.0 | 1924.0 | 0 | 0.52 | 1.000 | 610 | 2279 | 0.0 |
| flat int8 + rerank4x [SSD buffered] | 388.0 | 388.0 | 1536 | 2.58 | 1.000 | 766 | 5022 | 0.0 |
| flat int8 + rerank4x [SSD direct] | 388.0 | 388.0 | 1536 | 2.58 | 1.000 | 6925 | 18169 | 0.0 |
| turbovec 2-bit | 100.4 | 100.4 | 0 | 9.96 | 0.924 | 84 | 303 | 0.1 |
| turbovec 2-bit + rerank4x [RAM] | 100.4 | 1636.4 | 0 | 0.61 | 1.000 | 114 | 1070 | 0.1 |
| turbovec 2-bit + rerank4x [SSD buffered] | 100.4 | 100.4 | 1536 | 9.96 | 1.000 | 333 | 2085 | 0.1 |
| turbovec 2-bit + rerank4x [SSD direct] | 100.4 | 100.4 | 1536 | 9.96 | 1.000 | 4117 | 10218 | 0.1 |
| turbovec 2-bit + rerank8x [RAM] | 100.4 | 1636.4 | 0 | 0.61 | 1.000 | 148 | 206 | 0.1 |
| turbovec 2-bit + rerank8x [SSD buffered] | 100.4 | 100.4 | 1536 | 9.96 | 1.000 | 507 | 1288 | 0.1 |
| turbovec 2-bit + rerank8x [SSD direct] | 100.4 | 100.4 | 1536 | 9.96 | 1.000 | 10842 | 45034 | 0.1 |
| turbovec 2-bit + rerank16x [RAM] | 100.4 | 1636.4 | 0 | 0.61 | 1.000 | 298 | 2115 | 0.1 |
| turbovec 2-bit + rerank16x [SSD buffered] | 100.4 | 100.4 | 1536 | 9.96 | 1.000 | 1094 | 3061 | 0.1 |
| turbovec 2-bit + rerank16x [SSD direct] | 100.4 | 100.4 | 1536 | 9.96 | 1.000 | 18168 | 42742 | 0.1 |
| turbovec 3-bit | 196.5 | 196.5 | 0 | 5.09 | 0.960 | 125 | 152 | 0.1 |
| turbovec 3-bit + rerank2x [RAM] | 196.5 | 1732.5 | 0 | 0.58 | 1.000 | 140 | 171 | 0.1 |
| turbovec 3-bit + rerank4x [RAM] | 196.5 | 1732.5 | 0 | 0.58 | 1.000 | 163 | 251 | 0.1 |
| turbovec 3-bit + rerank4x [SSD buffered] | 196.5 | 196.5 | 1536 | 5.09 | 1.000 | 327 | 450 | 0.1 |
| turbovec 3-bit + rerank4x [SSD direct] | 196.5 | 196.5 | 1536 | 5.09 | 1.000 | 3784 | 15934 | 0.1 |
| turbovec 3-bit + rerank8x [RAM] | 196.5 | 1732.5 | 0 | 0.58 | 1.000 | 204 | 237 | 0.1 |
| turbovec 3-bit + rerank8x [SSD buffered] | 196.5 | 196.5 | 1536 | 5.09 | 1.000 | 533 | 697 | 0.1 |
| turbovec 3-bit + rerank8x [SSD direct] | 196.5 | 196.5 | 1536 | 5.09 | 1.000 | 6806 | 8869 | 0.1 |
| turbovec 4-bit | 196.5 | 196.5 | 0 | 5.09 | 0.977 | 341 | 380 | 0.1 |
| turbovec 4-bit + rerank2x [RAM] | 196.5 | 1732.5 | 0 | 0.58 | 1.000 | 358 | 385 | 0.1 |
| turbovec 4-bit + rerank2x [SSD buffered] | 196.5 | 196.5 | 1536 | 5.09 | 1.000 | 447 | 506 | 0.1 |
| turbovec 4-bit + rerank2x [SSD direct] | 196.5 | 196.5 | 1536 | 5.09 | 1.000 | 2019 | 2941 | 0.1 |
| turbovec 4-bit + rerank4x [RAM] | 196.5 | 1732.5 | 0 | 0.58 | 1.000 | 372 | 411 | 0.1 |
| turbovec 4-bit + rerank4x [SSD buffered] | 196.5 | 196.5 | 1536 | 5.09 | 1.000 | 543 | 627 | 0.1 |
| turbovec 4-bit + rerank4x [SSD direct] | 196.5 | 196.5 | 1536 | 5.09 | 1.000 | 3863 | 5979 | 0.1 |
| turbovec 4-bit + rerank8x [RAM] | 196.5 | 1732.5 | 0 | 0.58 | 1.000 | 425 | 582 | 0.1 |
| turbovec 4-bit + rerank8x [SSD buffered] | 196.5 | 196.5 | 1536 | 5.09 | 1.000 | 745 | 1206 | 0.1 |
| turbovec 4-bit + rerank8x [SSD direct] | 196.5 | 196.5 | 1536 | 5.09 | 1.000 | 7370 | 10719 | 0.1 |
| rabitq-style 1-bit | 77.5 | 77.5 | 0 | 12.91 | 0.867 | 250 | 1004 | 0.2 |
| rabitq-style 1-bit + rerank8x [RAM] | 77.5 | 1613.5 | 0 | 0.62 | 1.000 | 292 | 416 | 0.2 |
| rabitq-style 1-bit + rerank8x [SSD buffered] | 77.5 | 77.5 | 1536 | 12.91 | 1.000 | 616 | 783 | 0.2 |
| rabitq-style 1-bit + rerank8x [SSD direct] | 77.5 | 77.5 | 1536 | 12.91 | 1.000 | 6805 | 9336 | 0.2 |
| rabitq-style 1-bit + rerank16x [RAM] | 77.5 | 1613.5 | 0 | 0.62 | 1.000 | 359 | 654 | 0.2 |
| rabitq-style 2-bit | 125.5 | 125.5 | 0 | 7.97 | 0.929 | 447 | 1103 | 0.6 |
| rabitq-style 2-bit + rerank2x [RAM] | 125.5 | 1661.5 | 0 | 0.60 | 0.999 | 485 | 1097 | 0.6 |
| rabitq-style 2-bit + rerank4x [RAM] | 125.5 | 1661.5 | 0 | 0.60 | 1.000 | 513 | 2127 | 0.6 |
| rabitq-style 2-bit + rerank4x [SSD buffered] | 125.5 | 125.5 | 1536 | 7.97 | 1.000 | 650 | 1255 | 0.6 |
| rabitq-style 2-bit + rerank4x [SSD direct] | 125.5 | 125.5 | 1536 | 7.97 | 1.000 | 4339 | 7649 | 0.6 |
| rabitq-style 2-bit + rerank8x [RAM] | 125.5 | 1661.5 | 0 | 0.60 | 1.000 | 522 | 634 | 0.6 |
| rabitq-style 3-bit | 173.5 | 173.5 | 0 | 5.76 | 0.959 | 648 | 683 | 0.5 |
| rabitq-style 3-bit + rerank2x [RAM] | 173.5 | 1709.5 | 0 | 0.58 | 1.000 | 654 | 708 | 0.5 |
| rabitq-style 3-bit + rerank4x [RAM] | 173.5 | 1709.5 | 0 | 0.58 | 1.000 | 697 | 788 | 0.5 |
| rabitq-style 3-bit + rerank4x [SSD buffered] | 173.5 | 173.5 | 1536 | 5.76 | 1.000 | 870 | 1054 | 0.5 |
| rabitq-style 3-bit + rerank4x [SSD direct] | 173.5 | 173.5 | 1536 | 5.76 | 1.000 | 6918 | 11557 | 0.5 |
| rabitq-style 3-bit + rerank8x [RAM] | 173.5 | 1709.5 | 0 | 0.58 | 1.000 | 698 | 837 | 0.5 |
| rabitq-style 4-bit | 221.5 | 221.5 | 0 | 4.52 | 0.972 | 884 | 3592 | 0.5 |
| rabitq-style 4-bit + rerank2x [RAM] | 221.5 | 1757.5 | 0 | 0.57 | 1.000 | 907 | 2755 | 0.5 |
| rabitq-style 4-bit + rerank4x [RAM] | 221.5 | 1757.5 | 0 | 0.57 | 1.000 | 822 | 1040 | 0.5 |
| rabitq-style 4-bit + rerank4x [SSD buffered] | 221.5 | 221.5 | 1536 | 4.52 | 1.000 | 1104 | 2371 | 0.5 |
| rabitq-style 4-bit + rerank4x [SSD direct] | 221.5 | 221.5 | 1536 | 4.52 | 1.000 | 4142 | 5325 | 0.5 |
| rabitq-style 4-bit + rerank8x [RAM] | 221.5 | 1757.5 | 0 | 0.57 | 1.000 | 867 | 972 | 0.5 |
| rabitq-style 5-bit | 269.5 | 269.5 | 0 | 3.71 | 0.987 | 954 | 1065 | 0.4 |
| rabitq-style 6-bit | 317.5 | 317.5 | 0 | 3.15 | 0.992 | 1188 | 2391 | 0.5 |
| HNSW M32 efc100 ef24 [f32lib stored] | 6561.8 | 6561.8 | 0 | 0.15 | 0.992 | 127 | 232 | 3.2 |
| HNSW M32 efc100 ef24 [bfp16 stored] | 5815.9 | 5815.9 | 0 | 0.17 | 0.994 | 120 | 221 | 2.5 |
| HNSW M32 efc100 ef24 [int8 stored] | 5432.3 | 5432.3 | 0 | 0.18 | 0.988 | 103 | 168 | 2.3 |
| HNSW M16 efc100 ef24 [f32lib stored] | 5856.3 | 5856.3 | 0 | 0.17 | 0.992 | 142 | 269 | 3.5 |
| HNSW M16 efc100 ef24 [bfp16 stored] | 5124.1 | 5124.1 | 0 | 0.20 | 0.994 | 106 | 209 | 2.4 |
| HNSW M16 efc100 ef24 [int8 stored] | 4695.3 | 4695.3 | 0 | 0.21 | 0.993 | 116 | 215 | 2.1 |
| HNSW M48 efc200 ef16 [f32lib stored] | 7134.6 | 7134.6 | 0 | 0.14 | 0.994 | 135 | 241 | 8.2 |
| HNSW M48 efc200 ef16 [bfp16 stored] | 6418.7 | 6418.7 | 0 | 0.16 | 0.995 | 103 | 205 | 7.3 |
| HNSW M48 efc200 ef16 [int8 stored] | 6003.1 | 6003.1 | 0 | 0.17 | 0.992 | 75 | 146 | 4.8 |

#### h8_memory_synth384-n100000.json n=100000 dim=384
| config | index heap B/vec | RAM B/vec | SSD B/vec | M vec/GB RAM | recall@10 | p50 us | p99 us | build s |
|---|---|---|---|---|---|---|---|---|
| flat f32 exact (avx2) | 1536.0 | 1536.0 | 0 | 0.65 | 1.000 | 3225 | 4717 | 0.0 |
| flat f16 | 768.0 | 768.0 | 0 | 1.30 | 0.999 | 3004 | 3144 | 0.0 |
| flat blockfp16(32,i16+f32 scale) | 816.0 | 816.0 | 0 | 1.23 | 1.000 | 2540 | 3240 | 0.1 |
| flat int8(per-vector scale) | 388.0 | 388.0 | 0 | 2.58 | 0.976 | 2100 | 2235 | 0.1 |
| flat int8 + rerank4x [RAM] | 388.0 | 1924.0 | 0 | 0.52 | 1.000 | 1475 | 2078 | 0.1 |
| flat int8 + rerank4x [SSD buffered] | 388.0 | 388.0 | 1536 | 2.58 | 1.000 | 2374 | 2502 | 0.1 |
| flat int8 + rerank4x [SSD direct] | 388.0 | 388.0 | 1536 | 2.58 | 1.000 | 4828 | 6044 | 0.1 |
| turbovec 2-bit | 100.1 | 100.1 | 0 | 9.99 | 0.557 | 614 | 840 | 0.1 |
| turbovec 2-bit + rerank4x [RAM] | 100.1 | 1636.1 | 0 | 0.61 | 0.957 | 616 | 1109 | 0.1 |
| turbovec 2-bit + rerank4x [SSD buffered] | 100.1 | 100.1 | 1536 | 9.99 | 0.957 | 588 | 940 | 0.1 |
| turbovec 2-bit + rerank4x [SSD direct] | 100.1 | 100.1 | 1536 | 9.99 | 0.957 | 3762 | 4543 | 0.1 |
| turbovec 2-bit + rerank8x [RAM] | 100.1 | 1636.1 | 0 | 0.61 | 1.000 | 274 | 507 | 0.1 |
| turbovec 2-bit + rerank8x [SSD buffered] | 100.1 | 100.1 | 1536 | 9.99 | 1.000 | 797 | 1211 | 0.1 |
| turbovec 2-bit + rerank8x [SSD direct] | 100.1 | 100.1 | 1536 | 9.99 | 1.000 | 6656 | 8205 | 0.1 |
| turbovec 2-bit + rerank16x [RAM] | 100.1 | 1636.1 | 0 | 0.61 | 1.000 | 471 | 755 | 0.1 |
| turbovec 2-bit + rerank16x [SSD buffered] | 100.1 | 100.1 | 1536 | 9.99 | 1.000 | 1595 | 1869 | 0.1 |
| turbovec 2-bit + rerank16x [SSD direct] | 100.1 | 100.1 | 1536 | 9.99 | 1.000 | 12818 | 18115 | 0.1 |
| turbovec 3-bit | 196.1 | 196.1 | 0 | 5.10 | 0.726 | 1037 | 1369 | 0.1 |
| turbovec 3-bit + rerank2x [RAM] | 196.1 | 1732.1 | 0 | 0.58 | 0.946 | 918 | 1138 | 0.1 |
| turbovec 3-bit + rerank4x [RAM] | 196.1 | 1732.1 | 0 | 0.58 | 0.998 | 946 | 1121 | 0.1 |
| turbovec 3-bit + rerank4x [SSD buffered] | 196.1 | 196.1 | 1536 | 5.10 | 0.998 | 1305 | 1618 | 0.1 |
| turbovec 3-bit + rerank4x [SSD direct] | 196.1 | 196.1 | 1536 | 5.10 | 0.998 | 4165 | 4939 | 0.1 |
| turbovec 3-bit + rerank8x [RAM] | 196.1 | 1732.1 | 0 | 0.58 | 1.000 | 925 | 1152 | 0.1 |
| turbovec 3-bit + rerank8x [SSD buffered] | 196.1 | 196.1 | 1536 | 5.10 | 1.000 | 1440 | 1925 | 0.1 |
| turbovec 3-bit + rerank8x [SSD direct] | 196.1 | 196.1 | 1536 | 5.10 | 1.000 | 7374 | 18994 | 0.1 |
| turbovec 4-bit | 196.1 | 196.1 | 0 | 5.10 | 0.847 | 765 | 993 | 0.1 |
| turbovec 4-bit + rerank2x [RAM] | 196.1 | 1732.1 | 0 | 0.58 | 0.995 | 768 | 969 | 0.1 |
| turbovec 4-bit + rerank2x [SSD buffered] | 196.1 | 196.1 | 1536 | 5.10 | 0.995 | 891 | 1088 | 0.1 |
| turbovec 4-bit + rerank2x [SSD direct] | 196.1 | 196.1 | 1536 | 5.10 | 0.995 | 2506 | 2926 | 0.1 |
| turbovec 4-bit + rerank4x [RAM] | 196.1 | 1732.1 | 0 | 0.58 | 1.000 | 811 | 969 | 0.1 |
| turbovec 4-bit + rerank4x [SSD buffered] | 196.1 | 196.1 | 1536 | 5.10 | 1.000 | 1038 | 1346 | 0.1 |
| turbovec 4-bit + rerank4x [SSD direct] | 196.1 | 196.1 | 1536 | 5.10 | 1.000 | 4415 | 5082 | 0.1 |
| turbovec 4-bit + rerank8x [RAM] | 196.1 | 1732.1 | 0 | 0.58 | 1.000 | 752 | 997 | 0.1 |
| turbovec 4-bit + rerank8x [SSD buffered] | 196.1 | 196.1 | 1536 | 5.10 | 1.000 | 1211 | 1432 | 0.1 |
| turbovec 4-bit + rerank8x [SSD direct] | 196.1 | 196.1 | 1536 | 5.10 | 1.000 | 7327 | 16850 | 0.1 |
| rabitq-style 1-bit | 57.9 | 57.9 | 0 | 17.27 | 0.347 | 952 | 1100 | 0.6 |
| rabitq-style 1-bit + rerank8x [RAM] | 57.9 | 1593.9 | 0 | 0.63 | 0.987 | 1003 | 1093 | 0.6 |
| rabitq-style 1-bit + rerank8x [SSD buffered] | 57.9 | 57.9 | 1536 | 17.27 | 0.987 | 1422 | 1839 | 0.6 |
| rabitq-style 1-bit + rerank8x [SSD direct] | 57.9 | 57.9 | 1536 | 17.27 | 0.987 | 7202 | 8128 | 0.6 |
| rabitq-style 1-bit + rerank16x [RAM] | 57.9 | 1593.9 | 0 | 0.63 | 1.000 | 1561 | 1630 | 0.6 |
| rabitq-style 2-bit | 105.9 | 105.9 | 0 | 9.44 | 0.554 | 2610 | 2692 | 1.8 |
| rabitq-style 2-bit + rerank2x [RAM] | 105.9 | 1641.9 | 0 | 0.61 | 0.787 | 2694 | 2781 | 1.8 |
| rabitq-style 2-bit + rerank4x [RAM] | 105.9 | 1641.9 | 0 | 0.61 | 0.958 | 2254 | 2734 | 1.8 |
| rabitq-style 2-bit + rerank4x [SSD buffered] | 105.9 | 105.9 | 1536 | 9.44 | 0.958 | 2529 | 3059 | 1.8 |
| rabitq-style 2-bit + rerank4x [SSD direct] | 105.9 | 105.9 | 1536 | 9.44 | 0.958 | 4605 | 6119 | 1.8 |
| rabitq-style 2-bit + rerank8x [RAM] | 105.9 | 1641.9 | 0 | 0.61 | 0.999 | 1794 | 2233 | 1.8 |
| rabitq-style 3-bit | 153.9 | 153.9 | 0 | 6.50 | 0.720 | 2515 | 3874 | 1.7 |
| rabitq-style 3-bit + rerank2x [RAM] | 153.9 | 1689.9 | 0 | 0.59 | 0.934 | 2914 | 3910 | 1.7 |
| rabitq-style 3-bit + rerank4x [RAM] | 153.9 | 1689.9 | 0 | 0.59 | 0.999 | 2684 | 3877 | 1.7 |
| rabitq-style 3-bit + rerank4x [SSD buffered] | 153.9 | 153.9 | 1536 | 6.50 | 0.999 | 3018 | 3876 | 1.7 |
| rabitq-style 3-bit + rerank4x [SSD direct] | 153.9 | 153.9 | 1536 | 6.50 | 0.999 | 5116 | 7080 | 1.7 |
| rabitq-style 3-bit + rerank8x [RAM] | 153.9 | 1689.9 | 0 | 0.59 | 1.000 | 3890 | 3925 | 1.7 |
| rabitq-style 4-bit | 201.9 | 201.9 | 0 | 4.95 | 0.842 | 3904 | 5076 | 1.6 |
| rabitq-style 4-bit + rerank2x [RAM] | 201.9 | 1737.9 | 0 | 0.58 | 0.994 | 4589 | 5154 | 1.6 |
| rabitq-style 4-bit + rerank4x [RAM] | 201.9 | 1737.9 | 0 | 0.58 | 1.000 | 3318 | 5131 | 1.6 |
| rabitq-style 4-bit + rerank4x [SSD buffered] | 201.9 | 201.9 | 1536 | 4.95 | 1.000 | 4047 | 5382 | 1.6 |
| rabitq-style 4-bit + rerank4x [SSD direct] | 201.9 | 201.9 | 1536 | 4.95 | 1.000 | 5975 | 8622 | 1.6 |
| rabitq-style 4-bit + rerank8x [RAM] | 201.9 | 1737.9 | 0 | 0.58 | 1.000 | 3122 | 4929 | 1.6 |
| rabitq-style 5-bit | 249.9 | 249.9 | 0 | 4.00 | 0.908 | 4372 | 6225 | 1.7 |
| rabitq-style 6-bit | 297.9 | 297.9 | 0 | 3.36 | 0.950 | 4544 | 7496 | 1.6 |
| HNSW M32 efc100 ef24 [f32lib stored] | 6843.0 | 6843.0 | 0 | 0.15 | 0.936 | 165 | 331 | 29.5 |
| HNSW M32 efc100 ef24 [bfp16 stored] | 6129.5 | 6129.5 | 0 | 0.16 | 0.942 | 109 | 240 | 21.4 |
| HNSW M32 efc100 ef24 [int8 stored] | 5684.2 | 5684.2 | 0 | 0.18 | 0.927 | 152 | 329 | 18.2 |
| HNSW M16 efc100 ef24 [f32lib stored] | 6087.2 | 6087.2 | 0 | 0.16 | 0.818 | 197 | 456 | 26.7 |
| HNSW M16 efc100 ef24 [bfp16 stored] | 5371.4 | 5371.4 | 0 | 0.19 | 0.842 | 119 | 291 | 19.6 |
| HNSW M16 efc100 ef24 [int8 stored] | 4884.2 | 4884.2 | 0 | 0.20 | 0.815 | 92 | 207 | 17.4 |
| HNSW M48 efc200 ef16 [f32lib stored] | 7423.8 | 7423.8 | 0 | 0.13 | 0.962 | 200 | 352 | 59.8 |
| HNSW M48 efc200 ef16 [bfp16 stored] | 6706.9 | 6706.9 | 0 | 0.15 | 0.969 | 144 | 285 | 53.3 |
| HNSW M48 efc200 ef16 [int8 stored] | 6276.9 | 6276.9 | 0 | 0.16 | 0.945 | 95 | 175 | 40.6 |
