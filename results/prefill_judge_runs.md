# OKC judge results (H200), muse_prefill_layer

Per-shape times are the judge's medians (us); `x` is the judge's paired speedup for that shape. SGLang's own time
varies run to run because the judge draws a new router seed per run (different expert loads).

## check: SGLang baseline as the submission (env probe)  job=4d5f497ae658422a  speedup=1.0007  score_us=387134
| load | layer | B | ctx | L | weight | SGLang us | ours us | x |
|---|---|---|---|---|---|---|---|---|
| turn 1 | sliding | 1 | 80000 | 753 | 46 | 1079.0 | 1079.0 | 1.000 |
| turn 1 | full | 1 | 80000 | 753 | 16 | 1469.7 | 1467.2 | 1.002 |
| turn 4 | sliding | 4 | 80000 | 753 | 46 | 3275.2 | 3273.7 | 1.000 |
| turn 4 | full | 4 | 80000 | 753 | 16 | 4930.2 | 4919.2 | 1.002 |
| first turn | full | 1 | 0 | 16384 | 16 | 18144.0 | 17959.9 | 1.010 |
| first turn | sliding | 1 | 32768 | 16384 | 138 | 17045.0 | 17043.2 | 1.000 |
| first turn | full | 1 | 32768 | 16384 | 16 | 21887.8 | 21781.3 | 1.005 |
| first turn | full | 1 | 65536 | 16384 | 16 | 24611.6 | 24812.3 | 0.992 |

## check: v1 (humming single launch, FA3)  job=568e8393157c4acc  speedup=1.4198  CI=[1.3985, 1.4411]  score_us=272862  checks={'flagged': [], 'ok': True, 'rejected': []}
| load | layer | B | ctx | L | weight | SGLang us | ours us | x |
|---|---|---|---|---|---|---|---|---|
| turn 1 | sliding | 1 | 80000 | 753 | 46 | 1093.6 | 844.6 | 1.295 |
| turn 1 | full | 1 | 80000 | 753 | 16 | 1471.7 | 1240.2 | 1.187 |
| turn 4 | sliding | 4 | 80000 | 753 | 46 | 3293.2 | 2180.8 | 1.510 |
| turn 4 | full | 4 | 80000 | 753 | 16 | 4963.8 | 3799.5 | 1.306 |
| first turn | full | 1 | 0 | 16384 | 16 | 17041.2 | 10591.7 | 1.609 |
| first turn | sliding | 1 | 32768 | 16384 | 138 | 17072.4 | 10336.2 | 1.652 |
| first turn | full | 1 | 32768 | 16384 | 16 | 21452.6 | 14656.4 | 1.464 |
| first turn | full | 1 | 65536 | 16384 | 16 | 24673.5 | 18185.9 | 1.357 |

Tests: pass, pass, pass, pass, pass

## check: v2 (humming big/small split)  job=c0cad17ab6094279  speedup=1.4928  CI=[1.4704, 1.5152]  score_us=259515  checks={'flagged': [], 'ok': True, 'rejected': []}
| load | layer | B | ctx | L | weight | SGLang us | ours us | x |
|---|---|---|---|---|---|---|---|---|
| turn 1 | sliding | 1 | 80000 | 753 | 46 | 1006.6 | 689.2 | 1.461 |
| turn 1 | full | 1 | 80000 | 753 | 16 | 1408.6 | 1083.3 | 1.300 |
| turn 4 | sliding | 4 | 80000 | 753 | 46 | 3352.7 | 2108.2 | 1.590 |
| turn 4 | full | 4 | 80000 | 753 | 16 | 4769.2 | 3612.0 | 1.320 |
| first turn | full | 1 | 0 | 16384 | 16 | 18408.8 | 11197.6 | 1.644 |
| first turn | sliding | 1 | 32768 | 16384 | 138 | 17233.1 | 10375.9 | 1.661 |
| first turn | full | 1 | 32768 | 16384 | 16 | 20696.8 | 14256.6 | 1.452 |
| first turn | full | 1 | 65536 | 16384 | 16 | 23908.8 | 18100.3 | 1.321 |

Tests: pass, pass, pass, pass, pass

## check: probe_moe (v2 + custom attention; H200 diagnostics)  job=9b1529465ddc4684  speedup=1.5114  CI=[1.4888, 1.5341]  score_us=256316  checks={'flagged': [], 'ok': True, 'rejected': []}
| load | layer | B | ctx | L | weight | SGLang us | ours us | x |
|---|---|---|---|---|---|---|---|---|
| turn 1 | sliding | 1 | 80000 | 753 | 46 | 1058.5 | 708.4 | 1.494 |
| turn 1 | full | 1 | 80000 | 753 | 16 | 1417.4 | 1070.5 | 1.324 |
| turn 4 | sliding | 4 | 80000 | 753 | 46 | 3280.2 | 2081.4 | 1.576 |
| turn 4 | full | 4 | 80000 | 753 | 16 | 4839.2 | 3510.1 | 1.379 |
| first turn | full | 1 | 0 | 16384 | 16 | 17959.9 | 10943.7 | 1.641 |
| first turn | sliding | 1 | 32768 | 16384 | 138 | 16810.0 | 10207.9 | 1.647 |
| first turn | full | 1 | 32768 | 16384 | 16 | 20729.4 | 13939.4 | 1.487 |
| first turn | full | 1 | 65536 | 16384 | 16 | 24530.1 | 17186.2 | 1.427 |

Tests: pass, pass, pass, pass, pass

## submit 1 (+ chunked boundaries, tuned split)  job=e8ee3b4002c24932  speedup=1.5011  CI=[1.4786, 1.5236]  score_us=258078  checks={'flagged': [], 'ok': True, 'rejected': []}
| load | layer | B | ctx | L | weight | SGLang us | ours us | x |
|---|---|---|---|---|---|---|---|---|
| turn 1 | sliding | 1 | 80000 | 753 | 46 | 1065.3 | 749.9 | 1.421 |
| turn 1 | full | 1 | 80000 | 753 | 16 | 1453.8 | 1110.3 | 1.309 |
| turn 4 | sliding | 4 | 80000 | 753 | 46 | 3292.2 | 2083.7 | 1.580 |
| turn 4 | full | 4 | 80000 | 753 | 16 | 4821.8 | 3522.4 | 1.369 |
| first turn | full | 1 | 0 | 16384 | 16 | 17871.6 | 10807.7 | 1.654 |
| first turn | sliding | 1 | 32768 | 16384 | 138 | 17244.4 | 10279.3 | 1.678 |
| first turn | full | 1 | 32768 | 16384 | 16 | 20928.9 | 13764.3 | 1.521 |
| first turn | full | 1 | 65536 | 16384 | 16 | 24309.0 | 16827.6 | 1.445 |

Tests: pass, pass, pass, pass, pass

## submit 2 (+ concurrent small-expert launch)  job=dd81791432fe4b81  speedup=1.5429  CI=[1.5197, 1.5660]  score_us=251094  checks={'flagged': [], 'ok': True, 'rejected': []}
| load | layer | B | ctx | L | weight | SGLang us | ours us | x |
|---|---|---|---|---|---|---|---|---|
| turn 1 | sliding | 1 | 80000 | 753 | 46 | 1049.6 | 693.7 | 1.513 |
| turn 1 | full | 1 | 80000 | 753 | 16 | 1484.9 | 1102.0 | 1.348 |
| turn 4 | sliding | 4 | 80000 | 753 | 46 | 3367.7 | 2069.9 | 1.627 |
| turn 4 | full | 4 | 80000 | 753 | 16 | 4938.5 | 3554.6 | 1.389 |
| first turn | full | 1 | 0 | 16384 | 16 | 18896.7 | 11168.9 | 1.692 |
| first turn | sliding | 1 | 32768 | 16384 | 138 | 16903.5 | 10017.6 | 1.687 |
| first turn | full | 1 | 32768 | 16384 | 16 | 21171.8 | 13793.6 | 1.535 |
| first turn | full | 1 | 65536 | 16384 | 16 | 25041.9 | 17078.5 | 1.466 |

Tests: pass, pass, pass, pass, pass

