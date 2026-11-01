# OKC judge results (H200), muse_bf16_proj

## check: first compressed kernel (v1)  job=6fbebdcb2321404c node=n-fwq5zu7sgu2oyb5nsz3gf5n23p  speedup=1.0244 CI=[1.009, 1.0398]
| load | layer | m | SGLang us | ours us | x |
|---|---|---|---|---|---|
| decode 1 | qkv_gate | 1 | 14.35 | 13.90 | 1.032 |
| decode 1 | o | 1 | 6.72 | 7.54 | 0.892 |
| decode 1 | pooler | 1 | 22.08 | 21.27 | 1.038 |
| decode 1 | unpooler | 1 | 7.07 | 7.42 | 0.954 |
| decode 1 | router | 1 | 10.27 | 6.59 | 1.558 |
| decode 16 | qkv_gate | 16 | 14.48 | 14.94 | 0.969 |
| decode 16 | o | 16 | 6.62 | 9.51 | 0.697 |
| decode 16 | pooler | 16 | 21.23 | 21.59 | 0.983 |
| decode 16 | unpooler | 16 | 18.50 | 22.58 | 0.819 |
| decode 16 | router | 16 | 14.85 | 8.00 | 1.856 |
| decode 32 | qkv_gate | 32 | 14.38 | 17.07 | 0.842 |
| decode 32 | o | 32 | 6.88 | 10.91 | 0.630 |
| decode 32 | pooler | 32 | 22.03 | 24.84 | 0.887 |
| decode 32 | unpooler | 32 | 18.93 | 26.86 | 0.705 |
| decode 32 | router | 32 | 15.46 | 11.21 | 1.379 |
| verify 1 | qkv_gate | 8 | 14.42 | 14.49 | 0.995 |
| verify 1 | o | 8 | 6.85 | 8.75 | 0.782 |
| verify 1 | pooler | 8 | 21.31 | 21.30 | 1.001 |
| verify 1 | unpooler | 8 | 7.23 | 7.82 | 0.925 |
| verify 1 | router | 8 | 26.43 | 7.23 | 3.657 |
| verify 16 | qkv_gate | 128 | 17.84 | 17.35 | 1.028 |
| verify 16 | o | 128 | 7.39 | 7.42 | 0.996 |
| verify 16 | pooler | 128 | 25.10 | 25.80 | 0.973 |
| verify 16 | unpooler | 128 | 20.11 | 20.15 | 0.998 |
| verify 16 | router | 128 | 16.42 | 16.31 | 1.006 |
| verify 32 | qkv_gate | 256 | 20.00 | 19.82 | 1.009 |
| verify 32 | o | 256 | 9.18 | 9.33 | 0.985 |
| verify 32 | pooler | 256 | 30.99 | 30.82 | 1.006 |
| verify 32 | unpooler | 256 | 24.54 | 24.42 | 1.005 |
| verify 32 | router | 256 | 17.47 | 17.27 | 1.012 |

## submit 1  job=7f8363a536d44a17 node=n-djsogg5ih26gwdcnhyhbnqzfkb  speedup=1.1728 CI=[1.1552, 1.1904]
| load | layer | m | SGLang us | ours us | x |
|---|---|---|---|---|---|
| decode 1 | qkv_gate | 1 | 14.22 | 12.83 | 1.108 |
| decode 1 | o | 1 | 6.90 | 6.27 | 1.099 |
| decode 1 | pooler | 1 | 21.09 | 16.82 | 1.254 |
| decode 1 | unpooler | 1 | 7.07 | 5.43 | 1.302 |
| decode 1 | router | 1 | 10.30 | 6.66 | 1.548 |
| decode 16 | qkv_gate | 16 | 13.30 | 13.77 | 0.966 |
| decode 16 | o | 16 | 6.66 | 6.83 | 0.974 |
| decode 16 | pooler | 16 | 21.28 | 18.83 | 1.130 |
| decode 16 | unpooler | 16 | 18.67 | 18.89 | 0.988 |
| decode 16 | router | 16 | 15.34 | 7.46 | 2.057 |
| decode 32 | qkv_gate | 32 | 14.29 | 14.37 | 0.994 |
| decode 32 | o | 32 | 6.93 | 7.05 | 0.983 |
| decode 32 | pooler | 32 | 21.66 | 21.79 | 0.994 |
| decode 32 | unpooler | 32 | 18.78 | 18.98 | 0.989 |
| decode 32 | router | 32 | 15.39 | 8.13 | 1.893 |
| verify 1 | qkv_gate | 8 | 14.16 | 13.53 | 1.047 |
| verify 1 | o | 8 | 7.07 | 7.26 | 0.975 |
| verify 1 | pooler | 8 | 21.38 | 17.79 | 1.202 |
| verify 1 | unpooler | 8 | 6.91 | 5.39 | 1.281 |
| verify 1 | router | 8 | 26.11 | 6.58 | 3.968 |
| verify 16 | qkv_gate | 128 | 16.93 | 17.01 | 0.995 |
| verify 16 | o | 128 | 7.33 | 7.44 | 0.985 |
| verify 16 | pooler | 128 | 25.66 | 26.09 | 0.984 |
| verify 16 | unpooler | 128 | 20.19 | 20.22 | 0.999 |
| verify 16 | router | 128 | 16.58 | 10.79 | 1.536 |
| verify 32 | qkv_gate | 256 | 19.70 | 19.95 | 0.987 |
| verify 32 | o | 256 | 9.38 | 9.40 | 0.998 |
| verify 32 | pooler | 256 | 30.53 | 31.03 | 0.984 |
| verify 32 | unpooler | 256 | 24.05 | 24.39 | 0.986 |
| verify 32 | router | 256 | 17.52 | 11.40 | 1.537 |

## submit 2  job=dfffd1b41a394753 node=n-hg25ugy45aa3trrcfa4kf5iljz  speedup=1.1361 CI=[1.119, 1.1531]
| load | layer | m | SGLang us | ours us | x |
|---|---|---|---|---|---|
| decode 1 | qkv_gate | 1 | 14.06 | 14.43 | 0.975 |
| decode 1 | o | 1 | 6.90 | 6.57 | 1.049 |
| decode 1 | pooler | 1 | 21.46 | 17.70 | 1.212 |
| decode 1 | unpooler | 1 | 6.75 | 5.56 | 1.213 |
| decode 1 | router | 1 | 10.35 | 7.76 | 1.333 |
| decode 16 | qkv_gate | 16 | 13.92 | 14.38 | 0.968 |
| decode 16 | o | 16 | 7.01 | 7.00 | 1.001 |
| decode 16 | pooler | 16 | 21.50 | 19.07 | 1.128 |
| decode 16 | unpooler | 16 | 18.61 | 18.50 | 1.006 |
| decode 16 | router | 16 | 14.85 | 8.72 | 1.702 |
| decode 32 | qkv_gate | 32 | 14.32 | 14.68 | 0.976 |
| decode 32 | o | 32 | 6.66 | 6.80 | 0.979 |
| decode 32 | pooler | 32 | 21.84 | 21.51 | 1.015 |
| decode 32 | unpooler | 32 | 18.85 | 18.98 | 0.993 |
| decode 32 | router | 32 | 15.20 | 9.62 | 1.580 |
| verify 1 | qkv_gate | 8 | 14.11 | 15.29 | 0.923 |
| verify 1 | o | 8 | 6.75 | 7.14 | 0.946 |
| verify 1 | pooler | 8 | 21.52 | 18.54 | 1.161 |
| verify 1 | unpooler | 8 | 7.01 | 6.16 | 1.138 |
| verify 1 | router | 8 | 26.34 | 8.12 | 3.244 |
| verify 16 | qkv_gate | 128 | 16.90 | 17.27 | 0.979 |
| verify 16 | o | 128 | 7.30 | 7.34 | 0.995 |
| verify 16 | pooler | 128 | 24.83 | 25.14 | 0.988 |
| verify 16 | unpooler | 128 | 20.30 | 20.08 | 1.011 |
| verify 16 | router | 128 | 16.27 | 10.19 | 1.597 |
| verify 32 | qkv_gate | 256 | 19.62 | 19.71 | 0.995 |
| verify 32 | o | 256 | 9.44 | 9.44 | 1.000 |
| verify 32 | pooler | 256 | 30.78 | 31.10 | 0.990 |
| verify 32 | unpooler | 256 | 24.18 | 24.04 | 1.006 |
| verify 32 | router | 256 | 17.63 | 11.21 | 1.573 |

## submit 3  job=7dc55ce4438c4006 node=n-wsbc7j2d6xnjjdziaz2kmgurhe  speedup=1.1334 CI=[1.1164, 1.1504]
| load | layer | m | SGLang us | ours us | x |
|---|---|---|---|---|---|
| decode 1 | qkv_gate | 1 | 14.06 | 14.56 | 0.966 |
| decode 1 | o | 1 | 6.91 | 6.73 | 1.028 |
| decode 1 | pooler | 1 | 21.54 | 17.74 | 1.214 |
| decode 1 | unpooler | 1 | 6.94 | 5.56 | 1.249 |
| decode 1 | router | 1 | 10.43 | 7.89 | 1.322 |
| decode 16 | qkv_gate | 16 | 13.87 | 14.05 | 0.987 |
| decode 16 | o | 16 | 6.90 | 7.79 | 0.885 |
| decode 16 | pooler | 16 | 21.31 | 18.90 | 1.128 |
| decode 16 | unpooler | 16 | 18.58 | 18.59 | 0.999 |
| decode 16 | router | 16 | 15.28 | 9.01 | 1.696 |
| decode 32 | qkv_gate | 32 | 14.37 | 14.94 | 0.961 |
| decode 32 | o | 32 | 6.78 | 6.90 | 0.984 |
| decode 32 | pooler | 32 | 22.11 | 21.87 | 1.011 |
| decode 32 | unpooler | 32 | 18.85 | 18.96 | 0.994 |
| decode 32 | router | 32 | 15.25 | 9.66 | 1.578 |
| verify 1 | qkv_gate | 8 | 13.92 | 15.30 | 0.910 |
| verify 1 | o | 8 | 6.94 | 7.19 | 0.966 |
| verify 1 | pooler | 8 | 21.58 | 18.40 | 1.173 |
| verify 1 | unpooler | 8 | 7.04 | 5.88 | 1.198 |
| verify 1 | router | 8 | 26.37 | 8.38 | 3.148 |
| verify 16 | qkv_gate | 128 | 17.28 | 17.00 | 1.017 |
| verify 16 | o | 128 | 7.54 | 7.46 | 1.010 |
| verify 16 | pooler | 128 | 25.74 | 26.18 | 0.983 |
| verify 16 | unpooler | 128 | 19.81 | 19.82 | 0.999 |
| verify 16 | router | 128 | 16.18 | 11.08 | 1.460 |
| verify 32 | qkv_gate | 256 | 19.70 | 19.44 | 1.013 |
| verify 32 | o | 256 | 9.47 | 9.38 | 1.010 |
| verify 32 | pooler | 256 | 30.19 | 30.16 | 1.001 |
| verify 32 | unpooler | 256 | 24.32 | 24.18 | 1.006 |
| verify 32 | router | 256 | 17.65 | 11.48 | 1.537 |

## submit 4  job=6cf17e9c38f24c60 node=n-djsogg5ih26gwdcnhyhbnqzfkb  speedup=1.1933 CI=[1.1754, 1.2112]
| load | layer | m | SGLang us | ours us | x |
|---|---|---|---|---|---|
| decode 1 | qkv_gate | 1 | 13.98 | 11.56 | 1.209 |
| decode 1 | o | 1 | 6.69 | 6.55 | 1.021 |
| decode 1 | pooler | 1 | 21.86 | 17.24 | 1.267 |
| decode 1 | unpooler | 1 | 7.04 | 5.78 | 1.218 |
| decode 1 | router | 1 | 10.40 | 6.45 | 1.612 |
| decode 16 | qkv_gate | 16 | 13.89 | 13.31 | 1.043 |
| decode 16 | o | 16 | 6.56 | 7.51 | 0.874 |
| decode 16 | pooler | 16 | 21.70 | 19.16 | 1.132 |
| decode 16 | unpooler | 16 | 18.56 | 17.94 | 1.035 |
| decode 16 | router | 16 | 14.90 | 7.22 | 2.063 |
| decode 32 | qkv_gate | 32 | 14.08 | 14.13 | 0.997 |
| decode 32 | o | 32 | 7.07 | 6.97 | 1.015 |
| decode 32 | pooler | 32 | 22.10 | 21.63 | 1.022 |
| decode 32 | unpooler | 32 | 18.50 | 18.48 | 1.001 |
| decode 32 | router | 32 | 15.06 | 8.05 | 1.870 |
| verify 1 | qkv_gate | 8 | 13.81 | 11.69 | 1.181 |
| verify 1 | o | 8 | 6.72 | 7.12 | 0.943 |
| verify 1 | pooler | 8 | 21.58 | 17.39 | 1.241 |
| verify 1 | unpooler | 8 | 6.98 | 5.83 | 1.196 |
| verify 1 | router | 8 | 26.37 | 6.72 | 3.926 |
| verify 16 | qkv_gate | 128 | 16.94 | 16.91 | 1.002 |
| verify 16 | o | 128 | 7.22 | 7.30 | 0.989 |
| verify 16 | pooler | 128 | 25.36 | 25.42 | 0.998 |
| verify 16 | unpooler | 128 | 20.30 | 20.17 | 1.006 |
| verify 16 | router | 128 | 16.46 | 10.44 | 1.577 |
| verify 32 | qkv_gate | 256 | 19.98 | 19.99 | 1.000 |
| verify 32 | o | 256 | 9.38 | 9.49 | 0.988 |
| verify 32 | pooler | 256 | 30.91 | 30.72 | 1.006 |
| verify 32 | unpooler | 256 | 24.75 | 24.59 | 1.007 |
| verify 32 | router | 256 | 17.73 | 11.26 | 1.575 |

## submit 5  job=8f7525605d264b82 node=n-wsbc7j2d6xnjjdziaz2kmgurhe  speedup=1.2005 CI=[1.1825, 1.2185]
| load | layer | m | SGLang us | ours us | x |
|---|---|---|---|---|---|
| decode 1 | qkv_gate | 1 | 13.90 | 11.47 | 1.212 |
| decode 1 | o | 1 | 6.78 | 6.32 | 1.073 |
| decode 1 | pooler | 1 | 21.04 | 17.06 | 1.233 |
| decode 1 | unpooler | 1 | 6.98 | 5.34 | 1.306 |
| decode 1 | router | 1 | 10.43 | 6.20 | 1.684 |
| decode 16 | qkv_gate | 16 | 13.86 | 12.79 | 1.083 |
| decode 16 | o | 16 | 6.98 | 6.95 | 1.003 |
| decode 16 | pooler | 16 | 20.94 | 18.38 | 1.139 |
| decode 16 | unpooler | 16 | 18.59 | 17.81 | 1.044 |
| decode 16 | router | 16 | 15.01 | 7.09 | 2.116 |
| decode 32 | qkv_gate | 32 | 14.24 | 14.59 | 0.976 |
| decode 32 | o | 32 | 6.90 | 6.97 | 0.990 |
| decode 32 | pooler | 32 | 21.46 | 21.59 | 0.994 |
| decode 32 | unpooler | 32 | 18.93 | 18.88 | 1.002 |
| decode 32 | router | 32 | 15.15 | 7.75 | 1.955 |
| verify 1 | qkv_gate | 8 | 14.14 | 12.15 | 1.164 |
| verify 1 | o | 8 | 6.72 | 6.65 | 1.011 |
| verify 1 | pooler | 8 | 20.98 | 17.52 | 1.197 |
| verify 1 | unpooler | 8 | 7.07 | 5.87 | 1.205 |
| verify 1 | router | 8 | 26.32 | 6.44 | 4.086 |
| verify 16 | qkv_gate | 128 | 16.99 | 17.03 | 0.998 |
| verify 16 | o | 128 | 7.33 | 7.17 | 1.022 |
| verify 16 | pooler | 128 | 25.66 | 25.87 | 0.992 |
| verify 16 | unpooler | 128 | 19.92 | 19.82 | 1.005 |
| verify 16 | router | 128 | 16.24 | 10.02 | 1.621 |
| verify 32 | qkv_gate | 256 | 19.63 | 19.50 | 1.007 |
| verify 32 | o | 256 | 9.44 | 9.32 | 1.013 |
| verify 32 | pooler | 256 | 29.09 | 29.26 | 0.994 |
| verify 32 | unpooler | 256 | 24.46 | 24.23 | 1.010 |
| verify 32 | router | 256 | 17.44 | 11.09 | 1.573 |

## submit 6 (final)  job=6249a4554d96493f node=n-hg25ugy45aa3trrcfa4kf5iljz  speedup=1.2021 CI=[1.1841, 1.2202]
| load | layer | m | SGLang us | ours us | x |
|---|---|---|---|---|---|
| decode 1 | qkv_gate | 1 | 14.18 | 11.76 | 1.206 |
| decode 1 | o | 1 | 6.62 | 6.35 | 1.044 |
| decode 1 | pooler | 1 | 21.06 | 16.93 | 1.244 |
| decode 1 | unpooler | 1 | 7.04 | 5.63 | 1.252 |
| decode 1 | router | 1 | 10.37 | 6.30 | 1.645 |
| decode 16 | qkv_gate | 16 | 14.14 | 13.06 | 1.083 |
| decode 16 | o | 16 | 6.82 | 6.88 | 0.991 |
| decode 16 | pooler | 16 | 21.18 | 18.25 | 1.161 |
| decode 16 | unpooler | 16 | 18.67 | 17.78 | 1.050 |
| decode 16 | router | 16 | 14.64 | 6.85 | 2.136 |
| decode 32 | qkv_gate | 32 | 14.24 | 14.04 | 1.014 |
| decode 32 | o | 32 | 7.10 | 7.06 | 1.006 |
| decode 32 | pooler | 32 | 21.82 | 21.57 | 1.012 |
| decode 32 | unpooler | 32 | 18.58 | 18.54 | 1.002 |
| decode 32 | router | 32 | 15.20 | 7.85 | 1.935 |
| verify 1 | qkv_gate | 8 | 14.54 | 12.31 | 1.182 |
| verify 1 | o | 8 | 6.53 | 6.56 | 0.995 |
| verify 1 | pooler | 8 | 21.41 | 17.26 | 1.240 |
| verify 1 | unpooler | 8 | 7.07 | 5.81 | 1.218 |
| verify 1 | router | 8 | 26.06 | 6.23 | 4.184 |
| verify 16 | qkv_gate | 128 | 16.94 | 16.67 | 1.016 |
| verify 16 | o | 128 | 7.52 | 7.57 | 0.994 |
| verify 16 | pooler | 128 | 25.28 | 25.58 | 0.988 |
| verify 16 | unpooler | 128 | 19.74 | 20.17 | 0.979 |
| verify 16 | router | 128 | 16.58 | 10.61 | 1.562 |
| verify 32 | qkv_gate | 256 | 19.87 | 19.70 | 1.009 |
| verify 32 | o | 256 | 9.10 | 9.29 | 0.980 |
| verify 32 | pooler | 256 | 30.74 | 30.53 | 1.007 |
| verify 32 | unpooler | 256 | 24.29 | 24.42 | 0.995 |
| verify 32 | router | 256 | 17.52 | 11.27 | 1.554 |
