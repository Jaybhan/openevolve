# R3 attack matrix (verified branch)

| attack | kind | V0 table | VR (R1) | VR, GT labels | VR, lower-bound labels | F1 excess>2k | F2 tie tail | F3 a_I from lower bound, Wmin 1e6 | F4 GEN Wmin 1e5 | VR* = F1+F2+F3+F4 | F1b excess>20k | VR** = F1b+F3 | VR** on GT labels | VR* on GT labels | work removed (lb, train+target) | of which target |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref: schema_recipe_frozen | real | 0.2594 | 0.3403 | 0.2998 | 0.3000 | 0.3392 | 0.3456 | 0.2790 | 0.3449 | 0.2835 | 0.3263 | 0.2619 | 0.2484 | 0.2623 | 4.79e+07 | 3.77e+07 |
| ref: dgh_lib | real | 0.2000 | 0.3182 | 0.3928 | 0.3931 | 0.3171 | 0.3181 | 0.3863 | 0.3204 | 0.3860 | 0.3029 | 0.3836 | 0.3929 | 0.3940 | 1.9e+07 | 6.33e+05 |
| ref: recipe_dgh | real | 0.2594 | 0.3634 | 0.4303 | 0.4308 | 0.3622 | 0.3686 | 0.4285 | 0.3702 | 0.4330 | 0.3467 | 0.4139 | 0.4180 | 0.4322 | 6.69e+07 | 3.83e+07 |
| ref: f_weak | real | 0.2286 | 0.2226 | 0.2227 | 0.2220 | 0.2212 | 0.2249 | 0.2219 | 0.2241 | 0.2230 | 0.2180 | 0.2173 | 0.2181 | 0.2214 | 2.08e+07 | 1.98e+07 |
| ref: cert_target_12_18 | real | 0.2099 | 0.2117 | 0.2117 | 0.2083 | 0.2117 | 0.2126 | 0.2105 | 0.2117 | 0.2114 | 0.2117 | 0.2105 | 0.2105 | 0.2110 | 7.57e+06 | 7.57e+06 |
| r1: schema_pool | real | 0.2618 | 0.3487 | 0.3072 | 0.3075 | 0.3476 | 0.3540 | 0.2875 | 0.3533 | 0.2918 | 0.3346 | 0.2701 | 0.2551 | 0.2693 | 6.09e+07 | 3.77e+07 |
| r1: cert_one_train | real | 0.2222 | 0.2201 | 0.2201 | 0.2201 | 0.2165 | 0.2201 | 0.2050 | 0.2210 | 0.2057 | 0.2000 | 0.2000 | 0.2000 | 0.2057 | 6.54e+04 | 0 |
| r1: inst_dgh_10_23 | real | 0.2000 | 0.2037 | 0.2037 | 0.2029 | 0.2037 | 0.2036 | 0.2030 | 0.2037 | 0.2029 | 0.2037 | 0.2030 | 0.2030 | 0.2029 | 6.33e+05 | 6.33e+05 |
| r1: e1_dgh_s2 | real | 0.2000 | 0.2021 | 0.2021 | 0.2021 | 0.2000 | 0.2021 | 0.2021 | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0 | 0 |
| r1: e2_easy_certs | real | 0.2068 | 0.2076 | 0.2075 | 0.2075 | 0.2008 | 0.2076 | 0.2045 | 0.2066 | 0.2008 | 0.2003 | 0.2003 | 0.2003 | 0.2008 | 4.54e+05 | 0 |
| r1: e3_relist | real | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0 | 0 |
| A1 clear GEN (9,9;4,4) | oracle | 0.2800 | 0.2539 | 0.2539 | 0.2539 | 0.2000 | 0.2539 | 0.2539 | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0 | 0 |
| A2 clear (9,12,64) | oracle | 0.2000 | 0.2406 | 0.2403 | 0.2404 | 0.2041 | 0.2406 | 0.2084 | 0.2423 | 0.2011 | 0.2000 | 0.2000 | 0.2000 | 0.2010 | 9.61e+03 | 0 |
| A2 clear (10,10,61) | oracle | 0.2635 | 0.2382 | 0.2382 | 0.2382 | 0.2064 | 0.2382 | 0.2056 | 0.2392 | 0.2013 | 0.2000 | 0.2000 | 0.2000 | 0.2013 | 9.86e+03 | 0 |
| A3 clear every table with W_lb<1e5 | oracle | 0.4705 | 0.3929 | 0.3531 | 0.3545 | 0.3130 | 0.3929 | 0.2949 | 0.3423 | 0.2152 | 0.3066 | 0.2105 | 0.2043 | 0.2111 | 1.12e+05 | 5.83e+04 |
| A5 real: pool certs on small tables only | real | 0.2428 | 0.3187 | 0.2791 | 0.2803 | 0.3141 | 0.3187 | 0.2207 | 0.3189 | 0.2163 | 0.3066 | 0.2105 | 0.2043 | 0.2123 | 1.32e+05 | 5.83e+04 |
| A4 real: pool certs + DGH on small tables only | real | 0.2428 | 0.3375 | 0.4057 | 0.4073 | 0.3319 | 0.3375 | 0.4026 | 0.3391 | 0.3973 | 0.3228 | 0.3900 | 0.3940 | 0.4018 | 1.51e+07 | 5.83e+04 |
| B easy exact d<=2000 | oracle | 0.4100 | 0.2966 | 0.2963 | 0.2963 | 0.2000 | 0.2966 | 0.2820 | 0.2457 | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 3.09e+05 | 1.41e+04 |
| B easy exact d<=5000 | oracle | 0.5109 | 0.3422 | 0.3414 | 0.3417 | 0.2274 | 0.3422 | 0.3077 | 0.2945 | 0.2161 | 0.2000 | 0.2000 | 0.2000 | 0.2158 | 1.07e+06 | 6.81e+04 |
| B easy exact d<=20000 | oracle | 0.5893 | 0.4136 | 0.4100 | 0.4110 | 0.3289 | 0.4136 | 0.3671 | 0.3734 | 0.2908 | 0.2000 | 0.2000 | 0.2000 | 0.2857 | 7.79e+06 | 1.2e+06 |
| B4 real: pool certs killing only exact d<=20k | real | 0.2489 | 0.2272 | 0.2272 | 0.2272 | 0.2184 | 0.2272 | 0.2182 | 0.2279 | 0.2112 | 0.2000 | 0.2000 | 0.2000 | 0.2112 | 2.83e+05 | 0 |
| C1 real: cert_close_10_22 | real | 0.2000 | 0.3087 | 0.2691 | 0.2702 | 0.3084 | 0.3087 | 0.2106 | 0.3087 | 0.2106 | 0.3066 | 0.2105 | 0.2043 | 0.2066 | 5.83e+04 | 5.83e+04 |
| C2 tail-index sniper (targets) | oracle | 0.2610 | 0.3222 | 0.2657 | 0.2491 | 0.3253 | 0.2779 | 0.2865 | 0.3222 | 0.2426 | 0.3353 | 0.2979 | 0.2671 | 0.2478 | 6.17e+07 | 6.17e+07 |
| C3a deepened censored, EASY half | oracle | 0.2672 | 0.2572 | 0.2297 | 0.2728 | 0.2572 | 0.2585 | 0.2599 | 0.2572 | 0.2613 | 0.2573 | 0.2603 | 0.2308 | 0.2257 | 9.67e+07 | 9.67e+07 |
| C3b deepened censored, HARD half | oracle | 0.2785 | 0.2604 | 0.3303 | 0.3855 | 0.2605 | 0.2592 | 0.2632 | 0.2604 | 0.2621 | 0.2606 | 0.2636 | 0.3362 | 0.3173 | 3.01e+08 | 3.01e+08 |
| D1 random 10% of m16_n17_w134 | oracle | 0.2000 | 0.2178 | 0.2173 | 0.2145 | 0.2180 | 0.2177 | 0.2169 | 0.2178 | 0.2170 | 0.2183 | 0.2175 | 0.2166 | 0.2165 | 1.26e+07 | 1.26e+07 |
| D1 random 10% of m13_n19_w123 | oracle | 0.2000 | 0.2162 | 0.2176 | 0.2232 | 0.2163 | 0.2160 | 0.2145 | 0.2162 | 0.2143 | 0.2165 | 0.2148 | 0.2160 | 0.2153 | 2.59e+07 | 2.59e+07 |
| D1 random 10% of m12_n18_w109 | oracle | 0.2140 | 0.2155 | 0.2158 | 0.2136 | 0.2155 | 0.2154 | 0.2141 | 0.2155 | 0.2140 | 0.2158 | 0.2144 | 0.2144 | 0.2142 | 1.23e+07 | 1.23e+07 |
| D1 random 10% of m9_n23_w104 | oracle | 0.2118 | 0.2102 | 0.2124 | 0.2128 | 0.2102 | 0.2108 | 0.2107 | 0.2102 | 0.2113 | 0.2102 | 0.2107 | 0.2125 | 0.2125 | 6.96e+06 | 6.96e+06 |
| G tail top-decile of tables with <=30 survivors | oracle | 0.3148 | 0.2896 | 0.2543 | 0.2553 | 0.2658 | 0.2872 | 0.2525 | 0.2751 | 0.2258 | 0.2625 | 0.2236 | 0.2182 | 0.2230 | 2.05e+06 | 3.42e+04 |

## R1 criteria per fix (this attack set)

```
{
 "VR (R1)": {
  "recipe": 0.3403,
  "dgh": 0.3182,
  "recipe_dgh": 0.3634,
  "C2 dgh/recipe uplift": 0.84,
  "C5 recipe uplift / V0": 2.36,
  "C3 real exploits >= genuine min": {},
  "exploit-shape oracles >= genuine min": {
   "A3 clear every table with W_lb<1e5": 0.3929,
   "B easy exact d<=5000": 0.3422,
   "B easy exact d<=20000": 0.4136
  },
  "tiny-table clears > f_weak": {
   "A1 clear GEN (9,9;4,4)": 0.2539,
   "A2 clear (9,12,64)": 0.2406,
   "A2 clear (10,10,61)": 0.2382
  }
 },
 "VR, GT labels": {
  "recipe": 0.2998,
  "dgh": 0.3928,
  "recipe_dgh": 0.4303,
  "C2 dgh/recipe uplift": 1.93,
  "C5 recipe uplift / V0": 1.68,
  "C3 real exploits >= genuine min": {},
  "exploit-shape oracles >= genuine min": {
   "A3 clear every table with W_lb<1e5": 0.3531,
   "B easy exact d<=5000": 0.3414,
   "B easy exact d<=20000": 0.41
  },
  "tiny-table clears > f_weak": {
   "A1 clear GEN (9,9;4,4)": 0.2539,
   "A2 clear (9,12,64)": 0.2403,
   "A2 clear (10,10,61)": 0.2382
  }
 },
 "VR, lower-bound labels": {
  "recipe": 0.3,
  "dgh": 0.3931,
  "recipe_dgh": 0.4308,
  "C2 dgh/recipe uplift": 1.93,
  "C5 recipe uplift / V0": 1.68,
  "C3 real exploits >= genuine min": {},
  "exploit-shape oracles >= genuine min": {
   "A3 clear every table with W_lb<1e5": 0.3545,
   "B easy exact d<=5000": 0.3417,
   "B easy exact d<=20000": 0.411
  },
  "tiny-table clears > f_weak": {
   "A1 clear GEN (9,9;4,4)": 0.2539,
   "A2 clear (9,12,64)": 0.2404,
   "A2 clear (10,10,61)": 0.2382
  }
 },
 "F1 excess>2k": {
  "recipe": 0.3392,
  "dgh": 0.3171,
  "recipe_dgh": 0.3622,
  "C2 dgh/recipe uplift": 0.84,
  "C5 recipe uplift / V0": 2.34,
  "C3 real exploits >= genuine min": {},
  "exploit-shape oracles >= genuine min": {
   "B easy exact d<=20000": 0.3289
  },
  "tiny-table clears > f_weak": {}
 },
 "F2 tie tail": {
  "recipe": 0.3456,
  "dgh": 0.3181,
  "recipe_dgh": 0.3686,
  "C2 dgh/recipe uplift": 0.81,
  "C5 recipe uplift / V0": 2.45,
  "C3 real exploits >= genuine min": {},
  "exploit-shape oracles >= genuine min": {
   "A3 clear every table with W_lb<1e5": 0.3929,
   "B easy exact d<=5000": 0.3422,
   "B easy exact d<=20000": 0.4136
  },
  "tiny-table clears > f_weak": {
   "A1 clear GEN (9,9;4,4)": 0.2539,
   "A2 clear (9,12,64)": 0.2406,
   "A2 clear (10,10,61)": 0.2382
  }
 },
 "F3 a_I from lower bound, Wmin 1e6": {
  "recipe": 0.279,
  "dgh": 0.3863,
  "recipe_dgh": 0.4285,
  "C2 dgh/recipe uplift": 2.36,
  "C5 recipe uplift / V0": 1.33,
  "C3 real exploits >= genuine min": {},
  "exploit-shape oracles >= genuine min": {
   "A3 clear every table with W_lb<1e5": 0.2949,
   "B easy exact d<=2000": 0.282,
   "B easy exact d<=5000": 0.3077,
   "B easy exact d<=20000": 0.3671
  },
  "tiny-table clears > f_weak": {
   "A1 clear GEN (9,9;4,4)": 0.2539
  }
 },
 "F4 GEN Wmin 1e5": {
  "recipe": 0.3449,
  "dgh": 0.3204,
  "recipe_dgh": 0.3702,
  "C2 dgh/recipe uplift": 0.83,
  "C5 recipe uplift / V0": 2.44,
  "C3 real exploits >= genuine min": {},
  "exploit-shape oracles >= genuine min": {
   "A3 clear every table with W_lb<1e5": 0.3423,
   "B easy exact d<=20000": 0.3734
  },
  "tiny-table clears > f_weak": {
   "A2 clear (9,12,64)": 0.2423,
   "A2 clear (10,10,61)": 0.2392
  }
 },
 "VR* = F1+F2+F3+F4": {
  "recipe": 0.2835,
  "dgh": 0.386,
  "recipe_dgh": 0.433,
  "C2 dgh/recipe uplift": 2.23,
  "C5 recipe uplift / V0": 1.4,
  "C3 real exploits >= genuine min": {},
  "exploit-shape oracles >= genuine min": {
   "B easy exact d<=20000": 0.2908
  },
  "tiny-table clears > f_weak": {}
 },
 "F1b excess>20k": {
  "recipe": 0.3263,
  "dgh": 0.3029,
  "recipe_dgh": 0.3467,
  "C2 dgh/recipe uplift": 0.81,
  "C5 recipe uplift / V0": 2.13,
  "C3 real exploits >= genuine min": {},
  "exploit-shape oracles >= genuine min": {
   "A3 clear every table with W_lb<1e5": 0.3066
  },
  "tiny-table clears > f_weak": {}
 },
 "VR** = F1b+F3": {
  "recipe": 0.2619,
  "dgh": 0.3836,
  "recipe_dgh": 0.4139,
  "C2 dgh/recipe uplift": 2.97,
  "C5 recipe uplift / V0": 1.04,
  "C3 real exploits >= genuine min": {},
  "exploit-shape oracles >= genuine min": {},
  "tiny-table clears > f_weak": {}
 },
 "VR** on GT labels": {
  "recipe": 0.2484,
  "dgh": 0.3929,
  "recipe_dgh": 0.418,
  "C2 dgh/recipe uplift": 3.99,
  "C5 recipe uplift / V0": 0.81,
  "C3 real exploits >= genuine min": {},
  "exploit-shape oracles >= genuine min": {},
  "tiny-table clears > f_weak": {}
 },
 "VR* on GT labels": {
  "recipe": 0.2623,
  "dgh": 0.394,
  "recipe_dgh": 0.4322,
  "C2 dgh/recipe uplift": 3.12,
  "C5 recipe uplift / V0": 1.05,
  "C3 real exploits >= genuine min": {},
  "exploit-shape oracles >= genuine min": {
   "B easy exact d<=20000": 0.2857
  },
  "tiny-table clears > f_weak": {}
 }
}
```

## VR components (table labels)

| attack | G_train | G_target | G_gen | Tail | Depth | Close |
|---|---|---|---|---|---|---|
| ref: schema_recipe_frozen | 0.082 | 0.117 | 0.000 | 0.037 | 0.409 | 0.409 |
| ref: dgh_lib | 0.078 | 0.003 | 0.038 | 0.027 | 0.396 | 0.396 |
| ref: recipe_dgh | 0.160 | 0.120 | 0.038 | 0.064 | 0.409 | 0.409 |
| ref: f_weak | 0.027 | 0.046 | 0.000 | 0.000 | 0.074 | 0.000 |
| ref: cert_target_12_18 | 0.000 | 0.013 | 0.000 | 0.000 | 0.079 | 0.000 |
| r1: schema_pool | 0.110 | 0.117 | 0.029 | 0.042 | 0.409 | 0.409 |
| r1: cert_one_train | 0.017 | 0.000 | 0.000 | 0.010 | 0.126 | 0.000 |
| r1: inst_dgh_10_23 | 0.000 | 0.003 | 0.000 | 0.003 | 0.025 | 0.000 |
| r1: e1_dgh_s2 | 0.000 | 0.000 | 0.038 | 0.000 | 0.000 | 0.000 |
| r1: e2_easy_certs | 0.011 | 0.000 | 0.029 | 0.000 | 0.030 | 0.000 |
| r1: e3_relist | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| A1 clear GEN (9,9;4,4) | 0.000 | 0.000 | 0.962 | 0.000 | 0.000 | 0.000 |
| A2 clear (9,12,64) | 0.031 | 0.000 | 0.000 | 0.013 | 0.134 | 0.134 |
| A2 clear (10,10,61) | 0.018 | 0.000 | 0.000 | 0.013 | 0.136 | 0.136 |
| A3 clear every table with W_lb<1e5 | 0.096 | 0.046 | 1.000 | 0.086 | 0.409 | 0.409 |
| A5 real: pool certs on small tables only | 0.033 | 0.046 | 0.029 | 0.035 | 0.409 | 0.409 |
| A4 real: pool certs + DGH on small tables only | 0.095 | 0.046 | 0.067 | 0.060 | 0.409 | 0.409 |
| B easy exact d<=2000 | 0.091 | 0.000 | 1.000 | 0.011 | 0.158 | 0.000 |
| B easy exact d<=5000 | 0.149 | 0.000 | 1.000 | 0.060 | 0.205 | 0.179 |
| B easy exact d<=20000 | 0.282 | 0.003 | 1.000 | 0.104 | 0.343 | 0.343 |
| B4 real: pool certs killing only exact d<=20k | 0.041 | 0.000 | 0.029 | 0.011 | 0.126 | 0.000 |
| C1 real: cert_close_10_22 | 0.000 | 0.046 | 0.000 | 0.025 | 0.409 | 0.409 |
| C2 tail-index sniper (targets) | 0.000 | 0.141 | 0.000 | 0.466 | 0.385 | 0.000 |
| C3a deepened censored, EASY half | 0.000 | 0.062 | 0.000 | 0.017 | 0.373 | 0.000 |
| C3b deepened censored, HARD half | 0.000 | 0.063 | 0.000 | 0.044 | 0.375 | 0.000 |
| D1 random 10% of m16_n17_w134 | 0.000 | 0.033 | 0.000 | 0.010 | 0.093 | 0.000 |
| D1 random 10% of m13_n19_w123 | 0.000 | 0.017 | 0.000 | 0.012 | 0.100 | 0.000 |
| D1 random 10% of m12_n18_w109 | 0.000 | 0.016 | 0.000 | 0.011 | 0.096 | 0.000 |
| D1 random 10% of m9_n23_w104 | 0.000 | 0.008 | 0.000 | 0.000 | 0.073 | 0.000 |
| G tail top-decile of tables with <=30 survivors | 0.033 | 0.044 | 0.292 | 0.110 | 0.385 | 0.000 |

## Suite without A1's *_pure_gt tables (VR, table labels)

{"clear (9,12,64)": 0.2485, "recipe": 0.3549, "dgh": 0.3363}

## Marginal score per 1e6 conflicts removed (uniform 1% gain on one table)

| table | W (table) | VR | VR* | VR** |
|---|---|---|---|---|
| m8_n9_s2_t2_w27_pure | 150 | 14.2 | 0 | 0 |
| m9_n12_s3_t3_w64_pure | 9.61e+03 | 2.4 | 6.71 | 0 |
| m10_n10_s3_t3_w61_pure | 9.86e+03 | 2.07 | 5.55 | 0 |
| m9_n9_s4_t4_w62_pure | 1.05e+04 | 5.14 | 0 | 0 |
| m9_n9_s3_t3_w50_pure | 1.55e+04 | 1.61 | 5.31 | 0 |
| m9_n10_s3_t3_w55_pure | 1.89e+04 | 1.42 | 6.05 | 0 |
| m10_n11_s3_t3_w65_pure | 1.78e+05 | 0.289 | 0.598 | 0 |
| m11_n21_s3_t3_w117_pure | 3.6e+05 | 0.166 | 0.182 | 0.306 |
| m10_n22_s3_t3_w111 | 4.24e+05 | 0.134 | 0.135 | 0.147 |
| m11_n11_s3_t3_w70_pure | 1.44e+06 | 0.0523 | 0.0738 | 0.251 |
| m11_n12_s3_t3_w75_pure | 2.65e+06 | 0.0311 | 0.0403 | 0.129 |
| m10_n14_s3_t3_w78_pure | 2.99e+06 | 0.032 | 0.0408 | 0 |
| m10_n20_s3_t3_w103_pure | 3.21e+06 | 0.0264 | 0.0288 | 0.0447 |
| m12_n12_s3_t3_w81_pure | 4.52e+06 | 0.0196 | 0.0231 | 0.0403 |
| m12_n13_s3_t3_w87_pure | 5e+06 | 0 | 0 | 0 |
| m13_n13_s3_t3_w93_pure | 1.24e+07 | 0 | 0 | 0 |
| m9_n18_s3_t3_w86_pure_gt | 1.98e+07 | 0.00535 | 0.0057 | 0.00647 |
| m9_n16_s3_t3_w78_pure_gt | 3.67e+07 | 0.00351 | 0.00404 | 0.00707 |
| m9_n23_s3_t3_w104 | 3.68e+07 | 0.00283 | 0.00284 | 0.00297 |
| m10_n19_s3_t3_w99_pure_gt | 4.97e+07 | 0.00234 | 0.00248 | 0.00263 |
| m11_n23_s3_t3_w124 | 1.34e+08 | 0.000876 | 0.00088 | 0.000919 |
| m16_n17_s3_t3_w134 | 3.97e+08 | 0.000422 | 0.000424 | 0.000443 |
| m10_n23_s3_t3_w113 | 4.75e+08 | 0.000276 | 0.000277 | 0.00029 |
| m12_n18_s3_t3_w109 | 5.9e+08 | 0.000242 | 0.000243 | 0.000254 |
| m13_n19_s3_t3_w123 | 1.03e+09 | 0.000144 | 0.000145 | 0.000151 |

## Tail tie-break on targets (deepened censored cases; lower-bound true d)

```
{
 "m12_n18_s3_t3_w109": {
  "H_size": 152,
  "H_index_max": 175,
  "deepened_in_H": 7,
  "median_lb_H": 2000001.0,
  "median_lb_rest": 826632.0,
  "deepened_rest": 81
 },
 "m9_n23_s3_t3_w104": {
  "H_size": 10,
  "H_index_max": 9,
  "deepened_in_H": 10,
  "median_lb_H": 1638122.0,
  "median_lb_rest": 465749.0,
  "deepened_rest": 82
 },
 "m10_n22_s3_t3_w111": {
  "H_size": 1,
  "H_index_max": 1,
  "deepened_in_H": 1,
  "median_lb_H": 34219.0,
  "median_lb_rest": null,
  "deepened_rest": 0
 },
 "m10_n23_s3_t3_w113": {
  "H_size": 119,
  "H_index_max": 118,
  "deepened_in_H": 0,
  "median_lb_H": null,
  "median_lb_rest": null,
  "deepened_rest": 0
 },
 "m11_n23_s3_t3_w124": {
  "H_size": 34,
  "H_index_max": 33,
  "deepened_in_H": 0,
  "median_lb_H": null,
  "median_lb_rest": null,
  "deepened_rest": 0
 },
 "m13_n19_s3_t3_w123": {
  "H_size": 265,
  "H_index_max": 275,
  "deepened_in_H": 11,
  "median_lb_H": 2000000.0,
  "median_lb_rest": 1809663.0,
  "deepened_rest": 78
 },
 "m16_n17_s3_t3_w134": {
  "H_size": 100,
  "H_index_max": 108,
  "deepened_in_H": 7,
  "median_lb_H": 2000000.0,
  "median_lb_rest": 2000000.0,
  "deepened_rest": 68
 }
}
```

## Do genuine rules kill truly easier censored target cases? (deepened sample)

```
{
 "schema_recipe_frozen": {
  "n_killed": 27,
  "median_lb_killed": 505187.0,
  "mean_lb_killed": 933117.7037037037,
  "n_not": 318,
  "median_lb_not": 1362127.0,
  "mean_lb_not": 1182627.8867924528
 },
 "dgh_lib": {
  "n_killed": 0,
  "median_lb_killed": null,
  "mean_lb_killed": null,
  "n_not": 345,
  "median_lb_not": 1265712.0,
  "mean_lb_not": 1163101.0028985508
 },
 "schema_pool": {
  "n_killed": 27,
  "median_lb_killed": 505187.0,
  "mean_lb_killed": 933117.7037037037,
  "n_not": 318,
  "median_lb_not": 1362127.0,
  "mean_lb_not": 1182627.8867924528
 }
}
```
