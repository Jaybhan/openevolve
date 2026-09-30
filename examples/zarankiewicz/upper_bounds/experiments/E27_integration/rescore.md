# E27 re-scored benchmark (final code, offline)

`experiments/E27_integration/rescore_benchmark.py`. Verified-branch combined_score; R1 masks are Lean-gated programs, R3 rows are attack shapes (oracle = not a program). 'old' = pre-E27 labels, 'new' = ground truth + hardness model. work = lower-bound conflicts removed over TRAIN ∪ TARGET of suite v3.

| program | kind | v2/S0 old | v2/S0 new | v2/S1 new | v3/S1 old | v3/S1 new | v3 G_train / G_target / Tail / Depth / Close | work removed (lb) |
|---|---|---|---|---|---|---|---|---|
| cert_close_10_22 | real (R1 Lean-gated) | 0.2000 | 0.2000 | 0.2423 | 0.2105 | 0.2041 | 0.000 / 0.018 / 0.009 / 0.000 / 0.000 | 5.83e+04 |
| cert_one_train | real (R1 Lean-gated) | 0.2222 | 0.2222 | 0.2115 | 0.2000 | 0.2000 | 0.000 / 0.000 / 0.000 / 0.000 / 0.000 | 6.54e+04 |
| cert_target_12_18 | real (R1 Lean-gated) | 0.2099 | 0.2061 | 0.2017 | 0.2095 | 0.2063 | 0.000 / 0.008 / 0.001 / 0.041 / 0.000 | 7.57e+06 |
| dgh_lib | real (R1 Lean-gated) | 0.2000 | 0.2000 | 0.2763 | 0.2272 | 0.3921 | 0.087 / 0.001 / 0.039 / 0.680 / 0.680 | 1.9e+07 |
| e1_dgh_s2 | real (R1 Lean-gated) | 0.2000 | 0.2000 | 0.2400 | 0.2000 | 0.2000 | 0.000 / 0.000 / 0.000 / 0.000 / 0.000 | 0 |
| e2_easy_certs | real (R1 Lean-gated) | 0.2068 | 0.2068 | 0.2038 | 0.2003 | 0.2003 | 0.000 / 0.000 / 0.000 / 0.001 / 0.000 | 4.52e+05 |
| e3_relist | real (R1 Lean-gated) | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.000 / 0.000 / 0.000 / 0.000 / 0.000 | 0 |
| f_weak | real (R1 Lean-gated) | 0.2286 | 0.2250 | 0.2168 | 0.2159 | 0.2126 | 0.003 / 0.027 / 0.004 / 0.057 / 0.000 | 2.08e+07 |
| initial | real (R1 Lean-gated) | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.000 / 0.000 / 0.000 / 0.000 / 0.000 | 0 |
| inst_dgh_10_23 | real (R1 Lean-gated) | 0.2000 | 0.2000 | 0.2002 | 0.2030 | 0.2005 | 0.000 / 0.001 / 0.000 / 0.003 / 0.000 | 6.33e+05 |
| instance_specific | real (R1 Lean-gated) | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.000 / 0.000 / 0.000 / 0.000 / 0.000 | 0 |
| lean_dgh4 | real (R1 Lean-gated) | 0.2000 | 0.2000 | 0.2763 | 0.2272 | 0.3921 | 0.087 / 0.001 / 0.039 / 0.680 / 0.680 | 1.9e+07 |
| recipe_dgh | real (R1 Lean-gated) | 0.2594 | 0.2556 | 0.3598 | 0.2659 | 0.4118 | 0.112 / 0.067 / 0.053 / 0.680 / 0.680 | 6.69e+07 |
| schema_pool | real (R1 Lean-gated) | 0.2618 | 0.2580 | 0.2920 | 0.2610 | 0.2497 | 0.046 / 0.067 / 0.019 / 0.195 / 0.000 | 6.09e+07 |
| schema_recipe_frozen | real (R1 Lean-gated) | 0.2594 | 0.2556 | 0.2835 | 0.2455 | 0.2430 | 0.025 / 0.067 / 0.014 / 0.195 / 0.000 | 4.79e+07 |
| R3 A1 clear GEN (9,9;4,4) | oracle (R3 attack shape) |  | 0.2800 |  |  | 0.2000 | 0.000 / 0.000 / 0.000 / 0.000 / 0.000 | 0 |
| R3 A2 clear (9,12,64) | oracle (R3 attack shape) |  | 0.2000 |  |  | 0.2000 | 0.000 / 0.000 / 0.000 / 0.000 / 0.000 | 0 |
| R3 A2 clear (10,10,61) | oracle (R3 attack shape) |  | 0.2635 |  |  | 0.2000 | 0.000 / 0.000 / 0.000 / 0.000 / 0.000 | 9.86e+03 |
| R3 A3 clear every table with W_lb<1e5 | oracle (R3 attack shape) |  | 0.4705 |  |  | 0.2041 | 0.000 / 0.018 / 0.009 / 0.000 / 0.000 | 1.03e+05 |
| R3 A5 real: pool certs on small tables only | real (R3 attack shape) |  | 0.2428 |  |  | 0.2041 | 0.000 / 0.018 / 0.009 / 0.000 / 0.000 | 1.3e+05 |
| R3 A4 real: pool certs + DGH on small tables only | real (R3 attack shape) |  | 0.2428 |  |  | 0.2041 | 0.000 / 0.018 / 0.009 / 0.000 / 0.000 | 1.3e+05 |
| R3 B easy exact d<=2000 | oracle (R3 attack shape) |  | 0.4100 |  |  | 0.2000 | 0.000 / 0.000 / 0.000 / 0.000 / 0.000 | 3.02e+05 |
| R3 B easy exact d<=5000 | oracle (R3 attack shape) |  | 0.5108 |  |  | 0.2000 | 0.000 / 0.000 / 0.000 / 0.000 / 0.000 | 1.06e+06 |
| R3 B easy exact d<=20000 | oracle (R3 attack shape) |  | 0.5892 |  |  | 0.2000 | 0.000 / 0.000 / 0.000 / 0.000 / 0.000 | 7.81e+06 |
| R3 B4 real: pool certs killing only exact d<=20k | real (R3 attack shape) |  | 0.2489 |  |  | 0.2000 | 0.000 / 0.000 / 0.000 / 0.000 / 0.000 | 2.81e+05 |
| R3 C1 real: cert_close_10_22 | real (R3 attack shape) |  | 0.2000 |  |  | 0.2041 | 0.000 / 0.018 / 0.009 / 0.000 / 0.000 | 5.83e+04 |
| R3 C2 tail-index sniper (targets) | oracle (R3 attack shape) |  | 0.2998 |  |  | 0.3186 | 0.000 / 0.193 / 0.519 / 0.233 / 0.000 | 1.89e+08 |
| R3 C3a deepened censored, EASY half | oracle (R3 attack shape) |  | 0.2206 |  |  | 0.2162 | 0.000 / 0.026 / 0.028 / 0.072 / 0.000 | 1.38e+08 |
| R3 C3b deepened censored, HARD half | oracle (R3 attack shape) |  | 0.2214 |  |  | 0.2210 | 0.000 / 0.026 / 0.070 / 0.072 / 0.000 | 1.38e+08 |
| R3 D1 random 10% of m16_n17_w134 | oracle (R3 attack shape) |  | 0.2000 |  |  | 0.2178 | 0.000 / 0.034 / 0.016 / 0.086 / 0.000 | 1.26e+07 |
| R3 D1 random 10% of m13_n19_w123 | oracle (R3 attack shape) |  | 0.2000 |  |  | 0.2149 | 0.000 / 0.017 / 0.014 / 0.087 / 0.000 | 2.59e+07 |
| R3 D1 random 10% of m12_n18_w109 | oracle (R3 attack shape) |  | 0.2140 |  |  | 0.2142 | 0.000 / 0.016 / 0.010 / 0.086 / 0.000 | 1.23e+07 |
| R3 D1 random 10% of m9_n23_w104 | oracle (R3 attack shape) |  | 0.2157 |  |  | 0.2124 | 0.000 / 0.009 / 0.008 / 0.082 / 0.000 | 6.96e+06 |
| R3 G tail top-decile of tables with <=30 survivors | oracle (R3 attack shape) |  | 0.3148 |  |  | 0.2223 | 0.011 / 0.018 / 0.048 / 0.091 / 0.000 | 2.05e+06 |

## Criteria (live column v3/S1 new)

```
{
 "C1 initial == 0.20": true,
 "C2 dgh uplift >= 0.02 and >= 0.5 x recipe uplift": true,
 "C2 ratio dgh/recipe uplift": 4.468811736167037,
 "C3 real exploits below min(recipe, dgh)": true,
 "C3 real exploit violations": {},
 "C3 R3 realizable attacks (A5, B4, C1) below min(recipe, dgh)": true,
 "C3 R3 realizable violations": {},
 "C3 oracle shapes below min(recipe, dgh)": false,
 "C3 oracle violations": {
  "R3 C2 tail-index sniper (targets)": 0.31856113258191077
 },
 "tiny-table clears below f_weak": {
  "R3 A1 clear GEN (9,9;4,4)": [
   0.2,
   true
  ],
  "R3 A2 clear (9,12,64)": [
   0.2,
   true
  ],
  "R3 A2 clear (10,10,61)": [
   0.2,
   true
  ]
 },
 "C5 recipe uplift retained vs v2/S0 old": 0.7235175114744592,
 "C5 recipe+dgh >= max(recipe, dgh)": true,
 "C4 live property test": {
  "mono_fail": 0,
  "trials": 60,
  "uniform_fail": []
 }
}
```

## Implementation check: R3's VR** reproduced by reward.score

```
{
 "schema_recipe_frozen": 0.2619,
 "dgh_lib": 0.3836,
 "recipe_dgh": 0.4139,
 "f_weak": 0.2173,
 "cert_close_10_22": 0.2105,
 "R3 expected": {
  "schema_recipe_frozen": 0.2619,
  "dgh_lib": 0.3836,
  "recipe_dgh": 0.4139,
  "f_weak": 0.2173,
  "cert_close_10_22": 0.2105
 }
}
```
