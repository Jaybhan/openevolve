# E12 — Farkas certificates over hidden pair codegrees

Per table: library survivors S_I, schema kills, certificates kept (greedy cover), gain_I = Σ d(killed) / Σ d(S_I),
tail kills (top decile by d), SAT-witnessed cases killed (must be 0).

| table | kind | cases | S_I | kills | certs | gain_I | tail | censored killed | sat killed | LP s |
|---|---|---|---|---|---|---|---|---|---|---|
| m9_n9_s3_t3_w50 | train | 36 | 17 | 5 | 1 | 0.097 | 0/1 | 0 | 0 | 0.26 |
| m9_n10_s3_t3_w55 | train | 45 | 22 | 6 | 1 | 0.153 | 0/2 | 0 | 0 | 0.02 |
| m10_n10_s3_t3_w61 | train | 25 | 10 | 1 | 1 | 0.148 | 0/1 | 0 | 0 | 0.01 |
| m10_n11_s3_t3_w65 | train | 195 | 66 | 32 | 1 | 0.367 | 2/6 | 0 | 0 | 0.04 |
| m11_n11_s3_t3_w70 | train | 625 | 237 | 77 | 2 | 0.100 | 1/23 | 0 | 0 | 0.21 |
| m11_n12_s3_t3_w75 | train | 420 | 227 | 14 | 1 | 0.024 | 0/22 | 0 | 0 | 0.26 |
| m12_n12_s3_t3_w81 | train | 225 | 137 | 29 | 1 | 0.067 | 0/13 | 0 | 0 | 0.16 |
| m10_n20_s3_t3_w103 | - | 130 | 67 | 33 | 2 | 0.411 | 0/6 | 33 | 0 | 0.05 |
| m12_n13_s3_t3_w87 | - | 360 | 102 | 6 | 3 | 0.015 | 0/10 | 0 | 0 | 0.16 |
| m13_n13_s3_t3_w93 | - | 1024 | 314 | 94 | 2 | 0.257 | 0/31 | 61 | 0 | 0.42 |
| m10_n14_s3_t3_w78 | - | 525 | 220 | 16 | 1 | 0.031 | 0/22 | 1 | 0 | 0.22 |
| m9_n9_s4_t4_w62 | gen | 49 | 25 | 2 | 1 | 0.030 | 0/2 | 0 | 0 | 0.02 |
| m12_n13_s3_t3_w86 | - | 1710 | 628 | 29 | 1 | 0.008 | 0/62 | 0 | 0 | 0.85 |
| m13_n13_s3_t3_w92 | - | 3969 | 1442 | 225 | 4 | 0.106 | 8/144 | 99 | 0 | 2.37 |
| m10_n14_s3_t3_w77 | - | 2064 | 1118 | 90 | 1 | 0.032 | 0/111 | 6 | 0 | 1.1 |

**TRAIN headline**: 164 of 716 library survivors killed over 7 tables, difficulty share 0.066, mean gain_I 0.137, 8 certificates.
**All tables**: 659 of 4632 survivors, d share 0.102.

SAT self-check: 11 witnessed cases in 24 tables, killed 0.

Gate (built ZarPrune.Schemas): one Lean process, 154.1 s, ok=True.

| table | ladder | axioms | Lean schema_mask | mirror | identical |
|---|---|---|---|---|---|
| m9_n9_s3_t3_w50 | 5 | ['propext', 'Classical.choice', 'Quot.sound'] | 18 | 18 | True |
| m9_n10_s3_t3_w55 | 5 | ['propext', 'Classical.choice', 'Quot.sound'] | 25 | 25 | True |
| m10_n10_s3_t3_w61 | 5 | ['propext', 'Classical.choice', 'Quot.sound'] | 5 | 5 | True |
| m10_n11_s3_t3_w65 | 5 | ['propext', 'Classical.choice', 'Quot.sound'] | 129 | 129 | True |
| m11_n11_s3_t3_w70 | 5 | ['propext', 'Classical.choice', 'Quot.sound'] | 292 | 292 | True |
| m11_n12_s3_t3_w75 | 5 | ['propext', 'Classical.choice', 'Quot.sound'] | 37 | 37 | True |
| m12_n12_s3_t3_w81 | 5 | ['propext', 'Classical.choice', 'Quot.sound'] | 61 | 61 | True |
| m10_n20_s3_t3_w103 | 5 | ['propext', 'Classical.choice', 'Quot.sound'] | 92 | 92 | True |
| m12_n13_s3_t3_w87 | 5 | ['propext', 'Classical.choice', 'Quot.sound'] | 40 | 40 | True |
| m13_n13_s3_t3_w93 | 5 | ['propext', 'Classical.choice', 'Quot.sound'] | 449 | 449 | True |
| m10_n14_s3_t3_w78 | 5 | ['propext', 'Classical.choice', 'Quot.sound'] | 247 | 247 | True |
| m9_n9_s4_t4_w62 | 5 | ['propext', 'Classical.choice', 'Quot.sound'] | 4 | 4 | True |
| m12_n13_s3_t3_w86 | 5 | ['propext', 'Classical.choice', 'Quot.sound'] | 494 | 494 | True |
| m13_n13_s3_t3_w92 | 5 | ['propext', 'Classical.choice', 'Quot.sound'] | 1372 | 1372 | True |
| m10_n14_s3_t3_w77 | 5 | ['propext', 'Classical.choice', 'Quot.sound'] | 720 | 720 | True |

Hardest killed survivors per table:

- m9_n9_s3_t3_w50: rows [7, 7, 6, 5, 5, 5, 5, 5, 5] cols [7, 6, 6, 6, 5, 5, 5, 5, 5] d=501
- m9_n9_s3_t3_w50: rows [7, 7, 6, 5, 5, 5, 5, 5, 5] cols [6, 6, 6, 6, 6, 6, 5, 5, 4] d=311
- m9_n10_s3_t3_w55: rows [8, 7, 6, 6, 6, 6, 6, 5, 5] cols [7, 6, 6, 6, 5, 5, 5, 5, 5, 5] d=1043
- m9_n10_s3_t3_w55: rows [8, 7, 6, 6, 6, 6, 6, 6, 4] cols [7, 6, 6, 6, 5, 5, 5, 5, 5, 5] d=486
- m10_n10_s3_t3_w61: rows [8, 7, 6, 6, 6, 6, 6, 6, 5, 5] cols [8, 7, 6, 6, 6, 6, 6, 6, 5, 5] d=1464
- m10_n11_s3_t3_w65: rows [8, 7, 7, 7, 7, 6, 6, 6, 6, 5] cols [7, 7, 6, 6, 6, 6, 6, 6, 5, 5, 5] d=10079
- m10_n11_s3_t3_w65: rows [8, 7, 7, 7, 7, 6, 6, 6, 6, 5] cols [7, 7, 6, 6, 6, 6, 6, 6, 6, 5, 4] d=6630
- m11_n11_s3_t3_w70: rows [8, 7, 7, 7, 6, 6, 6, 6, 6, 6, 5] cols [7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5] d=16756
- m11_n11_s3_t3_w70: rows [8, 7, 7, 7, 7, 6, 6, 6, 6, 5, 5] cols [7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5] d=11612
- m11_n12_s3_t3_w75: rows [9, 8, 7, 7, 7, 7, 6, 6, 6, 6, 6] cols [8, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 5] d=13322
- m11_n12_s3_t3_w75: rows [9, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6] cols [8, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 5] d=11998
- m12_n12_s3_t3_w81: rows [8, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6] cols [7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6] d=39712
- m12_n12_s3_t3_w81: rows [8, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 5] cols [7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6] d=26049
- m10_n20_s3_t3_w103: rows [11, 11, 11, 11, 11, 11, 10, 9, 9, 9] cols [7, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5] d=40000
- m10_n20_s3_t3_w103: rows [12, 11, 10, 10, 10, 10, 10, 10, 10, 10] cols [6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5] d=40000
- m12_n13_s3_t3_w87: rows [8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 5] cols [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5] d=21805
- m12_n13_s3_t3_w87: rows [8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 5] cols [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 4] d=20979
- m13_n13_s3_t3_w93: rows [9, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6] cols [8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6] d=49334
- m13_n13_s3_t3_w93: rows [9, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6] cols [8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 6, 6] d=49334
- m10_n14_s3_t3_w78: rows [10, 10, 8, 8, 7, 7, 7, 7, 7, 7] cols [7, 7, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 4] d=20000
- m10_n14_s3_t3_w78: rows [10, 10, 8, 8, 7, 7, 7, 7, 7, 7] cols [7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5] d=13062
- m9_n9_s4_t4_w62: rows [8, 8, 7, 7, 7, 7, 6, 6, 6] cols [7, 7, 7, 7, 7, 7, 7, 7, 6] d=168
- m9_n9_s4_t4_w62: rows [8, 8, 7, 7, 7, 7, 7, 6, 5] cols [7, 7, 7, 7, 7, 7, 7, 7, 6] d=150
- m12_n13_s3_t3_w86: rows [10, 9, 8, 7, 7, 7, 7, 7, 6, 6, 6, 6] cols [9, 8, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6] d=12942
- m12_n13_s3_t3_w86: rows [10, 9, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6] cols [9, 8, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6] d=9928
- m13_n13_s3_t3_w92: rows [9, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6] cols [8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6] d=50364
- m13_n13_s3_t3_w92: rows [9, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6] cols [9, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6] d=50364
- m10_n14_s3_t3_w77: rows [10, 10, 8, 8, 7, 7, 7, 7, 7, 6] cols [7, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 4] d=20000
- m10_n14_s3_t3_w77: rows [10, 10, 8, 8, 7, 7, 7, 7, 7, 6] cols [7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 4] d=20000
