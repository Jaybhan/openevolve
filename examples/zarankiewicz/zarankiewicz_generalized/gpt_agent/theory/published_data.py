"""
published_data.py — independently transcribed literature data for cross-checks.

Every block below was transcribed by the theory agent on 2026-07-28 directly
from the cited source (ar5iv HTML of the papers, parsed programmatically from
the table markup — NOT copied from ../../evaluator.py). This lets
verify_claims.py compare the local ground-truth table against the published
record as two independent transcriptions.

Sources
-------
[TAN]   Jeremy Tan, "An attack on Zarankiewicz's problem through SAT solving",
        arXiv:2203.02283v2 (2022). https://arxiv.org/abs/2203.02283
        Tables 2/3/4 parsed from https://ar5iv.labs.arxiv.org/html/2203.02283
        (TikZ cell spans; class ltx_font_bold == boldface == proven exact).
        Data repo: https://github.com/Parcly-Taxel/Kyoto (last push 2022-04-19).
[CRWR]  A. Collins, A. Riasanovsky, J. Wallace, S. Radziszowski,
        "Zarankiewicz Numbers and Bipartite Ramsey Numbers",
        J. Algorithms Comput. 47 (2016) 63-78; arXiv:1604.01257.
        Appendix Tables parsed from ar5iv alttext LaTeX arrays.
        Legend: bold = exact; '*' = unique extremal graph; '+' (dagger) =
        also unique (z-1)-graph; italic = obtained by exhaustive computation;
        plain = bound via their Lemmas 2-4.
[DGH]   S. Davies, P. Gill, D. Horsley, "Improved upper bounds on Zarankiewicz
        numbers", arXiv:2411.18842 (2024/2025), Discrete Mathematics 349
        (2026), 114924. Table 2 = improved UBs on z(m,n;3,3).
[OEIS]  A001197 (k_2(n)), A001198 (k_3(n)), A072567 (z(n,n;2,2)), fetched
        2026-07-28 from https://oeis.org (JSON API). k_a(n) = z(n,n;a,a)+1.
[BNL]   J. Bhan, N. Nobili, P. Langer, "New Bounds for Zarankiewicz Numbers
        via Reinforced LLM Evolutionary Search", arXiv:2605.01120 (May 2026).
Convention: z(m,n;s,t) = max ones in m x n 0/1 matrix with no s rows and t
columns forming an all-one s x t submatrix; z(m,n;s,t) = z(n,m;t,s).
"""

# ---------------------------------------------------------------------------
# [TAN] Table 3: z_3(m,n) = z(m,n;3,3). Rows m=3..16, columns n=m..23.
# Value marked exact (bold in the paper) iff the flag is 1.
# Transcription: ar5iv S4.T3.pic1 spans, in row-major order.
TAN_Z3 = {
    3:  [(8,1),(10,1),(12,1),(14,1),(16,1),(18,1),(20,1),(22,1),(24,1),(26,1),(28,1),(30,1),(32,1),(34,1),(36,1),(38,1),(40,1),(42,1),(44,1),(46,1),(48,1)],
    4:  [(13,1),(16,1),(18,1),(21,1),(24,1),(26,1),(28,1),(30,1),(32,1),(34,1),(36,1),(38,1),(40,1),(42,1),(44,1),(46,1),(48,1),(50,1),(52,1),(54,1)],
    5:  [(20,1),(22,1),(25,1),(28,1),(30,1),(33,1),(36,1),(38,1),(41,1),(44,1),(46,1),(49,1),(52,1),(54,1),(57,1),(60,1),(62,1),(64,1),(66,1)],
    6:  [(26,1),(29,1),(32,1),(36,1),(39,1),(42,1),(45,1),(48,1),(50,1),(53,1),(56,1),(58,1),(61,1),(64,1),(66,1),(69,1),(72,1),(74,1)],
    7:  [(33,1),(37,1),(40,1),(44,1),(47,1),(50,1),(53,1),(56,1),(60,1),(63,1),(66,1),(69,1),(72,1),(75,1),(78,1),(81,1),(84,1)],
    8:  [(42,1),(45,1),(50,1),(53,1),(57,1),(60,1),(64,1),(67,1),(70,1),(74,1),(77,1),(81,1),(84,1),(87,1),(90,1),(94,1)],
    9:  [(49,1),(54,1),(59,1),(64,1),(67,1),(70,1),(73,1),(77,1),(81,1),(85,1),(89,1),(93,1),(96,1),(100,1),(104,0)],
    10: [(60,1),(64,1),(68,1),(73,1),(77,1),(81,1),(85,1),(90,1),(94,1),(98,1),(102,1),(108,0),(112,0),(116,0)],
    11: [(69,1),(74,1),(80,1),(84,1),(88,1),(92,1),(96,1),(101,1),(109,0),(113,0),(117,0),(121,0),(125,0)],
    12: [(80,1),(86,1),(91,1),(96,1),(99,1),(108,0),(113,0),(118,0),(122,0),(127,0),(132,0),(136,0)],
    13: [(92,1),(98,1),(104,1),(107,1),(117,0),(122,0),(126,0),(131,0),(136,0),(140,0),(145,0)],
    14: [(105,1),(112,1),(115,1),(125,0),(130,0),(136,0),(141,0),(146,0),(151,0),(155,0)],
    15: [(120,1),(123,1),(134,0),(139,0),(144,0),(150,0),(155,0),(160,0),(166,0)],
    16: [(128,1),(142,0),(148,0),(154,0),(160,0),(165,0),(170,0),(176,0)],
}

# [TAN] square-case list (Section 5 rows), z_a(m) = z(m,m;a,a):
TAN_Z2_DIAG = {2:3,3:6,4:9,5:12,6:16,7:21,8:24,9:29,10:34,11:39,12:45,13:52,
               14:56,15:61,16:67,17:74,18:81,19:88,20:96,21:105,22:108,23:115,24:122}
TAN_Z3_DIAG = {3:8,4:13,5:20,6:26,7:33,8:42,9:49,10:60,11:69,12:80,13:92,
               14:105,15:120,16:128}
TAN_Z4_DIAG = {4:15,5:22,6:31,7:42,8:51,9:61,10:74,11:86,12:100,13:117}

# [TAN] Table 1: T_{3,3}(m) = max multiset of 4-subsets of an m-set covering
# every 3-subset at most twice (computed by Tan with Gurobi; "we found no
# corresponding results in the literature").  m = 2,4 mod 6 omitted by Tan
# because a perfect Steiner quadruple system SQS(m) exists (Hanani 1960),
# doubling to a perfect 2-fold packing:  T_{3,3}(m) = C(m,3)/2 there.
#
# CORRECTION 2026-07-28: an earlier revision of this file recorded
# T33[18] = C(18,3)/2 = 408.  That was a TRANSCRIPTION-COMPLETION ERROR by
# the theory agent (extrapolated from a truncated HTML parse), not Tan's
# number.  Tan's Table 1 prints  T_{3,3}(17) = 340  and  T_{3,3}(18) = 405
# (re-read verbatim from the ar5iv text, and independently re-proved:
# a perfect 2-fold packing on 18 points would be a 3-(18,4,2) design with
# r = 2*C(17,2)/3 = 272/3 blocks per point — non-integral, impossible; the
# per-point Johnson bound gives T <= floor(18*90/4) = 405; and Tan's cyclic
# construction (17 base blocks under a dihedral group of order 36) generates
# exactly 405 blocks forming a valid 2-fold packing.  See verify_claims C16.)
from math import comb as _comb
T33 = {3:0, 4:2, 5:5, 6:9, 7:15, 8:_comb(8,3)//2, 9:40, 10:_comb(10,3)//2,
       11:80, 12:108, 13:143, 14:_comb(14,3)//2, 15:225, 16:_comb(16,3)//2,
       17:340, 18:405}

# [TAN] Table 1 cyclic presentations for m=17,18 (verbatim; symbols 0-9,A-H):
T33_PRESENTATIONS = {
    17: (["(0123456789ABCDEFG)", "(1G)(2F)(3E)(4D)(5C)(6B)(7A)(89)"],
         "013F 014E 0156 018A 0246 0279 027C 037D 0128 013C 014B 0159 025D 036B".split()),
    18: (["(0123456789ABCDEFGH)", "(1H)(2G)(3F)(4E)(5D)(6C)(7B)(8A)"],
         "029B 039C 049D 0123 014F 0156 0167 0189 025F 028A 0138 014A 015C 0248 025D 026B 036A".split()),
}

# [TAN] witnesses, base64 per Tan's spec (row-major bits, right-padded to a
# byte multiple, LITTLE-endian bit order within each byte).
TAN_WITNESSES = {
    (3,3):  "/wA=",                     # 8 ones
    (5,5):  "7+7uAQ==",                 # 20 ones, sums (4^5)/(4^5)
    (10,10):"PzwzXZ2Zp2216bqyBw==",     # 60 ones, (6^10)/(6^10)
    (16,16):"/wAPDzMzwzxVVaVamWZpaZaWZplapaqqPMPMzPDwAP8=",  # 128, (8^16)^2
}
# Tan, kyoto/data/3x3: z(3,3,16,16) < 129 was proven by refuting exactly the
# row/column-sum cases (9,8^15)x(9,8^15) and (9,8^15)x(9,9,8^13,7).

# ---------------------------------------------------------------------------
# [CRWR] Appendix Table 4: bounds on z(m,n;3), rows m=6..18, cols n=m..18.
# flags: 'B' bold(exact), '*' unique extremal graph, '+' dagger, 'I' italic.
CRWR_Z3 = {
    6:  [("26","B*"),("29","B"),("32","B"),("36","B*"),("39","B*"),("42","B"),("45","B*"),("48","B*"),("50","B"),("53","B"),("56","B"),("58","B"),("61","")],
    7:  [("33","B*"),("37","B*"),("40","B"),("44","B*"),("47","B"),("50","B"),("53","B"),("56","B"),("60","B*"),("63","B*"),("66","B"),("69","")],
    8:  [("42","B*"),("45","B"),("50","B*"),("53","B"),("57","B*"),("60","B"),("64","B*"),("67","B"),("70","B"),("74","B*"),("78","")],
    9:  [("49","B"),("54","B"),("59","B*"),("64","B*"),("67","B*"),("70","B"),("73","B"),("77","B"),("81","B"),("85","")],
    10: [("60","B+"),("64","B*"),("68","B"),("73","B*"),("77","B"),("81","B*"),("85","B*"),("90","B*"),("94","")],
    11: [("69","B*"),("74","B"),("80","B"),("84","B"),("88","B"),("92","B"),("96","B"),("101","")],
    12: [("80","B"),("86","B*"),("91","B*"),("96","B"),("99","B"),("103","B*"),("109","")],
    13: [("92","B*"),("98","B*"),("104","B*"),("107","B"),("110","I"),("116","")],
    14: [("105","B*"),("112","B*"),("115","B*"),("118",""),("124","")],
    15: [("120","B+"),("123","B*"),("126",""),("132","")],
    16: [("128","B*"),("133","I"),("140","")],
    17: [("141",""),("148","")],
    18: [("156","")],
}

# [CRWR] Table 3: z(n;2) = z(n,n;2,2), n=1..31 exact (Guy 1969 for n<=21,
# Afzaly-McKay unpublished for 22<=n<=31); z(32;2) in {189,190}.
CRWR_Z2_DIAG = [1,3,6,9,12,16,21,24,29,34,39,45,52,56,61,67,74,81,88,96,105,
                108,115,122,130,138,147,156,165,175,186]   # n = 1..31

# ---------------------------------------------------------------------------
# [DGH] Table 2: improved upper bounds on z(m,n;3,3) (arXiv:2411.18842v?,
# parsed 2026-07-28; every improvement over the previously best known bound,
# which for these cells was the Roman bound as printed by Tan).
DGH_UB33 = {
    (10,22):111, (10,23):115,
    (11,19):108, (11,20):112, (11,21):116,
    (13,17):116, (13,18):121, (13,19):125, (13,20):130, (13,21):135,
    (14,17):124, (14,18):129, (14,19):135, (14,20):140, (14,21):145, (14,22):150,
    (15,17):132, (15,18):138, (15,19):143, (15,20):149, (15,21):154, (15,23):165,
    (16,17):141, (16,18):146, (16,19):152, (16,20):158, (16,21):164, (16,22):169, (16,23):175,
}

# ---------------------------------------------------------------------------
# [OEIS] fetched 2026-07-28.
# A001197: k_2(n), offset 2:  z(n,n;2,2) = a(n) - 1.
A001197 = [4,7,10,13,17,22,25,30,35,40,46,53,57,62,68,75,82,89,97,106,109,116,123]
# A001198: k_3(n), offset 3:  z(n,n;3,3) = a(n) - 1.  a(16)=129 added by Tan.
A001198 = [9,14,21,27,34,43,50,61,70,81,93,106,121,129]
# A072567: z(n,n;2,2) directly, offset 1.
A072567 = [1,3,6,9,12,16,21,24,29,34,39,45,52,56,61,67,74,81,88,96,105,108,115,122]

# ---------------------------------------------------------------------------
# [BNL] newly exact values proven in arXiv:2605.01120 (the project owner's
# own prior paper): LLM-evolved constructions meeting published UBs
# (UB (11,21)=116 from DGH; (11,22)=121 and (12,22)=132 already the best
# published UB).  NOTE: (11,22)=121 is NOT in the local evaluator's suite.
BNL_EXACT = {(11,21):116, (11,22):121, (12,22):132}
