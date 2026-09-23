import ZarPrune.Counting

/-!
# Conditional prunes and fact provenance (design §8.2)

A `Fact` is an upper bound `z(m,n;s,t) ≤ z` carried with a provenance `tag`
(`"tan2022"`, `"lean-here"`, …).  `FactHolds f` is what the fact asserts.  A
`CondPrune P facts` is a prune whose soundness proof may *assume* the listed facts:
the closure theorem of an instance then names exactly the facts that were
load-bearing, and `CondPrune.discharge` turns it back into an unconditional `Prune`
once each fact is proved (or accepted as a cited hypothesis).

Contents

* `Fact`, `FactHolds`, `Fact.transpose`, `factHolds_transpose`, `FactHolds.ofBound`,
  `factHolds_waterfill` (the pure-mode facts are provable here and now).
* `CondPrune P facts`, `discharge`, `ofPrune`, `weaken`, `or`, `orSame`, `ofList`,
  `never`, `transposed`, `restate`.
* `argDelColF` / `argDelRowF`: the deletion prunes of `Counting.lean` with `U := f.z`
  taken from a fact about the neighbour `(m, n-1)` / `(m-1, n)`.
* `ofPrefixF` / `ofPrefixRowF` (**Argument I**): the `k` heaviest columns (rows) of a
  valid matrix form a `K_{s,t}`-free `m × k` (`k × n`) minor, so their sums add up to
  at most `z(m,k)` (`z(k,n)`).  Stated for proper minors only (`k < n`, `k < m`).

Gate: no `sorry`, no `native_decide`, no new axioms.
-/

namespace ZarPrune

open Finset

/-! ## Facts -/

/-- An upper bound `z(m,n;s,t) ≤ z` with its provenance. -/
structure Fact where
  m : Nat
  n : Nat
  s : Nat
  t : Nat
  z : Nat
  tag : String
deriving Repr, DecidableEq

/-- What a fact asserts: every `K_{s,t}`-free `m × n` matrix has at most `z` ones.
The weight field of the `Params` is irrelevant to `HasKst`; `0` is a fixed choice. -/
def FactHolds (f : Fact) : Prop :=
  ∀ B : Mat f.m f.n, ¬ HasKst ⟨f.m, f.n, f.s, f.t, 0⟩ B → weight B ≤ f.z

/-- `z(m,n;s,t) = z(n,m;t,s)`: the same fact seen from the other side. -/
def Fact.transpose (f : Fact) : Fact := ⟨f.n, f.m, f.t, f.s, f.z, f.tag⟩

theorem factHolds_transpose (f : Fact) : FactHolds f.transpose ↔ FactHolds f := by
  constructor
  · intro h B hB
    have := h (transpose B)
      (fun hk => hB ((hasKst_transpose ⟨f.m, f.n, f.s, f.t, 0⟩ B).mp hk))
    exact Nat.le_trans (Nat.le_of_eq (weight_transpose B).symm) this
  · intro h B hB
    have := h (transpose B)
      (fun hk => hB ((hasKst_transpose ⟨f.n, f.m, f.t, f.s, 0⟩ B).mp hk))
    exact Nat.le_trans (Nat.le_of_eq (weight_transpose B).symm) this

/-- A proved bound in the shape produced by `upper_bound_succ_of_cover` is a fact. -/
theorem FactHolds.ofBound (P : Params) (z : Nat) (tag : String)
    (h : ∀ A : Mat P.m P.n, ¬ HasKst P A → weight A ≤ z) :
    FactHolds ⟨P.m, P.n, P.s, P.t, z, tag⟩ :=
  fun B hB => h B (fun hk => hB hk)

/-- The pure-mode fact: the waterfilled column budget (Argument A) is a proved bound. -/
theorem factHolds_waterfill (m n s t : Nat) (hs : 1 ≤ s) (tag : String) :
    FactHolds ⟨m, n, s, t, waterfillBound m n s (colBudgetOf ⟨m, n, s, t, 0⟩), tag⟩ :=
  fun B hB => weight_le_waterfill ⟨m, n, s, t, 0⟩ hs B hB

/-! ## Conditional prunes -/

/-- A prune whose soundness may assume the facts in `facts`. -/
structure CondPrune (P : Params) (facts : List Fact) where
  name : String := ""
  kill : Profile P.m P.n → Bool
  sound : (∀ f ∈ facts, FactHolds f) →
    ∀ A : Mat P.m P.n, kill (profileOf A) = true → ¬ Valid P A

namespace CondPrune

variable {P : Params}

/-- Discharge the hypotheses: an unconditional prune. -/
def discharge {facts : List Fact} (q : CondPrune P facts) (h : ∀ f ∈ facts, FactHolds f) :
    Prune P where
  name := q.name
  kill := q.kill
  sound := q.sound h

/-- An unconditional prune is a conditional prune with no facts. -/
def ofPrune (p : Prune P) : CondPrune P [] where
  name := p.name
  kill := p.kill
  sound := fun _ => p.sound

/-- Assuming more facts never hurts. -/
def weaken {F G : List Fact} (q : CondPrune P F) (h : ∀ f ∈ F, f ∈ G) : CondPrune P G where
  name := q.name
  kill := q.kill
  sound := fun hG => q.sound (fun f hf => hG f (h f hf))

/-- Re-express the hypotheses: `G` implies `F`. -/
def restate {F G : List Fact} (q : CondPrune P F)
    (h : (∀ f ∈ G, FactHolds f) → ∀ f ∈ F, FactHolds f) : CondPrune P G where
  name := q.name
  kill := q.kill
  sound := fun hG => q.sound (h hG)

/-- The conditional prune that kills nothing. -/
def never (P : Params) (facts : List Fact) : CondPrune P facts where
  name := "never"
  kill := fun _ => false
  sound := by intro _ A h; simp at h

/-- Disjunction: the facts accumulate. -/
def or {F G : List Fact} (p : CondPrune P F) (q : CondPrune P G) : CondPrune P (F ++ G) where
  name := p.name ++ " | " ++ q.name
  kill := fun pf => p.kill pf || q.kill pf
  sound := by
    intro hF A h
    simp only [Bool.or_eq_true] at h
    rcases h with h' | h'
    · exact p.sound (fun f hf => hF f (List.mem_append_left _ hf)) A h'
    · exact q.sound (fun f hf => hF f (List.mem_append_right _ hf)) A h'

/-- Disjunction over a shared fact list. -/
def orSame {F : List Fact} (p q : CondPrune P F) : CondPrune P F :=
  (p.or q).weaken (fun f hf => by
    rcases List.mem_append.mp hf with h | h
    · exact h
    · exact h)

/-- Fold a list of conditional prunes over a common fact list. -/
def ofList (P : Params) (facts : List Fact) (qs : List (CondPrune P facts)) :
    CondPrune P facts :=
  qs.foldr orSame (never P facts)

/-- A conditional prune for the transposed instance is one for the original. -/
def transposed {facts : List Fact} (q : CondPrune P.transpose facts) : CondPrune P facts where
  name := q.name ++ " [transposed]"
  kill := fun pf => q.kill pf.swap
  sound := fun hF A h hv => q.sound hF (transpose A) h ((valid_transpose P A).mpr hv)

end CondPrune

/-! ## Deletion prunes with a ledgered fact -/

/-- Delete the lightest column against the fact `z(m, n-1; s, t) ≤ f.z`
(`argDelCol` of `Counting.lean` with `U := f.z`). -/
def argDelColF (P : Params) (f : Fact)
    (h : f.m = P.m ∧ f.n + 1 = P.n ∧ f.s = P.s ∧ f.t = P.t) : CondPrune P [f] where
  name := s!"delete-lightest-column [{f.tag}: z({f.m},{f.n};{f.s},{f.t}) ≤ {f.z}]"
  kill := fun pf => (List.finRange P.n).any (fun j => decide (pf.col j + f.z < P.w))
  sound := by
    intro hF A hk
    have hf : FactHolds f := hF f (List.mem_singleton.mpr rfl)
    rw [List.any_eq_true] at hk
    obtain ⟨j, _, hj⟩ := hk
    have hj' : colSum A j + f.z < P.w := of_decide_eq_true hj
    obtain ⟨m, n, s, t, w⟩ := P
    obtain ⟨fm, fn, fs, ft, z, tag⟩ := f
    simp only at h hj' hf
    obtain ⟨rfl, rfl, rfl, rfl⟩ := h
    exact valid_deleteCol_bound A j hf hj'

/-- Delete the lightest row against the fact `z(m-1, n; s, t) ≤ f.z`. -/
def argDelRowF (P : Params) (f : Fact)
    (h : f.m + 1 = P.m ∧ f.n = P.n ∧ f.s = P.s ∧ f.t = P.t) : CondPrune P [f] where
  name := s!"delete-lightest-row [{f.tag}: z({f.m},{f.n};{f.s},{f.t}) ≤ {f.z}]"
  kill := fun pf => (List.finRange P.m).any (fun i => decide (pf.row i + f.z < P.w))
  sound := by
    intro hF A hk
    have hf : FactHolds f := hF f (List.mem_singleton.mpr rfl)
    rw [List.any_eq_true] at hk
    obtain ⟨i, _, hi⟩ := hk
    have hi' : rowSum A i + f.z < P.w := of_decide_eq_true hi
    obtain ⟨m, n, s, t, w⟩ := P
    obtain ⟨fm, fn, fs, ft, z, tag⟩ := f
    simp only at h hi' hf
    obtain ⟨rfl, rfl, rfl, rfl⟩ := h
    exact valid_deleteRow_bound A i hf hi'

/-! ## Argument I: the `k` heaviest lines -/

/-- Indices of the `k` largest entries of `c` (stable sort, non-increasing), as a list. -/
def topIdx {n : Nat} (k : Nat) (c : Fin n → Nat) : List (Fin n) :=
  ((List.finRange n).mergeSort (fun a b => decide (c b ≤ c a))).take k

/-- The sum of the `k` largest entries of `c`. -/
def topSum {n : Nat} (k : Nat) (c : Fin n → Nat) : Nat := ((topIdx k c).map c).sum

theorem topIdx_nodup {n : Nat} (k : Nat) (c : Fin n → Nat) : (topIdx k c).Nodup :=
  List.Nodup.sublist (List.take_sublist _ _)
    ((List.mergeSort_perm _ _).nodup_iff.mpr (List.nodup_finRange n))

theorem topIdx_length {n : Nat} (k : Nat) (c : Fin n → Nat) (hk : k ≤ n) :
    (topIdx k c).length = k := by
  unfold topIdx
  rw [List.length_take, List.length_mergeSort, List.length_finRange]
  omega

/-- The `m × k` minor of `A` on the columns enumerated by `e`. -/
def minorCols {m n k : Nat} (A : Mat m n) (e : Fin k → Fin n) : Mat m k :=
  fun i a => A i (e a)

theorem colSum_minorCols {m n k : Nat} (A : Mat m n) (e : Fin k → Fin n) (a : Fin k) :
    colSum (minorCols A e) a = colSum A (e a) := rfl

theorem weight_minorCols {m n k : Nat} (A : Mat m n) (e : Fin k → Fin n) :
    weight (minorCols A e) = ∑ a, colSum A (e a) := by
  rw [weight_eq_sum_colSum, sumFin_eq_sum]
  rfl

/-- A `K_{s,t}` in a column minor (taken along an increasing enumeration) is a
`K_{s,t}` in the original. -/
theorem not_hasKst_minorCols {m n k s t w w' : Nat} (A : Mat m n) (e : Fin k → Fin n)
    (he : Incr e) (h : ¬ HasKst ⟨m, n, s, t, w⟩ A) : ¬ HasKst ⟨m, k, s, t, w'⟩ (minorCols A e) := by
  rintro ⟨R, C, hR, hC, hall⟩
  exact h ⟨R, fun b => e (C b), hR, fun a b hab => he _ _ (hC a b hab), fun a b => hall a b⟩

/-- Summing over the increasing enumeration of `S` is summing over `S`. -/
theorem sum_orderEmbOfFin {n k : Nat} (S : Finset (Fin n)) (hS : S.card = k) (c : Fin n → Nat) :
    ∑ a, c (S.orderEmbOfFin hS a) = ∑ j ∈ S, c j := by
  conv_rhs => rw [← Finset.map_orderEmbOfFin_univ S hS]
  rw [Finset.sum_map]
  rfl

/-- **Argument I (columns).**  The `k` heaviest columns of a valid matrix form a
`K_{s,t}`-free `m × k` minor, so their sums total at most `z(m,k;s,t)`.  Kill when the
sum of the `k` largest column sums of the profile exceeds the fact.  `k < P.n`: only
proper minors, never the cell itself (E1 circularity rule). -/
def Prune.ofPrefixF (P : Params) (k : Nat) (f : Fact) (hk : k < P.n)
    (h : f.m = P.m ∧ f.n = k ∧ f.s = P.s ∧ f.t = P.t) : CondPrune P [f] where
  name := s!"prefix: {k} heaviest columns [{f.tag}: z({f.m},{f.n};{f.s},{f.t}) ≤ {f.z}]"
  kill := fun pf => decide (f.z < topSum k pf.col)
  sound := by
    intro hF A hkill
    have hf : FactHolds f := hF f (List.mem_singleton.mpr rfl)
    have hlt : f.z < topSum k (colSum A) := of_decide_eq_true hkill
    rintro ⟨hfree, _⟩
    obtain ⟨m, n, s, t, w⟩ := P
    obtain ⟨fm, fn, fs, ft, z, tag⟩ := f
    simp only at h
    obtain ⟨rfl, rfl, rfl, rfl⟩ := h
    have hk' : fn < n := hk
    have hS : (topIdx fn (colSum A)).toFinset.card = fn := by
      rw [List.toFinset_card_of_nodup (topIdx_nodup _ _), topIdx_length _ _ (by omega)]
    have hsum : topSum fn (colSum A)
        = ∑ a, colSum A ((topIdx fn (colSum A)).toFinset.orderEmbOfFin hS a) := by
      rw [sum_orderEmbOfFin, List.sum_toFinset _ (topIdx_nodup _ _)]
      rfl
    have he : Incr ((topIdx fn (colSum A)).toFinset.orderEmbOfFin hS) :=
      incr_of_strictMono _ ((topIdx fn (colSum A)).toFinset.orderEmbOfFin hS).strictMono
    have hw := hf (minorCols A _) (not_hasKst_minorCols A _ he hfree)
    rw [weight_minorCols] at hw
    rw [hsum] at hlt
    exact absurd hlt (Nat.not_lt.mpr hw)

/-- **Argument I (rows).**  The `k` heaviest rows form a `K_{s,t}`-free `k × n` minor:
`ofPrefixF` on the transposed instance, with the fact read from the other side. -/
def Prune.ofPrefixRowF (P : Params) (k : Nat) (f : Fact) (hk : k < P.m)
    (h : f.m = k ∧ f.n = P.n ∧ f.s = P.s ∧ f.t = P.t) : CondPrune P [f] :=
  ((Prune.ofPrefixF P.transpose k f.transpose hk ⟨h.2.1, h.1, h.2.2.2, h.2.2.1⟩).transposed).restate
    (fun hG g hg => by
      rw [List.mem_singleton] at hg
      subst hg
      exact (factHolds_transpose f).mpr (hG f (List.mem_singleton.mpr rfl)))

end ZarPrune

-- ===== scratch audit (not part of the module) =====
namespace ZarPrune
#print axioms FactHolds
#print axioms factHolds_transpose
#print axioms FactHolds.ofBound
#print axioms factHolds_waterfill
#print axioms CondPrune.discharge
#print axioms CondPrune.ofPrune
#print axioms CondPrune.weaken
#print axioms CondPrune.restate
#print axioms CondPrune.never
#print axioms CondPrune.or
#print axioms CondPrune.orSame
#print axioms CondPrune.ofList
#print axioms CondPrune.transposed
#print axioms argDelColF
#print axioms argDelRowF
#print axioms topIdx_nodup
#print axioms topIdx_length
#print axioms weight_minorCols
#print axioms not_hasKst_minorCols
#print axioms sum_orderEmbOfFin
#print axioms Prune.ofPrefixF
#print axioms Prune.ofPrefixRowF

-- semantic checks on (9,9,3,3,50) with facts z(9,8)<=45 (tan2022), z(8,9)<=45
def T : Params := ⟨9, 9, 3, 3, 50⟩
def f98 : Fact := ⟨9, 8, 3, 3, 45, "tan2022"⟩
def f89 : Fact := ⟨8, 9, 3, 3, 45, "tan2022"⟩
def f97 : Fact := ⟨9, 7, 3, 3, 40, "tan2022"⟩
def mk (r c : List Nat) : Profile 9 9 := ⟨fun i => r.getD i 0, fun j => c.getD j 0⟩
def pDel := argDelColF T f98 ⟨rfl, rfl, rfl, rfl⟩
def pDelR := argDelRowF T f89 ⟨rfl, rfl, rfl, rfl⟩
def pPre := Prune.ofPrefixF T 7 f97 (by decide) ⟨rfl, rfl, rfl, rfl⟩
def pPreR := Prune.ofPrefixRowF T 7 f97.transpose (by decide) ⟨rfl, rfl, rfl, rfl⟩
def both := (pDel.or pPre)
-- col 4 + 45 = 49 < 50 -> kill ; all cols >= 5 -> 5+45 = 50, not < 50 -> survive
#eval pDel.kill (mk [6,6,6,6,6,5,5,5,5] [6,6,6,6,6,5,5,5,5])
#eval pDel.kill (mk [6,6,6,6,6,6,6,4,4] [6,6,6,6,6,6,6,4,4])
#eval pDelR.kill (mk [6,6,6,6,6,6,6,4,4] [6,6,6,6,6,5,5,5,5])
-- 7 heaviest columns of [6,6,6,6,6,5,5,5,5] sum 40, not > 40 -> survive; [7,7,6,6,6,6,4,4,4]: 44 > 40 -> kill
#eval pPre.kill (mk [6,6,6,6,6,5,5,5,5] [6,6,6,6,6,5,5,5,5])
#eval pPre.kill (mk [6,6,6,6,6,5,5,5,5] [7,7,6,6,6,6,4,4,4])
#eval pPreR.kill (mk [7,7,6,6,6,6,4,4,4] [6,6,6,6,6,5,5,5,5])
#eval pPreR.kill (mk [6,6,6,6,6,5,5,5,5] [7,7,6,6,6,6,4,4,4])
-- unsorted input: topSum must pick the largest regardless of order
#eval topSum 3 (fun j : Fin 9 => [1,9,2,8,3,7,4,6,5].getD j 0)
#eval (topIdx 3 (fun j : Fin 9 => [1,9,2,8,3,7,4,6,5].getD j 0)).map (fun j => j.val)
#eval both.name
#eval (CondPrune.ofList T [f98] [pDel, pDel]).kill (mk [6,6,6,6,6,5,5,5,5] [6,6,6,6,6,6,6,4,4])
-- discharge with a proved (waterfill) fact and use it as a Prune
def fWF : Fact := ⟨9, 8, 3, 3, waterfillBound 9 8 3 (colBudgetOf ⟨9, 8, 3, 3, 0⟩), "lean-here"⟩
#eval fWF.z
def pWF : Prune T := (argDelColF T fWF ⟨rfl, rfl, rfl, rfl⟩).discharge
  (fun f hf => by rw [List.mem_singleton] at hf; subst hf; exact factHolds_waterfill 9 8 3 3 (by decide) _)
#print axioms pWF
#eval pWF.kill (mk [6,6,6,6,6,5,5,5,5] [6,6,6,6,6,6,6,4,4])
end ZarPrune
