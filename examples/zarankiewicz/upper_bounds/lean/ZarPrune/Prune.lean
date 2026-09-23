import ZarPrune.Basic

/-
The prune verifier.

A *prune* is a computable predicate on cases together with a proof that no valid
matrix lives in a killed case.  The harness may skip building and solving the SAT
instance for any case the predicate kills; `Prune.skip` is the theorem that makes
that skip legitimate, and `upper_bound_of_cover` is the theorem that turns
"pruned cases + refuted cases" into the Zarankiewicz upper bound.

Note what is NOT here: nothing lets a prune remove a case because an equivalent
copy of it survives.  That is the *adding* move (symmetry breaking), it needs a
permutation witness, and it belongs in an SR/VeriPB certificate rather than here.
A prune must kill only cases that are genuinely empty.
-/

namespace ZarPrune

/-- A verified pruning argument for the instance `P`.

* `kill` is what the search harness actually runs, on every candidate case.
* `sound` is the proof obligation.  This is the entire gate: a candidate prune
  proposed by the search is accepted exactly when this field elaborates. -/
structure Prune (P : Params) where
  /-- Human-readable label, carried through so rejected/accepted candidates can be
  logged by name. -/
  name : String := ""
  kill : Profile P.m P.n → Bool
  sound : ∀ A : Mat P.m P.n, kill (profileOf A) = true → ¬ Valid P A

namespace Prune

variable {P : Params}

/-- The prune that kills nothing. Sound, useless, and the unit for `or`. -/
def never (P : Params) : Prune P where
  name := "never"
  kill := fun _ => false
  sound := by intro A h; simp at h

/-- Prunes compose: killing a case for either reason is still killing it for a reason. -/
def or (p q : Prune P) : Prune P where
  name := p.name ++ " | " ++ q.name
  kill := fun pf => p.kill pf || q.kill pf
  sound := by
    intro A h
    simp only [Bool.or_eq_true] at h
    rcases h with h' | h'
    · exact p.sound A h'
    · exact q.sound A h'

/-- Fold a whole evolved population of accepted prunes into one filter. -/
def ofList (P : Params) (ps : List (Prune P)) : Prune P :=
  ps.foldr Prune.or (Prune.never P)

/-- **The skip theorem.**  If the prune fires on a case, every matrix belonging to
that case is already refuted, so the harness never has to emit or solve the SAT
instance for it. -/
theorem skip (p : Prune P) (pf : Profile P.m P.n) (h : p.kill pf = true)
    (A : Mat P.m P.n) (hA : profileOf A = pf) : ¬ Valid P A := by
  subst hA
  exact p.sound A h

end Prune

/-- **The closure theorem.**

Given
* a verified prune `p`,
* the list of cases that survived it and were actually handed to the solver,
* `cover`: every valid matrix is either killed by the prune or falls in a surviving
  case (this is the cover-completeness obligation of the decomposition -- in a
  cube-and-conquer setting it is itself a certificate, not a hand proof), and
* `refuted`: each surviving case came back UNSAT (this is where a checked
  LRAT/SR certificate plus an encoding-correctness theorem is discharged),

we get the Zarankiewicz upper bound `z(m,n;s,t) < w` in the form
"every `K_{s,t}`-free matrix has fewer than `w` ones". -/
theorem upper_bound_of_cover
    (P : Params) (p : Prune P) (survivors : List (Profile P.m P.n))
    (cover : ∀ A : Mat P.m P.n, Valid P A →
        p.kill (profileOf A) = true ∨ ∃ q ∈ survivors, profileOf A = q)
    (refuted : ∀ q ∈ survivors, ∀ A : Mat P.m P.n, profileOf A = q → ¬ Valid P A) :
    ∀ A : Mat P.m P.n, ¬ HasKst P A → weight A < P.w := by
  intro A hfree
  rcases Nat.lt_or_ge (weight A) P.w with hw | hw
  · exact hw
  · have hv : Valid P A := ⟨hfree, hw⟩
    rcases cover A hv with hk | ⟨q, hq, hqe⟩
    · exact absurd hv (p.sound A hk)
    · exact absurd hv (refuted q hq A hqe)

/-- Restated with the bound as `≤ w - 1`, the form the bound tables use. -/
theorem upper_bound_succ_of_cover
    (P : Params) (k : Nat) (hk : P.w = k + 1) (p : Prune P)
    (survivors : List (Profile P.m P.n))
    (cover : ∀ A : Mat P.m P.n, Valid P A →
        p.kill (profileOf A) = true ∨ ∃ q ∈ survivors, profileOf A = q)
    (refuted : ∀ q ∈ survivors, ∀ A : Mat P.m P.n, profileOf A = q → ¬ Valid P A) :
    ∀ A : Mat P.m P.n, ¬ HasKst P A → weight A ≤ k := by
  intro A hfree
  have h := upper_bound_of_cover P p survivors cover refuted A hfree
  omega

end ZarPrune

/-- **Weight monotonicity.**  A prune for weight `w` is a prune for every `w' ≥ w`:
a matrix with `≥ w'` ones has `≥ w` ones, and the `K_{s,t}` condition ignores `w`. -/
def ZarPrune.Prune.mono {P : ZarPrune.Params} (p : ZarPrune.Prune P) (w' : Nat) (h : P.w ≤ w') :
    ZarPrune.Prune {P with w := w'} where
  name := p.name
  kill := p.kill
  sound := fun A hk hv => p.sound A hk ⟨hv.1, Nat.le_trans h hv.2⟩
