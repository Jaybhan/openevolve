/-
Finite sums and quantifiers over `Fin k`, with no Mathlib dependency.

Deliberately Mathlib-free: the intended neighbour of this library is a verified
LRAT/SR certificate importer, and those do not depend on Mathlib either.  Keeping
the core dependency-light means the whole prune verifier elaborates in seconds,
which matters when it sits inside an evolutionary inner loop.
-/

namespace ZarPrune

/-- `sumFin k f = f 0 + f 1 + ... + f (k-1)`. -/
def sumFin : (k : Nat) → (Fin k → Nat) → Nat
  | 0,     _ => 0
  | k + 1, f => f 0 + sumFin k (fun i => f i.succ)

/-- `allFin k f` is true when `f` holds at every index. -/
def allFin : (k : Nat) → (Fin k → Bool) → Bool
  | 0,     _ => true
  | k + 1, f => f 0 && allFin k (fun i => f i.succ)

theorem sumFin_succ (k : Nat) (f : Fin (k + 1) → Nat) :
    sumFin (k + 1) f = f 0 + sumFin k (fun i => f i.succ) := rfl

theorem allFin_succ (k : Nat) (f : Fin (k + 1) → Bool) :
    allFin (k + 1) f = (f 0 && allFin k (fun i => f i.succ)) := rfl

theorem sumFin_congr : ∀ (k : Nat) (f g : Fin k → Nat), (∀ i, f i = g i) →
    sumFin k f = sumFin k g := by
  intro k
  induction k with
  | zero => intro _ _ _; rfl
  | succ k ih =>
      intro f g h
      rw [sumFin_succ, sumFin_succ, h 0, ih _ _ (fun i => h i.succ)]

theorem sumFin_const_zero : ∀ (k : Nat), sumFin k (fun _ => 0) = 0 := by
  intro k
  induction k with
  | zero => rfl
  | succ k ih => rw [sumFin_succ, ih]

theorem sumFin_add : ∀ (k : Nat) (f g : Fin k → Nat),
    sumFin k (fun i => f i + g i) = sumFin k f + sumFin k g := by
  intro k
  induction k with
  | zero => intro _ _; rfl
  | succ k ih =>
      intro f g
      rw [sumFin_succ, sumFin_succ, sumFin_succ,
        ih (fun i => f i.succ) (fun i => g i.succ)]
      omega

/-- Fubini for `sumFin`: summing rows-then-columns equals columns-then-rows. -/
theorem sumFin_swap : ∀ (m n : Nat) (g : Fin m → Fin n → Nat),
    sumFin m (fun i => sumFin n (g i)) = sumFin n (fun j => sumFin m (fun i => g i j)) := by
  intro m
  induction m with
  | zero =>
      intro n g
      exact (sumFin_const_zero n).symm
  | succ m ih =>
      intro n g
      rw [sumFin_succ, ih n (fun i j => g i.succ j)]
      rw [show (fun j => sumFin (m + 1) (fun i => g i j))
            = (fun j => g 0 j + sumFin m (fun i => g i.succ j)) from rfl]
      rw [sumFin_add n (fun j => g 0 j) (fun j => sumFin m (fun i => g i.succ j))]

theorem sumFin_le : ∀ (k : Nat) (f : Fin k → Nat) (c : Nat), (∀ i, f i ≤ c) →
    sumFin k f ≤ k * c := by
  intro k
  induction k with
  | zero => intro _ _ _; simp [sumFin]
  | succ k ih =>
      intro f c h
      have := ih (fun i => f i.succ) c (fun i => h i.succ)
      have h0 := h 0
      rw [sumFin_succ, Nat.succ_mul]
      omega

theorem allFin_iff : ∀ (k : Nat) (f : Fin k → Bool),
    allFin k f = true ↔ ∀ i, f i = true := by
  intro k
  induction k with
  | zero =>
      intro f
      exact ⟨fun _ i => i.elim0, fun _ => rfl⟩
  | succ k ih =>
      intro f
      rw [allFin_succ, Bool.and_eq_true, ih (fun i => f i.succ)]
      constructor
      · rintro ⟨h0, hs⟩ i
        refine Fin.cases ?_ ?_ i
        · exact h0
        · intro j; exact hs j
      · intro h
        exact ⟨h 0, fun j => h j.succ⟩

/-- Discharging a negated `allFin`: if `g` holds everywhere it cannot be that
`allFin` is false.  Stated with `g` implicit so it unifies against whatever lambda
a prune's `kill` function happens to produce. -/
theorem not_allFin_elim {k : Nat} {g : Fin k → Bool} (hg : ∀ i, g i = true)
    (hh : (!allFin k g) = true) : False := by
  rw [(allFin_iff k g).mpr hg] at hh
  exact Bool.noConfusion hh

end ZarPrune
