import ZarPrune.Sum

/-
The Zarankiewicz problem as a Boolean-matrix statement, plus the row/column sum
profile that indexes the case decomposition.
-/

namespace ZarPrune

/-- An instance of the Zarankiewicz decision problem:
does an `m x n` `K_{s,t}`-free 0/1 matrix with at least `w` ones exist? -/
structure Params where
  m : Nat
  n : Nat
  s : Nat
  t : Nat
  w : Nat
deriving Repr, DecidableEq

/-- The candidate bipartite adjacency matrix. One Boolean per cell, matching the
one-variable-per-cell SAT encoding. -/
abbrev Mat (m n : Nat) := Fin m → Fin n → Bool

/-- 0/1 indicator, kept separate from `Bool.toNat` so the arithmetic lemmas below
are about one fixed definition. -/
def ind (b : Bool) : Nat := if b then 1 else 0

theorem ind_le_one (b : Bool) : ind b ≤ 1 := by cases b <;> decide

def rowSum {m n : Nat} (A : Mat m n) (i : Fin m) : Nat := sumFin n (fun j => ind (A i j))

def colSum {m n : Nat} (A : Mat m n) (j : Fin n) : Nat := sumFin m (fun i => ind (A i j))

/-- Total number of ones. This is the quantity the cardinality constraint bounds. -/
def weight {m n : Nat} (A : Mat m n) : Nat := sumFin m (rowSum A)

theorem rowSum_le {m n : Nat} (A : Mat m n) (i : Fin m) : rowSum A i ≤ n := by
  have h := sumFin_le n (fun j => ind (A i j)) 1 (fun j => ind_le_one (A i j))
  rw [Nat.mul_one] at h
  exact h

theorem colSum_le {m n : Nat} (A : Mat m n) (j : Fin n) : colSum A j ≤ m := by
  have h := sumFin_le m (fun i => ind (A i j)) 1 (fun i => ind_le_one (A i j))
  rw [Nat.mul_one] at h
  exact h

/-- Counting the ones by columns gives the same total as counting them by rows. -/
theorem weight_eq_sum_colSum {m n : Nat} (A : Mat m n) :
    weight A = sumFin n (colSum A) := by
  show sumFin m (fun i => sumFin n (fun j => ind (A i j)))
      = sumFin n (fun j => sumFin m (fun i => ind (A i j)))
  exact sumFin_swap m n (fun i j => ind (A i j))

/-- Strictly increasing index tuple. Using increasing rather than merely injective
tuples matches the SAT encoding, which emits one clause per increasing tuple. -/
def Incr {k N : Nat} (f : Fin k → Fin N) : Prop := ∀ a b : Fin k, a < b → f a < f b

/-- `A` contains an all-ones `s x t` submatrix, i.e. a `K_{s,t}`. -/
def HasKst (P : Params) (A : Mat P.m P.n) : Prop :=
  ∃ R : Fin P.s → Fin P.m, ∃ C : Fin P.t → Fin P.n,
    Incr R ∧ Incr C ∧ ∀ a b, A (R a) (C b) = true

/-- A witness to `z(m,n;s,t) ≥ w`: `K_{s,t}`-free with at least `w` ones.
The whole upper-bound pipeline exists to prove `∀ A, ¬ Valid P A`. -/
def Valid (P : Params) (A : Mat P.m P.n) : Prop :=
  ¬ HasKst P A ∧ P.w ≤ weight A

/-- The case index: the *vector* of row sums and the vector of column sums.
Tan's decomposition cases are unordered versions of these; a `Profile` refines
that, so a prune stated on profiles transfers to a prune on partitions. -/
structure Profile (m n : Nat) where
  row : Fin m → Nat
  col : Fin n → Nat

def profileOf {m n : Nat} (A : Mat m n) : Profile m n :=
  { row := rowSum A, col := colSum A }

@[simp] theorem profileOf_row {m n : Nat} (A : Mat m n) : (profileOf A).row = rowSum A := rfl
@[simp] theorem profileOf_col {m n : Nat} (A : Mat m n) : (profileOf A).col = colSum A := rfl

end ZarPrune
