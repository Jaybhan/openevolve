import ZarPrune.Prunes

/-
Tests, including the important negative one.
-/

namespace ZarPrune
namespace Demo

/-- `z(2,2;2,2)`-shaped toy instance: 2x2, forbid an all-ones 2x2, ask for 1 one. -/
abbrev demoP : Params := { m := 2, n := 2, s := 2, t := 2, w := 1 }

/-- A single one in the bottom-right cell.  Row sums `(0,1)`, column sums `(0,1)`. -/
def badA : Mat 2 2 := fun i j => decide (i.val = 1 ∧ j.val = 1)

theorem badA_valid : Valid demoP badA := by
  refine ⟨?_, ?_⟩
  · rintro ⟨R, C, _, hC, h⟩
    have e0 : (C 0).val = 1 := (of_decide_eq_true (h 0 0)).2
    have e1 : (C 1).val = 1 := (of_decide_eq_true (h 0 1)).2
    have hlt : (C 0).val < (C 1).val := hC 0 1 (by decide)
    omega
  · decide

/-- The reflex move: "an extremal matrix can be assumed to have its rows sorted,
so kill every case where they are not."  Written as a filter on cases. -/
def notDescending : Profile 2 2 → Bool := fun pf => decide (pf.row 0 < pf.row 1)

theorem notDescending_fires : notDescending (profileOf badA) = true := by decide

/-- **The negative test.**  `notDescending` cannot be given a soundness proof:
`badA` is a valid matrix living in a case it kills.

This is the adding/pruning distinction made concrete.  Row ordering is sound only
as an *addition* justified by a row permutation witness, which is an SR/VeriPB
redundancy step -- not a prune, because the case it deletes is not empty. -/
theorem notDescending_unsound : ¬ ∃ p : Prune demoP, p.kill = notDescending := by
  rintro ⟨p, hp⟩
  exact p.sound badA (by rw [hp]; exact notDescending_fires) badA_valid

/-- Positive test: the baseline filter is silent on a case that really occurs. -/
theorem baseline_silent : (baseline demoP).kill (profileOf badA) = false := by decide

/-- Positive test: the baseline filter kills an arithmetically impossible case. -/
theorem baseline_fires :
    (baseline demoP).kill { row := fun _ => 0, col := fun _ => 1 } = true := by decide

end Demo
end ZarPrune
