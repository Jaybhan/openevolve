import ZarPrune.Counting

/-!
Axiom audit and `#eval` smoke tests for `ZarPrune/Counting.lean`.
Run with: `lake env lean Attempts/CountingAudit.lean` (after `lake build`).
Every `#print axioms` line must be within {propext, Quot.sound, Classical.choice}.
-/

namespace ZarPrune

-- Bridge
#print axioms sumFin_eq_sum
#print axioms rowSum_eq_sum
#print axioms colSum_eq_sum
#print axioms weight_eq_sum
#print axioms sumFin_succAbove
-- Supports
#print axioms support
#print axioms rowSupport
#print axioms mem_support
#print axioms mem_rowSupport
#print axioms card_support
#print axioms card_rowSupport
-- HasKst from finsets
#print axioms incr_of_strictMono
#print axioms hasKst_of_subsets
-- Double counting
#print axioms card_powersetCard_eq_sum
#print axioms budget_general
#print axioms colBudget
#print axioms rowLocalBudget
-- Transposition
#print axioms Params.transpose
#print axioms transpose
#print axioms Profile.swap
#print axioms rowSum_transpose
#print axioms colSum_transpose
#print axioms weight_transpose
#print axioms profileOf_transpose
#print axioms hasKst_transpose
#print axioms valid_transpose
#print axioms Prune.transposed
#print axioms rowBudget
-- Argument A prunes
#print axioms argA
#print axioms argAT
-- Argument D
#print axioms fD
#print axioms gD
#print axioms cntLt
#print axioms boundD
#print axioms fD_mono
#print axioms fD_layer_cake
#print axioms sum_range_eq_sum_ite
#print axioms fD_layer_cake_le
#print axioms sum_fD_eq
#print axioms card_filter_ge
#print axioms boundD_le
#print axioms argD
#print axioms argDT
-- Deletion
#print axioms deleteCol
#print axioms deleteRow
#print axioms colSum_deleteCol
#print axioms rowSum_deleteRow
#print axioms weight_deleteCol
#print axioms weight_deleteRow
#print axioms Incr.succAbove_comp
#print axioms hasKst_of_deleteCol
#print axioms not_hasKst_deleteCol
#print axioms hasKst_of_deleteRow
#print axioms not_hasKst_deleteRow
-- Waterfilling
#print axioms choose_tangent
#print axioms equalCost
#print axioms equalCost_le_sum_choose
#print axioms waterfillBound
#print axioms sum_le_waterfillBound
#print axioms colBudgetOf
#print axioms weight_le_waterfill
-- Deletion prunes
#print axioms valid_deleteCol_bound
#print axioms valid_deleteRow_bound
#print axioms argDelCol
#print axioms argDelRow
#print axioms argDelColWF
#print axioms argDelRowWF
#print axioms argWF
-- Bundles
#print axioms countingA
#print axioms countingD
#print axioms deletion
#print axioms counting

/-! ### `#eval` smoke tests (all kills must compute) -/

def P99 : Params := ⟨9, 9, 2, 2, 50⟩
def pf1 : Profile 9 9 :=
  { row := fun i => [6,6,6,6,6,5,5,5,5].getD i.val 0,
    col := fun j => [6,6,6,6,6,5,5,5,5].getD j.val 0 }
def pf3 : Profile 9 9 := { row := fun _ => 3, col := fun _ => 3 }
def pf4 : Profile 9 9 :=
  { row := fun _ => 4, col := fun j => [4,4,4,3,3,3,3,3,3].getD j.val 0 }
def pf2 : Profile 9 9 :=
  { row := fun i => [7,5,5,5,5,5,5,5,5].getD i.val 0,
    col := fun j => [5,5,5,5,5,5,5,5,7].getD j.val 0 }

-- argA / argAT (hand: 5*15+4*10 = 115 > 36 kills; 9*3 = 27 ≤ 36 survives)
#eval ((argA P99).kill pf1, (argAT P99).kill pf1)      -- (true, true)
#eval ((argA P99).kill pf3, (argAT P99).kill pf3)      -- (false, false)
#eval ((argA P99).kill pf4, (argAT P99).kill pf4)      -- (false, true): cols 3*6+6*3=36 ≤ 36, rows 9*6=54 > 36
-- argD / argDT (D3 hand values: boundD pf1 6 = ..., budget (t-1)C(8,1) = 8)
#eval boundD P99 pf1 6
#eval (P99.t - 1) * (P99.m - 1).choose (P99.s - 1)
#eval ((argD P99).kill pf1, (argDT P99).kill pf1)
#eval ((argD P99).kill pf2, (argDT P99).kill pf2)
#eval ((argD P99).kill pf3, (argDT P99).kill pf3)
-- Real cases from cache/case_table_m10_n11_s3_t3_w65_pure.json;
-- Python reference (kill_row_argument_d, kill_col_argument_d) = (F,F), (F,T), (T,F)
def P1011 : Params := ⟨10, 11, 3, 3, 65⟩
def c1 : Profile 10 11 :=
  { row := fun i => [7,7,7,7,7,6,6,6,6,6].getD i.val 0,
    col := fun j => [6,6,6,6,6,6,6,6,6,6,5].getD j.val 0 }
def c2 : Profile 10 11 :=
  { row := fun i => [7,7,7,7,7,6,6,6,6,6].getD i.val 0,
    col := fun j => [8,6,6,6,6,6,6,6,5,5,5].getD j.val 0 }
def c3 : Profile 10 11 :=
  { row := fun i => [8,7,7,7,6,6,6,6,6,6].getD i.val 0,
    col := fun j => [6,6,6,6,6,6,6,6,6,6,5].getD j.val 0 }
#eval ((argD P1011).kill c1, (argDT P1011).kill c1)   -- (false, false)
#eval ((argD P1011).kill c2, (argDT P1011).kill c2)   -- (false, true)
#eval ((argD P1011).kill c3, (argDT P1011).kill c3)   -- (true, false)
#eval ((argA P1011).kill c1, (argAT P1011).kill c1)   -- (false, false)
-- Waterfilling (E2 hand values)
#eval waterfillBound 9 9 2 (2 * Nat.choose 9 2)   -- 40
#eval waterfillBound 9 8 2 (2 * Nat.choose 9 2)   -- 38
#eval (argWF ⟨9, 9, 2, 3, 41⟩).kill pf1            -- true
#eval (argWF ⟨9, 9, 2, 3, 40⟩).kill pf1            -- false
def pfL : Profile 9 9 :=
  { row := fun i => [5,5,5,5,5,5,5,5,1].getD i.val 0,
    col := fun j => [5,5,5,5,5,5,5,5,1].getD j.val 0 }
#eval (argDelColWF ⟨9, 9, 2, 3, 40⟩).kill pfL      -- true  (40 - 1 = 39 > 38)
#eval (argDelColWF ⟨9, 9, 2, 3, 40⟩).kill pf1      -- false (40 - 5 = 35 ≤ 38)
#eval (argDelRowWF ⟨9, 9, 2, 3, 40⟩).kill pfL      -- true
#eval (argDelColWF ⟨9, 0, 2, 3, 40⟩).kill ⟨fun _ => 0, fun j => j.elim0⟩  -- false (n = 0)
-- Bundles
#eval (counting P99).kill pf1                       -- true
#eval (counting P99).kill pf3                       -- false
#eval (counting P1011).kill c1                      -- false
#eval (counting P1011).kill c3                      -- true
#eval (counting P99).name
-- Degenerate parameters must not crash
#eval (counting ⟨9, 9, 0, 0, 5⟩).kill pf3
#eval (counting ⟨0, 0, 2, 2, 1⟩).kill ⟨fun i => i.elim0, fun j => j.elim0⟩
#eval (counting ⟨9, 9, 1, 1, 5⟩).kill pf3
-- A couple of `decide`s on kills (no native_decide)
example : (argD P1011).kill c3 = true := by decide
example : (argA P99).kill pf3 = false := by decide

end ZarPrune
