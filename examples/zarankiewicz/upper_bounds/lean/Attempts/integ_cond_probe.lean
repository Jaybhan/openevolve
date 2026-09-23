import ZarPrune
set_option autoImplicit false
namespace ZarPrune
namespace Cand
abbrev target : Params := { m := 9, n := 9, s := 3, t := 3, w := 50 }
def fact98 : Fact := { m := 9, n := 8, s := 3, t := 3, z := 45, tag := "tan2022" }
def condDelCol : CondPrune target [fact98] where
  name := "delete a column vs z(9,8;3,3)=45 [tan2022]"
  kill := fun pf => (List.finRange target.n).any (fun j => decide (pf.col j + fact98.z < target.w))
  sound := by
    intro hF A h
    have hf : FactHolds fact98 := hF fact98 (List.mem_singleton.mpr rfl)
    exact (argDelCol target fact98.z (fun B hB => hf B hB)).sound A h
def candidateF : CondPrune target [fact98] := condDelCol
def candidate (P : Params) : Prune P := Prune.or (baseline P) (counting P)
end Cand
end ZarPrune
def ZarPrune.Cand.gateFacts : List ZarPrune.Fact := [⟨9, 8, 3, 3, 45, "tan2022"⟩, ⟨8, 9, 3, 3, 45, "tan2022"⟩]
def ZarPrune.Cand.gateFacts1 : List ZarPrune.Fact := []
def ZarPrune.Cand.candInst : ZarPrune.Prune ZarPrune.Cand.target := by
  first
  | exact ZarPrune.Cand.candidate
  | exact ZarPrune.Cand.candidate ZarPrune.Cand.target
def ZarPrune.Cand.condF : ZarPrune.CondPrune ZarPrune.Cand.target ZarPrune.Cand.gateFacts := by
  first
  | exact ZarPrune.Cand.candidateF ZarPrune.Cand.target ZarPrune.Cand.gateFacts
  | exact ZarPrune.CondPrune.weaken ZarPrune.Cand.candidateF (by decide)
  | exact ZarPrune.CondPrune.weaken (ZarPrune.Cand.candidateF ZarPrune.Cand.target) (by decide)
  | exact ZarPrune.CondPrune.never _ _
def ZarPrune.Cand.condF1 : ZarPrune.CondPrune ZarPrune.Cand.target ZarPrune.Cand.gateFacts1 := by
  first
  | exact ZarPrune.Cand.candidateF ZarPrune.Cand.target ZarPrune.Cand.gateFacts1
  | exact ZarPrune.CondPrune.weaken ZarPrune.Cand.candidateF (by decide)
  | exact ZarPrune.CondPrune.weaken (ZarPrune.Cand.candidateF ZarPrune.Cand.target) (by decide)
  | exact ZarPrune.CondPrune.never _ _
def ZarPrune.Cand.gateInst (hF : ∀ f ∈ ZarPrune.Cand.gateFacts, ZarPrune.FactHolds f) : ZarPrune.Prune ZarPrune.Cand.target :=
  ZarPrune.Prune.or ZarPrune.Cand.candInst (ZarPrune.Cand.condF.discharge hF)
def ZarPrune.Cand.gateKill : ZarPrune.Profile ZarPrune.Cand.target.m ZarPrune.Cand.target.n → Bool :=
  fun pf => ZarPrune.Cand.candInst.kill pf || ZarPrune.Cand.condF.kill pf
theorem ZarPrune.Cand.gateKill_eq : ∀ hF pf, (ZarPrune.Cand.gateInst hF).kill pf = ZarPrune.Cand.gateKill pf := fun _ _ => rfl
#print axioms ZarPrune.Cand.gateInst
#print axioms ZarPrune.Cand.gateKill_eq
#eval IO.println ("COND " ++ ZarPrune.Cand.condF.name)
#eval IO.println ("COND1 " ++ ZarPrune.Cand.condF1.name)
def ZarPrune.Cand.gateProfile (r c : List Nat) : ZarPrune.Profile ZarPrune.Cand.target.m ZarPrune.Cand.target.n :=
  { row := fun i => r.getD i.val 0, col := fun j => c.getD j.val 0 }
#eval ZarPrune.Cand.gateKill (ZarPrune.Cand.gateProfile [7,7,6,5,5,5,5,5,5] [8,8,8,8,8,4,2,2,2])
#eval ZarPrune.Cand.gateKill (ZarPrune.Cand.gateProfile [6,6,6,6,6,5,5,5,5] [6,6,6,6,6,5,5,5,5])
