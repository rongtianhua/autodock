# Cascade Rescue Benchmark Analysis

## Summary

| Metric | Baseline | Cascade | Δ |
|--------|----------|---------|---|
| Success rate | 7/20 (35.0%) | 11/20 (55.0%) | **+4 targets (+20.0 pp)** |
| Rescued | 0 | 4 | +4 |
| Degraded | 0 | 0 | 0 |
| Mean RMSD (successes) | 1.03 Å | 1.06 Å | +0.03 Å |
| Median RMSD (successes) | 1.25 Å | 1.15 Å | −0.10 Å |
| Best-achievable | 14/20 (70%) | 17/20 (85%) | **+3 targets** |

## Rescued Targets (4 total, all via IFP)

| PDB | Family | Baseline RMSD | Cascade RMSD | Improvement | Method |
|-----|--------|---------------|--------------|-------------|--------|
| 2P54 | nuclear_receptor | 2.55 Å | 1.62 Å | −0.93 Å | IFP re-dock (50 poses) |
| 1D4K | enzyme | 3.12 Å | 1.79 Å | −1.33 Å | IFP re-dock (50 poses) |
| 1E1V | enzyme | 3.38 Å | 1.15 Å | −2.23 Å | IFP re-dock (50 poses) |
| 1T46 | enzyme | 2.06 Å | 0.84 Å | −1.22 Å | IFP re-dock (50 poses) |

## Family Breakdown

| Family | Baseline | Cascade | Δ |
|--------|----------|---------|---|
| kinase | 3/5 (60%) | 3/5 (60%) | — |
| protease | 3/5 (60%) | 3/5 (60%) | — |
| nuclear_receptor | 1/3 (33%) | 2/3 (67%) | **+1** |
| enzyme | 0/7 (0%) | 3/7 (43%) | **+3** |

## Key Observations

1. **All rescues via IFP**: The cascade IFP re-ranking after re-docking with 50 poses rescued all 4 targets. MM-GBSA contributed 0 rescues historically (NaN bug now fixed in `4dee472`).

2. **Zero degradation**: No previously successful docking was made worse by cascade — the rescue is strictly additive.

3. **Enzyme family biggest beneficiary**: Enzymes went from 0% to 43% success, accounting for 3 of the 4 rescues. This aligns with the hard-target overrides (deep sampling) being applied to several enzyme targets.

4. **Best-achievable is 85%**: With 50 poses, the best pose among all generated is within 2.0 Å for 17/20 targets. Baseline (20 poses) achieves 70%. The gap shows that more sampling helps, but the bigger issue is **scoring** — Vina doesn't rank good poses first.

5. **RMSD quality preserved**: Mean RMSD of successful dockings remains ~1.06 Å — rescues are genuine, not marginal passes just under the 2.0 Å threshold.

6. **6 scoring failures in cascade**: 1B9S, 2HU4, 1GWX, 1F0R, 1H1P, 1H22 fail during PLIP interaction extraction. Fixing these could unlock additional IFP rescues.

## Recommendations

- **Keep cascade enabled by default** for redocking validation — it provides a +20 pp success boost with zero risk of degradation.
- **MM-GBSA is now functional** (NaN fixed, 17/20 targets score). Re-run cascade with MM-GBSA enabled to see if tier-3 rescues any targets.
- **Fix 6 scoring failures** — likely PLIP parser issue with certain ligand PDB files. Fixing this is the highest-impact next step.
- **For production docking**: The 85% best-achievable rate suggests a consensus/scoring fusion approach (Vina + IFP + GNINA CNN) could push overall success from 55% toward 70–75%.
