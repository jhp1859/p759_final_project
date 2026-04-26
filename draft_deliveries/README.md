# Final Project Reproducibility Bundle

This folder contains the report and the main files needed to reproduce the results referenced in the report's `Repository and Reproducibility` section.

## Report

- `paper_draft/final_project_report.pdf`  
  Final compiled report.
- `paper_draft/final_project_report.tex` and section files in `paper_draft/`  
  LaTeX source for the report.

## File Map

| File | What it reproduces |
| --- | --- |
| `Fair_Subspace_Benchmark.ipynb` | Small-system H2 end-to-end Quantum Krylov versus UCCSD benchmark. |
| `Fair_Subspace_Benchmark_H4.ipynb` | Small-system H4 end-to-end Quantum Krylov versus UCCSD benchmark. |
| `Fair_Subspace_Benchmark_LiH.ipynb` | Small-system LiH end-to-end Quantum Krylov versus UCCSD benchmark. |
| `validate_trimci_subspace.py` | TrimCI reduced-space validation against exact FCI. Produces the validation data used for `TrimCI_FCI_Subspace_Validation.pdf`. |
| `TrimCI_FCI_Subspace_Validation.pdf` | Pre-generated TrimCI-versus-FCI validation figure. |
| `warmstart_uccsd_compare.py` | MP2- and CCSD-parameterized one-shot UCCSD comparisons before and after screening. |
| `uccsd_screening_benchmark.py` | Compact full-versus-screened UCCSD benchmark for H2, H4, and LiH. Produces the LiH screening figure and the small-system screening summary table used in the report. |
| `krylov_vs_uccsd_k_comparison.py` | Main six-system head-to-head comparison of Quantum Krylov and MP2/CCSD-ranked UCCSD as basis size K increases. Produces the main K-comparison data and figures. |
| `h2o_trimci_krylov_uccsd_compare.py` | Auxiliary larger-system reduced-space comparison used as an additional workflow check. |
| `Resource_Estimation_Subspace.ipynb` | Resource-estimation notebook for the benchmark ansatz families. Produces the resource summaries used in the full-stack section. |

## Suggested Order

1. Read `paper_draft/final_project_report.pdf`.
2. Run `validate_trimci_subspace.py` to check that the TrimCI reduced space agrees with exact FCI on small systems.
3. Run the three `Fair_Subspace_Benchmark*.ipynb` notebooks for the small-system comparisons.
4. Run `uccsd_screening_benchmark.py` and `warmstart_uccsd_compare.py` for the full-versus-screened UCCSD results.
5. Run `krylov_vs_uccsd_k_comparison.py` for the main six-system K-scaling comparison.
6. Run `Resource_Estimation_Subspace.ipynb` for the resource estimates.
