This folder collects the files explicitly listed in Section VI
(`Repository and Reproducibility`) of the report draft.

Included files:
- `Fair_Subspace_Benchmark.ipynb`
- `Fair_Subspace_Benchmark_H4.ipynb`
- `Fair_Subspace_Benchmark_LiH.ipynb`
- `validate_trimci_subspace.py`
- `TrimCI_FCI_Subspace_Validation.pdf`
- `warmstart_uccsd_compare.py`
- `uccsd_screening_benchmark.py`
- `krylov_vs_uccsd_k_comparison.py`
- `h2o_trimci_krylov_uccsd_compare.py`
- `Resource_Estimation_Subspace.ipynb`

Additional folder:
- `paper_draft/`
  Contains the current report source files and compiled PDF, including the
  corrected repository address in `Reproducibility.tex`.

Suggested reproduction steps:
1. Clone or download the repository.
2. Create the project environment and install the
required packages for PennyLane, PySCF, and
TrimCI.
3. Run validate trimci subspace.py to reproduce
the TrimCI-versus-FCI reference-validation figure
and CSV file.
4. Run the benchmark notebooks for H2, H4, and LiH
to reproduce the small-system Krylov and UCCSD
comparisons.
5. Run the larger-system reduced-space scripts to re-
produce the warm-start and H2O comparison fig-
ures used to motivate the scaling discussion.
