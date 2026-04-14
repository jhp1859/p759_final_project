# p765_final_project

I plan to study different effective quantum state constructions for ground-state energy estimation, such as Krylov states, UCCSD, and symmetry-adapted references. Using a common measurement framework to build overlap and Hamiltonian matrices, I will compare their performance(sampling cost and gate depth) by analyzing ground-state energy errors across molecular systems.

## Suggested reproduction steps

1. Clone or download the repository.
2. Create the project environment and install the required packages for PennyLane, PySCF, and TrimCI.
3. Run `validate_trimci_subspace.py` to reproduce the TrimCI-versus-FCI reference-validation figure and CSV file.
4. Run the benchmark notebooks for `H2`, `H4`, and `LiH` to reproduce the small-system Krylov and UCCSD comparisons.
5. Run the larger-system reduced-space scripts to reproduce the warm-start and `H2O` comparison figures used to motivate the scaling discussion.
