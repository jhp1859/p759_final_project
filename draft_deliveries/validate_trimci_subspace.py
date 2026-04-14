"""Validate TrimCI-selected determinant subspaces against exact FCI.

This script uses a shared PySCF FCIDUMP so that TrimCI and exact FCI work in
the same molecular-orbital basis. For each supported molecule, it:

1. runs exact RHF + FCI with PySCF
2. runs TrimCI on the matching FCIDUMP
3. sorts TrimCI determinants by |c_i|^2
4. projects the exact Hamiltonian into the span of the top-M determinants
5. compares the projected ground-state energy against exact FCI
6. computes the captured exact FCI weight in that selected determinant span

The main output figure is a two-panel validation plot showing:
- projected energy error E_M - E_FCI (mHa)
- captured FCI weight (%)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl"))

import matplotlib.pyplot as plt
import numpy as np
import trimci
from pyscf import ao2mo, fci, gto, scf, tools
from pyscf.fci import cistring, direct_spin1

CHEMICAL_ACCURACY_MHA = 1.5936
DEFAULT_UNIT = "Angstrom"
DEFAULT_BASIS = "sto-3g"

DEFAULT_SYSTEMS = {
    "H2": {
        "atom": "H 0 0 0; H 0 0 0.74",
        "spin": 0,
        "charge": 0,
    },
    "H4": {
        "atom": "H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22",
        "spin": 0,
        "charge": 0,
    },
    "LiH": {
        "atom": "Li 0 0 0; H 0 0 1.5474",
        "spin": 0,
        "charge": 0,
    },
}


@dataclass
class ExactFCIContext:
    label: str
    mol: gto.Mole
    mf: scf.hf.SCF
    fci_energy: float
    fci_coeffs: np.ndarray
    full_hamiltonian: np.ndarray
    alpha_addr: dict[int, int]
    beta_addr: dict[int, int]
    nbeta: int


@dataclass
class TrimCISelection:
    label: str
    trimci_energy: float
    determinant_addresses: list[int]
    determinant_coeffs: np.ndarray
    determinant_count: int


@contextmanager
def suppress_native_output(enabled: bool = True):
    """Temporarily redirect stdout/stderr, including native extension output."""

    if not enabled:
        yield
        return

    sys.stdout.flush()
    sys.stderr.flush()
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)

    try:
        with open(os.devnull, "w", encoding="utf-8") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            yield
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(old_stdout)
        os.close(old_stderr)


def build_exact_fci_context(label: str, atom: str, *, basis: str, unit: str, spin: int, charge: int) -> ExactFCIContext:
    """Build the exact determinant-space Hamiltonian and FCI wavefunction."""

    mol = gto.M(atom=atom, basis=basis, spin=spin, charge=charge, unit=unit)
    mf = scf.RHF(mol).run(verbose=0)

    solver = fci.FCI(mf)
    fci_energy, fci_coeffs = solver.kernel(verbose=0)
    fci_coeffs = np.asarray(fci_coeffs).reshape(-1)

    mo_coeff = mf.mo_coeff
    norb = mo_coeff.shape[1]
    h1e = mo_coeff.T @ mf.get_hcore() @ mo_coeff
    eri = ao2mo.kernel(mol, mo_coeff)

    _, h_elec = direct_spin1.pspace(h1e, eri, norb, mol.nelec, np=fci_coeffs.size)
    full_hamiltonian = np.asarray(h_elec, dtype=float) + np.eye(fci_coeffs.size) * mol.energy_nuc()

    neleca, nelecb = mol.nelec
    alpha_strings = cistring.make_strings(range(norb), neleca)
    beta_strings = cistring.make_strings(range(norb), nelecb)
    alpha_addr = {int(value): idx for idx, value in enumerate(alpha_strings)}
    beta_addr = {int(value): idx for idx, value in enumerate(beta_strings)}

    return ExactFCIContext(
        label=label,
        mol=mol,
        mf=mf,
        fci_energy=float(fci_energy),
        fci_coeffs=fci_coeffs,
        full_hamiltonian=full_hamiltonian,
        alpha_addr=alpha_addr,
        beta_addr=beta_addr,
        nbeta=len(beta_strings),
    )


def run_trimci_selection(
    context: ExactFCIContext,
    *,
    ndets: int,
    ndets_explore: int | None,
    nexploration: int | None,
    suppress_output: bool,
) -> TrimCISelection:
    """Run TrimCI on a FCIDUMP derived from the same RHF reference."""

    with tempfile.TemporaryDirectory() as tmpdir:
        fcidump_path = os.path.join(tmpdir, "FCIDUMP")
        tools.fcidump.from_scf(context.mf, fcidump_path)

        kwargs = {
            "fcidump_path": fcidump_path,
            "goal": "speed",
            "ndets": ndets,
            "verbose": False,
            "n_parallel": 1,
        }
        if ndets_explore is not None:
            kwargs["ndets_explore"] = ndets_explore
        if nexploration is not None:
            kwargs["nexploration"] = nexploration

        with suppress_native_output(suppress_output):
            trimci_energy, dets, coeffs, _details, _args = trimci.run_auto(**kwargs)

    coeff_by_address: dict[int, complex] = {}

    for det, coeff in zip(dets, coeffs, strict=True):
        alpha = int(det.alpha)
        beta = int(det.beta)
        if alpha not in context.alpha_addr or beta not in context.beta_addr:
            raise KeyError(
                f"TrimCI determinant ({alpha}, {beta}) is not present in the exact FCI determinant basis."
            )
        address = context.alpha_addr[alpha] * context.nbeta + context.beta_addr[beta]
        coeff_by_address[address] = coeff_by_address.get(address, 0.0) + complex(coeff)

    ranked_items = sorted(coeff_by_address.items(), key=lambda item: -abs(item[1]) ** 2)
    determinant_addresses = [int(address) for address, _coeff in ranked_items]
    determinant_coeffs = np.array([complex(coeff) for _address, coeff in ranked_items], dtype=complex)

    return TrimCISelection(
        label=context.label,
        trimci_energy=float(trimci_energy),
        determinant_addresses=determinant_addresses,
        determinant_coeffs=determinant_coeffs,
        determinant_count=len(determinant_addresses),
    )


def validation_rows(context: ExactFCIContext, selection: TrimCISelection) -> list[dict[str, float | int | str]]:
    """Build cumulative subspace-validation data for the selected determinants."""

    rows: list[dict[str, float | int | str]] = []
    ranked_addresses = list(selection.determinant_addresses)
    ranked_weights = np.abs(selection.determinant_coeffs) ** 2

    full_selected_h = context.full_hamiltonian[np.ix_(ranked_addresses, ranked_addresses)]
    projected_ground_energy = float(np.linalg.eigvalsh(full_selected_h)[0])

    for m in range(1, selection.determinant_count + 1):
        addresses = ranked_addresses[:m]
        sub_h = context.full_hamiltonian[np.ix_(addresses, addresses)]
        projected_energy = float(np.linalg.eigvalsh(sub_h)[0])
        captured_weight = float(np.sum(np.abs(context.fci_coeffs[addresses]) ** 2))

        rows.append(
            {
                "molecule": context.label,
                "M": m,
                "projected_energy": projected_energy,
                "fci_energy": context.fci_energy,
                "trimci_energy": selection.trimci_energy,
                "selected_subspace_energy": projected_ground_energy,
                "gap_mha": 1000.0 * (projected_energy - context.fci_energy),
                "captured_fci_weight": captured_weight,
                "selected_subspace_weight_sum": float(np.sum(ranked_weights[:m])),
                "total_selected_dets": selection.determinant_count,
            }
        )

    return rows


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    """Persist validation rows for later report use."""

    fieldnames = [
        "molecule",
        "M",
        "projected_energy",
        "fci_energy",
        "trimci_energy",
        "selected_subspace_energy",
        "gap_mha",
        "captured_fci_weight",
        "selected_subspace_weight_sum",
        "total_selected_dets",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_validation(all_rows: list[dict[str, float | int | str]], systems: list[str], out_png: Path, out_pdf: Path) -> None:
    """Render the energy-gap and captured-weight validation plot."""

    nrows = len(systems)
    fig, axes = plt.subplots(nrows, 1, figsize=(6.4, 4.5 * nrows), constrained_layout=True)
    if len(systems) == 1:
        axes = [axes]

    for ax, system in zip(axes, systems, strict=True):
        rows = [row for row in all_rows if row["molecule"] == system]
        m_values = np.array([int(row["M"]) for row in rows], dtype=int)
        gaps = np.array([float(row["gap_mha"]) for row in rows], dtype=float)
        weights = 100.0 * np.array([float(row["captured_fci_weight"]) for row in rows], dtype=float)

        ax.plot(m_values, gaps, marker="o", color="#1d4e89", linewidth=2.5, markersize=7.0)
        ax.axhline(CHEMICAL_ACCURACY_MHA, color="black", linestyle="--", linewidth=1.3, alpha=0.7)
        ax.set_title(system, fontsize=18)
        ax.set_xlabel("Top TrimCI determinants kept", fontsize=18)
        ax.set_ylabel(r"$E_{\mathrm{ref}}^{(M)} - E_{\mathrm{FCI}}$ (mHa)", color="#1d4e89", fontsize=18)
        ax.tick_params(axis="x", labelsize=15, width=1.1, length=5)
        ax.tick_params(axis="y", labelcolor="#1d4e89", labelsize=15, width=1.1, length=5)
        ax.grid(alpha=0.25)
        ax.set_xlim(1, int(m_values[-1]))
        ax.set_ylim(bottom=0.0)

        ax2 = ax.twinx()
        ax2.plot(m_values, weights, marker="s", color="#e07a1f", linewidth=2.2, markersize=5.5, linestyle=":")
        ax2.set_ylabel("Captured FCI weight (%)", color="#e07a1f", fontsize=18)
        ax2.tick_params(axis="y", labelcolor="#e07a1f", labelsize=15, width=1.1, length=5)
        ax2.set_ylim(0.0, 102.0)

        final_gap = gaps[-1]
        final_weight = weights[-1]
        ax.text(
            0.03,
            0.97,
            f"Final gap: {final_gap:.3f} mHa\nFCI weight: {final_weight:.2f}%",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=13,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
        )

    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecules", nargs="+", default=["H2", "H4", "LiH"], choices=sorted(DEFAULT_SYSTEMS))
    parser.add_argument("--basis", default=DEFAULT_BASIS)
    parser.add_argument("--unit", default=DEFAULT_UNIT)
    parser.add_argument("--ndets", type=int, default=40)
    parser.add_argument("--ndets-explore", type=int, default=None)
    parser.add_argument("--nexploration", type=int, default=None)
    parser.add_argument("--show-trimci-output", action="store_true")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent
    png_path = out_dir / "TrimCI_FCI_Subspace_Validation.png"
    pdf_path = out_dir / "TrimCI_FCI_Subspace_Validation.pdf"
    csv_path = out_dir / "TrimCI_FCI_Subspace_Validation.csv"

    all_rows: list[dict[str, float | int | str]] = []

    for system in args.molecules:
        spec = DEFAULT_SYSTEMS[system]
        context = build_exact_fci_context(
            system,
            spec["atom"],
            basis=args.basis,
            unit=args.unit,
            spin=int(spec["spin"]),
            charge=int(spec["charge"]),
        )
        selection = run_trimci_selection(
            context,
            ndets=args.ndets,
            ndets_explore=args.ndets_explore,
            nexploration=args.nexploration,
            suppress_output=not args.show_trimci_output,
        )
        rows = validation_rows(context, selection)
        all_rows.extend(rows)

        full_gap = rows[-1]["gap_mha"]
        full_weight = 100.0 * float(rows[-1]["captured_fci_weight"])
        projected_total = float(rows[-1]["selected_subspace_energy"])
        print(
            f"{system:4s}  FCI = {context.fci_energy: .12f} Ha   "
            f"TrimCI = {selection.trimci_energy: .12f} Ha   "
            f"Proj(top {selection.determinant_count:2d}) = {projected_total: .12f} Ha   "
            f"gap = {float(full_gap):.4f} mHa   "
            f"captured FCI weight = {full_weight:.2f}%"
        )

    write_csv(all_rows, csv_path)
    plot_validation(all_rows, args.molecules, png_path, pdf_path)

    print(csv_path)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
