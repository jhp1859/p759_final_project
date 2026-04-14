"""Benchmark full and screened warm-started UCCSD variants on small systems.

This script compares no-opt UCCSD energies initialized from MP2 or CCSD
amplitudes against exact FCI on the three tractable calibration systems:
H2, H4, and LiH. It produces:

- a CSV summary of all systems and variants
- a LaTeX table for the report
- two representative LiH plots:
  - accuracy gap relative to FCI
  - resource estimate via retained excitation-operator count
"""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl"))

import matplotlib.pyplot as plt
import numpy as np

from warmstart_uccsd_compare import molecule_geometry
from uccsd_initializers import OneShotUCCSDResult, warm_started_uccsd_baselines

CHEMICAL_ACCURACY_MHA = 1.5936
SCREEN_THRESHOLD = 1.0e-2
SYSTEMS = ("H2", "H4", "LiH")
REPRESENTATIVE_SYSTEM = "LiH"
METHOD_ORDER = (
    ("mp2_no_opt", "MP2 full"),
    ("mp2_screened_no_opt", "MP2 screened"),
    ("ccsd_no_opt", "CCSD full"),
    ("ccsd_screened_no_opt", "CCSD screened"),
)
PLOT_LABELS = {
    "MP2 full": "MP2\nfull",
    "MP2 screened": "MP2\nscreened",
    "CCSD full": "CCSD\nfull",
    "CCSD screened": "CCSD\nscreened",
}
COLORS = {
    "mp2_no_opt": "#2a9d8f",
    "mp2_screened_no_opt": "#8ecae6",
    "ccsd_no_opt": "#e76f51",
    "ccsd_screened_no_opt": "#f4a261",
}


def electron_sector_indices(n_qubits: int, electrons: int) -> list[int]:
    """Return the computational-basis indices in a fixed-electron sector."""

    return [index for index in range(1 << n_qubits) if index.bit_count() == electrons]


def exact_fci_energy_from_results(results: dict[str, object]) -> float:
    """Diagonalize the qubit Hamiltonian in the fixed-electron sector."""

    reference = results["reference"]
    hamiltonian_matrix = results["hamiltonian_matrix"]
    n_qubits = int(reference.n_spin_orbitals)
    electrons = int(reference.n_electrons)
    indices = electron_sector_indices(n_qubits, electrons)
    sector = hamiltonian_matrix[indices, :][:, indices].toarray()
    sector = 0.5 * (sector + sector.conj().T)
    eigvals = np.linalg.eigvalsh(sector)
    return float(np.real(eigvals[0]))


def collect_system_rows(system: str) -> list[dict[str, object]]:
    """Collect all reported UCCSD rows for one benchmark system."""

    symbols, geometry = molecule_geometry(system)
    results = warm_started_uccsd_baselines(
        symbols,
        geometry,
        basis="sto-3g",
        unit="angstrom",
        screen_threshold=SCREEN_THRESHOLD,
        include_ccsd=True,
        device_name="lightning.qubit",
    )
    e_fci = exact_fci_energy_from_results(results)

    rows: list[dict[str, object]] = []
    for key, label in METHOD_ORDER:
        result = results[key]
        if not isinstance(result, OneShotUCCSDResult):
            continue
        rows.append(
            {
                "system": system,
                "method_key": key,
                "label": label,
                "energy_ha": float(result.energy),
                "gap_to_fci_mha": 1000.0 * (float(result.energy) - e_fci),
                "operators": int(result.operator_count),
            }
        )
    return rows


def write_summary_csv(rows: list[dict[str, object]], path: Path) -> None:
    """Write the summary rows to CSV."""

    fieldnames = ["system", "label", "energy_ha", "gap_to_fci_mha", "operators"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def write_latex_table(rows: list[dict[str, object]], path: Path) -> None:
    """Write a report-ready LaTeX table."""

    ordered_rows = []
    for system in SYSTEMS:
        ordered_rows.extend([row for row in rows if row["system"] == system])

    lines = [
        r"\begin{tabular}{llcc}",
        r"    \toprule",
        r"    System & Variant & Gap to FCI (mHa) & Ops \\",
        r"    \midrule",
    ]

    for idx, row in enumerate(ordered_rows):
        system_cell = row["system"] if idx == 0 or ordered_rows[idx - 1]["system"] != row["system"] else ""
        system_cell = (
            system_cell.replace("H2", r"H$_2$")
            .replace("H4", r"H$_4$")
            .replace("LiH", "LiH")
        )
        lines.append(
            "    "
            + f"{system_cell} & {row['label']} & {row['gap_to_fci_mha']:.3f} & {row['operators']} \\\\"
        )
    lines.extend([r"    \bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_accuracy(rows: list[dict[str, object]], out_path: Path) -> None:
    """Plot the representative-system UCCSD energy gaps."""

    x = np.arange(len(rows))
    gaps = np.array([float(row["gap_to_fci_mha"]) for row in rows], dtype=float)
    labels = [PLOT_LABELS[str(row["label"])] for row in rows]
    colors = [COLORS[str(row["method_key"])] for row in rows]

    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    bars = ax.bar(x, gaps, color=colors, edgecolor="black", linewidth=1.0)
    ax.axhline(CHEMICAL_ACCURACY_MHA, color="black", linestyle="--", linewidth=1.4, alpha=0.8)
    ax.set_ylabel(r"$E - E_{\mathrm{FCI}}$ (mHa)", fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=15)
    ax.tick_params(axis="y", labelsize=15)
    ax.grid(axis="y", alpha=0.25)

    ymax = max(float(np.max(gaps)), CHEMICAL_ACCURACY_MHA)
    ax.set_ylim(0.0, 1.22 * ymax)
    for bar, gap in zip(bars, gaps, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(gap) + 0.035 * ymax,
            f"{gap:.2f}",
            ha="center",
            va="bottom",
            fontsize=13,
        )

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_resources(rows: list[dict[str, object]], out_path: Path) -> None:
    """Plot the representative-system retained operator counts."""

    x = np.arange(len(rows))
    ops = np.array([int(row["operators"]) for row in rows], dtype=float)
    labels = [PLOT_LABELS[str(row["label"])] for row in rows]
    colors = [COLORS[str(row["method_key"])] for row in rows]

    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    bars = ax.bar(x, ops, color=colors, edgecolor="black", linewidth=1.0)
    ax.set_ylabel("Retained excitation operators", fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=15)
    ax.tick_params(axis="y", labelsize=15)
    ax.grid(axis="y", alpha=0.25)

    ymax = max(float(np.max(ops)), 1.0)
    ax.set_ylim(0.0, 1.22 * ymax)
    for bar, count in zip(bars, ops, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(count) + 0.03 * ymax,
            f"{int(count)}",
            ha="center",
            va="bottom",
            fontsize=13,
        )

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_combined(rows: list[dict[str, object]], out_path: Path) -> None:
    """Plot the representative-system accuracy and resource panels together."""

    x = np.arange(len(rows))
    gaps = np.array([float(row["gap_to_fci_mha"]) for row in rows], dtype=float)
    ops = np.array([int(row["operators"]) for row in rows], dtype=float)
    labels = [PLOT_LABELS[str(row["label"])] for row in rows]
    colors = [COLORS[str(row["method_key"])] for row in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 8.8))

    bars1 = ax1.bar(x, gaps, color=colors, edgecolor="black", linewidth=1.0)
    ax1.axhline(CHEMICAL_ACCURACY_MHA, color="black", linestyle="--", linewidth=1.4, alpha=0.8)
    ax1.set_ylabel(r"$E - E_{\mathrm{FCI}}$ (mHa)", fontsize=18)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=15)
    ax1.tick_params(axis="y", labelsize=15)
    ax1.grid(axis="y", alpha=0.25)
    ymax1 = max(float(np.max(gaps)), CHEMICAL_ACCURACY_MHA)
    ax1.set_ylim(0.0, 1.22 * ymax1)
    for bar, gap in zip(bars1, gaps, strict=True):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(gap) + 0.035 * ymax1,
            f"{gap:.2f}",
            ha="center",
            va="bottom",
            fontsize=13,
        )

    bars2 = ax2.bar(x, ops, color=colors, edgecolor="black", linewidth=1.0)
    ax2.set_ylabel("Retained excitation operators", fontsize=18)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=15)
    ax2.tick_params(axis="y", labelsize=15)
    ax2.grid(axis="y", alpha=0.25)
    ymax2 = max(float(np.max(ops)), 1.0)
    ax2.set_ylim(0.0, 1.22 * ymax2)
    for bar, count in zip(bars2, ops, strict=True):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(count) + 0.03 * ymax2,
            f"{int(count)}",
            ha="center",
            va="bottom",
            fontsize=13,
        )

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    rows: list[dict[str, object]] = []
    for system in SYSTEMS:
        rows.extend(collect_system_rows(system))

    write_summary_csv(rows, out_dir / "UCCSD_Screening_Summary.csv")
    write_latex_table(rows, out_dir / "UCCSD_Screening_Summary_Table.tex")

    representative_rows = [row for row in rows if row["system"] == REPRESENTATIVE_SYSTEM]
    plot_accuracy(representative_rows, out_dir / f"UCCSD_Screening_{REPRESENTATIVE_SYSTEM}_Accuracy")
    plot_resources(representative_rows, out_dir / f"UCCSD_Screening_{REPRESENTATIVE_SYSTEM}_Resources")
    plot_combined(representative_rows, out_dir / f"UCCSD_Screening_{REPRESENTATIVE_SYSTEM}_Combined")

    print(out_dir / "UCCSD_Screening_Summary.csv")
    print(out_dir / "UCCSD_Screening_Summary_Table.tex")
    print(out_dir / f"UCCSD_Screening_{REPRESENTATIVE_SYSTEM}_Accuracy.pdf")
    print(out_dir / f"UCCSD_Screening_{REPRESENTATIVE_SYSTEM}_Resources.pdf")
    print(out_dir / f"UCCSD_Screening_{REPRESENTATIVE_SYSTEM}_Combined.pdf")


if __name__ == "__main__":
    main()
