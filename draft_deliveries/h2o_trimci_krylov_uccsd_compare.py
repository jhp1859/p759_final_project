"""Compare Krylov and warm-started UCCSD against a TrimCI reference for H2O.

This script uses:
- PySCF RHF to define a shared molecular-orbital basis
- TrimCI on the matching FCIDUMP as the main reference energy
- one-shot warm-started UCCSD baselines from MP2 and CCSD amplitudes
- cached-grid Krylov projected subspaces built from the HF reference

Outputs:
- H2O_TrimCI_Krylov_UCCSD_Comparison.csv
- H2O_TrimCI_Krylov_UCCSD_Comparison.png
- H2O_TrimCI_Krylov_UCCSD_Comparison.pdf
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl"))

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import trimci
from pyscf import fci, gto, scf, tools

from uccsd_initializers import warm_started_uccsd_baselines

CHEMICAL_ACCURACY_MHA = 1.5936
H2O_SYMBOLS = ["O", "H", "H"]
H2O_GEOMETRY = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, -0.757, 0.587],
        [0.0, 0.757, 0.587],
    ],
    dtype=float,
)
H2O_ATOM = "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587"


@dataclass
class MethodResult:
    method: str
    energy: float
    gap_to_trimci_mha: float
    detail: str


@contextmanager
def suppress_native_output(enabled: bool = True):
    """Temporarily redirect stdout/stderr, including native-extension output."""

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


def hf_statevector(hf_bitstring: np.ndarray) -> np.ndarray:
    """Construct the computational-basis HF statevector."""

    index = int("".join(map(str, hf_bitstring.tolist())), 2)
    state = np.zeros(1 << len(hf_bitstring), dtype=complex)
    state[index] = 1.0
    return state


def normalize_statevector(state: np.ndarray) -> np.ndarray:
    """Normalize a statevector and reject invalid values."""

    state = np.asarray(state, dtype=complex)
    if not np.all(np.isfinite(state)):
        raise ValueError("Encountered a non-finite statevector.")
    norm = np.linalg.norm(state)
    if not np.isfinite(norm) or np.isclose(norm, 0.0):
        raise ValueError("Encountered an invalid or zero-norm statevector.")
    return state / norm


def solve_projected_subspace(states, h_sparse, s_eval_cutoff: float = 1.0e-8) -> float:
    """Solve the projected subspace problem for a list of basis states."""

    state_matrix = np.asarray(states, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        overlap = state_matrix.conj() @ state_matrix.T
    overlap = 0.5 * (overlap + overlap.conj().T)
    if not np.all(np.isfinite(overlap)):
        raise ValueError("Projected overlap matrix became non-finite.")
    s_evals, s_vecs = np.linalg.eigh(overlap)
    s_evals = np.real_if_close(s_evals).real
    keep = s_evals > float(s_eval_cutoff)

    if keep.sum() != len(s_evals):
        raise ValueError("Projected subspace became linearly dependent.")

    h_states = np.vstack([h_sparse @ psi for psi in state_matrix])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        h_sub = state_matrix.conj() @ h_states.T
    h_sub = 0.5 * (h_sub + h_sub.conj().T)
    if not np.all(np.isfinite(h_sub)):
        raise ValueError("Projected Hamiltonian became non-finite.")

    transform = s_vecs[:, keep] / np.sqrt(s_evals[keep])[None, :]
    h_ortho = transform.conj().T @ h_sub @ transform
    h_ortho = 0.5 * (h_ortho + h_ortho.conj().T)
    if not np.all(np.isfinite(h_ortho)):
        raise ValueError("Orthogonalized projected Hamiltonian became non-finite.")
    return float(np.real(np.linalg.eigvalsh(h_ortho)[0]))


def build_h2o_context(*, basis: str, unit: str, device_name: str, krylov_trotter_steps: int, krylov_trotter_order: int):
    """Build the PennyLane Hamiltonian and Krylov QNode for H2O."""

    h_op, n_qubits = qml.qchem.molecular_hamiltonian(
        H2O_SYMBOLS,
        H2O_GEOMETRY,
        charge=0,
        mult=1,
        basis=basis,
        unit=unit,
    )
    electrons = 10
    hf = np.array(qml.qchem.hf_state(electrons, n_qubits), dtype=int)
    h_sparse = qml.pauli.pauli_sentence(h_op).to_mat(wire_order=list(range(n_qubits)), format="csr")

    dev = qml.device(device_name, wires=list(range(n_qubits)))

    @qml.qnode(dev)
    def krylov_qnode(time_value):
        qml.BasisState(hf, wires=list(range(n_qubits)))
        qml.TrotterProduct(h_op, time=float(time_value), order=int(krylov_trotter_order), n=int(krylov_trotter_steps))
        return qml.state()

    return {
        "h_sparse": h_sparse,
        "hf": hf,
        "hf_vector": hf_statevector(hf),
        "krylov_qnode": krylov_qnode,
        "n_qubits": n_qubits,
        "n_terms": len(qml.pauli.pauli_sentence(h_op)),
    }


def run_trimci_reference(*, basis: str, unit: str, ndets: int, ndets_explore: int | None, nexploration: int | None):
    """Run TrimCI and exact FCI in the same RHF orbital basis for H2O."""

    mol = gto.M(atom=H2O_ATOM, basis=basis, spin=0, charge=0, unit=unit)
    mf = scf.RHF(mol).run(verbose=0)
    fci_energy, _ = fci.FCI(mf).kernel(verbose=0)

    with tempfile.TemporaryDirectory() as tmpdir:
        fcidump_path = os.path.join(tmpdir, "FCIDUMP")
        tools.fcidump.from_scf(mf, fcidump_path)

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

        with suppress_native_output(True):
            trimci_energy, dets, _coeffs, _details, _args = trimci.run_auto(**kwargs)

    return {
        "trimci_energy": float(trimci_energy),
        "fci_energy": float(fci_energy),
        "trimci_minus_fci_mha": 1000.0 * (float(trimci_energy) - float(fci_energy)),
        "det_count": len(dets),
        "hf_energy": float(mf.e_tot),
    }


def run_krylov_grid_scan(
    context,
    *,
    trimci_energy: float,
    time_grid: np.ndarray,
    max_k: int,
    s_eval_cutoff: float,
) -> list[MethodResult]:
    """Evaluate best Krylov projected energies from a cached time grid."""

    cached_states = {}
    for time_value in time_grid:
        cached_states[float(time_value)] = normalize_statevector(context["krylov_qnode"](float(time_value)))

    rows: list[MethodResult] = []

    for extra_states in range(1, int(max_k)):
        best_energy = None
        best_combo = None
        for combo in itertools.combinations(time_grid, extra_states):
            combo = tuple(float(t) for t in combo)
            states = [context["hf_vector"]] + [cached_states[t] for t in combo]
            try:
                energy = solve_projected_subspace(states, context["h_sparse"], s_eval_cutoff=s_eval_cutoff)
            except ValueError:
                continue
            if best_energy is None or energy < best_energy:
                best_energy = energy
                best_combo = combo

        if best_energy is None or best_combo is None:
            continue

        rows.append(
            MethodResult(
                method=f"Krylov K={extra_states + 1}",
                energy=float(best_energy),
                gap_to_trimci_mha=1000.0 * (float(best_energy) - float(trimci_energy)),
                detail="times=" + ",".join(f"{t:.2f}" for t in best_combo),
            )
        )

    return rows


def run_uccsd_warmstarts(*, trimci_energy: float, basis: str, unit: str, device_name: str) -> list[MethodResult]:
    """Evaluate one-shot warm-started UCCSD baselines for H2O."""

    baselines = warm_started_uccsd_baselines(
        H2O_SYMBOLS,
        H2O_GEOMETRY,
        basis=basis,
        unit=unit,
        screen_threshold=1.0e-2,
        include_ccsd=True,
        device_name=device_name,
    )

    rows = [
        MethodResult(
            method="HF",
            energy=float(baselines["hf_energy"]),
            gap_to_trimci_mha=1000.0 * (float(baselines["hf_energy"]) - float(trimci_energy)),
            detail="-",
        )
    ]

    for key, label in [
        ("mp2_no_opt", "UCCSD MP2"),
        ("mp2_screened_no_opt", "UCCSD scr. MP2"),
        ("ccsd_no_opt", "UCCSD CCSD"),
        ("ccsd_screened_no_opt", "UCCSD scr. CCSD"),
    ]:
        value = baselines[key]
        if isinstance(value, dict):
            continue
        rows.append(
            MethodResult(
                method=label,
                energy=float(value.energy),
                gap_to_trimci_mha=1000.0 * (float(value.energy) - float(trimci_energy)),
                detail=f"operators={value.operator_count}",
            )
        )

    return rows


def write_csv(rows: list[MethodResult], trimci_info: dict[str, float], out_path: Path) -> None:
    """Save the comparison data."""

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "energy_ha",
                "gap_to_trimci_mha",
                "detail",
                "trimci_energy_ha",
                "fci_energy_ha",
                "trimci_minus_fci_mha",
                "trimci_det_count",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row.method,
                    "energy_ha": row.energy,
                    "gap_to_trimci_mha": row.gap_to_trimci_mha,
                    "detail": row.detail,
                    "trimci_energy_ha": trimci_info["trimci_energy"],
                    "fci_energy_ha": trimci_info["fci_energy"],
                    "trimci_minus_fci_mha": trimci_info["trimci_minus_fci_mha"],
                    "trimci_det_count": trimci_info["det_count"],
                }
            )


def make_plot(rows: list[MethodResult], out_png: Path, out_pdf: Path) -> None:
    """Render the H2O comparison plot."""

    labels = [row.method for row in rows]
    gaps = [row.gap_to_trimci_mha for row in rows]
    colors = []
    for row in rows:
        if row.method.startswith("Krylov"):
            colors.append("#2a6fbb")
        elif "MP2" in row.method:
            colors.append("#2a9d8f")
        elif "CCSD" in row.method:
            colors.append("#e76f51")
        else:
            colors.append("#6c757d")

    fig, ax = plt.subplots(figsize=(11.2, 4.8), constrained_layout=True)
    bars = ax.bar(np.arange(len(rows)), gaps, color=colors, edgecolor="black", linewidth=0.7)
    ax.axhline(CHEMICAL_ACCURACY_MHA, color="black", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.set_ylabel(r"$E - E_{\mathrm{TrimCI}}$ (mHa)")
    ax.set_title(r"H$_2$O / STO-3G: Krylov and Warm-Started UCCSD vs TrimCI")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0.0, max(gaps) * 1.18)

    for bar, row in zip(bars, rows, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.03 * max(gaps),
            f"{row.gap_to_trimci_mha:.2f}\n{row.detail}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--unit", default="Angstrom")
    parser.add_argument("--device", default="lightning.qubit")
    parser.add_argument("--krylov-max-k", type=int, default=6)
    parser.add_argument("--krylov-time-min", type=float, default=0.02)
    parser.add_argument("--krylov-time-max", type=float, default=0.20)
    parser.add_argument("--krylov-grid-points", type=int, default=10)
    parser.add_argument("--krylov-trotter-steps", type=int, default=1)
    parser.add_argument("--krylov-trotter-order", type=int, default=2)
    parser.add_argument("--trimci-ndets", type=int, default=100)
    parser.add_argument("--trimci-ndets-explore", type=int, default=50)
    parser.add_argument("--trimci-nexploration", type=int, default=1)
    args = parser.parse_args()

    trimci_info = run_trimci_reference(
        basis=args.basis,
        unit=args.unit,
        ndets=args.trimci_ndets,
        ndets_explore=args.trimci_ndets_explore,
        nexploration=args.trimci_nexploration,
    )

    context = build_h2o_context(
        basis=args.basis,
        unit=args.unit,
        device_name=args.device,
        krylov_trotter_steps=args.krylov_trotter_steps,
        krylov_trotter_order=args.krylov_trotter_order,
    )
    time_grid = np.linspace(args.krylov_time_min, args.krylov_time_max, args.krylov_grid_points)

    rows = []
    rows.extend(run_uccsd_warmstarts(trimci_energy=trimci_info["trimci_energy"], basis=args.basis, unit=args.unit, device_name=args.device))
    rows.extend(
        run_krylov_grid_scan(
            context,
            trimci_energy=trimci_info["trimci_energy"],
            time_grid=time_grid,
            max_k=args.krylov_max_k,
            s_eval_cutoff=1.0e-8,
        )
    )

    out_dir = Path(__file__).resolve().parent
    csv_path = out_dir / "H2O_TrimCI_Krylov_UCCSD_Comparison.csv"
    png_path = out_dir / "H2O_TrimCI_Krylov_UCCSD_Comparison.png"
    pdf_path = out_dir / "H2O_TrimCI_Krylov_UCCSD_Comparison.pdf"

    write_csv(rows, trimci_info, csv_path)
    make_plot(rows, png_path, pdf_path)

    print(f"TrimCI energy: {trimci_info['trimci_energy']:.12f} Ha")
    print(f"FCI energy:    {trimci_info['fci_energy']:.12f} Ha")
    print(f"TrimCI - FCI:  {trimci_info['trimci_minus_fci_mha']:.6f} mHa")
    print(f"TrimCI selected determinants: {trimci_info['det_count']}")
    print()
    for row in rows:
        print(f"{row.method:16s}  E = {row.energy: .12f} Ha   gap = {row.gap_to_trimci_mha: .4f} mHa   {row.detail}")
    print()
    print(csv_path)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
