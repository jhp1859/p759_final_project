"""Compare one-shot warm-started UCCSD baselines for the project molecules.

Examples
--------
Run the default LiH comparison with a simple screened-UCCSD threshold:

    ./bin/python warmstart_uccsd_compare.py --molecule LiH --screen-threshold 1e-2

The script reports:
- Hartree-Fock energy
- MP2-initialized UCCSD energy with no optimization
- screened MP2-initialized UCCSD energy with no optimization
- optional CCSD-based variants when PySCF is installed
"""

from __future__ import annotations

import argparse
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl"))

import numpy as np

from uccsd_initializers import OneShotUCCSDResult, warm_started_uccsd_baselines

DEFAULT_BOND_LENGTHS = {
    "H2": 0.74,
    "H4": 0.74,
    "LIH": 1.5474,
}


def molecule_geometry(name: str, bond_length: float | None = None) -> tuple[list[str], np.ndarray]:
    """Return the default project geometry for a supported molecule."""

    mol = name.strip().upper()
    bond = DEFAULT_BOND_LENGTHS[mol] if bond_length is None else float(bond_length)

    if mol == "H2":
        return ["H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, bond]], dtype=float)
    if mol == "H4":
        return ["H", "H", "H", "H"], np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, bond], [0.0, 0.0, 2.0 * bond], [0.0, 0.0, 3.0 * bond]],
            dtype=float,
        )
    if mol == "LIH":
        return ["Li", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, bond]], dtype=float)

    raise ValueError("Supported molecules: H2, H4, LiH")


def format_result(result: OneShotUCCSDResult, hf_energy: float) -> str:
    """Format one baseline row for terminal output."""

    delta = result.energy - hf_energy
    return (
        f"{result.label:28s}  E = {result.energy: .12f} Ha   "
        f"dE(HF) = {delta: .12f} Ha   operators = {result.operator_count}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule", default="LiH", choices=["H2", "H4", "LiH"])
    parser.add_argument("--bond-length", type=float, default=None)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--unit", default="angstrom")
    parser.add_argument("--screen-threshold", type=float, default=1e-2)
    parser.add_argument("--max-screened-excitations", type=int, default=None)
    parser.add_argument("--device", default="lightning.qubit")
    parser.add_argument("--skip-ccsd", action="store_true")
    args = parser.parse_args()

    symbols, geometry = molecule_geometry(args.molecule, args.bond_length)
    results = warm_started_uccsd_baselines(
        symbols,
        geometry,
        basis=args.basis,
        unit=args.unit,
        screen_threshold=args.screen_threshold,
        max_screened_excitations=args.max_screened_excitations,
        include_ccsd=not args.skip_ccsd,
        device_name=args.device,
    )

    hf_energy = float(results["hf_energy"])
    print(f"Molecule: {args.molecule}")
    print(f"Geometry ({args.unit}):\n{geometry}")
    print(f"Hartree-Fock energy           E = {hf_energy: .12f} Ha")

    for key in ("mp2_no_opt", "mp2_screened_no_opt", "ccsd_no_opt", "ccsd_screened_no_opt"):
        if key not in results:
            continue
        value = results[key]
        if isinstance(value, dict):
            print(f"{key:28s}  skipped: {value['reason']}")
            continue
        print(format_result(value, hf_energy))


if __name__ == "__main__":
    main()
