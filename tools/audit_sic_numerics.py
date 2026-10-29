"""Read-only diagnostic of the archived SIC's near-zero hard decisions.

This does not change the estimator or the reference results. Run with the same
interpreter as reproduce.py; floating-point details depend on NumPy and BLAS.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "extensions" / "shared_source"
sys.path.insert(0, str(EXT))
import numpy as np
from isac.shared_source import make_case, received_sensing, sensing


def main():
    cfg = json.loads((EXT / "reference/config_used.json").read_text(encoding="utf-8"))
    power = cfg["reference_power"]
    case = make_case("BIA", "lfm", .3, power, cfg["blocks"], cfg["waveform_seed"])
    _, observed, _, _, _ = received_sensing(case, 10, cfg["sensing_trials"], cfg["seed"])
    known = case.direct.sensing[:, case.sensed]
    coefficients = case.direct.communication[:, case.sensed]
    estimate = observed["direct"] @ np.linalg.pinv(known).T
    active = np.flatnonzero(np.sum(abs(coefficients) ** 2, axis=(0, 1)) > 1e-14)
    signatures = np.einsum("tjs,nj->nts", coefficients[:, :, active], estimate, optimize=True)
    residual = observed["direct"] - estimate @ known.T
    soft = np.sum(signatures.conj() * residual[:, :, None], axis=1) / np.maximum(np.sum(abs(signatures) ** 2, axis=1), 1e-30)
    result = {"numpy": np.__version__, "scenario": "BIA/lfm/rho=0.3/SNR=10",
              "soft_decisions": soft.size,
              "real_components_below_1e_12": int(np.count_nonzero(abs(soft.real) < 1e-12)),
              "imag_components_below_1e_12": int(np.count_nonzero(abs(soft.imag) < 1e-12)),
              "one_ulp_power_perturbation": []}
    for value in (power, float(np.nextafter(power, np.inf))):
        trial = make_case("BIA", "lfm", .3, value, cfg["blocks"], cfg["waveform_seed"])
        error, _, ber = sensing(trial, 10, cfg["sensing_trials"], cfg["seed"], cfg["sic_iterations"])
        result["one_ulp_power_perturbation"].append({"reference_power": value,
            "sic_mean_cee": float(error["SIC"].mean()), "sensor_decoder_ber": float(ber.mean())})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
