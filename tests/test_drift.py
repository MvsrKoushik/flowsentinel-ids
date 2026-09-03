import numpy as np

from flowsentinel import population_stability_index


def test_psi_detects_shift() -> None:
    reference = np.linspace(0, 1, 1000)
    assert population_stability_index(reference, reference) < 0.01
    assert population_stability_index(reference, reference + 2) > 0.2
