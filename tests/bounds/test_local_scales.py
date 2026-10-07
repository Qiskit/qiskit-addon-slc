# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the local scales computation."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import BoxOp
from qiskit.quantum_info import PauliLindbladMap, QubitSparsePauliList
from qiskit_addon_slc.bounds import compute_local_scales
from samplomatic import InjectNoise

PAULIS = ["XI", "IX"]


def _build_inputs(
    bounds: list[float], noise_rates: list[float]
) -> tuple[QuantumCircuit, dict[str, PauliLindbladMap], dict[str, PauliLindbladMap]]:
    """Builds a minimal single-box circuit plus matching bounds and noise rates.

    Only ``BoxOp`` instructions carrying an :class:`~samplomatic.InjectNoise` annotation are
    inspected, so a hand-built circuit is enough; no transpilation is needed.

    Args:
        bounds: the commutator bounds to use, one per Pauli in ``PAULIS``.
        noise_rates: the learned noise rates to use, one per Pauli in ``PAULIS``.

    Returns:
        A tuple of the circuit, the bounds and the noise rates.
    """
    body = QuantumCircuit(2)
    body.cx(0, 1)
    annotation = InjectNoise(ref="noise", modifier_ref="box")

    circuit = QuantumCircuit(2)
    circuit.append(BoxOp(body, annotations=[annotation]), [0, 1])

    paulis = QubitSparsePauliList.from_list(PAULIS)
    return (
        circuit,
        {"box": PauliLindbladMap.from_components(np.asarray(bounds), paulis)},
        {"noise": PauliLindbladMap.from_components(np.asarray(noise_rates), paulis)},
    )


def test_mitigates_everything_within_an_unlimited_budget():
    """Test that every term is mitigated when the sampling cost is unconstrained."""
    circuit, bounds, noise_rates = _build_inputs([1.0, 0.5], [1e-2, 2e-2])

    scales, sampling_cost, bias_remaining = compute_local_scales(
        circuit, bounds, noise_rates, sampling_cost_budget=np.inf
    )

    np.testing.assert_allclose(scales["box"], [1.0, 1.0])
    assert bias_remaining == 0.0
    # Mitigating costs more than doing nothing, which would be an overhead of exactly 1.
    assert sampling_cost > 1.0


@pytest.mark.parametrize(
    ("kwargs", "bounds", "noise_rates", "reason"),
    [
        # The cheapest single term already costs exp(4 * 1e-2) ~ 1.041, so a budget of 1.0 affords
        # nothing at all.
        ({"sampling_cost_budget": 1.0}, [1.0, 0.5], [1e-2, 2e-2], "budget too small"),
        # A tolerance above the total bias bound means no term needs mitigating.
        ({"bias_tolerance": 10.0}, [1.0, 0.5], [1e-3, 2e-3], "tolerance already met"),
        # Zero bounds contribute no bias, so there is nothing worth mitigating.
        ({"sampling_cost_budget": np.inf}, [0.0, 0.0], [1e-3, 2e-3], "all bounds zero"),
    ],
    ids=["budget_too_small", "tolerance_already_met", "all_bounds_zero"],
)
def test_mitigating_nothing_is_not_an_error(kwargs, bounds, noise_rates, reason, subtests):
    """Test that computing local scales succeeds when no term gets mitigated.

    Each of the three conditions making up the selection mask can independently rule out every term.
    That used to index an empty array and raise an ``IndexError``, even though mitigating nothing is
    a perfectly meaningful outcome.

    Args:
        kwargs: the keyword arguments provoking an empty selection.
        bounds: the commutator bounds to use.
        noise_rates: the learned noise rates to use.
        reason: why no term gets mitigated, for the failure message.
        subtests: the pytest-subtests fixture.
    """
    circuit, bounds_map, noise_rates_map = _build_inputs(bounds, noise_rates)

    scales, sampling_cost, bias_remaining = compute_local_scales(
        circuit, bounds_map, noise_rates_map, **kwargs
    )

    with subtests.test(f"no term is mitigated ({reason})"):
        np.testing.assert_allclose(scales["box"], [0.0, 0.0])

    with subtests.test("the sampling cost is that of not mitigating anything"):
        # `samp_cost_accum` is `exp(4 * cumsum(rates))`, so mitigating nothing means a cumulative
        # rate of zero and hence an overhead of exactly 1.
        assert sampling_cost == 1.0

    with subtests.test("the full bias bound remains"):
        exp_rates = np.exp(-2 * np.asarray(noise_rates))
        expected = (np.asarray(bounds) * (1 - exp_rates) / 2).sum()
        np.testing.assert_allclose(bias_remaining, expected, rtol=1e-6)


def test_rejects_both_budget_and_tolerance():
    """Test that specifying both a sampling cost budget and a bias tolerance raises."""
    circuit, bounds, noise_rates = _build_inputs([1.0, 0.5], [1e-2, 2e-2])

    with pytest.raises(ValueError, match="Only one of either"):
        compute_local_scales(
            circuit, bounds, noise_rates, sampling_cost_budget=10.0, bias_tolerance=0.1
        )


def test_rejects_a_missing_noise_rate():
    """Test that a noise model identifier absent from ``noise_rates`` raises."""
    circuit, bounds, _ = _build_inputs([1.0, 0.5], [1e-2, 2e-2])

    with pytest.raises(KeyError, match="Missing noise rate"):
        compute_local_scales(circuit, bounds, {"noise": None}, sampling_cost_budget=np.inf)
