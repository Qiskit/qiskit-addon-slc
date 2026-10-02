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

PAULIS = ["XI", "IX", "ZI"]


def _build_inputs(
    bound_rates: list[float], noise_rates: list[float]
) -> tuple[QuantumCircuit, dict[str, PauliLindbladMap], dict[str, PauliLindbladMap]]:
    """Builds a minimal single-box circuit plus matching bounds and noise rates.

    Args:
        bound_rates: the commutator bounds to use, one per Pauli in ``PAULIS``.
        noise_rates: the learned noise rates to use, one per Pauli in ``PAULIS``.

    Returns:
        A tuple of the circuit, the bounds and the noise rates.
    """
    body = QuantumCircuit(2)
    body.cx(0, 1)
    box = BoxOp(body, annotations=[InjectNoise(ref="n0", modifier_ref="m0")])

    circuit = QuantumCircuit(2)
    circuit.append(box, [0, 1])

    paulis = QubitSparsePauliList.from_list(PAULIS)
    bounds = {"m0": PauliLindbladMap.from_components(np.asarray(bound_rates), paulis)}
    rates = {"n0": PauliLindbladMap.from_components(np.asarray(noise_rates), paulis)}

    return circuit, bounds, rates


@pytest.mark.parametrize("multiple_observables", [False, True])
def test_mitigates_everything_within_budget(multiple_observables):
    """Test that all non-zero terms get mitigated when the budget is unlimited.

    Args:
        multiple_observables: whether to use the multi-observable prioritization.
    """
    circuit, bounds, rates = _build_inputs([1.0, 0.5, 0.2], [1e-3, 2e-3, 3e-3])

    scales, sampling_cost, bias_remaining = compute_local_scales(
        circuit,
        bounds,
        rates,
        sampling_cost_budget=np.inf,
        multiple_observables=multiple_observables,
    )

    np.testing.assert_allclose(scales["m0"], [1.0, 1.0, 1.0])
    assert bias_remaining == 0.0
    assert sampling_cost > 1.0


def test_prioritization_differs_from_default():
    """Test that the multi-observable mode can rank terms differently than the default.

    The default prioritizes by ``bound * exp(-2 * rate)``, which penalizes a large rate only
    exponentially. The multi-observable mode prioritizes by "value density", i.e. the bias bound
    divided by the rate, whose numerator saturates for a large rate so that the penalty becomes
    roughly linear. An expensive term with a large bound is therefore ranked relatively higher by
    the default than by the multi-observable mode.
    """
    # The first term has the largest bound but a rate three orders of magnitude above the others.
    # Default priority ranks it last ([1, 2, 0]); value density ranks it second ([1, 0, 2]).
    bound_rates = [1.0, 0.6, 0.2]
    noise_rates = [0.9, 1e-3, 1e-3]

    circuit, bounds, rates = _build_inputs(bound_rates, noise_rates)

    # This budget affords the two cheap terms but never the expensive one. Under the default
    # ranking those two come first, so both are mitigated; under value density the expensive term
    # sits between them, blocking the second cheap one.
    budget = 1.02

    scales_default, _, bias_default = compute_local_scales(
        circuit, bounds, rates, sampling_cost_budget=budget, multiple_observables=False
    )
    scales_multi, _, bias_multi = compute_local_scales(
        circuit, bounds, rates, sampling_cost_budget=budget, multiple_observables=True
    )

    np.testing.assert_allclose(scales_default["m0"], [0.0, 1.0, 1.0])
    np.testing.assert_allclose(scales_multi["m0"], [0.0, 1.0, 0.0])
    assert not np.array_equal(scales_default["m0"], scales_multi["m0"])

    # The multi-observable bias bound uses ``1 - exp(-2 * rate)**2`` rather than
    # ``1 - exp(-2 * rate)``, reflecting the doubled error rates of the unmitigated terms. The
    # residual bias is the sum of the bias bounds of the terms left unmitigated, which differs
    # between the two modes both in the formula and in which terms remain. Asserting on it pins the
    # doubled-rate correction itself, not just the resulting ranking.
    bounds_arr = np.asarray(bound_rates)
    exp_rates = np.exp(-2 * np.asarray(noise_rates))

    # Default mode leaves only the first term unmitigated.
    expected_bias_default = (bounds_arr * (1 - exp_rates) / 2)[0]
    # Multi-observable mode leaves the first and third terms unmitigated.
    expected_bias_multi = (bounds_arr * (1 - exp_rates**2) / 2)[[0, 2]].sum()

    np.testing.assert_allclose(bias_default, expected_bias_default, rtol=1e-6)
    np.testing.assert_allclose(bias_multi, expected_bias_multi, rtol=1e-6)
    assert bias_multi > bias_default


def test_zero_rate_does_not_produce_nan(recwarn):
    """Test that a noise rate of exactly zero is handled without dividing by zero.

    The multi-observable priority is a ratio whose denominator is the noise rate. Where the rate is
    exactly zero the numerator is zero too, so the priority must be defined as zero rather than
    evaluating to ``nan``. A ``nan`` would silently corrupt the ``argsort`` that ranks the terms.

    Args:
        recwarn: the pytest warning-recorder fixture.
    """
    # The final term has a rate of exactly zero.
    circuit, bounds, rates = _build_inputs([1.0, 0.5, 0.2], [1e-3, 2e-3, 0.0])

    with np.errstate(divide="raise", invalid="raise"):
        scales, sampling_cost, bias_remaining = compute_local_scales(
            circuit,
            bounds,
            rates,
            bias_tolerance=0.0,
            multiple_observables=True,
        )

    assert not np.any(np.isnan(scales["m0"]))
    assert not np.isnan(sampling_cost)
    assert not np.isnan(bias_remaining)

    # A zero-rate term contributes no bias, so there is nothing to mitigate for it.
    np.testing.assert_allclose(scales["m0"], [1.0, 1.0, 0.0])

    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_rejects_both_budget_and_tolerance():
    """Test that specifying both a sampling cost budget and a bias tolerance raises."""
    circuit, bounds, rates = _build_inputs([1.0, 0.5, 0.2], [1e-3, 2e-3, 3e-3])

    with pytest.raises(ValueError, match="Only one of either"):
        compute_local_scales(
            circuit,
            bounds,
            rates,
            sampling_cost_budget=10.0,
            bias_tolerance=0.1,
            multiple_observables=True,
        )
