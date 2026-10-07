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
    bounds: list[float], noise_rates: list[float]
) -> tuple[QuantumCircuit, dict[str, PauliLindbladMap], dict[str, PauliLindbladMap]]:
    """Builds a minimal single-box circuit plus matching bounds and noise rates.

    Only ``BoxOp`` instructions carrying an :class:`~samplomatic.InjectNoise` annotation are
    inspected, so a hand-built circuit is enough; no transpilation is needed.

    The first ``len(bounds)`` entries of ``PAULIS`` are used as the error terms.

    Args:
        bounds: the commutator bounds to use, one per error term.
        noise_rates: the learned noise rates to use, one per error term.

    Returns:
        A tuple of the circuit, the bounds and the noise rates.
    """
    body = QuantumCircuit(2)
    body.cx(0, 1)
    annotation = InjectNoise(ref="noise", modifier_ref="box")

    circuit = QuantumCircuit(2)
    circuit.append(BoxOp(body, annotations=[annotation]), [0, 1])

    paulis = QubitSparsePauliList.from_list(PAULIS[: len(bounds)])
    return (
        circuit,
        {"box": PauliLindbladMap.from_components(np.asarray(bounds), paulis)},
        {"noise": PauliLindbladMap.from_components(np.asarray(noise_rates), paulis)},
    )


@pytest.mark.parametrize("apply_in_post", [False, True])
def test_mitigates_everything_within_an_unlimited_budget(apply_in_post):
    """Test that every term is mitigated when the sampling cost is unconstrained.

    Args:
        apply_in_post: whether to use the post-processing prioritization.
    """
    circuit, bounds, noise_rates = _build_inputs([1.0, 0.5], [1e-2, 2e-2])

    scales, sampling_cost, bias_remaining = compute_local_scales(
        circuit, bounds, noise_rates, sampling_cost_budget=np.inf, apply_in_post=apply_in_post
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


def test_prioritization_differs_from_default():
    """Test that the post-processing mode can rank terms differently than the default.

    The default prioritizes by ``bound * exp(-2 * rate)``, which penalizes a large rate only
    exponentially. The post-processing mode prioritizes by "value density", i.e. the bias bound
    divided by the rate, whose numerator saturates for a large rate so that the penalty becomes
    roughly linear. An expensive term with a large bound is therefore ranked relatively higher by
    the default than by the post-processing mode.
    """
    # The first term has the largest bound but a rate three orders of magnitude above the others.
    # Default priority ranks it last ([1, 2, 0]); value density ranks it second ([1, 0, 2]).
    bound_rates = [1.0, 0.6, 0.2]
    noise_rates = [0.9, 1e-3, 1e-3]

    circuit, bounds_map, noise_rates_map = _build_inputs(bound_rates, noise_rates)

    # This budget affords the two cheap terms but never the expensive one. Under the default
    # ranking those two come first, so both are mitigated; under value density the expensive term
    # sits between them, blocking the second cheap one.
    budget = 1.02

    scales_default, _, bias_default = compute_local_scales(
        circuit, bounds_map, noise_rates_map, sampling_cost_budget=budget, apply_in_post=False
    )
    scales_post, _, bias_post = compute_local_scales(
        circuit, bounds_map, noise_rates_map, sampling_cost_budget=budget, apply_in_post=True
    )

    np.testing.assert_allclose(scales_default["box"], [0.0, 1.0, 1.0])
    np.testing.assert_allclose(scales_post["box"], [0.0, 1.0, 0.0])
    assert not np.array_equal(scales_default["box"], scales_post["box"])

    # The post-processing bias bound uses ``1 - exp(-2 * rate)**2`` rather than
    # ``1 - exp(-2 * rate)``, reflecting the doubled error rates of the unmitigated terms. The
    # residual bias is the sum of the bias bounds of the terms left unmitigated, which differs
    # between the two modes both in the formula and in which terms remain. Asserting on it pins the
    # doubled-rate correction itself, not just the resulting ranking.
    bounds_arr = np.asarray(bound_rates)
    exp_rates = np.exp(-2 * np.asarray(noise_rates))

    # Default mode leaves only the first term unmitigated.
    expected_bias_default = (bounds_arr * (1 - exp_rates) / 2)[0]
    # Post-processing mode leaves the first and third terms unmitigated.
    expected_bias_post = (bounds_arr * (1 - exp_rates**2) / 2)[[0, 2]].sum()

    np.testing.assert_allclose(bias_default, expected_bias_default, rtol=1e-6)
    np.testing.assert_allclose(bias_post, expected_bias_post, rtol=1e-6)
    assert bias_post > bias_default


def test_zero_rate_does_not_produce_nan(recwarn):
    """Test that a noise rate of exactly zero is handled without dividing by zero.

    The post-processing priority is a ratio whose denominator is the noise rate. Where the rate is
    exactly zero the numerator is zero too, so the priority must be defined as zero rather than
    evaluating to ``nan``. A ``nan`` would silently corrupt the ``argsort`` that ranks the terms.

    Args:
        recwarn: the pytest warning-recorder fixture.
    """
    # The final term has a rate of exactly zero.
    circuit, bounds, noise_rates = _build_inputs([1.0, 0.5, 0.2], [1e-3, 2e-3, 0.0])

    with np.errstate(divide="raise", invalid="raise"):
        scales, sampling_cost, bias_remaining = compute_local_scales(
            circuit,
            bounds,
            noise_rates,
            bias_tolerance=0.0,
            apply_in_post=True,
        )

    assert not np.any(np.isnan(scales["box"]))
    assert not np.isnan(sampling_cost)
    assert not np.isnan(bias_remaining)

    # A zero-rate term contributes no bias, so there is nothing to mitigate for it.
    np.testing.assert_allclose(scales["box"], [1.0, 1.0, 0.0])

    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


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
