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

"""Tests for the animated shaded lightcone visualization."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import BoxOp
from qiskit.quantum_info import PauliLindbladMap
from qiskit.transpiler import CouplingMap
from qiskit_addon_slc.utils import generate_noise_model_paulis
from qiskit_addon_slc.visualization import animate_shaded_lightcone
from qiskit_addon_slc.visualization._lattice_layout import _get_coords
from qiskit_addon_slc.visualization.animate_shaded_lightcone import (
    _NUM_HUES,
    _PIE_RADIUS,
    _get_rgb_color,
    _pie_slice,
)
from samplomatic import InjectNoise
from samplomatic.utils import find_unique_box_instructions

# A miniature coupling map following the heavy-hex numbering convention of IBM's processors: a row
# of consecutively-numbered qubits (0-3), then the connectors branching off it (4, 5), then the next
# row (6-9). This mirrors how e.g. Eagle numbers its first row 0-13, its connectors 14-17 and its
# second row from 18 onwards.
HEAVY_HEX_LIKE = CouplingMap(
    [
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 4),
        (4, 6),
        (2, 5),
        (5, 8),
        (6, 7),
        (7, 8),
        (8, 9),
    ]
)


def _build_circuit_and_bounds(
    coupling_map: CouplingMap, num_boxes: int = 2
) -> tuple[QuantumCircuit, dict[str, PauliLindbladMap]]:
    """Builds a boxed circuit on the coupling map plus synthetic bounds for each box.

    Args:
        coupling_map: the coupling map to build the circuit for.
        num_boxes: the number of boxes to place in the circuit.

    Returns:
        A tuple of the circuit and its bounds.
    """
    num_qubits = coupling_map.size()
    edges = sorted({tuple(sorted(edge)) for edge in coupling_map.get_edges()})

    circuit = QuantumCircuit(num_qubits)
    for idx in range(num_boxes):
        body = QuantumCircuit(num_qubits)
        # Use a different, non-overlapping edge per box so the frames differ.
        qubit_0, qubit_1 = edges[idx % len(edges)]
        body.cx(qubit_0, qubit_1)
        annotation = InjectNoise(ref=f"noise_{idx}", modifier_ref=f"box_{idx}")
        circuit.append(BoxOp(body, annotations=[annotation]), range(num_qubits))

    noise_model_paulis = generate_noise_model_paulis(
        find_unique_box_instructions(circuit), coupling_map=coupling_map, circuit=circuit
    )

    rates = {"X": 0.01, "Y": 0.02, "Z": 0.03}
    bounds = {}
    for idx in range(num_boxes):
        terms = noise_model_paulis[f"noise_{idx}"]
        values = np.array(
            [sum(rates[pauli] for pauli in label) for label, _ in terms.to_sparse_list()]
        ) * (idx + 1)
        bounds[f"box_{idx}"] = PauliLindbladMap.from_components(values, terms)

    return circuit, bounds


@pytest.mark.parametrize("reverse", [False, True])
def test_animation_has_one_frame_per_box(reverse):
    """Test that the animation contains one frame for each box with bounds.

    Args:
        reverse: whether to animate the circuit layers in reverse order.
    """
    circuit, bounds = _build_circuit_and_bounds(HEAVY_HEX_LIKE, num_boxes=3)

    figure = animate_shaded_lightcone(circuit, bounds, HEAVY_HEX_LIKE, reverse=reverse)

    assert len(figure.frames) == 3


def test_animation_skips_boxes_without_bounds():
    """Test that boxes missing from ``bounds`` are skipped rather than truncating the animation.

    This is what happens when the bounds computation was limited via ``max_num_boxes``.
    """
    circuit, bounds = _build_circuit_and_bounds(HEAVY_HEX_LIKE, num_boxes=3)
    # Drop the middle box, as if bounds had only been computed for a subset.
    del bounds["box_1"]

    figure = animate_shaded_lightcone(circuit, bounds, HEAVY_HEX_LIKE)

    # The two remaining boxes are still animated; the gap does not cut the animation short.
    assert len(figure.frames) == 2


def _shape_extent(shape) -> tuple[float, float, float, float] | None:
    """Returns the ``(xlo, xhi, ylo, yhi)`` bounding box of a plotly shape, or ``None`` to skip it.

    Args:
        shape: the plotly shape to measure.

    Returns:
        The bounding box, or ``None`` if the shape is of a type that is not measured here.
    """
    if shape.type == "rect":
        return shape.x0, shape.x1, shape.y0, shape.y1
    if shape.type == "path":
        cleaned = shape.path
        for token in ("M", "L", "A", "Z", ","):
            cleaned = cleaned.replace(token, " ")
        numbers = [float(token) for token in cleaned.split()]
        x_coords, y_coords = numbers[0::2], numbers[1::2]
        return min(x_coords), max(x_coords), min(y_coords), max(y_coords)
    return None


@pytest.mark.parametrize(
    "coupling_map",
    [
        CouplingMap.from_ring(4),
        CouplingMap.from_grid(2, 2),
        CouplingMap.from_line(3),
        CouplingMap.from_line(7),
        CouplingMap.from_grid(4, 5),
        HEAVY_HEX_LIKE,
    ],
    ids=["ring_4", "grid_2x2", "line_3", "line_7", "grid_4x5", "heavy_hex_like"],
)
def test_legend_is_fully_inside_the_axes(coupling_map):
    """Test that the legend is drawn entirely within the axis ranges.

    The legend is a fixed-size key drawn in data coordinates, so the axis ranges have to make room
    for it. They previously did not: its top sat half a unit above the y range on every device, and
    on a map narrower than the key itself it also ran off the left edge.

    Args:
        coupling_map: the coupling map to animate.
    """
    circuit, bounds = _build_circuit_and_bounds(coupling_map)

    figure = animate_shaded_lightcone(circuit, bounds, coupling_map)

    x_range, y_range = figure.layout.xaxis.range, figure.layout.yaxis.range
    for frame in figure.frames:
        for shape in frame.layout.shapes or ():
            extent = _shape_extent(shape)
            if extent is None:
                continue
            x_low, x_high, y_low, y_high = extent
            assert x_low >= x_range[0] and x_high <= x_range[1], (
                f"shape spans x {x_low}..{x_high}, outside the axis range {x_range}"
            )
            assert y_low >= y_range[0] and y_high <= y_range[1], (
                f"shape spans y {y_low}..{y_high}, outside the axis range {y_range}"
            )


@pytest.mark.parametrize(
    "coupling_map",
    [CouplingMap.from_line(3), CouplingMap.from_grid(2, 2), CouplingMap.from_line(7)],
    ids=["line_3", "grid_2x2", "line_7"],
)
def test_legend_sits_above_the_lattice(coupling_map):
    """Test that the legend occupies its reserved band and never covers a qubit.

    The legend used to be placed by subtracting a fixed offset from the rightmost column, which put
    it on top of the qubits whenever the layout was narrower than the legend itself.

    Args:
        coupling_map: the coupling map to animate.
    """
    circuit, bounds = _build_circuit_and_bounds(coupling_map)

    figure = animate_shaded_lightcone(circuit, bounds, coupling_map)

    # ``_get_coords`` already returns plotting coordinates, so the topmost row sits at the maximum
    # ``y``. The bounds on a qubit are drawn as a pie of radius ``_PIE_RADIUS`` around it, so
    # anything belonging to the lattice stays within that of a row.
    top_of_lattice = max(y for _, y in _get_coords(coupling_map))
    lattice_ceiling = top_of_lattice + _PIE_RADIUS

    # The legend is the only thing drawn clear of the lattice, so every shape above that ceiling is
    # part of it -- and all of it must stay above the ceiling rather than reaching onto a qubit.
    legend_shapes = [
        extent
        for frame in figure.frames
        for shape in frame.layout.shapes or ()
        if (extent := _shape_extent(shape)) is not None and extent[3] > lattice_ceiling
    ]
    assert legend_shapes, "expected the legend to be drawn above the lattice"
    for _, _, y_low, _ in legend_shapes:
        assert y_low >= lattice_ceiling, (
            f"a legend shape reaches down to y={y_low}, onto the lattice which tops out at "
            f"y={lattice_ceiling}"
        )


def test_animation_rejects_a_circuit_wider_than_the_coupling_map():
    """Test that a circuit acting on more qubits than the map has raises a clear error.

    The rates are keyed by the map's physical qubits while the Pauli indices come from the circuit
    width, so a mismatch used to surface as a bare ``KeyError`` deep inside the rate accumulation.
    """
    coupling_map = CouplingMap.from_line(6)
    circuit, bounds = _build_circuit_and_bounds(coupling_map)

    with pytest.raises(ValueError, match="acts on 6 qubits but the coupling map has 4"):
        animate_shaded_lightcone(circuit, bounds, CouplingMap.from_line(4))


def test_animation_rejects_bounds_with_no_frames():
    """Test that bounds matching none of the circuit's boxes raise rather than an ``IndexError``.

    This is what a ``bounds`` dict with mistyped keys looks like, and it used to fail with a bare
    ``IndexError`` from indexing the empty list of frames.
    """
    coupling_map = CouplingMap.from_line(4)
    circuit, _ = _build_circuit_and_bounds(coupling_map)

    with pytest.raises(ValueError, match="nothing to animate"):
        animate_shaded_lightcone(circuit, {}, coupling_map)


def test_animation_rejects_a_two_qubit_bound_off_the_coupling_map():
    """Test that a weight-2 bound on an uncoupled pair raises a clear error.

    ``generate_noise_model_paulis`` defaults to a 1D line when called without a coupling map, so
    bounds built that way and then drawn on a real device can carry terms the device has no edge
    for. That used to surface as a bare ``KeyError`` naming only the qubit pair.
    """
    # A tree on 4 qubits, deliberately lacking the ``(1, 2)`` coupling that a line would have.
    coupling_map = CouplingMap([(0, 1), (0, 2), (2, 3)])
    circuit = QuantumCircuit(4)
    body = QuantumCircuit(4)
    body.cx(0, 1)
    annotation = InjectNoise(ref="noise_0", modifier_ref="box_0")
    circuit.append(BoxOp(body, annotations=[annotation]), range(4))

    # Omitting ``coupling_map`` makes this fall back to a line, which includes the ``(1, 2)`` pair.
    terms = generate_noise_model_paulis(find_unique_box_instructions(circuit), circuit=circuit)[
        "noise_0"
    ]
    rates = np.full(len(terms.to_sparse_list()), 0.01)
    bounds = {"box_0": PauliLindbladMap.from_components(rates, terms)}

    with pytest.raises(ValueError, match=r"2-qubit term on qubits \(1, 2\), which are not coupled"):
        animate_shaded_lightcone(circuit, bounds, coupling_map)


def test_animation_rejects_a_higher_weight_bound():
    """Test that a bound acting on three or more qubits raises rather than being dropped.

    Only weight-1 and weight-2 terms have somewhere to be drawn, so a higher-weight term used to be
    silently ignored, which under-reports the error without saying so.
    """
    coupling_map = CouplingMap.from_line(3)
    circuit = QuantumCircuit(3)
    body = QuantumCircuit(3)
    body.cx(0, 1)
    annotation = InjectNoise(ref="noise_0", modifier_ref="box_0")
    circuit.append(BoxOp(body, annotations=[annotation]), range(3))

    bounds = {
        "box_0": PauliLindbladMap.from_sparse_list(
            [("XXX", [0, 1, 2], 0.01), ("X", [0], 0.02)], num_qubits=3
        )
    }

    with pytest.raises(ValueError, match="term acting on 3 qubits"):
        animate_shaded_lightcone(circuit, bounds, coupling_map)


def test_rgb_color_bounds(subtests):
    """Test the colorscale mapping at and beyond its bounds.

    Args:
        subtests: the pytest-subtests fixture.
    """
    colorscale = [f"rgb(0,0,{idx})" for idx in range(_NUM_HUES)]

    with subtests.test("zero returns the default"):
        assert _get_rgb_color(colorscale, 0.0, "default", "out") == "default"

    with subtests.test("one returns the last hue"):
        assert _get_rgb_color(colorscale, 1.0, "default", "out") == colorscale[-1]

    with subtests.test("above one returns the out-of-scale color"):
        assert _get_rgb_color(colorscale, 1.5, "default", "out") == "out"

    with subtests.test("below zero returns the out-of-scale color"):
        # Error rates are non-negative, so a negative value is as out of scale as one above 1.
        assert _get_rgb_color(colorscale, -0.5, "default", "out") == "out"

    with subtests.test("an intermediate value indexes into the scale"):
        assert _get_rgb_color(colorscale, 0.5, "default", "out") == colorscale[500]

    with subtests.test("a value that rounds up to one stays inside the scale"):
        # ``np.round(val, 3)`` carries anything from 0.9995 up to 1.0, which used to index one past
        # the end of the colorscale and raise an ``IndexError``.
        for value in (0.9995, 0.9996, 0.99999):
            assert _get_rgb_color(colorscale, value, "default", "out") == colorscale[-1]

    with (
        subtests.test("a wrongly sized colorscale raises"),
        pytest.raises(ValueError, match="Expected a colorscale"),
    ):
        _get_rgb_color(["rgb(0,0,0)"], 0.5, "default", "out")


def test_pie_slice_is_a_closed_path():
    """Test that a pie slice path starts at its arc and closes back on the center."""
    path = _pie_slice(0, 120, 1.0, 2.0, 0.5)

    assert path.startswith("M ")
    # The path returns to the center of the pie and closes.
    assert path.endswith("L1.0,2.0 Z")
