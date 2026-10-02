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
from qiskit_addon_slc.visualization.animate_shaded_lightcone import (
    _NUM_HUES,
    _get_coords,
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


@pytest.mark.parametrize(
    "coupling_map",
    [
        HEAVY_HEX_LIKE,
        CouplingMap.from_grid(4, 5),
        CouplingMap.from_heavy_hex(3),
        CouplingMap.from_heavy_hex(5),
        CouplingMap.from_line(7),
        CouplingMap.from_ring(8),
    ],
    ids=["heavy_hex_like", "grid", "heavy_hex_3", "heavy_hex_5", "line", "ring"],
)
def test_coords_place_every_qubit_on_a_lattice(coupling_map, subtests):
    """Test that the layout places every qubit so that each coupling is a unit grid step.

    The layout is derived from the graph alone, so it must work for square connectivity (a grid or a
    Nighthawk-style device) as well as for heavy-hex, and regardless of the qubit numbering.

    Args:
        coupling_map: the coupling map to lay out.
        subtests: the pytest-subtests fixture.
    """
    coords = _get_coords(coupling_map)

    with subtests.test("every qubit is placed"):
        assert len(coords) == coupling_map.size()

    with subtests.test("no two qubits share a cell"):
        assert len(set(coords)) == coupling_map.size()

    with subtests.test("every coupling spans exactly one cell"):
        for qubit_0, qubit_1 in {tuple(sorted(edge)) for edge in coupling_map.get_edges()}:
            (row_0, col_0), (row_1, col_1) = coords[qubit_0], coords[qubit_1]
            assert abs(row_0 - row_1) + abs(col_0 - col_1) == 1, (
                f"qubits {qubit_0} and {qubit_1} are coupled but not grid-adjacent"
            )


def test_coords_are_numbering_independent():
    """Test that relabelling the qubits does not change whether the layout succeeds.

    The previous layout inferred rows from the qubit numbering, so permuting the labels of an
    otherwise identical device made it fail outright.
    """
    permutation = [3, 1, 4, 0, 2, 7, 5, 9, 6, 8]
    relabelled = CouplingMap(
        [(permutation[qubit_0], permutation[qubit_1]) for qubit_0, qubit_1 in HEAVY_HEX_LIKE]
    )

    coords = _get_coords(relabelled)

    assert len(coords) == relabelled.size()
    for qubit_0, qubit_1 in {tuple(sorted(edge)) for edge in relabelled.get_edges()}:
        (row_0, col_0), (row_1, col_1) = coords[qubit_0], coords[qubit_1]
        assert abs(row_0 - row_1) + abs(col_0 - col_1) == 1


def test_coords_reject_a_disconnected_map():
    """Test that a coupling map in two disconnected halves raises."""
    with pytest.raises(ValueError, match="not connected"):
        _get_coords(CouplingMap([(0, 1), (2, 3)]))


def test_coords_reject_a_non_planar_map():
    """Test that a coupling map too densely connected for a grid raises.

    A qubit coupled to five others cannot be drawn on a grid, where a cell has only four neighbors.
    """
    with pytest.raises(ValueError, match="at most 4"):
        _get_coords(CouplingMap([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]))


def test_square_lattice_is_placed_densely(subtests):
    """Test that a square lattice is placed with no wasted grid cells.

    A square lattice is uniquely placeable, so its layout should be exactly as large as the device.

    Args:
        subtests: the pytest-subtests fixture.
    """
    coupling_map = CouplingMap.from_grid(4, 5)

    coords = _get_coords(coupling_map)

    num_rows = max(row for row, _ in coords) + 1
    num_cols = max(col for _, col in coords) + 1
    with subtests.test("the grid has one cell per qubit"):
        assert num_rows * num_cols == coupling_map.size()

    with subtests.test("the grid has the dimensions of the device"):
        assert sorted((num_rows, num_cols)) == [4, 5]


def test_heavy_hex_rows_are_laid_out_straight():
    """Test that a heavy-hex device's rows come out as straight, evenly-spaced lines.

    A plain lattice search finds a valid but visibly distorted embedding for heavy-hex, so the rows
    are pre-placed instead. This checks that they really do end up straight: the qubits of each row
    must share a lattice row and occupy consecutive columns.
    """
    # Three rows of five, joined by single-qubit rungs, numbered the way IBM numbers its devices.
    coupling_map = CouplingMap(
        [
            *[(0, 1), (1, 2), (2, 3), (3, 4)],  # row 0
            *[(0, 5), (5, 7), (2, 6), (6, 9)],  # rungs down to row 1
            *[(7, 8), (8, 9), (9, 10), (10, 11)],  # row 1
            *[(7, 12), (12, 14), (9, 13), (13, 16)],  # rungs down to row 2
            *[(14, 15), (15, 16), (16, 17), (17, 18)],  # row 2
        ]
    )

    coords = _get_coords(coupling_map)

    for row in ([0, 1, 2, 3, 4], [7, 8, 9, 10, 11], [14, 15, 16, 17, 18]):
        lattice_rows = {coords[qubit][0] for qubit in row}
        assert len(lattice_rows) == 1, f"row {row} is not straight: {[coords[q] for q in row]}"
        columns = sorted(coords[qubit][1] for qubit in row)
        assert columns == list(range(columns[0], columns[0] + len(row)))


def test_coords_respect_the_search_budget():
    """Test that an exhausted search budget raises rather than running unbounded."""
    with pytest.raises(ValueError, match="search steps"):
        _get_coords(CouplingMap.from_heavy_hex(5), max_search_steps=1)


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

    with subtests.test("an intermediate value indexes into the scale"):
        assert _get_rgb_color(colorscale, 0.5, "default", "out") == colorscale[500]

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
