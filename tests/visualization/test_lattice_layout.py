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

"""Tests for placing the qubits of a coupling map on a square lattice."""

from __future__ import annotations

import pytest
from qiskit.transpiler import CouplingMap
from qiskit_addon_slc.visualization import _lattice_layout
from qiskit_addon_slc.visualization._lattice_layout import _get_coords

# A miniature coupling map following the heavy-hex numbering convention of IBM's processors: a row
# of consecutively-numbered qubits (0-3), then the connectors branching off it (4, 5), then the next
# row (6-9).
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


@pytest.mark.parametrize(
    "coupling_map",
    [
        HEAVY_HEX_LIKE,
        CouplingMap.from_grid(4, 5),
        CouplingMap.from_heavy_hex(3),
        CouplingMap.from_heavy_hex(5),
        CouplingMap.from_line(7),
        CouplingMap.from_ring(4),
        CouplingMap.from_ring(8),
        CouplingMap.from_ring(14),
    ],
    ids=[
        "heavy_hex_like",
        "grid",
        "heavy_hex_3",
        "heavy_hex_5",
        "line",
        "ring_4",
        "ring_8",
        "ring_14",
    ],
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
            (x_0, y_0), (x_1, y_1) = coords[qubit_0], coords[qubit_1]
            assert abs(x_0 - x_1) + abs(y_0 - y_1) == 1, (
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
        (x_0, y_0), (x_1, y_1) = coords[qubit_0], coords[qubit_1]
        assert abs(x_0 - x_1) + abs(y_0 - y_1) == 1


def test_coords_reject_a_disconnected_map():
    """Test that a coupling map in two disconnected halves raises."""
    with pytest.raises(ValueError, match="not connected"):
        _get_coords(CouplingMap([(0, 1), (2, 3)]))


@pytest.mark.parametrize("num_qubits", [3, 5, 6, 7, 9])
def test_coords_reject_an_undrawable_ring(num_qubits):
    """Test that a ring which cannot be folded into a rectangle raises a ring-specific error.

    A grid is bipartite and therefore contains no odd cycle, so no odd ring can be drawn. The
    6-ring is even but equally impossible: its only placement is the perimeter of a 2x3 block, whose
    two middle qubits land in adjacent cells without being coupled, which would draw a coupling that
    does not exist.

    Args:
        num_qubits: the size of the ring to reject.
    """
    with pytest.raises(ValueError, match=f"ring of {num_qubits} qubits"):
        _get_coords(CouplingMap.from_ring(num_qubits))


def test_coords_draw_no_uncoupled_adjacency(subtests):
    """Test that no two qubits are placed in adjacent cells unless they are actually coupled.

    Drawing two uncoupled qubits side by side would imply a coupling the device does not have.

    Args:
        subtests: the pytest-subtests fixture.
    """
    for coupling_map, name in [
        (CouplingMap.from_ring(8), "ring_8"),
        (CouplingMap.from_ring(10), "ring_10"),
        (CouplingMap.from_line(7), "line_7"),
        (CouplingMap.from_heavy_hex(3), "heavy_hex_3"),
    ]:
        with subtests.test(name):
            coords = _get_coords(coupling_map)
            edges = {tuple(sorted(edge)) for edge in coupling_map.get_edges()}
            occupant = {cell: qubit for qubit, cell in enumerate(coords)}
            for (row, col), qubit in occupant.items():
                for row_step, col_step in ((0, 1), (1, 0)):
                    neighbor = occupant.get((row + row_step, col + col_step))
                    if neighbor is not None:
                        assert tuple(sorted((qubit, neighbor))) in edges, (
                            f"qubits {qubit} and {neighbor} are drawn adjacent but not coupled"
                        )


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

    num_cols = max(x for x, _ in coords) - min(x for x, _ in coords) + 1
    num_rows = max(y for _, y in coords) - min(y for _, y in coords) + 1
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
        heights = {coords[qubit][1] for qubit in row}
        assert len(heights) == 1, f"row {row} is not straight: {[coords[q] for q in row]}"
        columns = sorted(coords[qubit][0] for qubit in row)
        assert columns == list(range(columns[0], columns[0] + len(row)))


# The layouts the real IBM devices are expected to come out as, keyed by fake-backend name. These
# pin the shape of each layout rather than its exact coordinates: a change to the seeding or the
# scoring legitimately moves individual qubits, but it should not make a device wider, push qubit 0
# off the top-left corner, or bend the straight lines of its numbering. Those are what make a layout
# recognisable as the device, and a plain lattice search can satisfy every invariant above while
# getting all three wrong.
#
# ``width``/``height`` are the extents in cells, ``longest_line`` the longest run of consecutively
# numbered qubits sharing a row, and ``max_wasted`` the fraction of cells that may be left empty.
_EXPECTED_DEVICE_LAYOUTS = {
    "FakeAlgiers": {"width": 9, "height": 6, "longest_line": 4, "max_wasted": 0.55},
    "FakeSherbrooke": {"width": 15, "height": 13, "longest_line": 15, "max_wasted": 0.40},
    "FakeTorino": {"width": 15, "height": 14, "longest_line": 15, "max_wasted": 0.40},
    "FakePittsburgh": {"width": 16, "height": 15, "longest_line": 16, "max_wasted": 0.40},
    # A square-connectivity device tiles its grid exactly, with no wasted cells at all.
    "FakeNighthawk": {"width": 10, "height": 12, "longest_line": 10, "max_wasted": 0.0},
}


@pytest.mark.parametrize("backend_name", sorted(_EXPECTED_DEVICE_LAYOUTS))
def test_real_devices_lay_out_as_expected(backend_name, subtests):
    """Test that each real IBM device comes out in its known-good shape.

    Args:
        backend_name: the name of the fake backend to lay out.
        subtests: the pytest-subtests fixture.
    """
    fake_provider = pytest.importorskip("qiskit_ibm_runtime.fake_provider")
    coupling_map = CouplingMap(getattr(fake_provider, backend_name)().coupling_map)
    expected = _EXPECTED_DEVICE_LAYOUTS[backend_name]

    coords = _get_coords(coupling_map)

    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]
    width = max(xs) - min(xs) + 1
    height = max(ys) - min(ys) + 1

    with subtests.test("the layout has the expected extents"):
        assert (width, height) == (expected["width"], expected["height"])

    with subtests.test("qubit 0 is in the top-left corner"):
        assert coords[0] == (min(xs), max(ys))

    with subtests.test("the numbering lines are straight"):
        longest_line = run = 1
        for qubit in range(1, len(coords)):
            run = run + 1 if coords[qubit][1] == coords[qubit - 1][1] else 1
            longest_line = max(longest_line, run)
        assert longest_line == expected["longest_line"]

    with subtests.test("the layout is no sparser than expected"):
        wasted = 1 - coupling_map.size() / (width * height)
        assert wasted <= expected["max_wasted"]

    with subtests.test("every coupling still spans exactly one cell"):
        for qubit_0, qubit_1 in {tuple(sorted(edge)) for edge in coupling_map.get_edges()}:
            (x_0, y_0), (x_1, y_1) = coords[qubit_0], coords[qubit_1]
            assert abs(x_0 - x_1) + abs(y_0 - y_1) == 1


def test_coords_respect_the_search_budget(monkeypatch):
    """Test that an exhausted search budget raises rather than running unbounded.

    The budget is a module-wide constant rather than an argument, since nothing has a reason to vary
    it, so this patches the constant to a value no real search can fit inside.

    Args:
        monkeypatch: the pytest monkeypatch fixture.
    """
    monkeypatch.setattr(_lattice_layout, "_MAX_SEARCH_STEPS", 1)

    with pytest.raises(ValueError, match="search steps"):
        _get_coords(CouplingMap.from_heavy_hex(5))
