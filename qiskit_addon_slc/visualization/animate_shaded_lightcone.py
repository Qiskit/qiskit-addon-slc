# This code is a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Warning: this module is not documented and it does not have an RST file.
# If we ever publicly expose interfaces users can import from this module,
# we should set up its RST file.

"""An animated visualization of a shaded lightcone, powered by ``plotly``."""

from collections import deque

import numpy as np
from plotly import graph_objects as go
from plotly.colors import sample_colorscale
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import PauliLindbladMap
from qiskit.transpiler import CouplingMap

from ..bounds.commutator_bounds import Bounds
from ..utils import find_indices, iter_circuit

# The number of hues in the discreet colorscales built by this module.
_NUM_HUES = 1000

# The four unit steps of a square lattice, used to lay the qubits out on a grid.
_LATTICE_STEPS = ((0, 1), (1, 0), (0, -1), (-1, 0))

# The default search budget of :func:`_get_coords`, per candidate layout. Every IBM device tried
# needs fewer than ~10_000 steps, so this leaves a wide margin while still bounding pathological
# inputs.
_MAX_SEARCH_STEPS = 200_000

# The shortest run of consecutively-numbered qubits that :func:`_lattice_rows` treats as a row rather
# than as a rung between rows. The rungs of IBM's heavy-hex devices are single qubits and the rows are
# 14 or more, so anything in between separates the two cleanly.
_MIN_ROW_LENGTH = 5

# How far a heavy-hex row may be indented relative to the one above it. The rows of a device are not
# all the same length and the shorter ones sit inset by a qubit or two.
_MAX_ROW_INDENT = 2

# How many starting qubits :func:`_get_coords` tries before keeping the most compact layout.
_NUM_LAYOUT_CANDIDATES = 12


def _get_rgb_color(
    discreet_colorscale: list[str], val: float, default: str, color_out_of_scale: str
) -> str:
    """Maps a float to an RGB color based on a discreet colorscale.

    Args:
        discreet_colorscale: a discreet colorscale of :data:`_NUM_HUES` hues.
        val: a value in ``[0, 1]`` to map to a color.
        default: the color returned when ``val`` is ``0``.
        color_out_of_scale: the color returned when ``val`` exceeds ``1``.

    Returns:
        The color as a string.

    Raises:
        ValueError: if the colorscale does not contain :data:`_NUM_HUES` hues.
    """
    if len(discreet_colorscale) != _NUM_HUES:
        raise ValueError(
            f"Expected a colorscale of {_NUM_HUES} hues, but got {len(discreet_colorscale)}."
        )

    if val > 1:
        return color_out_of_scale
    if val == 1:
        return discreet_colorscale[-1]
    if val == 0:
        return default
    return discreet_colorscale[int(np.round(val, 3) * _NUM_HUES)]


def _pie_slice(angle_st: float, angle_end: float, x: float, y: float, radius: float) -> str:
    """Returns an SVG path drawing a slice of a pie chart.

    Pie charts are drawn as paths and shapes rather than :class:`~plotly.graph_objects.Pie` objects
    because those are easier to place at a specific location.

    Args:
        angle_st: the angle (in degrees) at which the slice begins.
        angle_end: the angle (in degrees) at which the slice ends.
        x: the ``x`` coordinate of the center of the pie.
        y: the ``y`` coordinate of the center of the pie.
        radius: the radius of the pie.

    Returns:
        The path as a string.
    """
    angles = np.linspace(angle_st * np.pi / 180, angle_end * np.pi / 180, 10)

    path_xs = x + radius * np.cos(angles)
    path_ys = y + radius * np.sin(angles)
    path = f"M {path_xs[0]},{path_ys[0]}"

    for x_coord, y_coord in zip(path_xs[1:], path_ys[1:], strict=True):
        path += f" L{x_coord},{y_coord}"
    path += f"L{x},{y} Z"

    return path


def _lattice_rows(coupling_map: CouplingMap, min_length: int) -> list[list[int]]:
    """Finds the rows of a heavy-hex coupling map from its qubit numbering.

    IBM numbers its heavy-hex processors row by row: a row is a maximal run of consecutively-numbered
    qubits that are also coupled, and the qubits numbered in between the runs are the rungs bridging
    one row to the next. On an Eagle device this yields rows of 14-15 qubits; on Heron, 15-16.

    This is a deliberate use of the numbering, which :func:`_embed_in_lattice` does not rely on. The
    reason is that heavy-hex offers no *local* way to tell a row-interior qubit from a rung: both are
    degree-2 qubits sitting between two degree-3 qubits, and because the lattice has no short cycles
    (its shortest is a 12-cycle) there is no equivalent of the 4-cycle test that makes square
    lattices easy. Distinguishing them from the graph alone needs global structure, so the numbering
    is used as a hint and the result is verified afterwards.

    Args:
        coupling_map: the coupling map whose rows to find.
        min_length: the minimum number of qubits for a run to count as a row.

    Returns:
        The rows, in numbering order. Empty if fewer than two rows were found, which means the
        numbering does not follow the convention and the caller should fall back to a plain search.
    """
    graph = coupling_map.graph.to_undirected()
    neighbors = {qubit: set(graph.neighbors(qubit)) for qubit in range(coupling_map.size())}

    runs: list[list[int]] = [[0]]
    for qubit in range(1, coupling_map.size()):
        if qubit - 1 in neighbors[qubit]:
            runs[-1].append(qubit)
        else:
            runs.append([qubit])

    rows = [run for run in runs if len(run) >= min_length]
    # A single row tells us nothing about the spacing between rows, so it is not worth seeding with.
    return rows if len(rows) >= 2 else []


def _place_square_lattice(coupling_map: CouplingMap) -> dict[int, tuple[int, int]] | None:
    """Places the qubits of a square-lattice coupling map, without any search.

    A square lattice is uniquely placeable because of its 4-cycles: stepping from one qubit to the
    next, exactly one neighbor continues in a straight line, since turning would close a 4-cycle.
    Walking the two straight lines out of a corner therefore recovers the axes directly, and the rest
    of the grid follows by walking each row parallel to the first.

    Args:
        coupling_map: the coupling map to place.

    Returns:
        A mapping of qubit to its ``(row, column)`` cell, or ``None`` if the map does not have the
        expected structure, in which case the caller should fall back to a plain search.
    """
    graph = coupling_map.graph.to_undirected()
    num_qubits = coupling_map.size()
    neighbors = {qubit: set(graph.neighbors(qubit)) for qubit in range(num_qubits)}

    def _straight_line(previous: int, current: int) -> list[int]:
        """Walks in a straight line, continuing until the next step is ambiguous."""
        line = [previous, current]
        while True:
            previous, current = line[-2], line[-1]
            # Turning left or right closes a 4-cycle with ``previous``, so the straight continuation
            # is the neighbor sharing no *other* qubit with it. ``current`` itself is always shared
            # and must be excluded.
            straight = [
                candidate
                for candidate in neighbors[current]
                if candidate != previous
                and not (neighbors[candidate] & neighbors[previous]) - {current}
            ]
            if len(straight) != 1:
                return line
            line.append(straight[0])

    # The corners of a square lattice are its only degree-2 qubits.
    corners = [qubit for qubit in range(num_qubits) if len(neighbors[qubit]) == 2]
    if not corners:
        return None

    axes = sorted(
        (_straight_line(corners[0], step) for step in sorted(neighbors[corners[0]])), key=len
    )
    if len(axes) != 2:
        return None
    first_row, first_column = axes

    positions = {qubit: (0, column) for column, qubit in enumerate(first_row)}
    positions.update({qubit: (row, 0) for row, qubit in enumerate(first_column)})

    # Every remaining row starts at the corresponding qubit of the first column and runs parallel to
    # the first row.
    for row, anchor in enumerate(first_column):
        if row == 0:
            continue
        unplaced = sorted(neighbors[anchor] - positions.keys())
        if not unplaced:
            continue
        for column, qubit in enumerate(_straight_line(anchor, unplaced[0])):
            positions.setdefault(qubit, (row, column))

    return positions if len(positions) == num_qubits else None


def _embed_in_lattice(
    coupling_map: CouplingMap,
    seed: int,
    max_search_steps: int,
    seed_rows: list[list[int]] | None = None,
) -> dict[int, tuple[int, int]]:
    """Embeds a coupling map into a square lattice via backtracking search.

    Every qubit is assigned an integer ``(row, column)`` cell such that each coupling-map edge spans
    exactly one cell, horizontally or vertically. The search itself is driven purely by the graph.

    Passing ``seed_rows`` pre-places those rows as straight horizontal lines two lattice rows apart,
    leaving the search only the rungs in between. Without that help the search finds an embedding
    that is valid but visibly distorted: it lays the first ten or so qubits out correctly and then
    bends the row upwards, because backtracking only revisits a placement once some *later* qubit
    becomes unplaceable, and the bent layout stays feasible forever.

    Args:
        coupling_map: the coupling map to embed.
        seed: the qubit to place first, at the origin. Ignored when ``seed_rows`` is given.
        max_search_steps: the maximum number of search steps before giving up.
        seed_rows: rows to pre-place as straight lines, as returned by :func:`_lattice_rows`.

    Returns:
        A mapping of qubit to its ``(row, column)`` cell.

    Raises:
        ValueError: if the coupling map does not embed into a square lattice, or if the search budget
            is exhausted before an embedding is found.
    """
    graph = coupling_map.graph.to_undirected()
    num_qubits = coupling_map.size()
    neighbors = {qubit: set(graph.neighbors(qubit)) for qubit in range(num_qubits)}

    positions: dict[int, tuple[int, int]] = {}
    occupied: dict[tuple[int, int], int] = {}
    steps = 0

    def _fits(qubit: int, cell: tuple[int, int]) -> bool:
        """Checks whether placing ``qubit`` at ``cell`` keeps the embedding consistent."""
        if cell in occupied:
            return False
        for row_step, col_step in _LATTICE_STEPS:
            adjacent = (cell[0] + row_step, cell[1] + col_step)
            # A qubit may only be lattice-adjacent to qubits it is actually coupled to, otherwise the
            # drawing would imply a coupling that does not exist.
            if adjacent in occupied and occupied[adjacent] not in neighbors[qubit]:
                return False
        for neighbor in neighbors[qubit]:
            if neighbor in positions:
                row, col = positions[neighbor]
                if abs(row - cell[0]) + abs(col - cell[1]) != 1:
                    return False
        return True

    def _candidate_cells(qubit: int) -> list[tuple[int, int]]:
        """Returns the cells a qubit could occupy, i.e. those next to an already-placed neighbor."""
        cells = []
        for neighbor in sorted(neighbors[qubit]):
            if neighbor in positions:
                row, col = positions[neighbor]
                cells += [(row + dr, col + dc) for dr, dc in _LATTICE_STEPS]
        # ``dict.fromkeys`` de-duplicates while keeping the order deterministic.
        return list(dict.fromkeys(cells))

    def _place(remaining: list[int], index: int) -> bool:
        """Places ``remaining[index:]``, backtracking on dead ends."""
        nonlocal steps
        steps += 1
        if steps > max_search_steps:
            raise ValueError(
                f"Could not embed this coupling map within {max_search_steps} search steps. Pass "
                "explicit coordinates, or raise `max_search_steps` if the map really is a lattice."
            )
        if index == len(remaining):
            return True

        qubit = remaining[index]
        cells = [(0, 0)] if not positions else _candidate_cells(qubit)
        for cell in cells:
            if _fits(qubit, cell):
                positions[qubit] = cell
                occupied[cell] = qubit
                if _place(remaining, index + 1):
                    return True
                del occupied[cell]
                del positions[qubit]
        return False

    if seed_rows:
        seeded = {qubit for row in seed_rows for qubit in row}
        rungs = [qubit for qubit in range(num_qubits) if qubit not in seeded]

        def _place_rows(index: int) -> bool:
            """Lays each seed row out straight, trying a few indents, then places the rungs."""
            if index == len(seed_rows):
                return _place(rungs, 0)
            # Rows of a heavy-hex device are not all the same length; the shorter ones are indented.
            # Trying a couple of offsets per row is enough to line them all up.
            for indent in range(_MAX_ROW_INDENT + 1):
                cells = [(2 * index, indent + column) for column in range(len(seed_rows[index]))]
                if any(cell in occupied for cell in cells):
                    continue
                for qubit, cell in zip(seed_rows[index], cells, strict=True):
                    positions[qubit] = cell
                    occupied[cell] = qubit
                if _place_rows(index + 1):
                    return True
                for qubit, cell in zip(seed_rows[index], cells, strict=True):
                    del positions[qubit]
                    del occupied[cell]
            return False

        if not _place_rows(0):
            raise ValueError("Could not lay out this coupling map from its rows.")
        return positions

    # Visit the qubits breadth-first, so each one is reached from an already-placed neighbor.
    visit_order = [seed]
    seen = {seed}
    queue = deque([seed])
    while queue:
        qubit = queue.popleft()
        for neighbor in sorted(neighbors[qubit]):
            if neighbor not in seen:
                seen.add(neighbor)
                visit_order.append(neighbor)
                queue.append(neighbor)

    if len(visit_order) != num_qubits:
        raise ValueError(
            "Cannot lay out a coupling map that is not connected: "
            f"only {len(visit_order)} of {num_qubits} qubits are reachable from qubit {seed}."
        )

    if not _place(visit_order, 0):
        raise ValueError(
            "This coupling map does not embed into a square lattice, so its qubits cannot be laid "
            "out on a grid. Pass explicit coordinates instead."
        )

    return positions


def _normalize_layout(
    positions: dict[int, tuple[int, int]], num_qubits: int
) -> list[tuple[int, int]]:
    """Compacts and orients a lattice embedding.

    An embedding is only defined up to translation, rotation and reflection, so the raw output is
    normalized: unused rows and columns are dropped, the layout is transposed so that
    consecutively-numbered qubits run along the rows, and it is reflected so that qubit ``0`` falls
    in the top-left quadrant. All of this is derived from the embedding itself, so it adapts to any
    device size.

    Args:
        positions: the raw embedding.
        num_qubits: the number of qubits, i.e. the length of the returned list.

    Returns:
        The ``(row, column)`` coordinate of each qubit, indexed by qubit number.
    """
    # Drop rows and columns that no qubit occupies, so the grid is as tight as the embedding allows.
    used_rows = sorted({row for row, _ in positions.values()})
    used_cols = sorted({col for _, col in positions.values()})
    row_index = {row: idx for idx, row in enumerate(used_rows)}
    col_index = {col: idx for idx, col in enumerate(used_cols)}
    positions = {qubit: (row_index[row], col_index[col]) for qubit, (row, col) in positions.items()}
    num_rows, num_cols = len(used_rows), len(used_cols)

    # Transpose if consecutively-numbered qubits run down the columns rather than across the rows.
    # IBM numbers its devices line by line, so following the numbering reproduces the conventional
    # orientation. Judging by the grid's shape instead would get the Heron devices wrong, since those
    # embed into a lattice that is taller than it is wide.
    row_travel = sum(
        abs(positions[qubit][0] - positions[qubit - 1][0]) for qubit in range(1, num_qubits)
    )
    col_travel = sum(
        abs(positions[qubit][1] - positions[qubit - 1][1]) for qubit in range(1, num_qubits)
    )
    if row_travel > col_travel:
        positions = {qubit: (col, row) for qubit, (row, col) in positions.items()}
        num_rows, num_cols = num_cols, num_rows

    # Orient so that qubit 0 lies in the top-left quadrant. It will not always reach the exact
    # corner, since qubit 0 need not sit on the boundary of the lattice.
    if positions[0][0] > (num_rows - 1) / 2:
        positions = {qubit: (num_rows - 1 - row, col) for qubit, (row, col) in positions.items()}
    if positions[0][1] > (num_cols - 1) / 2:
        positions = {qubit: (row, num_cols - 1 - col) for qubit, (row, col) in positions.items()}

    return [positions[qubit] for qubit in range(num_qubits)]


def _search_for_layout(
    coupling_map: CouplingMap, max_search_steps: int
) -> dict[int, tuple[int, int]]:
    """Searches for a lattice embedding, trying several starting qubits.

    Args:
        coupling_map: the coupling map to lay out.
        max_search_steps: the maximum number of search steps per starting qubit.

    Returns:
        A mapping of qubit to its ``(row, column)`` cell.

    Raises:
        ValueError: if no starting qubit yields an embedding.
    """
    graph = coupling_map.graph.to_undirected()
    num_qubits = coupling_map.size()

    # Seed from the lowest-degree qubits: those sit on the boundary of the lattice, where there are
    # fewest ways for the search to strand itself. Seeding from a high-degree qubit in the interior
    # costs orders of magnitude more steps.
    seeds = sorted(range(num_qubits), key=lambda qubit: (len(list(graph.neighbors(qubit))), qubit))

    best: dict[int, tuple[int, int]] | None = None
    best_score: tuple[int, int] | None = None
    failure: ValueError | None = None
    for seed in seeds[:_NUM_LAYOUT_CANDIDATES]:
        try:
            positions = _embed_in_lattice(coupling_map, seed, max_search_steps)
        except ValueError as exc:
            failure = exc
            continue
        candidate = _normalize_layout(positions, num_qubits)
        # Prefer the layout that keeps the device's numbering lines straight, falling back on the
        # tightest bounding box. Ranking by area alone yields valid but visibly tangled layouts.
        longest_run = run = 1
        for qubit in range(1, num_qubits):
            run = run + 1 if candidate[qubit][0] == candidate[qubit - 1][0] else 1
            longest_run = max(longest_run, run)
        num_rows = max(row for row, _ in candidate) + 1
        num_cols = max(col for _, col in candidate) + 1
        score = (-longest_run, num_rows * num_cols)
        if best_score is None or score < best_score:
            best = positions
            best_score = score

    if best is None:
        raise failure if failure is not None else ValueError("Could not lay out this coupling map.")
    return best


def _get_coords(
    coupling_map: CouplingMap, *, max_search_steps: int = _MAX_SEARCH_STEPS
) -> list[tuple[int, int]]:
    """Computes ``(row, column)`` coordinates for each qubit of a coupling map.

    The coupling maps of IBM's processors are subgraphs of a square lattice: every coupling connects
    two qubits that are neighbors on a grid. This function recovers such a grid placement, choosing
    its strategy from the map's maximum connectivity:

    * **degree 1 or 2** -- a path or a ring, which lays out in a single line with no choices to make;
    * **degree 3** -- a heavy-hex lattice. Its rows are pre-placed as straight lines and only the
      rungs between them are searched for, which reproduces the familiar even comb. A plain search
      finds a valid but visibly distorted layout here, because heavy-hex has no short cycles to
      indicate which way is "straight";
    * **degree 4** -- a square lattice, which is placed directly by walking its axes, no search at
      all;
    * **anything higher** -- not a planar lattice, so there is no grid placement to find.

    Each strategy falls back on a plain backtracking search if the map does not have the structure
    its generation implies -- for instance :meth:`~qiskit.transpiler.CouplingMap.from_heavy_hex`
    numbers all data qubits before all rungs, so its rows cannot be read off the numbering.

    Args:
        coupling_map: the coupling map whose qubits to place.
        max_search_steps: the maximum number of search steps before giving up.

    Returns:
        The ``(row, column)`` coordinate of each qubit, indexed by qubit number.

    Raises:
        ValueError: if the coupling map is not connected, is too densely connected to lie on a grid,
            or needs more than ``max_search_steps`` steps to lay out.
    """
    graph = coupling_map.graph.to_undirected()
    num_qubits = coupling_map.size()
    if num_qubits == 0:
        return []

    max_degree = max(len(list(graph.neighbors(qubit))) for qubit in range(num_qubits))

    positions: dict[int, tuple[int, int]] | None = None
    if max_degree <= 2:
        # A path or a ring: walk it from one end and lay it out along a single row.
        positions = _embed_in_lattice(coupling_map, 0, max_search_steps)
    elif max_degree == 3:
        rows = _lattice_rows(coupling_map, _MIN_ROW_LENGTH)
        if rows:
            try:
                positions = _embed_in_lattice(coupling_map, 0, max_search_steps, seed_rows=rows)
            except ValueError:
                # The numbering looked like rows but does not lay out, so fall through to the search.
                positions = None
    elif max_degree == 4:
        positions = _place_square_lattice(coupling_map)
    else:
        raise ValueError(
            f"Cannot lay out a coupling map whose qubits have up to {max_degree} couplings: a grid "
            "placement allows at most 4. Pass explicit coordinates instead."
        )

    if positions is None:
        positions = _search_for_layout(coupling_map, max_search_steps)

    return _normalize_layout(positions, num_qubits)


def _restrict_num_bodies(plm: PauliLindbladMap, num_qubits: int) -> PauliLindbladMap:
    if num_qubits < 0:
        raise ValueError("``num_qubits`` must be ``0`` or larger.")
    paulis = plm.get_qubit_sparse_pauli_list_copy().to_pauli_list()
    mask = np.sum(paulis.x | paulis.z, axis=1) == num_qubits
    return paulis[mask], plm.rates[mask] + 1e-17


def animate_shaded_lightcone(
    circuit: QuantumCircuit,
    bounds: Bounds,
    coupling_map: CouplingMap,
    *,
    reverse: bool = False,
) -> go.Figure:
    """Animates a shaded lightcone.

    This animation permits visualization of a shaded lightcone on top of a QPU's coupling map,
    making interpretation of shaded lightcones easier for circuits that act on qubits with
    connectivity higher than a 1D line.

    The qubits are placed by embedding the ``coupling_map`` into a square lattice, so that every
    coupling is drawn as a unit step. This works for heavy-hex and square connectivity alike and does
    not depend on the qubit numbering; see :func:`_get_coords`.

    Args:
        circuit: the circuit whose shaded lightcone to animate.
        bounds: the bounds to use for the shaded lightcone.
        coupling_map: the qubit connectivity map onto which to project the 1- and 2-weight bounds.
        reverse: whether to animate the circuit layers in reverse order.

    Returns:
        The ``plotly`` figure.

    Raises:
        ValueError: if the ``coupling_map`` cannot be laid out on a grid.
    """
    color_no_data = "lightgray"
    color_out_of_scale = "lightcoral"
    background_color = "white"
    highest_rate = 2.01
    edge_width = 4
    radius = 0.25
    height = 1000
    width = 1000
    colorscale = "viridis"

    frames = []

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 16},
            "prefix": "Box: ",
            "visible": True,
            "xanchor": "right",
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"t": 0},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [],
    }

    coordinates = _get_coords(coupling_map)

    dag = circuit_to_dag(circuit)
    idle_qubits = set(dag.idle_wires(ignore=["barrier"]))
    active_qubits_ = set(dag.qubits) - idle_qubits
    active_qubit_indices = set(find_indices(circuit, list(active_qubits_)))  # type: ignore[arg-type]

    # The coordinates come in the format ``(row, column)`` and place qubit ``0`` in the bottom row.
    # We turn them into ``(x, y)`` coordinates for convenience, multiplying the ``ys`` by ``-1`` so
    # that the map matches the map displayed on the ibmq website.
    ys = [-row for row, _ in coordinates]
    xs = [col for _, col in coordinates]

    # Add a line for each edge
    all_edges = set(tuple(sorted(edge)) for edge in list(coupling_map))
    data = []
    for q1, q2 in all_edges:
        x0 = xs[q1]
        x1 = xs[q2]
        y0 = ys[q1]
        y1 = ys[q2]

        edge = go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            hoverinfo="skip",
            mode="lines",
            line={
                "color": color_no_data,
                "width": edge_width,
            },
            showlegend=False,
            name="",
        )
        data.append(edge)

    colorbar = go.Scatter(
        x=[float("NaN")],
        y=[float("NaN")],
        marker=dict(
            size=0,
            cmax=2,
            cmin=0,
            color=[float("NaN")],
            colorbar={
                "title": "",
                "x": 0.95,
            },
            colorscale=colorscale,
        ),
        showlegend=False,
        name="",
    )
    data.append(colorbar)

    for _, qargs, box_id, _ in iter_circuit(circuit, reverse=reverse, log_process=False):
        if box_id is None:
            continue

        if box_id not in bounds:
            # No bounds were computed for this box, for example because it lies beyond the
            # ``max_num_boxes`` limit of the bounds computation. Skip it rather than truncating the
            # animation at this point.
            continue

        layer_error = bounds[box_id].apply_layout(qargs, circuit.num_qubits)

        layout = go.Layout(width=width, height=height)

        # A set of unique edges ``(i, j)``, with ``i < j``.
        edges = set(tuple(sorted(edge)) for edge in list(coupling_map))

        # The highest rate
        max_rate = 0

        # Initialize a dictionary of one-qubit errors
        paulis_1q, rates_1q_ = _restrict_num_bodies(layer_error, 1)
        rates_1q: dict[int, dict[str, float]] = {
            qubit: {} for qubit in coupling_map.physical_qubits
        }
        for pauli, rate in zip(paulis_1q, rates_1q_, strict=True):
            qubit_idx = np.where(pauli.x | pauli.z)[0][0]
            rates_1q[qubit_idx][str(pauli[qubit_idx])] = rate
            max_rate = max(max_rate, rate)

        # Initialize a dictionary of two-qubit errors
        paulis_2q, rates_2q_ = _restrict_num_bodies(layer_error, 2)
        rates_2q: dict[tuple[int, int], dict[str, float]] = {edge: {} for edge in edges}
        for pauli, rate in zip(paulis_2q, rates_2q_, strict=True):
            err_idxs = tuple(sorted([i for i, q in enumerate(pauli) if str(q) != "I"]))
            edge = (err_idxs[0], err_idxs[1])
            rates_2q[edge][str(pauli[[err_idxs[0], err_idxs[1]]])] = rate
            max_rate = max(max_rate, rate)

        highest_rate = highest_rate if highest_rate else max_rate

        # A discrete colorscale that contains 1000 hues.
        discrete_colorscale = sample_colorscale(colorscale, np.linspace(0, 1, 1000))

        # Plot the pie charts showing X, Y, and Z for each qubit
        shapes = []
        # hoverinfo_1q = []  # the info displayed when hovering over the pie charts
        for qubit, (x, y) in enumerate(zip(xs, ys, strict=True)):
            # hoverinfo = ""
            for pauli, angle in [("Z", -30), ("X", 90), ("Y", 210)]:
                rate = rates_1q.get(qubit, {}).get(pauli, 0)
                # print(qubit, pauli, rate)
                fillcolor = _get_rgb_color(
                    discrete_colorscale, rate / highest_rate, color_no_data, color_out_of_scale
                )
                line_color = "black"
                if fillcolor == color_no_data:
                    line_color = color_no_data
                if qubit in active_qubit_indices:
                    line_color = "black"
                shapes += [
                    {
                        "type": "path",
                        "path": _pie_slice(angle, angle + 120, x, y, radius),
                        "fillcolor": fillcolor,
                        "line_color": line_color,
                        "line_width": 1,
                    },
                ]

                # if rate:
                #     hoverinfo += f"<br>{pauli}: {rate}"
            # hoverinfo_1q += [hoverinfo or "No data"]

            # Add annotation with qubit label
            # fig.add_annotation(x=x + 0.3, y=y + 0.4, text=f"{qubit}", showarrow=False)

        for q1, q2 in edges:
            # NOTE: x > 0
            x0 = xs[q1]
            x1 = xs[q2]
            xmin = min(x0, x1) + 0.25
            xmax = max(x0, x1) - 0.25
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            # NOTE: y < 0
            y0 = ys[q1]
            y1 = ys[q2]
            ymin = min(y0, y1) + 0.25
            ymax = max(y0, y1) - 0.25
            if ymin > ymax:
                ymin, ymax = ymax, ymin

            locs = {
                "XX": {
                    "x0": xmin,
                    "x1": xmax - 1 / 3,
                    "y0": ymin + 1 / 3,
                    "y1": ymax,
                    "line_width": 2,
                },
                "XY": {
                    "x0": xmin + 1 / 6,
                    "x1": xmax - 1 / 6,
                    "y0": ymin + 1 / 3,
                    "y1": ymax,
                },
                "XZ": {
                    "x0": xmin + 1 / 3,
                    "x1": xmax,
                    "y0": ymin + 1 / 3,
                    "y1": ymax,
                },
                "YX": {
                    "x0": xmin,
                    "x1": xmax - 1 / 3,
                    "y0": ymin + 1 / 6,
                    "y1": ymax - 1 / 6,
                },
                "YY": {
                    "x0": xmin + 1 / 6,
                    "x1": xmax - 1 / 6,
                    "y0": ymin + 1 / 6,
                    "y1": ymax - 1 / 6,
                },
                "YZ": {
                    "x0": xmin + 1 / 3,
                    "x1": xmax,
                    "y0": ymin + 1 / 6,
                    "y1": ymax - 1 / 6,
                },
                "ZX": {
                    "x0": xmin,
                    "x1": xmax - 1 / 3,
                    "y0": ymin,
                    "y1": ymax - 1 / 3,
                },
                "ZY": {
                    "x0": xmin + 1 / 6,
                    "x1": xmax - 1 / 6,
                    "y0": ymin,
                    "y1": ymax - 1 / 3,
                },
                "ZZ": {
                    "x0": xmin + 1 / 3,
                    "x1": xmax,
                    "y0": ymin,
                    "y1": ymax - 1 / 3,
                },
            }

            if rates_2q[(q1, q2)].values():
                for pauli, rate in rates_2q[(q1, q2)].items():
                    if pauli not in locs:
                        continue
                    fillcolor = _get_rgb_color(
                        discrete_colorscale, rate / highest_rate, color_no_data, color_out_of_scale
                    )
                    shapes += [
                        {
                            "type": "rect",
                            "fillcolor": fillcolor,
                            "line_color": "black",
                            "line_width": 1,
                            **locs[pauli],
                        },
                    ]

                # hoverinfo_2q = ""
                # for pauli, rate in rates_2q[(q1, q2)].items():
                #     hoverinfo_2q += f"<br>{pauli}: {rate}"

            elif q1 in active_qubit_indices and q2 in active_qubit_indices:
                for pauli in locs:
                    shapes += [
                        {
                            "type": "rect",
                            "fillcolor": color_no_data,
                            "line_color": "black",
                            "line_width": 1,
                            **locs[pauli],
                        },
                    ]

        # Add a "legend" pie to show how pies work
        x_legend = max(xs) - 3.0
        y_legend = 1
        for pauli, angle in [("Z", -30), ("X", 90), ("Y", 210)]:
            shapes += [
                {
                    "type": "path",
                    "path": _pie_slice(angle, angle + 120, x_legend, y_legend, 0.5),
                    "fillcolor": color_no_data,
                    "line_color": "black",
                    "line_width": 1,
                    "label": {"text": f"<b>{pauli}</b>"},
                },
            ]

        # Add a "legend" square to show how edges work
        xmin = x_legend + 1.0
        xmax = x_legend + 3.0
        ymin = y_legend - 0.5
        ymax = y_legend + 0.5

        locs = {
            "XX": {
                "x0": xmin,
                "x1": xmax - 4 / 3,
                "y0": ymin + 2 / 3,
                "y1": ymax,
                "line_width": 2,
            },
            "XY": {
                "x0": xmin + 4 / 6,
                "x1": xmax - 4 / 6,
                "y0": ymin + 2 / 3,
                "y1": ymax,
            },
            "XZ": {
                "x0": xmin + 4 / 3,
                "x1": xmax,
                "y0": ymin + 2 / 3,
                "y1": ymax,
            },
            "YX": {
                "x0": xmin,
                "x1": xmax - 4 / 3,
                "y0": ymin + 2 / 6,
                "y1": ymax - 2 / 6,
            },
            "YY": {
                "x0": xmin + 4 / 6,
                "x1": xmax - 4 / 6,
                "y0": ymin + 2 / 6,
                "y1": ymax - 2 / 6,
            },
            "YZ": {
                "x0": xmin + 4 / 3,
                "x1": xmax,
                "y0": ymin + 2 / 6,
                "y1": ymax - 2 / 6,
            },
            "ZX": {
                "x0": xmin,
                "x1": xmax - 4 / 3,
                "y0": ymin,
                "y1": ymax - 2 / 3,
            },
            "ZY": {
                "x0": xmin + 4 / 6,
                "x1": xmax - 4 / 6,
                "y0": ymin,
                "y1": ymax - 2 / 3,
            },
            "ZZ": {
                "x0": xmin + 4 / 3,
                "x1": xmax,
                "y0": ymin,
                "y1": ymax - 2 / 3,
            },
        }
        for pauli in locs:
            shapes += [
                {
                    "type": "rect",
                    "fillcolor": color_no_data,
                    "line_color": "black",
                    "line_width": 1,
                    "label": {"text": f"<b>{pauli}</b>"},
                    **locs[pauli],
                },
            ]

        layout.shapes = shapes

        frame = go.Frame(data=[], layout=layout, name=box_id)
        frames.append(frame)

        slider_step = {
            "args": [
                [box_id],
                {
                    "frame": {"duration": 300, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 300},
                },
            ],
            "label": box_id,
            "method": "animate",
        }
        sliders_dict["steps"].append(slider_step)  # type: ignore[attr-defined]

    # Set x and y range
    fig = go.Figure(
        data=data,
        layout=frames[0].layout,
        frames=frames,
    )

    fig.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 500, "redraw": False},
                                "fromcurrent": True,
                                "transition": {"duration": 300, "easing": "quadratic-in-out"},
                            },
                        ],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 22},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ],
        sliders=[sliders_dict],
    )

    fig.update_xaxes(
        range=[min(xs) - 1, max(xs) + 2],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(
        range=[min(ys) - 1, max(ys) + 1],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
    )

    # Ensure that the circle is non-deformed
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(plot_bgcolor=background_color)

    return fig
