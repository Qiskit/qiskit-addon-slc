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

"""Placing the qubits of a coupling map on a square lattice.

The coupling maps of IBM's processors are subgraphs of a square lattice: every coupling connects two
qubits that are neighbors on a grid. This module recovers such a grid placement from the graph alone,
without reading the qubit numbering, so that a visualization can draw every coupling as a unit step.

:func:`_get_coords` is the entry point; everything else here is a strategy it dispatches to or a
helper those strategies share.
"""

from collections import deque

from qiskit.transpiler import CouplingMap

# The four unit steps of a square lattice, used to lay the qubits out on a grid.
_LATTICE_STEPS = ((0, 1), (1, 0), (0, -1), (-1, 0))

# The default search budget of :func:`_get_coords`, per candidate layout. Every IBM device tried
# needs fewer than ~10_000 steps, so this leaves an order of magnitude of margin while still bounding
# pathological inputs.
_MAX_SEARCH_STEPS = 100_000

# The shortest run of consecutively-numbered qubits that :func:`_lattice_rows` accepts as a row.
#
# A run is only evidence of row-by-row numbering if it is long enough that it cannot have arisen by
# accident: on a heavy-hex device the rungs are single qubits, so every run of 2 or more already
# separates a row from a rung, but a short run says nothing about which *direction* the numbering
# travels and column-wise or irregularly numbered devices produce plenty of them. The Falcon-27 maps
# are the case in point, with runs of up to 4 and no rows at all.
#
# The genuine rows of IBM's heavy-hex devices hold 14 or more qubits (14-15 on Eagle, 15-16 on
# Heron), so anything from 5 up to 14 admits every real row while rejecting the accidental runs. The
# midpoint is as defensible as either end and leaves room on both sides.
_MIN_ROW_LENGTH = 5

# How far a heavy-hex row may be indented relative to the one above it. The rows of a device are not
# all the same length and the shorter ones sit inset by a qubit or two.
_MAX_ROW_INDENT = 2

# How many starting qubits :func:`_search_for_layout` tries before keeping the best layout it found.
#
# The seeds are tried in increasing order of degree, and the boundary qubits that make good seeds
# come first, so the later candidates rarely win. Eight is enough to settle every device tried:
# raising it to 12 only changes Falcon-27, and for the worse (a 9x7 placement of a device that is
# recognisably 6x9), while costing ~3x the search time on heavy-hex.
_NUM_LAYOUT_CANDIDATES = 8


def _neighbors_of(coupling_map: CouplingMap) -> dict[int, set[int]]:
    """Returns the undirected neighbors of every qubit of a coupling map.

    Every strategy in this module needs random access to a qubit's neighbors, which the coupling
    map's own graph does not offer cheaply, so each one starts by building this.

    Args:
        coupling_map: the coupling map whose adjacency to collect.

    Returns:
        A mapping of qubit to the set of qubits it is coupled to.
    """
    graph = coupling_map.graph.to_undirected()
    return {qubit: set(graph.neighbors(qubit)) for qubit in range(coupling_map.size())}


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
    neighbors = _neighbors_of(coupling_map)

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
    num_qubits = coupling_map.size()
    neighbors = _neighbors_of(coupling_map)

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


class _LatticeSearch:
    """The mutable state of a backtracking search for a square-lattice embedding.

    The search assigns each qubit an integer ``(row, column)`` cell. It keeps that assignment in two
    directions at once -- :attr:`positions` from qubit to cell and :attr:`occupied` from cell to
    qubit -- because every placement has to be checked against both the qubit's couplings and the
    cells already taken, and it counts the placements it has tried so that a hopeless search gives up
    rather than running forever.

    The two mappings are only ever changed through :meth:`occupy` and :meth:`vacate`, which keep them
    in step. :meth:`fits` reads one for its no-accidental-neighbors check and the other for its
    unit-step check, so a half-applied placement does not fail where it happens -- it leaves a stale
    entry that over-constrains later placements, and the search then rejects an embedding that exists
    and blames the coupling map for it.

    Args:
        neighbors: the adjacency of the coupling map being embedded.
    """

    def __init__(self, neighbors: dict[int, set[int]]) -> None:
        self.neighbors = neighbors
        self.positions: dict[int, tuple[int, int]] = {}
        self.occupied: dict[tuple[int, int], int] = {}
        self.steps = 0

    def fits(self, qubit: int, cell: tuple[int, int]) -> bool:
        """Checks whether placing ``qubit`` at ``cell`` keeps the embedding consistent.

        Consistency runs in both directions, so this checks two things. First, that the cell's
        already-occupied lattice neighbors are all qubits this one is coupled to, since drawing two
        uncoupled qubits side by side would imply a coupling that does not exist. Second, the
        converse: that every one of this qubit's couplings whose other end is already placed lands
        exactly one step away, since a coupling has to be drawn as a unit step.

        Args:
            qubit: the qubit to place.
            cell: the ``(row, column)`` cell to place it in.

        Returns:
            Whether the placement is consistent.
        """
        if cell in self.occupied:
            return False

        # No accidental neighbors: everything adjacent to this cell must be a coupled qubit.
        for row_step, col_step in _LATTICE_STEPS:
            adjacent = (cell[0] + row_step, cell[1] + col_step)
            if adjacent in self.occupied and self.occupied[adjacent] not in self.neighbors[qubit]:
                return False

        # No stretched couplings: every placed neighbor must be exactly one step away.
        for neighbor in self.neighbors[qubit]:
            if neighbor in self.positions:
                row, col = self.positions[neighbor]
                if abs(row - cell[0]) + abs(col - cell[1]) != 1:
                    return False

        return True

    def candidate_cells(self, qubit: int) -> list[tuple[int, int]]:
        """Returns the cells a qubit could occupy, i.e. those next to an already-placed neighbor.

        Args:
            qubit: the qubit to find candidate cells for.

        Returns:
            The candidate cells, in a deterministic order.
        """
        cells = []
        for neighbor in sorted(self.neighbors[qubit]):
            if neighbor in self.positions:
                row, col = self.positions[neighbor]
                cells += [(row + dr, col + dc) for dr, dc in _LATTICE_STEPS]
        # ``dict.fromkeys`` de-duplicates while keeping the order deterministic.
        return list(dict.fromkeys(cells))

    def breadth_first_order(self, seed: int) -> list[int]:
        """Orders the qubits so that each is reached from an already-placed neighbor.

        Args:
            seed: the qubit to start from.

        Returns:
            The qubits reachable from ``seed``, in breadth-first order. Shorter than the coupling map
            if it is not connected.
        """
        visit_order = [seed]
        seen = {seed}
        queue = deque([seed])
        while queue:
            qubit = queue.popleft()
            for neighbor in sorted(self.neighbors[qubit]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    visit_order.append(neighbor)
                    queue.append(neighbor)
        return visit_order

    def place(self, remaining: list[int], index: int) -> bool:
        """Places ``remaining[index:]``, backtracking on dead ends.

        Args:
            remaining: the qubits to place, in the order to try them.
            index: how far into ``remaining`` this call starts.

        Returns:
            Whether every remaining qubit could be placed.

        Raises:
            ValueError: if the search budget is exhausted first.
        """
        self.steps += 1
        if self.steps > _MAX_SEARCH_STEPS:
            raise ValueError(
                f"Could not embed this coupling map within {_MAX_SEARCH_STEPS} search steps, so it "
                "is most likely not a square lattice. Pass explicit coordinates instead."
            )
        if index == len(remaining):
            return True

        qubit = remaining[index]
        cells = [(0, 0)] if not self.positions else self.candidate_cells(qubit)
        for cell in cells:
            if self.fits(qubit, cell):
                self.occupy(qubit, cell)
                if self.place(remaining, index + 1):
                    return True
                self.vacate(qubit, cell)
        return False

    def place_rows(self, rows: list[list[int]], index: int, rungs: list[int]) -> bool:
        """Lays each row out straight, trying a few indents, then places the rungs between them.

        Args:
            rows: the rows to pre-place, as returned by :func:`_lattice_rows`.
            index: how far into ``rows`` this call starts.
            rungs: the qubits that are not part of any row, to place once the rows are down.

        Returns:
            Whether every row and rung could be placed.

        Raises:
            ValueError: if the search budget is exhausted while placing the rungs.
        """
        if index == len(rows):
            return self.place(rungs, 0)

        # Rows of a heavy-hex device are not all the same length; the shorter ones are indented.
        # Trying a couple of offsets per row is enough to line them all up.
        for indent in range(_MAX_ROW_INDENT + 1):
            cells = [(2 * index, indent + column) for column in range(len(rows[index]))]
            if any(cell in self.occupied for cell in cells):
                continue
            for qubit, cell in zip(rows[index], cells, strict=True):
                self.occupy(qubit, cell)
            if self.place_rows(rows, index + 1, rungs):
                return True
            for qubit, cell in zip(rows[index], cells, strict=True):
                self.vacate(qubit, cell)
        return False

    def occupy(self, qubit: int, cell: tuple[int, int]) -> None:
        """Places ``qubit`` in ``cell``, keeping both views of the assignment in step.

        Args:
            qubit: the qubit being placed.
            cell: the cell it occupies.
        """
        self.positions[qubit] = cell
        self.occupied[cell] = qubit

    def vacate(self, qubit: int, cell: tuple[int, int]) -> None:
        """Undoes an :meth:`occupy`, when the search backtracks past it.

        Args:
            qubit: the qubit being unplaced.
            cell: the cell it was occupying.
        """
        del self.positions[qubit]
        del self.occupied[cell]


def _embed_in_lattice(
    coupling_map: CouplingMap,
    seed: int,
    *,
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
        seed_rows: rows to pre-place as straight lines, as returned by :func:`_lattice_rows`.

    Returns:
        A mapping of qubit to its ``(row, column)`` cell.

    Raises:
        ValueError: if the coupling map is not connected.
        ValueError: if the coupling map does not embed into a square lattice.
        ValueError: if ``seed_rows`` were given but could not be laid out.
        ValueError: if the search budget is exhausted before an embedding is found.
    """
    num_qubits = coupling_map.size()
    search = _LatticeSearch(_neighbors_of(coupling_map))

    if seed_rows:
        seeded = {qubit for row in seed_rows for qubit in row}
        rungs = [qubit for qubit in range(num_qubits) if qubit not in seeded]
        if not search.place_rows(seed_rows, 0, rungs):
            raise ValueError("Could not lay out this coupling map from its rows.")
        return search.positions

    visit_order = search.breadth_first_order(seed)
    if len(visit_order) != num_qubits:
        raise ValueError(
            "Cannot lay out a coupling map that is not connected: "
            f"only {len(visit_order)} of {num_qubits} qubits are reachable from qubit {seed}."
        )

    if not search.place(visit_order, 0):
        raise ValueError(
            "This coupling map does not embed into a square lattice, so its qubits cannot be laid "
            "out on a grid. Pass explicit coordinates instead."
        )

    return search.positions


def _normalize_layout(
    positions: dict[int, tuple[int, int]], num_qubits: int
) -> list[tuple[int, int]]:
    """Compacts and orients a lattice embedding, as ``(x, y)`` plotting coordinates.

    An embedding is only defined up to translation, rotation and reflection, so the raw output is
    normalized: unused rows and columns are dropped, the layout is transposed so that
    consecutively-numbered qubits run along the rows, and it is reflected so that qubit ``0`` falls
    in the top-left quadrant. All of this is derived from the embedding itself, so it adapts to any
    device size.

    Settling the orientation is the last of those steps: the cells are returned as ``(x, y)``, with
    ``x`` the column and ``y`` the negated row, so that row ``0`` sits at the top of a figure. That
    matches the orientation the ibmq website draws its devices in, and fixing it here means a caller
    never has to decide which way is up.

    Args:
        positions: the raw embedding.
        num_qubits: the number of qubits, i.e. the length of the returned list.

    Returns:
        The ``(x, y)`` coordinate of each qubit, indexed by qubit number.
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

    # Negating the row turns a lattice cell, whose rows count downwards, into a plotting coordinate
    # whose ``y`` grows upwards.
    return [(positions[qubit][1], -positions[qubit][0]) for qubit in range(num_qubits)]


def _search_for_layout(coupling_map: CouplingMap) -> dict[int, tuple[int, int]]:
    """Searches for a lattice embedding, trying several starting qubits.

    Args:
        coupling_map: the coupling map to lay out.

    Returns:
        A mapping of qubit to its ``(row, column)`` cell.

    Raises:
        ValueError: if no starting qubit yields an embedding.
    """
    num_qubits = coupling_map.size()
    neighbors = _neighbors_of(coupling_map)

    # Seed from the lowest-degree qubits: those sit on the boundary of the lattice, where there are
    # fewest ways for the search to strand itself. Seeding from a high-degree qubit in the interior
    # costs orders of magnitude more steps.
    seeds = sorted(range(num_qubits), key=lambda qubit: (len(neighbors[qubit]), qubit))

    best: dict[int, tuple[int, int]] | None = None
    best_score: tuple[int, int] | None = None
    failure: ValueError | None = None
    for seed in seeds[:_NUM_LAYOUT_CANDIDATES]:
        try:
            positions = _embed_in_lattice(coupling_map, seed)
        except ValueError as exc:
            failure = exc
            continue
        candidate = _normalize_layout(positions, num_qubits)
        # Prefer the layout that keeps the device's numbering lines straight, falling back on the
        # tightest bounding box. Ranking by area alone yields valid but visibly tangled layouts.
        # ``candidate`` holds ``(x, y)`` coordinates, so a numbering line is a run of equal ``y``.
        longest_run = run = 1
        for qubit in range(1, num_qubits):
            run = run + 1 if candidate[qubit][1] == candidate[qubit - 1][1] else 1
            longest_run = max(longest_run, run)
        xs = [x for x, _ in candidate]
        ys = [y for _, y in candidate]
        area = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
        score = (-longest_run, area)
        if best_score is None or score < best_score:
            best = positions
            best_score = score

    if best is None:
        raise failure if failure is not None else ValueError("Could not lay out this coupling map.")
    return best


def _get_coords(coupling_map: CouplingMap) -> list[tuple[int, int]]:
    """Computes ``(row, column)`` coordinates for each qubit of a coupling map.

    The coupling maps of IBM's processors are subgraphs of a square lattice: every coupling connects
    two qubits that are neighbors on a grid. This function recovers such a grid placement, choosing
    its strategy from the map's maximum connectivity:

    * **degree 1 or 2** -- a path, which lays out along a single row with no choices to make, or a
      ring, which folds into a rectangle. Rings of 4 or of 8 or more even qubits can be drawn; odd
      rings cannot, since a grid is bipartite, and neither can a 6-ring, whose only placement would
      imply a coupling that does not exist;
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

    The result is returned as ``(x, y)`` plotting coordinates rather than as lattice cells: ``x`` is
    the column, and ``y`` is the *negated* row, so that row ``0`` sits at the top of the figure. That
    matches the orientation the ibmq website draws its devices in, and fixing the convention here
    means every caller agrees on which way is up.

    Args:
        coupling_map: the coupling map whose qubits to place.

    Returns:
        The ``(x, y)`` coordinate of each qubit, indexed by qubit number.

    Raises:
        ValueError: if the coupling map is not connected.
        ValueError: if the coupling map is too densely connected to lie on a grid.
        ValueError: if the coupling map is a ring whose length does not fold into a rectangle.
        ValueError: if the layout needs more than :data:`_MAX_SEARCH_STEPS` steps to find.
    """
    num_qubits = coupling_map.size()
    if num_qubits == 0:
        return []

    # The degrees decide which strategy to use, so they are collected once up front.
    degrees = [len(adjacent) for adjacent in _neighbors_of(coupling_map).values()]
    max_degree = max(degrees)

    positions: dict[int, tuple[int, int]] | None = None
    if max_degree <= 2:
        # A path or a ring. A path lays out along a single row; a ring has to fold back on itself,
        # which only works for some lengths, so it is worth reporting the impossible ones precisely.
        if min(degrees) == 2 and (num_qubits % 2 or num_qubits == 6):
            # Every qubit has two couplings, so this is a single cycle rather than a path. A grid is
            # bipartite and so contains no odd cycle; and the only grid placement of a 6-cycle is the
            # perimeter of a 2x3 block, whose two middle qubits end up adjacent without being
            # coupled, which would draw a coupling that does not exist. Every other even ring folds
            # into a rectangle: 4 into a 2x2 block, and 2n >= 8 into a 3-row "U".
            raise ValueError(
                f"Cannot lay out a ring of {num_qubits} qubits on a grid: a grid contains no odd "
                "cycle, and a 6-ring can only be drawn by implying a coupling that does not exist. "
                "Rings of 4 or of 8 or more even qubits are supported; otherwise pass explicit "
                "coordinates."
            )
        positions = _embed_in_lattice(coupling_map, 0)
    elif max_degree == 3:
        rows = _lattice_rows(coupling_map, _MIN_ROW_LENGTH)
        if rows:
            try:
                positions = _embed_in_lattice(coupling_map, 0, seed_rows=rows)
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
        positions = _search_for_layout(coupling_map)

    return _normalize_layout(positions, num_qubits)
