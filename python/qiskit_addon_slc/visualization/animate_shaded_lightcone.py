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

import numpy as np
from plotly import graph_objects as go
from plotly.colors import sample_colorscale
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import PauliLindbladMap, PauliList
from qiskit.transpiler import CouplingMap

from ..bounds.commutator_bounds import Bounds
from ..utils import find_indices, iter_circuit
from ._lattice_layout import _get_coords

# The number of hues in the discrete colorscales built by this module.
_NUM_HUES = 1000

# Added to every rate that is present in the noise model, so that a rate of exactly zero still maps
# to a color: the drawing reserves a rate of ``0`` for "this Pauli is not in the noise model", which
# it greys out. Far smaller than any rate that is physically meaningful.
_RATE_EPSILON = 1e-17

# The upper end of the color scale, as a total Pauli-error rate. A rate of 2 is the theoretical
# maximum for a depolarizing channel, and the small excess keeps a bound of exactly 2 inside the
# scale rather than flagging it as out of scale.
_MAX_RATE = 2.01

# The radius of the pie drawn on each qubit to show its 1-qubit bounds, in data coordinates. Qubits
# sit one unit apart, so this keeps a clear gap between neighboring pies.
_PIE_RADIUS = 0.25

# The size of the legend key, in data coordinates. The key is a fixed-size drawing -- a pie of radius
# 0.5 followed by a 2x1 block of edge swatches -- so :func:`animate_shaded_lightcone` reserves a band
# of this size above the lattice and extends the axis ranges to keep all of it on canvas.
_LEGEND_WIDTH = 4.0
_LEGEND_HEIGHT = 2.0


# The fixed styling of the animated figure: the colors it draws with, the size of the canvas and the
# colorscale the bounds are shaded by. Collected here rather than spread through the drawing code so
# that the whole look of the figure can be seen, and changed, in one place. These are separate
# constants rather than one dict so that each keeps its own type.
_COLOR_NO_DATA = "lightgray"
_COLOR_OUT_OF_SCALE = "lightcoral"
_BACKGROUND_COLOR = "white"
_COLORSCALE = "viridis"
_EDGE_WIDTH = 4
_FIGURE_HEIGHT = 1000
_FIGURE_WIDTH = 1000

# The slider that steps through the boxes. ``steps`` is added per figure, since it names the boxes.
_SLIDER = {
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
}

# The Play and Pause buttons that drive the animation.
_PLAY_PAUSE_BUTTONS = {
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

# The angle at which each Pauli's pie slice starts, in degrees, and the width of a slice. Three
# slices of 120 degrees make up the pie drawn on a qubit.
_PIE_SLICE_ANGLES = {"Z": -30, "X": 90, "Y": 210}
_PIE_SLICE_WIDTH = 120


def _pauli_pair_cells(
    xmin: float, xmax: float, ymin: float, ymax: float, *, x_scale: float, y_scale: float
) -> dict[str, dict[str, float]]:
    """Lays the nine 2-qubit Pauli pairs out as a 3x3 grid of swatches.

    The two Paulis of a pair are read as coordinates into the grid: the first picks the row and the
    second the column, both in ``XYZ`` order, so ``X`` is the top row and the left column while ``Z``
    is the bottom row and the right one. Neighboring swatches overlap by design, so that the grid reads as one block
    rather than nine separate rectangles.

    Args:
        xmin: the left edge of the block.
        xmax: the right edge of the block.
        ymin: the bottom edge of the block.
        ymax: the top edge of the block.
        x_scale: how far, in thirds of this value, the columns are inset from the block's edges.
        y_scale: the same for the rows.

    Returns:
        A mapping of 2-qubit Pauli label to the ``x0``/``x1``/``y0``/``y1`` of its swatch.
    """
    x_third, x_sixth = x_scale / 3, x_scale / 6
    y_third, y_sixth = y_scale / 3, y_scale / 6

    # The inset of a swatch from each edge, by its index along that axis.
    from_low = (0.0, x_sixth, x_third)
    from_high = (x_third, x_sixth, 0.0)
    down_from_low = (y_third, y_sixth, 0.0)
    down_from_high = (0.0, y_sixth, y_third)

    return {
        f"{row_pauli}{col_pauli}": {
            "x0": xmin + from_low[col],
            "x1": xmax - from_high[col],
            "y0": ymin + down_from_low[row],
            "y1": ymax - down_from_high[row],
        }
        for row, row_pauli in enumerate("XYZ")
        for col, col_pauli in enumerate("XYZ")
    }


def _get_rgb_color(
    discrete_colorscale: list[str], val: float, default: str, color_out_of_scale: str
) -> str:
    """Maps a float to an RGB color based on a discrete colorscale.

    Args:
        discrete_colorscale: a discrete colorscale of :data:`_NUM_HUES` hues.
        val: a value in ``[0, 1]`` to map to a color.
        default: the color returned when ``val`` is ``0``.
        color_out_of_scale: the color returned when ``val`` falls outside ``[0, 1]``.

    Returns:
        The color as a string.

    Raises:
        ValueError: if the colorscale does not contain :data:`_NUM_HUES` hues.
    """
    if len(discrete_colorscale) != _NUM_HUES:
        raise ValueError(
            f"Expected a colorscale of {_NUM_HUES} hues, but got {len(discrete_colorscale)}."
        )

    # Error rates are non-negative, so a negative value means the caller scaled something wrongly.
    # It is as much out of scale as a value above 1, and treating it as such keeps this total rather
    # than indexing the colorscale from the wrong end.
    if val < 0 or val > 1:
        return color_out_of_scale
    if val == 1:
        return discrete_colorscale[-1]
    if val == 0:
        return default
    # ``val`` is strictly between 0 and 1 here, but rounding can still carry it up to 1.0 -- anything
    # from 0.9995 up rounds to 1.0 and would index one past the end -- so clamp to the last hue.
    return discrete_colorscale[min(int(np.round(val, 3) * _NUM_HUES), _NUM_HUES - 1)]


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


def _restrict_num_bodies(plm: PauliLindbladMap, num_qubits: int) -> tuple[PauliList, np.ndarray]:
    """Selects the terms of a Pauli-Lindblad map that act on exactly ``num_qubits`` qubits.

    The returned rates are nudged up by :data:`_RATE_EPSILON` so that a term which is present in the
    noise model but carries a rate of exactly zero still renders as a color. The drawing uses a rate
    of ``0`` to mean "this Pauli is not in the noise model at all" and greys it out, so without the
    nudge those two cases would be indistinguishable.

    Args:
        plm: the Pauli-Lindblad map whose terms to restrict.
        num_qubits: the exact number of qubits a term must act on to be kept.

    Returns:
        A tuple of the selected Pauli terms and their rates.

    Raises:
        ValueError: if ``num_qubits`` is negative.
    """
    if num_qubits < 0:
        raise ValueError("``num_qubits`` must be ``0`` or larger.")
    # ``to_pauli_list`` preserves the term order of ``plm.rates``, and the same boolean mask is
    # applied to both, so the returned Paulis and rates stay aligned and the caller may zip them.
    paulis = plm.get_qubit_sparse_pauli_list_copy().to_pauli_list()
    mask = np.sum(paulis.x | paulis.z, axis=1) == num_qubits
    return paulis[mask], plm.rates[mask] + _RATE_EPSILON


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
    not depend on the qubit numbering.

    Args:
        circuit: the circuit whose shaded lightcone to animate.
        bounds: the bounds to use for the shaded lightcone.
        coupling_map: the qubit connectivity map onto which to project the 1- and 2-weight bounds.
        reverse: whether to animate the circuit layers in reverse order.

    Returns:
        The ``plotly`` figure.

    Raises:
        ValueError: if the ``coupling_map`` cannot be laid out on a grid.
        ValueError: if the ``bounds`` contain a term acting on more than 2 qubits, which has no
            place in the drawing.
        ValueError: if the ``circuit`` does not act on exactly the qubits of the ``coupling_map``.
        ValueError: if none of the circuit's boxes have bounds.
        ValueError: if the ``bounds`` contain a 2-qubit term on an uncoupled pair of qubits.
    """
    if circuit.num_qubits != coupling_map.size():
        raise ValueError(
            f"This circuit acts on {circuit.num_qubits} qubits but the coupling map has "
            f"{coupling_map.size()}. Transpile the circuit to the coupling map first, so that its "
            "qubits are the physical qubits being drawn."
        )

    color_no_data = _COLOR_NO_DATA
    color_out_of_scale = _COLOR_OUT_OF_SCALE
    background_color = _BACKGROUND_COLOR
    highest_rate = _MAX_RATE
    edge_width = _EDGE_WIDTH
    height = _FIGURE_HEIGHT
    width = _FIGURE_WIDTH
    colorscale = _COLORSCALE

    frames = []

    # ``steps`` is filled in per box below, so this starts from a copy of the template.
    sliders_dict: dict = {**_SLIDER, "steps": []}

    coordinates = _get_coords(coupling_map)

    dag = circuit_to_dag(circuit)
    idle_qubits = set(dag.idle_wires(ignore=["barrier"]))
    active_qubits_ = set(dag.qubits) - idle_qubits
    active_qubit_indices = set(find_indices(circuit, list(active_qubits_)))  # type: ignore[arg-type]

    xs = [x for x, _ in coordinates]
    ys = [y for _, y in coordinates]

    # Add a line for each edge
    all_edges = set(tuple(sorted(edge)) for edge in list(coupling_map))
    data = []
    for q1, q2 in all_edges:
        x0, x1, y0, y1 = xs[q1], xs[q2], ys[q1], ys[q2]

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
            # The same ceiling the shading is normalized by, so the hues the colorbar explains match
            # the ones actually drawn.
            cmax=_MAX_RATE,
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

    # A discrete colorscale of :data:`_NUM_HUES` hues. Sampling it is not cheap and it does not
    # depend on the box, so it is built once rather than per frame.
    discrete_colorscale = sample_colorscale(colorscale, np.linspace(0, 1, _NUM_HUES))

    for _, qargs, box_id, _ in iter_circuit(circuit, reverse=reverse, log_process=False):
        if box_id is None:
            continue

        if box_id not in bounds:
            # No bounds were computed for this box, for example because it lies beyond the
            # ``max_num_boxes`` limit of the bounds computation. Skip it rather than truncating the
            # animation at this point.
            continue

        layer_error = bounds[box_id].apply_layout(qargs, circuit.num_qubits)

        # Only weight-1 and weight-2 terms have somewhere to be drawn: the former shade a qubit's
        # pie, the latter an edge's swatch block. A higher-weight term has no such home, and silently
        # dropping it would quietly under-report the error, so refuse it instead.
        all_paulis = layer_error.get_qubit_sparse_pauli_list_copy().to_pauli_list()
        weights = np.sum(all_paulis.x | all_paulis.z, axis=1)
        if np.any(weights > 2):
            raise ValueError(
                f"The bounds for box '{box_id}' contain a term acting on {int(max(weights))} "
                "qubits, but this animation can only draw 1- and 2-qubit error terms: a qubit's "
                "pie and a coupling's swatch block, respectively. Filter the bounds down to "
                "weight-1 and weight-2 terms before animating them."
            )

        layout = go.Layout(width=width, height=height)

        # Initialize a dictionary of one-qubit errors
        paulis_1q, rates_1q_ = _restrict_num_bodies(layer_error, 1)
        rates_1q: dict[int, dict[str, float]] = {
            qubit: {} for qubit in coupling_map.physical_qubits
        }
        for pauli, rate in zip(paulis_1q, rates_1q_, strict=True):
            qubit_idx = np.where(pauli.x | pauli.z)[0][0]
            rates_1q[qubit_idx][str(pauli[qubit_idx])] = rate

        # Initialize a dictionary of two-qubit errors
        paulis_2q, rates_2q_ = _restrict_num_bodies(layer_error, 2)
        rates_2q: dict[tuple[int, int], dict[str, float]] = {edge: {} for edge in all_edges}
        for pauli, rate in zip(paulis_2q, rates_2q_, strict=True):
            err_idxs = tuple(sorted([i for i, q in enumerate(pauli) if str(q) != "I"]))
            edge = (err_idxs[0], err_idxs[1])
            if edge not in rates_2q:
                raise ValueError(
                    f"The bounds for box '{box_id}' contain a 2-qubit term on qubits {edge}, which "
                    "are not coupled in this coupling map, so there is no edge to draw it on. The "
                    "bounds must be computed for the same coupling map that is being drawn."
                )
            rates_2q[edge][str(pauli[[err_idxs[0], err_idxs[1]]])] = rate

        # Plot the pie charts showing X, Y, and Z for each qubit
        shapes = []
        for qubit, (x, y) in enumerate(zip(xs, ys, strict=True)):
            for pauli, angle in _PIE_SLICE_ANGLES.items():
                rate = rates_1q.get(qubit, {}).get(pauli, 0)
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
                        "path": _pie_slice(angle, angle + _PIE_SLICE_WIDTH, x, y, _PIE_RADIUS),
                        "fillcolor": fillcolor,
                        "line_color": line_color,
                        "line_width": 1,
                    },
                ]

        for q1, q2 in all_edges:
            x0, x1, y0, y1 = xs[q1], xs[q2], ys[q1], ys[q2]

            # The swatch block spans the gap between the two qubits' pies, so each end is inset by
            # the pie radius. Coupled qubits differ along one axis only, so along the other axis the
            # inset inverts the interval and the bounds have to be swapped back.
            xmin, xmax = min(x0, x1) + _PIE_RADIUS, max(x0, x1) - _PIE_RADIUS
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            ymin, ymax = min(y0, y1) + _PIE_RADIUS, max(y0, y1) - _PIE_RADIUS
            if ymin > ymax:
                ymin, ymax = ymax, ymin

            locs = _pauli_pair_cells(xmin, xmax, ymin, ymax, x_scale=1.0, y_scale=1.0)
            # The XX swatch carries a heavier outline, as the anchor of the 3x3 block.
            locs["XX"]["line_width"] = 2

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

        # Add a "legend" pie to show how pies work. The legend is a fixed-size key drawn in data
        # coordinates, so it is placed in the band reserved for it above the lattice (see
        # ``_LEGEND_WIDTH`` and ``_LEGEND_HEIGHT``) and the axis ranges below are widened to match.
        # Right-aligning it to the widest of the lattice and the legend keeps it on canvas even when
        # the map is narrower than the key itself.
        x_legend = max(max(xs), min(xs) + _LEGEND_WIDTH) - 3.0
        y_legend = max(ys) + _LEGEND_HEIGHT / 2
        for pauli, angle in _PIE_SLICE_ANGLES.items():
            shapes += [
                {
                    "type": "path",
                    "path": _pie_slice(angle, angle + _PIE_SLICE_WIDTH, x_legend, y_legend, 0.5),
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

        locs = _pauli_pair_cells(xmin, xmax, ymin, ymax, x_scale=4.0, y_scale=2.0)
        locs["XX"]["line_width"] = 2
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

    if not frames:
        raise ValueError(
            "None of the boxes in this circuit have bounds, so there is nothing to animate. Check "
            "that `bounds` is keyed by the circuit's box ids."
        )

    # Set x and y range
    fig = go.Figure(
        data=data,
        layout=frames[0].layout,
        frames=frames,
    )

    fig.update_layout(
        updatemenus=[_PLAY_PAUSE_BUTTONS],
        sliders=[sliders_dict],
    )

    # Leave room for the legend: it occupies a band of height ``_LEGEND_HEIGHT`` above the lattice
    # and, on a map narrower than the legend, extends further right than any qubit. Without this the
    # top of the legend fell outside the y range and was clipped on every device.
    legend_right = max(max(xs), min(xs) + _LEGEND_WIDTH)
    fig.update_xaxes(
        range=[min(xs) - 1, legend_right + 2],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(
        range=[min(ys) - 1, max(ys) + _LEGEND_HEIGHT + 1],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
    )

    # Ensure that the circle is non-deformed
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(plot_bgcolor=background_color)

    return fig
