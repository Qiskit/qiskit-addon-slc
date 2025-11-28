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
from qiskit.quantum_info import PauliLindbladMap
from qiskit.transpiler import CouplingMap
from qiskit_ibm_runtime.visualization.utils import get_rgb_color, pie_slice

from ..bounds.commutator_bounds import Bounds
from ..utils import find_indices, iter_circuit


def _get_coords(coupling_map: CouplingMap) -> list[tuple[int, int]]:
    graph = coupling_map.graph
    assigned = set()
    rows: list[set[int]] = [set()]
    for node in graph.node_indices()[1:]:
        prev_node = node - 1
        if prev_node in graph.neighbors_undirected(node):
            rows[-1].update({prev_node, node})
            assigned.update({prev_node, node})
            continue
        rows.append(set())

    rows = [row for row in rows if len(row) > 0]

    unassigned = set(graph.node_indices()) - assigned

    cur_y = 0
    cur_row = 0
    next_row_start = min(rows[cur_row + 1])

    # fix 0 to x=0 and the resulting first row
    x_pos = {i: i for i in sorted(rows[0])}
    y_pos = {i: cur_y for i in rows[0]}

    for node in sorted(unassigned):
        if node > next_row_start:
            cur_row += 1
            cur_y += 2
            for node_ in rows[cur_row]:
                y_pos[node_] = cur_y
                if node_ in x_pos:
                    continue
                for other_node in rows[cur_row]:
                    if other_node in x_pos:
                        x_pos[node_] = x_pos[other_node] + (node_ - other_node)
                        break
            next_row_start = min(rows[cur_row + 1])

        for neighbor in sorted(graph.neighbors_undirected(node)):
            if neighbor in x_pos:
                x = x_pos[neighbor]

        for neighbor in graph.neighbors_undirected(node):
            if neighbor not in x_pos:
                x_pos[neighbor] = x
        x_pos[node] = x
        y_pos[node] = cur_y + 1

    # final row
    cur_row += 1
    cur_y += 2
    for node in rows[cur_row]:
        y_pos[node] = cur_y
        if node in x_pos:
            continue
        for other_node in rows[cur_row]:
            if other_node in x_pos:
                x_pos[node] = x_pos[other_node] + (node - other_node)
                break

    coords = [(y_pos[qb], x_pos[qb]) for qb in sorted(x_pos)]
    return coords


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
    reverse: bool = False,
) -> go.Figure:
    """Animates a shaded lightcone.

    Args:
        circuit: the circuit whose shaded lightcone to draw.
        bounds: the bounds to use for the shaded lightcone.
        coupling_map: the qubit connectivity map onto which to project the 1- and 2-weight bounds.
        reverse: whether to animate the circuit layers in reverse order.

    Returns:
        The ``plotly`` figure.
    """
    color_no_data = "lightgray"
    color_out_of_scale = "lightred"
    background_color = "white"
    highest_rate = 2.01
    edge_width = 4
    radius = 0.25
    height = 1000
    width = 1000
    colorscale = "viridis"

    # fig = go.Figure(layout=go.Layout(width=width, height=height))
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
            # hovertemplate="No data",
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
            # HACK: remove me!
            break

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
                fillcolor = get_rgb_color(
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
                        "path": pie_slice(angle, angle + 120, x, y, radius),
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
                    fillcolor = get_rgb_color(
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
                    "path": pie_slice(angle, angle + 120, x_legend, y_legend, 0.5),
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
