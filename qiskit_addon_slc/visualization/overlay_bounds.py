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

"""The bounds circuit overlaying method."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import rustworkx as rx
from qiskit import QuantumCircuit
from qiskit.circuit import Gate

from ..utils import iter_circuit


def overlay_bounds_onto_circuit(
    pauli_bounds: dict[str, dict[tuple[int, ...], float]],
    circuit: QuantumCircuit,
    *,
    include_empty_boxes: bool = True,
) -> QuantumCircuit:
    """Overlays the bounds onto the circuit.

    This produces a new "QuantumCircuit" with "fake" gates reproducing those :class:`.BoxOp`
    instructions from the original ``circuit`` that had an :class:`.InjectNoise` annotation.
    Those boxes are replaced by visual representations of the computed bounds for Pauli errors of
    their respective support (see also :func:`accumulate_filtered_bounds`).

    Args:
        pauli_bounds: the accumulated and filtered bounds.
        circuit: the circuit on which to overlay the bounds.
        include_empty_boxes: whether to include empty boxes or not.

    Returns:
        A "fake" quantum circuit with "Gate" instructions displaying the computed bound values.
        The returned quantum circuit's :attr:`.QuantumCircuit.metadata` contains the maximum bound
        value under `"max_bound"`.
    """
    bounds_circuit = circuit.copy_empty_like()
    max_bound_value = 2.0

    invisible_gate = Gate("hide", 1, [], label="")

    for _, qargs, box_id, _ in iter_circuit(circuit, reverse=False, log_process=False):
        if box_id is None:
            continue

        box_body = QuantumCircuit(len(qargs))

        pauli_bounds_this_box = pauli_bounds[box_id]

        if len(pauli_bounds_this_box) == 0 and include_empty_boxes:
            box_body.append(invisible_gate, [0])
        else:
            for support in _sort_support_keys_for_drawing(pauli_bounds_this_box.keys()):
                bound = pauli_bounds_this_box[support]
                max_bound_value = max(bound, max_bound_value)
                gate = Gate(f"{round(100 * bound) / 100:.2f}", len(support), [], label="")
                box_body.append(gate, support)

        bounds_circuit.box(box_body, qubits=qargs, clbits=[])

    bounds_circuit.metadata["max_bound"] = max_bound_value
    return bounds_circuit


def _sort_support_keys_for_drawing(supports: Iterable[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """Sort the supports for optimal plotting depth.

    Given a list of integer tuples (indicating the gate supports in the circuit to be plotted), sort
    them to ensure a plotting order that gives optimal circuit depth.

    Args:
        supports: the support tuples to be sorted.

    Returns:
        The ordered support tuples.
    """
    sorted_wt1: list[tuple[int]] = []
    sorted_wt2: list[tuple[int, int]] = []
    sorted_heavy: list[tuple[int, ...]] = []
    for support in supports:
        length = len(support)
        if length == 1:
            sorted_wt1.append(cast(tuple[int], support))
        elif length == 2:
            sorted_wt2.append(cast(tuple[int, int], support))
        else:
            sorted_heavy.append(support)
    # Ordering of weight 1 shouldn't matter for plot.
    # Group weight-2 terms into dense layers:
    sorted_wt2 = _group_qb_pairs_into_layers(sorted_wt2)
    # TODO: Ordering of weight 3+ is left unchanged. Can we reasonably do something about this?
    # NOTE: for the foreseeable future, we do not expect to encounter weight >2 noise terms.
    sorted_all = sorted_wt1 + sorted_wt2 + sorted_heavy
    return sorted_all


def _group_qb_pairs_into_layers(wt2_supports: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Group weight-2 supports for optimal plotting depth.

    Given a list of pairs of integers (i.e. weight-2 support tuples), reorder them to ensure a
    plotting order that gives optimal circuit depth.

    Args:
        wt2_supports: the weight-2 support tuples to be grouped.

    Returns:
        The ordered weight-2 support tuples.
    """
    # Generate graph coloring for coupling map
    graph = rx.PyGraph()
    unique_qbs = list({qb for gate in wt2_supports for qb in gate})
    node_indices = graph.add_nodes_from(unique_qbs)
    qb_to_node = dict(zip(unique_qbs, node_indices, strict=False))  # type: ignore[call-overload]
    edges = [(qb_to_node[q1], qb_to_node[q2]) for q1, q2 in wt2_supports]
    edge_indices = graph.add_edges_from_no_data(edges)

    # Do the coloring, which gives mapping from edge_idx to layer_idx
    try:
        # This works for most common coupling maps
        # (e.g. 2d square grid or lower connectivity)
        coloring = rx.graph_bipartite_edge_color(graph)
    except rx.GraphNotBipartite:
        coloring = rx.graph_greedy_edge_color(graph)

    # Group gates into layers:
    layers: list[list[tuple[int, int]]] = [[] for _ in range(len(set(coloring.values())))]
    for gate_idx, gate in enumerate(wt2_supports):
        edge_idx = edge_indices[gate_idx]
        layers[coloring[edge_idx]].append(gate)

    sorted_pairs = [pair for layer in layers for pair in layer]
    return sorted_pairs
