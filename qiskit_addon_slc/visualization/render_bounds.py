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

"""The bounds rendering method."""

from __future__ import annotations

from typing import cast

import matplotlib as mpl
import numpy as np
from matplotlib import cm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit.visualization.circuit.matplotlib import MPLDefaultStyle


def render_bounds(
    bounds_circuit: QuantumCircuit,
    *,
    pauli_filter: Pauli | str | int | None = None,
    **kwargs,
) -> mpl.figure.Figure:
    """Renders a quantum circuit with overlaid bounds and according styling.

    Args:
        bounds_circuit: the quantum circuit with overlaid bounds. See also
            :func:`overlay_bounds_onto_circuit`.
        pauli_filter: the optional Pauli type by which the bounds were filtered. This will be
            indicated in the produced figure's title.
        kwargs: any additional keyword arguments are forwarded to :meth:`.QuantumCircuit.draw`.

    Returns:
        The ``mpl`` figure.
    """
    max_bound = bounds_circuit.metadata.get("max_bound", 2.0)
    max_plot_val = float(f"{round(100 * max_bound) / 100:.2f}")
    num_colors = 1 + round(max_plot_val * 100)
    colors = [cm.viridis(x) for x in np.linspace(0, 1, num_colors)]  # type: ignore[attr-defined]
    values = np.linspace(0, max_plot_val, num_colors)

    style_dict = MPLDefaultStyle().style
    style_dict["displaycolor"] = {
        f"{v:.2f}": (mpl.colors.to_hex(color), "0.5")
        for color, v in zip(colors, values, strict=False)  # type: ignore[call-overload]
    }
    style_dict["displaycolor"]["h"] = ("darkblue", "white")
    style_dict["displaycolor"]["hide"] = ("#00000000", "#00000000")
    style_dict |= {
        "textcolor": "0.5",
        "backgroundcolor": "black",
        "wirecolor": "0.5",
    }
    fig = bounds_circuit.draw(output="mpl", style=style_dict, **kwargs)

    if pauli_filter is not None:
        if isinstance(pauli_filter, int):
            title = f"{pauli_filter}-qubit errors"
        else:
            pauli_str = Pauli(pauli_filter).to_label()
            if len(pauli_str) > 1:
                pauli_str = "".join([f"{c}_{{{i}}}" for i, c in enumerate(pauli_str[::-1])][::-1])
            title = f"${pauli_str}$ errors"

        fig.suptitle(title, y=0.91, color="white")

    return cast(mpl.figure.Figure, fig)
