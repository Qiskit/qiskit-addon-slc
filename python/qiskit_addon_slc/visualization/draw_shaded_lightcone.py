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

"""The single-call shaded lightcone drawing method."""

from __future__ import annotations

import matplotlib as mpl
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, QubitSparsePauliList

from ..bounds.commutator_bounds import Bounds
from .accumulate_filtered_bounds import accumulate_filtered_bounds
from .overlay_bounds import overlay_bounds_onto_circuit
from .render_bounds import render_bounds


def draw_shaded_lightcone(
    circuit: QuantumCircuit,
    bounds: Bounds,
    noise_model_paulis: dict[str, QubitSparsePauliList],
    *,
    pauli_filter: Pauli | str | int | None = None,
    include_empty_boxes: bool = True,
    **rendering_kwargs,
) -> mpl.figure.Figure:
    """Draws a shaded lightcone.

    First, the provided ``bounds`` are accumulated and filtered according to ``pauli_filter``. See
    also :func:`.accumulate_filtered_bounds`.
    Then, the resulting bounds are overlaid onto the provided ``circuit`` (see
    :func:`.overlay_bounds_onto_circuit`) and subsequently rendered (see :func:`.render_bounds`).

    This is the single-call entry point for the three-step pipeline; use it unless you need the
    intermediate results.

    The noise models below come from :func:`~qiskit_addon_slc.utils.generate_noise_model_paulis`,
    which produces the 1- and 2-weight Pauli terms that are actually learned for each box. The bound
    values are made up, with :math:`Z` errors taken three times as likely as :math:`X` ones so that
    the effect of filtering is visible.

    .. plot::
       :include-source:
       :context: reset
       :alt: A two-box quantum circuit on three qubits whose boxes are shaded by their accumulated
             error bound, with a vertical colorbar to the right of the circuit.

       >>> import numpy as np
       >>> from qiskit import QuantumCircuit
       >>> from qiskit.circuit import BoxOp
       >>> from qiskit.quantum_info import PauliLindbladMap
       >>> from samplomatic import InjectNoise
       >>> from samplomatic.utils import find_unique_box_instructions
       >>> from qiskit_addon_slc.utils import generate_noise_model_paulis
       >>> from qiskit_addon_slc.visualization import draw_shaded_lightcone

       >>> circuit = QuantumCircuit(3)
       >>> for idx, (qubit_0, qubit_1) in enumerate([(0, 1), (1, 2)]):
       ...     body = QuantumCircuit(3)
       ...     _ = body.cx(qubit_0, qubit_1)
       ...     annotation = InjectNoise(ref=f"noise_{idx}", modifier_ref=f"box_{idx}")
       ...     _ = circuit.append(BoxOp(body, annotations=[annotation]), [0, 1, 2])

       >>> noise_model_paulis = generate_noise_model_paulis(
       ...     find_unique_box_instructions(circuit)
       ... )
       >>> rates = {"X": 0.01, "Y": 0.02, "Z": 0.03}
       >>> bounds = {
       ...     f"box_{idx}": PauliLindbladMap.from_components(
       ...         np.array(
       ...             [
       ...                 sum(rates[pauli] for pauli in label)
       ...                 for label, _ in noise_model_paulis[f"noise_{idx}"].to_sparse_list()
       ...             ]
       ...         )
       ...         * (idx + 1),
       ...         noise_model_paulis[f"noise_{idx}"],
       ...     )
       ...     for idx in range(2)
       ... }

       >>> figure = draw_shaded_lightcone(circuit, bounds, noise_model_paulis)

    The ``pauli_filter`` is forwarded to :func:`.accumulate_filtered_bounds` and also recorded in the
    figure title. An ``int`` selects a Pauli weight, still mixing the Pauli types:

    .. plot::
       :include-source:
       :context: close-figs
       :alt: The same shaded circuit restricted to weight-1 error terms, titled "1-qubit errors".

       >>> figure = draw_shaded_lightcone(
       ...     circuit, bounds, noise_model_paulis, pauli_filter=1
       ... )

    A ``str`` or :class:`~qiskit.quantum_info.Pauli` matches each noise term reduced to its own
    non-identity support. Since the drawings above accumulate every Pauli type sharing a support,
    this is how to visualize one type on its own — and the two differ here by the factor of three
    between the :math:`X` and :math:`Z` rates:

    .. plot::
       :include-source:
       :nofigs:
       :context:

       >>> from qiskit_addon_slc.visualization import accumulate_filtered_bounds
       >>> for pauli_filter in (None, "X", "Z"):
       ...     filtered = accumulate_filtered_bounds(
       ...         circuit, bounds, noise_model_paulis, pauli_filter=pauli_filter
       ...     )
       ...     single_qubit = {
       ...         support[0]: round(float(bound), 3)
       ...         for support, bound in sorted(filtered["box_0"].items())
       ...         if len(support) == 1
       ...     }
       ...     print(f"{str(pauli_filter):>4}: {single_qubit}")
       None: {0: 0.06, 1: 0.06, 2: 0.06}
          X: {0: 0.01, 1: 0.01, 2: 0.01}
          Z: {0: 0.03, 1: 0.03, 2: 0.03}

    .. plot::
       :include-source:
       :context: close-figs
       :alt: The same shaded circuit restricted to single-qubit Z error terms, showing lighter
             shading than the unfiltered drawing and titled with the Z Pauli label.

       >>> figure = draw_shaded_lightcone(
       ...     circuit, bounds, noise_model_paulis, pauli_filter="Z"
       ... )

    Because the match uses the reduced support, a weight-2 term is selected by ``"XX"`` rather than
    by a full-width label such as ``"XXI"``:

    .. plot::
       :include-source:
       :context: close-figs
       :alt: The same circuit restricted to the weight-2 XX error terms, shading only the qubit
             pairs they act on and titled with subscripted Pauli labels.

       >>> figure = draw_shaded_lightcone(
       ...     circuit, bounds, noise_model_paulis, pauli_filter="XX"
       ... )

    .. seealso::
       :func:`.accumulate_filtered_bounds`, :func:`.overlay_bounds_onto_circuit`,
       :func:`.render_bounds`

    Args:
        circuit: the circuit whose shaded lightcone to draw.
        bounds: the bounds to use for the shaded lightcone.
        noise_model_paulis: the Pauli error terms of the circuit's noise models.
        pauli_filter: the optional Pauli type by which the bounds were filtered.
        include_empty_boxes: whether to include empty boxes or not.
        rendering_kwargs: any additional keyword arguments are forwarded to
            :meth:`.QuantumCircuit.draw`.

    Returns:
        The ``mpl`` figure.
    """
    pauli_bounds = accumulate_filtered_bounds(circuit, bounds, noise_model_paulis, pauli_filter)
    bounds_circuit = overlay_bounds_onto_circuit(
        pauli_bounds, circuit, include_empty_boxes=include_empty_boxes
    )
    return render_bounds(bounds_circuit, pauli_filter=pauli_filter, **rendering_kwargs)
