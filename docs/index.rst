#######################
Shaded lightcones (SLC)
#######################

.. image:: images/lightcones.png

`qiskit-addon-slc` is a package for computing the shaded lightcone (SLC) `[1] <ref1_>`__ of an
observable with respect to a quantum circuit. In the context of probabilistic error cancellation (PEC), shaded lightcones
are similar to conventional binary lightcones in that not mitigating errors outside the
lightcone can reduce the variance (i.e. sampling cost). Lightcone shading allows this strategy to
be pushed further by assigning scales to Pauli error generators within the causal lightcone.
Errors that are assigned smaller scales have less effect on the observable; truncating them from the
noise model can reduce variance (i.e. sampling cost) at the cost of some bias. The `qiskit-addon-slc` package
gives users the ability to do lightcone shading and control the tradeoff between sampling cost and bias.

Getting started
---------------

A simple guide to help you get started quickly with this package is available in the :doc:`quickstart guide <guides/quickstart>`.

Use case examples
-----------------

This technique has been used to improve the sampling cost of PEC on a 20-qubit mirrored Ising circuit `[tutorial] <https://quantum.cloud.ibm.com/docs/en/tutorials/pec-with-shaded-lightcones>`__.

Technical discussion
--------------------

Method overview
"""""""""""""""

Shaded lightcones are calculated and used in 5 steps:

1. Compute a bound on the effect of each Pauli error term on the observable at the end of the circuit (forward bound)
2. Compute a bound on the effect of each Pauli error term on the initial state at the beginning of the circuit (backward bound)

.. image:: images/bounds.png

3. Approximate a bias contribution for each Pauli error term using the forward/backward bounds and the term's error rate

.. image:: images/merge_bounds.png

4. Prioritize error terms based on their error rate and bounds. Truncate terms from the noise model which have the least effect on the observable expectation value until the user-specified bias tolerance is hit. Alternatively, one can add the most impactful error terms to a noise model until the user-specified sampling cost budget is filled.

.. image:: images/prioritize_and_truncate.png

5. Mitigate the truncated noise model.

.. image:: images/reduced_pec.png

Software features
"""""""""""""""""

- Parallel asynchronous bound computation
- `Rust-accelerated propagation <https://quantum.cloud.ibm.com/docs/addons/pauli-prop>`_
- Permits ahead-of-time bound computation (i.e. prior to the actual noise learning)

Known issues
""""""""""""

- Windows not supported
- `InjectNoise(site="before")` not supported
- Does not support fine-grained bound merging

Future work
"""""""""""

- Rust-accelerated eigenvalue computation for computing forward bounds
- Additional guides coming soon

Contributing
------------

The source code is available `on GitHub <https://github.com/Qiskit/qiskit-addon-slc>`_.

The developer guide is located at `CONTRIBUTING.md <https://github.com/Qiskit/qiskit-addon-slc/blob/main/CONTRIBUTING.md>`_
in the root of this project's repository.
By participating, you are expected to uphold Qiskit's `code of conduct <https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md>`_.

We use `GitHub issues <https://github.com/Qiskit/qiskit-addon-slc/issues/new/choose>`_ for tracking requests and bugs.

Citing this package
-------------------

If you use this package in your research, use the `CITATION.bib <https://github.com/Qiskit/qiskit-addon-slc/blob/main/CITATION.bib>`_ file in this project's repository to cite the appropriate reference(s).

License
-------

`Apache License 2.0 <https://github.com/Qiskit/qiskit-addon-slc/blob/main/LICENSE.txt>`_

Deprecation Policy
------------------

We follow `semantic versioning <https://semver.org/>`_. We may occasionally make breaking changes in order to
improve the user experience. When possible, we will keep old interfaces and mark them as deprecated, as long
as they can co-exist with the new ones. Each substantial improvement, breaking change, or deprecation will be
documented in the `release notes <https://quantum.cloud.ibm.com/docs/api/qiskit-addon-slc/release-notes>`_.

Contributing
------------

The source code is available `on GitHub <https://github.com/Qiskit/qiskit-addon-slc>`_.

The developer guide is located at `CONTRIBUTING.md <https://github.com/Qiskit/qiskit-addon-slc/blob/main/CONTRIBUTING.md>`_.
By participating, you are expected to uphold Qiskit's `code of conduct <https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md>`_.

We use `GitHub issues <https://github.com/Qiskit/qiskit-addon-slc/issues/new/choose>`_ for tracking requests and bugs.

.. _references:

References
----------

.. _ref1:
   
1. Andrew Eddins, et al., `Lightcone shading for classically accelerated quantum error mitigation <https://arxiv.org/abs/2409.04401v1>`_, arXiv:2409.04401v1 [quant-ph].

.. toctree::
   :hidden:

   Documentation home <self>
   Installation instructions <install>
   Guides <guides/index>
   GitHub <https://github.com/Qiskit/qiskit-addon-slc>

.. toctree::
   :hidden:
   :caption: Tutorials

   Probabilistic error cancellation with shaded lightcones <https://quantum.cloud.ibm.com/docs/tutorials/pec-with-shaded-lightcones>

.. toctree::
   :hidden:
   :caption: API reference

   Python API reference <https://quantum.cloud.ibm.com/docs/api/qiskit-addon-slc>
   Release notes <release-notes>
