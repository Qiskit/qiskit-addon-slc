Installation instructions
=========================

First, choose how you're going to run and install the packages. There are two primary ways to do this:

- :ref:`Option 1`
- :ref:`Option 2`

Prerequisites
^^^^^^^^^^^^^

First, create a minimal environment with only Python installed in it. We recommend using `Python virtual environments <https://docs.python.org/3.10/tutorial/venv.html>`__.

.. code:: sh

    python3 -m venv /path/to/virtual/environment

Activate your new environment.

.. code:: sh

    source /path/to/virtual/environment/bin/activate


.. _Option 1:

Option 1: Install from PyPI
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most straightforward way to install the ``qiskit-addon-slc`` package is via ``PyPI``.

.. code:: sh

    pip install 'qiskit-addon-slc'


.. _Option 2:

Option 2: Install from Source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you plan to develop in the repository or run the notebooks locally, install from source.

If so, the first step is to clone the ``qiskit-addon-slc`` repository.

.. code:: sh

    git clone git@github.com:Qiskit/qiskit-addon-slc.git

Next, upgrade ``pip`` and enter the repository.

.. code:: sh

    pip install --upgrade pip
    cd qiskit-addon-slc

The next step is to install ``qiskit-addon-slc`` to the virtual environment.
Install the notebook dependencies if you plan to run all the visualizations in the notebooks.
If you plan on developing in the repository, install the ``dev`` dependencies.

Adjust the options below to suit your needs.

.. code:: sh

    pip install tox notebook -e '.[notebook-dependencies,dev]'

If you installed the notebook dependencies, you can get started by running the notebooks in the docs.

.. code::

    cd docs/
    jupyter lab
