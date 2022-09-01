rxnutils documentation
============================

rxnutils is a collection of routines for working with reactions, reaction templates and template extraction

Introduction
------------

The package is divided into (currently) three sub-packages:

* `chem` - chemistry routines like template extraction or reaction cleaning
* `data` - routines for manipulating various reaction data sources
* `pipeline` - routines for building and executing simple pipelines for modifying and analyzing reactions

Auto-generated API documentation is available, as well as guides for common tasks.  See the menu to the left.

Installation
------------

For most users it is as simple as

.. code-block::

    pip install reaction-utils


`For developers`, first clone the repository using Git.

Then execute the following commands in the root of the repository

.. code-block::

    conda env create -f env-dev.yml
    conda activate rxn-env
    poetry install


the `rxnutils` package is now installed in editable mode.

Lastly, make sure to install pre-commits that are run on every commit

.. code-block::

    pre-commit install


Limitations
-----------

* Some old RDKit wheels on pypi did not include the `Contrib` folder, preventing the usage of the `rdkit_RxnRoleAssignment` action
* The pipeline for the Open reaction database requires some additional dependencies, see the documentation for this pipeline
* Using the data piplines for the USPTO and Open reaction database requires you to setup a second python environment
* The RInChI capabilities are not supported on MacOS


.. toctree::
    :hidden:
    
    templates
    uspto
    ord
    pipeline
    rxnutils
