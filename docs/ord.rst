Open reaction database
=======================

``rxnutils`` contain two pipelines that together imports and prepares the reaction data from the `Open reaction database <https://open-reaction-database.org/>`_ so that it can be used on modelling.

It is a complete end-to-end pipeline that is designed to be transparent and reproducible.

Pre-requisites
--------------

The reason the pipeline is divided into two blocks is because the dependencies of the atom-mapper package (``rxnmapper``) is incompatible with 
the dependencies ``rxnutils`` package. Therefore, to be able to use to full pipeline, you need to setup two python environment. 

1. Install ``rxnutils`` according to the instructions in the `README`-file

2. Install the ``ord-schema`` package in the `` rxnutils`` environment

    conda activate rxn-env
    python -m pip install ord-schema

3. Download/Clone the ``ord-data`` repository according to the instructions here: https://github.com/Open-Reaction-Database/ord-data

    git clone https://github.com/open-reaction-database/ord-data.git .

Note down the path to the repository as this needs to be given to the preparation pipeline

4. Install ``rxnmapper`` according to the instructions in the repo: https://github.com/rxn4chemistry/rxnmapper


.. code-block::
            
    conda create -n rxnmapper python=3.6 -y
    conda activate rxnmapper
    conda install -c rdkit rdkit=2020.03.3.0
    python -m pip install rxnmapper


5. Install ``Metaflow`` and ``rxnutils`` in the new environment


.. code-block::

    python -m pip install metaflow
    python -m pip install --no-deps --ignore-requires-python . 


Usage
-----

Create a folder for the ORD data and in that folder execute this command in the ``rxnutils`` environment


.. code-block::

    conda activate rxn-env
    python -m rxnutils.data.ord.preparation_pipeline run --nbatches 200  --max-workers 8 --max-num-splits 200 --ord-data ORD_DATA_REPO_PATH


and then in the environment with the ``rxnmapper`` run


.. code-block::

    conda activate rxnmapper
    python -m rxnutils.data.mapping_pipeline run --data-prefix ord --nbatches 200  --max-workers 8 --max-num-splits 200


The ``-max-workers`` flag should be set to the number of CPUs available.

On 8 CPUs and 1 GPU the pipeline takes a couple of hours.


Artifacts
---------

The pipelines creates a number of `tab-separated` CSV files:

    * `ord_data.csv` is the imported ORD data
    * `ord_data_cleaned.csv` is the cleaned and filter data
    * `ord_data_mapped.csv` is the atom-mapped, modelling-ready data


The cleaning is done to be able to atom-map the reactions and are performing the following tasks:
    * Ignore extended SMILES information in the SMILES strings 
    * Remove molecules not sanitizable by RDKit
    * Remove reactions without any reactants or products 
    * Move all reagents to reactants
    * Remove the existing atom-mapping
    * Remove reactions with more than 200 atoms when summing reactants and products 

(the last is a requisite for ``rxnmapper`` that was trained on a maximum token size roughly corresponding to 200 atoms)


The ``ord_data_mapped.csv`` files will have the following columns:

    * ID - unique ID from the original database
    * Dataset - the name of the dataset from which this is reaction is taken
    * Date - the date of the experiment as given in the database
    * ReactionSmiles - the original reaction SMILES
    * Yield - the yield of the first product of the first outcome, if provided
    * ReactionSmilesClean - the reaction SMILES after cleaning
    * BadMolecules - molecules not sanitizable by RDKit
    * ReactantSize - number of atoms in reactants
    * ProductSize - number of atoms in products
    * mapped_rxn - the mapped reaction SMILES
    * confidence - the confidence of the mapping as provided by ``rxnmapper`` 
