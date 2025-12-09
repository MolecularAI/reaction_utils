USPTO
=====

``rxnutils`` contain two pipelines that together downloads and prepares the USPTO reaction data so that it can be used on modelling.

It is a complete end-to-end pipeline that is designed to be transparent and reproducible.

Pre-requisites
--------------

The reason the pipeline is divided into two blocks is because the dependencies of the atom-mapper package (``rxnmapper``) is incompatible with 
the dependencies ``rxnutils`` package. Therefore, to be able to use to full pipeline, you need to setup two python environment. 

1. Install ``rxnutils`` according to the instructions in the `README`-file

2. Install ``rxnmapper`` according to the instructions in the repo: https://github.com/rxn4chemistry/rxnmapper


.. code-block::
            
    conda create -n rxnmapper python=3.6 -y
    conda activate rxnmapper
    conda install -c rdkit rdkit=2020.03.3.0
    python -m pip install rxnmapper


3. Install ``Metaflow`` and ``rxnutils`` in the new environment


.. code-block::

    python -m pip install metaflow
    python -m pip install --no-deps --ignore-requires-python . 


Usage
-----

Create a folder for the USPTO data and in that folder execute this command in the ``rxnutils`` environment


.. code-block::

    conda activate rxn-env
    python -m rxnutils.data.uspto.preparation_pipeline run --nbatches 200  --max-workers 8 --max-num-splits 200


and then in the environment with the ``rxnmapper`` run


.. code-block::

    conda activate rxnmapper
    python -m rxnutils.data.mapping_pipeline run --data-prefix uspto --nbatches 200  --max-workers 8 --max-num-splits 200


The ``-max-workers`` flag should be set to the number of CPUs available.

On 8 CPUs and 1 GPU the pipeline takes a couple of hours.


Artifacts
---------

The pipelines creates a number of `tab-separated` CSV files:

    * `1976_Sep2016_USPTOgrants_smiles.rsmi` and `2001_Sep2016_USPTOapplications_smiles.rsmi` is the original USPTO data downloaded from Figshare
    * `uspto_data.csv` is the combined USPTO data, with selected columns and a unique ID for each reaction
    * `uspto_data_cleaned.csv` is the cleaned and filter data
    * `uspto_data_mapped.csv` is the atom-mapped, modelling-ready data


The cleaning is done to be able to atom-map the reactions and are performing the following tasks:
    * Ignore extended SMILES information in the SMILES strings 
    * Remove molecules not sanitizable by RDKit
    * Remove reactions without any reactants or products 
    * Move all reagents to reactants
    * Remove the existing atom-mapping
    * Remove reactions with more than 200 atoms when summing reactants and products 

(the last is a requisite for ``rxnmapper`` that was trained on a maximum token size roughly corresponding to 200 atoms)


The ``uspo_data_mapped.csv`` files will have the following columns:

    * ID - unique ID created by concatenated patent number, paragraph and row index  in the original data file
    * Year - the year of the patent filing
    * ReactionSmiles - the original reaction SMILES
    * ReactionSmilesClean - the reaction SMILES after cleaning
    * BadMolecules - molecules not sanitizable by RDKit
    * ReactantSize - number of atoms in reactants
    * ProductSize - number of atoms in products
    * mapped_rxn - the mapped reaction SMILES
    * confidence - the confidence of the mapping as provided by ``rxnmapper`` 
