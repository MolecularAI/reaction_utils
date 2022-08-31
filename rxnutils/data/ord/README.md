# Open reaction database

This sub-package contains tools to import, process and atom-map data from the [Open reaction database](https://open-reaction-database.org/)

It is divided into two parts:
- Preparation
- Atom-mapping


## Preparation

This parts contists of two tasks
1. Import the reactions from the database (`import_ord_dataset.py`)
2. Clean-up

Each of the tasks can be run individually or as a Metaflow pipeline, which is the recommended way.

First install the `rxnutils` package and activate that python environment. 

Second, install the `ord-schema` package in this environment

    conda activate rxn-env
    python -m pip install ord-schema

Third, clone the ord-data repository as detailed [here](https://github.com/Open-Reaction-Database/ord-data).

    git clone https://github.com/open-reaction-database/ord-data.git .

Finally you can run, 

    python -m rxnutils.data.ord.preparation_pipeline run --nbatches 200  --max-workers 8 --max-num-splits 200 --ord-data ORD_DATA_REPOSITORY_FOLDER


## Atom-mapping

This part will atom-map the data prepared by the preparation pipeline using the [rxnmapper](https://github.com/rxn4chemistry/rxnmapper) package.

This needs to be installed and run in a new environment

    conda create -n rxnmapper python=3.6 -y
    conda activate rxnmapper
    conda install -c rdkit rdkit=2020.03.3.0
    python -m pip install rxnmapper metaflow
    python -m pip install --no-deps --ignore-requires-python . 

Then you can run a command like this

    python -m rxnutils.data.mapping_pipeline run --data-prefix ord --nbatches 200  --max-workers 8 --max-num-splits 200



