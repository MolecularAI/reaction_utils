# USPTO data

This sub-package contains tools to download, process and atom-map USPTO data

It is divided into two parts:
- Preparation
- Atom-mapping


## Preparation

This parts contists of three tasks
1. Download USPTO data from Figshare (`download.py`)
2. Combine USPTO data from different files and create row indices (`combine.py`)
3. Clean-up

Each of the tasks can be run individually or as a Metaflow pipeline, which is the recommended way.

First install the `rxnutils` package and activate that python environment. 

Then 

    python -m rxnutils.data.uspto.preparation_pipeline run --nbatches 200  --max-workers 8 --max-num-splits 200


## Atom-mapping

This part will atom-map the data prepared by the preparation pipeline using the [rxnmapper](https://github.com/rxn4chemistry/rxnmapper) package.

This needs to be installed and run in a new environment

    conda create -n rxnmapper python=3.6 -y
    conda activate rxnmapper
    conda install -c rdkit rdkit=2020.03.3.0
    python -m pip install rxnmapper metaflow
    python -m pip install --no-deps --ignore-requires-python . 

Then you can run a command like this

    python -m rxnutils.data.mapping_pipeline run --data-prefix uspto --nbatches 200  --max-workers 8 --max-num-splits 200



