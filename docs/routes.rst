Routes
======

``rxnutils`` contains routines to analyse synthesis routes. There are a number of readers that can be used to read routes from a number of 
formats, and there are routines to score the different routes.

Reading
-------

The simplest route format supported is a text file, where each reaction is written as a reaction SMILES in a line. 
Routes are separated by new-line

For instance:

.. code-block::

    CC(C)N.Clc1cccc(Nc2ccoc2)n1>>CC(C)Nc1cccc(Nc2ccoc2)n1
    Brc1ccoc1.Nc1cccc(Cl)n1>>Clc1cccc(Nc2ccoc2)n1

    Nc1cccc(NC(C)C)n1.Brc1ccoc1>>CC(C)Nc1cccc(Nc2ccoc2)n1
    CC(C)N.Nc1cccc(Cl)n1>>Nc1cccc(NC(C)C)n1


If this is saved to ``routes.txt``, these can be read into route objects with 

.. code-block::

    from rxnutils.routes.readers import read_reaction_lists
    routes = read_reaction_lists("reactions.txt")


If you have an environment with ``rxnmapper`` installed and the NextMove software ``namerxn`` in your PATH then you can
add atom-mapping and reaction classes to these routes with

.. code-block::

    # This can be set on the command-line as well
    import os
    os.environ["RXNMAPPER_ENV_PATH"] = "/home/username/miniconda/envs/rxnmapper/"

    for route in routes:
        route.assign_atom_mapping(only_rxnmapper=True)
    routes[1].remap(routes[0])


The last line of code also make sure that the second route shares mapping with the first route. 


Other readers are available

* ``read_aizynthcli_dataframe`` - for reading routes from aizynthcli output dataframe
* ``read_reactions_dataframe`` - for reading routes stored as reactions in a dataframe


For instance, to read routes from a dataframe with reactions. You can do something like what follows.
The dataframe has column ``reaction_smiles`` that holds the reaction SMILES, and the individual routes
are identified by a ``target_smiles`` and ``route_id`` column. The dataframe also has a column ``classification``,
holding the NextMove classification. The dataframe is called ``data``.

.. code-block::

    from rxnutils.routes.readers import read_reactions_dataframe
    routes = read_reactions_dataframe(
        data, 
        "reaction_smiles", 
        group_by=["target_smiles", "route_id"], 
        metadata_columns=["classification"]
    )
