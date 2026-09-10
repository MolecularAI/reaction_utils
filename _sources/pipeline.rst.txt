Pipeline
========

``rxnutils`` provide a simple pipeline to perform simple tasks on reaction SMILES and templates in a CSV-file.


The pipeline works on  `tab-separated` CSV files (TSV files) 


Usage
-----

To exemplify the pipeline capabilities, we will have a look at the pipeline used to clean the USPTO data.

The input to the pipeline is a simple YAML-file that specifies each action to take. The actions will be executed
sequentially, one after the other and each action takes a number of input arguments. 

This is the YAML-file used to clean the USPTO data:

.. code-block:: yaml

    trim_rxn_smiles:
        in_column: ReactionSmiles
        out_column: ReactionSmilesClean 
    remove_unsanitizable:
        in_column: ReactionSmilesClean
        out_column: ReactionSmilesClean
    reagents2reactants:
        in_column: ReactionSmilesClean
        out_column: ReactionSmilesClean
    remove_atom_mapping:
        in_column: ReactionSmilesClean
        out_column: ReactionSmilesClean
    reactantsize:
        in_column: ReactionSmilesClean
    productsize:
        in_column: ReactionSmilesClean
    query_dataframe1:
        query: "ReactantSize>0"
    query_dataframe2:
        query: "ProductSize>0"
    query_dataframe3:
        query: "ReactantSize+ProductSize<200"


The first action is called ``trim_rxn_smiles`` and two arguments are given: ``in_column`` specifying which column to use as input and ``out_column`` specifying which column
to use as output. 

The following actions ``remove_unsanitizable``, ``reagents2reactants``, ``remove_atom_mapping``, ``reactantsize``, ``productsize`` works the same way, but might use other columns to specified for output. 

The last three actions are actually the same action but executed with different arguments. They therefore have to be postfixed with 1, 2 and 3. 
The action ``query_dataframe`` takes a ``query`` argument and removes a number of rows not matching the query.

If we save this to ``clean_pipeline.yml`` and given that we have a tab-separated file with USPTO data called ``uspto_data.csv`` we can run the following command

.. code-block::

    python -m rxnutils.pipeline.runner --pipeline clean_pipeline.yml --data uspto_data.csv --output uspto_cleaned.csv


or we can alternatively run it from a python method like this

.. code-block::

    from rxnutils.pipeline.runner import main as validation_runner
    
    validation_runner(
        [
            "--pipeline",
            "clean_pipeline.yml",
            "--data",
            "uspto_data.csv",
            "--output",
            "uspto_cleaned.csv",
        ]
    )

Actions
-------

To find out what actions are available, you can type

.. code-block::

    python -m rxnutils.pipeline.runner --list

Development
-----------

New actions can easily be added to the pipeline framework. All of the actions are implemented in one of four modules 


    * ``rxnutils.pipeline.actions.dataframe_mod`` - actions that modify the dataframe, e.g., removing rows or columns
    * ``rxnutils.pipeline.actions.reaction_mod`` - actions that modify reaction SMILES
    * ``rxnutils.pipeline.actions.dataframe_props`` - actions that compute properties from reaction SMILES
    * ``rxnutils.pipeline.actions.templates`` - actions that process reaction templates


To exemplify, let's have a look at the ``productsize`` action


.. code-block:: python

    @action
    @dataclass
    class ProductSize:
    """Action for counting product size"""

    pretty_name: ClassVar[str] = "productsize"
    in_column: str
    out_column: str = "ProductSize"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        smiles_col = global_apply(data, self._row_action, axis=1)
        return data.assign(**{self.out_column: smiles_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (number of heavy atoms in product)"

    def _row_action(self, row: pd.Series) -> str:
        _, _, products = row[self.in_column].split(">")
        products_mol = Chem.MolFromSmiles(products)

        if products_mol:
            product_atom_count = products_mol.GetNumHeavyAtoms()
        else:
            product_atom_count = 0

        return product_atom_count

The action is defined as a class ``ProductSize`` that has two class-decorators. 
The first ``@action`` will register the action in a global action list and second ``@dataclass`` is dataclass decorator from the standard library.
The ``pretty_name`` class variable is used to identify the action in the pipeline, that is what you are specifying in the YAML-file. 
The other two ``in_column`` and ``out_column`` are the arguments you can specify in the YAML file for executing the action, they can have default 
values in case they don't need to be specified in the YAML file.

When the action is executed by the pipeline the ``__call__`` method is invoked with the current Pandas dataframe as the only argument. This method 
should return the modified dataframe.

Lastly, it is nice to implement a ``__str__`` method which is used by the pipeline to print useful information about the action that is executed.
