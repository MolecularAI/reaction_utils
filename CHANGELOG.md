# CHANGELOG

## Version 1.8.0 - 2024-12-18

### Features

- DeepSet route scoring routines
- SCScore featurizer for molecules
- Routine for identifying symmetric sites on molecules
- Batch utilities for numpy arrays
- Reader for synthesis route from aizynthfinder dictionary

### Trivial changes

- Update dependencies

## Version 1.7.0 - 2024-10-15

### Features

- Support for novel route comparison metric
- Support for tree edit distance (TED) calculations previously in the `route_distances` package
- Support for Retro-BLEU and Badowski et al. route scoring
- Update to USPTO pre-processing pipeline to support extracting yields
- Extended route methods taken from `route_distances`
- Support for augmenting single-reactant reactions

### Trivial changes

- Updates to reaction tagging routines

## Version 1.6.0 - 2024-06-13

### Trivial changes

- rdkit version requirements have been updated to versions above 2023.9.1.

## Version 1.5.0 - 2024-05-27

### Features

- Adding support for tagging reaction sites in SMILES
- Adding more options for re-mapping routes

### Miscellaneous

- Improving batch routines
- Updating InChI tools download URL

## Version 1.4.0 - 2024-03-12

### Features

- Adding support for reading and processing routes
- Extracting co-reactant for ChemicalReaction class

### Trivial changes

- Making help for pipeline runner simpler

## Version 1.3.0 - 2024-01-09

### Features

- Adding support for Condensed Graph of Reaction
- Adding support for flagging stereocontrolled reactions
- Adding routines for desalting reactions
- Adding routines for identifying reaction centers
- Adding several dataframe modifying actions
- Adding several reaction modifying actions

### Trival changes

- Unifying routines to make and concatenate batches

## Version 1.2.0 - 2022-11-21

### Features

- Adding pipeline actions for identifying RingBreaker reactions
- Adding support for extracting new type of RingBreaker templates

## Version 1.1.1 - 2022-08-31

### Trival changes

- Change project name to reaction-utils
- Change RDChiral dependency to pypi package

## Version 1.1.0 - 2022-08-31

### Features

- Support for importing and preparing reactions from the Open reaction database

### Bug-fixes

- Multiplatform support for RInChI features
- Checking if RDKit Contrib folder is properly installed
- Documentation errors corrected

### Trivial changes

- Updating RDkit dependency to use pypi package
- Replacing used timeout package

## Version 1.0.0 - 2022-07-01

- First stable version
