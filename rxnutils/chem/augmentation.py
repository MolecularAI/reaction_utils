""" Routines for augmenting chemical reactions
"""

from rxnutils.chem.utils import split_rsmi

_SINGLE_REACTANT_REAGENTS = {"10.1.1": "Br", "10.1.2": "Cl"}


def single_reactant_augmentation(smiles: str, classification: str) -> str:
    """
    Augment single-reactant reaction with additional reagent if possible
    based on the classification of the reaction
    :param smiles: the reaction SMILES to augment
    :param classification: the classification of the reaction or an empty string
    :return: the processed SMILES
    """
    reactants = split_rsmi(smiles)[0]
    if "." in reactants:
        return smiles
    classification = classification.split(" ")[0]
    new_reactant = _SINGLE_REACTANT_REAGENTS.get(classification)
    if new_reactant:
        return new_reactant + "." + smiles
    return smiles
