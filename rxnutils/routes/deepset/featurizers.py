""" Module with featurizers for molecules and reactions
"""

from operator import itemgetter
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from rdkit.Chem import AllChem, DataStructs

from rxnutils.chem.reaction import ChemicalReaction


def collect_reaction_features(
    reaction_data: List[Dict[str, Any]],
    target_fingerprint: np.ndarray,
    reaction_class_ranks: Dict[str, int],
    featurizer: Callable[[Dict[str, Any]], np.ndarray],
    prefactor: float = 1.1,
    default_rank: int = 5,
    default_cls: str = "0.0",
) -> Tuple[float, np.ndarray]:
    """
    Collect all reaction features and compute the prior reaction feasibility score

    :params reaction_data: the reactions of the route
    :params target_fingerprint: the precomputed fingerprint of the target
    :params reaction_class_ranks: the ranks of the reaction classes
    :params featurizer: the featurizer to use for reactions
    :params prefactor: the factor for the feasibility score
    :params default_rank: the rank of reactions not parameterized
    :params default_cls: the class of reactions not parameterized
    :returns: the prior feasibility score
    :returns: the reaction features
    """
    weights = [prefactor**idx for idx in range(1, len(reaction_data) + 1)]
    weights_sum = sum(weights)

    classes = []
    ranks = []
    fingerprints = []
    rank_score = 0.0
    # This result in reactions being traversed breadth-first instead of depth-first
    reaction_data_sorted = sorted(reaction_data, key=itemgetter("tree_depth"))
    for weight, reaction_data in zip(weights, reaction_data_sorted):
        norm_weight = weight / weights_sum
        cls = reaction_data.get("classification", default_cls)
        if " " in cls:
            cls = cls.split(" ")[0]
        if cls in reaction_class_ranks:
            rank = reaction_class_ranks[cls]
        else:
            rank = default_rank
            cls = default_cls
        rank_score += norm_weight * rank

        class_vector = [int(item) for item in cls.split(".")]
        class_vector += [0] * (3 - len(class_vector))
        classes.append(class_vector)
        ranks.append(rank)
        fingerprints.append(featurizer(reaction_data))

    features = []
    for class_, rank, fingerprint in zip(classes, ranks, fingerprints):
        vector = np.asarray(class_ + [rank])
        vector = np.concatenate([vector, target_fingerprint, fingerprint])
        features.append(vector)
    return rank_score, np.asarray(features)


def default_reaction_featurizer(reaction_data: Dict[str, Any], radius: int = 2, numbits: int = 64) -> np.ndarray:
    """
    Given an item of reaction data, returned by the `reaction_data` method
    of `SynthesisRoute` this features the reaction using a difference ECFP

    Currently it featurizes the reaction in the retro-sense

    :params reaction_data: the reaction as a dictionary
    :params radius: the radius of the fingerprint
    :params numbits: the number of bits
    :returns: the fingerprint
    """
    rsmi = ">>".join(reaction_data["reaction_smiles"].split(">>")[::-1])
    rxn = ChemicalReaction(rsmi, clean_smiles=False)
    return reaction_difference_fingerprint(rxn, radius, numbits)


def ecfp_fingerprint(mol: AllChem.rdchem.Mol, radius: int = 2, numbits: int = 64) -> np.ndarray:
    """
    Computes a ECFP with a given radius and length

    :params mol: the RDKit molecule
    :params radius: the radius of the fingerprint
    :params numbits: the number of bits
    :returns: the fingerprint
    """
    array = np.zeros((0,), dtype=np.int8)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=numbits)
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    return array


def reaction_difference_fingerprint(reaction: ChemicalReaction, radius: int = 2, numbits: int = 64) -> np.ndarray:
    """
    Computes the difference fingerprint of given reaction

    :params reaction: the reaction to featurize
    :params radius: the radius of the fingerprint
    :params numbits: the number of bits
    :returns: the fingerprint
    """
    fp = np.zeros(numbits)
    for product in reaction.products:
        if product is not None:
            fp += ecfp_fingerprint(product, radius, numbits)
    for reactant in reaction.reactants:
        if reactant is not None:
            fp -= ecfp_fingerprint(reactant, radius, numbits)
    return fp
