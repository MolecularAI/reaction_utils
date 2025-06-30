""" Calculates reaction centre RDKit fingerprints
"""

from itertools import permutations
from dataclasses import dataclass
from typing import Optional, List, Union

import numpy as np
from scipy.spatial.distance import cdist
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdChemReactions as Reactions

from rxnutils.chem.reaction import ChemicalReaction
from rxnutils.chem.utils import reaction_centres


@dataclass
class ReactionCentreFingerprint:
    """
    Reaction featurizer based on a native RDKit functions for reaction centre atoms

    if max_centers is not set, the output of the featurization will be a list of each
    centre's fingerprint - but if max_centers is given the output is a flattend and
    concatenated list of all centres up to max_centers. the fingerprint is padded
    with zeros, and an initial bit indicate the number of fingerprints that has
    been concatenated

    :params numbits: length of fingerprint
    :params max_centers: if given, will concatenate the individual FPs up to a maximum of centres
    """

    numbits: int = 1024
    max_centers: Optional[int] = None

    def __call__(
        self, reaction: ChemicalReaction
    ) -> Union[List[List[float]], List[float]]:
        fps = self._calc_fingerprint(reaction)

        if self.max_centers is None:
            return [arr.tolist() for arr in fps]

        fps = fps[: self.max_centers]
        fps_cat = np.zeros(self.max_centers * self.numbits + 1)
        fps_cat[0] = len(fps)
        for idx, fp in enumerate(fps):
            fps_cat[idx * self.numbits + 1 : (idx + 1) * self.numbits + 1] = fp
        return fps_cat.tolist()

    def _calc_fingerprint(self, reaction: ChemicalReaction) -> List[np.ndarray]:
        rdkit_rxn = Reactions.ReactionFromSmarts(reaction.rsmi, useSmiles=True)
        rdkit_rxn.Initialize()
        rxncenters = reaction_centres(rdkit_rxn)
        reactants = rdkit_rxn.GetReactants()

        fps = []
        for centers, reactant in zip(rxncenters, reactants):
            for index in centers:
                reactantfp = Chem.RDKFingerprint(
                    reactant,
                    minPath=1,
                    maxPath=3,
                    fpSize=self.numbits,
                    fromAtoms=[index],
                )
                array = np.zeros((0,), dtype=np.int8)
                DataStructs.ConvertToNumpyArray(reactantfp, array)
                fps.append(array)
        return fps


def reaction_center_similarity(
    fingerprints1: List[List[float]], fingerprints2: List[List[float]]
) -> float:
    """
    Calculate the maximum similarity between two sets of reaction center fingerprints

    :params fingerprints1: the first set of center fingerprints
    :params fingerprints2: the second set of center fingerprints
    :return: the maximum Jaccard distance
    """
    if len(fingerprints1) == 0 or len(fingerprints1) != len(fingerprints2):
        return 0.0

    indices = list(range(len(fingerprints1)))
    similarities = 1 - cdist(fingerprints1, fingerprints2, "jaccard")
    max_value = 0
    for ind_perm in permutations(indices):
        dist = similarities[indices, ind_perm].sum()
        max_value = max(dist, max_value)
    return max_value / len(fingerprints1)
