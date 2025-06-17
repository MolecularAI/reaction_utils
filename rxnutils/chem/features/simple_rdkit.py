""" Calculates RDKit fingerprints for reactions
"""

import sys
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import RDKFingerprint

from rxnutils.chem.reaction import ChemicalReaction


@dataclass
class SimpleRdkitFingerprint:
    """
    Reaction featurizer based on a native RDKit functions

    The featurizer is used by calling it with a reaction SMILES and
    a list of float constituting the fingerprints

    :params featurizer: the type of featurizer
    :params numbits: length of fingerprint
    :params product_bits: the number of bits for the product
    """

    featurizer: str
    numbits: int = 2048
    product_bits: int = 0

    def __call__(self, reaction: ChemicalReaction) -> Optional[List[float]]:
        product_smi = reaction.products_smiles
        reactants = [Chem.MolFromSmiles(smiles) for smiles in reaction.reactants_list]

        product = Chem.MolFromSmiles(product_smi)
        if product is None:
            return None

        featurizer = getattr(sys.modules[self.__module__], self.featurizer)
        fp = featurizer(product, self.numbits)
        for reactant in reactants:
            if reactant is not None:
                fp -= featurizer(reactant, self.numbits)

        if self.product_bits > 0:
            product_fp = featurizer(product, self.product_bits)
            fp = np.concatenate([fp, product_fp])

        return fp.tolist()


def fingerprint_mixed(mol: Chem.rdchem.Mol, numbits: int) -> np.ndarray:
    """
    Calculates a mixed fingerprint first half being an ECFP3
    and the second half being an RDKit fingerprint

    :params mol: the molecule
    :params numbits: the length of the finger fingerprint
    """
    numbits = numbits // 2

    array1 = fingerprint_ecfp(mol, numbits)

    array2 = np.zeros((0,), dtype=np.int8)
    fp2 = RDKFingerprint(
        mol,
        minPath=1,
        maxPath=7,
        fpSize=numbits,
        useHs=True,
        branchedPaths=True,
        useBondOrder=True,
    )
    DataStructs.ConvertToNumpyArray(fp2, array2)

    return np.concatenate([array1, array2])


def fingerprint_ecfp(mol: Chem.rdchem.Mol, numbits: int) -> np.ndarray:
    """
    Calculates an ECFP6

    :params mol: the molecule
    :params numbits: the length of the fingal fingerprint
    """
    array = np.zeros((0,), dtype=np.int8)
    fingerprint = AllChem.GetHashedMorganFingerprint(mol, 3, nBits=numbits)
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    return array
