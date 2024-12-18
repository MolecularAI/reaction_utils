""" Module containing utilities for TED calculations """

from __future__ import annotations

import random
from enum import Enum
from operator import itemgetter
from typing import Any, Callable, Dict, List

import numpy as np
from apted import Config as BaseAptedConfig
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy.spatial.distance import jaccard as jaccard_dist

StrDict = Dict[str, Any]


class TreeContent(str, Enum):
    """Possibilities for distance calculations on reaction trees"""

    MOLECULES = "molecules"
    REACTIONS = "reactions"
    BOTH = "both"


class AptedConfig(BaseAptedConfig):
    """
    This is a helper class for the tree edit distance
    calculation. It defines how the substitution
    cost is calculated and how to obtain children nodes.

    :param randomize: if True, the children will be shuffled
    :param sort_children: if True, the children will be sorted
    :param dist_func: the distance function used for renaming nodes, Jaccard by default
    """

    def __init__(
        self,
        randomize: bool = False,
        sort_children: bool = False,
        dist_func: Callable[[np.ndarray, np.ndarray], float] = None,
    ) -> None:
        super().__init__()
        self._randomize = randomize
        self._sort_children = sort_children
        self._dist_func = dist_func or jaccard_dist

    def rename(self, node1: StrDict, node2: StrDict) -> float:
        if node1["type"] != node2["type"]:
            return 1

        fp1 = node1["fingerprint"]
        fp2 = node2["fingerprint"]
        return self._dist_func(fp1, fp2)

    def children(self, node: StrDict) -> List[StrDict]:
        if self._sort_children:
            return sorted(node["children"], key=itemgetter("sort_key"))
        if not self._randomize:
            return node["children"]
        children = list(node["children"])
        random.shuffle(children)
        return children


class StandardFingerprintFactory:
    """
    Calculate Morgan fingerprint for molecules, and difference fingerprints for reactions

    :param radius: the radius of the fingerprint
    :param nbits: the fingerprint lengths
    """

    def __init__(self, radius: int = 2, nbits: int = 2048) -> None:
        self._fp_params = (radius, nbits)

    def __call__(self, tree: StrDict, parent: StrDict = None) -> None:
        if tree["type"] == "reaction":
            if parent is None:
                raise ValueError("Must specify parent when making Morgan fingerprints for reaction nodes")
            self._add_rxn_fingerprint(tree, parent)
        else:
            self._add_mol_fingerprints(tree)

    def _add_mol_fingerprints(self, tree: StrDict) -> None:
        if "fingerprint" not in tree:
            mol = Chem.MolFromSmiles(tree["smiles"])
            rd_fp = AllChem.GetMorganFingerprintAsBitVect(mol, *self._fp_params)
            tree["fingerprint"] = np.zeros((1,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(rd_fp, tree["fingerprint"])
        tree["sort_key"] = "".join(f"{digit}" for digit in tree["fingerprint"])
        if "children" not in tree:
            tree["children"] = []

        for child in tree["children"]:
            for grandchild in child["children"]:
                self._add_mol_fingerprints(grandchild)

    def _add_rxn_fingerprint(self, node: StrDict, parent: StrDict) -> None:
        if "fingerprint" not in node:
            node["fingerprint"] = parent["fingerprint"].copy()
            for reactant in node["children"]:
                node["fingerprint"] -= reactant["fingerprint"]
        node["sort_key"] = "".join(f"{digit}" for digit in node["fingerprint"])

        for child in node["children"]:
            for grandchild in child.get("children", []):
                self._add_rxn_fingerprint(grandchild, child)
