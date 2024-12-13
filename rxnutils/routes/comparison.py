""" Contains routines for computing route similarities
"""

import functools
from typing import Any, Callable, Dict, List, Sequence, Set, Tuple

import numpy as np

from rxnutils.chem.reaction import ChemicalReaction
from rxnutils.chem.utils import atom_mapping_numbers, split_rsmi
from rxnutils.routes.base import SynthesisRoute
from rxnutils.routes.ted.distances_calculator import ted_distances_calculator

RouteDistancesCalculator = Callable[[Sequence[SynthesisRoute]], np.ndarray]


def simple_route_similarity(routes: Sequence[SynthesisRoute]) -> np.ndarray:
    """
    Returns the geometric mean of the simple bond forming similarity, and
    the atom matching bonanza similarity

    :param routes: the sequence of routes to compare
    :return: the pairwise similarity
    """
    bond_score = simple_bond_forming_similarity(routes)
    atom_score = atom_matching_bonanza_similarity(routes)
    return np.sqrt(bond_score * atom_score)


def atom_matching_bonanza_similarity(routes: Sequence[SynthesisRoute]) -> np.ndarray:
    """
    Calculates the pairwise similarity of a sequence of routes
    based on the overlap of the atom-mapping numbers of the compounds
    in the routes.

    :param routes: the sequence of routes to compare
    :return: the pairwise similarity
    """
    sims = np.zeros((len(routes), len(routes)))
    for idx1, route1 in enumerate(routes):
        mols1 = _extract_atom_mapping_numbers(route1)
        n1 = len(mols1) - 1
        mols1 = [mol for mol in mols1 if mol]
        for idx2, route2 in enumerate(routes):
            if idx1 == idx2:
                sims[idx1, idx2] = 1.0
                continue
            # If both of the routes have not reactions, we assume
            # they are identical and it is assignes a score of 1.0
            if route1.max_depth == 0 and route2.max_depth == 0:
                sims[idx1, idx2] = 1.0
                continue
            # If just one the route has no reactions, we assume
            # they are maximal dissimilar and assigns a score 0.0 (see how `sims` is initialized)
            if route1.max_depth == 0 or route2.max_depth == 0:
                continue

            mols2 = _extract_atom_mapping_numbers(route2)
            n2 = len(mols2) - 1
            mols2 = [mol for mol in mols2 if mol]

            o = _calc_overlap_matrix(mols1[1:], mols2[1:])
            sims[idx1, idx2] = (o.max(axis=1).sum() + o.max(axis=0).sum()) / (n1 + n2)
    return sims


def simple_bond_forming_similarity(routes: Sequence[SynthesisRoute]) -> np.ndarray:
    """
    Calculates the pairwise similarity of a sequence of routes
    based on the overlap of formed bonds in the reactions.

    :param routes: the sequence of routes to compare
    :return: the pairwise similarity
    """
    sims = np.ones((len(routes), len(routes)))
    for idx1, route1 in enumerate(routes):
        bonds1 = _extract_formed_bonds(route1)
        for idx2, route2 in enumerate(routes):
            if idx2 <= idx1:
                continue
            # If both of the routes have not reactions, we assume
            # they are identical and it is assignes a score of 1.0 (see how `sims` is initialized)
            if route1.max_depth == 0 and route2.max_depth == 0:
                continue
            # If just one the route has no reactions, we assume
            # they are maximal dissimilar and assigns a score 0.0
            if route1.max_depth == 0 or route2.max_depth == 0:
                sims[idx1, idx2] = sims[idx2, idx1] = 0.0
                continue

            bonds2 = _extract_formed_bonds(route2)
            sims[idx1, idx2] = _bond_formed_overlap_score(bonds1, bonds2)
            sims[idx2, idx1] = sims[idx1, idx2]  # Score is symmetric
    return sims


def route_distances_calculator(model: str, **kwargs: Any) -> RouteDistancesCalculator:
    """
    Return a callable that given a list routes as dictionaries
    calculate the squared distance matrix

    :param model: the route distance model name
    :param kwargs: additional keyword arguments for the model
    :return: the appropriate route distances calculator
    """
    if model not in ["ted", "lstm"]:
        raise ValueError("Model must be either 'ted' or 'lstm'")

    if model == "ted":
        model_kwargs = _copy_kwargs(["content", "timeout"], **kwargs)
        return functools.partial(ted_distances_calculator, **model_kwargs)

    # Placeholder for LSTM distances calculation
    # model_kwargs = _copy_kwargs(["model_path"], **kwargs)
    # return lstm_distances_calculator(**model_kwargs)
    raise NotImplementedError("LSTM route distances calculator not implemented yet.")


def _copy_kwargs(keys_to_copy: List[str], **kwargs: Any) -> Dict[str, Any]:
    """Copy selected keyword arguments."""
    new_kwargs = {}
    for key in keys_to_copy:
        if key in kwargs:
            new_kwargs[key] = kwargs[key]
    return new_kwargs


def _calc_overlap_matrix(mols1: List[List[int]], mols2: List[List[int]]) -> np.ndarray:
    """
    Calculate the pairwise overlap matrix between
    the molecules in the two input vectors

    :param mols1: the atom-mapping numbers of the first molecule
    :param mols2: the atom-mapping numbers of the second molecule
    :return: the computed matrix
    """

    def mol_overlap(mol1, mol2):
        if not mol1 and not mol2:
            return 0
        return len(set(mol1).intersection(mol2)) / max(len(mol1), len(mol2))

    overlaps = np.zeros((len(mols1), len(mols2)))
    for idx1, mol1 in enumerate(mols1):
        for idx2, mol2 in enumerate(mols2):
            overlaps[idx1, idx2] = mol_overlap(mol1, mol2)
    return overlaps


def _extract_atom_mapping_numbers(route: SynthesisRoute) -> List[List[int]]:
    """
    Extract all the compounds in a synthesis routes as a set of
    atom-mapping numbers of the atoms in the compounds.

    Only account for atom-mapping numbers that exists in the root compound
    """
    if route.max_depth == 0:
        return []

    root_atom_numbers = sorted(atom_mapping_numbers(route.mapped_root_smiles))
    mapping_list = [root_atom_numbers]
    for reaction_smiles in route.atom_mapped_reaction_smiles():
        reactants_smiles = split_rsmi(reaction_smiles)[0]
        for smi in reactants_smiles.split("."):
            atom_numbers = sorted([num for num in atom_mapping_numbers(smi) if num in root_atom_numbers])
            mapping_list.append(atom_numbers)
    return mapping_list


def _extract_formed_bonds(route: SynthesisRoute) -> List[Tuple[int, int]]:
    """
    Extract a set of bonds formed in the synthesis routes.
    Only bonds contributing to the root compound is accounted for.
    """
    if route.max_depth == 0:
        return []

    formed_bonds = []
    root_atom_numbers = set(atom_mapping_numbers(route.mapped_root_smiles))
    for reaction_smiles in route.atom_mapped_reaction_smiles():
        rxn_obj = ChemicalReaction(reaction_smiles, clean_smiles=False)
        product_bonds = set()
        product_bonds = _extract_molecule_bond_tuple(rxn_obj.products)
        reactants_bonds = _extract_molecule_bond_tuple(rxn_obj.reactants)
        for idx1, idx2 in product_bonds - reactants_bonds:
            if idx1 in root_atom_numbers and idx2 in root_atom_numbers:
                formed_bonds.append(tuple(sorted([idx1, idx2])))
    return formed_bonds


def _extract_molecule_bond_tuple(molecules: List) -> Set[int]:
    """
    Returns the sorted set of atom-mapping number pairs for each
    bonds in a list of rdkit molecules
    """
    bonds = set()
    for bond in [bond for mol in molecules for bond in mol.GetBonds()]:
        bonds.add(
            tuple(
                sorted(
                    [
                        bond.GetBeginAtom().GetAtomMapNum(),
                        bond.GetEndAtom().GetAtomMapNum(),
                    ]
                )
            )
        )
    return bonds


def _bond_formed_overlap_score(bonds1: List[Tuple[int]], bonds2: List[Tuple[int]]) -> float:
    """
    Computes a similarity score of two routes by comparing the overlap
    of bonds formed in the synthesis routes.
    """
    overlap = len(set(bonds1).intersection(bonds2))
    if not bonds1 and not bonds2:
        return 0
    return overlap / max(len(bonds1), len(bonds2))
