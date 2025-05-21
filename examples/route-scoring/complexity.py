""" Module containing routines to calculate molecular complexity
"""
from collections import Counter
from typing import Tuple, Dict, Any, Iterator, List

import numpy as np
from rdkit import Chem

_AtomType = Tuple[str, int, int]
_Atom = Tuple[int, _AtomType]
_AtomDict = Dict[_Atom, List[_Atom]]


def calc_cse(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Unsanitizble molecule")
    return calculate_molecular_complexity(mol)[0]


def calc_cm(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Unsanitizble molecule")
    return calculate_molecular_complexity(mol)[1]


def calc_cm_star(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Unsanitizble molecule")
    return calculate_molecular_complexity(mol)[2]


def calculate_molecular_complexity(mol: Chem.rdchem.Mol) -> Tuple[float, float, float]:
    """
    This is a function to calculate the CM and CM* molecular complexity metrics described in
    Proudfoot, Bioorganic & Medicinal Chemistry Letters 27 (2017) 2014-2017.
    https://doi.org/10.1016/j.bmcl.2017.03.008

    This function takes an rdkit mol object, identifies the connection paths 1 and 2 atoms away,
    and the calculates the complexity environment for each atom CA as

    CA = - Sum (pi*log2(pi)) + log2(N)

    where pi is the fractional occurrence of each path type emanating from a particular atom and N
    is the total number of paths emanating from that atom.

    Molecular complexity CM can be defined as either the simple sum of the CA,
    or CM* which is the log-sum of the exponentials of the CA.

    CM = Sum (CA)

    CM* = log2(Sum (2**CA))

    Cse = - Sum (qi*log2(qi))

    where qi is the fractional occurrence of an atom (or atom environment).
    """
    # get atom types for each atom in the molecule
    atoms = [(atom.GetIdx(), _get_atom_type(atom)) for atom in mol.GetAtoms()]

    # create dict with neighbors of each atom
    neighbors = {
        atom: [
            atoms[neighbor.GetIdx()]
            for neighbor in mol.GetAtomWithIdx(atom[0]).GetNeighbors()
        ]
        for atom in atoms
    }

    atom_paths = _collect_atom_paths(neighbors)

    cas = np.zeros(len(atom_paths))
    for i, paths in enumerate(atom_paths):
        total_paths = len(paths)
        pi = _fractional_occurrence(paths)
        cas[i] = -np.sum(pi * np.log2(pi)) + np.log2(total_paths)

    cm = np.sum(cas)

    cm_star = np.log2(np.sum(2 ** cas))

    # sort and concatenate the individual paths to compare the atom environments
    atom_environments = [tuple(sorted((paths))) for paths in atom_paths]

    # Now we can calculate the Cse metric as the fractional occurrence of each atom environment
    qi = _fractional_occurrence(atom_environments)
    cse = -np.sum(qi * np.log2(qi))

    return float(cse), float(cm), float(cm_star)


def _non_h_items(data: Dict[_Atom, Any]) -> Iterator[Tuple[_Atom, Any]]:
    """
    Generator for non-H items from a dictionary where the keys are atom tuples.

    Expected keys: (index, (symbol, total degree, non-h degree))
    """
    for key, val in data.items():
        if key[1][0] != "H":
            yield key, val


def _collect_atom_paths(neighbors: _AtomDict) -> List[List[Tuple[int, int, int]]]:
    """
    Returns list of atom paths for each atom.

    An atom path is a tuple of atom types.
    """
    atom_paths = []
    for atom, nbs in _non_h_items(neighbors):
        paths = []
        for nb in nbs:
            if nb[1][0] == "H" or neighbors[nb] == [atom]:
                # No second neighbors
                paths.append((atom[1], nb[1]))
            else:
                paths.extend(
                    (atom[1], nb[1], nb2[1]) for nb2 in neighbors[nb] if nb2 != atom
                )

        atom_paths.append(paths)

    return atom_paths


def _get_atom_type(atom: Chem.rdchem.Atom) -> _AtomType:
    """
    Return a tuple describing the atom type.

    Considers element, total number of connections, and number of non-H connections.
    """
    symbol = atom.GetSymbol()
    degree = atom.GetTotalDegree()
    h_count = atom.GetTotalNumHs(includeNeighbors=True)
    non_h = degree - h_count
    return (symbol, degree, non_h)


def _fractional_occurrence(data: list) -> np.ndarray:
    """
    Calculate the fractional occurrence of unique items in the input data.

    Uniqueness determined by collections.Counter.

    Returns:
        np.ndarray: fractional occurrence of unique items
    """
    counter = Counter(data)
    counts = np.array(list(counter.values()))
    return counts / len(data)
