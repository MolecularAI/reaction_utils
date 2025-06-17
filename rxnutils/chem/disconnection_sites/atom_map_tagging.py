from __future__ import annotations

import argparse
from collections import OrderedDict
from typing import List, Optional, Sequence

import pandas as pd
from rdkit import Chem

from rxnutils.chem.utils import split_rsmi
from rxnutils.chem.disconnection_sites.tag_converting import smiles_tokens


def _get_atom_identifier(atom: Chem.rdchem.Atom) -> str:
    """
    Get atom identifier for neighborhood identification.
    The identifier is either the atom-map number if available, otherwise the symbol.
    :param atom: rdkit atom
    :return: an atom identifier string
    """
    atom_id = atom.GetAtomMapNum()
    if atom_id == 0:
        atom_id = atom.GetSymbol()
    return str(atom_id)


def _get_bond_environment_identifier(
    atoms: Sequence[Chem.rdchem.Atom], bond: Chem.rdchem.Bond
) -> str:
    """
    Get the environment of a specific bond.

    :param atoms: atoms in the molecule.
    :param bond: bond for which the environment should be specified
    :return: string representation of the bond environment
    """
    atom_map1 = _get_atom_identifier(atoms[bond.GetBeginAtomIdx()])
    atom_map2 = _get_atom_identifier(atoms[bond.GetEndAtomIdx()])
    bond_order = bond.GetBondType()
    atom_map1, atom_map2 = sorted([atom_map1, atom_map2])
    return f"{atom_map1}_{atom_map2}_{bond_order}"


def _get_atomic_neighborhoods(smiles: str) -> OrderedDict[int, List[str]]:
    """
    Obtains a dictionary containing each atom (atomIdx) and a list of its
    bonding environment.

    :param smiles: Atom-mapped SMILES string
    :return: A dictionary containing each atom (atomIdx) and a list of its
        bonding environment identifiers.
    """

    mol = Chem.MolFromSmiles(smiles)
    atoms = mol.GetAtoms()

    neighbor_dict = {}
    for atom in atoms:
        bonds_list = []
        if atom.GetAtomMapNum() != 0:
            for bond in atom.GetBonds():

                bonds_list.append(_get_bond_environment_identifier(atoms, bond))

            neighbor_dict[atom.GetAtomMapNum()] = sorted(bonds_list)
    ordered_neighbor_dict = OrderedDict(sorted(neighbor_dict.items()))

    return ordered_neighbor_dict


def _strip_tokens(smiles: str) -> str:
    """Remove brackets around single letter"""
    tokens = smiles_tokens(smiles)
    output_smiles = ""
    for token in tokens:
        if len(token) == 3 and token.startswith("[") and token.endswith("]"):
            token = token[1]
        output_smiles += token
    return output_smiles


def get_atom_list(reactants_smiles: str, product_smiles: str) -> List[int]:
    """
    Given two sets of SMILES strings corresponding to a set of reactants and products,
    returns a list of atomIdxs for which the atomic environment has changed,
    as defined by a change in the bonds.

    :param reactants_smiles: Atom-mapped SMILES string for the reactant(s)
    :param product_smiles: Atom-mapped SMILES string for the product(s)
    :return: List of atoms (atom-map-nums) for which the atomic environment has changed
    """

    ordered_reactant_neighbor_dict = _get_atomic_neighborhoods(reactants_smiles)
    ordered_product_neighbor_dict = _get_atomic_neighborhoods(product_smiles)

    all_indices = set(ordered_product_neighbor_dict.keys()) | set(
        ordered_reactant_neighbor_dict.keys()
    )

    # Checks to see equivlence of atomic enviroments.
    # If environment changed, then add atom to list
    atom_list = [
        atom_map
        for atom_map in all_indices
        if ordered_reactant_neighbor_dict.get(atom_map, [])
        != ordered_product_neighbor_dict.get(atom_map, [])
    ]

    return atom_list


def atom_map_tag_site(mapped_smiles: str, atom_maps: List[int]) -> str:
    """
    Remove atom-tagging on all atoms except those in atom_maps.
    Tag atom_maps atoms as [<atom>:1] where <atom> is the atom.

    :param mapped_smiles: SMILES with atom-mapping
    :param atom_maps: atom-map nums in site that will be tagged
    :return: atom-map tagged SMILES
    """
    mol = Chem.MolFromSmiles(mapped_smiles)

    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() in atom_maps:
            atom.SetAtomMapNum(1)
        else:
            atom.SetAtomMapNum(0)

    smiles = Chem.MolToSmiles(mol)
    return _strip_tokens(smiles)


def atom_map_tag_reactants(mapped_rxn: str) -> str:
    """
    Given atom-mapped reaction, returns disconnection site-tagged reactants where atoms
    with changed atom environment are represented by [<atom>:1].

    :param mapped_rxn: Atom-mapped reaction SMILES
    :return: SMILES of the reactants containing tags corresponding to atoms changed in the
        reaction.
    """
    reactants_smiles, _, product_smiles = split_rsmi(mapped_rxn)

    reactants_mol = Chem.MolFromSmiles(reactants_smiles)
    atom_list = get_atom_list(reactants_smiles, product_smiles)

    # Set atoms in product with a different environment in reactants to 1
    for atom in reactants_mol.GetAtoms():
        if atom.GetAtomMapNum() in atom_list:
            atom.SetAtomMapNum(1)
        else:
            atom.SetAtomMapNum(0)

    return Chem.MolToSmiles(reactants_mol)


def atom_map_tag_products(mapped_rxn: str) -> str:
    """
    Given atom-mapped reaction, returns disconnection site-tagged product where atoms
    with changed atom environment are represented by [<atom>:1].

    :param mapped_rxn: Atom-mapped reaction SMILES
    :return: SMILES of the product containing tags corresponding to atoms changed in the
        reaction.
    """
    reactants_smiles, _, product_smiles = split_rsmi(mapped_rxn)

    product_mol = Chem.MolFromSmiles(product_smiles)
    atom_list = get_atom_list(reactants_smiles, product_smiles)

    # Set atoms in product with a different environment in reactants to 1
    for atom in product_mol.GetAtoms():
        if atom.GetAtomMapNum() in atom_list:
            atom.SetAtomMapNum(1)
        else:
            atom.SetAtomMapNum(0)

    return Chem.MolToSmiles(product_mol)


def main(args: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--in_column", default="RxnSmilesClean")
    parser.add_argument("--out_column", default="products_atom_map_tagged")
    parser.add_argument("--output")

    args = parser.parse_args(args)

    data = pd.read_csv(args.input, sep="\t")

    smiles_col = data[args.in_column].apply(atom_map_tag_products)
    data = data.assign(**{args.out_column: smiles_col})
    data.to_csv(args.output, sep="\t", index=False)


if __name__ == "__main__":
    main()
