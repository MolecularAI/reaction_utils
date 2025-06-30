""" Module to attach protection groups to amino acids
"""

import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize as MolStandardize

from rxnutils.chem.smartslib import SmartsLibrary, SmartsHit


_UNCHARGER = MolStandardize.Uncharger()


def preprocess_amino_acids(original_smiles: str) -> str:
    """
    Remove all charges of the amino acids so that the protection
    reaction can be applied.
    """
    mol = AllChem.MolFromSmiles(original_smiles)
    uncharged_mol = _UNCHARGER.uncharge(mol)
    uncharged_smiles = AllChem.MolToSmiles(uncharged_mol, isomericSmiles=True)
    return remove_backbone_charges(uncharged_smiles)


def remove_backbone_charges(original_smiles: str) -> str:
    """
    Remove the backbone charges of the amino acid, this is
    needed for the protection chemistry.
    """
    modified_smiles = original_smiles.replace("[O-]", "O")
    return modified_smiles


@dataclass
class ProtectedAminoAcid:
    """
    An output from the protection strategy on an amino acid

    :param smiles: the protected amino acid SMILES
    :param protection_groups: the protection groups that was used
                              to generate the amino acid
    """

    smiles: str
    protection_groups: Tuple[str, ...]


class AminoAcidProtectionEngine:
    """
    Engine for protecting amino acids

    The file with protection reactions, specified with `reaction_rules_path` should
    have the following columns:
        * functional_group - the functional group that can be protected by this reaction
        * protection_group - the name of the protection group
        * primary_reaction - the first reaction SMARTS to apply
        * secondary_reaction - a second, optional reaction SMARTS to apply

    The file with the protection groups, specified with `protection_groups_path`should
    have the following columns:
        * name - the name of the protection group
        * smiles - the SMILES of the group
        * smarts - the SMARTS of the group

    :param smartslib_path: the path to a file with SMARTS patterns for functional group
    :param reaction_rules_path: the path to the file with the protection reaction SMARTS
    :param protection_groups: the path to the file with SMILES for protection groups.
    """

    def __init__(
        self, smartslib_path: str, reaction_rules_path: str, protection_groups_path: str
    ) -> None:
        self._smarts_lib = SmartsLibrary(smartslib_path)

        # Reads the protection groups and make a mapping from
        # name of protection group to the RdKit Mol object the group
        df_protection_groups = pd.read_csv(protection_groups_path, sep="\t")
        mols = [AllChem.MolFromSmarts(smiles) for smiles in df_protection_groups.smiles]
        self._protection2mol = dict(zip(df_protection_groups.name, mols))

        # Reads the protection rules and make a mapping from
        # functional group to a list of dictionaries where each
        # dictionary represent a protection group that can be attached
        # to the functional group and the RdKit reaction objects to do this
        df_protection_rules = pd.read_csv(reaction_rules_path, sep="\t")
        rxns1 = [
            AllChem.ReactionFromSmarts(smarts)
            for smarts in df_protection_rules.primary_reaction
        ]
        rxns2 = [
            AllChem.ReactionFromSmarts(smarts) if not pd.isna(smarts) else None
            for smarts in df_protection_rules.secondary_reaction
        ]
        df_protection_rules = df_protection_rules.assign(
            primary_rxn=rxns1, secondary_rxn=rxns2
        )
        self._func2reaction_data = (
            df_protection_rules.groupby("functional_group")
            .apply(lambda df: df.to_dict("records"))
            .to_dict()
        )
        # Keep a set of all unique functional groups for easy use
        self._available_funcs = set(df_protection_rules.functional_group)

    def __call__(self, smiles: str) -> List[ProtectedAminoAcid]:
        return self.protect(smiles)

    def protect(self, smiles: str) -> List[ProtectedAminoAcid]:
        """
        Protect the given SMILES.

        The SMILES need to have neutralized backbone charges

        :param smiles: the smiles of an amino acid
        :return: a list of possible protected amino acids
        """
        # Construct a list of unique functional groups to protect
        smarts_hits = self._smarts_lib.match_smarts(smiles)
        smarts_hits = {
            func: hit
            for func, hit in smarts_hits.items()
            if func in self._available_funcs
        }
        unique_hits = _remove_functional_group_redundancies(smarts_hits)

        # Extract possible protection strategies for each functional group
        possible_strategies = [
            self._func2reaction_data[func] for func in unique_hits.keys()
        ]
        # This is how many of a particular functional group that needs to
        # be protected, i.e. how many times to attach the protection group
        napplications = [len(atoms) for atoms in unique_hits.values()]

        # Now iterate over all possible combination of protection groups
        # for each functional group
        results = []
        for protection_strategy in itertools.product(*possible_strategies):
            protected_amino_acid = self._apply_protection_strategy(
                smiles, protection_strategy, napplications
            )
            results.append(protected_amino_acid)
        return results

    def _apply_protection_strategy(
        self,
        smiles,
        protection_strategy: List[Dict[str, Any]],
        napplications: List[int],
    ) -> ProtectedAminoAcid:
        """Apply one possible protection group for each functional group"""
        protected_smiles = smiles
        protection_groups = []
        for protection_data, nruns in zip(protection_strategy, napplications):
            protection_groups.append(protection_data["protection_group"])
            protected_smiles = self._attach_protection_group(
                protected_smiles, nruns, protection_data
            )
        return ProtectedAminoAcid(
            smiles=protected_smiles, protection_groups=tuple(protection_groups)
        )

    def _attach_protection_group(
        self, protected_smiles: str, napplications: int, protection_data: Dict[str, Any]
    ) -> str:
        """Attach one possible protection group `napplication` times on a functional group"""
        protect_reactant = self._protection2mol[protection_data["protection_group"]]
        for _ in range(napplications):
            reactants = (AllChem.MolFromSmiles(protected_smiles), protect_reactant)
            products = protection_data["primary_rxn"].RunReactants(reactants)
            if not products:
                continue
            protected_smiles = AllChem.MolToSmiles(products[0][0])
            if not protection_data["secondary_rxn"]:
                continue
            reactant = AllChem.MolFromSmiles(protected_smiles)
            products = protection_data["secondary_rxn"].RunReactant(reactant, 0)
            if products:
                protected_smiles = AllChem.MolToSmiles(products[0][0])
        return protected_smiles


# pylint: disable=too-many-locals
def _remove_functional_group_redundancies(
    smarts_hits: Dict[str, SmartsHit]
) -> Dict[str, Set[int]]:
    reactive_funcs = list(smarts_hits.keys())
    # Put AmineConjugatedSecondary at the end if it is in the list
    try:
        reactive_funcs.remove("AmineConjugatedSecondary")
    except ValueError:
        pass
    else:
        reactive_funcs.append("AmineConjugatedSecondary")

    highlights = [
        {atoms[0] for atoms in smarts_hits[func].match} for func in reactive_funcs
    ]

    # Keep only the first unique hit for each match
    unique_highlights = []
    indices = []
    reactive_funcs2 = []
    for idx, (atoms, func) in enumerate(zip(highlights, reactive_funcs)):
        if atoms not in unique_highlights:
            unique_highlights.append(atoms)
            indices.append(idx)
            reactive_funcs2.append(func)
    reactive_funcs = reactive_funcs2

    # Sort reactive functions and unique atom indices by alphabetical order of reactive function
    sort_indices = sorted(range(len(reactive_funcs)), key=reactive_funcs.__getitem__)
    reactive_funcs.sort()
    unique_highlights = [unique_highlights[idx] for idx in sort_indices]

    unique_hits = dict(zip(reactive_funcs, unique_highlights))

    # Remove sidechain atoms from backbone atoms
    all_set = set(unique_hits.keys())
    if {"AcidAliphatic", "AcidAliphaticAlphaCarbon"} <= all_set:
        sidechain_atoms = unique_hits["AcidAliphatic"]
        backbone_atoms = unique_hits["AcidAliphaticAlphaCarbon"]
        backbone_atoms = set(backbone_atoms).difference(set(sidechain_atoms))
        unique_hits["AcidAliphaticAlphaCarbon"] = backbone_atoms

    # Remove atom of a specific group from a general group
    set1 = {"AmineConjugatedSecondary", "AmineHetero6AromaticSecondary"}
    set2 = {"AmineConjugatedSecondary", "AminePhenylSecondary"}
    if set1 <= all_set or set2 <= all_set:
        specific_func = (
            "AmineHetero6AromaticSecondary"
            if "AmineHetero6AromaticSecondary" in unique_hits
            else "AminePhenylSecondary"
        )
        general_atoms = unique_hits["AmineConjugatedSecondary"]
        specific_atoms = unique_hits[specific_func]

        difference = list(set(specific_atoms).difference(set(general_atoms)))
        if not difference:
            unique_hits.pop(specific_func)
        else:
            unique_hits[specific_func] = difference
    return unique_hits
