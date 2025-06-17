""" Module containing routines to deal with a SMARTS library of chemical functions
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem

from rxnutils.chem.reaction import ChemicalReaction
from rxnutils.chem.utils import join_smiles_from_reaction


@dataclass
class SmartsHit:
    """
    A hit/match for a substructure match

    :param number_of_hits: the number of times the SMARTS match
    :param match: the result from the substructure match, i.e. a tuple of each atom match
    :param atoms: the flattened list of atoms matching the smarts
    """

    number_of_hits: int
    match: Tuple[Tuple[int, ...]]
    atoms: List[int]
    smarts_mol: Chem.rdchem.Mol


class SmartsLibrary:
    """
    Class for manipulating and working with a library of SMARTS
    defining chemical functions.

    The created instances of the class are stored in an internal
    storage dictionary for easy retrieval.

    The library supports reading two kinds of file:

    1. a JSON-file with a dictionary with pattern names as keys and
       SMARTS strings as values, e.g.

        {
            "ARC": "[c]",
            "ARN3": "[n;X3]"
        }

    2. An "ontology-file" that should have 6 tab-separated columns for each pattern
        * The super-group name of the SMARTS
        * The group name of the SMARTS
        * The name of the SMARTS
        * The more generic version of the SMARTS
        * The more specific version of the SMARTS
        * The type of SMARTS, i.e. if it can be used for clipping or labeling

        For an example, look at the tests/data/simple_smartslib.txt file in the repository

    :params filename: the path of the library on disc or an environmental variable pointing to a path
    :params label: the label for the library in the internal storage, if not given will use basename
    """

    _STORE: Dict[str, SmartsLibrary] = {}

    def __init__(self, filename: str, label: Optional[str] = None) -> None:
        self.names: List[str] = []
        self.smarts_objects: Dict[str, Chem.rdchem.Mol] = {}
        self.general_smarts_objects: Dict[str, Chem.rdchem.Mol] = {}
        self.type_: Dict[str, str] = {}
        self.name_aliases: Dict[str, Any] = {}
        self._rank: Dict[str, int] = {}

        if filename.startswith("$"):
            filename = os.environ[filename[2:-1]]

        if filename.endswith(".json"):
            self._load_smarts_json(filename)
        else:
            self._load_smarts_ontology(filename)

        label = label or os.path.basename(filename)
        self._STORE[label] = self

    @classmethod
    def load(cls, source: str) -> "SmartsLibrary":
        """
        Loads a SMARTS library either from the storage
        if `source` exists as a label or creates a new
        library using `source` as the path

        :params source: either the label or a path of a SMARTS library
        :returns: the loaded library
        """
        try:
            return cls._STORE[source]
        except KeyError:
            return SmartsLibrary(source)

    def detect_reactive_functions(  # pylint: disable=[R0913, R0917, R0914, R1702, R0912]
        self,
        reaction: ChemicalReaction,
        sort: bool = False,
        add_none: bool = True,
        none_str: str = "None",
        target_size: Optional[int] = 2,
        max_reactants: Optional[int] = None,
        product_functions: Optional[Dict[str, SmartsHit]] = None,
        reactants_functions: Optional[List[Dict[str, SmartsHit]]] = None,
    ) -> Tuple[Tuple[str, ...], str]:
        """
        Find the reactive functions in the reactants, i.e. the functional
        groups that changes during the reaction

        :param reaction: the reaction to manipulate
        :param sort: if True will sort the reactants based on the alphabetical order of the reactive functions
        :param add_none: if True will add the `none_str` for reactants without any hits
        :param none_str: the placeholder string for the non-matching reactants
        :param target_size: the desired length of the returned lists when adding placeholders
        :param max_reactants: if number of reactant larger than this, it will raise an exception
        :param product_functions: Dictionary of pre-identified SMARTS hits for the products
        :param reactants_functions: List of dictionaries of pre-identified SMARTS hits for each reactant
        :return: tuple of reactive function and reaction SMILES

        :raises ValueError: when there is no reactants or no product or number of reactant are larger than limit
        """
        self._validate_reaction(reaction, max_reactants)

        product_functions = product_functions or self.match_smarts(
            reaction.products_smiles
        )
        product_hits = {
            name: hit.number_of_hits for name, hit in product_functions.items()
        }

        reactants_functions = reactants_functions or [
            self.match_smarts(reactant) for reactant in reaction.reactants_list
        ]

        reactive_functions = self._find_reactive_functions(
            reaction, reactants_functions, product_hits, none_str
        )

        reactive_functions, reactants_ordered = self._adjust_reactive_functions(
            reactive_functions,
            list(reaction.reactants_list),
            target_size,
            none_str,
            sort,
            add_none,
        )

        reactants = join_smiles_from_reaction([smi for smi in reactants_ordered if smi])
        rsmi_new = ">".join(
            [
                reactants,
                reaction.agents_smiles,
                reaction.products_smiles,
            ]
        )
        return tuple(reactive_functions), rsmi_new

    def match_smarts(self, smiles: str) -> Dict[str, SmartsHit]:
        """
        Check the SMARTS library for hits, i.e. substructure matches
        in the given SMILES

        :param smiles: the query SMILES
        :return: dictionary of SMARTS hits
        """
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return {}

        match_list: Dict[str, SmartsHit] = {}
        for name, smarts in self.smarts_objects.items():
            hit_atoms = []
            match = mol.GetSubstructMatches(smarts)
            if match:
                hit_atoms = [element for tupl in match for element in tupl]
                match_list[name] = SmartsHit(
                    number_of_hits=len(match),
                    match=match,
                    atoms=hit_atoms,
                    smarts_mol=smarts,
                )

        return match_list

    def _load_smarts_json(self, filename: str) -> None:
        with open(filename, "r") as fileobj:
            dict_ = json.load(fileobj)

        for idx, (func_name, smarts_str) in enumerate(dict_.items()):
            self.smarts_objects[func_name] = Chem.MolFromSmarts(smarts_str)
            self.names.append(func_name)
            self._rank[func_name] = idx

    def _load_smarts_ontology(self, filename: str) -> None:
        with open(filename, "r") as fileobj:
            lines = fileobj.read().splitlines()

        idx = 0
        for line in lines:
            if not line or line.startswith("#"):
                continue
            (
                func_supergroup,
                func_group,
                func_name,
                general_smarts,
                specific_smarts,
                _type,
            ) = line.split("\t")[0:6]

            self.name_aliases[func_name] = [func_name, func_group, func_supergroup]
            self.type_[func_name] = _type
            self.general_smarts_objects[func_name] = Chem.MolFromSmarts(general_smarts)
            self.smarts_objects[func_name] = Chem.MolFromSmarts(specific_smarts)
            self.names.append(func_name)
            self._rank[func_name] = idx
            idx += 1

    def _sort_key(self, sort_tuple: Tuple[str, str]) -> int:
        reactive_function, _ = sort_tuple
        default_val = len(self.names) + 1
        return self._rank.get(reactive_function, default_val)

    # Helper functions to detect_reactive_function

    def _validate_reaction(
        self, reaction: ChemicalReaction, max_reactants: Optional[int]
    ) -> None:
        """
        Check validity of reaction.
        :raises: ValueError for missing product or reactants, and if the number of reactants exceed
        max_reactants.
        """
        if not reaction.products_smiles or not reaction.reactants_list:
            raise ValueError(
                f"Cannot detect reactive functions on SMILES lacking either reactant or product: {reaction.rsmi}"
            )
        if max_reactants is not None and len(reaction.reactants_list) > max_reactants:
            raise ValueError(
                f"Too many reactants in SMILES {len(reaction.reactants_list)}: {reaction.rsmi}"
            )

    def _find_reactive_functions(
        self,
        reaction: ChemicalReaction,
        reactants_functions: List[Dict[str, SmartsHit]],
        product_hits: Dict[str, int],
        none_str: str,
    ) -> List[str]:
        """
        Find reactive function of the reaction given the reactant functions and product
        functions.

        :param reaction: Chemical reaction object.
        :param reactants_functions: Reactive functions in the reactants.
        :param product_hits: Mapping product functions to number of SMARTS hits.
        :param none_str: the placeholder string for the non-matching reactants
        :return: List of reactive function names (or None)
        """
        reactive_functions = []
        for reactant_idx, reactant_functions in enumerate(reactants_functions):
            found = self._find_function_by_count(reactant_functions, product_hits)
            if not found:
                found = self._find_function_by_subtraction(
                    reactant_functions,
                    dict(product_hits),
                    reactant_idx,
                    reaction.reactants_list,
                )
            reactive_functions.append(found or none_str)
        return reactive_functions

    def _find_function_by_count(
        self, reactant_functions: Dict[str, SmartsHit], product_hits: Dict[str, int]
    ) -> Optional[str]:
        """
        Try to find the reactive function by just comparing the counts in the product,
        this fails if another reactant also have the reactive function but it remains
        intact during the reaction.
        :param reactants_functions: SMARTS hits for each reactant.
        :param product_hits: Mapping product functions to number of SMARTS hits.
        :return: Reactive function name (or None)
        """
        for name, hit in reactant_functions.items():
            if hit.number_of_hits > product_hits.get(name, 0):
                return name
        return None

    def _find_function_by_subtraction(
        self,
        reactant_functions: Dict[str, SmartsHit],
        product_hits: Dict[str, int],
        reactant_idx: int,
        reactants_list: List[str],
    ) -> Optional[str]:
        """
        Try to find the reactive function by subtracting the counts of the reactive
        functions of the other reactants.
        :param reactants_functions: SMARTS hits for each reactant.
        :param product_hits: Mapping product functions to number of SMARTS hits.
        :param reactant_idx: Index of reactant in list of reactants.
        :param reactants_list: Reactant molecules in the reaction.
        :return: Reactive function name (or None)
        """
        for name1, hit1 in reactant_functions.items():
            if name1 not in product_hits:
                continue

            for reactant_idx2, reactant2 in enumerate(reactants_list):
                if reactant_idx == reactant_idx2:
                    continue

                reactant_functions2 = self.match_smarts(reactant2)
                product_hits[name1] -= sum(
                    matched_hit.number_of_hits
                    for name2, matched_hit in reactant_functions2.items()
                    if name1 == name2
                )

            if hit1.number_of_hits > product_hits.get(name1, 0):
                return name1

        return None

    def _adjust_reactive_functions(
        self,
        reactive_functions: List[str],
        reactants_ordered: List[str],
        target_size: Optional[int],
        none_str: str,
        sort: bool,
        add_none: bool,
    ) -> Tuple[List[str], List[str]]:
        """
        Modify the reactive functions and reactants:
        Sort reactive functions and reactants, add 'none_str' as padding.
        :param reactive_functions: List of reactive function names.
        :param reactants_ordered: List of reactant SMILES.
        :param target_size: the desired length of the returned lists when adding placeholders
        :param none_str: the placeholder string for the non-matching reactants
        :param sort: if True will sort the reactants based on the alphabetical order of the reactive functions
        :param add_none: if True will add the `none_str` for reactants without any hits
        :return: Updated reactive functions and reactants
        """
        diff = (
            target_size - len(reactive_functions) if target_size is not None else None
        )
        if diff is not None and diff > 0:
            reactive_functions.extend([none_str] * diff)
            reactants_ordered.extend([""] * diff)

        if sort:
            reactive_functions, reactants_ordered = zip(  # type: ignore[assignment]
                *sorted(zip(reactive_functions, reactants_ordered), key=self._sort_key)
            )

        # None:s should only be removed after sorting
        if not add_none:
            reactive_functions_tmp = []
            reactants_ordered_tmp = []
            for name, reactant in zip(reactive_functions, reactants_ordered):
                if name != none_str:
                    reactive_functions_tmp.append(name)
                    reactants_ordered_tmp.append(reactant)
            reactive_functions, reactants_ordered = (
                reactive_functions_tmp,
                reactants_ordered_tmp,
            )

        return list(reactive_functions), list(reactants_ordered)
