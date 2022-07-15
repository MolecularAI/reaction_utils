"""Module containing actions on reactions that doesn't modify the reactions only compute properties of them"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import ClassVar, Tuple, List

from dataclasses import dataclass
import pandas as pd
from rdkit import RDLogger
from rdkit import Chem

from rxnutils.pipeline.base import action, global_apply, ReactionActionMixIn
from rxnutils.chem.utils import (
    has_atom_mapping,
    split_smiles_from_reaction,
    atom_mapping_numbers,
)

rd_logger = RDLogger.logger()
rd_logger.setLevel(RDLogger.CRITICAL)


@action
@dataclass
class CountComponents(ReactionActionMixIn):
    """Action for counting reaction components"""

    pretty_name: ClassVar[str] = "count_components"
    in_column: str
    nreactants_column: str = "NReactants"
    nmapped_reactants_column: str = "NMappedReactants"
    nreagents_column: str = "NReagents"
    nmapped_reagents_column: str = "NMappedReagents"
    nproducts_column: str = "NProducts"
    nmapped_products_column: str = "NMappedProducts"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        new_data = global_apply(data, self._apply_row, axis=1)
        return data.assign(**{column: new_data[column] for column in new_data.columns})

    def _apply_row(self, row: pd.Series) -> pd.Series:
        def _process_smiles(smiles_list: List[str]) -> Tuple[int, int]:
            if not smiles_list:
                return 0, 0
            smiles_list_mapped = [
                smi for smi in smiles_list if has_atom_mapping(smi, sanitize=False)
            ]
            return len(smiles_list), len(smiles_list_mapped)

        reactants_list, reagents_list, products_list = self.split_lists(row)
        nreactants, nmapped_reactants = _process_smiles(reactants_list)
        nreagents, nmapped_reagents = _process_smiles(reagents_list)
        nproducts, nmapped_products = _process_smiles(products_list)
        return pd.Series(
            {
                self.nreactants_column: nreactants,
                self.nmapped_reactants_column: nmapped_reactants,
                self.nreagents_column: nreagents,
                self.nmapped_reagents_column: nmapped_reagents,
                self.nproducts_column: nproducts,
                self.nmapped_products_column: nmapped_products,
            }
        )

    def __str__(self) -> str:
        return f"{self.pretty_name} (counting reactants, reagents, products and mapped versions of these)"


@action
@dataclass
class CountElements:
    """Action for counting elements in reactants"""

    pretty_name: ClassVar[str] = "count_elements"
    in_column: str
    out_column: str = "ElementCount"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        hash_col = global_apply(data, self._row_action, axis=1)
        return data.assign(**{self.out_column: hash_col})

    def __str__(self) -> str:
        return (
            f"{self.pretty_name} (calculate the occurence of elements in the reactants)"
        )

    def _row_action(self, row: pd.Series) -> str:
        def count_elements(smiles, counts):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return
            for atom in mol.GetAtoms():
                counts[atom.GetAtomicNum()] += 1

        counts = defaultdict(int)
        reactants, _, _ = row[self.in_column].split(">")
        for smiles in split_smiles_from_reaction(reactants):
            count_elements(smiles, counts)

        return json.dumps(counts)


@action
@dataclass
class HasStereoInfo:
    """Action for checking stereo info"""

    pretty_name: ClassVar[str] = "has_stereo_info"
    in_column: str
    out_column: str = "HasStereo"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        smiles_col = global_apply(data, self._row_action, axis=1)
        return data.assign(**{self.out_column: smiles_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (returns boolean if SMILES has stereo info (contains '@'))"

    def _row_action(self, row: pd.Series) -> str:
        smiles = row[self.in_column]
        return "@" in smiles


@action
@dataclass
class ProductAtomMappingStats:
    """Action for collecting statistics of product atom mapping"""

    pretty_name: ClassVar[str] = "product_atommapping_stats"
    in_column: str
    unmapped_column: str = "UnmappedProdAtoms"
    widow_column: str = "WidowAtoms"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        new_data = global_apply(data, self._row_action, axis=1)
        return data.assign(**{column: new_data[column] for column in new_data.columns})

    def __str__(self) -> str:
        return f"{self.pretty_name} (count number of number of unmapped and widow product atoms)"

    def _row_action(self, row: pd.Series) -> str:
        reactants, _, products = row[self.in_column].split(">")
        prod_mol = Chem.MolFromSmiles(products)
        nprod_atoms = prod_mol.GetNumAtoms() if prod_mol else 0

        prod_atommappings = set(atom_mapping_numbers(products))
        react_atommappings = []
        for smi in split_smiles_from_reaction(reactants):
            react_atommappings.extend(atom_mapping_numbers(smi))
        react_atommappings = set(react_atommappings)

        return pd.Series(
            {
                self.unmapped_column: nprod_atoms - len(prod_atommappings),
                self.widow_column: len(prod_atommappings - react_atommappings),
            }
        )


@action
@dataclass
class ProductSize:
    """Action for counting product size"""

    pretty_name: ClassVar[str] = "productsize"
    in_column: str
    out_column: str = "ProductSize"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        smiles_col = global_apply(data, self._row_action, axis=1)
        return data.assign(**{self.out_column: smiles_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (number of heavy atoms in product)"

    def _row_action(self, row: pd.Series) -> str:
        _, _, products = row[self.in_column].split(">")
        products_mol = Chem.MolFromSmiles(products)

        if products_mol:
            product_atom_count = products_mol.GetNumHeavyAtoms()
        else:
            product_atom_count = 0

        return product_atom_count


@action
@dataclass
class PseudoReactionHash:
    """Action for creating a reaction hash based on InChI keys"""

    pretty_name: ClassVar[str] = "pseudo_reaction_hash"
    in_column: str
    out_column: str = "PseudoHash"
    no_reagents: bool = False

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        hash_col = global_apply(data, self._row_action, axis=1)
        return data.assign(**{self.out_column: hash_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (calculate hash based on InChI key of components)"

    def _row_action(self, row: pd.Series) -> str:
        def components_to_inchi(smiles):
            mols = [
                Chem.MolFromSmiles(smi) for smi in split_smiles_from_reaction(smiles)
            ]
            return ".".join(sorted([Chem.MolToInchiKey(mol) for mol in mols if mol]))

        if self.no_reagents:
            reactants, _, products = row[self.in_column].split(">")
            return ">>".join(
                [components_to_inchi(smi) for smi in [reactants, products]]
            )

        return ">".join(
            components_to_inchi(smi) for smi in row[self.in_column].split(">")
        )


@action
@dataclass
class PseudoSmilesHash:
    """Action for creating a reaction hash based on SMILES"""

    pretty_name: ClassVar[str] = "pseudo_smiles_hash"
    in_column: str
    out_column: str = "PseudoSmilesHash"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        hash_col = global_apply(data, self._row_action, axis=1)
        return data.assign(**{self.out_column: hash_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (calculate hash based on InChI key of components of SMILES)"

    def _row_action(self, row: pd.Series) -> str:
        def components_to_inchi(smiles):
            mols = [
                Chem.MolFromSmiles(smi) for smi in split_smiles_from_reaction(smiles)
            ]
            return ".".join(sorted([Chem.MolToInchiKey(mol) for mol in mols if mol]))

        return components_to_inchi(row[self.in_column])


@action
@dataclass
class ReactantProductAtomBalance:
    """Action for computing atom balance"""

    pretty_name: ClassVar[str] = "atombalance"
    in_column: str
    out_column: str = "RxnAtomBalance"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        smiles_col = global_apply(data, self._row_action, axis=1)
        return data.assign(**{self.out_column: smiles_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (product atom count minus reactants atom count)"

    def _row_action(self, row: pd.Series) -> str:
        reactants, _, products = row[self.in_column].split(">")

        reactants_mol = Chem.MolFromSmiles(reactants)
        if reactants_mol:
            reactant_atom_count = reactants_mol.GetNumHeavyAtoms()
        else:
            reactant_atom_count = 0

        products_mol = Chem.MolFromSmiles(products)
        if products_mol:
            product_atom_count = products_mol.GetNumHeavyAtoms()
        else:
            product_atom_count = 0

        return product_atom_count - reactant_atom_count


@action
@dataclass
class ReactantSize:
    """Action for counting reactant size"""

    pretty_name: ClassVar[str] = "reactantsize"
    in_column: str
    out_column: str = "ReactantSize"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        smiles_col = global_apply(data, self._row_action, axis=1)
        return data.assign(**{self.out_column: smiles_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (number of heavy atoms in reactant)"

    def _row_action(self, row: pd.Series) -> str:
        reactants, _, _ = row[self.in_column].split(">")
        reactants_mol = Chem.MolFromSmiles(reactants)

        if reactants_mol:
            reactant_atom_count = reactants_mol.GetNumHeavyAtoms()
        else:
            reactant_atom_count = 0

        return reactant_atom_count


@action
@dataclass
class SmilesLength:
    """Action for counting SMILES length"""

    pretty_name: ClassVar[str] = "smiles_length"
    in_column: str
    out_column: str = "SmilesLength"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        smiles_col = global_apply(data, self._row_action, axis=1)
        return data.assign(**{self.out_column: smiles_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (calculate length of smiles)"

    def _row_action(self, row: pd.Series) -> str:
        return len(row[self.in_column])


@action
@dataclass
class SmilesSanitizable:
    """Action for checking if SMILES are sanitizable"""

    pretty_name: ClassVar[str] = "smiles_sanitizable"
    in_column: str
    out_column: str = "SmilesSanitizable"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        smiles_col = global_apply(data, self._row_action, axis=1)
        return data.assign(**{self.out_column: smiles_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (check if SMILES are sanitizable by RDKit)"

    def _row_action(self, row: pd.Series) -> str:
        smiles = row[self.in_column]
        mol = Chem.MolFromSmiles(smiles)
        try:  # Extra check that the molecule can generate SMILES that are again parsable
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        except Exception:  # pylint: disable=broad-except # noqa
            return False
        else:
            return bool(mol)


@action
@dataclass
class StereoInvention:
    """
    Flags reactions where non-stereo compounds (No "@"s in SMILES)
    turn into stereo compounds (containing "@")
    """

    pretty_name: ClassVar[str] = "stereo_invention"
    in_column: str
    out_column: str = "StereoInvention"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        smiles_col = global_apply(data, self._row_action, axis=1)
        return data.assign(**{self.out_column: smiles_col})

    def __str__(self) -> str:
        return (
            f"{self.pretty_name} (Reactants with no stereo, turn into stereo compounds)"
        )

    def _has_stereo(self, smiles: str) -> bool:
        return "@" in smiles

    def _row_action(self, row: pd.Series) -> bool:
        reactants, _, products = row[self.in_column].split(">")

        return self._has_stereo(products) and not self._has_stereo(reactants)
