"""Module containing actions on reactions that doesn't modify the reactions only compute properties of them"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import ClassVar, List, Optional, Set, Tuple

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.MolKey.InchiInfo import InchiInfo
from rdkit.Chem.rdMolDescriptors import CalcNumRings
from rdkit.Chem.rdmolops import FindPotentialStereo

import rxnutils.chem.smartslib as smartslib
from rxnutils.chem.cgr import CondensedGraphReaction
from rxnutils.chem.reaction import ChemicalReaction
from rxnutils.chem.utils import (
    atom_mapping_numbers,
    has_atom_mapping,
    reaction_centres,
    split_rsmi,
    split_smiles_from_reaction,
)
from rxnutils.pipeline.base import ReactionActionMixIn, action, global_apply

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
        reactants, _, _ = split_rsmi(row[self.in_column])
        for smiles in split_smiles_from_reaction(reactants):
            count_elements(smiles, counts)

        return json.dumps(counts)


@action
@dataclass
class DetectReactiveFunctions:
    """
    Maps reactive SMART functions to reactants.
    """

    pretty_name: ClassVar[str] = "detect_reactive_functions"

    in_column: str
    func_column: str = "ReactiveFunction"
    rsmi_column: str = "RxnProcessed"

    smarts_lib: str = "ontology"
    alphabetic_order: bool = True
    add_none: bool = True
    max_reactants: Optional[int] = None

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            self.lib = smartslib.SmartsLibrary.load(self.smarts_lib)
        except KeyError:
            smarts_lib_path = self.smarts_lib
            if smarts_lib_path.startswith("$"):
                smarts_lib_path = os.environ[smarts_lib_path[2:-1]]
            self.lib = smartslib.SmartsLibrary(smarts_lib_path)
        new_data = global_apply(data, self._row_action, axis=1)
        return data.assign(**{column: new_data[column] for column in new_data.columns})

    def __str__(self) -> str:
        return f"{self.pretty_name} (maps reactive functions to atoms in reactants)"

    def _row_action(self, row: pd.Series) -> bool:
        try:
            reactive_functions, rsmi = self.lib.detect_reactive_functions(
                ChemicalReaction(row[self.in_column], clean_smiles=False),
                sort=self.alphabetic_order,
                add_none=self.add_none,
                target_size=None,
                max_reactants=self.max_reactants,
            )
        except ValueError:
            return pd.Series(
                {self.func_column: None, self.rsmi_column: row[self.in_column]}
            )
        return pd.Series(
            {self.func_column: "|".join(reactive_functions), self.rsmi_column: rsmi}
        )


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
class HasUnmappedRadicalAtom:
    """Action for flagging if reaction has any unmapped radical atoms"""

    pretty_name: ClassVar[str] = "hasunmappedradicalatom"
    in_column: str
    out_column: str = "HasUnmappedRadicalAtom"

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        smiles_col = global_apply(df, self._row_action, axis=1)
        return df.assign(**{self.out_column: smiles_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (detect if there is an unmapped radical in the reaction SMILES)"

    def _row_action(self, row: pd.Series) -> str:
        reactants, _, product = split_rsmi(row["rsmi_processed"])

        react_mol = Chem.MolFromSmiles(reactants)
        if Descriptors.NumRadicalElectrons(react_mol) == 0:
            return False

        prod_mol = Chem.MolFromSmiles(product)
        unmapped_product_atom_element = None
        for atom in prod_mol.GetAtoms():
            if atom.GetAtomMapNum() == 0:
                unmapped_product_atom_element = atom.GetAtomicNum()
                break

        if unmapped_product_atom_element is None:
            return False

        has_radical = False
        for atom in react_mol.GetAtoms():
            if (
                atom.GetAtomMapNum() == 0
                and atom.GetNumRadicalElectrons() > 0
                and atom.GetAtomicNum() == unmapped_product_atom_element
            ):
                has_radical = True
                break
        return has_radical


@action
@dataclass
class HasUnsanitizableReactants:
    """Action for flagging if reaction has any unsanitizable reactants"""

    pretty_name: ClassVar[str] = "unsanitizablereactants"
    rsmi_column: str
    bad_columns: List[str]
    out_column: str = "HasUnsanitizableReactants"

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        smiles_col = global_apply(df, self._row_action, axis=1)
        return df.assign(**{self.out_column: smiles_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (detect if there is unsanitizable reactants)"

    def _row_action(self, row: pd.Series) -> str:
        unsanitize_atom_map_num = set()
        for column in self.bad_columns:
            if not isinstance(row[column], str):
                continue
            unsanitize_atom_map_num |= self._process_bad_column(row[column])

        if not unsanitize_atom_map_num:
            return False

        _, _, product = split_rsmi(row[self.rsmi_column])
        prod_atom_map_num = set(atom_mapping_numbers(product))
        return bool(prod_atom_map_num.intersection(unsanitize_atom_map_num))

    @staticmethod
    def _process_bad_column(smiles_list: str) -> Set[int]:
        unsanitize_atom_map_num = set()
        for smiles in smiles_list.split(","):
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if not mol:
                continue
            unsanitize_atom_map_num |= {
                atom.GetAtomMapNum() for atom in mol.GetAtoms() if atom.GetAtomMapNum()
            }
        return unsanitize_atom_map_num


@action
@dataclass
class CgrCreated:
    """Action for determining if a CGR can be created from the reaction smiles"""

    in_column: str
    pretty_name: ClassVar[str] = "cgr_created"
    out_column: str = "CGRCreated"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        out_col = global_apply(data, self._create_cgr_column, axis=1)
        return data.assign(**{self.out_column: out_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (flag if a CGR can be created for the reaction)"

    def _create_cgr_column(self, row: pd.Series) -> bool:
        rxn = ChemicalReaction(row[self.in_column], clean_smiles=False)
        try:
            _ = CondensedGraphReaction(rxn)
        except Exception:
            return False
        return True


@action
@dataclass
class CgrNumberOfDynamicBonds:
    """Action for calculating the number of dynamic bonds"""

    in_column: str
    pretty_name: ClassVar[str] = "cgr_dynamic_bonds"
    out_column: str = "NDynamicBonds"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        out_col = global_apply(data, self._process_row, axis=1)
        return data.assign(**{self.out_column: out_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (number of dynamic bonds in the CGR)"

    def _process_row(self, row: pd.Series) -> Optional[int]:
        rxn = ChemicalReaction(row[self.in_column], clean_smiles=False)
        try:
            cgr = CondensedGraphReaction(rxn)
        except Exception:
            return None
        return cgr.bonds_changed


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
        reactants, _, products = split_rsmi(row[self.in_column])
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
        _, _, products = split_rsmi(row[self.in_column])
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
            reactants, _, products = split_rsmi(row[self.in_column])
            return ">>".join(
                [components_to_inchi(smi) for smi in [reactants, products]]
            )

        return ">".join(
            components_to_inchi(smi) for smi in split_rsmi(row[self.in_column])
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
        reactants, _, products = split_rsmi(row[self.in_column])

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
        reactants, _, _ = split_rsmi(row[self.in_column])
        reactants_mol = Chem.MolFromSmiles(reactants)

        if reactants_mol:
            reactant_atom_count = reactants_mol.GetNumHeavyAtoms()
        else:
            reactant_atom_count = 0

        return reactant_atom_count


@action
@dataclass
class MaxRingNumber:
    """
    Action for calculating the maximum number of rings in either the product or reactant
    For a reaction without reactants or products, it will return 0 to enable easy arithmetic comparison
    """

    pretty_name: ClassVar[str] = "maxrings"
    in_column: str
    out_column: str = "MaxRings"

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        smiles_col = global_apply(df, self._row_action, axis=1)
        return df.assign(**{self.out_column: smiles_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (maximum number of rings)"

    def _row_action(self, row: pd.Series) -> str:
        reactants, _, products = split_rsmi(row[self.in_column])

        reactants_mols = [Chem.MolFromSmiles(smi) for smi in reactants.split(".")]
        nrings_reactants = [CalcNumRings(mol) for mol in reactants_mols if mol]
        max_rings_reactants = 0 if not nrings_reactants else max(nrings_reactants)

        prod_mol = Chem.MolFromSmiles(products)
        if not prod_mol:
            return max_rings_reactants

        nrings_product = CalcNumRings(prod_mol)
        return max([nrings_product, max_rings_reactants])


@action
@dataclass
class RingNumberChange:
    """
    Action for calculating if reaction has change in number of rings

    A positive number from this action implies that a ring was formed during the reaction
    """

    pretty_name: ClassVar[str] = "ringnumberchange"
    in_column: str
    out_column: str = "NRingChange"

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        smiles_col = global_apply(df, self._row_action, axis=1)
        return df.assign(**{self.out_column: smiles_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (ring change based on number of rings)"

    def _row_action(self, row: pd.Series) -> str:
        reactants, _, products = split_rsmi(row[self.in_column])
        prod_mol = Chem.MolFromSmiles(products)

        if not prod_mol:
            return False

        reactants_mols = [Chem.MolFromSmiles(smi) for smi in reactants.split(".")]
        nrings_reactants = sum(CalcNumRings(mol) for mol in reactants_mols if mol)
        nrings_product = CalcNumRings(prod_mol)
        return nrings_product - nrings_reactants


@action
@dataclass
class RingBondMade:
    """Action for flagging if reaction has made a ring bond in the product"""

    pretty_name: ClassVar[str] = "ringbondmade"
    in_column: str
    out_column: str = "RingBondMade"

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        smiles_col = global_apply(df, self._row_action, axis=1)
        return df.assign(**{self.out_column: smiles_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (ring change based on ring bond made)"

    def _row_action(self, row: pd.Series) -> str:
        reactants, _, products = split_rsmi(row[self.in_column])

        prod_mol = Chem.MolFromSmiles(products)
        if not prod_mol:
            return False
        ring_bonds = [
            (bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum())
            for bond in prod_mol.GetBonds()
            if bond.IsInRing()
        ]

        for smiles in split_smiles_from_reaction(reactants):
            reactant_mol = Chem.MolFromSmiles(smiles)
            if not reactant_mol:
                continue
            r_mappings = self._mapping_to_index(reactant_mol)
            for atom_map1, atom_map2 in ring_bonds:
                if atom_map1 not in r_mappings and atom_map2 not in r_mappings:
                    continue

                # If not both of the atoms are in the same molecule the bond is new
                if atom_map1 not in r_mappings and atom_map2 in r_mappings:
                    return True
                if atom_map2 not in r_mappings and atom_map1 in r_mappings:
                    return True

                atom_idx1 = r_mappings[atom_map1]
                atom_idx2 = r_mappings[atom_map2]
                bond = reactant_mol.GetBondBetweenAtoms(atom_idx1, atom_idx2)
                if bond is None or not bond.IsInRing():
                    return True
        return False

    @staticmethod
    def _mapping_to_index(mol):
        return {
            atom.GetAtomMapNum(): atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetAtomMapNum()
        }


@action
@dataclass
class RingMadeSize:
    """Action for computing the size of a newly formed ring"""

    pretty_name: ClassVar[str] = "ringmadesize"
    in_column: str
    out_column: str = "RingMadeSize"

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        smiles_col = global_apply(df, self._row_action, axis=1)
        return df.assign(**{self.out_column: smiles_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (largest ring made)"

    def _row_action(self, row: pd.Series) -> str:
        reactants, _, products = split_rsmi(row[self.in_column])

        prod_mol = Chem.MolFromSmiles(products)
        if not prod_mol:
            return None

        reactant_rings = []
        for smiles in split_smiles_from_reaction(reactants):
            reactant_mol = Chem.MolFromSmiles(smiles)
            if not reactant_mol:
                continue

            reactant_rings.extend(self._find_rings(reactant_mol))

        largest_made_ring = 0
        for ring in self._find_rings(prod_mol):
            if ring not in reactant_rings:
                largest_made_ring = max(largest_made_ring, len(ring))
        return largest_made_ring

    @staticmethod
    def _find_rings(mol):
        rings = []
        for ring in mol.GetRingInfo().AtomRings():
            ring_atoms = [mol.GetAtomWithIdx(idx).GetAtomMapNum() for idx in ring]
            # Only consider at least partially atom-mapped rings
            if set(ring_atoms) == {0}:
                continue
            rings.append(tuple(sorted(ring_atoms)))
        return rings


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
        if not smiles:
            return False

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
        reactants, _, products = split_rsmi(row[self.in_column])

        return self._has_stereo(products) and not self._has_stereo(reactants)


@action
@dataclass
class StereoCentreChanges:
    """
    Action for checking if stereogenic centre in reaction center is changing
    during the reaction

    Will create two columns:
        1. A boolean column indicating True or False if it has stereochanges
        2. A description of the stereo information before and after the reaction
    """

    pretty_name: ClassVar[str] = "stereo_centre_changes"
    in_column: str
    out_column: str = "HasStereoChanges"
    stereo_changes_column: str = "StereoChanges"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        new_df = global_apply(data, self._row_action, axis=1)
        return data.assign(**{col: new_df[col] for col in new_df.columns})

    def __str__(self) -> str:
        return f"{self.pretty_name} (check if stereochemistry changes in reaction)"

    def _row_action(self, row: pd.Series) -> str:
        smiles = row[self.in_column]
        rdkit_rxn = AllChem.ReactionFromSmarts(smiles, useSmiles=True)
        rdkit_rxn.Initialize()
        rxncenters = reaction_centres(rdkit_rxn)

        reactants = rdkit_rxn.GetReactants()
        reactants_chiral_flags = {}
        for centers, reactant in zip(rxncenters, reactants):
            for index in centers:
                atom = reactant.GetAtomWithIdx(index)
                reactants_chiral_flags[atom.GetAtomMapNum()] = str(atom.GetChiralTag())

        product_chiral_flags = {}
        for atom in rdkit_rxn.GetProducts()[0].GetAtoms():
            if atom.GetAtomMapNum() in reactants_chiral_flags:
                product_chiral_flags[atom.GetAtomMapNum()] = str(atom.GetChiralTag())

        if reactants_chiral_flags == product_chiral_flags:
            return pd.Series({self.out_column: False, self.stereo_changes_column: None})

        output = {}
        for atom_map, reactant_chirality in reactants_chiral_flags.items():
            product_chirality = product_chiral_flags.get(atom_map, None)
            if product_chirality != reactant_chirality:
                output[atom_map] = (reactant_chirality, product_chirality)
        return pd.Series(
            {self.out_column: True, self.stereo_changes_column: json.dumps(output)}
        )


@action
@dataclass
class StereoHasChiralReagent:
    """
    Action for checking if reagent has stereo centres
    """

    pretty_name: ClassVar[str] = "stereo_chiral_reagent"
    in_column: str
    out_column: str = "HasChiralReagent"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        if len(data) == 0:
            return data
        out_col = global_apply(data, self._row_action, axis=1)
        return data.assign(**{self.out_column: out_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (check if reagent is chiral)"

    def _row_action(self, row: pd.Series) -> bool:
        smiles = row[self.in_column]
        return "@" in split_rsmi(smiles)[1]


@action
@dataclass
class StereoCenterIsCreated:
    """
    Action for checking if stereo centre is created during reaction
    """

    pretty_name: ClassVar[str] = "stereo_centre_created"
    in_column: str = "StereoChanges"
    out_column: str = "StereoCentreCreated"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        if len(data) == 0:
            return data
        out_col = global_apply(data, self._row_action, axis=1)
        return data.assign(**{self.out_column: out_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (check if stereo centre is created)"

    def _row_action(self, row: pd.Series) -> str:
        try:
            chirality_changes = json.loads(row[self.in_column])
        except TypeError:
            return False
        for reactant_chirality, _ in chirality_changes.values():
            if reactant_chirality == "CHI_UNSPECIFIED":
                return True
        return False


@action
@dataclass
class StereoCenterIsRemoved:
    """
    Action for checking if stereo centre is removed during reaction
    """

    pretty_name: ClassVar[str] = "stereo_centre_removed"
    in_column: str = "StereoChanges"
    out_column: str = "StereoCentreRemoved"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        if len(data) == 0:
            return data
        out_col = global_apply(data, self._row_action, axis=1)
        return data.assign(**{self.out_column: out_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (check if stereo centre is removed)"

    def _row_action(self, row: pd.Series) -> str:
        try:
            chirality_changes = json.loads(row[self.in_column])
        except TypeError:
            return False
        for _, product_chirality in chirality_changes.values():
            if product_chirality == "CHI_UNSPECIFIED":
                return True
        return False


@action
@dataclass
class StereoCenterInReactantPotential:
    """
    Action for checking if there is a potential stereo centre in the reaction

    Do not consider changes to bond stereochemistry
    """

    pretty_name: ClassVar[str] = "potential_stereo_center"
    in_column: str
    out_column: str = "PotentialStereoCentre"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        if len(data) == 0:
            return data
        out_col = global_apply(data, self._row_action, axis=1)
        return data.assign(**{self.out_column: out_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (check for existence of potential stereo centre)"

    def _row_action(self, row: pd.Series) -> str:
        smiles = row[self.in_column]
        rdkit_rxn = AllChem.ReactionFromSmarts(smiles, useSmiles=True)
        rdkit_rxn.Initialize()
        rxncenters = reaction_centres(rdkit_rxn)
        reactants = rdkit_rxn.GetReactants()

        atom_maps = set()
        for centers, reactant0 in zip(rxncenters, reactants):
            reactant = self._get_clean_mol_copy(reactant0)
            stereo_info = FindPotentialStereo(reactant)
            if any(
                not str(info.type).startswith("Bond")
                and info.centeredOn in centers
                and str(reactant.GetAtomWithIdx(info.centeredOn).GetChiralTag())
                == "CHI_UNSPECIFIED"
                for info in stereo_info
            ):
                return True
            for index in centers:
                atom = reactant0.GetAtomWithIdx(index)
                atom_maps.add(atom.GetAtomMapNum())

        product0 = rdkit_rxn.GetProducts()[0]
        product = self._get_clean_mol_copy(product0)
        stereo_info = FindPotentialStereo(product)
        if any(
            not str(info.type).startswith("Bond")
            and product0.GetAtomWithIdx(info.centeredOn).GetAtomMapNum() in atom_maps
            and str(product.GetAtomWithIdx(info.centeredOn).GetChiralTag())
            == "CHI_UNSPECIFIED"
            for info in stereo_info
        ):
            return True

        return False

    @staticmethod
    def _get_clean_mol_copy(mol0):
        """
        Return a copy of the molecule
        that is sanitized and without atom-map numbers
        """
        mol = AllChem.rdchem.Mol(mol0)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        AllChem.SanitizeMol(mol)
        return mol


@action
@dataclass
class StereoCenterOutsideReaction:
    """
    Action for checking if there is a stereo centre outside the reaction centre
    """

    pretty_name: ClassVar[str] = "stereo_centre_outside"
    in_column: str
    out_column: str = "StereoOutside"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        if len(data) == 0:
            return data
        out_col = global_apply(data, self._row_action, axis=1)
        return data.assign(**{self.out_column: out_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (check for existence of chirality outside reaction centre)"

    def _row_action(self, row: pd.Series) -> str:
        smiles = row[self.in_column]
        rdkit_rxn = AllChem.ReactionFromSmarts(smiles, useSmiles=True)
        rdkit_rxn.Initialize()
        rxncenters = reaction_centres(rdkit_rxn)

        reactants = rdkit_rxn.GetReactants()
        for centers, reactant in zip(rxncenters, reactants):
            for index, atom in enumerate(reactant.GetAtoms()):
                if index in centers:
                    continue
                if str(atom.GetChiralTag()) != "CHI_UNSPECIFIED":
                    return True
        return False


@action
@dataclass
class StereoMesoProduct:
    """
    Action for checking if the product is a meso compound
    """

    pretty_name: ClassVar[str] = "meso_product"
    in_column: str
    out_column: str = "MesoProduct"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        if len(data) == 0:
            return data
        out_col = global_apply(data, self._row_action, axis=1)
        return data.assign(**{self.out_column: out_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (check for existence of meso compound)"

    def _row_action(self, row: pd.Series) -> str:
        smiles = row[self.in_column]
        rdkit_rxn = AllChem.ReactionFromSmarts(smiles, useSmiles=True)
        rdkit_rxn.Initialize()
        product_mol = rdkit_rxn.GetProducts()[0]
        AllChem.SanitizeMol(product_mol)
        info = InchiInfo(AllChem.MolToInchi(product_mol))
        return info.get_sp3_stereo()["main"]["non-isotopic"][2]
