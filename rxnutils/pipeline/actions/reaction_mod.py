"""Module containing actions on reactions that modify the reaction in some way"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, List, Tuple

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import RDConfig

from rxnutils.chem.disconnection_sites.atom_map_tagging import atom_map_tag_products
from rxnutils.chem.disconnection_sites.tag_converting import convert_atom_map_tag
from rxnutils.chem.utils import (
    atom_mapping_numbers,
    desalt_molecules,
    join_smiles_from_reaction,
    neutralize_molecules,
    recreate_rsmi,
    remove_atom_mapping,
    split_rsmi,
    split_smiles_from_reaction,
)
from rxnutils.pipeline.base import ReactionActionMixIn, action, global_apply

CONTRIB_INSTALLED = os.path.exists(RDConfig.RDContribDir)
if CONTRIB_INSTALLED:
    # RDKit contrib is not default part of PYTHONPATH
    sys.path.append(RDConfig.RDContribDir)
    sys.path.append(f"{RDConfig.RDContribDir}/RxnRoleAssignment")  # to make Nadines code import "utils"
    # fmt: off
    from RxnRoleAssignment.identifyReactants import reassignReactionRoles  # pylint: disable=all # noqa

    # fmt: on

rd_logger = RDLogger.logger()
rd_logger.setLevel(RDLogger.CRITICAL)


@action
@dataclass
class DesaltMolecules(ReactionActionMixIn):
    """Action for desalting molecules"""

    pretty_name: ClassVar[str] = "desalt_molecules"
    in_column: str
    out_column: str = "RxnDesalted"
    keep_something: bool = False

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        smiles_col = global_apply(data, self._apply_row, axis=1)
        return data.assign(**{self.out_column: smiles_col})

    def _apply_row(self, row: pd.Series) -> pd.Series:
        reactants_list, reagents_list, products_list = self.split_lists(row)
        reactants_list = desalt_molecules(reactants_list, keep_something=self.keep_something)
        if reagents_list:
            reagents_list = desalt_molecules(reagents_list, keep_something=self.keep_something)
        products_list = desalt_molecules(products_list, keep_something=self.keep_something)
        return self.join_lists(reactants_list, reagents_list, products_list)

    def __str__(self) -> str:
        return f"{self.pretty_name} (desalt molecules using RDKit SaltRemover)"


@action
@dataclass
class NameRxn:
    """Action for calling namrxn"""

    pretty_name: ClassVar[str] = "namerxn"
    in_column: str
    options: str = ""
    nm_rxn_column: str = "NextMoveRxnSmiles"
    nmc_column: str = "NMC"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        _, infile = tempfile.mkstemp(suffix=".smi")
        _, outfile = tempfile.mkstemp(suffix=".smi")
        data[self.in_column].to_csv(infile, index=False, header=False)
        subprocess.call(["namerxn"] + self.options.split() + [infile, outfile])
        if not os.path.exists(outfile) or os.path.getsize(outfile) == 0:
            raise FileNotFoundError("Could not produce namerxn output. Make sure 'namerxn' program is in path")
        # Set the data types to 'str' classification id must be a string
        namerxn_data = pd.read_csv(
            outfile,
            sep=" ",
            names=[self.nm_rxn_column, self.nmc_column],
            dtype=str,
        )
        return data.assign(**{col: namerxn_data[col] for col in namerxn_data.columns})

    def __str__(self) -> str:
        return f"{self.pretty_name} (running 'namerxn' to do atom mapping and classification)"


@action
@dataclass
class NeutralizeMolecules(ReactionActionMixIn):
    """Action for neutralizing molecules"""

    pretty_name: ClassVar[str] = "neutralize_molecules"
    in_column: str
    out_column: str = "RxnNeutralized"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        smiles_col = global_apply(data, self._apply_row, axis=1)
        return data.assign(**{self.out_column: smiles_col})

    def _apply_row(self, row: pd.Series) -> pd.Series:
        reactants_list, reagents_list, products_list = self.split_lists(row)
        reactants_list = neutralize_molecules(reactants_list)
        if reagents_list:
            reagents_list = neutralize_molecules(reagents_list)
        products_list = neutralize_molecules(products_list)
        return self.join_lists(reactants_list, reagents_list, products_list)

    def __str__(self) -> str:
        return f"{self.pretty_name} (neutralize molecules using RDKit neutralizer)"


@action
@dataclass
class ReactantsToReagents(ReactionActionMixIn):
    """Action for converting reactants to reagents"""

    pretty_name: ClassVar[str] = "reactants2reagents"
    in_column: str
    out_column: str = "RxnSmilesWithTrueReagents"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        smiles_col = global_apply(data, self._row_action, axis=1)
        return data.assign(**{self.out_column: smiles_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (putting all non-reacting reactants as reagents)"

    def _row_action(self, row: pd.Series) -> str:
        reactants_list_old, reagents_list, products_list = self.split_lists(row)

        product_indices = {idx for smi in products_list for idx in atom_mapping_numbers(smi)}

        reactants_list = []
        for reactant in reactants_list_old:
            reactant_indices = atom_mapping_numbers(reactant)
            if not reactant_indices or not any(idx in product_indices for idx in reactant_indices):
                reagents_list.append(reactant)
            else:
                reactants_list.append(reactant)
        return self.join_lists(reactants_list, reagents_list, products_list)


@action
@dataclass
class ReagentsToReactants:
    """Action for converting reagents to reactants"""

    pretty_name: ClassVar[str] = "reagents2reactants"
    in_column: str
    out_column: str = "RxnSmilesAllReactants"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        smiles_col = global_apply(data, self._row_action, axis=1)
        return data.assign(**{self.out_column: smiles_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (putting all reagents to reactants)"

    def _row_action(self, row: pd.Series) -> str:
        reactants, reagents, products = split_rsmi(row[self.in_column])
        if reagents:
            reactants = ".".join([reactants, reagents])
        return reactants + ">>" + products


@action
@dataclass
class RemoveAtomMapping(ReactionActionMixIn):
    """Action for removing all atom mapping"""

    pretty_name: ClassVar[str] = "remove_atom_mapping"
    in_column: str
    out_column: str = "RxnSmilesNoAtomMap"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        smiles_col = global_apply(data, self._row_action, axis=1)
        return data.assign(**{self.out_column: smiles_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (remove atom mapping)"

    def _row_action(self, row: pd.Series) -> str:
        reactants_list, reagents_list, products_list = self.split_lists(row)
        reactants_list = [remove_atom_mapping(smi, sanitize=False) for smi in reactants_list]
        reagents_list = [remove_atom_mapping(smi, sanitize=False) for smi in reagents_list]
        products_list = [remove_atom_mapping(smi, sanitize=False) for smi in products_list]

        return self.join_lists(reactants_list, reagents_list, products_list)


@action
@dataclass
class RemoveStereoInfo:
    """Action for removing stero information"""

    pretty_name: ClassVar[str] = "remove_stereo_info"
    in_column: str
    out_column: str = "RxnSmilesNoStereo"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        smiles_col = global_apply(data, self._row_action, axis=1)
        return data.assign(**{self.out_column: smiles_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (remove stereo information from the SMILES)"

    def _row_action(self, row: pd.Series) -> str:
        smiles = row[self.in_column]
        smiles_nostereo = smiles.replace("@", "")
        return smiles_nostereo


@action
@dataclass
class InvertStereo:
    """Action for inverting stero information"""

    pretty_name: ClassVar[str] = "invert_stereo"
    in_column: str
    out_column: str = "RxnSmilesInvertedStereo"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        smiles_col = global_apply(data, self._row_action, axis=1)
        return data.assign(**{self.out_column: smiles_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (invert stereo in SMILES (@@ => @ and @ => @@))"

    def _row_action(self, row: pd.Series) -> str:
        smiles = row[self.in_column]
        smiles_temp = smiles.replace("@@", "£").replace("@", "$")
        return smiles_temp.replace("$", "@@").replace("£", "@")


@action
@dataclass
class IsotopeInfo:
    """Action creating and modifying isotope information"""

    pretty_name: ClassVar[str] = "isotope_info"
    in_column: str
    isotope_column: str = "Isotope"
    out_column: str = "RxnSmilesWithoutIsotopes"
    match_regex: str = r"\[(?P<mass>[0-9]+)(?P<symbol>[A-Za-z][a-z]*)(?P<spec>[^\]]+)*\]"
    sub_regex: str = r"[\g<symbol>\g<spec>]"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        isotop_info = data[self.in_column].str.extract(self.match_regex)
        isotop_col = isotop_info["mass"] + isotop_info["symbol"]
        rsmi_col = data[self.in_column].str.replace(
            self.match_regex, self.sub_regex, regex=True
        )
        return data.assign(**{self.isotope_column: isotop_col, self.out_column: rsmi_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (extract and remove isotope information from reactions)"


@action
@dataclass
class RemoveExtraAtomMapping(ReactionActionMixIn):
    """Action for removing extra atom mapping"""

    pretty_name: ClassVar[str] = "remove_extra_atom_mapping"
    in_column: str
    out_column: str = "RxnSmilesReassignedAtomMap"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        smiles_col = global_apply(data, self._row_action, axis=1)
        return data.assign(**{self.out_column: smiles_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (removing atom maps in reactants and reagents not in products)"

    def _row_action(self, row: pd.Series) -> str:
        def reassign_atom_mapping(smiles: str) -> str:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles
            for atom in mol.GetAtoms():
                if atom.GetAtomMapNum() not in product_indices:
                    atom.SetAtomMapNum(0)
            return Chem.MolToSmiles(mol)

        reactants_list, reagents_list, products_list = self.split_lists(row)

        product_indices = {idx for smi in products_list for idx in atom_mapping_numbers(smi)}

        reactants_list = [reassign_atom_mapping(smi) for smi in reactants_list]
        reagents_list = [reassign_atom_mapping(smi) for smi in reagents_list]
        return self.join_lists(reactants_list, reagents_list, products_list)


@action
@dataclass
class RemoveUnchangedProducts(ReactionActionMixIn):
    """Compares the products with the reagents and reactants and remove unchanged products.

    Protonation is considered a difference, As example, if there's a HCl in the reagents and
    a Cl- in the products, it will not be removed.
    """

    pretty_name: ClassVar[str] = "remove_unchanged_products"
    in_column: str
    out_column: str = "RxnNoUnchangedProd"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        smiles_col = global_apply(data, self._row_action, axis=1)
        return data.assign(**{self.out_column: smiles_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (Remove unchanged products)"

    def _row_action(self, row: pd.Series) -> str:
        reactants_list, reagents_list, products_list = self.split_lists(row)

        reactants_mols = [Chem.MolFromSmiles(m) for m in reactants_list]
        reagents_mols = [Chem.MolFromSmiles(m) for m in reagents_list]
        products_mols = [Chem.MolFromSmiles(m) for m in products_list]

        new_products = []
        for product, product_smi in zip(products_mols, products_list):
            matches = [self._is_equal(product, r) for r in reactants_mols + reagents_mols]
            if not any(matches):
                new_products.append(product_smi)

        return self.join_lists(reactants_list, reagents_list, new_products)

    @staticmethod
    def _is_equal(mol1, mol2) -> bool:
        try:
            inchi1 = Chem.MolToInchiKey(mol1)
            inchi2 = Chem.MolToInchiKey(mol2)
        except Exception:  # pylint: disable=broad-except # noqa
            return False
        else:
            return inchi1 == inchi2


@action
@dataclass
class RemoveUnsanitizable(ReactionActionMixIn):
    """Action for removing unsanitizable reactions"""

    pretty_name: ClassVar[str] = "remove_unsanitizable"
    in_column: str
    out_column: str = "RxnSanitizable"
    bad_column: str = "BadMolecules"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        new_data = global_apply(data, self._apply_row, axis=1)
        return data.assign(
            **{
                self.out_column: new_data.rxn_smi,
                self.bad_column: new_data.bad_smiles,
            }
        )

    def _apply_row(self, row: pd.Series) -> pd.Series:
        def process_smiles(smiles_list: List[str]) -> Tuple[List[str], List[str]]:
            good_list = []
            bad_list = []
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    try:
                        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
                    except Exception:  # pylint: disable=broad-except # noqa
                        bad_list.append(smiles)
                    else:
                        if mol:
                            good_list.append(smiles)
                        else:
                            bad_list.append(smiles)
                else:
                    bad_list.append(smiles)
            return good_list, bad_list

        reactants_list, reagents_list, products_list = self.split_lists(row)
        good_reactants, bad_reactants = process_smiles(reactants_list)
        good_reagents, bad_reagents = process_smiles(reagents_list)
        good_products, bad_products = process_smiles(products_list)

        bad_smiles = ",".join(bad_reactants + bad_reagents + bad_products)
        rxn_smi = self.join_lists(good_reactants, good_reagents, good_products)
        return pd.Series({"rxn_smi": rxn_smi, "bad_smiles": bad_smiles})

    def __str__(self) -> str:
        return f"{self.pretty_name} (removing molecules that is not sanitizable by RDKit)"


@action
@dataclass
class RDKitRxnRoles:
    """Action for assigning roles based on RDKit algorithm"""

    pretty_name: ClassVar[str] = "rdkit_RxnRoleAssignment"
    in_column: str
    out_column: str = "RxnRoleAssigned"

    def __post_init__(self):
        if not CONTRIB_INSTALLED:
            raise ImportError("This action cannot be used because the RDKit Contrib folder is not installed")

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:

        smiles_col = global_apply(data, lambda row: reassignReactionRoles(row[self.in_column]), axis=1)
        return data.assign(**{self.out_column: smiles_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (reaction role assignment using RDKit contrib (Nadine and Greg's))"


# fmt: off
_MAP_SCRIPT = Path(__file__).parent.parent.parent / "data" / "mapping.py"
# fmt: on


@action
@dataclass
class RxnMapper:
    """Action for mapping reactions with the RXNMapper tool"""

    pretty_name: ClassVar[str] = "rxnmapper"
    in_column: str
    out_column: str = "RxnmapperRxnSmiles"
    rxnmapper_command: str = "conda run -p ${RXNMAPPER_ENV_PATH} python " + str(_MAP_SCRIPT)

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        new_column = self._do_action(data[self.in_column])
        return data.assign(**{self.out_column: new_column})

    def _do_action(self, column: pd.Series) -> pd.Series:
        input_handle, input_path = tempfile.mkstemp(suffix=".csv")
        output_handle, output_path = tempfile.mkstemp(suffix=".csv")
        # Recreate Reaction SMILES by removing intra-molecular complexes
        tmp_column = column.apply(recreate_rsmi)
        tmp_column.to_csv(input_path, sep="\t", index=False)

        cmd = self.rxnmapper_command
        if "CONDA_PATH" in os.environ:
            cmd = os.environ["CONDA_PATH"] + os.sep + cmd
        arglist = cmd.split() + [
            "--input",
            input_path,
            "--column",
            self.in_column,
            "--output",
            output_path,
        ]
        _ = subprocess.check_output(arglist).decode("utf-8")

        mapped_data = pd.read_csv(output_path, sep="\t")

        os.remove(input_path)
        os.remove(output_path)
        os.close(input_handle)
        os.close(output_handle)

        return mapped_data["mapped_rxn"]

    def __str__(self) -> str:
        return f"{self.pretty_name} (RFP by rxnfp)"


@action
@dataclass
class SplitReaction(ReactionActionMixIn):
    """Action for splitting reaction into components"""

    pretty_name: ClassVar[str] = "split_reaction"
    in_column: str
    out_columns: list[str]

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        new_columns = global_apply(data, self._apply_row, axis=1)
        if len(self.out_columns) == 2:
            return data.assign(
                **{
                    self.out_columns[0]: new_columns.reactants,
                    self.out_columns[1]: new_columns.products,
                }
            )
        if len(self.out_columns) == 3:
            return data.assign(
                **{
                    self.out_columns[0]: new_columns.reactants,
                    self.out_columns[1]: new_columns.reagents,
                    self.out_columns[2]: new_columns.products,
                }
            )
        raise ValueError(f"Don't know what to return with {len(self.out_columns)} output columns")

    def __str__(self) -> str:
        return f"{self.pretty_name} (rxn => reactants, [reagents], products)"

    def _apply_row(self, row: pd.Series) -> pd.Series:
        reactants, reagents, products = self.split_smiles(row)
        return pd.Series({"reactants": reactants, "reagents": reagents, "products": products})


@action
@dataclass
class AtomMapTagDisconnectionSite(ReactionActionMixIn):
    """Action for tagging disconnection site in products with atom-map '[<atom>:1]'."""

    pretty_name: ClassVar[str] = "atom_map_tag_disconnection_site"
    in_column: str = "RxnSmilesClean"
    out_column: str = "products_atom_map_tagged"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        smiles_col = global_apply(data, self._row_action, axis=1)
        return data.assign(**{self.out_column: smiles_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (tag disconnection sites in products with '[<atom>:1]')"

    def _row_action(self, row: pd.Series) -> str:
        return atom_map_tag_products(row[self.in_column])


@action
@dataclass
class ConvertAtomMapDisconnectionTag(ReactionActionMixIn):
    """Action for converting atom-map tagging to exclamation mark tagging.

    yaml example:

    convert_atom_map_disconnection_tag:
        in_column_tagged: products_atom_map_tagged
        in_column_untagged: products
        out_column_tagged: products_tagged
        out_column_reconstructed: products_reconstructed
    """

    pretty_name: ClassVar[str] = "convert_atom_map_disconnection_tag"
    in_column: str = "products_atom_map_tagged"
    out_column_tagged: str = "products_tagged"
    out_column_reconstructed: str = "products_reconstructed"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        smiles_tagged_col = global_apply(data, self._row_action, axis=1)
        smiles_reconstructed_col = smiles_tagged_col.str.replace("!", "")

        return data.assign(
            **{
                self.out_column_tagged: smiles_tagged_col,
                self.out_column_reconstructed: smiles_reconstructed_col,
            }
        )

    def __str__(self) -> str:
        return f"{self.pretty_name} (convert disconnection tagging '[<atom>:1]' to '<atom>!')"

    def _row_action(self, row: pd.Series) -> str:
        product_tagged = convert_atom_map_tag(row[self.in_column])
        return product_tagged


@action
@dataclass
class TrimRxnSmiles:
    """Action from trimming reaction SMILES"""

    pretty_name: ClassVar[str] = "trim_rxn_smiles"
    in_column: str
    out_column: str = "RxnSmiles"
    smiles_column_index: int = 0

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        smiles_col = global_apply(
            data,
            lambda row: row[self.in_column].split(" ")[self.smiles_column_index],
            axis=1,
        )
        return data.assign(**{self.out_column: smiles_col})

    def __str__(self) -> str:
        return f"{self.pretty_name} (removing junk from the reaction smiles)"
