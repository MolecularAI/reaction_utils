"""Module containing template validation actions"""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Set, Sequence

import pandas as pd
from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import AllChem

from rxnutils.pipeline.base import action, global_apply
from rxnutils.chem.template import ReactionTemplate
from rxnutils.chem.utils import split_smiles_from_reaction

rd_logger = RDLogger.logger()
rd_logger.setLevel(RDLogger.CRITICAL)


@action
@dataclass
class CountTemplateComponents:
    """Action for counting template components"""

    pretty_name: ClassVar[str] = "count_template_components"
    in_column: str
    nreactants_column: str = "nreactants"
    nreagents_column: str = "nreagents"
    nproducts_column: str = "nproducts"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        new_data = global_apply(data, self._apply_row, axis=1)
        return data.assign(**{column: new_data[column] for column in new_data.columns})

    def _apply_row(self, row: pd.Series) -> pd.Series:
        if not isinstance(row[self.in_column], str):
            return pd.Series(
                {
                    self.nreactants_column: None,
                    self.nreagents_column: None,
                    self.nproducts_column: None,
                }
            )

        rd_reaction = AllChem.ReactionFromSmarts(row[self.in_column])
        return pd.Series(
            {
                self.nproducts_column: rd_reaction.GetNumReactantTemplates(),
                self.nreagents_column: rd_reaction.GetNumAgentTemplates(),
                self.nreactants_column: rd_reaction.GetNumProductTemplates(),
            }
        )

    def __str__(self) -> str:
        return (
            f"{self.pretty_name} (counting reactants, reagents, products in template)"
        )


@action
@dataclass
class RetroTemplateReproduction:
    """Action for checking template reproduction"""

    pretty_name: ClassVar[str] = "retro_template_reproduction"
    template_column: str
    smiles_column: str
    expected_reactants_column: str = "TemplateGivesTrueReactants"
    other_reactants_column: str = "TemplateGivesOtherReactants"
    noutcomes_column: str = "TemplateGivesNOutcomes"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        new_data = global_apply(data, self._apply_row, axis=1)
        return data.assign(**{column: new_data[column] for column in new_data.columns})

    def _apply_row(self, row: pd.Series) -> pd.Series:
        no_oututcome_response = pd.Series(
            {
                self.noutcomes_column: 0,
                self.expected_reactants_column: False,
                self.other_reactants_column: False,
            }
        )
        if not isinstance(row[self.template_column], str):
            return no_oututcome_response

        reactants, _, products = row[self.smiles_column].split(">")
        # Flattening the list of SMILES, i.e. intramolecular -> intermolecular,
        # because all produced templates will be intermolecular
        expected_reactants = self._smiles_to_inchiset(
            [
                smi
                for reactant in split_smiles_from_reaction(reactants)
                for smi in reactant.split(".")
            ]
        )
        template = ReactionTemplate(row[self.template_column], direction="retro")

        try:
            outcome = template.apply(products)
        except Exception:  # pylint: disable=broad-except # noqa
            return no_oututcome_response
        if not outcome:
            return no_oututcome_response

        found_expected_reactants = False
        found_other_reactants = False
        for outcome_item in outcome:
            if self._smiles_to_inchiset(outcome_item).issubset(expected_reactants):
                found_expected_reactants = True
            else:
                found_other_reactants = True
        return pd.Series(
            {
                self.noutcomes_column: len(outcome),
                self.expected_reactants_column: found_expected_reactants,
                self.other_reactants_column: found_other_reactants,
            }
        )

    @staticmethod
    def _smiles_to_inchiset(smiles_list: Sequence[str]) -> Set[str]:
        inchi_list = []
        for smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smi)
            except Exception:  # pylint: disable=broad-except # noqa
                pass
            else:
                if mol:
                    inchi_list.append(Chem.MolToInchiKey(mol))
        return set(inchi_list)

    def __str__(self) -> str:
        return f"{self.pretty_name} (checking that retro template produce the expected reactants)"
