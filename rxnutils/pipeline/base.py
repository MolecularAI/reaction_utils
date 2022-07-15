"""Module containing routines for the validation framework"""
from __future__ import annotations
from typing import Any
from typing import Callable
from typing import Dict, Optional, Tuple, List
from collections import defaultdict
from dataclasses import fields

import pandas as pd
import swifter  # noqa #pylint: disable=unused-import
from tqdm import tqdm

from rxnutils.chem.utils import split_smiles_from_reaction


ActionType = Callable[[pd.DataFrame], pd.DataFrame]

_REGISTERED_ACTIONS: Dict[str, ActionType] = {}


def action(obj: ActionType) -> ActionType:
    """
    Decorator that register a callable as a validation action.

    An action will be called with a `pandas.DataFrame` object
    and return a new `pandas.DataFrame` object.

    An action needs to have an attribute `pretty_name`.

    :param obj: the callable to register as an action
    :return: the same as `obj`.
    """
    global _REGISTERED_ACTIONS
    if not hasattr(obj, "pretty_name"):
        raise ValueError("A validation action needs to have an attribue 'pretty_name'")
    if not callable(obj):
        raise ValueError("A validation action needs to be callable")
    _REGISTERED_ACTIONS[obj.pretty_name] = obj
    return obj


def list_actions() -> None:
    """List all available actions in a nice table"""
    dict_ = defaultdict(list)
    for name, obj in _REGISTERED_ACTIONS.items():
        dict_["name"].append(name)
        optionals = []
        requireds = []
        kwargs = {}
        for field in fields(obj):
            if "MISSING" in repr(field.default):
                requireds.append(f"{field.name} ({field.type})")
                kwargs[field.name] = ""
            else:
                optionals.append(f"{field.name} ({field.type})")
        dict_["required arguments"].append(", ".join(requireds))
        dict_["optional arguments"].append(", ".join(optionals))
        dict_["description"].append(str(obj(**kwargs)))
    print(pd.DataFrame(dict_).to_string(index=False, max_colwidth=100))


def create_action(pretty_name: str, *args: Any, **kwargs: Any) -> ActionType:
    """
    Create an action that can be called

    :param pretty_name: the name of the action
    :return: the instantiated actions
    """
    global _REGISTERED_ACTIONS
    if pretty_name not in _REGISTERED_ACTIONS:
        # Check for pretty_name variance...
        # We can have the same action more than once, hopefuly with different parameters...
        for action_pretty_name, reg_action in _REGISTERED_ACTIONS.items():
            if pretty_name.startswith(action_pretty_name):
                return reg_action(*args, **kwargs)
        raise KeyError(f"Unknown action: {pretty_name}")
    return _REGISTERED_ACTIONS[pretty_name](*args, **kwargs)


class _DataFrameApplier:
    def __init__(self) -> None:
        self.max_workers: Optional[int] = None
        tqdm.pandas()

    def __call__(self, data: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
        if self.max_workers is None:
            return data.swifter.apply(*args, **kwargs)
        if self.max_workers == 1:
            return data.apply(*args, **kwargs)
        return data.swifter.set_npartitions(self.max_workers).apply(*args, **kwargs)


global_apply = _DataFrameApplier()


class ReactionActionMixIn:
    """Mixin class with standard routines for splitting and joining reaction SMILES"""

    def join_lists(
        self,
        reactants_list: List[str],
        reagents_list: List[str],
        products_list: List[str],
    ) -> str:
        """
        Join list of components into a reaction SMILES

        :param reactants_list: the list of reactant SMILES
        :param reagents_list: the list of reagent SMILES
        :param products_list: the list of product SMILES
        :return: the concatenated reaction SMILES
        """
        return self.join_smiles(
            self._join_list(reactants_list),
            self._join_list(reagents_list),
            self._join_list(products_list),
        )

    def join_smiles(self, reactants: str, reagents: str, products: str) -> str:
        """
        Join component SMILES into a reaction SMILES

        :param reactants: the reactant SMILES
        :param reagents: the reagent SMILES
        :param products: the product SMILES
        :return: the concatenated reaction SMILES
        """
        return ">".join([reactants, reagents, products])

    def split_lists(self, row: pd.Series) -> Tuple[List[str], List[str], List[str]]:
        """
        Split a reaction SMILES into list of component SMILES

        :param row: the row with the SMILES
        :return: the list of SMILES of the components
        """
        reactants, reagents, products = self.split_smiles(row)

        reactants_list = split_smiles_from_reaction(reactants)
        reagents_list = split_smiles_from_reaction(reagents)
        products_list = split_smiles_from_reaction(products)
        return reactants_list, reagents_list, products_list

    def split_smiles(self, row: pd.Series) -> Tuple[str, str, str]:
        """
        Split a reaction SMILES into components SMILES

        :param row: the row with the SMILES
        :return: the SMILES of the components
        """
        return row[self.in_column].split(">")

    @staticmethod
    def _join_list(list_: List[str]) -> str:
        return ".".join([f"({item})" if "." in item else item for item in list_])
