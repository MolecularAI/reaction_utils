""" Module containing routes to validate AiZynthFinder-like input dictionaries """

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, StringConstraints, ValidationError, conlist
from typing_extensions import Annotated

StrDict = Dict[str, Any]


class ReactionNode(BaseModel):
    """Node representing a reaction"""

    type: Annotated[str, StringConstraints(pattern=r"^reaction$")]
    children: List[MoleculeNode]


class MoleculeNode(BaseModel):
    """Node representing a molecule"""

    smiles: str
    type: Annotated[str, StringConstraints(pattern=r"^mol$")]
    children: Optional[conlist(ReactionNode, min_length=1, max_length=1)] = None


MoleculeNode.update_forward_refs()


def validate_dict(dict_: StrDict) -> None:
    """
    Check that the route dictionary is a valid structure

    :param dict_: the route as dictionary
    """
    try:
        MoleculeNode(**dict_, extra="ignore")
    except ValidationError as err:
        raise ValueError(f"Invalid input: {err.json()}")
