import pandas as pd
import pytest

from rxnutils.pipeline.actions.reaction_props import CountComponents
from rxnutils.pipeline.actions.reaction_mod import (
    ReactantsToReagents,
    ReagentsToReactants,
    RemoveAtomMapping,
    RemoveExtraAtomMapping,
    RemoveUnsanitizable,
    RDKitRxnRoles,
    SplitReaction,
    RemoveUnchangedProducts,
    CONTRIB_INSTALLED,
)
from rxnutils.pipeline.base import global_apply

global_apply.max_workers = 1


def test_reactants_to_reagents(make_reaction_dataframe):
    action1 = ReactantsToReagents(in_column="rsmi")
    action2 = CountComponents(in_column="rsmi")
    action3 = CountComponents(in_column="RxnSmilesWithTrueReagents")
    df = make_reaction_dataframe

    df2 = action2(df)

    assert df2["NReactants"].to_list() == [3, 2]
    assert df2["NMappedReactants"].to_list() == [1, 2]
    assert df2["NReagents"].to_list() == [2, 1]
    assert df2["NMappedReagents"].to_list() == [1, 0]

    df2 = action1(df2)
    df2 = action3(df2)

    assert df2["NReactants"].to_list() == [1, 1]
    assert df2["NMappedReactants"].to_list() == [1, 1]
    assert df2["NReagents"].to_list() == [4, 2]
    assert df2["NMappedReagents"].to_list() == [1, 1]


def test_reagents_to_reactants(make_reaction_dataframe):
    action1 = ReagentsToReactants(in_column="rsmi")
    action2 = CountComponents(in_column="rsmi")
    action3 = CountComponents(in_column="RxnSmilesAllReactants")
    df = make_reaction_dataframe

    df2 = action2(df)

    assert df2["NReactants"].to_list() == [3, 2]
    assert df2["NMappedReactants"].to_list() == [1, 2]
    assert df2["NReagents"].to_list() == [2, 1]
    assert df2["NMappedReagents"].to_list() == [1, 0]

    df2 = action1(df2)
    df2 = action3(df2)

    assert df2["NReactants"].to_list() == [5, 3]
    assert df2["NMappedReactants"].to_list() == [2, 2]
    assert df2["NReagents"].to_list() == [0, 0]
    assert df2["NMappedReagents"].to_list() == [0, 0]


def test_remove_mapping(make_reaction_dataframe):
    action1 = RemoveAtomMapping(in_column="rsmi")
    action2 = CountComponents(in_column="rsmi")
    action3 = CountComponents(in_column="RxnSmilesNoAtomMap")
    df = make_reaction_dataframe

    df2 = action2(df)

    assert df2["NReactants"].to_list() == [3, 2]
    assert df2["NMappedReactants"].to_list() == [1, 2]
    assert df2["NReagents"].to_list() == [2, 1]
    assert df2["NMappedReagents"].to_list() == [1, 0]

    df2 = action1(df2)
    df2 = action3(df2)

    assert df2["NReactants"].to_list() == [3, 2]
    assert df2["NMappedReactants"].to_list() == [0, 0]
    assert df2["NReagents"].to_list() == [2, 1]
    assert df2["NMappedReagents"].to_list() == [0, 0]


def test_remove_extra_mapping(make_reaction_dataframe):
    action1 = RemoveExtraAtomMapping(in_column="rsmi")
    action2 = CountComponents(in_column="rsmi")
    action3 = CountComponents(in_column="RxnSmilesReassignedAtomMap")
    df = make_reaction_dataframe

    df2 = action2(df)

    assert df2["NReactants"].to_list() == [3, 2]
    assert df2["NMappedReactants"].to_list() == [1, 2]
    assert df2["NReagents"].to_list() == [2, 1]
    assert df2["NMappedReagents"].to_list() == [1, 0]

    df2 = action1(df2)
    df2 = action3(df2)

    assert df2["NReactants"].to_list() == [3, 2]
    assert df2["NMappedReactants"].to_list() == [1, 1]
    assert df2["NReagents"].to_list() == [2, 1]
    assert df2["NMappedReagents"].to_list() == [1, 0]


def test_remove_unchanged_product():
    smi1 = (
        "[CH3:1]I.[CH3:6][N:5]1[CH2:4][CH2:3][NH:2][CH2:31][CH2:16]1>>"
        "[CH3:1][N:2]1[CH2:3][CH2:4][N:5]([CH3:6])[CH2:16][CH2:31]1"
    )
    smi2 = (
        "[CH3:1]I.[CH3:6][N:5]1[CH2:4][CH2:3][NH:2][CH2:31][CH2:16]1>>"
        "[CH3:1]I.[CH3:6][N:5]1[CH2:4][CH2:3][NH:2][CH2:31][CH2:16]1"
    )
    df = pd.DataFrame({"rsmi": [smi1, smi2]})
    action1 = RemoveUnchangedProducts(in_column="rsmi")
    action2 = CountComponents(in_column="rsmi")
    action3 = CountComponents(in_column="RxnNoUnchangedProd")

    df2 = action2(df)

    assert df2["NReactants"].to_list() == [2, 2]
    assert df2["NReagents"].to_list() == [0, 0]
    assert df2["NProducts"].to_list() == [1, 2]

    df2 = action1(df2)
    df2 = action3(df2)

    assert df2["NReactants"].to_list() == [2, 2]
    assert df2["NReagents"].to_list() == [0, 0]
    assert df2["NProducts"].to_list() == [1, 0]


def test_remove_unsanitizable(make_reaction_dataframe):
    action = RemoveUnsanitizable(in_column="rsmi")
    df = make_reaction_dataframe

    df2 = action(df)

    assert df2["BadMolecules"].to_list() == ["C(C(CC)(O[Mg+2])C)C", ""]


@pytest.mark.xfail(
    not CONTRIB_INSTALLED, reason="RDKit Contrib folder is not installed"
)
def test_rdit_rxn_roles(make_reaction_dataframe):
    action1 = RDKitRxnRoles(in_column="rsmi")
    action2 = CountComponents(in_column="rsmi")
    action3 = CountComponents(in_column="RxnRoleAssigned")
    df = make_reaction_dataframe

    df2 = action2(df)

    assert df2["NReactants"].to_list() == [3, 2]
    assert df2["NMappedReactants"].to_list() == [1, 2]
    assert df2["NReagents"].to_list() == [2, 1]
    assert df2["NMappedReagents"].to_list() == [1, 0]

    df2 = action1(df2)
    df2 = action3(df2)

    assert df2["NReactants"].to_list() == [3, 2]
    assert df2["NMappedReactants"].to_list() == [2, 2]
    assert df2["NReagents"].to_list() == [2, 1]
    assert df2["NMappedReagents"].to_list() == [0, 0]


def test_split_reaction(make_reaction_dataframe):
    action = SplitReaction(
        in_column="rsmi", out_columns=["Reactants", "Reagents", "Products"]
    )
    df = make_reaction_dataframe

    df2 = action(df)

    assert [smi.count(".") + 1 for smi in df2["Reactants"]] == [3, 2]
    assert [smi.count(".") + 1 for smi in df2["Reagents"]] == [2, 1]
    assert [smi.count(".") + 1 for smi in df2["Products"]] == [1, 1]
