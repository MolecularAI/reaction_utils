import json

import pandas as pd

from rxnutils.pipeline.actions.reaction_props import (
    CountComponents,
    CountElements,
    HasUnsanitizableReactants,
    ProductAtomMappingStats,
    ProductSize,
    PseudoReactionHash,
    ReactantProductAtomBalance,
    ReactantSize,
    SmilesLength,
    SmilesSanitizable,
    RingBondMade,
    RingMadeSize,
    RingNumberChange,
)
from rxnutils.pipeline.actions.reaction_mod import RemoveUnsanitizable
from rxnutils.pipeline.base import global_apply

global_apply.max_workers = 1


def test_count_components(make_reaction_dataframe):
    action = CountComponents(in_column="rsmi")
    df = make_reaction_dataframe

    df2 = action(df)

    assert df2["NReactants"].to_list() == [3, 2]
    assert df2["NMappedReactants"].to_list() == [1, 2]
    assert df2["NReagents"].to_list() == [2, 1]
    assert df2["NMappedReagents"].to_list() == [1, 0]
    assert df2["NProducts"].to_list() == [1, 1]
    assert df2["NMappedProducts"].to_list() == [1, 1]


def test_count_elements(make_reaction_dataframe):
    action = CountElements(in_column="rsmi")
    df = make_reaction_dataframe

    df2 = action(df)

    elements1 = json.loads(df2["ElementCount"].iloc[0])
    assert elements1 == {"16": 1, "17": 1, "6": 21, "8": 7}

    elements2 = json.loads(df2["ElementCount"].iloc[1])
    assert elements2 == {"16": 1, "17": 1, "6": 6, "8": 3}


def test_product_atom_mapping_stats(make_reaction_dataframe):
    action = ProductAtomMappingStats(in_column="rsmi")
    df = make_reaction_dataframe

    df2 = action(df)

    assert df2["UnmappedProdAtoms"].to_list() == [0, 4]
    assert df2["WidowAtoms"].to_list() == [1, 0]


def test_product_size(make_reaction_dataframe):
    action = ProductSize(in_column="rsmi")
    df = make_reaction_dataframe

    df2 = action(df)

    assert df2["ProductSize"].to_list() == [25, 9]


def test_pseudo_hash(make_reaction_dataframe):
    action = PseudoReactionHash(in_column="rsmi")
    df = make_reaction_dataframe

    df2 = action(df)

    assert ">>" not in df2["PseudoHash"].iloc[0]
    assert df2["PseudoHash"].iloc[0].count(".") == 2
    assert (
        df2["PseudoHash"].iloc[0]
        == "GWIBCCZNAYLLCD-UHFFFAOYSA-N.QAOWNCQODCNURD-UHFFFAOYSA-N>"
        "OFOBLEOULBTSOW-UHFFFAOYSA-L.UHOVQNZJYSORNB-UHFFFAOYSA-N>KOJXGMJOTRYLBD-UHFFFAOYSA-N"
    )


def test_reactant_product_balance(make_reaction_dataframe):
    action = ReactantProductAtomBalance(in_column="rsmi")
    df = make_reaction_dataframe

    df2 = action(df)

    assert df2["RxnAtomBalance"].to_list() == [25, -2]


def test_reactant_size(make_reaction_dataframe):
    action = ReactantSize(in_column="rsmi")
    df = make_reaction_dataframe

    df2 = action(df)

    assert df2["ReactantSize"].to_list() == [0, 11]


def test_smiles_length(make_reaction_dataframe):
    action = SmilesLength(in_column="rsmi")
    df = make_reaction_dataframe

    df2 = action(df)

    assert df2["SmilesLength"].to_list() == [434, 133]


def test_smiles_sanitizable(make_reaction_dataframe):
    action = SmilesSanitizable(in_column="new_column")
    df = make_reaction_dataframe
    df["new_column"] = df["rsmi"].apply(lambda row: row.split(">")[0])

    df2 = action(df)

    assert df2["SmilesSanitizable"].to_list() == [False, True]


def test_unsanitizable_reactants(make_reaction_dataframe):
    action1 = RemoveUnsanitizable(in_column="rsmi")
    action2 = HasUnsanitizableReactants(
        rsmi_column="rsmi", bad_columns=["BadMolecules"]
    )
    df = make_reaction_dataframe

    df2 = action1(df)
    df3 = action2(df2)

    assert df3["HasUnsanitizableReactants"].to_list() == [False, False]


def test_ring_actions(shared_datadir):
    df = pd.read_csv(shared_datadir / "example_ring_reactions.csv")
    action1 = RingBondMade(in_column="rsmi")
    action2 = RingMadeSize(in_column="rsmi")
    action3 = RingNumberChange(in_column="rsmi")

    df2 = action3(action2(action1(df)))

    assert df2["NRingChange"].to_list() == [1, 1]
    assert df2["RingBondMade"].to_list() == [True, True]
    assert df2["RingMadeSize"].to_list() == [5, 4]


def test_ring_actions_no_rings(make_reaction_dataframe):
    df = make_reaction_dataframe
    action1 = RingBondMade(in_column="rsmi")
    action2 = RingMadeSize(in_column="rsmi")
    action3 = RingNumberChange(in_column="rsmi")

    df2 = action3(action2(action1(df)))

    assert df2["NRingChange"].to_list() == [0, 0]
    assert df2["RingBondMade"].to_list() == [False, False]
    assert df2["RingMadeSize"].to_list() == [0, 0]
