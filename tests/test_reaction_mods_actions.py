import pandas as pd
import pytest

from rxnutils.chem.disconnection_sites.tag_converting import smiles_tokens
from rxnutils.chem.utils import split_rsmi
from rxnutils.pipeline.actions.reaction_mod import (
    CONTRIB_INSTALLED,
    AtomMapTagDisconnectionSite,
    ConvertAtomMapDisconnectionTag,
    DesaltMolecules,
    IsotopeInfo,
    NameRxn,
    RDKitRxnRoles,
    ReactantsToReagents,
    ReagentsToReactants,
    RemoveAtomMapping,
    RemoveExtraAtomMapping,
    RemoveUnchangedProducts,
    RemoveUnsanitizable,
    RxnMapper,
    SplitReaction,
)
from rxnutils.pipeline.actions.reaction_props import CountComponents
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

    assert df2["BadMolecules"].to_list() == ["C(C(CC)(O[Mg+2])C)CN(C)(C)(C)", ""]


@pytest.mark.xfail(not CONTRIB_INSTALLED, reason="RDKit Contrib folder is not installed")
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
    action = SplitReaction(in_column="rsmi", out_columns=["Reactants", "Reagents", "Products"])
    df = make_reaction_dataframe

    df2 = action(df)

    assert [smi.count(".") + 1 for smi in df2["Reactants"]] == [3, 2]
    assert [smi.count(".") + 1 for smi in df2["Reagents"]] == [2, 1]
    assert [smi.count(".") + 1 for smi in df2["Products"]] == [1, 1]


def test_isotope_info():
    # Don't have to be sensible SMILES, the action works on strings
    df = pd.DataFrame(
        {
            "rsmi": [
                "c1ccccc1",
                "[13C]c1ccccc1",
                "[13CH2]c1ccccc1",
                "[2H:3]",
                "[2H-:3][13CH2]",
            ]
        }
    )
    action = IsotopeInfo(in_column="rsmi")

    df2 = action(df)

    assert pd.isna(df2["Isotope"]).to_list() == [
        True,
        False,
        False,
        False,
        False,
    ]
    assert df2["Isotope"].to_list()[1:] == ["13C", "13C", "2H", "2H"]
    assert df2["RxnSmilesWithoutIsotopes"].to_list() == [
        "c1ccccc1",
        "[C]c1ccccc1",
        "[CH2]c1ccccc1",
        "[H:3]",
        "[H-:3][CH2]",
    ]


def test_disconnection_tagging(shared_datadir):

    df = pd.read_csv(shared_datadir / "mapped_tests_reactions.csv", sep="\t")

    action_atom_map_tag = AtomMapTagDisconnectionSite(in_column="RxnmapperRxnSmiles")
    action_convert_tag = ConvertAtomMapDisconnectionTag()

    df_atom_map_tag = action_atom_map_tag(df)
    df_tag = action_convert_tag(df_atom_map_tag)

    df_ground_truth = pd.Series(["Cl!c!1ccccc1", "CO!c!1ccccc1"], name="products_tagged")

    assert df_ground_truth.equals(df_tag["products_tagged"])


def test_smiles_tokenization_unknown_token_error(shared_datadir):

    df = pd.read_csv(shared_datadir / "mapped_tests_reactions.csv", sep="\t")

    action_atom_map_tag = AtomMapTagDisconnectionSite(in_column="RxnmapperRxnSmiles")
    df_atom_map_tag = action_atom_map_tag(df)

    product_atom_map_tagged = df_atom_map_tag["products_atom_map_tagged"].values[0]

    with pytest.raises(AssertionError):
        smiles_tokens(product_atom_map_tagged + "{")


def test_converting_no_atom_map_tag(shared_datadir):

    df = pd.read_csv(shared_datadir / "mapped_tests_reactions.csv", sep="\t")
    df["products"] = [split_rsmi(rxn)[-1] for rxn in df.smiles]

    action_convert_tag = ConvertAtomMapDisconnectionTag(in_column="products")

    df_tag = action_convert_tag(df)

    df_ground_truth = pd.Series(["", ""], name="products_tagged")

    assert df_ground_truth.equals(df_tag["products_tagged"])


def test_desalting():
    smi1 = "OCC.(C.[Na+].[Cl-])>>OC(=O)CC"
    smi2 = "OCC>>OC(=O)CC"
    smi3 = "OCC.O>>OC(=O)CC"
    df = pd.DataFrame({"rsmi": [smi1, smi2, smi3]})
    action = DesaltMolecules(in_column="rsmi")

    df2 = action(df)

    assert df2["RxnDesalted"].to_list() == [
        "CCO.C>>CCC(=O)O",
        "CCO>>CCC(=O)O",
        "CCO>>CCC(=O)O",
    ]

    action.keep_something = True

    df2 = action(df)

    assert df2["RxnDesalted"].to_list() == [
        "CCO.C>>CCC(=O)O",
        "CCO>>CCC(=O)O",
        "CCO.O>>CCC(=O)O",
    ]


def test_namerxn_fail(make_reaction_dataframe):
    df = make_reaction_dataframe
    action1 = RemoveAtomMapping(in_column="rsmi", out_column="rsmi_nomap")
    action2 = NameRxn(in_column="rsmi_nomap")

    df1 = action1(df)
    with pytest.raises(FileNotFoundError, match="No such file or directory: 'namerxn'"):
        df1 = action2(df1)


def test_namerxn(make_reaction_dataframe, mocker):
    df = make_reaction_dataframe
    action1 = RemoveAtomMapping(in_column="rsmi", out_column="rsmi_nomap")
    action2 = NameRxn(in_column="rsmi_nomap")

    # Mock namerxn call
    # Use the mocker fixture to mock subprocess.call
    mock_suprocess_call = mocker.patch("rxnutils.pipeline.actions.reaction_mod.subprocess.call")
    # Set up the mock to return a specific response
    mock_result = mocker.MagicMock()
    mock_suprocess_call.return_value = mock_result

    df1 = action1(df)
    with pytest.raises(FileNotFoundError, match="Could not produce namerxn output."):
        df1 = action2(df1)

    mock_suprocess_call.assert_called_once()

    # Mock os.path action
    mock_os_path_exists = mocker.patch("rxnutils.pipeline.actions.reaction_mod.os.path.exists")
    mock_os_path_getsize = mocker.patch("rxnutils.pipeline.actions.reaction_mod.os.path.getsize")
    # Set up the mock to return a specific response
    mock_os_path_exists.return_value = True
    mock_os_path_getsize.return_value = 1

    # Mock pd.read_csv
    mock_pd_read_csv = mocker.patch("rxnutils.pipeline.actions.reaction_mod.pd.read_csv")
    # Set up the mock to return a specific response
    mock_result = pd.DataFrame(data={"NextMoveRxnSmiles": list(df["rsmi"]), "NMC": ["1.1", "1.2"]})
    mock_pd_read_csv.return_value = mock_result

    df1 = action2(df1)

    mock_os_path_exists.assert_called_once()
    mock_os_path_getsize.assert_called_once()

    assert list(df1.columns) == ["rsmi", "rsmi_nomap", "NextMoveRxnSmiles", "NMC"]
    assert df1.shape == (2, 4)
    assert df1["NMC"].to_list() == ["1.1", "1.2"]
    assert df1["NextMoveRxnSmiles"].to_list() == df1["rsmi"].to_list()


def test_rxnmapper(make_reaction_dataframe, mocker):
    df = make_reaction_dataframe
    action1 = RemoveAtomMapping(in_column="rsmi", out_column="rsmi_nomap")
    action2 = RxnMapper(in_column="rsmi_nomap")

    df1 = action1(df)

    # Mock rnxmapper call
    # Use the mocker fixture to mock subprocess.call
    mock_suprocess = mocker.patch("rxnutils.pipeline.actions.reaction_mod.subprocess.check_output")
    # Set up the mock to return a specific response
    mock_result = mocker.MagicMock()
    mock_suprocess.return_value = mock_result

    # Mock pd.read_csv
    mock_pd_read_csv = mocker.patch("rxnutils.pipeline.actions.reaction_mod.pd.read_csv")
    # Set up the mock to return a specific response
    mock_result = pd.DataFrame(data={"mapped_rxn": list(df["rsmi"]), "confidence": [1.0, 1.0]})
    mock_pd_read_csv.return_value = mock_result

    df1 = action2(df1)

    assert list(df1.columns) == ["rsmi", "rsmi_nomap", "RxnmapperRxnSmiles"]
    assert df1.shape == (2, 3)
    assert df1["RxnmapperRxnSmiles"].to_list() == df1["rsmi"].to_list()
