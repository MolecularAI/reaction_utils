import json

import pytest

from rxnutils.chem.reaction import ChemicalReaction
from rxnutils.chem.reaction import ReactionException


@pytest.fixture
def load_templates(shared_datadir):
    filename = str(shared_datadir / "sample_templates.json")
    with open(filename, "r") as fileobj:
        return json.load(fileobj)


def test_create_reaction():
    rxn1 = ChemicalReaction(
        "[CH3:1]I.[CH3:6][N:5]1[CH2:4][CH2:3][NH:2][CH2:31][CH2:16]1"
        ">O=C(O)O.[Na+].[Na+]>"
        "[CH3:1][N:2]1[CH2:3][CH2:4][N:5]([CH3:6])[CH2:16][CH2:31]1"
    )

    rxn2 = ChemicalReaction(
        "[CH3:1]I.[CH3:6][N:5]1[CH2:4][CH2:3][NH:2][CH2:31][CH2:16]1."
        "[O:90]=[C:91]([OH:92])[OH:93].[Na+].[Na+]"
        ">>[CH3:1][N:2]1[CH2:3][CH2:4][N:5]([CH3:6])[CH2:16][CH2:31]1"
    )

    rxn3 = ChemicalReaction(
        "[CH3:1]I.[CH3:6][N:5]1[CH2:4][CH2:3][NH:2][CH2:31][CH2:16]1."
        "O=C(O)O.[Na+].[Na+]>"
        ">[CH3:1][N:2]1[CH2:3][CH2:4][N:5]([CH3:6])[CH2:16][CH2:31]1",
        clean_smiles=False,
    )

    expected_agents = [
        "OC(O)=O.[Na+].[Na+]",
        "O=C(O)O.[Na+].[Na+]",
        "O=C([OH])[OH].[Na+].[Na+]",
    ]
    assert rxn1.agents_smiles in expected_agents
    assert rxn2.agents_smiles in expected_agents
    assert rxn3.agents_smiles == ""

    assert rxn2.reactants_smiles == rxn1.reactants_smiles
    assert rxn2.reactants_smiles != rxn3.reactants_smiles
    assert len(rxn2.reactants) == len(rxn2.reactants) == 2
    assert len(rxn3.reactants) == 5

    assert rxn1.products_smiles == rxn2.products_smiles == rxn3.products_smiles


@pytest.mark.parametrize(
    ("rsmi", "expected"),
    [
        (
            "[CH3:1]I.[CH3:6][N:5]1[CH2:4][CH2:3][NH:2][CH2:31][CH2:16]1>>"
            "[CH3:1][N:2]1[CH2:3][CH2:4][N:5]([CH3:6])[CH2:16][CH2:31]1",
            True,
        ),
        (
            "[CH3:1]I.[CH3:6][N:5]1[CH2:4][CH2:3][NH:2][CH2:31][CH2:16]1>>",
            False,
        ),
    ],
)
def test_completeness(rsmi, expected):
    rxn = ChemicalReaction(rsmi)

    assert rxn.is_complete() == expected


@pytest.mark.parametrize(
    ("rsmi", "expected"),
    [
        (
            "[CH3:1]I.[CH3:6][N:5]1[CH2:4][CH2:3][NH:2][CH2:31][CH2:16]1>>"
            "[CH3:1][N:2]1[CH2:3][CH2:4][N:5]([CH3:6])[CH2:16][CH2:31]1",
            False,
        ),
        (
            "[CH3:1]I.[CH3:6][N:5]1[CH2:4][CH2:3][NH:2][CH2:31][CH2:16]1>>"
            "[CH3:1]I.[CH3:6][N:5]1[CH2:4][CH2:3][NH:2][CH2:31][CH2:16]1",
            True,
        ),
    ],
)
def test_nochangeness(rsmi, expected):
    rxn = ChemicalReaction(rsmi)

    assert rxn.no_change() == expected


@pytest.mark.parametrize(
    ("rsmi", "expected"),
    [
        (
            "[CH3:1]I.[CH3:6][N:5]1[CH2:4][CH2:3][NH:2][CH2:31][CH2:16]1>>"
            "[CH3:1][N:2]1[CH2:3][CH2:4][N:5]([CH3:6])[CH2:16][CH2:31]1",
            True,
        ),
        (
            "[CH3:1]I.[CH3:6][N:5]1[CH2:4][CH2:3][NH:2][CH2:31][CH2:16]1>C(C(CC)(O[Mg+2])C)C>"
            "[CH3:1]I.[CH3:6][N:5]1[CH2:4][CH2:3][NH:2][CH2:31][CH2:16]1",
            False,
        ),
    ],
)
def test_sanitization(rsmi, expected):
    rxn = ChemicalReaction(rsmi)

    assert rxn.sanitization_check() == expected


def test_no_template_creation(load_templates):
    # Ommit last test, raises ReactionException
    for record in load_templates[:-1]:
        rxn = ChemicalReaction(record["rsmi"])

        with pytest.raises(ValueError):
            _ = rxn.canonical_template.smarts

        with pytest.raises(ValueError):
            _ = rxn.retro_template.smarts


@pytest.mark.parametrize(
    ("radius", "expected"),
    [
        (
            0,
            "[CH3;D1;+0:1]-[N;H0;D3;+0:2]>>I-[CH3;D1;+0:1].[NH;D2;+0:2]",
        ),
        (
            1,
            "[C:2]-[N;H0;D3;+0:3](-[C:4])-[CH3;D1;+0:1]>>I-[CH3;D1;+0:1].[C:2]-[NH;D2;+0:3]-[C:4]",
        ),
    ],
)
def test_template_creation_different_radius(radius, expected):
    rxn_smiles = (
        "[CH3:1]I.[CH3:6][N:5]1[CH2:4][CH2:3][NH:2][CH2:31][CH2:16]1>"
        ">[CH3:1][N:2]1[CH2:3][CH2:4][N:5]([CH3:6])[CH2:16][CH2:31]1"
    )
    rxn = ChemicalReaction(rxn_smiles)

    rxn.generate_reaction_template(radius=radius)
    assert rxn.retro_template.smarts == expected


def test_template_creation(make_template_dataframe):
    failures = []
    # Ommit last test, no RSMI
    for _, record in make_template_dataframe.iloc[:-1].iterrows():
        rxn = ChemicalReaction(record["rsmi"], clean_smiles=False)
        rxn.generate_reaction_template()

        if record["retrotemplate"] != rxn.retro_template.smarts:
            failures.append(
                (record["rsmi"], rxn.retro_template.smarts, record["retrotemplate"])
            )

    if failures:
        print(
            "\n"
            + "\n\n".join(
                f"{failed[0]}\t{failed[1]}\t{failed[2]}" for failed in failures
            )
        )
    assert len(failures) == 0


@pytest.mark.parametrize(
    ("rsmi", "expected"),
    [
        (
            "O[CH2:16][CH2:15][CH2:14][CH:13]([c:11]1[c:10]2[c:5]([n:4][c:3]([C:2]([F:1])"
            "([F:22])[F:23])[cH:12]1)[c:6]([C:18]([F:19])([F:20])[F:21])[cH:7][cH:8][cH:9]2)[OH:17]"
            ">>[F:1][C:2]([c:3]1[n:4][c:5]2[c:6]([C:18]([F:19])([F:20])[F:21])[cH:7][cH:8]"
            "[cH:9][c:10]2[c:11]([CH:13]2[CH2:14][CH2:15][CH2:16][O:17]2)[cH:12]1)([F:22])[F:23]",
            False,
        ),
        (
            "[CH3:1]I.[CH3:6][N:5]1[CH2:4][CH2:3][NH:2][CH2:31][CH2:16]1>"
            ">[CH3:1][N:2]1[CH2:3][CH2:4][N:5]([CH3:6])[CH2:16][CH2:31]1",
            True,
        ),
    ],
)
def test_canonical_template_validation(rsmi, expected):
    rxn = ChemicalReaction(
        rsmi,
        clean_smiles=False,
    )
    rxn.generate_reaction_template()

    assert rxn.canonical_template_generate_outcome() == expected


@pytest.mark.parametrize(
    ("rsmi", "valid_template", "selectivity"),
    [
        (
            "[Cl:1][c:3]1[n:4][cH:5][cH:6][c:7]2[cH:8][cH:9][cH:10][cH:11][c:12]12>Cc1cc2c([N+](=O)[O-])cccc2cn1>[ClH:1]",
            False,
            0,
        ),
        (
            "[CH3:1]I.[CH3:6][N:5]1[CH2:4][CH2:3][NH:2][CH2:31][CH2:16]1>"
            ">[CH3:1][N:2]1[CH2:3][CH2:4][N:5]([CH3:6])[CH2:16][CH2:31]1",
            True,
            1,
        ),
    ],
)
def test_retro_template_validation(rsmi, valid_template, selectivity):
    rxn = ChemicalReaction(
        rsmi,
        clean_smiles=False,
    )
    rxn.generate_reaction_template()

    assert rxn.retro_template_generate_outcome() == valid_template
    assert rxn.retro_template_selectivity() == selectivity


def test_timedout_template_creation():
    rsmi = (
        "Cc1c(C)c(S(=O)(=O)NC(=N)NCCC[C@H](NC(=O)[C@@H]2CCCN2C(=O)[C@H](CCC(=O)NC(c2ccccc2)(c2ccccc2)"
        "c2ccccc2)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CCCCNC(=O)OC(C)(C)C)NC(=O)[C@H](C)NC(=O)[C@@H]2CCCN2C"
        "(=O)[C@@H]2CCCN2C(=O)[C@H](CCCCNC(=O)OC(C)(C)C)NC(=O)[C@H](CCCCNC(=O)OC(C)(C)C)NC(=O)[C@H]"
        "(COC(C)(C)C)NC(=O)[C@H](CCC(=O)OC(C)(C)C)NC(=O)[C@H](CCCCNC(=O)OC(C)(C)C)NC(=O)[C@H](CCCNC(=N)"
        "NS(=O)(=O)c2c(C)c(C)c3c(c2C)CCC(C)(C)O3)NC(=O)[C@H](CCC(=O)NC(c2ccccc2)(c2ccccc2)c2ccccc2)NC(=O)"
        "[C@H](CCC(=O)NC(c2ccccc2)(c2ccccc2)c2ccccc2)NC(=O)[C@@H](NC(=O)[C@H](CCCNC(=N)NS(=O)(=O)c2c(C)c"
        "(C)c3c(c2C)CCC(C)(C)O3)NC(=O)[C@H](CCC(=O)NC(c2ccccc2)(c2ccccc2)c2ccccc2)NC(=O)[C@H](Cc2cn"
        "(C(=O)OC(C)(C)C)cn2)NC(=O)[C@H](CCC(=O)OC(C)(C)C)NC(=O)[C@@H]2CCCN2C(=O)[C@H](COC(C)(C)C)NC(=O)"
        "[C@H](CC(C)C)NC(=O)[C@H](Cc2ccccc2)NC(=O)[C@H](COC(c2ccccc2)(c2ccccc2)c2ccccc2)NC(=O)[C@H]"
        "(COC(C)(C)C)N[C:3]([CH2:2][NH:1][C:5](=[O:6])[O:7][C:8]([CH3:9])([CH3:10])[CH3:11])=[O:4])C(C)C)C(=O)O)"
        "c(C)c2c1OC(C)(C)CC2.Cc1c(C)c(S(=O)(NC(=N)NCCC[C@H](NC(=O)OCC2c3ccccc3-c3ccccc32)C(=O)O)=[O:12])c(C)c2c1O"
        "C(C)(C)CC2>>[NH:1]([CH2:2][C:3]([OH:4])=[O:12])[C:5](=[O:6])[O:7][C:8]([CH3:9])([CH3:10])[CH3:11]"
    )
    rxn = ChemicalReaction(rsmi)
    # Capture ReactionException
    with pytest.raises(
        ReactionException, match="Template generation failed with message: Timed out"
    ):
        rxn.generate_reaction_template()
