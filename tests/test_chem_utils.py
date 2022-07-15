import pytest
from rdkit import Chem

from rxnutils.chem.utils import (
    has_atom_mapping,
    remove_atom_mapping,
    remove_atom_mapping_template,
    neutralize_molecules,
    same_molecule,
    atom_mapping_numbers,
    split_smiles_from_reaction,
    reassign_rsmi_atom_mapping,
    get_special_groups,
)


@pytest.mark.parametrize(
    ("smiles", "is_smarts", "expected"),
    [
        ("CCCC(=O)O", False, False),
        ("CCCC(=O)[O:1]", False, True),
        ("C-[C;H0]", True, False),
        ("C-[C;H0:8]", True, True),
        ("", False, False),
        ("[C:1](C)(C)(C)(C)(C)", False, False),
    ],
)
def test_has_atom_mapping(smiles, is_smarts, expected):
    assert has_atom_mapping(smiles, is_smarts) == expected


@pytest.mark.parametrize(
    ("smiles", "is_smarts", "expected"),
    [
        ("CCCC(=O)O", False, "CCCC(=O)O"),
        ("CCCC(=O)[O:1]", False, "CCCC([O])=O"),
        ("C-[C;H0]", True, "C-[C&H0]"),
        ("C-[C;H0:8]", True, "C-[C&H0]"),
        ("", False, ""),
        ("[C:1](C)(C)(C)(C)(C)", False, "[C:1](C)(C)(C)(C)(C)"),
    ],
)
def test_remove_atom_mapping(smiles, is_smarts, expected):
    assert remove_atom_mapping(smiles, is_smarts) == expected


def test_remove_atom_mapping_from_template():
    template = "C-[C;H0:1]-[O:2]>>C-[C;H0:1].[O:2]"
    assert remove_atom_mapping_template(template) == "C-[C&H0]-O>>C-[C&H0].O"


def test_neutralize_molecules():
    smiles = ["C[N+](C)(C)CCC(=O)[O-]", "CP([O-])(=O)OC[NH3+]"]
    expected = ["C[N+](C)(C)CCC(=O)[O-]", "CP(=O)(O)OCN"]
    assert neutralize_molecules(smiles) == expected


@pytest.mark.parametrize(
    ("smiles1", "smiles2", "expected"),
    [
        ("CCCC(=O)O", "CCCC(=O)O", True),
        ("CCCC(=O)O", "CCCC(O)O", False),
        ("CCCC(=O)O", "[CH3:1]CCC(=O)O", True),
    ],
)
def test_same_molecule(smiles1, smiles2, expected):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    assert same_molecule(mol1, mol2) == expected


def test_atom_mapping_numbers():
    assert atom_mapping_numbers("[CH3:2]CCC(=O)[O:1]") == [2, 1]


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("AA.BB", ["AA", "BB"]),
        ("A(CC)A.BB", ["A(CC)A", "BB"]),
        ("A(CC)A.B(DD)B", ["A(CC)A", "B(DD)B"]),
        ("(A(CC)A.B(DD)B).PP", ["A(CC)A.B(DD)B", "PP"]),
        ("(A(CC)A.B(DD)B).P(R)P", ["A(CC)A.B(DD)B", "P(R)P"]),
        ("(A(CC)A.B(DD)B).(P(R)P.Q)", ["A(CC)A.B(DD)B", "P(R)P.Q"]),
        ("A(CC)A.B(DD)B.(P(R)P.Q)", ["A(CC)A", "B(DD)B", "P(R)P.Q"]),
        ("", []),
    ],
)
def test_split_smiles(smiles, expected):
    assert split_smiles_from_reaction(smiles) == expected


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        (
            "[C:7]-[C;H0:1]-[O:2]>>[C:7]-[C;H0:1].[O:2]",
            ("[C:7]-[C&H0:1]-[O:2]>>[C:7]-[C&H0:1].[O:2]"),
        ),
        (
            "[Br:99][CH2:11][CH2:12][Br:100].[N:1]#[C:2][CH2:3][c:4]1[c:5]([Br:6])[cH:7][cH:8][cH:9][cH:10]1"
            ">>[N:1]#[C:2][C:3]1([c:4]2[c:5]([Br:6])[cH:7][cH:8][cH:9][cH:10]2)[CH2:11][CH2:12]1",
            (
                "Br[C&H2:11][C&H2:12]Br.[N:1]#[C:2][C&H2:3][c:4]1[c:5](-,:[Br:6])[c&H1:7][c&H1:8][c&H1:9][c&H1:10]1"
                ">>[N:1]#[C:2][C:3]1(-,:[c:4]2[c:5](-,:[Br:6])[c&H1:7][c&H1:8][c&H1:9][c&H1:10]2)[C&H2:11][C&H2:12]1",
                "Br[C&H2:11][C&H2:12]Br.[N:1]#[C:2][C&H2:3][c:4]1[c:5]([Br:6])[c&H1:7][c&H1:8][c&H1:9][c&H1:10]1"
                ">>[N:1]#[C:2][C:3]1([c:4]2[c:5]([Br:6])[c&H1:7][c&H1:8][c&H1:9][c&H1:10]2)[C&H2:11][C&H2:12]1",
            ),
        ),
    ],
)
def test_reassign_atom_numbers(template, expected):
    assert reassign_rsmi_atom_mapping(template) in expected


def test_special_groups():
    # This test is rather pointless, we should test each and everyone of the special groups
    # if this should make sense
    mol = Chem.MolFromSmiles("c1ccccc1B(O)O")
    assert get_special_groups(mol)[0][1] == (6, 7, 8)
