import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from rxnutils.chem.augmentation import single_reactant_augmentation
from rxnutils.chem.utils import (
    atom_mapping_numbers,
    canonicalize_tautomer,
    get_mol_weight,
    get_special_groups,
    get_symmetric_sites,
    has_atom_mapping,
    is_valid_mol,
    neutralize_molecules,
    reaction_centres,
    reassign_rsmi_atom_mapping,
    recreate_rsmi,
    remove_atom_mapping,
    remove_atom_mapping_template,
    remove_stereochemistry,
    same_molecule,
    split_rsmi,
    split_smiles_from_reaction,
)


def test_canonicalize_tautomer():
    input_smiles = ["c1nc2cncnc2[nH]1", "c1ncc2[nH]cnc2n1", "c1cc[nH]cn1"]
    expected_outputs = ["c1ncc2[nH]cnc2n1", "c1ncc2[nH]cnc2n1", "c1cc[nH]cn1"]

    for smiles, expected in zip(input_smiles, expected_outputs):
        assert canonicalize_tautomer(smiles) == expected


def test_remove_stereochemistry() -> None:
    input_smiles = ["C[C@@H]CCO[C@H]", "CC(=O)C", "C[C@H]CCO[C@@H]", "C[C@H]CCOC"]

    expected_output = [
        "[CH]OCC[CH]C",
        "CC(C)=O",
        "[CH]OCC[CH]C",
        "C[CH]CCOC",
    ]
    for smiles, expected in zip(input_smiles, expected_output):
        output = remove_stereochemistry(smiles)
        assert output == expected

    smiles_invalid = "not-a-smiles"
    assert smiles_invalid == remove_stereochemistry(smiles_invalid)


def test_is_valid_mol():
    assert is_valid_mol("CCO")
    assert not is_valid_mol("C!O")


def test_get_mol_weight():
    smiles = "CCCCCC(=O)CCCC"
    smiles_invalid = "not-a-smiles"
    assert round(get_mol_weight(smiles), 2) == 156.15
    assert get_mol_weight(smiles_invalid) is None


@pytest.mark.parametrize(
    ("smiles", "atoms_inds", "expected"),
    [
        ("Oc1cccc(O)c1", [0], [[0, 6]]),
        ("Oc1cccc(O)c1", [2], [[2, 4]]),
        ("Oc1cccc(O)c1", [0, 2], [[0, 6], [2, 4]]),
        ("Oc1cccc(O)c1", [0, 2, 6], [[0, 6], [2, 4]]),
        ("Oc1cccc(O)c1", [3], []),
    ],
)
def test_get_symmetric_sites(smiles, atoms_inds, expected):
    mol = Chem.MolFromSmiles(smiles)
    assert get_symmetric_sites(mol, atoms_inds) == expected


def test_get_symmetric_sites_error():
    mol = Chem.MolFromSmiles("Oc1cccc(O)c1")
    with pytest.raises(ValueError, match="At least one candidate atom-idx is out of bounds"):
        get_symmetric_sites(mol, [13])


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


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        (
            "[ClH:1].[cH:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1>>[Cl:1][c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1",
            ((0,), (0,)),
        ),
        (
            "Br[CH2:9][CH2:10][CH2:11]Br.[CH3:1][CH2:2][O:3][C:4](=[O:5])[CH2:6][C:7]#[N:8]>>[CH3:1][CH2:2][O:3][C:4](=[O:5])[C:6]1([C:7]#[N:8])[CH2:9][CH2:10][CH2:11]1",
            ((1, 3), (5,)),
        ),
    ],
)
def test_reaction_centers(smiles, expected):
    rxn = AllChem.ReactionFromSmarts(smiles, useSmiles=True)
    rxn.Initialize()

    assert reaction_centres(rxn) == expected


@pytest.mark.parametrize(
    ("smiles", "classification", "expected"),
    [
        (
            "A.B>>C",
            "",
            "A.B>>C",
        ),
        (
            "A.B>>C",
            "0.0",
            "A.B>>C",
        ),
        (
            "A>>C",
            "0.0",
            "A>>C",
        ),
        (
            "A>>C",
            "10.1.1",
            "Br.A>>C",
        ),
        (
            "A>>C",
            "10.1.2 Chlorination",
            "Cl.A>>C",
        ),
    ],
)
def test_single_reactant_agumentation(smiles, classification, expected):
    assert single_reactant_augmentation(smiles, classification) == expected


@pytest.mark.parametrize(
    ("rsmi", "expected_components"),
    [
        (
            "A.B>>C",
            ("A.B", "", "C"),
        ),
        (
            "A>B>C",
            ("A", "B", "C"),
        ),
        (
            "A->B>>C",
            ("A->B", "", "C"),
        ),
        (
            "A>B->C>D",
            ("A", "B->C", "D"),
        ),
    ],
)
def test_split_rsmi(rsmi, expected_components):
    reaction_components = split_rsmi(rsmi)
    assert reaction_components == expected_components


def test_split_rsmi_error():
    with pytest.raises(ValueError, match="Expected 3 reaction components but got 4 for 'A>B>C>D'"):
        _ = split_rsmi("A>B>C>D")

    with pytest.raises(ValueError, match="Expected 3 reaction components but got 2 for 'A>B'"):
        _ = split_rsmi("A>B")


@pytest.mark.parametrize(
    ("rsmi", "expected_rsmi"),
    [
        (
            "A.B>>C",
            "A.B>>C",
        ),
        (
            "(A.B).C>>D",
            "A.B.C>>D",
        ),
        (
            "(A->B.C).D>E.F>G",
            "A->B.C.D>E.F>G",
        ),
        (
            "A.(B.C).D>(E.F)>G",
            "A.B.C.D>E.F>G",
        ),
        (
            "(A.B).C>(E.F)>(D.G)",
            "A.B.C>E.F>D.G",
        ),
    ],
)
def test_recreate_rsmi(rsmi, expected_rsmi):
    new_rsmi = recreate_rsmi(rsmi)

    assert new_rsmi == expected_rsmi
