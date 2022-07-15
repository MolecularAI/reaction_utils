import pytest
from rdkit import Chem
from rdkit import RDLogger

from rxnutils.chem.template import TemplateMolecule

rd_logger = RDLogger.logger()
rd_logger.setLevel(RDLogger.CRITICAL)


@pytest.fixture
def create_mol():
    return Chem.MolFromSmarts(
        "[c:7]:[c;H0;D3;+0:6](:[c:8])-[c;H0;D3;+0:5]1:[n;H0;D2;+0:4]:[n;H0;D2;+0:1]:[n;H0;D2;+0:2]:[nH;D2;+0:3]:1"
    )


@pytest.mark.parametrize(
    ("smarts", "expected"),
    [("C", False), ("[#7;a]", True), ("[#7;a:1]", True), ("[#7;a;H0]", True)],
)
def test_fix_aromaticity(smarts, expected):
    mol = TemplateMolecule(smarts=smarts)
    mol.sanitize()

    assert not mol.atom_properties()["IsAromatic"][0]

    mol.fix_atom_properties()

    assert mol.atom_properties()["IsAromatic"][0] == expected


@pytest.mark.parametrize(
    ("smarts", "expected"),
    [("C", 0), ("[#7;+]", +1), ("[#7;+2:1]", +2), ("[#7;-;H0]", -1)],
)
def test_fix_formal_charge(smarts, expected):
    mol = TemplateMolecule(smarts=smarts)
    mol.sanitize()

    assert mol.atom_properties()["FormalCharge"][0] == 0

    mol.fix_atom_properties()

    assert mol.atom_properties()["FormalCharge"][0] == expected


@pytest.mark.parametrize(
    ("smarts", "hs_before", "hs_after"),
    [
        ("C", 0, 0),
        ("[NH]", 1, 1),
        ("[N;H1]", 0, 1),
        ("[N;H1:1]", 0, 1),
        ("[N;-;H1]", 0, 1),
    ],
)
def test_fix_explicit_hydrogens(smarts, hs_before, hs_after):
    mol = TemplateMolecule(smarts=smarts)
    mol.sanitize()

    assert mol.atom_properties()["NumExplicitHs"][0] == hs_before

    mol.fix_atom_properties()

    assert mol.atom_properties()["NumExplicitHs"][0] == hs_after


@pytest.mark.parametrize(
    ("smarts", "degree", "num_hs"),
    [("C", 0, 4), ("[C;D3]", 3, 1), ("[C;D3:1]", 3, 1), ("[C;D3;H0]", 3, 1)],
)
def test_fix_degree(smarts, degree, num_hs):
    mol = TemplateMolecule(smarts=smarts)
    mol.sanitize()
    assert mol.atom_properties()["Degree"][0] == 0

    mol.fix_atom_properties()

    assert mol.atom_properties()["Degree"][0] == 0
    assert mol.atom_properties()["comp degree"][0] == degree


def test_fix_atom_props_full_mol(create_mol):
    tmpl_mol = TemplateMolecule(create_mol)

    tmpl_mol.fix_atom_properties()

    explicit_hs_after = [atom.GetNumExplicitHs() for atom in tmpl_mol.atoms()]
    assert explicit_hs_after == [0, 0, 0, 0, 0, 0, 0, 1]


def test_atom_invariants(create_mol):
    tmpl_mol = TemplateMolecule(create_mol)
    tmpl_mol.fix_atom_properties()

    assert tmpl_mol.atom_invariants() == [
        3539526920,
        440107023,
        3539526920,
        607308987,
        454492884,
        454492884,
        454492884,
        2869507918,
    ]


def test_fingerprint_bits(create_mol):
    tmpl_mol = TemplateMolecule(create_mol)

    assert tmpl_mol.fingerprint_bits(0) == set(tmpl_mol.atom_invariants())

    assert tmpl_mol.fingerprint_bits(1) == {
        239760326,
        241552887,
        440107023,
        454492884,
        607308987,
        1303532158,
        2343775396,
        2567109925,
        2657007424,
        2807016243,
        2869507918,
        3539526920,
    }


def test_hash_from_smiles(create_mol):
    tmpl_mol = TemplateMolecule(create_mol)

    expected = "608ed1c582519649113267b68463d3446959f5dbff6b7f9a39751f32"
    assert tmpl_mol.hash_from_smiles() == expected


def test_hash_from_smarts(create_mol):
    tmpl_mol = TemplateMolecule(create_mol)

    expected = "2ca4d01cfffa9adcf724d9cc4cb498da8195a4f038b12278456e94f4"
    assert tmpl_mol.hash_from_smarts() == expected


def test_template_with_aromaticity():
    rd_mol = Chem.MolFromSmarts(
        "C-C-O-[C;H0;D3;+0:1](=[O;D1;H0:2])-[c:3](:[#7;a:4]):[#8;a:5]:[#7;a:6]"
    )
    tmpl_mol = TemplateMolecule(rd_mol)

    assert tmpl_mol.fingerprint_bits() != {}


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("C-[AlH3]", "C-[#13&H3]"),
        ("[C@&H1&D3&+0]", "[C@H;D3;+0]"),
    ],
)
def test_template_equality(first, second):
    mol1 = TemplateMolecule(smarts=first)
    mol2 = TemplateMolecule(smarts=second)

    assert mol1.fingerprint_bits() == mol2.fingerprint_bits()


def test_template_equality_chiral():
    mol1 = TemplateMolecule(smarts="C-S(=O)(=O)-O-[C@&H1&D3&+0](-C)-C")
    mol2 = TemplateMolecule(smarts="C-S(=O)(=O)-O-[C@H;D3;+0:1](-[C:2])-[C:3]")

    assert mol1.fingerprint_bits() != mol2.fingerprint_bits()
    assert mol1.fingerprint_bits(use_chirality=False) == mol2.fingerprint_bits(
        use_chirality=False
    )
