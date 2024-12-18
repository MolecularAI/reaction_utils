import pytest

from rxnutils.chem.cgr import CondensedGraphReaction
from rxnutils.chem.reaction import ChemicalReaction


@pytest.fixture
def rsmi2cgr():
    def wrapper(rsmi):
        rxn = ChemicalReaction(rsmi, clean_smiles=False)
        return CondensedGraphReaction(rxn)

    return wrapper


def test_create_cgr(rsmi2cgr):
    rsmi = (
        "[O:2]=[C:3]([OH:4])[CH2:5][CH2:6][CH2:7][O:8][c:9]1[cH:10][cH:11][cH:12][cH:23][cH:24]1.O=C1CCC(=O)N1[Cl:1]>"
        ">[Cl:1][c:10]1[c:9]([O:8][CH2:7][CH2:6][CH2:5][C:3](=[O:2])[OH:4])[cH:24][cH:23][cH:12][cH:11]1"
    )

    cgr = rsmi2cgr(rsmi)

    assert cgr.bonds_broken == 1
    assert cgr.bonds_formed == 1
    assert cgr.bonds_changed == 2
    assert cgr.total_centers == 5


def test_compare_cgrs(rsmi2cgr):
    rsmi1 = (
        "[O:2]=[C:3]([OH:4])[CH2:5][CH2:6][CH2:7][O:8][c:9]1[cH:10][cH:11][cH:12][cH:23][cH:24]1.O=C1CCC(=O)N1[Cl:1]"
        ">>[Cl:1][c:10]1[c:9]([O:8][CH2:7][CH2:6][CH2:5][C:3](=[O:2])[OH:4])[cH:24][cH:23][cH:12][cH:11]1"
    )
    rsmi2 = (
        "[O:2]=[C:3]([OH:4])[CH2:5][CH2:6][CH2:7][O:8][c:9]1[cH:10][cH:11][cH:12][cH:23][cH:24]1"
        ">>[Cl:1][c:10]1[c:9]([O:8][CH2:7][CH2:6][CH2:5][C:3](=[O:2])[OH:4])[cH:24][cH:23][cH:12][cH:11]1"
    )

    cgr1 = rsmi2cgr(rsmi1)
    cgr2 = rsmi2cgr(rsmi2)

    assert cgr1 == cgr1
    assert cgr1.distance_to(cgr1) == 0

    assert cgr1 != cgr2
    assert cgr1.distance_to(cgr2) == 2
    assert cgr2.distance_to(cgr1) == 2
