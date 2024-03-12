import copy

import pytest
import pandas as pd

from rxnutils.routes.base import SynthesisRoute


def test_collect_reaction_smiles(synthesis_route):
    smiles = synthesis_route.reaction_smiles()
    assert len(smiles) == 2
    assert smiles[0] == "CO.Clc1ccccc1>>COc1ccccc1"


def test_atom_mapping(synthesis_route, setup_mapper):

    synthesis_route.assign_atom_mapping()

    smiles = synthesis_route.atom_mapped_reaction_smiles()
    assert len(smiles) == 2
    assert (
        smiles[0]
        == "[CH3:1][OH:2].[cH:6]1[cH:5][cH:4][c:3]([cH:8][cH:7]1)Cl>>[CH3:1][O:2][c:3]1[cH:4][cH:5][cH:6][cH:7][cH:8]1"
    )

    synthesis_route.assign_atom_mapping(only_rxnmapper=True, overwrite=True)

    smiles = synthesis_route.atom_mapped_reaction_smiles()
    assert (
        smiles[0]
        == "Cl[c:3]1[cH:4][cH:5][cH:6][cH:7][cH:8]1.[CH3:1][OH:2]>>[CH3:1][O:2][c:3]1[cH:4][cH:5][cH:6][cH:7][cH:8]1"
    )


def test_root_smiles(synthesis_route, setup_mapper):
    with pytest.raises(ValueError):
        SynthesisRoute({"smiles": "C"}).mapped_root_smiles

    with pytest.raises(ValueError):
        synthesis_route.mapped_root_smiles

    synthesis_route.assign_atom_mapping()
    assert (
        synthesis_route.mapped_root_smiles
        == "[CH3:1][O:2][c:3]1[cH:4][cH:5][cH:6][cH:7][cH:8]1"
    )


def test_reaction_data(synthesis_route, setup_mapper):
    synthesis_route.assign_atom_mapping()

    data = pd.DataFrame(synthesis_route.reaction_data())

    assert len(data) == 2
    assert data["classification"].to_list() == ["1.7.11", "10.1.2"]
    assert data["tree_depth"].to_list() == [1, 2]
    assert data["forward_step"].to_list() == [2, 1]


def test_remap(synthesis_route, setup_mapper):
    route1 = synthesis_route
    route2 = copy.deepcopy(synthesis_route)

    route1.assign_atom_mapping()
    route2.assign_atom_mapping(only_rxnmapper=True)

    old_reaction_smiles = route1.atom_mapped_reaction_smiles()

    route1.remap(route2)

    assert route1.atom_mapped_reaction_smiles() != old_reaction_smiles


def test_extract_chains(synthesis_route):
    complexity = {"COc1ccccc1": 5, "CO": 0, "Clc1ccccc1": 1, "c1ccccc1": 1, "Cl": 0}

    chains = synthesis_route.chains(lambda smi: complexity[smi])

    assert len(chains) == 1
    assert len(chains[0]) == 3
    assert chains[0][0]["smiles"] == "c1ccccc1"
    assert chains[0][0]["step"] == 0
    assert chains[0][0]["chain"] == "lls"
    assert chains[0][0]["type"] == "sm"
    assert chains[0][2]["step"] == 2
    assert chains[0][2]["chain"] == "lls"
    assert chains[0][2]["type"] == "target"

    # Faking the selection of another starting material
    complexity["Cl"] = 5

    chains = synthesis_route.chains(lambda smi: complexity[smi])

    assert len(chains) == 1
    assert len(chains[0]) == 3
    assert chains[0][0]["smiles"] == "Cl"
    assert chains[0][0]["step"] == 0
    assert chains[0][0]["chain"] == "lls"
    assert chains[0][0]["type"] == "sm"


def test_extract_branched_chains(branched_synthesis_route):
    complexity = {"COc1ccccc1": 5, "CO": 1, "Clc1ccccc1": 2, "c1ccccc1": 1, "Cl": 0}

    chains = branched_synthesis_route.chains(lambda smi: complexity.get(smi, 0))

    assert len(chains) == 2
    assert len(chains[0]) == 3
    assert len(chains[1]) == 3
    assert chains[0][0]["smiles"] == "c1ccccc1"

    assert chains[1][0]["smiles"] == "C"
    assert chains[1][0]["step"] == 0
    assert chains[1][0]["chain"] == "sub1"
    assert chains[1][0]["type"] == "sm"
    assert chains[1][2]["smiles"] == "COc1ccccc1"
    assert chains[1][2]["step"] == 2
    assert chains[1][2]["chain"] == "sub1"
    assert chains[1][2]["type"] == "branch"
