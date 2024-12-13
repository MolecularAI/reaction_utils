import copy

import pandas as pd
import pytest
from rdkit import Chem

from rxnutils.chem.utils import split_rsmi
from rxnutils.routes.base import SynthesisRoute


def test_collect_reaction_smiles(synthesis_route):
    smiles = synthesis_route.reaction_smiles()
    assert synthesis_route.nsteps == 2
    assert len(smiles) == 2
    assert smiles[0] == "CO.Clc1ccccc1>>COc1ccccc1"


def test_collect_reaction_smiles_augmented(augmentable_sythesis_route):
    smiles = augmentable_sythesis_route.reaction_smiles(augment=True)

    assert len(smiles) == 3
    assert smiles == [
        "Cl.c1ccccc1>>Clc1ccccc1",
        "Cc1ccccc1>>c1ccccc1",
        "Oc1ccccc1.C>>Cc1ccccc1",
    ]


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


def test_atom_mapping_no_namerxn(synthesis_route, setup_mapper_no_namerxn, recwarn):
    synthesis_route.assign_atom_mapping()

    assert len(recwarn) == 1
    assert "namerxn" in str(recwarn[0])

    data = pd.DataFrame(synthesis_route.reaction_data())

    assert len(data) == 2
    assert data["classification"].to_list() == ["0.0", "0.0"]
    assert (
        data["mapped_reaction_smiles"][0]
        == "Cl[c:3]1[cH:4][cH:5][cH:6][cH:7][cH:8]1.[CH3:1][OH:2]>>[CH3:1][O:2][c:3]1[cH:4][cH:5][cH:6][cH:7][cH:8]1"
    )


def test_atom_mapping_no_namerxn_choose_rxnmapper(synthesis_route, setup_mapper_no_namerxn, recwarn):
    synthesis_route.assign_atom_mapping(only_rxnmapper=True)

    assert len(recwarn) == 0

    data = pd.DataFrame(synthesis_route.reaction_data())

    assert len(data) == 2
    assert data["classification"].to_list() == ["0.0", "0.0"]
    assert (
        data["mapped_reaction_smiles"][0]
        == "Cl[c:3]1[cH:4][cH:5][cH:6][cH:7][cH:8]1.[CH3:1][OH:2]>>[CH3:1][O:2][c:3]1[cH:4][cH:5][cH:6][cH:7][cH:8]1"
    )


def test_root_smiles(synthesis_route, setup_mapper):
    with pytest.raises(ValueError):
        SynthesisRoute({"smiles": "C"}).mapped_root_smiles

    with pytest.raises(ValueError):
        synthesis_route.mapped_root_smiles

    synthesis_route.assign_atom_mapping()
    assert synthesis_route.mapped_root_smiles == "[CH3:1][O:2][c:3]1[cH:4][cH:5][cH:6][cH:7][cH:8]1"


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


def test_remap_ref_smiles(synthesis_route, setup_mapper):
    route1 = synthesis_route
    route1.assign_atom_mapping()
    old_reaction_smiles = route1.atom_mapped_reaction_smiles()
    reactants, _, products = split_rsmi(old_reaction_smiles[0])
    rsmi_old = Chem.MolToSmiles(Chem.MolFromSmiles(reactants))
    psmi_old = Chem.MolToSmiles(Chem.MolFromSmiles(products))

    route1.remap(products)

    reactants, _, products = split_rsmi(route1.atom_mapped_reaction_smiles()[0])
    rsmi = Chem.MolToSmiles(Chem.MolFromSmiles(reactants))
    psmi = Chem.MolToSmiles(Chem.MolFromSmiles(products))
    assert rsmi == rsmi_old
    assert psmi == psmi_old

    route1.remap("[CH3:10][O:2][c:3]1[cH:4][cH:5][cH:6][cH:7][cH:8]1")

    reactants, _, products = split_rsmi(route1.atom_mapped_reaction_smiles()[0])
    rsmi = Chem.MolToSmiles(Chem.MolFromSmiles(reactants))
    psmi = Chem.MolToSmiles(Chem.MolFromSmiles(products))
    assert rsmi != rsmi_old
    assert psmi != psmi_old


def test_remap_ref_dict(synthesis_route, setup_mapper):
    route1 = synthesis_route
    route1.assign_atom_mapping()
    old_reaction_smiles = route1.atom_mapped_reaction_smiles()
    reactants, _, products = split_rsmi(old_reaction_smiles[0])
    rsmi_old = Chem.MolToSmiles(Chem.MolFromSmiles(reactants))
    psmi_old = Chem.MolToSmiles(Chem.MolFromSmiles(products))

    route1.remap({1: 10, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8})

    reactants, _, products = split_rsmi(route1.atom_mapped_reaction_smiles()[0])
    rsmi = Chem.MolToSmiles(Chem.MolFromSmiles(reactants))
    psmi = Chem.MolToSmiles(Chem.MolFromSmiles(products))
    assert rsmi != rsmi_old
    assert psmi != psmi_old


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


def test_route_leaves(synthesis_route):
    leaves = synthesis_route.leaves()

    assert leaves == {"c1ccccc1", "Cl", "CO"}


def test_route_leaves_count(synthesis_route):
    counts = synthesis_route.leaf_counts()

    assert counts == {"c1ccccc1": 1, "Cl": 1, "CO": 1}


def test_route_intermediates(synthesis_route):
    intermediates = synthesis_route.intermediates()

    assert intermediates == {"Clc1ccccc1"}


def test_route_intermediates_count(synthesis_route):
    counts = synthesis_route.intermediate_counts()

    assert counts == {"Clc1ccccc1": 1}


def test_route_is_solved(synthesis_route, setup_stock):
    assert synthesis_route.is_solved()

    setup_stock(synthesis_route, {"c1ccccc1", "Cl", "CO"})
    assert synthesis_route.is_solved()

    setup_stock(synthesis_route, {"Cl", "CO"})
    assert not synthesis_route.is_solved()


def test_extract_monograms(synthesis_route):
    monograms = synthesis_route.reaction_ngrams(1, "reaction_smiles")

    assert len(monograms) == 2
    assert monograms[0] == ("CO.Clc1ccccc1>>COc1ccccc1",)
    assert monograms[1] == ("Cl.c1ccccc1>>Clc1ccccc1",)


def test_extract_bigrams(synthesis_route):
    bigrams = synthesis_route.reaction_ngrams(2, "reaction_smiles")

    assert len(bigrams) == 1
    assert bigrams[0] == ("CO.Clc1ccccc1>>COc1ccccc1", "Cl.c1ccccc1>>Clc1ccccc1")


def test_extract_bigrams_branched(branched_synthesis_route):
    bigrams = branched_synthesis_route.reaction_ngrams(2, "reaction_smiles")

    assert len(bigrams) == 2
    assert bigrams[0] == ("CO.Clc1ccccc1>>COc1ccccc1", "C.O>>CO")
    assert bigrams[1] == ("CO.Clc1ccccc1>>COc1ccccc1", "Cl.c1ccccc1>>Clc1ccccc1")


def test_extract_trigrams_too_short(synthesis_route):
    trigrams = synthesis_route.reaction_ngrams(3, "reaction_smiles")

    assert len(trigrams) == 0


def test_extract_grams_augmentable(augmentable_sythesis_route):
    route = augmentable_sythesis_route
    bigrams = route.reaction_ngrams(2, "hash")

    assert bigrams == [("xyz", "xyy"), ("xyy", "xxz")]

    trigrams = route.reaction_ngrams(3, "hash")
    assert trigrams == [("xyz", "xyy", "xxz")]
