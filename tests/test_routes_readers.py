import json

import pandas as pd

from rxnutils.routes.readers import (
    reactions2route,
    read_aizynthcli_dataframe,
    read_aizynthfinder_dict,
    read_rdf_file,
    read_reactions_dataframe,
)


def test_create_route_from_reaction(synthesis_route):
    route = reactions2route(["Cl.C1=CC=CC=C1>>ClC1=CC=CC=C1", "CO.ClC1=CC=CC=C1>>COC1=CC=CC=C1"])

    assert route.reaction_tree == synthesis_route.reaction_tree


def test_create_route_from_reaction_w_meta():
    route = reactions2route(
        ["Cl.C1=CC=CC=C1>>ClC1=CC=CC=C1", "CO.ClC1=CC=CC=C1>>COC1=CC=CC=C1"],
        [{}, {"foo": "bar"}],
    )

    data = route.reaction_data()
    assert list(data[0].keys()) == [
        "foo",
        "reaction_smiles",
        "tree_depth",
        "forward_step",
    ]
    assert list(data[1].keys()) == ["reaction_smiles", "tree_depth", "forward_step"]


def test_create_route_from_dataframe():
    df = pd.DataFrame(
        {
            "smiles": [
                "Cl.C1=CC=CC=C1>>ClC1=CC=CC=C1",
                "CO.ClC1=CC=CC=C1>>COC1=CC=CC=C1",
            ],
            "foo": ["bar", None],
            "id": [0, 0],
        }
    )
    routes = read_reactions_dataframe(df, "smiles", ["id"], ["foo"])

    assert len(routes) == 1
    data = routes[0].reaction_data()
    expected_keys = ["foo", "reaction_smiles", "tree_depth", "forward_step"]
    assert list(data[0].keys()) == expected_keys
    assert list(data[1].keys()) == expected_keys


def test_create_route_from_aizynth_dict(shared_datadir):
    with open(shared_datadir / "branched_route.json", "r") as fileobj:
        tree_dict = json.load(fileobj)

    route = read_aizynthfinder_dict(tree_dict)

    assert len(route.reaction_smiles()) == 4
    assert route.reaction_smiles()[-1] == "NC1CCCCC1.C1=CCC=C1>>NC1CCCC(C2C=CC=C2)C1"


def test_create_route_from_aizynth_cli(shared_datadir):
    with open(shared_datadir / "branched_route.json", "r") as fileobj:
        tree_dict = json.load(fileobj)

    df = pd.DataFrame({"trees": [[tree_dict]]})

    routes = read_aizynthcli_dataframe(df)

    assert len(routes) == 1
    assert len(routes[0]) == 1
    assert len(routes[0][0].reaction_smiles()) == 4
    assert routes[0][0].reaction_smiles()[-1] == "NC1CCCCC1.C1=CCC=C1>>NC1CCCC(C2C=CC=C2)C1"


def test_create_route_from_rdf(shared_datadir):
    filename = str(shared_datadir / "example_route.rdf")

    route = read_rdf_file(filename)

    reactions = route.reaction_smiles()
    assert len(reactions) == 2
    assert reactions[0] == "CCC=O>>CCC(=O)O"
    assert reactions[1] == "CCCN>>CCC=O"
