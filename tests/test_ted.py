import json

import pytest

from rxnutils.routes.base import SynthesisRoute
from rxnutils.routes.ted.distances_calculator import ted_distances_calculator
from rxnutils.routes.ted.reactiontree import ReactionTreeWrapper
from rxnutils.routes.ted.utils import AptedConfig, StandardFingerprintFactory, TreeContent


def collect_smiles(tree, query_type, smiles_list):
    if tree["type"] == query_type:
        smiles_list.append(tree["smiles"])
    for child in tree.get("children", []):
        collect_smiles(child, query_type, smiles_list)


node1 = {"type": "mol", "fingerprint": [0, 1, 0], "children": ["A", "B", "C"]}

node2 = {"type": "mol", "fingerprint": [1, 1, 0]}

example_tree = {
    "type": "mol",
    "smiles": "Cc1ccc2nc3ccccc3c(Nc3ccc(NC(=S)Nc4ccccc4)cc3)c2c1",
    "children": [
        {
            "type": "reaction",
            "metadata": {},
            "children": [
                {
                    "type": "mol",
                    "smiles": "Cc1ccc2nc3ccccc3c(Cl)c2c1",
                },
                {
                    "type": "mol",
                    "smiles": "Nc1ccc(NC(=S)Nc2ccccc2)cc1",
                },
            ],
        }
    ],
}


def test_rename_cost_different_types():
    config = AptedConfig()

    cost = config.rename({"type": "type1"}, {"type": "type2"})

    assert cost == 1


def test_rename_cost_same_types():
    config = AptedConfig()

    cost = config.rename(node1, node2)

    assert cost == 0.5


def test_get_children_fixed():
    config = AptedConfig()

    assert config.children(node1) == ["A", "B", "C"]


def test_get_children_random():
    config = AptedConfig(randomize=True)

    children = config.children(node1)

    assert len(children) == 3
    for expected_child in ["A", "B", "C"]:
        assert expected_child in children


@pytest.mark.parametrize(
    "route_index",
    [1, 2],
)
def test_create_wrapper(load_reaction_tree, route_index):
    tree = load_reaction_tree("example_routes.json", route_index)
    route = SynthesisRoute(tree)
    wrapper = ReactionTreeWrapper(route)

    assert wrapper.info["content"] == TreeContent.MOLECULES
    assert wrapper.info["tree count"] == 4
    assert wrapper.first_tree["type"] == "mol"
    assert len(wrapper.trees) == 4

    wrapper = ReactionTreeWrapper(route, TreeContent.REACTIONS)

    assert wrapper.info["content"] == TreeContent.REACTIONS
    assert wrapper.info["tree count"] == 1
    assert wrapper.first_tree["type"] == "reaction"
    assert len(wrapper.trees) == 1

    wrapper = ReactionTreeWrapper(route, TreeContent.BOTH)

    assert wrapper.info["content"] == TreeContent.BOTH
    assert wrapper.info["tree count"] == 4
    assert len(wrapper.trees) == 4


def test_create_wrapper_no_reaction():
    tree = {"smiles": "CCC", "type": "mol"}
    route = SynthesisRoute(tree)

    wrapper = ReactionTreeWrapper(route)
    assert wrapper.info["tree count"] == 1
    assert len(wrapper.trees) == 1

    with pytest.raises(ValueError):
        ReactionTreeWrapper(route, TreeContent.REACTIONS)

    wrapper = ReactionTreeWrapper(route, TreeContent.BOTH)
    assert wrapper.info["tree count"] == 1
    assert wrapper.first_tree["type"] == "mol"
    assert len(wrapper.trees) == 1


def test_create_one_tree_of_molecules(load_reaction_tree):
    tree = load_reaction_tree("example_routes.json", 0)
    route = SynthesisRoute(tree)

    wrapper = ReactionTreeWrapper(route, exhaustive_limit=1)

    assert wrapper.info["tree count"] == 2
    assert len(wrapper.trees) == 1

    assert wrapper.first_tree["smiles"] == tree["smiles"]
    assert len(wrapper.first_tree["children"]) == 2

    child_smiles = [child["smiles"] for child in wrapper.first_tree["children"]]
    expected_smiles = [node["smiles"] for node in tree["children"][0]["children"]]
    assert child_smiles == expected_smiles


def test_create_one_tree_of_reactions(load_reaction_tree):
    tree = load_reaction_tree("example_routes.json", 0)
    route = SynthesisRoute(tree)

    wrapper = ReactionTreeWrapper(route, content=TreeContent.REACTIONS, exhaustive_limit=1)

    assert wrapper.info["tree count"] == 1
    assert len(wrapper.trees) == 1

    rxn_nodes = []
    collect_smiles(tree, "reaction", rxn_nodes)
    assert wrapper.first_tree["smiles"] == rxn_nodes[0]
    assert len(wrapper.first_tree["children"]) == 0


def test_create_one_tree_of_everything(load_reaction_tree):
    tree = load_reaction_tree("example_routes.json", 0)
    route = SynthesisRoute(tree)

    wrapper = ReactionTreeWrapper(route, content=TreeContent.BOTH, exhaustive_limit=1)

    assert wrapper.info["tree count"] == 2
    assert len(wrapper.trees) == 1

    mol_nodes = []
    collect_smiles(tree, "mol", mol_nodes)
    rxn_nodes = []
    collect_smiles(tree, "reaction", rxn_nodes)
    assert wrapper.first_tree["smiles"] == tree["smiles"]
    assert len(wrapper.first_tree["children"]) == 1

    child1 = wrapper.first_tree["children"][0]
    assert child1["smiles"] == rxn_nodes[0]
    assert len(child1["children"]) == 2

    child_smiles = [child["smiles"] for child in child1["children"]]
    assert child_smiles == mol_nodes[1:]


def test_create_all_trees_of_molecules(load_reaction_tree):
    tree = load_reaction_tree("example_routes.json", 0)
    route = SynthesisRoute(tree)

    wrapper = ReactionTreeWrapper(route)

    assert wrapper.info["tree count"] == 2
    assert len(wrapper.trees) == 2

    mol_nodes = []
    collect_smiles(tree, "mol", mol_nodes)
    # Assert first tree
    assert wrapper.first_tree["smiles"] == mol_nodes[0]
    assert len(wrapper.first_tree["children"]) == 2

    child_smiles = [child["smiles"] for child in wrapper.first_tree["children"]]
    assert child_smiles == mol_nodes[1:]

    # Assert second tree
    assert wrapper.trees[1]["smiles"] == mol_nodes[0]
    assert len(wrapper.trees[1]["children"]) == 2

    child_smiles = [child["smiles"] for child in wrapper.trees[1]["children"]]
    assert child_smiles == mol_nodes[1:][::-1]


def test_create_two_trees_of_everything(load_reaction_tree):
    tree = load_reaction_tree("example_routes.json", 0)
    route = SynthesisRoute(tree)

    wrapper = ReactionTreeWrapper(route, content=TreeContent.BOTH)

    assert wrapper.info["tree count"] == 2
    assert len(wrapper.trees) == 2

    mol_nodes = []
    collect_smiles(tree, "mol", mol_nodes)
    rxn_nodes = []
    collect_smiles(tree, "reaction", rxn_nodes)
    # Assert first tree
    assert wrapper.first_tree["smiles"] == mol_nodes[0]
    assert len(wrapper.first_tree["children"]) == 1

    child1 = wrapper.first_tree["children"][0]
    assert child1["smiles"] == rxn_nodes[0]
    assert len(child1["children"]) == 2

    child_smiles = [child["smiles"] for child in child1["children"]]
    assert child_smiles == mol_nodes[1:]

    # Assert second tree
    assert wrapper.trees[1]["smiles"] == mol_nodes[0]
    assert len(wrapper.trees[1]["children"]) == 1

    child1 = wrapper.trees[1]["children"][0]
    assert child1["smiles"] == rxn_nodes[0]
    assert len(child1["children"]) == 2

    child_smiles = [child["smiles"] for child in child1["children"]]
    assert child_smiles == mol_nodes[1:][::-1]


def test_route_self_distance(load_reaction_tree):
    tree = load_reaction_tree("example_routes.json", 0)
    route = SynthesisRoute(tree)

    wrapper = ReactionTreeWrapper(route, exhaustive_limit=1)

    assert wrapper.distance_to(wrapper) == 0.0


def test_route_distances_random(load_reaction_tree):
    tree1 = load_reaction_tree("example_routes.json", 0)
    route1 = SynthesisRoute(tree1)
    wrapper1 = ReactionTreeWrapper(route1, exhaustive_limit=1)

    tree2 = load_reaction_tree("example_routes.json", 1)
    route2 = SynthesisRoute(tree2)
    wrapper2 = ReactionTreeWrapper(route2, exhaustive_limit=1)

    distances = list(wrapper1.distance_iter(wrapper2, exhaustive_limit=1))

    assert len(distances) == 2
    assert pytest.approx(distances[0], abs=1e-2) == 2.6522


def test_route_distances_exhaustive(load_reaction_tree):
    tree1 = load_reaction_tree("example_routes.json", 0)
    route1 = SynthesisRoute(tree1)
    wrapper1 = ReactionTreeWrapper(route1, exhaustive_limit=2)

    tree2 = load_reaction_tree("example_routes.json", 1)
    route2 = SynthesisRoute(tree2)
    wrapper2 = ReactionTreeWrapper(route2, exhaustive_limit=2)

    distances = list(wrapper1.distance_iter(wrapper2, exhaustive_limit=40))

    assert len(distances) == 2
    assert pytest.approx(distances[0], abs=1e-2) == 2.6522
    assert pytest.approx(min(distances), abs=1e-2) == 2.6522


def test_route_distances_semi_exhaustive(load_reaction_tree):
    tree1 = load_reaction_tree("example_routes.json", 0)
    route1 = SynthesisRoute(tree1)
    wrapper1 = ReactionTreeWrapper(route1, exhaustive_limit=1)

    tree2 = load_reaction_tree("example_routes.json", 1)
    route2 = SynthesisRoute(tree2)
    wrapper2 = ReactionTreeWrapper(route2, exhaustive_limit=2)

    distances = list(wrapper1.distance_iter(wrapper2, exhaustive_limit=1))

    assert len(distances) == 2
    assert pytest.approx(distances[0], abs=1e-2) == 2.6522
    assert pytest.approx(min(distances), abs=1e-2) == 2.6522


def test_route_distances_longer_routes(load_reaction_tree):
    tree1 = load_reaction_tree("longer_routes.json", 0)
    route1 = SynthesisRoute(tree1)
    wrapper1 = ReactionTreeWrapper(route1, content="both")

    tree2 = load_reaction_tree("longer_routes.json", 1)
    route2 = SynthesisRoute(tree2)
    wrapper2 = ReactionTreeWrapper(route2, content="both")

    distances = list(wrapper1.distance_iter(wrapper2))

    assert len(distances) == 21
    assert pytest.approx(distances[0], abs=1e-2) == 4.14


def test_distance_matrix(load_reaction_tree):
    routes = [SynthesisRoute(load_reaction_tree("example_routes.json", idx)) for idx in range(3)]

    dist_mat = ted_distances_calculator(routes, content="molecules")

    assert len(dist_mat) == 3
    assert pytest.approx(dist_mat[0, 1], abs=1e-2) == 2.6522
    assert pytest.approx(dist_mat[0, 2], abs=1e-2) == 3.0779
    assert pytest.approx(dist_mat[2, 1], abs=1e-2) == 0.7483


def test_distance_matrix_timeout(load_reaction_tree):
    routes = [SynthesisRoute(load_reaction_tree("example_routes.json", idx)) for idx in range(3)]

    with pytest.raises(ValueError):
        ted_distances_calculator(routes, content="molecules", timeout=0)


def test_fingerprint_calculations():
    example_route = SynthesisRoute(example_tree)
    wrapper = ReactionTreeWrapper(example_route, content="both", fp_factory=StandardFingerprintFactory(nbits=128))

    fp = wrapper.first_tree["sort_key"]
    mol1 = "1000010000000000000010001000100101000101100000010000010000100001"
    mol2 = "1100000001110110011000100010000001001000000100100000110000100100"
    assert fp == mol1 + mol2

    fp = wrapper.first_tree["children"][0]["sort_key"]
    rxn1 = "00000-1000000-1000-100-2000000000000000000000000000-10-20000000000000-1"
    rxn2 = "-10000000001000100-10000-100-10000000000-10-1000000000000-11000-10000100"
    assert fp == rxn1 + rxn2


def test_custom_fingerprint_calculations():
    def factory(tree, parent):
        if tree["type"] != "reaction":
            return
        tree["fingerprint"] = [1, 2, 3, 4]

    example_route = SynthesisRoute(example_tree)
    wrapper = ReactionTreeWrapper(example_route, content="both", fp_factory=factory)

    assert wrapper.first_tree["sort_key"] == ""
    assert wrapper.first_tree["children"][0]["sort_key"] == "1234"
