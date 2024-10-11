import copy
import json

import pandas as pd
import pytest

from rxnutils.routes.base import SynthesisRoute
from rxnutils.routes.readers import reactions2route


@pytest.fixture
def make_reaction_dataframe(shared_datadir):
    return pd.read_csv(shared_datadir / "example_reactions.csv")


@pytest.fixture
def make_template_dataframe(shared_datadir):
    return pd.read_csv(shared_datadir / "example_templates.csv")


TEST_TREE = {
    "type": "mol",
    "smiles": "COc1ccccc1",
    "children": [
        {
            "type": "reaction",
            "metadata": {"reaction_smiles": "CO.Clc1ccccc1>>COc1ccccc1"},
            "children": [
                {"type": "mol", "smiles": "CO"},
                {
                    "type": "mol",
                    "smiles": "Clc1ccccc1",
                    "children": [
                        {
                            "type": "reaction",
                            "metadata": {"reaction_smiles": "Cl.c1ccccc1>>Clc1ccccc1"},
                            "children": [
                                {"type": "mol", "smiles": "Cl"},
                                {"type": "mol", "smiles": "c1ccccc1"},
                            ],
                        }
                    ],
                },
            ],
        }
    ],
}


@pytest.fixture
def synthesis_route():
    return SynthesisRoute(copy.deepcopy(TEST_TREE))


@pytest.fixture
def branched_synthesis_route():
    dict_ = copy.deepcopy(TEST_TREE)
    dict_["children"][0]["children"][0]["children"] = [
        {
            "type": "reaction",
            "metadata": {"reaction_smiles": "C.O>>CO"},
            "children": [
                {"type": "mol", "smiles": "C"},
                {"type": "mol", "smiles": "O"},
            ],
        }
    ]
    return SynthesisRoute(dict_)


@pytest.fixture
def augmentable_sythesis_route():
    smiles0 = ["c1ccccc1>>Clc1ccccc1", "Cc1ccccc1>>c1ccccc1", "c1ccccc1O.C>>c1ccccc1C"]
    metadata = [
        {"classification": "10.1.2 chlorination", "hash": "xyz"},
        {"hash": "xyy"},
        {"hash": "xxz"},
    ]
    return reactions2route(smiles0, metadata)


@pytest.fixture
def setup_mapper(mocker, shared_datadir):
    df = pd.read_csv(shared_datadir / "mapped_tests_reactions.csv", sep="\t")
    namerxn_mock = mocker.patch("rxnutils.routes.base.NameRxn")
    namerxn_mock.return_value.return_value = df
    namerxn_mock = mocker.patch("rxnutils.routes.base.RxnMapper")
    namerxn_mock.return_value.return_value = df


@pytest.fixture
def setup_mapper_no_namerxn(mocker, shared_datadir):
    df = pd.read_csv(shared_datadir / "mapped_tests_reactions.csv", sep="\t")
    namerxn_mock = mocker.patch("rxnutils.routes.base.NameRxn")
    namerxn_mock.return_value.side_effect = FileNotFoundError("No namerxn")
    namerxn_mock = mocker.patch("rxnutils.routes.base.RxnMapper")
    df["NMC"] = "0.0"
    namerxn_mock.return_value.return_value = df


@pytest.fixture
def load_reaction_tree(shared_datadir):
    def wrapper(filename, index=0):
        filename = str(shared_datadir / filename)
        with open(filename, "r") as fileobj:
            trees = json.load(fileobj)
        if isinstance(trees, dict):
            return trees
        elif index == -1:
            return trees
        else:
            return trees[index]

    return wrapper


@pytest.fixture
def setup_stock():
    def traverse(tree_dict, stock):
        if tree_dict["type"] == "mol":
            tree_dict["in_stock"] = tree_dict["smiles"] in stock
        for child in tree_dict.get("children", []):
            traverse(child, stock)

    def wrapper(route, stock):
        traverse(route.reaction_tree, stock)

    return wrapper
