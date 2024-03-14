import copy

import pandas as pd
import pytest

from rxnutils.routes.base import SynthesisRoute


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
def setup_mapper(mocker, shared_datadir):
    df = pd.read_csv(shared_datadir / "mapped_tests_reactions.csv", sep="\t")
    namerxn_mock = mocker.patch("rxnutils.routes.base.NameRxn")
    namerxn_mock.return_value.return_value = df
    namerxn_mock = mocker.patch("rxnutils.routes.base.RxnMapper")
    namerxn_mock.return_value.return_value = df
