import copy
import json
import os

import pandas as pd
import pytest
import requests
from rdkit import Chem

from rxnutils.chem.features.sc_score import SCORE_SUPPORTED, SCScore
from rxnutils.routes.deepset.featurizers import collect_reaction_features, default_reaction_featurizer, ecfp_fingerprint
from rxnutils.routes.retro_bleu.ngram_collection import NgramCollection
from rxnutils.routes.scoring import (
    DeepsetModelClient,
    badowski_route_score,
    deepset_route_score,
    ngram_overlap_score,
    reaction_class_rank_score,
    retro_bleu_score,
    route_ranks,
    route_sorter,
)


def test_route_sorter(synthesis_route, setup_stock):
    route1 = copy.deepcopy(synthesis_route)
    setup_stock(route1, {"Cl", "CO"})
    route2 = copy.deepcopy(synthesis_route)
    setup_stock(route2, {"Cl", "CO", "c1ccccc1"})
    routes = [route1, route2]

    sorted_routes, route_scores = route_sorter(routes, badowski_route_score)

    assert route_scores == pytest.approx([6.625, 20.6875], abs=1e-4)
    assert sorted_routes[0] is route2
    assert sorted_routes[1] is route1


def test_route_rank():

    assert route_ranks([4.0, 5.0, 5.0]) == [1, 2, 2]
    assert route_ranks([4.0, 4.0, 5.0]) == [1, 1, 2]
    assert route_ranks([4.0, 5.0, 6.0]) == [1, 2, 3]
    assert route_ranks([4.0, 5.0, 5.0, 6.0]) == [1, 2, 2, 3]


def test_badowski_score(synthesis_route, setup_stock):
    setup_stock(synthesis_route, {"Cl", "CO"})

    assert badowski_route_score(synthesis_route) == pytest.approx(20.6875, abs=1e-4)

    setup_stock(synthesis_route, {"Cl", "CO", "c1ccccc1"})

    assert badowski_route_score(synthesis_route) == pytest.approx(6.625, abs=1e-4)


def test_class_rank_score(synthesis_route, setup_mapper):
    synthesis_route.assign_atom_mapping()

    score = reaction_class_rank_score(synthesis_route, {"10.1.2": 15, "1.7.11": 2}, ["10.1.2"])

    assert pytest.approx(score, abs=0.01) == 0.33


def test_read_and_write_ngram_collection(tmpdir):
    collection = NgramCollection(2, "dummy", {("a", "b"), ("a", "c")})

    filename = str(tmpdir / "ngrams.json")
    collection.save_to_file(filename)

    assert os.path.exists(filename)

    collection2 = NgramCollection.from_file(filename)

    assert collection2.nitems == collection.nitems
    assert collection2.metadata_key == collection.metadata_key
    assert collection2.ngrams == collection.ngrams


def test_create_ngram_collection(augmentable_sythesis_route, tmpdir):
    filename = str(tmpdir / "routes.json")
    with open(filename, "w") as fileobj:
        json.dump([augmentable_sythesis_route.reaction_tree], fileobj)

    collection = NgramCollection.from_tree_collection(filename, 2, "hash")

    assert collection.nitems == 2
    assert collection.metadata_key == "hash"
    assert collection.ngrams == {("xyz", "xyy"), ("xyy", "xxz")}


def test_ngram_overlap_score(augmentable_sythesis_route):
    ref = NgramCollection(2, "hash", {("xyz", "xyy"), ("xxyy", "xxz")})

    score = ngram_overlap_score(augmentable_sythesis_route, ref)

    assert score == 0.5


def test_retro_bleu_score(augmentable_sythesis_route):
    ref = NgramCollection(2, "hash", {("xyz", "xyy"), ("xxyy", "xxz")})

    score = retro_bleu_score(augmentable_sythesis_route, ref)

    assert pytest.approx(score, abs=1e-4) == 4.3670


def test_retro_bleu_score_short_nsteps(augmentable_sythesis_route):
    ref = NgramCollection(2, "hash", {("xyz", "xyy"), ("xxyy", "xxz")})

    score = retro_bleu_score(augmentable_sythesis_route, ref, 1)

    assert pytest.approx(score, abs=1e-4) == 3.0443


def test_ecfp_featurizer():
    mol = Chem.MolFromSmiles("O")

    fp = ecfp_fingerprint(mol, 2, 10)

    assert len(fp) == 10
    assert sum(fp) == 1


def test_reaction_featurizer():
    reaction_data = {"reaction_smiles": "C.O>>CO"}

    fp = default_reaction_featurizer(reaction_data, 2, 10)

    assert len(fp) == 10
    assert sum(fp) == 0.0


def test_collect_reaction_features():
    reaction_data = {
        "reaction_smiles": "C.O>>CO",
        "classification": "5.1.1",
        "tree_depth": 1,
    }
    class_ranks = {"5.1.1": 3}
    target_fp = ecfp_fingerprint(Chem.MolFromSmiles("CO"), 2, 10)

    score, features = collect_reaction_features([reaction_data], target_fp, class_ranks, default_reaction_featurizer)

    assert score == 3.0
    assert features.shape == (1, 78)
    assert features[0, :3].tolist() == [5, 1, 1]
    assert features[0, 3] == 3.0
    assert features[0, 4:14].tolist() == target_fp.tolist()


@pytest.mark.xfail(condition=not SCORE_SUPPORTED, reason="onnx support not installed")
def test_deepsetscorer(mocker, shared_datadir, augmentable_sythesis_route):
    filename = str(shared_datadir / "scscore_dummy_model.onnx")
    scscorer = SCScore(filename, 5)
    mocked_model = mocker.patch("rxnutils.routes.deepset.scoring.onnxruntime.InferenceSession")
    mocked_model.return_value.run.return_value = [5.0]
    model = DeepsetModelClient("dummy")
    class_ranks = {"5.1.1": 3}

    score = deepset_route_score(augmentable_sythesis_route, model, scscorer, class_ranks)

    model._deepnet.run.assert_called_once()
    assert score == 5.0
    mocked_model.return_value.run.assert_called_once()

    call_args = mocked_model.return_value.run.call_args
    assert isinstance(call_args[0][1], dict)
    assert call_args[0][1]["reaction_features"].shape == (3, 132)
    assert len(call_args[0][1]["route_features"]) == 4
