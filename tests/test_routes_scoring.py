import copy
import json
import os

import pandas as pd
import pytest
import requests

from rxnutils.routes.scoring import (
    route_sorter,
    route_ranks,
    badowski_route_score,
    retro_bleu_score,
    ngram_overlap_score,
)
from rxnutils.routes.retro_bleu.ngram_collection import NgramCollection


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
