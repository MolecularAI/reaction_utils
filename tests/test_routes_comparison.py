import json
from socket import timeout

import pytest

from rxnutils.routes.base import SynthesisRoute
from rxnutils.routes.comparison import (
    atom_matching_bonanza_similarity,
    route_distances_calculator,
    simple_bond_forming_similarity,
    simple_route_similarity,
)


@pytest.fixture
def example_routes(shared_datadir):
    with open(shared_datadir / "atorvastatin_routes.json", "r") as fileobj:
        dicts = json.load(fileobj)
    return [SynthesisRoute(dicts[0]), SynthesisRoute(dicts[1])]


def test_bond_similarity(example_routes):
    similarity = simple_bond_forming_similarity(example_routes)

    assert similarity.tolist() == [[1, 0.5], [0.5, 1.0]]


def test_atom_similarity(example_routes):
    similarity = atom_matching_bonanza_similarity(example_routes)

    assert pytest.approx(similarity.tolist()[0], rel=1e-4) == [1, 0.6871]
    assert pytest.approx(similarity.tolist()[1], rel=1e-4) == [0.6871, 1]


def test_simple_route_similarity(example_routes):
    similarity = simple_route_similarity(example_routes)

    assert pytest.approx(similarity.tolist()[0], rel=1e-4) == [1, 0.5861]
    assert pytest.approx(similarity.tolist()[1], rel=1e-4) == [0.5861, 1]


def test_simple_route_similarity_one_no_rxns(example_routes):
    example_routes[0].max_depth = 0.0
    example_routes[0].reaction_tree["children"] = []

    similarity = simple_route_similarity(example_routes)

    assert pytest.approx(similarity.tolist()[0], rel=1e-4) == [1, 0.0]
    assert pytest.approx(similarity.tolist()[1], rel=1e-4) == [0.0, 1]


def test_simple_route_similarity_two_no_rxns(example_routes):
    for route in example_routes:
        route.max_depth = 0.0
        route.reaction_tree["children"] = []

    similarity = simple_route_similarity(example_routes)

    assert pytest.approx(similarity.tolist()[0], rel=1e-4) == [1, 1.0]
    assert pytest.approx(similarity.tolist()[1], rel=1e-4) == [1.0, 1]


def test_ted_distance_calculator(example_routes):
    ted_distance_calc = route_distances_calculator(model="ted")

    route_distances = ted_distance_calc(example_routes)

    assert route_distances.shape == (2, 2)
    assert route_distances[0, 1] == route_distances[1, 0]
    assert route_distances[0, 0] == route_distances[1, 1] == 0


def test_ted_distance_calculator_with_kwargs(example_routes):
    ted_distance_calc = route_distances_calculator(model="ted", timeout=10)

    route_distances = ted_distance_calc(example_routes)

    assert route_distances.shape == (2, 2)
    assert route_distances[0, 1] == route_distances[1, 0]
    assert route_distances[0, 0] == route_distances[1, 1] == 0


def test_lstm_distance_calulator_not_implemented(example_routes):
    with pytest.raises(NotImplementedError):
        _ = route_distances_calculator(model="lstm")
