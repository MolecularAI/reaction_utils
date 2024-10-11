""" Routines for scoring synthesis routes
"""

from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from rxnutils.routes.base import SynthesisRoute
from rxnutils.routes.retro_bleu.scoring import (
    ngram_overlap_score,  # noqa
    retro_bleu_score,
)


def route_sorter(
    routes: List[SynthesisRoute], scorer: Callable[..., float], **kwargs: Any
) -> Tuple[List[SynthesisRoute], List[float]]:
    """
    Scores and sort a list of routes.
    Returns a tuple of the sorted routes and their scores.

    :param routes: the routes to score
    :param scorer: the scorer function
    :param kwargs: additional argument given to the scorer
    :return: the sorted routes and their scores
    """
    scores = np.asarray([scorer(route, **kwargs) for route in routes])
    sorted_idx = np.argsort(scores)
    routes = [routes[idx] for idx in sorted_idx]
    return routes, scores[sorted_idx].tolist()


def route_ranks(scores: List[float]) -> List[int]:
    """
    Compute the rank of route scores. Rank starts at 1

    :param scores: the route scores
    :return: a list of ranks for each route
    """
    ranks = [1]
    for idx in range(1, len(scores)):
        if abs(scores[idx] - scores[idx - 1]) < 1e-8:
            ranks.append(ranks[idx - 1])
        else:
            ranks.append(ranks[idx - 1] + 1)
    return ranks


def badowski_route_score(
    route: SynthesisRoute,
    mol_costs: Dict[bool, float] = None,
    average_yield: float = 0.8,
    reaction_cost: float = 1.0,
) -> float:
    """
    Calculate the score of route using the method from
    (Badowski et al. Chem Sci. 2019, 10, 4640).

    The reaction cost is constant and the yield is an average yield.
    The starting materials are assigned a cost based on whether they are in
    stock or not. By default starting material in stock is assigned a
    cost of 1 and starting material not in stock is assigned a cost of 10.

    To be accurate, each molecule node need to have an extra
    boolean property called `in_stock`.

    :param route: the route to analyze
    :param mol_costs: the starting material cost
    :param average_yield: the average yield, defaults to 0.8
    :param reaction_cost: the reaction cost, defaults to 1.0
    :return: the computed cost
    """

    def traverse(tree_dict):
        mol_cost = mol_costs or {True: 1, False: 10}

        reactions = tree_dict.get("children", [])
        if not reactions:
            return mol_cost[tree_dict.get("in_stock", True)]

        child_sum = sum(
            1 / average_yield * traverse(child) for child in reactions[0]["children"]
        )
        return reaction_cost + child_sum

    return traverse(route.reaction_tree)
