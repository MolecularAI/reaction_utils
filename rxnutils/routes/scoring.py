""" Routines for scoring synthesis routes
"""

from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from rxnutils.routes.base import SynthesisRoute
from rxnutils.routes.chemformer_feasibility import (
    reaction_feasibility_score,
)  # noqa # pylint: disable=unused-import
from rxnutils.routes.chemformer_feasibility import (  # noqa # pylint: disable=unused-import
    ChemformerReactionFeasibilityCalculator,
)
from rxnutils.routes.deepset.scoring import (  # noqa # pylint: disable=unused-import
    DeepsetModelClient,
    deepset_route_score,
)
from rxnutils.routes.retro_bleu.scoring import (  # noqa # pylint: disable=unused-import
    ngram_overlap_score,
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


def reaction_class_rank_score(
    route: SynthesisRoute,
    reaction_class_ranks: Dict[str, int],
    preferred_classes: List[str],
    non_preferred_factor: float = 0.25,
) -> float:
    """
    Calculates a score of a route based on the reaction class rank score, i.e.
    how likely a particular reaction class is to succeed.

    Each step in the route is scored based on the following factors:
        * The reaction class rank
        * The step in the synthesis sequence
        * The preference of the reaction class

    The score is min-max normalized relative to the maximum depth of the three
    and the max/min of the class ranks.

    :param route: the route to score
    :param reaction_class_ranks: the rank score of NextMove classes
    :param preferred_classes: the preferred reaction classes
    :param non_preferred_factor: steps with non-preferred classes are multiplied by this
    :return: the computed score
    """

    def _score_reaction_tree(tree_dict):
        cls = tree_dict["metadata"]["classification"]
        if " " in cls:
            cls = cls.split(" ")[0]
        if _nextmove_superclass(cls) in preferred_superclasses:
            pref_factor = 1.0
        else:
            pref_factor = non_preferred_factor
        step_weight = tree_dict["metadata"]["forward_step"]
        score = pref_factor * step_weight * reaction_class_ranks.get(cls, min_rank)

        child_scores = []
        for child in tree_dict["children"]:
            grandchildren = child.get("children", [])
            if grandchildren:
                child_score = _score_reaction_tree(grandchildren[0])
                child_scores.append(child_score)
        if child_scores:
            score += min(child_scores)

        return score

    if route.nsteps == 0:
        return 1.0

    min_rank = min(reaction_class_ranks.values())
    max_step_weight = sum(range(1, route.max_depth + 1))
    min_score = max_step_weight * min_rank * non_preferred_factor
    max_score = max_step_weight * max(reaction_class_ranks.values())

    preferred_superclasses = [_nextmove_superclass(cls) for cls in preferred_classes]
    first_reaction = route.reaction_tree["children"][0]
    score = _score_reaction_tree(first_reaction)
    return (score - min_score) / (max_score - min_score)


def _nextmove_superclass(cls: str) -> str:
    """Extracting the NextMove superclass"""
    cls = str(cls).strip()
    if cls == "nan":
        return cls
    if cls.count(".") == 1:
        return cls
    return ".".join(cls.split(".")[:2])
