""" Module contain method to compute distance matrix using TED """

import time
from typing import Sequence

import numpy as np

from rxnutils.routes.base import SynthesisRoute
from rxnutils.routes.ted.reactiontree import ReactionTreeWrapper


def ted_distances_calculator(
    routes: Sequence[SynthesisRoute], content: str = "both", timeout: int = None
) -> np.ndarray:
    """
    Compute the TED distances between each pair of routes

    :param routes: the routes to calculate pairwise distance on
    :param content: determine what part of the synthesis trees to include in the calculation,
            options 'molecules', 'reactions' and 'both', default 'both'
    :param timeout: if given, raises an exception if timeout is taking longer time
    :return: the square distance matrix
    """
    distances = np.zeros([len(routes), len(routes)])
    distance_wrappers = [ReactionTreeWrapper(route, content) for route in routes]
    time0 = time.perf_counter()
    for i, iwrapper in enumerate(distance_wrappers):
        # fmt: off
        for j, jwrapper in enumerate(distance_wrappers[i + 1:], i + 1):
            distances[i, j] = iwrapper.distance_to(jwrapper)
            distances[j, i] = distances[i, j]
        # fmt: on
        time_past = time.perf_counter() - time0
        if timeout is not None and time_past > timeout:
            raise ValueError(f"Unable to compute distance matrix in {timeout} s")
    return distances
