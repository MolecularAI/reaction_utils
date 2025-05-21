"""
Module containing helper classes to compute the distance between to reaction trees using the APTED method
Since APTED is based on ordered trees and the reaction trees are unordered, plenty of
heuristics are implemented to deal with this.
"""

from __future__ import annotations

import itertools
import math
from copy import deepcopy
from logging import getLogger
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from apted import APTED as Apted

from rxnutils.routes.base import SynthesisRoute
from rxnutils.routes.ted.utils import AptedConfig, StandardFingerprintFactory, TreeContent

StrDict = Dict[str, Any]
_FloatIterator = Iterable[float]


class ReactionTreeWrapper:
    """
    Wrapper for a reaction tree that can calculate distances between
    trees.

    :param route: the synthesis route to wrap
    :param content: the content of the route to consider in the distance calculation, default 'molecules'
    :param exhaustive_limit: if the number of possible ordered trees are below this limit create them all, default 20
    :param fp_factory: the factory of the fingerprint, Morgan fingerprint for molecules and reactions by default
    :param dist_func: the distance function to use when renaming nodes
    """

    _index_permutations = {n: list(itertools.permutations(range(n), n)) for n in range(1, 8)}

    def __init__(
        self,
        route: SynthesisRoute,
        content: Union[str, TreeContent] = TreeContent.MOLECULES,
        exhaustive_limit: int = 20,
        fp_factory: Callable[[StrDict, Optional[StrDict]], None] = None,
        dist_func: Callable[[np.ndarray, np.ndarray], float] = None,
    ) -> None:
        single_node_tree = not bool(route.reaction_smiles())
        if single_node_tree and content == TreeContent.REACTIONS:
            raise ValueError("Cannot create wrapping with content = reactions for a tree without reactions")

        self._logger = getLogger()
        # Will convert string input automatically
        self._content = TreeContent(content)
        self._base_tree = deepcopy(route.reaction_tree)

        self._fp_factory = fp_factory or StandardFingerprintFactory()
        self._add_fingerprints(self._base_tree)

        if self._content != TreeContent.MOLECULES and not single_node_tree:
            self._add_fingerprints(self._base_tree["children"][0], self._base_tree)

        if self._content == TreeContent.MOLECULES:
            self._base_tree = self._remove_children_nodes(self._base_tree)
        elif not single_node_tree and self._content == TreeContent.REACTIONS:
            self._base_tree = self._remove_children_nodes(self._base_tree["children"][0])

        self._trees = []
        self._tree_count, self._node_index_list = self._inspect_tree()
        self._enumeration = self._tree_count <= exhaustive_limit

        if self._enumeration:
            self._create_all_trees()
        else:
            self._trees.append(self._base_tree)

        self._dist_func = dist_func

    @property
    def info(self) -> StrDict:
        """Return a dictionary with internal information about the wrapper"""
        return {
            "content": self._content,
            "tree count": self._tree_count,
            "enumeration": self._enumeration,
        }

    @property
    def first_tree(self) -> StrDict:
        """Return the first created ordered tree"""
        return self._trees[0]

    @property
    def trees(self) -> List[StrDict]:
        """Return a list of all created ordered trees"""
        return self._trees

    def distance_iter(self, other: "ReactionTreeWrapper", exhaustive_limit: int = 20) -> _FloatIterator:
        """
        Iterate over all distances computed between this and another tree

        There are three possible enumeration of distances possible dependent
        on the number of possible ordered trees for the two routes that are compared

        * If the product of the number of possible ordered trees for both routes are
          below `exhaustive_limit` compute the distance between all pair of trees
        * If both self and other has been fully enumerated (i.e. all ordered trees has been created)
          compute the distances between all trees of the route with the most ordered trees and
          the first tree of the other route
        * Compute `exhaustive_limit` number of distances by shuffling the child order for
          each of the routes.

        The rules are applied top-to-bottom.

        :param other: another tree to calculate distance to
        :param exhaustive_limit: used to determine what type of enumeration to do
        :yield: the next computed distance between self and other
        """
        if self._tree_count * other.info["tree count"] < exhaustive_limit:
            yield from self._distance_iter_exhaustive(other)
        elif self._enumeration or other.info["enumeration"]:
            yield from self._distance_iter_semi_exhaustive(other)
        else:
            yield from self._distance_iter_random(other, exhaustive_limit)

    def distance_to(self, other: "ReactionTreeWrapper", exhaustive_limit: int = 20) -> float:
        """
        Calculate the minimum distance from this route to another route

        Enumerate the distances using `distance_iter`.

        :param other: another tree to calculate distance to
        :param exhaustive_limit: used to determine what type of enumeration to do
        :return: the minimum distance
        """
        min_dist = 1e6
        min_iter = -1
        for iteration, distance in enumerate(self.distance_iter(other, exhaustive_limit)):
            if distance < min_dist:
                min_iter = iteration
                min_dist = distance
        self._logger.debug(f"Found minimum after {min_iter} iterations")
        return min_dist

    def distance_to_with_sorting(self, other: "ReactionTreeWrapper") -> float:
        """
        Compute the distance to another tree, by simpling sorting the children
        of both trees. This is not guaranteed to return the minimum distance.

        :param other: another tree to calculate distance to
        :return: the distance
        """
        config = AptedConfig(sort_children=True, dist_func=self._dist_func)
        return Apted(self.first_tree, other.first_tree, config).compute_edit_distance()

    def _add_fingerprints(self, tree: StrDict, parent: StrDict = None) -> None:
        if "fingerprint" not in tree:
            try:
                self._fp_factory(tree, parent)
            except ValueError:
                pass
        if "fingerprint" not in tree:
            tree["fingerprint"] = []
        tree["sort_key"] = "".join(f"{digit}" for digit in tree["fingerprint"])
        if "children" not in tree:
            tree["children"] = []

        for child in tree["children"]:
            for grandchild in child["children"]:
                self._add_fingerprints(grandchild, child)

    def _create_all_trees(self) -> None:
        self._trees = []
        # Iterate over all possible combinations of child order
        for order_list in itertools.product(*self._node_index_list):
            self._trees.append(self._create_tree_recursively(self._base_tree, list(order_list)))

    def _create_tree_recursively(
        self,
        node: StrDict,
        order_list: List[List[int]],
    ) -> StrDict:
        new_tree = self._make_base_copy(node)
        children = node.get("children", [])
        if children:
            child_order = order_list.pop(0)
            assert len(child_order) == len(children)
            new_children = [self._create_tree_recursively(child, order_list) for child in children]
            new_tree["children"] = [new_children[idx] for idx in child_order]
        return new_tree

    def _distance_iter_exhaustive(self, other: "ReactionTreeWrapper") -> _FloatIterator:
        self._logger.debug(f"APTED: Exhaustive search. {len(self.trees)} {len(other.trees)}")
        config = AptedConfig(randomize=False, dist_func=self._dist_func)
        for tree1, tree2 in itertools.product(self.trees, other.trees):
            yield Apted(tree1, tree2, config).compute_edit_distance()

    def _distance_iter_random(self, other: "ReactionTreeWrapper", ntimes: int) -> _FloatIterator:
        self._logger.debug(f"APTED: Heuristic search. {len(self.trees)} {len(other.trees)}")
        config = AptedConfig(randomize=False, dist_func=self._dist_func)
        yield Apted(self.first_tree, other.first_tree, config).compute_edit_distance()

        config = AptedConfig(randomize=True, dist_func=self._dist_func)
        for _ in range(ntimes):
            yield Apted(self.first_tree, other.first_tree, config).compute_edit_distance()

    def _distance_iter_semi_exhaustive(self, other: "ReactionTreeWrapper") -> _FloatIterator:
        self._logger.debug(f"APTED: Semi-exhaustive search. {len(self.trees)} {len(other.trees)}")
        if len(self.trees) < len(other.trees):
            first_wrapper = self
            second_wrapper = other
        else:
            first_wrapper = other
            second_wrapper = self

        config = AptedConfig(randomize=False, dist_func=self._dist_func)
        for tree1 in first_wrapper.trees:
            yield Apted(tree1, second_wrapper.first_tree, config).compute_edit_distance()

    def _inspect_tree(self) -> Tuple[int, List[List[int]]]:
        """
        Find the number of children for each node in the tree, which
        will be used to compute the number of possible combinations of child orders

        Also accumulate the possible child orders for the nodes.
        """

        def _recurse_tree(node):
            children = node.get("children", [])
            nchildren = len(children)
            permutations.append(math.factorial(nchildren))

            if nchildren > 0:
                node_index_list.append(list(self._index_permutations[nchildren]))
            for child in children:
                _recurse_tree(child)

        permutations: List[int] = []
        node_index_list: List[List[int]] = []
        _recurse_tree(self._base_tree)
        if not permutations:
            return 0, []
        return int(np.prod(permutations)), node_index_list

    @staticmethod
    def _make_base_copy(node: StrDict) -> StrDict:
        return {
            "type": node["type"],
            "smiles": node.get("smiles", ""),
            "metadata": node.get("metadata"),
            "fingerprint": node["fingerprint"],
            "sort_key": node["sort_key"],
            "children": [],
        }

    @staticmethod
    def _remove_children_nodes(tree: StrDict) -> StrDict:
        new_tree = ReactionTreeWrapper._make_base_copy(tree)

        if tree.get("children"):
            new_tree["children"] = []
            for child in tree["children"]:
                new_tree["children"].extend(
                    [ReactionTreeWrapper._remove_children_nodes(grandchild) for grandchild in child.get("children", [])]
                )
        return new_tree
