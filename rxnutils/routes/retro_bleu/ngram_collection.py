"""
Contains routines for creating, reading, and writing n-gram collections

Can be run as a module to create a collection from a set of routes:

    python -m rxnutils.routes.retro_bleu.ngram_collection --filename routes.json --output ngrams.json --nitems 2 --metadata template_hash

"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Set, Tuple

from tqdm import tqdm

from rxnutils.routes.base import SynthesisRoute


@dataclass
class NgramCollection:
    """
    Class to create, read, and write a collection of n-grams

    :param nitems: the length of each n-gram
    :param metadata_key: the key used to extract the n-grams from the reactions
    :param ngrams: the extracted n-grams
    """

    nitems: int
    metadata_key: str
    ngrams: Set[Tuple[str, ...]]

    @classmethod
    def from_file(cls, filename: str) -> "NgramCollection":
        """
        Read an n-gram collection from a JSON-file

        :param filename: the path to the file
        :return: the n-gram collection
        """
        with open(filename, "r") as fileobj:
            dict_ = json.load(fileobj)
        ngrams = {tuple(item.split("\t")) for item in dict_["ngrams"]}
        return NgramCollection(dict_["nitems"], dict_["metadata_key"], ngrams)

    @classmethod
    def from_tree_collection(cls, filename: str, nitems: int, metadata_key: str) -> "NgramCollection":
        """
        Make a n-gram collection by extracting them from a collection of
        synthesis routes.

        :param filename: the path to a file with a list of synthesis routes
        :param nitems: the length of the gram
        :param metadata_key: the metadata to extract
        :return: the n-gram collection
        """
        with open(filename, "r") as fileobj:
            tree_list = json.load(fileobj)

        ngrams = set()
        for tree_dict in tqdm(tree_list, leave=False):
            route = SynthesisRoute(tree_dict)
            ngrams = ngrams.union(route.reaction_ngrams(nitems, metadata_key))
        return NgramCollection(nitems, metadata_key, ngrams)

    def save_to_file(self, filename: str) -> None:
        """
        Save an n-gram collection to a JSON-file

        :param filename: the path to the file
        """
        dict_ = {
            "nitems": self.nitems,
            "metadata_key": self.metadata_key,
            "ngrams": ["\t".join(ngram) for ngram in self.ngrams],
        }
        with open(filename, "w") as fileobj:
            json.dump(dict_, fileobj)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Tool for making n-gram collections from a set of synthesis routes")
    parser.add_argument("--filename", nargs="+", help="the path to the synthesis routes")
    parser.add_argument("--output", help="the path to the n-gram collection")
    parser.add_argument("--nitems", type=int, help="the length of the gram")
    parser.add_argument("--metadata", help="the reaction metadata to extract for making the n-grams")
    args = parser.parse_args()

    collection = None
    for filename in args.filename:
        temp = NgramCollection.from_tree_collection(filename, args.nitems, args.metadata)
        if collection is None:
            collection = temp
        else:
            collection.ngrams = collection.ngrams.union(temp.ngrams)
    print(f"Collected unique {len(collection.ngrams)} {args.nitems}-grams")
    collection.save_to_file(args.output)
