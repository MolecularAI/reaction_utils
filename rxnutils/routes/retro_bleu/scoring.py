""" Contains routine to score routes according to Retro-BLEU paper
"""

import numpy as np

from rxnutils.routes.base import SynthesisRoute
from rxnutils.routes.retro_bleu.ngram_collection import NgramCollection


def ngram_overlap_score(
    route: SynthesisRoute,
    ref: NgramCollection,
) -> float:
    """
    Calculate the fractional n-gram overlap of the n-grams in the given
    route and the reference n-gram collection

    :param route: the route to score
    :param ref: the reference n-gram collection
    :return: the calculated score
    """
    route_ngrams = set(route.reaction_ngrams(ref.nitems, ref.metadata_key))
    if not route_ngrams:
        return np.nan
    return len(route_ngrams.intersection(ref.ngrams)) / len(route_ngrams)


def retro_bleu_score(route: SynthesisRoute, ref: NgramCollection, ideal_steps: int = 3) -> float:
    """
    Calculate the Retro-BLEU score according to the paper:

    Li, Junren, Lei Fang, och Jian-Guang Lou.
    ”Retro-BLEU: quantifying chemical plausibility of retrosynthesis routes through reaction template sequence analysis”.
    Digital Discovery 3, nr 3 (2024): 482–90. https://doi.org/10.1039/D3DD00219E.

    :param route: the route to score
    :param ref: the reference n-gram collection
    :param ideal_steps: a length-penalization hyperparameter (see Eq 2 in ref)
    :return: the calculated score
    """
    overlap = ngram_overlap_score(route, ref)

    nreactions = len(route.reaction_smiles())
    heuristic_score = ideal_steps / max(nreactions, ideal_steps)

    return np.exp(overlap) + np.exp(heuristic_score)
