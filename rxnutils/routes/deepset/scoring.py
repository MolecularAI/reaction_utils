""" 
Module containing routines for Deepset route scoring

This is an implementation of the code used in the following publication,
original implementation by Guo Yujia

Yujia G, Kabeshov M, Le THD, Genheden S, Bergonzini G, Engkvist O, et al. 
A Deep Learning with Expert Augmentation Approach for Route Scoring in Organic Synthesis. ChemRxiv. 2024; 
doi:10.26434/chemrxiv-2024-tp7rh 
"""

from typing import Dict

import numpy as np
from rdkit import Chem

from rxnutils.chem.features.sc_score import SCScore
from rxnutils.routes.base import SynthesisRoute
from rxnutils.routes.deepset.featurizers import collect_reaction_features, default_reaction_featurizer, ecfp_fingerprint

try:
    import onnxruntime
except ImportError:
    SCORE_SUPPORTED = False
else:
    SCORE_SUPPORTED = True


class DeepsetModelClient:
    """
    Interface for an in-memory instance of a Deepset model for route scoring

    :params model_path: the path to an ONNX model file
    """

    def __init__(self, model_path: str):
        if not SCORE_SUPPORTED:
            raise ImportError("Cannot score routes with Deepset model. Install rxnutils with optional dependencies")

        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        self._deepnet = onnxruntime.InferenceSession(model_path, sess_options=session_options)

    def __call__(self, reaction_features: np.ndarray, route_features: np.ndarray) -> float:
        """
        Computes the route score using reaction and route features

        :param reaction_features: the packed reaction features
        :param route_features: the route features
        """
        input_ = {
            "reaction_features": reaction_features.astype(np.float32),
            "route_features": route_features.astype(np.float32),
        }
        return float(self._deepnet.run(None, input_)[0])


def deepset_route_score(
    route: SynthesisRoute,
    model_client: DeepsetModelClient,
    scscorer: SCScore,
    reaction_class_ranks: Dict[str, int],
) -> float:
    """
    Scores a synthesis route using a Deepset model

    Currently, it uses defaults for featurizers as
    described in the original publication.

    :params route: the route to score
    :params model_client: the interface to the model
    :params scscorer: the interface to the SCScore
    :params reaction_class_ranks: the lookup from reaction class to rank
    :returns: the computed score
    """
    # To avoid circular imports
    from rxnutils.routes.scoring import badowski_route_score

    target_mol = Chem.MolFromSmiles(route.reaction_tree["smiles"])
    target_fp = ecfp_fingerprint(target_mol)

    rank_score, reaction_features = collect_reaction_features(
        route.reaction_data(),
        target_fp,
        reaction_class_ranks,
        default_reaction_featurizer,
    )
    cost_score = badowski_route_score(route)
    volume_score = sum(route.leaf_counts().values())
    complexity_score = _complexity_score(route, scscorer)
    route_features = np.asarray([cost_score, volume_score, complexity_score, rank_score])
    return model_client(reaction_features, route_features)


def _complexity_score(route: SynthesisRoute, scscorer: SCScore) -> float:
    """Sums up the SCScore of all the intermediates in the route"""
    score = 0
    for smiles, count in route.intermediate_counts().items():
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            continue
        scscore = scscorer(mol)
        score += count * scscore
    return score
