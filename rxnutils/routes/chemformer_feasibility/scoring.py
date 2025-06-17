""" Scoring function for feasibility using Chemformer model
"""

from typing import List, Dict


from rxnutils.routes.base import SynthesisRoute
from rxnutils.routes.chemformer_feasibility import (
    ChemformerReactionFeasibilityCalculator,
)
from rxnutils.chem.augmentation import single_reactant_augmentation
from rxnutils.chem.utils import join_smiles_from_reaction, split_rsmi


def reaction_feasibility_score(
    route: SynthesisRoute,
    feasibility_calculator: ChemformerReactionFeasibilityCalculator,
    use_reactant_heuristics: bool = True,
) -> float:
    """
    A scorer that uses the Chemformer API to calculate the feasibility of each
    reaction and then combine it into a route score.

    :param route: the route to score
    :param feasibility_calculator: the interface to the Chemformer API
    :param use_reactant_heuristics: if True, will augment the calculations with
                                    heuristics for reactant reactions
    :return: the computed score
    """
    smiles = _get_reaction_smiles(route, use_reactant_heuristics)
    if len(smiles) == 0:
        return 1.0

    # Send the reaction SMILES of the route to the Chemformer API
    reaction2feasibility = feasibility_calculator(smiles)

    first_reaction = route.reaction_tree["children"][0]
    scores = _score_reaction_tree(
        first_reaction,
        route.reaction_tree["smiles"],
        reaction2feasibility,
        use_reactant_heuristics,
    )
    return scores


def _get_reaction_smiles(
    route: SynthesisRoute, use_reactant_heuristics: bool
) -> List[str]:
    """
    Get the reaction SMILES that should be sent to the Chemformer model
    If `use_reactant_heuristics` is True it will augment the SMILES if
    possible for single-reactant reactions, otherwise it will exclude it.

    :route: the route to assemble the SMILES for
    :use_reactant_heuristics: if to apply the reaction augmentation
    :return: the reaction SMILES
    """
    smiles_list = route.reaction_smiles(augment=use_reactant_heuristics)
    if not use_reactant_heuristics:
        return smiles_list
    smiles_list_pruned = []
    for smiles in smiles_list:
        reactants = split_rsmi(smiles)[0]
        if "." in reactants:
            smiles_list_pruned.append(smiles)
    return smiles_list_pruned


def _score_reaction_tree(
    tree_dict: Dict,
    product_smiles: str,
    feasibilities: Dict[str, float],
    use_reactant_heuristics: bool,
) -> float:
    """Recursive scorer for feasibility

    :param tree_dict: the current parent node
    :param product_smiles: the SMILES of the previous product in the tree
    :param feasibility: a map between reaction SMILES and feasibilities
    :param use_reactant_heuristics: if to apply the reaction augmentation
    :return: the score
    """
    reactants = join_smiles_from_reaction(
        [child["smiles"] for child in tree_dict["children"]]
    )
    reaction_smiles = f"{reactants}>>{product_smiles}"

    if use_reactant_heuristics:
        reaction_smiles = single_reactant_augmentation(
            reaction_smiles,
            tree_dict["metadata"].get("classification", ""),
        )
    score = feasibilities.get(reaction_smiles, 1.0)

    child_scores = []
    for child in tree_dict["children"]:
        grandchildren = child.get("children", [])
        if grandchildren:
            child_score = _score_reaction_tree(
                grandchildren[0],
                child["smiles"],
                feasibilities,
                use_reactant_heuristics,
            )
            child_scores.append(child_score)
    if child_scores:
        score *= min(child_scores)
    return score
