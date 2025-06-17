"""
This module contains an interface for calculating reaction feasibility
with a REST API to a Chemformer model or output from batch predictions
"""

from typing import List, Dict, Any

import scipy
import requests
import numpy as np
import pandas as pd
from rdkit import Chem
from requests.exceptions import ConnectionError  # pylint: disable=redefined-builtin

from rxnutils.chem.utils import split_rsmi


class ChemformerReactionFeasibilityCalculator:
    """
    Interface to the Chemformer API to calculate forward feasibility

    Once instantiated, the calculator can be called as a function to calculate
    feasibilities of a set of reactions

        scores = calculator(list_of_smiles)

    the return value is a dictionary mapping reaction SMILES to the calculated
    feasibility.

    :param api_url: the URL to the REST API
    """

    def __init__(self, api_url: str) -> None:
        self._api_url = api_url
        self._cache: Dict[str, float] = {}
        self.retries = 3

    def __call__(self, reactions: List[str]) -> Dict[str, float]:
        unchached_reactions = [
            reaction for reaction in reactions if reaction not in self._cache
        ]
        if unchached_reactions:
            self._update_cache(unchached_reactions)

        return {reaction: self._cache[reaction] for reaction in reactions}

    def load_batch_output(self, data: pd.DataFrame):
        """
        Given an output from the `predict` function of the chemformer
        package, and an additional column called `reactants` this will
        update the cache with those predictions.

        :param data: the batch output
        """

        def process_row(row):
            predictions = [
                row[column]
                for column in row.index
                if column.startswith("sampled_smiles")
            ]
            llhs = [
                row[column]
                for column in row.index
                if column.startswith("loglikelihood")
            ]
            true_product = row["target_smiles"]
            return self._calculate_reaction_feasibility(
                true_product, predictions, scipy.special.softmax(llhs)
            )

        smiles = data["reactants"] + ">>" + data["target_smiles"]
        feasibility = data.apply(process_row, axis=1)
        self._cache.update(dict(zip(smiles, feasibility)))

    def _submit_request(self, reactants: List[str]) -> Dict[str, Any]:
        headers = {"accept": "application/json", "Content-Type": "application/json"}
        response = None
        for _ in range(self.retries):
            try:
                response = requests.post(self._api_url, headers=headers, json=reactants)
            except ConnectionError:
                continue
            else:
                if response.status_code == requests.codes.ok:
                    break

        if not response:
            raise ConnectionError(
                "Could not retrieve Chemformer predictions. Could not connect."
            )
        if response.status_code != requests.codes.ok:
            raise ValueError(
                f"Could not retrieve Chemformer predictions. Response: {response.content}"
            )

        return response.json()

    def _update_cache(self, uncached_reactions: List[str]) -> None:
        uncached_reactants = [
            split_rsmi(reaction)[0] for reaction in uncached_reactions
        ]
        predictions = self._submit_request(uncached_reactants)
        for reaction, prediction in zip(uncached_reactions, predictions):
            product = split_rsmi(reaction)[-1]
            self._cache[reaction] = self._calculate_reaction_feasibility(
                product, prediction["output"], scipy.special.softmax(prediction["lhs"])
            )

    @staticmethod
    def _calculate_reaction_feasibility(
        true_product: str,
        predicted_products: List[str],
        prediction_probabilities: np.ndarray,
    ) -> float:
        """
        Lookup the true product among the predicted products and return the
        probability of it. It is not found return 0.0
        """
        true_inchikey = Chem.MolToInchiKey(Chem.MolFromSmiles(true_product))
        for pred, prob in zip(predicted_products, prediction_probabilities):
            mol = Chem.MolFromSmiles(pred)
            if not mol:
                continue
            pred_inchikey = Chem.MolToInchiKey(mol)
            if pred_inchikey == true_inchikey:
                return prob
        return 0.0
