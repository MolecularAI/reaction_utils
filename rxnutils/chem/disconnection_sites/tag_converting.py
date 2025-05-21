from __future__ import annotations

import re
from typing import List, Tuple

from rdkit import Chem

from rxnutils.chem.utils import remove_atom_mapping


def smiles_tokens(smiles: str) -> List[str]:
    """
    Tokenize SMILES using basic regex pattern for Chemformer.

    :param smiles: SMILES to tokenize
    :return: List of tokens identified in SMILES.
    """
    pattern = (
        r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\\|\/|:|~|@|\?|>|\*|\!|\$|\%[0-9]{2}|[0-9])"
    )
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smiles)]

    tokenized_smiles = "".join(tokens)
    if smiles != tokenized_smiles:
        raise AssertionError(
            f"tokenized SMILES not the same as input SMILES: {tokenized_smiles}, " "{smiles}, tokens: {tokens}"
        )
    return tokens


def _next_tagged_token(product_tagged_tokens: List[str], untagged_token: str, tagged_token_idx: int) -> Tuple[str, int]:
    """
    Get the next tagged token in the sequence. Includes checks and fixes for
    stereochemistry changes due to removing atom mapping.

    :param product_tagged_tokens: tokens of product tagged with [<atom>:1]
    :param untagged_token: the current token from the untagged product
    :param tagged_token_idx: the current token index of the tagged product
    :return: the next (tagged-product) token and the corresponding token index
    """
    tagged_token = product_tagged_tokens[tagged_token_idx]

    # Check if the stereo chemistry has changed after removing atom-mapping and
    # handle each specific case.
    if tagged_token != untagged_token and (tagged_token == "/" or tagged_token == "\\"):
        if untagged_token == "/" or untagged_token == "\\":
            return untagged_token, tagged_token_idx
        else:
            tagged_token_idx += 1
            return product_tagged_tokens[tagged_token_idx], tagged_token_idx

    if tagged_token != untagged_token and not ":1" in tagged_token and "@" in tagged_token:
        return untagged_token, tagged_token_idx

    return tagged_token, tagged_token_idx


def tagged_smiles_from_tokens(product_tagged_tokens: List[str], product_untagged_tokens: List[str]) -> Tuple[str, str]:
    """
    Convert the tagged SMILES from atom-mapping to unmapped-token + '!'

    :param product_tagged_tokens: tokens of product tagged with [<atom>:1]
    :param product_untagged_tokens: tokens of the untagged product

    :return: Tuple of SMILES of the product containing tags corresponding to atoms changed in the
        reaction using "<atom>!", and SMILES of the (reconstructed) untagged product
    """

    product_converted = ""
    product_untagged = ""

    tagged_token_idx = 0

    for untagged_token in product_untagged_tokens:

        tagged_token, tagged_token_idx = _next_tagged_token(product_tagged_tokens, untagged_token, tagged_token_idx)

        if tagged_token != untagged_token and (untagged_token == "/" or untagged_token == "\\"):
            continue

        if tagged_token == untagged_token:
            product_converted += untagged_token
        else:
            # Remove brackets around a single letter
            if len(untagged_token) == 3 and untagged_token.startswith("[") and untagged_token.endswith("]"):
                untagged_token = untagged_token[1]
            product_converted += untagged_token + "!"

        product_untagged += untagged_token

        tagged_token_idx += 1

    return product_converted, product_untagged


def _canonicalize_tagged_smiles(product_tagged: str, product_untagged: str = None) -> Tuple[str, str]:
    """
    Reorder the tagged-product SMILES on canonical form using the canonicalized
    untagged product.

    :param product_tagged: SMILES of tagged product
    :param product_untagged: SMILES of untagged product
    :return: canonicalized untagged and tagged product SMILES
    """
    mol = Chem.MolFromSmiles(product_tagged)
    mol_untagged = Chem.MolFromSmiles(product_untagged)

    _, canonical_atom_order = tuple(zip(*sorted([(j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(mol_untagged))])))

    mol = Chem.RenumberAtoms(mol, canonical_atom_order)
    mol_untagged = Chem.RenumberAtoms(mol_untagged, canonical_atom_order)
    return Chem.MolToSmiles(mol, canonical=False), Chem.MolToSmiles(mol_untagged)


def convert_atom_map_tag(product_atom_map_tagged: str) -> str:
    """
    Replace product tagged by atom-mapping [<atom>:1] to product tagged by "<atom>!".
    Returns empty string if no atom-map tagging or the failed to create untagged product.

    :param product_tagged: SMILES of the product containing tags corresponding to
        atoms changed in the reaction using [<atom>:1]
    :return: SMILES of the product containing tags corresponding to atoms changed in the
        reaction using "<atom>!"
    """

    # Check number of tags
    n_tags = len(re.findall(r"\[[^\]]+:1]", product_atom_map_tagged))

    if n_tags < 1:
        return ""

    product_untagged = remove_atom_mapping(product_atom_map_tagged, canonical=False)

    if not Chem.MolFromSmiles(product_untagged):
        return ""

    product_tagged, product_untagged = _canonicalize_tagged_smiles(product_atom_map_tagged, product_untagged)

    # Update the SMILES string to remove atom-mapping brackets and explicit [H]:s and
    # replace by <atom>!
    product_tagged_tokens = smiles_tokens(product_tagged)
    product_untagged_tokens = smiles_tokens(product_untagged)

    product_tagged_converted, product_untagged = tagged_smiles_from_tokens(
        product_tagged_tokens, product_untagged_tokens
    )

    n_new_tags = product_tagged_converted.count("!")

    if n_new_tags != n_tags:
        raise AssertionError(
            f"The number of tags is not the same after converting to '!' tagging. "
            f"product_tagged_atom_map: {product_atom_map_tagged}"
            f"product_tagged_converted: {product_tagged_converted}."
        )

    if product_tagged_converted.replace("!", "") != product_untagged:
        raise AssertionError(
            f"product_tagged.replace('!', '') != product_untagged."
            f"product_tagged: {product_tagged_converted}, product_untagged: {product_untagged}"
        )

    return product_tagged_converted
