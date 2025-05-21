"""Routines for reading routes from various formats"""

import copy
from typing import Any, Dict, List, Sequence

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from rxnutils.chem.utils import join_smiles_from_reaction, split_rsmi, split_smiles_from_reaction
from rxnutils.routes.base import SynthesisRoute, smiles2inchikey


def read_reaction_lists(filename: str) -> List[SynthesisRoute]:
    """
    Read one or more simple lists of reactions into one or more
    retrosynthesis trees.

    Each list of reactions should be separated by an empty line.
    Each row of each reaction should contain the reaction SMILES (reactants>>products)
    and nothing else.

    Example:
    A.B>>C
    D.E>>B

    A.X>>Y
    Z>>X

    defines two retrosynthesis trees, and the first being
         A
    C ->      D
         B ->
              E

    :params filename: the path to the file with the reactions
    :returns: the list of the created trees
    """
    with open(filename, "r") as fileobj:
        all_lines = fileobj.read()

    reaction_lists = [lines.splitlines() for lines in all_lines.split("\n\n")]
    return [reactions2route(reactions) for reactions in reaction_lists]


def read_aizynthcli_dataframe(data: pd.DataFrame) -> pd.Series:
    """
    Read routes as produced by the `aizynthcli` tool of the `AiZynthFinder` package.

    :param data: the dataframe as output by `aizynthcli`
    :return: the created routes
    """

    def read_row(row: pd.Series) -> List[SynthesisRoute]:
        return [read_aizynthfinder_dict(tree) for tree in row.trees]

    return data.apply(read_row, axis=1)


def read_aizynthfinder_dict(tree: Dict[str, Any]) -> SynthesisRoute:
    """
    Read a single aizynthfinder dictionary

    :param tree: the aizynthfinder structure
    :return: the created routes
    """
    dict_ = copy.deepcopy(tree)
    _transform_retrosynthesis_atom_mapping(dict_)
    return SynthesisRoute(dict_)


def read_reactions_dataframe(
    data: pd.DataFrame,
    smiles_column: str,
    group_by: List[str],
    metadata_columns: List[str] = None,
) -> pd.Series:
    """
    Read routes from reactions stored in a pandas dataframe. The different
    routes are groupable by one or more column. Additional metadata columns
    can be extracted from the dataframe as well.

    The dataframe is grouped by the columns specified by `group_by` and
    then one routes is extracted from each subset dataframe. The function
    returns a series with the routes, which is indexable by the columns
    in the `group_by` list.

    :param data: the dataframe with reaction data
    :param smiles_column: the column with the reaction SMILES
    :param group_by: the columns that uniquely identifies each route
    :param metadata_column: additional columns to be added as metadata to each route
    :return: the created series with route.
    """

    def reaction_dataframe2routes(subdata):
        smiles = subdata[smiles_column]
        metadata = subdata[metadata_columns].to_dict("records")
        try:
            route = reactions2route(smiles, metadata)
        except ValueError:
            id_str = ", ".join([str(subdata.iloc[0][col]) for col in group_by])
            print(f"Failed to re-create synthesis tree for: ({id_str})")
            route = None
        dict_ = {col: subdata[col].iloc[0] for col in group_by}
        dict_["route"] = route
        return route

    metadata_columns = metadata_columns or []
    grouped = data.groupby(group_by)[data.columns.to_list()]
    return grouped.apply(reaction_dataframe2routes)


def reactions2route(reactions: Sequence[str], metadata: Sequence[Dict[str, Any]] = None) -> SynthesisRoute:
    """
    Convert a list of reactions into a retrosynthesis tree

    This is based on matching partial InChI keys of the reactants in one
    reaction with the partial InChI key of a product.

    :params reactions: list of reaction SMILES
    :returns: the created trees
    """

    def make_dict(product_smiles):
        dict_ = {
            "type": "mol",
            "smiles": product_smiles,
        }
        product_inchi = smiles2inchikey(product_smiles, ignore_stereo=True)
        reaction = product2reaction.get(product_inchi)
        if reaction is not None:
            metadata = dict(reaction["metadata"])
            metadata["reaction_smiles"] = join_smiles_from_reaction(reaction["reactants"]) + ">>" + reaction["product"]
            dict_["children"] = [
                {
                    "type": "reaction",
                    "metadata": metadata,
                    "children": [make_dict(smiles) for smiles in reaction["reactants"]],
                }
            ]
        return dict_

    metadata = metadata or [dict() for _ in range(len(reactions))]
    all_reactants = set()
    product2reaction = {}
    inchi_map = {}
    for reaction, meta in zip(reactions, metadata):
        reactants_smiles, _, product_smiles = split_rsmi(reaction)
        product_smiles = Chem.CanonSmiles(product_smiles)
        partial_product_inchi = smiles2inchikey(product_smiles, ignore_stereo=True)
        inchi_map[partial_product_inchi] = product_smiles
        reactants = split_smiles_from_reaction(reactants_smiles)
        try:
            reactants = [Chem.CanonSmiles(smi) for smi in reactants]
        except:
            raise ValueError(f"Cannot canonicalize SMILES: {reactants_smiles}")
        all_reactants = all_reactants.union([smiles2inchikey(smi, ignore_stereo=True) for smi in reactants])
        product2reaction[partial_product_inchi] = {
            "smiles": reaction,
            "product": product_smiles,
            "reactants": reactants,
            "metadata": meta,
        }

    only_products = set(product2reaction.keys()) - all_reactants
    if len(only_products) != 1:
        raise ValueError(f"Could not identify one and only one target product: {only_products}")

    target_inchi = list(only_products)[0]
    return SynthesisRoute(make_dict(inchi_map[target_inchi]))


def read_rdf_file(filename: str) -> SynthesisRoute:
    def finish_reaction():
        if not rxnblock:
            return
        rxn = AllChem.ReactionFromRxnBlock("\n".join(rxnblock), sanitize=False, strictParsing=False)
        reactions.append(AllChem.ReactionToSmiles(rxn))

    with open(filename, "r") as fileobj:
        lines = fileobj.read().splitlines()

    reactions = []
    rxnblock = []
    read_rxn = skip_entry = False
    for line in lines:
        if line.startswith("$RFMT"):
            read_rxn = skip_entry = False
            finish_reaction()
            rxnblock = []
        elif line.startswith(("$MFMT", "$DATUM")):
            # Ignore MFMT and DATUM entries for now
            skip_entry = True
        elif skip_entry:
            continue
        elif line.startswith("$RXN"):
            rxnblock = [line]
            read_rxn = True
        elif read_rxn:
            rxnblock.append(line)
    finish_reaction()

    return reactions2route(reactions)


def _transform_retrosynthesis_atom_mapping(tree_dict: Dict[str, Any]) -> None:
    """
    Routes output from AiZynth has atom-mapping from the template-based model,
    but it needs to be processed
    1. Remove atom-mapping from reactants not in product
    2. Reverse reaction SMILES
    """
    if not tree_dict.get("children", []):
        return

    reaction_dict = tree_dict["children"][0]
    mapped_rsmi = reaction_dict["metadata"]["mapped_reaction_smiles"]

    product, _, reactants = split_rsmi(mapped_rsmi)
    product_mol = Chem.MolFromSmiles(product)
    product_maps = {atom.GetAtomMapNum() for atom in product_mol.GetAtoms() if atom.GetAtomMapNum()}
    reactant_mols = []
    for smiles in reactants.split("."):
        mol = Chem.MolFromSmiles(smiles)
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() and atom.GetAtomMapNum() not in product_maps:
                atom.SetAtomMapNum(0)
        reactant_mols.append(mol)

    mapped_rsmi = ">".join(
        [
            join_smiles_from_reaction([Chem.MolToSmiles(mol) for mol in reactant_mols]),
            "",
            Chem.MolToSmiles(product_mol),
        ]
    )
    reaction_dict["metadata"]["mapped_reaction_smiles"] = mapped_rsmi

    for grandchild in reaction_dict["children"]:
        _transform_retrosynthesis_atom_mapping(grandchild)
