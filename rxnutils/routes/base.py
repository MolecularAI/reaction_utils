"""
Contains a class encapsulating a synthesis route,
as well as routines for assigning proper atom-mapping
and drawing the route
"""

import warnings
from collections import defaultdict
from copy import deepcopy
from operator import itemgetter
from typing import Any, Callable, Dict, List, Set, Tuple, Union

import pandas as pd
from PIL.Image import Image as PilImage
from rdkit import Chem

from rxnutils.chem.augmentation import single_reactant_augmentation
from rxnutils.chem.utils import (
    atom_mapping_numbers,
    join_smiles_from_reaction,
    split_rsmi,
    split_smiles_from_reaction,
)
from rxnutils.pipeline.actions.reaction_mod import NameRxn, RxnMapper
from rxnutils.routes.image import RouteImageFactory
from rxnutils.routes.utils.validation import validate_dict


class SynthesisRoute:
    """
    This encapsulates a synthesis route or a reaction tree.
    It provide convinient methods for assigning atom-mapping
    to the reactions, and for providing reaction-level data
    of the route

    It is typically initiallized by one of the readers in the
    `rxnutils.routes.readers` module.

    The tree depth and the forward step is automatically assigned
    to each reaction node.

    The `max_depth` attribute holds the longest-linear-sequence (LLS)

    :param reaction_tree: the tree structure representing the route
    """

    def __init__(self, reaction_tree: Dict[str, Any]) -> None:
        validate_dict(reaction_tree)
        self.reaction_tree = reaction_tree
        self.max_depth = _assign_tree_depth(reaction_tree)
        _assign_forward_step(reaction_tree, self.max_depth)
        self._nsteps = -1

    @property
    def mapped_root_smiles(self) -> str:
        """
        Return the atom-mapped SMILES of the root compound

        Will raise an exception if the route is a just a single
        compound, or if the route has not been assigned atom-mapping.
        """
        if len(self.reaction_tree.get("children", [])) == 0:
            raise ValueError("Single-compound root does not have this property")

        first_reaction = self.reaction_tree["children"][0]
        if "mapped_reaction_smiles" not in first_reaction.get("metadata", {}):
            raise ValueError(
                "It appears that the route has no atom-mapping information"
            )

        return split_rsmi(first_reaction["metadata"]["mapped_reaction_smiles"])[-1]

    @property
    def nsteps(self) -> int:
        """Return the number of reactions in the route"""
        if self._nsteps == -1:
            self._nsteps = len(self.reaction_smiles())
        return self._nsteps

    def atom_mapped_reaction_smiles(self) -> List[str]:
        """Returns a list of the atom-mapped reaction SMILES in the route"""
        smiles = []
        _collect_atom_mapped_smiles(self.reaction_tree, smiles)
        return smiles

    def assign_atom_mapping(
        self, overwrite: bool = False, only_rxnmapper: bool = False
    ) -> None:
        """
        Assign atom-mapping to each reaction in the route and
        ensure that is is consistent from root compound and throughout
        the route.

        It will use NameRxn to assign classification and possiblty atom-mapping,
        as well as rxnmapper to assign atom-mapping in case NameRxn cannot classify
        a reaction.

        :param overwrite: if True will overwrite existing mapping
        :param only_rxnmapper: if True will disregard NameRxn mapping and use only rxnmapper
        """
        self._assign_mapping(overwrite, only_rxnmapper)
        map_nums = atom_mapping_numbers(self.mapped_root_smiles)
        if not map_nums:
            raise ValueError("Assigning atom-mapping failed")
        max_atomnum = max(map_nums)
        _inherit_atom_mapping(self.reaction_tree, max_atomnum + 100)

    def chains(
        self, complexity_func: Callable[[str], float]
    ) -> List[List[Dict[str, Any]]]:
        """
        Returns linear sequences or chains extracted from the route.

        Each chain is a list of a dictionary representing the molecules, only the most
        complex molecule is kept for each reaction - making the chain a sequence of molecule
        to molecule transformation.

        The first chain will be the longest linear sequence (LLS), and the second chain
        will be longest branch if this is a convergent route. This branch will be processed
        further, but the other branches can probably be discarded as they have not been
        investigated thoroughly.

        :param complexity_func: a function that takes a SMILES and returns a
                            complexity metric of the molecule
        :returns: a list of chains where each chain is a list of molecules
        """
        mols = []
        _extract_chain_molecules(self.reaction_tree, mols)
        chains = _extract_chains(mols, complexity_func)

        # Process the longest branch for convergent routes
        # note that other chains will not be processed
        # and thus should probably be discarded
        # TODO: analyse other chains
        if len(chains) > 1:
            last_step = chains[1][-1]["step"]
            mol_copy = dict(chains[0][last_step + 1])
            mol_copy["serial"] += "x"
            chains[1].append(mol_copy)

        # Add name of chain and type of molecule to each molecule
        for idx, chain in enumerate(chains):
            chain_name = "lls" if idx == 0 else f"sub{idx}"
            for mol in chain:
                mol["chain"] = chain_name
                if mol["parent_hash"] is None:
                    mol["type"] = "target"
                elif "hash" not in mol:
                    mol["type"] = "sm"
                else:
                    mol["type"] = "inter"

                if mol["serial"].endswith("x"):
                    mol["type"] = "branch"
                    mol["chain"] = "sub1"
        return chains

    def image(
        self, show_atom_mapping: bool = False, factory_kwargs: Dict[str, Any] = None
    ) -> PilImage:
        """
        Depict the route.

        :param show_atom_mapping: if True, will show the atom-mapping
        :param factory_kwargs: additional keyword arguments sent to the `RouteImageFactory`
        :returns: the image of the route
        """
        factory_kwargs = factory_kwargs or {}
        if show_atom_mapping:
            dict_ = self._create_atom_mapping_tree()
        else:
            dict_ = self.reaction_tree
        return RouteImageFactory(dict_, **factory_kwargs).image

    def intermediate_counts(self) -> Dict[str, int]:
        """
        Extract the counts of all intermediates

        return: the counts
        """
        intermediates = defaultdict(int)
        children = self.reaction_tree.get("children", [])
        if children:
            _collect_intermediates(children[0], intermediates)
        return intermediates

    def intermediates(
        self,
    ) -> Set[str]:
        """
        Extract a set with the SMILES of all the intermediates nodes

        :return: a set of SMILES strings
        """
        return set(self.intermediate_counts().keys())

    def is_solved(self) -> bool:
        """
        Find if this route is solved, i.e. if all starting material
        is in stock.

        To be accurate, each molecule node need to have an extra
        boolean property called `in_stock`.
        """
        try:
            _find_leaves_not_in_stock(self.reaction_tree)
        except ValueError:
            return False
        return True

    def leaf_counts(self) -> Dict[str, int]:
        """
        Extract the counts of all leaf nodes, i.e. starting material

        return: the counts
        """
        leaves = defaultdict(int)
        _collect_leaves(self.reaction_tree, leaves)
        return leaves

    def leaves(
        self,
    ) -> Set[str]:
        """
        Extract a set with the SMILES of all the leaf nodes, i.e.
        starting material

        :return: a set of SMILES strings
        """
        return set(self.leaf_counts().keys())

    def reaction_data(self) -> List[Dict[str, Any]]:
        """
        Returns a list of dictionaries for each reaction
        in the route. This is metadata of the reactions
        augmented with reaction SMILES and depth of the reaction
        """
        data = []
        _collect_reaction_data(self.reaction_tree, data)
        return data

    def reaction_ngrams(self, nitems: int, metadata_key: str) -> List[Tuple[Any, ...]]:
        """
        Extract an n-gram representation of the route by building up n-grams
        of the reaction metadata.

        :param nitems: the length of the gram
        :param metadata_key: the metadata to extract
        :return: the collected n-grams
        """
        if self.max_depth < nitems:
            return []
        first_reaction = self.reaction_tree["children"][0]
        ngrams = []
        _collect_ngrams(first_reaction, nitems, metadata_key, ngrams)
        return ngrams

    def reaction_smiles(self, augment: bool = False) -> List[str]:
        """
        Returns a list of the un-mapped reaction SMILES
        :param augment: if True will add reagents to single-reactant
                          reagents whenever possible
        """
        if not augment:
            smiles_list = []
            _collect_reaction_smiles(self.reaction_tree, smiles_list)
            return smiles_list

        smiles_list = []
        for data in self.reaction_data():
            smiles = data["reaction_smiles"]
            classification = data.get("classification", "")
            smiles_list.append(single_reactant_augmentation(smiles, classification))
        return smiles_list

    def remap(self, other: Union["SynthesisRoute", str, Dict[int, int]]) -> None:
        """
        Remap the reaction so that it follows the mapping of a
        1) root compound in a reference route, 2) a ref compound given
        as a SMILES, or 3) using a raw mapping

        :param other: the reference for re-mapping
        """
        if isinstance(other, SynthesisRoute):
            if len(self.reaction_smiles()) == 0 or len(other.reaction_smiles()) == 0:
                return
            mapping_dict = _find_remapping(
                other.mapped_root_smiles, self.mapped_root_smiles
            )
        elif isinstance(other, str):
            if len(self.reaction_smiles()) == 0:
                return
            mapping_dict = _find_remapping(other, self.mapped_root_smiles)
        elif isinstance(other, dict):
            mapping_dict = other
        else:
            raise ValueError(f"Cannot perform re-mapping using a {type(other)}")
        _remap_reactions(self.reaction_tree, mapping_dict)

    def update_stock(self, stock: Set[str], molfunc: Callable[[str], str]) -> None:
        """
        Update the `in_stock` property of the molecule nodes
        in the route.

        :param stock: the set of SMILES of molecules in stock
        :param molfunc: a function that takes a SMILES and returns a
                        string that can be checked against the stock
        """
        _update_stock(self.reaction_tree, stock, molfunc)

    def _assign_mapping(
        self, overwrite: bool = False, only_rxnmapper: bool = False
    ) -> None:
        if not overwrite:
            try:
                self.atom_mapped_reaction_smiles()
            except KeyError:
                # We will just create the mapping
                pass
            else:
                return

        df = pd.DataFrame({"smiles": list(set(self.reaction_smiles()))})
        nextmove_action = NameRxn(in_column="smiles", nm_rxn_column="mapped_smiles")
        rxnmapper_action = RxnMapper(in_column="smiles")
        if not only_rxnmapper:
            try:
                df = nextmove_action(df)
            # Raised by nextmove_action if namerxn not in path
            except FileNotFoundError:
                df = df.assign(NMC=["0.0"] * len(df), mapped_smiles=[""] * len(df))
                warnings.warn(
                    "namerxn does not appear to be in $PATH. Run failed and proceeding with rxnmapper only"
                )
        else:
            df = df.assign(NMC=["0.0"] * len(df), mapped_smiles=[""] * len(df))
        df = rxnmapper_action(df)
        if only_rxnmapper:
            df["mapped_smiles"] = df["RxnmapperRxnSmiles"]
        else:
            sel = df["NMC"] == "0.0"
            df.loc[sel, "mapped_smiles"] = df.loc[sel, "RxnmapperRxnSmiles"]
        datamap = df.set_index("smiles").to_dict("index")
        _copy_mapping_from_datamap(self.reaction_tree, datamap)

    def _create_atom_mapping_tree(self) -> Dict[str, Any]:
        dict_ = deepcopy(self.reaction_tree)
        _assign_atom_mapped_smiles(dict_)
        return dict_


def _apply_remapping(
    reaction_smiles: str,
    remapping: Dict[int, int],
    keep_original: bool,
) -> str:
    """
    Apply a specific re-mapping to a given reaction

    If an atom has an atom-mapping number not given
    by the re-mapping dictionary, the `keep_original` argument
    determines the action. If `keep_original` is True, the atom
    keeps the atom-mapping number, else it is set to 0 and
    becomes and un-mapped atom.

    :param reaction_smiles: the reaction to remap
    :param remapping: the mapping from old to new atom-map number
    :param keep_original: determines the behavior of atoms not in `remapping`
    :return: the remapped reaction SMILES
    """
    reactants_smiles, reagent_smiles, product_smiles = split_rsmi(reaction_smiles)
    product_mol = Chem.MolFromSmiles(product_smiles)
    reactant_mols = [
        Chem.MolFromSmiles(smiles)
        for smiles in split_smiles_from_reaction(reactants_smiles)
    ]
    atoms_to_renumber = list(product_mol.GetAtoms())
    for mol in reactant_mols:
        atoms_to_renumber.extend(mol.GetAtoms())
    for atom in atoms_to_renumber:
        if atom.GetAtomMapNum() and atom.GetAtomMapNum() in remapping:
            atom.SetAtomMapNum(remapping[atom.GetAtomMapNum()])
        elif not keep_original:
            atom.SetAtomMapNum(0)
    return ">".join(
        [
            join_smiles_from_reaction([Chem.MolToSmiles(mol) for mol in reactant_mols]),
            reagent_smiles,
            Chem.MolToSmiles(product_mol),
        ]
    )


def _extract_chains(
    chain_molecules: List[Dict[str, Any]], complexity_func: Callable[[str], float]
) -> List[List[Dict[str, Any]]]:
    """
    Take extracted molecules from a route and compile them into chains or sequences.

    The molecules will first be sorted based on complexity and then by step.
    Therefore the chains will only contain the most complex reactant of a reaction,
    and the first chain will be the longest linear sequence (LLS)

    Single-molecule chains will be discarded.

    :param chain_molecules: the list of molecules
    :param complexity_func: a function that takes a SMILES and returns a
                            complexity metric of the molecule
    :returns: a list of chains where each chain is a list of molecules
    """
    # Add complexity and serial number to each molecule
    for idx, mol in enumerate(chain_molecules):
        mol["complexity"] = complexity_func(mol["smiles"])
        mol["serial"] = str(len(chain_molecules) - idx - 1)
    chain_molecules.sort(key=itemgetter("complexity"), reverse=True)
    chain_molecules.sort(key=itemgetter("step"))

    chains = []
    residuals = list(chain_molecules)
    while len(residuals) > 0:
        if len(residuals) == 1:
            chains.append([residuals[0]])
            break

        # Start a new chain with the first residual molecule
        # then find the linear sequence from that molecule
        new_chain = [residuals[0]]
        residuals = residuals[1:]
        # Loop until we don't find a parent molecule
        found = True
        while found:
            found = False
            # Look for a parent molecule in the residuals
            for mol in residuals:
                if "hash" in mol and mol["hash"] == new_chain[-1]["parent_hash"]:
                    new_chain.append(mol)
                    found = True
                    break
        residuals = [mol for mol in residuals if not mol in new_chain]
        chains.append(new_chain)

    chains.sort(key=len, reverse=True)
    chains = [chain for chain in chains if len(chain) > 1]
    return chains


def _find_remapping(parent_smiles: str, target_smiles: str) -> Dict[int, int]:
    """
    Find the mapping between two identical copies of a compound
    that have different atom-mapping.

    :param parent_smiles: the reference SMILES for which the mapping should be changed to
    :param target_smiles: the other SMILES that will be re-mapped
    :return: the translation of atom mapping from `product_smiles` to `parent_smiles`
    """
    remapping = {}
    pmol = Chem.MolFromSmiles(parent_smiles)
    tmol = Chem.MolFromSmiles(target_smiles)

    for atom_idx1, atom_idx2 in enumerate(pmol.GetSubstructMatch(tmol)):
        atom1 = tmol.GetAtomWithIdx(atom_idx1)
        atom2 = pmol.GetAtomWithIdx(atom_idx2)
        if atom1.GetAtomMapNum() > 0 and atom2.GetAtomMapNum() > 0:
            remapping[atom1.GetAtomMapNum()] = atom2.GetAtomMapNum()
    return remapping


def smiles2inchikey(smiles: str, ignore_stereo: bool = False) -> str:
    """Converts a SMILES to an InChI key"""
    inchi = Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))
    if not ignore_stereo:
        return inchi
    return inchi.split("-")[0]


# Recursive functions acting on reaction trees


def _assign_atom_mapped_smiles(
    tree_dict: Dict[str, Any], reactants_smiles: List[str] = None
) -> None:
    """
    Used to copy the atom-mapped SMILES from the reaction metadata
    to the SMILES property of each molecule node. This is used to
    display atom-mapping when depicting routes.

    The matching of atom-mapped SMILES and unmapped SMILES are
    based on partial InChI key without stereoinformation

    :param tree_dict: the current molecule node
    :param reactants_smiles: the list of reactants SMILES of the parent reaction
    """
    tree_dict["unmapped_smiles"] = tree_dict["smiles"]
    # For leaf nodes
    if not tree_dict.get("children", []):
        inchi = smiles2inchikey(tree_dict["smiles"], ignore_stereo=True)
        found = -1
        for idx, smi in enumerate(reactants_smiles):
            if inchi == smiles2inchikey(smi, ignore_stereo=True):
                found = idx
                break
        if found > -1:
            tree_dict["smiles"] = reactants_smiles.pop(found)
    else:
        reaction_dict = tree_dict["children"][0]
        reaction_smiles = reaction_dict["metadata"]["mapped_reaction_smiles"]
        reactants, _, tree_dict["smiles"] = split_rsmi(reaction_smiles)
        reactants_smiles = split_smiles_from_reaction(reactants)
        for grandchild in reaction_dict["children"]:
            _assign_atom_mapped_smiles(grandchild, reactants_smiles)


def _assign_forward_step(
    tree_dict: Dict[str, Any], max_depth: int, depth: int = 1
) -> None:
    """
    Assign the forward_step property of each reaction node.
    The forward step is defined as "maximum depth - node depth + 1"

    :param tree_dict: the current molecule node
    :param max_depth: the maximum depth of the full tree
    :param depth: the depth of the current molecule node + 1
    """
    children = tree_dict.get("children")
    if children is None:
        return
    grandchildren = children[0]["children"]
    children[0]["metadata"]["forward_step"] = max_depth - depth + 1
    for grandchild in grandchildren:
        _assign_forward_step(grandchild, max_depth, depth + 1)


def _assign_tree_depth(tree_dict: Dict[str, Any], depth: int = 1) -> int:
    """
    Assign the tree_depth property of each reaction node,
    which is the depth of the node in the tree relative to the root node.
    First reaction is at depth 1.

    :param tree_dict: the current molecule node
    :param depth: the depth of the current molecule node + 1
    :return: the maximum depth of the tree
    """
    children = tree_dict.get("children")
    if children is None:
        return depth - 1
    grandchildren = children[0]["children"]
    children[0]["metadata"]["tree_depth"] = depth
    child_depths = []
    for grandchild in grandchildren:
        child_depths.append(_assign_tree_depth(grandchild, depth + 1))
    return max(child_depths)


def _collect_atom_mapped_smiles(tree_dict: Dict[str, Any], smiles: List[str]) -> None:
    """
    Save atom-mapped SMILES from the reaction metadata in a list

    :param tree_dict: the current molecule node
    :param smiles: the list of collect reaction SMILES
    """
    children = tree_dict.get("children")
    if children is None:
        return
    smiles.append(children[0]["metadata"]["mapped_reaction_smiles"])
    for grandchild in children[0]["children"]:
        _collect_atom_mapped_smiles(grandchild, smiles)


def _collect_intermediates(
    tree_dict: Dict[str, Any], intermediates: Dict[str, int]
) -> None:
    """
    Traverse the tree and collect SMILES of molecule nodes that has children

    :param tree_dict: the current molecule node
    :param intermediates: the list of collected leaves and their counts
    """
    if tree_dict["type"] == "mol" and tree_dict.get("children"):
        intermediates[tree_dict["smiles"]] += 1
    for child in tree_dict.get("children", []):
        _collect_intermediates(child, intermediates)


def _collect_leaves(tree_dict: Dict[str, Any], leaves: Dict[str, int]) -> None:
    """
    Traverse the tree and collect SMILES of molecule nodes that has no children

    :param tree_dict: the current molecule node
    :param leaves: the list of collected leaves and their counts
    """
    children = tree_dict.get("children", [])
    if children:
        for child in children:
            _collect_leaves(child, leaves)
    else:
        leaves[tree_dict["smiles"]] += 1


def _collect_ngrams(
    tree_dict: Dict[str, Any],
    nitems: int,
    metadata_key: str,
    result: List[Tuple[Any, ...]],
    accumulation: List[str] = None,
):
    """
    Collect ngrams from reaction metadata

    :param tree_dict: the current reaction node in the recursion
    :param nitems: the length of the gram
    :param metadata_key: the metadata to extract
    :param result: the collected ngrams
    :param accumulation: the accumulate items in the recursion
    :raise ValueError: if this routine is initialized from a molecule node
    """
    accumulation = accumulation or []
    if tree_dict["type"] == "mol":
        raise ValueError(
            "Found _collect_ngrams at molecule node. This should not happen."
        )

    data = tree_dict.get("metadata", {}).get(metadata_key)
    accumulation.append(data)

    if len(accumulation) == nitems:
        result.append(tuple(accumulation))
        accumulation.pop(0)

    for mol_child in tree_dict["children"]:
        for rxn_grandchild in mol_child.get("children", []):
            _collect_ngrams(
                rxn_grandchild, nitems, metadata_key, result, list(accumulation)
            )


def _collect_reaction_data(
    tree_dict: Dict[str, Any], data: List[Dict[str, Any]]
) -> None:
    """
    Save the reaction metadata to a list and augment it
    with the un-mapped reaction SMILES and reaction depth

    :param tree_dict: the current molecule node
    :param smiles: the list of collect reaction data
    """
    children = tree_dict.get("children")
    if children is None:
        return
    grandchildren = children[0]["children"]
    reactants = join_smiles_from_reaction(
        [grandchild["smiles"] for grandchild in grandchildren]
    )
    metadata = deepcopy(children[0]["metadata"])
    metadata["reaction_smiles"] = f"{reactants}>>{tree_dict['smiles']}"
    data.append(metadata)
    for grandchild in grandchildren:
        _collect_reaction_data(grandchild, data)


def _collect_reaction_smiles(tree_dict: Dict[str, Any], smiles: List[str]) -> None:
    """
    Save reaction SMILES from the molecule nodes

    :param tree_dict: the current molecule node
    :param smiles: the list of collect reaction SMILES
    """
    children = tree_dict.get("children")
    if children is None:
        return
    grandchildren = children[0]["children"]
    reactants = join_smiles_from_reaction(
        [grandchild["smiles"] for grandchild in grandchildren]
    )
    smiles.append(f"{reactants}>>{tree_dict['smiles']}")
    for grandchild in grandchildren:
        _collect_reaction_smiles(grandchild, smiles)


def _copy_mapping_from_datamap(
    tree_dict: Dict[str, Any], datamap: Dict[str, Dict[str, str]]
) -> None:
    """
    Store the generated atom-mapping and classification to the reaction
    metadata.

    :param tree_dict: the current molecule node
    :param datamap: a mapping from original reaction SMILES to
                    NameRxn classication and mapping as well
                    as rxnmapper mapping
    """
    children = tree_dict.get("children")
    if children is None:
        return
    grandchildren = children[0]["children"]
    reactants = join_smiles_from_reaction(
        [grandchild["smiles"] for grandchild in grandchildren]
    )
    rxnsmi = f"{reactants}>>{tree_dict['smiles']}"
    metadata = children[0].get("metadata", {})
    metadata["classification"] = datamap[rxnsmi]["NMC"]
    metadata["mapped_reaction_smiles"] = datamap[rxnsmi]["mapped_smiles"]
    metadata = children[0]["metadata"] = metadata
    for grandchild in grandchildren:
        _copy_mapping_from_datamap(grandchild, datamap)


def _extract_chain_molecules(
    tree_dict: Dict[str, Any],
    chain_molecules: List[Dict[str, Any]],
    step: int = None,
    parent_hash: str = None,
) -> None:
    """
    Extract molecules in the route in a format that is suitable for
    extracting the chains or sequences of the route.

    Each molecule is represented by a dictionary that contains
        * smiles - the SMILES string of the molecule
        * parent_hash - the hash of the parent molecule
        * step - the synthesis (forward) step number
        * classification - the reaction classification of the reaction
                           applied to the molecule
        * reaction_id - the reaction id of the reaction applied to the molecule
        * hash - the hash of the molecule (SMILES+step), for non-leaf molecules

    :param tree_dict: the current molecule node
    :param chain_molecules: the extracted molecules
    :param step: the forward step of the molecule
    :param parent_hash: the hash of the previous molecule node
    """
    mol_dict = {
        "smiles": tree_dict["smiles"],
        "parent_hash": parent_hash,
    }
    children = tree_dict.get("children")
    if not children:
        mol_dict["step"] = step or 0
        mol_dict["classification"] = ""
        mol_dict["reaction_id"] = None
    else:
        metadata = children[0]["metadata"]
        step = step or metadata["forward_step"]
        mol_dict["step"] = step
        mol_dict["classification"] = metadata.get("classification", "")
        mol_dict["reaction_id"] = metadata.get("id", metadata.get("ID"))
        mol_dict["hash"] = hash((mol_dict["smiles"], mol_dict["step"]))
        for grandchild in children[0]["children"]:
            _extract_chain_molecules(
                grandchild, chain_molecules, step - 1, mol_dict["hash"]
            )
    chain_molecules.append(mol_dict)


def _find_leaves_not_in_stock(tree_dict: Dict[str, Any]) -> None:
    """
    Traverse the tree and check the `in_stock` value of molecule
    nodes without children (leaves). If one is found that is not
    in stock this raises an exception, which stops the traversal.
    """
    children = tree_dict.get("children", [])
    if not children and not tree_dict.get("in_stock", True):
        raise ValueError(f"child not in stock {tree_dict}")
    elif children:
        for child in children:
            _find_leaves_not_in_stock(child)


def _inherit_atom_mapping(
    tree_dict: Dict[str, Any], new_atomnum: int, parent_smiles: str = ""
) -> None:
    """
    Replace atom-mapping in the route so that it keeps the atom-mapping
    from the root node, i.e. the target molecule

    The matching of atom-mapped SMILES and unmapped SMILES are
    based on partial InChI key without stereoinformation. Very rarely
    the atom-mapping tool inverts stereochemistry, and this works
    in those situations.

    :param tree_dict: the current molecule node
    :param new_atomnum: the atom mapping number to assign to unmapped atoms
    :param parent_smiles: the SMILE of the parent molecule node
    """
    # If at leaf node, just return
    if not tree_dict.get("children", []):
        return

    # Extract reaction and atom-mapped reaction SMILES
    reaction_dict = tree_dict["children"][0]
    mapped_rsmi0 = reaction_dict["metadata"]["mapped_reaction_smiles"]

    # If we are at the root, i.e. target node, we just use the atom-mapping as-is
    if parent_smiles == "":
        mapped_rsmi = mapped_rsmi0
    else:
        _, _, product0 = split_rsmi(mapped_rsmi0)
        remapping = _find_remapping(parent_smiles, product0)
        product_mol = Chem.MolFromSmiles(product0)
        for atom in product_mol.GetAtoms():
            if atom.GetAtomMapNum() and atom.GetAtomMapNum() not in remapping:
                new_atomnum += 1
                remapping[atom.GetAtomMapNum()] = new_atomnum
        mapped_rsmi = _apply_remapping(mapped_rsmi0, remapping, keep_original=False)

    reaction_dict["metadata"]["mapped_reaction_smiles"] = mapped_rsmi

    # Recursing to reactants
    inchi2mapped = {
        smiles2inchikey(smiles, ignore_stereo=True): smiles
        for smiles in split_smiles_from_reaction(split_rsmi(mapped_rsmi)[0])
    }
    for grandchild in reaction_dict["children"]:
        # When a molecule is a complex use each part to inherit atom mapping
        for smiles in split_smiles_from_reaction(grandchild["smiles"]):
            inchi = smiles2inchikey(smiles, ignore_stereo=True)
            _inherit_atom_mapping(grandchild, new_atomnum, inchi2mapped[inchi])


def _remap_reactions(tree_dict: Dict[str, Any], remapping: Dict[int, int]) -> None:
    """
    Remapp each of the reaction using a remapping dictionary

    :param tree_dict: the current molecule node
    :param remapping: the remapping dictionary
    """
    children = tree_dict.get("children")
    if children is None:
        return
    reaction_smiles = children[0]["metadata"]["mapped_reaction_smiles"]
    children[0]["metadata"]["mapped_reaction_smiles"] = _apply_remapping(
        reaction_smiles, remapping, keep_original=True
    )
    for grandchild in children[0]["children"]:
        _remap_reactions(grandchild, remapping)


def _update_stock(
    tree_dict: Dict[str, Any], stock: set[str], molfunc: Callable[[str], str]
) -> None:
    """
    Update the `in_stock` property of the molecule nodes"""
    tree_dict["in_stock"] = molfunc(tree_dict["smiles"]) in stock

    children = tree_dict.get("children", [])
    if not children:
        return

    for grandchildren in children[0]["children"]:
        _update_stock(grandchildren, stock, molfunc)
