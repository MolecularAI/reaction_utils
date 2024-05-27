"""
Contains a class encapsulating a synthesis route,
as well as routines for assigning proper atom-mapping
and drawing the route
"""

from typing import Dict, Any, List, Callable, Union
from copy import deepcopy
from operator import itemgetter

import pandas as pd
from PIL.Image import Image as PilImage
from rdkit import Chem

from rxnutils.pipeline.actions.reaction_mod import NameRxn, RxnMapper
from rxnutils.routes.image import RouteImageFactory
from rxnutils.chem.utils import (
    atom_mapping_numbers,
    split_smiles_from_reaction,
    join_smiles_from_reaction,
)


class SynthesisRoute:
    """
    This encapsulates a synthesis route or a reaction tree.
    It provide convenient methods for assigning atom-mapping
    to the reactions, and for providing reaction-level data
    of the route

    It is typically initiallized by one of the readers in the
    `rxnutils.routes.readers` module.

    The tree depth and the forward step are automatically assigned
    to each reaction node.

    :param reaction_tree: the tree structure representing the route
    """

    def __init__(self, reaction_tree: Dict[str, Any]) -> None:
        self.reaction_tree = reaction_tree
        self.max_depth = _assign_tree_depth(reaction_tree)
        _assign_forward_step(reaction_tree, self.max_depth)

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

        return first_reaction["metadata"]["mapped_reaction_smiles"].split(">")[-1]

    def atom_mapped_reaction_smiles(self) -> List[str]:
        """Returns a list of the atom-mapped reaction SMILES in the route"""
        smiles = []
        _collect_atom_mapped_smiles(self.reaction_tree, smiles)
        return smiles

    def assign_atom_mapping(
        self,
        overwrite: bool = False,
        only_rxnmapper: bool = False,
    ) -> None:
        """
        Assign atom-mapping to each reaction in the route and
        ensure that it is consistent from root compound and throughout
        the route.

        It will use NameRxn to assign classification and possiblty atom-mapping,
        as well as rxnmapper to assign atom-mapping in case NameRxn cannot classify
        a reaction.

        :param overwrite: if True will overwrite existing mapping
        :param only_rxnmapper: if True will disregard NameRxn mapping and use only rxnmapper
        """
        self._assign_mapping(overwrite, only_rxnmapper)
        max_atomnum = max(atom_mapping_numbers(self.mapped_root_smiles))
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

    def image(self, show_atom_mapping=False) -> PilImage:
        """
        Depict the route.

        :param show_atom_mapping: if True, will show the atom-mapping
        :returns: the image of the route
        """
        if show_atom_mapping:
            dict_ = self._create_atom_mapping_tree()
        else:
            dict_ = self.reaction_tree
        return RouteImageFactory(dict_).image

    def reaction_data(self) -> List[Dict[str, Any]]:
        """
        Returns a list of dictionaries for each reaction
        in the route. This is metadata of the reactions
        augmented with reaction SMILES and depth of the reaction
        """
        data = []
        _collect_reaction_data(self.reaction_tree, data)
        return data

    def reaction_smiles(self) -> List[str]:
        """Returns a list of the un-mapped reaction SMILES"""
        smiles = []
        _collect_reaction_smiles(self.reaction_tree, smiles)
        return smiles

    def remap(self, other: "SynthesisRoute") -> None:
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
        df = rxnmapper_action(nextmove_action(df))
        if only_rxnmapper:
            df["mapped_smiles"] = df["RxnmapperRxnSmiles"]
        else:
            sel = df["NMC"] == "0.0"
            df["mapped_smiles"].mask(sel, df["RxnmapperRxnSmiles"], inplace=True)
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
    reactants_smiles, reagent_smiles, product_smiles = reaction_smiles.split(">")
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
    tree_dict: Dict[str, Any], reaction_smiles: str = ""
) -> None:
    """
    Used to copy the atom-mapped SMILES from the reaction metadata
    to the SMILES property of each molecule node. This is used to
    display atom-mapping when depicting routes.

    The matching of atom-mapped SMILES and unmapped SMILES are
    based on partial InChI key without stereoinformation

    :param tree_dict: the current molecule node
    :param reaction_smiles: the reaction SMILES of the parent reaction
    """
    tree_dict["unmapped_smiles"] = tree_dict["smiles"]
    # For leaf nodes
    if not tree_dict.get("children", []):
        inchi2mapped = {
            smiles2inchikey(smiles, ignore_stereo=True): smiles
            for smiles in split_smiles_from_reaction(reaction_smiles.split(">")[0])
        }
        inchi = smiles2inchikey(tree_dict["smiles"], ignore_stereo=True)
        tree_dict["smiles"] = inchi2mapped.get(inchi, tree_dict["smiles"])
    else:
        reaction_dict = tree_dict["children"][0]
        reaction_smiles = reaction_dict["metadata"]["mapped_reaction_smiles"]
        tree_dict["smiles"] = reaction_smiles.split(">")[-1]
        for grandchild in reaction_dict["children"]:
            _assign_atom_mapped_smiles(grandchild, reaction_smiles)


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
        _, _, product0 = mapped_rsmi0.split(">")
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
        for smiles in split_smiles_from_reaction(mapped_rsmi.split(">>")[0])
    }
    for grandchild in reaction_dict["children"]:
        inchi = smiles2inchikey(grandchild["smiles"], ignore_stereo=True)
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
