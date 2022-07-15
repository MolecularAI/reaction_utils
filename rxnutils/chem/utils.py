"""Module containing various chemical utility routines"""
import logging
import functools
from typing import List, Tuple


import rdchiral.template_extractor

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize


def get_special_groups(mol) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """
    Given an RDKit molecule, this function returns a list of tuples, where
    each tuple contains the AtomIdx's for a special group of atoms which should
    be included in a fragment all together. This should only be done for the
    reactants, otherwise the products might end up with mapping mismatches
    We draw a distinction between atoms in groups that trigger that whole
    group to be included, and "unimportant" atoms in the groups that will not
    be included if another atom matches.
    """
    # Define templates
    group_templates = [
        (
            range(3),
            "[OH0,SH0]=C[O,Cl,I,Br,F]",
        ),  # carboxylic acid / halogen
        (
            range(3),
            "[OH0,SH0]=CN",
        ),  # amide/sulfamide
        (
            range(4),
            "S(O)(O)[Cl]",
        ),  # sulfonyl chloride
        (
            range(3),
            "B(O)O",
        ),  # boronic acid/ester
        ((0,), "[Si](C)(C)C"),  # trialkyl silane
        ((0,), "[Si](OC)(OC)(OC)"),  # trialkoxy silane, default to methyl
        (
            range(3),
            "[N;H0;$(N-[#6]);D2]-,=[N;D2]-,=[N;D1]",
        ),  # azide
        (
            range(8),
            "O=C1N([Br,I,F,Cl])C(=O)CC1",
        ),  # NBS brominating agent
        (range(11), "Cc1ccc(S(=O)(=O)O)cc1"),  # Tosyl
        ((7,), "CC(C)(C)OC(=O)[N]"),  # N(boc)
        ((4,), "[CH3][CH0]([CH3])([CH3])O"),  # t-Butyl
        (
            range(2),
            "[C,N]=[C,N]",
        ),  # alkene/imine
        (
            range(2),
            "[C,N]#[C,N]",
        ),  # alkyne/nitrile
        (
            (2,),
            "C=C-[*]",
        ),  # adj to alkene
        (
            (2,),
            "C#C-[*]",
        ),  # adj to alkyne
        (
            (2,),
            "O=C-[*]",
        ),  # adj to carbonyl
        ((3,), "O=C([CH3])-[*]"),  # adj to methyl ketone
        (
            (3,),
            "O=C([O,N])-[*]",
        ),  # adj to carboxylic acid/amide/ester
        (
            range(4),
            "ClS(Cl)=O",
        ),  # thionyl chloride
        (
            range(2),
            "[Mg,Li,Zn,Sn][Br,Cl,I,F]",
        ),  # grinard/metal (non-disassociated)
        (
            range(3),
            "S(O)(O)",
        ),  # SO2 group
        (
            range(2),
            "N~N",
        ),  # diazo
        (
            (1,),
            "[!#6;R]@[#6;R]",
        ),  # adjacency to heteroatom in ring
        (
            (2,),
            "[a!c]:a:a",
        ),  # two-steps away from heteroatom in aromatic ring
        # ((1,), 'c(-,=[*]):c([Cl,I,Br,F])',), # ortho to halogen on ring - too specific?
        # ((1,), 'c(-,=[*]):c:c([Cl,I,Br,F])',), # meta to halogen on ring - too specific?
        ((0,), "[B,C](F)(F)F"),  # CF3, BF3 should have the F3 included
    ]

    # Stereo-specific ones (where we will need to include neighbors)
    # Tetrahedral centers should already be okay...
    group_templates += [
        (
            (
                1,
                2,
            ),
            "[*]/[CH]=[CH]/[*]",
        ),  # trans with two hydrogens
        (
            (
                1,
                2,
            ),
            "[*]/[CH]=[CH]\\[*]",
        ),  # cis with two hydrogens
        (
            (
                1,
                2,
            ),
            "[*]/[CH]=[CH0]([*])\\[*]",
        ),  # trans with one hydrogens
        (
            (
                1,
                2,
            ),
            "[*]/[D3;H1]=[!D1]",
        ),  # specified on one end, can be N or C
    ]

    # Amol addition
    # Carried over from previous iteration (before merge with rdchiral)
    # There is some redundancy from merge with rdchiral and templates are not general
    group_templates += [
        (range(3), "C(=O)Cl"),  # acid chloride
        (range(4), "[$(N-!@[#6])](=!@C=!@O)"),  # isocyanate
        (range(2), "C=O"),  # carbonyl
        (range(4), "ClS(Cl)=O"),  # thionyl chloride
        (range(5), "[#6]S(=O)(=O)[O]"),  # RSO3 leaving group
        (range(5), "[O]S(=O)(=O)[O]"),  # SO4 group
        (range(3), "[N-]=[N+]=[C]"),  # diazo-alkyl
        (
            range(15),
            "[#6][C]([#6])([#6])[#8]-[#6](=[O])-[#8]-[#6](=O)-[#8]C([#6])([#6])[#6]",
        ),  # Boc Anhydride
        # Protecting Groups
        # Amino
        (range(18), "[#7]-[#6](=O)-[#8]-[#6]-[#6]-1-c2ccccc2-c2ccccc-12"),  # Fmoc
        (range(8), "[#6]C([#6])([#6])[#8]-[#6](-[#7])=O"),  # Boc
        (range(11), "[#7]-[#6](=O)-[#8]-[#6]-c1ccccc1"),  # Cbz
        (range(6), "[#7]-[#6](=O)-[#8]-[#6]=[#6]"),  # Voc
        (range(7), "[#7]-[#6](=O)-[#8]-[#6]-[#6]=[#6]"),  # Alloc
        (range(4), "[#6]-[#6](-[#7])=O"),  # Acetamide
        (range(7), "[#7]-[#6](=O)C(F)(F)F"),  # Trifluoroacetamide
        (range(7), "[#7]-[#6](=O)C(Cl)(Cl)Cl"),  # Trichloroacetamide
        (range(11), "O=[#6]-1-[#7]-[#6](=O)-c2ccccc-12"),  # Phthalimide
        (range(8), "[#7]-[#6]-c1ccccc1"),  # Benzylamine
        (range(10), "[#6]-[#8]-c1ccc(-[#6]-[#7])cc1"),  # p-Methoxybenzyl (PMB)
        (range(9), "[#6]-[#8]-c1ccc(-[#7])cc1"),  # PMP
        (range(20), "[#7]C(c1ccccc1)(c1ccccc1)c1ccccc1"),  # Triphenylmethylamine
        (range(8), "[#7]=[#6]-c1ccccc1"),  # N-Benzylidene
        (range(11), "[#6]-c1ccc(cc1)S([#7])(=O)=O"),  # p-Toluenesulfonamide
        (range(13), "[#7]S(=O)(=O)c1ccc(cc1)-[#7+](-[#8-])=O"),  # p-Nosylsulfonamide
        (range(5), "[#6]S([#7])(=O)=O"),  # Mesylate
        (
            range(7),
            "[#7]-[#6]-1-[#6]-[#6]-[#6]-[#6]-[#8]-1",
        ),  # N-Tetrahydropyranyl ether (N-THP)
        (range(14), "[#7]=[#6](-c1ccccc1)-c1ccccc1"),  # N-Benzhydrylidene
        (range(14), "[#7]-[#6](-c1ccccc1)-c1ccccc1"),  # N-Benzhydryl
        (range(5), "[#6]-[#7](-[#6])-[#6]=[#7]"),  # Dimethylaminomethylene
        (range(9), "[#7]-[#6](=O)-c1ccccc1"),  # N-Benzoate (N-Bz)
        (range(8), "[#7]=[#6]-c1ccccc1"),  # N-Benzylidene
        # Carbonyl
        (range(6), "[#6]-[#8]-[#6](-[#6])-[#8]-[#6]"),  # Dimethyl acetal
        (range(6), "[#6]-1-[#6]-[#8]-[#6]-[#8]-[#6]-1"),  # 1,3-Dioxane
        (range(5), "[#6]-1-[#6]-[#8]-[#6]-[#8]-1"),  # 1,3-Dioxolane
        (range(5), "[#6]-1-[#6]-[#16]-[#6]-[#16]-1"),  # 1,3-Dithiane
        (range(6), "[#6]-1-[#6]-[#16]-[#6]-[#16]-[#6]-1"),  # 1,3-Dithiolane
        (range(5), "[#6]-[#7](-[#6])-[#7]=[#6]"),  # N,N-Dimethylhydrazone
        (range(8), "[#6]C1([#6])[#6]-[#8]-[#6]-[#8]-[#6]1"),  # 1,3-Dioxolane (dimethyl)
        # Carboxyl
        (range(5), "[#6]-[#8]-[#6](-[#6])=O"),  # Methyl ester
        (range(6), "[#6]-[#6]-[#8]-[#6](-[#6])=O"),  # Ethyl ester
        (range(8), "[#6]-[#6](=O)-[#8]C([#6])([#6])[#6]"),  # t-Butyl ester
        (
            range(11),
            "[#6]-[#6](=O)-[#8]-[#6]-[#6]-1=[#6]-[#6]=[#6]-[#6]=[#6]-1",
        ),  # Benzyl ester
        (range(8), "[#6]-[#6](=O)-[#8][Si]([#6])([#6])[#6]"),  # Silyl Ester
        (range(8), "[#6]-[#6](=O)-[#16]C([#6])([#6])[#6]"),  # S-t-Butyl ester
        (range(6), "[#6]-[#6]-1=[#7]-[#6]-[#6]-[#8]-1"),  # 2-Alkyl-1,3-oxazoline
        # Hydroxyl
        (range(5), "[#6]-[#8]-[#6]-[#8]-[#6]"),  # Methoxymethyl ether
        (
            range(8),
            "[#6]-[#8]-[#6]-[#6]-[#8]-[#6]-[#8]-[#6]",
        ),  # 2-Methoxyethoxymethyl ether
        (
            range(9),
            "[#8]-[#6](=O)-[#8]-[#6]C(Cl)(Cl)Cl",
        ),  # 2,2,2-Trichloroethyl carbonate
        (range(7), "[#6]-[#8]C([#6])([#6])[#8]-[#6]"),  # Methoxypropyl acetal
        (range(7), "[#6]-[#6]-[#8]-[#6](-[#6])-[#8]-[#6]"),  # Ethoxyethyl acetal
        (
            range(11),
            "[#6]-[#8]-[#6]-[#8]-[#6]-[#6]-1=[#6]-[#6]=[#6]-[#6]=[#6]-1",
        ),  # Benzyloxymethyl acetal
        (
            range(8),
            "[#6]-[#8]-[#6]-1-[#6]-[#6]-[#6]-[#6]-[#8]-1",
        ),  # Tetrahydropyranyl ether
        (range(10), "[#6]-[#8]-[#6](=O)-c1ccccc1"),  # Benzoate (O-Bz)
        (range(6), "[#6]-[#8]C([#6])([#6])[#6]"),  # t-Butyl ether
        (range(5), "[#6]-[#8]-[#6]-[#6]=[#6]"),  # Allyl ether
        (range(13), "[#6]-[#8]-[#6]-c1ccc2ccccc2c1"),  # Napthyl ether
        (range(11), "[#6]-[#8]-[#6]-c1ccc(-[#8]-[#6])cc1"),  # p-Methoxybenzyl ether
        (range(6), "[#6]-[#8][Si]([#6])([#6])[#6]"),  # Trimethylsilyl ether
        (
            range(8),
            "[#6]-[#6](-[#6])-[#6][Si]([#6])([#6])[#8]",
        ),  # dimethylisopropylsilyl ether
        (
            range(9),
            "[#6]-[#6][Si]([#6]-[#6])([#6]-[#6])[#8]-[#6]",
        ),  # Triethylsilyl ether
        (
            range(12),
            "[#6]-[#8][Si]([#6](-[#6])-[#6])([#6](-[#6])-[#6])[#6](-[#6])-[#6]",
        ),  # Triisopropylsilyl ether
        (
            range(9),
            "[#6]-[#8][Si]([#6])([#6])C([#6])([#6])[#6]",
        ),  # t-Butyldimethylsilyl ether
        (
            range(10),
            "[#6]-[#8]-[#6]-[#8]-[#6]-[#6][Si]([#6])([#6])[#6]",
        ),  # [2-(trimethylsilyl)ethoxy]methyl
        (
            range(19),
            "[#6]-[#8][Si]([#6]-1=[#6]-[#6]=[#6]-[#6]=[#6]-1)([#6]-1=[#6]-[#6]=[#6]-[#6]=[#6]-1)C([#6])([#6])[#6]",
        ),  # t-Butyldiphenylsilyl ether
        (
            range(9),
            "[#6]-[#8][Si]([#6])([#6])C([#6])([#6])[#6]",
        ),  # t-Butyldimethylsilyl ether
        (
            range(20),
            "[#6]-[#6]-[#8][Si]([#6]-[#6]-[#6]-c1ccc(-[#6])cc1)([#6](-[#6])-[#6])[#6](-[#6])-[#6]",
        ),  # propyl-p-toluenediisopropylsilyl ether
        (range(5), "[#6]-[#8]-[#6](-[#6])=O"),  # Acetic acid ester (acetate)
        (range(8), "[#6]-[#8]-[#6](=O)C([#6])([#6])[#6]"),  # Pivalic acid ester
        (range(10), "[#6]-[#8]-[#6](=O)-c1ccccc1"),  # Benzoic acid ester
        (range(9), "[#6]-[#8]-[#6]-c1ccccc1"),  # Benzyl ether (O-Bn)
        # 1',2- 1',3-Diols
        (range(8), "[#6]C1([#6])[#8]-[#6]-[#6]-[#6]-[#8]1"),  # Acetonide
        (
            range(12),
            "[#6]-1-[#6]-[#8]-[#6](-[#8]-[#6]-1)-c1ccccc1",
        ),  # Benzylidene acetal
        # Alkyne
        (range(6), "[#6][Si]([#6])([#6])C#C"),  # Trimethylsilyl ether (TMS-alkyne)
        (
            range(9),
            "[#6]-[#6][Si]([#6]-[#6])([#6]-[#6])C#C",
        ),  # Triethylsilyl ether (TES-alkyne)
        # Thiol
        (range(5), "[#6]-[#16]-[#6](-[#6])=O"),  # S-Carbonyl deprotection (thioester)
        (
            range(15),
            "[#6]-[#6]-[#6](-[#6]-[#6])-[#6]C1([#6]-[#6]-[#6]-[#6]-[#6]1)[#6](-[#16])=O",
        ),  # S-Carbonyl deprotection (thioester)
        (
            range(9),
            "[#6]-[#6]-[#7](-[#6]-[#6])-[#6](=O)-[#16]-[#6]",
        ),  # N,N-ethyl thiocarbamate
        (range(7), "[#6]-[#16]-[#6](=O)-[#7](-[#6])-[#6]"),  # N,N-methyl thiocarbamate
    ]

    # Build list
    groups = []
    for (add_if_match, template) in group_templates:
        matches = mol.GetSubstructMatches(
            Chem.MolFromSmarts(template), useChirality=True
        )
        for match in matches:
            add_if = []
            for pattern_idx, atom_idx in enumerate(match):
                if pattern_idx in add_if_match:
                    add_if.append(atom_idx)
            groups.append((add_if, match))
    return groups


# Monkey patch RDChiral
rdchiral.template_extractor.get_special_groups = get_special_groups


def has_atom_mapping(
    smiles: str, is_smarts: bool = False, sanitize: bool = True
) -> bool:
    """
    Returns True if a molecule has atom mapping, else False.

    :param smiles: the SMILES/SMARTS representing the molecule
    :param is_smarts: if True, will interpret the SMILES as a SMARTS
    :param sanitize: if True, will sanitize the molecule
    :return: True if the SMILES string has atom-mapping, else False
    """
    if not smiles:
        return False

    from_func = {
        False: functools.partial(Chem.MolFromSmiles, sanitize=sanitize),
        True: Chem.MolFromSmarts,
    }
    mol = from_func[is_smarts](smiles)
    if not mol:
        return False

    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() > 0:
            return True
    return False


def remove_atom_mapping(
    smiles: str, is_smarts: bool = False, sanitize: bool = True
) -> str:
    """
    Returns a molecule without atom mapping

    :param smiles: the SMILES/SMARTS representing the molecule
    :param is_smarts: if True, will interpret the SMILES as a SMARTS
    :param sanitize: if True, will sanitize the molecule
    :return: the molecule without atom-mapping
    """
    if not smiles:
        return ""

    # is_smarts=True: Chem.MolFromSmarts
    # is_smarts=False: Chem.MolFromSmiles
    from_func = {
        False: functools.partial(Chem.MolFromSmiles, sanitize=sanitize),
        True: Chem.MolFromSmarts,
    }
    # is_smarts=True: Chem.MolToSmarts
    # is_smarts=False: Chem.MolToSmiles
    to_func = {
        False: Chem.MolToSmiles,
        True: Chem.MolToSmarts,
    }
    mol = from_func[is_smarts](smiles)

    if not mol:
        return smiles

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return to_func[is_smarts](mol)


def remove_atom_mapping_template(template_smarts: str) -> str:
    """Remove atom mapping from a template SMARTS string"""
    if not template_smarts:
        return ""

    rxn = AllChem.ReactionFromSmarts(template_smarts)
    rxn.Initialize()
    AllChem.RemoveMappingNumbersFromReactions(rxn)
    return AllChem.ReactionToSmarts(rxn)


def neutralize_molecules(smiles_list: List[str]) -> List[str]:
    """
    Neutralize a set of molecules using RDKit routines

    :param smiles_list: the molecules as SMILES
    :return: the neutralized molecules
    """
    # Neutralize/Uncharge Molecules
    uncharger = rdMolStandardize.Uncharger()
    # Create RDKit MolObjs
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    # Neutralize Molecules
    neutral_mols = [uncharger.uncharge(mol) for mol in mols if mol]
    if len(mols) > len(neutral_mols):
        logging.warning(
            f"No. of Neutralised molecules ({len(neutral_mols)}) are less than No. of input molecules ({len(mols)})."
        )

    return [Chem.MolToSmiles(mol) for mol in neutral_mols]


def same_molecule(mol1, mol2) -> bool:
    """Test if two molecules are the same.
    First number of atoms and bonds are compared to guard the potentially more expensive
    substructure match. If mol1 is a substructure of mol2 and vice versa, the molecules
    are considered to be the same.

    :param mol1: First molecule
    :param mol2: Second molecule for comparison
    :return: if the molecules match
    """
    return (
        mol1.GetNumHeavyAtoms() == mol2.GetNumHeavyAtoms()
        and (mol1.GetNumBonds() == mol2.GetNumBonds())
        and ((mol1.HasSubstructMatch(mol2)) and (mol2.HasSubstructMatch(mol1)))
    )


def atom_mapping_numbers(smiles: str) -> List[int]:
    """
    Return the numbers in the atom mapping

    :param smiles: the molecule as SMILES
    :return: the atom mapping numbers
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return []
    return [atom.GetAtomMapNum() for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0]


def reassign_rsmi_atom_mapping(rsmi: str, as_smiles: bool = False) -> str:
    """Reassign reaction's atom mapping.
    Remove atom maps for atoms in reactants and reactents not found in product's atoms.

    :param rsmi: Reaction SMILES
    :param as_smiles: Return reaction SMILES or SMARTS, defaults to False
    :return: Reaction SMILES or SMARTS
    """
    rxn = AllChem.ReactionFromSmarts(rsmi)
    logging.debug(f"Original rsmi: {rsmi}")
    logging.debug(f"Reactants: {rxn.GetNumReactantTemplates()}")
    logging.debug(f"Reagents: {rxn.GetNumAgentTemplates()}")
    logging.debug(f"Products: {rxn.GetNumProductTemplates()}")

    # Get Product(s) Atom Maps
    products_atommaps = {
        atom.GetAtomMapNum()
        for product in rxn.GetProducts()
        for atom in product.GetAtoms()
        if atom.GetAtomMapNum()
    }
    logging.debug(f"Product atom maps:{products_atommaps}")

    # Reassign Atom Maps for Rectants
    for reactant in rxn.GetReactants():
        reactant_atommaps = {
            atom.GetIdx(): atom.GetAtomMapNum()
            for atom in reactant.GetAtoms()
            if atom.GetAtomMapNum()
        }
        logging.debug(f"Reactant atom maps:{reactant_atommaps}")
        for atom_idx, atom_map in reactant_atommaps.items():
            # If atom map exists then continue
            if atom_map in products_atommaps:
                continue
            logging.debug(
                f"Atom {atom_idx} map num ({atom_map}) not found in product!!!"
            )
            atom = reactant.GetAtomWithIdx(atom_idx)
            atom.SetAtomMapNum(0)
        logging.debug(f"Updated reactant: {Chem.MolToSmiles(reactant)}")
    # Reassign Atom Maps for Reagents
    for reagent in rxn.GetAgents():
        reagent_atommaps = {
            atom.GetIdx(): atom.GetAtomMapNum()
            for atom in reagent.GetAtoms()
            if atom.GetAtomMapNum()
        }
        logging.debug(f"Reagent atom maps: {reagent_atommaps}")
        for atom_idx, atom_map in reagent_atommaps.items():
            # If atom map exists then continue
            if atom_map in products_atommaps:
                continue
            logging.debug(
                f"Atom {atom_idx} map num ({atom_map}) not found in product!!!"
            )
            atom = reagent.GetAtomWithIdx(atom_idx)
            atom.SetAtomMapNum(0)
        logging.debug(f"Updated reagent: {Chem.MolToSmiles(reagent)}")
    # Get updated rsmi
    updated_rsmi = (
        AllChem.ReactionToSmiles(rxn) if as_smiles else AllChem.ReactionToSmarts(rxn)
    )
    logging.debug(f"Updated rsmi: {updated_rsmi}")

    return updated_rsmi


def split_smiles_from_reaction(smiles: str) -> List[str]:
    """
    Split a part of reaction SMILES, e.g. reactants or products
    into components. Taking care of intra-molecular complexes

    Taken from RDKit:
    https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/ChemReactions/DaylightParser.cpp

    :param smiles: the SMILES/SMARTS
    :return: the individual components.
    """
    pos = 0
    block_start = 0
    level = 0
    in_block = 0
    components = []
    while pos < len(smiles):
        if smiles[pos] == "(":
            if pos == block_start:
                in_block = 1
            level += 1
        elif smiles[pos] == ")":
            if level == 1 and in_block:
                in_block = 2
            level -= 1
        elif level == 0 and smiles[pos] == ".":
            if in_block == 2:
                components.append(smiles[block_start + 1 : pos - 1])
            else:
                components.append(smiles[block_start:pos])
            block_start = pos + 1
            in_block = 0
        pos += 1
    if block_start < pos:
        if in_block == 2:
            components.append(smiles[block_start + 1 : pos - 1])
        else:
            components.append(smiles[block_start:pos])
    return components
