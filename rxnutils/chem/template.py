"""Module containing useful representations of templates
"""
import re
import hashlib
import logging
from collections import defaultdict
from itertools import permutations
from typing import List, Dict, Set, Iterator, Tuple, Any

import numpy as np
import rdchiral.main as rdc
from xxhash import xxh32
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, SanitizeFlags  # pylint: disable=all


DELIM_REGEX_STR = r"[&:\]]"
AROMATIC_REGEX_STR = r"&a" + DELIM_REGEX_STR
AROMATIC_REGEX = re.compile(AROMATIC_REGEX_STR)
CHARGE_REGEX_STR = r"&([+-]\d?)" + DELIM_REGEX_STR
CHARGE_REGEX = re.compile(CHARGE_REGEX_STR)
HYDROGEN_REGEX_STR = r"&H(\d)" + DELIM_REGEX_STR
HYDROGEN_REGEX = re.compile(HYDROGEN_REGEX_STR)
DEGREE_REGEX_STR = r"&D(\d)" + DELIM_REGEX_STR
DEGREE_REGEX = re.compile(DEGREE_REGEX_STR)


class TemplateMolecule:
    """
    Representation of a molecule created from a SMARTS string

    :param rd_mol: the RDKit molecule to be represented by this class
    """

    def __init__(self, rd_mol: Chem.rdchem.Mol = None, smarts: str = None) -> None:
        if rd_mol is None and smarts is None:
            raise ValueError("Both rd_mol and smarts argument is None")

        if rd_mol is None:
            self._rd_mol = Chem.MolFromSmarts(smarts)
            self._smarts = smarts
        else:
            self._rd_mol = rd_mol
            self._smarts = Chem.MolToSmarts(self._rd_mol)

    def atoms(self) -> Iterator[Chem.rdchem.Atom]:
        """
        Generate the atom object of this molecule

        :yield: the next atom object
        """
        for atom in self._rd_mol.GetAtoms():
            yield atom

    def atom_invariants(self) -> List[int]:
        """
        Calculate invariants on similar properties as in RDKit but ignore mass and add aromaticity

        :return: a list of the atom invariants
        """
        self.fix_atom_properties()
        invariants = []
        for atom in self.atoms():
            arr = np.asarray(
                [
                    atom.GetAtomicNum(),
                    atom.GetIntProp("comp_degree"),
                    atom.GetTotalNumHs(),
                    atom.GetFormalCharge(),
                    int(atom.IsInRing()),
                    int(atom.GetIsAromatic()),
                ],
                dtype=np.uint8,
            )
            invariants.append(xxh32(arr.tobytes()).intdigest())
        return invariants

    def atom_properties(self) -> Dict[str, List[object]]:
        """
        Return a dictionary with atomic properties

        Example:
            import pandas
            pandas.DataFrame(my_mol.atom_properties())
        """
        props: Dict[str, List[Any]] = defaultdict(list)
        for idx, atom in enumerate(self.atoms()):
            props["index"].append(idx)
            for key in dir(atom):
                if key.startswith("Get") and "Prop" not in key:
                    ret = getattr(atom, key)()
                    if isinstance(ret, (list, tuple)):
                        props[f"# {key[3:]}"].append(len(ret))
                    else:
                        props[key[3:]].append(ret)
            try:
                props["comp degree"].append(atom.GetIntProp("comp_degree"))
            except KeyError:  # Raised if fix_atom_properties hasn't been called
                pass
        return dict(props)

    def fingerprint_bits(self, radius: int = 2, use_chirality: bool = True) -> Set[int]:
        """
        Calculate the unique fingerprint bits

        Will sanitize molecule if necessary

        :param radius: the radius of the Morgan calculation
        :param use_chirality: determines if chirality should be taken into account
        :return: the set of unique bits
        """

        def calc_fp():
            AllChem.GetMorganFingerprint(
                self._rd_mol,
                radius,
                bitInfo=bits,
                invariants=invariants,
                useChirality=use_chirality,
            )

        invariants = self.atom_invariants()
        bits: Dict[int, List[Tuple[int, int]]] = {}
        try:
            calc_fp()
        except RuntimeError:
            self.sanitize()
            calc_fp()
        return set(bits.keys())

    def fingerprint_vector(
        self, radius: int = 2, nbits: int = 1024, use_chirality: bool = True
    ) -> np.ndarray:
        """
        Calculate the finger bit vector

        Will sanitize molecule if necessary

        :param radius: the radius of the Morgan calculation
        :param nbits: the length of the bit vector
        :param use_chirality: determines if chirality should be taken into account
        :return: the bit vector
        """

        def calc_fp():
            return AllChem.GetMorganFingerprintAsBitVect(
                self._rd_mol,
                radius,
                nbits,
                invariants=invariants,
                useChirality=use_chirality,
            )

        invariants = self.atom_invariants()
        try:
            bitvect = calc_fp()
        except RuntimeError:
            self.sanitize()
            bitvect = calc_fp()
        array = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(bitvect, array)
        return array

    def fix_atom_properties(self) -> None:
        """
        Copy over some properties from the SMARTS specification to the atom object
        1. Set IsAromatic flag is lower-case a is in the SMARTS
        2. Fix formal charges
        3. Explicit number of hydrogen atoms

        Also extract explicit degree from SMARTS and is stored in
        the `comp_degree` property.
        """
        for atom in self.atoms():
            atom.SetNoImplicit(False)
            aromatic_match = AROMATIC_REGEX.search(atom.GetSmarts())
            if aromatic_match:
                atom.SetIsAromatic(True)

            charge_match = CHARGE_REGEX.search(atom.GetSmarts())
            if charge_match:
                amount = "1" if len(charge_match.group(1)) == 1 else ""
                charge = int(charge_match.group(1) + amount)
                atom.SetFormalCharge(charge)

            hydrogen_match = HYDROGEN_REGEX.search(atom.GetSmarts())
            if hydrogen_match:
                atom.SetNoImplicit(True)
                nhydrogens = int(hydrogen_match.group(1))
                atom.SetNumExplicitHs(nhydrogens)

            degree_match = DEGREE_REGEX.search(atom.GetSmarts())
            if degree_match:
                degrees = int(degree_match.group(1))
                atom.SetIntProp("comp_degree", degrees)
            else:
                atom.SetIntProp("comp_degree", atom.GetDegree())
            atom.UpdatePropertyCache()

    def hash_from_smiles(self) -> str:
        """
        Create a hash of the template based on a cleaned-up template SMILES string

        :return: the hash string
        """
        other_template = TemplateMolecule(smarts=self._smarts)
        other_template.remove_atom_mapping()
        # pylint: disable=protected-access
        smiles = Chem.MolToSmiles(other_template._rd_mol)
        return hashlib.sha224(smiles.encode("utf8")).hexdigest()

    def hash_from_smarts(self) -> str:
        """
        Create a hash of the template based on a cleaned-up template SMARTS string

        :return: the hash string
        """
        other_template = TemplateMolecule(smarts=self._smarts)
        other_template.remove_atom_mapping()
        # pylint: disable=protected-access
        smarts = Chem.MolToSmarts(other_template._rd_mol)
        return hashlib.sha224(smarts.encode("utf8")).hexdigest()

    def remove_atom_mapping(self) -> None:
        """Remove the atom mappings from the molecule"""
        for atom in self.atoms():
            atom.SetAtomMapNum(0)

    def sanitize(self) -> None:
        """
        Will do selective sanitation - skip some procedures that causes problems due to "hanging" aromatic atoms

        All possible flags:
            SANITIZE_ADJUSTHS
            SANITIZE_ALL
            SANITIZE_CLEANUP
            SANITIZE_CLEANUPCHIRALITY
            SANITIZE_FINDRADICALS
            SANITIZE_KEKULIZE
            SANITIZE_NONE
            SANITIZE_PROPERTIES
            SANITIZE_SETAROMATICITY
            SANITIZE_SETCONJUGATION
            SANITIZE_SETHYBRIDIZATION
            SANITIZE_SYMMRINGS
        """
        AllChem.SanitizeMol(
            self._rd_mol,
            sanitizeOps=SanitizeFlags.SANITIZE_ADJUSTHS
            | SanitizeFlags.SANITIZE_CLEANUP
            | SanitizeFlags.SANITIZE_CLEANUPCHIRALITY
            | SanitizeFlags.SANITIZE_PROPERTIES
            | SanitizeFlags.SANITIZE_SETAROMATICITY
            | SanitizeFlags.SANITIZE_SETCONJUGATION
            | SanitizeFlags.SANITIZE_SETHYBRIDIZATION
            | SanitizeFlags.SANITIZE_SYMMRINGS,
        )


class ReactionTemplate:
    """
    Representation of a reaction template created with RDChiral

    :param smarts: the SMARTS string representation of the reaction
    :param direction: if equal to "retro" reverse the meaning of products and reactants
    """

    def __init__(self, smarts: str, direction: str = "canonical") -> None:
        self.smarts = smarts
        self.direction = direction
        self._rd_reaction = AllChem.ReactionFromSmarts(smarts)
        self._reactants = [
            TemplateMolecule(self._rd_reaction.GetReactantTemplate(i))
            for i in range(self._rd_reaction.GetNumReactantTemplates())
        ]
        self._products = [
            TemplateMolecule(self._rd_reaction.GetProductTemplate(i))
            for i in range(self._rd_reaction.GetNumProductTemplates())
        ]

        if direction == "retro":
            temp = self._products
            self._products = self._reactants
            self._reactants = temp

    def apply(self, mols: str) -> Tuple[Tuple[str, ...], ...]:
        """
        Applies the template on the given molecule

        :param mols: the molecule as a SMILES
        :return: the list of reactants
        """
        if self.direction == "retro":
            outcome = rdc.rdchiralRunText(self.smarts, mols)
            return tuple(tuple(str_.split(".")) for str_ in outcome)

        mols_objs = [Chem.MolFromSmiles(mol) for mol in mols.split(".")]
        # Get all permutations of molecules
        outcome = []
        num_reactant_templates = self._rd_reaction.GetNumReactantTemplates()
        logging.debug(
            f"#Reactants: {len(mols_objs)} Vs. #Reactant Templates: {num_reactant_templates}"
        )
        reactants_permutations = permutations(mols_objs, num_reactant_templates)
        for idx, reactants in enumerate(reactants_permutations):
            outcome = self._rd_reaction.RunReactants(reactants)
            # Break as soon as one outcome has been produced
            # or 100 (0-99) permutations have been tried...
            if outcome or idx == 99:
                break

        def create_smiles(mol_list):
            smiles_list: List[str] = []
            for mol in mol_list:
                try:
                    Chem.SanitizeMol(mol)
                except Exception as error:  # pylint: disable=broad-except
                    logging.error(f"{error}")
                else:
                    smiles_list.append(Chem.MolToSmiles(mol))

            return tuple(smiles_list)

        return tuple(create_smiles(list_) for list_ in outcome)

    def fingerprint_bits(
        self, radius: int = 2, use_chirality: bool = True
    ) -> Dict[int, int]:
        """
        Calculate the difference count of the fingerprint bits set of the reactants and products

        :param radius: the radius of the Morgan calculation
        :param use_chirality: determines if chirality should be taken into account
        :return: a dictionary of the difference count for each bit
        """
        bit_sum = defaultdict(int)
        for mol in self._products:
            for bit in mol.fingerprint_bits(radius, use_chirality):
                bit_sum[bit] = 1
        for mol in self._reactants:
            for bit in mol.fingerprint_bits(radius, use_chirality):
                bit_sum[bit] -= 1
        return bit_sum

    def fingerprint_vector(
        self, radius: int = 2, nbits: int = 1024, use_chirality: bool = True
    ) -> np.ndarray:
        """
        Calculate the difference fingerprint vector

        :param radius: the radius of the Morgan calculation
        :param nbits: the length of the bit vector
        :param use_chirality: determines if chirality should be taken into account
        :return: the bit vector
        """
        fp_ = np.zeros(nbits)
        for mol in self._products:
            fp_ += mol.fingerprint_vector(radius, nbits, use_chirality)
        for mol in self._reactants:
            fp_ -= mol.fingerprint_vector(radius, nbits, use_chirality)
        return fp_

    def hash_from_bits(self, radius: int = 2, use_chirality: bool = True) -> str:
        """
        Create a hash of the template based on the difference counts of the fingerprint bits

        :param radius: the radius of the Morgan calculation
        :param use_chirality: determines if chirality should be taken into account
        :return: the hash string
        """
        bit_sum = self.fingerprint_bits(radius, use_chirality)
        hash_obj = hashlib.sha256()
        sorted_bits = sorted(bit_sum.keys())
        for bit in sorted_bits:
            hash_obj.update(f"{bit}:{bit_sum[bit]}".encode())
        return hash_obj.hexdigest()

    def hash_from_smiles(self) -> str:
        """
        Create a hash of the template based on a cleaned-up template SMILES string

        :return: the hash string
        """
        reaction = self._cleaned_reaction_copy()
        AllChem.RemoveMappingNumbersFromReactions(reaction)
        rxn_smiles = AllChem.ReactionToSmiles(reaction)
        return hashlib.sha224(rxn_smiles.encode()).hexdigest()

    def hash_from_smarts(self) -> str:
        """
        Create a hash of the template based on a cleaned-up template SMARTS string

        :return: the hash string
        """
        reaction = self._cleaned_reaction_copy()
        AllChem.RemoveMappingNumbersFromReactions(reaction)
        rxn_smarts = AllChem.ReactionToSmarts(reaction)
        return hashlib.sha224(rxn_smarts.encode()).hexdigest()

    def _cleaned_reaction_copy(self) -> AllChem.ChemicalReaction:
        """Clean RDKit chemical reaction.

        :return: RDKit reaction
        :rtype: AllChem.ChemicalReaction
        """
        reaction = AllChem.ReactionFromSmarts(self.smarts)
        mols = [
            TemplateMolecule(reaction.GetProductTemplate(i))
            for i in range(reaction.GetNumProductTemplates())
        ]
        mols.extend(
            [
                TemplateMolecule(reaction.GetReactantTemplate(i))
                for i in range(reaction.GetNumReactantTemplates())
            ]
        )

        for mol in mols:
            mol.fix_atom_properties()

        return reaction

    def rdkit_validation(self) -> bool:
        """Checks if the template is valid in RDKit"""
        rxn = AllChem.ReactionFromSmarts(self.smarts)
        rxn.Initialize()
        return rxn.Validate()[1] == 0
