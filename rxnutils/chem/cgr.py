""" Wrapper class for the CGRTools library
"""

import io
import warnings
from typing import List

from CGRtools.containers.molecule import MoleculeContainer
from CGRtools.containers.reaction import ReactionContainer
from CGRtools.files.SDFrw import SDFRead
from rdkit import Chem

from rxnutils.chem.reaction import ChemicalReaction
from rxnutils.chem.utils import atom_mapping_numbers


class CondensedGraphReaction:
    """
    The Condensed Graph of Reaction (CGR) representation of a reaction

    :ivar reaction_container: the CGRTools container of the reaction
    :ivar cgr_container: the CGRTools container of the CGR
    :param reaction: the reaction composed of RDKit molecule to start from
    :raises ValueError: if it is not possible to create the CGR from the reaction
    """

    def __init__(self, reaction: ChemicalReaction) -> None:
        self.reaction = reaction
        self._cgr_reactants = []
        self._cgr_products = []
        self._make_cgr_containers()
        self.reaction_container = ReactionContainer(reactants=self._cgr_reactants, products=self._cgr_products)
        try:
            self.cgr_container = self.reaction_container.compose()
        except ValueError as err:
            if str(err) == "mapping of graphs is not disjoint":
                raise ValueError("Reaction contains inconsistent atom-mapping, perhaps duplicates")
            elif str(err).endswith("} not equal"):
                raise ValueError("Atom with the same atom-mapping in reactant and product is not equal")
            else:
                raise ValueError(f"Unknown problem with generating CGR: {err}")

    def __eq__(self, obj: object) -> bool:
        if not isinstance(obj, CondensedGraphReaction):
            return False
        return self.cgr_container == obj.cgr_container

    @property
    def bonds_broken(self) -> int:
        """Returns the number of broken bonds in the reaction"""
        return sum(bond.p_order is None for _, _, bond in self.cgr_container.bonds())

    @property
    def bonds_changed(self) -> int:
        """Returns the number of broken or formed bonds in the reaction"""
        return sum(bond.p_order is None or bond.order is None for _, _, bond in self.cgr_container.bonds())

    @property
    def bonds_formed(self) -> int:
        """Returns the number of formed bonds in the reaction"""
        return sum(bond.order is None for _, _, bond in self.cgr_container.bonds())

    @property
    def total_centers(self) -> int:
        """Returns the number of atom and bond centers in the reaction"""
        return len(self.cgr_container.center_atoms) + len(self.cgr_container.center_bonds)

    def distance_to(self, other: "CondensedGraphReaction") -> int:
        """
        Returns the chemical distance between two reactions, i.e. the absolute difference
        between the total number of centers.

        Used for some atom-mapping comparison statistics

        :param other: the reaction to compare to
        :returns: the computed distance
        """
        return abs(self.total_centers - other.total_centers)

    def _make_cgr_containers(self):
        nreactants = len(self.reaction.reactants)
        renumbered_mols = self._make_renumbered_mols()
        self._cgr_reactants = _convert_rdkit_molecules(renumbered_mols[:nreactants])
        self._cgr_products = _convert_rdkit_molecules(renumbered_mols[nreactants:])

        if len(self._cgr_reactants) != nreactants:
            warnings.warn("Warning not all reactants could be converted to CGRtools")
        if len(self._cgr_products) != len(self.reaction.products):
            warnings.warn("Warning not all products could be converted to CGRtools")

    def _make_renumbered_mols(self):
        # It is necessary that all atoms have atom-mapping
        # otherwise CGRTools will start adding bad atom-mappings
        # so this adds safe atom-mapping to un-mapped atoms
        renumbered_mols = []
        max_atom_map_numb = max(
            max(atom_mapping_numbers(smi) or [0]) for smi in self.reaction.reactants_list + self.reaction.products_list
        )
        for mol0 in self.reaction.reactants + self.reaction.products:
            if mol0 is None:
                raise ValueError("Cannot create CGR for this reaction, some molecules are None")
            mol = Chem.rdchem.Mol(mol0)
            for atom in mol.GetAtoms():
                if not atom.GetAtomMapNum():
                    max_atom_map_numb += 1
                    atom.SetAtomMapNum(max_atom_map_numb)
            renumbered_mols.append(mol)
        return renumbered_mols


def _convert_rdkit_molecules(
    mols: List[Chem.rdchem.Mol],
) -> List[MoleculeContainer]:
    # This connects the SDWriter of RDKit with the SDFRead of CGRTools
    with io.StringIO() as sd_buffer:
        with Chem.SDWriter(sd_buffer) as sd_writer_obj:
            for mol in mols:
                sd_writer_obj.write(mol)
        sd_buffer.seek(0)
        with SDFRead(sd_buffer, remap=False) as sd_reader_obj:
            cgr_molecules = sd_reader_obj.read()

    # This make the aromatization of the molecules, which were
    # lost when writing to SD format. This is essential to keep
    # the reactant and product structures as equal as possible
    for mol in cgr_molecules:
        mol.thiele()
    return cgr_molecules
