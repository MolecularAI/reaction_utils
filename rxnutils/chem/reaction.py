"""Module containing a class to handle chemical reactions"""
import hashlib
from typing import List, Tuple, Optional, Dict, Any

import wrapt_timeout_decorator
from rdkit import Chem
from rdkit.Chem import AllChem
from rdchiral import template_extractor as extractor

from rxnutils.chem.rinchi import rinchi_api
from rxnutils.chem import utils
from rxnutils.chem.template import ReactionTemplate
from rxnutils.chem.utils import (
    reassign_rsmi_atom_mapping,
    split_smiles_from_reaction,
)


class ReactionException(Exception):
    """Custom exception raised when failing operations on a chemical reaction"""


class ChemicalReaction:
    """
    Representation of chemical reaction

    :param smiles: the reaction SMILES
    :param id_: an optional database ID of the reaction
    :param clean_smiles: if True, will standardize the reaction SMILES
    """

    def __init__(self, smiles: str, id_: str = None, clean_smiles: bool = True) -> None:
        self.rsmi = smiles
        self.clean_rsmi = ""
        self.rid = id_

        self.agents_smiles: str = ""
        self.products_smiles: str = ""
        self.reactants_smiles: str = ""
        if clean_smiles:
            self.rsmi = reassign_rsmi_atom_mapping(self.rsmi, as_smiles=True)
            self._clean_reaction()
        else:
            self._split_reaction()

        self.reactants: List[Chem.rdchem.Mol] = extractor.mols_from_smiles_list(
            self.reactants_list
        )
        self.agents: List[Chem.rdchem.Mol] = extractor.mols_from_smiles_list(
            self.agents_list
        )
        self.products: List[Chem.rdchem.Mol] = extractor.mols_from_smiles_list(
            self.products_list
        )

        self._pseudo_rinchi: Optional[str] = None
        self._pseudo_rinchi_key: Optional[str] = None
        self._rinchi_data: Optional[rinchi_api.RInChIStructure] = None

        self._canonical_template: Optional[ReactionTemplate] = None
        self._retro_template: Optional[ReactionTemplate] = None

    @property
    def agents_list(self) -> List[str]:
        """Gives all the agents as strings"""
        return split_smiles_from_reaction(self.agents_smiles)

    @property
    def canonical_template(self) -> ReactionTemplate:
        """Gives the canonical (forward) template"""
        if self._canonical_template is None:
            raise ValueError(
                f"Call {self.__class__.__name__}.generate_reaction_template() "
                "before using {self.__class__.__name__}.canonical_template"
            )
        return self._canonical_template

    @property
    def products_list(self) -> List[str]:
        """Gives all products as strings"""
        return split_smiles_from_reaction(self.products_smiles)

    @property
    def pseudo_rinchi(self) -> str:
        """Gives pseudo RInChI"""
        if self._pseudo_rinchi is None:
            self._pseudo_rinchi = self._make_pseudo_rinchi()
        return self._pseudo_rinchi

    @property
    def pseudo_rinchi_key(self) -> str:
        """Gives a pseudo reaction InChI key"""
        if self._pseudo_rinchi_key is None:
            self._pseudo_rinchi_key = self._make_pseudo_rinchi_key()
        return self._pseudo_rinchi_key

    @property
    def hashed_rid(self) -> str:
        """Gives a reaction hashkey based on Reaction SMILES & reaction id."""
        return hashlib.sha224(f"{self.rsmi}_{self.rid}".encode("utf8")).hexdigest()

    @property
    def reactants_list(self) -> List[str]:
        """Gives all reactants as strings"""
        return split_smiles_from_reaction(self.reactants_smiles)

    @property
    def retro_template(self) -> ReactionTemplate:
        """Gives the retro template"""
        if self._retro_template is None:
            raise ValueError(
                f"Call {self.__class__.__name__}.generate_reaction_template() "
                "before using {self.__class__.__name__}.retro_template"
            )
        return self._retro_template

    @property
    def rinchi(self) -> str:
        """Gives the reaction InChI"""
        if self._rinchi_data is None:
            self._build_rinchi_info()
        return self._rinchi_data.rinchi

    @property
    def rinchi_key_long(self) -> str:
        """Gives the long reaction InChI key"""
        if self._rinchi_data is None:
            self._build_rinchi_info()
        return self._rinchi_data.long_rinchikey

    @property
    def rinchi_key_short(self) -> str:
        """Gives the short reaction InChI key"""
        if self._rinchi_data is None:
            self._build_rinchi_info()
        return self._rinchi_data.short_rinchikey

    def generate_reaction_template(
        self,
        radius: int = 1,
        expand_ring: bool = False,
        expand_hetero: bool = False,
    ) -> Tuple[ReactionTemplate, ReactionTemplate]:
        """
        Extracts the forward(canonical) and retro reaction template with the specified radius.

        Uses a modified version of:
            https://github.com/connorcoley/ochem_predict_nn/blob/master/data/generate_reaction_templates.py
            https://github.com/connorcoley/rdchiral/blob/master/templates/template_extractor.py

        :param radius: the radius refers to the number of atoms away from the reaction
                       centre to be extracted (the enivronment) i.e. radius = 1 (default)
                       returns the first neighbours around the reaction centre
        :param expand_ring: if True will include all atoms in the same ring as the reaction centre in the template
        :param expand_hetero: if True will extend the template with all bonded hetero atoms
        :returns: the canonical and retrosynthetic templates
        """
        if not self.sanitization_check():
            raise ReactionException(
                "Template generation failed: sanitation check failed"
            )

        reactants = self.reactants
        products = self.products

        if self.no_change():
            raise ReactionException("Template generation failed: no change in reaction")
        if None in reactants + products:
            raise ReactionException(
                "Template generation failed: None in reactants or products"
            )

        # Similar to has_partial mapping
        are_unmapped_product_atoms = self.has_partial_mapping()
        extra_reactant_fragment = ""

        if are_unmapped_product_atoms:  # add fragment to template
            for product in products:
                # Get unmapped atoms
                unmapped_ids = [
                    atom.GetIdx()
                    for atom in product.GetAtoms()
                    if not atom.HasProp("molAtomMapNumber")
                ]
                if len(unmapped_ids) > extractor.MAXIMUM_NUMBER_UNMAPPED_PRODUCT_ATOMS:
                    raise ReactionException(
                        f"Template generation failed: too many unmapped atoms ({len(unmapped_ids)})"
                    )
                # Define new atom symbols for fragment with atom maps, generalizing fully
                atom_symbols = [f"[{atom.GetSymbol()}]" for atom in product.GetAtoms()]
                # And bond symbols...
                bond_symbols = ["~" for _ in product.GetBonds()]
                if unmapped_ids:
                    extra_reactant_fragment += (
                        AllChem.MolFragmentToSmiles(
                            product,
                            unmapped_ids,
                            allHsExplicit=False,
                            isomericSmiles=extractor.USE_STEREOCHEMISTRY,
                            atomSymbols=atom_symbols,
                            bondSymbols=bond_symbols,
                        )
                        + "."
                    )
            if extra_reactant_fragment:
                extra_reactant_fragment = extra_reactant_fragment[:-1]
                if extractor.VERBOSE:
                    print(f"    extra reactant fragment: {extra_reactant_fragment}")

            # Consolidate repeated fragments (stoichometry)
            extra_reactant_fragment = ".".join(
                sorted(list(set(extra_reactant_fragment.split("."))))
            )

        extra_atoms = {}
        if expand_ring:
            ring_atom_maps = self._product_unique_ring_atoms()
            for reactant in self.reactants:
                extra_atoms.update(
                    {
                        str(atom.GetAtomMapNum()): atom
                        for atom in reactant.GetAtoms()
                        if atom.GetAtomMapNum() in ring_atom_maps
                    }
                )

        try:
            # Generate canonical reaction template with rdChiral
            canonical_smarts = self._generate_rdchiral_template(
                reactants=reactants,
                products=products,
                radius=radius,
                extra_atoms=extra_atoms,
                expand_hetero=expand_hetero,
            )
            # Canonical ReactionTemplate
            self._canonical_template = ReactionTemplate(canonical_smarts)

            # Retro ReactionTemplate
            reactants_string = canonical_smarts.split(">>")[0]
            products_string = canonical_smarts.split(">>")[1]
            retro_smarts = products_string + ">>" + reactants_string
            self._retro_template = ReactionTemplate(retro_smarts, "retro")

            # Validate ReactionTemplates
            if not (
                self._retro_template.rdkit_validation()
                and self._canonical_template.rdkit_validation()
            ):
                raise ReactionException(
                    "Template generation failed: RDkit validation of extracted templates failed"
                )
        except Exception as err:
            raise ReactionException(f"Template generation failed with message: {err}")

        return self.canonical_template, self.retro_template

    @staticmethod
    @wrapt_timeout_decorator.timeout(
        30,
        use_signals=False,
        timeout_exception=ReactionException,
        exception_message="Timed out",
    )
    def _generate_rdchiral_template(
        reactants: List[Chem.Mol],
        products: List[Chem.Mol],
        radius: int = 1,
        extra_atoms: Dict[str, Any] = None,
        expand_hetero: bool = False,
    ) -> str:
        """Generate a reaction template with rdChiral.

        :param reactants: List of reactant molecules
        :param products: List of product molecules
        :param radius: Template radius, defaults to 1
        :raises ReactionException: Template generation failed: could not obtained changed atoms
        :raises ReactionException: Template generation failed: no atoms changes
        :raises ReactionException: Template generation failed: Timed out
        :return: Canonical reaction template
        """
        changed_atoms, changed_atom_tags, err = extractor.get_changed_atoms(
            reactants=reactants, products=products
        )

        if expand_hetero:
            for atom in changed_atoms:
                for atom2 in atom.GetNeighbors():
                    if atom2.GetSymbol() not in ["C", "H"] and atom2.GetAtomMapNum():
                        extra_atoms[str(atom2.GetAtomMapNum())] = atom2

        old_tags = list(changed_atom_tags)
        for atom_num, atom in extra_atoms.items():
            if atom_num not in old_tags:
                changed_atoms.append(atom)
                changed_atom_tags.append(atom_num)

        if err:
            raise ReactionException(
                "Template generation failed: could not obtained changed atoms"
            )
        if not changed_atom_tags:
            raise ReactionException("Template generation failed: no atoms changes")

        # Get fragments for reactants
        (reactant_fragments, _, _,) = extractor.get_fragments_for_changed_atoms(
            reactants,
            changed_atom_tags,
            radius=radius,
            expansion=[],
            category="reactants",
        )
        # Get fragments for products %%!
        # (WITHOUT matching groups but WITH the addition of reactant fragments)
        product_fragments, _, _ = extractor.get_fragments_for_changed_atoms(
            products,
            changed_atom_tags,
            radius=radius,
            expansion=extractor.expand_changed_atom_tags(
                changed_atom_tags,
                reactant_fragments,
            ),
            category="products",
        )

        rxn_string = f"{reactant_fragments}>>{product_fragments}"
        canonical_template = extractor.canonicalize_transform(rxn_string)
        # Change from inter-molecular to intra-molecular
        canonical_template_split = canonical_template.split(">>")
        canonical_smarts = (
            canonical_template_split[0][1:-1].replace(").(", ".")
            + ">>"
            + canonical_template_split[1][1:-1].replace(").(", ".")
        )

        return canonical_smarts

    def has_partial_mapping(self) -> bool:
        """Check product atom mapping."""
        for product in self.products:
            sum_with = sum(
                atom.HasProp("molAtomMapNumber") for atom in product.GetAtoms()
            )
            if sum_with < product.GetNumAtoms():
                return True
        return False

    def is_complete(self) -> bool:
        """Check that the product is not among the reactants"""
        return len(self.reactants) > 0 and len(self.products) > 0

    def no_change(self) -> bool:
        """
        Checks to see if the product appears in the reactant set.

        Compares InChIs to rule out possible variations in SMILES notation.

        :return: True the product is present in the reactants set, else False
        """
        reactant_inchi = {Chem.MolToInchi(reactant) for reactant in self.reactants}
        product_inchi = {Chem.MolToInchi(product) for product in self.products}
        return bool(reactant_inchi & product_inchi)

    def is_fuzzy(self) -> bool:
        """Checks to see if there is fuzziness in the reaction.

        :return: True if there is fuzziness, False otherwise
        """
        return bool(self.rsmi.count("*"))

    def sanitization_check(self) -> bool:
        """
        Checks if the reactant and product mol objects can be sanitized in RDKit.

        The actualy sanitization is carried out when the reaction is instansiated,
        this method will only check that all molecules objects were created.

        :return: True if all the molecule objects were successfully created, else False
        """
        return all(
            mol is not None for mol in self.reactants + self.agents + self.products
        )

    def canonical_template_generate_outcome(self) -> bool:
        """Checks whether the canonical template produces"""
        try:
            return len(self.canonical_template.apply(self.reactants_smiles)) > 0
        except Exception:  # pylint: disable=broad-except # noqa
            return False

    def retro_template_generate_outcome(self) -> bool:
        """Checks whether the retrosynthetic template produces an outcome"""
        try:
            return len(self.retro_template.apply(self.products_smiles)) > 0
        except Exception:  # pylint: disable=broad-except # noqa
            return False

    def retro_template_selectivity(self) -> float:
        """
        Checks whether the recorded reactants belong to the set of generated precursors.

        :returns: selectivity, i.e. the fraction of generated precursors matching the recorded precursors
                i.e. 1.0 - match or match.match or match.match.match etc.
                    0.5 - match.none or match.none.match.none etc.
                    0.0 - none
        """
        # pylint: disable=too-many-branches
        try:
            retro_outcomes = self.retro_template.apply(self.products_smiles)
        except Exception:  # pylint: disable=broad-except # noqa
            return 0.0

        if not extractor.USE_STEREOCHEMISTRY:
            reactant_mol_list = []
            for smiles in self.reactants_list:
                if not smiles:
                    continue
                reactant_mol_list.append(Chem.MolFromSmiles(smiles.replace("@", "")))
        else:
            reactant_mol_list = self.reactants

        reactant_inchi = [Chem.MolToInchi(reactant) for reactant in reactant_mol_list]
        precursor_set = []
        for outcome_set in retro_outcomes:
            if not extractor.USE_STEREOCHEMISTRY:
                outcome_set_inchi = [
                    Chem.MolToInchi(Chem.MolFromSmiles(outcome.replace("@", "")))
                    for outcome in outcome_set
                ]
            else:
                outcome_set_inchi = [
                    Chem.MolToInchi(Chem.MolFromSmiles(outcome))
                    for outcome in outcome_set
                ]
            precursor_set.append(outcome_set_inchi)

        assessment = []
        for precursor in precursor_set:
            # There must be a match between the generated outcomes and recorded reactants
            if len(list(set(precursor) & set(reactant_inchi))) != 0:
                assessment.append(2)
            # No match or error
            elif len(list(set(precursor) & set(reactant_inchi))) == 0:
                assessment.append(1)
            else:
                print("Template error")
                assessment.append(0)

        # Quantify the level of selectivity, if an error has occured set to 0
        if assessment.count(0) != 0:
            return 0
        return assessment.count(2) / len(assessment)

    def _build_rinchi_info(self) -> None:
        try:
            rinchi_data = rinchi_api.generate_rinchi(self.rsmi)
        except rinchi_api.RInChIError as err:
            raise ReactionException(f"Could not generate reaction InChI: {err}")
        self._rinchi_data = rinchi_data

    def _clean_reaction(self) -> None:
        """This function updates the reaction smiles by moving mapped molecules from agents to reactants and
        unmapped molecules from reactants to the agents to have reaction smiles in the form of:
            mapped_reactants>unmapped_reagents>products

        Stores:
            rsmi (str): Reaction SMILES in the form mapped_reactants>unmapped_reagents>products
            clean_rsmi (str): Reaction SMILES in the form mapped_reactants>unmapped_reagents>mapped_products
        """

        def _atom_mapping(smiles: str) -> Tuple[int, int]:
            """Return the number of atoms and number of mapped atoms of molecule.

            :param smiles: SMILES of molecule
            :type smiles: string
            :return: Numbers of atoms and mapped atoms of molecule.
            :rtype: Tuple[int, int]
            """
            mol = Chem.MolFromSmiles(smiles)
            num_atoms_ = 0
            num_mapped_atoms_ = 0
            if mol:
                num_mapped_atoms_ = sum(
                    atom.HasProp("molAtomMapNumber") for atom in mol.GetAtoms()
                )
                num_atoms_ = mol.GetNumAtoms()

            return num_atoms_, num_mapped_atoms_

        split_reaction = self.rsmi.split(">")
        self.reactants_smiles = extractor.replace_deuterated(split_reaction[0])
        self.agents_smiles = extractor.replace_deuterated(split_reaction[1])
        self.products_smiles = extractor.replace_deuterated(split_reaction[2])

        # If any agent has mapped atoms then move it to reactants
        updated_agents = []
        add_reactants = []
        for agent in self.agents_list:
            _, num_mapped_atoms = _atom_mapping(agent)
            if num_mapped_atoms:
                add_reactants.append(agent)
            else:
                updated_agents.append(agent)
        self.agents_smiles = self._join_components(updated_agents)

        # If any reactant has 0 mapped atoms then move it to agents
        updated_reactants = []
        add_agents = []
        for reactant in self.reactants_list:
            _, num_mapped_atoms = _atom_mapping(reactant)
            if num_mapped_atoms == 0:
                add_agents.append(reactant)
            else:
                updated_reactants.append(reactant)
        self.reactants_smiles = self._join_components(updated_reactants)

        # Update reactants with molecules from agents
        if add_reactants:
            self.reactants_smiles = ".".join(
                [self.reactants_smiles, self._join_components(add_reactants)]
            )
        # Update agents with molecules from reactants
        if add_agents:
            if self.agents_smiles:
                self.agents_smiles = ".".join(
                    [self.agents_smiles, self._join_components(add_agents)]
                )
            else:
                self.agents_smiles = self._join_components(add_agents)

        # Neutralize/Uncharge Reactants & Products
        # Neutralize reactants
        neutral_reactants = utils.neutralize_molecules(self.reactants_list)
        self.reactants_smiles = self._join_components(neutral_reactants)
        # Neutralize products
        neutral_products = utils.neutralize_molecules(self.products_list)
        self.products_smiles = self._join_components(neutral_products)

        # Reaction SMILES
        self.rsmi = ">".join(
            [self.reactants_smiles, self.agents_smiles, self.products_smiles]
        )

        # Clean Reaction SMILES by RDKit
        reaction = AllChem.ReactionFromSmarts(self.rsmi)
        reaction.Initialize()
        reaction.RemoveUnmappedReactantTemplates(thresholdUnmappedAtoms=0.01)
        reaction.RemoveUnmappedProductTemplates(thresholdUnmappedAtoms=0.01)
        self.clean_rsmi = AllChem.ReactionToSmiles(reaction)

    def _join_components(self, list_: List[str]) -> str:
        """Join SMILES components to one SMILES.

        :param list_: List of SMILES components
        :return: A single SMILES
        """
        return ".".join([f"({item})" if "." in item else item for item in list_])

    def _product_unique_ring_atoms(self) -> List[int]:
        def find_ring_atom_maps(mol):
            ring_atoms_sets = []
            for ring in mol.GetRingInfo().AtomRings():
                ring_atoms_sets.append(
                    tuple(
                        sorted(mol.GetAtomWithIdx(idx).GetAtomMapNum() for idx in ring)
                    )
                )
            return ring_atoms_sets

        reactant_ring_atom_sets = []
        for reactant in self.reactants:
            reactant_ring_atom_sets.extend(find_ring_atom_maps(reactant))

        product_ring_atom_sets = []
        for product in self.products:
            for set_ in find_ring_atom_maps(product):
                if set_ not in reactant_ring_atom_sets:
                    product_ring_atom_sets.append(set_)

        return [idx for ring_atoms in product_ring_atom_sets for idx in ring_atoms]

    def _split_reaction(self) -> None:
        """
        This function is used instead of `_clean_reaction` if cleaning is not required.

        Stores:
            rsmi (str): Reaction SMILES in the form mapped_reactants>unmapped_reagents>products
        """
        split_reaction = self.rsmi.split(">")
        self.reactants_smiles = extractor.replace_deuterated(split_reaction[0])
        self.agents_smiles = extractor.replace_deuterated(split_reaction[1])
        self.products_smiles = extractor.replace_deuterated(split_reaction[2])
        self.rsmi = ">".join(
            [self.reactants_smiles, self.agents_smiles, self.products_smiles]
        )
        self.clean_rsmi = self.rsmi

    def _make_pseudo_rinchi(self) -> str:
        """Generate pseudo Reaction InChI by concatenate the InChIs of reactants and products.

        :return: Pseudo Reaction InChI
        :rtype: str
        """
        reactant_inchi = (
            Chem.MolToInchi(
                Chem.MolFromSmiles(utils.remove_atom_mapping(self.reactants_smiles))
            )
            if self.reactants_smiles
            else ""
        )
        product_inchi = (
            Chem.MolToInchi(
                Chem.MolFromSmiles(utils.remove_atom_mapping(self.products_smiles))
            )
            if self.products_smiles
            else ""
        )
        concatenated_rinchi = "++".join([reactant_inchi, product_inchi])
        return concatenated_rinchi

    def _make_pseudo_rinchi_key(self) -> str:
        """Generate pseudo Reaction InChI-Key by concatenate the InChIs of reactants and products.

        :return: Pseudo Reaction InChI-Key
        :rtype: str
        """
        reactant_inchi_key = Chem.MolToInchiKey(
            Chem.MolFromSmiles(utils.remove_atom_mapping(self.reactants_smiles))
        )
        product_inchi_key = Chem.MolToInchiKey(
            Chem.MolFromSmiles(utils.remove_atom_mapping(self.products_smiles))
        )
        concatenated_rinchi_key = "++".join([reactant_inchi_key, product_inchi_key])
        return concatenated_rinchi_key
