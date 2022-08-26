"""Module containing an API to the Reaction InChI program"""
import logging
import os
import sys
import subprocess
import tempfile
from collections import namedtuple

from rdkit.Chem import AllChem

from rxnutils.chem.rinchi import download_rinchi
from rxnutils.chem.rinchi.download_rinchi import RInChIError, PLATFORM2FOLDER


RInChIStructure = namedtuple(
    "RInChI", "rinchi rauxinfo long_rinchikey short_rinchikey web_rinchikey"
)


def generate_rinchi(reaction_smiles: str) -> RInChIStructure:
    """Generate RInChI from Reaction SMILES.

    :param reaction_smiles: Reaction SMILES
    :raises RInChIError: When there is an error with RInChI generation.
    :return: Namedtuple with the generated RInChI.
    """
    if sys.platform not in PLATFORM2FOLDER:
        raise RInChIError("RInChI software not supported on this platform")

    reaction = AllChem.ReactionFromSmarts(reaction_smiles)
    reaction.Initialize()
    nwarn, nerror, nreactants, nproducts, labels = AllChem.PreprocessReaction(reaction)
    logging.debug(f"Number of warnings: {nwarn}")
    logging.debug(f"Number of preprocessing errors: {nerror}")
    logging.debug(f"Number of reactants in reaction: {nreactants}")
    logging.debug(f"Number of products in reaction: {nproducts}")
    logging.debug(f"Preprocess labels added: {labels}")

    rxn_block = AllChem.ReactionToRxnBlock(reaction)

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as fileobj:
        tmp_file = fileobj.name
        fileobj.write(rxn_block)
    logging.debug(f"Temp RXN file: {tmp_file}")

    try:
        rinchi_data = _rinchi_cli(rxn_file=tmp_file)
        logging.debug(rinchi_data)
    except RInChIError:  # pylint: disable=try-except-raise
        raise
    finally:
        os.unlink(tmp_file)

    return rinchi_data


def _rinchi_cli(rxn_file: str, options: str = None) -> RInChIStructure:
    """Run the commandline to generate RInChI.

    :param rxn_file: Path for RXN file
    :param options: Additional options from rinchi_cmdline, defaults to None
    :raises RInChIError: When there is an error with RInChI generation.
    :return: Namedtuple with the generated RInChI.
    """
    rinchi_path = download_rinchi.main()
    cmd = [os.path.join(rinchi_path, "rinchi_cmdline"), rxn_file]
    if options:
        cmd += [options]

    # Note: rinchi_cmdline doesn't like fuzzyness...
    # I.e.: Neighbouring bonds to double bond, expected to be defined...
    # pylint: disable=subprocess-run-check
    process = subprocess.run(
        cmd,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if process.returncode:
        raise RInChIError(process.stderr)
    lines = process.stdout.strip().split("\n")
    rinchi_data = RInChIStructure(
        rinchi=lines[0],
        rauxinfo=lines[1],
        long_rinchikey=lines[2],
        short_rinchikey=lines[3],
        web_rinchikey=lines[4],
    )

    return rinchi_data


if __name__ == "__main__":
    import sys

    print(generate_rinchi(sys.argv[1]))
