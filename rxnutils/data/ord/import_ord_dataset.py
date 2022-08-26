"""
Module containing script to import ORD dataset to a CSV file
"""
import re
import argparse
import os
from collections import defaultdict
from typing import Optional, Sequence

import pandas as pd
from tqdm import tqdm

try:
    from ord_schema import message_helpers
except ImportError:
    DEPENDENCY_OK = False
else:
    from ord_schema.proto import dataset_pb2

    DEPENDENCY_OK = True


def main(input_args: Optional[Sequence[str]] = None) -> None:
    """Function for command-line tool"""

    if not DEPENDENCY_OK:
        raise ImportError(
            "You need to install the ord-schema package and download the ord-data repository"
        )

    parser = argparse.ArgumentParser("Script to import ORD datasets")
    parser.add_argument("--filenames", nargs="+", help="the files to import")
    parser.add_argument("--output", default="ord_data.csv", help="the output filename")
    parser.add_argument("--include-only", help="name or ID of dataset to extract")
    args = parser.parse_args(input_args)

    if not args.filenames:
        print("No filenames given. Quiting early.")
        return

    include_only = args.include_only
    data = defaultdict(list)
    for filename in tqdm(args.filenames):
        ord_dataset = message_helpers.load_message(filename, dataset_pb2.Dataset)
        if include_only and not (
            ord_dataset.dataset_id == include_only or ord_dataset.name == include_only
        ):
            continue
        for reaction in tqdm(ord_dataset.reactions, desc=os.path.basename(filename)):
            smiles = message_helpers.get_reaction_smiles(
                reaction, generate_if_missing=True
            )
            # Trying to remove spurious spaces in reaction SMILES
            smiles = smiles.strip()
            smiles = re.sub(r"\.\s+", ".", smiles)
            smiles = re.sub(r"\s+\.", ".", smiles)
            # Replace dative bonds
            smiles = smiles.replace("->", "-").replace("<-", "-")
            if smiles.count(">") != 2:
                continue
            data["ID"].append(reaction.reaction_id)
            data["Dataset"].append(ord_dataset.name)
            data["Date"].append(reaction.provenance.experiment_start.value)
            data["ReactionSmiles"].append(smiles)
            try:
                data["Yield"].append(
                    message_helpers.get_product_yield(reaction.outcomes[0].products[0])
                )
            except IndexError:
                data["Yield"].append(None)

    pd.DataFrame(data).to_csv(args.output, sep="\t", index=False)


if __name__ == "__main__":
    main()
