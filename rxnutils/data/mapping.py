"""Module containing script to atom-map USPTO or ORD reactions
"""

import argparse
from typing import Optional, Sequence

import pandas as pd

from rxnutils.data.batch_utils import read_csv_batch

try:
    from rxnmapper import BatchedMapper  # pylint: disable=all
except ImportError:
    DEPENDENCY_OK = False
else:
    DEPENDENCY_OK = True
    from transformers import logging

    logging.set_verbosity_error()


def main(input_args: Optional[Sequence[str]] = None) -> None:
    """Function for command-line tool"""

    if not DEPENDENCY_OK:
        raise ImportError("You need to run this tool in an environment where rxnmapper is installed")

    parser = argparse.ArgumentParser("Script to atom-map USPTO or ORD reactions")
    parser.add_argument("--input", default="data.csv", help="the file with reactions")
    parser.add_argument("--output", default="data_mapped.csv", help="the output filename")
    parser.add_argument(
        "--batch",
        type=int,
        nargs=2,
        help="Line numbers to start and stop reading rows",
    )
    parser.add_argument(
        "--column",
        default="ReactionSmilesClean",
        help="The column with the reaction SMILES",
    )
    parser.add_argument(
        "--mapper_batch_size",
        type=int,
        default=5,
        help="how many SMILES to mapped at the same time",
    )
    args = parser.parse_args(input_args)

    data = read_csv_batch(args.input, sep="\t", index_col=False, batch=args.batch)

    rxn_mapper = BatchedMapper(batch_size=args.mapper_batch_size)
    mapped_data = []
    # Get RXNMapper results in batches with error handling (returning and empty dict for failed reaction)
    for start in range(0, len(data), args.mapper_batch_size):
        batch = data[args.column].iloc[start : start + args.mapper_batch_size].tolist()
        batch_mapped = list(rxn_mapper.map_reactions_with_info(batch))
        # Check RXNMapper results and if there is a failure (empty dict) then use the original Reaction SMILES
        for idx, (original_rsmi, mapped_rsmi) in enumerate(zip(batch, batch_mapped)):
            # In case of failure return the original Reaction SMILES
            if not mapped_rsmi:
                batch_mapped[idx] = {
                    "mapped_rxn": original_rsmi,
                    "confidence": 0.0,
                }
        mapped_data.extend(batch_mapped)
    mapped_data_df = pd.DataFrame(mapped_data)

    data = data.assign(**{column: mapped_data_df[column] for column in mapped_data_df.columns})
    data.to_csv(args.output, sep="\t", index=False)


if __name__ == "__main__":
    main()
