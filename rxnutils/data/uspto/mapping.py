"""Module containing script to atom-map USPTO reactions
"""
import argparse
from typing import Optional, Sequence

import pandas as pd

try:
    from rxnmapper import RXNMapper  # pylint: disable=all
except ImportError:
    DEPENDENCY_OK = False
else:
    DEPENDENCY_OK = True


def main(args: Optional[Sequence[str]] = None) -> None:
    """Function for command-line tool"""

    if not DEPENDENCY_OK:
        raise ImportError(
            "You need to run this tool in an environment where rxnmapper is installed"
        )

    parser = argparse.ArgumentParser("Script to atom-map USPTO reactions")
    parser.add_argument(
        "--input", default="uspto_data.csv", help="the file with USPTO reactions"
    )
    parser.add_argument(
        "--output", default="uspto_data_mapped.csv", help="the output filename"
    )
    parser.add_argument(
        "--batch",
        type=int,
        nargs=2,
        help="Line numbers to start and stop reading rows",
    )
    parser.add_argument(
        "--mapper_batch_size",
        type=int,
        default=5,
        help="how many SMILES to mapped at the same time",
    )
    args = parser.parse_args(args)

    if args.batch:
        start, end = args.batch
        data = pd.read_csv(
            args.input,
            sep="\t",
            nrows=end - start,
            skiprows=range(1, start),
        )
    else:
        data = pd.read_csv(args.input, sep="\t")

    rxn_mapper = RXNMapper()
    mapped_data = []
    for start in range(0, len(data), args.mapper_batch_size):
        batch = (
            data["ReactionSmilesClean"]
            .iloc[start : start + args.mapper_batch_size]
            .tolist()
        )
        try:
            batch_mapped = rxn_mapper.get_attention_guided_atom_maps(
                batch, canonicalize_rxns=False
            )
        except ValueError:
            print(batch)
            raise
        except RuntimeError:
            print("\n\n".join(batch))
            raise
        mapped_data.extend(batch_mapped)
    mapped_data = pd.DataFrame(mapped_data)

    data = data.assign(
        **{column: mapped_data[column] for column in mapped_data.columns}
    )
    data.to_csv(args.output, sep="\t", index=False)


if __name__ == "__main__":
    main()
