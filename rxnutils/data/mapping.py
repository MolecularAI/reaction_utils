"""Module containing script to atom-map USPTO or ORD reactions
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


def main(input_args: Optional[Sequence[str]] = None) -> None:
    """Function for command-line tool"""

    if not DEPENDENCY_OK:
        raise ImportError(
            "You need to run this tool in an environment where rxnmapper is installed"
        )

    parser = argparse.ArgumentParser("Script to atom-map USPTO or ORD reactions")
    parser.add_argument("--input", default="data.csv", help="the file with reactions")
    parser.add_argument(
        "--output", default="data_mapped.csv", help="the output filename"
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
    args = parser.parse_args(input_args)

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
        fail_data = [
            {
                "mapped_rxn": None,
                "confidence": 0.0,
            }
            for _ in range(len(batch))
        ]
        try:
            batch_mapped = rxn_mapper.get_attention_guided_atom_maps(
                batch, canonicalize_rxns=False
            )
        except ValueError:
            print(batch)
            mapped_data.extend(fail_data)
        except RuntimeError:
            print("\n\n".join(batch))
            mapped_data.extend(fail_data)
        else:
            mapped_data.extend(batch_mapped)
    mapped_data_df = pd.DataFrame(mapped_data)

    data = data.assign(
        **{column: mapped_data_df[column] for column in mapped_data_df.columns}
    )
    data.to_csv(args.output, sep="\t", index=False)


if __name__ == "__main__":
    main()
