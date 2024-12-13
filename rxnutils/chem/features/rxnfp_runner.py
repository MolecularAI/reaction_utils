"""Module containing script to calculate RXNFP for a set of reactions
"""

import argparse
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from rxnutils.data.batch_utils import read_csv_batch

try:
    from rxnfp.transformer_fingerprints import (  # pylint: disable=all
        RXNBERTFingerprintGenerator,
        get_default_model_and_tokenizer,
    )
except ImportError:
    DEPENDENCY_OK = False
else:
    DEPENDENCY_OK = True


def main(input_args: Optional[Sequence[str]] = None) -> None:
    """Function for command-line tool"""

    if not DEPENDENCY_OK:
        raise ImportError("You need to run this tool in an environment where rxnfp is installed")

    parser = argparse.ArgumentParser("Script to calculate RXNFP")
    parser.add_argument("--input", default="data.csv", help="the file with reactions")
    parser.add_argument("--output", default="data_rxnfp.npz", help="the output filename")
    parser.add_argument(
        "--column",
        default="reaction_smiles",
        help="The column with the reaction SMILES",
    )
    parser.add_argument(
        "--batch",
        type=int,
        nargs=2,
        help="Line numbers to start and stop reading rows",
    )
    parser.add_argument(
        "--fp_batch_size",
        type=int,
        default=10,
        help="how many SMILES to calculate at the same time",
    )
    args = parser.parse_args(input_args)

    data = read_csv_batch(args.input, sep="\t", index_col=False, batch=args.batch)

    model, tokenizer = get_default_model_and_tokenizer(force_no_cuda=True)
    rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer, force_no_cuda=True)

    fingerprints = []
    for start in range(0, len(data), args.fp_batch_size):
        batch = data[args.column].iloc[start : start + args.fp_batch_size].tolist()
        # Adding zeros for all failed SMILES
        fail_data = [[0.0] * 256 for _ in range(args.fp_batch_size)]
        try:
            fps_batch = rxnfp_generator.convert_batch(batch)
        except ValueError as err:
            print(err)
            print("\n\n".join(batch))
            fingerprints.extend(fail_data)
        except RuntimeError as err:
            print(err)
            print("\n\n".join(batch))
            fingerprints.extend(fail_data)
        else:
            fingerprints.extend(fps_batch)

    fingerprints = np.asarray(fingerprints)
    np.savez(args.output, rxnfps=fingerprints)


if __name__ == "__main__":
    main()
