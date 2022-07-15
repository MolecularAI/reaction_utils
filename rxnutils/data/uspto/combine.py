"""
Module containing script to combine raw USPTO files

It will:
    * preserve the ReactionSmiles and Year columns
    * create an ID from PatentNumber and ParagraphNum and row index in the original file
"""
import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

DEFAULT_FILENAMES = [
    "1976_Sep2016_USPTOgrants_smiles.rsmi",
    "2001_Sep2016_USPTOapplications_smiles.rsmi",
]


def main(args: Optional[Sequence[str]] = None) -> None:
    """Function for command-line tool"""
    parser = argparse.ArgumentParser(
        "Script to combine USPTO SMILES data from Figshare"
    )
    parser.add_argument(
        "--filenames", nargs="+", default=DEFAULT_FILENAMES, help="the files to combine"
    )
    parser.add_argument(
        "--output", default="uspto_data.csv", help="the output filename"
    )
    parser.add_argument("--folder", default=".", help="folder with downloaded files")
    args = parser.parse_args(args)

    filenames = [Path(args.folder) / filename for filename in args.filenames]
    data = pd.concat(
        [
            pd.read_csv(filename, sep="\t", dtype={"ParagraphNum": "str"})
            for filename in filenames
        ]
    )

    para_num = data["ParagraphNum"].fillna("")
    row_num = data.index.astype(str)
    data["ID"] = data["PatentNumber"] + ";" + para_num + ";" + row_num
    data2 = data[["ID", "Year", "ReactionSmiles"]]

    print(f"Total number of unique IDs: {len(set(data2['ID']))}")
    print(f"Total number of records: {len(data2)}")

    data2.to_csv(Path(args.folder) / args.output, sep="\t", index=False)


if __name__ == "__main__":
    main()
