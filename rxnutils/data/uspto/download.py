"""Module containing a script to download USPTO files Figshare
"""
import os
import argparse
from pathlib import Path
from typing import Optional, Sequence

import tqdm
import requests
import py7zr


FILES_TO_DOWNLOAD = [
    {
        "filename": "2001_Sep2016_USPTOapplications_smiles.rsmi.7z",
        "url": "https://ndownloader.figshare.com/files/8664370",
    },
    {
        "filename": "1976_Sep2016_USPTOgrants_smiles.rsmi.7z",
        "url": "https://ndownloader.figshare.com/files/8664379",
    },
]


def _download_file(url: str, filename: str) -> None:
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        pbar = tqdm.tqdm(
            total=total_size, desc=os.path.basename(filename), unit="B", unit_scale=True
        )
        with open(filename, "wb") as fileobj:
            for chunk in response.iter_content(chunk_size=1024):
                fileobj.write(chunk)
                pbar.update(len(chunk))
        pbar.close()


def main(args: Optional[Sequence[str]] = None) -> None:
    """Function for command-line tool"""
    parser = argparse.ArgumentParser(
        "Script to download USPTO SMILES data from Figshare"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="if given will overwrite existing files",
    )
    parser.add_argument("--folder", default=".", help="folder for downloading")
    args = parser.parse_args(args)

    for file_spec in FILES_TO_DOWNLOAD:
        filename = Path(args.folder) / file_spec["filename"]
        if not args.overwrite and filename.exists():
            continue
        _download_file(file_spec["url"], filename)

        archive = py7zr.SevenZipFile(filename, mode="r")
        archive.extractall(path=args.folder)
        archive.close()


if __name__ == "__main__":
    main()
