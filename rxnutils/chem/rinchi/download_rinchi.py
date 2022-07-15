"""Module for downloading InChI Trust Reaction InChI."""
import os
import sys
import stat
from zipfile import ZipFile
import logging

import requests

CONFIG = {
    "download_folder": ".",
    "download_url": "http://www.inchi-trust.org/download/RInChI/RInChI-V1-00.zip",
}
PATH = os.path.dirname(__file__)


def main() -> str:
    """Check if Reaction InchI application is present.
    Download it if it's required to do so.

    :return: Path of the folder containing the appropriate
             command line executable based on system type.
    """
    rinchi_url = CONFIG.get("download_url")
    rinchi_fn = rinchi_url.split("/")[-1]
    download_loc = CONFIG.get("download_folder")
    download_loc = os.path.join(PATH, download_loc)
    rinchi_fn = os.path.join(download_loc, rinchi_fn)
    if not os.path.exists(rinchi_fn):
        if not os.path.exists(download_loc):
            logging.info(f"Creating: {download_loc}")
            os.makedirs(download_loc)
        logging.info(f"Downloading: {rinchi_url}")
        req = requests.get(rinchi_url)
        logging.debug(f"{req.status_code}")
        logging.debug(f"{req.headers}")
        req.raise_for_status()
        logging.info(f"Creating: {rinchi_fn}")
        with open(rinchi_fn, "wb") as fileobj:
            fileobj.write(req.content)
        logging.info("Download completed...")
        logging.info(f"Unziping: {rinchi_fn}")
        with ZipFile(rinchi_fn, "r") as fileobj:
            bin_path = [
                x
                for x in fileobj.namelist()
                if x.endswith(f"bin/rinchi_cmdline/{sys.platform}/x86_64/")
            ]
            logging.debug(bin_path)
            fileobj.extractall(download_loc)
        logging.info("Completed...")
        rinchi_cli_path = os.path.join(download_loc, bin_path[0])
        logging.info(f"RInChI CLI: {rinchi_cli_path}")
        os.chmod(os.path.join(rinchi_cli_path, "rinchi_cmdline"), stat.S_IEXEC)
    else:
        logging.info(f"RInChI exists at: {rinchi_fn}")
        bin_path = [
            r
            for r, d, f in os.walk(download_loc)
            if r.endswith(f"bin/rinchi_cmdline/{sys.platform}/x86_64")
        ]
        logging.debug(bin_path)
        rinchi_cli_path = bin_path[0]
        logging.info(f"RInChI CLI: {rinchi_cli_path}")
        os.chmod(os.path.join(rinchi_cli_path, "rinchi_cmdline"), stat.S_IEXEC)

    return rinchi_cli_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
