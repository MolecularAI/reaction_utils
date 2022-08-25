"""Module for downloading InChI Trust Reaction InChI."""
import os
import sys
import stat
from zipfile import ZipFile
import logging

import requests

PLATFORM2FOLDER = {
    "linux": "linux",
    "win32": "windows",
}

PLATFORM2EXTENSION = {"linux": "", "win32": ".exe"}

CONFIG = {
    "download_folder": ".",
    "download_url": "http://www.inchi-trust.org/download/RInChI/RInChI-V1-00.zip",
}
PATH = os.path.dirname(__file__)


class RInChIError(Exception):
    """Exception raised by RInChI API"""


def main() -> str:
    """Check if Reaction InchI application is present.
    Download it if it's required to do so.

    :return: Path of the folder containing the appropriate
             command line executable based on system type.
    """
    if sys.platform not in PLATFORM2FOLDER:
        raise RInChIError("RInChI software not supported on this platform")

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
                if x.endswith(_exec_folder_ending(os_sep=False) + "/")
            ]
            logging.debug(bin_path)
            fileobj.extractall(download_loc)
        logging.info("Completed...")
        rinchi_cli_path = os.path.join(download_loc, bin_path[0])
        logging.info(f"RInChI CLI: {rinchi_cli_path}")
        if sys.platform == "linux":
            os.chmod(os.path.join(rinchi_cli_path, "rinchi_cmdline"), stat.S_IEXEC)
    else:
        logging.info(f"RInChI exists at: {rinchi_fn}")
        bin_path = [
            r
            for r, d, f in os.walk(download_loc)
            if r.endswith(_exec_folder_ending(os_sep=True))
        ]
        logging.debug(bin_path)
        rinchi_cli_path = bin_path[0]
        logging.info(f"RInChI CLI: {rinchi_cli_path}")
        if sys.platform == "linux":
            os.chmod(
                os.path.join(
                    rinchi_cli_path, f"rinchi_cmdline{PLATFORM2EXTENSION[sys.platform]}"
                ),
                stat.S_IEXEC,
            )

    return rinchi_cli_path


def _exec_folder_ending(os_sep: bool) -> str:
    sep = os.sep if os_sep else "/"
    return sep.join(
        ["bin", "rinchi_cmdline", f"{PLATFORM2FOLDER[sys.platform]}", "x86_64"]
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
