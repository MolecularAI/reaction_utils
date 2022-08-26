import os
import sys

import pytest

from rxnutils.chem.rinchi.download_rinchi import (
    main as download_rinchi,
    PLATFORM2FOLDER,
)
from rxnutils.chem.rinchi.rinchi_api import generate_rinchi


@pytest.fixture(scope="session")
def rinchi_download():
    download_rinchi()


@pytest.mark.xfail(sys.platform not in PLATFORM2FOLDER, reason="Platform not supported")
def test_download_rinchi(mocker, tmpdir):
    config = {
        "download_folder": str(tmpdir),
        "download_url": "http://www.inchi-trust.org/download/RInChI/RInChI-V1-00.zip",
    }
    mocker.patch("rxnutils.chem.rinchi.download_rinchi.CONFIG", config)

    download_rinchi()

    assert os.path.exists(tmpdir / "RInChI-V1-00.zip")


@pytest.mark.xfail(sys.platform not in PLATFORM2FOLDER, reason="Platform not supported")
def test_generate_rinchi(rinchi_download):
    rsmi = (
        "[ClH;D0;+0:1]>>"
        "[Cl;H0;D1;+0:1]-[c;H0;D3;+0:2]1:[n;H0;D2;+0:3]:[cH;D2;+0:4]:[cH;D2;+0:5]:"
        "[c;H0;D3;+0:6]2:[cH;D2;+0:7]:[cH;D2;+0:8]:[cH;D2;+0:9]:[cH;D2;+0:10]:[c;H0;D3;+0:11]:1:2"
    )

    resp = generate_rinchi(rsmi)

    assert (
        resp.long_rinchikey
        == "Long-RInChIKey=SA-BUHFF-MSQCQINLJMEVNJ-UHFFFAOYSA-N--VEXZGXHMUGYJMC-UHFFFAOYSA-N"
    )
    assert (
        resp.short_rinchikey
        == "Short-RInChIKey=SA-BUHFF-MSQCQINLJM-VEXZGXHMUG-UHFFFADPSC-NUHFF-NUHFF-NUHFF-ZZZ"
    )
