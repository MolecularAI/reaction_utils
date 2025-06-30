import os
import json

import pytest

from rxnutils.chem.smartslib import SmartsLibrary
from rxnutils.chem.reaction import ChemicalReaction


def test_load_smartslib(shared_datadir):
    filename = str(shared_datadir / "simple_smartslib.txt")

    lib = SmartsLibrary(filename, "default")

    assert len(lib.general_smarts_objects) == 6
    assert lib.name_aliases["AcidAliphatic"] == ["AcidAliphatic", "Y", "X"]
    assert len(lib.smarts_objects["AcidAliphatic"].GetAtoms()) == 3
    assert len(lib.smarts_objects["Ammonia"].GetAtoms()) == 1
    assert len(lib.general_smarts_objects["AcidAliphatic"].GetAtoms()) == 0
    assert len(lib.general_smarts_objects["Ammonia"].GetAtoms()) == 1
    assert lib.type_["AcidAliphatic"] == "clip"


def test_load_smartslib_from_env(shared_datadir):
    filename = str(shared_datadir / "simple_smartslib.txt")
    os.environ["SMARTS_LIB"] = filename

    lib = SmartsLibrary("${SMARTS_LIB}", "default")

    assert len(lib.general_smarts_objects) == 6


def test_load_smartslib_from_json(tmpdir):
    filename = str(tmpdir / "smarts.json")
    with open(filename, "w") as fileobj:
        json.dump({"ARC": "[c]", "ARN3": "[n;X3]"}, fileobj)

    lib = SmartsLibrary(filename, "default")

    assert len(lib.smarts_objects) == 2
    assert len(lib.general_smarts_objects) == 0
    assert len(lib.name_aliases) == 0
    assert len(lib.type_) == 0


def test_load_smartslib_from_load(shared_datadir):
    filename = str(shared_datadir / "simple_smartslib.txt")

    lib = SmartsLibrary.load(filename)

    assert len(lib.general_smarts_objects) == 6

    lib2 = SmartsLibrary(filename)

    assert lib is not lib2


def test_reload_smartslib(shared_datadir):
    filename = str(shared_datadir / "simple_smartslib.txt")
    lib0 = SmartsLibrary(filename, "default")

    assert SmartsLibrary.load("default") is lib0


def test_match_smarts(shared_datadir):
    filename = str(shared_datadir / "simple_smartslib.txt")
    lib = SmartsLibrary(filename, "default")

    hits = lib.match_smarts("CC(C(O)=O)C(O)=O")

    assert len(hits) == 1
    assert hits["AcidAliphatic"].number_of_hits == 2
    assert hits["AcidAliphatic"].match == ((3, 2, 1), (6, 5, 1))
    assert hits["AcidAliphatic"].atoms == [3, 2, 1, 6, 5, 1]


@pytest.mark.parametrize(
    ("rsmi", "kwargs", "expected"),
    [
        (
            "N.CC(=O)O>>CC(=O)N",
            {"sort": True},
            (("AcidAliphatic", "Ammonia"), "CC(=O)O.N>>CC(=O)N"),
        ),
        (
            "N.CC(=O)O>>CC(=O)N",
            {},
            (("Ammonia", "AcidAliphatic"), "N.CC(=O)O>>CC(=O)N"),
        ),
        (
            "CC(=O)O>>CC(=O)N",
            {"sort": True},
            (("AcidAliphatic", "None"), "CC(=O)O>>CC(=O)N"),
        ),
        (
            "CC(=O)O>>CC(=O)N",
            {"sort": True, "add_none": False},
            (("AcidAliphatic",), "CC(=O)O>>CC(=O)N"),
        ),
        (
            "CC(=O)O>>CC(=O)N",
            {"sort": True, "none_str": "???"},
            (("AcidAliphatic", "???"), "CC(=O)O>>CC(=O)N"),
        ),
        (
            "CC(=O)O>>CC(=O)N",
            {"sort": True, "target_size": None},
            (("AcidAliphatic",), "CC(=O)O>>CC(=O)N"),
        ),
        (
            "CC(=O)O>>CC(=O)N",
            {"sort": True, "target_size": 3},
            (("AcidAliphatic", "None", "None"), "CC(=O)O>>CC(=O)N"),
        ),
    ],
)
def test_detect_reactive_function(rsmi, kwargs, expected, shared_datadir):
    filename = str(shared_datadir / "simple_smartslib.txt")
    lib = SmartsLibrary(filename, "default")

    reaction = ChemicalReaction(rsmi, clean_smiles=False)
    assert lib.detect_reactive_functions(reaction, **kwargs) == expected


def test_detect_reactive_function_no_product(shared_datadir):
    filename = str(shared_datadir / "simple_smartslib.txt")
    lib = SmartsLibrary(filename, "default")

    reaction = ChemicalReaction("C.N>>", clean_smiles=False)
    with pytest.raises(ValueError, match="Cannot detect"):
        lib.detect_reactive_functions(reaction)


def test_detect_reactive_function_too_many_reactants(shared_datadir):
    filename = str(shared_datadir / "simple_smartslib.txt")
    lib = SmartsLibrary(filename, "default")

    reaction = ChemicalReaction("C.N>>CN", clean_smiles=False)
    with pytest.raises(ValueError, match="Too many reactants"):
        lib.detect_reactive_functions(reaction, max_reactants=1)
