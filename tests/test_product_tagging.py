import pytest

from rxnutils.chem.disconnection_sites.atom_map_tagging import (
    atom_map_tag_products,
    atom_map_tag_reactants,
    get_atom_list,
)
from rxnutils.chem.disconnection_sites.tag_converting import convert_atom_map_tag, tagged_smiles_from_tokens


@pytest.mark.parametrize(
    ("reactants_smiles", "product_smiles", "expected"),
    [
        (
            "[Cl:2].[CH:1]1=[CH:7][CH:6]=[CH:5][CH:4]=[CH:3]1",
            "[Cl:2][C:1]1=[CH:7][CH:6]=[CH:5][CH:4]=[CH:3]1",
            [1, 2],
        ),
        (
            "Cl.[CH:1]1=[CH:7][CH:6]=[CH:5][CH:4]=[CH:3]1",
            "Cl[C:1]1=[CH:7][CH:6]=[CH:5][CH:4]=[CH:3]1",
            [1],
        ),
    ],
)
def test_get_atom_list(reactants_smiles, product_smiles, expected):
    atom_list = get_atom_list(reactants_smiles, product_smiles)
    assert sorted(atom_list) == expected


@pytest.mark.parametrize(
    ("reactants_smiles", "product_smiles", "expected"),
    [
        (
            "[Cl:2].[CH:1]1=[CH:7][CH:6]=[CH:5][CH:4]=[CH:3]1",
            "[Cl:2][C:1]1=[CH:7][CH:6]=[CH:5][CH:4]=[CH:3]1",
            "c1cc[c:1]([Cl:1])cc1",
        ),
        (
            "Cl.[CH:1]1=[CH:7][CH:6]=[CH:5][CH:4]=[CH:3]1",
            "Cl[C:1]1=[CH:7][CH:6]=[CH:5][CH:4]=[CH:3]1",
            "Cl[c:1]1ccccc1",
        ),
    ],
)
def test_atom_map_tag_products(reactants_smiles, product_smiles, expected):
    tagged_product = atom_map_tag_products(f"{reactants_smiles}>>{product_smiles}")
    assert tagged_product == expected


@pytest.mark.parametrize(
    ("reactants_smiles", "product_smiles", "expected"),
    [
        (
            "[Cl:2].[CH:1]1=[CH:7][CH:6]=[CH:5][CH:4]=[CH:3]1",
            "[Cl:2][C:1]1=[CH:7][CH:6]=[CH:5][CH:4]=[CH:3]1",
            "[Cl:1].c1cc[cH:1]cc1",
        ),
        (
            "Cl.[CH:1]1=[CH:7][CH:6]=[CH:5][CH:4]=[CH:3]1",
            "Cl[C:1]1=[CH:7][CH:6]=[CH:5][CH:4]=[CH:3]1",
            "Cl.c1cc[cH:1]cc1",
        ),
    ],
)
def test_atom_map_tag_reactants(reactants_smiles, product_smiles, expected):
    tagged_reactants = atom_map_tag_reactants(f"{reactants_smiles}>>{product_smiles}")
    assert tagged_reactants == expected


@pytest.mark.parametrize(
    ("product_smiles", "expected"),
    [
        ("c1cc[c:1]([Cl:1])cc1", "Cl!c!1ccccc1"),
        ("Cl[c:1]1ccccc1", "Clc!1ccccc1"),
        ("Clc1ccccc1", ""),
    ],
)
def test_tag_converting(product_smiles, expected):
    tagged_product = convert_atom_map_tag(product_smiles)
    assert tagged_product == expected


@pytest.mark.parametrize(
    ("tagged_tokens", "untagged_tokens", "expected"),
    [
        (["[C:1]", "[C@H]", "O"], ["C", "[C@@H]", "O"], ("C![C@@H]O", "C[C@@H]O")),
        (["[C:1]", "/", "C", "O"], ["C", "C", "O"], ("C!CO", "CCO")),
        (["[C:1]", "C", "O"], ["C", "/", "C", "O"], ("C!CO", "CCO")),
        (["[C:1]", "/", "C", "O"], ["C", "\\", "C", "O"], ("C!\\CO", "C\\CO")),
        (["[C:1]", "C", "O"], ["[C]", "C", "O"], ("C!CO", "CCO")),
    ],
)
def test_tagged_smiles_from_tokens(tagged_tokens, untagged_tokens, expected):
    output = tagged_smiles_from_tokens(tagged_tokens, untagged_tokens)
    assert output == expected
