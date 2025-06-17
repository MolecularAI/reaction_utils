from rxnutils.chem.protection.amino_acids import (
    AminoAcidProtectionEngine,
    preprocess_amino_acids,
    remove_backbone_charges,
)

import pytest


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        (
            "Cc1nc(O)c([C@H](N)C(=O)O)c(=O)[nH]1",
            "Cc1nc(O)c([C@H](N)C(=O)O)c(=O)[nH]1",
        ),
        (
            "Cc1[nH]c(=O)c(c(n1)[O-])[C@@H](C(=O)[O-])[NH3+]",
            "Cc1nc(O)c([C@H](N)C(=O)O)c(=O)[nH]1",
        ),
    ],
)
def test_preprocess_smiles(smiles, expected):
    assert preprocess_amino_acids(smiles) == expected


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        (
            "Cc1[nH]c(=O)c(c(n1)O)[C@@H](C(=O)O)[NH3+]",
            "Cc1[nH]c(=O)c(c(n1)O)[C@@H](C(=O)O)[NH3+]",
        ),
        (
            "Cc1[nH]c(=O)c(c(n1)[O-])[C@@H](C(=O)[O-])[NH3+]",
            "Cc1[nH]c(=O)c(c(n1)O)[C@@H](C(=O)O)[NH3+]",
        ),
    ],
)
def test_remove_backbone_charges(smiles, expected):
    assert remove_backbone_charges(smiles) == expected


@pytest.mark.parametrize(
    ("smiles", "expected_smiles", "expected_groups"),
    [
        (
            "Cc1[nH]c(=O)c(c(n1)O)[C@@H](C(=O)O)[NH3+]",
            "C=CCOc1nc(C)n(-c2ccc([N+](=O)[O-])cc2[N+](=O)[O-])c(=O)c1[C@H]([NH3+])C(=O)OC(C)(C)C",
            ("tBu", "allyl", "DNP"),
        ),
        (
            "COc1c2c(ncn1)N[C@@H](C(=O)N2)[C@@H](C(=O)O)[NH3+]",
            "COc1ncnc2c1N(c1ccc([N+](=O)[O-])cc1[N+](=O)[O-])C(=O)[C@@H]([C@H]([NH3+])C(=O)OC(C)(C)C)N2c1ccc([N+](=O)[O-])cc1[N+](=O)[O-]",
            ("tBu", "DNP"),
        ),
        (
            "c1csc(c1SCC(=O)O)[C@@H](C(=O)O)[NH3+]",
            "CC(C)(C)OC(=O)[C@@H]([NH3+])c1sccc1SCC(=O)OCC=C",
            ("allyl", "tBu"),
        ),
    ],
)
def test_protection_engine(smiles, expected_smiles, expected_groups, shared_datadir):
    engine = AminoAcidProtectionEngine(
        smartslib_path=str(shared_datadir / "simple_smartslib_nnaa.txt"),
        reaction_rules_path=str(shared_datadir / "simple_protection_reactions.csv"),
        protection_groups_path=str(shared_datadir / "simple_protection_groups.csv"),
    )

    results = engine(smiles)

    assert len(results) == 1

    assert results[0].protection_groups == expected_groups
    assert results[0].smiles == expected_smiles
