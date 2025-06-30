import json

import numpy as np
from rdkit import Chem

from rxnutils.chem.features.reaction_centre_fp import ReactionCentreFingerprint, reaction_center_similarity
from rxnutils.chem.features.simple_rdkit import SimpleRdkitFingerprint, fingerprint_ecfp, fingerprint_mixed
from rxnutils.chem.reaction import ChemicalReaction


def test_ecfp():
    mol = Chem.MolFromSmiles("O")

    fp = fingerprint_ecfp(mol, 10)

    assert len(fp) == 10
    assert sum(fp) == 1


def test_mixed_fp():
    mol = Chem.MolFromSmiles("O")

    fp_ecfp = fingerprint_ecfp(mol, 5)

    fp = fingerprint_mixed(mol, 10)

    assert len(fp) == 10
    assert sum(fp) == 1
    assert fp[:5].tolist() == fp_ecfp.tolist()


def test_rdkit_fp():
    rxn = ChemicalReaction("O.N.C>>OCN", clean_smiles=False)

    fp1 = SimpleRdkitFingerprint("fingerprint_ecfp", 10)(rxn)

    assert len(fp1) == 10
    assert fp1 == [-2, 2, 1, 0, -1, 1, 1, 0, 0, 1]

    fp2 = SimpleRdkitFingerprint("fingerprint_ecfp", 10, product_bits=10)(rxn)
    mol = Chem.MolFromSmiles("OCN")
    fp_ecfp = fingerprint_ecfp(mol, 10)

    assert len(fp2) == 20
    assert fp2[10:] == fp_ecfp.tolist()
    assert fp2[:10] == fp1


def test_center_fingerprint(shared_datadir):
    with open(shared_datadir / "precomputed_fps.json", "r") as fileobj:
        expected = np.asarray(json.load(fileobj)["reaction_centre"])

    rxn = ChemicalReaction(
        "[c:2]1([OH:1])[cH:3][cH:4][cH:5][cH:6][cH:7]1>>[Cl:1][c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1",
        clean_smiles=False,
    )

    fp1 = np.asarray(ReactionCentreFingerprint()(rxn))
    fp2 = np.asarray(ReactionCentreFingerprint(max_centers=2)(rxn))

    assert len(fp1) == 1
    assert (fp1[0] - expected).sum() == 0
    assert len(fp2) == (2049)
    assert fp2[0] == 1
    assert (fp2[1:1025] - fp1[0]).sum() == 0
    assert all(bit == 0 for bit in fp2[1025:])


def test_center_similarity_non_comparable():
    fps1 = [[0, 1, 1, 0, 1], [1, 1, 1, 0, 1]]
    fps2 = [[0, 1, 1, 0, 1]]

    assert reaction_center_similarity(fps1, fps2) == 0.0
    assert reaction_center_similarity(fps2, fps1) == 0.0


def test_center_similarity_comparable():
    fps1 = [[0, 1, 1, 0, 1], [1, 1, 1, 0, 1]]
    fps2 = [[1, 0, 0, 1, 0], [1, 1, 1, 0, 1]]
    fps3 = [[1, 1, 1, 0, 1], [1, 0, 0, 1, 0]]

    assert reaction_center_similarity(fps1, fps2) == 0.50
    assert reaction_center_similarity(fps1, fps3) == 0.50
    assert reaction_center_similarity(fps2, fps3) == 1.0
