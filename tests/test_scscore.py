import pytest
from rdkit import Chem

from rxnutils.chem.features.sc_score import SCORE_SUPPORTED, SCScore


@pytest.mark.xfail(condition=not SCORE_SUPPORTED, reason="onnx support not installed")
def test_scscore(shared_datadir):
    filename = str(shared_datadir / "scscore_dummy_model.onnx")
    scorer = SCScore(filename, 5)
    mol = Chem.MolFromSmiles("C")

    assert pytest.approx(scorer(mol), abs=1e-3) == 4.523
