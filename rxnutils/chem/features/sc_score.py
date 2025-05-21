""" Module containing the implementation of the SC-score model for synthetic complexity scoring. """

from __future__ import annotations

import numpy as np
from rdkit.Chem import AllChem

RdMol = AllChem.rdchem.Mol

try:
    import onnxruntime
except ImportError:
    SCORE_SUPPORTED = False
else:
    SCORE_SUPPORTED = True


class SCScore:
    """
    Encapsulation of the SCScore model

    Re-write of the SCScorer from the scscorer package using ONNX

    The predictions of the score is made with a sanitized instance of an RDKit molecule

    .. code-block::

        mol = Chem.MolFromSmiles(CCC)
        scscorer = SCScorer("path_to_model")
        score = scscorer(mol)

    :param model_path: the filename of the model weights and biases
    :param fingerprint_length: the number of bits in the fingerprint
    :param fingerprint_radius: the radius of the fingerprint
    """

    def __init__(
        self,
        model_path: str,
        fingerprint_length: int = 1024,
        fingerprint_radius: int = 2,
    ) -> None:
        if not SCORE_SUPPORTED:
            raise ImportError("Cannot use SCScore model. Install rxnutils with optional dependencies")
        self._fingerprint_length = fingerprint_length
        self._fingerprint_radius = fingerprint_radius
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        self._model = onnxruntime.InferenceSession(model_path, sess_options=session_options)

    def __call__(self, rd_mol: RdMol) -> float:
        fingerprint = self._make_fingerprint(rd_mol)
        return self._model.run(None, {"fingerprint": fingerprint})[0][0]

    def _make_fingerprint(self, rd_mol: RdMol) -> np.ndarray:
        """Returns the molecule's Morgan fingerprint"""
        fp_vec = AllChem.GetMorganFingerprintAsBitVect(
            rd_mol,
            self._fingerprint_radius,
            nBits=self._fingerprint_length,
            useChirality=True,
        )
        return np.array(fp_vec, dtype=float)
