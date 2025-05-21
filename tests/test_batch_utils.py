import os

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from rxnutils.data.batch_utils import (
    combine_csv_batches,
    combine_numpy_array_batches,
    combine_sparse_matrix_batches,
    create_csv_batches,
    nlines,
    read_csv_batch,
)


@pytest.fixture
def create_dummy_file(tmpdir):
    def wrapper(line_count):
        filename = str(tmpdir / "temp.csv")
        pd.DataFrame({"any": list(range(line_count))}).to_csv(filename, index=False)
        return filename

    return wrapper


@pytest.mark.parametrize(
    ("line_count"),
    [
        (0),
        (1),
        (5),
        (10),
        (100),
    ],
)
def test_nlines(line_count, create_dummy_file):
    filename = create_dummy_file(line_count)

    assert nlines(filename) == line_count + 1


@pytest.mark.parametrize(
    ("nbatches", "expected"),
    [
        (2, [(0, 1, 6), (1, 6, 11)]),
        (3, [(0, 1, 5), (1, 5, 8), (2, 8, 11)]),
        (
            10,
            [
                (0, 1, 2),
                (1, 2, 3),
                (2, 3, 4),
                (3, 4, 5),
                (4, 5, 6),
                (5, 6, 7),
                (6, 7, 8),
                (7, 8, 9),
                (8, 9, 10),
                (9, 10, 11),
            ],
        ),
        (
            15,
            [
                (0, 1, 2),
                (1, 2, 3),
                (2, 3, 4),
                (3, 4, 5),
                (4, 5, 6),
                (5, 6, 7),
                (6, 7, 8),
                (7, 8, 9),
                (8, 9, 10),
                (9, 10, 11),
            ],
        ),
    ],
)
def test_csv_chunks(nbatches, expected, create_dummy_file):
    filename = create_dummy_file(10)

    assert create_csv_batches(filename, nbatches) == expected


@pytest.mark.parametrize(
    ("nbatches", "expected"),
    [
        (2, [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
        (3, [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        (10, [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]),
        (15, [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]),
    ],
)
def test_csv_chunks_end2end(nbatches, expected, create_dummy_file):
    filename = create_dummy_file(10)

    chunks = create_csv_batches(filename, nbatches)
    for batch, this_expected in zip(chunks, expected):
        csv_data = read_csv_batch(filename, batch)
        assert csv_data["any"].to_list() == this_expected


def test_combine_csv_batches(tmpdir):
    filename1 = str(tmpdir / "temp.csv.0")
    pd.DataFrame({"any": [1, 2, 3]}).to_csv(filename1, sep="\t", index=False)
    filename2 = str(tmpdir / "temp.csv.1")
    pd.DataFrame({"any": [4, 5, 6]}).to_csv(filename2, sep="\t", index=False)
    filename = str(tmpdir / "temp.csv")

    assert os.path.exists(filename1)
    assert os.path.exists(filename2)

    combine_csv_batches(filename, 2)

    assert not os.path.exists(filename1)
    assert not os.path.exists(filename2)
    assert pd.read_csv(filename)["any"].to_list() == [1, 2, 3, 4, 5, 6]


def test_combine_sparse_matrix_batches(tmpdir):
    filename1 = str(tmpdir / "temp.0.npz")
    sparse.save_npz(filename1, sparse.csr_array([[0, 0, 0, 1]]), compressed=True)
    filename2 = str(tmpdir / "temp.1.npz")
    sparse.save_npz(filename2, sparse.csr_array([[1, 1, 1, 0]]), compressed=True)
    filename = str(tmpdir / "temp.npz")

    assert os.path.exists(filename1)
    assert os.path.exists(filename2)

    combine_sparse_matrix_batches(filename, 2)

    assert not os.path.exists(filename1)
    assert not os.path.exists(filename2)
    expected = [
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
    ]
    assert sparse.load_npz(filename).toarray().flatten().tolist() == expected


def test_combine_numpy_array_batches(tmpdir):
    filename1 = str(tmpdir / "temp.0.npz")
    np.savez(filename1, np.array([0, 0, 0, 1]), compressed=True)
    filename2 = str(tmpdir / "temp.1.npz")
    np.savez(filename2, np.array([1, 1, 1, 0]), compressed=True)
    filename = str(tmpdir / "temp.npz")

    assert os.path.exists(filename1)
    assert os.path.exists(filename2)

    combine_numpy_array_batches(filename, 2)

    assert not os.path.exists(filename1)
    assert not os.path.exists(filename2)
    expected = [
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
    ]
    assert np.load(filename)["arr_0"].tolist() == expected
