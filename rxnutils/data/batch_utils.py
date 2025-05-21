import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


def nlines(filename: str) -> int:
    """Count and return the number of lines in a file"""
    with open(filename, "rb") as fileobj:
        return sum(1 for line in fileobj)


def combine_batches(
    filename: str,
    nbatches: int,
    read_func: Any,
    write_func: Any,
    combine_func: Any,
) -> None:
    if Path(filename).exists():
        return

    data = None
    for idx in range(nbatches):
        temp_data, filename2 = read_func(filename, idx)
        if data is None:
            data = temp_data
        else:
            data = combine_func(data, temp_data)
        os.remove(filename2)
    write_func(data, filename)


def combine_csv_batches(filename: str, nbatches: int) -> None:
    """
    Combine CSV batches to one master file

    The batch files are removed from disc

    :param filename: the filename of the master file
    :param nbatches: the number of batches
    """

    def _combine_csv(data: pd.DataFrame, temp_data: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([data, temp_data])

    def _read_csv(filename: str, idx: int) -> pd.DataFrame:
        filename2 = f"{filename}.{idx}"
        return pd.read_csv(filename2, sep="\t"), filename2

    def _write_csv(data: pd.DataFrame, filename: str) -> None:
        data.to_csv(filename, index=False, sep="\t")

    combine_batches(filename, nbatches, _read_csv, _write_csv, _combine_csv)


def combine_numpy_array_batches(filename: str, nbatches: int) -> None:
    """
    Combine numpy array batches to one master file
    The batch files are removed from disc
    :param filename: the filename of the master file
    :param nbatches: the number of batches
    """

    def _read_array(filename: str, idx: int) -> Any:
        filename2 = filename.replace(".npz", f".{idx}.npz")
        return np.load(filename2)["arr_0"], filename2

    def _write_array(data: np.ndarray, filename: str) -> None:
        np.savez(filename, data, compressed=True)

    def _combine_array(data: np.ndarray, temp_data: np.ndarray) -> np.ndarray:
        return np.hstack([data, temp_data])

    return combine_batches(filename, nbatches, _read_array, _write_array, _combine_array)


def combine_sparse_matrix_batches(filename: str, nbatches: int) -> None:
    """
    Combine sparse matrix batches to one master file

    The batch files are removed from disc

    :param filename: the filename of the master file
    :param nbatches: the number of batches
    """

    def _read_matrix(filename: str, idx: int) -> Any:
        filename2 = filename.replace(".npz", f".{idx}.npz")
        return sparse.load_npz(filename2), filename2

    def _write_matrix(data: Any, filename: str) -> None:
        sparse.save_npz(filename, data, compressed=True)

    def _combine_matrix(data: pd.DataFrame, temp_data: pd.DataFrame) -> pd.DataFrame:
        return sparse.vstack([data, temp_data])

    combine_batches(filename, nbatches, _read_matrix, _write_matrix, _combine_matrix)


def create_csv_batches(
    filename: str, nbatches: int, output_filename: Optional[str] = None
) -> List[Tuple[int, int, int]]:
    """
    Create batches for reading a splitted CSV-file

    The batches will be in  the form of a tuple with three indices:
        * Batch index
        * Start index
        * End index

    :param filename: the CSV file to make batches of
    :param nbatches: the number of batches
    :param output_filename:
    :return: the created batches
    """
    if output_filename and Path(output_filename).exists():
        return [(-1, None, None)]

    file_size = nlines(filename) - 1  # Header should not be counted for batch size calculations
    nbatches = min(file_size, nbatches)  # Adjust the number of batches to the size of the file
    batch_size, remainder = divmod(file_size, nbatches)
    stop = 1  # 1-indexed to account for header in the .csv file
    batches = []
    for partition_idx in range(1, nbatches + 1):
        start = stop
        stop += batch_size + 1 if partition_idx <= remainder else batch_size
        batches.append((partition_idx - 1, start, stop))
    return batches


def read_csv_batch(filename: str, batch: Tuple[int, ...] = None, **kwargs: Any) -> pd.DataFrame:
    """
    Read parts of a CSV file as specified by a batch

    :param filename: the path to the CSV file on disc
    :param batch: the batch specification as returned by `create_csv_batches`
    """
    if batch is None:
        return pd.read_csv(filename, **kwargs)
    if len(batch) == 3:
        _, batch_start, batch_end = batch
    elif len(batch) == 2:
        batch_start, batch_end = batch
    else:
        raise ValueError(f"The batch specification can only be 2 or 3 not {len(batch)}")
    return pd.read_csv(
        filename,
        nrows=batch_end - batch_start,
        skiprows=range(1, batch_start),
        **kwargs,
    )
