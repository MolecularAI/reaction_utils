"""Module containging base class for USPTO pipelines
"""
import os
import math
from pathlib import Path

import pandas as pd
from metaflow import FlowSpec, Parameter


def nlines(filename: str) -> int:
    """Count and return the number of lines in a file"""
    with open(filename, "rb") as fileobj:
        return sum(1 for line in fileobj)


class UsptoBaseFlow(FlowSpec):
    """Base-class for pipelines for processing USPTO data"""

    nbatches = Parameter("nbatches", type=int, required=True)
    folder = Parameter("folder", default=".")

    def _combine_batches(self, filename):
        if Path(filename).exists():
            return

        data = None
        for idx in range(self.nbatches):
            filename2 = f"{filename}.{idx}"
            data_temp = pd.read_csv(filename2, sep="\t")
            if data is None:
                data = data_temp
            else:
                data = pd.concat([data, data_temp])
            os.remove(filename2)
        data.to_csv(filename, index=False, sep="\t")

    def _create_batches(self, input_filename, output_filename):
        if Path(output_filename).exists():
            return [(-1, None, None)]

        file_size = (
            nlines(input_filename) - 1
        )  # Header should not be counted for chunk size calculations
        chunk_size = math.ceil(file_size / self.nbatches)
        partition_limits = [
            (
                idx,
                idx * chunk_size + 1,
                (idx + 1) * chunk_size + 1,
            )  # +1 to account for header
            for idx in range(self.nbatches)
        ]
        return partition_limits
