"""Module containing base class for data pipelines
"""
import os
import math
from pathlib import Path

import pandas as pd
from metaflow import FlowSpec, Parameter

# This is hack to only import the validation_runner if rxnmapper is not installed
try:
    import rxnmapper  # noqa
except ImportError:
    from rxnutils.pipeline.runner import main as validation_runner


def nlines(filename: str) -> int:
    """Count and return the number of lines in a file"""
    with open(filename, "rb") as fileobj:
        return sum(1 for line in fileobj)


class DataBaseFlow(FlowSpec):
    """Base-class for pipelines for processing data"""

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


class DataPreparationBaseFlow(DataBaseFlow):
    """Base pipeline for preparing datasets and doing clean-up"""

    data_prefix = ""

    def _setup_cleaning(self):
        """Setup cleaning"""
        # pylint: disable=attribute-defined-outside-init
        self.partitions = self._create_batches(
            Path(self.folder) / f"{self.data_prefix}_data.csv",
            Path(self.folder) / f"{self.data_prefix}_data_cleaned.csv",
        )

    def _do_cleaning(self):
        """Perform cleaning of data"""
        idx, start, end = self.input
        pipeline_path = str(Path(__file__).parent / "clean_pipeline.yml")
        if idx > -1:
            validation_runner(
                [
                    "--pipeline",
                    pipeline_path,
                    "--data",
                    str(Path(self.folder) / f"{self.data_prefix}_data.csv"),
                    "--output",
                    str(
                        Path(self.folder) / f"{self.data_prefix}_data_cleaned.csv.{idx}"
                    ),
                    "--batch",
                    str(start),
                    str(end),
                    "--max-workers",
                    "1",
                    "--no-intermediates",
                ]
            )

    def _join_cleaning(self):
        """Combined cleaned batches of data"""
        self._combine_batches(
            Path(self.folder) / f"{self.data_prefix}_data_cleaned.csv"
        )
