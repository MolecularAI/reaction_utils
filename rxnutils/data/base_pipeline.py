"""Module containing base class for data pipelines
"""

from pathlib import Path
from typing import List, Tuple

from metaflow import FlowSpec, Parameter

from rxnutils.data.batch_utils import combine_csv_batches, create_csv_batches

# This is hack to only import the validation_runner if rxnmapper is not installed
try:
    import rxnmapper  # noqa
except ImportError:
    from rxnutils.pipeline.runner import main as validation_runner


class DataBaseFlow(FlowSpec):
    """Base-class for pipelines for processing data"""

    nbatches = Parameter("nbatches", type=int, required=True)
    folder = Parameter("folder", default=".")

    def _combine_batches(self, filename: str) -> None:
        combine_csv_batches(filename, self.nbatches)

    def _create_batches(self, input_filename: str, output_filename: str) -> List[Tuple[int, int, int]]:
        return create_csv_batches(
            input_filename,
            self.nbatches,
            output_filename,
        )


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
                    str(Path(self.folder) / f"{self.data_prefix}_data_cleaned.csv.{idx}"),
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
        self._combine_batches(Path(self.folder) / f"{self.data_prefix}_data_cleaned.csv")
