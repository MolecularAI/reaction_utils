"""
Module containing pipeline for downloading, transforming and cleaning USPTO data
This needs to be run in an environment with rxnutils installed
"""
from pathlib import Path

from metaflow import step

from rxnutils.data.uspto.base_pipeline import UsptoBaseFlow
from rxnutils.data.uspto.download import main as download_uspto
from rxnutils.data.uspto.combine import main as combine_uspto
from rxnutils.pipeline.runner import main as validation_runner


class UsptoDataPreparationFlow(UsptoBaseFlow):
    """Pipeline for download UPSTO source file, combining them and do some clean-up"""

    @step
    def start(self):
        """Download USPTO data from Figshare"""
        download_uspto(
            [
                "--folder",
                self.folder,
            ]
        )
        self.next(self.combine_files)

    @step
    def combine_files(self):
        """Combine USPTO data files and add IDs"""
        combine_uspto(["--folder", self.folder])
        self.next(self.setup_cleaning)

    @step
    def setup_cleaning(self):
        """Setup cleaning"""
        # pylint: disable=attribute-defined-outside-init
        self.partitions = self._create_batches(
            Path(self.folder) / "uspto_data.csv",
            Path(self.folder) / "uspto_data_cleaned.csv",
        )
        self.next(self.do_cleaning, foreach="partitions")

    @step
    def do_cleaning(self):
        """Perform cleaning of USPTO data"""
        idx, start, end = self.input
        pipeline_path = str(Path(__file__).parent / "clean_pipeline.yml")
        if idx > -1:
            validation_runner(
                [
                    "--pipeline",
                    pipeline_path,
                    "--data",
                    str(Path(self.folder) / "uspto_data.csv"),
                    "--output",
                    str(Path(self.folder) / f"uspto_data_cleaned.csv.{idx}"),
                    "--batch",
                    str(start),
                    str(end),
                    "--max-workers",
                    "1",
                    "--no-intermediates",
                ]
            )
        self.next(self.join_cleaning)

    @step
    def join_cleaning(self, _):
        """Combined cleaned batches of data"""
        self._combine_batches(Path(self.folder) / "uspto_data_cleaned.csv")
        self.next(self.end)

    @step
    def end(self):
        """Final step, just print information"""
        print(
            f"Processed file is locate here: {Path(self.folder) / 'uspto_data_cleaned.csv'}"
        )


if __name__ == "__main__":
    UsptoDataPreparationFlow()
