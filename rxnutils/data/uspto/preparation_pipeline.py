"""
Module containing pipeline for downloading, transforming and cleaning USPTO data
This needs to be run in an environment with rxnutils installed
"""
from pathlib import Path

from metaflow import step

from rxnutils.data.base_pipeline import DataPreparationBaseFlow
from rxnutils.data.uspto.download import main as download_uspto
from rxnutils.data.uspto.combine import main as combine_uspto


class UsptoDataPreparationFlow(DataPreparationBaseFlow):
    """Pipeline for download UPSTO source file, combining them and do some clean-up"""

    data_prefix = "uspto"

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
        self._setup_cleaning()
        self.next(self.do_cleaning, foreach="partitions")

    @step
    def do_cleaning(self):
        """Perform cleaning of data"""
        self._do_cleaning()
        self.next(self.join_cleaning)

    @step
    def join_cleaning(self, _):
        """Combined cleaned batches of data"""
        self._join_cleaning()
        self.next(self.end)

    @step
    def end(self):
        """Final step, just print information"""
        print(
            f"Processed file is locate here: {Path(self.folder) / 'uspto_data_cleaned.csv'}"
        )


if __name__ == "__main__":
    UsptoDataPreparationFlow()
