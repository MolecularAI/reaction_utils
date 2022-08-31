"""
Module containing pipeline for extracting, transforming and cleaning Open reaction database data
This needs to be run in an environment with rxnutils installed
"""
import glob
import os
from pathlib import Path

from metaflow import step, Parameter

from rxnutils.data.base_pipeline import DataPreparationBaseFlow
from rxnutils.data.ord.import_ord_dataset import main as import_data


class OrdDataPreparationFlow(DataPreparationBaseFlow):
    """Pipeline for extracting ORD data and do some clean-up"""

    ord_data = Parameter("ord-data")

    data_prefix = "ord"

    @step
    def start(self):
        """Import ORD data"""
        output_name = str(Path(self.folder) / "ord_data.csv")
        if not Path(output_name).exists():
            filenames = glob.glob(os.path.join(self.ord_data, "data", "*", "*"))
            import_data(
                [
                    "--output",
                    output_name,
                    "--filenames",
                ]
                + filenames
            )
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
            f"Processed file is locate here: {Path(self.folder) / 'ord_data_cleaned.csv'}"
        )


if __name__ == "__main__":
    OrdDataPreparationFlow()
