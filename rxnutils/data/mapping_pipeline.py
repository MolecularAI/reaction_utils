"""
Module containing pipeline for mapping with rxnmapper
This needs to be run in an environment with rxnmapper installed
"""
from pathlib import Path

from metaflow import step, Parameter

from rxnutils.data.base_pipeline import DataBaseFlow
from rxnutils.data.mapping import main as map_data


class RxnMappingFlow(DataBaseFlow):
    """Pipeline for atom-map USPTO or ORD data with rxnmapper"""

    data_prefix = Parameter("data-prefix")

    @step
    def start(self):
        """Setup batches for mapping"""
        # pylint: disable=attribute-defined-outside-init
        self.partitions = self._create_batches(
            Path(self.folder) / f"{self.data_prefix}_data_cleaned.csv",
            Path(self.folder) / f"{self.data_prefix}_data_mapped.csv",
        )
        self.next(self.do_mapping, foreach="partitions")

    @step
    def do_mapping(self):
        """Perform atom-mapping of reactions"""
        idx, start, end = self.input
        output_filename = (
            Path(self.folder) / f"{self.data_prefix}_data_mapped.csv.{idx}"
        )
        if idx > -1 and not output_filename.exists():
            map_data(
                [
                    "--input",
                    str(Path(self.folder) / f"{self.data_prefix}_data_cleaned.csv"),
                    "--output",
                    str(output_filename),
                    "--batch",
                    str(start),
                    str(end),
                    "--mapper_batch_size",
                    "2",
                ]
            )
        self.next(self.join_mapping)

    @step
    def join_mapping(self, _):
        """Join batches from mapping"""
        self._combine_batches(Path(self.folder) / f"{self.data_prefix}_data_mapped.csv")
        self.next(self.end)

    @step
    def end(self):
        """Final step, just print information"""
        filename = Path(self.folder) / f"{self.data_prefix}_data_mapped.csv"
        print(f"Processed file is locate here: {filename}")


if __name__ == "__main__":
    RxnMappingFlow()
