"""
Module containing pipeline for mapping with rxnmapper
This needs to be run in an environment with rxnmapper installed
"""
from pathlib import Path

from metaflow import step

from rxnutils.data.uspto.base_pipeline import UsptoBaseFlow
from rxnutils.data.uspto.mapping import main as map_uspto


class RxnMappingFlow(UsptoBaseFlow):
    """Pipeline for atom-map USPTO data with rxnmapper"""

    @step
    def start(self):
        """Setup batches for mapping"""
        # pylint: disable=attribute-defined-outside-init
        self.partitions = self._create_batches(
            Path(self.folder) / "uspto_data_cleaned.csv",
            Path(self.folder) / "uspto_data_mapped.csv",
        )
        self.next(self.do_mapping, foreach="partitions")

    @step
    def do_mapping(self):
        """Perform atom-mapping of reactions"""
        idx, start, end = self.input
        if idx > -1:
            map_uspto(
                [
                    "--input",
                    str(Path(self.folder) / "uspto_data_cleaned.csv"),
                    "--output",
                    str(Path(self.folder) / f"uspto_data_mapped.csv.{idx}"),
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
        self._combine_batches(Path(self.folder) / "uspto_data_mapped.csv")
        self.next(self.end)

    @step
    def end(self):
        """Final step, just print information"""
        print(
            f"Processed file is locate here: {Path(self.folder) / 'uspto_data_mapped.csv'}"
        )


if __name__ == "__main__":
    RxnMappingFlow()
