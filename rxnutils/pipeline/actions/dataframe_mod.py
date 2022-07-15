"""Module containing actions that modify the dataframe in some way"""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

import pandas as pd
from rdkit import RDLogger
from rxnutils.pipeline.base import action

rd_logger = RDLogger.logger()
rd_logger.setLevel(RDLogger.CRITICAL)


@action
@dataclass
class DropColumns:
    """Drops columns specified in 'columns'

    yaml example:

    drop_columns:
        columns:
            - NRingChange
            - RingBondMade
    """

    pretty_name: ClassVar[str] = "drop_columns"
    columns: list[str]

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.drop(self.columns, 1)

    def __str__(self) -> str:
        return f"{self.pretty_name} (drop columns in specified list, keeps the rest)"


@action
@dataclass
class DropDuplicates:
    """Action for dropping duplicates from dataframe"""

    pretty_name: ClassVar[str] = "drop_duplicates"
    key_columns: list[str]

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.drop_duplicates(subset=self.key_columns)

    def __str__(self) -> str:
        return f"{self.pretty_name} (remove duplicates based on the provided list of columns for comparison)"


@action
@dataclass
class KeepColumns:
    """Drops columns not specified in 'columns'

    yaml example:

    keep_columns:
        columns:
            - id
            - classification
            - rsmi_processed
    """

    pretty_name: ClassVar[str] = "keep_columns"
    columns: list[str]

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        drop_columns = list(data.columns)
        for name in self.columns:
            drop_columns.remove(name)

        return data.drop(drop_columns, 1)

    def __str__(self) -> str:
        return f"{self.pretty_name} (keeps columns in specified list, drops the rest)"


@action
@dataclass
class QueryDataframe:
    """Uses dataframe query to produce a new (smaller) dataframe. Query must conform to pandas.query()

    yaml file example: Keeping only rows where the has_stereo columns is True:

    query_dataframe:
        query: has_stereo == True
    """

    pretty_name: ClassVar[str] = "query_dataframe"
    query: str

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.query(self.query)

    def __str__(self) -> str:
        return f"{self.pretty_name} (query the dataframe using {self.query} and returns the filtered dataframe)"


@action
@dataclass
class StackColumns:
    """Stacks the specified in_columns under a new column name (out_column),
    multiplies the rest of the columns as appropriate

    yaml control file example:

    stack_columns:
        in_columns:
            - rsmi_processed
            - rsmi_inverted_stereo
        out_column: rsmi_processed
    """

    pretty_name: ClassVar[str] = "stack_columns"
    in_columns: list[str]
    out_column: str = "StackedColumns"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        var_column_name = "D5B447A5B810"  # Temporary name with low risk of collision

        keep_columns = list(data.columns)
        for name in self.in_columns:

            keep_columns.remove(name)

        return data.melt(
            id_vars=keep_columns,
            value_vars=self.in_columns,
            var_name=var_column_name,
            value_name=self.out_column,
        ).drop(var_column_name, 1)

    def __str__(self) -> str:
        return f"{self.pretty_name} (stacks two columns into a single column)"


@action
@dataclass
class StackMultiColumns:
    """Stacks the specified target_columns on top of the stack_columns after renaming.

    Example Yaml:

    stack_multi_columns:
        stack_columns:
            - rsmi_inverted_stereo
            - PseudoHash_inverted_stereo
        target_columns:
            - rsmi_processed
            - PseudoHash
    """

    pretty_name: ClassVar[str] = "stack_multi_columns"
    stack_columns: list[str]
    target_columns: list[str]

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        # Wonder if this creates two copies in memory giving memory issues, or reference the original data?
        df_target = data.drop(columns=self.stack_columns)
        df_stack = data.drop(columns=self.target_columns)

        rename_dict = dict(zip(self.stack_columns, self.target_columns))
        df_stack = df_stack.rename(columns=rename_dict)

        return pd.concat([df_target, df_stack])

    def __str__(self) -> str:
        return f"{self.pretty_name} (stacks multiple columns onto multiple columns)"
