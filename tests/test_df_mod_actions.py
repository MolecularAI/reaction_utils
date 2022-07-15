import pandas as pd

from rxnutils.pipeline.actions.dataframe_mod import (
    DropColumns,
    DropDuplicates,
    KeepColumns,
    QueryDataframe,
    StackColumns,
    StackMultiColumns,
)


def test_drop_columns():
    action = DropColumns(columns=["B"])
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    df2 = action(df)

    assert "B" not in df2.columns


def test_drop_duplicates():
    action = DropDuplicates(key_columns=["B"])
    df = pd.DataFrame({"A": [1, 1, 1], "B": [3, 4, 3]})

    df2 = action(df)

    assert len(df2) == 2


def test_keep_columns():
    action = KeepColumns(columns=["B"])
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    df2 = action(df)

    assert "A" not in df2.columns


def test_query_df():
    action = QueryDataframe(query=("B==3"))
    df = pd.DataFrame({"A": [1, 1, 2], "B": [3, 4, 3]})

    df2 = action(df)

    assert len(df2) == 2
    assert df2["B"].to_list() == [3, 3]
    assert df2["A"].to_list() == [1, 2]


def test_stack_columns():
    action = StackColumns(in_columns=["A", "B"], out_column="D")
    df = pd.DataFrame({"A": [1, 1, 1], "B": [3, 4, 3], "C": ["A", "B", "C"]})

    df2 = action(df)

    assert len(df2) == 6
    assert df2.columns.to_list() == ["C", "D"]
    assert df2["C"].to_list() == ["A", "B", "C", "A", "B", "C"]
    assert df2["D"].to_list() == [1, 1, 1, 3, 4, 3]


def test_stack_multi_columns():
    action = StackMultiColumns(stack_columns=["A", "B"], target_columns=["C", "D"])
    df = pd.DataFrame(
        {"A": [1, 1, 1], "B": ["A", "B", "C"], "C": [3, 4, 3], "D": ["X", "Y", "Z"]}
    )

    df2 = action(df)

    assert len(df2) == 6
    assert df2.columns.to_list() == ["C", "D"]
    assert df2["C"].to_list() == [3, 4, 3, 1, 1, 1]
    assert df2["D"].to_list() == ["X", "Y", "Z", "A", "B", "C"]
