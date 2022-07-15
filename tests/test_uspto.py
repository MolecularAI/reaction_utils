import pandas as pd

from rxnutils.data.uspto.combine import main as combine_uspto


def test_combine_uspto(shared_datadir, tmpdir):
    combine_uspto(
        ["--filenames", "uspto_example_reactions.rsmi", "--folder", str(shared_datadir)]
    )

    data = pd.read_csv(shared_datadir / "uspto_data.csv", sep="\t")

    assert len(data) == 9
    assert data.columns.to_list() == ["ID", "Year", "ReactionSmiles"]
    assert data["ID"].to_list()[:2] == ["US03930836;;0", "US03930836;;1"]
