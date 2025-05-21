import pandas as pd

from rxnutils.data.uspto.combine import main as combine_uspto
from rxnutils.data.uspto.uspto_yield import UsptoYieldCuration


def test_combine_uspto(shared_datadir):
    combine_uspto(["--filenames", "uspto_example_reactions.rsmi", "--folder", str(shared_datadir)])

    data = pd.read_csv(shared_datadir / "uspto_data.csv", sep="\t")

    assert len(data) == 9
    assert data.columns.to_list() == ["ID", "Year", "ReactionSmiles"]
    assert data["ID"].to_list()[:2] == ["US03930836;;0", "US03930836;;1"]


def test_uspto_yield_curation():

    data = pd.DataFrame(
        {
            "TextMinedYield": [
                "~10",
                "50%",
                ">40",
                ">=50",
                "<30",
                "20 to 50",
                "-50",
                "100",
                "50",
                "",
                "",
            ],
            "CalculatedYield": [
                "12%",
                "110",
                "40",
                "51",
                "",
                "40",
                "-40",
                "120",
                "40",
                "70",
                "",
            ],
        }
    )
    action = UsptoYieldCuration()

    data2 = action(data)

    expected = [False] * 6 + [True] + [False] * 3 + [True]
    assert data2["CuratedYield"].isna().tolist() == expected
    assert data2["CuratedYield"].tolist()[:6] == [12, 50, 40, 51, 30, 50]
    assert data2["CuratedYield"].tolist()[7:-1] == [100, 50, 70]
