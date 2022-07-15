import pandas as pd

from rxnutils.pipeline.runner import main as pipline_runner


def test_pipeline(shared_datadir, tmpdir):
    pipline_runner(
        [
            "--pipeline",
            str(shared_datadir / "example_pipeline.yml"),
            "--data",
            str(shared_datadir / "example_reactions.csv"),
            "--output",
            str(tmpdir / "data.csv"),
            "--max-workers",
            "1",
        ]
    )

    data = pd.read_csv(tmpdir / "data.csv", sep="\t")

    assert data.columns.to_list() == [
        "rsmi",
        "BadMolecules",
        "rsmi_clean",
        "NReactants",
        "NMappedReactants",
        "NReagents",
        "NMappedReagents",
        "NProducts",
        "NMappedProducts",
    ]

    assert data["NReactants"].to_list() == [4, 3]
    assert data["NMappedReactants"].to_list() == [2, 2]
    assert data["NReagents"].to_list() == [0, 0]
    assert data["NMappedReagents"].to_list() == [0, 0]
    assert data["BadMolecules"].to_list()[0] == "C(C(CC)(O[Mg+2])C)C"
    assert data["BadMolecules"].isna()[1]


def test_pipeline_batched(shared_datadir, tmpdir):
    pipline_runner(
        [
            "--pipeline",
            str(shared_datadir / "example_pipeline.yml"),
            "--data",
            str(shared_datadir / "example_reactions.csv"),
            "--output",
            str(tmpdir / "data.csv"),
            "--max-workers",
            "1",
            "--batch",
            "0",
            "1",
        ]
    )

    data = pd.read_csv(tmpdir / "data.csv", sep="\t")

    assert data["NReactants"].to_list() == [4]
    assert data["NMappedReactants"].to_list() == [2]
    assert data["NReagents"].to_list() == [0]
    assert data["NMappedReagents"].to_list() == [0]
    assert data["BadMolecules"].to_list() == ["C(C(CC)(O[Mg+2])C)C"]
