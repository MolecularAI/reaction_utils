import pandas as pd
import pytest

from rxnutils.pipeline.actions.templates import (
    CountTemplateComponents,
    RetroTemplateReproduction,
)
from rxnutils.pipeline.base import global_apply

global_apply.max_workers = 1


def test_count_components(make_template_dataframe):
    action = CountTemplateComponents(in_column="retrotemplate")
    df = make_template_dataframe

    df2 = action(df)

    assert df2["nreactants"].to_list()[:2] == [1, 1]
    assert df2["nreagents"].to_list()[:2] == [0, 0]
    assert df2["nproducts"].to_list()[:2] == [1, 1]

    assert df2["nreactants"].isna().to_list() == [False, False, True]
    assert df2["nreagents"].isna().to_list() == [False, False, True]
    assert df2["nproducts"].isna().to_list() == [False, False, True]


def test_template_reproduction(make_template_dataframe):
    action = RetroTemplateReproduction(
        template_column="retrotemplate", smiles_column="rsmi"
    )
    df = make_template_dataframe

    df2 = action(df)

    assert df2["TemplateGivesTrueReactants"].to_list() == [True, False, False]
    assert df2["TemplateGivesOtherReactants"].to_list() == [True, False, False]
    assert df2["TemplateGivesNOutcomes"].to_list() == [2, 0, 0]
