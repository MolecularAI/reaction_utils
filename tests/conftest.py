import pandas as pd
import pytest


@pytest.fixture
def make_reaction_dataframe(shared_datadir):
    return pd.read_csv(shared_datadir / "example_reactions.csv")


@pytest.fixture
def make_template_dataframe(shared_datadir):
    return pd.read_csv(shared_datadir / "example_templates.csv")
