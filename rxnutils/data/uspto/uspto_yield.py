"""
Code for curating USPTO yields.

Inspiration from this code: https://github.com/DocMinus/Yield_curation_USPTO

This could potentially be an action, but since it only make sens to use it
with USPTO data, it resides here for now.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class UsptoYieldCuration:
    """
    Action for curating USPTO yield columns
    """

    text_yield_column: str = "TextMinedYield"
    calc_yield_column: str = "CalculatedYield"
    out_column: str = "CuratedYield"

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        calc_yield = data[self.calc_yield_column].str.rstrip("%")
        calc_yield = pd.to_numeric(calc_yield, errors="coerce")
        calc_yield[(calc_yield < 0) | (calc_yield > 100)] = np.nan

        text_yield = data[self.text_yield_column].str.lstrip("~")
        text_yield = text_yield.str.rstrip("%")
        text_yield = text_yield.str.replace(">=", "", regex=False)
        text_yield = text_yield.str.replace(">", "", regex=False)
        text_yield = text_yield.str.replace("<", "", regex=False)
        text_yield = text_yield.str.replace(r"\d{1,2}\sto\s", "", regex=True)
        text_yield = pd.to_numeric(text_yield, errors="coerce")
        text_yield[(text_yield < 0) | (text_yield > 100)] = np.nan

        curated_yield = text_yield.copy()

        sel = (~calc_yield.isna()) & (~text_yield.isna())
        curated_yield[sel] = np.maximum(calc_yield[sel], text_yield[sel])

        sel = (~calc_yield.isna()) & (text_yield.isna())
        curated_yield[sel] = calc_yield[sel]

        return data.assign(**{self.out_column: curated_yield})

    def __str__(self) -> str:
        return f"{self.pretty_name} (create one column with curated yield values)"
