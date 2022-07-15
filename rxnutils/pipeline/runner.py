"""Module containg routines and interface to run pipelines"""
import argparse
from typing import Dict, Any, Optional, Sequence

import yaml
import pandas as pd

# imports needed to register all the actions
# pylint: disable=unused-import
import rxnutils.pipeline.actions.reaction_mod  # noqa
import rxnutils.pipeline.actions.reaction_props  # noqa
import rxnutils.pipeline.actions.templates  # noqa
import rxnutils.pipeline.actions.dataframe_mod  # noqa
from rxnutils.pipeline.base import create_action, global_apply, list_actions


def run_pipline(
    data: pd.DataFrame,
    pipeline: Dict[str, Any],
    filename: str,
    save_intermediates: bool = True,
) -> pd.DataFrame:
    """
    Run a given pipeline on a dataset

    The actions are applied sequentials as they are defined in the pipeline

    The intermediate results of the pipeline will be written to separate
    tab-separated CSV files.

    :param data: the dataset
    :param pipeline: the action specifications
    :param filename: path to the final output file
    :param save_intermediates: if True will save intermediate results
    :return: the dataset after completing the pipeline
    """
    actions = [
        create_action(name, **(options or {})) for name, options in pipeline.items()
    ]

    for idx, action in enumerate(actions):
        print(f"Running {action}", flush=True)
        data = action(data)
        if save_intermediates:
            data.to_csv(f"{filename}.{idx}", index=False, sep="\t")
    return data


def main(args: Optional[Sequence[str]] = None) -> None:
    """Function for command line argument"""
    parser = argparse.ArgumentParser("Runner of validation pipeline")
    parser.add_argument("--pipeline", help="the yaml file with a pipeline")
    parser.add_argument(
        "--data", help="the data to be processed. Should be a tab-separated CSV-file"
    )
    parser.add_argument(
        "--output", help="the processed data. Will be a tab-separated CSV-file"
    )
    parser.add_argument("--max-workers", type=int, help="the maximum number of works")
    parser.add_argument(
        "--batch", type=int, nargs=2, help="Line numbers to start and stop reading rows"
    )
    parser.add_argument("--no-intermediates", action="store_true", default=False)
    parser.add_argument("--list", action="store_true", default=False)
    args = parser.parse_args(args)

    if args.list:
        print("The available actions are as follow:")
        list_actions()
        return

    global_apply.max_workers = args.max_workers

    with open(args.pipeline, "r") as fileobj:
        pipeline = yaml.load(fileobj, Loader=yaml.SafeLoader)

    if args.batch:
        start, end = args.batch
        data = pd.read_csv(
            args.data,
            sep="\t",
            index_col=False,
            nrows=end - start,
            skiprows=range(1, start),
        )
    else:
        data = pd.read_csv(args.data, sep="\t")
    data = run_pipline(data, pipeline, args.output, not args.no_intermediates)
    data.to_csv(args.output, index=False, sep="\t")


if __name__ == "__main__":
    main()
