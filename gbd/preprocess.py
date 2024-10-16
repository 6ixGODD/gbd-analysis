from __future__ import annotations

import io
import pathlib
import typing

import joblib
import pandas as pd
from tabulate import tabulate

from gbd.common import utils as _utils

LOGGER = _utils.Logger()


def load_csv_files(root_path: pathlib.Path) -> pd.DataFrame:
    data_frames: typing.List[pd.DataFrame] = []
    LOGGER.info(f"Loading data from {root_path.name}")

    for dir_ in root_path.iterdir():
        if dir_.is_dir():
            for file_path in dir_.glob("*.csv"):
                df = pd.read_csv(file_path)
                data_frames.append(df)
                LOGGER.info(
                    f"Loaded {_utils.ANSIFormatter.format(str(file_path), _utils.ANSIFormatter.UNDERLINE)}"
                    f" into DataFrame"
                )

    if len(data_frames):
        concatenated_df = pd.concat(data_frames, ignore_index=True)
        LOGGER.info(f"Concatenated {len(data_frames)} files from {root_path.name}")
        return concatenated_df
    else:
        LOGGER.warning(f"No CSV files found in {root_path}")
        return pd.DataFrame()


def display_details(df: pd.DataFrame, name: str) -> None:
    print(f"=== Details of {name} ===")

    # Head
    print(f"--- Head of {name} ---")
    print(df.head())

    # Info
    print(f"--- Info of {name} ---")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    print(info_str)

    # Description
    print(f"--- Description of {name} ---")
    description = df.describe(include='all')
    details = {
        "Shape":       [df.shape],
        "Columns":     [list(df.columns)],
        "Description": [description]
    }
    print(tabulate(details.items(), headers=["Details", "Values"], tablefmt="grid"))

    # Unique Values for Specific Columns
    unique_columns = {
        "measure_name":  "Measures",
        "location_name": "Locations",
        "year":          "Years",
        "cause_name":    "Causes",
        "metric_name":   "Metrics",
        "age_name":      "Age Groups",
        "rei_name":      "Risk Factors",
        "sex_name":      "Gender"
    }

    for col, desc in unique_columns.items():
        if col in df.columns:
            print(f"--- Unique {desc} in {name} ---")
            unique_values = df[col].unique()
            print(f"{desc}: {unique_values}\n")


def run(
    path: pathlib.Path,
    *args: str,
    serialize: bool = True
) -> None:
    if not path.exists():
        LOGGER.error(f"Path not found: {path}")
        raise FileNotFoundError(f"Path not found: {path}")
    if path.is_dir():
        for dir_ in args:
            data = load_csv_files(path / dir_)
            LOGGER.info(f"Data loading for {dir_} completed successfully.")

            if serialize:
                joblib.dump(data, path / f"data_{_utils.to_snake_case(dir_)}.pkl")
                LOGGER.info(f"Data serialization for {dir_} completed successfully.")

            display_details(data, dir_)
    else:
        data: pd.DataFrame = joblib.load(path)
        LOGGER.info(f"Data loaded from {path}")
        display_details(data, path.stem)


if __name__ == "__main__":
    import os

    os.chdir(pathlib.Path(__file__).parent.parent)

    # run(
    #     pathlib.Path("data"),
    #     "Cause of death or injury",
    #     "Risk factors",
    #     serialize=False,
    # )

    run(pathlib.Path("data") / "data_risk_factors.pkl")
    run(pathlib.Path("data") / "data_cause_of_death_or_injury.pkl")