"""
Data Loading and Splitting Module for NYC Taxi Trip Duration Pipeline.

- Loads raw NYC Taxi dataset.
- Performs time-aware temporal splitting (Train: Jan-Apr 2016, Eval: May 2016, Holdout: June 2016).
- Saves raw splits to data/processed/ (or custom output_dir for isolated testing).
"""

from pathlib import Path
from typing import Tuple, Union
import pandas as pd

from src.config import settings


def load_and_split_data(
    raw_path: Union[Path, str, None] = None,
    output_dir: Union[Path, str, None] = None,
    eval_cutoff: str = "2016-05-01",
    holdout_cutoff: str = "2016-06-01",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw NYC dataset and split into train, eval, and holdout sets by date.

    Parameters
    ----------
    raw_path : Path | str | None
        Path to raw CSV file. Defaults to settings.raw_data_path.
    output_dir : Path | str | None
        Directory to save split CSVs. Defaults to settings.processed_dir.
    eval_cutoff : str
        Start date for the evaluation set (YYYY-MM-DD).
    holdout_cutoff : str
        Start date for the holdout set (YYYY-MM-DD).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, eval_df, holdout_df)
    """
    raw_file = Path(raw_path) if raw_path is not None else settings.raw_data_path
    out_dir = Path(output_dir) if output_dir is not None else settings.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw dataset from: {raw_file}")
    df = pd.read_csv(raw_file)

    # Ensure datetime format for time-based split
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df = df.sort_values("pickup_datetime").reset_index(drop=True)

    t_eval = pd.Timestamp(eval_cutoff)
    t_holdout = pd.Timestamp(holdout_cutoff)

    # Time-based splitting
    # Temporal splitting: We split chronologically (Jan-Apr for train, May for eval,
    # June for holdout) rather than random train_test_split. In real-world transportation
    # and taxi demand, random splitting causes temporal data leakage because trips on the
    # same rainy day or holiday would appear in both train and test sets.
    train_df = df[df["pickup_datetime"] < t_eval].copy()
    eval_df = df[(df["pickup_datetime"] >= t_eval) & (df["pickup_datetime"] < t_holdout)].copy()
    holdout_df = df[df["pickup_datetime"] >= t_holdout].copy()

    # Save to disk
    train_df.to_csv(out_dir / "raw_train.csv", index=False)
    eval_df.to_csv(out_dir / "raw_eval.csv", index=False)
    holdout_df.to_csv(out_dir / "raw_holdout.csv", index=False)

    print(f"Data split completed and saved to {out_dir}:")
    print(f"   Train:   {train_df.shape} (up to {t_eval.date()})")
    print(f"   Eval:    {eval_df.shape} ({t_eval.date()} to {t_holdout.date()})")
    print(f"   Holdout: {holdout_df.shape} (from {t_holdout.date()} onwards)")

    return train_df, eval_df, holdout_df


if __name__ == "__main__":
    load_and_split_data()
