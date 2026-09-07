"""
Data Preprocessing Module for NYC Taxi Trip Duration Pipeline.

- Drops leakage columns (dropoff_datetime, id).
- Drops redundant/low-variance columns (store_and_fwd_flag, passenger_count).
- Filters duration outliers (100 < trip_duration <= 4000).
- Cleans geographic coordinate outliers outside the NYC bounding box.
- Removes duplicates.
"""

from pathlib import Path
from typing import Tuple, Union
import pandas as pd

from src.config import settings


def drop_leakage_and_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Drop target leakage and unneeded metadata columns.
    
    - dropoff_datetime is direct target leakage (trip_duration = dropoff - pickup).
    - id is an arbitrary identifier with zero predictive value.
    - store_and_fwd_flag has >99% identical values (near zero variance).
    - passenger_count showed negligible correlation with duration during EDA.
    """
    cols_to_drop = ["dropoff_datetime", "id", "store_and_fwd_flag", "passenger_count"]
    existing_cols = [c for c in cols_to_drop if c in df.columns]
    if existing_cols:
        df = df.drop(columns=existing_cols)
    return df


def clean_coordinates(
    df: pd.DataFrame,
    min_lat: float = settings.min_latitude,
    max_lat: float = settings.max_latitude,
    min_lon: float = settings.min_longitude,
    max_lon: float = settings.max_longitude,
) -> pd.DataFrame:
    """Filter out trips with GPS coordinates outside the specified bounding box."""
    coord_cols = {"pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"}
    if not coord_cols.issubset(df.columns):
        return df

    before = len(df)
    mask = (
        (df["pickup_latitude"] >= min_lat)
        & (df["pickup_latitude"] <= max_lat)
        & (df["pickup_longitude"] >= min_lon)
        & (df["pickup_longitude"] <= max_lon)
        & (df["dropoff_latitude"] >= min_lat)
        & (df["dropoff_latitude"] <= max_lat)
        & (df["dropoff_longitude"] >= min_lon)
        & (df["dropoff_longitude"] <= max_lon)
    )
    df = df[mask].reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"   Dropped {before - after} rows outside NYC coordinate bounding box.")
    return df


def filter_duration_outliers(
    df: pd.DataFrame,
    min_duration: int = settings.min_trip_duration,
    max_duration: int = settings.max_trip_duration,
) -> pd.DataFrame:
    """Filter trip duration outliers. Only executed if target column exists (training/eval)."""
    if "trip_duration" not in df.columns:
        return df

    before = len(df)
    mask = (df["trip_duration"] > min_duration) & (df["trip_duration"] <= max_duration)
    df = df[mask].reset_index(drop=True)
    after = len(df)
    print(f"   Filtered duration outliers: {before} -> {after} rows (kept {min_duration}s < duration <= {max_duration}s).")
    return df


def preprocess_data(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """Full preprocessing pipeline on a single DataFrame."""
    df = drop_leakage_and_metadata(df)
    df = df.drop_duplicates().reset_index(drop=True)

    if is_training:
        df = clean_coordinates(df)
        df = filter_duration_outliers(df)

    return df


def preprocess_split(
    split: str,
    input_dir: Union[Path, str, None] = None,
    output_dir: Union[Path, str, None] = None,
) -> pd.DataFrame:
    """Preprocess a specific split file (raw_<split>.csv) and write to cleaned_<split>.csv."""
    in_dir = Path(input_dir) if input_dir is not None else settings.processed_dir
    out_dir = Path(output_dir) if output_dir is not None else settings.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    in_path = in_dir / f"raw_{split}.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Input split file not found: {in_path}")

    print(f"Preprocessing {split} from {in_path}...")
    df = pd.read_csv(in_path)
    cleaned_df = preprocess_data(df, is_training=(split in ["train", "eval"]))

    out_path = out_dir / f"cleaned_{split}.csv"
    cleaned_df.to_csv(out_path, index=False)
    print(f"Cleaned {split} saved to {out_path} ({cleaned_df.shape})")
    return cleaned_df


def run_preprocess(
    splits: Tuple[str, ...] = ("train", "eval", "holdout"),
    input_dir: Union[Path, str, None] = None,
    output_dir: Union[Path, str, None] = None,
):
    """Run preprocessing for all splits."""
    for split in splits:
        preprocess_split(split, input_dir=input_dir, output_dir=output_dir)


if __name__ == "__main__":
    run_preprocess()
