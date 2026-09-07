"""
Batch Prediction Runner for NYC Taxi Trip Duration.

- Processes batches of unlabelled or holdout trips.
- Leverages the production inference engine.
- Saves timestamped prediction artifacts to data/predictions/.
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
import pandas as pd

from src.config import settings
from src.inference_pipeline.inference import predict


def run_batch_predictions(
    input_path: Union[Path, str, None] = None,
    output_dir: Union[Path, str, None] = None,
    sample_limit: Optional[int] = 10000,
) -> pd.DataFrame:
    """Execute batch predictions on a dataset."""
    in_file = Path(input_path) if input_path is not None else settings.processed_dir / "raw_holdout.csv"
    out_dir = Path(output_dir) if output_dir is not None else settings.predictions_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 Starting batch inference from {in_file}...")
    if sample_limit:
        df = pd.read_csv(in_file, nrows=sample_limit)
        print(f"   Sampled {len(df)} rows for batch run.")
    else:
        df = pd.read_csv(in_file)
        print(f"   Loaded full dataset: {len(df)} rows.")

    preds_df = predict(df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = out_dir / f"batch_preds_{timestamp}.csv"
    preds_df.to_csv(output_file, index=False)

    print(f"✅ Batch predictions complete! Saved to {output_file}")
    print(preds_df[["pickup_datetime", "predicted_duration_seconds", "predicted_duration_minutes"]].head())

    return preds_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch predictions on NYC taxi data.")
    parser.add_argument("--input", type=str, default=None, help="Input raw CSV path")
    parser.add_argument("--limit", type=int, default=5000, help="Number of rows to predict")
    args = parser.parse_args()

    run_batch_predictions(input_path=args.input, sample_limit=args.limit)
