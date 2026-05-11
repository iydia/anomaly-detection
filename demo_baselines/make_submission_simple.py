"""
Demo 1: Simple Test-Only Baseline
The following script reads all test.npy files from the dataset and generates a valid JSON file.
It uses a simple z-score rule:
- Compute the mean and standard deviation of one test window.
- Mark points with abs(z-score) > 2.0 as anomalies.
- Mark all other points as normal.
This method does not train a model. It is only for testing the submission process.

Save the following code as make_submission_simple.py:
Before running the script, usually you only need to check these two lines:
DATASET_ROOT = Path("student_dataset")
OUTPUT_JSON = Path("submission.json")
Set DATASET_ROOT to the folder where you extracted the dataset. For example, if your dataset folder is named student, change it to Path("student").
OUTPUT_JSON is the output file name. You can keep it as submission.json.
"""
import json
from pathlib import Path

import numpy as np

DATASET_ROOT = Path("student_dataset")
OUTPUT_JSON = Path("submission.json")


def predict_window(test_npy_path: Path, threshold: float = 2.0) -> list[int]:
    x = np.asarray(np.load(test_npy_path), dtype=np.float64).ravel()

    if x.size == 0:
        return []

    mean = float(np.mean(x))
    std = float(np.std(x))

    if std < 1e-9:
        return [0] * int(x.size)

    z = (x - mean) / (std + 1e-9)
    pred = (np.abs(z) > threshold).astype(np.int64)
    return [int(v) for v in pred.tolist()]


def main() -> None:
    predictions = {}

    window_dirs = sorted(
        p for p in DATASET_ROOT.iterdir()
        if p.is_dir() and (p / "test.npy").is_file()
    )

    if not window_dirs:
        raise RuntimeError(f"No test windows found under {DATASET_ROOT}")

    for window_dir in window_dirs:
        window_id = window_dir.name.split("_", 1)[0]
        predictions[window_id] = predict_window(window_dir / "test.npy")

    payload = {"predictions": predictions}

    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    print(f"Wrote {OUTPUT_JSON}")
    print(f"Number of windows: {len(predictions)}")


if __name__ == "__main__":
    main()
