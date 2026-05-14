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

To compile .venv:
pip install numpy

To generate submission.json:
source /Users/lydiahe/anomaly-detection/.venv/bin/activate
python3 demo_baselines/make_submission_simple.py

"""

import json
from pathlib import Path
import numpy as np

DATASET_ROOT = Path("student_dataset")
OUTPUT_JSON = Path("submission.json")


def predict_window(test_npy_path: Path, threshold: float = 2.0) -> list[int]:
    x = np.asarray(np.load(test_npy_path), dtype=np.float64).ravel() # load test.npy file, ensure it's a 1D array of floats

    if x.size == 0:
        return []

    mean = float(np.mean(x))
    std = float(np.std(x))

    if std < 1e-9: # atdev very small, cannot compute z-scores reliably
        return [0] * int(x.size) # mark points as normal

    z = (x - mean) / (std + 1e-9) # compute z-scores, add small epsilon to avoid division by zero
    pred = (np.abs(z) > threshold).astype(np.int64) # mark abs(z-score) > threshold as anomalies (1), else normal (0)
    return [int(v) for v in pred.tolist()]


def main() -> None:
    predictions = {}
    window_dirs = sorted(
        p for p in DATASET_ROOT.iterdir()
        if p.is_dir() and (p / "test.npy").is_file() # only consider directories that contain test.npy
    )
    if not window_dirs:
        raise RuntimeError(f"No test windows found under {DATASET_ROOT}")

    for window_dir in window_dirs: # each window_dir is expected to be a directory containing test.npy
        window_id = window_dir.name.split("_", 1)[0] # extract window_id from the directory name
        predictions[window_id] = predict_window(window_dir / "test.npy") # generate predictions for the test.npy file in the window_dir

    payload = {"predictions": predictions}

    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Number of windows: {len(predictions)}")


if __name__ == "__main__":
    main()
