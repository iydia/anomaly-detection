"""
To compile .venv:
pip install numpy

To generate submission.json:
source /Users/lydiahe/anomaly-detection/.venv/bin/activate
python3 demo_baselines/make_submission_trained.py
"""
import json
from pathlib import Path

import numpy as np

DATASET_ROOT = Path("student_dataset")
OUTPUT_JSON = Path("submission_trained.json")


def binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def anomaly_score(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        values = x.reshape(-1, 1)
    else:
        values = x.reshape(x.shape[0], -1)

    z = np.abs((values - mean) / (std + 1e-9))
    return np.max(z, axis=1)


def fit_threshold(train_x: np.ndarray, train_y: np.ndarray) -> tuple[float, float, float]:
    train_x = np.asarray(train_x, dtype=np.float64)
    train_y = np.asarray(train_y).astype(int).ravel()

    mean = float(np.mean(train_x))
    std = float(np.std(train_x))

    if std < 1e-9:
        return mean, std, float("inf")

    scores = anomaly_score(train_x, mean, std)
    if scores.shape[0] != train_y.shape[0]:
        raise ValueError(f"train length mismatch: score={scores.shape[0]} label={train_y.shape[0]}")

    if int(np.sum(train_y == 1)) == 0:
        return mean, std, float(np.max(scores) + 1e-9)

    candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 101)))
    best_threshold = float(candidates[0])
    best_f1 = -1.0

    for threshold in candidates:
        pred = (scores > threshold).astype(np.int64)
        f1 = binary_f1(train_y, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return mean, std, best_threshold


def predict_window(window_dir: Path) -> list[int]:
    train_x = np.load(window_dir / "train.npy")
    train_y = np.load(window_dir / "train_label.npy")
    test_x = np.load(window_dir / "test.npy")

    mean, std, threshold = fit_threshold(train_x, train_y)
    test_scores = anomaly_score(test_x, mean, std)
    pred = (test_scores > threshold).astype(np.int64)
    return [int(v) for v in pred.tolist()]


def main() -> None:
    predictions = {}

    window_dirs = sorted(
        p for p in DATASET_ROOT.iterdir()
        if p.is_dir()
        and (p / "train.npy").is_file()
        and (p / "train_label.npy").is_file()
        and (p / "test.npy").is_file()
    )

    if not window_dirs:
        raise RuntimeError(f"No valid windows found under {DATASET_ROOT}")

    for window_dir in window_dirs:
        window_id = window_dir.name.split("_", 1)[0]
        predictions[window_id] = predict_window(window_dir)

    payload = {"predictions": predictions}

    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    print(f"Wrote {OUTPUT_JSON}")
    print(f"Number of windows: {len(predictions)}")


if __name__ == "__main__":
    main()
