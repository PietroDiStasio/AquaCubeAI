"""Paired FNU regression and threshold-exceedance metrics."""

from typing import Dict, Sequence, Tuple
import numpy as np
import torch
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from .data import Dataset, load_norm_params, invert_minmax_np
from .models import RegressNN

TURBIDITY_RANGES = [(0.0, 40.0), (0.0, 80.0), (0.0, 100.0), (0.0, 140.0), (80.0, 140.0)]


def torch_to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def compute_metrics_np(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    diff = y_pred - y_true
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    r2 = float(r2_score(y_true, y_pred))
    pr, _ = pearsonr(y_true, y_pred)
    return {"mse": mse, "rmse": rmse, "mae": mae, "bias": bias, "r2": r2, "pearson": float(pr)}


def metrics_by_ranges(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ranges: Sequence[Tuple[float, float]] = TURBIDITY_RANGES,
    min_n: int = 25,
) -> Dict[str, Dict[str, float]]:
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    out: Dict[str, Dict[str, float]] = {}
    for lo, hi in ranges:
        mask = (yt >= lo) & (yt < hi)
        n = int(mask.sum())
        key = f"{int(lo)}-{int(hi)}"
        if n < min_n:
            out[key] = {"n": n}
            continue
        m = compute_metrics_np(yt[mask], yp[mask])
        m["n"] = n
        out[key] = m
    return out


def threshold_metrics(y_reference, y_prediction, threshold=100.0):
    reference = np.asarray(y_reference, dtype=float).ravel()
    prediction = np.asarray(y_prediction, dtype=float).ravel()
    if reference.size == 0 or reference.shape != prediction.shape:
        raise ValueError("Reference and prediction arrays must be paired.")
    if not (np.isfinite(reference).all() and np.isfinite(prediction).all()):
        raise ValueError("Metric inputs contain non-finite values.")
    reference_positive = reference > threshold
    prediction_positive = prediction > threshold
    tp = int(np.sum(reference_positive & prediction_positive))
    fp = int(np.sum(~reference_positive & prediction_positive))
    tn = int(np.sum(~reference_positive & ~prediction_positive))
    fn = int(np.sum(reference_positive & ~prediction_positive))

    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator > 0 else float("nan")

    return {
        "N": int(reference.size),
        "Reference positives": int(reference_positive.sum()),
        "Predicted positives": int(prediction_positive.sum()),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Precision": ratio(tp, tp + fp),
        "Recall": ratio(tp, tp + fn),
        "F1": ratio(2 * tp, 2 * tp + fp + fn),
    }


def load_best_model(input_size, n_targets, hidden_layers, path, device="cpu"):
    model = RegressNN(input_size, list(hidden_layers), n_targets).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.eval()


def predict_loader(model, X_tensor, Y_tensor, batch_size=4096):
    device = next(model.parameters()).device
    loader = DataLoader(
        Dataset(X_tensor, Y_tensor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    predictions, references = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            predictions.append(model(xb).detach().cpu())
            references.append(yb.detach().cpu())
    return torch.cat(predictions), torch.cat(references)


def denorm_if_needed(y_true_t, y_pred_t, cfg):
    if cfg.normalization_enable:
        params = load_norm_params(cfg.norm_params_path)
        if params is None:
            raise FileNotFoundError(cfg.norm_params_path)
        return (
            invert_minmax_np(torch_to_np(y_true_t), params),
            np.maximum(invert_minmax_np(torch_to_np(y_pred_t), params), 0.0),
        )
    return torch_to_np(y_true_t), torch_to_np(y_pred_t)


def evaluate_checkpoint(checkpoint_path, hidden_layers, tensors, cfg):
    x, y = tensors["train"]
    model = load_best_model(x.shape[1], y.shape[1], hidden_layers, checkpoint_path, cfg.device)
    predictions, metrics, by_range = {}, {}, {}
    for name, (features, targets) in tensors.items():
        prediction, reference = predict_loader(model, features, targets)
        true, pred = denorm_if_needed(reference, prediction, cfg)
        predictions[name] = (true, pred)
        metrics[name] = compute_metrics_np(true, pred)
        by_range[name] = metrics_by_ranges(true, pred)
    return {"metrics": metrics, "preds": predictions, "metrics_by_range": by_range}


def regression_metrics(reference, prediction):
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if reference.size == 0 or reference.shape != prediction.shape:
        raise ValueError("Metrics require non-empty paired arrays of equal length.")
    if not np.isfinite(reference).all() or not np.isfinite(prediction).all():
        raise ValueError("Non-finite values in metric inputs.")
    error = prediction - reference
    sst = np.sum((reference - reference.mean()) ** 2)
    return {
        "n": int(reference.size),
        "rmse_fnu": float(np.sqrt(np.mean(error**2))),
        "mae_fnu": float(np.mean(np.abs(error))),
        "bias_fnu": float(np.mean(error)),
        "r2": float(1 - np.sum(error**2) / sst) if sst > 0 else None,
    }
