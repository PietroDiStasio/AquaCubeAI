"""Dense inference at the supplied CMEMS reference locations."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pyproj import Transformer
from .config import ROOT, data_directory, set_seed, configuration_file
from .data import get_dataset_from_csv, load_norm_params
from .models import load_area_model
from .evaluation import regression_metrics
from .plotting import read_background


def predict_map(model, patch, device, low, denominator, patch_hw=20):
    array = np.asarray(patch, dtype=np.float32)
    if array.ndim != 3 or array.shape[:2] != (patch_hw, patch_hw):
        raise ValueError(f"Unexpected input patch shape: {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Non-finite input features in a selected patch.")
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
    with torch.no_grad():
        output = model(tensor.unsqueeze(0).to(device)).detach().cpu().numpy()
    expected_shape = (1, 1, array.shape[0], array.shape[1])
    if output.shape != expected_shape:
        raise ValueError(f"Unexpected model output shape: {output.shape}")
    prediction = np.maximum(output[0, 0] * denominator + low, 0.0)
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite predictions; evaluation stopped.")
    return prediction.astype(np.float32)


def run_reference_inference(config_path=None, data_root=None, device="cpu", limit=None):
    """Return predictions and plotting data; no files are written here.

    `limit` is for local smoke checks only. Leave it unset for the complete scene.
    """
    set_seed(5)
    config = json.loads(
        (
            Path(config_path)
            if config_path is not None
            else configuration_file("inference_20x20.json")
        ).read_text(encoding="utf-8")
    )
    root = Path(data_root) if data_root is not None else data_directory()
    paths = {key: root / config[key] for key in ("scene", "patch_dir", "csv")}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not included. Set AQUACUBEAI_DATA_DIR. Missing: {path}"
            )
    rgb, bounds, crs = read_background(paths["scene"], "EPSG:3857")
    patches, targets, metadata = get_dataset_from_csv(
        str(paths["patch_dir"]), str(paths["csv"]), coord_eps=2.5, verbose=False, return_meta=True
    )
    targets = np.asarray(targets, dtype=np.float32).reshape(-1)
    xs = np.asarray([float(item["longitude_utm"]) for item in metadata])
    ys = np.asarray([float(item["latitude_utm"]) for item in metadata])
    transformer = Transformer.from_crs(f"EPSG:{config['source_epsg']}", crs, always_xy=True)
    xp, yp = [np.asarray(values) for values in transformer.transform(xs, ys)]
    west, east, south, north = bounds
    valid = np.isfinite(targets) & (targets >= 0.0) & (targets < 140.0)
    inside = (
        np.isfinite(xp)
        & np.isfinite(yp)
        & (xp >= west)
        & (xp <= east)
        & (yp >= south)
        & (yp <= north)
    )
    indices = np.flatnonzero(valid & inside)
    if limit is not None:
        indices = indices[:limit]
    if not indices.size:
        raise ValueError("No valid reference targets inside the scene.")
    model_dir = ROOT / config["model_dir"]
    params = load_norm_params(model_dir / "norm_params_tur.npz")
    if params is None:
        raise FileNotFoundError(model_dir / "norm_params_tur.npz")
    model = load_area_model(model_dir, device)
    maps = np.stack(
        [
            predict_map(
                model,
                patches[idx],
                device,
                float(params["min"].ravel()[0]),
                float(params["denom"].ravel()[0]),
            )
            for idx in indices
        ]
    )
    means = maps.mean(axis=(1, 2), dtype=np.float64)
    return dict(
        config=config,
        scene=paths["scene"].stem,
        rgb=rgb,
        bounds=bounds,
        crs=crs,
        loader_indices=indices,
        reference_fnu=targets[indices],
        x=xp[indices],
        y=yp[indices],
        predicted_maps_fnu=maps,
        mean_dense_prediction_fnu=means,
        anomaly_maps=(maps > config["anomaly_threshold_fnu"]).astype(np.uint8),
        metrics=regression_metrics(targets[indices], means),
    )


def save_inference_outputs(result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / result["scene"]
    prefix.with_name(prefix.name + "_metrics.json").write_text(
        json.dumps(
            {
                "scene": result["scene"],
                "evaluation": "CMEMS vs mean dense prediction per reference patch",
                "bias_convention": "prediction - reference",
                "metrics": result["metrics"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            key: result[key]
            for key in ("loader_indices", "reference_fnu", "mean_dense_prediction_fnu", "x", "y")
        }
    ).to_csv(str(prefix) + "_paired_reference_predictions.csv", index=False)
    np.savez_compressed(
        str(prefix) + "_numerical_outputs.npz",
        **{
            key: result[key]
            for key in (
                "loader_indices",
                "reference_fnu",
                "mean_dense_prediction_fnu",
                "predicted_maps_fnu",
                "anomaly_maps",
                "x",
                "y",
            )
        },
    )
