#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AquaCubeAI model-only inference on OpenVINO devices.

This example does not generate a land/sea/cloud mask. Operational use requires
preliminary screening, as discussed by Goudemant et al. (IEEE JSTARS, 2024):
https://doi.org/10.1109/JSTARS.2024.3382394

The example evaluates the dense turbidity model on source-valid scene pixels.
Source NoData handling is separate from land/sea/cloud screening.
Compatible with the legacy OpenVINO 2020.3 Python 3.6 runtime.
"""
import argparse
import csv
import glob
import json
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from tqdm import tqdm


def load_model(xml_path, device):
    """Return a callable for one fixed [1,8,20,20] input tensor."""
    xml_path = Path(xml_path)
    bin_path = xml_path.with_suffix(".bin")
    for path in (xml_path, bin_path):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    try:
        from openvino.runtime import Core
    except ImportError:
        Core = None
    if Core is not None:
        core = Core()
        network = core.read_model(str(xml_path), str(bin_path))
        compiled = core.compile_model(network, device)
        input_port, output_port = compiled.input(0), compiled.output(0)
        shape = tuple(int(value) for value in input_port.shape)

        def infer(tensor):
            return np.asarray(compiled({input_port: tensor})[output_port])

        return infer, shape, "runtime"

    from openvino.inference_engine import IECore

    core = IECore()
    network = core.read_network(model=str(xml_path), weights=str(bin_path))
    if hasattr(network, "input_info"):
        input_name = next(iter(network.input_info))
        shape = tuple(network.input_info[input_name].input_data.shape)
    else:
        input_name = next(iter(network.inputs))
        shape = tuple(network.inputs[input_name].shape)
    output_name = next(iter(network.outputs))
    compiled = core.load_network(network=network, device_name=device)

    def infer(tensor):
        return np.asarray(compiled.infer(inputs={input_name: tensor})[output_name])

    return infer, shape, "ie"


def load_target_scaling(path):
    with np.load(str(path)) as values:
        minimum = np.asarray(values["min"], dtype=np.float32).reshape(-1)
        denominator = np.asarray(
            values["denom"] if "denom" in values else values["max"] - values["min"],
            dtype=np.float32,
        ).reshape(-1)
    if minimum.size != 1 or denominator.size != 1:
        raise ValueError("Expected one turbidity target in the scaling file.")
    denominator = np.where(denominator == 0, 1.0, denominator)
    if not np.isfinite(minimum).all() or not np.isfinite(denominator).all():
        raise ValueError("Non-finite target scaling parameters.")
    return minimum[0], denominator[0]


def resolve_image_paths(config, config_dir):
    paths = []
    for value in config.get("image_paths", []):
        path = Path(value)
        paths.append(str(path if path.is_absolute() else (config_dir / path).resolve()))
    if config.get("image_glob"):
        pattern = Path(config["image_glob"])
        pattern = str(pattern if pattern.is_absolute() else config_dir / pattern)
        paths.extend(glob.glob(pattern))
    return sorted(set(paths))


def predict_scene(image_path, infer, minimum, denominator, threshold):
    """Apply the model to native, non-overlapping 20 x 20 windows.

    Boundary windows are zero-padded to the fixed model input size and cropped
    back to their original extent. No land/sea/cloud selection is performed.
    """
    start = time.perf_counter()
    patch_size = 20
    model_patches, infer_seconds = 0, 0.0
    with rasterio.open(str(image_path)) as source:
        if source.count != 8:
            raise ValueError("Expected eight bands in the original simulator order.")
        height, width = source.height, source.width
        turbidity = np.full((height, width), -9999.0, dtype=np.float32)
        anomaly = np.full((height, width), 255, dtype=np.uint8)
        total = ((height + 19) // 20) * ((width + 19) // 20)
        windows = (
            (row, col)
            for row in range(0, height, patch_size)
            for col in range(0, width, patch_size)
        )
        for row, col in tqdm(windows, total=total, desc=Path(image_path).name):
            h, w = min(patch_size, height - row), min(patch_size, width - col)
            raw = source.read(window=Window(col, row, w, h), masked=True, out_dtype="float32")
            values = raw.filled(0.0)
            valid = ~np.ma.getmaskarray(raw).any(axis=0) & np.isfinite(values).all(axis=0)
            if not valid.any():
                continue
            values[:, ~valid] = 0.0
            tensor = np.zeros((1, 8, patch_size, patch_size), dtype=np.float32)
            tensor[0, :, :h, :w] = values
            inference_start = time.perf_counter()
            output = np.asarray(infer(tensor), dtype=np.float32)
            infer_seconds += time.perf_counter() - inference_start
            if output.shape != (1, 1, patch_size, patch_size):
                raise ValueError("Unexpected model output shape: {}".format(output.shape))
            prediction = np.maximum(output[0, 0, :h, :w] * denominator + minimum, 0.0)
            if not np.isfinite(prediction[valid]).all():
                raise ValueError("Non-finite model predictions.")
            turbidity[row : row + h, col : col + w] = np.where(valid, prediction, -9999.0)
            anomaly[row : row + h, col : col + w] = np.where(
                valid, prediction > threshold, 255
            ).astype(np.uint8)
            model_patches += 1
        profile = {
            "driver": "GTiff",
            "width": width,
            "height": height,
            "count": 1,
            "crs": source.crs,
            "transform": source.transform,
            "compress": "deflate",
        }
    valid_values = turbidity[turbidity != -9999.0]
    summary = {
        "image_name": Path(image_path).name,
        "input_patches": total,
        "model_patches": model_patches,
        "source_valid_pixels": int(valid_values.size),
        "mean_predicted_fnu": (
            float(valid_values.mean(dtype=np.float64)) if valid_values.size else float("nan")
        ),
        "max_predicted_fnu": float(valid_values.max()) if valid_values.size else float("nan"),
        "threshold_exceedance_pixels": int(np.sum(anomaly == 1)),
        "model_inference_seconds": infer_seconds,
        "scene_compute_seconds": time.perf_counter() - start,
    }
    return turbidity, anomaly, profile, summary


def save_maps(turbidity, anomaly, profile, output_dir, stem):
    for key, array, nodata in (("turbidity", turbidity, -9999.0), ("anomaly", anomaly, 255)):
        directory = output_dir / key
        directory.mkdir(parents=True, exist_ok=True)
        settings = dict(profile, dtype=str(array.dtype), nodata=nodata)
        with rasterio.open(str(directory / (stem + "_" + key + ".tif")), "w", **settings) as dest:
            dest.write(array, 1)


def save_preview(image_path, turbidity, anomaly, output_path, rgb_indices, threshold):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    step = max(1, int(np.ceil(max(turbidity.shape) / 1200.0)))
    prediction = turbidity[::step, ::step]
    labels = anomaly[::step, ::step]
    height, width = prediction.shape
    with rasterio.open(str(image_path)) as source:
        rgb = source.read(
            [int(i) + 1 for i in rgb_indices],
            out_shape=(3, height, width),
            resampling=Resampling.bilinear,
        ).transpose(1, 2, 0)
    rgb = rgb.astype(np.float32)
    finite = rgb[np.isfinite(rgb)]
    low, high = np.percentile(finite, [2, 98]) if finite.size else (0.0, 1.0)
    rgb = np.clip((np.nan_to_num(rgb) - low) / (high - low if high > low else 1.0), 0.0, 1.0)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("Simulated PhiSat-2 scene")
    im = axes[1].imshow(np.ma.masked_equal(prediction, -9999.0), cmap="YlOrRd")
    axes[1].set_title("Predicted turbidity")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="TUR [FNU]")
    im = axes[2].imshow(
        np.ma.masked_equal(labels, 255),
        cmap=ListedColormap(["purple", "yellow"]),
        norm=BoundaryNorm([-0.5, 0.5, 1.5], 2),
    )
    axes[2].set_title("Threshold exceedance (> {:.0f} FNU)".format(threshold))
    fig.colorbar(im, ax=axes[2], ticks=[0, 1], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the AquaCubeAI turbidity model on OpenVINO devices."
    )
    parser.add_argument("--config", default=str(Path(__file__).with_name("config_myriad.json")))
    parser.add_argument("--load_path", required=True, help="OpenVINO IR .xml, with .bin alongside.")
    parser.add_argument("--device", default="MYRIAD")
    parser.add_argument(
        "--norm_params", default="", help="Override the target scaling file in the JSON."
    )
    parser.add_argument("--output_dir", default="", help="Override output_dir from the JSON.")
    parser.add_argument(
        "--limit", type=int, default=0, help="Maximum scenes to process; 0 means all."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print the model path, shape and OpenVINO API."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config.get("patch_size", 20)) != 20 or config.get("use_overlap", False):
        raise ValueError("This example uses native, non-overlapping 20 x 20 patches.")
    images = resolve_image_paths(config, config_dir)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise RuntimeError("No input scenes found. Check image_paths or image_glob.")
    output_dir = Path(args.output_dir or config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = (config_dir / output_dir).resolve()
    scale_path = Path(args.norm_params or config["norm_params_path"])
    if not scale_path.is_absolute():
        scale_path = (config_dir / scale_path).resolve()
    minimum, denominator = load_target_scaling(scale_path)
    infer, shape, api = load_model(Path(args.load_path).resolve(), args.device)
    if tuple(shape) != (1, 8, 20, 20):
        raise ValueError("Expected model input [1,8,20,20], got {}".format(shape))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "Device: {} | Scenes: {} | Model-only inference (no land/sea/cloud screening)".format(
            args.device, len(images)
        )
    )
    if args.verbose:
        print("Model: {} | Input: {} | OpenVINO API: {}".format(args.load_path, shape, api))
    rows = []
    threshold = float(config.get("anomaly_threshold_fnu", 100.0))
    start = time.perf_counter()
    for image in images:
        prediction, anomaly, profile, row = predict_scene(
            image, infer, minimum, denominator, threshold
        )
        stem = Path(image).stem
        save_maps(prediction, anomaly, profile, output_dir, stem)
        if config.get("save_preview", True):
            save_preview(
                image,
                prediction,
                anomaly,
                output_dir / "preview" / (stem + "_preview.png"),
                config.get("rgb_indices", [2, 1, 0]),
                threshold,
            )
        rows.append(row)
        print(
            "{} | Model patches: {} | Inference: {:.3f} s".format(
                Path(image).name, row["model_patches"], row["model_inference_seconds"]
            )
        )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Outputs: {} | Total elapsed: {:.3f} s".format(output_dir, time.perf_counter() - start))


if __name__ == "__main__":
    main()
