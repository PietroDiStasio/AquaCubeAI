"""Paper-style diagnostics and the common-scale three-panel mapping figure."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FormatStrFormatter
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.transform import array_bounds
from pyproj import CRS


def read_background(path, fallback_crs):
    with rasterio.open(path) as src:
        if src.crs is None or src.count < 3:
            raise ValueError("Background TIFF requires a CRS and at least 3 bands.")
        rgb_bgr = src.read([1, 2, 3]).astype(np.float32)
        if CRS.from_user_input(src.crs).is_projected:
            data = rgb_bgr
            transform = src.transform
            plot_crs = src.crs.to_string()
        else:
            plot_crs = fallback_crs
            transform, width, height = calculate_default_transform(
                src.crs, plot_crs, src.width, src.height, *src.bounds
            )
            data = np.zeros((3, height, width), dtype=np.float32)
            for band in range(3):
                reproject(
                    source=rgb_bgr[band],
                    destination=data[band],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=plot_crs,
                    resampling=Resampling.bilinear,
                )
        if transform.b != 0 or transform.d != 0:
            raise ValueError("Rotated background TIFF requires rectification before plotting.")
        west, south, east, north = array_bounds(data.shape[1], data.shape[2], transform)
    rgb = np.transpose(data, (1, 2, 0))[..., ::-1].copy()
    rgb -= np.nanpercentile(rgb, 2)
    high = np.nanpercentile(rgb, 98)
    rgb = np.clip(rgb / (high if high > 0 else 1.0), 0, 1)
    bounds = (west, east, south, north)
    return (rgb, bounds, plot_crs)


def plot_learning_curves(train_history, val_history):
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.plot(train_history, label="Training")
    ax.plot(val_history, label="Validation")
    ax.set(xlabel="Epoch (zero-based)", ylabel="Huber loss", title="Training history")
    ax.legend()
    return fig


def plot_split_histograms(targets, indices):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), layout="constrained")
    for ax, name, idx in zip(axes, ("Training", "Validation", "Test"), indices):
        ax.hist(targets[idx].ravel(), bins=15, alpha=0.7)
        ax.set(title=name, xlabel="Turbidity [FNU]", ylabel="Count")
    return fig


def plot_regression(predictions):
    fig, axes = plt.subplots(
        1,
        len(predictions),
        figsize=(5 * len(predictions), 4.5),
        squeeze=False,
        layout="constrained",
    )
    for ax, (name, (reference, prediction)) in zip(axes.ravel(), predictions.items()):
        reference, prediction = np.asarray(reference).ravel(), np.asarray(prediction).ravel()
        ax.scatter(reference, prediction, s=5, alpha=0.25)
        lo = min(reference.min(), prediction.min())
        hi = max(reference.max(), prediction.max())
        ax.plot([lo, hi], [lo, hi], "k--", label="1:1")
        slope, intercept = np.polyfit(reference, prediction, 1)
        ax.plot(
            [lo, hi],
            slope * np.array([lo, hi]) + intercept,
            label=f"Fit: y={slope:.3f}x{intercept:+.3f}",
        )
        ax.set(title=name.title(), xlabel="Reference TUR [FNU]", ylabel="Predicted TUR [FNU]")
        ax.legend(fontsize=8)
    return fig


def plot_distributions(predictions):
    fig, axes = plt.subplots(
        1,
        len(predictions),
        figsize=(5 * len(predictions), 3.5),
        squeeze=False,
        layout="constrained",
    )
    for ax, (name, (reference, prediction)) in zip(axes.ravel(), predictions.items()):
        ax.hist(np.asarray(reference).ravel(), bins=35, alpha=0.5, label="Reference")
        ax.hist(np.asarray(prediction).ravel(), bins=35, alpha=0.5, label="Prediction")
        ax.set(title=name.title(), xlabel="Turbidity [FNU]", ylabel="Count")
        ax.legend()
    return fig


def plot_mapping(result, save_path=None, dpi=300, colour_limits=None):
    """Use a shared reference/prediction colour scale; raw predictions are unchanged.

    Retains the legacy Figure-10 visualization footprint of 300 m and lower-origin
    patch display. This visualization is not a georeferenced full-scene product.
    """
    cfg = result["config"]
    values = result["reference_fnu"]
    if colour_limits is None:
        ref_low, ref_high = np.percentile(values, [2.0, 98.0])
        pred_low, pred_high = np.percentile(result["predicted_maps_fnu"], [2.0, 98.0])
        low, high = min(ref_low, pred_low), max(ref_high, pred_high)
    else:
        low, high = colour_limits
    low = max(0.0, float(low))
    high = float(high) if high > low else low + 1e-3
    norm = Normalize(vmin=low, vmax=high, clip=False)
    ticks = np.linspace(low, high, 6)
    fig = plt.figure(figsize=(24.0, 8.8), layout="none")
    axes = [fig.add_axes([left, 0.08, 0.265, 0.78]) for left in (0.01, 0.335, 0.66)]
    west, east, south, north = result["bounds"]
    for ax in axes:
        ax.imshow(result["rgb"], extent=result["bounds"], origin="upper", interpolation="nearest")
        ax.set(xlim=(west, east), ylim=(south, north), aspect="equal")
        ax.axis("off")
    scatter = axes[0].scatter(
        result["x"],
        result["y"],
        c=values,
        s=24.0,
        cmap="YlOrRd",
        norm=norm,
        alpha=1.0,
        edgecolors="none",
        zorder=10,
    )
    anomaly_cmap = ListedColormap(["purple", "yellow"])
    anomaly_norm = BoundaryNorm([-0.5, 0.5, 1.5], anomaly_cmap.N)
    half = 0.5 * cfg["patch_size_m"]
    for x, y, pred, anomaly in zip(
        result["x"], result["y"], result["predicted_maps_fnu"], result["anomaly_maps"]
    ):
        extent = (x - half, x + half, y - half, y + half)
        axes[1].imshow(
            pred,
            extent=extent,
            origin="lower",
            cmap="YlOrRd",
            norm=norm,
            alpha=1.0,
            interpolation="nearest",
            zorder=10,
        )
        axes[2].imshow(
            anomaly,
            extent=extent,
            origin="lower",
            cmap=anomaly_cmap,
            norm=anomaly_norm,
            alpha=0.8,
            interpolation="nearest",
            zorder=10,
        )
    bars = [
        (scatter, "CMEMS TUR [FNU]", ticks, False),
        (ScalarMappable(norm=norm, cmap="YlOrRd"), "Pred TUR [FNU]", ticks, False),
        (
            ScalarMappable(norm=anomaly_norm, cmap=anomaly_cmap),
            f"Anom (> {cfg['anomaly_threshold_fnu']:.0f})",
            [0, 1],
            True,
        ),
    ]
    for ax, (mappable, label, tick_values, discrete), title in zip(
        axes, bars, ("CMEMS TUR Values", "Predicted TUR Values", "Anomaly Map")
    ):
        ax.apply_aspect()
        pos = ax.get_position()
        cax = fig.add_axes([pos.x1 + 0.008, pos.y0, 0.0085, pos.height])
        cb = fig.colorbar(mappable, cax=cax, extend="neither" if discrete else "both")
        cb.set_ticks(tick_values)
        if not discrete:
            cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        cb.set_label(label, labelpad=6, fontsize=15)
        cb.ax.tick_params(length=3, pad=2.5, labelsize=13)
        ax.set_title(title, pad=10, fontsize=18)
    fig.suptitle(f"Global Mapping - {cfg['region']} {cfg['date']}", fontsize=22, y=0.95)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig
