"""CSV matching, patch descriptors, target scaling and the original group split."""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Sequence
import numpy as np
import pandas as pd
import rasterio
import torch
from .config import Config, ROOT, data_directory, configuration_file

PHISAT2_N_BANDS = 8
S2_N_BANDS = 13


def _read_multiband_raster(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read()
    arr = np.moveaxis(arr, 0, -1)
    return arr


def read_PhiSat2_image(path: str, normalize: bool = False, norm_band: bool = False) -> np.ndarray:
    img = _read_multiband_raster(path)
    if normalize:
        img = _normalize_image(img, norm_band=norm_band)
    return img[:, :, :PHISAT2_N_BANDS]


def read_S2_image(path: str, normalize: bool = False, norm_band: bool = False) -> np.ndarray:
    img = _read_multiband_raster(path)
    if normalize:
        img = _normalize_image(img, norm_band=norm_band)
    return img[:, :, :S2_N_BANDS]


def reject_outliers(img: np.ndarray, perc: float = 98) -> np.ndarray:
    p = np.percentile(img, perc)
    return np.clip(img, 0.0, p)


def _normalize_image(img: np.ndarray, norm_band: bool = False) -> np.ndarray:
    img = reject_outliers(img)
    if norm_band:
        for b in range(img.shape[-1]):
            band = img[:, :, b]
            bmin = np.nanmin(band)
            bmax = np.nanmax(band)
            img[:, :, b] = (band - bmin) / (bmax - bmin + 1e-06)
    else:
        gmin = np.nanmin(img)
        gmax = np.nanmax(img)
        img = (img - gmin) / (gmax - gmin + 1e-06)
    if np.isnan(img).any():
        img[np.isnan(img)] = np.nanmean(img)
    return img


def parse_phisat2_coords_from_filename(path: str) -> Tuple[float, float]:
    base = os.path.basename(path)
    base = os.path.splitext(base)[0]
    parts = base.split("_")
    try:
        lat_idx = parts.index("lat")
        lon_idx = parts.index("lon")
    except ValueError as e:
        raise ValueError(f"Cannot find 'lat'/'lon' tokens in filename: {base}") from e
    utm_y = float(parts[lat_idx + 1])
    utm_x = float(parts[lon_idx + 1])
    return (utm_y, utm_x)


def get_targets_from_csv(csv_path: str, pollutant_col: str = "TUR") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = ["time", "latitude_utm", "longitude_utm", pollutant_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in CSV {csv_path}. Found: {list(df.columns)}")
    return df


def get_dataset_from_csv(
    root_path: str,
    csv_path: str,
    satellite_sensor: str = "PhiSat-2",
    pollutant_col: str = "TUR",
    coord_eps: float = 2.5,
    verbose: bool = True,
    return_meta: bool = False,
) -> Tuple[List[np.ndarray], np.ndarray, Optional[List[Dict[str, Any]]]]:
    df = get_targets_from_csv(csv_path, pollutant_col=pollutant_col)
    sensor_images = []
    for root, _, files in os.walk(root_path):
        for f in files:
            if satellite_sensor == "PhiSat-2" and f.endswith(".tiff"):
                sensor_images.append(os.path.join(root, f))
            elif satellite_sensor == "S2" and f.endswith(".tif"):
                sensor_images.append(os.path.join(root, f))
    if not sensor_images:
        raise ValueError(f"No images found in {root_path} for sensor {satellite_sensor}.")
    if verbose:
        print(f"Number of images {satellite_sensor} found in '{root_path}': {len(sensor_images)}")
        print("Matching CSV rows to patch images...")
    img_infos = []
    for path in sensor_images:
        utm_y, utm_x = parse_phisat2_coords_from_filename(path)
        img_infos.append((path, utm_y, utm_x))
    img_coords = np.array([[utm_y, utm_x] for _, utm_y, utm_x in img_infos], dtype=np.float64)
    coords_csv = df[["latitude_utm", "longitude_utm"]].values.astype(np.float64)
    y_csv = df[pollutant_col].values.astype(np.float64)
    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    meta_list: List[Dict[str, Any]] = []
    time_vals = df["time"].values
    for i in range(coords_csv.shape[0]):
        lat_utm, lon_utm = coords_csv[i]
        target_val = float(y_csv[i])
        dists = np.sqrt((img_coords[:, 0] - lat_utm) ** 2 + (img_coords[:, 1] - lon_utm) ** 2)
        idx_min = int(np.argmin(dists))
        dist_min = float(dists[idx_min])
        if dist_min <= coord_eps:
            img_path = img_infos[idx_min][0]
            if satellite_sensor == "PhiSat-2":
                img = read_PhiSat2_image(img_path, normalize=False, norm_band=False)
            else:
                img = read_S2_image(img_path, normalize=False, norm_band=False)
            X_list.append(img)
            y_list.append(target_val)
            if return_meta:
                meta_list.append(
                    {
                        "time": time_vals[i],
                        "latitude_utm": float(lat_utm),
                        "longitude_utm": float(lon_utm),
                        "img_path": img_path,
                        "csv": os.path.basename(csv_path),
                        "dist_m": dist_min,
                    }
                )
    if not X_list:
        raise ValueError("No CSV/image matches. Check coordinates, filenames and coord_eps.")
    y_array = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    if verbose:
        print(f"Matching complete: {len(X_list)} samples (image + {pollutant_col}).")
    if return_meta:
        return (X_list, y_array, meta_list)
    return (X_list, y_array)


class Dataset(torch.utils.data.Dataset):
    """Simple PyTorch Dataset: returns X and y as provided."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        return (X, y)


def geo_cell_id_utm(y_utm: float, x_utm: float, cell_m: float) -> str:
    y_bin = int(np.floor(float(y_utm) / cell_m))
    x_bin = int(np.floor(float(x_utm) / cell_m))
    return f"{y_bin}_{x_bin}"


def best_group_partition_stratified(
    group_list: List[str],
    group_sizes: Dict[str, int],
    group_sums: Dict[str, float],
    targets: Tuple[int, int, int],
    global_mean: float,
    n_restarts: int,
    seed: int,
    mean_weight: float,
    size_tolerance: float,
):
    rng = np.random.default_rng(seed)
    tgts = np.array(targets, dtype=float)
    max_sizes = tgts * (1.0 + size_tolerance)
    best = None
    best_score = float("inf")
    for _ in range(n_restarts):
        perm = rng.permutation(group_list)
        sets = [[], [], []]
        sizes = np.array([0.0, 0.0, 0.0])
        sums = np.array([0.0, 0.0, 0.0])
        for gid in perm:
            sz = float(group_sizes[gid])
            sm = float(group_sums[gid])
            feasible = [k for k in range(3) if sizes[k] + sz <= max_sizes[k]]
            if not feasible:
                feasible = [0, 1, 2]
            cand = []
            for k in feasible:
                new_sizes = sizes.copy()
                new_sizes[k] += sz
                new_sums = sums.copy()
                new_sums[k] += sm
                size_err = np.sum(np.abs(new_sizes - tgts) / np.maximum(1.0, tgts))
                mus = np.divide(new_sums, np.maximum(1.0, new_sizes))
                mean_err = np.sum(np.abs(mus - global_mean) / max(1e-06, global_mean))
                cand.append((size_err + mean_weight * mean_err, k))
            _, kbest = min(cand, key=lambda t: t[0])
            sets[kbest].append(gid)
            sizes[kbest] += sz
            sums[kbest] += sm
        final_mus = np.divide(sums, np.maximum(1.0, sizes))
        size_err = np.sum(np.abs(sizes - tgts) / np.maximum(1.0, tgts))
        mean_err = np.sum(np.abs(final_mus - global_mean) / max(1e-06, global_mean))
        total = size_err + mean_weight * mean_err
        if any((len(s) == 0 for s in sets)):
            total += 1000000.0
        if total < best_score:
            best_score = total
            best = (sets[0], sets[1], sets[2], int(sizes[0]), int(sizes[1]), int(sizes[2]))
    return (best, best_score)


def build_group_split(x: np.ndarray, y: np.ndarray, groups: np.ndarray, cfg: Config):
    N = len(x)
    n_train = int(round(cfg.train_perc * N))
    n_val = int(round(cfg.val_perc * N))
    n_test = N - n_train - n_val
    global_mean = float(np.mean(y))
    idx_by_group = defaultdict(list)
    group_sums = defaultdict(float)
    y_flat = np.asarray(y, dtype=np.float32).reshape(-1)
    for i, gid in enumerate(groups):
        idx_by_group[gid].append(i)
        group_sums[gid] += float(y_flat[i].item())
    group_list = list(idx_by_group.keys())
    group_sizes = {gid: len(idxs) for gid, idxs in idx_by_group.items()}
    print(f"Total groups: {len(group_list)}")
    print("Searching for a group-aware partition...")
    best, score = best_group_partition_stratified(
        group_list=group_list,
        group_sizes=group_sizes,
        group_sums=group_sums,
        targets=(n_train, n_val, n_test),
        global_mean=global_mean,
        n_restarts=cfg.split_restarts,
        seed=cfg.split_seed,
        mean_weight=cfg.split_mean_weight,
        size_tolerance=cfg.split_size_tolerance,
    )
    train_groups, val_groups, test_groups, _, _, _ = best
    train_idx = np.concatenate([idx_by_group[g] for g in train_groups]).astype(int)
    val_idx = np.concatenate([idx_by_group[g] for g in val_groups]).astype(int)
    test_idx = np.concatenate([idx_by_group[g] for g in test_groups]).astype(int)
    rng = np.random.default_rng(cfg.split_seed)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    gtr = set(groups[train_idx].tolist())
    gva = set(groups[val_idx].tolist())
    gte = set(groups[test_idx].tolist())
    assert len(gtr & gva) == 0, "Overlapping train/validation groups."
    assert len(gtr & gte) == 0, "Overlapping train/test groups."
    assert len(gva & gte) == 0, "Overlapping validation/test groups."
    return (train_idx, val_idx, test_idx, score)


def patch_mean_features(X_obj: np.ndarray) -> np.ndarray:
    """Match the contiguous float32 reduction used by the original object arrays."""
    feats = []
    for img in X_obj:
        # Original loading materialized HWC object arrays in C order. Preserve
        # that reduction order when processing Rasterio's strided HWC views.
        arr = np.asarray(img, dtype=np.float32, order="C")
        feats.append(arr.mean(axis=(0, 1)))
    return np.asarray(feats, dtype=np.float32)


def fit_minmax_np(y_train: np.ndarray) -> Dict[str, np.ndarray]:
    y_min = np.min(y_train, axis=0)
    y_max = np.max(y_train, axis=0)
    denom = y_max - y_min
    denom = np.where(denom == 0, 1.0, denom)
    return {
        "min": y_min.astype(np.float32),
        "max": y_max.astype(np.float32),
        "denom": denom.astype(np.float32),
    }


def apply_minmax_np(y: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    return (y - params["min"]) / params["denom"]


def invert_minmax_np(y_norm: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    return y_norm * params["denom"] + params["min"]


def save_norm_params(path: str, params: Dict[str, np.ndarray]) -> None:
    np.savez(path, min=params["min"], max=params["max"], denom=params["denom"])


def load_norm_params(path: str) -> Optional[Dict[str, np.ndarray]]:
    if not os.path.exists(path):
        return None
    d = np.load(path)
    denom = d["denom"] if "denom" in d else d["max"] - d["min"]
    denom = np.asarray(denom, dtype=np.float32)
    denom[denom == 0] = 1.0
    return {
        "min": np.asarray(d["min"], dtype=np.float32),
        "max": np.asarray(d["max"], dtype=np.float32),
        "denom": denom,
    }


def clean_tensor(t: torch.Tensor) -> torch.Tensor:
    if torch.isnan(t).any() or torch.isinf(t).any():
        print("[WARNING] Replacing NaN/Inf tensor values with zero.")
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    return t


def source_pairs(manifest=None, data_root=None, subset="training"):
    """Resolve dataset identifiers without changing source order or multiplicity."""
    manifest = Path(manifest) if manifest is not None else configuration_file("sources_20x20.json")
    root = Path(data_root) if data_root is not None else data_directory()
    entries = json.loads(manifest.read_text(encoding="utf-8"))[subset]
    if isinstance(entries, dict):
        entries = [entries]
    pairs = [(str(root / item["patch_dir"]), str(root / item["csv"])) for item in entries]
    missing = [p for pair in pairs for p in pair if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset files are not included. Set AQUACUBEAI_DATA_DIR to the supplied data root. Missing: "
            + missing[0]
        )
    return pairs


def load_feature_dataset(sources, cfg):
    """Compute the unchanged float32 patch mean one acquisition at a time.

    This avoids retaining all image patches as Python object arrays. Sample order,
    target filtering, matching and group identifiers follow the original pipeline.
    """
    features, targets, group_ids = [], [], []
    for patch_dir, csv_path in sources:
        patches, y, meta = get_dataset_from_csv(
            patch_dir, csv_path, coord_eps=2.5, verbose=cfg.verbose, return_meta=True
        )
        for patch in patches:
            if np.asarray(patch).shape != (cfg.patch_hw, cfg.patch_hw, 8):
                raise ValueError("The paper workflow requires native 20 x 20 x 8 patches.")
        features.append(patch_mean_features(patches))
        targets.append(np.asarray(y, dtype=np.float32).reshape(-1, 1))
        group_ids.extend(
            f"{Path(csv_path).name}|{geo_cell_id_utm(m['latitude_utm'], m['longitude_utm'], cfg.cell_m)}"
            for m in meta
        )
    x = np.concatenate(features).astype(np.float32)
    y = np.concatenate(targets).astype(np.float32)
    groups = np.asarray(group_ids, dtype=object)
    keep = np.isfinite(y.ravel()) & (y.ravel() >= cfg.tur_min) & (y.ravel() < cfg.tur_max)
    return x[keep], y[keep], groups[keep]


def prepare_splits(features, targets, indices, cfg):
    """Fit optional PCA and target scaling on training data only."""
    if cfg.augment_enable:
        raise NotImplementedError("Augmentation is not part of the paper configuration.")
    xs = [features[idx] for idx in indices]
    ys = [targets[idx] for idx in indices]
    pca = None
    if cfg.pca_enable:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=cfg.pca_components, random_state=cfg.seed)
        xs = [
            pca.fit_transform(xs[0]).astype(np.float32),
            pca.transform(xs[1]).astype(np.float32),
            pca.transform(xs[2]).astype(np.float32),
        ]
    if cfg.normalization_enable:
        params = fit_minmax_np(ys[0])
        Path(cfg.norm_params_path).parent.mkdir(parents=True, exist_ok=True)
        save_norm_params(cfg.norm_params_path, params)
        ys = [apply_minmax_np(y, params) for y in ys]
    tensors = {
        name: (
            clean_tensor(torch.tensor(x, dtype=torch.float32)),
            clean_tensor(torch.tensor(y, dtype=torch.float32)),
        )
        for name, x, y in zip(("train", "val", "test"), xs, ys)
    }
    return tensors, pca
