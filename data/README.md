# Dataset access and placement

The dataset is not distributed in this repository. It is available upon request
to the authors: [p.distasio@studenti.unisannio.it](mailto:p.distasio@studenti.unisannio.it).

After obtaining the data, either place the supplied directories here or set
`AQUACUBEAI_DATA_DIR` to their parent directory. Windows PowerShell example:

```powershell
$env:AQUACUBEAI_DATA_DIR = 'C:\path\to\AquaCubeAI\data'
jupyter lab
```

By default, the notebooks look for the supplied dataset in `AquaCubeAI/data/`.
They also recognize an organized sibling `../data/` containing `patches/`,
`references/` and `metadata/sources_20x20.json`. An explicit `AQUACUBEAI_DATA_DIR`
setting takes precedence and allows the data to be kept in any other directory.

Linux/macOS example:

```bash
export AQUACUBEAI_DATA_DIR=/path/to/AquaCubeAI/data
jupyter lab
```

Example organization:

```text
data/
  patches/<region>/<YYYY-MM-DD>/<scene_id>/
  references/<region>/<YYYY-MM-DD>/<scene_id>.csv
  scenes/<region>/<YYYY-MM-DD>/<scene_id>.tiff
  metadata/                  # optional acquisition catalogue and runnable configurations
    sources_20x20.json        # ordered development/evaluation sources, when supplied
    inference_20x20.json      # mapping configuration, when supplied
```

Use distinct scene identifiers for separate acquisitions, including those from
the same region and date. Patch TIFF filenames retain the coordinates needed
for matching. Acquisition metadata can retain original source identifiers.

Folder names and paths are configurable. The bundled JSON files are generic
templates. Notebook configurations supplied in the dataset's `metadata/` directory
are used when present. Otherwise, set the patch-directory/CSV pairs in
`configs/sources_20x20.json` and the mapping paths and labels in
`configs/inference_20x20.json` to match the supplied data.
See `configs/README.md` for configuration options.

Patch filenames encode UTM coordinates using `lat_<northing>_lon_<easting>`.
CSV columns are `time`, `latitude_utm`, `longitude_utm`, and `TUR` in FNU.
Matching uses the closest patch coordinates within 2.5 m and preserves CSV row order.

Native model inputs use eight bands in the stored order:

| Channel (zero-based) | Simulator source channel |
| --- | --- |
| 0 | B02 |
| 1 | B03 |
| 2 | B04 |
| 3 | PAN |
| 4 | B08 |
| 5 | B05 |
| 6 | B06 |
| 7 | B07 |

These names identify Sentinel-2 bands used by the simulator. PAN is synthesized
from B02, B03, B04, B05 and B06. Values are used as stored, cast to float32;
no additional input normalization is applied. Training uses the spatial mean
of each native 20 x 20 patch. Dense inference uses the per-pixel spectra.

Raw data, generated predictions and run logs are ignored by Git.
The supplied notebooks have cleared execution outputs.
