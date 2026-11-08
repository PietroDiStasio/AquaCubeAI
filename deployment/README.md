# OpenVINO device inference

This standalone example runs the AquaCubeAI turbidity model on native 20 x 20
patches and reconstructs scene-sized outputs. It supports the legacy OpenVINO
Inference Engine API used with Myriad 2, as well as the OpenVINO Runtime API.

For operational use, a land/sea/cloud mask should be implemented and validated
as an initial screening stage before turbidity inference, following Goudemant
et al., [*Onboard Anomaly Detection for Marine Environmental Protection*](https://doi.org/10.1109/JSTARS.2024.3382394),
IEEE JSTARS, vol. 17, pp. 7918–7931, 2024.

This model-only example does not implement land/sea/cloud screening. All
source-valid pixels are processed; source NoData handling is not a water/cloud
mask. The notebook workflow separately selects patches at valid CMEMS reference
locations.

Outputs are:

- georeferenced turbidity and threshold-exceedance TIFFs;
- one three-panel preview per scene (disable with `save_preview: false` in the JSON);
- `summary.csv` and a concise timing summary.

The preview shows RGB, predicted turbidity and TUR > 100 FNU exceedances.
No upper clipping is applied to predictions. Invalid source pixels are written
as NoData (-9999 for turbidity, 255 for threshold exceedances). Boundary patches
are zero-padded for the fixed model input and cropped to the original scene size.
The `--verbose` option prints model and OpenVINO interface details.

Use the previously validated `openvino/ubuntu18_dev:2020.3` runtime on a Linux
computer with a connected Movidius device. Run the following command in a terminal
on the Linux host, before entering Docker, with the repository root as the current
directory. Replace `/path/to/AquaCubeAI` below with the absolute path to your
cloned repository:

```bash
cd /path/to/AquaCubeAI
```

`$PWD` is the host terminal's current directory. The bind mount makes that directory
available at `/workspace/AquaCubeAI` inside the container; `-w` selects it as the
container's working directory. `/workspace/AquaCubeAI` is an internal container
path, not a directory that must already exist on the host.

If the supplied dataset is inside the repository's `data/` directory, use:

```bash
docker run -it --privileged --device /dev/bus/usb \
  -v /dev:/dev -v "$PWD":/workspace/AquaCubeAI \
  -w /workspace/AquaCubeAI --user root \
  openvino/ubuntu18_dev:2020.3
```

If the dataset is stored outside the repository, use this version instead.
Replace `/path/to/data` with the absolute path to that dataset directory. The
second bind mount exposes the data inside the container without copying it:

```bash
docker run -it --privileged --device /dev/bus/usb \
  -v /dev:/dev \
  -v "$PWD":/workspace/AquaCubeAI \
  -v "/path/to/data":/workspace/AquaCubeAI/data:ro \
  -w /workspace/AquaCubeAI --user root \
  openvino/ubuntu18_dev:2020.3
```

The repository mount is writable, so files saved under `outputs/` in the container
appear under the repository's `outputs/` on the host. The shared-data mount is
read-only.

Inside the container:

```bash
pip3 install "numpy==1.16.2" "rasterio==1.2.10" tqdm matplotlib
python3 deployment/run_turbidity_myriad.py \
  --config deployment/config_myriad.json \
  --load_path models/myriad_fp16/model_turbidity_area.xml \
  --norm_params /workspace/AquaCubeAI/models/norm_params_tur.npz \
  --device MYRIAD \
  --output_dir /workspace/AquaCubeAI/outputs/deployment/myriad \
  --limit 1
```

The JSON uses a generic example path, `data/scenes/region_name/YYYY-MM-DD/scene_001.tiff`.
Set `image_paths` to the supplied scene before running the command. Image paths in the JSON
are resolved relative to the configuration file. For an OpenVINO CPU run, use
`--device CPU` and a separate output directory.

The model input is `[1,8,20,20]`, preserving the original band order and stored
input values cast to float32. The output is denormalized with the matching
training parameters. Source-invalid pixels are filled with zero for inference
and marked as NoData in saved outputs. The source GeoTIFF supplies the output
transform and CRS.

The included FP16 IR can be regenerated in the legacy runtime with:

```bash
python3 /opt/intel/openvino_2020.3.194/deployment_tools/model_optimizer/mo.py \
  --input_model models/model_turbidity_area.onnx \
  --data_type FP16 --input_shape "[1,8,20,20]" \
  --model_name model_turbidity_area \
  --output_dir outputs/ir_fp16 --progress --stream_output
```

Hardware execution must be checked on the MYRIAD-equipped computer. The public
example contains no land/sea/cloud screening implementation.
