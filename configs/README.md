# Configuration templates

The JSON files describe the expected structure using generic example identifiers:
`region_name`, `YYYY-MM-DD` and `scene_001`. They contain no personal dataset paths.
The repository runs independently; no separate personal-configuration folder is required.

| File | Settings |
| --- | --- |
| `sources_20x20.json` | Ordered patch-directory/CSV pairs for development and evaluation |
| `inference_20x20.json` | Patches, reference CSV, scene, source EPSG and plot labels |
| `../deployment/config_myriad.json` | Scenes and output options for device inference |

Before execution, either customize these templates to the data received from the
authors or use the corresponding JSON files supplied in the dataset's metadata/
directory. Dataset metadata take precedence over public templates. Set
`AQUACUBEAI_CONFIG_DIR` to explicitly select another configuration directory.

Notebook data paths are relative to the selected dataset directory. By default,
this is the repository's data/ directory. An organized sibling ../data/ is also
recognized when it contains patches/, references/ and metadata/sources_20x20.json.
Set `AQUACUBEAI_DATA_DIR` to select any other location explicitly. Data inside
the repository take precedence over automatic sibling discovery.

Model and notebook output paths are relative to the repository root. Set the
mapping template's example EPSG to your reference-coordinate CRS and replace
the study-area and date labels. Preserve source order when reproducing a run.

Device JSON paths are relative to that JSON file. Select a customized device
configuration with --config; the device runner does not discover dataset metadata
automatically.
