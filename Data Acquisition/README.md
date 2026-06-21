# Flood Data Generator — Sen1Floods11-Compatible

A self-hosted tool that pairs Sentinel-1 SAR and Sentinel-2 optical imagery from Google Earth Engine into fixed-size GeoTIFF chips, for any area and date you choose, in the same chip size, resolution, band layout, and raw pixel format used by the [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) dataset. The goal is to let you generate additional training/evaluation imagery for flood detection deep learning models, beyond the 11 flood events covered by the original published dataset.

This is a data-generation tool only: it exports raw, unlabeled SAR/optical imagery. It does not produce flood or water labels, and it is not the official Sen1Floods11 dataset.

## Screenshots

> Drop your screenshots into `docs/images/` using the filenames below and they will render automatically.

| | |
|---|---|
| Drawing an AOI | ![Draw AOI](docs/images/aoi-draw.png) |
| Paired-date discovery | ![Paired dates](docs/images/paired-dates.png) |
| Download with pause/resume/stop | ![Download progress](docs/images/download-progress.png) |
| Tile visualization (SAR + optical) | ![Visualization panel](docs/images/visualization.png) |
| Download history | ![History](docs/images/history.png) |

## Why this exists

Sen1Floods11 (Bonafilia et al., 2020) is a fixed dataset of 4,831 hand-labeled and weakly-labeled 512x512 chips covering 11 historical flood events. It is widely used to train and benchmark SAR/optical flood segmentation models, but it is frozen in time and location. This pipeline lets you point at a new area and a new date — a recent flood, a region not covered by the original dataset, a holdout region for testing generalization — and pull down imagery with matching characteristics:

- Same chip size: 512x512 pixels at 10 m resolution (a 5.12 km square tile).
- Same Sentinel-1 format: raw float32 dB backscatter, bands VH then VV, no normalization.
- Same Sentinel-2 format: raw int16 L1C TOA digital numbers, all 13 bands in official order (B1–B9, B8A, B10–B12), no scaling.
- Same projection convention: chips are exported in the local UTM zone, so pixels are uniformly 10 m on the ground.

## How it works

1. **Draw an AOI.** The frontend is a Leaflet map with Leaflet.Draw; you draw a polygon over the area you want imagery for.
2. **Find paired dates.** The frontend sends the polygon (and an optional date range) to `POST /api/process_polygon`. The backend reverse-geocodes the AOI's centroid for a location label, computes the area and how many chips it would produce, then queries Earth Engine for every Sentinel-2 scene under a cloud-cover threshold and every dual-pol Sentinel-1 IW scene in the range. A Sentinel-2 date is kept as "paired" only if a Sentinel-1 acquisition exists within a few days of it — both sensors are required to build a usable chip pair.
3. **Select dates and confirm.** You check one or more paired dates; the UI shows an estimate of total chips, storage, and time before you confirm.
4. **Tiled background export.** For each selected date, `POST /api/download_tiles` queues a background job. The backend splits the AOI into a grid of fixed-size tiles in the local UTM projection, builds a Sentinel-1 mosaic and a Sentinel-2 mosaic for a date window around the target date, and exports each tile as a separate SAR GeoTIFF and optical GeoTIFF.
5. **Progress, pause, resume, stop.** The frontend polls `GET /api/status/{task_id}` and can call `POST /api/task_control/{task_id}` to pause, resume, or stop the export between tiles.
6. **Visualize what you downloaded.** `GET /api/visualize/{location}/{date}/{tile_idx}` reads a saved tile pair and returns contrast-stretched PNG previews (SAR VH, SAR VV, an SAR false-color composite, and optical true color) for quick inspection, without modifying the saved GeoTIFFs.
7. **Download history.** `GET /api/list_downloads` walks everything saved on disk so the sidebar can list past downloads and reopen them in the visualization panel.

## Output format

```
flood_data_exports/
└── <City>_<Country>/
    └── <YYYY-MM-DD>/
        ├── SAR_ALL_BANDS/
        │   ├── tile_0.tif        band 1 = VH, band 2 = VV, float32 dB, raw
        │   ├── tile_1.tif
        │   └── ...
        └── OPTICAL_ALL_BANDS/
            ├── tile_0.tif        bands B1..B9,B8A,B10..B12, int16 DN, raw
            ├── tile_1.tif
            └── ...
```

- `<City>_<Country>` comes from reverse-geocoding the AOI centroid (Nominatim); it falls back to `Custom_Region` if geocoding fails.
- Tile indices match the order of Earth Engine's `coveringGrid()` output over the AOI.
- Every chip is 512x512 pixels at 10 m/pixel, in the AOI's local UTM CRS.
- The optical true-color preview shown in the UI uses bands B4/B3/B2 (red/green/blue).

## Hyperparameters

All defined as module-level constants near the top of `main.py`.

| Name | Default | What it controls |
|---|---|---|
| `PROJECT_ID` | `"discord-reports-439616"` | GCP project used to initialize/bill Earth Engine. **Change this to your own project.** |
| `BASE_DIR` | `"flood_data_exports"` | Root folder where all exported tiles are written. |
| `PORT` | `8002` | Local port the FastAPI/uvicorn server listens on. |
| `S1_DB_MIN` / `S1_DB_MAX` | `-30` / `5` | Reference dB bounds for scaling SAR to a preview range. Defined but not currently wired into the preview code — see Implementation Notes. |
| `S1_DATE_WINDOW_DAYS` | `5` | ± day window used (a) to decide if a Sentinel-2 date is "paired" with Sentinel-1, and (b) to filter Sentinel-1 scenes when building the export mosaic. |
| `S2_DATE_WINDOW_DAYS` | `10` | ± day window used only when building the Sentinel-2 export mosaic for a chosen date. |
| `S2_MAX_CLOUD_PCT` | `40` | Maximum scene-level `CLOUDY_PIXEL_PERCENTAGE` allowed for a Sentinel-2 scene. Permissive on purpose, so flood-period scenes with some cloud aren't excluded. |
| `S2_REFLECTANCE_MAX` | `10000` | Reference reflectance ceiling for preview scaling. Defined but not currently used — see Implementation Notes. |
| `S2_CLOUD_BLUE_MAX` | `0.35` | Per-pixel cloud threshold on the Sentinel-2 blue band (B2/10000), used by `mask_s2_clouds()`. That function exists but is not currently called in the export path — see Implementation Notes. |
| `CHIP_SIZE_M` | `5120` | Tile edge length in meters. With `EXPORT_SCALE_M = 10`, gives exactly 512x512 px chips. |
| `EXPORT_SCALE_M` | `10` | Output pixel resolution in meters/pixel (native S1/S2 resolution). |
| `percentile_stretch(lo, hi)` | `2`, `98` | Percentile bounds for the adaptive contrast stretch used **only** when rendering preview PNGs in `/api/visualize`; never touches the saved GeoTIFFs. |

## Requirements

- Python 3.10+
- A Google Cloud project with the Earth Engine API enabled
- A Google account approved for Earth Engine access, with at least `roles/serviceusage.serviceUsageConsumer` on that project

```bash
pip install -r requirements.txt
```

## Setup

1. Create or choose a GCP project and enable the Earth Engine API for it.
2. Make sure your Google account has Earth Engine access (https://earthengine.google.com/) and at least `roles/serviceusage.serviceUsageConsumer` on the project (grant it from the IAM console).
3. Open `main.py` and set `PROJECT_ID` to your project's ID.
4. Install dependencies: `pip install -r requirements.txt`.
5. Run the server:

```bash
python main.py
```

On first run, `authenticate_gee()` will open a browser window for the standard Earth Engine OAuth flow. Once approved, credentials are cached under `~/.config/earthengine/`, so future runs initialize silently.

6. Open `http://127.0.0.1:8002` in a browser.

## Usage

1. Click **Draw**, trace a polygon over your area of interest, double-click to finish.
2. Set a date range and click **Find Paired Dates**.
3. Check one or more dates from the returned list.
4. Click **Download Selected**, review the estimated chip count / storage / time, and confirm.
5. Use **Pause**, **Resume**, or **Stop** under the progress bar if you need to control a running export. Stop cancels the rest of the queue; tiles already saved are kept.
6. Once a date finishes, find it under **Downloaded** in the sidebar and click **View** to inspect SAR VH, SAR VV, SAR false color, and optical true color for each tile.

## API reference

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Serves the frontend (`index.html`). |
| POST | `/api/process_polygon` | Given a GeoJSON feature (+ optional `start_date`/`end_date`), returns location name, paired dates, tile count, and area. |
| POST | `/api/download_tiles` | Given `{date, geometry}`, queues a background export and returns a `task_id`. |
| GET | `/api/status/{task_id}` | Returns the task's current status, progress, total, message, and control state. |
| POST | `/api/task_control/{task_id}` | Given `{"action": "pause" \| "resume" \| "stop"}`, controls a running export. |
| GET | `/api/visualize/{location}/{date}/{tile_idx}` | Returns base64 PNG previews (`vh`, `vv`, `false_color`, `rgb`) for one tile. |
| GET | `/api/tile_count/{location}/{date}` | Returns how many SAR tiles exist on disk for a location/date. |
| GET | `/api/list_downloads` | Lists every location/date with at least one saved tile, for the history panel. |

## Implementation notes & known limitations

- **`mask_s2_clouds()` is defined but not called.** The export path filters clouds at the scene level only (`S2_MAX_CLOUD_PCT`); the per-pixel blue-band cloud mask described by `S2_CLOUD_BLUE_MAX` exists as a function but is not wired into `run_tiled_export`, so individual cloud pixels can still appear inside an exported optical chip.
- **`S1_DB_MIN`, `S1_DB_MAX`, and `S2_REFLECTANCE_MAX` are currently inert.** The preview renderer (`build_visualizations`) uses an adaptive 2nd–98th percentile stretch per image instead of these fixed bounds, so changing them has no effect on output today.
- **Mosaic priority and cloud sorting.** Earth Engine's `ImageCollection.mosaic()` gives priority to the *last* image in the collection wherever images overlap. The optical collection is sorted ascending by `CLOUDY_PIXEL_PERCENTAGE` before mosaicking; since no per-pixel mask is applied, this matters only when two Sentinel-2 footprints overlap the same part of the AOI within the same date window — worth knowing if you're debugging an unexpectedly cloudy chip in an overlap zone.
- **In-memory task state.** `task_registry` is a plain Python dict; restarting the server loses progress history for any incomplete task (saved files on disk are unaffected).
- **Pause/Stop granularity.** The control flag is checked between tiles, not mid-tile, so a tile already being exported by Earth Engine will finish before a new pause or stop takes effect.
- **No authentication.** CORS is open and there is no auth layer; this is meant to run locally for a single user, not to be exposed publicly as-is.
- **Earth Engine quotas and archive coverage apply.** Some AOI/date combinations will legitimately have no usable scene.

## Project structure

```
.
├── main.py              FastAPI backend: GEE queries, tiled export, preview rendering
├── index.html            Frontend: map, AOI drawing, date selection, download controls
├── requirements.txt
├── .gitignore
├── flood_data_exports/   generated at runtime, not committed
└── docs/
    └── images/           screenshots referenced by this README
```

## Citation

If you use Sen1Floods11 itself (the original dataset, not the chips generated by this tool), cite:

> Bonafilia, D., Tellman, B., Anderson, T., & Issenberg, E. (2020). Sen1Floods11: A georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops*, 835–845.

## Acknowledgments

- [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) for the dataset format and chip convention this pipeline targets.
- [Google Earth Engine](https://earthengine.google.com/) and [geemap](https://geemap.org/) for data access and export.
- Copernicus Sentinel-1 and Sentinel-2 missions (ESA) for the source imagery.

## License

No license file is included yet. Add a `LICENSE` (MIT, Apache-2.0, etc.) before publishing this repository publicly, and update this section to point to it.
