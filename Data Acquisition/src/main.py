import ee
import geemap
import os, time, uuid
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
from geopy.geocoders import Nominatim
import numpy as np
import rasterio
from PIL import Image
from pathlib import Path
import io, base64

# -----------------------------------------------------------------------------
# Hyperparameters
#
# SAR (Sentinel-1 GRD)
#   S1_DB_MIN / S1_DB_MAX     : dB bounds used when scaling SAR to uint8 for
#                               preview images. -30 dB ~ open water / noise
#                               floor, +5 dB covers urban double-bounce.
#   S1_DATE_WINDOW_DAYS       : search window (+/- days) around the target
#                               date when looking for a Sentinel-1 scene.
#
# Optical (Sentinel-2 L1C TOA - matches Sen1Floods11)
#   S2_DATE_WINDOW_DAYS       : search window (+/- days) for Sentinel-2.
#   S2_MAX_CLOUD_PCT          : scene-level cloud percentage ceiling. 40 is
#                               permissive so flood events are not excluded.
#   S2_REFLECTANCE_MAX        : reference reflectance ceiling for uint8 scaling.
#   S2_CLOUD_BLUE_MAX         : per-pixel cloud threshold on B2/10000,
#                               following the Sen1Floods11 paper.
#
# Chip geometry
#   CHIP_SIZE_M               : tile edge length in meters. 5120 m at 10 m
#                               resolution gives 512x512 px tiles, matching
#                               Sen1Floods11.
#   EXPORT_SCALE_M            : output pixel resolution in meters (native
#                               S1/S2 resolution is 10 m).
# -----------------------------------------------------------------------------

PROJECT_ID          = "discord-reports-439616"
BASE_DIR            = "flood_data_exports"
PORT                = 8002

S1_DB_MIN           = -30
S1_DB_MAX           =   5
S1_DATE_WINDOW_DAYS =   5

S2_DATE_WINDOW_DAYS =   10
S2_MAX_CLOUD_PCT    =  40
S2_REFLECTANCE_MAX  = 10000
S2_CLOUD_BLUE_MAX   =   0.35

CHIP_SIZE_M         = 5120
EXPORT_SCALE_M      =   10

# Setup
os.makedirs(BASE_DIR, exist_ok=True)


def authenticate_gee():
    """
    Authenticate with Google Earth Engine and initialize the client.

    On the first run this opens a browser window for the standard OAuth
    flow; the resulting credentials are cached under
    ~/.config/earthengine/, so subsequent runs initialize silently without
    further interaction.

    The Google account used must have at least the
    roles/serviceusage.serviceUsageConsumer role on the target GCP project
    (see PROJECT_ID), which can be granted from the IAM console.
    """
    try:
        ee.Initialize(project=PROJECT_ID)
        print(f"GEE initialized from cached credentials | project: {PROJECT_ID}")
    except Exception:
        print("No cached credentials found, starting browser authentication")
        ee.Authenticate(auth_mode="localhost")
        ee.Initialize(project=PROJECT_ID)
        print(f"GEE initialized | project: {PROJECT_ID}")


try:
    authenticate_gee()
except Exception as exc:
    print(f"GEE initialization failed: {exc}")
    print("Confirm the account has IAM access to the GCP project:")
    print(f"https://console.developers.google.com/iam-admin/iam?project={PROJECT_ID}")

# FastAPI app setup
app = FastAPI(title="Flood Data Generator - Sen1Floods11 Compatible")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
geolocator = Nominatim(user_agent="flood_gen_sen1floods11_2026")
task_registry: dict = {}


# Request/response models
class GeoJSON(BaseModel):
    type: str
    geometry: dict
    properties: Optional[dict] = {}

class DownloadRequest(BaseModel):
    date: str
    geometry: dict

class TaskControlRequest(BaseModel):
    action: str  # "pause", "resume", or "stop"


# Helper functions
def get_location_name(geometry: dict) -> str:
    """
    Resolve a human-readable location label for a geometry.

    Computes the centroid of the given geometry, reverse-geocodes it with
    Nominatim, and returns a "City_Country" string suitable for use as a
    folder name. Falls back to "Custom_Region" if reverse geocoding fails.
    """
    try:
        centroid = ee.Geometry(geometry).centroid().getInfo()["coordinates"]
        loc  = geolocator.reverse(f"{centroid[1]}, {centroid[0]}", language="en")
        addr = loc.raw.get("address", {})
        city = addr.get("city") or addr.get("town") or addr.get("state", "Area")
        country = addr.get("country", "Unknown")
        return f"{city}_{country}".replace(" ", "_").replace(",", "")
    except Exception:
        return "Custom_Region"


def get_utm_epsg(geometry: dict) -> str:
    """
    Determine the appropriate UTM EPSG code for a geometry.

    Uses the geometry's centroid longitude/latitude to compute the UTM zone
    and hemisphere, then returns the matching EPSG code (e.g. "EPSG:32636").
    """
    coords = ee.Geometry(geometry).centroid(1).coordinates().getInfo()
    lon, lat = coords[0], coords[1]
    zone = int((lon + 180) / 6) + 1
    base = 32600 if lat >= 0 else 32700
    epsg = f"EPSG:{base + zone}"
    print(f"UTM CRS: {epsg}")
    return epsg


def mask_s2_clouds(image):
    """
    Apply a per-pixel cloud mask to a Sentinel-2 image.

    Masks out pixels where the blue band (B2) reflectance is at or above
    S2_CLOUD_BLUE_MAX, following the thresholding method described in the
    Sen1Floods11 paper.
    """
    blue = image.select("B2").divide(10000)
    return image.updateMask(blue.lt(S2_CLOUD_BLUE_MAX))


def percentile_stretch(arr: np.ndarray, lo: int = 2, hi: int = 98) -> np.ndarray:
    """
    Rescale an array to uint8 using a percentile contrast stretch.

    Clips values to the [lo, hi] percentile range, then linearly maps that
    range to 0-255. Used to turn raw float SAR/optical bands into viewable
    preview images. Returns an all-zero array if there are no finite values.
    """
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    vmin, vmax = np.percentile(valid, [lo, hi])
    return np.clip((arr - vmin) / (vmax - vmin + 1e-8) * 255, 0, 255).astype(np.uint8)


def ndarray_to_b64_png(arr: np.ndarray) -> str:
    """Encode a numpy image array as a base64 PNG string."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def build_visualizations(sar_path, opt_path) -> dict:
    """
    Build preview images for a downloaded tile pair.

    Reads the SAR GeoTIFF (band 1 = VH, band 2 = VV) and the optical GeoTIFF
    (band 2 = blue, band 3 = green, band 4 = red), applies a percentile
    stretch to each band, and returns a dict of base64-encoded PNGs:
    "vh", "vv", "false_color" (VV/VH composite), and "rgb" (true color).
    Any input that is missing or does not exist on disk is skipped.
    """
    out = {}

    if sar_path and os.path.exists(sar_path):
        with rasterio.open(sar_path) as src:
            vh = src.read(1).astype(np.float32)
            vv = src.read(2).astype(np.float32)
        out["vh"] = ndarray_to_b64_png(percentile_stretch(vh))
        out["vv"] = ndarray_to_b64_png(percentile_stretch(vv))
        out["false_color"] = ndarray_to_b64_png(np.stack([
            percentile_stretch(vv),
            percentile_stretch(vh),
            np.zeros(vh.shape, dtype=np.uint8),
        ], axis=-1))

    if opt_path and os.path.exists(opt_path):
        with rasterio.open(opt_path) as src:
            r = src.read(4).astype(np.float32)
            g = src.read(3).astype(np.float32)
            b = src.read(2).astype(np.float32)
        out["rgb"] = ndarray_to_b64_png(np.stack([
            percentile_stretch(r),
            percentile_stretch(g),
            percentile_stretch(b),
        ], axis=-1))

    return out


# Core export logic
def run_tiled_export(task_id: str, date_str: str,
                     geometry: dict, location_name: str):
    """
    Download a full set of SAR/optical tiles for one date and area.

    Builds a Sentinel-1 (VH, VV) mosaic and a Sentinel-2 L1C TOA mosaic
    for the requested date window, splits the area of interest into a grid
    of CHIP_SIZE_M tiles, and exports each tile pair as a GeoTIFF to
    BASE_DIR/<location_name>/<date_str>/. Progress and status are written
    to task_registry[task_id] so the API can report progress to the client.
    Runs as a background task; any exception is caught and recorded in the
    task registry rather than raised.
    """
    try:
        task_registry[task_id].update(
            {"status": "running", "message": "Initializing GEE query"})

        date_obj   = ee.Date(date_str)
        roi        = ee.Geometry(geometry)
        native_crs = get_utm_epsg(geometry)

        grid_proj     = ee.Projection(native_crs).atScale(CHIP_SIZE_M)
        grid_cells    = roi.coveringGrid(grid_proj)
        grid_features = grid_cells.getInfo()["features"]
        total         = len(grid_features)

        task_registry[task_id].update(
            {"total": total, "message": f"Found {total} tiles"})

        path_s1 = os.path.join(BASE_DIR, location_name, date_str, "SAR_ALL_BANDS")
        path_s2 = os.path.join(BASE_DIR, location_name, date_str, "OPTICAL_ALL_BANDS")
        os.makedirs(path_s1, exist_ok=True)
        os.makedirs(path_s2, exist_ok=True)

        # Sentinel-1 GRD, VH then VV, raw dB float32 (Sen1Floods11 compatible)
        s1_img = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(roi)
            .filterDate(
                date_obj.advance(-S1_DATE_WINDOW_DAYS, "day"),
                date_obj.advance( S1_DATE_WINDOW_DAYS, "day"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select(["VH", "VV"])
            .mosaic()
            # Raw dB values (float32), no scaling, matches Sen1Floods11
        )

        # Sentinel-2 L1C TOA, matches Sen1Floods11 exactly
        s2_img = (
            ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            .filterBounds(roi)
            .filterDate(
                date_obj.advance(-S2_DATE_WINDOW_DAYS, "day"),
                date_obj.advance( S2_DATE_WINDOW_DAYS, "day"))
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", S2_MAX_CLOUD_PCT))
            .sort("CLOUDY_PIXEL_PERCENTAGE")
            .mosaic()
            .select(["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B10","B11","B12"])
            # Raw DN values (int16), no scaling, matches Sen1Floods11
        )

        for i, feature in enumerate(grid_features):
            while task_registry[task_id].get("control") == "pause":
                task_registry[task_id]["message"] = f"Paused at tile {i + 1} / {total}"
                time.sleep(1)

            if task_registry[task_id].get("control") == "stop":
                task_registry[task_id].update({
                    "status": "stopped",
                    "message": f"Stopped at tile {i} / {total}"
                })
                print(f"Export stopped by user: {location_name} | {date_str} | {i}/{total} tiles")
                return

            tile_geom = ee.Geometry(feature["geometry"])
            task_registry[task_id].update({
                "progress": i,
                "message": f"Downloading tile {i + 1} / {total}"
            })
            geemap.ee_export_image(
                s1_img,
                filename=os.path.join(path_s1, f"tile_{i}.tif"),
                region=tile_geom, scale=EXPORT_SCALE_M,
                crs=native_crs, file_per_band=False,
            )
            geemap.ee_export_image(
                s2_img,
                filename=os.path.join(path_s2, f"tile_{i}.tif"),
                region=tile_geom, scale=EXPORT_SCALE_M,
                crs=native_crs, file_per_band=False,
            )

        task_registry[task_id].update({
            "status": "done", "progress": total,
            "message": f"{total} tiles saved - {location_name}/{date_str}"
        })
        print(f"Export complete: {location_name} | {date_str} | {total} tiles")

    except Exception as exc:
        task_registry[task_id].update({"status": "error", "message": str(exc)})
        print(f"Export error: {exc}")


# API endpoints
@app.get("/")
async def serve_frontend():
    """Serve the frontend single-page application."""
    return FileResponse("index.html")


@app.post("/api/process_polygon")
async def process_polygon(feature: GeoJSON):
    """
    Inspect an area of interest and return available paired dates.

    Given a GeoJSON feature (with optional start_date/end_date properties),
    computes the area and tile count, queries Earth Engine for Sentinel-1
    and Sentinel-2 acquisition timestamps in that window, and returns the
    dates where a Sentinel-2 scene has a matching Sentinel-1 scene within
    S1_DATE_WINDOW_DAYS. Also returns a resolved location name for display.
    """
    try:
        location_name = get_location_name(feature.geometry)
        roi   = ee.Geometry(feature.geometry)
        start = feature.properties.get("start_date", "2019-01-01")
        end   = feature.properties.get("end_date",   "2025-12-31")

        # Compute polygon area and tile count once, shared by all dates
        area_m2    = roi.area(1).getInfo()
        area_km2   = round(area_m2 / 1e6, 2)
        native_crs = get_utm_epsg(feature.geometry)
        grid_proj  = ee.Projection(native_crs).atScale(CHIP_SIZE_M)
        tile_count = roi.coveringGrid(grid_proj).size().getInfo()

        # Fetch all timestamps in two Earth Engine calls
        s2_timestamps = (
            ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            .filterBounds(roi).filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", S2_MAX_CLOUD_PCT))
            .aggregate_array("system:time_start").getInfo()
        )
        s1_timestamps = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(roi).filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .aggregate_array("system:time_start").getInfo()
        )

        window_ms = S1_DATE_WINDOW_DAYS * 86400 * 1000
        paired_dates = []
        for ts in s2_timestamps:
            if any(abs(ts - s1) <= window_ms for s1 in s1_timestamps):
                paired_dates.append(time.strftime("%Y-%m-%d", time.gmtime(ts / 1000.0)))

        unique_dates = sorted(set(paired_dates))
        return {
            "location":       location_name,
            "available_dates": unique_dates,
            "count":          len(unique_dates),
            "tile_count":     tile_count,
            "area_km2":       area_km2,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/download_tiles")
async def download_tiles(req: DownloadRequest, background_tasks: BackgroundTasks):
    """
    Queue a tile export job for a given date and geometry.

    Creates a task entry in task_registry and schedules run_tiled_export
    as a FastAPI background task. Returns the task_id immediately so the
    client can poll /api/status/{task_id} for progress.
    """
    location_name = get_location_name(req.geometry)
    task_id       = str(uuid.uuid4())[:8]
    task_registry[task_id] = {
        "status": "queued", "progress": 0, "total": 0,
        "message": "Queued", "location": location_name, "date": req.date,
        "control": "run",
    }
    background_tasks.add_task(
        run_tiled_export, task_id, req.date, req.geometry, location_name)
    return {"task_id": task_id, "location": location_name, "date": req.date}


@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    """Return the current status/progress record for a background task."""
    if task_id not in task_registry:
        raise HTTPException(status_code=404, detail="Task not found")
    return task_registry[task_id]


@app.post("/api/task_control/{task_id}")
async def task_control(task_id: str, req: TaskControlRequest):
    """
    Pause, resume, or stop a running download task.

    Sets the control flag on the task's registry entry, which
    run_tiled_export checks between tiles. "pause" makes the export loop
    wait, "resume" lets it continue, and "stop" ends the export early and
    marks the task as "stopped".
    """
    if task_id not in task_registry:
        raise HTTPException(status_code=404, detail="Task not found")
    if req.action not in ("pause", "resume", "stop"):
        raise HTTPException(status_code=400, detail="Invalid action")
    task_registry[task_id]["control"] = "run" if req.action == "resume" else req.action
    return {"task_id": task_id, "control": task_registry[task_id]["control"]}


@app.get("/api/visualize/{location}/{date}/{tile_idx}")
async def visualize_tile(location: str, date: str, tile_idx: int):
    """
    Return preview images for a single downloaded tile.

    Locates the SAR and optical GeoTIFFs for the given location, date, and
    tile index, and returns base64-encoded PNG previews via
    build_visualizations(). Raises 404 if neither file exists.
    """
    base     = os.path.join(BASE_DIR, location, date)
    sar_path = os.path.join(base, "SAR_ALL_BANDS",     f"tile_{tile_idx}.tif")
    opt_path = os.path.join(base, "OPTICAL_ALL_BANDS", f"tile_{tile_idx}.tif")
    if not os.path.exists(sar_path) and not os.path.exists(opt_path):
        raise HTTPException(status_code=404, detail="Tile not found")
    return build_visualizations(
        sar_path if os.path.exists(sar_path) else None,
        opt_path if os.path.exists(opt_path) else None,
    )


@app.get("/api/tile_count/{location}/{date}")
async def tile_count(location: str, date: str):
    """Return the number of SAR tiles downloaded for a location/date."""
    sar_dir = Path(BASE_DIR) / location / date / "SAR_ALL_BANDS"
    n = len(list(sar_dir.glob("*.tif"))) if sar_dir.exists() else 0
    return {"count": n}


@app.get("/api/list_downloads")
async def list_downloads():
    """
    List all completed downloads on disk.

    Walks BASE_DIR for location/date subfolders and reports the tile count
    found in each, used to populate the download history list in the
    frontend.
    """
    results = []
    base = Path(BASE_DIR)
    if base.exists():
        for loc in sorted(base.iterdir()):
            if not loc.is_dir():
                continue
            for date_dir in sorted(loc.iterdir()):
                if not date_dir.is_dir():
                    continue
                sar_dir = date_dir / "SAR_ALL_BANDS"
                opt_dir = date_dir / "OPTICAL_ALL_BANDS"
                n_sar = len(list(sar_dir.glob("*.tif"))) if sar_dir.exists() else 0
                n_opt = len(list(opt_dir.glob("*.tif"))) if opt_dir.exists() else 0
                if n_sar or n_opt:
                    results.append({
                        "location": loc.name,
                        "date":     date_dir.name,
                        "tiles":    max(n_sar, n_opt),
                    })
    return results


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, reload=False)
