# 🌊 Flood Building Impact Assessment Pipeline

This pipeline evaluates multi-modal SegFormer models on satellite imagery to detect flooded areas, then overlays those predictions with the **Google Open Buildings v3** dataset to assess structural impact. Each building inside the image bounds is classified as **flooded** or **safe** based on whether the model predicted water at that location. Results are displayed on an interactive satellite map.

## Table of Contents
- [The "Unknown Region" Challenge & Solution](#-the-unknown-region-challenge--solution)
- [Pipeline Architecture](#pipeline-architecture)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Multi-Modal SegFormer Inference](#2-multi-modal-segformer-inference)
  - [3. Automatic Geolocation & Vectorization](#3-automatic-geolocation--vectorization)
  - [4. Building Impact Analysis](#4-building-impact-analysis)
  - [5. Visualization](#5-visualization)
- [Testing Dataset](#-testing-dataset)
- [Disclaimer](#️-disclaimer)

## 🌍 The "Unknown Region" Challenge & Solution

A major challenge in automated disaster response is assessing areas where the region name or administrative boundaries are unknown. Because we only have the raw satellite images and the generated flood masks, we cannot rely on geocoders, region names, or external bounding box APIs.

**Our solution:** the pipeline extracts coordinates automatically, directly from the TIFF files, using `rasterio`. It reads each image's CRS (Coordinate Reference System) and bounding box, then transforms those coordinates to `EPSG:4326` (latitude/longitude). This means no region name needs to be specified — the pipeline determines the exact geographic location of each image tile purely from its metadata, allowing it to seamlessly query building footprints for any location on Earth.

## Pipeline Architecture

### 1. Data Preprocessing

- **Sentinel-1 SAR:** VV and VH bands are clipped to the valid dB range of **-50 to 0**, then normalized to **0–1**. NaN values are replaced with **-25 dB** before clipping.
- **Sentinel-2 Optical:** 13 bands are normalized to **0–1**.
- **Spectral Indices:** NDVI and NDWI are computed from Sentinel-2 bands and appended as extra channels to provide richer spectral context.

### 2. Multi-Modal SegFormer Inference

The pipeline supports five SegFormer model variants to evaluate different sensor combinations:

| Variant | Description |
|---|---|
| `fusion_pca` | PCA fusion — normalizes channels, projects to 3 channels |
| `fusion_projector` | Projector fusion — 1x1 convolutions, sums SAR and optical features |
| `fusion_attention` | Attention fusion — cross-attention between SAR and optical tokens |
| `s1_only` | Sentinel-1 SAR only |
| `s2_only` | Sentinel-2 Optical only |

### 3. Automatic Geolocation & Vectorization

Flood predictions (raster masks) are vectorized into polygons. The pipeline uses the TIFF metadata (CRS and affine transform) to map pixel coordinates to real-world coordinates, converting them to `EPSG:4326` for global compatibility.

### 4. Building Impact Analysis

Using the extracted WGS84 bounding boxes, the pipeline calculates the required **S2 Cells (Level 4)** and downloads the corresponding building footprints from the **Google Open Buildings v3** dataset. A spatial join determines which buildings intersect with the flooded polygons.

### 5. Visualization

Generates both:
- **Static satellite maps** using [Contextily](https://contextily.readthedocs.io/)
- **Interactive HTML maps** using [Folium](https://python-visualization.github.io/folium/)

Both feature Esri World Imagery basemaps, flood polygons, and color-coded building markers:
- 🔴 **Red** = Flooded
- 🟢/⚪ **Gray/Green** = Safe

## 🧪 Testing Dataset

The pipeline was tested using the **Bolivia** split of the [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) dataset, and by extracting the **Somalia** samples from its test set. These two splits contain paired Sentinel-1 SAR, Sentinel-2 optical, and hand-labeled water masks from flood events in the two regions.

## ⚠️ Disclaimer

### Google Open Buildings V3 Dataset Coverage

This analysis uses the Google Open Buildings V3 dataset.

- ✅ **Covers:** Africa, South Asia, Southeast Asia, Latin America, and the Caribbean
- ❌ **Does NOT cover:** North America, Europe, Australia/New Zealand, most of East Asia, or Antarctica

If your area of interest is outside the covered regions, no building data will be found.
