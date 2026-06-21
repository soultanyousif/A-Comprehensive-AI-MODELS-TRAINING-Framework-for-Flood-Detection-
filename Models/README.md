# 🧠 Model Architectures

This project's flood detection and segmentation models are built on **MUST-Former**, a family of transformer-based, multi-modal flood mapping architectures.

> All model architectures used across the **Training and Preprocessing** and **Damaged Building Detection** modules come from a related project of mine:
>
> 🔗 **[MUST-Former: Transformer-Based Models for Multi-Modal Flood Mapping](https://github.com/soultanyousif/MUST-Former-Transformer-Based-Models-for-Multi-Modal-Flood-Mapping)**

This project does not implement new model architectures — it consumes the full MUST-Former model family from that repository and wraps it in the data acquisition → annotation → training → impact-assessment pipeline described in this repo.

## Variants Used

All five MUST-Former variants are used by this pipeline, depending on which sensor data is available and which fusion strategy is being evaluated:

| Variant | Sensor(s) | Strategy |
|---|---|---|
| **MUST-Former-SAR** (`s1_only`) | Sentinel-1 only | Radar-only, 2 input channels |
| **MUST-Former-Optical** (`s2_only`) | Sentinel-2 only | Optical-only, 13–15 input channels |
| **MUST-Former-Projector** (`fusion_projector`) | Sentinel-1 + Sentinel-2 | 1x1-convolution projection of each modality to 3 channels, then summed |
| **MUST-Former-CrossAttn** (`fusion_attention`) | Sentinel-1 + Sentinel-2 | Bi-directional cross-attention between SAR and optical tokens before classification |
| **MUST-Former-PCA** (`fusion_pca`) | Sentinel-1 + Sentinel-2 | Channel standardization (zero mean, unit variance) followed by projection to 3 channels |

The `fusion_pca` (PCA fusion) variant is the default checkpoint used in the Damaged Building Detection module.

## How They're Used Here

- **Training and Preprocessing module** — trains each MUST-Former variant from scratch on paired Sentinel-1/Sentinel-2 chips produced by the Data Acquisition and Data Annotation modules.
- **Damaged Building Detection module** — loads a trained MUST-Former checkpoint (`.pth`) for inference, generates flood masks, vectorizes them into polygons, and cross-references them against the Google Open Buildings v3 dataset.

## Credit

All model architecture design, implementation, and benchmarking for MUST-Former lives in its own dedicated repository — see the link above for architecture details, training code, and experiment results.
