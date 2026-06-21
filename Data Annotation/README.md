# Satellite Water Annotation Pipeline

A weak-supervision pipeline that generates water/land masks for Sentinel-1 (SAR) and Sentinel-2 (optical) tile pairs, then fuses the two into a single label per pixel. Designed for building training data for flood-mapping models (Sen1Floods11-style).

> **Note:** the area and date used in this notebook (`AREA_NAME`, `DATE_STR`, `BASE`) are placeholder values for local testing only. Replace them with your own export paths before running on real data.

---

## 1. Input data

The pipeline expects two directories of co-registered GeoTIFF tiles, indexed by the same tile ID (`tile_<id>.tif`):

| Directory | Contents |
|---|---|
| `SAR_ALL_BANDS/` | 2-band Sentinel-1 GRD tiles: band 1 = VV, band 2 = VH (in dB) |
| `OPTICAL_ALL_BANDS/` | 13-band Sentinel-2 L2A tiles, band order `B1..B12, B8A` as listed in `BAND_NAMES` |

A tile is **paired** if the same tile ID exists in both directories. Only paired tiles are run through the fusion step; `sar_only` / `opt_only` report any tile IDs missing their counterpart (e.g. due to a partial export).

---

## 2. Configuration / hyperparameters

| Symbol | Variable | Value | Meaning |
|---|---|---|---|
| — | `S1_DB_MIN`, `S1_DB_MAX` | −30, +5 | dB range used to stretch SAR backscatter for display |
| $\tau_{VH}$ | `S1_WATER_VH_THRESH` | −20.0 dB | Default VH water threshold (fallback / cap) |
| $ENL$ | `LEE_ENL` | 4.4 | Equivalent number of looks, used by the Lee filter |
| $W$ | `LEE_WINDOW` | 7 | Lee filter window size (pixels) |
| — | `KI_MAX_ITER` | 50 | Max iterations for Kittler–Illingworth thresholding |
| $\tau_{SDWI}$ | `SDWI_THRESH` | 17.5 | SDWI confidence margin reference |
| $\tau_{HOT}$ | `HOT_THRESH` | 0.05 | Haze-Optimized Transform threshold (cloud test) |
| $\tau_{B/SWIR}$ | `BLUE_SWIR_THRESH` | 2.5 | Blue/SWIR ratio threshold (cloud test) |
| $\tau_{NIR}$ | `NIR_DROP_THRESH` | 0.10 | NIR reflectance threshold (shadow test) |
| $\tau_{MNDWI}$ | `MNDWI_THRESH` | 0.2 | MNDWI water threshold |
| $\tau_{AWEI_{nsh}}$ | `AWEI_NSH_THRESH` | 0.0 | AWEI (no-shadow) water threshold |
| $\tau_{AWEI_{sh}}$ | `AWEI_SH_THRESH` | 0.0 | AWEI (shadow) water threshold |
| $\tau_{NDVI}$ | `NDVI_VEG_THRESH` | 0.3 | Vegetation veto threshold |

---

## 3. SAR processing

### 3.1 Speckle filtering — refined Lee filter

For each polarization ($VV$, $VH$, in dB), a local mean $\mu$ and local variance $\sigma^2$ are computed over a $W \times W$ window:

$$
\mu = \mathrm{mean}_{W}(x), \qquad \sigma^2 = \max\big(\mathrm{mean}_{W}(x^2) - \mu^2,\ 0\big)
$$

The adaptive weight is:

$$
C_u = \frac{1}{\sqrt{ENL}}, \qquad
C_i = \frac{\sigma}{|\mu| + \epsilon}, \qquad
\beta = \mathrm{clip}\!\left(1 - \left(\frac{C_u}{C_i + \epsilon}\right)^2,\ 0,\ 1\right)
$$

and the filtered output is:

$$
x_{\text{filtered}} = \mu + \beta\,(x - \mu)
$$

Here $\beta \to 1$ in heterogeneous (edge/texture) areas, preserving detail, and $\beta \to 0$ in homogeneous areas, where the pixel is replaced by the local mean to suppress speckle.

### 3.2 Adaptive threshold — Kittler–Illingworth (minimum-error thresholding)

The VH histogram is assumed to be a mixture of two populations (water / land). For a candidate threshold $t$, pixels are split into $\{x \le t\}$ and $\{x > t\}$ with weights $w_1, w_2$, means $\mu_1, \mu_2$, and variances $\sigma_1^2, \sigma_2^2$. The minimum-error criterion solves:

$$
t^{*} = \frac{1}{2}(\mu_1+\mu_2) + \frac{\sigma_1^2 \ln(\sigma_2/\sigma_1)}{\sigma_2^2-\sigma_1^2} + \frac{\sigma_1^2\sigma_2^2}{\sigma_2^2-\sigma_1^2}\ln\frac{w_2}{w_1}
$$

solved iteratively until $|t^{*}-t| < 0.001$ or `KI_MAX_ITER` is reached. The result, $KI$, is the data-driven VH threshold below which a pixel is considered water.

The **effective threshold** used for classification is capped to a plausible range:

$$
\tau_{\text{eff}} = \mathrm{clip}\big(\min(KI,\ \tau_{VH}),\ -25,\ -15\big) \quad \text{[dB]}
$$

### 3.3 SAR dual-pol water index (SDWI)

$$
SDWI = -\frac{VH_{f} + VV_{f}}{2}
$$

A higher SDWI indicates lower combined backscatter — consistent with specular reflection from open water.

### 3.4 SAR classification and probability

A pixel is **water** if:

$$
VH_f < \tau_{\text{eff}}
$$

Confidence scores, both clipped to $[0,1]$ over a 10 dB margin:

$$
c_k = \mathrm{clip}\!\left(\frac{\tau_{\text{eff}} - VH_f}{10},\ 0,\ 1\right), \qquad
c_s = \mathrm{clip}\!\left(\frac{SDWI - \tau_{SDWI}}{10},\ 0,\ 1\right)
$$

$$
\text{water\_conf} = \frac{c_k + c_s}{2}, \qquad
\text{land\_dist} = \mathrm{clip}\!\left(\frac{VH_f - \tau_{\text{eff}}}{10},\ 0,\ 1\right)
$$

The SAR water probability $P_{SAR} \in [0,1]$:

$$
P_{SAR} =
\begin{cases}
0.5 & \text{no data} \\
0.5 + 0.5\cdot\text{water\_conf} & \text{water } (VH_f < \tau_{\text{eff}}) \\
0.5 - 0.5\cdot\text{land\_dist} & \text{land}
\end{cases}
$$

$P_{SAR} = 0.5$ is "uncertain"; values above/below 0.5 indicate water/land evidence respectively.

---

## 4. Optical processing (Sentinel-2)

All reflectance bands are scaled from digital number to reflectance: $b = DN / 10000$.

### 4.1 Cloud and shadow masking

A pixel is flagged **bright** if $b_2 > 0.15$ (used to gate the cloud tests so dark water isn't misclassified):

$$
\text{HOT} = b_2 - 0.5\,b_4 > \tau_{HOT}, \qquad
\text{Blue/SWIR} = \frac{b_2}{b_{11}+\epsilon} > \tau_{B/SWIR}
$$

$$
\text{cloud} = (\text{HOT} \lor \text{Blue/SWIR}) \land \text{bright}
$$

A simple NDWI identifies water-like pixels, which are excluded from the shadow test:

$$
NDWI_{\text{simple}} = \frac{b_3-b_8}{b_3+b_8+\epsilon}, \qquad
\text{is\_water\_like} = NDWI_{\text{simple}} > 0
$$

$$
\text{shadow} = (b_8 < \tau_{NIR}) \land \lnot\text{cloud} \land \lnot\text{is\_water\_like}
$$

$$
\text{clear} = \lnot\text{cloud} \land \lnot\text{shadow}
$$

### 4.2 Water indices

**MNDWI** (Modified Normalized Difference Water Index, Xu 2006):

$$
MNDWI = \frac{b_3-b_{11}}{b_3+b_{11}+\epsilon}
$$

Water has high green ($b_3$) and near-zero SWIR ($b_{11}$), giving $MNDWI > 0$.

**AWEI**, no-shadow and shadow variants (Feyisa et al. 2014):

$$
AWEI_{nsh} = 4(b_3-b_{11}) - (0.25\,b_8 + 2.75\,b_{12})
$$

$$
AWEI_{sh} = b_2 + 2.5\,b_3 - 1.5(b_8+b_{11}) - 0.25\,b_{12}
$$

**NDVI** (vegetation veto):

$$
NDVI = \frac{b_8-b_4}{b_8+b_4+\epsilon}
$$

### 4.3 Voting and classification

Each index casts a vote if it exceeds its threshold:

$$
\text{votes} = \mathbb{1}[MNDWI > \tau_{MNDWI}] + \mathbb{1}[AWEI_{nsh} > \tau_{AWEI_{nsh}}] + \mathbb{1}[AWEI_{sh} > \tau_{AWEI_{sh}}]
$$

$$
\text{water} = (\text{votes} \ge 2) \land (NDVI \le \tau_{NDVI}) \land \text{clear}
$$

The NDVI veto prevents dense, well-lit vegetation from being voted "water" by chance.

### 4.4 Optical water probability

Each index is converted into a score centered on its threshold, so the score equals 0.5 exactly at the threshold:

$$
s_{\text{index}} = \mathrm{clip}\!\left(\frac{\text{index} - \tau_{\text{index}}}{\text{scale}},\ -1,\ 1\right)\cdot 0.5 + 0.5
$$

with scale $0.30$ for MNDWI and $0.15$ for both AWEI variants. The combined index score is the mean of the three:

$$
\text{index\_score} = \frac{s_{MNDWI}+s_{AWEI_{nsh}}+s_{AWEI_{sh}}}{3} \in [0,1]
$$

$0$ = strongly dry, $0.5$ = at threshold, $1$ = open water. The final optical probability:

$$
P_{OPT} =
\begin{cases}
0.5 & \text{cloud, shadow, or no data} \\
\text{index\_score} & \text{clear}
\end{cases}
$$

---

## 5. Fusion

SAR and optical probabilities are combined with an optical weight that decays with cloud cover $c_{\text{frac}}$ (the fraction of the tile flagged cloud or shadow):

$$
w_{OPT} = (1 - c_{\text{frac}})^2
$$

$$
P_{\text{fused}} = \frac{P_{SAR} + P_{OPT}\cdot w_{OPT}}{1 + w_{OPT}}
$$

SAR always contributes with full weight (since SAR is largely unaffected by cloud cover); optical contributes proportionally to how clear the scene is. The final label per pixel:

$$
\text{label} =
\begin{cases}
-1 & \text{both SAR and optical are no-data} \\
1 & P_{\text{fused}} > 0.5 \quad \text{(water)} \\
0 & \text{otherwise} \quad \text{(land)}
\end{cases}
$$

---

## 6. Glossary

| Term | Meaning |
|---|---|
| **VV / VH** | Radar polarizations: transmit vertical / receive vertical (VV), transmit vertical / receive horizontal (VH) |
| **σ⁰ (sigma naught)** | Radar backscatter coefficient, in dB |
| **SDWI** | SAR Dual-pol Water Index, derived from combined VV/VH backscatter |
| **ENL** | Equivalent Number of Looks — a measure of speckle averaging in SAR |
| **MNDWI** | Modified Normalized Difference Water Index (green vs. SWIR) |
| **AWEI** | Automated Water Extraction Index (no-shadow / shadow variants) |
| **NDVI** | Normalized Difference Vegetation Index |
| **NDWI** | Normalized Difference Water Index (green vs. NIR), used here only for the shadow test |
| **HOT** | Haze-Optimized Transform, a cloud-detection test |
| **Clear / cloud / shadow** | Per-pixel optical scene classification used to gate the optical water index |
| **No-data (`-1`)** | Pixel has no valid value in either SAR or optical input |

---

## 7. Outputs

| File | Description |
|---|---|
| `QC_WEAK_SUPERVISION/tile_<id>.tif` | Single-band int16 GeoTIFF mask: `1` = water, `0` = land, `-1` = no-data |
| `vis_sar_all_tiles.png` | VV / VH / VV-VH composite per tile |
| `vis_optical_all_tiles.png` | True-color and false-color composites per tile |
| `vis_tile_<id>_dashboard.png` | 9-panel diagnostic dashboard per tile (indices, masks, probabilities) |
| `vis_summary_stats.png` | Bar charts of fused water %, cloud %, and optical weight per tile |
