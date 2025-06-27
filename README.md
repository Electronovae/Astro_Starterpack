# 🔭 ZTF-Tools: Public ZTF Data Download & Supernova Detection Framework

## 🌌 Project Overview

This open-source project is designed to help anyone download and process public science images from the Zwicky Transient Facility (ZTF), with the goal of detecting astronomical transients like Type Ia supernovae.

It provides a Python-based, modular pipeline that walks through the entire workflow — from querying data to producing scientifically usable light curves — using only public ZTF data.

## 🧪 Main Notebook

All steps of the pipeline are integrated in a single Jupyter notebook:

### 👉 `Astro_StarterPack.ipynb`

This is the core of the project: it brings together data acquisition, image calibration, transient detection, and light curve extraction.

## 🛰️ Pipeline Steps

1. **Indexing the Public Archive**  
   - Pre-downloads IRSA index files  
   - Extracts all science images available for a given field/CCD/quadrant/date

2. **Image Download**  
   - Selects relevant `.fits` files by field, filter, CCD, quadrant, and time range  
   - Based on the indexed `fichiers_par_field.csv` file (generated beforehand)

3. **Image Alignment**  
   - Aligns all FITS images to a common WCS reference using `reproject`

4. **Reference Image Construction**  
   - Filters best-seeing images  
   - Builds a median stack as a stable sky background reference

5. **(Optional) Temporal Stacking**  
   - Groups aligned images in triplets  
   - Improves signal-to-noise ratio (SNR) by ~0.6 mag  
   - Enables detection of fainter transients

6. **Image Subtraction**  
   - Subtracts the reference image from each science frame  
   - Reveals transient sources as residuals

7. **Photometry & Light Curve Extraction**  
   - Performs aperture photometry on transient locations  
   - Outputs light curves in flux and magnitude

## ✨ Example: ZTF17aadlxmv

A ready-to-use analysis is provided in `Your_First_SNIa.ipynb` for the known Type Ia supernova **ZTF17aadlxmv**. This includes:

- Image downloading  
- Reference image creation  
- Stacking  
- Subtraction  
- Light curve generation

📈 The resulting curve reproduces the brightness peak and the secondary bump, as expected for SNe Ia.

## 📦 Requirements

Main Python packages:

- `astropy`
- `numpy`
- `matplotlib`
- `scipy`
- `reproject`
- `tqdm`

(see notebook for full environment)

## 📁 File Structure

- `Astro_StarterPack.ipynb` – 📌 full pipeline (main notebook)
- `fichiers_par_field.csv` – pre-generated observation index
- `ztf_transient_pipeline.py` – `AstroTools` class (modular backend)
- `Your_First_SNIa.ipynb` – example on SN ZTF17aadlxmv
- `CSVDownloader.ipynb` – archive index generation (run once)

## 🔓 Open Science Philosophy

This framework is fully reproducible, modular, and built on public data only. It aims to democratize access to time-domain astronomy and support community-driven discoveries.
