# Satellite Imagery-Based Property Valuation

**Multimodal Regression Pipeline for Real Estate Pricing**

---

## Project Overview

This project predicts **property market value (price)** by fusing:

* **Tabular real-estate data** (bedrooms, sqft, coordinates, etc.)
* **Satellite imagery embeddings** extracted using a frozen **ResNet-18 CNN**
* **Dimensionality reduction** via **PCA**
* **Regression modeling** using **XGBoost**

The goal is to enhance valuation by capturing environmental and neighborhood context such as green cover, road density, and waterfront proximity.

---

## Repository Structure

```
src/
  ├── data_fetcher.py
  ├── extract_image_features.py
  ├── image_dataset.py
notebooks/
  ├── preprocessing.ipynb
  ├── image_feature_extraction.ipynb
  ├── model_training.ipynb
outputs/
  ├── submission.csv
  ├── tabular_baseline.csv
  ├── gradcam_samples/ (screenshots saved manually)
```

---

## Setup Instructions

1. **Clone the repo**

2. **Create and activate virtual environment**

   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Fetch satellite images (already included in fused training, optional to re-run)**

   ```bash
   python src/data_fetcher.py
   ```

---

## Run the Model

1. **Open Jupyter Notebook**

   ```bash
   jupyter notebook
   ```

2. Run notebooks in this order:

   ```
  a) notebooks/image_feature_extraction.ipynb   → generates image embeddings(CSV)
  b) notebooks/model_training.ipynb             → trains fused model and saves final predictions

   ```

   and run all cells top-to-bottom.

3. The final predictions are saved at:

   ```
   outputs/23117123_final.csv  (id, predicted_price)
   ```

---

## Model Results (to be added after training runs)

1) Tabular-Only Baseline (Random Forest)

Validation R²: 0.8666

Validation RMSE: 129,376

Captures structural pricing drivers effectively, serving as a strong classical benchmark.

2) Multimodal Regression (ResNet-18 + PCA + XGBoost)

Validation R²: 0.8333

Validation RMSE: 144,632

Learns environmental pricing cues (water, roads, greenery) from satellite imagery, verified using Grad-CAM sample explanations.

3) Final Test Predictions

Total predictions: 5,404

Output CSV: outputs/submission.csv

Columns: id, predicted_price

---

## Explainability

Grad-CAM overlays were generated on sample satellite images and saved manually in `outputs/gradcam_samples/` for inclusion in the final PDF report.

---


