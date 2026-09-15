# Nepal Multi-Hazard Early Warning Prototype

**Machine learning, GIS and an interactive dashboard for municipality-level hazard assessment in Nepal.**

This MSc Data Science dissertation project combines flood risk modelling, landslide susceptibility modelling and GIS-based glacial lake outburst flood (GLOF) scoring. It brings the three assessments together to help users explore spatial patterns and identify municipalities for further investigation.

The Streamlit dashboard supports municipality search, hazard comparison and boundary mapping. The project demonstrates a workflow from prepared datasets to model evaluation, spatial integration and interactive presentation.

> **Research prototype:** This project assesses relative hazard priority. It does not provide real-time forecasts, operational alerts or evacuation guidance.

## Project at a glance

| Component | Approach |
|---|---|
| Flood assessment | Supervised classification using municipality-level predictors |
| Landslide assessment | Supervised classification using municipality-level predictors |
| GLOF assessment | GIS-based scoring using lake characteristics and available spatial inputs |
| Multi-hazard integration | Highest of the three hazard scores determines overall priority |
| Spatial outputs | CSV and GeoPackage results for QGIS and the dashboard |
| Interactive interface | Streamlit dashboard with municipality search and mapped boundaries |

## Repository contents

| File | Purpose |
|---|---|
| `Flood_Landslide_ML_Code.py` | Data exploration, model training, evaluation and prediction exports |
| `GLOF_Risk_Scoring_Code.py` | Lake-level and municipality-level GLOF scoring |
| `Multi_Hazard_Integration_Code.py` | Combines hazard results using municipality identifiers |
| `app.py` | Interactive Streamlit dashboard |
| `admin3_flood_ml_dataset.csv` | Prepared flood modelling dataset |
| `admin3_landslide_ml_dataset.csv` | Prepared landslide modelling dataset |
| `ml_predictor_lists.json` | Predictor configuration used by the ML script |
| `Nepal_Multi_Hazard_Final_Maps.qgz` | QGIS project for map presentation; requires its linked spatial files |

Generated tables, plots, models and spatial layers belong in the local `Project_Outputs` folder. They do not need to be committed to GitHub. Final map images are exported separately from QGIS.

## Methods

### Flood and landslide modelling

The ML script compares Logistic Regression, Random Forest, Support Vector Machine and XGBoost classifiers. It uses a stratified 70:30 train–test split with a random seed of 42, together with five-fold stratified cross-validation.

Predictor preparation includes checks for constant features and removal of highly correlated predictors using a correlation threshold of 0.95. Evaluation includes accuracy, precision, recall, F1-score and ROC-AUC. Cross-validation ROC-AUC is calculated from predicted probabilities using a custom scorer.

The script exports model comparisons, feature importance results, trained models and municipality-level prediction tables. Model selection in the current implementation ranks held-out test ROC-AUC first, followed by F1-score, recall and precision. These results are exploratory; independent validation would be needed for deployment.

### GLOF scoring

GLOF assessment uses GIS-based scoring rather than a trained classifier. Required inputs include glacial lake polygons and municipality boundaries. Glacier, elevation and prepared exposure or terrain inputs support additional components when available.

The method does not simulate dam breach, flood routing, inundation depth or flow velocity. Optional inputs should match those used in the original analysis to reproduce its scores.

### Multi-hazard priority

For each municipality:

```text
Overall priority = max(flood score, landslide score, GLOF score)
```

The output also includes a mean score for reference, but overall priority uses the maximum. The dominant hazard is the hazard with the highest score. In a tie, the current code returns the first matching hazard in the order Flood, Landslide, GLOF.

| Score | Category |
|---|---|
| Below 0.25 | Low |
| 0.25 to below 0.50 | Medium |
| 0.50 to below 0.75 | High |
| 0.75 to 1.00 | Very High |

The combined score is a prioritisation index. It is not a calibrated probability that any hazard will occur.

## Installation

Download this repository or clone it:

```bash
git clone https://github.com/iamramchandra/nepal-multi-hazard-early-warning.git
cd nepal-multi-hazard-early-warning
```

Create and activate a Python environment. For Windows Command Prompt:

```bat
python -m venv .venv
.venv\Scripts\activate
```

Install the packages imported by the scripts:

```bash
python -m pip install numpy pandas scipy matplotlib seaborn scikit-learn xgboost joblib geopandas pyogrio rasterio streamlit plotly
```

QGIS is required separately to open the map project and export its layouts. Package versions are not pinned here; this installation command is a dependency starting point, not a verified reproduction of the original environment.

## Run the project

### 1. Generate flood and landslide results

Keep the two prepared CSV files and `ml_predictor_lists.json` beside the ML script, then run:

```bash
python Flood_Landslide_ML_Code.py
```

This stage uses the included prepared datasets. It creates its output folders automatically. It does not require rerunning the original raw-data preparation workflow.

### 2. Prepare the additional GIS inputs

**The two modelling CSVs are not sufficient to regenerate the full GIS workflow.** The GLOF and integration scripts currently discover inputs from the original project structure. Those source GIS files are not included among the core repository files listed above.

To use the scripts unchanged, place this repository within `11_project_workspace` under a common project root, and supply these sibling folders:

| Location relative to the common root | Required contents |
|---|---|
| `01_admin_boundary/` | Municipality shapefile matching `*admin3*.shp`, `*adm3*.shp` or `*ADM3*.shp`, including its companion files |
| `12_glacial_lakes_glof/` | Glacial lake shapefile, such as `NepalGlacialLake2011.shp`; glacier polygons if available |
| `05_dem_terrain/` | DEM raster in TIFF format, if used |
| `11_features/` | Prepared population, OSM, hydrology and terrain feature CSVs, if used |
| `11_project_workspace/nepal-multi-hazard-early-warning/` | This repository and its scripts |

The optional feature filenames expected by the GLOF script are:

- `admin3_population_exposure_features.csv`
- `admin3_osm_exposure_features.csv`
- `admin3_hydrology_features.csv`
- `admin3_dem_terrain_features.csv`

Use the original source datasets and preprocessing outputs for faithful reproduction. The scripts do not download these inputs automatically. Alternatively, adapt their input-discovery functions to your own data locations before running them.

### 3. Generate GLOF results and combine the hazards

After supplying the GIS inputs and completing the ML stage, run these commands in order from the repository folder:

```bash
python GLOF_Risk_Scoring_Code.py
python Multi_Hazard_Integration_Code.py
```

The integration script reads flood and landslide prediction CSVs from `Project_Outputs/Final_QGIS_Layers/ML_Prediction_Layers` and GLOF results from `Project_Outputs/Final_QGIS_Layers/GLOF_Layers`.

### 4. Launch the dashboard

After generating the required results:

```bash
python -m streamlit run app.py
```

The dashboard reads existing files; launching it does not train models or generate missing outputs. For the municipality boundary map, it uses:

```text
Project_Outputs/Final_QGIS_Layers/Multi_Hazard_Layers/admin3_multi_hazard_risk_scores.gpkg
```

Users can search by municipality, district, province or admin3 code, compare the three hazard categories, and view overall priority and dominant hazard.

### 5. Open and export the QGIS maps

Open `Nepal_Multi_Hazard_Final_Maps.qgz` in QGIS after generating the spatial layers. Repair any missing data-source links, especially the province, district and municipality boundary layers that reference the original project folders.

Export the final map images into `Project_Outputs/Final_Maps`. The dashboard recognises these filenames:

- `flood_ml_risk_map.png`
- `landslide_ml_risk_map.png`
- `glof_gis_risk_map.png`
- `GLOF_Lake_Level_Risk_Map.png`
- `multi_hazard_priority_map.png`
- `dominant_hazard_map.png`

## Local output folders

| Folder under `Project_Outputs/` | Contents |
|---|---|
| `Final_Tables/` | Dataset summaries, evaluation metrics and hazard summaries |
| `Final_Plots/` | Exploratory charts, evaluation plots and feature importance |
| `Final_Models/` | Trained ML models |
| `Final_QGIS_Layers/` | Prediction CSVs and spatial GeoPackages |
| `Final_Notes/` | Generated method and run notes |
| `Working_Outputs/` | Intermediate files and logs |
| `Final_Maps/` | Map images exported from QGIS |

The scripts create their own output directories. Preserve their names so subsequent stages and the dashboard can locate the results.

## Limitations

- Historical recorded-event labels depend on the coverage and quality of the underlying records. No recorded event does not establish absence of hazard.
- Municipality-level summaries do not identify property-level exposure or local inundation boundaries.
- Stratified validation does not establish performance in unseen geographic regions or future events.
- GLOF scores and ML probabilities represent different methods; their combination supports relative prioritisation.
- Full reproduction requires the additional GIS inputs and a compatible software environment.
- The dashboard has no live monitoring feed or automated warning-delivery service.

## Author

**Ramchandra Paudel**  
MSc Data Science, University of East London  
Dissertation: *Machine Learning and GIS-Based Early Warning Prototype for Multi-Hazard Risk in Nepal*

[GitHub profile](https://github.com/iamramchandra)
