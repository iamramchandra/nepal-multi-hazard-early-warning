from pathlib import Path
from datetime import datetime
import traceback
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# Configuration

SCRIPT_DIR = Path(__file__).resolve().parent
METRIC_CRS = "EPSG:32645"

OUTPUT_ROOT = SCRIPT_DIR / "Project_Outputs"

OUTPUT_FOLDERS = {
    "tables": OUTPUT_ROOT / "Final_Tables" / "GLOF_Tables",
    "plots": OUTPUT_ROOT / "Final_Plots" / "GLOF_Plots",
    "qgis_layers": OUTPUT_ROOT / "Final_QGIS_Layers" / "GLOF_Layers",
    "final_notes": OUTPUT_ROOT / "Final_Notes",
    "logs": OUTPUT_ROOT / "Working_Outputs" / "Logs",
}

ADMIN_COLUMNS = {
    "adm1_name": ["adm1_name", "ADM1_EN", "ADM1_NAME", "Province", "province"],
    "adm2_name": ["adm2_name", "ADM2_EN", "ADM2_NAME", "District", "district"],
    "adm3_name": ["adm3_name", "ADM3_EN", "ADM3_NAME", "Municipality", "municipality"],
    "adm3_pcode": ["adm3_pcode", "ADM3_PCODE", "ADM3_CODE", "adm3_code", "pcode", "PCODE"],
}

RISK_ORDER = ["Low", "Medium", "High", "Very High"]


# Helper functions

def create_output_folders():
    for folder in OUTPUT_FOLDERS.values():
        folder.mkdir(parents=True, exist_ok=True)


def setup_plot_style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9


def print_section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_subsection(title):
    print("\n" + "-" * 90)
    print(title)
    print("-" * 90)


def save_table(df, filename):
    path = OUTPUT_FOLDERS["tables"] / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_plot(fig, filename):
    path = OUTPUT_FOLDERS["plots"] / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def find_project_root(start_dir):
    candidates = [start_dir] + list(start_dir.parents)

    for folder in candidates:
        if (folder / "12_glacial_lakes_glof").exists() and (folder / "01_admin_boundary").exists():
            return folder

    raise FileNotFoundError(
        "Could not find Nepal_Early_Warning_Project root folder. "
        "Please keep this script inside the project workspace folder."
    )


def find_first_file(base_folder, patterns, required=True):
    base_folder = Path(base_folder)

    for pattern in patterns:
        matches = sorted(base_folder.glob(pattern))

        if matches:
            return matches[0]

    if required:
        raise FileNotFoundError(f"No file found in {base_folder} using patterns: {patterns}")

    return None


def find_column(df, candidates):
    lower_map = {str(col).lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def standardise_admin_columns(gdf):
    gdf = gdf.copy()

    for standard_name, candidates in ADMIN_COLUMNS.items():
        found_col = find_column(gdf, candidates)

        if found_col is not None:
            gdf[standard_name] = gdf[found_col].astype(str)
        else:
            if standard_name == "adm3_pcode":
                gdf[standard_name] = ["ADM3_" + str(i + 1).zfill(4) for i in range(len(gdf))]
            else:
                gdf[standard_name] = ""

    return gdf


def standardise_admin_columns_table(df):
    df = df.copy()

    for standard_name, candidates in ADMIN_COLUMNS.items():
        found_col = find_column(df, candidates)

        if found_col is not None:
            df[standard_name] = df[found_col].astype(str)

    return df


def minmax_score(series, lower_q=0.02, upper_q=0.98):
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)

    if values.notna().sum() == 0:
        return pd.Series(np.zeros(len(values)), index=values.index)

    lower = values.quantile(lower_q)
    upper = values.quantile(upper_q)

    if pd.isna(lower) or pd.isna(upper) or upper == lower:
        return pd.Series(np.zeros(len(values)), index=values.index)

    clipped = values.clip(lower, upper)
    score = (clipped - lower) / (upper - lower)

    return score.fillna(0).clip(0, 1)


def inverse_distance_score(distance_km, max_distance_km):
    values = pd.to_numeric(distance_km, errors="coerce").replace([np.inf, -np.inf], np.nan)
    score = 1 - (values / max_distance_km)
    return score.fillna(0).clip(0, 1)


def classify_risk(score):
    if score < 0.25:
        return "Low"
    elif score < 0.50:
        return "Medium"
    elif score < 0.75:
        return "High"
    else:
        return "Very High"


def risk_category_order(category):
    order = {
        "Low": 1,
        "Medium": 2,
        "High": 3,
        "Very High": 4,
    }

    return order.get(category, 0)


def pick_first_existing(df, candidates):
    for col in candidates:
        found = find_column(df, [col])

        if found is not None:
            return found

    return None


def read_vector(path, layer=None):
    if layer is None:
        return gpd.read_file(path, engine="pyogrio")

    return gpd.read_file(path, layer=layer, engine="pyogrio")


def parse_pdl_score(value):
    text = str(value).strip().lower()

    if text in ["", "nan", "none", "no", "false", "0", "not available", "na"]:
        return 0.0

    if "very high" in text:
        return 1.0

    if "high" in text:
        return 0.9

    if "medium" in text or "moderate" in text:
        return 0.6

    if "low" in text:
        return 0.3

    if text in ["1", "yes", "true", "pdl"]:
        return 1.0

    return 0.7


# Data discovery

def discover_input_files(project_root):
    files = {}

    files["glacial_lakes"] = find_first_file(
        project_root / "12_glacial_lakes_glof",
        ["**/NepalGlacialLake2011.shp", "**/*Glacial*Lake*.shp", "**/*lake*.shp"],
        required=True,
    )

    files["glims_glaciers"] = find_first_file(
        project_root / "12_glacial_lakes_glof",
        ["**/glims_polygons.shp", "**/*glims*polygon*.shp", "**/*glacier*.shp"],
        required=False,
    )

    files["admin3"] = find_first_file(
        project_root / "01_admin_boundary",
        ["**/*admin3*.shp", "**/*adm3*.shp", "**/*ADM3*.shp"],
        required=True,
    )

    files["dem"] = find_first_file(
        project_root / "05_dem_terrain",
        ["**/*.tif", "**/*.tiff"],
        required=False,
    )

    files["admin3_population_features"] = find_first_file(
        project_root / "11_features",
        ["**/admin3_population_exposure_features.csv"],
        required=False,
    )

    files["admin3_osm_features"] = find_first_file(
        project_root / "11_features",
        ["**/admin3_osm_exposure_features.csv"],
        required=False,
    )

    files["admin3_hydrology_features"] = find_first_file(
        project_root / "11_features",
        ["**/admin3_hydrology_features.csv"],
        required=False,
    )

    files["admin3_terrain_features"] = find_first_file(
        project_root / "11_features",
        ["**/admin3_dem_terrain_features.csv"],
        required=False,
    )

    return files


def create_input_inventory(files):
    rows = []

    for name, path in files.items():
        rows.append({
            "input_name": name,
            "path": str(path) if path is not None else "",
            "exists": bool(path is not None and Path(path).exists()),
        })

    return pd.DataFrame(rows)


# Load spatial data

def load_admin3(admin3_path):
    admin3 = read_vector(admin3_path)
    admin3 = admin3[admin3.geometry.notna()].copy()

    if admin3.crs is None:
        admin3 = admin3.set_crs("EPSG:4326")

    admin3 = standardise_admin_columns(admin3)
    admin3 = admin3.to_crs("EPSG:4326")

    needed_cols = ["adm1_name", "adm2_name", "adm3_name", "adm3_pcode", "geometry"]
    admin3 = admin3[needed_cols].copy()

    admin3["admin3_area_km2"] = admin3.to_crs(METRIC_CRS).geometry.area / 1_000_000

    return admin3


def load_lakes(lake_path):
    lakes = read_vector(lake_path)
    lakes = lakes[lakes.geometry.notna()].copy()

    if lakes.crs is None:
        lakes = lakes.set_crs("EPSG:4326")

    lakes = lakes.to_crs("EPSG:4326")
    lakes = lakes.reset_index(drop=True)

    lake_code_col = find_column(lakes, ["Gl_Code", "gl_code", "lake_code", "gid"])
    lake_name_col = find_column(lakes, ["Gl_Name", "gl_name", "lake_name", "name"])
    basin_col = find_column(lakes, ["Basin", "basin"])
    sub_basin_col = find_column(lakes, ["Sub_Basin", "sub_basin"])
    area_col = find_column(lakes, ["Gl_Area", "gl_area", "area", "area_km2"])
    length_col = find_column(lakes, ["Gl_Length", "gl_length", "length"])
    elevation_col = find_column(lakes, ["Elevation", "elevation", "elev"])
    class_col = find_column(lakes, ["Class", "class"])
    drainage_col = find_column(lakes, ["Drainage", "drainage"])
    pdl_col = find_column(lakes, ["PDL_Level", "pdl_level"])

    clean = gpd.GeoDataFrame(geometry=lakes.geometry, crs=lakes.crs)

    clean["lake_row_id"] = np.arange(len(lakes))

    if lake_code_col is not None:
        clean["lake_id"] = lakes[lake_code_col].astype(str)
    else:
        clean["lake_id"] = ["lake_" + str(i + 1).zfill(5) for i in range(len(lakes))]

    clean["lake_name"] = lakes[lake_name_col].astype(str) if lake_name_col is not None else ""
    clean["basin"] = lakes[basin_col].astype(str) if basin_col is not None else ""
    clean["sub_basin"] = lakes[sub_basin_col].astype(str) if sub_basin_col is not None else ""

    if area_col is not None:
        clean["lake_area_km2"] = pd.to_numeric(lakes[area_col], errors="coerce")
    else:
        clean["lake_area_km2"] = np.nan

    if length_col is not None:
        clean["lake_length_m"] = pd.to_numeric(lakes[length_col], errors="coerce")
    else:
        clean["lake_length_m"] = np.nan

    if elevation_col is not None:
        clean["lake_elevation_m_attribute"] = pd.to_numeric(lakes[elevation_col], errors="coerce")
    else:
        clean["lake_elevation_m_attribute"] = np.nan

    clean["lake_class"] = lakes[class_col].astype(str) if class_col is not None else ""
    clean["drainage_type"] = lakes[drainage_col].astype(str) if drainage_col is not None else ""
    clean["pdl_level"] = lakes[pdl_col].astype(str).replace("nan", "") if pdl_col is not None else ""

    projected = clean.to_crs(METRIC_CRS)
    geometry_area = projected.geometry.area / 1_000_000

    clean["lake_area_km2"] = clean["lake_area_km2"].fillna(geometry_area)
    clean["lake_centroid_lon"] = clean.geometry.representative_point().x
    clean["lake_centroid_lat"] = clean.geometry.representative_point().y

    return clean


def load_glaciers(glacier_path, lakes):
    if glacier_path is None:
        return None

    lake_bounds = lakes.total_bounds
    buffer_degrees = 1.0

    bbox = (
        lake_bounds[0] - buffer_degrees,
        lake_bounds[1] - buffer_degrees,
        lake_bounds[2] + buffer_degrees,
        lake_bounds[3] + buffer_degrees,
    )

    try:
        glaciers = gpd.read_file(glacier_path, bbox=bbox, engine="pyogrio")
    except Exception:
        glaciers = read_vector(glacier_path)

    glaciers = glaciers[glaciers.geometry.notna()].copy()

    if glaciers.empty:
        return None

    if glaciers.crs is None:
        glaciers = glaciers.set_crs("EPSG:4326")

    glaciers = glaciers.to_crs("EPSG:4326")

    return glaciers


# DEM sampling

def sample_dem_elevation_for_lakes(lakes, dem_path):
    if dem_path is None:
        lakes["lake_elevation_m_dem"] = np.nan
        lakes["lake_elevation_m"] = lakes["lake_elevation_m_attribute"]
        return lakes

    lakes = lakes.copy()

    try:
        with rasterio.open(dem_path) as src:
            lake_points = lakes.copy()
            lake_points["geometry"] = lake_points.geometry.representative_point()
            lake_points = lake_points.to_crs(src.crs)

            coords = [(geom.x, geom.y) for geom in lake_points.geometry]
            sampled_values = []

            for value in src.sample(coords):
                elev = float(value[0])

                if src.nodata is not None and elev == src.nodata:
                    sampled_values.append(np.nan)
                else:
                    sampled_values.append(elev)

            lakes["lake_elevation_m_dem"] = sampled_values

    except Exception as e:
        print(f"DEM sampling skipped because of this issue: {e}")
        lakes["lake_elevation_m_dem"] = np.nan

    lakes["lake_elevation_m"] = lakes["lake_elevation_m_dem"].fillna(lakes["lake_elevation_m_attribute"])

    return lakes


# Spatial joins and distance

def attach_admin3_to_lakes(lakes, admin3):
    lakes = lakes.copy().reset_index(drop=True)

    lake_points = lakes.copy()
    lake_points["geometry"] = lake_points.geometry.representative_point()

    joined = gpd.sjoin(
        lake_points[["lake_row_id", "geometry"]],
        admin3[["adm1_name", "adm2_name", "adm3_name", "adm3_pcode", "geometry"]],
        how="left",
        predicate="within",
    )

    joined = joined.drop(columns=["index_right"], errors="ignore")

    joined = (
        joined
        .sort_values("lake_row_id")
        .drop_duplicates(subset="lake_row_id", keep="first")
    )

    unmatched_ids = joined.loc[joined["adm3_pcode"].isna(), "lake_row_id"].tolist()

    if len(unmatched_ids) > 0:
        print(f"{len(unmatched_ids)} lakes did not match by within join. Trying nearest admin3 join.")

        unmatched_lakes = lake_points[lake_points["lake_row_id"].isin(unmatched_ids)].copy()

        nearest = gpd.sjoin_nearest(
            unmatched_lakes[["lake_row_id", "geometry"]].to_crs(METRIC_CRS),
            admin3[["adm1_name", "adm2_name", "adm3_name", "adm3_pcode", "geometry"]].to_crs(METRIC_CRS),
            how="left",
            distance_col="nearest_admin3_distance_m",
        )

        nearest = nearest.drop(columns=["index_right"], errors="ignore")

        nearest = (
            nearest
            .sort_values(["lake_row_id", "nearest_admin3_distance_m"])
            .drop_duplicates(subset="lake_row_id", keep="first")
            .to_crs("EPSG:4326")
        )

        matched = joined[joined["adm3_pcode"].notna()].copy()
        joined = pd.concat([matched, nearest], ignore_index=True)

    admin_cols = joined[["lake_row_id", "adm1_name", "adm2_name", "adm3_name", "adm3_pcode"]].copy()

    lakes = lakes.drop(columns=["adm1_name", "adm2_name", "adm3_name", "adm3_pcode"], errors="ignore")
    lakes = lakes.merge(admin_cols, on="lake_row_id", how="left")

    return lakes


def calculate_nearest_glacier_distance(lakes, glaciers):
    lakes = lakes.copy().reset_index(drop=True)

    if glaciers is None or glaciers.empty:
        lakes["nearest_glacier_distance_km"] = np.nan
        lakes["glacier_proximity_score"] = 0
        return lakes

    lake_points = lakes.copy()
    lake_points["geometry"] = lake_points.geometry.representative_point()

    nearest = gpd.sjoin_nearest(
        lake_points[["lake_row_id", "geometry"]].to_crs(METRIC_CRS),
        glaciers[["geometry"]].to_crs(METRIC_CRS),
        how="left",
        distance_col="nearest_glacier_distance_m",
    )

    nearest = nearest.drop(columns=["index_right"], errors="ignore")

    nearest = (
        nearest
        .sort_values(["lake_row_id", "nearest_glacier_distance_m"])
        .drop_duplicates(subset="lake_row_id", keep="first")
        .sort_values("lake_row_id")
    )

    distance_table = nearest[["lake_row_id", "nearest_glacier_distance_m"]].copy()
    lakes = lakes.merge(distance_table, on="lake_row_id", how="left")

    lakes["nearest_glacier_distance_km"] = lakes["nearest_glacier_distance_m"] / 1000

    lakes["glacier_proximity_score"] = inverse_distance_score(
        lakes["nearest_glacier_distance_km"],
        max_distance_km=10,
    )

    lakes = lakes.drop(columns=["nearest_glacier_distance_m"], errors="ignore")

    return lakes


# Exposure features

def load_admin3_feature_tables(files):
    feature_frames = []

    feature_sources = [
        ("population", files.get("admin3_population_features")),
        ("osm", files.get("admin3_osm_features")),
        ("hydrology", files.get("admin3_hydrology_features")),
        ("terrain", files.get("admin3_terrain_features")),
    ]

    for source_name, path in feature_sources:
        if path is None:
            continue

        try:
            df = pd.read_csv(path, low_memory=False)
            df = standardise_admin_columns_table(df)

            if "adm3_pcode" not in df.columns:
                print(f"Skipping {source_name} features because adm3_pcode was not found.")
                continue

            df = df.drop_duplicates(subset=["adm3_pcode"]).copy()
            df["adm3_pcode"] = df["adm3_pcode"].astype(str)

            drop_cols = ["adm1_name", "adm2_name", "adm3_name", "geometry"]
            keep_cols = ["adm3_pcode"] + [col for col in df.columns if col not in drop_cols + ["adm3_pcode"]]

            df = df[keep_cols].copy()
            df = df.loc[:, ~df.columns.duplicated()].copy()

            feature_frames.append(df)

        except Exception as e:
            print(f"Could not load {source_name} features: {e}")

    if not feature_frames:
        return pd.DataFrame(columns=["adm3_pcode"])

    merged = feature_frames[0]

    for frame in feature_frames[1:]:
        duplicate_cols = [col for col in frame.columns if col in merged.columns and col != "adm3_pcode"]
        frame = frame.drop(columns=duplicate_cols, errors="ignore")
        merged = merged.merge(frame, on="adm3_pcode", how="outer")

    return merged


def build_admin3_exposure_score(admin3, feature_table):
    admin3 = admin3.copy()
    admin3["adm3_pcode"] = admin3["adm3_pcode"].astype(str)

    if feature_table.empty:
        admin3["exposure_score"] = 0
        return admin3, pd.DataFrame()

    exposure = feature_table.copy()
    exposure["adm3_pcode"] = exposure["adm3_pcode"].astype(str)

    admin3 = admin3.merge(exposure, on="adm3_pcode", how="left")

    pop_col = pick_first_existing(admin3, [
        "worldpop_population_sum",
        "population_sum",
        "pop_sum",
        "total_population",
    ])

    pop_density_col = pick_first_existing(admin3, [
        "worldpop_population_density_per_km2",
        "population_density_per_km2",
        "pop_density",
    ])

    road_col = pick_first_existing(admin3, [
        "osm_road_length_km",
        "road_length_km",
        "total_road_length_km",
    ])

    road_density_col = pick_first_existing(admin3, [
        "osm_road_density_km_per_km2",
        "road_density_km_per_km2",
    ])

    building_col = pick_first_existing(admin3, [
        "osm_building_count",
        "building_count",
        "total_buildings",
    ])

    building_density_col = pick_first_existing(admin3, [
        "osm_building_density_per_km2",
        "building_density_per_km2",
    ])

    place_col = pick_first_existing(admin3, [
        "osm_place_count",
        "place_count",
        "settlement_count",
    ])

    nearest_place_col = pick_first_existing(admin3, [
        "osm_nearest_place_distance_km",
        "nearest_place_distance_km",
        "nearest_settlement_distance_km",
    ])

    component_records = []
    score_components = []

    def add_component(output_col, source_col, weight, inverse=False, max_distance_km=20):
        if source_col is None:
            return

        if inverse:
            admin3[output_col] = inverse_distance_score(admin3[source_col], max_distance_km=max_distance_km)
        else:
            admin3[output_col] = minmax_score(admin3[source_col])

        score_components.append((output_col, weight))

        component_records.append({
            "component": output_col,
            "source_column": source_col,
            "weight": weight,
            "inverse_distance": inverse,
        })

    add_component("population_exposure_score", pop_col or pop_density_col, 0.35)
    add_component("road_exposure_score", road_density_col or road_col, 0.20)
    add_component("building_exposure_score", building_density_col or building_col, 0.25)
    add_component("settlement_exposure_score", place_col, 0.10)
    add_component("settlement_proximity_score", nearest_place_col, 0.10, inverse=True, max_distance_km=20)

    if not score_components:
        admin3["exposure_score"] = 0
    else:
        total_weight = sum(weight for _, weight in score_components)
        score = 0

        for component_col, weight in score_components:
            score = score + admin3[component_col].fillna(0) * (weight / total_weight)

        admin3["exposure_score"] = score.clip(0, 1)

    component_df = pd.DataFrame(component_records)

    return admin3, component_df


# Risk scoring

def calculate_lake_hazard_score(lakes):
    lakes = lakes.copy()

    lakes["lake_area_score"] = minmax_score(lakes["lake_area_km2"])
    lakes["lake_length_score"] = minmax_score(lakes["lake_length_m"])
    lakes["lake_elevation_score"] = minmax_score(lakes["lake_elevation_m"])

    lakes["pdl_score"] = lakes["pdl_level"].apply(parse_pdl_score)
    lakes["pdl_flag"] = (lakes["pdl_score"] > 0).astype(int)

    lakes["glof_hazard_score"] = (
        0.35 * lakes["lake_area_score"] +
        0.20 * lakes["lake_length_score"] +
        0.15 * lakes["lake_elevation_score"] +
        0.20 * lakes["glacier_proximity_score"] +
        0.10 * lakes["pdl_score"]
    ).clip(0, 1)

    return lakes


def attach_exposure_to_lakes(lakes, admin3_exposure):
    lakes = lakes.copy()

    exposure_cols = [
        "adm3_pcode",
        "exposure_score",
        "population_exposure_score",
        "road_exposure_score",
        "building_exposure_score",
        "settlement_exposure_score",
        "settlement_proximity_score",
    ]

    existing_cols = [col for col in exposure_cols if col in admin3_exposure.columns]
    exposure_table = admin3_exposure[existing_cols].copy()

    lakes = lakes.merge(exposure_table, on="adm3_pcode", how="left")

    for col in existing_cols:
        if col != "adm3_pcode":
            lakes[col] = pd.to_numeric(lakes[col], errors="coerce").fillna(0)

    if "exposure_score" not in lakes.columns:
        lakes["exposure_score"] = 0

    return lakes


def calculate_final_lake_risk(lakes):
    lakes = lakes.copy()

    lakes["glof_final_risk_score"] = (
        0.65 * lakes["glof_hazard_score"] +
        0.35 * lakes["exposure_score"]
    ).clip(0, 1)

    lakes["glof_risk_category"] = lakes["glof_final_risk_score"].apply(classify_risk)
    lakes["glof_risk_category_order"] = lakes["glof_risk_category"].apply(risk_category_order)

    return lakes


def aggregate_to_admin3(admin3_exposure, lakes_risk):
    admin3 = admin3_exposure.copy()

    lake_summary = (
        lakes_risk
        .groupby("adm3_pcode", dropna=False)
        .agg(
            glof_lake_count=("lake_id", "count"),
            glof_total_lake_area_km2=("lake_area_km2", "sum"),
            glof_max_lake_area_km2=("lake_area_km2", "max"),
            glof_mean_hazard_score=("glof_hazard_score", "mean"),
            glof_max_hazard_score=("glof_hazard_score", "max"),
            glof_mean_final_lake_risk_score=("glof_final_risk_score", "mean"),
            glof_max_final_lake_risk_score=("glof_final_risk_score", "max"),
            glof_pdl_lake_count=("pdl_flag", "sum"),
            nearest_glacier_distance_km_min=("nearest_glacier_distance_km", "min"),
        )
        .reset_index()
    )

    admin3 = admin3.merge(lake_summary, on="adm3_pcode", how="left")

    fill_cols = [
        "glof_lake_count",
        "glof_total_lake_area_km2",
        "glof_max_lake_area_km2",
        "glof_mean_hazard_score",
        "glof_max_hazard_score",
        "glof_mean_final_lake_risk_score",
        "glof_max_final_lake_risk_score",
        "glof_pdl_lake_count",
    ]

    for col in fill_cols:
        if col in admin3.columns:
            admin3[col] = pd.to_numeric(admin3[col], errors="coerce").fillna(0)

    admin3["lake_presence_score"] = (admin3["glof_lake_count"] > 0).astype(int)
    admin3["admin3_lake_area_score"] = minmax_score(admin3["glof_total_lake_area_km2"])
    admin3["admin3_pdl_score"] = minmax_score(admin3["glof_pdl_lake_count"])

    admin3["glof_admin3_risk_score"] = (
        0.60 * admin3["glof_max_final_lake_risk_score"] +
        0.20 * admin3["admin3_lake_area_score"] +
        0.10 * admin3["admin3_pdl_score"] +
        0.10 * admin3["exposure_score"] * admin3["lake_presence_score"]
    ).clip(0, 1)

    admin3.loc[admin3["glof_lake_count"] == 0, "glof_admin3_risk_score"] = 0

    admin3["glof_admin3_risk_category"] = admin3["glof_admin3_risk_score"].apply(classify_risk)
    admin3["glof_admin3_risk_category_order"] = admin3["glof_admin3_risk_category"].apply(risk_category_order)

    return admin3


# Summary tables

def create_lake_summary(lakes_risk):
    rows = [{
        "total_glacial_lakes": len(lakes_risk),
        "total_lake_area_km2": round(lakes_risk["lake_area_km2"].sum(), 4),
        "mean_lake_area_km2": round(lakes_risk["lake_area_km2"].mean(), 4),
        "max_lake_area_km2": round(lakes_risk["lake_area_km2"].max(), 4),
        "pdl_lake_count": int(lakes_risk["pdl_flag"].sum()),
        "mean_hazard_score": round(lakes_risk["glof_hazard_score"].mean(), 4),
        "mean_final_risk_score": round(lakes_risk["glof_final_risk_score"].mean(), 4),
        "max_final_risk_score": round(lakes_risk["glof_final_risk_score"].max(), 4),
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }]

    return pd.DataFrame(rows)


def create_admin3_summary(admin3_risk):
    rows = [{
        "total_admin3_units": len(admin3_risk),
        "admin3_units_with_glacial_lakes": int((admin3_risk["glof_lake_count"] > 0).sum()),
        "admin3_units_without_glacial_lakes": int((admin3_risk["glof_lake_count"] == 0).sum()),
        "mean_admin3_risk_score": round(admin3_risk["glof_admin3_risk_score"].mean(), 4),
        "max_admin3_risk_score": round(admin3_risk["glof_admin3_risk_score"].max(), 4),
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }]

    return pd.DataFrame(rows)


def risk_category_counts(df, category_col, task_name):
    counts = (
        df[category_col]
        .value_counts()
        .reindex(RISK_ORDER, fill_value=0)
        .reset_index()
    )

    counts.columns = ["risk_category", "count"]
    counts.insert(0, "task", task_name)

    return counts


# Plots

def plot_risk_category_counts(count_df, title, filename):
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.barplot(
        data=count_df,
        x="risk_category",
        y="count",
        order=RISK_ORDER,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("Risk category")
    ax.set_ylabel("Count")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3)

    return save_plot(fig, filename)


def plot_score_histogram(df, score_col, title, filename):
    fig, ax = plt.subplots(figsize=(9, 5))

    sns.histplot(
        data=df,
        x=score_col,
        bins=30,
        kde=True,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("Risk score")
    ax.set_ylabel("Frequency")

    return save_plot(fig, filename)


def plot_top_lakes(lakes_risk):
    top = lakes_risk.sort_values("glof_final_risk_score", ascending=False).head(20).copy()
    top["label"] = top["lake_id"].astype(str)

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.barplot(
        data=top,
        y="label",
        x="glof_final_risk_score",
        ax=ax,
    )

    ax.set_title("Top 20 glacial lakes by GLOF risk score")
    ax.set_xlabel("Final GLOF risk score")
    ax.set_ylabel("Lake ID")

    return save_plot(fig, "top_20_glacial_lakes_by_glof_risk.png")


def plot_top_admin3(admin3_risk):
    top = admin3_risk.sort_values("glof_admin3_risk_score", ascending=False).head(20).copy()

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.barplot(
        data=top,
        y="adm3_name",
        x="glof_admin3_risk_score",
        ax=ax,
    )

    ax.set_title("Top 20 admin3 municipalities by GLOF risk score")
    ax.set_xlabel("Admin3 GLOF risk score")
    ax.set_ylabel("Admin3 municipality")

    return save_plot(fig, "top_20_admin3_by_glof_risk.png")


def plot_lake_area_vs_risk(lakes_risk):
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.scatterplot(
        data=lakes_risk,
        x="lake_area_km2",
        y="glof_final_risk_score",
        hue="glof_risk_category",
        hue_order=RISK_ORDER,
        ax=ax,
    )

    ax.set_title("Lake area and final GLOF risk score")
    ax.set_xlabel("Lake area km²")
    ax.set_ylabel("Final GLOF risk score")
    ax.legend(title="Risk category", bbox_to_anchor=(1.02, 1), loc="upper left")

    return save_plot(fig, "lake_area_vs_glof_risk_score.png")


def plot_hazard_exposure_scatter(lakes_risk):
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.scatterplot(
        data=lakes_risk,
        x="glof_hazard_score",
        y="exposure_score",
        hue="glof_risk_category",
        hue_order=RISK_ORDER,
        ax=ax,
    )

    ax.set_title("GLOF hazard score and exposure score")
    ax.set_xlabel("Hazard score")
    ax.set_ylabel("Exposure score")
    ax.legend(title="Risk category", bbox_to_anchor=(1.02, 1), loc="upper left")

    return save_plot(fig, "hazard_score_vs_exposure_score.png")


# Save QGIS outputs

def save_qgis_layers(lakes_risk, admin3_risk):
    lake_cols = [
        "lake_id",
        "lake_name",
        "basin",
        "sub_basin",
        "adm1_name",
        "adm2_name",
        "adm3_name",
        "adm3_pcode",
        "lake_area_km2",
        "lake_length_m",
        "lake_elevation_m",
        "nearest_glacier_distance_km",
        "lake_area_score",
        "lake_length_score",
        "lake_elevation_score",
        "glacier_proximity_score",
        "pdl_level",
        "pdl_flag",
        "pdl_score",
        "glof_hazard_score",
        "exposure_score",
        "glof_final_risk_score",
        "glof_risk_category",
        "glof_risk_category_order",
        "geometry",
    ]

    lake_cols = [col for col in lake_cols if col in lakes_risk.columns]

    admin_cols = [
        "adm1_name",
        "adm2_name",
        "adm3_name",
        "adm3_pcode",
        "admin3_area_km2",
        "exposure_score",
        "glof_lake_count",
        "glof_total_lake_area_km2",
        "glof_max_lake_area_km2",
        "glof_pdl_lake_count",
        "glof_mean_hazard_score",
        "glof_max_hazard_score",
        "glof_mean_final_lake_risk_score",
        "glof_max_final_lake_risk_score",
        "glof_admin3_risk_score",
        "glof_admin3_risk_category",
        "glof_admin3_risk_category_order",
        "geometry",
    ]

    admin_cols = [col for col in admin_cols if col in admin3_risk.columns]

    lake_gpkg = OUTPUT_FOLDERS["qgis_layers"] / "glof_lake_risk_scores.gpkg"
    admin_gpkg = OUTPUT_FOLDERS["qgis_layers"] / "glof_admin3_risk_scores.gpkg"

    if lake_gpkg.exists():
        lake_gpkg.unlink()

    if admin_gpkg.exists():
        admin_gpkg.unlink()

    lakes_risk[lake_cols].to_file(lake_gpkg, layer="glof_lake_risk_scores", driver="GPKG")
    admin3_risk[admin_cols].to_file(admin_gpkg, layer="glof_admin3_risk_scores", driver="GPKG")

    lake_csv = OUTPUT_FOLDERS["qgis_layers"] / "glof_lake_risk_scores_for_qgis.csv"
    admin_csv = OUTPUT_FOLDERS["qgis_layers"] / "glof_admin3_risk_scores_for_qgis.csv"

    lakes_risk.drop(columns="geometry", errors="ignore").to_csv(lake_csv, index=False, encoding="utf-8-sig")
    admin3_risk.drop(columns="geometry", errors="ignore").to_csv(admin_csv, index=False, encoding="utf-8-sig")

    return lake_gpkg, admin_gpkg, lake_csv, admin_csv


# Notes and logs

def write_methodology_note(project_root, files, lake_summary, admin3_summary, lake_category_counts, admin_category_counts):
    note_path = OUTPUT_FOLDERS["final_notes"] / "glof_methodology_note.txt"

    note = f"""GLOF GIS risk scoring methodology note

Project:
Machine Learning and GIS-Based Early Warning Prototype for Multi-Hazard Risk in Nepal

Purpose:
This script creates a GIS-based GLOF risk scoring output. It is separate from the flood and landslide machine learning workflow because labelled GLOF event data are limited.

Input data:
- Nepal glacial lakes 2011
- GLIMS glacier outlines
- Nepal admin3 boundaries
- Copernicus DEM, used where possible for lake elevation checking
- Admin3 population and OSM exposure features created during earlier preprocessing

Risk scoring structure:
Final lake-level GLOF risk is calculated using:
- Hazard score
- Exposure score

Hazard score uses:
- Lake area
- Lake length
- Lake elevation
- Distance to nearest glacier
- Potentially dangerous lake flag where available

Exposure score uses available admin3 exposure features:
- Population
- Roads
- Buildings
- Settlements or nearest settlement distance

Lake-level final risk:
0.65 * hazard score + 0.35 * exposure score

Admin3-level risk:
Admin3 risk is aggregated from lake-level risk, total lake area, potentially dangerous lake count and exposure. Admin3 units without mapped glacial lakes are assigned a risk score of 0.

Risk categories:
- Low: 0.00 to 0.25
- Medium: 0.25 to 0.50
- High: 0.50 to 0.75
- Very High: 0.75 to 1.00

Important limitation:
This is a GIS-based risk scoring prototype, not an operational GLOF warning system. It does not model dam breach, flood routing, real-time lake level, rainfall, seismic triggers or downstream hydraulic depth.

Project root:
{project_root}

Input files:
{create_input_inventory(files).to_string(index=False)}

Lake summary:
{lake_summary.to_string(index=False)}

Admin3 summary:
{admin3_summary.to_string(index=False)}

Lake risk category counts:
{lake_category_counts.to_string(index=False)}

Admin3 risk category counts:
{admin_category_counts.to_string(index=False)}
"""

    with open(note_path, "w", encoding="utf-8") as f:
        f.write(note)

    return note_path


def write_run_log(project_root, files, outputs):
    log_path = OUTPUT_FOLDERS["logs"] / "glof_risk_scoring_run_log.txt"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("GLOF_Risk_Scoring_Code.py run log\n")
        f.write(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Project root: {project_root}\n")
        f.write(f"Metric CRS: {METRIC_CRS}\n\n")

        f.write("Input files:\n")
        for key, value in files.items():
            f.write(f"{key}: {value}\n")

        f.write("\nOutput files:\n")
        for key, value in outputs.items():
            f.write(f"{key}: {value}\n")

    return log_path


# Main workflow

def main():
    warnings.filterwarnings("ignore")

    create_output_folders()
    setup_plot_style()

    print_section("GLOF GIS Risk Scoring")
    print("This script performs GIS-based risk scoring only. It does not train an ML model.")

    try:
        project_root = find_project_root(SCRIPT_DIR)
        files = discover_input_files(project_root)

        print_subsection("Project and input files")
        print(f"Project root: {project_root}")

        inventory = create_input_inventory(files)
        print(inventory.to_string(index=False))
        save_table(inventory, "glof_input_file_inventory.csv")

        print_subsection("Loading admin3 boundaries")
        admin3 = load_admin3(files["admin3"])
        print(f"Admin3 units loaded: {len(admin3)}")

        print_subsection("Loading glacial lakes")
        lakes = load_lakes(files["glacial_lakes"])
        print(f"Glacial lakes loaded: {len(lakes)}")

        print_subsection("Loading glaciers")
        glaciers = load_glaciers(files["glims_glaciers"], lakes)

        if glaciers is None:
            print("No glacier polygons loaded. Glacier proximity score will be 0.")
        else:
            print(f"Glacier polygons loaded near Nepal: {len(glaciers)}")

        print_subsection("Sampling DEM elevation where possible")
        lakes = sample_dem_elevation_for_lakes(lakes, files["dem"])

        print_subsection("Matching lakes to admin3 municipalities")
        lakes = attach_admin3_to_lakes(lakes, admin3)

        matched_count = int(lakes["adm3_pcode"].notna().sum())
        print(f"Lakes matched to admin3: {matched_count} / {len(lakes)}")

        print_subsection("Calculating nearest glacier distance")
        lakes = calculate_nearest_glacier_distance(lakes, glaciers)

        print_subsection("Loading admin3 exposure features")
        feature_table = load_admin3_feature_tables(files)
        print(f"Admin3 feature table rows: {len(feature_table)}")
        print(f"Admin3 feature table columns: {len(feature_table.columns)}")

        admin3_exposure, exposure_component_table = build_admin3_exposure_score(admin3, feature_table)
        save_table(exposure_component_table, "glof_exposure_score_components.csv")

        print_subsection("Calculating GLOF hazard and exposure scores")
        lakes = calculate_lake_hazard_score(lakes)
        lakes = attach_exposure_to_lakes(lakes, admin3_exposure)
        lakes_risk = calculate_final_lake_risk(lakes)

        print_subsection("Aggregating GLOF risk to admin3")
        admin3_risk = aggregate_to_admin3(admin3_exposure, lakes_risk)

        lake_summary = create_lake_summary(lakes_risk)
        admin3_summary = create_admin3_summary(admin3_risk)

        lake_category_counts = risk_category_counts(
            lakes_risk,
            "glof_risk_category",
            "lake_level_glof_risk",
        )

        admin_category_counts = risk_category_counts(
            admin3_risk,
            "glof_admin3_risk_category",
            "admin3_glof_risk",
        )

        save_table(lake_summary, "glof_lake_summary.csv")
        save_table(admin3_summary, "glof_admin3_summary.csv")
        save_table(lake_category_counts, "glof_lake_risk_category_counts.csv")
        save_table(admin_category_counts, "glof_admin3_risk_category_counts.csv")

        lake_table = lakes_risk.drop(columns="geometry", errors="ignore")
        admin_table = admin3_risk.drop(columns="geometry", errors="ignore")

        save_table(lake_table, "glof_lake_risk_scores.csv")
        save_table(admin_table, "glof_admin3_risk_scores.csv")

        print_subsection("Creating plots")
        plot_risk_category_counts(
            lake_category_counts,
            "Lake-level GLOF risk category counts",
            "lake_level_glof_risk_category_counts.png",
        )

        plot_risk_category_counts(
            admin_category_counts,
            "Admin3 GLOF risk category counts",
            "admin3_glof_risk_category_counts.png",
        )

        plot_score_histogram(
            lakes_risk,
            "glof_final_risk_score",
            "Lake-level final GLOF risk score distribution",
            "lake_level_glof_risk_score_distribution.png",
        )

        plot_score_histogram(
            admin3_risk,
            "glof_admin3_risk_score",
            "Admin3 GLOF risk score distribution",
            "admin3_glof_risk_score_distribution.png",
        )

        plot_top_lakes(lakes_risk)
        plot_top_admin3(admin3_risk)
        plot_lake_area_vs_risk(lakes_risk)
        plot_hazard_exposure_scatter(lakes_risk)

        print_subsection("Saving QGIS layers")
        lake_gpkg, admin_gpkg, lake_csv, admin_csv = save_qgis_layers(lakes_risk, admin3_risk)

        outputs = {
            "lake_gpkg": lake_gpkg,
            "admin3_gpkg": admin_gpkg,
            "lake_csv": lake_csv,
            "admin3_csv": admin_csv,
            "lake_summary": OUTPUT_FOLDERS["tables"] / "glof_lake_summary.csv",
            "admin3_summary": OUTPUT_FOLDERS["tables"] / "glof_admin3_summary.csv",
        }

        note_path = write_methodology_note(
            project_root,
            files,
            lake_summary,
            admin3_summary,
            lake_category_counts,
            admin_category_counts,
        )

        log_path = write_run_log(project_root, files, outputs)

        print_section("FINAL GLOF SUMMARY")

        print("\nLake summary:")
        print(lake_summary.to_string(index=False))

        print("\nAdmin3 summary:")
        print(admin3_summary.to_string(index=False))

        print("\nLake risk category counts:")
        print(lake_category_counts.to_string(index=False))

        print("\nAdmin3 risk category counts:")
        print(admin_category_counts.to_string(index=False))

        print_subsection("Main QGIS outputs")
        print(f"Lake-level GeoPackage: {lake_gpkg}")
        print(f"Admin3 GeoPackage: {admin_gpkg}")
        print(f"Lake-level CSV: {lake_csv}")
        print(f"Admin3 CSV: {admin_csv}")

        print_subsection("Logs")
        print(f"Methodology note: {note_path}")
        print(f"Run log: {log_path}")

        print("\nCompleted successfully.")
        print("Check Project_Outputs for GLOF tables, plots, QGIS layers, notes and logs.")

    except Exception as e:
        error_path = OUTPUT_FOLDERS["logs"] / "glof_risk_scoring_error_log.txt"

        with open(error_path, "w", encoding="utf-8") as f:
            f.write("GLOF_Risk_Scoring_Code.py failed.\n")
            f.write(str(e))
            f.write("\n\n")
            f.write(traceback.format_exc())

        print("\nERROR: The GLOF script failed.")
        print(f"Error log saved to: {error_path}")
        raise


if __name__ == "__main__":
    main()
