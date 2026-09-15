from pathlib import Path
from datetime import datetime
import traceback
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# Configuration

SCRIPT_DIR = Path(__file__).resolve().parent
METRIC_CRS = "EPSG:32645"

OUTPUT_ROOT = SCRIPT_DIR / "Project_Outputs"

INPUT_FOLDERS = {
    "ml_layers": OUTPUT_ROOT / "Final_QGIS_Layers" / "ML_Prediction_Layers",
    "glof_layers": OUTPUT_ROOT / "Final_QGIS_Layers" / "GLOF_Layers",
}

OUTPUT_FOLDERS = {
    "tables": OUTPUT_ROOT / "Final_Tables" / "Multi_Hazard_Tables",
    "plots": OUTPUT_ROOT / "Final_Plots" / "Multi_Hazard_Plots",
    "qgis_layers": OUTPUT_ROOT / "Final_QGIS_Layers" / "Multi_Hazard_Layers",
    "final_notes": OUTPUT_ROOT / "Final_Notes",
    "logs": OUTPUT_ROOT / "Working_Outputs" / "Logs",
}

RISK_ORDER = ["Low", "Medium", "High", "Very High"]

ADMIN_COLUMNS = {
    "adm1_name": ["adm1_name", "ADM1_EN", "ADM1_NAME", "Province", "province"],
    "adm2_name": ["adm2_name", "ADM2_EN", "ADM2_NAME", "District", "district"],
    "adm3_name": ["adm3_name", "ADM3_EN", "ADM3_NAME", "Municipality", "municipality"],
    "adm3_pcode": ["adm3_pcode", "ADM3_PCODE", "ADM3_CODE", "adm3_code", "pcode", "PCODE"],
}


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
        if (folder / "01_admin_boundary").exists() and (folder / "11_project_workspace").exists():
            return folder

    raise FileNotFoundError("Could not find Nepal_Early_Warning_Project root folder.")


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


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


# Find inputs

def discover_input_files(project_root):
    files = {}

    files["admin3_boundary"] = find_first_file(
        project_root / "01_admin_boundary",
        ["**/*admin3*.shp", "**/*adm3*.shp", "**/*ADM3*.shp"],
        required=True,
    )

    files["flood_predictions"] = find_first_file(
        INPUT_FOLDERS["ml_layers"],
        ["admin3_flood_risk_predictions_for_qgis.csv"],
        required=True,
    )

    files["landslide_predictions"] = find_first_file(
        INPUT_FOLDERS["ml_layers"],
        ["admin3_landslide_risk_predictions_for_qgis.csv"],
        required=True,
    )

    files["glof_admin3_predictions"] = find_first_file(
        INPUT_FOLDERS["glof_layers"],
        ["glof_admin3_risk_scores_for_qgis.csv", "glof_admin3_risk_scores.csv"],
        required=True,
    )

    return files


def create_input_inventory(files):
    rows = []

    for name, path in files.items():
        rows.append({
            "input_name": name,
            "path": str(path),
            "exists": Path(path).exists(),
        })

    return pd.DataFrame(rows)


# Load data

def load_admin3_boundary(path):
    admin3 = gpd.read_file(path, engine="pyogrio")
    admin3 = admin3[admin3.geometry.notna()].copy()

    if admin3.crs is None:
        admin3 = admin3.set_crs("EPSG:4326")

    admin3 = admin3.to_crs("EPSG:4326")
    admin3 = standardise_admin_columns(admin3)

    keep_cols = ["adm1_name", "adm2_name", "adm3_name", "adm3_pcode", "geometry"]
    admin3 = admin3[keep_cols].copy()

    admin3["adm3_pcode"] = admin3["adm3_pcode"].astype(str)
    admin3["admin3_area_km2"] = admin3.to_crs(METRIC_CRS).geometry.area / 1_000_000

    return admin3


def load_flood_predictions(path):
    df = pd.read_csv(path, low_memory=False)
    df["adm3_pcode"] = df["adm3_pcode"].astype(str)

    needed = [
        "adm3_pcode",
        "flood_risk_probability",
        "flood_predicted_class",
        "flood_risk_category",
        "flood_risk_category_order",
    ]

    existing = [col for col in needed if col in df.columns]
    df = df[existing].copy()

    df["flood_score"] = safe_numeric(df["flood_risk_probability"]).fillna(0).clip(0, 1)
    df["flood_category"] = df["flood_score"].apply(classify_risk)
    df["flood_category_order"] = df["flood_category"].apply(risk_category_order)

    return df[[
        "adm3_pcode",
        "flood_score",
        "flood_category",
        "flood_category_order",
    ]]


def load_landslide_predictions(path):
    df = pd.read_csv(path, low_memory=False)
    df["adm3_pcode"] = df["adm3_pcode"].astype(str)

    needed = [
        "adm3_pcode",
        "landslide_risk_probability",
        "landslide_predicted_class",
        "landslide_risk_category",
        "landslide_risk_category_order",
    ]

    existing = [col for col in needed if col in df.columns]
    df = df[existing].copy()

    df["landslide_score"] = safe_numeric(df["landslide_risk_probability"]).fillna(0).clip(0, 1)
    df["landslide_category"] = df["landslide_score"].apply(classify_risk)
    df["landslide_category_order"] = df["landslide_category"].apply(risk_category_order)

    return df[[
        "adm3_pcode",
        "landslide_score",
        "landslide_category",
        "landslide_category_order",
    ]]


def load_glof_predictions(path):
    df = pd.read_csv(path, low_memory=False)
    df["adm3_pcode"] = df["adm3_pcode"].astype(str)

    score_col = None

    for candidate in ["glof_admin3_risk_score", "glof_risk_score", "glof_final_risk_score"]:
        if candidate in df.columns:
            score_col = candidate
            break

    if score_col is None:
        raise ValueError("Could not find GLOF admin3 risk score column.")

    category_col = None

    for candidate in ["glof_admin3_risk_category", "glof_risk_category"]:
        if candidate in df.columns:
            category_col = candidate
            break

    keep_cols = ["adm3_pcode", score_col]

    optional_cols = [
        "glof_lake_count",
        "glof_total_lake_area_km2",
        "glof_max_lake_area_km2",
        "glof_pdl_lake_count",
        "glof_mean_hazard_score",
        "glof_max_hazard_score",
        "glof_mean_final_lake_risk_score",
        "glof_max_final_lake_risk_score",
        "exposure_score",
    ]

    for col in optional_cols:
        if col in df.columns:
            keep_cols.append(col)

    if category_col is not None:
        keep_cols.append(category_col)

    df = df[keep_cols].copy()

    df["glof_score"] = safe_numeric(df[score_col]).fillna(0).clip(0, 1)
    df["glof_category"] = df["glof_score"].apply(classify_risk)
    df["glof_category_order"] = df["glof_category"].apply(risk_category_order)

    rename_map = {
        "exposure_score": "glof_exposure_score",
    }

    df = df.rename(columns=rename_map)

    final_cols = [
        "adm3_pcode",
        "glof_score",
        "glof_category",
        "glof_category_order",
    ]

    for col in optional_cols:
        renamed_col = rename_map.get(col, col)
        if renamed_col in df.columns:
            final_cols.append(renamed_col)

    return df[final_cols]


# Multi-hazard scoring

def dominant_hazard(row):
    scores = {
        "Flood": row["flood_score"],
        "Landslide": row["landslide_score"],
        "GLOF": row["glof_score"],
    }

    return max(scores, key=scores.get)


def calculate_multi_hazard_scores(admin3_merged):
    df = admin3_merged.copy()

    for col in ["flood_score", "landslide_score", "glof_score"]:
        df[col] = safe_numeric(df[col]).fillna(0).clip(0, 1)

    df["multi_hazard_mean_score"] = df[["flood_score", "landslide_score", "glof_score"]].mean(axis=1)
    df["multi_hazard_max_score"] = df[["flood_score", "landslide_score", "glof_score"]].max(axis=1)

    # Final warning-focused score: high if any hazard is high
    df["multi_hazard_priority_score"] = df["multi_hazard_max_score"]

    df["multi_hazard_category"] = df["multi_hazard_priority_score"].apply(classify_risk)
    df["multi_hazard_category_order"] = df["multi_hazard_category"].apply(risk_category_order)

    df["dominant_hazard"] = df.apply(dominant_hazard, axis=1)

    df["flood_high_flag"] = df["flood_score"].ge(0.50).astype(int)
    df["landslide_high_flag"] = df["landslide_score"].ge(0.50).astype(int)
    df["glof_high_flag"] = df["glof_score"].ge(0.50).astype(int)

    df["number_of_high_hazards"] = (
        df["flood_high_flag"] +
        df["landslide_high_flag"] +
        df["glof_high_flag"]
    )

    df["multi_hazard_type"] = np.select(
        [
            df["number_of_high_hazards"] == 0,
            df["number_of_high_hazards"] == 1,
            df["number_of_high_hazards"] == 2,
            df["number_of_high_hazards"] >= 3,
        ],
        [
            "No high hazard",
            "Single high hazard",
            "Two high hazards",
            "Three high hazards",
        ],
        default="No high hazard",
    )

    return df


# Summary tables

def risk_count_table(df, hazard_name, category_col):
    counts = (
        df[category_col]
        .value_counts()
        .reindex(RISK_ORDER, fill_value=0)
        .reset_index()
    )

    counts.columns = ["risk_category", "count"]
    counts.insert(0, "hazard", hazard_name)

    return counts


def create_all_risk_counts(df):
    tables = [
        risk_count_table(df, "Flood", "flood_category"),
        risk_count_table(df, "Landslide", "landslide_category"),
        risk_count_table(df, "GLOF", "glof_category"),
        risk_count_table(df, "Multi-hazard", "multi_hazard_category"),
    ]

    return pd.concat(tables, ignore_index=True)


def create_summary_table(df):
    rows = [{
        "total_admin3_units": len(df),
        "mean_flood_score": round(df["flood_score"].mean(), 4),
        "mean_landslide_score": round(df["landslide_score"].mean(), 4),
        "mean_glof_score": round(df["glof_score"].mean(), 4),
        "mean_multi_hazard_priority_score": round(df["multi_hazard_priority_score"].mean(), 4),
        "max_multi_hazard_priority_score": round(df["multi_hazard_priority_score"].max(), 4),
        "high_or_very_high_multi_hazard_units": int(df["multi_hazard_category"].isin(["High", "Very High"]).sum()),
        "two_or_more_high_hazard_units": int((df["number_of_high_hazards"] >= 2).sum()),
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }]

    return pd.DataFrame(rows)


def create_province_summary(df):
    summary = (
        df
        .groupby("adm1_name", dropna=False)
        .agg(
            admin3_units=("adm3_pcode", "count"),
            mean_flood_score=("flood_score", "mean"),
            mean_landslide_score=("landslide_score", "mean"),
            mean_glof_score=("glof_score", "mean"),
            mean_multi_hazard_priority_score=("multi_hazard_priority_score", "mean"),
            max_multi_hazard_priority_score=("multi_hazard_priority_score", "max"),
            high_multi_hazard_units=("multi_hazard_category", lambda x: x.isin(["High", "Very High"]).sum()),
            two_or_more_high_hazard_units=("number_of_high_hazards", lambda x: (x >= 2).sum()),
        )
        .reset_index()
    )

    return summary.sort_values("mean_multi_hazard_priority_score", ascending=False)


def create_district_summary(df):
    summary = (
        df
        .groupby(["adm1_name", "adm2_name"], dropna=False)
        .agg(
            admin3_units=("adm3_pcode", "count"),
            mean_flood_score=("flood_score", "mean"),
            mean_landslide_score=("landslide_score", "mean"),
            mean_glof_score=("glof_score", "mean"),
            mean_multi_hazard_priority_score=("multi_hazard_priority_score", "mean"),
            max_multi_hazard_priority_score=("multi_hazard_priority_score", "max"),
            high_multi_hazard_units=("multi_hazard_category", lambda x: x.isin(["High", "Very High"]).sum()),
            two_or_more_high_hazard_units=("number_of_high_hazards", lambda x: (x >= 2).sum()),
        )
        .reset_index()
    )

    return summary.sort_values("mean_multi_hazard_priority_score", ascending=False)


def create_merge_quality_table(admin3, flood, landslide, glof, merged):
    rows = [{
        "admin3_boundary_rows": len(admin3),
        "flood_prediction_rows": len(flood),
        "landslide_prediction_rows": len(landslide),
        "glof_prediction_rows": len(glof),
        "final_merged_rows": len(merged),
        "missing_flood_score": int(merged["flood_score"].isna().sum()),
        "missing_landslide_score": int(merged["landslide_score"].isna().sum()),
        "missing_glof_score": int(merged["glof_score"].isna().sum()),
    }]

    return pd.DataFrame(rows)


# Plots

def plot_all_risk_counts(risk_counts):
    fig, ax = plt.subplots(figsize=(11, 6))

    sns.barplot(
        data=risk_counts,
        x="hazard",
        y="count",
        hue="risk_category",
        hue_order=RISK_ORDER,
        ax=ax,
    )

    ax.set_title("Risk category counts by hazard")
    ax.set_xlabel("Hazard")
    ax.set_ylabel("Number of admin3 municipalities")
    ax.legend(title="Risk category", bbox_to_anchor=(1.02, 1), loc="upper left")

    return save_plot(fig, "risk_category_counts_by_hazard.png")


def plot_dominant_hazard_counts(df):
    counts = df["dominant_hazard"].value_counts().reset_index()
    counts.columns = ["dominant_hazard", "count"]

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.barplot(
        data=counts,
        x="dominant_hazard",
        y="count",
        ax=ax,
    )

    ax.set_title("Dominant hazard by admin3 municipality")
    ax.set_xlabel("Dominant hazard")
    ax.set_ylabel("Number of admin3 municipalities")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3)

    return save_plot(fig, "dominant_hazard_counts.png")


def plot_multi_hazard_type_counts(df):
    order = ["No high hazard", "Single high hazard", "Two high hazards", "Three high hazards"]

    counts = (
        df["multi_hazard_type"]
        .value_counts()
        .reindex(order, fill_value=0)
        .reset_index()
    )

    counts.columns = ["multi_hazard_type", "count"]

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.barplot(
        data=counts,
        x="multi_hazard_type",
        y="count",
        order=order,
        ax=ax,
    )

    ax.set_title("Number of high hazards per admin3 municipality")
    ax.set_xlabel("Multi-hazard type")
    ax.set_ylabel("Number of admin3 municipalities")
    plt.xticks(rotation=20, ha="right")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3)

    return save_plot(fig, "multi_hazard_type_counts.png")


def plot_top_admin3(df):
    top = df.sort_values("multi_hazard_priority_score", ascending=False).head(20).copy()

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.barplot(
        data=top,
        y="adm3_name",
        x="multi_hazard_priority_score",
        ax=ax,
    )

    ax.set_title("Top 20 admin3 municipalities by multi-hazard priority score")
    ax.set_xlabel("Multi-hazard priority score")
    ax.set_ylabel("Admin3 municipality")

    return save_plot(fig, "top_20_admin3_multi_hazard_priority.png")


def plot_score_distributions(df):
    plot_df = df[[
        "flood_score",
        "landslide_score",
        "glof_score",
        "multi_hazard_priority_score",
    ]].copy()

    plot_df = plot_df.rename(columns={
        "flood_score": "Flood",
        "landslide_score": "Landslide",
        "glof_score": "GLOF",
        "multi_hazard_priority_score": "Multi-hazard",
    })

    long_df = plot_df.melt(var_name="hazard", value_name="score")

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.kdeplot(
        data=long_df,
        x="score",
        hue="hazard",
        fill=False,
        common_norm=False,
        ax=ax,
    )

    ax.set_title("Risk score distributions")
    ax.set_xlabel("Risk score")
    ax.set_ylabel("Density")

    return save_plot(fig, "risk_score_distributions.png")


def plot_flood_vs_landslide(df):
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.scatterplot(
        data=df,
        x="flood_score",
        y="landslide_score",
        hue="multi_hazard_category",
        hue_order=RISK_ORDER,
        ax=ax,
    )

    ax.set_title("Flood score vs landslide score")
    ax.set_xlabel("Flood risk score")
    ax.set_ylabel("Landslide risk score")
    ax.legend(title="Multi-hazard category", bbox_to_anchor=(1.02, 1), loc="upper left")

    return save_plot(fig, "flood_score_vs_landslide_score.png")


def plot_glof_vs_multi_hazard(df):
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.scatterplot(
        data=df,
        x="glof_score",
        y="multi_hazard_priority_score",
        hue="dominant_hazard",
        ax=ax,
    )

    ax.set_title("GLOF score vs multi-hazard priority score")
    ax.set_xlabel("GLOF risk score")
    ax.set_ylabel("Multi-hazard priority score")
    ax.legend(title="Dominant hazard", bbox_to_anchor=(1.02, 1), loc="upper left")

    return save_plot(fig, "glof_score_vs_multi_hazard_priority_score.png")


# Save QGIS layer

def save_qgis_outputs(final_gdf):
    qgis_gpkg = OUTPUT_FOLDERS["qgis_layers"] / "admin3_multi_hazard_risk_scores.gpkg"
    qgis_csv = OUTPUT_FOLDERS["qgis_layers"] / "admin3_multi_hazard_risk_scores_for_qgis.csv"

    if qgis_gpkg.exists():
        qgis_gpkg.unlink()

    final_gdf.to_file(qgis_gpkg, layer="admin3_multi_hazard_risk_scores", driver="GPKG")
    final_gdf.drop(columns="geometry", errors="ignore").to_csv(qgis_csv, index=False, encoding="utf-8-sig")

    return qgis_gpkg, qgis_csv


# Methodology note

def write_methodology_note(project_root, files, summary_table):
    note_path = OUTPUT_FOLDERS["final_notes"] / "multi_hazard_integration_methodology_note.txt"

    note = f"""Multi-hazard integration methodology note

Project:
Machine Learning and GIS-Based Early Warning Prototype for Multi-Hazard Risk in Nepal

Purpose:
This script combines the final admin3-level outputs from flood machine learning, landslide machine learning and GLOF GIS risk scoring into one multi-hazard risk layer for QGIS mapping and prototype communication.

Input outputs used:
- Flood ML risk predictions
- Landslide ML risk predictions
- GLOF GIS risk scores
- Nepal admin3 boundary layer

Risk score handling:
Flood and landslide use model probability scores from the supervised ML workflow.
GLOF uses the GIS-based admin3 risk score.

Multi-hazard score:
The final multi-hazard priority score is calculated using the maximum of the three hazard scores:
max(flood score, landslide score, GLOF score)

Reason:
This is suitable for a warning-focused prototype because a municipality should be prioritised if any one hazard has high risk, even if the other hazards are lower.

Additional outputs:
The script also saves the mean score across hazards, dominant hazard, number of high hazards, and multi-hazard type.

Risk categories:
- Low: 0.00 to 0.25
- Medium: 0.25 to 0.50
- High: 0.50 to 0.75
- Very High: 0.75 to 1.00

Important interpretation:
This is a research prototype output. It should not be presented as an operational government warning system.

Project root:
{project_root}

Input files:
{create_input_inventory(files).to_string(index=False)}

Summary:
{summary_table.to_string(index=False)}
"""

    with open(note_path, "w", encoding="utf-8") as f:
        f.write(note)

    return note_path


def write_run_log(project_root, files, outputs):
    log_path = OUTPUT_FOLDERS["logs"] / "multi_hazard_integration_run_log.txt"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Multi_Hazard_Integration_Code.py run log\n")
        f.write(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Project root: {project_root}\n\n")

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

    print_section("Multi-Hazard Integration")
    print("Combining Flood ML, Landslide ML and GLOF GIS risk into one admin3 layer.")

    try:
        project_root = find_project_root(SCRIPT_DIR)
        files = discover_input_files(project_root)

        inventory = create_input_inventory(files)

        print_subsection("Input files")
        print(inventory.to_string(index=False))
        save_table(inventory, "multi_hazard_input_file_inventory.csv")

        print_subsection("Loading data")
        admin3 = load_admin3_boundary(files["admin3_boundary"])
        flood = load_flood_predictions(files["flood_predictions"])
        landslide = load_landslide_predictions(files["landslide_predictions"])
        glof = load_glof_predictions(files["glof_admin3_predictions"])

        print(f"Admin3 boundary rows: {len(admin3)}")
        print(f"Flood rows: {len(flood)}")
        print(f"Landslide rows: {len(landslide)}")
        print(f"GLOF rows: {len(glof)}")

        print_subsection("Merging hazard layers")
        merged = admin3.merge(flood, on="adm3_pcode", how="left")
        merged = merged.merge(landslide, on="adm3_pcode", how="left")
        merged = merged.merge(glof, on="adm3_pcode", how="left")

        quality_table = create_merge_quality_table(admin3, flood, landslide, glof, merged)
        save_table(quality_table, "multi_hazard_merge_quality_check.csv")
        print(quality_table.to_string(index=False))

        print_subsection("Calculating multi-hazard scores")
        final_gdf = calculate_multi_hazard_scores(merged)

        risk_counts = create_all_risk_counts(final_gdf)
        summary_table = create_summary_table(final_gdf)
        province_summary = create_province_summary(final_gdf.drop(columns="geometry", errors="ignore"))
        district_summary = create_district_summary(final_gdf.drop(columns="geometry", errors="ignore"))

        top_30 = final_gdf.drop(columns="geometry", errors="ignore").sort_values(
            "multi_hazard_priority_score",
            ascending=False,
        ).head(30)

        high_priority = final_gdf.drop(columns="geometry", errors="ignore")
        high_priority = high_priority[
            high_priority["multi_hazard_category"].isin(["High", "Very High"])
        ].sort_values("multi_hazard_priority_score", ascending=False)

        save_table(summary_table, "multi_hazard_summary.csv")
        save_table(risk_counts, "multi_hazard_risk_category_counts.csv")
        save_table(province_summary, "multi_hazard_province_summary.csv")
        save_table(district_summary, "multi_hazard_district_summary.csv")
        save_table(top_30, "top_30_admin3_multi_hazard_priority.csv")
        save_table(high_priority, "high_priority_admin3_municipalities.csv")
        save_table(final_gdf.drop(columns="geometry", errors="ignore"), "admin3_multi_hazard_risk_scores.csv")

        print_subsection("Creating plots")
        plot_all_risk_counts(risk_counts)
        plot_dominant_hazard_counts(final_gdf)
        plot_multi_hazard_type_counts(final_gdf)
        plot_top_admin3(final_gdf)
        plot_score_distributions(final_gdf)
        plot_flood_vs_landslide(final_gdf)
        plot_glof_vs_multi_hazard(final_gdf)

        print_subsection("Saving QGIS outputs")
        qgis_gpkg, qgis_csv = save_qgis_outputs(final_gdf)

        outputs = {
            "qgis_gpkg": qgis_gpkg,
            "qgis_csv": qgis_csv,
            "summary_table": OUTPUT_FOLDERS["tables"] / "multi_hazard_summary.csv",
            "risk_counts": OUTPUT_FOLDERS["tables"] / "multi_hazard_risk_category_counts.csv",
            "top_30": OUTPUT_FOLDERS["tables"] / "top_30_admin3_multi_hazard_priority.csv",
            "high_priority": OUTPUT_FOLDERS["tables"] / "high_priority_admin3_municipalities.csv",
        }

        note_path = write_methodology_note(project_root, files, summary_table)
        log_path = write_run_log(project_root, files, outputs)

        print_section("FINAL MULTI-HAZARD SUMMARY")
        print(summary_table.to_string(index=False))

        print("\nRisk category counts:")
        print(risk_counts.to_string(index=False))

        print("\nTop 10 admin3 municipalities by multi-hazard priority:")
        print(top_30[[
            "adm1_name",
            "adm2_name",
            "adm3_name",
            "flood_score",
            "landslide_score",
            "glof_score",
            "multi_hazard_priority_score",
            "multi_hazard_category",
            "dominant_hazard",
            "number_of_high_hazards",
        ]].head(10).to_string(index=False))

        print_subsection("Main QGIS outputs")
        print(f"GeoPackage: {qgis_gpkg}")
        print(f"CSV: {qgis_csv}")

        print_subsection("Logs")
        print(f"Methodology note: {note_path}")
        print(f"Run log: {log_path}")

        print("\nCompleted successfully.")
        print("Check Project_Outputs for multi-hazard tables, plots, QGIS layers, notes and logs.")

    except Exception as e:
        error_path = OUTPUT_FOLDERS["logs"] / "multi_hazard_integration_error_log.txt"

        with open(error_path, "w", encoding="utf-8") as f:
            f.write("Multi_Hazard_Integration_Code.py failed.\n")
            f.write(str(e))
            f.write("\n\n")
            f.write(traceback.format_exc())

        print("\nERROR: The multi-hazard integration script failed.")
        print(f"Error log saved to: {error_path}")
        raise

if __name__ == "__main__":
    main()
