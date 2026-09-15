from pathlib import Path
from datetime import datetime
import json
import traceback
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from xgboost import XGBClassifier
import joblib


# Configuration

PROJECT_DIR = Path(__file__).resolve().parent

FLOOD_FILE = PROJECT_DIR / "admin3_flood_ml_dataset.csv"
LANDSLIDE_FILE = PROJECT_DIR / "admin3_landslide_ml_dataset.csv"
PREDICTOR_JSON = PROJECT_DIR / "ml_predictor_lists.json"

ID_COLUMNS = ["adm3_pcode", "adm1_name", "adm2_name", "adm3_name"]

TASKS = {
    "flood": {
        "dataset_path": FLOOD_FILE,
        "target_col": "flood_label",
        "display_name": "Flood Risk Mapping",
    },
    "landslide": {
        "dataset_path": LANDSLIDE_FILE,
        "target_col": "landslide_label",
        "display_name": "Landslide Susceptibility Mapping",
    },
}

TEST_SIZE = 0.30
RANDOM_STATE = 42
CV_SPLITS = 5

REMOVE_HIGH_CORRELATION = True
HIGH_CORRELATION_THRESHOLD = 0.95

TOP_FEATURES_FOR_INDIVIDUAL_PLOTS = 15
TOP_FEATURES_FOR_CORRELATION_PLOT = 20
TOP_FEATURES_FOR_IMPORTANCE = 20
MAX_BIVARIATE_SCATTER_PLOTS = 12

OUTPUT_ROOT = PROJECT_DIR / "Project_Outputs"

OUTPUT_FOLDERS = {
    "tables": OUTPUT_ROOT / "Final_Tables" / "Data_Summary_Tables",
    "metrics": OUTPUT_ROOT / "Final_Tables" / "ML_Model_Tables",
    "plots": OUTPUT_ROOT / "Final_Plots" / "ML_Plots",
    "feature_importance": OUTPUT_ROOT / "Final_Plots" / "ML_Feature_Importance_Plots",
    "models": OUTPUT_ROOT / "Final_Models",
    "predictions_for_mapping": OUTPUT_ROOT / "Final_QGIS_Layers" / "ML_Prediction_Layers",
    "final_notes": OUTPUT_ROOT / "Final_Notes",
    "train_test_data": OUTPUT_ROOT / "Working_Outputs" / "Train_Test_Data",
    "logs": OUTPUT_ROOT / "Working_Outputs" / "Logs",
}

PLOT_FOLDERS = {
    "class_balance": OUTPUT_FOLDERS["plots"] / "class_balance",
    "histograms": OUTPUT_FOLDERS["plots"] / "histograms",
    "boxplots": OUTPUT_FOLDERS["plots"] / "boxplots",
    "qqplots": OUTPUT_FOLDERS["plots"] / "qqplots",
    "correlation": OUTPUT_FOLDERS["plots"] / "correlation",
    "pca": OUTPUT_FOLDERS["plots"] / "pca",
    "bivariate": OUTPUT_FOLDERS["plots"] / "bivariate",
    "model_evaluation": OUTPUT_FOLDERS["plots"] / "model_evaluation",
    "risk_categories": OUTPUT_FOLDERS["plots"] / "risk_categories",
}


# General helper functions

def create_output_folders():
    for folder in OUTPUT_FOLDERS.values():
        folder.mkdir(parents=True, exist_ok=True)

    for folder in PLOT_FOLDERS.values():
        folder.mkdir(parents=True, exist_ok=True)


def setup_plot_style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9


def print_section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_subsection(title):
    print("\n" + "-" * 90)
    print(title)
    print("-" * 90)


def save_table(df, folder_key, filename):
    path = OUTPUT_FOLDERS[folder_key] / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_plot(fig, plot_group, filename):
    path = PLOT_FOLDERS[plot_group] / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_feature_importance_plot(fig, filename):
    path = OUTPUT_FOLDERS["feature_importance"] / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def safe_name(text):
    return (
        str(text)
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "percent")
        .replace(".", "_")
    )


def clean_label(text):
    return str(text).replace("_", " ")


def load_predictor_list():
    if not PREDICTOR_JSON.exists():
        raise FileNotFoundError(f"Predictor JSON not found: {PREDICTOR_JSON}")

    with open(PREDICTOR_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "admin3" not in data:
        raise ValueError("The predictor JSON does not contain an admin3 predictor list.")

    predictors = data["admin3"]

    if not isinstance(predictors, list) or len(predictors) == 0:
        raise ValueError("The admin3 predictor list is empty or invalid.")

    return predictors


def get_id_columns(df):
    return [col for col in ID_COLUMNS if col in df.columns]


def risk_category(probability):
    if probability < 0.25:
        return "Low"
    if probability < 0.50:
        return "Medium"
    if probability < 0.75:
        return "High"
    return "Very High"


def risk_order():
    return ["Low", "Medium", "High", "Very High"]

# Data loading and summary

def load_task_dataset(task_key, dataset_path, target_col, predictor_cols):
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path, low_memory=False)

    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)

    available_predictors = [col for col in predictor_cols if col in df.columns]
    missing_predictors = [col for col in predictor_cols if col not in df.columns]

    if len(available_predictors) == 0:
        raise ValueError(f"No predictors found for {task_key}.")

    if missing_predictors:
        missing_df = pd.DataFrame({
            "task": task_key,
            "missing_predictor": missing_predictors,
        })
        save_table(missing_df, "logs", f"{task_key}_missing_predictors_from_json.csv")

    for col in available_predictors:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df, available_predictors, missing_predictors


def dataset_summary(task_key, df, target_col, predictors):
    total = len(df)
    positive = int((df[target_col] == 1).sum())
    negative = int((df[target_col] == 0).sum())

    return pd.DataFrame([{
        "task": task_key,
        "rows": total,
        "columns": len(df.columns),
        "target_column": target_col,
        "predictor_count_available": len(predictors),
        "positive_units": positive,
        "negative_units": negative,
        "positive_percent": round(positive / total * 100, 3),
        "negative_percent": round(negative / total * 100, 3),
        "total_missing_values": int(df.isna().sum().sum()),
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }])


def class_balance_table(task_key, df, target_col):
    counts = df[target_col].value_counts().sort_index()
    rows = []

    for class_value, count in counts.items():
        rows.append({
            "task": task_key,
            "class_value": int(class_value),
            "class_name": "No recorded event" if class_value == 0 else "Recorded event",
            "count": int(count),
            "percent": round(float(count / len(df) * 100), 3),
        })

    return pd.DataFrame(rows)


def missing_value_table(task_key, df, predictors):
    rows = []

    for col in df.columns:
        missing_count = int(df[col].isna().sum())

        rows.append({
            "task": task_key,
            "column": col,
            "missing_count": missing_count,
            "missing_percent": round(float(missing_count / len(df) * 100), 3),
            "is_predictor": col in predictors,
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(["missing_count", "column"], ascending=[False, True])
    return out


def descriptive_statistics_table(task_key, df, predictors):
    stats_df = df[predictors].describe().T.reset_index()
    stats_df = stats_df.rename(columns={"index": "feature"})
    stats_df.insert(0, "task", task_key)
    return stats_df


def target_correlation_table(task_key, df, target_col, predictors):
    rows = []

    for feature in predictors:
        values = pd.to_numeric(df[feature], errors="coerce")

        if values.nunique(dropna=True) <= 1:
            corr_value = np.nan
        else:
            corr_value = values.corr(df[target_col])

        rows.append({
            "task": task_key,
            "feature": feature,
            "correlation_with_target": corr_value,
            "absolute_correlation_with_target": abs(corr_value) if pd.notna(corr_value) else np.nan,
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(
        "absolute_correlation_with_target",
        ascending=False,
        na_position="last",
    )
    return out


def high_correlation_pairs_table(task_key, df, predictors, threshold=0.80):
    corr = df[predictors].corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    rows = []

    for feature_1 in upper.index:
        for feature_2 in upper.columns:
            value = upper.loc[feature_1, feature_2]

            if pd.notna(value) and value >= threshold:
                rows.append({
                    "task": task_key,
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "absolute_correlation": round(float(value), 5),
                })

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values("absolute_correlation", ascending=False)

    return out


def feature_distribution_table(task_key, df, predictors):
    rows = []

    for feature in predictors:
        values = pd.to_numeric(df[feature], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).dropna()

        if len(values) == 0:
            continue

        rows.append({
            "task": task_key,
            "feature": feature,
            "count": int(len(values)),
            "missing_count": int(df[feature].isna().sum()),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std()) if len(values) > 1 else np.nan,
            "min": float(values.min()),
            "max": float(values.max()),
            "skewness": float(values.skew()) if len(values) > 2 else np.nan,
            "kurtosis": float(values.kurtosis()) if len(values) > 3 else np.nan,
            "unique_values": int(values.nunique()),
        })

    out = pd.DataFrame(rows)
    return out


# Feature selection for EDA plots

def select_eda_features(task_key, df, predictors, target_corr_df):
    top_by_target = (
        target_corr_df
        .dropna(subset=["absolute_correlation_with_target"])
        .head(TOP_FEATURES_FOR_INDIVIDUAL_PLOTS)["feature"]
        .tolist()
    )

    dist_df = feature_distribution_table(task_key, df, predictors)

    if dist_df.empty:
        top_by_skew = top_by_target
        top_by_variation = top_by_target
    else:
        top_by_skew = (
            dist_df
            .dropna(subset=["skewness"])
            .assign(abs_skewness=lambda x: x["skewness"].abs())
            .sort_values("abs_skewness", ascending=False)
            .head(TOP_FEATURES_FOR_INDIVIDUAL_PLOTS)["feature"]
            .tolist()
        )

        top_by_variation = (
            dist_df
            .dropna(subset=["std"])
            .sort_values("std", ascending=False)
            .head(TOP_FEATURES_FOR_INDIVIDUAL_PLOTS)["feature"]
            .tolist()
        )

    if len(top_by_skew) < TOP_FEATURES_FOR_INDIVIDUAL_PLOTS:
        top_by_skew = list(dict.fromkeys(top_by_skew + top_by_target))[:TOP_FEATURES_FOR_INDIVIDUAL_PLOTS]

    if len(top_by_variation) < TOP_FEATURES_FOR_INDIVIDUAL_PLOTS:
        top_by_variation = list(dict.fromkeys(top_by_variation + top_by_target))[:TOP_FEATURES_FOR_INDIVIDUAL_PLOTS]

    rows = []

    for plot_type, features in {
        "histogram": top_by_target,
        "boxplot": top_by_target,
        "qqplot": top_by_skew,
        "bivariate_candidate": top_by_target,
        "high_variation_candidate": top_by_variation,
    }.items():
        for rank, feature in enumerate(features, start=1):
            rows.append({
                "task": task_key,
                "plot_type": plot_type,
                "rank": rank,
                "feature": feature,
            })

    selection_df = pd.DataFrame(rows)

    return top_by_target, top_by_target, top_by_skew, top_by_variation, selection_df


def select_bivariate_pairs(task_key, df, target_corr_df):
    top_features = (
        target_corr_df
        .dropna(subset=["absolute_correlation_with_target"])
        .head(12)["feature"]
        .tolist()
    )

    pairs = []

    if len(top_features) >= 2:
        anchor = top_features[0]

        for feature in top_features[1:]:
            pairs.append((anchor, feature, "top_feature_against_other_top_features"))

    if len(top_features) >= 4:
        for i in range(1, len(top_features) - 1, 2):
            pairs.append((top_features[i], top_features[i + 1], "paired_top_features"))

    clean_pairs = []
    seen = set()

    for x_feature, y_feature, reason in pairs:
        if x_feature == y_feature:
            continue

        key = tuple(sorted([x_feature, y_feature]))

        if key in seen:
            continue

        seen.add(key)
        clean_pairs.append((x_feature, y_feature, reason))

        if len(clean_pairs) >= MAX_BIVARIATE_SCATTER_PLOTS:
            break

    rows = []

    for rank, (x_feature, y_feature, reason) in enumerate(clean_pairs, start=1):
        rows.append({
            "task": task_key,
            "rank": rank,
            "x_feature": x_feature,
            "y_feature": y_feature,
            "selection_reason": reason,
        })

    return clean_pairs, pd.DataFrame(rows)


# EDA plots

def plot_class_balance(task_key, class_df):
    plot_df = class_df.copy()
    plot_df["class_label"] = plot_df["class_value"].map({
        0: "Negative",
        1: "Positive",
    })

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.barplot(
        data=plot_df,
        x="class_label",
        y="count",
        ax=ax,
    )

    ax.set_title(f"{task_key.title()} class balance")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of admin3 municipalities")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3)

    return save_plot(fig, "class_balance", f"{task_key}_class_balance.png")


def plot_missing_values_if_needed(task_key, missing_df):
    plot_df = missing_df[missing_df["missing_count"] > 0].head(20).copy()

    if plot_df.empty:
        print(f"{task_key}: No missing values found.")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=plot_df,
        y="column",
        x="missing_count",
        ax=ax,
    )

    ax.set_title(f"{task_key.title()} missing values")
    ax.set_xlabel("Missing count")
    ax.set_ylabel("Column")

    return save_plot(fig, "class_balance", f"{task_key}_missing_values.png")


def plot_target_correlations(task_key, corr_df):
    plot_df = corr_df.dropna().head(TOP_FEATURES_FOR_CORRELATION_PLOT).copy()

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.barplot(
        data=plot_df,
        y="feature",
        x="correlation_with_target",
        ax=ax,
    )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"{task_key.title()} top feature correlations with target")
    ax.set_xlabel("Correlation with target")
    ax.set_ylabel("Feature")

    return save_plot(fig, "correlation", f"{task_key}_top_feature_target_correlations.png")


def plot_correlation_heatmap(task_key, df, target_col, corr_df):
    top_features = corr_df.dropna().head(TOP_FEATURES_FOR_CORRELATION_PLOT)["feature"].tolist()
    heatmap_columns = [target_col] + top_features

    corr = df[heatmap_columns].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(13, 10))

    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        annot=False,
        linewidths=0.3,
        cbar_kws={"label": "Correlation"},
        ax=ax,
    )

    ax.set_title(f"{task_key.title()} correlation heatmap")

    return save_plot(fig, "correlation", f"{task_key}_correlation_heatmap.png")


def plot_individual_histograms(task_key, df, features):
    saved = []

    for i, feature in enumerate(features, start=1):
        values = pd.to_numeric(df[feature], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).dropna()

        if len(values) < 2:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.histplot(
            values,
            kde=True,
            bins=30,
            ax=ax,
        )

        ax.axvline(values.median(), linestyle="--", linewidth=1.2, label="Median")
        ax.set_title(f"{task_key.title()} histogram: {clean_label(feature)}")
        ax.set_xlabel(clean_label(feature))
        ax.set_ylabel("Frequency")
        ax.legend()

        path = save_plot(
            fig,
            "histograms",
            f"{task_key}_histogram_{i:02d}_{safe_name(feature)}.png",
        )
        saved.append(str(path))

    return saved


def plot_individual_boxplots(task_key, df, target_col, features):
    saved = []

    for i, feature in enumerate(features, start=1):
        plot_df = df[[target_col, feature]].copy()
        plot_df[feature] = pd.to_numeric(plot_df[feature], errors="coerce")
        plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()

        if plot_df.empty:
            continue

        plot_df["target_class"] = plot_df[target_col].map({
            0: "Negative",
            1: "Positive",
        })

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.boxplot(
            data=plot_df,
            x="target_class",
            y=feature,
            ax=ax,
        )

        sns.stripplot(
            data=plot_df,
            x="target_class",
            y=feature,
            ax=ax,
            alpha=0.25,
            size=2,
        )

        ax.set_title(f"{task_key.title()} boxplot: {clean_label(feature)}")
        ax.set_xlabel("Target class")
        ax.set_ylabel(clean_label(feature))

        path = save_plot(
            fig,
            "boxplots",
            f"{task_key}_boxplot_{i:02d}_{safe_name(feature)}.png",
        )
        saved.append(str(path))

    return saved


def plot_individual_qqplots(task_key, df, features):
    saved = []

    for i, feature in enumerate(features, start=1):
        values = pd.to_numeric(df[feature], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).dropna()

        if len(values) < 5:
            continue

        fig, ax = plt.subplots(figsize=(7, 6))

        stats.probplot(values, dist="norm", plot=ax)
        ax.set_title(f"{task_key.title()} Q-Q plot: {clean_label(feature)}")
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Ordered values")

        path = save_plot(
            fig,
            "qqplots",
            f"{task_key}_qqplot_{i:02d}_{safe_name(feature)}.png",
        )
        saved.append(str(path))

    return saved


def plot_bivariate_scatterplots(task_key, df, target_col, pairs):
    saved = []

    for i, (x_feature, y_feature, reason) in enumerate(pairs, start=1):
        plot_df = df[[target_col, x_feature, y_feature]].copy()
        plot_df[x_feature] = pd.to_numeric(plot_df[x_feature], errors="coerce")
        plot_df[y_feature] = pd.to_numeric(plot_df[y_feature], errors="coerce")
        plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()

        if len(plot_df) < 5:
            continue

        plot_df["target_class"] = plot_df[target_col].map({
            0: "Negative",
            1: "Positive",
        })

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.scatterplot(
            data=plot_df,
            x=x_feature,
            y=y_feature,
            hue="target_class",
            alpha=0.75,
            s=35,
            ax=ax,
        )

        ax.set_title(
            f"{task_key.title()} bivariate plot:\n"
            f"{clean_label(x_feature)} vs {clean_label(y_feature)}"
        )
        ax.set_xlabel(clean_label(x_feature))
        ax.set_ylabel(clean_label(y_feature))
        ax.legend(title="Target class")

        path = save_plot(
            fig,
            "bivariate",
            f"{task_key}_scatter_{i:02d}_{safe_name(x_feature)}_vs_{safe_name(y_feature)}.png",
        )
        saved.append(str(path))

    return saved


def run_pca_and_scree_plot(task_key, df, predictors):
    X = df[predictors].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(len(predictors), len(df))

    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    rows = []

    for i, (var, cum) in enumerate(zip(explained, cumulative), start=1):
        rows.append({
            "task": task_key,
            "principal_component": i,
            "explained_variance_ratio": float(var),
            "cumulative_explained_variance": float(cum),
        })

    pca_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 6))

    shown = pca_df.head(min(20, len(pca_df)))

    ax.plot(
        shown["principal_component"],
        shown["explained_variance_ratio"],
        marker="o",
        label="Individual explained variance",
    )

    ax.plot(
        shown["principal_component"],
        shown["cumulative_explained_variance"],
        marker="s",
        label="Cumulative explained variance",
    )

    ax.axhline(
        0.80,
        linestyle="--",
        linewidth=1,
        color="gray",
        label="80% cumulative variance",
    )

    ax.set_title(f"{task_key.title()} PCA scree plot")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.legend()

    save_plot(fig, "pca", f"{task_key}_pca_scree_plot.png")

    return pca_df


# Feature preparation

def remove_constant_predictors(task_key, df, predictors):
    kept = []
    removed = []

    for feature in predictors:
        values = pd.to_numeric(df[feature], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan)

        if values.nunique(dropna=True) <= 1:
            removed.append({
                "task": task_key,
                "feature": feature,
                "reason": "constant_or_near_constant",
            })
        else:
            kept.append(feature)

    return kept, pd.DataFrame(removed)


def remove_highly_correlated_predictors(task_key, df, predictors):
    if not REMOVE_HIGH_CORRELATION:
        return predictors, pd.DataFrame()

    if len(predictors) <= 1:
        return predictors, pd.DataFrame()

    X = df[predictors].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    corr = X.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    dropped = set()
    records = []

    for feature_1 in upper.index:
        for feature_2 in upper.columns:
            pair_corr = upper.loc[feature_1, feature_2]

            if pd.notna(pair_corr) and pair_corr >= HIGH_CORRELATION_THRESHOLD:
                if feature_1 in dropped or feature_2 in dropped:
                    continue

                keep_feature = feature_1
                drop_feature = feature_2
                dropped.add(drop_feature)

                records.append({
                    "task": task_key,
                    "feature": drop_feature,
                    "reason": "highly_correlated_with_kept_feature",
                    "kept_feature": keep_feature,
                    "absolute_pair_correlation": round(float(pair_corr), 5),
                    "correlation_threshold": HIGH_CORRELATION_THRESHOLD,
                })

    final_predictors = [feature for feature in predictors if feature not in dropped]
    removed_df = pd.DataFrame(records)

    return final_predictors, removed_df


def prepare_predictors_for_modelling(task_key, df, predictors):
    df = df.copy()

    for feature in predictors:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
        df[feature] = df[feature].replace([np.inf, -np.inf], np.nan)
        df[feature] = df[feature].fillna(0)

    non_constant_predictors, removed_constant = remove_constant_predictors(
        task_key,
        df,
        predictors,
    )

    final_predictors, removed_correlated = remove_highly_correlated_predictors(
        task_key,
        df,
        non_constant_predictors,
    )

    removed_frames = []

    if not removed_constant.empty:
        removed_frames.append(removed_constant)

    if not removed_correlated.empty:
        removed_frames.append(removed_correlated)

    if removed_frames:
        removed_predictors = pd.concat(removed_frames, ignore_index=True)
    else:
        removed_predictors = pd.DataFrame(columns=["task", "feature", "reason"])

    return df, final_predictors, removed_predictors


# Model creation and evaluation

def create_models(y_train):
    positive_count = int((y_train == 1).sum())
    negative_count = int((y_train == 0).sum())

    scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                )),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "SVM": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                )),
            ]
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    return models


def get_probability(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype="float64")

        if scores.max() == scores.min():
            return np.full(len(scores), 0.5)

        return (scores - scores.min()) / (scores.max() - scores.min())
    return model.predict(X)

def probability_roc_auc_scorer(model, X, y_true):
    y_probability = get_probability(model, X)
    return roc_auc_score(y_true, y_probability)

def calculate_test_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def cross_validation_metrics(model, X, y):
    min_class_count = int(y.value_counts().min())
    n_splits = min(CV_SPLITS, min_class_count)

    if n_splits < 2:
        return {
            "cv_splits": n_splits,
            "cv_accuracy_mean": np.nan,
            "cv_precision_mean": np.nan,
            "cv_recall_mean": np.nan,
            "cv_f1_mean": np.nan,
            "cv_roc_auc_mean": np.nan,
            "cv_f1_std": np.nan,
            "cv_roc_auc_std": np.nan,
        }

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": probability_roc_auc_scorer,
    }

    scores = cross_validate(
        clone(model),
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        error_score=np.nan,
    )

    return {
        "cv_splits": n_splits,
        "cv_accuracy_mean": np.nanmean(scores["test_accuracy"]),
        "cv_precision_mean": np.nanmean(scores["test_precision"]),
        "cv_recall_mean": np.nanmean(scores["test_recall"]),
        "cv_f1_mean": np.nanmean(scores["test_f1"]),
        "cv_roc_auc_mean": np.nanmean(scores["test_roc_auc"]),
        "cv_f1_std": np.nanstd(scores["test_f1"]),
        "cv_roc_auc_std": np.nanstd(scores["test_roc_auc"]),
    }


def plot_confusion_matrix(task_key, model_name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted negative", "Predicted positive"],
        yticklabels=["Actual negative", "Actual positive"],
        ax=ax,
    )

    ax.set_title(f"{task_key.title()} - {model_name} confusion matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")

    return save_plot(
        fig,
        "model_evaluation",
        f"{task_key}_{safe_name(model_name)}_confusion_matrix.png",
    )


def plot_roc_curves(task_key, roc_records):
    fig, ax = plt.subplots(figsize=(7, 6))

    for record in roc_records:
        ax.plot(
            record["fpr"],
            record["tpr"],
            linewidth=2,
            label=f"{record['model']} AUC={record['roc_auc']:.3f}",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title(f"{task_key.title()} ROC curves")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=9)

    return save_plot(fig, "model_evaluation", f"{task_key}_roc_curves.png")


def plot_model_comparison(task_key, metrics_df):
    plot_df = metrics_df.melt(
        id_vars=["task", "model"],
        value_vars=["accuracy", "precision", "recall", "f1_score", "roc_auc"],
        var_name="metric",
        value_name="score",
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    sns.barplot(
        data=plot_df,
        x="model",
        y="score",
        hue="metric",
        ax=ax,
    )

    ax.set_ylim(0, 1.05)
    ax.set_title(f"{task_key.title()} model comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=20, ha="right")

    return save_plot(fig, "model_evaluation", f"{task_key}_model_comparison.png")


def extract_feature_importance(task_key, model_name, fitted_model, predictors):
    records = []

    try:
        if model_name == "Logistic Regression":
            coefficients = fitted_model.named_steps["model"].coef_[0]

            for feature, value in zip(predictors, coefficients):
                records.append({
                    "task": task_key,
                    "model": model_name,
                    "feature": feature,
                    "importance": abs(float(value)),
                    "raw_value": float(value),
                    "importance_type": "absolute_logistic_coefficient",
                })

        elif model_name in ["Random Forest", "XGBoost"]:
            importances = fitted_model.feature_importances_

            for feature, value in zip(predictors, importances):
                records.append({
                    "task": task_key,
                    "model": model_name,
                    "feature": feature,
                    "importance": float(value),
                    "raw_value": float(value),
                    "importance_type": "tree_feature_importance",
                })

    except Exception as e:
        print(f"Feature importance skipped for {task_key} - {model_name}: {e}")

    if not records:
        return pd.DataFrame(columns=[
            "task",
            "model",
            "feature",
            "importance",
            "raw_value",
            "importance_type",
        ])

    out = pd.DataFrame(records)
    out = out.sort_values("importance", ascending=False)
    return out


def permutation_importance_for_best_model(task_key, model_name, fitted_model, X_test, y_test, predictors):
    result = permutation_importance(
        fitted_model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scoring="roc_auc",
    )

    rows = []

    for feature, mean_value, std_value in zip(
        predictors,
        result.importances_mean,
        result.importances_std,
    ):
        rows.append({
            "task": task_key,
            "model": model_name,
            "feature": feature,
            "importance": float(mean_value),
            "importance_std": float(std_value),
            "importance_type": "permutation_importance_roc_auc",
        })

    out = pd.DataFrame(rows)
    out = out.sort_values("importance", ascending=False)
    return out


def plot_feature_importance(task_key, model_name, importance_df):
    if importance_df.empty:
        return None

    plot_df = importance_df.head(TOP_FEATURES_FOR_IMPORTANCE).copy()

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.barplot(
        data=plot_df,
        y="feature",
        x="importance",
        ax=ax,
    )

    ax.set_title(f"{task_key.title()} - {model_name} feature importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    return save_feature_importance_plot(
        fig,
        f"{task_key}_{safe_name(model_name)}_feature_importance.png",
    )


# Main task workflow

def run_task_workflow(task_key, config, predictor_cols):
    print_section(config["display_name"])

    df, available_predictors, missing_predictors = load_task_dataset(
        task_key,
        config["dataset_path"],
        config["target_col"],
        predictor_cols,
    )

    target_col = config["target_col"]
    id_cols = get_id_columns(df)

    summary_df = dataset_summary(task_key, df, target_col, available_predictors)
    class_df = class_balance_table(task_key, df, target_col)
    missing_df = missing_value_table(task_key, df, available_predictors)
    stats_df = descriptive_statistics_table(task_key, df, available_predictors)
    target_corr_df = target_correlation_table(task_key, df, target_col, available_predictors)
    high_corr_df = high_correlation_pairs_table(task_key, df, available_predictors, threshold=0.80)
    distribution_df = feature_distribution_table(task_key, df, available_predictors)
    pca_df = run_pca_and_scree_plot(task_key, df, available_predictors)

    histogram_features, boxplot_features, qqplot_features, variation_features, plot_selection_df = select_eda_features(
        task_key,
        df,
        available_predictors,
        target_corr_df,
    )

    bivariate_pairs, bivariate_pairs_df = select_bivariate_pairs(task_key, df, target_corr_df)

    save_table(summary_df, "tables", f"{task_key}_dataset_summary.csv")
    save_table(class_df, "tables", f"{task_key}_class_balance.csv")
    save_table(missing_df, "tables", f"{task_key}_missing_values_summary.csv")
    save_table(stats_df, "tables", f"{task_key}_descriptive_statistics.csv")
    save_table(target_corr_df, "tables", f"{task_key}_target_correlations.csv")
    save_table(high_corr_df, "tables", f"{task_key}_high_correlation_pairs.csv")
    save_table(distribution_df, "tables", f"{task_key}_feature_distribution_summary.csv")
    save_table(pca_df, "tables", f"{task_key}_pca_explained_variance.csv")
    save_table(plot_selection_df, "tables", f"{task_key}_eda_plot_feature_selection.csv")
    save_table(bivariate_pairs_df, "tables", f"{task_key}_bivariate_plot_pairs.csv")

    plot_class_balance(task_key, class_df)
    plot_missing_values_if_needed(task_key, missing_df)
    plot_target_correlations(task_key, target_corr_df)
    plot_correlation_heatmap(task_key, df, target_col, target_corr_df)
    plot_individual_histograms(task_key, df, histogram_features)
    plot_individual_boxplots(task_key, df, target_col, boxplot_features)
    plot_individual_qqplots(task_key, df, qqplot_features)
    plot_bivariate_scatterplots(task_key, df, target_col, bivariate_pairs)

    df_prepared, final_predictors, removed_predictors_df = prepare_predictors_for_modelling(
        task_key,
        df,
        available_predictors,
    )

    save_table(
        removed_predictors_df,
        "tables",
        f"{task_key}_removed_predictors_before_modelling.csv",
    )

    print_subsection("Dataset summary")
    print(summary_df.to_string(index=False))

    print("\nClass balance:")
    print(class_df.to_string(index=False))

    print(f"\nPredictors available in dataset: {len(available_predictors)}")
    print(f"Final predictors used in modelling: {len(final_predictors)}")
    print(f"Predictors removed before modelling: {len(removed_predictors_df)}")
    print(f"Individual histogram plots created for up to {len(histogram_features)} features")
    print(f"Individual boxplots created for up to {len(boxplot_features)} features")
    print(f"Individual Q-Q plots created for up to {len(qqplot_features)} features")
    print(f"Bivariate scatter plots created for up to {len(bivariate_pairs)} feature pairs")

    X = df_prepared[final_predictors].copy()
    y = df_prepared[target_col].copy()

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X,
        y,
        df_prepared.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_df = df_prepared.loc[train_idx, id_cols + [target_col] + final_predictors].copy()
    test_df = df_prepared.loc[test_idx, id_cols + [target_col] + final_predictors].copy()

    save_table(train_df, "train_test_data", f"{task_key}_train_data.csv")
    save_table(test_df, "train_test_data", f"{task_key}_test_data.csv")

    print_subsection("Train/test split")
    print(f"Training rows: {len(train_df)}")
    print(f"Testing rows: {len(test_df)}")

    print("\nTraining class counts:")
    print(y_train.value_counts().sort_index().to_string())

    print("\nTesting class counts:")
    print(y_test.value_counts().sort_index().to_string())

    models = create_models(y_train)

    metrics_records = []
    classification_report_records = []
    roc_records = []
    all_importance_frames = []
    fitted_models = {}

    test_predictions = df_prepared.loc[test_idx, id_cols + [target_col]].copy()
    test_predictions = test_predictions.reset_index(drop=True)

    all_unit_predictions = df_prepared[id_cols + [target_col]].copy()

    for model_name, model in models.items():
        print_subsection(f"Training model: {model_name}")

        fitted_model = clone(model)
        fitted_model.fit(X_train, y_train)

        y_pred = fitted_model.predict(X_test)
        y_prob = get_probability(fitted_model, X_test)

        test_metrics = calculate_test_metrics(y_test, y_pred, y_prob)
        cv_metrics = cross_validation_metrics(fitted_model, X, y)

        metric_row = {
            "task": task_key,
            "model": model_name,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "predictor_count": len(final_predictors),
            "positive_class_total": int((y == 1).sum()),
            "negative_class_total": int((y == 0).sum()),
        }

        metric_row.update(test_metrics)
        metric_row.update(cv_metrics)

        for key, value in metric_row.items():
            if isinstance(value, (float, np.floating)):
                metric_row[key] = round(float(value), 4)

        metrics_records.append(metric_row)

        print(pd.DataFrame([metric_row])[[
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "cv_f1_mean",
            "cv_roc_auc_mean",
        ]].to_string(index=False))

        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Negative", "Positive"],
            output_dict=True,
            zero_division=0,
        )

        for report_key, values in report.items():
            if isinstance(values, dict):
                row = {
                    "task": task_key,
                    "model": model_name,
                    "class_or_average": report_key,
                }
                row.update(values)
                classification_report_records.append(row)

        fpr, tpr, _ = roc_curve(y_test, y_prob)

        roc_records.append({
            "model": model_name,
            "fpr": fpr,
            "tpr": tpr,
            "roc_auc": test_metrics["roc_auc"],
        })

        plot_confusion_matrix(task_key, model_name, y_test, y_pred)

        test_predictions[f"{safe_name(model_name)}_predicted_class"] = y_pred
        test_predictions[f"{safe_name(model_name)}_predicted_probability"] = y_prob

        all_probability = get_probability(fitted_model, X)
        all_unit_predictions[f"{safe_name(model_name)}_predicted_probability"] = all_probability
        all_unit_predictions[f"{safe_name(model_name)}_predicted_class"] = (all_probability >= 0.5).astype(int)

        model_path = OUTPUT_FOLDERS["models"] / f"{task_key}_{safe_name(model_name)}.joblib"
        joblib.dump(fitted_model, model_path)

        fitted_models[model_name] = fitted_model

        importance_df = extract_feature_importance(
            task_key,
            model_name,
            fitted_model,
            final_predictors,
        )

        if not importance_df.empty:
            save_table(
                importance_df,
                "feature_importance",
                f"{task_key}_{safe_name(model_name)}_feature_importance.csv",
            )

            plot_feature_importance(task_key, model_name, importance_df)
            all_importance_frames.append(importance_df)

    metrics_df = pd.DataFrame(metrics_records)

    metrics_df = metrics_df.sort_values(
        by=["roc_auc", "f1_score", "recall", "precision"],
        ascending=False,
    ).reset_index(drop=True)

    metrics_df["rank"] = np.arange(1, len(metrics_df) + 1)

    classification_reports_df = pd.DataFrame(classification_report_records)

    save_table(metrics_df, "metrics", f"{task_key}_model_metrics.csv")
    save_table(classification_reports_df, "metrics", f"{task_key}_classification_reports.csv")
    save_table(test_predictions, "predictions_for_mapping", f"{task_key}_test_set_predictions.csv")
    save_table(all_unit_predictions, "predictions_for_mapping", f"{task_key}_all_model_probabilities_all_admin3_units.csv")

    plot_roc_curves(task_key, roc_records)
    plot_model_comparison(task_key, metrics_df)

    best_model_name = metrics_df.iloc[0]["model"]
    best_model_from_test = fitted_models[best_model_name]

    print_subsection("Best model")
    print(f"Best {task_key} model: {best_model_name}")

    best_model_final = clone(models[best_model_name])
    best_model_final.fit(X, y)

    best_probability_all = get_probability(best_model_final, X)
    best_class_all = (best_probability_all >= 0.5).astype(int)

    qgis_df = df_prepared[id_cols + [target_col]].copy()
    qgis_df[f"{task_key}_best_model"] = best_model_name
    qgis_df[f"{task_key}_risk_probability"] = best_probability_all
    qgis_df[f"{task_key}_predicted_class"] = best_class_all
    qgis_df[f"{task_key}_risk_category"] = [risk_category(p) for p in best_probability_all]

    qgis_df[f"{task_key}_risk_category_order"] = pd.Categorical(
        qgis_df[f"{task_key}_risk_category"],
        categories=risk_order(),
        ordered=True,
    ).codes + 1

    qgis_path = save_table(
        qgis_df,
        "predictions_for_mapping",
        f"admin3_{task_key}_risk_predictions_for_qgis.csv",
    )

    final_model_path = OUTPUT_FOLDERS["models"] / f"{task_key}_best_model_refit_all_data_{safe_name(best_model_name)}.joblib"
    joblib.dump(best_model_final, final_model_path)

    risk_counts = (
        qgis_df[f"{task_key}_risk_category"]
        .value_counts()
        .reindex(risk_order(), fill_value=0)
        .reset_index()
    )

    risk_counts.columns = ["risk_category", "admin3_unit_count"]
    risk_counts.insert(0, "task", task_key)

    save_table(risk_counts, "tables", f"{task_key}_risk_category_counts.csv")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.barplot(
        data=risk_counts,
        x="risk_category",
        y="admin3_unit_count",
        order=risk_order(),
        ax=ax,
    )

    ax.set_title(f"{task_key.title()} risk category counts")
    ax.set_xlabel("Risk category")
    ax.set_ylabel("Number of admin3 municipalities")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3)

    save_plot(fig, "risk_categories", f"{task_key}_risk_category_counts.png")

    best_importance_df = extract_feature_importance(
        task_key,
        best_model_name,
        best_model_final,
        final_predictors,
    )

    if best_importance_df.empty:
        print("Native feature importance not available for this best model.")
        print("Running permutation importance for best model.")

        best_importance_df = permutation_importance_for_best_model(
            task_key,
            best_model_name,
            best_model_from_test,
            X_test,
            y_test,
            final_predictors,
        )

    if not best_importance_df.empty:
        save_table(
            best_importance_df,
            "feature_importance",
            f"{task_key}_best_model_feature_importance.csv",
        )

        plot_feature_importance(
            task_key,
            f"Best Model - {best_model_name}",
            best_importance_df,
        )

    if all_importance_frames:
        all_importance_df = pd.concat(all_importance_frames, ignore_index=True)
    else:
        all_importance_df = pd.DataFrame()

    return {
        "task": task_key,
        "target_col": target_col,
        "dataset_summary": summary_df,
        "class_balance": class_df,
        "missing_values": missing_df,
        "descriptive_statistics": stats_df,
        "target_correlations": target_corr_df,
        "high_correlation_pairs": high_corr_df,
        "distribution_summary": distribution_df,
        "pca": pca_df,
        "plot_selection": plot_selection_df,
        "bivariate_pairs": bivariate_pairs_df,
        "removed_predictors": removed_predictors_df,
        "final_predictors": final_predictors,
        "metrics": metrics_df,
        "classification_reports": classification_reports_df,
        "test_predictions": test_predictions,
        "all_unit_predictions": all_unit_predictions,
        "qgis_predictions": qgis_df,
        "risk_counts": risk_counts,
        "feature_importance": all_importance_df,
        "best_model": best_model_name,
        "best_model_path": str(final_model_path),
        "qgis_prediction_path": str(qgis_path),
    }


# Final summary and logs

def create_final_summary(results):
    rows = []

    for task_key, result in results.items():
        best = result["metrics"].iloc[0]

        rows.append({
            "task": task_key,
            "target_column": result["target_col"],
            "best_model": result["best_model"],
            "accuracy": best["accuracy"],
            "precision": best["precision"],
            "recall": best["recall"],
            "f1_score": best["f1_score"],
            "roc_auc": best["roc_auc"],
            "cv_f1_mean": best["cv_f1_mean"],
            "cv_roc_auc_mean": best["cv_roc_auc_mean"],
            "final_predictor_count": len(result["final_predictors"]),
            "best_model_file": result["best_model_path"],
            "qgis_prediction_file": result["qgis_prediction_path"],
            "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    return pd.DataFrame(rows)


def plot_final_best_model_summary(final_summary):
    plot_df = final_summary.melt(
        id_vars=["task", "best_model"],
        value_vars=["accuracy", "precision", "recall", "f1_score", "roc_auc"],
        var_name="metric",
        value_name="score",
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=plot_df,
        x="task",
        y="score",
        hue="metric",
        ax=ax,
    )

    ax.set_ylim(0, 1.05)
    ax.set_title("Best model performance summary")
    ax.set_xlabel("Hazard model")
    ax.set_ylabel("Score")
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")

    return save_plot(fig, "model_evaluation", "final_best_model_performance_summary.png")


def create_output_manifest():
    rows = []

    for folder_name, folder_path in OUTPUT_FOLDERS.items():
        for file_path in folder_path.rglob("*"):
            if file_path.is_file():
                rows.append({
                    "folder": folder_name,
                    "file_name": file_path.name,
                    "path": str(file_path),
                    "size_kb": round(file_path.stat().st_size / 1024, 3),
                })

    manifest = pd.DataFrame(rows)

    if not manifest.empty:
        manifest = manifest.sort_values(["folder", "file_name"])

    return manifest


def write_methodology_note(results, final_summary, output_manifest):
    note_path = OUTPUT_FOLDERS["final_notes"] / "methodology_and_modelling_note.txt"

    note = f"""Methodology and modelling note

Project:
Machine Learning and GIS-Based Early Warning Prototype for Multi-Hazard Risk in Nepal

Purpose:
This script performs the final exploratory data analysis and supervised machine learning stage for flood risk mapping and landslide susceptibility mapping using cleaned admin3 municipality-level datasets.

Input datasets:
- admin3_flood_ml_dataset.csv
- admin3_landslide_ml_dataset.csv
- ml_predictor_lists.json

Spatial modelling unit:
Admin3 municipality-level data are used because they provide 775 spatial units and both positive and negative classes for flood and landslide modelling. Admin2 district-level outputs remain useful for wider summaries and mapping, but admin2 flood classification is not suitable because all districts are flood-positive.

EDA outputs:
The script creates class balance tables, missing-value summaries, descriptive statistics, target-correlation tables, high-correlation tables, PCA scree plots, correlation heatmaps, individual histograms, individual boxplots, individual Q-Q plots and selected bivariate scatter plots.

Plot folder structure:
- plots/class_balance
- plots/histograms
- plots/boxplots
- plots/qqplots
- plots/correlation
- plots/pca
- plots/bivariate
- plots/model_evaluation
- plots/risk_categories

Feature preparation:
Predictors are loaded from the clean predictor list created during preprocessing. Numeric predictors are checked, missing values are filled with zero, constant predictors are removed, and highly correlated redundant predictors are removed using a threshold of {HIGH_CORRELATION_THRESHOLD}.

Machine learning models:
Four supervised classification models are trained for both flood and landslide:
- Logistic Regression
- Random Forest
- SVM
- XGBoost

Evaluation:
Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- ROC curve
- 5-fold stratified cross-validation

Train-test split:
A 70:30 stratified train-test split is used with random_state={RANDOM_STATE}.

Risk categories:
Best-model probabilities are converted into four fixed risk categories:
- Low: 0.00 to 0.25
- Medium: 0.25 to 0.50
- High: 0.50 to 0.75
- Very High: 0.75 to 1.00

Important interpretation:
Test-set model metrics are used for model evaluation. The QGIS-ready full-admin3 prediction outputs are created by refitting the best model on all available admin3 data for mapping and prototype communication. These outputs support a research prototype and should not be presented as a real-time operational government warning system.

Final best model summary:
{final_summary.to_string(index=False)}
"""

    with open(note_path, "w", encoding="utf-8") as f:
        f.write(note)

    log_path = OUTPUT_FOLDERS["logs"] / "Flood_Landslide_ML_run_log.txt"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Flood_Landslide_ML_Code.py run log\n")
        f.write(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Project folder: {PROJECT_DIR}\n")
        f.write("Train/test split: 70/30\n")
        f.write(f"Cross-validation folds: {CV_SPLITS}\n")
        f.write(f"High-correlation removal: {REMOVE_HIGH_CORRELATION}\n")
        f.write(f"High-correlation threshold: {HIGH_CORRELATION_THRESHOLD}\n")
        f.write("Report figures folder used: No\n")
        f.write("Cleaned datasets folder used: No\n\n")

        f.write("Final summary:\n")
        f.write(final_summary.to_string(index=False))
        f.write("\n\n")

        for task_key, result in results.items():
            f.write(f"\n{task_key.upper()} final predictors:\n")

            for feature in result["final_predictors"]:
                f.write(f"- {feature}\n")

        f.write("\n\nOutput manifest:\n")

        if output_manifest.empty:
            f.write("No outputs found.\n")
        else:
            f.write(output_manifest.to_string(index=False))

    return note_path, log_path


# Main script

def main():
    warnings.filterwarnings("ignore")

    create_output_folders()
    setup_plot_style()

    print_section("Flood and Landslide ML Code")
    print(f"Project folder: {PROJECT_DIR}")
    print("Using final cleaned admin3 datasets.")
    print("Models: Logistic Regression, Random Forest, SVM and XGBoost.")
    print("Evaluation: 70/30 train-test split and 5-fold stratified cross-validation.")
    print("Plots are saved as PNG files inside the plots subfolders.")

    try:
        predictor_cols = load_predictor_list()

        print_subsection("Input check")
        print(f"Flood dataset exists: {FLOOD_FILE.exists()} | {FLOOD_FILE}")
        print(f"Landslide dataset exists: {LANDSLIDE_FILE.exists()} | {LANDSLIDE_FILE}")
        print(f"Predictor JSON exists: {PREDICTOR_JSON.exists()} | {PREDICTOR_JSON}")
        print(f"Predictors loaded from JSON: {len(predictor_cols)}")

        results = {}

        for task_key, config in TASKS.items():
            result = run_task_workflow(
                task_key=task_key,
                config=config,
                predictor_cols=predictor_cols,
            )

            results[task_key] = result

        all_dataset_summaries = pd.concat(
            [result["dataset_summary"] for result in results.values()],
            ignore_index=True,
        )

        all_class_balance = pd.concat(
            [result["class_balance"] for result in results.values()],
            ignore_index=True,
        )

        all_metrics = pd.concat(
            [result["metrics"] for result in results.values()],
            ignore_index=True,
        )

        all_plot_selection = pd.concat(
            [result["plot_selection"] for result in results.values()],
            ignore_index=True,
        )

        all_bivariate_pairs = pd.concat(
            [result["bivariate_pairs"] for result in results.values()],
            ignore_index=True,
        )

        removed_frames = [
            result["removed_predictors"]
            for result in results.values()
            if not result["removed_predictors"].empty
        ]

        if removed_frames:
            all_removed_predictors = pd.concat(removed_frames, ignore_index=True)
        else:
            all_removed_predictors = pd.DataFrame(columns=["task", "feature", "reason"])

        final_summary = create_final_summary(results)

        save_table(all_dataset_summaries, "tables", "all_dataset_summary.csv")
        save_table(all_class_balance, "tables", "all_class_balance_summary.csv")
        save_table(all_metrics, "metrics", "all_model_metrics_combined.csv")
        save_table(all_removed_predictors, "tables", "all_removed_predictors_before_modelling.csv")
        save_table(all_plot_selection, "tables", "all_eda_plot_feature_selection.csv")
        save_table(all_bivariate_pairs, "tables", "all_bivariate_plot_pairs.csv")
        save_table(final_summary, "metrics", "final_best_model_summary.csv")

        plot_final_best_model_summary(final_summary)

        output_manifest = create_output_manifest()
        save_table(output_manifest, "logs", "output_manifest.csv")

        note_path, log_path = write_methodology_note(results, final_summary, output_manifest)

        print_section("FINAL SUMMARY")
        print(final_summary.to_string(index=False))

        print_subsection("Main QGIS-ready prediction files")

        for task_key, result in results.items():
            print(f"{task_key}: {result['qgis_prediction_path']}")

        print_subsection("Important output folders")

        for folder_name, folder_path in OUTPUT_FOLDERS.items():
            print(f"{folder_name}: {folder_path}")

        print("\nPlot subfolders:")
        for folder_name, folder_path in PLOT_FOLDERS.items():
            print(f"{folder_name}: {folder_path}")

        print_subsection("Logs")
        print(f"Run log: {log_path}")
        print(f"Methodology note: {note_path}")

        print("\nCompleted successfully.")
        print("Check the folders for CSV tables, PNG plots, saved models and QGIS prediction files.")

    except Exception as e:
        error_path = OUTPUT_FOLDERS["logs"] / "Flood_Landslide_ML_error_log.txt"

        with open(error_path, "w", encoding="utf-8") as f:
            f.write("Flood_Landslide_ML_Code.py failed.\n")
            f.write(str(e))
            f.write("\n\n")
            f.write(traceback.format_exc())

        print("\nERROR: The script failed.")
        print(f"Error log saved to: {error_path}")
        raise


if __name__ == "__main__":
    main()
