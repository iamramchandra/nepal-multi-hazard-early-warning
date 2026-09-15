from pathlib import Path
import base64
import html
import json
import math

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

try:
    import geopandas as gpd
except ImportError:
    gpd = None


# Dashboard page setup
st.set_page_config(
    page_title="Nepal Multi-Hazard Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Main folder paths
PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_OUTPUTS = PROJECT_DIR / "Project_Outputs"

FINAL_MAPS = PROJECT_OUTPUTS / "Final_Maps"
FINAL_MODELS = PROJECT_OUTPUTS / "Final_Models"
FINAL_NOTES = PROJECT_OUTPUTS / "Final_Notes"
FINAL_PLOTS = PROJECT_OUTPUTS / "Final_Plots"
FINAL_QGIS = PROJECT_OUTPUTS / "Final_QGIS_Layers"
FINAL_TABLES = PROJECT_OUTPUTS / "Final_Tables"


# Dashboard styling
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 0.35rem;
        padding-bottom: 0.5rem;
        max-width: 1500px;
    }

    h1 {
        font-size: 2.0rem !important;
        line-height: 1.2 !important;
        margin-bottom: 0 !important;
    }

    h2 {
        font-size: 1.45rem !important;
        margin-top: 0.7rem;
    }

    h3 {
        font-size: 1.12rem !important;
    }

    .small-text {
        font-size: 0.89rem;
        color: #666;
        margin-bottom: 0.4rem;
    }

    .overview-hero {
        background: linear-gradient(120deg, #123b66 0%, #176b78 58%, #2c8b72 100%);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 15px;
        box-shadow: 0 8px 22px rgba(10, 35, 60, 0.2);
        color: #ffffff;
        margin: 0.15rem 0 0.55rem 0;
        padding: 14px 20px;
    }

    .overview-hero-kicker {
        color: #c8f2e7;
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.085em;
        margin-bottom: 4px;
        text-transform: uppercase;
    }

    .overview-hero-title {
        font-size: 1.24rem;
        font-weight: 800;
        line-height: 1.22;
        margin-bottom: 4px;
    }

    .overview-hero-text {
        color: #e6f6f3;
        font-size: 0.85rem;
        line-height: 1.35;
    }

    .metric-card {
        background-color: #f7f9fc;
        border: 1px solid #e6eaf0;
        border-top: 4px solid #2878a8;
        box-shadow: 0 3px 9px rgba(18, 59, 102, 0.07);
        padding: 7px 10px 8px 10px;
        border-radius: 10px;
        margin-bottom: 0.4rem;
        text-align: center;
    }

    .metric-number {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1f4e79;
        line-height: 1.18;
    }

    .metric-label {
        font-size: 0.77rem;
        color: #555;
        line-height: 1.18;
    }

    .image-box {
        width: 100%;
        height: 72vh;
        display: flex;
        justify-content: center;
        align-items: center;
        background: white;
        border: 1px solid #eeeeee;
        border-radius: 10px;
        padding: 6px;
    }

    .image-box img {
        max-width: 100%;
        max-height: 70vh;
        object-fit: contain;
    }

    .plot-box {
        width: 100%;
        height: 68vh;
        display: flex;
        justify-content: center;
        align-items: center;
        background: white;
        border: 1px solid #eeeeee;
        border-radius: 10px;
        padding: 6px;
    }

    .plot-box img {
        max-width: 100%;
        max-height: 66vh;
        object-fit: contain;
    }

    .selected-municipality-bar {
        align-items: center;
        background: linear-gradient(100deg, #f7f9fc 0%, #eef5f7 100%);
        border: 1px solid #dfe7ee;
        border-left: 5px solid #2878a8;
        border-radius: 10px;
        color: #273142;
        display: flex;
        gap: 16px;
        justify-content: space-between;
        margin: 0.2rem 0 0.45rem 0;
        padding: 8px 12px;
    }

    .selected-place {
        color: #173f65;
        font-size: 1rem;
        font-weight: 800;
        line-height: 1.15;
    }

    .selected-meta {
        color: #657487;
        font-size: 0.72rem;
        line-height: 1.2;
        margin-top: 2px;
    }

    .selected-status {
        align-items: center;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        justify-content: flex-end;
    }

    .dominant-chip {
        background-color: #e7eef5;
        border-radius: 999px;
        color: #1f4e79;
        font-size: 0.74rem;
        font-weight: 750;
        padding: 4px 9px;
    }

    .map-title-card {
        align-items: center;
        background-color: #f7f9fc;
        border: 1px solid #e1e8ef;
        border-radius: 11px;
        color: #173f65;
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.3rem;
        padding: 6px 10px;
    }

    .map-title-name {
        font-size: 0.92rem;
        font-weight: 800;
    }

    .risk-table {
        border-collapse: separate;
        border-spacing: 0;
        border: 1px solid #e6eaf0;
        border-radius: 12px;
        color: #273142;
        overflow: hidden;
        font-size: 0.85rem;
        width: 100%;
    }

    .risk-table th {
        background-color: #edf2f7;
        font-weight: 700;
        padding: 7px 9px;
        text-align: left;
    }

    .risk-table td {
        background-color: #ffffff;
        border-top: 1px solid #edf0f4;
        padding: 7px 9px;
    }

    .risk-badge {
        border-radius: 999px;
        color: #ffffff;
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 700;
        min-width: 74px;
        padding: 3px 8px;
        text-align: center;
    }

    @media (max-width: 900px) {
        .selected-municipality-bar {
            align-items: flex-start;
            flex-direction: column;
            gap: 6px;
        }

        .selected-status {
            justify-content: flex-start;
        }
    }

    div[data-testid="stDataFrame"] {
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Helper functions
def clean_name(text):
    text = str(text).replace("_", " ").replace("-", " ")
    return " ".join(text.split()).title()


def find_file(root, include_words, suffixes):
    """Find one file using words in file name."""
    if not root.exists():
        return None

    include_words = [w.lower() for w in include_words]
    suffixes = [s.lower() for s in suffixes]

    matches = []
    for file in root.rglob("*"):
        if file.is_file() and file.suffix.lower() in suffixes:
            file_name = file.stem.lower()
            if all(word in file_name for word in include_words):
                matches.append(file)

    if not matches:
        return None

    return sorted(matches, key=lambda p: len(str(p)))[0]


def list_files(root, suffixes):
    """List files by type."""
    if not root.exists():
        return []

    suffixes = [s.lower() for s in suffixes]
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes]
    )


def read_csv_file(path):
    """Read CSV safely."""
    if path is None or not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, encoding="latin1")
        except Exception:
            return pd.DataFrame()


def display_image_fit(path, box_type="image"):
    """Show image fitted in one screen frame."""
    if path is None or not path.exists():
        st.warning("Image file was not found.")
        return

    image_bytes = path.read_bytes()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    image_type = path.suffix.replace(".", "").lower()

    css_class = "plot-box" if box_type == "plot" else "image-box"

    st.markdown(
        f"""
        <div class="{css_class}">
            <img src="data:image/{image_type};base64,{image_base64}">
        </div>
        """,
        unsafe_allow_html=True,
    )

def show_download_button(df, file_name, label):
    """Download table as CSV."""
    if df.empty:
        return

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_data,
        file_name=file_name,
        mime="text/csv",
        use_container_width=True,
    )

def first_matching_column(df, keywords):
    """Find first column using keywords."""
    if df.empty:
        return None

    for col in df.columns:
        col_lower = col.lower()
        if any(k.lower() in col_lower for k in keywords):
            return col

    return None


def exact_column(df, candidates):
    """Find a column from a list of accepted names."""
    if df.empty:
        return None

    column_lookup = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        match = column_lookup.get(candidate.lower())
        if match is not None:
            return match

    return None


def municipality_columns(df):
    """Match municipality and risk fields in the integrated table."""
    accepted_names = {
        "province": ["adm1_name", "province", "province_name", "adm1_en"],
        "district": ["adm2_name", "district", "district_name", "adm2_en"],
        "municipality": ["adm3_name", "municipality", "municipality_name", "adm3_en"],
        "code": ["adm3_pcode", "adm3_code", "municipality_code", "pcode"],
        "flood_score": ["flood_score", "flood_risk_score", "risk_probability"],
        "flood_category": ["flood_category", "flood_risk_category"],
        "landslide_score": ["landslide_score", "landslide_risk_score"],
        "landslide_category": ["landslide_category", "landslide_risk_category"],
        "glof_score": ["glof_score", "glof_risk_score", "admin3_glof_risk_score"],
        "glof_category": ["glof_category", "glof_risk_category"],
        "overall_score": ["multi_hazard_priority_score", "multi_hazard_max_score"],
        "overall_category": ["multi_hazard_category", "multi_hazard_risk_category"],
        "dominant_hazard": ["dominant_hazard", "dominant hazard"],
    }

    return {
        field: exact_column(df, candidates)
        for field, candidates in accepted_names.items()
    }


def safe_text(value, default="Not available"):
    """Return clean text for one table value."""
    if value is None or pd.isna(value):
        return default

    value = str(value).strip()
    return value if value else default


def row_value(row, column, default=None):
    """Read a selected row safely when a column is optional."""
    if column is None or column not in row.index:
        return default
    return row[column]


def numeric_score(value):
    """Convert one risk score to a number when possible."""
    score = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(score) else float(score)


def risk_colour(category):
    """Use consistent colours for the four project risk categories."""
    colours = {
        "low": "#2e7d32",
        "medium": "#fbc02d",
        "high": "#ef6c00",
        "very high": "#c62828",
    }
    return colours.get(safe_text(category, "").lower(), "#607d8b")


def risk_text_colour(category):
    """Keep category labels readable on their risk colour."""
    return "#332b00" if safe_text(category, "").lower() == "medium" else "#ffffff"


def risk_badge_html(category, suffix=""):
    """Create one reusable colour-coded risk badge."""
    category_text = safe_text(category)
    label = f"{category_text}{suffix}"
    return (
        "<span class='risk-badge' "
        f"style='background-color:{risk_colour(category_text)};"
        f"color:{risk_text_colour(category_text)};'>"
        f"{html.escape(label)}</span>"
    )


def prepare_municipality_search(df, columns):
    """Create searchable municipality labels without changing source data."""
    search_df = df.copy()

    province = search_df[columns["province"]].fillna("").astype(str).str.strip()
    district = search_df[columns["district"]].fillna("").astype(str).str.strip()
    municipality = search_df[columns["municipality"]].fillna("").astype(str).str.strip()
    code = search_df[columns["code"]].fillna("").astype(str).str.strip()

    search_df["_search_text"] = (
        municipality + " " + district + " " + province + " " + code
    ).str.casefold()
    search_df["_display_label"] = (
        municipality + " — " + district + ", " + province + " (" + code + ")"
    )

    return search_df.sort_values(
        by=[columns["municipality"], columns["district"], columns["province"]],
        kind="stable",
    )


def create_municipality_risk_table(row, columns):
    """Create the four-row municipality risk result table."""
    risk_fields = [
        ("Flood", "flood_score", "flood_category"),
        ("Landslide", "landslide_score", "landslide_category"),
        ("GLOF", "glof_score", "glof_category"),
        ("Overall multi-hazard", "overall_score", "overall_category"),
    ]

    rows = []
    for hazard, score_field, category_field in risk_fields:
        rows.append(
            {
                "Hazard": hazard,
                "Risk score": numeric_score(
                    row_value(row, columns.get(score_field))
                ),
                "Risk category": safe_text(
                    row_value(row, columns.get(category_field))
                ),
            }
        )

    return pd.DataFrame(rows)


def display_municipality_risk_table(risk_df):
    """Display risk results as a clear colour-coded table."""
    table_rows = []

    for _, result in risk_df.iterrows():
        score = result["Risk score"]
        score_text = "Not available" if pd.isna(score) else f"{score:.3f}"
        category = safe_text(result["Risk category"])

        table_rows.append(
            "<tr>"
            f"<td><strong>{html.escape(str(result['Hazard']))}</strong></td>"
            f"<td>{html.escape(score_text)}</td>"
            "<td>"
            f"{risk_badge_html(category)}"
            "</td>"
            "</tr>"
        )

    st.markdown(
        "<table class='risk-table'>"
        "<thead><tr><th>Hazard</th><th>Risk score</th><th>Risk category</th></tr></thead>"
        f"<tbody>{''.join(table_rows)}</tbody></table>",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_municipality_boundaries(path_text):
    """Load the integrated admin3 GeoPackage and convert it for web mapping."""
    if gpd is None:
        return None

    path = Path(path_text)
    try:
        spatial_df = gpd.read_file(
            path,
            layer="admin3_multi_hazard_risk_scores",
        )
    except Exception:
        spatial_df = gpd.read_file(path)

    if spatial_df.empty:
        return spatial_df
    if spatial_df.crs is None:
        raise ValueError("The municipality GeoPackage has no coordinate reference system.")

    return spatial_df.to_crs("EPSG:4326")


def municipality_search_source(gpkg_path, fallback_df):
    """Use the integrated GeoPackage for selection, with a CSV fallback."""
    if gpd is not None and gpkg_path is not None and gpkg_path.exists():
        try:
            spatial_df = load_municipality_boundaries(str(gpkg_path))
            if spatial_df is not None and not spatial_df.empty:
                return spatial_df, "Integrated admin3 GeoPackage"
        except Exception:
            pass

    if fallback_df is not None and not fallback_df.empty:
        return fallback_df.copy(), "Integrated admin3 CSV fallback"

    return pd.DataFrame(), "Unavailable"


def display_municipality_map(gpkg_path, municipality_code, overall_category):
    """Highlight the selected municipality using the integrated GeoPackage."""
    if gpd is None:
        st.warning(
            "The risk results are available, but the map requires GeoPandas. "
            "Run the dashboard in the ds7010_gis environment."
        )
        return

    if gpkg_path is None or not gpkg_path.exists():
        st.warning(
            "The municipality GeoPackage was not found in "
            "Project_Outputs/Final_QGIS_Layers/Multi_Hazard_Layers."
        )
        return

    try:
        spatial_df = load_municipality_boundaries(str(gpkg_path))
    except Exception as exc:
        st.warning(f"The municipality map could not be loaded: {exc}")
        return

    if spatial_df is None or spatial_df.empty:
        st.warning("The municipality GeoPackage does not contain any features.")
        return

    spatial_code_col = exact_column(
        spatial_df,
        ["adm3_pcode", "adm3_code", "municipality_code", "pcode"],
    )
    if spatial_code_col is None:
        st.warning("The admin3 municipality code was not found in the GeoPackage.")
        return

    code_text = safe_text(municipality_code, "")
    selected_geo = spatial_df.loc[
        spatial_df[spatial_code_col].fillna("").astype(str).str.strip() == code_text
    ].copy()
    selected_geo = selected_geo.loc[
        selected_geo.geometry.notna() & ~selected_geo.geometry.is_empty
    ].copy()

    if selected_geo.empty:
        st.warning("The selected municipality boundary was not found in the GeoPackage.")
        return

    selected_geo = selected_geo.reset_index(drop=True)
    selected_geo["_map_id"] = selected_geo.index.astype(str)
    geojson_data = json.loads(selected_geo.to_json())

    min_x, min_y, max_x, max_y = selected_geo.total_bounds
    longitude_span = max(float(max_x - min_x), 0.02)
    latitude_span = max(float(max_y - min_y), 0.02)
    map_span = max(longitude_span, latitude_span)
    map_zoom = max(5.5, min(10.5, math.log2(360 / map_span) - 1.4))
    map_colour = risk_colour(overall_category)

    name_col = exact_column(
        selected_geo,
        ["adm3_name", "municipality", "municipality_name", "adm3_en"],
    )
    district_col = exact_column(
        selected_geo,
        ["adm2_name", "district", "district_name", "adm2_en"],
    )
    province_col = exact_column(
        selected_geo,
        ["adm1_name", "province", "province_name", "adm1_en"],
    )

    map_row = selected_geo.iloc[0]
    hover_text = (
        f"<b>{html.escape(safe_text(row_value(map_row, name_col)))}</b><br>"
        f"District: {html.escape(safe_text(row_value(map_row, district_col)))}<br>"
        f"Province: {html.escape(safe_text(row_value(map_row, province_col)))}<br>"
        f"Overall risk: {html.escape(safe_text(overall_category))}"
    )

    trace_values = {
        "geojson": geojson_data,
        "locations": selected_geo["_map_id"],
        "z": [1] * len(selected_geo),
        "featureidkey": "properties._map_id",
        "colorscale": [[0, map_colour], [1, map_colour]],
        "zmin": 0,
        "zmax": 1,
        "marker_opacity": 0.82,
        "marker_line_color": "#ffffff",
        "marker_line_width": 2.5,
        "showscale": False,
        "text": [hover_text] * len(selected_geo),
        "hovertemplate": "%{text}<extra></extra>",
    }
    map_settings = {
        "style": "carto-positron",
        "center": {
            "lon": float((min_x + max_x) / 2),
            "lat": float((min_y + max_y) / 2),
        },
        "zoom": map_zoom,
    }

    # Use Plotly's current MapLibre trace, with support for older Plotly versions.
    if hasattr(go, "Choroplethmap"):
        figure = go.Figure(go.Choroplethmap(**trace_values))
        figure.update_layout(map=map_settings)
    else:
        figure = go.Figure(go.Choroplethmapbox(**trace_values))
        figure.update_layout(mapbox=map_settings)

    figure.update_layout(
        height=290,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    st.plotly_chart(figure, use_container_width=True)

def category_count_plot(df, title):
    """Create category count bar chart."""
    if df.empty:
        st.info("Table not available.")
        return

    category_col = first_matching_column(df, ["category", "risk", "hazard"])
    count_col = first_matching_column(df, ["count", "number", "total"])

    if category_col and count_col:
        plot_df = df[[category_col, count_col]].copy()
        plot_df[count_col] = pd.to_numeric(plot_df[count_col], errors="coerce")
        plot_df = plot_df.dropna(subset=[count_col])

        fig = px.bar(
            plot_df,
            x=category_col,
            y=count_col,
            text=count_col,
            title=title,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=55, b=30))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True, height=260)

def top_score_plot(df, title, top_n=15):
    """Create top score horizontal bar chart."""
    if df.empty:
        st.info("Table not available.")
        return

    score_cols = [c for c in df.columns if "score" in c.lower()]
    if not score_cols:
        st.dataframe(df.head(top_n), use_container_width=True, height=300)
        return

    score_col = score_cols[0]

    name_col = first_matching_column(
        df,
        [
            "municipality",
            "admin3",
            "adm3",
            "palika",
            "local",
            "lake",
            "name",
            "id",
        ],
    )

    if name_col is None:
        name_col = df.columns[0]

    plot_df = df[[name_col, score_col]].copy()
    plot_df[score_col] = pd.to_numeric(plot_df[score_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[score_col])
    plot_df = plot_df.sort_values(score_col, ascending=False).head(top_n)

    fig = px.bar(
        plot_df.sort_values(score_col),
        x=score_col,
        y=name_col,
        orientation="h",
        title=title,
    )
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=55, b=30))
    st.plotly_chart(fig, use_container_width=True)


def prepare_metric_plot(df):
    """Prepare model metrics for plotting."""
    if df.empty:
        return pd.DataFrame()

    model_col = first_matching_column(df, ["model"])
    task_col = first_matching_column(df, ["hazard", "task", "dataset", "target"])

    metric_cols = []
    for col in df.columns:
        low = col.lower()
        if any(k in low for k in ["accuracy", "precision", "recall", "f1", "auc"]):
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            if numeric_series.notna().sum() > 0:
                metric_cols.append(col)

    if not model_col or not metric_cols:
        return pd.DataFrame()

    id_cols = [model_col]
    if task_col and task_col != model_col:
        id_cols.append(task_col)

    long_df = df[id_cols + metric_cols].copy()

    for col in metric_cols:
        long_df[col] = pd.to_numeric(long_df[col], errors="coerce")

    long_df = long_df.melt(
        id_vars=id_cols,
        value_vars=metric_cols,
        var_name="Metric",
        value_name="Value",
    )

    long_df = long_df.dropna(subset=["Value"])
    long_df = long_df.rename(columns={model_col: "Model"})

    if task_col and task_col in long_df.columns:
        long_df = long_df.rename(columns={task_col: "Hazard"})
    else:
        long_df["Hazard"] = "Model"

    return long_df

def make_file_table(files, base_folder):
    """Create simple file list table."""
    rows = []
    for file in files:
        try:
            relative_path = file.relative_to(base_folder)
        except Exception:
            relative_path = file

        rows.append(
            {
                "File": str(relative_path),
                "Folder": file.parent.name,
                "Size KB": round(file.stat().st_size / 1024, 1),
            }
        )

    return pd.DataFrame(rows)


# Load files from Project_Outputs
map_files = {
    "Flood ML Risk Map": find_file(FINAL_MAPS, ["flood", "ml", "risk"], [".png", ".jpg", ".jpeg"]),
    "Landslide ML Risk Map": find_file(FINAL_MAPS, ["landslide", "ml", "risk"], [".png", ".jpg", ".jpeg"]),
    "GLOF GIS Risk Map": find_file(FINAL_MAPS, ["glof", "gis", "risk"], [".png", ".jpg", ".jpeg"]),
    "GLOF Lake-Level Risk Map": find_file(FINAL_MAPS, ["glof", "lake"], [".png", ".jpg", ".jpeg"]),
    "Multi-Hazard Priority Map": find_file(FINAL_MAPS, ["multi", "hazard", "priority"], [".png", ".jpg", ".jpeg"]),
    "Dominant Hazard Map": find_file(FINAL_MAPS, ["dominant", "hazard"], [".png", ".jpg", ".jpeg"]),
}

all_csv_files = (
    list_files(FINAL_TABLES, [".csv"])
    + list_files(FINAL_QGIS, [".csv"])
)
all_png_files = (
    list_files(FINAL_MAPS, [".png", ".jpg", ".jpeg"])
    + list_files(FINAL_PLOTS, [".png", ".jpg", ".jpeg"])
)
all_model_files = list_files(FINAL_MODELS, [".joblib"])
all_note_files = list_files(FINAL_NOTES, [".txt", ".md"])
all_qgis_files = list_files(FINAL_QGIS, [".csv", ".gpkg"])

# Important tables
model_metrics_path = find_file(FINAL_TABLES, ["model", "metrics", "combined"], [".csv"])
best_model_path = find_file(FINAL_TABLES, ["best", "model", "summary"], [".csv"])

glof_lake_counts_path = find_file(FINAL_TABLES, ["glof", "lake", "category", "counts"], [".csv"])
glof_admin_counts_path = find_file(FINAL_TABLES, ["glof", "admin3", "category", "counts"], [".csv"])
glof_lake_scores_path = find_file(FINAL_TABLES, ["glof", "lake", "risk", "scores"], [".csv"])
glof_admin_scores_path = find_file(FINAL_TABLES, ["glof", "admin3", "risk", "scores"], [".csv"])

multi_counts_path = find_file(FINAL_TABLES, ["multi", "hazard", "risk", "category", "counts"], [".csv"])
multi_summary_path = find_file(FINAL_TABLES, ["multi", "hazard", "summary"], [".csv"])
multi_top30_path = find_file(FINAL_TABLES, ["top", "30", "admin3", "multi"], [".csv"])
multi_admin_scores_path = find_file(FINAL_TABLES, ["admin3", "multi", "hazard", "risk", "scores"], [".csv"])
multi_admin_gpkg_path = find_file(
    FINAL_QGIS,
    ["admin3", "multi", "hazard", "risk", "scores"],
    [".gpkg"],
)

model_metrics = read_csv_file(model_metrics_path)
best_model_summary = read_csv_file(best_model_path)

glof_lake_counts = read_csv_file(glof_lake_counts_path)
glof_admin_counts = read_csv_file(glof_admin_counts_path)
glof_lake_scores = read_csv_file(glof_lake_scores_path)
glof_admin_scores = read_csv_file(glof_admin_scores_path)

multi_counts = read_csv_file(multi_counts_path)
multi_summary = read_csv_file(multi_summary_path)
multi_top30 = read_csv_file(multi_top30_path)
multi_admin_scores = read_csv_file(multi_admin_scores_path)

# Plot groups
plot_groups = {
    "ML EDA and Evaluation Plots": FINAL_PLOTS / "ML_Plots",
    "ML Feature Importance Plots": FINAL_PLOTS / "ML_Feature_Importance_Plots",
    "GLOF Plots": FINAL_PLOTS / "GLOF_Plots",
    "Multi-Hazard Plots": FINAL_PLOTS / "Multi_Hazard_Plots",
}


# Sidebar
st.sidebar.title("Dashboard Menu")

page = st.sidebar.radio(
    "Select section",
    [
        "Overview",
        "Final Maps",
        "Final Plots",
        "ML Model Results",
        "GLOF Results",
        "Multi-Hazard Results",
        "Data Explorer",
        "Project Files Check",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Data source: Project_Outputs folder")

# Header
st.title("Nepal Multi-Hazard Early Warning Dashboard")
st.markdown(
    "<div class='small-text'>Machine Learning and GIS-Based Early Warning Prototype for Multi-Hazard Risk in Nepal</div>",
    unsafe_allow_html=True,
)

# Overview
if page == "Overview":
    st.markdown(
        """
        <style>
        div[data-testid="stMainBlockContainer"] > div[data-testid="stVerticalBlock"] {
            gap: 0.66rem;
        }

        div[data-baseweb="select"] > div {
            min-height: 44px;
        }

        div[data-testid="stSelectbox"] {
            margin-top: 0.15rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="overview-hero">
            <div class="overview-hero-kicker">Interactive municipality-level research prototype</div>
            <div class="overview-hero-title">Explore flood, landslide, GLOF and integrated risk across Nepal</div>
            <div class="overview-hero-text">
                Search any municipality to compare its three hazard results, overall priority zone
                and mapped admin3 boundary in one view.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Loading municipality risk explorer..."):
        municipality_source, _ = municipality_search_source(
            multi_admin_gpkg_path,
            multi_admin_scores,
        )

    if municipality_source.empty:
        municipality_count = 0
    else:
        source_columns = municipality_columns(municipality_source)
        code_column = source_columns.get("code")
        municipality_count = (
            municipality_source[code_column].nunique()
            if code_column is not None
            else len(municipality_source)
        )

    highlight_col1, highlight_col2, highlight_col3, highlight_col4 = st.columns(4)
    highlight_cards = [
        (highlight_col1, municipality_count, "Municipalities", "#2878a8"),
        (highlight_col2, 3, "Hazards", "#2e7d32"),
        (highlight_col3, 4, "ML models", "#7b5cb8"),
        (
            highlight_col4,
            len([path for path in map_files.values() if path is not None]),
            "Final maps",
            "#ef6c00",
        ),
    ]

    for card_column, number, label, colour in highlight_cards:
        with card_column:
            st.markdown(
                f"<div class='metric-card' style='border-top-color:{colour};'>"
                f"<div class='metric-number'>{number}</div>"
                f"<div class='metric-label'>{html.escape(label)}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:0.2rem;'></div>", unsafe_allow_html=True)

    if municipality_source.empty:
        st.warning(
            "The integrated municipality GeoPackage and CSV table were not found."
        )
    else:
        columns = municipality_columns(municipality_source)
        required_fields = {
            "province": columns["province"],
            "district": columns["district"],
            "municipality": columns["municipality"],
            "admin3 code": columns["code"],
            "flood category": columns["flood_category"],
            "landslide category": columns["landslide_category"],
            "GLOF category": columns["glof_category"],
            "multi-hazard category": columns["overall_category"],
        }
        missing_fields = [
            label for label, column in required_fields.items() if column is None
        ]

        if missing_fields:
            st.warning(
                "The municipality explorer cannot be displayed because these fields "
                f"were not found: {', '.join(missing_fields)}."
            )
        else:
            search_df = prepare_municipality_search(municipality_source, columns)
            selected_label = st.selectbox(
                "Search and select municipality",
                search_df["_display_label"].tolist(),
                index=None,
                placeholder="Type municipality, district, province or admin3 code...",
                key="overview_municipality_selection",
            )

            if selected_label is not None:
                selected_row = search_df.loc[
                    search_df["_display_label"] == selected_label
                ].iloc[0]

                municipality_name = safe_text(
                    row_value(selected_row, columns["municipality"])
                )
                district_name = safe_text(
                    row_value(selected_row, columns["district"])
                )
                province_name = safe_text(
                    row_value(selected_row, columns["province"])
                )
                municipality_code = safe_text(
                    row_value(selected_row, columns["code"])
                )
                dominant_hazard = safe_text(
                    row_value(selected_row, columns["dominant_hazard"])
                )
                overall_category = safe_text(
                    row_value(selected_row, columns["overall_category"])
                )

                st.markdown(
                    "<div class='selected-municipality-bar'>"
                    "<div>"
                    f"<div class='selected-place'>{html.escape(municipality_name)}, "
                    f"{html.escape(district_name)}</div>"
                    f"<div class='selected-meta'>{html.escape(province_name)} &nbsp;•&nbsp; "
                    f"{html.escape(municipality_code)}</div>"
                    "</div>"
                    "<div class='selected-status'>"
                    f"<span class='dominant-chip'>Dominant: {html.escape(dominant_hazard)}</span>"
                    f"{risk_badge_html(overall_category, ' overall risk')}"
                    "</div></div>",
                    unsafe_allow_html=True,
                )

                result_col, map_col = st.columns([0.95, 1.25], gap="large")

                with result_col:
                    risk_df = create_municipality_risk_table(selected_row, columns)
                    display_municipality_risk_table(risk_df)
                    st.caption(
                        "The overall priority is the maximum of the flood, landslide "
                        "and GLOF scores."
                    )

                with map_col:
                    st.markdown(
                        "<div class='map-title-card'>"
                        f"<div class='map-title-name'>{html.escape(municipality_name)}, "
                        f"{html.escape(district_name)}</div>"
                        f"{risk_badge_html(overall_category, ' risk')}"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    with st.spinner("Loading selected municipality boundary..."):
                        display_municipality_map(
                            multi_admin_gpkg_path,
                            municipality_code,
                            overall_category,
                        )

    with st.expander("About the prototype and final project outputs"):
        st.write(
            "This research prototype combines flood and landslide machine-learning "
            "risk, GLOF GIS-based scoring and an integrated warning-focused "
            "multi-hazard priority result at admin3 municipality level."
        )
        output_col1, output_col2 = st.columns(2)
        with output_col1:
            st.markdown(
                """
                - Flood and landslide ML risk maps
                - GLOF municipality and lake-level maps
                - Multi-hazard priority and dominant-hazard maps
                """
            )
        with output_col2:
            st.markdown(
                """
                - Model evaluation and feature-importance results
                - Municipality-level risk tables and QGIS layers
                - Final plots, models and methodology notes
                """
            )

# Final maps
elif page == "Final Maps":
    st.subheader("Final map viewer")

    selected_map = st.selectbox("Select map", list(map_files.keys()))
    display_image_fit(map_files[selected_map], box_type="image")

    map_path = map_files[selected_map]
    if map_path is not None and map_path.exists():
        st.caption(f"File: {map_path.name}")

# Final plots

elif page == "Final Plots":
    st.subheader("Final plots viewer")

    selected_group = st.selectbox("Select plot group", list(plot_groups.keys()))
    selected_folder = plot_groups[selected_group]

    plot_files = list_files(selected_folder, [".png", ".jpg", ".jpeg"])

    if not plot_files:
        st.warning("No plot images found in this folder.")
    else:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.write(f"Plots found: {len(plot_files)}")

            selected_plot = st.selectbox(
                "Select plot",
                plot_files,
                format_func=lambda p: p.name,
            )

            st.caption(f"Folder: {selected_folder.name}")

        with col2:
            st.write(clean_name(selected_plot.stem))

        display_image_fit(selected_plot, box_type="plot")

        with st.expander("View all plot files in this group"):
            plot_table = make_file_table(plot_files, FINAL_PLOTS)
            st.dataframe(plot_table, use_container_width=True, height=260)

# ML model results

elif page == "ML Model Results":
    st.subheader("Flood and landslide ML model results")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Model comparison", "Best model summary", "Feature importance tables", "Model plots"]
    )

    with tab1:
        st.markdown("### Model performance comparison")

        if model_metrics.empty:
            st.warning("Model metrics table was not found.")
        else:
            metric_long = prepare_metric_plot(model_metrics)

            if metric_long.empty:
                st.dataframe(model_metrics, use_container_width=True, height=360)
            else:
                selected_metric = st.selectbox(
                    "Select metric",
                    sorted(metric_long["Metric"].dropna().unique()),
                )

                plot_df = metric_long[metric_long["Metric"] == selected_metric].copy()

                fig = px.bar(
                    plot_df,
                    x="Model",
                    y="Value",
                    color="Hazard",
                    barmode="group",
                    title=f"Model comparison by {clean_name(selected_metric)}",
                )
                fig.update_layout(height=390, margin=dict(l=20, r=20, t=55, b=80))
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(model_metrics, use_container_width=True, height=260)
                show_download_button(model_metrics, "model_metrics.csv", "Download model metrics")

    with tab2:
        st.markdown("### Best model summary")

        if best_model_summary.empty:
            st.warning("Best model summary table was not found.")
        else:
            st.dataframe(best_model_summary, use_container_width=True, height=320)
            show_download_button(best_model_summary, "best_model_summary.csv", "Download best model summary")

    with tab3:
        st.markdown("### Feature importance tables")

        feature_files = [
            p for p in list_files(PROJECT_OUTPUTS, [".csv"])
            if "feature_importance" in p.stem.lower()
        ]

        if not feature_files:
            st.warning("Feature importance CSV files were not found.")
        else:
            selected_feature_file = st.selectbox(
                "Select feature importance table",
                feature_files,
                format_func=lambda p: p.name,
            )

            feature_df = read_csv_file(selected_feature_file)
            st.dataframe(feature_df.head(30), use_container_width=True, height=280)

            if not feature_df.empty:
                score_col = first_matching_column(
                    feature_df,
                    ["importance", "coefficient", "score", "value"],
                )
                feature_col = first_matching_column(
                    feature_df,
                    ["feature", "variable", "predictor"],
                )

                if feature_col and score_col:
                    plot_df = feature_df[[feature_col, score_col]].copy()
                    plot_df[score_col] = pd.to_numeric(plot_df[score_col], errors="coerce")
                    plot_df = plot_df.dropna(subset=[score_col])
                    plot_df = plot_df.sort_values(score_col, ascending=False).head(15)

                    fig = px.bar(
                        plot_df.sort_values(score_col),
                        x=score_col,
                        y=feature_col,
                        orientation="h",
                        title="Top 15 important features",
                    )
                    fig.update_layout(height=430, margin=dict(l=20, r=20, t=55, b=30))
                    st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("### ML evaluation and feature importance plots")

        ml_plot_choices = {
            "Evaluation / EDA plots": FINAL_PLOTS / "ML_Plots",
            "Feature importance plots": FINAL_PLOTS / "ML_Feature_Importance_Plots",
        }

        selected_ml_plot_group = st.selectbox("Select ML plot type", list(ml_plot_choices.keys()))
        ml_plot_files = list_files(ml_plot_choices[selected_ml_plot_group], [".png", ".jpg", ".jpeg"])

        if not ml_plot_files:
            st.warning("No ML plot files were found.")
        else:
            selected_ml_plot = st.selectbox(
                "Select ML plot",
                ml_plot_files,
                format_func=lambda p: p.name,
            )
            display_image_fit(selected_ml_plot, box_type="plot")

# GLOF results

elif page == "GLOF Results":
    st.subheader("GLOF GIS risk results")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Risk category counts", "Top lake risks", "Top municipality risks", "GLOF plots"]
    )

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            category_count_plot(glof_lake_counts, "Lake-level GLOF risk category counts")

        with col2:
            category_count_plot(glof_admin_counts, "Municipality-level GLOF risk category counts")

    with tab2:
        top_score_plot(glof_lake_scores, "Top glacial lakes by GLOF risk score", top_n=15)

        with st.expander("View lake-level GLOF table"):
            st.dataframe(glof_lake_scores, use_container_width=True, height=320)
            show_download_button(glof_lake_scores, "glof_lake_risk_scores.csv", "Download lake risk scores")

    with tab3:
        top_score_plot(glof_admin_scores, "Top municipalities by GLOF risk score", top_n=15)

        with st.expander("View municipality-level GLOF table"):
            st.dataframe(glof_admin_scores, use_container_width=True, height=320)
            show_download_button(glof_admin_scores, "glof_admin3_risk_scores.csv", "Download admin3 GLOF scores")

    with tab4:
        glof_plot_files = list_files(FINAL_PLOTS / "GLOF_Plots", [".png", ".jpg", ".jpeg"])

        if not glof_plot_files:
            st.warning("No GLOF plot files were found.")
        else:
            selected_glof_plot = st.selectbox(
                "Select GLOF plot",
                glof_plot_files,
                format_func=lambda p: p.name,
            )
            display_image_fit(selected_glof_plot, box_type="plot")

# Multi-hazard results
elif page == "Multi-Hazard Results":
    st.subheader("Integrated multi-hazard results")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Priority categories",
            "Top priority municipalities",
            "Dominant hazards",
            "Summary tables",
            "Multi-hazard plots",
        ]
    )

    with tab1:
        category_count_plot(multi_counts, "Multi-hazard priority category counts")

    with tab2:
        if not multi_top30.empty:
            top_score_plot(multi_top30, "Top priority municipalities", top_n=15)
            st.dataframe(multi_top30, use_container_width=True, height=300)
            show_download_button(
                multi_top30,
                "top_30_admin3_multi_hazard_priority.csv",
                "Download top 30 table",
            )
        else:
            top_score_plot(multi_admin_scores, "Top priority municipalities", top_n=15)

    with tab3:
        if not multi_admin_scores.empty:
            dominant_col = first_matching_column(
                multi_admin_scores,
                ["dominant_hazard", "dominant hazard"],
            )

            if dominant_col:
                dom_df = multi_admin_scores[dominant_col].value_counts().reset_index()
                dom_df.columns = ["Dominant hazard", "Count"]

                fig = px.pie(
                    dom_df,
                    names="Dominant hazard",
                    values="Count",
                    title="Dominant hazard distribution",
                    hole=0.35,
                )
                fig.update_layout(height=380, margin=dict(l=20, r=20, t=55, b=20))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Dominant hazard column was not found in the table.")
        else:
            st.warning("Multi-hazard admin3 score table was not found.")

    with tab4:
        st.markdown("### Multi-hazard summary")

        if multi_summary.empty:
            st.warning("Multi-hazard summary table was not found.")
        else:
            st.dataframe(multi_summary, use_container_width=True, height=260)
            show_download_button(
                multi_summary,
                "multi_hazard_summary.csv",
                "Download multi-hazard summary",
            )

        with st.expander("View full multi-hazard admin3 table"):
            st.dataframe(multi_admin_scores, use_container_width=True, height=340)

    with tab5:
        multi_plot_files = list_files(FINAL_PLOTS / "Multi_Hazard_Plots", [".png", ".jpg", ".jpeg"])

        if not multi_plot_files:
            st.warning("No multi-hazard plot files were found.")
        else:
            selected_multi_plot = st.selectbox(
                "Select multi-hazard plot",
                multi_plot_files,
                format_func=lambda p: p.name,
            )
            display_image_fit(selected_multi_plot, box_type="plot")

# Data explorer

elif page == "Data Explorer":
    st.subheader("Final output data explorer")

    csv_files = all_csv_files

    if not csv_files:
        st.warning("No final CSV files found in Project_Outputs.")
    else:
        selected_csv = st.selectbox(
            "Select a CSV file",
            csv_files,
            format_func=lambda p: str(p.relative_to(PROJECT_OUTPUTS)),
        )

        df = read_csv_file(selected_csv)

        st.caption(f"File: {selected_csv.relative_to(PROJECT_OUTPUTS)}")
        st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

        st.dataframe(df, use_container_width=True, height=470)
        show_download_button(df, selected_csv.name, "Download selected table")

# Project files check

elif page == "Project Files Check":
    st.subheader("Project files check")

    checks = [
        ("Project outputs folder", PROJECT_OUTPUTS),
        ("Final maps", FINAL_MAPS),
        ("Final models", FINAL_MODELS),
        ("Final notes", FINAL_NOTES),
        ("Final plots", FINAL_PLOTS),
        ("Final QGIS layers", FINAL_QGIS),
        ("Final tables", FINAL_TABLES),
    ]

    check_df = pd.DataFrame(
        [
            {
                "Item": name,
                "Path": str(path),
                "Available": "Yes" if path.exists() else "No",
            }
            for name, path in checks
        ]
    )

    st.dataframe(check_df, use_container_width=True, height=260)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Map/image files", len(all_png_files))

    with col2:
        st.metric("CSV tables", len(all_csv_files))

    with col3:
        st.metric("Model files", len(all_model_files))

    with st.expander("View discovered map and plot files"):
        image_list = make_file_table(all_png_files, PROJECT_OUTPUTS)
        st.dataframe(image_list, use_container_width=True, height=280)

    with st.expander("View discovered CSV files"):
        csv_list = make_file_table(all_csv_files, PROJECT_OUTPUTS)
        st.dataframe(csv_list, use_container_width=True, height=300)

    with st.expander("View discovered QGIS files"):
        qgis_list = make_file_table(all_qgis_files, PROJECT_OUTPUTS)
        st.dataframe(qgis_list, use_container_width=True, height=260)

    with st.expander("View discovered methodology note files"):
        note_list = make_file_table(all_note_files, PROJECT_OUTPUTS)
        st.dataframe(note_list, use_container_width=True, height=260)

# Footer
st.markdown("---")
st.caption(
    "Dashboard source folder: Project_Outputs | Prototype outputs for flood ML, landslide ML, GLOF GIS risk, and multi-hazard integration."
)
