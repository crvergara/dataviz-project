from pathlib import Path
from html import escape
import json
import sys
import unicodedata
from urllib.parse import quote

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.final_infographic import (  # noqa: E402
    COLOR_ALMOND,
    COLOR_BG,
    COLOR_RESTO,
    COLOR_RM,
    COLOR_TEXT,
    COLOR_WARM,
    REGION_NAMES,
    build_regional_vulnerability_metrics,
    calculate_spatial_autocorrelation,
    geometry_to_mainland_polygons,
    polygon_area,
    plot_donut_vulnerability,
    plot_ds49_yoy_growth,
    plot_historical_line,
    plot_radar_rm_vs_regions,
    prepare_ds49_yoy,
    read_ds49_region_totals,
)


st.set_page_config(
    page_title="Viviendas Sociales en Chile",
    layout="wide",
    initial_sidebar_state="collapsed",
)


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Roboto", "Arial", "DejaVu Sans"],
        "figure.facecolor": COLOR_BG,
        "axes.facecolor": COLOR_BG,
        "text.color": COLOR_TEXT,
        "axes.labelcolor": COLOR_TEXT,
        "xtick.color": COLOR_TEXT,
        "ytick.color": COLOR_TEXT,
    }
)


CUSTOM_CSS = f"""
<style>
    .stApp {{
        background: {COLOR_BG};
        color: {COLOR_TEXT};
    }}

    header[data-testid="stHeader"],
    div[data-testid="stToolbar"] {{
        display: none;
    }}

    .block-container {{
        padding-top: 1.45rem;
        padding-bottom: 2.8rem;
        max-width: 1380px;
    }}

    section[data-testid="stSidebar"] {{
        background-color: #f1ead8;
        border-right: 1px solid rgba(211, 139, 93, 0.32);
    }}

    h1, h2, h3, h4, p, label, span {{
        color: {COLOR_TEXT};
    }}

    .main-title {{
        color: {COLOR_TEXT};
        font-size: clamp(2.3rem, 5vw, 4.6rem);
        line-height: 1.0;
        font-weight: 900;
        letter-spacing: 0;
        text-align: center;
        margin: 0.10rem auto 0.45rem;
        text-transform: uppercase;
    }}

    .lead {{
        max-width: 1040px;
        color: rgba(76, 46, 5, 0.78);
        font-size: clamp(0.96rem, 1.35vw, 1.18rem);
        line-height: 1.50;
        text-align: center;
        margin: 0 auto 0.45rem;
        text-wrap: pretty;
    }}

    .svg-map-shell {{
        width: 100%;
        max-width: 1280px;
        margin: 0.1rem auto 0.15rem;
    }}

    .chile-hero-map {{
        display: block;
        width: 100%;
        height: min(70vh, 710px);
        min-height: 540px;
        overflow: visible;
    }}

    .hero-map-note {{
        color: rgba(76, 46, 5, 0.58);
        font-size: 0.78rem;
        text-align: center;
        margin: -0.15rem auto 0.75rem;
    }}

    .hero-panel {{
        background: #fffaf0;
        border: 1px solid rgba(211, 139, 93, 0.36);
        border-radius: 8px;
        padding: 1.15rem 1.2rem;
        min-height: 420px;
        box-shadow: 0 1px 0 rgba(76, 46, 5, 0.05);
    }}

    .hero-panel-kicker {{
        color: rgba(76, 46, 5, 0.66);
        font-size: 0.78rem;
        font-weight: 900;
        line-height: 1.15;
        text-transform: uppercase;
        margin-bottom: 0.42rem;
    }}

    .hero-panel-title {{
        color: {COLOR_TEXT};
        font-size: 1.55rem;
        line-height: 1.08;
        font-weight: 900;
        margin-bottom: 0.65rem;
    }}

    .hero-panel-copy {{
        color: rgba(76, 46, 5, 0.74);
        font-size: 0.92rem;
        line-height: 1.42;
        margin-bottom: 1.0rem;
    }}

    .hero-panel-value {{
        color: {COLOR_RM};
        font-size: 2.35rem;
        line-height: 1.0;
        font-weight: 900;
        margin: 0.55rem 0 0.15rem;
    }}

    .hero-panel-row {{
        border-top: 1px solid rgba(211, 139, 93, 0.24);
        padding-top: 0.72rem;
        margin-top: 0.72rem;
        color: rgba(76, 46, 5, 0.76);
        font-size: 0.86rem;
        line-height: 1.35;
    }}

    .map-legend {{
        height: 10px;
        border-radius: 999px;
        background: linear-gradient(90deg, #ead8b8, #f2a85c, #d96a2b, #823519);
        margin: 0.85rem 0 0.32rem;
        border: 1px solid rgba(76, 46, 5, 0.08);
    }}

    .legend-scale {{
        display: flex;
        justify-content: space-between;
        color: rgba(76, 46, 5, 0.62);
        font-size: 0.74rem;
        font-weight: 800;
    }}

    .analysis-title {{
        color: {COLOR_TEXT};
        font-size: 1.42rem;
        line-height: 1.1;
        font-weight: 900;
        margin: 0.25rem 0 0.85rem;
    }}

    .section-label {{
        color: {COLOR_TEXT};
        font-weight: 900;
        font-size: 1.03rem;
        margin: 0.25rem 0 0.75rem;
    }}

    .chart-explainer {{
        background: #fffaf0;
        border: 1px solid rgba(211, 139, 93, 0.34);
        border-left: 4px solid {COLOR_RM};
        border-radius: 8px;
        padding: 1.05rem 1.15rem 1.1rem;
        box-shadow: 0 1px 0 rgba(76, 46, 5, 0.04);
    }}

    .explainer-kicker {{
        color: {COLOR_RM};
        font-size: 0.72rem;
        font-weight: 900;
        letter-spacing: 0.04em;
        line-height: 1.15;
        text-transform: uppercase;
        margin-bottom: 0.34rem;
    }}

    .explainer-title {{
        color: {COLOR_TEXT};
        font-size: 1.16rem;
        line-height: 1.12;
        font-weight: 900;
        margin-bottom: 0.5rem;
    }}

    .explainer-body {{
        color: rgba(76, 46, 5, 0.78);
        font-size: 0.9rem;
        line-height: 1.46;
    }}

    .explainer-body b {{
        color: {COLOR_TEXT};
    }}

    .explainer-takeaway {{
        margin-top: 0.7rem;
        padding-top: 0.65rem;
        border-top: 1px solid rgba(211, 139, 93, 0.26);
        color: rgba(76, 46, 5, 0.86);
        font-size: 0.86rem;
        line-height: 1.4;
    }}

    .tab-intro {{
        color: rgba(76, 46, 5, 0.74);
        font-size: 0.92rem;
        line-height: 1.5;
        max-width: 980px;
        margin: -0.1rem 0 1.1rem;
    }}

    .metric-card {{
        background: #fffaf0;
        border: 1px solid rgba(211, 139, 93, 0.34);
        border-top: 4px solid {COLOR_RM};
        border-radius: 8px;
        padding: 0.82rem 0.9rem 0.74rem;
        min-height: 116px;
        box-shadow: 0 1px 0 rgba(76, 46, 5, 0.04);
    }}

    .metric-label {{
        color: rgba(76, 46, 5, 0.70);
        font-size: 0.76rem;
        font-weight: 800;
        line-height: 1.18;
        text-transform: uppercase;
    }}

    .metric-value {{
        color: {COLOR_TEXT};
        font-size: 1.72rem;
        line-height: 1.08;
        font-weight: 900;
        margin-top: 0.36rem;
    }}

    .metric-detail {{
        color: rgba(76, 46, 5, 0.68);
        font-size: 0.78rem;
        line-height: 1.2;
        margin-top: 0.28rem;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-color: rgba(211, 139, 93, 0.34);
        background: rgba(255, 250, 240, 0.42);
        border-radius: 8px;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.35rem;
    }}

    .stTabs [data-baseweb="tab"] {{
        background: #fffaf0;
        border: 1px solid rgba(211, 139, 93, 0.30);
        border-radius: 7px;
        color: {COLOR_TEXT};
        font-weight: 800;
        padding: 0.55rem 0.95rem;
    }}

    .stTabs [aria-selected="true"] {{
        color: {COLOR_RM};
        border-color: rgba(241, 145, 67, 0.70);
    }}

    button[kind="primary"], div[data-testid="stDownloadButton"] button {{
        background-color: {COLOR_RM};
        border-color: {COLOR_RM};
        color: #fffaf0;
        font-weight: 800;
    }}
</style>
"""


def normalize_text(value):
    text = str(value).strip().lower()
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def build_historical_region_dataset(raw_dir, cache_path):
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    years = {
        2015: ("casen_2015.dta", "v38e", "expr", "v16"),
        2017: ("casen_2017.dta", "v38e", "expr", "v15"),
        2022: ("casen_2022.dta", "v36e", "expr", "v15"),
        2024: ("casen_2024.dta", "v36e", "expr", "v15"),
    }

    frames = []
    for year, (filename, balacera_col, weight_col, subsidy_col) in years.items():
        path = raw_dir / filename
        if not path.exists():
            continue

        target_cols = ["region", "r", weight_col, subsidy_col, balacera_col, "pco1", "id_persona", "parentesco"]
        sample = pd.read_stata(path, iterator=True).read(1)
        cols_to_load = [col for col in target_cols if col in sample.columns]
        data = pd.read_stata(path, columns=cols_to_load, convert_categoricals=False)

        if "r" in data.columns and "region" not in data.columns:
            data = data.rename(columns={"r": "region"})

        head_col = next((col for col in ["pco1", "id_persona", "parentesco"] if col in data.columns), None)
        if head_col is not None:
            data = data[data[head_col].eq(1)]

        data = data.dropna(subset=["region", balacera_col])
        data = data[data[subsidy_col].isin([1, 2])].copy()
        data["region"] = data["region"].astype(int)
        data["ponderador"] = data[weight_col]
        data["is_balacera"] = np.where(data[balacera_col].isin([3, 4]), 1, 0)
        data["year"] = year
        frames.append(data[["year", "region", "is_balacera", "ponderador"]])

    if not frames:
        return pd.DataFrame(columns=["year", "region", "is_balacera", "ponderador"])

    result = pd.concat(frames, ignore_index=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cache_path, engine="pyarrow")
    return result


@st.cache_data(show_spinner=False)
def load_project_data():
    df_master = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "master_dataset.parquet")
    df_hist = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "balaceras_historico_zonas.parquet")
    df_hist_regions = build_historical_region_dataset(
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "data" / "processed" / "balaceras_historico_regiones.parquet",
    )
    ds49_totals = read_ds49_region_totals(PROJECT_ROOT / "data" / "raw" / "SUDS49Mar2026.xlsx")
    df_ds49_yoy = prepare_ds49_yoy(ds49_totals)

    with open(PROJECT_ROOT / "data" / "raw" / "regiones.json", encoding="utf-8-sig") as file:
        geojson_regions = json.load(file)

    regional_metrics = build_regional_vulnerability_metrics(df_master)
    moran_i, moran_p = calculate_spatial_autocorrelation(regional_metrics, geojson_regions)
    return df_master, df_hist, df_hist_regions, ds49_totals, df_ds49_yoy, geojson_regions, regional_metrics, moran_i, moran_p


def add_vulnerability_flags(df):
    result = df.copy()
    result["trampa_violencia"] = np.where((result["v36b"] >= 3) | (result["v36c"] >= 3), 1, 0)
    result["trampa_economica"] = np.where(
        (result["yautcorh"] < 600000) | (result["subsidy_ratio"] > 0.20),
        1,
        0,
    )
    result["vulnerabilidad_estructural"] = np.where(
        (result["trampa_violencia"] == 1) | (result["trampa_economica"] == 1),
        1,
        0,
    )
    result["bajos_ingresos"] = np.where(result["yautcorh"] < 600000, 1, 0)
    result["alta_dependencia"] = np.where(result["subsidy_ratio"] > 0.20, 1, 0)
    return result


def weighted_pct(df, mask):
    total = df["expr"].sum()
    if total <= 0:
        return np.nan
    return df.loc[mask, "expr"].sum() / total * 100


def fmt_pct(value, decimals=1):
    if pd.isna(value):
        return "-"
    return f"{value:.{decimals}f}%"


def fmt_int(value):
    if pd.isna(value):
        return "-"
    return f"{int(round(value)):,.0f}".replace(",", ".")


def get_region_options():
    ordered_regions = [REGION_NAMES[code] for code in sorted(REGION_NAMES) if code != 13]
    return ["Chile completo", "Santiago (RM)", "Regiones"] + ordered_regions


def region_display_name(region_code):
    return "Santiago (RM)" if int(region_code) == 13 else REGION_NAMES.get(int(region_code), str(region_code))


def territory_to_region_code(selection):
    if selection == "Santiago (RM)":
        return 13
    reverse = {name: code for code, name in REGION_NAMES.items()}
    return reverse.get(selection)


def filter_by_territory(df, selection):
    if selection == "Chile completo":
        return df.copy()
    if selection == "Santiago (RM)":
        return df[df["region"].eq(13)].copy()
    if selection == "Regiones":
        return df[~df["region"].eq(13)].copy()
    region_code = territory_to_region_code(selection)
    if region_code is None:
        return df.copy()
    return df[df["region"].eq(region_code)].copy()


def ds49_beneficiados_2025(ds49_totals, selection):
    valid = ds49_totals[
        ~ds49_totals["region"].str.startswith("Total Pa")
        & ~ds49_totals["region"].str.startswith("Sin Informaci")
    ].copy()

    if selection == "Chile completo":
        return valid[2025].sum()
    if selection == "Santiago (RM)":
        return valid.loc[valid["region"].eq("Metropolitana"), 2025].sum()
    if selection == "Regiones":
        return valid.loc[~valid["region"].eq("Metropolitana"), 2025].sum()

    normalized_selection = normalize_text(selection)
    rows = valid[valid["region"].map(normalize_text).eq(normalized_selection)]
    return rows[2025].sum() if not rows.empty else np.nan


def calculate_summary_metrics(df_context, ds49_totals, selection):
    df_flags = add_vulnerability_flags(df_context)
    df_sub = df_flags[df_flags["grupo_vivienda"].eq("Subsidiada")].copy()

    return {
        "total_subsidiadas": df_sub["expr"].sum(),
        "vulnerabilidad": weighted_pct(df_sub, df_sub["vulnerabilidad_estructural"].eq(1)),
        "violencia": weighted_pct(df_sub, df_sub["trampa_violencia"].eq(1)),
        "economia": weighted_pct(df_sub, df_sub["trampa_economica"].eq(1)),
        "ds49_2025": ds49_beneficiados_2025(ds49_totals, selection),
    }


def metric_card(label, value, detail, accent):
    st.markdown(
        f"""
        <div class="metric-card" style="border-top-color: {accent};">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_matplotlib(fig):
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def chart_explainer(kicker, title, body, takeaway=None):
    takeaway_html = (
        f'<div class="explainer-takeaway">{takeaway}</div>' if takeaway else ""
    )
    st.markdown(
        f"""
        <div class="chart-explainer">
            <div class="explainer-kicker">{escape(kicker)}</div>
            <div class="explainer-title">{escape(title)}</div>
            <div class="explainer-body">{body}</div>
            {takeaway_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


DASH_TITLE_FS = 12.5
DASH_LABEL_FS = 8.5
DASH_TICK_FS = 8
DASH_LINE_FIGSIZE = (8.8, 3.6)
DASH_SQUARE_FIGSIZE = (5.6, 4.8)
DASH_BUTTERFLY_FIGSIZE = (7.2, 4.8)
# Tamaño uniforme para los graficos de la vista narrativa (misma altura/ancho en todos)
DASH_STORY_FIGSIZE = (7.6, 4.7)


def normalize_dashboard_title(ax, title=None, loc="left", pad=12):
    title = title if title is not None else ax.get_title(loc=loc) or ax.get_title()
    ax.set_title(
        title,
        loc=loc,
        fontsize=DASH_TITLE_FS,
        fontweight="black",
        color=COLOR_TEXT,
        pad=pad,
    )


def plot_dashboard_map(ax, metrics, geojson, moran_i, moran_p, selected_region=None):
    metric_lookup = metrics.set_index("region")["pct_vulnerabilidad"].to_dict()

    boundaries = np.linspace(39, 70, 11)
    cmap = ListedColormap(
        [
            "#ead8b8",
            "#f0c997",
            "#f4ba75",
            "#f2a85c",
            "#ee9445",
            "#e77f33",
            "#d96a2b",
            "#c45725",
            "#a94620",
            "#823519",
        ],
        name="vulnerability_steps",
    )
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)

    ax.set_xlim(-76.2, -66.5)
    ax.set_ylim(-56.4, -17.0)
    ax.set_aspect("auto")
    ax.set_anchor("NW")
    ax.axis("off")

    for feature in geojson["features"]:
        region_code = int(feature["properties"]["codregion"])
        value = metric_lookup.get(region_code, np.nan)
        facecolor = "#ddd6c7" if np.isnan(value) else cmap(norm(value))
        selected = selected_region is not None and region_code == selected_region

        for coords in geometry_to_mainland_polygons(feature["geometry"]):
            ax.add_patch(
                patches.Polygon(
                    coords,
                    closed=True,
                    facecolor=facecolor,
                    edgecolor=COLOR_TEXT if selected else "#fff8ea",
                    linewidth=1.6 if selected else 0.48,
                    joinstyle="round",
                    zorder=5 if selected else 2,
                )
            )

    ax.set_title(
        "1. Vulnerabilidad estructural territorial (%)",
        fontsize=12.2,
        fontweight="black",
        color=COLOR_TEXT,
        loc="left",
        pad=10,
    )
    ax.text(
        0.0,
        0.945,
        "Tasa regional en viviendas subsidiadas",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        color=COLOR_TEXT,
        alpha=0.74,
    )
    ax.text(
        0.0,
        0.900,
        f"Moran's I: {moran_i:.2f}  ·  p = {moran_p:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        fontweight="bold",
        color=COLOR_RESTO,
    )

    cax = ax.inset_axes([0.80, 0.28, 0.040, 0.38])
    gradient = np.linspace(boundaries[0], boundaries[-1], 512).reshape(-1, 1)
    cax.imshow(gradient, aspect="auto", cmap=cmap, norm=norm, origin="lower")
    cax.axis("off")
    cax.text(
        1.35,
        0.00,
        f"{boundaries[0]:.0f}%",
        transform=cax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        fontweight="bold",
        color=COLOR_TEXT,
    )
    cax.text(
        1.35,
        0.52,
        "55%",
        transform=cax.transAxes,
        ha="left",
        va="center",
        fontsize=7,
        fontweight="bold",
        color=COLOR_TEXT,
    )
    cax.text(
        1.35,
        1.00,
        f"{boundaries[-1]:.0f}%+",
        transform=cax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        fontweight="bold",
        color=COLOR_TEXT,
    )
    ax.text(
        0.80,
        0.245,
        "% regional",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color=COLOR_TEXT,
        alpha=0.72,
    )


def vulnerability_color(value):
    colors = [
        "#ead8b8",
        "#f0c997",
        "#f4ba75",
        "#f2a85c",
        "#ee9445",
        "#e77f33",
        "#d96a2b",
        "#c45725",
        "#a94620",
        "#823519",
    ]
    if pd.isna(value):
        return "#ddd6c7"
    boundaries = np.linspace(39, 70, 11)
    idx = np.searchsorted(boundaries, value, side="right") - 1
    idx = int(np.clip(idx, 0, len(colors) - 1))
    return colors[idx]


def build_vertical_chile_svg(metrics, geojson, selected_region=None):
    metric_lookup = metrics.set_index("region")["pct_vulnerabilidad"].to_dict()
    paths = []
    all_x = []
    all_y = []

    for feature in geojson["features"]:
        region_code = int(feature["properties"]["codregion"])
        region_name = REGION_NAMES.get(region_code, str(region_code))
        value = metric_lookup.get(region_code, np.nan)
        fill = vulnerability_color(value)
        selected = selected_region is not None and region_code == selected_region

        for coords in geometry_to_mainland_polygons(feature["geometry"]):
            coords = np.asarray(coords, dtype=float)
            x = coords[:, 0]
            y = -coords[:, 1]
            all_x.extend(x.tolist())
            all_y.extend(y.tolist())
            commands = [f"M {x[0]:.3f} {y[0]:.3f}"]
            commands.extend(f"L {px:.3f} {py:.3f}" for px, py in zip(x[1:], y[1:]))
            commands.append("Z")
            tooltip = f"{region_name} · Vulnerabilidad subsidiada: {value:.1f}%"
            paths.append(
                f'''
                <path
                    class="region-path{' selected-region' if selected else ''}"
                    data-region="{escape(region_name)}"
                    d="{' '.join(commands)}"
                    fill="{fill}"
                    stroke="{'#4c2e05' if selected else '#fff8ea'}"
                    stroke-width="{'0.145' if selected else '0.060'}"
                    vector-effect="non-scaling-stroke"
                >
                    <title>{escape(tooltip)}</title>
                </path>
                '''
            )

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    pad_x = (x_max - x_min) * 0.18
    pad_y = (y_max - y_min) * 0.035
    view_box = f"{x_min - pad_x:.2f} {y_min - pad_y:.2f} {(x_max - x_min) + 2 * pad_x:.2f} {(y_max - y_min) + 2 * pad_y:.2f}"

    return f"""
    <style>
        html,
        body {{
            margin: 0;
            padding: 0;
            background: {COLOR_BG};
            overflow: hidden;
        }}
        .svg-map-shell {{
            width: 100%;
            height: 760px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .chile-hero-map {{
            display: block;
            width: 100%;
            height: 100%;
            overflow: visible;
            filter: drop-shadow(0 8px 18px rgba(76, 46, 5, 0.10));
        }}
    </style>
    <div class="svg-map-shell" id="hero-map">
        <svg
            class="chile-hero-map"
            viewBox="{view_box}"
            preserveAspectRatio="xMidYMid meet"
            role="img"
            aria-label="Mapa interactivo de Chile por región"
        >
            <style>
                .region-path {{
                    cursor: pointer;
                    transition: opacity 120ms ease, stroke-width 120ms ease, filter 120ms ease;
                }}
                .region-path:hover {{
                    opacity: 0.95;
                    stroke: #4c2e05;
                    stroke-width: 0.18;
                    filter: drop-shadow(0 0 0.08rem rgba(76, 46, 5, 0.45));
                }}
                .selected-region {{
                    filter: drop-shadow(0 0 0.11rem rgba(76, 46, 5, 0.55));
                }}
            </style>
            {''.join(paths)}
        </svg>
    </div>
    <script>
        document.querySelectorAll(".region-path").forEach((region) => {{
            region.addEventListener("click", () => {{
                const selected = encodeURIComponent(region.dataset.region);
                const parentLocation = window.parent.location;
                window.parent.location.href = parentLocation.pathname + "?territory=" + selected;
            }});
        }});
    </script>
    """


def get_region_panel_stats(metrics, selected_region, national_pct):
    if selected_region is None:
        return None

    ordered = metrics.sort_values("pct_vulnerabilidad", ascending=False).reset_index(drop=True)
    row = ordered[ordered["region"].eq(selected_region)]
    if row.empty:
        return None

    row = row.iloc[0]
    rank = int(row.name) + 1
    value = float(row["pct_vulnerabilidad"])
    delta = value - national_pct if not pd.isna(national_pct) else np.nan
    return {
        "region": region_display_name(row["region"]),
        "value": value,
        "rank": rank,
        "delta": delta,
    }


def build_vertical_chile_hero_html(metrics, geojson, selected_region=None):
    metric_lookup = metrics.set_index("region")["pct_vulnerabilidad"].to_dict()
    regional_avg = float(metrics["pct_vulnerabilidad"].mean())

    ordered = metrics.sort_values("pct_vulnerabilidad", ascending=False).reset_index(drop=True)
    selected_stats = None
    if selected_region is not None:
        row = ordered[ordered["region"].eq(selected_region)]
        if not row.empty:
            row = row.iloc[0]
            selected_stats = {
                "region": row["region_name"],
                "value": float(row["pct_vulnerabilidad"]),
                "rank": int(row.name) + 1,
                "delta": float(row["pct_vulnerabilidad"]) - regional_avg,
            }

    paths = []
    all_x = []
    all_y = []
    for feature in geojson["features"]:
        region_code = int(feature["properties"]["codregion"])
        region_name = REGION_NAMES.get(region_code, str(region_code))
        value = metric_lookup.get(region_code, np.nan)
        selected = selected_region is not None and region_code == selected_region
        for coords in geometry_to_mainland_polygons(feature["geometry"]):
            coords = np.asarray(coords, dtype=float)
            x = coords[:, 0]
            y = -coords[:, 1]
            all_x.extend(x.tolist())
            all_y.extend(y.tolist())
            commands = [f"M {x[0]:.3f} {y[0]:.3f}"]
            commands.extend(f"L {px:.3f} {py:.3f}" for px, py in zip(x[1:], y[1:]))
            commands.append("Z")
            tooltip = f"{region_name} · Vulnerabilidad subsidiada: {value:.1f}%"
            paths.append(
                f"""
                <path
                    class="region-path{' selected-region' if selected else ''}"
                    data-region="{escape(region_name)}"
                    d="{' '.join(commands)}"
                    fill="{vulnerability_color(value)}"
                    stroke="{'#4c2e05' if selected else '#fff8ea'}"
                    stroke-width="{'0.145' if selected else '0.060'}"
                    vector-effect="non-scaling-stroke"
                >
                    <title>{escape(tooltip)}</title>
                </path>
                """
            )

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    pad_x = (x_max - x_min) * 0.20
    pad_y = (y_max - y_min) * 0.035
    view_box = (
        f"{x_min - pad_x:.2f} {y_min - pad_y:.2f} "
        f"{(x_max - x_min) + 2 * pad_x:.2f} {(y_max - y_min) + 2 * pad_y:.2f}"
    )

    if selected_stats is None:
        panel_html = f"""
        <div class="side-panel">
            <div class="panel-kicker">Mapa interactivo</div>
            <div class="panel-title">Selecciona una región</div>
            <p>El color muestra la tasa regional de vulnerabilidad estructural dentro de viviendas subsidiadas.</p>
            <p>Haz click en una región para abrir los gráficos filtrados por ese territorio.</p>
            <div class="panel-legend"></div>
            <div class="legend-row"><span>Menor</span><span>Mayor vulnerabilidad</span></div>
            <div class="panel-divider"></div>
            <p class="panel-small">Promedio regional simple: <b>{regional_avg:.1f}%</b>.</p>
        </div>
        """
    else:
        delta = selected_stats["delta"]
        if delta > 0:
            delta_text = f"{delta:.1f} pp sobre el promedio regional"
        elif delta < 0:
            delta_text = f"{abs(delta):.1f} pp bajo el promedio regional"
        else:
            delta_text = "igual al promedio regional"
        panel_html = f"""
        <div class="side-panel">
            <div class="panel-kicker">Región seleccionada</div>
            <div class="panel-title">{escape(selected_stats["region"])}</div>
            <p>Vulnerabilidad estructural estimada en viviendas subsidiadas.</p>
            <div class="panel-value">{selected_stats["value"]:.1f}%</div>
            <div class="panel-divider"></div>
            <p class="panel-small">Ranking nacional: <b>{selected_stats["rank"]} de 16</b>, de mayor a menor vulnerabilidad.</p>
            <p class="panel-small">Comparación: <b>{delta_text}</b>.</p>
            <div class="panel-legend"></div>
            <div class="legend-row"><span>Menor</span><span>Mayor vulnerabilidad</span></div>
        </div>
        """

    return f"""
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            background: {COLOR_BG};
            overflow: hidden;
            font-family: Arial, sans-serif;
        }}
        .hero-grid {{
            height: 790px;
            display: grid;
            grid-template-columns: minmax(0, 1.08fr) minmax(300px, 0.92fr);
            gap: 28px;
            align-items: center;
            box-sizing: border-box;
            padding: 4px 8px;
        }}
        .map-wrap {{
            height: 790px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        svg {{
            width: 100%;
            height: 100%;
            overflow: visible;
            filter: drop-shadow(0 8px 18px rgba(76, 46, 5, 0.10));
        }}
        .region-path {{
            cursor: pointer;
            transition: opacity 120ms ease, stroke-width 120ms ease, filter 120ms ease;
        }}
        .region-path:hover {{
            opacity: 0.95;
            stroke: #4c2e05;
            stroke-width: 0.18;
            filter: drop-shadow(0 0 0.08rem rgba(76, 46, 5, 0.45));
        }}
        .selected-region {{
            filter: drop-shadow(0 0 0.11rem rgba(76, 46, 5, 0.55));
        }}
        .side-panel {{
            background: #fffaf0;
            border: 1px solid rgba(211, 139, 93, 0.36);
            border-radius: 8px;
            padding: 24px 25px;
            color: {COLOR_TEXT};
            box-sizing: border-box;
            box-shadow: 0 12px 28px rgba(76, 46, 5, 0.07);
        }}
        .panel-kicker {{
            color: rgba(76, 46, 5, 0.66);
            font: 900 12px/1.15 Arial, sans-serif;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}
        .panel-title {{
            font: 900 28px/1.05 Arial, sans-serif;
            margin-bottom: 12px;
        }}
        .side-panel p {{
            color: rgba(76, 46, 5, 0.76);
            font: 400 15px/1.42 Arial, sans-serif;
            margin: 0 0 12px;
        }}
        .panel-value {{
            color: {COLOR_RM};
            font: 900 48px/1 Arial, sans-serif;
            margin: 12px 0 4px;
        }}
        .panel-divider {{
            border-top: 1px solid rgba(211, 139, 93, 0.25);
            margin: 16px 0 14px;
        }}
        .panel-small {{
            font-size: 14px !important;
        }}
        .panel-legend {{
            height: 10px;
            border-radius: 999px;
            background: linear-gradient(90deg, #ead8b8, #f2a85c, #d96a2b, #823519);
            margin: 16px 0 6px;
            border: 1px solid rgba(76, 46, 5, 0.08);
        }}
        .legend-row {{
            display: flex;
            justify-content: space-between;
            color: rgba(76, 46, 5, 0.62);
            font: 800 12px/1 Arial, sans-serif;
        }}
        .panel-button {{
            width: 100%;
            margin-top: 20px;
            border: 0;
            border-radius: 7px;
            background: {COLOR_RM};
            color: #fffaf0;
            cursor: pointer;
            font: 900 15px/1 Arial, sans-serif;
            padding: 13px 14px;
        }}
        .panel-button:hover {{
            filter: brightness(0.96);
        }}
    </style>
    <div class="hero-grid">
        <div class="map-wrap">
            <svg viewBox="{view_box}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Mapa interactivo de Chile por región">
                {''.join(paths)}
            </svg>
        </div>
        {panel_html}
    </div>
    <script>
        function goToRegion(region) {{
            const selected = encodeURIComponent(region);
            const parentLocation = window.parent.location;
            window.parent.location.href = parentLocation.pathname + "?territory=" + selected;
        }}
        document.querySelectorAll(".region-path").forEach((region) => {{
            region.addEventListener("click", () => goToRegion(region.dataset.region));
        }});
        document.querySelectorAll(".panel-button").forEach((button) => {{
            button.addEventListener("click", () => goToRegion(button.dataset.region));
        }});
    </script>
    """


def build_region_click_points(metrics, geojson):
    metric_lookup = metrics.set_index("region")["pct_vulnerabilidad"].to_dict()
    rows = []

    for feature in geojson["features"]:
        region_code = int(feature["properties"]["codregion"])
        if region_code not in metric_lookup:
            continue

        for coords in geometry_to_mainland_polygons(feature["geometry"]):
            if len(coords) < 3:
                continue

            centroid = coords.mean(axis=0)
            sample_count = min(10, max(4, len(coords) // 12))
            vertex_idx = np.linspace(0, len(coords) - 1, sample_count, dtype=int)
            sampled_vertices = coords[vertex_idx]
            click_points = [centroid]

            for factor in (0.38, 0.68):
                click_points.extend(centroid + (sampled_vertices - centroid) * factor)

            if polygon_area(coords) > 0.12:
                click_points.extend(sampled_vertices)

            for lon, lat in click_points:
                rows.append(
                    {
                        "region": region_code,
                        "region_name": region_display_name(region_code),
                        "pct_vulnerabilidad": metric_lookup[region_code],
                        "lon": float(lon),
                        "lat": float(lat),
                    }
                )

    return pd.DataFrame(rows)


def build_plotly_chile_map(metrics, geojson, selected_region=None):
    plot_data = metrics.sort_values("region").copy()
    display_names = plot_data["region"].map(region_display_name)
    selectedpoints = None
    if selected_region is not None:
        selected_idx = plot_data.index[plot_data["region"].eq(selected_region)].tolist()
        if selected_idx:
            selectedpoints = [plot_data.index.get_loc(selected_idx[0])]

    colorscale = [
        [0.00, "#ead8b8"],
        [0.12, "#f0c997"],
        [0.24, "#f4ba75"],
        [0.36, "#f2a85c"],
        [0.48, "#ee9445"],
        [0.60, "#e77f33"],
        [0.72, "#d96a2b"],
        [0.84, "#c45725"],
        [0.94, "#a94620"],
        [1.00, "#823519"],
    ]

    fig = go.Figure(
        go.Choropleth(
            geojson=geojson,
            featureidkey="properties.codregion",
            locations=plot_data["region"],
            z=plot_data["pct_vulnerabilidad"],
            zmin=39,
            zmax=70,
            text=display_names,
            customdata=np.stack(
                [
                    display_names.to_numpy(),
                    plot_data["region"].to_numpy(),
                    plot_data["pct_vulnerabilidad"].round(1).to_numpy(),
                ],
                axis=-1,
            ),
            colorscale=colorscale,
            showscale=False,
            marker_line_color="#fff8ea",
            marker_line_width=0.9,
            selectedpoints=selectedpoints,
            selected={"marker": {"opacity": 1.0}},
            unselected={"marker": {"opacity": 0.72 if selected_region is not None else 1.0}},
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Vulnerabilidad subsidiada: %{customdata[2]:.1f}%"
                "<extra></extra>"
            ),
        )
    )

    click_points = build_region_click_points(plot_data, geojson)
    selected_click_points = None
    if selected_region is not None and not click_points.empty:
        selected_click_points = click_points.index[click_points["region"].eq(selected_region)].tolist()

    if not click_points.empty:
        fig.add_trace(
            go.Scattergeo(
                lon=click_points["lon"],
                lat=click_points["lat"],
                mode="markers",
                customdata=np.stack(
                    [
                        click_points["region_name"].to_numpy(),
                        click_points["region"].to_numpy(),
                        click_points["pct_vulnerabilidad"].round(1).to_numpy(),
                    ],
                    axis=-1,
                ),
                marker={
                    "size": 24,
                    "color": "rgba(76, 46, 5, 0)",
                    "line": {"width": 0},
                },
                selectedpoints=selected_click_points,
                selected={"marker": {"color": "rgba(76, 46, 5, 0)", "opacity": 0, "size": 26}},
                unselected={"marker": {"opacity": 0}},
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Vulnerabilidad subsidiada: %{customdata[2]:.1f}%"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        projection_type="mercator",
        bgcolor=COLOR_BG,
        showland=False,
        showcountries=False,
        showcoastlines=False,
        showframe=False,
    )
    fig.update_layout(
        height=880,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        dragmode=False,
        clickmode="event+select",
        hoverlabel={
            "bgcolor": "#fffaf0",
            "bordercolor": COLOR_ALMOND,
            "font": {"color": COLOR_TEXT, "size": 14},
        },
    )
    return fig


def extract_plotly_selected_region(event):
    if not event:
        return None

    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    if not selection:
        return None

    points = getattr(selection, "points", None)
    if points is None and isinstance(selection, dict):
        points = selection.get("points")
    if not points:
        return None

    point = points[0]
    if not isinstance(point, dict):
        return None

    customdata = point.get("customdata")
    if isinstance(customdata, (list, tuple, np.ndarray)) and len(customdata) >= 1:
        return str(customdata[0])

    location = point.get("location")
    if location is not None:
        try:
            return region_display_name(int(location))
        except (TypeError, ValueError):
            return str(location)

    text = point.get("text")
    if text:
        return str(text)
    return None


def hero_panel_html(stats, national_pct):
    if stats is None:
        return f"""
        <div class="hero-panel">
            <div class="hero-panel-kicker">Mapa interactivo</div>
            <div class="hero-panel-title">Selecciona una región</div>
            <div class="hero-panel-copy">
                El color muestra la tasa regional de vulnerabilidad estructural dentro de las viviendas subsidiadas.
                Haz click en una región para abrir los gráficos filtrados por ese territorio.
            </div>
            <div class="map-legend"></div>
            <div class="legend-scale">
                <span>Menor</span>
                <span>Mayor vulnerabilidad</span>
            </div>
            <div class="hero-panel-row">
                Promedio nacional en viviendas subsidiadas: <b>{fmt_pct(national_pct)}</b>.
            </div>
        </div>
        """

    delta = stats["delta"]
    delta_text = "sin diferencia relevante"
    if not pd.isna(delta):
        if delta > 0:
            delta_text = f"{delta:.1f} pp sobre el promedio nacional"
        elif delta < 0:
            delta_text = f"{abs(delta):.1f} pp bajo el promedio nacional"

    return f"""
    <div class="hero-panel">
        <div class="hero-panel-kicker">Región seleccionada</div>
        <div class="hero-panel-title">{escape(stats["region"])}</div>
        <div class="hero-panel-copy">
            Vulnerabilidad estructural estimada en viviendas subsidiadas.
        </div>
        <div class="hero-panel-value">{stats["value"]:.1f}%</div>
        <div class="hero-panel-row">
            Ranking nacional: <b>{stats["rank"]} de 16</b> regiones, ordenado de mayor a menor vulnerabilidad.
        </div>
        <div class="hero-panel-row">
            Comparación territorial: <b>{delta_text}</b>.
        </div>
        <div class="map-legend"></div>
        <div class="legend-scale">
            <span>Menor</span>
            <span>Mayor vulnerabilidad</span>
        </div>
    </div>
    """


def scroll_to_charts_once():
    if not st.session_state.get("_scroll_to_charts"):
        return
    st.session_state["_scroll_to_charts"] = False
    components.html(
        """
        <script>
            const doc = window.parent.document;
            const target = doc.getElementById("charts-anchor");
            if (target) {
                target.scrollIntoView({ behavior: "smooth", block: "start" });
            }
        </script>
        """,
        height=0,
    )


def plot_regional_ranking(metrics, selected_region=None):
    ordered = metrics.sort_values("pct_vulnerabilidad", ascending=True).copy()
    colors = [
        COLOR_RM if selected_region is not None and region == selected_region else COLOR_RESTO
        for region in ordered["region"]
    ]

    fig, ax = plt.subplots(figsize=(8.2, 5.4), facecolor=COLOR_BG)
    ax.barh(ordered["region_name"], ordered["pct_vulnerabilidad"], color=colors, height=0.66)
    ax.set_title(
        "Ranking regional de vulnerabilidad estructural en viviendas subsidiadas",
        loc="left",
        fontsize=9.5,
        fontweight="black",
        color=COLOR_TEXT,
        pad=10,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xlim(0, max(75, ordered["pct_vulnerabilidad"].max() + 5))
    ax.xaxis.grid(True, color=COLOR_ALMOND, alpha=0.25, linestyle="--")
    ax.yaxis.grid(False)
    ax.tick_params(axis="both", labelsize=9, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, value in enumerate(ordered["pct_vulnerabilidad"]):
        ax.text(value + 1, i, f"{value:.1f}%", va="center", ha="left", fontsize=8.5, fontweight="bold")
    return fig


def ds49_long_totals(ds49_totals):
    years = [col for col in ds49_totals.columns if isinstance(col, int) and col <= 2025]
    valid = ds49_totals[
        ~ds49_totals["region"].str.startswith("Total Pa")
        & ~ds49_totals["region"].str.startswith("Sin Informaci")
    ]
    return valid.melt(id_vars="region", value_vars=years, var_name="year", value_name="beneficiados")


def plot_ds49_selected_region(ds49_totals, selection, figsize=DASH_LINE_FIGSIZE):
    data = ds49_long_totals(ds49_totals)

    if selection == "Chile completo":
        selected = data.groupby("year", as_index=False)["beneficiados"].sum()
        label = "Chile completo"
    elif selection == "Santiago (RM)":
        selected = data[data["region"].eq("Metropolitana")].copy()
        label = "Santiago (RM)"
    elif selection == "Regiones":
        selected = data[~data["region"].eq("Metropolitana")].groupby("year", as_index=False)["beneficiados"].sum()
        label = "Regiones"
    else:
        selected = data[data["region"].map(normalize_text).eq(normalize_text(selection))].copy()
        label = selection

    fig, ax = plt.subplots(figsize=figsize, facecolor=COLOR_BG)
    sns.lineplot(
        data=selected,
        x="year",
        y="beneficiados",
        marker="o",
        markersize=5.5,
        linewidth=2.5,
        color=COLOR_RM,
        ax=ax,
    )
    normalize_dashboard_title(ax, f"2. Beneficiados DS49 en {label}", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.yaxis.grid(True, linestyle="--", alpha=0.25, color=COLOR_ALMOND)
    ax.xaxis.grid(False)
    ax.tick_params(axis="both", labelsize=DASH_TICK_FS, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(COLOR_ALMOND)
    ax.spines["bottom"].set_alpha(0.45)
    return fig


def plot_historical_line_streamlit(ax, df_hist, df_hist_regions, selected_region):
    def draw_default():
        plot_historical_line(ax, df_hist)
        normalize_dashboard_title(ax, "3. Evolución de balaceras en viviendas subsidiadas (%)", pad=18)
        ax.tick_params(axis="both", labelsize=DASH_TICK_FS, length=0)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(DASH_LABEL_FS)

    if selected_region is None or selected_region == 13:
        draw_default()
        return

    selected_name = region_display_name(selected_region)
    compare = df_hist_regions[df_hist_regions["region"].isin([13, selected_region])].copy()
    if compare.empty:
        draw_default()
        return

    compare["zona"] = np.where(compare["region"].eq(13), "Santiago (RM)", selected_name)
    compare["weighted_balacera"] = compare["is_balacera"] * compare["ponderador"]
    plot_data = compare.groupby(["year", "zona"], as_index=False).agg(
        weighted_balacera=("weighted_balacera", "sum"),
        ponderador=("ponderador", "sum"),
    )
    plot_data["pct"] = np.where(
        plot_data["ponderador"] > 0,
        plot_data["weighted_balacera"] / plot_data["ponderador"] * 100,
        np.nan,
    )
    plot_data = plot_data.pivot(index="year", columns="zona", values="pct")

    required = ["Santiago (RM)", selected_name]
    if any(col not in plot_data.columns for col in required):
        draw_default()
        return

    years = plot_data.index.to_numpy()
    rm_values = plot_data["Santiago (RM)"]
    region_values = plot_data[selected_name]

    ax.fill_between(years, region_values, rm_values, color=COLOR_RM, alpha=0.07, zorder=1)
    for year in years:
        val_rm = plot_data.loc[year, "Santiago (RM)"]
        val_region = plot_data.loc[year, selected_name]
        ax.plot([year, year], [val_region, val_rm], color=COLOR_RM, linestyle="--", linewidth=1, alpha=0.34, zorder=2)
        ax.text(year, val_rm + 1.2, f"{int(round(val_rm))}%", color=COLOR_RM,
                fontweight="bold", ha="center", fontsize=DASH_LABEL_FS)
        region_va = "top" if val_region <= val_rm else "bottom"
        region_offset = -1.2 if region_va == "top" else 1.2
        ax.text(year, val_region + region_offset, f"{int(round(val_region))}%", color=COLOR_RESTO,
                fontweight="bold", ha="center", fontsize=DASH_LABEL_FS, va=region_va)

    ax.plot(years, rm_values, marker="o", color=COLOR_RM, linewidth=2.8,
            markersize=7, label="Santiago (RM)", zorder=4)
    ax.plot(years, region_values, marker="o", color=COLOR_RESTO, linewidth=2.8,
            markersize=7, label=selected_name, zorder=3)

    max_value = float(plot_data[required].to_numpy().max())
    y_max = max(60, int(np.ceil((max_value + 6) / 10) * 10))
    ax.set_xticks(years)
    ax.set_xticklabels([str(int(year)) for year in years], fontsize=DASH_TICK_FS)
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max + 1, 20))
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, color=COLOR_ALMOND)
    ax.xaxis.grid(False)
    ax.tick_params(axis="y", labelsize=DASH_TICK_FS, length=0)
    ax.tick_params(axis="x", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(COLOR_ALMOND)
    ax.spines["bottom"].set_alpha(0.45)
    ax.set_xlabel("")
    ax.set_ylabel("")
    normalize_dashboard_title(
        ax,
        f"3. Evolución de balaceras:\nSantiago vs {selected_name} (%)",
        pad=18,
    )
    ax.legend(loc="upper left", frameon=False, facecolor="none", edgecolor="none", fontsize=DASH_LABEL_FS)


def plot_radar_streamlit(ax, df, selected_region):
    def draw_default():
        plot_radar_rm_vs_regions(ax, df)
        normalize_dashboard_title(ax, "4. Vulnerabilidades subsidiadas:\nSantiago vs Regiones (%)", loc="center", pad=4)
        for label in ax.get_xticklabels():
            label.set_fontsize(DASH_LABEL_FS - 1)
            label.set_fontweight("bold")
        for label in ax.get_yticklabels():
            label.set_fontsize(DASH_TICK_FS - 1)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(DASH_TICK_FS - 1)

    if selected_region is None or selected_region == 13:
        draw_default()
        return

    selected_name = region_display_name(selected_region)
    df_sub = df[df["grupo_vivienda"].eq("Subsidiada")].copy()
    df_sub["dim_multidim"] = np.where(df_sub["pobreza_multi"].eq(1), 1, 0)
    df_sub["dim_hacinamiento"] = np.where(df_sub["ind_hacina"] >= 2, 1, 0)
    df_sub["dim_balaceras"] = np.where(df_sub["v36e"] >= 3, 1, 0)
    df_sub["dim_narco"] = np.where(df_sub["v36c"] >= 3, 1, 0)
    df_sub["dim_alumbrado"] = np.where(df_sub["v35a"].eq(2), 1, 0)
    df_sub["dim_basura"] = np.where(df_sub["v35c"].eq(2), 1, 0)

    dimensions = ["dim_multidim", "dim_hacinamiento", "dim_balaceras", "dim_narco", "dim_alumbrado", "dim_basura"]
    labels = ["Pobreza\nmulti", "Hacinam.", "Balaceras", "Narco", "Sin\nluz", "Basura"]

    def wpct(data, dim):
        weight = data["expr"].sum()
        if weight <= 0:
            return 0.0
        return data.loc[data[dim].eq(1), "expr"].sum() / weight * 100

    df_rm = df_sub[df_sub["region"].eq(13)]
    df_region = df_sub[df_sub["region"].eq(selected_region)]
    if df_region.empty:
        draw_default()
        return

    vals_rm = [wpct(df_rm, dim) for dim in dimensions]
    vals_region = [wpct(df_region, dim) for dim in dimensions]
    max_value = max(vals_rm + vals_region)
    radial_max = max(40, int(np.ceil((max_value + 4) / 10) * 10))
    radial_ticks = [tick for tick in [10, 20, 30, 40, 50, 60] if tick < radial_max]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    vals_rm += vals_rm[:1]
    vals_region += vals_region[:1]

    ax.set_facecolor(COLOR_BG)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(angles, vals_region, color=COLOR_RESTO, linewidth=1.8, marker="o",
            markersize=3.5, label=selected_name, zorder=3)
    ax.fill(angles, vals_region, color=COLOR_RESTO, alpha=0.16, zorder=2)
    ax.plot(angles, vals_rm, color=COLOR_RM, linewidth=2.0, marker="o",
            markersize=3.5, label="Santiago (RM)", zorder=4)
    ax.fill(angles, vals_rm, color=COLOR_RM, alpha=0.18, zorder=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=DASH_LABEL_FS - 1, fontweight="bold", color=COLOR_TEXT)
    ax.tick_params(axis="x", pad=3)
    ax.set_ylim(0, radial_max)
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([f"{tick}%" for tick in radial_ticks], fontsize=DASH_TICK_FS - 1, color=COLOR_ALMOND)
    ax.set_rlabel_position(78)
    ax.grid(color=COLOR_ALMOND, linestyle="--", linewidth=0.6, alpha=0.36)
    ax.spines["polar"].set_visible(False)
    normalize_dashboard_title(
        ax,
        f"4. Vulnerabilidades subsidiadas:\nSantiago vs {selected_name} (%)",
        loc="center",
        pad=4,
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.19), ncol=2,
              frameon=False, fontsize=DASH_TICK_FS - 1, handlelength=1.4, columnspacing=0.8)


def plot_subsidy_butterfly_streamlit(ax, df):
    df = df.copy()
    df["Balaceras Frecuentes"] = np.where(df["v36b"] >= 3, 1, 0)
    df["Narco-trafico Frecuente"] = np.where(df["v36c"] >= 3, 1, 0)
    df["Alta Dependencia (>20%)"] = np.where(df["subsidy_ratio"] > 0.20, 1, 0)
    df["Ingreso Autonomo <$600k"] = np.where(df["yautcorh"] < 600000, 1, 0)
    df["Pobreza por Ingresos"] = np.where(df["pobreza"].isin([1, 2]), 1, 0)

    vars_to_plot = [
        "Balaceras Frecuentes",
        "Narco-trafico Frecuente",
        "Pobreza por Ingresos",
        "Alta Dependencia (>20%)",
        "Ingreso Autonomo <$600k",
    ]
    label_map = {
        "Balaceras Frecuentes": "BALACERAS",
        "Narco-trafico Frecuente": "NARCO FRECUENTE",
        "Pobreza por Ingresos": "POBREZA INGRESOS",
        "Alta Dependencia (>20%)": "DEPENDENCIA >20%",
        "Ingreso Autonomo <$600k": "INGRESO <$600K",
    }

    df_sub = df[df["is_subsidized"].eq(1)]
    df_nosub = df[df["is_subsidized"].eq(0)]

    def wpct(subset, var):
        weight = subset["expr"].sum()
        if weight <= 0:
            return 0.0
        return subset.loc[subset[var].eq(1), "expr"].sum() / weight * 100

    plot_df = pd.DataFrame(
        [
            {
                "Variable": var,
                "No_Subsidiada": wpct(df_nosub, var),
                "Subsidiada": wpct(df_sub, var),
            }
            for var in vars_to_plot
        ]
    )

    max_pct = float(plot_df[["No_Subsidiada", "Subsidiada"]].to_numpy().max())
    track_max = max(42, int(np.ceil(max_pct / 5) * 5))
    label_pad = max(3.6, track_max * 0.07)
    xlim = track_max + max(12, track_max * 0.24)

    row_gap = 1.06
    bar_h = 0.16
    blue_lit = COLOR_RESTO
    blue_dim = "#7485ad"
    orange_lit = COLOR_RM
    orange_dim = "#f4b06f"
    top_y = (len(vars_to_plot) - 1) * row_gap

    for i, row in plot_df.iterrows():
        y = top_y - i * row_gap
        val_nosub = row["No_Subsidiada"]
        val_sub = row["Subsidiada"]
        nosub_color = blue_lit if val_nosub >= val_sub else blue_dim
        sub_color = orange_lit if val_sub >= val_nosub else orange_dim

        ax.barh(y, -track_max, left=0, color=COLOR_ALMOND, alpha=0.18, height=bar_h, edgecolor="none", zorder=1)
        ax.barh(y, track_max, left=0, color=COLOR_ALMOND, alpha=0.18, height=bar_h, edgecolor="none", zorder=1)
        ax.barh(y, -val_nosub, left=0, color=nosub_color, height=bar_h, edgecolor="none", zorder=3)
        ax.barh(y, val_sub, left=0, color=sub_color, height=bar_h, edgecolor="none", zorder=3)

        ax.text(
            0,
            y + 0.36,
            label_map[row["Variable"]],
            ha="center",
            va="bottom",
            color=COLOR_TEXT,
            fontsize=DASH_LABEL_FS,
            fontweight="black",
            zorder=5,
        )
        ax.text(
            -track_max - label_pad,
            y,
            f"{val_nosub:.0f}%",
            ha="right",
            va="center",
            color=nosub_color,
            fontsize=DASH_LABEL_FS + 1.5,
            fontweight="black",
            zorder=5,
        )
        ax.text(
            track_max + label_pad,
            y,
            f"{val_sub:.0f}%",
            ha="left",
            va="center",
            color=sub_color,
            fontsize=DASH_LABEL_FS + 1.5,
            fontweight="black",
            zorder=5,
        )

    ax.text(-track_max * 0.28, top_y + 0.64, "SIN SUBSIDIO", ha="right", va="bottom",
            color=COLOR_RESTO, fontsize=DASH_LABEL_FS + 1, fontweight="black")
    ax.text(track_max * 0.28, top_y + 0.64, "CON SUBSIDIO", ha="left", va="bottom",
            color=COLOR_RM, fontsize=DASH_LABEL_FS + 1, fontweight="black")
    ax.text(
        -xlim,
        top_y + 1.72,
        "6. Vulnerabilidades:\nCon subsidio vs Sin subsidio (%)",
        ha="left",
        va="top",
        fontsize=DASH_TITLE_FS,
        fontweight="black",
        color=COLOR_TEXT,
        linespacing=1.05,
    )

    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-0.82, top_y + 1.78)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.axis("off")


DATA_PREVIEW_RENAME = {
    "region_nombre": "Región",
    "grupo_vivienda": "Grupo vivienda",
    "expr": "Factor de expansión",
    "yautcorh": "Ingreso autónomo hogar",
    "subsidy_ratio": "Razón subsidio (%)",
    "v36b": "Balaceras (frecuencia)",
    "v36c": "Narcotráfico (frecuencia)",
    "pobreza": "Pobreza por ingresos",
    "pobreza_multi": "Pobreza multidimensional",
    "commute_total_hrs": "Traslado diario (hrs)",
}


def build_filtered_dataset_preview(df_context):
    cols = [
        "region",
        "grupo_vivienda",
        "expr",
        "yautcorh",
        "subsidy_ratio",
        "v36b",
        "v36c",
        "pobreza",
        "pobreza_multi",
        "commute_total_hrs",
    ]
    available = [col for col in cols if col in df_context.columns]
    preview = df_context[available].copy()
    preview["region_nombre"] = preview["region"].map(REGION_NAMES)
    if "subsidy_ratio" in preview.columns:
        preview["subsidy_ratio"] = preview["subsidy_ratio"] * 100
    ordered = ["region_nombre"] + [col for col in available if col != "region"]
    return preview[ordered].rename(columns=DATA_PREVIEW_RENAME)


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

LANDING_FLOW_VERSION = 2
if st.session_state.get("_landing_flow_version") != LANDING_FLOW_VERSION:
    st.session_state["territory"] = "Chile completo"
    st.session_state["has_selected_territory"] = False
    st.session_state["_scroll_to_charts"] = False
    st.session_state["_landing_flow_version"] = LANDING_FLOW_VERSION

if "territory" not in st.session_state:
    st.session_state["territory"] = "Chile completo"
if "has_selected_territory" not in st.session_state:
    st.session_state["has_selected_territory"] = False
if "_scroll_to_charts" not in st.session_state:
    st.session_state["_scroll_to_charts"] = False

query_territory = st.query_params.get("territory")
options = get_region_options()
if query_territory in options:
    if st.session_state["territory"] != query_territory or not st.session_state["has_selected_territory"]:
        st.session_state["territory"] = query_territory
        st.session_state["has_selected_territory"] = True
        st.session_state["_scroll_to_charts"] = True
else:
    components.html(
        """
        <script>
            const loc = window.parent.location;
            if (loc.hash || loc.search) {
                window.parent.history.replaceState(null, "", loc.pathname);
            }
            window.parent.scrollTo({ top: 0, behavior: "auto" });
        </script>
        """,
        height=0,
    )

st.markdown('<div class="main-title">Viviendas Sociales en Chile</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="lead">
    La visualización muestra que el acceso a una vivienda subsidiada no siempre implica superar la vulnerabilidad.
    Aunque los beneficiarios DS49 han aumentado en distintos periodos, los hogares en viviendas sociales siguen
    concentrando desventajas estructurales: bajos ingresos, dependencia del subsidio, problemas de seguridad y
    condiciones territoriales desiguales entre Santiago y regiones.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.spinner("Cargando mapa y datos..."):
    (
        df_master,
        df_hist,
        df_hist_regions,
        ds49_totals,
        df_ds49_yoy,
        geojson_regions,
        regional_metrics,
        moran_i,
        moran_p,
    ) = load_project_data()

selected_region = territory_to_region_code(st.session_state["territory"])
national_pct = calculate_summary_metrics(df_master, ds49_totals, "Chile completo")["vulnerabilidad"]
panel_stats = get_region_panel_stats(
    regional_metrics,
    selected_region if st.session_state["has_selected_territory"] else None,
    national_pct,
)

hero_map_col, hero_panel_col = st.columns([0.74, 0.26], gap="large")
with hero_map_col:
    map_event = st.plotly_chart(
        build_plotly_chile_map(
            regional_metrics,
            geojson_regions,
            selected_region if st.session_state["has_selected_territory"] else None,
        ),
        key="chile_region_map",
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False},
        on_select="rerun",
        selection_mode="points",
    )

clicked_region = extract_plotly_selected_region(map_event)
if clicked_region and clicked_region != st.session_state["territory"]:
    st.session_state["territory"] = clicked_region
    st.session_state["has_selected_territory"] = True
    st.session_state["_scroll_to_charts"] = True
    st.rerun()

with hero_panel_col:
    st.markdown(hero_panel_html(panel_stats, national_pct), unsafe_allow_html=True)
    if st.button("Ver Chile completo", type="primary", width="stretch"):
        st.session_state["territory"] = "Chile completo"
        st.session_state["has_selected_territory"] = True
        st.session_state["_scroll_to_charts"] = True
        st.rerun()

st.markdown(
    '<div class="hero-map-note">Pasa el mouse para ver la región. Haz click sobre una región o usa Chile completo para abrir los gráficos.</div>',
    unsafe_allow_html=True,
)

st.markdown('<div id="charts-anchor"></div>', unsafe_allow_html=True)
scroll_to_charts_once()

if not st.session_state["has_selected_territory"]:
    st.stop()

current_index = options.index(st.session_state["territory"]) if st.session_state["territory"] in options else 0
analysis_left, analysis_right = st.columns([0.68, 0.32], gap="large")
with analysis_left:
    st.markdown(
        f'<div class="analysis-title">Análisis para {st.session_state["territory"]}</div>',
        unsafe_allow_html=True,
    )
with analysis_right:
    territory = st.selectbox("Territorio analizado", options, index=current_index)

if territory != st.session_state["territory"]:
    st.session_state["territory"] = territory
    st.session_state["_scroll_to_charts"] = True
    st.rerun()

selected_region = territory_to_region_code(territory)
df_context = filter_by_territory(df_master, territory)
summary = calculate_summary_metrics(df_context, ds49_totals, territory)

metric_cols = st.columns(5)
with metric_cols[0]:
    metric_card(
        "Viviendas subsidiadas",
        fmt_int(summary["total_subsidiadas"]),
        f"Estimación ponderada · {territory}",
        COLOR_ALMOND,
    )
with metric_cols[1]:
    metric_card(
        "Vulnerabilidad estructural",
        fmt_pct(summary["vulnerabilidad"]),
        "Violencia o presión económica",
        COLOR_RM,
    )
with metric_cols[2]:
    metric_card(
        "Entorno violento",
        fmt_pct(summary["violencia"]),
        "Balaceras o narcotráfico frecuente",
        COLOR_WARM,
    )
with metric_cols[3]:
    metric_card(
        "Presión económica",
        fmt_pct(summary["economia"]),
        "Bajos ingresos o alta dependencia",
        "#a94f24",
    )
with metric_cols[4]:
    metric_card(
        "Beneficiados DS49 2025",
        fmt_int(summary["ds49_2025"]),
        f"MINVU · {territory}",
        COLOR_RESTO,
    )


tab_story, tab_region, tab_data = st.tabs(["Vista narrativa", "Exploración regional", "Datos y metodología"])

with tab_story:
    st.markdown('<div class="section-label">Lectura principal</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="tab-intro">Recorrido visual de la situación de las viviendas subsidiadas en '
        f'<b>{escape(territory)}</b>. Cada gráfico se acompaña de una breve lectura para interpretarlo.</div>',
        unsafe_allow_html=True,
    )

    moran_text = (
        "**Existe un agrupamiento espacial significativo.** "
        "Las regiones con alta vulnerabilidad en viviendas subsidiadas tienden a "
        "estar geográficamente juntas, demostrando empíricamente que la segregación territorial no es un fenómeno aleatorio."
    ) if moran_p < 0.05 else "No se detecta un agrupamiento espacial estadísticamente significativo."

    comparison_name = region_display_name(selected_region) if (selected_region is not None and selected_region != 13) else None

    # --- Gráfico 2: Acceso al subsidio DS49 ---
    chart_col, text_col = st.columns([0.58, 0.42], gap="large", vertical_alignment="center")
    with chart_col:
        with st.container(border=True):
            show_matplotlib(plot_ds49_selected_region(ds49_totals, territory, figsize=DASH_STORY_FIGSIZE))
    with text_col:
        chart_explainer(
            "Gráfico 2 · Cobertura",
            "Acceso al subsidio DS49",
            f"Hogares que accedieron al subsidio habitacional <b>DS49</b> cada año en "
            f"<b>{escape(territory)}</b>. La pendiente revela si la cobertura del programa "
            f"se ha expandido o contraído en el tiempo.",
            takeaway=f"Beneficiados DS49 2025: <b>{fmt_int(summary['ds49_2025'])}</b> (fuente MINVU).",
        )

    # --- Gráfico 3: Evolución de balaceras ---
    chart_col, text_col = st.columns([0.58, 0.42], gap="large", vertical_alignment="center")
    with chart_col:
        with st.container(border=True):
            fig, ax = plt.subplots(figsize=DASH_STORY_FIGSIZE, facecolor=COLOR_BG)
            plot_historical_line_streamlit(ax, df_hist, df_hist_regions, selected_region)
            show_matplotlib(fig)
    with text_col:
        if comparison_name:
            balaceras_body = (
                f"Evolución del porcentaje de hogares subsidiados que reportan <b>balaceras "
                f"frecuentes</b> en su entorno, comparando <b>Santiago (RM)</b> con "
                f"<b>{escape(comparison_name)}</b>."
            )
        else:
            balaceras_body = (
                "Evolución del porcentaje de hogares subsidiados que reportan <b>balaceras "
                "frecuentes</b> en su entorno, desagregado por zona del país."
            )
        chart_explainer(
            "Gráfico 3 · Seguridad",
            "Evolución de balaceras",
            balaceras_body,
            takeaway="Permite ver si la exposición a la violencia se ha agravado o reducido con los años.",
        )

    # --- Gráfico 4: Perfil de vulnerabilidades (radar) ---
    chart_col, text_col = st.columns([0.58, 0.42], gap="large", vertical_alignment="center")
    with chart_col:
        with st.container(border=True):
            fig, ax = plt.subplots(figsize=DASH_STORY_FIGSIZE, subplot_kw={"projection": "polar"}, facecolor=COLOR_BG)
            plot_radar_streamlit(ax, df_master, selected_region)
            show_matplotlib(fig)
    with text_col:
        radar_target = comparison_name if comparison_name else "el resto de las regiones"
        chart_explainer(
            "Gráfico 4 · Perfil",
            "Perfil de vulnerabilidades",
            f"Cada eje resume una dimensión de precariedad —pobreza multidimensional, "
            f"hacinamiento, balaceras, narcotráfico, falta de alumbrado y basura— en viviendas "
            f"subsidiadas. Compara el perfil de <b>Santiago (RM)</b> con <b>{escape(radar_target)}</b>.",
            takeaway="Mientras más se aleja la línea del centro, mayor es la prevalencia de esa dimensión.",
        )

    # --- Gráfico 5: Vulnerabilidad estructural (donut) ---
    chart_col, text_col = st.columns([0.58, 0.42], gap="large", vertical_alignment="center")
    with chart_col:
        with st.container(border=True):
            fig, ax = plt.subplots(figsize=DASH_STORY_FIGSIZE, facecolor=COLOR_BG)
            plot_donut_vulnerability(ax, df_context)
            normalize_dashboard_title(
                ax,
                "5. Vulnerabilidad estructural en\nViviendas Subsidiadas (%)",
                pad=7,
            )
            show_matplotlib(fig)
    with text_col:
        chart_explainer(
            "Gráfico 5 · Síntesis",
            "Vulnerabilidad estructural",
            f"Proporción de viviendas subsidiadas que enfrentan al menos una <b>trampa "
            f"estructural</b> (entorno violento o presión económica) en <b>{escape(territory)}</b>.",
            takeaway=f"Vulnerabilidad estructural en {escape(territory)}: "
            f"<b>{fmt_pct(summary['vulnerabilidad'])}</b> de los hogares subsidiados.",
        )

    # --- Gráfico 6: Con subsidio vs sin subsidio (butterfly) ---
    chart_col, text_col = st.columns([0.58, 0.42], gap="large", vertical_alignment="center")
    with chart_col:
        with st.container(border=True):
            fig, ax = plt.subplots(figsize=DASH_STORY_FIGSIZE, facecolor=COLOR_BG)
            plot_subsidy_butterfly_streamlit(ax, df_context)
            show_matplotlib(fig)
    with text_col:
        chart_explainer(
            "Gráfico 6 · Contraste",
            "Con subsidio vs sin subsidio",
            "Contrasta, variable por variable, la prevalencia de cada factor de vulnerabilidad "
            "entre hogares <b>con</b> y <b>sin</b> subsidio.",
            takeaway="Revela si el acceso al subsidio se asocia a mayor o menor exposición a la precariedad.",
        )

    # --- Lectura espacial (Moran's I) ---
    st.markdown('<div class="section-label">Lectura espacial</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background-color: {COLOR_ALMOND}20; border-left: 5px solid {COLOR_RM}; padding: 18px 20px; border-radius: 6px; border: 1px solid rgba(211, 139, 93, 0.25); box-shadow: 0 4px 12px rgba(76,46,5,0.03);">
        <h4 style="margin-top: 0; margin-bottom: 8px; color: {COLOR_TEXT}; font-size: 17px; font-weight: 900;">Autocorrelación espacial (Moran's I)</h4>
        <p style="margin-bottom: 8px; color: {COLOR_TEXT}; font-size: 14.5px;">Índice global: <b>{moran_i:.3f}</b> &nbsp;|&nbsp; Valor p: <b>{moran_p:.3f}</b></p>
        <p style="margin-bottom: 0; color: rgba(76, 46, 5, 0.85); font-size: 14.5px; line-height: 1.5;">{moran_text}</p>
    </div>
    """, unsafe_allow_html=True)


with tab_region:
    st.markdown('<div class="section-label">Exploración regional</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="tab-intro">Ranking comparativo de las 16 regiones según su tasa de '
        'vulnerabilidad estructural en viviendas subsidiadas. La región seleccionada aparece '
        'destacada en color para situarla frente al resto del país.</div>',
        unsafe_allow_html=True,
    )
    region_left, region_right = st.columns([1.2, 0.8], gap="large", vertical_alignment="top")

    with region_left:
        with st.container(border=True):
            show_matplotlib(plot_regional_ranking(regional_metrics, selected_region))
            st.caption(
                "Cada barra es una región. La barra naranja corresponde al territorio "
                "seleccionado; las demás se muestran como referencia."
            )

    with region_right:
        with st.container(border=True):
            selected_metrics = regional_metrics.copy()
            if selected_region is not None:
                selected_metrics = selected_metrics[selected_metrics["region"].eq(selected_region)]

            st.markdown("#### Tasa regional")
            if selected_region is not None:
                val = selected_metrics['pct_vulnerabilidad'].iloc[0]
                delta_val = val - national_pct
                ordered = regional_metrics.sort_values("pct_vulnerabilidad", ascending=False).reset_index()
                rank = ordered.index[ordered["region"] == selected_region].tolist()[0] + 1

                st.metric(
                    label="Vulnerabilidad en viviendas subsidiadas",
                    value=f"{val:.1f}%",
                    delta=f"{delta_val:+.1f} pp vs Promedio Nacional",
                    delta_color="inverse",
                )
                st.markdown(f'''
                <div style="font-size: 13.5px; color: rgba(76, 46, 5, 0.85); margin-top: -5px; margin-bottom: 22px; padding: 12px; background-color: #fffaf0; border-left: 4px solid {COLOR_RM}; border-radius: 4px;">
                    🎯 <b>Insight:</b> Esta región ocupa el puesto <b>N°{rank} de 16</b> en el ranking nacional de mayor vulnerabilidad estructural.
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<p style="font-size: 13.5px; color: rgba(76, 46, 5, 0.75); margin: 0 0 10px;">'
                    'Selecciona una región para ver su tasa y ranking. Mientras tanto, esta es la '
                    'tabla comparativa completa.</p>',
                    unsafe_allow_html=True,
                )
                st.dataframe(
                    selected_metrics[["region_name", "pct_vulnerabilidad"]].sort_values("pct_vulnerabilidad", ascending=False).rename(
                        columns={
                            "region_name": "Región",
                            "pct_vulnerabilidad": "% vulnerabilidad subsidiadas",
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "% vulnerabilidad subsidiadas": st.column_config.NumberColumn(
                            "% vulnerabilidad", format="%.1f%%"
                        ),
                    },
                )


with tab_data:
    st.markdown('<div class="section-label">Datos y metodología</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="tab-intro">Datos de la encuesta <b>CASEN</b> filtrados para '
        f'<b>{escape(territory)}</b>, junto a las definiciones metodológicas empleadas en '
        f'todo el tablero. Puedes explorar la tabla y descargarla en formato CSV.</div>',
        unsafe_allow_html=True,
    )

    preview = build_filtered_dataset_preview(df_context)
    total_records = len(df_context)
    households = df_context["expr"].sum() if "expr" in df_context.columns else np.nan

    stat_cols = st.columns(3)
    with stat_cols[0]:
        metric_card("Registros en la muestra", fmt_int(total_records), f"Encuestas CASEN · {territory}", COLOR_RESTO)
    with stat_cols[1]:
        metric_card("Hogares representados", fmt_int(households), "Estimación ponderada (factor expansión)", COLOR_RM)
    with stat_cols[2]:
        metric_card("Variables mostradas", str(preview.shape[1]), "Indicadores seleccionados por hogar", COLOR_ALMOND)

    st.markdown('<div class="section-label" style="margin-top: 1.1rem;">Datos filtrados</div>', unsafe_allow_html=True)
    st.dataframe(
        preview.head(600),
        hide_index=True,
        width="stretch",
        column_config={
            "Factor de expansión": st.column_config.NumberColumn(format="%d"),
            "Ingreso autónomo hogar": st.column_config.NumberColumn(format="$ %d"),
            "Razón subsidio (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Traslado diario (hrs)": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    st.caption(
        f"Mostrando hasta 600 de {fmt_int(total_records)} registros. Descarga el CSV para el detalle completo."
    )

    csv_data = preview.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar datos filtrados (CSV)",
        data=csv_data,
        file_name=f"datos_filtrados_{normalize_text(territory).replace(' ', '_')}.csv",
        mime="text/csv",
    )

    st.markdown('<div class="section-label" style="margin-top: 1.3rem;">Metodología</div>', unsafe_allow_html=True)
    with st.expander("Definiciones y fuentes utilizadas en el tablero", expanded=False):
        st.markdown(
            """
            **Definiciones**

            - **Vivienda subsidiada:** hogares clasificados como `grupo_vivienda == "Subsidiada"`.
            - **Entorno violento:** presencia frecuente de balaceras o narcotráfico según variables CASEN.
            - **Presión económica:** ingreso autónomo del hogar menor a $600.000 o razón de subsidio superior a 20%.
            - **Vulnerabilidad estructural:** hogares subsidiados que presentan entorno violento o presión económica.
            - **Moran's I:** autocorrelación espacial regional calculada con contigüidad Queen sobre el mapa regional.

            **Fuentes**

            - **CASEN** (Encuesta de Caracterización Socioeconómica Nacional) — Ministerio de Desarrollo Social.
            - **DS49** — Registro de beneficiados del subsidio habitacional, MINVU.

            **Nota sobre la ponderación**

            Todos los porcentajes se calculan ponderando por el factor de expansión (`expr`),
            de modo que las cifras representan hogares a nivel poblacional y no el conteo simple de encuestas.
            """
        )
