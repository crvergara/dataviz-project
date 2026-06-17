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
    plot_donut_vulnerability,
    plot_ds49_yoy_growth,
    plot_historical_line,
    plot_radar_rm_vs_regions,
    plot_subsidy_butterfly,
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


@st.cache_data(show_spinner=False)
def load_project_data():
    df_master = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "master_dataset.parquet")
    df_hist = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "balaceras_historico_zonas.parquet")
    ds49_totals = read_ds49_region_totals(PROJECT_ROOT / "data" / "raw" / "SUDS49Mar2026.xlsx")
    df_ds49_yoy = prepare_ds49_yoy(ds49_totals)

    with open(PROJECT_ROOT / "data" / "raw" / "regiones.json", encoding="utf-8-sig") as file:
        geojson_regions = json.load(file)

    regional_metrics = build_regional_vulnerability_metrics(df_master)
    moran_i, moran_p = calculate_spatial_autocorrelation(regional_metrics, geojson_regions)
    return df_master, df_hist, ds49_totals, df_ds49_yoy, geojson_regions, regional_metrics, moran_i, moran_p


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


def build_horizontal_chile_svg(metrics, geojson, selected_region=None):
    metric_lookup = metrics.set_index("region")["pct_vulnerabilidad"].to_dict()
    paths = []
    all_x = []
    all_y = []
    y_scale = 1.04
    y_center = -71.35

    for feature in geojson["features"]:
        region_code = int(feature["properties"]["codregion"])
        region_name = REGION_NAMES.get(region_code, str(region_code))
        value = metric_lookup.get(region_code, np.nan)
        fill = vulnerability_color(value)
        selected = selected_region is not None and region_code == selected_region

        for coords in geometry_to_mainland_polygons(feature["geometry"]):
            coords = np.asarray(coords, dtype=float)
            x = -coords[:, 1]
            y = ((coords[:, 0] - y_center) * y_scale) + y_center
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
    pad_x = (x_max - x_min) * 0.035
    pad_y = (y_max - y_min) * 0.20
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
            height: 710px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .chile-hero-map {{
            display: block;
            width: 100%;
            height: 100%;
            overflow: visible;
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
        fontsize=13,
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


def plot_ds49_selected_region(ds49_totals, selection):
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

    fig, ax = plt.subplots(figsize=(7.8, 3.3), facecolor=COLOR_BG)
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
    ax.set_title(
        f"Beneficiados DS49 en {label}",
        loc="left",
        fontsize=12.2,
        fontweight="black",
        color=COLOR_TEXT,
        pad=10,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.yaxis.grid(True, linestyle="--", alpha=0.25, color=COLOR_ALMOND)
    ax.xaxis.grid(False)
    ax.tick_params(axis="both", labelsize=8.5, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(COLOR_ALMOND)
    ax.spines["bottom"].set_alpha(0.45)
    return fig


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
    ordered = ["region_nombre"] + [col for col in available if col != "region"]
    return preview[ordered]


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

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
        ds49_totals,
        df_ds49_yoy,
        geojson_regions,
        regional_metrics,
        moran_i,
        moran_p,
    ) = load_project_data()

selected_region = territory_to_region_code(st.session_state["territory"])

components.html(
    build_horizontal_chile_svg(
        regional_metrics,
        geojson_regions,
        selected_region if st.session_state["has_selected_territory"] else None,
    ),
    height=720,
    scrolling=False,
)
st.markdown(
    '<div class="hero-map-note">Pasa el mouse para ver la región. Haz click sobre una región para explorar sus gráficos.</div>',
    unsafe_allow_html=True,
)

button_left, button_center, button_right = st.columns([1, 0.28, 1])
with button_center:
    if st.button("Ver Chile completo", type="primary", width="stretch"):
        st.session_state["territory"] = "Chile completo"
        st.session_state["has_selected_territory"] = True
        st.session_state["_scroll_to_charts"] = True
        st.query_params["territory"] = "Chile completo"
        st.rerun()

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
    top_left, top_right = st.columns(2, gap="large")
    with top_left:
        with st.container(border=True):
            show_matplotlib(plot_ds49_selected_region(ds49_totals, territory))
    with top_right:
        with st.container(border=True):
            fig, ax = plt.subplots(figsize=(7.4, 3.35), facecolor=COLOR_BG)
            plot_historical_line(ax, df_hist)
            show_matplotlib(fig)

    mid_left, mid_right = st.columns(2, gap="large")
    with mid_left:
        with st.container(border=True):
            fig, ax = plt.subplots(figsize=(5.2, 4.4), subplot_kw={"projection": "polar"}, facecolor=COLOR_BG)
            plot_radar_rm_vs_regions(ax, df_master)
            show_matplotlib(fig)
    with mid_right:
        with st.container(border=True):
            fig, ax = plt.subplots(figsize=(5.2, 4.4), facecolor=COLOR_BG)
            plot_donut_vulnerability(ax, df_context)
            show_matplotlib(fig)

    with st.container(border=True):
        fig, ax = plt.subplots(figsize=(10.0, 4.6), facecolor=COLOR_BG)
        plot_subsidy_butterfly(ax, df_context)
        show_matplotlib(fig)


with tab_region:
    st.markdown('<div class="section-label">Territorio seleccionado</div>', unsafe_allow_html=True)
    region_left, region_right = st.columns([1.2, 0.8], gap="large")

    with region_left:
        with st.container(border=True):
            show_matplotlib(plot_regional_ranking(regional_metrics, selected_region))

    with region_right:
        with st.container(border=True):
            selected_metrics = regional_metrics.copy()
            if selected_region is not None:
                selected_metrics = selected_metrics[selected_metrics["region"].eq(selected_region)]

            st.markdown("#### Tasa regional")
            st.dataframe(
                selected_metrics[["region_name", "pct_vulnerabilidad"]].rename(
                    columns={
                        "region_name": "Región",
                        "pct_vulnerabilidad": "% vulnerabilidad subsidiadas",
                    }
                ),
                hide_index=True,
                width="stretch",
            )

            st.markdown("#### Evolución DS49")
            show_matplotlib(plot_ds49_selected_region(ds49_totals, territory))


with tab_data:
    st.markdown('<div class="section-label">Datos filtrados</div>', unsafe_allow_html=True)
    show_methodology = st.checkbox("Mostrar metodología", value=False)
    preview = build_filtered_dataset_preview(df_context)
    st.dataframe(preview.head(600), hide_index=True, width="stretch")

    csv_data = preview.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar datos filtrados",
        data=csv_data,
        file_name=f"datos_filtrados_{normalize_text(territory).replace(' ', '_')}.csv",
        mime="text/csv",
    )

    if show_methodology:
        st.markdown(
            """
            **Definiciones usadas en el dashboard**

            - **Vivienda subsidiada:** hogares clasificados como `grupo_vivienda == "Subsidiada"`.
            - **Entorno violento:** presencia frecuente de balaceras o narcotráfico según variables CASEN.
            - **Presión económica:** ingreso autónomo del hogar menor a $600.000 o razón de subsidio superior a 20%.
            - **Vulnerabilidad estructural:** hogares subsidiados que presentan entorno violento o presión económica.
            - **Moran's I:** autocorrelación espacial regional calculada con contigüidad Queen sobre el mapa regional.
            """
        )
    else:
        st.info("Activa 'Mostrar metodología' en esta pestaña para ver las definiciones usadas.")
