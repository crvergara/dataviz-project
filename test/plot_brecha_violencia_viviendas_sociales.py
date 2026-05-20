import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd


COLOR_BG = "#f8f4e3"
COLOR_TEXT = "#4c2e05"
COLOR_RM = "#f19143"
COLOR_RESTO = "#3c4f76"
COLOR_ALMOND = "#d38b5d"
COLOR_WARM = "#d35400"

DATA_PATH = "data/processed/master_dataset.parquet"
GEOJSON_PATH = "data/raw/regiones.json"
OUTPUT_DIR = "test/outputs"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "brecha_violencia_viviendas_sociales.png")


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Roboto", "Arial", "DejaVu Sans"],
    "text.color": COLOR_TEXT,
    "axes.labelcolor": COLOR_TEXT,
    "xtick.color": COLOR_TEXT,
    "ytick.color": COLOR_TEXT,
    "figure.facecolor": COLOR_BG,
    "axes.facecolor": COLOR_BG,
})


SHORT_REGION_NAMES = {
    1: "Tarapacá",
    2: "Antofagasta",
    3: "Atacama",
    4: "Coquimbo",
    5: "Valparaíso",
    6: "O'Higgins",
    7: "Maule",
    8: "Biobío",
    9: "La Araucanía",
    10: "Los Lagos",
    11: "Aysén",
    12: "Magallanes",
    13: "Metropolitana",
    14: "Los Ríos",
    15: "Arica y Parinacota",
    16: "Ñuble",
}


def weighted_pct(subset, mask):
    total = subset["expr"].sum()
    if total == 0:
        return np.nan
    return subset.loc[mask, "expr"].sum() / total * 100


def build_region_metrics(df):
    rows = []

    for region_code, name in SHORT_REGION_NAMES.items():
        region_df = df[df["region"].eq(region_code)]
        subsidized = region_df[region_df["grupo_vivienda"].eq("Subsidiada")]
        non_subsidized = region_df[~region_df["grupo_vivienda"].eq("Subsidiada")]

        violence_subsidized = (
            subsidized["v36b"].ge(3)
            | subsidized["v36c"].ge(3)
            | subsidized["v36e"].ge(3)
        )
        violence_non_subsidized = (
            non_subsidized["v36b"].ge(3)
            | non_subsidized["v36c"].ge(3)
            | non_subsidized["v36e"].ge(3)
        )

        pct_subsidized = weighted_pct(subsidized, violence_subsidized)
        pct_non_subsidized = weighted_pct(non_subsidized, violence_non_subsidized)

        rows.append({
            "region": region_code,
            "name": name,
            "violence_subsidized_pct": pct_subsidized,
            "violence_non_subsidized_pct": pct_non_subsidized,
            "violence_gap_pp": pct_subsidized - pct_non_subsidized,
        })

    return pd.DataFrame(rows)


def polygon_area(coords):
    x = coords[:, 0]
    y = coords[:, 1]
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2


def geometry_to_mainland_polygons(geometry, min_lon=-80):
    if geometry["type"] == "Polygon":
        polygons = [geometry["coordinates"]]
    elif geometry["type"] == "MultiPolygon":
        polygons = geometry["coordinates"]
    else:
        return []

    pieces = []
    for polygon in polygons:
        outer = np.asarray(polygon[0], dtype=float)
        if outer[:, 0].max() < min_lon:
            continue
        if polygon_area(outer) < 0.003:
            continue
        pieces.append(outer)
    return pieces


def draw_chile_map(ax, geojson, metrics, cmap, norm):
    metric_lookup = metrics.set_index("region").to_dict("index")

    for feature in geojson["features"]:
        region_code = int(feature["properties"]["codregion"])
        metric = metric_lookup[region_code]
        facecolor = cmap(norm(metric["violence_gap_pp"]))

        for coords in geometry_to_mainland_polygons(feature["geometry"]):
            ax.add_patch(
                patches.Polygon(
                    coords,
                    closed=True,
                    facecolor=facecolor,
                    edgecolor=COLOR_BG,
                    linewidth=0.55,
                    joinstyle="round",
                    zorder=2,
                )
            )

    ax.set_xlim(-76.3, -66.2)
    ax.set_ylim(-56.4, -17.0)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(-75.9, -18.0, "NORTE", fontsize=8, fontweight="black",
            color=COLOR_ALMOND, ha="left", va="top")
    ax.text(-75.9, -55.8, "SUR", fontsize=8, fontweight="black",
            color=COLOR_ALMOND, ha="left", va="bottom")


def draw_side_panel(ax, metrics, cmap, norm):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.0, 0.98, "Color del mapa", fontsize=10.5,
            fontweight="black", color=COLOR_TEXT, va="top")
    ax.text(
        0.0,
        0.935,
        "Brecha de violencia frecuente\nsubsidiadas - no subsidiadas",
        fontsize=8.5,
        fontweight="bold",
        color=COLOR_TEXT,
        va="top",
        linespacing=1.05,
    )

    gradient = np.linspace(-5, 12, 256).reshape(1, -1)
    ax.imshow(gradient, extent=(0.0, 0.72, 0.835, 0.87),
              cmap=cmap, norm=norm, aspect="auto")
    ax.text(0.0, 0.81, "-5 pp", fontsize=7.5, fontweight="bold",
            ha="left", color=COLOR_RESTO)
    ax.text(0.36, 0.81, "0", fontsize=7.5, fontweight="bold",
            ha="center", color=COLOR_TEXT)
    ax.text(0.72, 0.81, "+12 pp", fontsize=7.5, fontweight="bold",
            ha="right", color=COLOR_WARM)

    top_gap = metrics.sort_values("violence_gap_pp", ascending=False).head(5)
    ax.text(0.0, 0.735, "Mayor brecha contra no subsidiadas",
            fontsize=10.5, fontweight="black", color=COLOR_TEXT, va="top")
    for i, row in enumerate(top_gap.itertuples(index=False)):
        y = 0.685 - i * 0.07
        width = max(row.violence_gap_pp, 0) / 12 * 0.56
        ax.add_patch(
            patches.FancyBboxPatch(
                (0.0, y - 0.018),
                width,
                0.035,
                boxstyle="round,pad=0.002,rounding_size=0.008",
                facecolor=cmap(norm(row.violence_gap_pp)),
                edgecolor="none",
            )
        )
        ax.text(0.60, y, f"+{row.violence_gap_pp:.1f} pp", fontsize=8.6,
                fontweight="black", ha="right", va="center", color=COLOR_TEXT)
        ax.text(0.64, y, row.name, fontsize=8.4, fontweight="bold",
                ha="left", va="center", color=COLOR_TEXT)

    top_level = metrics.sort_values("violence_subsidized_pct", ascending=False).head(3)
    ax.text(0.0, 0.345, "Violencia frecuente en subsidiadas",
            fontsize=10.5, fontweight="black", color=COLOR_TEXT, va="top")
    ax.text(
        0.0,
        0.305,
        "% de hogares subsidiados con violencia frecuente",
        fontsize=8,
        fontweight="bold",
        color=COLOR_TEXT,
        va="top",
    )
    for i, row in enumerate(top_level.itertuples(index=False)):
        y = 0.235 - i * 0.068
        width = row.violence_subsidized_pct / 65 * 0.58
        ax.add_patch(
            patches.FancyBboxPatch(
                (0.0, y - 0.014),
                width,
                0.028,
                boxstyle="round,pad=0.002,rounding_size=0.008",
                facecolor=COLOR_RM,
                edgecolor="none",
                alpha=0.92,
            )
        )
        ax.text(0.62, y, f"{row.violence_subsidized_pct:.1f}%", fontsize=8.3,
                fontweight="black", ha="right", va="center", color=COLOR_TEXT)
        ax.text(0.66, y, row.name, fontsize=8.1, fontweight="bold",
                ha="left", va="center", color=COLOR_TEXT)

    ax.text(
        0.0,
        0.025,
        "Violencia frecuente: consumo de drogas/alcohol,\n"
        "narco-tráfico o balaceras reportadas muchas veces/siempre.",
        fontsize=7.6,
        color=COLOR_TEXT,
        va="bottom",
        linespacing=1.15,
        alpha=0.8,
    )


def plot():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_parquet(DATA_PATH)
    with open(GEOJSON_PATH, encoding="utf-8-sig") as file:
        geojson = json.load(file)

    metrics = build_region_metrics(df)
    cmap = LinearSegmentedColormap.from_list(
        "violence_gap",
        [COLOR_RESTO, "#f4ead6", COLOR_RM, COLOR_WARM],
    )
    norm = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=12)

    fig = plt.figure(figsize=(8.8, 7.1), facecolor=COLOR_BG)
    ax_map = fig.add_axes([0.04, 0.07, 0.55, 0.82])
    ax_side = fig.add_axes([0.62, 0.11, 0.34, 0.74])

    draw_chile_map(ax_map, geojson, metrics, cmap, norm)
    draw_side_panel(ax_side, metrics, cmap, norm)

    fig.text(
        0.055,
        0.965,
        "Brecha de violencia en viviendas sociales por región",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="black",
        color=COLOR_TEXT,
    )
    fig.text(
        0.055,
        0.922,
        "Diferencia en puntos porcentuales entre hogares subsidiados y no subsidiados con violencia frecuente en el entorno.",
        ha="left",
        va="top",
        fontsize=8.2,
        color=COLOR_TEXT,
        alpha=0.78,
    )
    fig.text(
        0.5,
        0.025,
        "Fuente: Elaboración propia basada en CASEN 2024. Porcentajes ponderados por expr. Islas oceánicas alejadas omitidas por escala.",
        ha="center",
        fontsize=7.2,
        color=COLOR_ALMOND,
    )

    fig.savefig(OUTPUT_PATH, dpi=300, facecolor=COLOR_BG)
    print(f"Mapa generado: {OUTPUT_PATH}")


if __name__ == "__main__":
    plot()
