import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd


COLOR_BG = "#f8f4e3"
COLOR_TEXT = "#4c2e05"
COLOR_RM = "#f19143"
COLOR_ALMOND = "#d38b5d"
COLOR_WARM = "#d35400"

DATA_PATH = "data/processed/master_dataset.parquet"
GEOJSON_PATH = "data/raw/regiones.json"
OUTPUT_DIR = "plots_alone"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "concentracion_nacional_viviendas_sociales.png")


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


def build_region_metrics(df):
    national_subsidized = df.loc[df["grupo_vivienda"].eq("Subsidiada"), "expr"].sum()
    rows = []

    for region_code, region_df in df.groupby("region"):
        subsidized = region_df[region_df["grupo_vivienda"].eq("Subsidiada")]
        rows.append({
            "region": int(region_code),
            "national_share_subsidized": subsidized["expr"].sum() / national_subsidized * 100,
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


def draw_map(ax, geojson, metrics, cmap, norm):
    metric_lookup = metrics.set_index("region")["national_share_subsidized"].to_dict()

    for feature in geojson["features"]:
        region_code = int(feature["properties"]["codregion"])
        facecolor = cmap(norm(metric_lookup[region_code]))

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


def draw_colorbar(fig, cmap, norm):
    cax = fig.add_axes([0.28, 0.105, 0.44, 0.026])
    gradient = np.linspace(norm.vmin, norm.vmax, 256).reshape(1, -1)
    cax.imshow(gradient, aspect="auto", cmap=cmap, norm=norm)
    cax.axis("off")

    fig.text(0.28, 0.085, "0%", ha="left", va="top",
             fontsize=8, fontweight="bold", color=COLOR_TEXT)
    fig.text(0.72, 0.085, "12%+", ha="right", va="top",
             fontsize=8, fontweight="bold", color=COLOR_TEXT)
    fig.text(0.50, 0.055, "% del total nacional subsidiado",
             ha="center", va="top", fontsize=8, color=COLOR_TEXT)


def plot():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_parquet(DATA_PATH)
    with open(GEOJSON_PATH, encoding="utf-8-sig") as file:
        geojson = json.load(file)

    metrics = build_region_metrics(df)
    cmap = LinearSegmentedColormap.from_list(
        "concentration",
        ["#f7ead8", "#f2c49d", COLOR_RM, COLOR_WARM],
    )
    norm = Normalize(vmin=0, vmax=12)

    fig = plt.figure(figsize=(4.1, 5.6), facecolor=COLOR_BG)
    ax = fig.add_axes([0.25, 0.18, 0.50, 0.68])

    draw_map(ax, geojson, metrics, cmap, norm)

    fig.text(0.08, 0.94, "Concentración nacional de\nviviendas sociales (%)",
             ha="left", va="top", fontsize=11, fontweight="black", color=COLOR_TEXT)
    draw_colorbar(fig, cmap, norm)

    fig.savefig(OUTPUT_PATH, dpi=300, facecolor=COLOR_BG)
    print(f"Gráfico generado: {OUTPUT_PATH}")


if __name__ == "__main__":
    plot()
