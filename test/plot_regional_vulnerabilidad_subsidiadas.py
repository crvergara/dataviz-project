import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import BoundaryNorm, ListedColormap
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
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "mapa_vulnerabilidad_subsidiadas_region.png")
OUTPUT_PATH_HIRES = os.path.join(OUTPUT_DIR, "mapa_vulnerabilidad_subsidiadas_region_hires.png")

REGION_NAMES = {
    1: "Tarapaca",
    2: "Antofagasta",
    3: "Atacama",
    4: "Coquimbo",
    5: "Valparaiso",
    6: "O'Higgins",
    7: "Maule",
    8: "Biobio",
    9: "Araucania",
    10: "Los Lagos",
    11: "Aysen",
    12: "Magallanes",
    13: "Metropolitana",
    14: "Los Rios",
    15: "Arica y Parinacota",
    16: "Nuble",
}


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


def build_region_metrics(df):
    df_sub = df[df["grupo_vivienda"].eq("Subsidiada")].copy()
    df_sub["trampa_violencia"] = np.where((df_sub["v36b"] >= 3) | (df_sub["v36c"] >= 3), 1, 0)
    df_sub["trampa_economica"] = np.where(
        (df_sub["yautcorh"] < 600000) | (df_sub["subsidy_ratio"] > 0.20),
        1,
        0,
    )
    df_sub["vulnerabilidad_estructural"] = np.where(
        (df_sub["trampa_violencia"] == 1) | (df_sub["trampa_economica"] == 1),
        1,
        0,
    )

    rows = []
    for region_code, region_df in df_sub.groupby("region"):
        total_weight = region_df["expr"].sum()
        vulnerable_weight = region_df.loc[region_df["vulnerabilidad_estructural"].eq(1), "expr"].sum()
        pct = vulnerable_weight / total_weight * 100 if total_weight > 0 else np.nan
        rows.append({
            "region": int(region_code),
            "region_name": REGION_NAMES.get(int(region_code), str(region_code)),
            "pct_vulnerabilidad": pct,
        })

    return pd.DataFrame(rows).sort_values("pct_vulnerabilidad", ascending=False)


def build_queen_weights(geojson, decimals=4):
    region_vertices = {}

    for feature in geojson["features"]:
        region_code = int(feature["properties"]["codregion"])
        vertices = set()
        for coords in geometry_to_mainland_polygons(feature["geometry"]):
            rounded = np.round(coords, decimals=decimals)
            vertices.update(map(tuple, rounded))
        region_vertices[region_code] = vertices

    regions = sorted(region_vertices)
    weights = np.zeros((len(regions), len(regions)), dtype=float)

    for i, region_i in enumerate(regions):
        for j, region_j in enumerate(regions):
            if j <= i:
                continue
            if region_vertices[region_i].intersection(region_vertices[region_j]):
                weights[i, j] = 1
                weights[j, i] = 1

    return regions, weights


def morans_i(values, weights, permutations=9999, seed=42):
    values = np.asarray(values, dtype=float)
    centered = values - values.mean()
    n = len(values)
    s0 = weights.sum()

    observed = (n / s0) * ((weights * np.outer(centered, centered)).sum() / (centered @ centered))

    rng = np.random.default_rng(seed)
    permuted = np.empty(permutations)
    for i in range(permutations):
        shuffled = rng.permutation(centered)
        permuted[i] = (n / s0) * ((weights * np.outer(shuffled, shuffled)).sum() / (shuffled @ shuffled))

    p_positive = (np.sum(permuted >= observed) + 1) / (permutations + 1)
    return observed, p_positive


def calculate_spatial_autocorrelation(metrics, geojson):
    regions, weights = build_queen_weights(geojson)
    metric_lookup = metrics.set_index("region")["pct_vulnerabilidad"].to_dict()
    values = [metric_lookup[region] for region in regions]
    return morans_i(values, weights)


def draw_map(ax, geojson, metrics, cmap, norm):
    metric_lookup = metrics.set_index("region")["pct_vulnerabilidad"].to_dict()

    for feature in geojson["features"]:
        region_code = int(feature["properties"]["codregion"])
        value = metric_lookup.get(region_code, np.nan)
        facecolor = "#ddd6c7" if np.isnan(value) else cmap(norm(value))

        for coords in geometry_to_mainland_polygons(feature["geometry"]):
            ax.add_patch(
                patches.Polygon(
                    coords,
                    closed=True,
                    facecolor=facecolor,
                    edgecolor="#fff8ea",
                    linewidth=0.62,
                    joinstyle="round",
                    zorder=2,
                )
            )

    ax.set_xlim(-76.3, -66.2)
    ax.set_ylim(-56.4, -17.0)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_colorbar(fig, cmap, norm, boundaries):
    cax = fig.add_axes([0.24, 0.11, 0.52, 0.024])
    gradient = np.linspace(boundaries[0], boundaries[-1], 512).reshape(1, -1)
    cax.imshow(gradient, aspect="auto", cmap=cmap, norm=norm)
    cax.axis("off")

    fig.text(0.24, 0.092, f"{boundaries[0]:.0f}%", ha="left", va="top",
             fontsize=8, fontweight="bold", color=COLOR_TEXT)
    fig.text(0.50, 0.092, "55%", ha="center", va="top",
             fontsize=8, fontweight="bold", color=COLOR_TEXT)
    fig.text(0.76, 0.092, f"{boundaries[-1]:.0f}%+", ha="right", va="top",
             fontsize=8, fontweight="bold", color=COLOR_TEXT)
    fig.text(0.50, 0.062, "% de viviendas subsidiadas con vulnerabilidad estructural",
             ha="center", va="top", fontsize=7.8, color=COLOR_TEXT)


def add_top_regions(fig, metrics):
    top = metrics.head(3)
    text = "Top regional: " + " | ".join(
        f"{row.region_name} {row.pct_vulnerabilidad:.0f}%" for row in top.itertuples()
    )
    fig.text(0.08, 0.19, text, ha="left", va="top",
             fontsize=7.2, fontweight="bold", color=COLOR_TEXT)


def add_morans_i(fig, moran_i, p_value):
    fig.text(0.08, 0.155, f"Autocorrelacion espacial queen: Moran's I = {moran_i:.2f}  |  p = {p_value:.3f}",
             ha="left", va="top", fontsize=7.2, fontweight="bold", color=COLOR_RESTO)


def plot():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_parquet(DATA_PATH)
    with open(GEOJSON_PATH, encoding="utf-8-sig") as file:
        geojson = json.load(file)

    metrics = build_region_metrics(df)
    moran_i, moran_p = calculate_spatial_autocorrelation(metrics, geojson)

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

    fig = plt.figure(figsize=(4.3, 5.8), facecolor=COLOR_BG)
    ax = fig.add_axes([0.25, 0.20, 0.50, 0.65])

    draw_map(ax, geojson, metrics, cmap, norm)

    fig.text(0.08, 0.95, "Vulnerabilidad estructural en\nviviendas subsidiadas (%)",
             ha="left", va="top", fontsize=11, fontweight="black", color=COLOR_TEXT)
    fig.text(0.08, 0.885, "Tasa regional interna, no concentracion nacional",
             ha="left", va="top", fontsize=8, color=COLOR_TEXT, alpha=0.72)
    add_top_regions(fig, metrics)
    add_morans_i(fig, moran_i, moran_p)
    draw_colorbar(fig, cmap, norm, boundaries)

    fig.savefig(OUTPUT_PATH, dpi=300, facecolor=COLOR_BG)
    fig.savefig(OUTPUT_PATH_HIRES, dpi=450, facecolor=COLOR_BG)
    print(f"Grafico generado: {OUTPUT_PATH}")
    print(f"Grafico alta resolucion generado: {OUTPUT_PATH_HIRES}")
    print(f"Moran's I queen: {moran_i:.4f} | p positive: {moran_p:.4f}")
    print(metrics[["region_name", "pct_vulnerabilidad"]].round(1).to_string(index=False))


if __name__ == "__main__":
    plot()
