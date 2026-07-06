import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from matplotlib.path import Path
from scipy.stats import gaussian_kde
from matplotlib.colors import BoundaryNorm, ListedColormap

# ─────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL Y COLORES (Rubrica: Estética Premium)
# ─────────────────────────────────────────────────────────
COLOR_BG     = '#f8f4e3'       # Ivory Mist
COLOR_TEXT   = '#4c2e05'       # Deep Walnut
COLOR_RM     = '#f19143'       # Sandy Brown (Naranja)
COLOR_RESTO  = '#3c4f76'       # Dusk Blue (Azul)
COLOR_ALMOND = '#d38b5d'       # Toasted Almond 
COLOR_WARM   = '#d35400'       # Terracota
TITLE_FS     = 9.5

REGION_NAMES = {
    1: "Tarapacá",
    2: "Antofagasta",
    3: "Atacama",
    4: "Coquimbo",
    5: "Valparaíso",
    6: "O'Higgins",
    7: "Maule",
    8: "Biobío",
    9: "Araucanía",
    10: "Los Lagos",
    11: "Aysén",
    12: "Magallanes",
    13: "Metropolitana",
    14: "Los Ríos",
    15: "Arica y Parinacota",
    16: "Ñuble",
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Roboto', 'Arial', 'DejaVu Sans'],
    'text.color': COLOR_TEXT,
    'axes.labelcolor': COLOR_TEXT,
    'xtick.color': COLOR_TEXT,
    'ytick.color': COLOR_TEXT,
    'figure.facecolor': COLOR_BG,
    'axes.facecolor': COLOR_BG,
    'axes.edgecolor': COLOR_ALMOND,
    'axes.linewidth': 0.8,
})

# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
def weighted_quantiles(values, weights, quantiles):
    df_clean = pd.DataFrame({'val': values, 'wt': weights}).dropna()
    sorter = np.argsort(df_clean['val'])
    val = df_clean['val'].iloc[sorter]
    wt = df_clean['wt'].iloc[sorter]
    wq = np.cumsum(wt) - 0.5 * wt
    wq /= np.sum(wt)
    return np.interp(quantiles, wq, val)

def thin_spines(ax, sides=['bottom', 'left']):
    for spine in ax.spines.values():
        spine.set_visible(False)
    for side in sides:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(COLOR_ALMOND)
        ax.spines[side].set_alpha(0.4)

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

def build_regional_vulnerability_metrics(df):
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

# ─────────────────────────────────────────────────────────
# COMPONENTES DE GRÁFICOS
# ─────────────────────────────────────────────────────────

def plot_quintile_heatmap(ax, df):
    """Lógica de Grfico_1.py: Heatmap de concentración de pobreza"""
    df_h = df.dropna(subset=['yautcorh', 'expr']).copy()
    quantiles = [0.2, 0.4, 0.6, 0.8]
    cortes = weighted_quantiles(df_h['yautcorh'], df_h['expr'], quantiles)
    
    bins = [-np.inf] + list(cortes) + [np.inf]
    labels = ['Q1\n(Más Pobres)', 'Q2', 'Q3', 'Q4', 'Q5\n(Más Ricos)']
    df_h['Quintil'] = pd.cut(df_h['yautcorh'], bins=bins, labels=labels, include_lowest=True)
    
    # Etiquetas Y mismo tamaño, con salto de línea para ahorrar espacio
    df_h['grupo_vivienda_label'] = np.where(
        df_h['grupo_vivienda'] == 'Subsidiada',
        'Con\nSubsidio',
        'No\nSubsidiada'
    )
    
    grouped = df_h.groupby(['grupo_vivienda_label', 'Quintil'], observed=False)['expr'].sum().unstack()
    props = grouped.div(grouped.sum(axis=1), axis=0) * 100
    props = props.loc[['Con\nSubsidio', 'No\nSubsidiada']]
    
    colors = [COLOR_BG, COLOR_RM, COLOR_TEXT]
    custom_cmap = sns.blend_palette(colors, as_cmap=True)
    
    cax = ax.inset_axes([1.012, 0.12, 0.014, 0.76])

    # Dibujar heatmap
    sns.heatmap(props, cmap=custom_cmap, annot=True, fmt=".1f", linewidths=4, linecolor=COLOR_BG, 
                vmin=0, vmax=35, cbar=True, cbar_ax=cax,
                cbar_kws={'ticks': [0, 35]}, ax=ax)
    cax.tick_params(labelsize=7, length=0, colors=COLOR_TEXT)
    cax.set_title('Conc.\n%', fontsize=6, color=COLOR_TEXT, pad=3)
    
    # Todos los números blancos y sin efectos de borde
    for t in ax.texts:
        if "." in t.get_text() and "%" not in t.get_text():
            val = float(t.get_text())
            t.set_text(f'{val:.1f}%')
            t.set_fontsize(11)
            t.set_fontweight('bold')
            t.set_color('white')  # Siempre blanco
            
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', length=0, labelsize=10, pad=10)
    ax.set_title("1. Concentración de hogares según subsidio y quintil de ingresos",
                 fontsize=TITLE_FS, fontweight='black', color=COLOR_TEXT, loc='left', pad=44)
    
    # Etiquetas Y mismo tamaño, discretas
    ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold', fontsize=9)
    ax.tick_params(axis='y', length=0, pad=8)

def plot_regional_vulnerability_map(ax, df, geojson):
    """Mapa regional: tasa interna de vulnerabilidad en viviendas subsidiadas."""
    metrics = build_regional_vulnerability_metrics(df)
    moran_i, moran_p = calculate_spatial_autocorrelation(metrics, geojson)
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
    # En la infografia el mapa funciona como columna visual; se permite
    # una leve deformacion para que use todo el alto asignado.
    ax.set_aspect("auto")
    ax.set_anchor("NW")
    ax.axis("off")

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
                    linewidth=0.48,
                    joinstyle="round",
                    zorder=2,
                )
            )

    ax.set_title("1. Vulnerabilidad estructural en\nViviendas Sociales (%)",
                 fontsize=TITLE_FS, fontweight="black", color=COLOR_TEXT, loc="left", pad=8)
    ax.text(0.0, 0.945, "Tasa regional\nen subsidiadas",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=5.7, color=COLOR_TEXT, alpha=0.72, linespacing=1.05)
    ax.text(0.0, 0.895, f"Moran's I: {moran_i:.2f}\np = {moran_p:.3f}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=5.7, fontweight="bold", color=COLOR_RESTO, linespacing=1.05)

    cax = ax.inset_axes([0.78, 0.40, 0.045, 0.24])
    gradient = np.linspace(boundaries[0], boundaries[-1], 512).reshape(-1, 1)
    cax.imshow(gradient, aspect="auto", cmap=cmap, norm=norm, origin="lower")
    cax.axis("off")
    cax.text(1.30, 0.00, f"{boundaries[0]:.0f}%", transform=cax.transAxes,
             ha="left", va="bottom", fontsize=5.6, fontweight="bold", color=COLOR_TEXT)
    cax.text(1.30, 0.52, "55%", transform=cax.transAxes,
             ha="left", va="center", fontsize=5.6, fontweight="bold", color=COLOR_TEXT)
    cax.text(1.30, 1.00, f"{boundaries[-1]:.0f}%+", transform=cax.transAxes,
             ha="left", va="top", fontsize=5.6, fontweight="bold", color=COLOR_TEXT)

    ax.text(0.78, 0.365, "% regional",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=5.5, color=COLOR_TEXT, alpha=0.74)

def read_ds49_region_totals(path="data/raw/SUDS49Mar2026.xlsx"):
    """Lee totales regionales DS49 desde la planilla MINVU."""
    raw = pd.read_excel(path, sheet_name="Total", header=None)

    first_col = raw.iloc[:, 0].astype(str).str.strip()
    header_idx = first_col[first_col.str.startswith("Regi")].index[0]
    year_row_idx = header_idx + 1

    year_cols = []
    years = []
    for col in raw.columns:
        value = raw.iat[year_row_idx, col]
        if pd.isna(value):
            continue
        try:
            year = int(value)
        except (TypeError, ValueError):
            continue
        if 2012 <= year <= 2026:
            year_cols.append(col)
            years.append(year)

    data = raw.iloc[year_row_idx + 1:].copy()
    data = data[data.iloc[:, 0].notna()]
    data = data[~data.iloc[:, 0].astype(str).str.startswith(("FUENTE", "NOTAS", "\xa0"))]

    totals = data.iloc[:, [0] + year_cols].copy()
    totals.columns = ["region"] + years
    totals["region"] = totals["region"].astype(str).str.strip()

    for year in years:
        totals[year] = pd.to_numeric(totals[year], errors="coerce").fillna(0)

    return totals

def prepare_ds49_yoy(totals):
    years = [col for col in totals.columns if isinstance(col, int) and col <= 2025]
    valid_regions = totals[
        ~totals["region"].str.startswith("Total Pa")
        & ~totals["region"].str.startswith("Sin Informaci")
    ]

    rm = valid_regions[valid_regions["region"].eq("Metropolitana")][years].iloc[0]
    regions = valid_regions[valid_regions["region"].ne("Metropolitana")][years].sum()

    wide = pd.DataFrame(
        {
            "year": years,
            "Santiago (RM)": rm.to_numpy(),
            "Regiones": regions.to_numpy(),
        }
    )

    long = wide.melt(id_vars="year", var_name="zona", value_name="beneficiados")
    long["yoy_pct"] = long.groupby("zona")["beneficiados"].pct_change() * 100
    return long.dropna(subset=["yoy_pct"])

def plot_ds49_yoy_growth(ax, data):
    palette = {"Santiago (RM)": COLOR_RM, "Regiones": COLOR_RESTO}

    sns.lineplot(
        data=data,
        x="year",
        y="yoy_pct",
        hue="zona",
        palette=palette,
        marker="o",
        markersize=5.2,
        linewidth=2.45,
        ax=ax,
        zorder=3,
    )

    ax.axhline(0, color=COLOR_TEXT, linewidth=0.95, alpha=0.52, zorder=1)
    ax.text(0.35, 1.090, "2. Beneficiados DS49: crecimiento interanual (%)",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=TITLE_FS, fontweight='black', color=COLOR_TEXT)
    ax.text(0.35, 1.040, "Santiago (RM) vs regiones, 2013-2025",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7.5, color=COLOR_TEXT, alpha=0.78)

    ax.set_ylim(-85, 210)
    ax.set_yticks([-50, 0, 50, 100, 150, 200])
    ax.set_xticks(sorted(data["year"].unique()))
    ax.set_xticklabels([str(int(y)) for y in sorted(data["year"].unique())], fontsize=6.8)
    ax.tick_params(axis='y', labelsize=7.2, length=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.yaxis.grid(True, linestyle='--', alpha=0.30, color=COLOR_ALMOND)
    ax.xaxis.grid(False)
    thin_spines(ax, ['bottom'])

    label_years = [2017, 2021, 2023, 2025]
    for _, row in data[data["year"].isin(label_years)].iterrows():
        color = palette[row["zona"]]
        offset = 8 if row["yoy_pct"] >= 0 else -10
        va = "bottom" if row["yoy_pct"] >= 0 else "top"
        ax.text(row["year"], row["yoy_pct"] + offset, f"{row['yoy_pct']:+.0f}%",
                ha="center", va=va, fontsize=6.8, fontweight="black", color=color)

    legend = ax.legend(title=None, loc="upper left", bbox_to_anchor=(0.01, 0.91),
                       frameon=False, fontsize=7.8, handlelength=1.8)
    for text in legend.get_texts():
        text.set_color(COLOR_TEXT)

def plot_butterfly_gaps(ax, df, compact=False):
    """Lógica de plot_diverging_bar.py: Comparación RM vs Regiones"""
    df['Viaje Crítico (>1h)']     = np.where(df['o28a_hr'] >= 1, 1, 0)
    df['Balaceras Frecuentes']    = np.where(df['v36b'] >= 3, 1, 0)
    df['Narco-tráfico Frecuente'] = np.where(df['v36c'] >= 3, 1, 0)
    df['Alta Dependencia (>20%)'] = np.where(df['subsidy_ratio'] > 0.20, 1, 0)
    df['Ingreso Autónomo <$600k'] = np.where(df['yautcorh'] < 600000, 1, 0)
    
    vars_to_plot = [
        'Ingreso Autónomo <$600k',
        'Alta Dependencia (>20%)',
        'Viaje Crítico (>1h)',
        'Narco-tráfico Frecuente',
        'Balaceras Frecuentes'
    ]
    
    df_rm = df[(df['grupo_vivienda'] == 'Subsidiada') & (df['region'] == 13)]
    df_re = df[(df['grupo_vivienda'] == 'Subsidiada') & (df['region'] != 13)]
    
    def wpct(sub, var):
        w = sub['expr'].sum()
        return (sub[sub[var] == 1]['expr'].sum() / w * 100) if w > 0 else 0

    rows = []
    for var in vars_to_plot:
        rows.append({'Variable': var, 'RM': wpct(df_rm, var), 'RE': wpct(df_re, var)})
    
    plot_df = pd.DataFrame(rows)
    row_gap = 1.18 if compact else 1.28
    bar_h = 0.58 if compact else 0.68
    label_fs = 5.6 if compact else 8
    value_fs = 7.4 if compact else 11
    header_fs = 7.8 if compact else 12
    title_fs = TITLE_FS
    label_map = {
        'Ingreso Autónomo <$600k': 'INGRESO <$600K',
        'Alta Dependencia (>20%)': 'DEPENDENCIA >20%',
        'Viaje Crítico (>1h)': 'VIAJE >1H',
        'Narco-tráfico Frecuente': 'NARCO FRECUENTE',
        'Balaceras Frecuentes': 'BALACERAS',
    }
    
    for i, row in plot_df.iterrows():
        y = i * row_gap
        ax.barh(y, -row['RE'], color=COLOR_RESTO, height=bar_h, zorder=3)
        ax.barh(y, row['RM'], color=COLOR_RM, height=bar_h, zorder=3)
        
        label = label_map[row['Variable']] if compact else row['Variable'].upper()
        ax.text(0, y + (0.38 if compact else 0.46), label,
                ha='center', va='bottom', fontsize=label_fs, fontweight='bold', color=COLOR_TEXT)
        ax.text(-row['RE'] - (1.4 if compact else 2), y, f"{row['RE']:.0f}%",
                ha='right', va='center', color=COLOR_RESTO, fontsize=value_fs, fontweight='black')
        ax.text(row['RM'] + (1.4 if compact else 2), y, f"{row['RM']:.0f}%",
                ha='left', va='center', color=COLOR_RM, fontsize=value_fs, fontweight='black')

    top_y = (len(vars_to_plot) - 1) * row_gap
    ax.vlines(0, ymin=-0.45, ymax=top_y + 0.82, color=COLOR_TEXT, lw=1.5, alpha=0.5)
    
    max_val = max(plot_df['RM'].max(), plot_df['RE'].max())
    lim = max_val + 9
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-1.25, top_y + 2.85)
    ax.set_yticks([])
    ax.axis('off')
    
    ax.text(-lim * 0.55 + (5.5 if compact else 8), top_y + (0.96 if compact else 1.12), 'REGIONES',
        ha='center', color=COLOR_RESTO, fontsize=header_fs, fontweight='black')

    ax.text(lim * 0.55, top_y + (0.96 if compact else 1.12), 'SANTIAGO (RM)',
        ha='center', color=COLOR_RM, fontsize=header_fs, fontweight='black')
    # Subtítulo indicando qué población se compara
    title_num = "6." if compact else "2."
    ax.text(-lim * (0.92 if compact else 0.7), top_y + (2.36 if compact else 2.72),
            f"{title_num} Brechas territoriales en\nViviendas Subsidiadas (%)",
            ha='left', va='top', fontsize=title_fs, fontweight='black', color=COLOR_TEXT)

def plot_subsidy_butterfly(ax, df):
    """Comparativa nacional: viviendas sin subsidio vs con subsidio."""
    df = df.copy()
    df['Balaceras Frecuentes'] = np.where(df['v36b'] >= 3, 1, 0)
    df['Narcotráfico Frecuente'] = np.where(df['v36c'] >= 3, 1, 0)
    df['Alta Dependencia (>20%)'] = np.where(df['subsidy_ratio'] > 0.20, 1, 0)
    df['Ingreso Autónomo <$600k'] = np.where(df['yautcorh'] < 600000, 1, 0)
    df['Pobreza por Ingresos'] = np.where(df['pobreza'].isin([1, 2]), 1, 0)

    vars_to_plot = [
        'Balaceras Frecuentes',
        'Narcotráfico Frecuente',
        'Pobreza por Ingresos',
        'Alta Dependencia (>20%)',
        'Ingreso Autónomo <$600k',
    ]
    label_map = {
        'Balaceras Frecuentes': 'BALACERAS',
        'Narcotráfico Frecuente': 'NARCO FRECUENTE',
        'Pobreza por Ingresos': 'POBREZA INGRESOS',
        'Alta Dependencia (>20%)': 'DEPENDENCIA >20%',
        'Ingreso Autónomo <$600k': 'INGRESO <$600K',
    }

    df_sub = df[df['is_subsidized'] == 1]
    df_nosub = df[df['is_subsidized'] == 0]

    def wpct(subset, var):
        w = subset['expr'].sum()
        return (subset.loc[subset[var] == 1, 'expr'].sum() / w * 100) if w > 0 else 0

    rows = []
    for var in vars_to_plot:
        rows.append({
            'Variable': var,
            'No_Subsidiada': wpct(df_nosub, var),
            'Subsidiada': wpct(df_sub, var),
        })
    plot_df = pd.DataFrame(rows)

    row_gap = 1.05
    track_max = 42
    bar_h = 0.13
    blue_lit = COLOR_RESTO
    blue_dim = "#6f7fa3"
    orange_lit = COLOR_RM
    orange_dim = "#f4b06f"
    top_y = (len(vars_to_plot) - 1) * row_gap

    for i, row in plot_df.iterrows():
        y = top_y - i * row_gap
        val_nosub = row['No_Subsidiada']
        val_sub = row['Subsidiada']
        nosub_wins = val_nosub >= val_sub
        sub_wins = val_sub >= val_nosub
        nosub_color = blue_lit if nosub_wins else blue_dim
        sub_color = orange_lit if sub_wins else orange_dim

        ax.barh(y, -track_max, left=0, color=COLOR_ALMOND, alpha=0.18,
                height=bar_h, edgecolor='none', zorder=1)
        ax.barh(y, track_max, left=0, color=COLOR_ALMOND, alpha=0.18,
                height=bar_h, edgecolor='none', zorder=1)
        ax.barh(y, -val_nosub, left=0, color=nosub_color,
                height=bar_h, edgecolor='none', zorder=3)
        ax.barh(y, val_sub, left=0, color=sub_color,
                height=bar_h, edgecolor='none', zorder=3)

        ax.text(0, y + 0.34, label_map[row['Variable']], ha='center', va='bottom',
                color=COLOR_TEXT, fontsize=5.5, fontweight='black', zorder=5)
        txt_nosub = ax.text(-track_max - 2.4, y, f"{val_nosub:.0f}%", ha='right', va='center',
                color=nosub_color, fontsize=7.5, fontweight='black', zorder=5)
        txt_sub = ax.text(track_max + 2.4, y, f"{val_sub:.0f}%", ha='left', va='center',
                color=sub_color, fontsize=7.5, fontweight='black', zorder=5)

    ax.text(-12, top_y + 0.60, 'SIN SUBSIDIO', ha='right', va='bottom',
            color=COLOR_RESTO, fontsize=7.9, fontweight='black')
    ax.text(12, top_y + 0.60, 'CON SUBSIDIO', ha='left', va='bottom',
            color=COLOR_RM, fontsize=7.9, fontweight='black')

    ax.text(-55, top_y + 1.80, "6. Vulnerabilidades nacionales:\nSin subsidio vs Con subsidio (%)",
            ha='left', va='top', fontsize=TITLE_FS, fontweight='black', color=COLOR_TEXT,
            linespacing=1.05)

    ax.set_xlim(-58, 58)
    ax.set_ylim(-0.75, top_y + 1.72)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.axis('off')

def plot_historical_line(ax, df_hist):
    """Lógica de Grafico3.py: Evolución de balaceras"""
    plot_data = df_hist.groupby(['year', 'zona'])[['is_balacera', 'ponderador']].apply(
        lambda x: np.average(x['is_balacera'], weights=x['ponderador']) * 100
    ).unstack()
    
    years = plot_data.index
    ax.fill_between(years, plot_data['Resto de Regiones'], plot_data['RM (Santiago)'], 
                    color=COLOR_RM, alpha=0.08, zorder=1)
    
    for year in years:
        val_rm = plot_data.loc[year, 'RM (Santiago)']
        val_resto = plot_data.loc[year, 'Resto de Regiones']
        ax.plot([year, year], [val_resto, val_rm], color=COLOR_RM, linestyle='--', linewidth=1, alpha=0.4, zorder=2)
        
        ax.text(year, val_rm + 1.2, f"{int(val_rm)}%", color=COLOR_RM, fontweight='bold', ha='center', fontsize=10)
        ax.text(year, val_resto - 1.2, f"{int(val_resto)}%", color=COLOR_RESTO, fontweight='bold', ha='center', fontsize=10, va='top')
        
        # Brecha en el centro de la línea punteada
        mid_y = (val_rm + val_resto) / 2
        brecha = int(val_rm - val_resto)
        ax.text(year + 0.15, mid_y, f"+{brecha}pp", color=COLOR_WARM,
                fontsize=8, fontweight='black', va='center', ha='left', alpha=0.85)

    ax.plot(years, plot_data['RM (Santiago)'], marker='o', color=COLOR_RM, linewidth=3, markersize=8, label='Santiago (RM)', zorder=3)
    ax.plot(years, plot_data['Resto de Regiones'], marker='o', color=COLOR_RESTO, linewidth=3, markersize=8, label='Regiones', zorder=3)
    
    ax.set_xticks(years)
    ax.set_xticklabels([str(int(y)) for y in years], fontsize=10)
    ax.set_ylim(0, 60)
    ax.set_yticks(np.arange(0, 61, 20))
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color=COLOR_ALMOND)
    thin_spines(ax, ['bottom'])
    
    # Título claro del gráfico
    ax.set_title("3. Evolución de balaceras en\nViviendas Subsidiadas (%)",
                 fontsize=TITLE_FS, fontweight='black', color=COLOR_TEXT, loc='left', pad=20)

    
    ax.legend(loc='upper left', frameon=False, facecolor='none', edgecolor='none', fontsize=9)

def plot_radar_rm_vs_regions(ax, df):
    """Radar compacto: RM vs regiones en viviendas subsidiadas."""
    df_sub = df[df['grupo_vivienda'] == 'Subsidiada'].copy()

    df_sub['dim_multidim'] = np.where(df_sub['pobreza_multi'] == 1, 1, 0)
    df_sub['dim_hacinamiento'] = np.where(df_sub['ind_hacina'] >= 2, 1, 0)
    df_sub['dim_balaceras'] = np.where(df_sub['v36e'] >= 3, 1, 0)
    df_sub['dim_narco'] = np.where(df_sub['v36c'] >= 3, 1, 0)
    df_sub['dim_alumbrado'] = np.where(df_sub['v35a'] == 2, 1, 0)
    df_sub['dim_basura'] = np.where(df_sub['v35c'] == 2, 1, 0)

    dimensions = [
        'dim_multidim',
        'dim_hacinamiento',
        'dim_balaceras',
        'dim_narco',
        'dim_alumbrado',
        'dim_basura'
    ]
    labels = ['Pobreza\nmulti', 'Hacinam.', 'Balaceras', 'Narco', 'Sin\nluz', 'Basura']

    def wpct(data, dim):
        w = data['expr'].sum()
        return (data.loc[data[dim] == 1, 'expr'].sum() / w * 100) if w > 0 else 0

    df_rm = df_sub[df_sub['region'] == 13]
    df_re = df_sub[df_sub['region'] != 13]
    vals_rm = [wpct(df_rm, dim) for dim in dimensions]
    vals_re = [wpct(df_re, dim) for dim in dimensions]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    vals_rm += vals_rm[:1]
    vals_re += vals_re[:1]

    ax.set_facecolor(COLOR_BG)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(angles, vals_re, color=COLOR_RESTO, linewidth=1.7, marker='o',
            markersize=3.2, label='Regiones', zorder=3)
    ax.fill(angles, vals_re, color=COLOR_RESTO, alpha=0.16, zorder=2)
    ax.plot(angles, vals_rm, color=COLOR_RM, linewidth=1.9, marker='o',
            markersize=3.2, label='Santiago (RM)', zorder=4)
    ax.fill(angles, vals_rm, color=COLOR_RM, alpha=0.18, zorder=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=5.5, fontweight='bold', color=COLOR_TEXT)
    ax.tick_params(axis='x', pad=2)
    ax.set_ylim(0, 40)
    ax.set_yticks([10, 20, 30])
    ax.set_yticklabels(['10%', '20%', '30%'], fontsize=5.2, color=COLOR_ALMOND)
    ax.set_rlabel_position(78)
    ax.grid(color=COLOR_ALMOND, linestyle='--', linewidth=0.6, alpha=0.36)
    ax.spines['polar'].set_visible(False)

    ax.set_title("4. Vulnerabilidades subsidiadas:\nSantiago vs Regiones (%)",
                 fontsize=TITLE_FS, fontweight='black', color=COLOR_TEXT, loc='left', pad=2)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.17), ncol=2,
              frameon=False, fontsize=5.7, handlelength=1.4, columnspacing=0.8)

def plot_donut_vulnerability(ax, df):
    """Dona compacta: desglose de vulnerabilidad estructural en viviendas subsidiadas."""
    df_sub = df[df['grupo_vivienda'] == 'Subsidiada'].copy()
    df_sub['trampa_violencia'] = np.where((df_sub['v36b'] >= 3) | (df_sub['v36c'] >= 3), 1, 0)
    df_sub['trampa_economica'] = np.where((df_sub['yautcorh'] < 600000) | (df_sub['subsidy_ratio'] > 0.20), 1, 0)

    def wpct(mask):
        w = df_sub['expr'].sum()
        return (df_sub[mask]['expr'].sum() / w * 100) if w > 0 else 0

    # Porcentajes crudos (las 4 categorias son mutuamente excluyentes y suman 100)
    raw_ambas = wpct((df_sub['trampa_violencia'] == 1) & (df_sub['trampa_economica'] == 1))
    raw_solo_v = wpct((df_sub['trampa_violencia'] == 1) & (df_sub['trampa_economica'] == 0))
    raw_solo_e = wpct((df_sub['trampa_violencia'] == 0) & (df_sub['trampa_economica'] == 1))
    raw_ninguna = max(0.0, 100 - (raw_ambas + raw_solo_v + raw_solo_e))
    raw_fallo = min(100.0, raw_ambas + raw_solo_v + raw_solo_e)

    # Geometria de la dona: valores crudos -> siempre no negativos (evita cuñas corruptas)
    geom_values = [raw_ambas, raw_solo_v, raw_solo_e, raw_ninguna]

    # Etiquetas: valores redondeados ("sin vulnerabilidad" como resto para sumar 100)
    pct_ambas = round(raw_ambas)
    pct_solo_v = round(raw_solo_v)
    pct_solo_e = round(raw_solo_e)
    pct_ninguna = max(0, 100 - (pct_ambas + pct_solo_v + pct_solo_e))
    pct_fallo = min(100, max(0, round(raw_fallo)))

    values = [pct_ambas, pct_solo_v, pct_solo_e, pct_ninguna]
    labels = [
        "Violencia + economía",
        "Entorno violento",
        "Presión económica",
        "Sin vulnerabilidad\nsevera",
    ]
    colors = ["#a94f24", "#c7662f", "#e07b3a", COLOR_RESTO]

    ax.set_title("5. Vulnerabilidad estructural en\nViviendas Subsidiadas (%)",
                 fontsize=TITLE_FS, fontweight='black', color=COLOR_TEXT, loc='left', pad=7)

    ax.pie(
        [raw_fallo, max(0.0, 100 - raw_fallo)],
        colors=["#e07b3a", (0, 0, 0, 0)],
        startangle=94,
        counterclock=False,
        radius=0.88,
        wedgeprops=dict(width=0.018, edgecolor="none"),
    )

    wedges, _ = ax.pie(
        geom_values,
        colors=colors,
        startangle=94,
        counterclock=False,
        radius=0.80,
        wedgeprops=dict(width=0.32, edgecolor=COLOR_BG, linewidth=2.5),
    )
    wedges[-1].set_edgecolor(COLOR_BG)
    wedges[-1].set_linewidth(2.5)

    ax.text(0, 0.08, f"{pct_fallo}%", ha='center', va='center',
            fontsize=18, fontweight='black', color=COLOR_TEXT)
    ax.text(0, -0.15, "Vulnerabilidad\nestructural", ha='center', va='center',
            fontsize=6.8, fontweight='bold', color=COLOR_TEXT, linespacing=1.05)

    label_positions = [
        (0.66, 0.94, "left", 0.46, 0.92),
        (1.02, 0.08, "left", 0.86, 0.08),
        (0.55, -1.14, "center", 0.03, -1.00),
        (-1.08, 0.36, "right", -0.86, 0.36),
    ]
    for wedge, value, label, color, (tx, ty, ha, elbow_x, elbow_y) in zip(wedges, values, labels, colors, label_positions):
        angle_deg = (wedge.theta1 + wedge.theta2) / 2
        angle = np.deg2rad(angle_deg)
        x = np.cos(angle) * (0.88 if not label.startswith("Sin") else 0.88)
        y = np.sin(angle) * (0.88 if not label.startswith("Sin") else 0.88)
        text_color = COLOR_RESTO if label.startswith("Sin") else color
        line_end_x = tx - 0.04 if ha == "left" else tx + 0.04 if ha == "right" else tx
        line_end_y = ty if ha != "center" else ty + 0.10
        ax.plot(
            [x, elbow_x, line_end_x],
            [y, elbow_y, line_end_y],
            color="#8a7b61",
            lw=0.8,
            alpha=0.85,
            solid_capstyle="round",
            zorder=5,
        )
        ax.text(
            tx,
            ty,
            f"{value}%\n{label}",
            ha=ha,
            va="center",
            fontsize=6.5,
            fontweight="black",
            color=text_color,
            linespacing=1.05,
        )

    ax.set_aspect('equal')
    ax.set_xlim(-1.26, 1.26)
    ax.set_ylim(-1.12, 1.12)
    ax.axis('off')

def plot_waffle_vulnerability(ax, df):
    """Lógica de plot_waffle_final.py: Vulnerabilidad compuesta con casitas"""
    df_sub = df[df['grupo_vivienda'] == 'Subsidiada'].copy()
    df_sub['trampa_violencia'] = np.where((df_sub['v36b'] >= 3) | (df_sub['v36c'] >= 3), 1, 0)
    df_sub['trampa_economica'] = np.where((df_sub['yautcorh'] < 600000) | (df_sub['subsidy_ratio'] > 0.20), 1, 0)

    def wpct(mask):
        w = df_sub['expr'].sum()
        return (df_sub[mask]['expr'].sum() / w * 100) if w > 0 else 0

    pct_ambas = round(wpct((df_sub['trampa_violencia']==1) & (df_sub['trampa_economica']==1)))
    pct_solo_v = round(wpct((df_sub['trampa_violencia']==1) & (df_sub['trampa_economica']==0)))
    pct_solo_e = round(wpct((df_sub['trampa_violencia']==0) & (df_sub['trampa_economica']==1)))
    pct_ninguna = 100 - (pct_ambas + pct_solo_v + pct_solo_e)

    colors_list = ([COLOR_TEXT] * pct_ambas + [COLOR_RM] * pct_solo_v + [COLOR_RESTO] * pct_solo_e + ['#8b969e'] * pct_ninguna)
    
    # Espaciar casitas: paso de 1.5 para que respiren bien
    x_coords, y_coords = [], []
    for y in range(10):
        for x in range(10):
            x_coords.append(x * 1.4)
            y_coords.append(y * 1.4)

    # Marcador de casita proporcionado
    verts = [(-0.4, -0.4), (0.4, -0.4), (0.4, 0.2), (0.0, 0.5), (-0.4, 0.2), (-0.4, -0.4)]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    house_path = Path(verts, codes)

    ax.scatter(x_coords, y_coords, s=360, c=colors_list, marker=house_path, alpha=0.9, edgecolors='none', zorder=3)
    ax.text(-0.35, 15.50, "5. Vulnerabilidad estructural en\nViviendas Subsidiadas (%)",
            ha='left', va='top', fontsize=11, fontweight='black', color=COLOR_TEXT)
    
    # Eje de leyenda bien a la derecha para no competir con las casitas
    text_x = 14.6

    # Subtítulo panorama nacional
    ax.text(text_x, 13.6, "Panorama nacional: de cada 100 viviendas subsidiadas", 
            color=COLOR_TEXT, fontsize=9, va='center', style='italic', alpha=0.75)
    ax.plot([text_x, text_x + 7.5], [13.2, 13.2], color=COLOR_TEXT, lw=0.7, alpha=0.2)

    # Desglose (leyenda compacta, bien espaciada)
    for i, (c, lbl, p) in enumerate([
        (COLOR_RM, "Trampa Violencia", pct_solo_v), 
        (COLOR_RESTO, "Trampa Económica", pct_solo_e), 
        (COLOR_TEXT, "Doble Vulnerabilidad", pct_ambas)
    ]):
        ax.scatter(text_x + 0.2, 11.5 - i*2.8, s=220, c=c, marker=house_path)
        ax.text(text_x + 1.0, 11.5 - i*2.8, f"{p}  {lbl}", color=COLOR_TEXT, fontsize=11, va='center', fontweight='bold')

    # Línea separadora
    ax.plot([text_x, text_x + 7.5], [3.8, 3.8], color=COLOR_TEXT, lw=0.8, alpha=0.3)

    # Porcentajes finales MUY abajo — solo etiqueta de cierre, discreta
    pct_fallo = pct_ambas + pct_solo_v + pct_solo_e
    ax.text(text_x + 1.5, 3.25, f"{pct_fallo}%", color=COLOR_TEXT, fontsize=16, fontweight='black', va='top', ha='center')
    ax.text(text_x + 1.5, 2.05, "Vulnerabilidad\nEstructural", color=COLOR_TEXT, fontsize=7.5, fontweight='bold', va='top', ha='center', linespacing=1.2)
    
    ax.text(text_x + 5.5, 3.25, f"{pct_ninguna}%", color='#8b969e', fontsize=16, fontweight='black', va='top', ha='center')
    ax.text(text_x + 5.5, 2.05, "Sin Vulnerabilidad\nSevera", color='#8b969e', fontsize=7.5, fontweight='bold', va='top', ha='center', linespacing=1.2)
    
    ax.set_xlim(-0.4, 23.4)
    ax.set_ylim(-1, 15.35)
    ax.axis('off')

# ─────────────────────────────────────────────────────────
# ENSAMBLAJE FINAL (A4 Gridspec)
# ─────────────────────────────────────────────────────────

def draw_justified_paragraph(ax, text, x0, x1, y_top, fontsize=7.2, color=COLOR_TEXT):
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bbox = ax.get_window_extent(renderer=renderer)
    target_width = (x1 - x0) * ax_bbox.width

    def text_bbox(label):
        probe = ax.text(0, 0, label, fontsize=fontsize, color=color, alpha=0)
        bbox = probe.get_window_extent(renderer=renderer)
        probe.remove()
        return bbox

    def text_width(label):
        return text_bbox(label).width

    words = text.split()
    lines = []
    current = []
    for word in words:
        candidate = current + [word]
        if current and text_width(" ".join(candidate)) > target_width:
            lines.append(current)
            current = [word]
        else:
            current = candidate
    if current:
        lines.append(current)

    line_height = text_bbox("Hg").height * 1.22 / ax_bbox.height
    for line_idx, line_words in enumerate(lines):
        y = y_top - line_idx * line_height
        is_last = line_idx == len(lines) - 1
        if len(line_words) == 1 or is_last:
            ax.text(x0, y, " ".join(line_words), transform=ax.transAxes,
                    ha="left", va="top", fontsize=fontsize, color=color, alpha=0.9)
            continue

        words_width = sum(text_width(word) for word in line_words)
        gap = (target_width - words_width) / (len(line_words) - 1)
        cursor = x0 * ax_bbox.width
        for word in line_words:
            ax.text(cursor / ax_bbox.width, y, word, transform=ax.transAxes,
                    ha="left", va="top", fontsize=fontsize, color=color, alpha=0.9)
            cursor += text_width(word) + gap

def add_story_block(fig):
    """Franja narrativa breve bajo los gráficos."""
    ax = fig.add_axes([0.075, 0.036, 0.85, 0.060])
    ax.axis("off")
    ax.plot([0, 1], [0.98, 0.98], color=COLOR_ALMOND, lw=0.8, alpha=0.38)
    ax.plot([0, 1], [0.02, 0.02], color=COLOR_ALMOND, lw=0.6, alpha=0.22)

    story = (
        "La infografía muestra que el acceso a una vivienda subsidiada no siempre implica superar la vulnerabilidad. "
        "Aunque los beneficiarios DS49 han aumentado en distintos periodos, los hogares en viviendas sociales siguen "
        "concentrando desventajas estructurales: bajos ingresos, dependencia del subsidio, presencia de barrios vulnerables, "
        "problemas de seguridad y condiciones territoriales desiguales entre Santiago y regiones. Así, la política habitacional "
        "logra entregar propiedad, pero no siempre garantiza integración urbana ni mejores oportunidades de vida."
    )
    draw_justified_paragraph(ax, story, x0=0.0, x1=1.0, y_top=0.82, fontsize=7.05)

def add_qr_code(fig, path="datavisualizationQR.png"):
    """Agrega QR discreto en el margen inferior derecho."""
    if not os.path.exists(path):
        return

    qr = plt.imread(path)
    ax = fig.add_axes([0.927, 0.044, 0.065, 0.046])
    ax.imshow(qr, interpolation="nearest")
    ax.axis("off")

def generate_infographic():
    print("Cargando datos...")
    df_master = pd.read_parquet("data/processed/master_dataset.parquet")
    df_hist = pd.read_parquet("data/processed/balaceras_historico_zonas.parquet")
    df_ds49 = prepare_ds49_yoy(read_ds49_region_totals())
    with open("data/raw/regiones.json", encoding="utf-8-sig") as file:
        geojson_regions = json.load(file)
    
    # Ajustar márgenes para darle más espacio útil y alinear los paneles
    fig = plt.figure(figsize=(8.27, 11.69), facecolor=COLOR_BG)
    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[0.06, 0.94],
        hspace=0.03,
        left=0.055, right=0.955, top=0.965, bottom=0.105
    )
    gs = gridspec.GridSpecFromSubplotSpec(
        3, 3,
        subplot_spec=outer[1],
        width_ratios=[0.66, 1, 1],
        height_ratios=[0.32, 0.31, 0.37],
        hspace=0.32,
        wspace=0.22,
    )
    
    # 1. HEADER
    ax_head = fig.add_subplot(outer[0])
    ax_head.axis('off')
    ax_head.text(0.5, 0.78, "VIVIENDAS SOCIALES EN CHILE", fontsize=20, fontweight='black', ha='center', color=COLOR_TEXT)
    
    # 2. PLOT 1 (MAPA GEO - COLUMNA 1 COMPLETA)
    ax1 = fig.add_subplot(gs[:, 0])
    plot_regional_vulnerability_map(ax1, df_master, geojson_regions)
    # Heatmap anterior, dejado como alternativa:
    # ax1 = fig.add_subplot(gs[0, 1:])
    # plot_quintile_heatmap(ax1, df_master)
    
    # 3. PLOT 2 (CRECIMIENTO DS49 - FILA 1, COLUMNAS 2-3)
    ax2 = fig.add_subplot(gs[0, 1:])
    plot_ds49_yoy_growth(ax2, df_ds49)
    
    # 4. PLOT 3 (LINE CHART - FILA 2, COLUMNA 2)
    ax3 = fig.add_subplot(gs[1, 1])
    plot_historical_line(ax3, df_hist)

    # 5. PLOT 4 (RADAR - FILA 2, COLUMNA 3)
    ax4 = fig.add_subplot(gs[1, 2], projection='polar')
    plot_radar_rm_vs_regions(ax4, df_master)
    
    # 6. PLOT 5 (DONA - FILA 3, COLUMNA 2)
    ax5 = fig.add_subplot(gs[2, 1])
    plot_donut_vulnerability(ax5, df_master)

    # 7. BUTTERFLY CHART - FILA 3, COLUMNA 3
    ax6 = fig.add_subplot(gs[2, 2])
    plot_subsidy_butterfly(ax6, df_master)

    add_story_block(fig)
    add_qr_code(fig)
    
    # Footer consolidado (Leyenda, Temporalidad y N) - Rúbrica
    fig.text(0.5, 0.024, "Fuente: Elaboración propia con CASEN 2024 (N=77.858 filas), histórico CASEN (2015-2024) y registros MINVU DS49 (2013-2025) · Universidad de Concepción", 
             ha='center', fontsize=7.4, color=COLOR_TEXT, fontweight='bold')
    fig.text(0.5, 0.010, "DS49: Fondo Solidario de Elección de Vivienda, subsidio habitacional para familias sin vivienda.", 
             ha='center', fontsize=7.0, color=COLOR_ALMOND)



    # Guardar
    output_path = "deliverables/infografia_final_maquina_dependencia.pdf"
    os.makedirs("deliverables", exist_ok=True)
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight', facecolor=COLOR_BG)
    plt.savefig(output_path.replace(".pdf", ".png"), format='png', dpi=300, bbox_inches='tight', facecolor=COLOR_BG)
    
    print(f"Infografía generada exitosamente en: {output_path}")

if __name__ == "__main__":
    generate_infographic()
