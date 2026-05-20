import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT_DIR / "data" / "raw" / "SUDS49Mar2026.xlsx"
OUTPUT_PATH = ROOT_DIR / "plots_alone" / "ds49_yoy_growth_rm_vs_regiones.png"

COLOR_BG = "#f8f4e3"
COLOR_TEXT = "#4c2e05"
COLOR_RM = "#f19143"
COLOR_RESTO = "#3c4f76"
COLOR_ALMOND = "#d38b5d"


def read_ds49_region_totals(path=INPUT_PATH):
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


def prepare_rm_vs_regions_yoy(totals):
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


def plot_yoy_growth(data, output_path=OUTPUT_PATH):
    os.makedirs(output_path.parent, exist_ok=True)

    sns.set_theme(
        style="whitegrid",
        rc={
            "axes.facecolor": COLOR_BG,
            "figure.facecolor": COLOR_BG,
            "grid.color": COLOR_ALMOND,
            "grid.alpha": 0.28,
            "axes.edgecolor": COLOR_ALMOND,
            "font.family": "sans-serif",
            "font.sans-serif": ["Roboto", "Arial", "DejaVu Sans"],
        },
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.1), facecolor=COLOR_BG)
    palette = {"Santiago (RM)": COLOR_RM, "Regiones": COLOR_RESTO}

    sns.lineplot(
        data=data,
        x="year",
        y="yoy_pct",
        hue="zona",
        palette=palette,
        marker="o",
        markersize=8,
        linewidth=3,
        ax=ax,
    )

    ax.axhline(0, color=COLOR_TEXT, linewidth=1.1, alpha=0.55)
    ax.fill_between(
        data["year"].sort_values().unique(),
        0,
        ax.get_ylim()[1],
        color=COLOR_RM,
        alpha=0.035,
        zorder=0,
    )

    ax.set_title(
        "Beneficiados DS49: crecimiento interanual",
        loc="left",
        fontsize=17,
        fontweight="black",
        color=COLOR_TEXT,
        pad=18,
    )
    ax.text(
        0,
        1.015,
        "Santiago (RM) vs regiones, 2013-2025",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        color=COLOR_TEXT,
        alpha=0.78,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Crecimiento interanual (%)", fontsize=10, color=COLOR_TEXT)
    ax.set_xticks(sorted(data["year"].unique()))
    ax.set_xticklabels([str(int(y)) for y in sorted(data["year"].unique())], rotation=0)

    ax.yaxis.grid(True, linestyle="--")
    ax.xaxis.grid(False)
    ax.tick_params(axis="both", colors=COLOR_TEXT, labelsize=9)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(COLOR_ALMOND)
    ax.spines["bottom"].set_alpha(0.5)

    label_years = [2017, 2021, 2023, 2025]
    for _, row in data[data["year"].isin(label_years)].iterrows():
        color = palette[row["zona"]]
        offset = 9 if row["yoy_pct"] >= 0 else -12
        va = "bottom" if row["yoy_pct"] >= 0 else "top"
        ax.text(
            row["year"],
            row["yoy_pct"] + offset,
            f"{row['yoy_pct']:+.0f}%",
            ha="center",
            va=va,
            fontsize=8.5,
            fontweight="black",
            color=color,
        )

    legend = ax.legend(
        title=None,
        loc="upper left",
        frameon=False,
        fontsize=10.5,
        bbox_to_anchor=(0.01, 0.93),
    )
    for text in legend.get_texts():
        text.set_color(COLOR_TEXT)

    ax.set_ylim(-85, 210)
    ax.text(
        0.99,
        -0.14,
        "Fuente: MINVU, DS49. 2026 excluido por ser dato a marzo.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=COLOR_ALMOND,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close(fig)


def main():
    totals = read_ds49_region_totals()
    data = prepare_rm_vs_regions_yoy(totals)
    plot_yoy_growth(data)
    print(f"Saved: {OUTPUT_PATH}")
    print(data.pivot(index="year", columns="zona", values="yoy_pct").round(1).to_string())


if __name__ == "__main__":
    main()
