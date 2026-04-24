import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from scipy.stats import gaussian_kde

# ─────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL Y COLORES (Rubrica: Estética Premium)
# ─────────────────────────────────────────────────────────
COLOR_BG     = '#f8f4e3'       # Ivory Mist
COLOR_TEXT   = '#4c2e05'       # Deep Walnut
COLOR_RM     = '#f19143'       # Sandy Brown (Naranja)
COLOR_RESTO  = '#3c4f76'       # Dusk Blue (Azul)
COLOR_ALMOND = '#d38b5d'       # Toasted Almond 
COLOR_WARM   = '#d35400'       # Terracota

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

# ─────────────────────────────────────────────────────────
# COMPONENTES DE GRÁFICOS
# ─────────────────────────────────────────────────────────

def plot_quintile_heatmap(ax, df):
    """Lógica de Grfico_1.py: Heatmap de concentración de pobreza"""
    df_h = df.dropna(subset=['yautcorh', 'expr']).copy()
    quantiles = [0.2, 0.4, 0.6, 0.8]
    cortes = weighted_quantiles(df_h['yautcorh'], df_h['expr'], quantiles)
    
    bins = [-np.inf] + list(cortes) + [np.inf]
    labels = ['Q1\n(Menores Ingresos)', 'Q2', 'Q3', 'Q4', 'Q5\n(Mayores Ingresos)']
    df_h['Quintil'] = pd.cut(df_h['yautcorh'], bins=bins, labels=labels, include_lowest=True)
    
    df_h['grupo_vivienda_label'] = np.where(df_h['v15'] == 1, 'Vivienda con Subsidio', 'Vivienda Sin Subsidio')
    
    grouped = df_h.groupby(['grupo_vivienda_label', 'Quintil'], observed=False)['expr'].sum().unstack()
    props = grouped.div(grouped.sum(axis=1), axis=0) * 100
    props = props.loc[['Vivienda con Subsidio', 'Vivienda Sin Subsidio']]
    
    colors = [COLOR_BG, COLOR_RM, COLOR_TEXT]
    custom_cmap = sns.blend_palette(colors, as_cmap=True)
    
    sns.heatmap(props, cmap=custom_cmap, annot=True, fmt=".1f", linewidths=4, linecolor=COLOR_BG, 
                vmin=0, vmax=35, cbar=False, ax=ax)
    
    for t in ax.texts:
        val = float(t.get_text())
        t.set_text(f'{val:.1f}%')
        t.set_fontsize(11)
        t.set_color('white')
        t.set_fontweight('bold')
        t.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='black')])
            
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', length=0, labelsize=10, pad=15)
    ax.tick_params(axis='y', length=0, labelsize=10, pad=10)

def plot_historical_line(ax, df_hist):
    """Lógica de Grafico3.py: Evolución de balaceras"""
    plot_data = df_hist.groupby(['year', 'zona']).apply(
        lambda x: np.average(x['is_balacera'], weights=x['ponderador']) * 100
    ).unstack()
    
    years = plot_data.index
    ax.fill_between(years, plot_data['Resto de Regiones'], plot_data['RM (Santiago)'], 
                    color=COLOR_RM, alpha=0.08, zorder=1)
    
    for year in years:
        val_rm = plot_data.loc[year, 'RM (Santiago)']
        val_resto = plot_data.loc[year, 'Resto de Regiones']
        ax.plot([year, year], [val_resto, val_rm], color=COLOR_RM, linestyle='--', linewidth=1, alpha=0.4, zorder=2)
        
        ax.text(year, val_rm + 2, f"{int(val_rm)}%", color=COLOR_RM, fontweight='bold', ha='center', fontsize=10)
        ax.text(year, val_resto - 4, f"{int(val_resto)}%", color=COLOR_RESTO, fontweight='bold', ha='center', fontsize=10, va='top')
        
        mid_y = (val_rm + val_resto) / 2
        ax.text(year + 0.1, mid_y, f"+{int(val_rm - val_resto)}%", color=COLOR_WARM, fontweight='black', 
                ha='left', va='center', fontsize=8, alpha=0.6)

    ax.plot(years, plot_data['RM (Santiago)'], marker='o', color=COLOR_RM, linewidth=3, markersize=8, label='Santiago (RM)', zorder=3)
    ax.plot(years, plot_data['Resto de Regiones'], marker='o', color=COLOR_RESTO, linewidth=3, markersize=8, label='Regiones', zorder=3)
    
    # Subtítulo decorativo
    ax.text(0, 1.15, "/ Evolución y brecha porcentual de balaceras", transform=ax.transAxes, 
            fontsize=9, color=COLOR_ALMOND, style='italic', va='bottom')
    
    ax.set_xticks(years)
    ax.set_xticklabels([str(int(y)) for y in years], fontsize=10)
    ax.set_ylim(0, 65)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color=COLOR_ALMOND)
    thin_spines(ax, ['bottom'])
    ax.legend(loc='lower center', frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02), ncol=2)

def plot_butterfly_gaps(ax, df):
    """Lógica de plot_diverging_bar.py: Comparación RM vs Regiones"""
    # Flags
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
    bar_h = 0.5
    
    for i, row in plot_df.iterrows():
        ax.barh(i, -row['RE'], color=COLOR_RESTO, height=bar_h, zorder=3)
        ax.barh(i, row['RM'], color=COLOR_RM, height=bar_h, zorder=3)
        
        ax.text(0, i + 0.35, row['Variable'].upper(), ha='center', va='bottom', fontsize=8, fontweight='bold', color=COLOR_TEXT)
        ax.text(-row['RE'] - 2, i, f"{row['RE']:.0f}%", ha='right', va='center', color=COLOR_RESTO, fontsize=11, fontweight='black')
        ax.text(row['RM'] + 2, i, f"{row['RM']:.0f}%", ha='left', va='center', color=COLOR_RM, fontsize=11, fontweight='black')

    ax.axvline(0, color=COLOR_TEXT, lw=1.5, alpha=0.5)
    
    ax.set_xlim(-85, 85)
    ax.set_yticks([])
    ax.axis('off')
    
    # Etiquetas de bando (Movidas más arriba para evitar solapamiento)
    ax.text(-40, len(vars_to_plot) + 0.1, 'REGIONES', ha='center', color=COLOR_RESTO, fontsize=10, fontweight='black')
    ax.text(40, len(vars_to_plot) + 0.1, 'SANTIAGO (RM)', ha='center', color=COLOR_RM, fontsize=10, fontweight='black')

def plot_waffle_vulnerability(ax, df):
    """Lógica de plot_waffle_final.py: Vulnerabilidad compuesta"""
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

    colors_list = ([COLOR_TEXT] * pct_ambas + [COLOR_RM] * pct_solo_v + [COLOR_RESTO] * pct_solo_e + ['#d1ccc0'] * pct_ninguna)
    x_coords, y_coords = [], []
    for y in range(10):
        for x in range(10):
            x_coords.append(x); y_coords.append(y)

    ax.scatter(x_coords, y_coords, s=200, c=colors_list, marker='o', alpha=0.9, edgecolors='none', zorder=3)
    
    text_x = 11.5
    ax.text(text_x, 8.5, f"{pct_ambas + pct_solo_v + pct_solo_e}%", color=COLOR_TEXT, fontsize=38, fontweight='black', va='center')
    ax.text(text_x + 5.5, 8.5, "VULNERABILIDAD\nESTRUCTURAL", color=COLOR_TEXT, fontsize=11, fontweight='bold', va='center')
    
    for i, (c, lbl, p) in enumerate([(COLOR_RM, "Trampa de Violencia", pct_solo_v), (COLOR_RESTO, "Trampa Económica", pct_solo_e), (COLOR_TEXT, "Doble Vulnerabilidad", pct_ambas)]):
        ax.scatter(text_x + 0.2, 6.2 - i*1.2, s=80, c=c, marker='o')
        ax.text(text_x + 0.8, 6.2 - i*1.2, f"{p}% {lbl}", color=COLOR_TEXT, fontsize=10, va='center')

    ax.text(text_x, 1.0, f"{pct_ninguna}%", color='#d1ccc0', fontsize=28, fontweight='black', va='center')
    ax.text(text_x + 4.5, 1.0, "SIN VULNERABILIDAD\nSEVERA", color='#d1ccc0', fontsize=9, fontweight='bold', va='center')
    
    ax.set_xlim(-1, 24)
    ax.set_ylim(-1, 10)
    ax.axis('off')

# ─────────────────────────────────────────────────────────
# ENSAMBLAJE FINAL (A4 Gridspec)
# ─────────────────────────────────────────────────────────

def generate_infographic():
    print("Cargando datos...")
    df_master = pd.read_parquet("data/processed/master_dataset.parquet")
    df_hist = pd.read_parquet("data/processed/balaceras_historico_zonas.parquet")
    
    # Configuración de la figura A4 (8.27 x 11.69 pulgadas)
    fig = plt.figure(figsize=(8.27, 11.69), facecolor=COLOR_BG)
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[0.08, 0.25, 0.35, 0.25], 
                           hspace=0.6, wspace=0.4, left=0.12, right=0.88, top=0.92, bottom=0.08)
    
    # 1. HEADER
    ax_head = fig.add_subplot(gs[0, :])
    ax_head.axis('off')
    ax_head.text(0.5, 0.9, "LA MÁQUINA DE LA DEPENDENCIA", fontsize=28, fontweight='black', ha='center', color=COLOR_TEXT)
    ax_head.text(0.5, 0.1, "Cómo la política habitacional chilena transforma el sueño de la casa propia en una trampa de segregación y estancamiento.", 
                 fontsize=11, ha='center', color=COLOR_TEXT, style='italic')
    
    # 2. PLOT 1 (HEATMAP)
    ax1 = fig.add_subplot(gs[1, :])
    plot_quintile_heatmap(ax1, df_master)
    
    # 3. PLOT 2 (BUTTERFLY CHART - IZQUIERDA)
    ax2 = fig.add_subplot(gs[2, 0])
    plot_butterfly_gaps(ax2, df_master)
    
    # 4. PLOT 3 (LINE CHART - DERECHA)
    ax3 = fig.add_subplot(gs[2, 1])
    plot_historical_line(ax3, df_hist)
    
    # 5. PLOT 4 (WAFFLE CHART)
    ax4 = fig.add_subplot(gs[3, :])
    plot_waffle_vulnerability(ax4, df_master)
    
    # Línea decorativa inferior
    fig.text(0.5, 0.02, "Fuente: Elaboración propia basada en CASEN 2024 y Censo 2024 · Universidad de Concepción", 
             ha='center', fontsize=8, color=COLOR_ALMOND)

    # Guardar
    output_path = "deliverables/infografia_final_maquina_dependencia.pdf"
    os.makedirs("deliverables", exist_ok=True)
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight', facecolor=COLOR_BG)
    plt.savefig(output_path.replace(".pdf", ".png"), format='png', dpi=300, bbox_inches='tight', facecolor=COLOR_BG)
    
    print(f"Infografía generada exitosamente en: {output_path}")
    plt.show()

if __name__ == "__main__":
    generate_infographic()
