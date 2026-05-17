import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from matplotlib.path import Path
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
                 fontsize=11, fontweight='black', color=COLOR_TEXT, loc='left', pad=44)
    
    # Etiquetas Y mismo tamaño, discretas
    ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold', fontsize=9)
    ax.tick_params(axis='y', length=0, pad=8)

def plot_butterfly_gaps(ax, df):
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
    row_gap = 1.28
    bar_h = 0.68 # Barras más gruesas
    
    for i, row in plot_df.iterrows():
        y = i * row_gap
        ax.barh(y, -row['RE'], color=COLOR_RESTO, height=bar_h, zorder=3)
        ax.barh(y, row['RM'], color=COLOR_RM, height=bar_h, zorder=3)
        
        ax.text(0, y + 0.46, row['Variable'].upper(), ha='center', va='bottom', fontsize=8, fontweight='bold', color=COLOR_TEXT)
        ax.text(-row['RE'] - 2, y, f"{row['RE']:.0f}%", ha='right', va='center', color=COLOR_RESTO, fontsize=11, fontweight='black')
        ax.text(row['RM'] + 2, y, f"{row['RM']:.0f}%", ha='left', va='center', color=COLOR_RM, fontsize=11, fontweight='black')

    top_y = (len(vars_to_plot) - 1) * row_gap
    ax.vlines(0, ymin=-0.45, ymax=top_y + 0.82, color=COLOR_TEXT, lw=1.5, alpha=0.5)
    
    max_val = max(plot_df['RM'].max(), plot_df['RE'].max())
    lim = max_val + 9
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-1.25, top_y + 2.85)
    ax.set_yticks([])
    ax.axis('off')
    
    ax.text(-lim * 0.55 +8, top_y + 1.12, 'REGIONES',
        ha='center', color=COLOR_RESTO, fontsize=12, fontweight='black')

    ax.text(lim * 0.55, top_y + 1.12, 'SANTIAGO (RM)',
        ha='center', color=COLOR_RM, fontsize=12, fontweight='black')
    # Subtítulo indicando qué población se compara
    ax.text(-lim * 0.7, top_y + 2.72, "2. Brechas territoriales en\nViviendas Subsidiadas (%)",
            ha='left', va='top', fontsize=11, fontweight='black', color=COLOR_TEXT)

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
                 fontsize=11, fontweight='black', color=COLOR_TEXT, loc='left', pad=8)

    
    ax.legend(loc='upper left', frameon=False, facecolor='none', edgecolor='none', fontsize=9)

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
    ax.text(-0.35, 15.50, "4. Vulnerabilidad estructural en\nViviendas Subsidiadas (%)",
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

def generate_infographic():
    print("Cargando datos...")
    df_master = pd.read_parquet("data/processed/master_dataset.parquet")
    df_hist = pd.read_parquet("data/processed/balaceras_historico_zonas.parquet")
    
    # Ajustar márgenes para darle más espacio útil y alinear los paneles
    fig = plt.figure(figsize=(8.27, 11.69), facecolor=COLOR_BG)
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[0.07, 0.28, 0.40, 0.31], 
                           hspace=0.24, wspace=0.18, left=0.055, right=0.955, top=0.965, bottom=0.045)
    
    # 1. HEADER
    ax_head = fig.add_subplot(gs[0, :])
    ax_head.axis('off')
    ax_head.text(0.5, 0.7, "VIVIENDAS SOCIALES EN CHILE", fontsize=20, fontweight='black', ha='center', color=COLOR_TEXT)
    
    # 2. PLOT 1 (HEATMAP)
    ax1 = fig.add_subplot(gs[1, :])
    plot_quintile_heatmap(ax1, df_master)
    
    # 3. PLOT 2 (BUTTERFLY CHART - IZQUIERDA)
    ax2 = fig.add_subplot(gs[2, 0])
    plot_butterfly_gaps(ax2, df_master)
    
    pos = ax2.get_position()
    ax2.set_position([pos.x0-0.06,
                      pos.y0 - 0.04, 
                      pos.width , 
                      pos.height + 0.08])  
    
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

if __name__ == "__main__":
    generate_infographic()
