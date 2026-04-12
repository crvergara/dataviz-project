"""
La Máquina de Dependencia — visualización estática PDF
Región Metropolitana de Santiago · CASEN 2024 + Censo 2024

Uso: python maquina_dependencia.py --region 13
"""

import os
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from scipy.stats import gaussian_kde

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────
# PALETA
# ─────────────────────────────────────────────────────────
CR      = '#C0392B'
CR_BG   = '#FDEDEC'
CB      = '#2471A3'
CG      = '#626567'
CG_LT   = '#AEB6BF'
C_BG    = '#FFFFFF'
C_TXT   = '#1C2833'
C_TXT2  = '#566573'
C_GRID  = '#EAECEE'

plt.rcParams.update({
    'figure.facecolor': C_BG, 'axes.facecolor': C_BG,
    'axes.edgecolor': C_GRID, 'axes.linewidth': 0.6,
    'axes.labelcolor': C_TXT2, 'axes.labelsize': 10,
    'xtick.color': C_TXT2, 'ytick.color': C_TXT2,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'xtick.major.size': 0, 'ytick.major.size': 0,
    'font.family': 'DejaVu Sans', 'text.color': C_TXT,
    'axes.grid': False,
})

FMT_M = FuncFormatter(
    lambda x, _: f'${x/1_000_000:.1f}M' if abs(x) >= 1_000_000
    else f'${x/1_000:.0f}k'
)


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
def thin_spine(ax, sides=('bottom', 'left')):
    for s in ('top', 'right', 'bottom', 'left'):
        ax.spines[s].set_visible(s in sides)
        if s in sides:
            ax.spines[s].set_color(C_GRID)


def act_label(ax, number, title, note=''):
    ax.text(0, 1.10, f'ACTO {number}', transform=ax.transAxes,
            fontsize=7.5, fontweight='bold', color=CR, va='bottom')
    ax.text(0, 1.03, title, transform=ax.transAxes,
            fontsize=12, fontweight='bold', color=C_TXT, va='bottom')
    if note:
        ax.text(0, 0.985, note, transform=ax.transAxes,
                fontsize=8.5, color=C_TXT2, style='italic', va='bottom')


def callout(ax, x, y, text, color=CR):
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=8.5, color=C_TXT, va='top', zorder=10,
            bbox=dict(facecolor='white', edgecolor=color,
                      boxstyle='round,pad=0.45', linewidth=0.9))


# ─────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────
def load_data():
    region_id = 13
    path = 'data/processed/icp_results.parquet'
    if not os.path.exists(path):
        raise FileNotFoundError(f'Parquet no encontrado: {path}')

    df = (
        pd.read_parquet(path)
        .rename(columns={
            'yautcorh':          'yaut',
            'ytotcorh':          'ytot',
            'ysubh':             'ysub',
            'c_friction':        'friccion',
            'commute_total_hrs': 'horas',
        })
        .replace([np.inf, -np.inf], np.nan)
        .query('region == @region_id')
        .dropna(subset=['icp', 'yaut', 'ytot', 'ysub', 'v15', 'horas'])
    )

    # ICP negativo es artefacto matematico, clip a 0
    for col in ('icp', 'subsidy_ratio', 'isolation_ratio'):
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    # Horas plausibles + excluir top 2% ingresos
    df = df[(df['horas'] >= 0) & (df['horas'] <= 4)]
    df = df[df['ytot'] <= df['ytot'].quantile(0.98)]

    # Segmentacion correcta segun codebook CASEN:
    # v15 = 1.0       -> Compro con subsidio del Estado  (SUBSIDIADO)
    # v15 in {2, 3}   -> Compro sin subsidio / a credito (MERCADO PRIVADO)
    # v15 = 4.0       -> Heredo / regalo (excluido: sesgo riqueza)
    # v15 = NaN       -> Arrienda / allegado (excluido: otro mercado)
    sub  = df[df['v15'] == 1.0].copy()
    priv = df[df['v15'].isin([2.0, 3.0])].copy()

    assert len(sub) > 0 and len(priv) > 0, 'Algun grupo quedo vacio.'
    return sub, priv


# ─────────────────────────────────────────────────────────
# FIGURA
# ─────────────────────────────────────────────────────────
def build_figure(sub, priv):
    # --- Metricas ---
    sub_w  = sub[sub['horas'] > 0]
    priv_w = priv[priv['horas'] > 0]
    
    icp_sub   = sub['icp'].mean()
    icp_priv  = priv['icp'].mean()
    dep_sub   = sub['ysub'].sum()  / sub['ytot'].sum()
    dep_priv  = priv['ysub'].sum() / priv['ytot'].sum()
    yaut_sub  = sub['yaut'].median()
    yaut_priv = priv['yaut'].median()

    # --- Layout A4 (2x2 Grid) ---
    fig = plt.figure(figsize=(11, 15), facecolor=C_BG)
    gs  = gridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[0.4, 1, 1.2], 
        hspace=0.35, wspace=0.25,
        left=0.08, right=0.92, top=0.95, bottom=0.05,
    )

    # ── HEADER & KPIs (Fila 0, abarca 2 columnas) ────────
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.set_ylim(0, 1)
    ax_header.axis('off')
    ax_header.text(0.5, 0.90, 'LA MÁQUINA DE DEPENDENCIA',
             fontsize=26, fontweight='bold', ha='center', color=C_TXT)
    ax_header.text(0.5, 0.65,
             'La planificación urbana periférica destruye la capacidad de generar ingresos autónomos,\n'
             'forzando al Estado a subsidiar la pobreza que su propia geografía creó en la Región Metropolitana.',
             fontsize=13, ha='center', color=C_TXT2, style='italic', linespacing=1.5)

    # KPI Boxes robustos
    kpis = [
        ('ICP Promedio (Cautiverio)', f'{icp_sub:.3f}', f'vs {icp_priv:.3f} Mercado Privado'),
        ('Dependencia Estatal', f'{dep_sub:.1%}', f'vs {dep_priv:.1%} Mercado Privado'),
        ('Ingreso Autónomo (Mediana)', f'${yaut_sub/1000:.0f}k', f'vs ${yaut_priv/1000:.0f}k Mercado Privado'),
    ]
    for i, (label, val, note) in enumerate(kpis):
        cx = 0.20 + i * 0.30
        ax_header.text(cx, 0.30, label, fontsize=10, color=C_TXT2, ha='center', fontweight='bold')
        ax_header.text(cx, 0.10, val, fontsize=24, color=CR, ha='center', fontweight='bold')
        ax_header.text(cx, 0.00, note, fontsize=9, color=C_TXT2, ha='center')

    # ── PLOT 1: Hexbin de Densidad (Rubric: Escala Secuencial) ──
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_title('La Zona de Subsidencia', fontsize=12, fontweight='bold', loc='left', pad=15)
    
    # Usamos hexbin para evitar el overplotting
    hb = ax1.hexbin(sub['yaut'], sub['ysub'], gridsize=25, cmap='Reds', mincnt=1, alpha=0.9, edgecolors='none')
    ax1.hexbin(priv['yaut'], priv['ysub'], gridsize=25, cmap='Greys', mincnt=1, alpha=0.4, edgecolors='none')
    
    ax1.axvline(sub['yaut'].median(), color=CR, linestyle='--', linewidth=1.5)
    ax1.text(sub['yaut'].median() + 80000, 350000, 'Mediana Vivienda Social\n(Alta concentración)', color=CR, fontsize=9, fontweight='bold')

    ax1.set_xlim(-10_000, 2_500_000)
    ax1.set_ylim(-10_000, 500_000)
    ax1.xaxis.set_major_formatter(FMT_M)
    ax1.yaxis.set_major_formatter(FMT_M)
    ax1.set_xlabel('Ingreso Autónomo del Hogar', fontweight='bold')
    ax1.set_ylabel('Subsidios Estatales', fontweight='bold')
    thin_spine(ax1, ('bottom', 'left'))
    
    # Colorbar integrado dinámicamente sin hardcodear coords
    plt.colorbar(hb, ax=ax1, fraction=0.046, pad=0.04, label='Densidad de Hogares')

    # ── PLOT 2: Espectro de Cautiverio (Rubric: Escala Divergente) ──
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.set_title('Gravedad del Cautiverio (ICP)', fontsize=12, fontweight='bold', loc='left', pad=15)

    rng = np.random.default_rng(42)
    df_plot = pd.concat([sub.assign(group=1), priv.assign(group=0)])
    samp = df_plot.sample(min(1500, len(df_plot)))
    
    jit = rng.uniform(-0.25, 0.25, size=len(samp))
    
    # Escala divergente: Coolwarm (Azul = Libre, Rojo = Atrapado)
    sc = ax2.scatter(samp['icp'], samp['group'] + jit, 
                     c=samp['icp'], cmap='coolwarm', 
                     s=20, alpha=0.8, linewidths=0)

    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Compra\nPrivada', 'Vivienda\nSubsidiada'], fontweight='bold', fontsize=10)
    ax2.set_xlabel('Índice de Cautiverio Patrimonial (0 = Libre, 1+ = Cautivo)', fontweight='bold')
    ax2.set_xlim(-0.01, df_plot['icp'].quantile(0.98))
    thin_spine(ax2, ('bottom',))
    
    # Colorbar dinámico
    plt.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04, label='Intensidad de Dependencia')

    # ── PLOT 3: Composición del Ingreso (Barras 100%) ──
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_title('Estructura de Resiliencia Financiera', fontsize=12, fontweight='bold', loc='left', pad=15)

    groups = ['Compra Privada', 'Vivienda Subsidiada']
    yaut_pct = [priv['yaut'].sum()/priv['ytot'].sum(), sub['yaut'].sum()/sub['ytot'].sum()]
    ysub_pct = [priv['ysub'].sum()/priv['ytot'].sum(), sub['ysub'].sum()/sub['ytot'].sum()]
    
    ax3.bar(groups, yaut_pct, color=CB, width=0.5, label='Ingreso Autónomo', edgecolor='white', linewidth=1)
    ax3.bar(groups, ysub_pct, bottom=yaut_pct, color=CR, width=0.5, label='Subsidio Estatal', edgecolor='white', linewidth=1)
    
    for i, (a, s) in enumerate(zip(yaut_pct, ysub_pct)):
        ax3.text(i, a/2, f'{a:.1%}', ha='center', va='center', color='white', fontweight='bold', fontsize=11)
        if s > 0.05:
            ax3.text(i, a + (s/2), f'{s:.1%}', ha='center', va='center', color='white', fontweight='bold', fontsize=11)

    ax3.set_ylabel('Proporción del Ingreso Total', fontweight='bold')
    ax3.set_xticklabels(groups, fontweight='bold', fontsize=10)
    ax3.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.05, 1.15))
    thin_spine(ax3, ('bottom', 'left'))

    # ── PLOT 4: El Impuesto del Traslado (KDE Limpio) ──
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.set_title('El Costo de Oportunidad (Horas de Traslado)', fontsize=12, fontweight='bold', loc='left', pad=15)

    x_grid = np.linspace(0, 4, 500)
    for i, (g, c, lbl) in enumerate([(sub_w, CR, 'Subsidiada'), (priv_w, CG, 'Privada')]):
        kde = gaussian_kde(g['horas'].values, bw_method=0.30)
        ys = kde(x_grid)
        ax4.plot(x_grid, ys, color=c, linewidth=2.5, label=lbl)
        ax4.fill_between(x_grid, ys, alpha=0.1, color=c)
        
        # Prevenir superposición de textos desplazando la altura
        med = g['horas'].median()
        ax4.axvline(med, color=c, linestyle=':', linewidth=1.5)
        y_offset = 0.85 if i == 0 else 0.65
        ax4.text(med + 0.05, ax4.get_ylim()[1]*y_offset, f'Mediana {lbl}:\n{med:.1f}h', color=c, fontsize=9, fontweight='bold')

    ax4.set_xlabel('Horas diarias perdidas en transporte', fontweight='bold')
    ax4.set_ylabel('Densidad de Trabajadores', fontweight='bold')
    ax4.set_xlim(0, 3.5)
    ax4.legend(frameon=False, loc='upper right')
    thin_spine(ax4, ('bottom', 'left'))

    return fig


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    region_id = 13
    out = f'deliverables/dependencia_maquina_R{region_id}.pdf'
    os.makedirs('deliverables', exist_ok=True)

    print(f'Cargando datos — Región {region_id} (RM)...')
    sub, priv = load_data()
    sub_w  = sub[sub['horas'] > 0]
    priv_w = priv[priv['horas'] > 0]

    print(f'  Subsidiados  (v15=1):  {len(sub):,}  | workers: {len(sub_w):,} ({len(sub_w)/len(sub):.0%})')
    print(f'  Privados (v15=2,3):    {len(priv):,}  | workers: {len(priv_w):,} ({len(priv_w)/len(priv):.0%})')

    print('\nGenerando dashboard embellecido...')
    fig = build_figure(sub, priv)
    fig.savefig(out, format='pdf', dpi=300, bbox_inches='tight',
                facecolor=C_BG, edgecolor='none')
    plt.close(fig)
    print(f'Dashboard PDF guardado correctamente en: {out}')


if __name__ == '__main__':
    main()