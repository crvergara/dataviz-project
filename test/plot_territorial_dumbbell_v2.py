import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

def generate_territorial_dumbbell_v2(df):
    print("Preparando datos...")

    # -- 1. FLAGS --
    df['Viaje Critico (>1h)']          = np.where(df['o28a_hr'] >= 1, 1, 0)
    df['Alta Dependencia (>20%)']      = np.where(df['subsidy_ratio'] > 0.20, 1, 0)
    df['Ingreso Autonomo <$600k']      = np.where(df['yautcorh'] < 600000, 1, 0)
    df['Entorno con Delincuencia']     = np.where(df['v36e'] == 1, 1, 0)
    df['Pobreza Multidimensional']     = np.where(df['pobreza_multi'] == 1, 1, 0)
    df['Sin Educacion Basica (<8 años)'] = np.where(df['e6a_no_asiste'] < 8, 1, 0)

    # Bloques narrativos: 4 peor en Regiones, 2 peor en RM
    variables_regio = [
        'Entorno con Delincuencia',
        'Ingreso Autonomo <$600k',
        'Alta Dependencia (>20%)',
        'Sin Educacion Basica (<8 años)',
    ]
    variables_metro = [
        'Viaje Critico (>1h)',
        'Pobreza Multidimensional',
    ]
    all_vars = variables_regio + variables_metro

    # -- 2. CALCULO --
    df_rm_s = df[(df['grupo_vivienda'] == 'Subsidiada') & (df['region'] == 13)]
    df_re_s = df[(df['grupo_vivienda'] == 'Subsidiada') & (df['region'] != 13)]

    def wpct(sub, var):
        w = sub['expr'].sum()
        return (sub[sub[var] == 1]['expr'].sum() / w * 100) if w > 0 else 0

    rows = []
    for var in all_vars:
        rows.append({
            'Variable': var,
            'Grupo': 'metro' if var in variables_metro else 'regio',
            'RM_Sub':  wpct(df_rm_s, var),
            'Re_Sub':  wpct(df_re_s, var),
        })
    plot_df = pd.DataFrame(rows)
    plot_df['brecha'] = plot_df['RM_Sub'] - plot_df['Re_Sub']

    # Ordenar dentro de cada bloque por magnitud de brecha
    pf_re = plot_df[plot_df['Grupo']=='regio'].sort_values('brecha', ascending=False)
    pf_me = plot_df[plot_df['Grupo']=='metro'].sort_values('brecha', ascending=True)
    plot_df = pd.concat([pf_re, pf_me]).reset_index(drop=True)

    print(plot_df[['Variable','RM_Sub','Re_Sub','brecha']].round(1).to_string())

    # -- 3. POSICIONES Y (hueco entre bloques) --
    y_pos, cy, last = [], 0, plot_df.iloc[0]['Grupo']
    for _, row in plot_df.iterrows():
        if row['Grupo'] != last:
            cy += 1.4   # espacio visual entre los dos bloques
            last = row['Grupo']
        y_pos.append(cy)
        cy += 1.0
    y_pos = np.array(y_pos, dtype=float)

    # -- 4. COLORES --
    c_rm  = '#C0392B'
    c_re  = '#2471A3'

    vmax = max(abs(plot_df['brecha'].min()), abs(plot_df['brecha'].max())) + 1e-9
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = mcolors.LinearSegmentedColormap.from_list('rw', [c_re, '#D5D8DC', c_rm])

    # -- 5. FIGURA --
    fig, ax = plt.subplots(figsize=(11, 8), dpi=150, facecolor='white')
    ax.set_facecolor('white')

    xlim_r = 82

    # Bandas de fondo para los dos bloques
    y_re_arr = y_pos[plot_df['Grupo'].values == 'regio']
    y_me_arr = y_pos[plot_df['Grupo'].values == 'metro']

    ax.axhspan(y_re_arr.min() - 0.55, y_re_arr.max() + 0.55,
               color='#EBF5FB', alpha=0.55, zorder=0)
    ax.axhspan(y_me_arr.min() - 0.55, y_me_arr.max() + 0.55,
               color='#FDEDEC', alpha=0.55, zorder=0)

    # Etiquetas de bloque dentro de la banda (margen derecho)
    ax.text(xlim_r - 0.5, np.mean(y_re_arr),
            'Friccion Social\n(Peor en Regiones)',
            color=c_re, fontsize=9, fontweight='bold', va='center', ha='right',
            bbox=dict(facecolor='#EBF5FB', edgecolor='none', pad=2), zorder=5)
    ax.text(xlim_r - 0.5, np.mean(y_me_arr),
            'Friccion Espacial\n(Peor en RM)',
            color=c_rm, fontsize=9, fontweight='bold', va='center', ha='right',
            bbox=dict(facecolor='#FDEDEC', edgecolor='none', pad=2), zorder=5)

    # Dibujar cada fila
    for i, (_, row) in enumerate(plot_df.iterrows()):
        yp = y_pos[i]
        rm, re = row['RM_Sub'], row['Re_Sub']
        lc = cmap(norm(row['brecha']))

        # Linea Dumbbell
        ax.plot([rm, re], [yp, yp], color=lc, lw=5,
                solid_capstyle='round', zorder=3, alpha=0.88)

        # Puntos
        ax.scatter(rm, yp, color=c_rm, s=175, zorder=4,
                   edgecolor='white', lw=1.2,
                   label='Subsidiada RM (Santiago)' if i == 0 else '')
        ax.scatter(re, yp, color=c_re, s=175, zorder=4,
                   edgecolor='white', lw=1.2,
                   label='Subsidiada Regiones' if i == 0 else '')

        # Valores encima de la linea
        ax.text(rm, yp + 0.28, f'{rm:.0f}%', ha='center', va='bottom',
                color=c_rm, fontsize=9, fontweight='bold', zorder=5)
        ax.text(re, yp + 0.28, f'{re:.0f}%', ha='center', va='bottom',
                color=c_re, fontsize=9, fontweight='bold', zorder=5)

    # -- 6. EJES Y DECORACION --
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['Variable'], fontsize=10.5, color='#2C3E50')
    ax.set_xlim(0, xlim_r)
    ax.set_ylim(y_pos[0] - 0.65, y_pos[-1] + 0.75)

    ax.set_xlabel('Hogares Afectados (%)', fontsize=10, color='#7F8C8D', labelpad=10)

    sns.despine(ax=ax, left=True, bottom=False, top=True, right=True)
    ax.grid(axis='x', ls='--', color='#EAECEE', lw=1)
    ax.tick_params(axis='y', length=0)
    ax.set_axisbelow(True)

    # Titulo centrado
    fig.suptitle(
        'Vulnerabilidades en la vivienda social en Chile\nRegiones vs Santiago',
        fontsize=14, fontweight='bold', color='#2C3E50', x=0.5, ha='center')

    # Leyenda debajo del grafico
    leg_handles = [
        mpatches.Patch(color=c_rm, label='Subsidiada RM (Santiago)'),
        mpatches.Patch(color=c_re, label='Subsidiada Regiones'),
    ]
    ax.legend(handles=leg_handles, loc='upper center',
              bbox_to_anchor=(0.5, -0.09), ncol=2,
              frameon=True, facecolor='white', edgecolor='#EAECEE', fontsize=9)

    plt.tight_layout()
    output_path = 'test/outputs/territorial_dumbbell_v2.png'
    os.makedirs('test/outputs', exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Guardado en: {output_path}")

if __name__ == '__main__':
    df_master = pd.read_parquet('data/processed/master_dataset.parquet')
    generate_territorial_dumbbell_v2(df_master)
