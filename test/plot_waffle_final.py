import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.path import Path

def generate_senior_waffle(df):
    print("Calculando vulnerabilidad nacional compuesta...")

    # -- 1. FILTRO Y BANDERAS --
    df_sub = df[df['grupo_vivienda'] == 'Subsidiada'].copy()

    df_sub['trampa_violencia'] = np.where((df_sub['v36b'] >= 3) | (df_sub['v36c'] >= 3), 1, 0)
    df_sub['trampa_economica'] = np.where((df_sub['yautcorh'] < 600000) | (df_sub['subsidy_ratio'] > 0.20), 1, 0)

    def wpct(mask):
        w = df_sub['expr'].sum()
        return (df_sub[mask]['expr'].sum() / w * 100) if w > 0 else 0

    pct_ambas = round(wpct((df_sub['trampa_violencia']==1) & (df_sub['trampa_economica']==1)))
    pct_solo_v = round(wpct((df_sub['trampa_violencia']==1) & (df_sub['trampa_economica']==0)))
    pct_solo_e = round(wpct((df_sub['trampa_violencia']==0) & (df_sub['trampa_economica']==1)))
    pct_ninguna = round(wpct((df_sub['trampa_violencia']==0) & (df_sub['trampa_economica']==0)))

    # Ajuste fino por redondeo
    total = pct_ambas + pct_solo_v + pct_solo_e + pct_ninguna
    if total != 100:
        pct_ninguna += (100 - total)

    # -- 2. COLORES MINIMALISTAS --
    c_deep_walnut   = '#4c2e05' # Ambas
    c_ivory_mist    = '#f8f4e3' # Fondo
    c_dusk_blue     = '#3c4f76' # Económica
    c_sandy_brown   = '#f19143' # Violencia
    c_empty         = '#8b969e' # Ninguna (Gris oscuro más visible)

    # -- 3. CREAR MARCADOR DE CASITA PERSONALIZADO --
    verts = [
        (-0.4, -0.5), # left bottom
        (0.4, -0.5),  # right bottom
        (0.4, 0.2),   # right top wall
        (0.0, 0.6),   # roof peak
        (-0.4, 0.2),  # left top wall
        (-0.4, -0.5)  # close
    ]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    house_path = Path(verts, codes)

    # -- 4. FIGURA --
    fig, ax = plt.subplots(figsize=(15, 8), dpi=200, facecolor=c_ivory_mist)
    ax.set_facecolor(c_ivory_mist)

    # -- 5. CREAR LA MATRIZ DE PUNTOS --
    colors_list = (
        [c_deep_walnut] * pct_ambas +
        [c_sandy_brown] * pct_solo_v +
        [c_dusk_blue] * pct_solo_e +
        [c_empty] * pct_ninguna
    )

    x_coords = []
    y_coords = []
    
    # Llenar de abajo hacia arriba
    for y in range(10):
        for x in range(10):
            x_coords.append(x)
            y_coords.append(y)

    # Dibujar usando el marcador de casita gigante
    ax.scatter(x_coords, y_coords, s=900, c=colors_list, marker=house_path, alpha=0.9, edgecolors='none', zorder=3)

    # -- 6. LEYENDA TIPOGRÁFICA Y AGREGADOS --
    text_x = 11.5
    
    # Indicador de Nivel Nacional
    ax.text(text_x, 9.8, "PANORAMA NACIONAL: VIVIENDAS SUBSIDIADAS", color=c_deep_walnut, fontsize=16, fontweight='black', va='center', alpha=0.8)
    ax.text(text_x, 9.2, "De cada 100 hogares que reciben una vivienda social en Chile:", color=c_deep_walnut, fontsize=14, style='italic', va='center', alpha=0.8)
    ax.plot([text_x, text_x + 9], [8.7, 8.7], color=c_deep_walnut, lw=1.5, alpha=0.2)

    # LEYENDA (Desglose gigante arriba)
    y_v = 7.5
    y_e = 5.5
    y_a = 3.5
    
    # Violencia
    ax.scatter(text_x + 0.3, y_v, s=500, c=c_sandy_brown, marker=house_path)
    ax.text(text_x + 1.2, y_v, f"{pct_solo_v}  Trampa de Violencia", color=c_deep_walnut, fontsize=21, fontweight='black', va='center')
    ax.text(text_x + 1.2, y_v - 0.6, "Expuestos frecuentemente a narcotráfico o balaceras.", color=c_deep_walnut, fontsize=11, fontweight='normal', va='center', alpha=0.8)

    # Economía
    ax.scatter(text_x + 0.3, y_e, s=500, c=c_dusk_blue, marker=house_path)
    ax.text(text_x + 1.2, y_e, f"{pct_solo_e}  Trampa Económica", color=c_deep_walnut, fontsize=21, fontweight='black', va='center')
    ax.text(text_x + 1.2, y_e - 0.6, "Incapacidad autónoma y altísima dependencia estatal.", color=c_deep_walnut, fontsize=11, fontweight='normal', va='center', alpha=0.8)

    # Ambas
    ax.scatter(text_x + 0.3, y_a, s=500, c=c_deep_walnut, marker=house_path)
    ax.text(text_x + 1.2, y_a, f"{pct_ambas}  Doble Vulnerabilidad", color=c_deep_walnut, fontsize=21, fontweight='black', va='center')
    ax.text(text_x + 1.2, y_a - 0.6, "Sufren extrema violencia y pobreza simultáneamente.", color=c_deep_walnut, fontsize=11, fontweight='normal', va='center', alpha=0.8)

    # Línea separadora sutil
    ax.plot([text_x, text_x + 9], [2.0, 2.0], color=c_deep_walnut, lw=0.8, alpha=0.3)

    # PORCENTAJES AGREGADOS (Esquina inferior derecha, LADO A LADO)
    pct_fallo = pct_ambas + pct_solo_v + pct_solo_e
    
    y_agg = 0.5
    # Fallo (64%) a la izquierda
    ax.text(text_x, y_agg, f"{pct_fallo}%", color=c_deep_walnut, fontsize=32, fontweight='black', va='center')
    ax.text(text_x + 1.8, y_agg, "VULNERABILIDAD\nESTRUCTURAL", color=c_deep_walnut, fontsize=12, fontweight='bold', va='center', linespacing=1.2)
    
    # Éxito (36%) a la derecha
    ax.text(text_x + 5.5, y_agg, f"{pct_ninguna}%", color=c_empty, fontsize=32, fontweight='black', va='center')
    ax.text(text_x + 7.3, y_agg, "SIN VULNERABILIDAD\nSEVERA", color=c_empty, fontsize=12, fontweight='bold', va='center', linespacing=1.2)

    # -- 7. LIMPIEZA DE EJES --
    ax.set_xlim(-1, 22)
    ax.set_ylim(-1, 10.5)
    ax.axis('off')

    plt.tight_layout()
    output_path = 'test/outputs/waffle_final_senior.png'
    os.makedirs('test/outputs', exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=200, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Guardado en: {output_path}")

if __name__ == '__main__':
    df_master = pd.read_parquet('data/processed/master_dataset.parquet')
    generate_senior_waffle(df_master)
