import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

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
    c_empty         = '#d1ccc0' # Ninguna

    # -- 3. FIGURA --
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200, facecolor=c_ivory_mist)
    ax.set_facecolor(c_ivory_mist)

    # -- 4. CREAR LA MATRIZ DE PUNTOS --
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

    # Usar puntos (scatter circles) para un look más "data science" moderno en vez de cuadrados gruesos
    ax.scatter(x_coords, y_coords, s=250, c=colors_list, marker='o', alpha=0.9, edgecolors='none', zorder=3)

    # -- 5. LEYENDA TIPOGRÁFICA (TUFTE STYLE) --
    text_x = 11.5
    
    # Bloque de Fallo (64%)
    pct_fallo = pct_ambas + pct_solo_v + pct_solo_e
    ax.text(text_x, 8.5, f"{pct_fallo}%", color=c_deep_walnut, fontsize=42, fontweight='black', va='center')
    ax.text(text_x + 3.2, 8.5, "VULNERABILIDAD\nESTRUCTURAL", color=c_deep_walnut, fontsize=12, fontweight='bold', va='center', linespacing=1.2)
    
    # Desglose minimalista
    ax.scatter(text_x + 0.2, 6.5, s=100, c=c_sandy_brown, marker='o')
    ax.text(text_x + 0.8, 6.5, f"{pct_solo_v}% Trampa de Violencia", color=c_deep_walnut, fontsize=11, fontweight='normal', va='center')
    
    ax.scatter(text_x + 0.2, 5.0, s=100, c=c_dusk_blue, marker='o')
    ax.text(text_x + 0.8, 5.0, f"{pct_solo_e}% Trampa Económica", color=c_deep_walnut, fontsize=11, fontweight='normal', va='center')

    ax.scatter(text_x + 0.2, 3.5, s=100, c=c_deep_walnut, marker='o')
    ax.text(text_x + 0.8, 3.5, f"{pct_ambas}% Doble Vulnerabilidad", color=c_deep_walnut, fontsize=11, fontweight='normal', va='center')

    # Línea separadora sutil
    ax.plot([text_x, text_x + 6], [2.0, 2.0], color=c_deep_walnut, lw=0.5, alpha=0.3)

    # Bloque de Éxito (36%)
    ax.text(text_x, 0.5, f"{pct_ninguna}%", color=c_empty, fontsize=32, fontweight='black', va='center')
    ax.text(text_x + 2.8, 0.5, "SIN VULNERABILIDAD\nSEVERA", color=c_empty, fontsize=10, fontweight='bold', va='center', linespacing=1.2)

    # -- 6. LIMPIEZA DE EJES --
    ax.set_xlim(-1, 18)
    ax.set_ylim(-1, 10)
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
