import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def create_historical_line_chart():
    # 1. Configuración de Estilo "Information Design"
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Roboto', 'Arial', 'DejaVu Sans']
    
    # Paleta de colores personalizada
    COLOR_BG = '#f8f4e3'       # Ivory Mist
    COLOR_TEXT = '#4c2e05'     # Deep Walnut
    COLOR_RM = '#f19143'       # Sandy Brown (Orange)
    COLOR_RESTO = '#3c4f76'    # Dusk Blue (Blue)
    COLOR_ALMOND = '#d38b5d'   # Toasted Almond 
    COLOR_WARM = '#d35400'     # Terracota Cálido para las brechas
    
    # 2. Carga de datos
    df_path = r"data\processed\balaceras_historico_zonas.parquet"
    if not os.path.exists(df_path):
        print(f"Error: No se encuentra {df_path}")
        return

    df = pd.read_parquet(df_path)
    
    # Calcular promedio ponderado por año y zona
    plot_data = df.groupby(['year', 'zona']).apply(
        lambda x: np.average(x['is_balacera'], weights=x['ponderador']) * 100
    ).unstack()
    
    # 3. Creación del Gráfico
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    
    years = plot_data.index

    # Área sombreada para la brecha
    ax.fill_between(years, 
                    plot_data['Resto de Regiones'], 
                    plot_data['RM (Santiago)'], 
                    color=COLOR_RM, alpha=0.08, zorder=1)
    
    # Líneas interlineadas verticales
    for year in years:
        val_rm = plot_data.loc[year, 'RM (Santiago)']
        val_resto = plot_data.loc[year, 'Resto de Regiones']
        
        ax.plot([year, year], [val_resto, val_rm], 
                color=COLOR_RM, linestyle='--', linewidth=1.5, alpha=0.6, zorder=2)
    
    # Dibujar líneas principales
    ax.plot(years, plot_data['RM (Santiago)'], marker='o', color=COLOR_RM, 
            linewidth=4, markersize=10, label='Santiago (RM)', zorder=3)
    ax.plot(years, plot_data['Resto de Regiones'], marker='o', color=COLOR_RESTO, 
            linewidth=4, markersize=10, label='Regiones', zorder=3)
    
    # --- ETIQUETAS DE DATOS Y BRECHAS ---
    for year in years:
        val_rm = plot_data.loc[year, 'RM (Santiago)']
        val_resto = plot_data.loc[year, 'Resto de Regiones']
        gap = val_rm - val_resto 
        
        # Etiqueta RM (Tamaño aumentado a 13)
        ax.text(year, val_rm + 1.8, f"{int(val_rm)}%", 
                color=COLOR_RM, fontweight='bold', ha='center', va='bottom', fontsize=13)
        
        # Etiqueta Regiones (Tamaño aumentado a 13)
        ax.text(year, val_resto - 2.2, f"{int(val_resto)}%", 
                color=COLOR_RESTO, fontweight='bold', ha='center', va='top', fontsize=13)

        # Texto de la Brecha (Color cálido, MÁS PEQUEÑO Y TRANSPARENTE)
        mid_y = (val_rm + val_resto) / 2
        # Sumamos 0.12 a "year" para moverlo a la derecha y usamos ha='left'
        ax.text(year + 0.12, mid_y, f"+{int(gap)}%", 
                color=COLOR_WARM, fontweight='black', ha='left', va='center', 
                fontsize=9, zorder=5, alpha=0.5) # <-- AQUÍ ESTÁN LOS CAMBIOS (fontsize=7, alpha=0.6)

    # 4. Pulido Estético
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color(COLOR_ALMOND)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, color=COLOR_ALMOND, zorder=0)
    ax.xaxis.grid(False)
    
    ax.set_xticks(years)
    ax.set_xticklabels([str(int(y)) for y in years], fontsize=12, color=COLOR_TEXT, fontweight='bold')
    
    ax.set_yticks(np.arange(0, 61, 10))
    ax.set_yticklabels([f"{int(y)}%" for y in np.arange(0, 61, 10)], 
                       fontsize=10, color=COLOR_TEXT)
    ax.tick_params(axis='both', length=0, pad=8)
    
    # Títulos
    plt.title("LA BRECHA ESTRUCTURAL DE LA INSEGURIDAD", 
              fontsize=16, fontweight='black', pad=30, loc='left', color=COLOR_TEXT)
    
    ax.text(years[0] - 0.1, 58, "/ Evolución y brecha porcentual (+%) de balaceras frecuentes por zona", 
            fontsize=11, color=COLOR_ALMOND, style='italic')
    
    # Leyenda
    ax.legend(loc='upper right', frameon=False, fontsize=11, labelcolor=COLOR_TEXT, 
              bbox_to_anchor=(1, 1.08))
    
    # Ajustar límites (Ampliamos un poco el derecho para dar espacio a los números corridos)
    ax.set_xlim(years[0] - 0.25, years[-1] + 0.4)
    ax.set_ylim(0, 60)
    
    plt.tight_layout()
    
    # Guardar deliverable
    out_img = r"deliverables\historico_balaceras_zonas_brecha_es.png"
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    plt.savefig(out_img, bbox_inches='tight', facecolor=COLOR_BG)
    print(f"Gráfico guardado en: {out_img}")
    plt.show()

if __name__ == "__main__":
    create_historical_line_chart()