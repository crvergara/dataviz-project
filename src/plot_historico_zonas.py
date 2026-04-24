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
    COLOR_ALMOND = '#d38b5d'   # Toasted Almond (para elementos secundarios)
    COLOR_GRID = '#d38b5d33'   # Toasted Almond con alpha
    
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
    
    # Dibujar líneas
    years = plot_data.index
    ax.plot(years, plot_data['RM (Santiago)'], marker='o', color=COLOR_RM, 
            linewidth=4, markersize=10, label='Santiago (RM)', zorder=3)
    ax.plot(years, plot_data['Resto de Regiones'], marker='o', color=COLOR_RESTO, 
            linewidth=4, markersize=10, label='Regiones', zorder=3)
    
    # 4. Pulido Estético (High Data-to-Ink Ratio)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color(COLOR_ALMOND)
    
    # Grid horizontal sutil
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, color=COLOR_ALMOND, zorder=1)
    ax.xaxis.grid(False)
    
    # Configurar ejes
    ax.set_xticks(years)
    ax.set_xticklabels(years, fontsize=11, color=COLOR_TEXT, fontweight='bold')
    
    # Eje Y con símbolos de %
    ax.set_yticks(np.arange(0, 45, 10))
    ax.set_yticklabels([f"{int(y)}%" for y in np.arange(0, 45, 10)], 
                       fontsize=10, color=COLOR_TEXT)
    
    # Títulos y Etiquetas (Narrativa)
    plt.title("LA BRECHA DE LA INSEGURIDAD EN VIVIENDA SUBSIDIADA", 
              fontsize=16, fontweight='black', pad=25, loc='left', color=COLOR_TEXT)
    
    # Subtítulo explicativo
    ax.text(years[0], 43, "/ Evolución del % de hogares que reportan balaceras frecuentes en su entorno", 
            fontsize=11, color=COLOR_ALMOND, style='italic')
    
    # Etiquetado Directo (Evitar leyenda tradicional para limpieza)
    # Etiqueta para RM
    ax.text(years[-1] + 0.1, plot_data['RM (Santiago)'].iloc[-1], 'Santiago (RM)', 
            color=COLOR_RM, fontweight='bold', va='center', fontsize=12)
    # Etiqueta para Resto
    ax.text(years[-1] + 0.1, plot_data['Resto de Regiones'].iloc[-1], 'Regiones', 
            color=COLOR_RESTO, fontweight='bold', va='center', fontsize=12)
    
    # Ajustar límites para las etiquetas de texto
    ax.set_xlim(years[0], years[-1] + 0.8)
    ax.set_ylim(0, 45)
    
    # Anotación de contexto (Insight)
    delta_2024 = plot_data['RM (Santiago)'].iloc[-1] - plot_data['Resto de Regiones'].iloc[-1]
    ax.annotate(f'Brecha de de {delta_2024:.1f} pp', 
                xy=(2024, (plot_data['RM (Santiago)'].iloc[-1] + plot_data['Resto de Regiones'].iloc[-1])/2),
                xytext=(2021.5, 25),
                arrowprops=dict(arrowstyle='<->', color=COLOR_ALMOND),
                fontsize=10, color=COLOR_TEXT, ha='center')

    plt.tight_layout()
    
    # Guardar deliverable
    out_img = r"deliverables\historico_balaceras_zonas.png"
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    plt.savefig(out_img, bbox_inches='tight', facecolor=COLOR_BG)
    print(f"Gráfico guardado en: {out_img}")
    plt.show()

if __name__ == "__main__":
    create_historical_line_chart()
