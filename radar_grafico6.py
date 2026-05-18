import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_radar_plot():
    # 1. Configuración de estilo exigida
    COLOR_BG = '#FDFBF7'
    COLOR_SUBSIDIO = '#D65A04'
    COLOR_NOSUBSIDIO = '#4A6572'
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Roboto', 'Arial', 'DejaVu Sans']
    
    try:
        df = pd.read_parquet('data/processed/master_dataset.parquet')
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        return
        
    # 2. Extraer exactamente 6 variables (Precariedad de Vivienda y Entorno)
    df['dim_multidim'] = np.where(df['pobreza_multi'] == 1, 1, 0)
    df['dim_hacinamiento'] = np.where(df['ind_hacina'] >= 2, 1, 0)
    df['dim_robos'] = np.where(df['v36e'] >= 3, 1, 0)
    df['dim_drogas'] = np.where(df['v36c'] >= 3, 1, 0)
    df['dim_alumbrado'] = np.where(df['v35a'] == 2, 1, 0)
    df['dim_basura'] = np.where(df['v35c'] == 2, 1, 0)
    
    dimensions = ['dim_multidim', 'dim_hacinamiento', 'dim_robos', 'dim_drogas', 'dim_alumbrado', 'dim_basura']
    labels = ['Pobreza\nMultidimensional', 'Hacinamiento\nMedio/Crítico', 'Robos en\nel Entorno', 
              'Tráfico de\nDrogas', 'Falta de\nAlumbrado', 'Microbasurales\nen el Entorno']
    
    # Calcular porcentajes reales ponderados
    def calculate_weighted_pct(data, dim):
        w = data['expr'].sum()
        if w == 0: return 0
        return (data[data[dim] == 1]['expr'].sum() / w) * 100
        
    df_sub = df[df['grupo_vivienda'] == 'Subsidiada']
    df_nosub = df[df['grupo_vivienda'] == 'No Subsidiada / Otro']
    
    vals_sub = [calculate_weighted_pct(df_sub, d) for d in dimensions]
    vals_nosub = [calculate_weighted_pct(df_nosub, d) for d in dimensions]
    
    # 3. Geometría del Radar Plot (Proyección Polar)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    
    # Cerrar los polígonos
    vals_sub += vals_sub[:1]
    vals_nosub += vals_nosub[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    
    # Ajuste de inicio angular a las 12 en punto
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1) 
    
    # Dibujar Sin Subsidio (Fondo protector)
    ax.plot(angles, vals_nosub, color=COLOR_NOSUBSIDIO, linewidth=2.5, linestyle='solid', 
            marker='o', markersize=6, label='Sin Subsidio (Mercado Libre)')
    ax.fill(angles, vals_nosub, color=COLOR_NOSUBSIDIO, alpha=0.25)
    
    # Dibujar Subsidio (Frente protagonista)
    ax.plot(angles, vals_sub, color=COLOR_SUBSIDIO, linewidth=2.5, linestyle='solid', 
            marker='o', markersize=6, label='Vivienda Subsidiada')
    ax.fill(angles, vals_sub, color=COLOR_SUBSIDIO, alpha=0.25)
    
    # 4. Estética de los ejes y grilla (Anti-Ugly Premium)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold', color='#333333')
    
    # Separar aún más las etiquetas del centro radial para que respiren bien
    ax.tick_params(axis='x', pad=35)
    
    # Ocultar el anillo exterior espantoso de matplotlib
    ax.spines['polar'].set_visible(False)
    
    # Grilla más amigable al ojo
    ax.grid(color='#E0DCD5', linestyle='--', linewidth=1)
    
    # Limite Radial Fijo para que nada se salga (Crítico)
    ax.set_ylim(0, 45)
    
    # Aros concéntricos limpios (10%, 20%, 30%, 40%) en gris muy tenue
    steps = [10, 20, 30, 40]
    ax.set_yticks(steps)
    ax.set_yticklabels([f"{s}%" for s in steps], color='#C2C2C2', fontsize=8)
    
    # 5. Títulos y Leyenda
    ax.set_title("Precariedad de Vida y Entorno Urbano:\nSubsidio Habitacional vs. Mercado Libre", 
                 fontsize=14, fontweight='black', color='#333333', pad=40)
                 
    # Leyenda compacta en la esquina superior derecha
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), frameon=False, 
              fontsize=10, labelcolor='#333333')
    
    output = "radar_grafico6_rediseñado.png"
    plt.savefig(output, dpi=300, bbox_inches='tight', facecolor=COLOR_BG)
    print(f"¡Radar rediseñado guardado en: {output}!")
    plt.show()

if __name__ == '__main__':
    create_radar_plot()
