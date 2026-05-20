import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
def get_project_root():
    """Encuentra la ruta raíz del proyecto dinámicamente."""
    # Asume que el script puede estar en la raíz o en una subcarpeta (ej. src)
    current_dir = Path(__file__).resolve().parent
    if (current_dir / 'data').exists() and (current_dir / 'deliverables').exists():
        return current_dir
    else:
        # Si está en una subcarpeta (ej. src/), subimos un nivel
        return current_dir.parent
def generate_vertical_bar_chart(df):
    print("Calculando valores porcentuales ponderados...")
    
    # -- 1. FLAGS --
    df['Balaceras Frecuentes']         = np.where(df['v36b'] >= 3, 1, 0)
    df['Narco-tráfico Frecuente']      = np.where(df['v36c'] >= 3, 1, 0)
    df['Alta Dependencia (>20%)']      = np.where(df['subsidy_ratio'] > 0.20, 1, 0)
    df['Ingreso Autónomo <$600k']      = np.where(df['yautcorh'] < 600000, 1, 0)
    df['Pobreza por Ingresos']         = np.where(df['pobreza'].isin([1, 2]), 1, 0)
    
    # Orden lógico de las variables (Agrupamos entorno vs ingresos)
    all_vars = [
        'Balaceras Frecuentes',
        'Narco-tráfico Frecuente',
        'Alta Dependencia (>20%)',
        'Ingreso Autónomo <$600k',
        'Pobreza por Ingresos'
    ]
    
    # Nombres más cortos y con saltos de línea para el eje X
    labels = [
        'Balaceras\nFrecuentes',
        'Narcotráfico\nFrecuente',
        'Alta\nDependencia',
        'Ingreso Autónomo\n<$600k',
        'Pobreza por\nIngresos'
    ]
    
    # -- 2. CALCULO --
    df_sub = df[df['is_subsidized'] == 1]
    df_nosub = df[df['is_subsidized'] == 0]
    
    def wpct(subset, var):
        w = subset['expr'].sum()
        if w == 0: return 0
        return (subset[subset[var] == 1]['expr'].sum() / w) * 100
        
    nosub_vals = [wpct(df_nosub, var) for var in all_vars]
    sub_vals = [wpct(df_sub, var) for var in all_vars]
    
    # -- 3. COLORES Y ESTILO --
    c_deep_walnut   = '#4c2e05'
    c_ivory_mist    = '#f8f4e3'
    c_dusk_blue     = '#3c4f76' # Sin Subsidio
    c_sandy_brown   = '#f19143' # Con Subsidio
    c_grid          = '#e0dcd3'
    
    # -- 4. FIGURA --
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=200, facecolor=c_ivory_mist)
    ax.set_facecolor(c_ivory_mist)
    
    x = np.arange(len(labels))
    width = 0.35  
    
    # -- 5. BARRAS VERTICALES --
    bars_nosub = ax.bar(x - width/2, nosub_vals, width, label='Sin Subsidio', color=c_dusk_blue, zorder=3)
    bars_sub = ax.bar(x + width/2, sub_vals, width, label='Con Subsidio', color=c_sandy_brown, zorder=3)
    
    # -- 6. ANOTACIONES EN BARRAS --
    def add_labels(rects, color_text):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.0f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 puntos de offset vertical
                        textcoords="offset points",
                        ha='center', va='bottom', color=color_text, 
                        fontsize=11, fontweight='bold')
    add_labels(bars_nosub, c_dusk_blue)
    add_labels(bars_sub, c_sandy_brown)
    
    # -- 7. EJES, CUADRÍCULA Y ESTILO LIMPIO --
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold', color=c_deep_walnut)
    ax.set_yticks([]) # Ocultamos el eje Y para que los números en las barras hablen por sí solos
    
    # Remover bordes (spines) excepto la base
    for spine in ax.spines.values():
        if spine != ax.spines['bottom']:
            spine.set_visible(False)
        else:
            spine.set_color(c_deep_walnut)
            spine.set_linewidth(1.5)
            
    # Líneas de grilla horizontales sutiles por detrás de las barras
    ax.grid(axis='y', color=c_grid, linestyle='-', linewidth=0.7, zorder=0, alpha=0.6)
    
    # Límite Y para dar espacio a los números y títulos
    ax.set_ylim(0, max(max(nosub_vals), max(sub_vals)) + 15)
    
    # Leyenda limpia en la parte superior derecha
    ax.legend(loc='upper right', frameon=False, fontsize=12, labelcolor=c_deep_walnut)
    
    # -- 8. TÍTULOS --
    fig.suptitle('6. Comparativa Nacional:\n viviendas con subsidio v/s viviendas sin subsidio',
                 fontsize=18, fontweight='black', color=c_deep_walnut, y=0.98, ha='left', x=0.05)
    
    plt.figtext(0.05, 0.85, 
                '"El desafío del subsidio: Entorno hostil y dependencia económica."',
                fontsize=12, color=c_deep_walnut, ha='left', alpha=0.9)
                
    plt.subplots_adjust(top=0.78, bottom=0.1, left=0.05, right=0.95)
    
    # -- 9. GUARDADO FINAL --
    project_root = get_project_root()
    out_dir = project_root / 'deliverables'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = out_dir / 'vertical_bar_sub_vs_nosub.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=200, facecolor=fig.get_facecolor())
    plt.close()
    print(f"¡Gráfico de Barras Verticales guardado exitosamente en: {output_path}!")
if __name__ == '__main__':
    project_root = get_project_root()
    data_path = project_root / 'data' / 'processed' / 'master_dataset.parquet'
    
    try:
        df_master = pd.read_parquet(data_path)
        generate_vertical_bar_chart(df_master)
    except Exception as e:
        print(f"Error cargando los datos: {e}")