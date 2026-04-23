import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def generate_territorial_dumbbell(df):
    """
    Genera un Dumbbell plot comparando RM vs Resto de Chile 
    en viviendas subsidiadas.
    """
    # 1. PREPARACIÓN DE DATOS
    df_sub = df[df['is_subsidized'] == 1].copy()
    df_sub['territorio'] = np.where(df_sub['region'] == 13, 'RM', 'Resto')
    
    # Flags
    df_sub['Viaje Crítico (>1h)'] = np.where(df_sub['o28a_hr'] >= 1, 1, 0)
    df_sub['Pobreza Multidimensional'] = np.where(df_sub['pobreza_multi'] == 1, 1, 0)
    df_sub['Alta Dependencia Estatal (>20%)'] = np.where(df_sub['subsidy_ratio'] > 0.20, 1, 0)
    df_sub['Ingreso Autónomo < $600k'] = np.where(df_sub['yautcorh'] < 600000, 1, 0)
    df_sub['Entorno con Delincuencia'] = np.where(df_sub['v36e'] == 1, 1, 0)

    variables = ['Entorno con Delincuencia', 'Ingreso Autónomo < $600k', 
                 'Alta Dependencia Estatal (>20%)', 'Pobreza Multidimensional', 'Viaje Crítico (>1h)']
    
    # Calcular porcentajes
    results = []
    for var in variables:
        for terr in ['RM', 'Resto']:
            subset = df_sub[df_sub['territorio'] == terr]
            if subset['expr'].sum() > 0:
                pct = (subset[subset[var] == 1]['expr'].sum() / subset['expr'].sum()) * 100
                results.append({'Variable': var, 'Territorio': terr, 'Valor': pct})
                
    plot_df = pd.DataFrame(results).pivot(index='Variable', columns='Territorio', values='Valor')
    plot_df['Brecha'] = plot_df['RM'] - plot_df['Resto']
    plot_df = plot_df.sort_values(by='Brecha', ascending=True)

    # 2. CONFIGURACIÓN VISUAL (Dumbbell Plot)
    plt.rcParams['font.family'] = 'sans-serif'
    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)

    # Colores
    color_rm = '#D73027'     # Rojo (Foco Urbano)
    color_resto = '#4575B4'  # Azul (Foco Regional)
    color_line = '#B0B0B0'   # Gris neutro para la conexión

    y_pos = np.arange(len(plot_df))

    # Dibujar líneas conectoras (Dumbbells)
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        ax.plot([row['RM'], row['Resto']], [i, i], color=color_line, linewidth=2, zorder=1)

    # Dibujar puntos
    ax.scatter(plot_df['RM'], y_pos, color=color_rm, s=120, label='Metropolitana', zorder=2, edgecolor='white')
    ax.scatter(plot_df['Resto'], y_pos, color=color_resto, s=120, label='Resto de Chile', zorder=2, edgecolor='white')

    # 3. ESTÉTICA "Data-to-Ink"
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df.index, fontsize=10, fontweight='medium')
    ax.set_xlim(0, max(plot_df.max(numeric_only=True)) + 10)
    
    ax.set_title("Dos Caras del Fracaso: La Vivienda Social en RM vs Regiones", 
                 fontsize=14, fontweight='bold', loc='left', pad=20)
    ax.set_xlabel("Hogares Afectados (%)", fontsize=10, color='#555555')
    
    sns.despine(left=True, bottom=False, top=True, right=True)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.tick_params(axis='y', length=0)

    # Anotar valores directamente en los puntos
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        rm_val = row['RM']
        resto_val = row['Resto']
        
        # Ajustar posición del texto para que no se superponga si están muy cerca
        offset_rm = -3 if rm_val < resto_val else 3
        offset_resto = 3 if rm_val < resto_val else -3
        
        align_rm = 'right' if rm_val < resto_val else 'left'
        align_resto = 'left' if rm_val < resto_val else 'right'

        ax.text(rm_val + offset_rm, i, f"{rm_val:.1f}%", va='center', ha=align_rm, color=color_rm, fontweight='bold', fontsize=9)
        ax.text(resto_val + offset_resto, i, f"{resto_val:.1f}%", va='center', ha=align_resto, color=color_resto, fontweight='bold', fontsize=9)

    # Leyenda personalizada
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False, fontsize=10)

    plt.tight_layout()
    
    output_path = "test/outputs/territorial_dumbbell.png"
    os.makedirs("test/outputs", exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    
    print(f"Gráfico Dumbbell guardado en: {output_path}")

if __name__ == "__main__":
    df_master = pd.read_parquet("data/processed/master_dataset.parquet")
    generate_territorial_dumbbell(df_master)
