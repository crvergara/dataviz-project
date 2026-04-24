import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

def weighted_quantiles(values, weights, quantiles):
    df_clean = pd.DataFrame({'val': values, 'wt': weights}).dropna()
    sorter = np.argsort(df_clean['val'])
    val = df_clean['val'].iloc[sorter]
    wt = df_clean['wt'].iloc[sorter]
    weighted_quantiles = np.cumsum(wt) - 0.5 * wt
    weighted_quantiles /= np.sum(wt)
    return np.interp(quantiles, weighted_quantiles, val)

def generate_premium_quintile_heatmap(df):
    # 1. PREPARACIÓN DE DATOS
    df = df.dropna(subset=['yautcorh', 'expr']).copy()
    quantiles = [0.2, 0.4, 0.6, 0.8]
    cortes_quintiles = weighted_quantiles(df['yautcorh'], df['expr'], quantiles)
    
    bins = [-np.inf] + list(cortes_quintiles) + [np.inf]
    # Etiquetas de quintiles ultra limpias
    labels = ['Q1\n(Menores Ingresos)', 'Q2', 'Q3', 'Q4', 'Q5\n(Mayores Ingresos)']
    df['Quintil'] = pd.cut(df['yautcorh'], bins=bins, labels=labels, include_lowest=True)
    
    df['grupo_vivienda'] = np.where(df['v15'] == 1, 'Vivienda con Subsidio', 'Vivienda Sin Subsidio')
    
    grouped = df.groupby(['grupo_vivienda', 'Quintil'], observed=False)['expr'].sum().unstack()
    props = grouped.div(grouped.sum(axis=1), axis=0) * 100
    props = props.loc[['Vivienda con Subsidio', 'Vivienda Sin Subsidio']]
    
    # 2. CONFIGURACIÓN VISUAL PREMIUM
    plt.rcParams['font.family'] = 'sans-serif'
    # Tamaño más ancho y achatado (panorámico) es mucho más estético para infografías A4
    fig, ax = plt.subplots(figsize=(11, 3.5), dpi=300)
    
    # Fondo completamente limpio
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Paleta secuencial perfecta de la Tabla Oficial: Ivory Mist -> Sandy Brown -> Deep Walnut
    # Se omitió el azul (Dusk Blue) en la gradación para que sea estrictamente secuencial de densidad.
    colors = ['#f8f4e3', '#f19143', '#4c2e05']
    custom_cmap = sns.blend_palette(colors, as_cmap=True)
    
    # Se elimina la colorbar por defecto porque rompe la limpieza
    # Fijamos el vmin y vmax para que los colores extremen el contraste
    sns.heatmap(
        props,
        cmap=custom_cmap,
        annot=True,
        fmt=".1f",
        linewidths=4,       # Separación más gruesa da aspecto de "tiles" modernos
        linecolor='white',
        vmin=0, vmax=35,    # Fuerza a que el naranja oscuro se vea solo en el Q1 subsidiado
        cbar=False,         # Apagamos barra lateral
        ax=ax
    )
    
    # 3. TYPOGRAPHY & DATA-TO-INK
    for t in ax.texts:
        val = float(t.get_text())
        t.set_text(f'{val:.1f}%')
        t.set_fontsize(11) # Números más pequeños
        
        # Color único para los números con contorno para garantizar legibilidad
        t.set_color('white') # Texto Blanco
        t.set_fontweight('bold')
        t.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')]) # Marco negro
            
    # Ejes limpios
    ax.set_ylabel("")
    ax.set_xlabel("")
    
    # Mover X-ticks arriba
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', length=0, labelsize=12, pad=10, colors='#4c2e05') # Mismo formato (Deep Walnut)
    ax.tick_params(axis='y', length=0, labelsize=12, pad=15, colors='#4c2e05', rotation=0) # Mismo formato (Deep Walnut)

    
    # 4. TÍTULOS CON COORDENADAS ABSOLUTAS (A prueba de overlaping)
    # Título principal redondeado para ser el único texto movido mucho más arriba (y=1.10)
    fig.text(0.125, 1.10, "La Trampa del Subsidio: Concentración de Pobreza Institucionalizada", 
             fontsize=17, fontweight='bold', color='#4c2e05', ha='left') # Deep Walnut
             
    # 5. LEYENDA (COLORBAR) ANCLADA EXACTAMENTE BAJO Q2, Q3 y Q4
    # inset_axes usa coordenadas relativas al cuadriculado del Heatmap:
    # x=0.2 arranca justo al terminar Q1. width=0.6 cubre exactamente los bloques Q2, Q3 y Q4.
    cax = ax.inset_axes([0.20, -0.35, 0.60, 0.05])
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=0, vmax=35))
    sm._A = []
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.outline.set_visible(False)
    cb.set_ticks([0, 35])
    # Dejamos la barra graficamente pura sin textos redundantes en las puntas
    cb.ax.set_xticklabels(['', ''])
    
    # Título del Colorbar
    cax.set_title("Nivel de Concentración", fontsize=11, color='#4c2e05', fontweight='bold', pad=8)

    plt.tight_layout()
    # Desplazar el mapa hacia abajo para títulos y hacer mucho espacio abajo para la leyenda
    plt.subplots_adjust(top=0.85, bottom=0.25)  
    
    output_path = "deliverables/national_income_premium_v2.png"
    os.makedirs("deliverables", exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', transparent=False, facecolor='white')
    print(f"Heatmap Premium exportado a: {output_path}")

if __name__ == "__main__":
    df_master = pd.read_parquet("data/processed/master_dataset.parquet")
    generate_premium_quintile_heatmap(df_master)
