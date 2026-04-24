import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- CONFIGURACIÓN DE PALETA EXACTA DEL USUARIO ---
C_BG            = '#f8f4e3' # ivory-mist (Fondo)
C_TXT           = '#4c2e05' # deep-walnut (Títulos y texto principal)
C_SUB           = '#f19143' # sandy-brown (Vivienda Subsidiada - foco de atención)
C_PRIV          = '#3c4f76' # dusk-blue (Vivienda Privada - comparativa)
C_GAP           = '#d38b5d' # toasted-almond (Relleno sutil para la brecha)

plt.rcParams.update({
    'figure.facecolor': C_BG,
    'axes.facecolor': C_BG,
    'font.family': 'DejaVu Sans', # Usaremos fuente estándar, editable luego en Ilustrator/Figma
    'text.color': C_TXT,
    'axes.labelcolor': C_TXT,
    'xtick.color': C_TXT,
    'ytick.color': C_TXT
})

def wm(value, weight):
    """Calcula el promedio ponderado."""
    return np.average(value, weights=weight)

def generate_historical_plot():
    in_path = r'data\processed\balaceras_historico.parquet'
    out_path = r'deliverables\03_brecha_historica.png'
    
    print("Cargando datos...")
    if not os.path.exists(in_path):
        print(f"Error: No se encuentra el archivo {in_path}")
        return
        
    df = pd.read_parquet(in_path)
    
    # Agregar % ponderado de balaceras frecuentes por año y grupo
    # Eliminamos nulos de la columna ponderador just in case
    df = df.dropna(subset=['ponderador', 'is_balacera'])
    
    # Calcular promedio ponderado
    agg_df = df.groupby(['year', 'is_subsidized']).apply(
        lambda x: wm(x['is_balacera'], x['ponderador'])
    ).reset_index(name='pct_balacera')
    
    # Separar series
    sub_df = agg_df[agg_df['is_subsidized'] == 1].sort_values('year')
    priv_df = agg_df[agg_df['is_subsidized'] == 0].sort_values('year')
    
    years = sub_df['year'].values
    y_sub = sub_df['pct_balacera'].values * 100
    y_priv = priv_df['pct_balacera'].values * 100
    
    # --- CREACIÓN DEL GRÁFICO ---
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Títulos - Data Storytelling
    fig.text(0.04, 0.90, "Frecuencia de balaceras según el tipo de solución habitacional en la Región MR", fontsize=18, fontweight='bold', color=C_TXT)

    # Cuadrícula sutil (Eje Y)
    ax.grid(axis='y', linestyle='--', linewidth=0.8, color=C_TXT, alpha=0.15, zorder=0)
    ax.set_yticks([10, 20, 30, 40, 50])
    ax.set_yticklabels([f"{y}%" for y in [10, 20, 30, 40, 50]], fontsize=10, color=C_TXT, alpha=0.6, fontweight='bold')
    
    # Eje X - Los años
    ax.set_xticks(years)
    ax.set_xticklabels([str(int(y)) for y in years], fontsize=12, fontweight='bold', color=C_TXT)
    ax.tick_params(axis='x', length=0, pad=2) # Quitar palitos del eje X
    ax.tick_params(axis='y', length=0, pad=0)
    
    # Eliminar spines (Bordes de la caja)
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)
    
    # Dibujar la línea base abajo sutil para anclar el gráfico
    ax.axhline(10, color=C_TXT, linewidth=2, alpha=0.5, zorder=1)

    # Relleno del área de la gap primero (Z-order 2)
    ax.fill_between(years, y_priv, y_sub, color=C_GAP, alpha=0.15, zorder=2)

    # Gráfico principal: Líneas gruesas y marcadores premium
    ax.plot(years, y_sub, color=C_SUB, linewidth=3, marker='o', markersize=10, 
            markeredgecolor=C_BG, markeredgewidth=1.0, zorder=2, label='Vivienda Subsidiada')
    ax.plot(years, y_priv, color=C_PRIV, linewidth=3, marker='o', markersize=10, 
            markeredgecolor=C_BG, markeredgewidth=1.0, zorder=2, label='Vivienda Sin Subsidio')
    
    # Líneas verticales tenues que anclan los puntos al eje X (Estética "Lollipop" invertida)
    for y in years:
        ax.axvline(y, ymin=0, ymax=1, color=C_TXT, alpha=0.08, linestyle=':', zorder=0)

    # Cajas con porcentajes directos
    bbox_props_sub = dict(boxstyle="round,pad=0.3", fc=C_SUB, ec="none", alpha=0.9)
    bbox_props_priv = dict(boxstyle="round,pad=0.3", fc=C_PRIV, ec="none", alpha=0.9)

    for i, year in enumerate(years):
        ax.text(year, y_sub[i] + 2.5, f"{y_sub[i]:.1f}%", color=C_BG, fontsize=10, fontweight='bold', ha='center', va='bottom', bbox=bbox_props_sub, zorder=5)
        ax.text(year, y_priv[i] - 2.5, f"{y_priv[i]:.1f}%", color=C_BG, fontsize=10, fontweight='bold', ha='center', va='top', bbox=bbox_props_priv, zorder=5)
    
    # Leyenda Estética (Forzada debajo del título usando coordenadas relativas al área del gráfico)
    leg = ax.legend(loc='upper right', bbox_to_anchor=(1.2, 0.85), frameon=False, ncol=1, fontsize=11)
    for text in leg.get_texts():
        text.set_color(C_TXT)
        text.set_fontweight('bold')

    # Anotación sobre The Gap al final
    gap_2024 = y_sub[-1] - y_priv[-1]
    ax.annotate(
        f"BRECHA ABSOLUTA\n+{gap_2024:.1f}%",
        xy=(years[-1], (y_sub[-1] + y_priv[-1])/2), 
        xytext=(years[-1] + 0.6, (y_sub[-1] + y_priv[-1])/2),
        ha='left', va='center',
        fontsize=10, fontweight='bold', color=C_TXT,
        arrowprops=dict(arrowstyle="wedge,tail_width=0.7", color=C_GAP, alpha=0.8, lw=0)
    )

    ax.set_xlim(years[0] - 0.5, years[-1] + 2.0) # Ajustado margen
    ax.set_ylim(5, max(y_sub) + 12) # Levantar el techo para dar respiro a las etiquetas superiores

    # Guardar
    os.makedirs('deliverables', exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=300)
    print(f"Gráfico guardado en: {out_path}")

if __name__ == "__main__":
    generate_historical_plot()
