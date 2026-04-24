import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

def generate_butterfly_chart(df):
    print("Calculando valores absolutos...")

    # -- 1. FLAGS (Enfocados en la Tesis) --
    df['Viaje Crítico (>1h)']          = np.where(df['o28a_hr'] >= 1, 1, 0)
    df['Balaceras Frecuentes']         = np.where(df['v36b'] >= 3, 1, 0)
    df['Narco-tráfico Frecuente']      = np.where(df['v36c'] >= 3, 1, 0)
    df['Alta Dependencia (>20%)']      = np.where(df['subsidy_ratio'] > 0.20, 1, 0)
    df['Ingreso Autónomo <$600k']      = np.where(df['yautcorh'] < 600000, 1, 0)
    df['Pobreza por Ingresos']         = np.where(df['pobreza'].isin([1, 2]), 1, 0)

    # Orden explícito de abajo hacia arriba
    all_vars = [
        'Ingreso Autónomo <$600k',
        'Alta Dependencia (>20%)',
        'Pobreza por Ingresos',
        'Viaje Crítico (>1h)',
        'Narco-tráfico Frecuente',
        'Balaceras Frecuentes'
    ]

    # -- 2. CALCULO --
    df_rm = df[(df['grupo_vivienda'] == 'Subsidiada') & (df['region'] == 13)]
    df_re = df[(df['grupo_vivienda'] == 'Subsidiada') & (df['region'] != 13)]

    def wpct(sub, var):
        w = sub['expr'].sum()
        return (sub[sub[var] == 1]['expr'].sum() / w * 100) if w > 0 else 0

    rows = []
    for var in all_vars:
        rm = wpct(df_rm, var)
        re = wpct(df_re, var)
        rows.append({
            'Variable': var,
            'RM_Sub': rm,
            'Re_Sub': re,
            'Brecha': rm - re,
        })

    plot_df = pd.DataFrame(rows)

    # -- 3. COLORES --
    c_deep_walnut   = '#4c2e05'
    c_ivory_mist    = '#f8f4e3'
    c_dusk_blue     = '#3c4f76' # Regiones
    c_sandy_brown   = '#f19143' # RM

    # -- 4. FIGURA --
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150, facecolor=c_ivory_mist)
    ax.set_facecolor(c_ivory_mist)

    bar_height = 0.40

    # Línea central 0
    ax.axvline(0, color=c_deep_walnut, lw=2.5, zorder=1, alpha=0.7)

    # -- 5. BARRAS Y ANOTACIONES LIMPIAS --
    max_val = max(plot_df['RM_Sub'].max(), plot_df['Re_Sub'].max())
    x_limit = max_val + 15 

    for i, row in plot_df.iterrows():
        yp = i
        rm = row['RM_Sub']
        re = row['Re_Sub']
        var_name = row['Variable']

        # Barra Regiones (Izquierda)
        ax.barh(yp, -re, color=c_dusk_blue, height=bar_height, edgecolor=c_ivory_mist, linewidth=1.5, zorder=3)
        # Barra RM (Derecha)
        ax.barh(yp, rm, color=c_sandy_brown, height=bar_height, edgecolor=c_ivory_mist, linewidth=1.5, zorder=3)

        # Nombre de la variable centrado (limpio)
        ax.text(0, yp + 0.35, var_name.upper(), 
                ha='center', va='bottom', color=c_deep_walnut, 
                fontsize=12, fontweight='bold', zorder=5,
                bbox=dict(facecolor=c_ivory_mist, edgecolor='none', pad=2, alpha=0.9))

        # Porcentaje Regiones (Izquierda, estándar azul)
        ax.text(-re - 2, yp, f"{re:.0f}%", 
                ha='right', va='center', color=c_dusk_blue, 
                fontsize=15, fontweight='black', zorder=5)

        # Porcentaje RM (Derecha, estándar naranja)
        ax.text(rm + 2, yp, f"{rm:.0f}%", 
                ha='left', va='center', color=c_sandy_brown, 
                fontsize=15, fontweight='black', zorder=5)

    # -- 6. SOMBREADOS Y TÍTULOS DE BANDO --
    # Volvemos al sombreado simple (dos lados enteros)
    ax.axvspan(-x_limit, 0, color=c_dusk_blue, alpha=0.12, zorder=0)
    ax.axvspan(0, x_limit, color=c_sandy_brown, alpha=0.12, zorder=0)

    # Títulos de Bando
    top_y = len(plot_df) + 0.1
    ax.text(-5, top_y, 'TASA ABSOLUTA EN REGIONES',
            ha='right', va='bottom', color=c_dusk_blue, fontsize=15, fontweight='black')
    ax.text(5, top_y, 'TASA ABSOLUTA EN RM (SANTIAGO)',
            ha='left', va='bottom', color=c_sandy_brown, fontsize=15, fontweight='black')

    # -- 7. EJES CON VALORES POSITIVOS A AMBOS LADOS --
    ax.set_yticks([])
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(-0.8, len(plot_df) + 0.5)
    
    # Eje X referencial positivo para ambos bandos
    ticks = np.arange(0, int(x_limit), 20)
    all_ticks = np.concatenate([-ticks[::-1][:-1], ticks])
    ax.set_xticks(all_ticks)
    ax.set_xticklabels([f"{abs(int(t))}%" for t in all_ticks], fontsize=10, color=c_deep_walnut, alpha=0.7)
    
    ax.tick_params(axis='x', length=4, color=c_deep_walnut, direction='out')
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Mostramos la línea base del eje x pero muy sutil
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color(c_deep_walnut)
    ax.spines['bottom'].set_alpha(0.2)

    # -- 8. TITULO GENERAL --
    fig.suptitle('Las Dos Caras de la Vivienda Social en Chile',
                 fontsize=22, fontweight='black', color=c_deep_walnut, y=1.02, ha='center')
                 
    # Subtítulo eliminado según solicitud.

    plt.tight_layout()
    output_path = 'test/outputs/butterfly_chart_clean.png'
    os.makedirs('test/outputs', exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Guardado en: {output_path}")

if __name__ == '__main__':
    df_master = pd.read_parquet('data/processed/master_dataset.parquet')
    generate_butterfly_chart(df_master)
