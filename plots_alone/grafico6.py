import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
def generate_pro_butterfly_chart(df):
    print("Calculando valores porcentuales ponderados...")
    
    # -- 1. FLAGS --
    # (Eliminamos Viaje Crítico)
    df['Balaceras Frecuentes']         = np.where(df['v36b'] >= 3, 1, 0)
    df['Narco-tráfico Frecuente']      = np.where(df['v36c'] >= 3, 1, 0)
    df['Alta Dependencia (>20%)']      = np.where(df['subsidy_ratio'] > 0.20, 1, 0)
    df['Ingreso Autónomo <$600k']      = np.where(df['yautcorh'] < 600000, 1, 0)
    df['Pobreza por Ingresos']         = np.where(df['pobreza'].isin([1, 2]), 1, 0)
    
    # Orden exacto de las variables (de arriba hacia abajo) sin viaje
    all_vars = [
        'Balaceras Frecuentes',
        'Narco-tráfico Frecuente',
        'Pobreza por Ingresos',
        'Alta Dependencia (>20%)',
        'Ingreso Autónomo <$600k'
    ]
    
    # -- 2. CALCULO --
    # Dividimos la base en Subsidiadas (1) y No Subsidiadas (0) a nivel nacional.
    df_sub = df[df['is_subsidized'] == 1]
    df_nosub = df[df['is_subsidized'] == 0]
    
    # Función para calcular porcentaje ponderado
    def wpct(subset, var):
        w = subset['expr'].sum()
        if w == 0: return 0
        return (subset[subset[var] == 1]['expr'].sum() / w) * 100
        
    rows = []
    for var in all_vars:
        pct_nosub = wpct(df_nosub, var)
        pct_sub = wpct(df_sub, var)
        rows.append({
            'Variable': var,
            'No_Subsidiada': pct_nosub,
            'Subsidiada': pct_sub
        })
        
    plot_df = pd.DataFrame(rows)
    
    # -- 3. COLORES --
    c_deep_walnut   = '#4c2e05'
    c_ivory_mist    = '#f8f4e3'
    c_dusk_blue     = '#3c4f76' # Usaremos este para Sin Subsidio
    c_sandy_brown   = '#f19143' # Usaremos este para Con Subsidio
    
    # -- 4. FIGURA --
    # Ajusté el alto a 5.8 para que no quede espacio vacío al sacar una barra
    fig, ax = plt.subplots(figsize=(12, 5.8), dpi=200, facecolor=c_ivory_mist)
    ax.set_facecolor(c_ivory_mist)
    bar_height = 0.45 
    ax.invert_yaxis()
    
    # -- 5. BARRAS Y ANOTACIONES LIMPIAS --
    for i, row in plot_df.iterrows():
        yp = i 
        val_nosub = row['No_Subsidiada']
        val_sub = row['Subsidiada']
        var_name = row['Variable'].upper()
        
        # Barras (Las no subsidiadas van hacia la izquierda, valores negativos)
        ax.barh(yp, -val_nosub, color=c_dusk_blue, height=bar_height, zorder=3)
        # Las subsidiadas van hacia la derecha (valores positivos)
        ax.barh(yp, val_sub, color=c_sandy_brown, height=bar_height, zorder=3)
        
        # Texto centrado de la variable
        ax.text(0, yp - 0.42, var_name, 
                ha='center', va='center', color=c_deep_walnut, 
                fontsize=11, fontweight='bold', zorder=5)
                
        # Porcentajes a los lados
        ax.text(-val_nosub - 2, yp, f"{val_nosub:.0f}%", 
                ha='right', va='center', color=c_dusk_blue, 
                fontsize=13, fontweight='black', zorder=5)
        ax.text(val_sub + 2, yp, f"{val_sub:.0f}%", 
                ha='left', va='center', color=c_sandy_brown, 
                fontsize=13, fontweight='black', zorder=5)
                
    # -- 6. ETIQUETAS DE BANDO Y EJES --
    ax.text(-10, -0.80, '← SIN SUBSIDIO', ha='right', va='bottom', color=c_dusk_blue, fontsize=14, fontweight='black')
    ax.text(10, -0.80, 'CON SUBSIDIO →', ha='left', va='bottom', color=c_sandy_brown, fontsize=14, fontweight='black')
    
    ax.set_xlim(-85, 85)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    # -- 7. TÍTULOS Y ESPACIADO SEGURO --
    fig.suptitle('6.Comparativa Nacional:\n viviendas sin subsidio v/s viviendas sin subsidio ',
                 fontsize=18, fontweight='black', color=c_deep_walnut, y=0.98)
    
    plt.figtext(0.5, 0.90, '.',
                fontsize=13, color=c_deep_walnut, ha='center', alpha=0.8)
                
    plt.subplots_adjust(top=0.78, bottom=0.05, left=0.05, right=0.95)
    
    # -- 8. GUARDADO FINAL --
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, 'deliverables')
    os.makedirs(out_dir, exist_ok=True)
    
    output_path = os.path.join(out_dir, 'butterfly_sub_vs_nosub.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=200, facecolor=fig.get_facecolor())
    plt.close()
    print(f"¡Gráfico Butterfly guardado exitosamente en: {output_path}!")
if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'processed', 'master_dataset.parquet')
    
    try:
        df_master = pd.read_parquet(data_path)
        generate_pro_butterfly_chart(df_master)
    except Exception as e:
        print(f"Error cargando los datos: {e}")