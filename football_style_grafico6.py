import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import matplotlib.colors as mcolors

def get_project_root():
    """Encuentra la ruta raíz del proyecto dinámicamente."""
    current_dir = Path(__file__).resolve().parent
    if (current_dir / 'data').exists() and (current_dir / 'deliverables').exists():
        return current_dir
    else:
        return current_dir.parent

def generate_football_style_chart(df):
    print("Calculando valores porcentuales ponderados...")
    
    # -- 1. FLAGS --
    df['Balaceras Frecuentes']         = np.where(df['v36b'] >= 3, 1, 0)
    df['Narco-tráfico Frecuente']      = np.where(df['v36c'] >= 3, 1, 0)
    df['Alta Dependencia (>20%)']      = np.where(df['subsidy_ratio'] > 0.20, 1, 0)
    df['Ingreso Autónomo <$600k']      = np.where(df['yautcorh'] < 600000, 1, 0)
    df['Pobreza por Ingresos']         = np.where(df['pobreza'].isin([1, 2]), 1, 0)
    
    all_vars = [
        'Balaceras Frecuentes',
        'Narco-tráfico Frecuente',
        'Pobreza por Ingresos',
        'Alta Dependencia (>20%)',
        'Ingreso Autónomo <$600k'
    ]
    
    # -- 2. CALCULO --
    df_sub = df[df['is_subsidized'] == 1]
    df_nosub = df[df['is_subsidized'] == 0]
    
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
    c_ivory_mist    = '#ffffff' # Fondo blanco para que se parezca más a la app deportiva
    c_dusk_blue     = '#3c4f76' # Sin Subsidio
    c_sandy_brown   = '#f19143' # Con Subsidio
    c_bg_track      = '#f0f0f0' # Gris claro para la pista de la barra
    
    # -- 4. FIGURA --
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=200, facecolor=c_ivory_mist)
    ax.set_facecolor(c_ivory_mist)
    ax.invert_yaxis()
    
    gap = 3 # Espacio vacío en el centro entre las dos barras
    max_track = 100 # Longitud máxima de la pista (100%)
    line_w = 8 # Grosor de las barras
    
    # -- 5. BARRAS Y ANOTACIONES ESTILO DEPORTIVO --
    for i, row in plot_df.iterrows():
        yp = i 
        val_nosub = row['No_Subsidiada']
        val_sub = row['Subsidiada']
        var_name = row['Variable']
        
        # Determinamos quién gana (quién tiene mayor porcentaje)
        if val_nosub > val_sub:
            alpha_nosub = 1.0
            alpha_sub = 0.35 # Perdedor apagado
            weight_nosub = 'black'
            weight_sub = 'normal'
            text_alpha_nosub = 1.0
            text_alpha_sub = 0.7
        else:
            alpha_nosub = 0.35 # Perdedor apagado
            alpha_sub = 1.0
            weight_nosub = 'normal'
            weight_sub = 'black'
            text_alpha_nosub = 0.7
            text_alpha_sub = 1.0
            
        # -- DIBUJO DE PISTAS GRISES (Background) --
        # Usamos plot en vez de barh para poder usar solid_capstyle='round' (bordes redondeados)
        ax.plot([-gap, -gap - max_track], [yp, yp], color=c_bg_track, linewidth=line_w, solid_capstyle='round', zorder=1)
        ax.plot([gap, gap + max_track], [yp, yp], color=c_bg_track, linewidth=line_w, solid_capstyle='round', zorder=1)
        
        # -- DIBUJO DE BARRAS DE DATOS --
        if val_nosub > 0:
            ax.plot([-gap, -gap - val_nosub], [yp, yp], color=c_dusk_blue, linewidth=line_w, 
                    solid_capstyle='round', alpha=alpha_nosub, zorder=2)
        if val_sub > 0:
            ax.plot([gap, gap + val_sub], [yp, yp], color=c_sandy_brown, linewidth=line_w, 
                    solid_capstyle='round', alpha=alpha_sub, zorder=2)
        
        # -- TEXTOS --
        text_y = yp - 0.35 # El texto va por encima de las barras
        
        # Nombre de la variable (centro)
        ax.text(0, text_y, var_name, 
                ha='center', va='bottom', color=c_deep_walnut, 
                fontsize=11, fontweight='bold', zorder=5)
                
        # Porcentajes a los extremos
        ax.text(-gap - max_track, text_y, f"{val_nosub:.0f}%", 
                ha='left', va='bottom', color=c_dusk_blue, 
                fontsize=13, fontweight=weight_nosub, alpha=text_alpha_nosub, zorder=5)
                
        ax.text(gap + max_track, text_y, f"{val_sub:.0f}%", 
                ha='right', va='bottom', color=c_sandy_brown, 
                fontsize=13, fontweight=weight_sub, alpha=text_alpha_sub, zorder=5)
                
    # -- 6. ETIQUETAS DE BANDO Y EJES --
    # Leyendas tipo "Equipo Local vs Equipo Visitante"
    ax.text(-gap - max_track, -0.80, 'SIN SUBSIDIO', ha='left', va='bottom', color=c_dusk_blue, fontsize=12, fontweight='black')
    ax.text(gap + max_track, -0.80, 'CON SUBSIDIO', ha='right', va='bottom', color=c_sandy_brown, fontsize=12, fontweight='black')
    
    # Límites asimétricos no son necesarios porque la pista llega a 100
    ax.set_xlim(-gap - max_track - 5, gap + max_track + 5)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    # -- 7. TÍTULOS Y ESPACIADO SEGURO --
    fig.suptitle('6. Comparativa Nacional:\n viviendas con subsidio v/s viviendas sin subsidio',
                 fontsize=18, fontweight='black', color=c_deep_walnut, y=1.0)
    
    plt.figtext(0.5, 0.92, '"El desafío del subsidio: Entorno hostil y dependencia económica."',
                fontsize=12, color=c_deep_walnut, ha='center', alpha=0.9)
                
    plt.subplots_adjust(top=0.82, bottom=0.05, left=0.05, right=0.95)
    
    # -- 8. GUARDADO FINAL --
    project_root = get_project_root()
    out_dir = project_root / 'deliverables'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = out_dir / 'football_style_sub_vs_nosub.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=200, facecolor=fig.get_facecolor())
    plt.close()
    print(f"¡Gráfico Estilo Deportivo guardado exitosamente en: {output_path}!")

if __name__ == '__main__':
    project_root = get_project_root()
    data_path = project_root / 'data' / 'processed' / 'master_dataset.parquet'
    
    try:
        df_master = pd.read_parquet(data_path)
        generate_football_style_chart(df_master)
    except Exception as e:
        print(f"Error cargando los datos: {e}")
