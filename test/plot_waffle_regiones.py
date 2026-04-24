import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_butterfly_waffle(df):
    print("Calculando datos para Waffle Chart (Regiones)...")

    # -- 1. FLAGS Y FILTRO REGIONES --
    df_re = df[df['region'] != 13].copy()
    
    df_re['Ingreso Autónomo <$600k'] = np.where(df_re['yautcorh'] < 600000, 1, 0)
    df_re['Alta Dependencia (>20%)'] = np.where(df_re['subsidy_ratio'] > 0.20, 1, 0)

    df_sub = df_re[df_re['grupo_vivienda'] == 'Subsidiada']
    df_priv = df_re[df_re['grupo_vivienda'] == 'No Subsidiada / Otro']

    def wpct(d, var):
        w = d['expr'].sum()
        return (d[d[var] == 1]['expr'].sum() / w * 100) if w > 0 else 0

    metrics = [
        {'var': 'Alta Dependencia (>20%)', 'label': 'Alta Dependencia del Estado\n(>20% del ingreso son bonos)'},
        {'var': 'Ingreso Autónomo <$600k', 'label': 'Incapacidad Financiera\n(Ingreso Autónomo < $600k)'}
    ]

    # Recopilar datos
    data = []
    for m in metrics:
        data.append({
            'label': m['label'],
            'priv': round(wpct(df_priv, m['var'])),
            'sub': round(wpct(df_sub, m['var']))
        })

    # -- 2. COLORES (Consistentes con Gráfico 2) --
    c_deep_walnut   = '#4c2e05'
    c_ivory_mist    = '#f8f4e3'
    c_dusk_blue     = '#3c4f76' # Subsidiado (Regiones)
    c_priv_grey     = '#95a5a6' # Privado (Neutro)

    # -- 3. FIGURA --
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150, facecolor=c_ivory_mist)
    ax.set_facecolor(c_ivory_mist)

    # Línea central 0
    ax.axvline(0, color=c_deep_walnut, lw=2.5, zorder=1, alpha=0.7)

    # -- 4. FUNCIÓN DIBUJO WAFFLE --
    def draw_waffle(ax, x_offset, y_offset, pct, color, mirror=False):
        pts_filled = []
        pts_empty = []
        count = 0
        for y in range(10):
            # Mirror: llena de derecha a izquierda (hacia afuera desde el centro)
            x_range = range(-1, -11, -1) if mirror else range(1, 11)
            for x in x_range:
                count += 1
                if count <= pct:
                    pts_filled.append((x + x_offset, y + y_offset))
                else:
                    pts_empty.append((x + x_offset, y + y_offset))
                    
        # Dibujar cuadritos llenos
        if pts_filled:
            fx, fy = zip(*pts_filled)
            ax.scatter(fx, fy, s=180, c=color, marker='s', edgecolors=c_ivory_mist, linewidths=1.5, zorder=3)
        # Dibujar cuadritos vacíos (muy tenues)
        if pts_empty:
            ex, ey = zip(*pts_empty)
            ax.scatter(ex, ey, s=180, c=color, marker='s', alpha=0.1, edgecolors=c_ivory_mist, linewidths=1.5, zorder=3)

    # -- 5. RENDERIZADO --
    y_spacing = 14
    for i, row in enumerate(data):
        y_off = i * y_spacing
        
        # Sombreado de "Bando Ganador" (peor) como en el Butterfly
        # En este caso, Subsidiada (derecha) es siempre peor, así que sombreamos la derecha
        import matplotlib.patches as mpatches
        rect = mpatches.Rectangle((0, y_off - 1), 16, 11.5, facecolor=c_dusk_blue, alpha=0.08, zorder=0)
        ax.add_patch(rect)

        # Waffle Privado (Izquierda, mirror=True para que crezca desde el centro)
        draw_waffle(ax, 0, y_off, row['priv'], c_priv_grey, mirror=True)
        # Waffle Subsidiado (Derecha)
        draw_waffle(ax, 0, y_off, row['sub'], c_dusk_blue, mirror=False)

        # Textos Centrales
        ax.text(0, y_off + 4.5, row['label'].upper(),
                ha='center', va='center', color=c_deep_walnut, 
                fontsize=11, fontweight='bold', zorder=5,
                bbox=dict(facecolor=c_ivory_mist, edgecolor='none', pad=4, alpha=0.9))

        # Porcentajes Extremos
        ax.text(-12.5, y_off + 4.5, f"{row['priv']}%", 
                ha='right', va='center', color=c_priv_grey, 
                fontsize=24, fontweight='black', zorder=5)
        
        ax.text(12.5, y_off + 4.5, f"{row['sub']}%", 
                ha='left', va='center', color=c_dusk_blue, 
                fontsize=24, fontweight='black', zorder=5)

    # -- 6. TÍTULOS DE BANDO --
    top_y = (len(data) - 1) * y_spacing + 11
    ax.text(-5.5, top_y, 'MERCADO PRIVADO\n(Regiones)',
            ha='center', va='bottom', color=c_priv_grey, fontsize=14, fontweight='black')
    ax.text(5.5, top_y, 'VIVIENDA SUBSIDIADA\n(Regiones)',
            ha='center', va='bottom', color=c_dusk_blue, fontsize=14, fontweight='black')

    # -- 7. EJES --
    ax.set_xlim(-18, 18)
    ax.set_ylim(-2, top_y + 3)
    ax.axis('off')

    # -- 8. TITULO GENERAL --
    fig.suptitle('El Techo de Cristal Financiero en Regiones',
                 fontsize=22, fontweight='black', color=c_deep_walnut, y=1.02, ha='center')
    ax.set_title('Tener casa propia estatal no mejora la autonomía: de cada 100 familias en vivienda subsidiada,\n40 no pueden generar 600 mil pesos por sí mismas, superando los promedios del mercado privado.',
                 fontsize=12, color=c_deep_walnut, pad=15, style='italic', alpha=0.8, ha='center')

    plt.tight_layout()
    output_path = 'test/outputs/waffle_regiones.png'
    os.makedirs('test/outputs', exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Guardado en: {output_path}")

if __name__ == '__main__':
    df_master = pd.read_parquet('data/processed/master_dataset.parquet')
    generate_butterfly_waffle(df_master)
