import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_rm_waffle(df):
    print("Calculando datos para Waffle Chart (Zoom Santiago)...")

    # -- 1. FLAGS Y FILTRO SANTIAGO (RM) --
    df_rm = df[df['region'] == 13].copy()
    
    df_rm['Balaceras Frecuentes']         = np.where(df_rm['v36b'] >= 3, 1, 0)
    df_rm['Narco-tráfico Frecuente']      = np.where(df_rm['v36c'] >= 3, 1, 0)

    df_sub = df_rm[df_rm['grupo_vivienda'] == 'Subsidiada']
    df_priv = df_rm[df_rm['grupo_vivienda'] == 'No Subsidiada / Otro']

    def wpct(d, var):
        w = d['expr'].sum()
        return (d[d[var] == 1]['expr'].sum() / w * 100) if w > 0 else 0

    metrics = [
        {'var': 'Narco-tráfico Frecuente', 'label': 'Tráfico de Drogas\n(Entorno cercano frecuentemente)'},
        {'var': 'Balaceras Frecuentes', 'label': 'Balaceras en el Entorno\n(Semanalmente o más)'}
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
    c_sandy_brown   = '#f19143' # Subsidiado (RM)
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
        
        # Sombreado de "Bando Ganador"
        # Vivienda Subsidiada siempre es peor en violencia, por ende fondo en la derecha
        import matplotlib.patches as mpatches
        rect = mpatches.Rectangle((0, y_off - 1), 16, 11.5, facecolor=c_sandy_brown, alpha=0.12, zorder=0)
        ax.add_patch(rect)

        # Waffle Privado (Izquierda, mirror=True)
        draw_waffle(ax, 0, y_off, row['priv'], c_priv_grey, mirror=True)
        # Waffle Subsidiado (Derecha)
        draw_waffle(ax, 0, y_off, row['sub'], c_sandy_brown, mirror=False)

        # Textos Centrales
        ax.text(0, y_off + 4.5, row['label'].upper(),
                ha='center', va='center', color=c_deep_walnut, 
                fontsize=11, fontweight='bold', zorder=5,
                bbox=dict(facecolor=c_ivory_mist, edgecolor='none', pad=4, alpha=0.9))

        # Porcentajes Extremos
        ax.text(-12.5, y_off + 4.5, f"{row['priv']}%", 
                ha='right', va='center', color=c_priv_grey, 
                fontsize=28, fontweight='black', zorder=5)
        
        ax.text(12.5, y_off + 4.5, f"{row['sub']}%", 
                ha='left', va='center', color=c_sandy_brown, 
                fontsize=28, fontweight='black', zorder=5)

    # -- 6. TÍTULOS DE BANDO --
    top_y = (len(data) - 1) * y_spacing + 11
    ax.text(-5.5, top_y, 'MERCADO PRIVADO\n(Santiago)',
            ha='center', va='bottom', color=c_priv_grey, fontsize=15, fontweight='black')
    ax.text(5.5, top_y, 'VIVIENDA SUBSIDIADA\n(Santiago)',
            ha='center', va='bottom', color=c_sandy_brown, fontsize=15, fontweight='black')

    # -- 7. EJES --
    ax.set_xlim(-18, 18)
    ax.set_ylim(-2, top_y + 3)
    ax.axis('off')

    # -- 8. TITULO GENERAL --
    fig.suptitle('La Trampa Violenta de la Periferia',
                 fontsize=24, fontweight='black', color=c_deep_walnut, y=1.02, ha='center')
    ax.set_title('El castigo territorial: cómo la vivienda social en Santiago condena a las familias a entornos controlados por el crimen,\nsuperando abismalmente los índices del mercado privado.',
                 fontsize=12, color=c_deep_walnut, pad=15, style='italic', alpha=0.8, ha='center')

    plt.tight_layout()
    output_path = 'test/outputs/waffle_santiago.png'
    os.makedirs('test/outputs', exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Guardado en: {output_path}")

if __name__ == '__main__':
    df_master = pd.read_parquet('data/processed/master_dataset.parquet')
    generate_rm_waffle(df_master)
