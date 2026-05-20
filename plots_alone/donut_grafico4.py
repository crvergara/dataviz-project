import numpy as np
import matplotlib.pyplot as plt

def create_donut_chart():
    # 1. Configuración General
    COLOR_BG = '#f8f4e3'
    COLOR_TEXT = '#4c2e05'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Roboto', 'Arial', 'DejaVu Sans']
    
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    
    # 2. Datos y Colores (Corregidos para ser mutuamente excluyentes)
    # Usamos la paleta de colores del proyecto para consistencia visual.
    colors = [
        '#4c2e05',  # Ambas Trampas (15.0%) - deep-walnut
        '#d38b5d',  # Solo Trampa de Violencia (9.4%) - toasted-almond
        '#f19143',  # Solo Trampa Económica (38.6%) - sandy-brown
        '#3c4f76'   # Sin Vulnerabilidad Severa (37.0%) - dusk-blue
    ]
    
    # Datos validados: Las categorías de vulnerabilidad ahora son exclusivas y suman 63%.
    sizes = [15.0, 9.4, 38.6, 37.0]
    labels = [
        "Doble Riesgo\n(Económico y Violencia)\n(15%)",
        "Riesgo por Violencia\n(9%)",
        "Riesgo Económico\n(39%)",
        "Sin Vulnerabilidad Severa\n(37%)"
    ]
    
    # 3. Anillo exterior para agrupar el 63% de vulnerabilidad
    vulnerable_pct = sum(sizes[:3])
    ax.pie(
        [vulnerable_pct, 100 - vulnerable_pct], 
        radius=1.05, 
        colors=['#f19143', 'none'], 
        startangle=90,
        counterclock=False,
        wedgeprops={'width': 0.02, 'edgecolor': 'none'}
    )

    # 4. Donut Chart principal
    wedges, texts = ax.pie(
        sizes, 
        radius=1.0,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={'width': 0.35, 'edgecolor': COLOR_BG, 'linewidth': 3}
    )
    
    # 5. Anotaciones externas con líneas conectoras (estilo mejorado)
    kw = dict(arrowprops=dict(arrowstyle="-", color=COLOR_TEXT, lw=0.8, alpha=0.6),
              bbox=None, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
            
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = f"angle,angleA=0,angleB={ang}"
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        
        # Ajustar el punto de origen de la flecha para que parta desde fuera del anillo exterior
        # si el segmento es parte del grupo vulnerable.
        if i < 3: # Los primeros 3 segmentos son los vulnerables
            arrow_start_radius = 1.07
        else:
            arrow_start_radius = 1.0
        
        # Posicionamiento del texto
        x_text = 1.4 * np.sign(x)
        y_text = 1.2 * y
        
        ax.annotate(labels[i], xy=(x * arrow_start_radius, y * arrow_start_radius), xytext=(x_text, y_text),
                    horizontalalignment=horizontalalignment,
                    fontsize=11, fontweight='bold', color=colors[i], **kw)
    
    # 6. Texto destacado y mejorado al centro del Donut
    # Se divide en dos partes para jerarquía visual y se elimina el círculo de fondo.
    ax.text(0, 0.15, "63%", ha='center', va='center',
            fontsize=30, fontweight='black', color='#f19143')
            
    ax.text(0, -0.1, "de los hogares\nenfrenta una\nvulnerabilidad severa",
            ha='center', va='center',
            fontsize=10, color=COLOR_TEXT, alpha=0.9)
            
    # 7. Títulos y Subtítulos
    fig.suptitle(
        "El Desafío de la Autonomía: Hogares que Reciben Subsidios",
        fontsize=16, fontweight='black', color=COLOR_TEXT, x=0.5, y=0.98, ha='center'
    )
    ax.set_title(
        "Desglose de los hogares que no logran la autonomía a pesar de la ayuda estatal.",
        fontsize=11, color=COLOR_TEXT, loc='center', pad=10, alpha=0.8
    )
                 
    # 8. Exportar gráfico
    output = "grafico_donut_vulnerabilidad.png"
    plt.savefig(output, dpi=300, bbox_inches='tight', facecolor=COLOR_BG)
    print(f"¡Gráfico con conectores optimizados guardado en: {output}!")

if __name__ == "__main__":
    create_donut_chart()
