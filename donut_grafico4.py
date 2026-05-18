import numpy as np
import matplotlib.pyplot as plt

def create_optimized_donut():
    # 1. Configuración General
    COLOR_BG = '#FDFBF7'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Roboto', 'Arial', 'DejaVu Sans']
    
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    
    # 2. Colores
    colors = [
        '#8B4513',  # Doble Vulnerabilidad (15%)
        '#D2691E',  # Trampa de Violencia (24%)
        '#CD853F',  # Trampa Económica (24%)
        '#607D8B'   # Sin Vulnerabilidad Severa (37%)
    ]
    
    sizes = [15, 24, 24, 37]
    labels = [
        "Doble Vulnerabilidad\n(15%)", 
        "Trampa de Violencia\n(24%)", 
        "Trampa Económica\n(24%)", 
        "Sin Vulnerabilidad\nSevera (37%)"
    ]
    
    # 3. Anillo Extra fino para agrupar el 63% (Vulnerables)
    ax.pie(
        [63, 37], 
        radius=1.06, 
        colors=['#D2691E', 'none'], 
        startangle=90,
        counterclock=False,
        wedgeprops={'width': 0.03, 'edgecolor': 'none'}
    )
    
    # 4. Donut Chart principal (interno)
    wedges, texts = ax.pie(
        sizes, 
        radius=1.0,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={'width': 0.35, 'edgecolor': COLOR_BG, 'linewidth': 2}
    )
    
    # 5. Anotaciones externas con líneas conectoras finas
    kw = dict(arrowprops=dict(arrowstyle="-", color="#555555", lw=0.7),
              bbox=None, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        
        # Ajustar el punto de origen de la flecha para que NO cruce el anillo extra
        # Si es del grupo vulnerable (los 3 primeros), el punto inicia en radio 1.08 (fuera del anillo 1.06)
        if i < 3:
            arrow_start_x = x * 1.08
            arrow_start_y = y * 1.08
        else:
            arrow_start_x = x * 1.0
            arrow_start_y = y * 1.0
            
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = f"angle,angleA=0,angleB={ang}"
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        
        # Empujamos el texto
        x_text = 1.35 * np.sign(x)
        y_text = 1.25 * y
        
        ax.annotate(labels[i], xy=(arrow_start_x, arrow_start_y), xytext=(x_text, y_text),
                    horizontalalignment=horizontalalignment,
                    fontsize=10, fontweight='bold', color=colors[i], **kw)

    # Texto destacado al centro del Donut
    ax.text(0, 0, "63%\nVulnerabilidad\nEstructural", ha='center', va='center', 
            fontsize=15, fontweight='black', color='#D2691E')
            
    # Título principal
    ax.set_title("Vulnerabilidad estructural en Viviendas Subsidiadas", 
                 fontsize=14, fontweight='black', color='#333333', pad=30)
                 
    # 6. Exportar gráfico
    output = "optimized_donut_demo.png"
    plt.savefig(output, dpi=300, bbox_inches='tight', facecolor=COLOR_BG)
    print(f"¡Gráfico con conectores optimizados guardado en: {output}!")
    plt.show()

if __name__ == "__main__":
    create_optimized_donut()
