import numpy as np
import matplotlib.pyplot as plt

def create_stacked_bar():
    # 1. Configuración General solicitada
    COLOR_BG = '#FDFBF7'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Roboto', 'Arial', 'DejaVu Sans']
    
    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    
    # 2. Datos y Colores
    sizes = [15, 24, 24, 37]
    labels = [
        "Doble\n(15%)", 
        "Trampa\nViolencia\n(24%)", 
        "Trampa\nEconómica\n(24%)", 
        "Sin Vulnerabilidad\nSevera (37%)"
    ]
    colors = [
        '#8B4513',  # Doble Vulnerabilidad (15%)
        '#D2691E',  # Trampa de Violencia (24%)
        '#CD853F',  # Trampa Económica (24%)
        '#607D8B'   # Sin Vulnerabilidad Severa (37%)
    ]
    
    # 3. Dibujar la barra apilada horizontal
    left = 0
    bar_height = 0.8
    for i in range(len(sizes)):
        # Barra
        ax.barh(0, sizes[i], left=left, color=colors[i], height=bar_height, edgecolor=COLOR_BG, linewidth=2.5)
        
        # Etiqueta interna centrada
        ax.text(left + sizes[i]/2, 0, labels[i], ha='center', va='center', 
                color='white', fontsize=10, fontweight='bold')
        left += sizes[i]
        
    # 4. Línea superior (corchete) para agrupar el 63%
    # La barra va de y=-0.4 a y=0.4 (ya que está centrada en 0 y height=0.8)
    y_bracket = 0.48
    y_tick = 0.42
    
    # Línea horizontal del corchete
    ax.plot([0, 63], [y_bracket, y_bracket], color='#D2691E', lw=2)
    # Ticks o "patitas" del corchete
    ax.plot([0, 0], [y_tick, y_bracket], color='#D2691E', lw=2)
    ax.plot([63, 63], [y_tick, y_bracket], color='#D2691E', lw=2)
    
    # Texto del corchete
    ax.text(63/2, y_bracket + 0.05, "63% Vulnerabilidad Estructural", ha='center', va='bottom',
            fontsize=12, fontweight='black', color='#D2691E')
            
    # 5. Limpieza y título
    ax.axis('off')
    ax.set_ylim(-0.6, 1.0)
    ax.set_title("Vulnerabilidad estructural en Viviendas Subsidiadas", 
                 fontsize=14, fontweight='black', color='#333333', loc='left', pad=20)
                 
    # 6. Exportar
    output = "stacked_bar_demo.png"
    plt.savefig(output, dpi=300, bbox_inches='tight', facecolor=COLOR_BG)
    print(f"¡Gráfico de barra agrupada guardado en: {output}!")
    plt.show()

if __name__ == "__main__":
    create_stacked_bar()
