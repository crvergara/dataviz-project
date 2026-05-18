import numpy as np
import matplotlib.pyplot as plt

def create_waterfall_chart():
    # 1. Configuración General
    COLOR_BG = '#FDFBF7'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Roboto', 'Arial', 'DejaVu Sans']
    
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    
    # 2. Datos para Cascada (Waterfall)
    increments = [15, 24, 24, 37]
    labels = [
        "Doble\nVulnerabilidad\n(+15%)", 
        "Trampa\nViolencia\n(+24%)", 
        "Trampa\nEconómica\n(+24%)", 
        "Sin Vulnerabilidad\nSevera (+37%)"
    ]
    colors = [
        '#8B4513',  # Doble Vulnerabilidad 
        '#D2691E',  # Trampa de Violencia 
        '#CD853F',  # Trampa Económica 
        '#607D8B'   # Sin Vulnerabilidad 
    ]
    
    x = np.arange(len(increments))
    bottoms = [0, 15, 39, 63] # Acumulados para usar como base (bottom) de cada barra
    
    # 3. Sombreado de la Zona de Vulnerabilidad (agrupa los 3 primeros)
    ax.axvspan(-0.5, 2.5, color='#F5E6D3', alpha=0.5, zorder=0)
    # Etiqueta de la zona
    ax.text(1, 105, "Vulnerabilidad Estructural (63% Acumulado)", 
            ha='center', va='bottom', fontsize=11, fontweight='black', color='#D2691E')
            
    # Corchete decorativo para enmarcar el 63% visualmente
    y_bracket = 103
    ax.plot([-0.4, 2.4], [y_bracket, y_bracket], color='#D2691E', lw=2)
    ax.plot([-0.4, -0.4], [100, y_bracket], color='#D2691E', lw=2)
    ax.plot([2.4, 2.4], [100, y_bracket], color='#D2691E', lw=2)
    
    # 4. Dibujar barras flotantes y líneas conectoras
    for i in range(len(increments)):
        # Barra Flotante
        ax.bar(x[i], increments[i], bottom=bottoms[i], color=colors[i], 
               width=0.6, zorder=3, edgecolor=COLOR_BG, linewidth=1.5)
               
        # Texto centrado verticalmente dentro o al lado de cada barra
        y_text = bottoms[i] + increments[i]/2
        ax.text(x[i], y_text, f"{increments[i]}%", ha='center', va='center', 
                color='white', fontsize=10, fontweight='bold', zorder=4)
        
        # Línea punteada conectora (salto) desde el techo de la barra actual al piso de la siguiente
        if i < len(increments) - 1:
            ax.plot([x[i], x[i+1]], [bottoms[i] + increments[i], bottoms[i] + increments[i]], 
                    color='#777777', linestyle='--', linewidth=1, zorder=2)
                    
    # Línea base en 0
    ax.axhline(0, color='#333333', linewidth=1.5, zorder=1)
    
    # 5. Estética de Ejes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color='#333333', fontweight='bold', fontsize=9)
    ax.tick_params(axis='x', length=0, pad=10)
    
    # Eje Y (Porcentajes)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], color='#777777', fontsize=9)
    ax.tick_params(axis='y', length=0, pad=5)
    
    # Grillas horizontales muy suaves para guiar el ojo
    ax.grid(axis='y', linestyle='-', color='#E5E5E5', linewidth=0.5, zorder=0)
    
    # Título principal
    ax.set_title("Construcción de la vulnerabilidad en Viviendas Subsidiadas", 
                 fontsize=14, fontweight='black', color='#333333', loc='left', pad=30)
                 
    # 6. Exportar
    output = "waterfall_demo.png"
    plt.savefig(output, dpi=300, bbox_inches='tight', facecolor=COLOR_BG)
    print(f"¡Gráfico Waterfall guardado en: {output}!")
    plt.show()

if __name__ == "__main__":
    create_waterfall_chart()
