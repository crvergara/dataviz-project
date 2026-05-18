import numpy as np
import matplotlib.pyplot as plt

def create_dot_strip_plot():
    # 1. Configuración General
    COLOR_BG = '#FDFBF7'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Roboto', 'Arial', 'DejaVu Sans']
    
    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    
    # 2. Datos Acumulados
    sizes = [15, 24, 24, 37]
    # Acumulamos para ubicar los puntos en el eje 0-100
    cumulative_x = np.cumsum(sizes) # [15, 39, 63, 100]
    
    labels = [
        "Doble\nVulnerabilidad\n(15%)", 
        "Trampa\nViolencia\n(24%)", 
        "Trampa\nEconómica\n(24%)", 
        "Sin Vulnerabilidad\nSevera (37%)"
    ]
    colors = [
        '#8B4513',  # Doble Vulnerabilidad 
        '#D2691E',  # Trampa de Violencia 
        '#CD853F',  # Trampa Económica 
        '#607D8B'   # Sin Vulnerabilidad 
    ]
    
    # 3. Dibujar Eje Horizontal y Sombreado
    ax.set_xlim(0, 105)
    ax.set_ylim(-0.2, 1.2)
    
    # Eje horizontal principal (y=0)
    ax.hlines(0, 0, 100, color='#333333', linewidth=1.5, zorder=1)
    
    # Sombreado de Zona de Vulnerabilidad (0 a 63%)
    ax.axvspan(0, 63, color='#F5E6D3', alpha=0.6, zorder=0) 
    
    # Texto de la Zona
    ax.text(63/2, -0.15, "Zona de Vulnerabilidad Estructural (63%)", 
            ha='center', va='top', fontsize=11, fontweight='black', color='#D2691E')
            
    # 4. Graficar Puntos y Líneas verticales
    for i in range(len(cumulative_x)):
        x_val = cumulative_x[i]
        
        # Línea vertical fina
        line_height = 0.5 + (i % 2) * 0.3 # Alternar alturas para evitar choques visuales
        ax.vlines(x_val, 0, line_height, color=colors[i], linestyle='-', linewidth=1.5, zorder=2)
        
        # Punto (Dot)
        ax.scatter(x_val, 0, color=colors[i], s=120, zorder=3, edgecolors=COLOR_BG, linewidth=1.5)
        
        # Etiqueta
        ax.text(x_val, line_height + 0.05, labels[i], ha='center', va='bottom', 
                color=colors[i], fontsize=10, fontweight='bold')
                
    # 5. Estética de los ejes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False) # Ocultamos el bottom porque ya dibujamos la hline
    
    # Configurar ticks del eje X
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], color='#555555', fontweight='bold', fontsize=9)
    ax.set_yticks([]) # Ocultar eje Y
    
    ax.set_title("Vulnerabilidad estructural en Viviendas Subsidiadas", 
                 fontsize=14, fontweight='black', color='#333333', loc='left', pad=15)
                 
    # 6. Exportar
    output = "dot_strip_demo.png"
    plt.savefig(output, dpi=300, bbox_inches='tight', facecolor=COLOR_BG)
    print(f"¡Gráfico Dot Strip guardado en: {output}!")
    plt.show()

if __name__ == "__main__":
    create_dot_strip_plot()
