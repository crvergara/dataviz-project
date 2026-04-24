import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patheffects as path_effects

def generate_crime_dumbbell():
    # Paleta corporativa
    c_walnut = '#4c2e05'
    c_ivory = '#f8f4e3'
    c_dusk = '#3c4f76'
    c_sandy = '#f19143'
    
    # 1. Base de datos depurada (Escala 0 a 60 aprox)
    labels = ['Robos y Vandalismo', 'Tráfico de Drogas', 'Balaceras Frecuentes']
    # Orden de pintado (de arriba hacia abajo en visualización)
    # matplotlib invierte el eje y, por lo que el [0] va abajo
    val_priv = np.array([45.9, 23.0, 22.0])
    val_sub = np.array([54.6, 33.5, 35.1])
    gaps = val_sub - val_priv
    
    # Eje vertical (Y coords)
    y_pos = np.arange(len(labels))

    # 2. Configurar Canvas
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=200)
    fig.patch.set_facecolor(c_ivory)
    ax.set_facecolor(c_ivory)
    
    # Limpiar bordes estruendosamente (Data-Ink ratio perfecto)
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    ax.get_xaxis().set_visible(False) # Cero grillas verticales

    # 3. Gráfica de Mancuernas (Dumbbell Geometries)
    # Trazamos el recorrido (el "mango" de la mancuerna)
    ax.hlines(y=y_pos, xmin=val_priv, xmax=val_sub, color=c_walnut, alpha=0.3, linewidth=6, zorder=1)
    
    # Puntos de Privados (Punto de Partida normal)
    ax.scatter(val_priv, y_pos, color=c_dusk, s=400, zorder=3, edgecolors=c_ivory, linewidths=2)
    # Puntos Subsidiados (El Destino Faltal)
    ax.scatter(val_sub, y_pos, color=c_sandy, s=400, zorder=3, edgecolors=c_ivory, linewidths=2)
    
    # 4. Textos Interiores/Exteriores Precisos
    for i in range(len(labels)):
        # Número Privado (dentro de o sobre la bola Azul)
        ax.text(val_priv[i] - 1.5, y_pos[i], f"{val_priv[i]:.0f}%", color=c_dusk, fontsize=12, 
                fontweight='bold', ha='right', va='center')
        
        # Número Subsidiado (dentro de o sobre la bola Naranja)
        ax.text(val_sub[i] + 1.5, y_pos[i], f"{val_sub[i]:.0f}%", color=c_sandy, fontsize=12, 
                fontweight='bold', ha='left', va='center')
        
        # Etiqueta Flotante de Brecha (+ pp) construida como píldora en el centro del mango
        mid_point = (val_priv[i] + val_sub[i]) / 2
        
        # Píldora moderna de texto central 
        bbox_props = dict(boxstyle="round,pad=0.3", fc=c_ivory, ec="none", alpha=1)
        ax.text(mid_point, y_pos[i], f"+{gaps[i]:.0f} pp", color=c_walnut, 
                fontsize=11, fontweight='bold', ha='center', va='center', zorder=4, bbox=bbox_props)

    # 5. Eje Y y Formato
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=14, fontweight='bold', color=c_walnut)
    ax.tick_params(axis='y', length=0, pad=15)
    
    # 6. Titulares
    fig.text(0.5, 1.05, "El Abandono del Entorno", fontsize=20, fontweight='bold', color=c_walnut, ha='center')
    story_text = "Diferencial de riesgo delictual habitacional en Santiago.\nA simple vista, vivir en subsidio garantiza un encierro en el ecosistema del crimen."
    fig.text(0.5, 0.95, story_text, fontsize=12, color=c_dusk, ha='center', linespacing=1.4)
    
    # 7. Leyenda Visual Súper Personalizada (Arriba derecha)
    leg_ax = fig.add_axes([0.65, 1.0, 0.3, 0.05])
    leg_ax.axis('off')
    leg_ax.scatter([0.1], [0.5], s=150, color=c_dusk)
    leg_ax.text(0.18, 0.45, "Privada (Base)", fontsize=11, color=c_walnut, va='center')
    leg_ax.scatter([0.65], [0.5], s=150, color=c_sandy)
    leg_ax.text(0.73, 0.45, "Subsidiada (+Riesgo)", fontsize=11, color=c_walnut, va='center')

    # Fuente en el pie
    fig.text(0.5, -0.05, "Elaboración propia en base a Encuesta CASEN 2024. Reportes de ocurrencia 'Siempre' o 'Frecuentemente'.",
             fontsize=9, color=c_dusk, ha='center', style='italic')

    # Ajuste e Impresión
    plt.tight_layout(rect=[0, 0.05, 1, 0.90])
    
    output_path = "deliverables/territorial_crime_dumbbell.png"
    os.makedirs("deliverables", exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', transparent=False, facecolor=fig.get_facecolor())
    print(f"Dumbbell exportado con exito a: {output_path}")

if __name__ == "__main__":
    generate_crime_dumbbell()
