import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import seaborn as sns
import os

# ==============================================================================
# CONFIGURACIÓN GENERAL Y ESTÉTICA (A4, ALTA TINTA-DATOS)
# ==============================================================================
# Paleta estricta: Rojo para Estado/Subsidio (Alerta/Dependencia), Azul/Gris para Privado
C_SUB = "#d62728"
C_NOSUB = "#333333"

# Configuración A4 Portrait (210 x 297 mm -> 8.27 x 11.69 inches)
fig = plt.figure(figsize=(8.27, 11.69), dpi=300, facecolor='white')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Grid 2x2 para acomodar los 4 gráficos ordenadamente
gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.45, top=0.88, bottom=0.08, left=0.08, right=0.95)

# Título Principal del Póster
fig.suptitle('THE DEPENDENCY MACHINE', fontsize=26, fontweight='black', ha='center', y=0.97)
fig.text(0.5, 0.935, 'Cómo la vivienda social frena el capital humano y encierra a las familias en guetos de violencia delictual y alta dependencia estatal.', 
         ha='center', fontsize=10, color='#555555', style='italic')

# ==============================================================================
# CARGA Y PREPARACIÓN DE DATOS (master_dataset.parquet)
# ==============================================================================
try:
    df = pd.read_parquet(r"data\processed\master_dataset.parquet")
    df = df[df['grupo_vivienda'].isin(['Subsidiada', 'No Subsidiada / Otro'])]
except FileNotFoundError:
    print("Error: Ejecuta primero src/build_master_dataset.py en la raíz para generar el parquet.")
    exit()

# Extraer factores de expansión para estadística oficial
weights = df['expr']

# ==============================================================================
# GRÁFICO 1: EL ELEVADOR ROTO (Lollipop - Educación Superior) [Top-Left]
# ==============================================================================
ax1 = fig.add_subplot(gs[0, 0])

df['edu_superior'] = df['e6a_no_asiste'] >= 12
df_edu = df.dropna(subset=['e6a_no_asiste']).copy()

# Cálculo Media Ponderada
def weighted_edu(x): return pd.Series({'val': np.average(x['edu_superior'], weights=x['expr']) * 100})
res_edu = df_edu.groupby('grupo_vivienda').apply(weighted_edu).reset_index()

grupos = res_edu['grupo_vivienda']
valores = res_edu['val']

# Dibujar Lollipop
ax1.hlines(y=grupos, xmin=0, xmax=valores, color=[C_NOSUB, C_SUB], linewidth=4, zorder=1)
ax1.scatter(valores, grupos, color=[C_NOSUB, C_SUB], s=300, zorder=2)

for i, val in enumerate(valores):
    color = C_NOSUB if i == 0 else C_SUB
    ax1.text(val + 3, i, f"{val:.1f}%", color=color, fontsize=11, fontweight='bold', va='center')

ax1.set_title("1. EL ELEVADOR SOCIAL ROTO\nJefes de Hogar que alcanzan Ed. Superior", loc='left', fontsize=12, fontweight='bold', pad=15)
ax1.set_xlabel("Población (%)", fontsize=9)
ax1.set_xlim(0, 60)
ax1.tick_params(axis='y', length=0)
ax1.set_yticklabels(grupos, fontsize=10, fontweight='bold')
ax1.spines['left'].set_visible(False)
ax1.grid(axis='x', linestyle='--', alpha=0.3)

# ==============================================================================
# GRÁFICO 2: EL ESTANCAMIENTO AUTÓNOMO (KDE Plot - Income) [Top-Right]
# ==============================================================================
ax2 = fig.add_subplot(gs[0, 1])

df_inc = df[(df['yautcorh'] > 0) & (df['yautcorh'] < 3000000)].copy()

# KDE
sns.kdeplot(data=df_inc[df_inc['grupo_vivienda'] == 'No Subsidiada / Otro'], x='yautcorh', weights='expr', 
            color=C_NOSUB, fill=True, alpha=0.2, ax=ax2, label="Privado")
sns.kdeplot(data=df_inc[df_inc['grupo_vivienda'] == 'Subsidiada'], x='yautcorh', weights='expr', 
            color=C_SUB, fill=True, alpha=0.4, ax=ax2, label="Subsidiada")

# Medianas
median_sub = np.average(df_inc[df_inc['grupo_vivienda'] == 'Subsidiada']['yautcorh'], weights=df_inc[df_inc['grupo_vivienda'] == 'Subsidiada']['expr'])
median_priv = np.average(df_inc[df_inc['grupo_vivienda'] == 'No Subsidiada / Otro']['yautcorh'], weights=df_inc[df_inc['grupo_vivienda'] == 'No Subsidiada / Otro']['expr'])

ax2.axvline(median_sub, color=C_SUB, linestyle='--', linewidth=2)
ax2.axvline(median_priv, color=C_NOSUB, linestyle='--', linewidth=2)

ax2.set_title("2. ESTANCAMIENTO DEL INGRESO\nDistribución de sueldos (Autonomía bloqueada)", loc='left', fontsize=12, fontweight='bold', pad=15)
ax2.set_xlabel("Ingreso Autónomo ($ CLP)", fontsize=9)
ax2.set_ylabel("Densidad de Hogares", fontsize=9)
ax2.ticklabel_format(style='plain', axis='x')
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${int(x/1000)}k"))
ax2.legend(frameon=False, loc='upper right')

# ==============================================================================
# GRÁFICO 3: LA DEPENDENCIA CAUTIVA (Salud Fonasa - Bar) [Bottom-Left]
# ==============================================================================
ax3 = fig.add_subplot(gs[1, 0])

df_health = df.dropna(subset=['s13']).copy()
df_health['es_fonasa'] = df_health['s13'] == 1

def weighted_health(x): return pd.Series({'val': np.average(x['es_fonasa'], weights=x['expr']) * 100})
res_health = df_health.groupby('grupo_vivienda').apply(weighted_health).reset_index()

bars = ax3.bar(res_health['grupo_vivienda'], res_health['val'], color=[C_NOSUB, C_SUB], width=0.5)

for bar, color in zip(bars, [C_NOSUB, C_SUB]):
    height = bar.get_height()
    ax3.annotate(f"{height:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, -25), textcoords="offset points", ha='center', va='bottom', color='white', fontweight='bold', fontsize=14)

ax3.set_title("3. CAUTIVOS DEL ESTADO\nAtrapados en la red pública de FONASA", loc='center', fontsize=12, fontweight='bold', pad=15)
ax3.set_ylabel("% de Familias en Fonasa", fontsize=9)
ax3.set_xticklabels(res_health['grupo_vivienda'], fontsize=10, fontweight='bold')
ax3.spines[['bottom', 'left']].set_visible(False)
ax3.tick_params(axis='x', length=0)
ax3.get_yaxis().set_visible(False) # Clean data-ink

# ==============================================================================
# GRÁFICO 4: EL MAPA DEL MIEDO (Dumbbell Plot - Regiones) [Bottom-Right]
# ==============================================================================
ax4 = fig.add_subplot(gs[1, 1])

region_map = {1:"Tarapacá", 2:"Antofagasta", 3:"Atacama", 4:"Coquimbo", 5:"Valparaíso", 6:"O'Higgins", 7:"Maule", 8:"Biobío", 9:"Araucanía", 10:"Los Lagos", 11:"Aysén", 12:"Magallanes", 13:"Metropolitana", 14:"Los Ríos", 15:"Arica", 16:"Ñuble"}
df['region_name'] = df['region'].map(region_map)
df['balaceras'] = df['v36e'].isin([3, 4])
df_bala = df.dropna(subset=['v36e', 'region_name']).copy()

def weighted_bala(x): return np.average(x['balaceras'], weights=x['expr']) * 100
grp = df_bala.groupby(['region_name', 'grupo_vivienda']).apply(weighted_bala).unstack()
grp['Brecha'] = grp['Subsidiada'] - grp['No Subsidiada / Otro']

# Extraer Top 6 peores (por espacio)
grp = grp.sort_values(by='Subsidiada', ascending=False).head(6).sort_values(by='Subsidiada', ascending=True)

y_r = np.arange(len(grp))
cmap = plt.colormaps.get_cmap('Reds')
norm = mcolors.Normalize(vmin=0, vmax=grp['Brecha'].max())

for i, (idx, row) in enumerate(grp.iterrows()):
    color = cmap(norm(row['Brecha'])) if row['Brecha'] > 0 else 'grey'
    ax4.plot([row['No Subsidiada / Otro'], row['Subsidiada']], [i, i], color=color, linewidth=3, zorder=1)
    ax4.scatter(row['No Subsidiada / Otro'], i, color=C_NOSUB, s=120, zorder=2)
    ax4.scatter(row['Subsidiada'], i, color=C_SUB, s=120, zorder=2)
    
    # Textos de impacto
    ax4.text(row['Subsidiada'] + 1, i, f"{row['Subsidiada']:.0f}%", color=C_SUB, va='center', fontweight='bold', fontsize=9)
    ax4.text(row['No Subsidiada / Otro'] - 4, i, f"{row['No Subsidiada / Otro']:.0f}%", color=C_NOSUB, va='center', fontweight='bold', fontsize=9)

ax4.set_yticks(y_r)
ax4.set_yticklabels(grp.index, fontsize=9)
ax4.set_title('4. EL MAPA DEL MIEDO URBANO\nExposición a balaceras en Guetos (Aísla capitales)', loc='left', fontsize=12, fontweight='bold', pad=15)
ax4.set_xlabel('Familias bajo balaceras frecuentes (%)', fontsize=9)
ax4.spines[['left', 'bottom']].set_visible(False)
ax4.grid(axis='x', linestyle='--', alpha=0.3)
ax4.tick_params(axis='y', length=0)

# ==============================================================================
# GUARDAR EL ARCHIVO (Alta Resolución)
# ==============================================================================
out_path = r"deliverables\the_dependency_machine.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Póster generado exitosamente en: {out_path}")
