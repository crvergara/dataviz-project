import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Cargar Datos Básicos (Subsidio y Deciles de ingreso)
df = pd.read_stata('data/raw/casen_2024.dta', convert_categoricals=False, columns=['v15', 'dau'])
df = df.dropna(subset=['v15', 'dau'])

# Filtros metodológicos oficiales
df = df[df['v15'].isin([1, 2, 3])]
df['subsidy'] = df['v15'].map({1: 'Vivienda Subsidiada', 2: 'Vivienda Subsidiada', 3: ' Mercado Libre\n(Sin Subsidio)'})

# Crear Quintiles (El concepto más básico y entendible en la sociedad chilena)
df['quintil'] = pd.cut(df['dau'], bins=[0, 2, 4, 6, 8, 10], labels=['Q1 (Más Pobre)', 'Q2', 'Q3', 'Q4', 'Q5 (Más Rico)'])

# Crear tabla de porcentajes al 100%
agg = pd.crosstab(df['subsidy'], df['quintil'], normalize='index') * 100
agg = agg.loc[['Vivienda Subsidiada', ' Mercado Libre\n(Sin Subsidio)']]

# -------------------------------------------------------------
# 2. CONFIGURACIÓN VISUAL (Mapa de Calor de Quintiles)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4), facecolor='#fafafa')
ax.set_facecolor('#fafafa')

# Generar el Mapa de Calor (Heatmap)
# Usaremos 'Reds' como la escala secuencial (De casi blanco cruzando por naranja hasta Rojo Oscuro)
sns.heatmap(agg, annot=True, fmt='.1f', cmap='Reds', 
            cbar_kws={'label': '% de Concentración Poblacional'},
            linewidths=3, ax=ax, vmin=0, vmax=30,
            annot_kws={'fontweight':'bold', 'fontsize':12})

# Títulos Premium
plt.title('La Trampa del Subsidio: Concentración en Quintiles de Ingreso', 
          fontsize=16, loc='left', pad=40, fontweight='black', color='#111827')

plt.text(0, 1.22, "El color secuencial representa la densidad poblacional.\nLa vivienda estatal atrapa a casi todo su volumen (Rojo oscuro) en los quintiles 1 y 2.", 
         fontsize=11, color='#4b5563', style='italic', transform=ax.transAxes)

# Estética de los Ejes
ax.set_xlabel('Quintiles de Ingreso Autónomo Estructurado', fontweight='bold', labelpad=15, color='#374151')
ax.set_ylabel('')

# Limpieza de Ticks
plt.xticks(fontweight='bold', color='#111827', fontsize=11)
plt.yticks(fontweight='black', color='#111827', fontsize=11, rotation=0)

plt.tight_layout()
plt.savefig('deliverables/grafico_secuencial_quintiles_heatmap.png', dpi=300, bbox_inches='tight', facecolor='#fafafa')
plt.show()

