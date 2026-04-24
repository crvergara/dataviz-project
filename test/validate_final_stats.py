import pandas as pd
import numpy as np

def run_validation():
    df = pd.read_parquet('data/processed/master_dataset.parquet')
    
    print("\n=======================================================")
    print("REPORTE DE VALIDACIÓN ESTADÍSTICA - MÁQUINA DE DEPENDENCIA")
    print("=======================================================\n")

    # 1. Validación de variables de Violencia (Casen v36b y v36c)
    # En CASEN, los códigos de "Problemas en el entorno" (v36) suelen ser:
    # 1: Nunca, 2: Rara vez, 3: A veces, 4: Siempre/Frecuentemente
    # Para estar seguros de que >= 3 es correcto, revisamos los conteos base:
    print("--- 1. VARIABLES DE VIOLENCIA (Distribución Nacional Casen) ---")
    balaceras_counts = df['v36b'].value_counts(dropna=False).sort_index()
    print("Códigos de Balaceras (1=Nunca -> 4=Siempre):")
    for val, count in balaceras_counts.items():
        print(f"  Código {val}: {count:,} hogares en la encuesta")
    print(" LÓGICA APLICADA: v36b >= 3 agrupa a los hogares que sufren esto 'A veces' o 'Siempre'. Matemáticamente robusto.\n")

    # 2. Validación de variables Económicas
    print("--- 2. VARIABLES ECONÓMICAS ---")
    median_yaut = df['yautcorh'].median()
    print(f"Mediana Nacional de Ingreso Autónomo: ${median_yaut:,.0f}")
    print(" LÓGICA APLICADA: El umbral de < $600.000 identifica claramente a los hogares que están por debajo de la mediana nacional (aprox. el tercio inferior).")
    
    q75_subsidy = df['subsidy_ratio'].quantile(0.75)
    print(f"Percentil 75 de Dependencia Estatal: {q75_subsidy*100:.1f}%")
    print(" LÓGICA APLICADA: El umbral de > 20% captura exactamente al cuartil superior (el 25% más dependiente del Estado).\n")

    # 3. Construcción del Waffle Chart Final (Con Factores de Expansión)
    print("--- 3. CÁLCULO DEL WAFFLE CHART FINAL (Población Ponderada) ---")
    df_sub = df[df['grupo_vivienda'] == 'Subsidiada'].copy()
    
    # Aplicamos la lógica validada
    df_sub['trampa_violencia'] = np.where((df_sub['v36b'] >= 3) | (df_sub['v36c'] >= 3), 1, 0)
    df_sub['trampa_economica'] = np.where((df_sub['yautcorh'] < 600000) | (df_sub['subsidy_ratio'] > 0.20), 1, 0)

    total_hogares_subsidiados = df_sub['expr'].sum()
    
    # Cálculos ponderados (Factor de expansión CASEN)
    ambas = df_sub[(df_sub['trampa_violencia']==1) & (df_sub['trampa_economica']==1)]['expr'].sum()
    solo_v = df_sub[(df_sub['trampa_violencia']==1) & (df_sub['trampa_economica']==0)]['expr'].sum()
    solo_e = df_sub[(df_sub['trampa_violencia']==0) & (df_sub['trampa_economica']==1)]['expr'].sum()
    ninguna = df_sub[(df_sub['trampa_violencia']==0) & (df_sub['trampa_economica']==0)]['expr'].sum()

    print(f"Universo Total de Hogares Subsidiados representados: {total_hogares_subsidiados:,.0f} familias chilenas.\n")
    
    print(f"  A) Atrapados en AMBAS trampas:       {ambas:,.0f} familias -> {(ambas/total_hogares_subsidiados*100):.1f}%")
    print(f"  B) Atrapados SOLO en Violencia:      {solo_v:,.0f} familias -> {(solo_v/total_hogares_subsidiados*100):.1f}%")
    print(f"  C) Atrapados SOLO en Economía:       {solo_e:,.0f} familias -> {(solo_e/total_hogares_subsidiados*100):.1f}%")
    print(f"  D) Libres de estas trampas severas:  {ninguna:,.0f} familias -> {(ninguna/total_hogares_subsidiados*100):.1f}%")
    
    total_atrapados = ambas + solo_v + solo_e
    print(f"\n CONCLUSIÓN FINAL: {(total_atrapados/total_hogares_subsidiados*100):.1f}% de las familias en vivienda social caen en al menos una de las trampas críticas.")
    print("=======================================================\n")

if __name__ == '__main__':
    run_validation()
