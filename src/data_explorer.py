import pandas as pd
import numpy as np
import os
import sys

class OutputLogger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def run_data_explorer(parquet_path='data/processed/icp_results.parquet'):
    # Create the output directory and start logging
    os.makedirs('data_explorer', exist_ok=True)
    sys.stdout = OutputLogger('data_explorer/icp_tesis.txt')
    
    try:
        df = pd.read_parquet(parquet_path)
        # Clean potential infinite values for statistical analysis
        df = df.replace([np.inf, -np.inf], np.nan)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {parquet_path}")
        return

    # Create grouping variable for easier analysis
    df['grupo_vivienda'] = np.where(df['v15'] == 1, 'Subsidiada', 'No Subsidiada / Otro')

    print("="*80)
    print(" EXPLORADOR DE DATOS: MÁQUINA DE DEPENDENCIA ")
    print("="*80)
    print(f"Total de Registros (Hogares/Personas): {len(df):,}")
    print(f"Total de Variables (Columnas): {len(df.columns)}")

    print("\n" + "="*80)
    print(" 1. DICCIONARIO DE VARIABLES ")
    print("="*80)
    data_dict = {
        'id_vivienda': 'Identificador único de la vivienda.',
        'folio': 'Identificador único del hogar dentro de la vivienda.',
        'id_persona': 'Identificador de la persona dentro del hogar.',
        'region': 'Región de residencia (1 a 16).',
        'expr': 'Factor de expansión regional (peso muestral CASEN).',
        'v13': 'Tenencia de la vivienda.',
        'v15': 'Adquisición con subsidio estatal (1 = Sí).',
        'o28a_hr': 'Horas de traslado al trabajo (Ida).',
        'o28a_min': 'Minutos de traslado al trabajo (Ida).',
        'yautcorh': 'Ingreso Autónomo del Hogar (A_potential).',
        'ysubh': 'Ingresos por Subsidios del Estado (S).',
        'ytotcorh': 'Ingreso Total del Hogar (I_total).',
        'pobreza': 'Situación de pobreza del hogar.',
        'commute_total_hrs': 'Tiempo total de viaje (Ida y vuelta en horas).',
        'c_friction': 'Costo de Fricción monetarizado.',
        'subsidy_ratio': 'Dependencia del Estado (S / I_total).',
        'isolation_ratio': 'Impacto del aislamiento (C_friction / A_potential).',
        'icp': 'Índice de Cautiverio Patrimonial.',
        'is_subsidized': 'Variable binaria (1 = Subsidiada).'
    }

    for col in df.columns:
        if col != 'grupo_vivienda':
            desc = data_dict.get(col, 'Descripción no disponible.')
            missing = df[col].isna().sum()
            missing_pct = (missing / len(df)) * 100
            print(f"- {col}: {desc} | Nulos: {missing:,} ({missing_pct:.1f}%)")

    print("\n" + "="*80)
    print(" 2. ESTADÍSTICAS DESCRIPTIVAS GENERALES (Variables Clave) ")
    print("="*80)
    key_vars = ['yautcorh', 'ysubh', 'ytotcorh', 'commute_total_hrs', 'c_friction', 'subsidy_ratio', 'isolation_ratio', 'icp']
    desc_stats = df[key_vars].describe(percentiles=[0.25, 0.5, 0.75, 0.95]).T
    desc_stats['missing'] = df[key_vars].isna().sum()
    desc_stats.index = desc_stats.index.map(lambda x: data_dict.get(x, x))
    pd.options.display.float_format = '{:,.2f}'.format
    print(desc_stats[['count', 'missing', 'mean', '50%', '95%', 'max']])

    print("\n" + "="*80)
    print(" 3. COMPARACIÓN MACRO: VIVIENDA SUBSIDIADA VS NO SUBSIDIADA ")
    print("="*80)
    comparison = df.groupby('grupo_vivienda')[key_vars].mean().T
    comparison.index = comparison.index.map(lambda x: data_dict.get(x, x))
    print("PROMEDIOS POR GRUPO:")
    print(comparison)

    print("\n" + "="*80)
    print(" 4. ANÁLISIS PROFUNDO DE TIEMPOS DE VIAJE (Commute Analysis) ")
    print("="*80)
    sub = df[df['v15'] == 1]
    priv = df[df['v15'] != 1]

    print("--- A. PROMEDIO GLOBAL (Incluyendo a todos, incluso 0 hrs) ---")
    print(f"Subsidiados: {sub['commute_total_hrs'].mean():.2f} horas")
    print(f"No Subsidiados: {priv['commute_total_hrs'].mean():.2f} horas")

    print("\n--- B. PROMEDIO REAL (Solo la gente que viaja al trabajo > 0 hrs) ---")
    print(f"Subsidiados: {sub[sub['commute_total_hrs'] > 0]['commute_total_hrs'].mean():.2f} horas")
    print(f"No Subsidiados: {priv[priv['commute_total_hrs'] > 0]['commute_total_hrs'].mean():.2f} horas")

    print("\n" + "="*80)
    print(" 5. ANÁLISIS REGIONAL DE TIEMPOS DE VIAJE (Regional Commute) ")
    print("="*80)
    df_viajeros = df[df['commute_total_hrs'] > 0]
    
    print("--- PROMEDIO REAL POR REGIÓN (Solo > 0 hrs) ---")
    real_commute = df_viajeros.groupby(['region', 'grupo_vivienda'])['commute_total_hrs'].mean().unstack()
    real_commute['Diferencia (Sub - No Sub)'] = real_commute['Subsidiada'] - real_commute['No Subsidiada / Otro']
    print(real_commute.round(3))

    print("\n" + "="*80)
    print(" 6. DETECCIÓN DE ANOMALÍAS (Data Quality) ")
    print("="*80)
    negative_commute = (df['commute_total_hrs'] < 0).sum()
    zero_commute = (df['commute_total_hrs'] == 0).sum()
    print(f"- Registros con tiempo de viaje negativo: {negative_commute:,} (Debería ser 0 tras limpieza)")
    print(f"- Registros con tiempo de viaje igual a cero: {zero_commute:,} (Posible desempleo, inactividad o trabajo en casa)")
    high_subsidy = (df['subsidy_ratio'] > 1).sum()
    print(f"- Hogares donde el subsidio es mayor al ingreso total (Ratio > 1): {high_subsidy:,}")

if __name__ == "__main__":
    run_data_explorer()
