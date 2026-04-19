import pandas as pd
import numpy as np
import os

def calculate_icp(df):
    """
    Calcula el Índice de Cautiverio Patrimonial (ICP) y sus dependencias
    (Costo de Fricción Monetarizado, Tiempos de Viaje, Ratios de Dependencia).
    """
    alpha = 0.5
    beta = 0.5
    
    print("  -> Limpiando tiempos de viaje...")
    # Limpiamos respuestas inválidas (-8, -88) limitando el rango realista
    mask_invalid_hr = (df['o28a_hr'].notna()) & ((df['o28a_hr'] < 0) | (df['o28a_hr'] > 9))
    mask_invalid_min = (df['o28a_min'].notna()) & ((df['o28a_min'] < 0) | (df['o28a_min'] > 59))
    df = df[~(mask_invalid_hr | mask_invalid_min)].copy()
    
    # Tiempo IDA Y VUELTA Mensual (20 días laborales)
    daily_commute_hrs = (df['o28a_hr'].fillna(0) + (df['o28a_min'].fillna(0) / 60)) * 2
    df['commute_total_hrs'] = daily_commute_hrs * 20
    
    print("  -> Monetarizando fricción...")
    # Costo de fricción = (Ingreso / 160 hrs mensuales) * horas de traslado mensuales
    hourly_rate = df['yautcorh'] / 160
    df['c_friction'] = df['commute_total_hrs'] * hourly_rate.clip(lower=0)
    
    print("  -> Calculando ratios...")
    # Ratio de Subsidio (S / I_total)
    df['subsidy_ratio'] = (df['ysubh'] / df['ytotcorh'].replace(0, np.nan)).fillna(0)
    
    # Ratio de Aislamiento (Fricción / Ingreso Potencial)
    df['isolation_ratio'] = (df['c_friction'] / df['yautcorh'].replace(0, np.nan)).fillna(0)
    
    # ICP Final
    df['icp'] = (alpha * df['subsidy_ratio'] + beta * df['isolation_ratio']) * 100
    return df

def build_master_dataset(raw_stata_path, output_parquet_path):
    """
    Pipeline unificado de extracción, transformación y carga (ETL).
    Construye la base final desde cero, sin depender de archivos intermedios.
    """
    print("1. Cargando la base de datos cruda de la CASEN 2024 (1-2 mins)...")
    
    columnas_clave = [
        # IDs y Ponderadores
        'id_vivienda', 'folio', 'id_persona', 'region', 'expr',
        # Dinero y Pobreza
        'yautcorh', 'ysubh', 'ytotcorh', 'pobreza', 'pobreza_multi',
        # Tiempos de Viaje
        'o28a_hr', 'o28a_min',
        # Características laborales y demográficas
        'v13', 'v15', 'sexo', 'activ',
        # NUEVAS: Entorno Delictual y Exclusión Físico-Estatal
        'v36e', 'v36c', 'v36b', 'v35a', 'v35c', 'ind_hacina', 'e6a_no_asiste', 's13'
    ]
    
    # Se extraen unificadas todas las variables (viejas y nuevas)
    df = pd.read_stata(raw_stata_path, columns=columnas_clave, convert_categoricals=False)

    print("2. Aplicando limpiezas primarias...")
    # Filtrar solo Jefes de Hogar
    df = df[df['id_persona'] == 1].copy()

    # Corrección metodológica V15 (Subsidios con y sin crédito vs Privados)
    # Corrección metodológica V15 (Subsidios con y sin crédito vs Privados/Herencia)
    df['is_subsidized'] = np.where(df['v15'].isin([1, 2]), 1, 0)
    df['grupo_vivienda'] = np.where(df['is_subsidized'] == 1, 'Subsidiada', 'No Subsidiada / Otro')

    print("3. Calculando el corpus matemático (ICP, Tiempos, Fricción)...")
    df = calculate_icp(df)

    print("4. Guardando el Universo Unificado...")
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    df.to_parquet(output_parquet_path, engine='pyarrow')
    print(f"\nExito! Base Creada V4 Definitiva -> {output_parquet_path}")

if __name__ == "__main__":
    INPUT_PATH = r"data\raw\casen_2024.dta"
    OUTPUT_PATH = r"data\processed\master_dataset.parquet"
    build_master_dataset(INPUT_PATH, OUTPUT_PATH)
