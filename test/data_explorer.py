import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

def run_data_explorer(parquet_path='data/processed/icp_results.parquet'):
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

    print("\n--- C. EXCLUSIÓN LABORAL (Hogares con 0 horas de viaje) ---")
    zero_sub = (sub['commute_total_hrs'] == 0).mean() * 100
    zero_priv = (priv['commute_total_hrs'] == 0).mean() * 100
    print(f"Subsidio: {zero_sub:.2f}% de los hogares NO reportan viajes al trabajo.")
    print(f"Privado: {zero_priv:.2f}% de los hogares NO reportan viajes al trabajo.")
    print(f"Diferencia: {zero_sub - zero_priv:+.2f} puntos porcentuales.")

    print("\n--- D. COMPOSICIÓN DEL MERCADO LABORAL (Variable 'activ') ---")
    # Mapping codes: 1: Ocupado, 2: Desocupado, 3: Inactivo
    # We only care about the distribution between Subsidized and Others
    labor_map = {1: 'Ocupado', 2: 'Desocupado', 3: 'Inactivo'}
    df['labor_status'] = df['activ'].map(labor_map)
    
    labor_analysis = df.groupby(['grupo_vivienda', 'labor_status']).size().unstack(fill_value=0)
    labor_pct = labor_analysis.div(labor_analysis.sum(axis=1), axis=0) * 100
    print(labor_pct.round(2))

    print("\n--- E. SESGO DE GÉNERO EN EL TRABAJO (Mujer + Subsidio) ---")
    # Mapping: 1: Hombre, 2: Mujer
    df['genero'] = df['sexo'].map({1: 'Hombre', 2: 'Mujer'})
    
    # Analyze the percentage of Women who are 'Inactivo' in subsidized vs non-subsidized
    inactivo_mujeres = df[df['genero'] == 'Mujer'].groupby('grupo_vivienda')['activ'].apply(lambda x: (x == 3).mean() * 100)
    print("Porcentaje de Mujeres en situación de INACTIVIDAD:")
    print(inactivo_mujeres.round(2))

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


    print("\n" + "="*80)
    print(" 7. BRECHAS REGIONALES DE POBREZA E INGRESOS ")
    print("="*80)
    df_gaps = df.dropna(subset=['region', 'grupo_vivienda', 'yautcorh', 'pobreza']).copy()
    region_map = {1: "Tarapacá", 2: "Antofagasta", 3: "Atacama", 4: "Coquimbo", 5: "Valparaíso", 6: "O'Higgins", 7: "Maule", 8: "Biobío", 9: "Araucanía", 10: "Los Lagos", 11: "Aysén", 12: "Magallanes", 13: "Metropolitana", 14: "Los Ríos", 15: "Arica y Parinacota", 16: "Ñuble"}
    df_gaps['region_name'] = df_gaps['region'].map(region_map)
    df_gaps['is_poor'] = df_gaps['pobreza'].isin([1.0, 2.0])
    grp = df_gaps.groupby(['region_name', 'grupo_vivienda']).apply(
        lambda x: pd.Series({
            'poverty_rate': np.average(x['is_poor'], weights=x['expr']) * 100 if x['expr'].sum() > 0 else 0,
            'median_income': x['yautcorh'].median()
        })
    ).round(2).reset_index()
    print(grp.to_string(index=False))

    print("\n" + "="*80)
    print(" 8. PROPORCIÓN DEL INGRESO POR SUBSIDIOS POR REGIÓN ")
    print("="*80)
    df_sub = df.dropna(subset=['region', 'grupo_vivienda', 'ysubh', 'yautcorh', 'ytotcorh']).copy()
    df_sub['region_name'] = df_sub['region'].map(region_map)
    
    grp_sub = df_sub.groupby(['region_name', 'grupo_vivienda']).apply(
        lambda x: pd.Series({
            'pct_ingreso_subsidio': (np.average(x['ysubh'], weights=x['expr']) / np.average(x['ytotcorh'], weights=x['expr'])) * 100 if np.average(x['ytotcorh'], weights=x['expr']) > 0 else 0,
            'pct_ingreso_autonomo': (np.average(x['yautcorh'], weights=x['expr']) / np.average(x['ytotcorh'], weights=x['expr'])) * 100 if np.average(x['ytotcorh'], weights=x['expr']) > 0 else 0
        })
    ).round(2).reset_index()
    print(grp_sub.to_string(index=False))

    print("\n" + "="*80)
    print(" 9. MULTIPLICADOR DE DEPENDENCIA ESTATAL (1x = Base Privada) ")
    print("="*80)
    pivot_mult = grp_sub.pivot(index='region_name', columns='grupo_vivienda', values='pct_ingreso_subsidio')
    pivot_mult['multiplicador_dependencia'] = (pivot_mult['Subsidiada'] / pivot_mult['No Subsidiada / Otro']).round(2)
    pivot_mult = pivot_mult.sort_values('multiplicador_dependencia', ascending=False).reset_index()
    print(pivot_mult.to_string(index=False))

    print("\n" + "="*80)
    print(" 10. ESTADO DE HACINAMIENTO: SUBSIDIADA VS NO SUBSIDIADA ")
    print("="*80)
    if 'ind_hacina' in df.columns:
        df_hacin = df.dropna(subset=['region', 'grupo_vivienda', 'ind_hacina']).copy()
        df_hacin['region_name'] = df_hacin['region'].map(region_map)
        
        # In CASEN, 1 = Sin hacinamiento. > 1 means some level of overcrowding.
        df_hacin['is_overcrowded'] = df_hacin['ind_hacina'] > 1.0
        
        grp_hacin = df_hacin.groupby(['region_name', 'grupo_vivienda']).apply(
            lambda x: pd.Series({
                'tasa_hacinamiento_pct': np.average(x['is_overcrowded'], weights=x['expr']) * 100 if x['expr'].sum() > 0 else 0
            })
        ).round(2).reset_index()
        
        pivot_hacin = grp_hacin.pivot(index='region_name', columns='grupo_vivienda', values='tasa_hacinamiento_pct')
        pivot_hacin['brecha_pp'] = (pivot_hacin['Subsidiada'] - pivot_hacin['No Subsidiada / Otro']).round(2)
        pivot_hacin = pivot_hacin.sort_values('Subsidiada', ascending=False).reset_index()
        print(pivot_hacin.to_string(index=False))
    else:
        print("La variable 'ind_hacina' no existe en el dataset.")

    print("\n" + "="*80)
    print(" 11. RELACIÓN HACINAMIENTO Y DEPENDENCIA ESTATAL ")
    print("="*80)
    if 'ind_hacina' in df.columns:
        df_hrel = df.dropna(subset=['ind_hacina', 'ysubh', 'ytotcorh', 'grupo_vivienda']).copy()
        
        # CASEN categorizes ind_hacina into 1, 2, 3, 4
        hacin_map = {1.0: '1. Sin hacinamiento', 2.0: '2. Medio bajo', 3.0: '3. Medio alto', 4.0: '4. Crítico', 5.0: '4. Crítico'}
        df_hrel['nivel_hacinamiento'] = df_hrel['ind_hacina'].map(hacin_map).fillna('Otro')
        
        grp_hrel = df_hrel.groupby(['grupo_vivienda', 'nivel_hacinamiento']).apply(
            lambda x: pd.Series({
                'pct_ingreso_subsidio': (np.average(x['ysubh'], weights=x['expr']) / np.average(x['ytotcorh'], weights=x['expr'])) * 100 if np.average(x['ytotcorh'], weights=x['expr']) > 0 else 0,
                'avg_ingreso_subsidio': np.average(x['ysubh'], weights=x['expr'])
            })
        ).round(2).reset_index()
        
        # Filter 'Otro' if there is any garbage
        grp_hrel = grp_hrel[grp_hrel['nivel_hacinamiento'] != 'Otro']
        grp_hrel = grp_hrel.sort_values(['grupo_vivienda', 'nivel_hacinamiento'])
        
        print(grp_hrel.to_string(index=False))

if __name__ == "__main__":
    run_data_explorer()
