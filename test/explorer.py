import pandas as pd
import numpy as np

def run_data_explorer(parquet_path='data/processed/icp_results_v3.parquet'):
    try:
        df = pd.read_parquet(parquet_path)
        df = df.replace([np.inf, -np.inf], np.nan)
        # Filtramos para analizar solo el contraste que nos importa
        df = df[df['grupo_vivienda'].isin(['Subsidiada', 'No Subsidiada / Otro'])]
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {parquet_path}.")
        print("Asegúrate de haber generado el parquet V3 primero.")
        return

    print("="*80)
    print(" EXPLORADOR DE DATOS: MÁQUINA DE DEPENDENCIA (V2 - ENTORNO Y EXCLUSIÓN) ")
    print("="*80)
    print(f"Total de Registros (Hogares): {len(df):,}")

    print("\n" + "="*80)
    print(" 1. ANÁLISIS DE ENTORNO CRÍTICO (Balaceras y Tráfico) ")
    print("="*80)
    # v36e: Balaceras (1=Nunca, 2=Pocas, 3=Muchas, 4=Siempre)
    # v36c: Tráfico de Drogas (1=Nunca, 2=Pocas, 3=Muchas, 4=Siempre)
    
    df['balaceras_frecuentes'] = df['v36e'].isin([3, 4])
    df['trafico_frecuente'] = df['v36c'].isin([3, 4])
    
    entorno = df.groupby('grupo_vivienda').apply(
        lambda x: pd.Series({
            '% Oye Balaceras (Muchas/Siempre)': np.average(x['balaceras_frecuentes'], weights=x['expr']) * 100 if pd.notna(x['balaceras_frecuentes']).any() else 0,
            '% Ve Tráfico (Muchas/Siempre)': np.average(x['trafico_frecuente'], weights=x['expr']) * 100 if pd.notna(x['trafico_frecuente']).any() else 0
        })
    ).round(1)
    
    print(entorno)

    print("\n" + "="*80)
    print(" 2. AISLAMIENTO ESTRUCTURAL (Transporte y Salud) ")
    print("="*80)
    # v35a: Cerca Transporte (1=Sí, 2=No)
    # v35c: Cerca Salud (1=Sí, 2=No)
    
    df['lejos_transporte'] = df['v35a'] == 2
    df['lejos_salud'] = df['v35c'] == 2
    
    aislamiento = df.groupby('grupo_vivienda').apply(
        lambda x: pd.Series({
            '% Lejos Transporte Público': np.average(x['lejos_transporte'], weights=x['expr']) * 100,
            '% Lejos Centro de Salud': np.average(x['lejos_salud'], weights=x['expr']) * 100
        })
    ).round(1)
    
    print(aislamiento)

    print("\n" + "="*80)
    print(" 3. CONDICIONES FÍSICAS (Hacinamiento) ")
    print("="*80)
    # ind_hacina: 1=Sin Hacinamiento, 2=Medio, 3=Alto, 4=Crítico
    df['hay_hacinamiento'] = df['ind_hacina'].isin([2, 3, 4])
    
    hacin = df.groupby('grupo_vivienda').apply(
        lambda x: pd.Series({
            '% Hogares Hacinados': np.average(x['hay_hacinamiento'], weights=x['expr']) * 100
        })
    ).round(1)
    print(hacin)

    print("\n" + "="*80)
    print(" 4. EXCLUSIÓN LABORAL Y DE GÉNERO ")
    print("="*80)
    # activ: 3 = Inactivo
    # sexo: 1 = Hombre, 2 = Mujer
    df['es_inactivo'] = df['activ'] == 3
    df['es_mujer'] = df['sexo'] == 2
    
    # Inactividad General
    inact = df.groupby('grupo_vivienda').apply(
        lambda x: pd.Series({
            '% Población Inactiva Total': np.average(x['es_inactivo'], weights=x['expr']) * 100
        })
    ).round(1)
    print("--- INACTIVIDAD GENERAL ---")
    print(inact)
    
    # Inactividad en Mujeres
    print("\n--- INACTIVIDAD EN MUJERES ---")
    df_mujeres = df[df['es_mujer']]
    inact_mujeres = df_mujeres.groupby('grupo_vivienda').apply(
        lambda x: pd.Series({
            '% Mujeres Inactivas': np.average(x['es_inactivo'], weights=x['expr']) * 100
        })
    ).round(1)
    print(inact_mujeres)


    print("\n" + "="*80)
    print(" 5. BRECHAS REGIONALES CLAVE (Pobreza e Ingreso) ")
    print("="*80)
    df_gaps = df.dropna(subset=['region', 'grupo_vivienda', 'yautcorh', 'pobreza_multi']).copy()
    region_map = {1: "Tarapacá", 2: "Antofagasta", 3: "Atacama", 4: "Coquimbo", 5: "Valparaíso", 6: "O'Higgins", 7: "Maule", 8: "Biobío", 9: "Araucanía", 10: "Los Lagos", 11: "Aysén", 12: "Magallanes", 13: "Metropolitana", 14: "Los Ríos", 15: "Arica", 16: "Ñuble"}
    df_gaps['region_name'] = df_gaps['region'].map(region_map)
    df_gaps['is_poor'] = df_gaps['pobreza_multi'].isin([1.0, 2.0])
    
    grp = df_gaps.groupby(['region_name', 'grupo_vivienda']).apply(
        lambda x: pd.Series({
            'Tasa Pobreza (%)': np.average(x['is_poor'], weights=x['expr']) * 100 if x['expr'].sum() > 0 else 0,
            'Mediana Autonomo ($)': x['yautcorh'].median()
        })
    ).round(1).reset_index()
    print(grp.to_string(index=False))

if __name__ == "__main__":
    run_data_explorer()
