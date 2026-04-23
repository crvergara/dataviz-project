import pandas as pd
import numpy as np

def analyze_territorial_gap(df):
    # 1. FILTRO: Solo Vivienda Subsidiada
    df_sub = df[df['is_subsidized'] == 1].copy()
    
    # 2. SEGMENTACIÓN: RM vs Resto del País
    df_sub['territorio'] = np.where(df_sub['region'] == 13, 'Metropolitana', 'Resto de Chile')
    
    # 3. DIMENSIONES CRÍTICAS
    df_sub['pobreza_ingreso'] = np.where(df_sub['yautcorh'] < 600000, 1, 0)
    df_sub['alta_dependencia'] = np.where(df_sub['subsidy_ratio'] > 0.20, 1, 0)
    df_sub['viaje_critico'] = np.where(df_sub['o28a_hr'] >= 1, 1, 0)
    df_sub['delincuencia'] = np.where(df_sub['v36e'] == 1, 1, 0)
    df_sub['hacinamiento'] = np.where(df_sub['ind_hacina'] >= 2, 1, 0)
    df_sub['pobreza_multi'] = np.where(df_sub['pobreza_multi'] == 1, 1, 0)

    variables = ['pobreza_ingreso', 'alta_dependencia', 'viaje_critico', 'delincuencia', 'hacinamiento', 'pobreza_multi']
    
    # 4. CÁLCULO PONDERADO (usando expr)
    results = []
    for var in variables:
        for terr in ['Metropolitana', 'Resto de Chile']:
            subset = df_sub[df_sub['territorio'] == terr]
            if subset['expr'].sum() > 0:
                pct = (subset[subset[var] == 1]['expr'].sum() / subset['expr'].sum()) * 100
                results.append({'Variable': var, 'Territorio': terr, 'Tasa (%)': pct})
                
    results_df = pd.DataFrame(results).pivot(index='Variable', columns='Territorio', values='Tasa (%)')
    results_df['Brecha (RM - Resto)'] = results_df['Metropolitana'] - results_df['Resto de Chile']
    
    print("\n--- ANÁLISIS TERRITORIAL: VIVIENDA SUBSIDIADA ---")
    print(results_df.round(1).sort_values(by='Brecha (RM - Resto)', ascending=False))

if __name__ == "__main__":
    df_master = pd.read_parquet("data/processed/master_dataset.parquet")
    analyze_territorial_gap(df_master)
