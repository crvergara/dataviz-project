import os
import pandas as pd
import numpy as np

def build_historical_dataset():
    raw_dir = r"data\raw"
    out_path = r"data\processed\balaceras_historico_zonas.parquet"
    
    years = {
        2015: ('casen_2015.dta', 'v38e', 'expr', 'v16'),
        2017: ('casen_2017.dta', 'v38e', 'expr', 'v15'),
        2022: ('casen_2022.dta', 'v36e', 'expr', 'v15'),
        2024: ('casen_2024.dta', 'v36e', 'expr', 'v15')
    }
    
    dfs = []
    
    for year, (filename, balacera_col, ponderador_col, sub_col) in years.items():
        file_path = os.path.join(raw_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"[{year}] No se encontró el archivo: {filename}. Saltando...")
            continue
            
        print(f"[{year}] Procesando {filename}...")
        
        target_cols = ['region', 'r', ponderador_col, sub_col, balacera_col, 'pco1', 'id_persona', 'parentesco']
        
        try:
            sample = pd.read_stata(file_path, iterator=True).read(1)
            actual_cols = list(sample.columns)
            cols_to_load = [c for c in target_cols if c in actual_cols]
            
            df = pd.read_stata(file_path, columns=cols_to_load, convert_categoricals=False)
            
            # 1. Región
            if 'r' in df.columns and 'region' not in df.columns:
                df = df.rename(columns={'r': 'region'})
            
            # 2. Jefe de Hogar
            jh_col = None
            if 'pco1' in df.columns: jh_col = 'pco1'
            elif 'id_persona' in df.columns: jh_col = 'id_persona'
            elif 'parentesco' in df.columns: jh_col = 'parentesco'
            
            if jh_col:
                df = df[df[jh_col] == 1]
            
            # --- CAMBIO 1 y 2: Mantener todo Chile y crear variable 'zona' ---
            df = df.dropna(subset=['region'])
            df['zona'] = np.where(df['region'] == 13, 'RM (Santiago)', 'Resto de Regiones')
            
            df['ponderador'] = df[ponderador_col]
            
            # --- CAMBIO 3: LÓGICA DE SUBSIDIO (Dejamos SOLO subsidiadas) ---
            df = df[df[sub_col].isin([1, 2])] # 1 y 2 son los códigos de subsidio
            
            # --- LÓGICA DE BALACERAS ---
            df = df.dropna(subset=[balacera_col])
            df['is_balacera'] = np.where(df[balacera_col].isin([3, 4]), 1, 0)
            df['year'] = year
            
            # Dejar solo las columnas limpias
            df_clean = df[['year', 'zona', 'is_balacera', 'ponderador']]
            dfs.append(df_clean)
            
            print(f"  -> {len(df_clean)} hogares subsidiados procesados a nivel nacional.")
            
        except Exception as e:
            print(f"[{year}] Error procesando: {e}")
            
    if dfs:
        master_df = pd.concat(dfs, ignore_index=True)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        master_df.to_parquet(out_path, engine='pyarrow')
        print(f"\n¡Éxito! Dataset guardado en: {out_path}")
        
        # Muestra rápida ponderada
        print("\n% Balaceras Frecuentes en Vivienda Subsidiada por Zona (Ponderado):")
        res = master_df.groupby(['year', 'zona']).apply(
            lambda x: np.average(x['is_balacera'], weights=x['ponderador']) * 100
        ).round(1)
        print(res.unstack())
        
if __name__ == "__main__":
    build_historical_dataset()
